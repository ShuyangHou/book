"""Select sharp keyframes from a shelf video with optional ROI cropping.

Typical usage:
    python _tools/select_video_keyframes.py --video dataset/001.mp4 --pick-roi

This script focuses on the front half of the inventory pipeline:
    video -> ROI -> sampled candidate frames -> blur filtering -> time-coverage keyframes
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    blur_score: float
    edge_score: float
    brightness: float


def parse_roi(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    x, y, w, h = [int(p) for p in parts]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def pick_roi_interactive(frame: np.ndarray) -> tuple[int, int, int, int]:
    rect = cv2.selectROI("Select Shelf ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Shelf ROI")
    x, y, w, h = [int(v) for v in rect]
    if w <= 0 or h <= 0:
        raise SystemExit("ROI selection cancelled.")
    return x, y, w, h


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_scores(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_score = float(edges.mean())
    brightness = float(gray.mean())
    return blur_score, edge_score, brightness


def suppress_near_duplicates(
    candidates: list[CandidateFrame],
    roi_snapshots: dict[int, np.ndarray],
    min_gap_sec: float,
    similarity_threshold: float,
) -> list[CandidateFrame]:
    kept: list[CandidateFrame] = []
    for candidate in candidates:
        if kept and candidate.timestamp_sec - kept[-1].timestamp_sec < min_gap_sec:
            continue

        current = roi_snapshots[candidate.frame_index]
        duplicate = False
        for prior in kept[-2:]:
            previous = roi_snapshots[prior.frame_index]
            diff = float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))
            if diff <= similarity_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def select_best_by_windows(
    candidates: list[CandidateFrame],
    roi_snapshots: dict[int, np.ndarray],
    target_count: int,
    min_gap_sec: float,
    similarity_threshold: float,
) -> list[CandidateFrame]:
    if len(candidates) <= target_count:
        return suppress_near_duplicates(candidates, roi_snapshots, min_gap_sec, similarity_threshold)

    start_ts = candidates[0].timestamp_sec
    end_ts = candidates[-1].timestamp_sec
    duration = max(end_ts - start_ts, 1e-6)
    window_size = duration / target_count

    windowed: list[CandidateFrame] = []
    for idx in range(target_count):
        left = start_ts + idx * window_size
        right = end_ts + 1e-6 if idx == target_count - 1 else left + window_size
        members = [c for c in candidates if left <= c.timestamp_sec < right]
        if not members:
            continue
        members.sort(key=lambda c: (c.blur_score, c.edge_score), reverse=True)
        windowed.append(members[0])

    windowed.sort(key=lambda c: c.timestamp_sec)
    return suppress_near_duplicates(windowed, roi_snapshots, min_gap_sec, similarity_threshold)


def format_stamp(seconds: float) -> str:
    rounded = round(seconds, 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded)):02d}s"
    return f"{rounded:05.2f}s"


def write_manifest(
    path: Path,
    *,
    video_path: Path,
    roi: tuple[int, int, int, int],
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    sampled_candidates: list[CandidateFrame],
    selected: list[CandidateFrame],
) -> None:
    payload = {
        "video_path": str(video_path),
        "roi_xywh": list(roi),
        "fps": fps,
        "frame_count": frame_count,
        "frame_size": {"width": width, "height": height},
        "sampled_candidates": [asdict(c) for c in sampled_candidates],
        "selected_keyframes": [asdict(c) for c in selected],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input shelf video path")
    parser.add_argument("--output-dir", default="output/keyframes", help="Directory to save selected keyframes")
    parser.add_argument("--manifest-dir", default="output/keyframes_manifest", help="Directory to save selection metadata")
    parser.add_argument("--video-id", default="", help="Override video id prefix, default is video stem")
    parser.add_argument("--roi", default="", help="Shelf ROI as x,y,w,h")
    parser.add_argument("--pick-roi", action="store_true", help="Pick ROI interactively from the first frame")
    parser.add_argument("--sample-sec", type=float, default=0.5, help="Sample interval for candidate frames")
    parser.add_argument("--target-count", type=int, default=4, help="Desired number of selected keyframes")
    parser.add_argument("--blur-threshold", type=float, default=40.0, help="Minimum Laplacian variance to keep")
    parser.add_argument("--min-gap-sec", type=float, default=1.5, help="Minimum time gap between selected frames")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=8.0,
        help="Mean absolute ROI difference threshold for duplicate suppression",
    )
    parser.add_argument("--save-roi-preview", action="store_true", help="Also save ROI-only crops for inspection")
    args = parser.parse_args()

    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise SystemExit(f"Invalid video metadata for {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit(f"Failed to read first frame from {video_path}")

    if args.pick_roi:
        roi = pick_roi_interactive(first_frame)
    elif args.roi:
        roi = parse_roi(args.roi, width, height)
    else:
        roi = (0, 0, width, height)

    x, y, w, h = roi
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sample_step = max(1, int(round(args.sample_sec * fps)))

    sampled_candidates: list[CandidateFrame] = []
    roi_snapshots: dict[int, np.ndarray] = {}
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_step != 0:
            frame_index += 1
            continue

        roi_frame = frame[y : y + h, x : x + w]
        blur_score, edge_score, brightness = compute_scores(roi_frame)
        if blur_score >= args.blur_threshold:
            candidate = CandidateFrame(
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                blur_score=blur_score,
                edge_score=edge_score,
                brightness=brightness,
            )
            sampled_candidates.append(candidate)
            snapshot = cv2.cvtColor(cv2.resize(roi_frame, (96, 96)), cv2.COLOR_BGR2GRAY)
            roi_snapshots[frame_index] = snapshot
        frame_index += 1

    cap.release()

    if not sampled_candidates:
        raise SystemExit(
            f"No candidate frames survived blur threshold {args.blur_threshold}. "
            "Try lowering --blur-threshold."
        )

    sampled_candidates.sort(key=lambda c: c.timestamp_sec)
    selected = select_best_by_windows(
        sampled_candidates,
        roi_snapshots,
        target_count=args.target_count,
        min_gap_sec=args.min_gap_sec,
        similarity_threshold=args.similarity_threshold,
    )
    if not selected:
        raise SystemExit("No keyframes selected after duplicate suppression.")

    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)
    ensure_dir(output_dir)
    ensure_dir(manifest_dir)
    roi_preview_dir = output_dir / "roi_preview"
    if args.save_roi_preview:
        ensure_dir(roi_preview_dir)

    video_id = args.video_id.strip() or video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    selected_by_index = {c.frame_index: c for c in selected}
    saved = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        candidate = selected_by_index.get(frame_index)
        if candidate is None:
            frame_index += 1
            continue

        stamp = format_stamp(candidate.timestamp_sec)
        image_name = f"{video_id}_{stamp}.jpg"
        image_path = output_dir / image_name
        cv2.imwrite(str(image_path), frame)

        if args.save_roi_preview:
            roi_frame = frame[y : y + h, x : x + w]
            cv2.imwrite(str(roi_preview_dir / image_name), roi_frame)

        saved += 1
        frame_index += 1
        if saved >= len(selected):
            break

    cap.release()

    manifest_path = manifest_dir / f"{video_id}.json"
    write_manifest(
        manifest_path,
        video_path=video_path,
        roi=roi,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        sampled_candidates=sampled_candidates,
        selected=selected,
    )

    print(f"video             -> {video_path}")
    print(f"roi               -> {roi}")
    print(f"sampled candidates-> {len(sampled_candidates)}")
    print(f"selected keyframes-> {len(selected)}")
    print(f"output dir        -> {output_dir}")
    print(f"manifest          -> {manifest_path}")


if __name__ == "__main__":
    main()
