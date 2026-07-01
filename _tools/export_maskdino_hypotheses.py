"""Export MaskDINO instance masks as DEVA-compatible frame-level hypotheses.

MaskDINO runs at maskdino_fps; all frames are exported at deva_fps.
Intermediate DEVA frames (no MaskDINO injection) are saved for later
mask propagation by track_masks_deva.py.

Pipeline:
    MP4 video
    -> ROI crop
    -> export ROI frames at deva_fps to deva_frames/
    -> run MaskDINO at maskdino_fps on injection frames
    -> save per-frame NPZ hypotheses + CSV metadata
    -> assemble maskdino_hypotheses.mp4 (mask-only visualization, no bboxes)
"""

from __future__ import annotations

import argparse
import csv
import json
from math import gcd
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASKDINO_ROOT = PROJECT_ROOT / "MaskDINO"
DEFAULT_CONFIG_FILE = (
    DEFAULT_MASKDINO_ROOT
    / "configs"
    / "coco"
    / "instance-segmentation"
    / "maskdino_R50_bs16_50ep_3s_dowsample1_2048_bitmask.yaml"
)

# ---------------------------------------------------------------------------
# helpers (kept from original)
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def normalize_roi_xyxy(x1: int, y1: int, x2: int, y2: int, frame_shape: tuple) -> dict:
    h, w = frame_shape[:2]
    return {
        "x1": max(0.0, min(1.0, x1 / w)),
        "y1": max(0.0, min(1.0, y1 / h)),
        "x2": max(0.0, min(1.0, x2 / w)),
        "y2": max(0.0, min(1.0, y2 / h)),
    }


def denormalize_roi_xyxy(roi: dict, frame_shape: tuple) -> list[int]:
    h, w = frame_shape[:2]
    x1 = int(round(float(roi["x1"]) * w))
    y1 = int(round(float(roi["y1"]) * h))
    x2 = int(round(float(roi["x2"]) * w))
    y2 = int(round(float(roi["y2"]) * h))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def save_roi_json(path: Path, roi: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(roi, ensure_ascii=False, indent=2), encoding="utf-8")


def load_roi_json(path: Path) -> dict:
    roi = json.loads(path.read_text(encoding="utf-8"))
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(roi):
        raise SystemExit(f"ROI json missing keys: {sorted(required - set(roi))}")
    return roi


def get_video_stats(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise SystemExit(f"Invalid source FPS for video: {video_path}")
    return fps, frame_count


def read_frame_at_index(video_path: Path, frame_index: int) -> np.ndarray:
    """Read a specific frame; falls back to sequential read on random-access failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok and frame is not None:
        cap.release()
        return frame

    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to reopen video for sequential read: {video_path}")

    current_index = 0
    while current_index <= frame_index:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise SystemExit(f"Failed to read frame {frame_index} from video: {video_path}")
        if current_index == frame_index:
            cap.release()
            return frame
        current_index += 1

    cap.release()
    raise SystemExit(f"Failed to read frame {frame_index} from video: {video_path}")


def read_reference_frame(video_path: Path, mode: str) -> np.ndarray:
    _, frame_count = get_video_stats(video_path)
    target_index = 0
    if mode == "middle" and frame_count > 0:
        target_index = frame_count // 2
    return read_frame_at_index(video_path, target_index)


def export_roi_candidate_frames(video_path: Path, out_dir: Path, count: int) -> list[dict]:
    if count <= 0:
        return []

    fps, frame_count = get_video_stats(video_path)
    if frame_count <= 0:
        raise SystemExit(f"Video has no readable frames: {video_path}")

    ensure_dir(out_dir)
    indices = np.linspace(0, max(0, frame_count - 1), num=count)
    unique_indices = sorted({int(round(v)) for v in indices})
    index_set = set(unique_indices)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    candidates = []
    current_index = 0
    target_ptr = 0
    max_target = unique_indices[-1]

    while current_index <= max_target and target_ptr < len(unique_indices):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if current_index in index_set:
            candidate_idx = target_ptr + 1
            frame_index = unique_indices[target_ptr]
            filename = f"candidate_{candidate_idx:02d}_frame_{frame_index:06d}.jpg"
            path = out_dir / filename
            cv2.imwrite(str(path), frame)
            candidates.append({
                "candidate_index": candidate_idx,
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "path": path,
            })
            target_ptr += 1
        current_index += 1

    cap.release()
    if len(candidates) != len(unique_indices):
        missing = unique_indices[len(candidates):]
        raise SystemExit(
            f"Failed to export ROI candidate frames for indices {missing} from video: {video_path}"
        )
    return candidates


def choose_roi_candidate(candidates: list[dict], requested_index: Optional[int]) -> dict:
    if not candidates:
        raise SystemExit("No ROI candidates were exported.")
    valid_indices = {row["candidate_index"] for row in candidates}
    if requested_index is not None:
        if requested_index not in valid_indices:
            raise SystemExit(
                f"Invalid --roi-candidate-index={requested_index}; choose from {sorted(valid_indices)}"
            )
        return next(row for row in candidates if row["candidate_index"] == requested_index)

    print("ROI candidate frames:")
    for row in candidates:
        print(f"  [{row['candidate_index']}] frame={row['frame_index']:06d} "
              f"time={row['time_sec']:.2f}s path={row['path']}")

    while True:
        raw = input(f"Choose ROI candidate index {sorted(valid_indices)}: ").strip()
        try:
            chosen = int(raw)
        except ValueError:
            print("Please enter an integer candidate index.")
            continue
        if chosen in valid_indices:
            return next(row for row in candidates if row["candidate_index"] == chosen)
        print(f"Invalid choice: {chosen}")


def select_roi_interactively(frame: np.ndarray) -> dict:
    win_name = "Select target shelf ROI"
    x, y, w, h = cv2.selectROI(win_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_name)
    if w <= 0 or h <= 0:
        raise SystemExit("ROI selection was cancelled or empty.")
    return normalize_roi_xyxy(int(x), int(y), int(x + w), int(y + h), frame.shape)


def draw_roi_outline(image: np.ndarray, roi_bbox: Optional[list[int]]) -> np.ndarray:
    if roi_bbox is None:
        return image
    vis = image.copy()
    x1, y1, x2, y2 = roi_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 215, 255), 2)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 215, 255), 1)
    return vis


# ---------------------------------------------------------------------------
# MaskDINO predictor (kept from original)
# ---------------------------------------------------------------------------

def build_predictor(weights: str, config_file: str, score_threshold: float):
    try:
        from detectron2.engine import DefaultPredictor
        from train_maskdino_r50 import build_config
    except ImportError as exc:
        raise SystemExit(
            "Detectron2 is not installed in this environment. "
            "Install detectron2 + MaskDINO first, then rerun this script."
        ) from exc

    cfg = build_config(
        data_root="book_spine_dataset/coco",
        output_dir="output/inference_tmp",
        max_iter=1,
        batch_size=1,
        lr=0.0001,
        num_gpus=1,
        weights=weights,
        config_file=config_file,
        num_workers=2,
        allow_fallback=False,
    )
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
    if hasattr(cfg.MODEL, "PANOPTIC_FPN"):
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_threshold
    return DefaultPredictor(cfg)


# ---------------------------------------------------------------------------
# dual-FPS frame iterator
# ---------------------------------------------------------------------------

def iter_deva_frames(video_path: Path, deva_fps: float):
    """Yield (deva_frame_index, source_frame_index, time_sec, frame) at deva_fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_fps <= 0:
        raise SystemExit(f"Invalid source FPS for video: {video_path}")

    duration = frame_count / src_fps if frame_count > 0 else 0.0
    print(f"video={video_path.name} src_fps={src_fps:.3f} "
          f"frames={frame_count} duration={duration:.2f}s "
          f"deva_fps={deva_fps:.3f}")

    interval = 1.0 / deva_fps
    next_sample_time = 0.0
    deva_frame_index = 0
    source_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        time_sec = source_index / src_fps
        if time_sec + 1e-9 >= next_sample_time:
            yield deva_frame_index, source_index, time_sec, frame
            deva_frame_index += 1
            next_sample_time += interval

        source_index += 1

    cap.release()


def should_run_maskdino(deva_frame_index: int, deva_fps: float, maskdino_fps: float) -> bool:
    """Determine if MaskDINO should inject at this DEVA frame.

    With deva_fps=6, maskdino_fps=3: injection every 2nd DEVA frame.
    With deva_fps=3, maskdino_fps=3: injection every frame.
    """
    if maskdino_fps >= deva_fps:
        return True
    step = max(1, int(round(deva_fps / maskdino_fps)))
    return (deva_frame_index % step) == 0


# ---------------------------------------------------------------------------
# mask quality analysis
# ---------------------------------------------------------------------------

def analyze_mask(mask: np.ndarray, frame_height: int, frame_width: int,
                 all_frame_masks: list[np.ndarray] | None = None) -> dict:
    """Analyze a single binary mask for quality and merge suspicion.

    Returns dict with:
        area, width, height, fill_ratio, is_empty, is_small,
        suspected_merged, touches_edge, accepted_for_deva
    """
    h, w = mask.shape
    area = int(mask.sum())
    bbox = mask_to_bbox(mask)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    bbox_area = max(1, bw * bh)
    fill_ratio = area / bbox_area

    is_empty = area == 0
    is_small = area < 256  # ~16x16 pixels
    is_too_short = bh < frame_height * 0.15  # book spine should be tall

    # Edge contact
    edge_margin = max(4, int(0.02 * w))
    touches_left = bbox[0] <= edge_margin
    touches_right = bbox[2] >= w - edge_margin
    touches_edge = touches_left or touches_right

    # Small edge fragment
    is_small_edge_fragment = is_small and touches_edge

    # Suspected merged: width > 2x median width among current frame masks
    suspected_merged = False
    if all_frame_masks and len(all_frame_masks) > 1:
        widths = []
        for m in all_frame_masks:
            if m is not mask:
                mb = mask_to_bbox(m)
                widths.append(mb[2] - mb[0])
        if widths:
            median_w = float(np.median(widths))
            if median_w > 0 and bw > median_w * 2.0:
                suspected_merged = True

    accepted = not is_empty and not is_small and not is_too_short and not is_small_edge_fragment

    return {
        "area": area,
        "width": bw,
        "height": bh,
        "bbox_area": bbox_area,
        "fill_ratio": round(fill_ratio, 4),
        "is_empty": is_empty,
        "is_small": is_small,
        "is_too_short": is_too_short,
        "is_small_edge_fragment": is_small_edge_fragment,
        "touches_edge": touches_edge,
        "suspected_merged": suspected_merged,
        "accepted_for_deva": accepted,
    }


# ---------------------------------------------------------------------------
# visualization (mask-only, no bbox)
# ---------------------------------------------------------------------------

def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (55, 200, 90), (65, 140, 255), (255, 170, 50),
        (220, 90, 200), (70, 220, 220), (255, 110, 110),
    ]
    return palette[(index - 1) % len(palette)]


def draw_hypotheses(image: np.ndarray, hypotheses: list[dict], display_threshold: float) -> np.ndarray:
    """Draw mask-only visualization (no bbox rectangles)."""
    overlay = image.copy()
    vis = image.copy()

    for hyp in hypotheses:
        color = color_for_index(hyp["source_instance_id"])
        mask = hyp["mask"]
        alpha = 0.60 if hyp["score"] >= display_threshold else 0.25
        overlay[mask] = (overlay[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)

    vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0.0)

    for hyp in hypotheses:
        color = color_for_index(hyp["source_instance_id"])
        mask_u8 = hyp["mask"].astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

        bbox = hyp["bbox"]
        label = f"#{hyp['source_instance_id']} {hyp['score']:.3f}"
        if hyp.get("suspected_merged"):
            label = f"{label} [!merged]"
        text_origin = (bbox[0], max(18, bbox[1] - 6))
        if hyp["score"] >= display_threshold:
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    return vis


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MaskDINO instance masks as DEVA-compatible frame hypotheses"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Trained MaskDINO checkpoint path")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE),
                        help="MaskDINO config path")
    parser.add_argument("--video-id", help="Video id; defaults to video stem")
    parser.add_argument("--job-dir", help="Job output directory; defaults to output/jobs/<video_id>")

    # Dual FPS
    parser.add_argument("--deva-fps", type=float, default=6.0,
                        help="FPS for DEVA frame sequence")
    parser.add_argument("--maskdino-fps", type=float, default=3.0,
                        help="FPS for MaskDINO injection (<= deva-fps)")

    # Thresholds
    parser.add_argument("--maskdino-score-threshold", type=float, default=0.50,
                        help="Minimum score to inject a mask hypothesis into DEVA")
    parser.add_argument("--display-score-threshold", type=float, default=0.50,
                        help="Minimum score for prominent visualization")

    # ROI
    parser.add_argument("--select-roi", action="store_true",
                        help="Interactively select a target shelf ROI")
    parser.add_argument("--roi-json", help="Path to normalized ROI json")
    parser.add_argument("--roi-source-frame", choices=("first", "middle"), default="middle")
    parser.add_argument("--roi-candidate-count", type=int, default=0)
    parser.add_argument("--roi-candidate-index", type=int)
    parser.add_argument("--roi-candidate-dir")

    # Output control
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--save-debug-masks", action="store_true",
                        help="Also save per-instance PNG masks for debugging")
    args = parser.parse_args()

    if args.deva_fps <= 0 or args.maskdino_fps <= 0:
        raise SystemExit("--deva-fps and --maskdino-fps must be > 0")
    if args.maskdino_fps > args.deva_fps:
        raise SystemExit("--maskdino-fps must be <= --deva-fps")
    if args.maskdino_score_threshold <= 0:
        raise SystemExit("--maskdino-score-threshold must be > 0")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    video_id = args.video_id or video_path.stem
    job_dir = Path(args.job_dir) if args.job_dir else Path("output") / "jobs" / video_id

    deva_frames_dir = job_dir / "deva_frames"
    hypotheses_dir = job_dir / "maskdino_hypotheses"
    debug_masks_dir = job_dir / "maskdino_masks_debug"
    vis_dir = job_dir / "visualizations"
    roi_json_path = Path(args.roi_json) if args.roi_json else job_dir / "roi.json"
    roi_candidate_dir = Path(args.roi_candidate_dir) if args.roi_candidate_dir else job_dir / "roi_candidates"
    frame_manifest_csv = job_dir / "frame_manifest.csv"
    instances_csv = job_dir / "maskdino_instances.csv"
    hyp_video_path = job_dir / "maskdino_hypotheses.mp4"

    for path in (job_dir, deva_frames_dir, hypotheses_dir, vis_dir):
        ensure_dir(path)
    if args.save_debug_masks:
        ensure_dir(debug_masks_dir)

    # --- ROI ---
    normalized_roi = None
    if args.select_roi:
        if args.roi_candidate_count > 0:
            candidates = export_roi_candidate_frames(video_path, roi_candidate_dir, args.roi_candidate_count)
            chosen_candidate = choose_roi_candidate(candidates, args.roi_candidate_index)
            print(f"selected candidate -> index={chosen_candidate['candidate_index']} "
                  f"frame={chosen_candidate['frame_index']:06d}")
            reference_frame = read_frame_at_index(video_path, chosen_candidate["frame_index"])
        else:
            reference_frame = read_reference_frame(video_path, args.roi_source_frame)
        normalized_roi = select_roi_interactively(reference_frame)
        save_roi_json(roi_json_path, normalized_roi)
        print(f"roi json          -> {roi_json_path}")
    elif args.roi_json and not roi_json_path.exists():
        raise SystemExit(f"ROI json not found: {roi_json_path}")
    elif roi_json_path.exists():
        normalized_roi = load_roi_json(roi_json_path)
        print(f"using roi json    -> {roi_json_path}")

    # --- Predictor ---
    predictor = build_predictor(
        weights=args.weights,
        config_file=args.config_file,
        score_threshold=args.maskdino_score_threshold,
    )

    # --- CSV writers ---
    manifest_f = frame_manifest_csv.open("w", encoding="utf-8-sig", newline="")
    manifest_writer = csv.writer(manifest_f)
    manifest_writer.writerow([
        "video_id", "deva_frame_index", "source_frame_index", "time_sec",
        "frame_path", "has_maskdino_hypotheses",
    ])

    inst_f = instances_csv.open("w", encoding="utf-8-sig", newline="")
    inst_writer = csv.writer(inst_f)
    inst_writer.writerow([
        "video_id", "deva_frame_index", "source_frame_index", "time_sec",
        "source_instance_id", "score",
        "mask_path", "mask_area", "mask_width", "mask_height",
        "mask_fill_ratio", "suspected_merged_mask", "accepted_for_deva",
    ])

    # --- Video writer ---
    hyp_writer = None
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)

    # --- Process ---
    deva_frame_count = 0
    injection_count = 0
    total_hypotheses = 0
    filtered_count = 0
    suspected_merged_count = 0
    injection_step = max(1, int(round(args.deva_fps / args.maskdino_fps))) if args.maskdino_fps < args.deva_fps else 1

    print(f"deva_fps={args.deva_fps:.1f} maskdino_fps={args.maskdino_fps:.1f} "
          f"injection_every={injection_step} deva_frame(s)")

    roi_bbox_abs = None  # absolute coords of ROI in full frame

    for deva_frame_index, source_index, time_sec, frame in iter_deva_frames(video_path, args.deva_fps):
        # Compute ROI once from first frame
        if roi_bbox_abs is None and normalized_roi is not None:
            roi_bbox_abs = denormalize_roi_xyxy(normalized_roi, frame.shape)

        # Crop to ROI
        if roi_bbox_abs is not None:
            roi_frame = frame[roi_bbox_abs[1]:roi_bbox_abs[3], roi_bbox_abs[0]:roi_bbox_abs[2]]
        else:
            roi_frame = frame

        roi_h, roi_w = roi_frame.shape[:2]

        # Save DEVA frame
        deva_frame_name = f"{deva_frame_index:06d}.jpg"
        deva_frame_path = deva_frames_dir / deva_frame_name
        cv2.imwrite(str(deva_frame_path), roi_frame)

        # Determine injection
        do_inject = should_run_maskdino(deva_frame_index, args.deva_fps, args.maskdino_fps)

        frame_hypotheses = []

        if do_inject:
            # --- Run MaskDINO ---
            outputs = predictor(roi_frame)
            instances = outputs["instances"].to("cpu")

            if not instances.has("pred_masks"):
                raise RuntimeError(
                    "MaskDINO output does not contain pred_masks; "
                    "DEVA integration requires real instance masks."
                )

            scores = instances.scores.tolist() if instances.has("scores") else []
            masks_all = instances.pred_masks.numpy() if instances.has("pred_masks") else None

            if masks_all is None or len(scores) == 0:
                print(f"deva_frame={deva_frame_index:06d} injection but no instances found")
            else:
                # Collect all binary masks for width-median analysis
                all_binary_masks = [masks_all[i].astype(bool) for i in range(len(scores))]

                for raw_idx, score in enumerate(scores):
                    if float(score) < args.maskdino_score_threshold:
                        continue

                    mask = masks_all[raw_idx].astype(bool)
                    quality = analyze_mask(mask, roi_h, roi_w, all_binary_masks)

                    if quality["is_empty"]:
                        continue

                    if not quality["accepted_for_deva"]:
                        filtered_count += 1

                    if quality["suspected_merged"]:
                        suspected_merged_count += 1

                    source_instance_id = raw_idx + 1
                    bbox = mask_to_bbox(mask)

                    frame_hypotheses.append({
                        "source_instance_id": source_instance_id,
                        "score": float(score),
                        "mask": mask,
                        "bbox": bbox,
                        "quality": quality,
                    })

            injection_count += 1

        # Save hypotheses NPZ (even if empty — DEVA needs to know)
        npz_path = hypotheses_dir / f"{deva_frame_index:06d}.npz"
        if frame_hypotheses:
            masks_stack = np.stack([h["mask"] for h in frame_hypotheses], axis=0)
            np.savez_compressed(
                npz_path,
                masks=masks_stack,
                scores=np.array([h["score"] for h in frame_hypotheses], dtype=np.float32),
                source_instance_ids=np.array([h["source_instance_id"] for h in frame_hypotheses], dtype=np.int32),
            )

            total_hypotheses += len(frame_hypotheses)

            # Optional debug PNGs
            if args.save_debug_masks:
                for h in frame_hypotheses:
                    png_name = f"{deva_frame_index:06d}_{h['source_instance_id']:03d}.png"
                    cv2.imwrite(str(debug_masks_dir / png_name), h["mask"].astype(np.uint8) * 255)

            # Write instance CSV rows
            for h in frame_hypotheses:
                q = h["quality"]
                inst_writer.writerow([
                    video_id, deva_frame_index, source_index, f"{time_sec:.2f}",
                    h["source_instance_id"], f"{h['score']:.4f}",
                    relative_posix(npz_path, job_dir),
                    q["area"], q["width"], q["height"],
                    q["fill_ratio"],
                    int(q["suspected_merged"]),
                    int(q["accepted_for_deva"]),
                ])
        else:
            # Empty NPZ for propagation-only frames
            np.savez_compressed(
                npz_path,
                masks=np.zeros((0, roi_h, roi_w), dtype=bool),
                scores=np.array([], dtype=np.float32),
                source_instance_ids=np.array([], dtype=np.int32),
            )

        # Write manifest
        rel_frame_path = relative_posix(deva_frame_path, job_dir)
        manifest_writer.writerow([
            video_id, deva_frame_index, source_index, f"{time_sec:.2f}",
            rel_frame_path, int(do_inject),
        ])

        # Visualization (on ROI frame)
        vis_frame = draw_hypotheses(roi_frame, frame_hypotheses, args.display_score_threshold)
        vis_frame = draw_roi_outline(vis_frame, None)  # ROI already cropped

        # HUD
        hud_lines = [
            f"deva_frame: {deva_frame_index:06d}",
            f"source_frame: {source_index}",
            f"time: {time_sec:.2f}s",
            f"injection: {'yes' if do_inject else 'no'}",
            f"hypotheses: {len(frame_hypotheses)}",
        ]
        y = 24
        for line in hud_lines:
            cv2.putText(vis_frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(vis_frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 22

        # Save visualization frame
        vis_name = f"{deva_frame_index:06d}.jpg"
        cv2.imwrite(str(vis_dir / vis_name), vis_frame)

        # Video
        if hyp_writer is None:
            vh, vw = vis_frame.shape[:2]
            hyp_writer = cv2.VideoWriter(str(hyp_video_path), fourcc, args.deva_fps, (vw, vh))
            if not hyp_writer.isOpened():
                raise SystemExit(f"Failed to create video: {hyp_video_path}")
        hyp_writer.write(vis_frame)

        deva_frame_count += 1

        status = "inject" if do_inject else "propagate"
        print(f"[{deva_frame_count}] deva_frame={deva_frame_index:06d} "
              f"source_frame={source_index:06d} time={time_sec:.2f}s "
              f"{status} hypots={len(frame_hypotheses)}")

    # --- Cleanup ---
    manifest_f.close()
    inst_f.close()
    if hyp_writer is not None:
        hyp_writer.release()

    print(f"\njob dir              -> {job_dir}")
    print(f"deva frames dir      -> {deva_frames_dir}")
    print(f"hypotheses dir       -> {hypotheses_dir}")
    print(f"manifest csv         -> {frame_manifest_csv}")
    print(f"instances csv        -> {instances_csv}")
    print(f"hypotheses video     -> {hyp_video_path}")
    if normalized_roi is not None:
        print(f"roi json             -> {roi_json_path}")
    if args.save_debug_masks:
        print(f"debug masks dir      -> {debug_masks_dir}")
    print(f"DEVA frames total    : {deva_frame_count}")
    print(f"MaskDINO injections  : {injection_count}")
    print(f"total hypotheses     : {total_hypotheses}")
    print(f"filtered (quality)   : {filtered_count}")
    print(f"suspected_merged     : {suspected_merged_count}")


if __name__ == "__main__":
    main()
