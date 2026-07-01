"""Track book spines across sampled frames with official BoT-SORT.

Inputs:
    output/jobs/<video_id>/
      frames/
      detections.csv
      optional roi.json

Outputs:
    output/jobs/<video_id>/
      tracked_video.mp4
      track_observations.csv
      unique_books.csv
      tracking_summary.json
      books/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOTSORT_ROOT = PROJECT_ROOT / "third_party" / "BoT-SORT"
if str(BOTSORT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOTSORT_ROOT))

from tracker.basetrack import TrackState  # type: ignore  # noqa: E402
from tracker.bot_sort import BoTSORT  # type: ignore  # noqa: E402


@dataclass
class DetectionRow:
    detection_id: int
    video_id: str
    frame_index: int
    time_sec: float
    instance_id: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask_path: str
    crop_path: str
    mask_abs_path: Path
    crop_abs_path: Path

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def bbox_area(self) -> int:
        return self.width * self.height


@dataclass
class Observation:
    video_id: str
    track_id: int
    book_id: str
    frame_index: int
    time_sec: float
    source_detection_id: int
    source_instance_id: int
    detection_score: float
    tracked_x1: int
    tracked_y1: int
    tracked_x2: int
    tracked_y2: int
    original_x1: int
    original_y1: int
    original_x2: int
    original_y2: int
    mask_path: str
    crop_path: str
    mask_abs_path: Path
    crop_abs_path: Path
    touches_left_edge: bool
    touches_right_edge: bool
    sharpness: float
    crop_area: int
    mask_fill_ratio: float
    normalized_sharpness: float = 0.0
    normalized_area: float = 0.0
    normalized_height: float = 0.0
    completeness_score: float = 0.0
    edge_penalty: float = 0.0
    quality_score: float = 0.0
    valid: bool = True

    @property
    def original_width(self) -> int:
        return max(0, self.original_x2 - self.original_x1)

    @property
    def original_height(self) -> int:
        return max(0, self.original_y2 - self.original_y1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def frame_index_from_path(path: Path) -> int:
    return int(path.stem)


def list_frame_paths(frames_dir: Path) -> dict[int, Path]:
    frame_paths = {frame_index_from_path(path): path for path in sorted(frames_dir.glob("*.jpg"))}
    if not frame_paths:
        raise SystemExit(f"No frame images found in {frames_dir}")
    return frame_paths


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def load_roi_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    roi = json.loads(path.read_text(encoding="utf-8"))
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(roi):
        raise SystemExit(f"ROI json missing keys: {sorted(required - set(roi))}")
    if not (0.0 <= float(roi["x1"]) < float(roi["x2"]) <= 1.0):
        raise SystemExit(f"Invalid ROI x range in {path}")
    if not (0.0 <= float(roi["y1"]) < float(roi["y2"]) <= 1.0):
        raise SystemExit(f"Invalid ROI y range in {path}")
    return roi


def denormalize_roi_xyxy(roi: dict, frame_shape: tuple[int, int, int]) -> list[int]:
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


def load_detections(csv_path: Path, job_dir: Path) -> tuple[str, dict[int, list[DetectionRow]], dict[int, DetectionRow]]:
    required_columns = {
        "video_id",
        "frame_index",
        "time_sec",
        "instance_id",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "mask_path",
        "crop_path",
    }
    rows_by_frame: dict[int, list[DetectionRow]] = defaultdict(list)
    rows_by_detection_id: dict[int, DetectionRow] = {}
    video_id = None
    detection_id = 1

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"detections.csv has no header: {csv_path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"detections.csv missing columns: {sorted(missing)}")
        for row in reader:
            video_id = row["video_id"] if video_id is None else video_id
            det = DetectionRow(
                detection_id=detection_id,
                video_id=row["video_id"],
                frame_index=int(row["frame_index"]),
                time_sec=float(row["time_sec"]),
                instance_id=int(row["instance_id"]),
                score=float(row["score"]),
                x1=int(float(row["x1"])),
                y1=int(float(row["y1"])),
                x2=int(float(row["x2"])),
                y2=int(float(row["y2"])),
                mask_path=row["mask_path"],
                crop_path=row["crop_path"],
                mask_abs_path=job_dir / row["mask_path"],
                crop_abs_path=job_dir / row["crop_path"],
            )
            rows_by_frame[det.frame_index].append(det)
            rows_by_detection_id[detection_id] = det
            detection_id += 1

    if video_id is None:
        raise SystemExit(f"No detections found in {csv_path}")

    for frame_rows in rows_by_frame.values():
        frame_rows.sort(key=lambda row: (row.x1, row.instance_id))

    return video_id, rows_by_frame, rows_by_detection_id


def infer_sample_fps(detections_by_frame: dict[int, list[DetectionRow]]) -> float:
    pairs = []
    ordered = sorted((frame_index, rows[0].time_sec) for frame_index, rows in detections_by_frame.items() if rows)
    for (frame_a, time_a), (frame_b, time_b) in zip(ordered, ordered[1:]):
        if frame_b <= frame_a or time_b <= time_a:
            continue
        pairs.append((frame_b - frame_a) / (time_b - time_a))
    if not pairs:
        return 3.0
    return float(np.median(pairs))


def make_tracker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        proximity_thresh=args.proximity_thresh,
        appearance_thresh=args.appearance_thresh,
        with_reid=args.with_reid,
        fast_reid_config="",
        fast_reid_weights="",
        device="cpu",
        cmc_method=args.cmc_method,
        name="book_spine",
        ablation=False,
        mot20=args.mot20,
    )


def compute_crop_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_32F).var())


def compute_mask_fill_ratio(mask_path: Path, bbox_area: int) -> float:
    if bbox_area <= 0:
        return 0.0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(bbox_area)


def clip_bbox(box: np.ndarray | list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def status_color(status: str) -> tuple[int, int, int]:
    if status == "confirmed":
        return (55, 200, 90)
    if status == "boundary_review":
        return (255, 180, 60)
    if status == "singleton_review":
        return (70, 160, 255)
    return (255, 110, 110)


def track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (55, 200, 90),
        (65, 140, 255),
        (255, 170, 50),
        (220, 90, 200),
        (70, 220, 220),
        (255, 110, 110),
        (120, 210, 255),
        (255, 210, 120),
    ]
    return palette[(track_id - 1) % len(palette)]


def overlay_mask(image: np.ndarray, mask_path: Path, color: tuple[int, int, int], cache: dict[Path, np.ndarray]) -> None:
    if mask_path not in cache:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        cache[mask_path] = mask if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cache[mask_path] > 0
    if np.any(mask):
        image[mask] = (image[mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)


def normalize_quality_scores(observations: list[Observation]) -> None:
    sharpness_values = [obs.sharpness for obs in observations if obs.sharpness > 0]
    area_values = [float(obs.crop_area) for obs in observations if obs.crop_area > 0]
    height_values = [float(obs.original_height) for obs in observations if obs.original_height > 0]

    sharpness_base = float(np.percentile(sharpness_values, 95)) if sharpness_values else 1.0
    area_base = float(np.percentile(area_values, 95)) if area_values else 1.0
    height_base = float(np.percentile(height_values, 95)) if height_values else 1.0

    for obs in observations:
        obs.normalized_sharpness = min(1.0, obs.sharpness / max(sharpness_base, 1.0))
        obs.normalized_area = min(1.0, obs.crop_area / max(area_base, 1.0))
        obs.normalized_height = min(1.0, obs.original_height / max(height_base, 1.0))
        edge_touch = obs.touches_left_edge or obs.touches_right_edge
        obs.edge_penalty = 0.15 if edge_touch else 0.0
        obs.completeness_score = 0.5 * (0.0 if edge_touch else 1.0) + 0.5 * obs.normalized_height
        obs.quality_score = (
            0.35 * obs.normalized_sharpness
            + 0.25 * obs.detection_score
            + 0.15 * obs.normalized_area
            + 0.15 * obs.mask_fill_ratio
            + 0.10 * obs.completeness_score
            - obs.edge_penalty
        )


def determine_track_status(observations: list[Observation]) -> tuple[str, int]:
    valid_observations = [obs for obs in observations if obs.valid]
    if not valid_observations:
        return "invalid", 0
    if len(valid_observations) >= 2:
        if any(not obs.touches_left_edge and not obs.touches_right_edge for obs in valid_observations):
            return "confirmed", 1
        return "boundary_review", 0
    if len(valid_observations) == 1:
        return "singleton_review", 0
    return "invalid", 0


def write_track_observations_csv(path: Path, observations: list[Observation]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "book_id",
                "video_id",
                "track_id",
                "frame_index",
                "time_sec",
                "source_detection_id",
                "source_instance_id",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
                "mask_path",
                "crop_path",
                "quality_score",
                "touches_left_edge",
                "touches_right_edge",
                "sharpness",
                "crop_area",
                "mask_fill_ratio",
            ]
        )
        for obs in sorted(observations, key=lambda row: (row.frame_index, row.track_id, row.source_detection_id)):
            writer.writerow(
                [
                    obs.book_id,
                    obs.video_id,
                    obs.track_id,
                    obs.frame_index,
                    f"{obs.time_sec:.2f}",
                    obs.source_detection_id,
                    obs.source_instance_id,
                    f"{obs.detection_score:.4f}",
                    obs.tracked_x1,
                    obs.tracked_y1,
                    obs.tracked_x2,
                    obs.tracked_y2,
                    obs.mask_path,
                    obs.crop_path,
                    f"{obs.quality_score:.4f}",
                    int(obs.touches_left_edge),
                    int(obs.touches_right_edge),
                    f"{obs.sharpness:.4f}",
                    obs.crop_area,
                    f"{obs.mask_fill_ratio:.4f}",
                ]
            )


def write_unique_books_csv(
    path: Path,
    tracks: dict[int, list[Observation]],
    statuses: dict[int, tuple[str, int]],
    best_observations: dict[int, Observation],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "book_id",
                "video_id",
                "track_id",
                "first_frame",
                "last_frame",
                "first_time_sec",
                "last_time_sec",
                "num_observations",
                "status",
                "count_as_book",
                "best_crop_path",
                "best_quality_score",
                "best_detection_score",
                "mean_width",
                "mean_height",
            ]
        )
        for track_id in sorted(tracks):
            observations = [obs for obs in tracks[track_id] if obs.valid]
            if observations:
                observations.sort(key=lambda row: (row.frame_index, row.source_detection_id))
                best = best_observations[track_id]
                status, count_as_book = statuses[track_id]
                mean_width = float(np.mean([obs.original_width for obs in observations]))
                mean_height = float(np.mean([obs.original_height for obs in observations]))
                writer.writerow(
                    [
                        observations[0].book_id,
                        observations[0].video_id,
                        track_id,
                        observations[0].frame_index,
                        observations[-1].frame_index,
                        f"{observations[0].time_sec:.2f}",
                        f"{observations[-1].time_sec:.2f}",
                        len(observations),
                        status,
                        count_as_book,
                        best.crop_path,
                        f"{best.quality_score:.4f}",
                        f"{best.detection_score:.4f}",
                        f"{mean_width:.4f}",
                        f"{mean_height:.4f}",
                    ]
                )
            else:
                writer.writerow(
                    [
                        f"unknown_book_{track_id:04d}",
                        "",
                        track_id,
                        "",
                        "",
                        "",
                        "",
                        0,
                        "invalid",
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )


def write_tracking_summary(
    path: Path,
    video_id: str,
    sample_fps: float,
    frame_count: int,
    detections_total: int,
    tracks_total: int,
    statuses: dict[int, tuple[str, int]],
    cmc_method: str,
) -> None:
    summary = {
        "video_id": video_id,
        "sample_fps": sample_fps,
        "frames_processed": frame_count,
        "detections_total": detections_total,
        "tracks_total": tracks_total,
        "confirmed_books": sum(1 for status, _ in statuses.values() if status == "confirmed"),
        "boundary_review": sum(1 for status, _ in statuses.values() if status == "boundary_review"),
        "singleton_review": sum(1 for status, _ in statuses.values() if status == "singleton_review"),
        "invalid_tracks": sum(1 for status, _ in statuses.values() if status == "invalid"),
        "cmc_method": cmc_method,
        "tracker": "BoT-SORT",
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tracked_video(
    path: Path,
    frame_paths: dict[int, Path],
    detections_by_frame: dict[int, list[DetectionRow]],
    observations_by_frame: dict[int, list[Observation]],
    statuses: dict[int, tuple[str, int]],
    roi_bbox: list[int] | None,
    sample_fps: float,
    cmc_method: str,
) -> None:
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_cache: dict[Path, np.ndarray] = {}

    track_first_frame: dict[int, int] = {}
    for rows in observations_by_frame.values():
        for obs in rows:
            if not obs.valid:
                continue
            track_first_frame[obs.track_id] = min(track_first_frame.get(obs.track_id, obs.frame_index), obs.frame_index)

    for frame_index in sorted(frame_paths):
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            raise SystemExit(f"Failed to read frame image: {frame_paths[frame_index]}")

        vis = frame.copy()
        rows = sorted(observations_by_frame.get(frame_index, []), key=lambda row: (row.tracked_x1, row.track_id))
        for obs in rows:
            status, _ = statuses[obs.track_id]
            color = track_color(obs.track_id)
            overlay_mask(vis, obs.mask_abs_path, color, mask_cache)
            cv2.rectangle(vis, (obs.tracked_x1, obs.tracked_y1), (obs.tracked_x2, obs.tracked_y2), color, 2)
            label = f"{obs.book_id} t{obs.track_id:04d}"
            if status != "confirmed":
                label = f"{label} {status}"
            text_origin = (obs.tracked_x1, max(18, obs.tracked_y1 - 6))
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color(status), 1)

        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 215, 255), 2)

        active_track_ids = {obs.track_id for obs in rows}
        started_track_ids = {
            track_id
            for track_id, first_frame in track_first_frame.items()
            if first_frame <= frame_index
        }
        confirmed_so_far = sum(
            1
            for track_id in started_track_ids
            if statuses.get(track_id, ("invalid", 0))[0] == "confirmed"
        )
        review_so_far = sum(
            1
            for track_id in started_track_ids
            if statuses.get(track_id, ("invalid", 0))[0] in {"boundary_review", "singleton_review"}
        )
        overlay_lines = [
            f"frame: {frame_index:06d}",
            f"detections: {len(detections_by_frame.get(frame_index, []))}",
            f"active tracks: {len(active_track_ids)}",
            f"confirmed: {confirmed_so_far}",
            f"review: {review_so_far}",
            f"gmc: {cmc_method}",
            f"unique books: {len(started_track_ids)}",
        ]
        y = 28
        for line in overlay_lines:
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)
            y += 28

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(path), fourcc, sample_fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f"Failed to create tracked video: {path}")
        writer.write(vis)

    if writer is not None:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Track sampled book-spine detections with official BoT-SORT")
    parser.add_argument("--job-dir", required=True, help="Job directory such as output/jobs/000")
    parser.add_argument("--sample-fps", type=float, help="Sample FPS used to generate frames; inferred from detections.csv when omitted")
    parser.add_argument("--cmc-method", choices=("orb", "ecc", "sparseOptFlow", "none"), default="orb", help="BoT-SORT global motion compensation method")
    parser.add_argument("--track-high-thresh", type=float, default=0.50, help="High-confidence association threshold")
    parser.add_argument("--track-low-thresh", type=float, default=0.10, help="Low-confidence association threshold")
    parser.add_argument("--new-track-thresh", type=float, default=0.60, help="Minimum score to initialize a new track")
    parser.add_argument("--track-buffer", type=int, default=30, help="Keep lost tracks for this many sampled frames")
    parser.add_argument("--match-thresh", type=float, default=0.80, help="Association threshold passed to BoT-SORT")
    parser.add_argument("--proximity-thresh", type=float, default=0.50, help="IoU-based proximity threshold")
    parser.add_argument("--appearance-thresh", type=float, default=0.25, help="Appearance threshold; inactive when --with-reid is not set")
    parser.add_argument("--with-reid", action="store_true", help="Enable BoT-SORT ReID branch")
    parser.add_argument("--mot20", action="store_true", help="Use MOT20 mode in BoT-SORT")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    frames_dir = job_dir / "frames"
    detections_csv = job_dir / "detections.csv"
    roi_json_path = job_dir / "roi.json"
    tracked_video_path = job_dir / "tracked_video.mp4"
    observations_csv_path = job_dir / "track_observations.csv"
    unique_books_csv_path = job_dir / "unique_books.csv"
    summary_json_path = job_dir / "tracking_summary.json"
    books_dir = job_dir / "books"

    if not detections_csv.exists():
        raise SystemExit(f"detections.csv not found: {detections_csv}")
    if not frames_dir.exists():
        raise SystemExit(f"frames directory not found: {frames_dir}")

    ensure_dir(books_dir)
    frame_paths = list_frame_paths(frames_dir)
    first_frame = cv2.imread(str(frame_paths[min(frame_paths)]))
    if first_frame is None:
        raise SystemExit(f"Failed to read frame image: {frame_paths[min(frame_paths)]}")
    frame_height, frame_width = first_frame.shape[:2]

    roi_json = load_roi_json(roi_json_path)
    roi_bbox = denormalize_roi_xyxy(roi_json, first_frame.shape) if roi_json is not None else None
    video_id, detections_by_frame, detections_by_id = load_detections(detections_csv, job_dir)
    sample_fps = float(args.sample_fps) if args.sample_fps is not None else infer_sample_fps(detections_by_frame)
    if sample_fps <= 0:
        raise SystemExit("--sample-fps must be > 0")

    tracker = BoTSORT(make_tracker_args(args), frame_rate=sample_fps)
    print(f"sample_fps          : {sample_fps:.2f}")
    print(f"track_buffer arg    : {args.track_buffer}")
    print(f"effective max_time_lost: {tracker.max_time_lost} frames ≈ {tracker.max_time_lost / sample_fps:.1f}s")
    all_observations: list[Observation] = []
    observations_by_track: dict[int, list[Observation]] = defaultdict(list)
    observations_by_frame: dict[int, list[Observation]] = defaultdict(list)

    left_bound = roi_bbox[0] if roi_bbox is not None else 0
    right_bound = roi_bbox[2] if roi_bbox is not None else frame_width
    edge_margin = max(8.0, 0.01 * (right_bound - left_bound))

    for frame_index in sorted(frame_paths):
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            raise SystemExit(f"Failed to read frame image: {frame_paths[frame_index]}")

        tracker_frame = frame
        roi_offset_x = 0
        roi_offset_y = 0
        if roi_bbox is not None:
            roi_offset_x, roi_offset_y = roi_bbox[0], roi_bbox[1]
            tracker_frame = frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]]

        frame_rows = detections_by_frame.get(frame_index, [])
        det_matrix = []
        source_detection_ids = []
        source_instance_ids = []
        valid_rows: list[DetectionRow] = []
        for row in frame_rows:
            x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
            if roi_bbox is not None:
                x1 = max(roi_bbox[0], min(roi_bbox[2] - 1, x1)) - roi_offset_x
                y1 = max(roi_bbox[1], min(roi_bbox[3] - 1, y1)) - roi_offset_y
                x2 = max(roi_bbox[0] + 1, min(roi_bbox[2], x2)) - roi_offset_x
                y2 = max(roi_bbox[1] + 1, min(roi_bbox[3], y2)) - roi_offset_y
            if x2 <= x1 or y2 <= y1:
                continue
            det_matrix.append([float(x1), float(y1), float(x2), float(y2), float(row.score)])
            source_detection_ids.append(row.detection_id)
            source_instance_ids.append(row.instance_id)
            valid_rows.append(row)

        detections_xyxy_score = np.asarray(det_matrix, dtype=np.float64).reshape(-1, 5) if det_matrix else np.empty((0, 5), dtype=np.float64)
        online_targets = tracker.update(
            output_results=detections_xyxy_score,
            img=tracker_frame,
            source_detection_ids=np.asarray(source_detection_ids, dtype=np.int64) if source_detection_ids else None,
            source_instance_ids=np.asarray(source_instance_ids, dtype=np.int64) if source_instance_ids else None,
        )

        current_observations = []
        for target in online_targets:
            if target.state != TrackState.Tracked or target.frame_id != tracker.frame_id:
                continue
            if getattr(target, "source_detection_id", None) is None:
                continue

            source_detection_id = int(target.source_detection_id)
            if source_detection_id not in detections_by_id:
                continue
            source_row = detections_by_id[source_detection_id]
            tracked_x1, tracked_y1, tracked_x2, tracked_y2 = clip_bbox(
                [
                    target.tlbr[0] + roi_offset_x,
                    target.tlbr[1] + roi_offset_y,
                    target.tlbr[2] + roi_offset_x,
                    target.tlbr[3] + roi_offset_y,
                ],
                frame_width,
                frame_height,
            )

            touches_left = source_row.x1 <= left_bound + edge_margin
            touches_right = source_row.x2 >= right_bound - edge_margin
            sharpness = compute_crop_sharpness(source_row.crop_abs_path)
            crop_area = 0
            crop_image = cv2.imread(str(source_row.crop_abs_path))
            if crop_image is not None:
                crop_area = int(crop_image.shape[0] * crop_image.shape[1])
            mask_fill_ratio = compute_mask_fill_ratio(source_row.mask_abs_path, source_row.bbox_area)
            valid = (
                source_row.x2 > source_row.x1
                and source_row.y2 > source_row.y1
                and source_row.bbox_area >= 32
                and source_row.crop_abs_path.exists()
                and source_row.mask_abs_path.exists()
            )
            book_id = f"{video_id}_book_{int(target.track_id):04d}"
            observation = Observation(
                video_id=video_id,
                track_id=int(target.track_id),
                book_id=book_id,
                frame_index=frame_index,
                time_sec=source_row.time_sec,
                source_detection_id=source_detection_id,
                source_instance_id=source_row.instance_id,
                detection_score=source_row.score,
                tracked_x1=tracked_x1,
                tracked_y1=tracked_y1,
                tracked_x2=tracked_x2,
                tracked_y2=tracked_y2,
                original_x1=source_row.x1,
                original_y1=source_row.y1,
                original_x2=source_row.x2,
                original_y2=source_row.y2,
                mask_path=source_row.mask_path,
                crop_path=source_row.crop_path,
                mask_abs_path=source_row.mask_abs_path,
                crop_abs_path=source_row.crop_abs_path,
                touches_left_edge=touches_left,
                touches_right_edge=touches_right,
                sharpness=sharpness,
                crop_area=crop_area,
                mask_fill_ratio=mask_fill_ratio,
                valid=valid,
            )
            current_observations.append(observation)
            observations_by_track[observation.track_id].append(observation)
            observations_by_frame[frame_index].append(observation)
            all_observations.append(observation)

        print(
            f"frame={frame_index:06d} detections={len(valid_rows)} "
            f"tracked={len(current_observations)} total_tracks={len(observations_by_track)}"
        )

    normalize_quality_scores(all_observations)
    statuses = {track_id: determine_track_status(rows) for track_id, rows in observations_by_track.items()}
    best_observations: dict[int, Observation] = {}
    for track_id, rows in observations_by_track.items():
        valid_rows = [row for row in rows if row.valid]
        if not valid_rows:
            continue
        best = max(valid_rows, key=lambda row: (row.quality_score, row.detection_score, row.crop_area))
        best_observations[track_id] = best
        shutil.copyfile(best.crop_abs_path, books_dir / f"{best.book_id}.jpg")

    write_track_observations_csv(observations_csv_path, all_observations)
    write_unique_books_csv(unique_books_csv_path, observations_by_track, statuses, best_observations)
    write_tracking_summary(
        summary_json_path,
        video_id=video_id,
        sample_fps=sample_fps,
        frame_count=len(frame_paths),
        detections_total=sum(len(rows) for rows in detections_by_frame.values()),
        tracks_total=len(observations_by_track),
        statuses=statuses,
        cmc_method=args.cmc_method,
    )
    write_tracked_video(
        tracked_video_path,
        frame_paths=frame_paths,
        detections_by_frame=detections_by_frame,
        observations_by_frame=observations_by_frame,
        statuses=statuses,
        roi_bbox=roi_bbox,
        sample_fps=sample_fps,
        cmc_method=args.cmc_method,
    )

    confirmed_count = sum(1 for status, _ in statuses.values() if status == "confirmed")
    boundary_count = sum(1 for status, _ in statuses.values() if status == "boundary_review")
    singleton_count = sum(1 for status, _ in statuses.values() if status == "singleton_review")
    invalid_count = sum(1 for status, _ in statuses.values() if status == "invalid")

    print(f"job dir            -> {job_dir}")
    print(f"tracked video      -> {tracked_video_path}")
    print(f"observations csv   -> {observations_csv_path}")
    print(f"unique books csv   -> {unique_books_csv_path}")
    print(f"tracking summary   -> {summary_json_path}")
    print(f"books dir          -> {books_dir}")
    print(f"frames processed   : {len(frame_paths)}")
    print(f"detections total   : {sum(len(rows) for rows in detections_by_frame.values())}")
    print(f"tracks total       : {len(observations_by_track)}")
    print(f"confirmed          : {confirmed_count}")
    print(f"boundary_review    : {boundary_count}")
    print(f"singleton_review   : {singleton_count}")
    print(f"invalid            : {invalid_count}")


if __name__ == "__main__":
    main()
