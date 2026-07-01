"""Track book spines across frames with DEVA mask propagation.

MaskDINO per-frame instance masks → DEVA temporal propagation → stable object_id

Input:
    output/jobs/<video_id>/
      frames/
      masks/
      detections.csv
      optional roi.json

Output:
    output/jobs/<video_id>/
      tracked_mask_video.mp4   — mask-only visualization (no bboxes)
      mask_observations.csv
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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVA_ROOT = PROJECT_ROOT / "third_party" / "Tracking-Anything-with-DEVA"
if str(DEVA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEVA_ROOT))

from deva.inference.inference_core import DEVAInferenceCore  # noqa: E402
from deva.inference.object_info import ObjectInfo  # noqa: E402
from deva.model.network import DEVA  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
class MaskObservation:
    video_id: str
    deva_object_id: int
    book_id: str
    frame_index: int
    time_sec: float
    source_detection_id: int
    source_instance_id: int
    detection_score: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask_path: str
    crop_path: str
    mask_abs_path: Path
    crop_abs_path: Path
    touches_left_edge: bool
    touches_right_edge: bool
    sharpness: float
    crop_area: int
    mask_fill_ratio: float
    bbox_area: int
    # quality scores (populated after normalization)
    normalized_sharpness: float = 0.0
    normalized_area: float = 0.0
    normalized_height: float = 0.0
    completeness_score: float = 0.0
    edge_penalty: float = 0.0
    quality_score: float = 0.0
    valid: bool = True

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def load_roi_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    roi = json.loads(path.read_text(encoding="utf-8"))
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(roi):
        raise SystemExit(f"ROI json missing keys: {sorted(required - set(roi))}")
    return roi


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


def load_detections(csv_path: Path, job_dir: Path) -> tuple[str, dict[int, list[DetectionRow]], dict[int, DetectionRow]]:
    required_columns = {
        "video_id", "frame_index", "time_sec", "instance_id",
        "score", "x1", "y1", "x2", "y2", "mask_path", "crop_path",
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


def compute_sharpness(path: Path) -> float:
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


def normalize_quality_scores(observations: list[MaskObservation]) -> None:
    sharpness_values = [obs.sharpness for obs in observations if obs.sharpness > 0]
    area_values = [float(obs.crop_area) for obs in observations if obs.crop_area > 0]
    height_values = [float(obs.height) for obs in observations if obs.height > 0]

    sharpness_base = float(np.percentile(sharpness_values, 95)) if sharpness_values else 1.0
    area_base = float(np.percentile(area_values, 95)) if area_values else 1.0
    height_base = float(np.percentile(height_values, 95)) if height_values else 1.0

    for obs in observations:
        obs.normalized_sharpness = min(1.0, obs.sharpness / max(sharpness_base, 1.0))
        obs.normalized_area = min(1.0, obs.crop_area / max(area_base, 1.0))
        obs.normalized_height = min(1.0, obs.height / max(height_base, 1.0))
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


def determine_track_status(observations: list[MaskObservation]) -> tuple[str, int]:
    valid_obs = [obs for obs in observations if obs.valid]
    if not valid_obs:
        return "invalid", 0
    if len(valid_obs) >= 2:
        if any(not obs.touches_left_edge and not obs.touches_right_edge for obs in valid_obs):
            return "confirmed", 1
        return "boundary_review", 0
    if len(valid_obs) == 1:
        return "singleton_review", 0
    return "invalid", 0


# ---------------------------------------------------------------------------
# DEVA model loading
# ---------------------------------------------------------------------------

def load_deva_model(weights_path: str, device: str = "cuda") -> tuple[DEVAInferenceCore, dict]:
    config = {
        "key_dim": 64,
        "value_dim": 512,
        "pix_feat_dim": 512,
        "mem_every": 5,
        "enable_long_term": True,
        "enable_long_term_count_usage": False,
        "max_mid_term_frames": 6,
        "min_mid_term_frames": 3,
        "max_long_term_elements": 500,
        "num_prototypes": 64,
        "top_k": 20,
        "chunk_size": 4,
        "max_missed_detection_count": 9,
        "max_num_objects": -1,
        "size": 480,
    }
    network = DEVA(config).to(device).eval()
    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    network.load_weights(model_weights)
    print(f"DEVA model loaded from {weights_path}")

    deva = DEVAInferenceCore(network, config=config)
    deva.enabled_long_id()
    return deva, config


# ---------------------------------------------------------------------------
# Mask composition: merge per-instance binary masks into one ID mask
# ---------------------------------------------------------------------------

def compose_id_mask(
    detections: list[DetectionRow],
    frame_shape: tuple,
    detection_score_threshold: float = 0.0,
) -> tuple[torch.Tensor, list[ObjectInfo], list[DetectionRow]]:
    """Merge per-instance binary masks into a single H*W ID mask.

    Smaller masks are rendered on top of larger ones to handle overlaps.
    Returns (id_mask, segments_info, valid_detections).
    """
    h, w = frame_shape[:2]
    id_mask = torch.zeros((h, w), dtype=torch.int32)

    # Sort by area descending: larger masks first, smaller on top
    dets_with_area = []
    for det in detections:
        if det.score < detection_score_threshold:
            continue
        mask_img = cv2.imread(str(det.mask_abs_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        area = int(np.count_nonzero(mask_img))
        dets_with_area.append((det, mask_img, area))

    dets_with_area.sort(key=lambda x: x[2], reverse=True)

    segments_info: list[ObjectInfo] = []
    valid_detections: list[DetectionRow] = []

    for det, mask_img, area in dets_with_area:
        mask_bool = mask_img > 127
        # Use instance_id as the object ID
        obj_id = det.instance_id
        obj_info = ObjectInfo(id=obj_id, score=det.score)
        segments_info.append(obj_info)
        valid_detections.append(det)
        id_mask[mask_bool] = obj_id

    return id_mask, segments_info, valid_detections


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def object_color(obj_id: int) -> tuple[int, int, int]:
    """Generate a stable color from an object ID."""
    np.random.seed(int(obj_id) % 10007)
    r = int(np.random.randint(60, 220))
    g = int(np.random.randint(60, 220))
    b = int(np.random.randint(60, 220))
    return (r, g, b)


def status_color(status: str) -> tuple[int, int, int]:
    if status == "confirmed":
        return (55, 200, 90)
    if status == "boundary_review":
        return (255, 180, 60)
    if status == "singleton_review":
        return (70, 160, 255)
    return (255, 110, 110)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_mask_observations_csv(path: Path, observations: list[MaskObservation]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "book_id", "video_id", "deva_object_id", "frame_index", "time_sec",
            "source_detection_id", "source_instance_id", "score",
            "x1", "y1", "x2", "y2",
            "mask_path", "crop_path",
            "quality_score", "touches_left_edge", "touches_right_edge",
            "sharpness", "crop_area", "mask_fill_ratio",
        ])
        for obs in sorted(observations, key=lambda r: (r.frame_index, r.deva_object_id)):
            writer.writerow([
                obs.book_id, obs.video_id, obs.deva_object_id,
                obs.frame_index, f"{obs.time_sec:.2f}",
                obs.source_detection_id, obs.source_instance_id,
                f"{obs.detection_score:.4f}",
                obs.x1, obs.y1, obs.x2, obs.y2,
                obs.mask_path, obs.crop_path,
                f"{obs.quality_score:.4f}",
                int(obs.touches_left_edge), int(obs.touches_right_edge),
                f"{obs.sharpness:.4f}", obs.crop_area,
                f"{obs.mask_fill_ratio:.4f}",
            ])


def write_unique_books_csv(
    path: Path,
    tracks: dict[int, list[MaskObservation]],
    statuses: dict[int, tuple[str, int]],
    best_observations: dict[int, MaskObservation],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "book_id", "video_id", "deva_object_id",
            "first_frame", "last_frame", "first_time_sec", "last_time_sec",
            "num_observations", "status", "count_as_book",
            "best_crop_path", "best_quality_score", "best_detection_score",
            "mean_width", "mean_height",
        ])
        for obj_id in sorted(tracks):
            obs_list = [o for o in tracks[obj_id] if o.valid]
            if obs_list:
                obs_list.sort(key=lambda r: (r.frame_index, r.source_detection_id))
                best = best_observations[obj_id]
                status, count_as_book = statuses[obj_id]
                mean_w = float(np.mean([o.width for o in obs_list]))
                mean_h = float(np.mean([o.height for o in obs_list]))
                writer.writerow([
                    obs_list[0].book_id, obs_list[0].video_id, obj_id,
                    obs_list[0].frame_index, obs_list[-1].frame_index,
                    f"{obs_list[0].time_sec:.2f}", f"{obs_list[-1].time_sec:.2f}",
                    len(obs_list), status, count_as_book,
                    best.crop_path,
                    f"{best.quality_score:.4f}", f"{best.detection_score:.4f}",
                    f"{mean_w:.4f}", f"{mean_h:.4f}",
                ])
            else:
                writer.writerow([
                    f"unknown_{obj_id:04d}", "", obj_id,
                    "", "", "", "", 0, "invalid", 0, "", "", "", "", "",
                ])


def write_tracking_summary(
    path: Path,
    video_id: str,
    frame_count: int,
    detections_total: int,
    objects_total: int,
    statuses: dict[int, tuple[str, int]],
    config: dict,
) -> None:
    summary = {
        "video_id": video_id,
        "tracker": "DEVA",
        "frames_processed": frame_count,
        "detections_total": detections_total,
        "deva_objects_total": objects_total,
        "confirmed_books": sum(1 for s, _ in statuses.values() if s == "confirmed"),
        "boundary_review": sum(1 for s, _ in statuses.values() if s == "boundary_review"),
        "singleton_review": sum(1 for s, _ in statuses.values() if s == "singleton_review"),
        "invalid_tracks": sum(1 for s, _ in statuses.values() if s == "invalid"),
        "max_missed_detection_count": config.get("max_missed_detection_count"),
        "suspected_merged_masks": 0,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tracked mask video
# ---------------------------------------------------------------------------

def write_tracked_mask_video(
    path: Path,
    frame_paths: dict[int, Path],
    detections_by_frame: dict[int, list[DetectionRow]],
    observations_by_frame: dict[int, list[MaskObservation]],
    statuses: dict[int, tuple[str, int]],
    roi_bbox: Optional[list[int]],
    sample_fps: float,
    job_dir: Path,
) -> None:
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Build set of known object IDs per frame for HUD
    obj_first_frame: dict[int, int] = {}
    for rows in observations_by_frame.values():
        for obs in rows:
            if not obs.valid:
                continue
            fid = obs.frame_index
            oid = obs.deva_object_id
            obj_first_frame[oid] = min(obj_first_frame.get(oid, fid), fid)

    for frame_index in sorted(frame_paths):
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            print(f"WARNING: failed to read frame {frame_index}")
            continue

        vis = frame.copy()
        frame_obs = observations_by_frame.get(frame_index, [])

        # Overlay semi-transparent colored masks
        for obs in sorted(frame_obs, key=lambda r: r.bbox_area, reverse=True):
            mask_img = cv2.imread(str(obs.mask_abs_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            color = object_color(obs.deva_object_id)
            mask_bool = mask_img > 127
            if np.any(mask_bool):
                vis[mask_bool] = (vis[mask_bool] * 0.45 + np.array(color[::-1]) * 0.55).astype(np.uint8)

        # Draw object ID labels (no bboxes)
        for obs in sorted(frame_obs, key=lambda r: r.x1):
            color = object_color(obs.deva_object_id)
            status, _ = statuses.get(obs.deva_object_id, ("unknown", 0))
            label = f"{obs.book_id}"
            if status != "confirmed":
                label = f"{label} {status}"
            text_origin = (obs.x1, max(18, obs.y1 - 6))
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3)
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color(status), 1)

        # ROI rectangle
        if roi_bbox is not None:
            cv2.rectangle(vis, (roi_bbox[0], roi_bbox[1]), (roi_bbox[2], roi_bbox[3]), (40, 215, 255), 2)

        # HUD
        active_ids = {obs.deva_object_id for obs in frame_obs}
        started_ids = {oid for oid, ff in obj_first_frame.items() if ff <= frame_index}
        confirmed = sum(1 for oid in started_ids if statuses.get(oid, ("invalid", 0))[0] == "confirmed")
        review = sum(1 for oid in started_ids if statuses.get(oid, ("invalid", 0))[0] in {"boundary_review", "singleton_review"})

        hud_lines = [
            f"frame: {frame_index:06d}",
            f"detections: {len(detections_by_frame.get(frame_index, []))}",
            f"active objects: {len(active_ids)}",
            f"confirmed: {confirmed}",
            f"review: {review}",
            f"unique objects: {len(started_ids)}",
        ]
        y = 28
        for line in hud_lines:
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)
            y += 28

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(path), fourcc, sample_fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f"Failed to create video: {path}")
        writer.write(vis)

    if writer is not None:
        writer.release()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Track book-spine masks with DEVA")
    parser.add_argument("--job-dir", required=True, help="Job directory e.g. output/jobs/000")
    parser.add_argument("--sample-fps", type=float, help="FPS of sampled frames; inferred if omitted")
    parser.add_argument("--deva-weights", default=str(DEVA_ROOT / "saves" / "DEVA-propagation.pth"),
                        help="Path to DEVA-propagation.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detection-score-threshold", type=float, default=0.10,
                        help="Minimum MaskDINO score to feed into DEVA")
    parser.add_argument("--max-missed-detection-count", type=int, default=9,
                        help="Frames before an undetected object is purged")
    parser.add_argument("--image-size", type=int, default=480,
                        help="DEVA resizes shorter side to this; -1 for original")
    parser.add_argument("--chunk-size", type=int, default=-1,
                        help="Objects per batch; -1 unlimited; reduce to save memory")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    frames_dir = job_dir / "frames"
    detections_csv = job_dir / "detections.csv"
    roi_json_path = job_dir / "roi.json"
    tracked_video_path = job_dir / "tracked_mask_video.mp4"
    observations_csv_path = job_dir / "mask_observations.csv"
    unique_books_csv_path = job_dir / "unique_books.csv"
    summary_json_path = job_dir / "tracking_summary.json"
    books_dir = job_dir / "books"

    if not detections_csv.exists():
        raise SystemExit(f"detections.csv not found: {detections_csv}")
    if not frames_dir.exists():
        raise SystemExit(f"frames directory not found: {frames_dir}")

    ensure_dir(books_dir)

    # Load input data
    video_id, detections_by_frame, detections_by_id = load_detections(detections_csv, job_dir)
    frame_paths = {
        int(p.stem): p for p in sorted(frames_dir.glob("*.jpg"))
    }
    if not frame_paths:
        raise SystemExit(f"No frames in {frames_dir}")

    first_frame_img = cv2.imread(str(frame_paths[min(frame_paths)]))
    if first_frame_img is None:
        raise SystemExit(f"Failed to read first frame")
    frame_height, frame_width = first_frame_img.shape[:2]

    # ROI
    roi_json = load_roi_json(roi_json_path)
    roi_bbox = denormalize_roi_xyxy(roi_json, first_frame_img.shape) if roi_json is not None else None

    # Infer sample FPS
    if args.sample_fps:
        sample_fps = float(args.sample_fps)
    else:
        from collections import OrderedDict
        ordered = sorted(detections_by_frame.items())
        time_diffs = []
        for (fa, ra), (fb, rb) in zip(ordered, ordered[1:]):
            if fb > fa and rb and ra and rb[0].time_sec > ra[0].time_sec:
                time_diffs.append((fb - fa) / (rb[0].time_sec - ra[0].time_sec))
        sample_fps = float(np.median(time_diffs)) if time_diffs else 3.0
    if sample_fps <= 0:
        raise SystemExit("sample_fps must be > 0")

    # Load DEVA model
    deva, config = load_deva_model(args.deva_weights, args.device)
    config["max_missed_detection_count"] = args.max_missed_detection_count
    config["size"] = args.image_size
    config["chunk_size"] = args.chunk_size
    print(f"sample_fps: {sample_fps:.2f}")
    print(f"max_missed_detection_count: {args.max_missed_detection_count} "
          f"≈ {args.max_missed_detection_count / sample_fps:.1f}s")
    print(f"detection score threshold: {args.detection_score_threshold}")

    # Per-frame processing
    all_observations: list[MaskObservation] = []
    observations_by_obj: dict[int, list[MaskObservation]] = defaultdict(list)
    observations_by_frame: dict[int, list[MaskObservation]] = defaultdict(list)

    # Track the mapping from DEVA object_id back to our observation structure
    # DEVA's object_manager keeps the authoritative mapping

    left_bound = roi_bbox[0] if roi_bbox else 0
    right_bound = roi_bbox[2] if roi_bbox else frame_width
    edge_margin = max(8.0, 0.01 * (right_bound - left_bound))

    # Resize logic: DEVA processes frames with shorter side = image_size
    if args.image_size > 0:
        orig_h, orig_w = frame_height, frame_width
        scale = args.image_size / min(orig_h, orig_w)
        proc_h = int(round(orig_h * scale))
        proc_w = int(round(orig_w * scale))
        # Ensure divisible by 16 for DEVA's pad_divide_by
        proc_h = (proc_h // 16) * 16
        proc_w = (proc_w // 16) * 16
        print(f"DEVA processing resolution: {proc_w}x{proc_h} (scale={scale:.3f})")
    else:
        proc_h, proc_w = frame_height, frame_width
        scale = 1.0

    for frame_index in sorted(frame_paths):
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            print(f"WARNING: skipping frame {frame_index}, failed to read")
            continue

        frame_detections = detections_by_frame.get(frame_index, [])

        # Compose ID mask from MaskDINO instance masks (at original resolution)
        id_mask_orig, segments_info, valid_dets = compose_id_mask(
            frame_detections,
            frame.shape,
            detection_score_threshold=args.detection_score_threshold,
        )

        # Resize for DEVA processing
        if scale != 1.0:
            frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            # Resize mask with nearest-neighbor to preserve IDs
            id_mask_np = id_mask_orig.numpy().astype(np.int32)
            id_mask_np = cv2.resize(id_mask_np, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            id_mask = torch.from_numpy(id_mask_np).to(args.device)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            id_mask = id_mask_orig.to(args.device)

        # Convert frame to torch tensor for DEVA
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(args.device)

        if len(segments_info) > 0:
            # Feed detections to DEVA
            with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
                pred_prob = deva.incorporate_detection(
                    image=frame_tensor,
                    new_mask=id_mask.to(args.device),
                    segments_info=segments_info,
                    incremental=True,
                )
        else:
            # No detections this frame — propagate from memory only
            if deva.memory.engaged:
                with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
                    pred_prob = deva.step(frame_tensor, mask=None, end=False)
            else:
                print(f"frame={frame_index:06d} no detections, no memory — skipping")
                continue

        # Map DEVA object IDs back to our format
        obj_to_tmp = deva.object_manager.obj_to_tmp_id
        tmp_to_obj = deva.object_manager.tmp_id_to_obj

        # Resize pred_mask back to original resolution for IoU matching
        pred_mask_proc = torch.argmax(pred_prob, dim=0)  # at processed resolution
        if scale != 1.0:
            pred_mask_np = pred_mask_proc.cpu().numpy().astype(np.int32)
            pred_mask_full = cv2.resize(pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            pred_mask_full = torch.from_numpy(pred_mask_full)
        else:
            pred_mask_full = pred_mask_proc.cpu()

        # Build tmp_id -> best detection via IoU at original resolution
        tmp_id_to_best_det: dict[int, DetectionRow] = {}

        for obj_info, tmp_id in obj_to_tmp.items():
            obj_mask = (pred_mask_full == tmp_id).numpy()
            obj_area = int(obj_mask.sum())
            if obj_area == 0:
                continue

            best_iou = 0.0
            best_det = None
            for det in valid_dets:
                det_mask = cv2.imread(str(det.mask_abs_path), cv2.IMREAD_GRAYSCALE)
                if det_mask is None:
                    continue
                det_bool = det_mask > 127
                intersection = int((obj_mask & det_bool).sum())
                union = int((obj_mask | det_bool).sum())
                iou = intersection / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou > 0.3:
                tmp_id_to_best_det[tmp_id] = best_det

        # Create observations for tracked objects
        current_obs = []
        for obj_info, tmp_id in obj_to_tmp.items():
            obj_id = obj_info.id  # DEVA's stable object ID
            det = tmp_id_to_best_det.get(tmp_id)
            if det is None:
                # Object exists in memory but wasn't detected this frame (propagated)
                continue

            touches_left = det.x1 <= left_bound + edge_margin
            touches_right = det.x2 >= right_bound - edge_margin
            sharpness = compute_sharpness(det.crop_abs_path)
            crop_area = 0
            crop_img = cv2.imread(str(det.crop_abs_path))
            if crop_img is not None:
                crop_area = int(crop_img.shape[0] * crop_img.shape[1])
            mask_fill = compute_mask_fill_ratio(det.mask_abs_path, det.bbox_area)
            valid = (
                det.x2 > det.x1
                and det.y2 > det.y1
                and det.bbox_area >= 32
                and det.crop_abs_path.exists()
                and det.mask_abs_path.exists()
            )

            book_id = f"{video_id}_book_{obj_id:04d}"
            obs = MaskObservation(
                video_id=video_id,
                deva_object_id=obj_id,
                book_id=book_id,
                frame_index=frame_index,
                time_sec=det.time_sec,
                source_detection_id=det.detection_id,
                source_instance_id=det.instance_id,
                detection_score=det.score,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                mask_path=det.mask_path,
                crop_path=det.crop_path,
                mask_abs_path=det.mask_abs_path,
                crop_abs_path=det.crop_abs_path,
                touches_left_edge=touches_left,
                touches_right_edge=touches_right,
                sharpness=sharpness,
                crop_area=crop_area,
                mask_fill_ratio=mask_fill,
                bbox_area=det.bbox_area,
                valid=valid,
            )
            current_obs.append(obs)
            observations_by_obj[obj_id].append(obs)
            observations_by_frame[frame_index].append(obs)
            all_observations.append(obs)

        print(
            f"frame={frame_index:06d} "
            f"detections={len(valid_dets)} "
            f"tracked={len(current_obs)} "
            f"total_objects={len(observations_by_obj)}"
        )

    # Post-processing
    normalize_quality_scores(all_observations)
    statuses = {
        obj_id: determine_track_status(obs_list)
        for obj_id, obs_list in observations_by_obj.items()
    }

    # Best observation per object
    best_observations: dict[int, MaskObservation] = {}
    for obj_id, obs_list in observations_by_obj.items():
        valid_list = [o for o in obs_list if o.valid]
        if valid_list:
            best = max(valid_list, key=lambda o: (o.quality_score, o.detection_score, o.crop_area))
            best_observations[obj_id] = best
            # Copy best crop to books/
            shutil.copyfile(best.crop_abs_path, books_dir / f"{best.book_id}.jpg")

    # Write outputs
    write_mask_observations_csv(observations_csv_path, all_observations)
    write_unique_books_csv(unique_books_csv_path, observations_by_obj, statuses, best_observations)
    write_tracking_summary(
        summary_json_path, video_id,
        frame_count=len(frame_paths),
        detections_total=sum(len(rows) for rows in detections_by_frame.values()),
        objects_total=len(observations_by_obj),
        statuses=statuses,
        config=config,
    )
    write_tracked_mask_video(
        tracked_video_path,
        frame_paths=frame_paths,
        detections_by_frame=detections_by_frame,
        observations_by_frame=observations_by_frame,
        statuses=statuses,
        roi_bbox=roi_bbox,
        sample_fps=sample_fps,
        job_dir=job_dir,
    )

    # Summary
    confirmed = sum(1 for s, _ in statuses.values() if s == "confirmed")
    boundary = sum(1 for s, _ in statuses.values() if s == "boundary_review")
    singleton = sum(1 for s, _ in statuses.values() if s == "singleton_review")
    invalid = sum(1 for s, _ in statuses.values() if s == "invalid")

    print(f"\njob dir              -> {job_dir}")
    print(f"tracked mask video   -> {tracked_video_path}")
    print(f"mask observations    -> {observations_csv_path}")
    print(f"unique books csv     -> {unique_books_csv_path}")
    print(f"tracking summary     -> {summary_json_path}")
    print(f"books dir            -> {books_dir}")
    print(f"frames processed     : {len(frame_paths)}")
    print(f"detections total     : {sum(len(rows) for rows in detections_by_frame.values())}")
    print(f"DEVA objects total   : {len(observations_by_obj)}")
    print(f"confirmed            : {confirmed}")
    print(f"boundary_review      : {boundary}")
    print(f"singleton_review     : {singleton}")
    print(f"invalid              : {invalid}")
    print(f"ground truth         : 165")


if __name__ == "__main__":
    main()
