"""Deduplicate repeated book-spine detections across sampled video frames.

Inputs:
    output/jobs/<video_id>/
      frames/
      detections.csv
      optional roi.json

Outputs:
    output/jobs/<video_id>/
      frame_motion.csv
      book_observations.csv
      book_tracks.csv

This is a first-pass tracker for the project stage:
    adjacent-frame registration
    -> estimate horizontal phone motion
    -> map detections to a shared shelf coordinate
    -> merge repeated detections into unique book_id
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

NUMPY_MAJOR_VERSION = int(str(np.__version__).split(".", 1)[0])
if NUMPY_MAJOR_VERSION >= 2:
    SCIPY_LINEAR_SUM_ASSIGNMENT = None
else:
    try:
        from scipy.optimize import linear_sum_assignment as SCIPY_LINEAR_SUM_ASSIGNMENT  # type: ignore
    except Exception:
        SCIPY_LINEAR_SUM_ASSIGNMENT = None


@dataclass
class Detection:
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
    crop_abs_path: Path
    mask_abs_path: Path
    segment_id: int = 0
    shelf_center_x: float = 0.0
    shelf_x1: float = 0.0
    shelf_x2: float = 0.0
    match_cost: float | None = None
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    area_score: float = 0.0
    completeness_score: float = 0.0
    edge_penalty: float = 0.0

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.x1), float(self.y1), float(self.x2), float(self.y2))


@dataclass
class Track:
    track_index: int
    video_id: str
    observations: list[Detection] = field(default_factory=list)
    best_detection: Detection | None = None
    best_quality_detection: Detection | None = None
    status: str = "singleton"
    count_as_book: bool = False
    boundary: bool = False

    def add(self, det: Detection) -> None:
        self.observations.append(det)
        if self.best_detection is None or det.score > self.best_detection.score:
            self.best_detection = det
        if self.best_quality_detection is None or det.quality_score > self.best_quality_detection.quality_score:
            self.best_quality_detection = det

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def last_detection(self) -> Detection:
        return self.observations[-1]

    @property
    def num_observations(self) -> int:
        return len(self.observations)


@dataclass
class MotionRow:
    frame_index: int
    time_sec: float
    dx: float
    dy: float
    cumulative_dx: float
    cumulative_dy: float
    response: float
    used_fallback: bool
    method: str
    segment_id: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_roi_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    roi = json.loads(path.read_text(encoding="utf-8"))
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(roi):
        raise SystemExit(f"ROI json missing keys: {sorted(required - set(roi))}")
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


def frame_index_from_path(path: Path) -> int:
    return int(path.stem)


def list_frame_paths(frames_dir: Path) -> dict[int, Path]:
    frame_paths = {frame_index_from_path(path): path for path in sorted(frames_dir.glob("*.jpg"))}
    if not frame_paths:
        raise SystemExit(f"No frame images found in {frames_dir}")
    return frame_paths


def load_detections(csv_path: Path, job_dir: Path) -> tuple[str, dict[int, list[Detection]]]:
    rows_by_frame: dict[int, list[Detection]] = defaultdict(list)
    video_id = None
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"] if video_id is None else video_id
            det = Detection(
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
                crop_abs_path=job_dir / row["crop_path"],
                mask_abs_path=job_dir / row["mask_path"],
            )
            rows_by_frame[det.frame_index].append(det)
    if video_id is None:
        raise SystemExit(f"No detections found in {csv_path}")
    for frame_rows in rows_by_frame.values():
        frame_rows.sort(key=lambda det: det.center_x)
    return video_id, rows_by_frame


def preprocess_frame_for_motion(image: np.ndarray, roi_bbox: list[int] | None, scale: float) -> np.ndarray:
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        image = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        new_w = max(32, int(round(gray.shape[1] * scale)))
        new_h = max(32, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    gray = gray.astype(np.float32)
    return gray


def estimate_pair_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    roi_bbox: list[int] | None,
    scale: float,
) -> tuple[float, float, float]:
    prev_gray = preprocess_frame_for_motion(prev_frame, roi_bbox, scale)
    curr_gray = preprocess_frame_for_motion(curr_frame, roi_bbox, scale)
    h, w = prev_gray.shape[:2]
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(prev_gray, curr_gray, window)
    dx = float(shift[0]) / scale
    dy = float(shift[1]) / scale
    return dx, dy, float(response)


def estimate_pair_motion_akaze(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    roi_bbox: list[int] | None,
) -> tuple[float, float, float] | None:
    prev_gray = preprocess_frame_for_motion(prev_frame, roi_bbox, 1.0).astype(np.uint8)
    curr_gray = preprocess_frame_for_motion(curr_frame, roi_bbox, 1.0).astype(np.uint8)
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(prev_gray, None)
    kp2, des2 = detector.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return None

    matches = sorted(matches, key=lambda match: match.distance)[:80]
    pts1 = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None:
        return None

    inlier_ratio = float(inliers.sum()) / len(inliers) if inliers is not None and len(inliers) else 0.0
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    return dx, dy, inlier_ratio


def estimate_frame_motion_series(
    frame_paths: dict[int, Path],
    roi_json: dict | None,
    phase_scale: float,
    min_response: float,
    max_abs_dx: float,
    max_abs_dy: float,
    sample_fps: float,
) -> dict[int, MotionRow]:
    frame_indices = sorted(frame_paths)
    first_image = cv2.imread(str(frame_paths[frame_indices[0]]))
    if first_image is None:
        raise SystemExit(f"Failed to read frame image: {frame_paths[frame_indices[0]]}")
    roi_bbox = denormalize_roi_xyxy(roi_json, first_image.shape) if roi_json is not None else None

    rows: dict[int, MotionRow] = {}
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    valid_dx_history: deque[float] = deque(maxlen=5)
    valid_dy_history: deque[float] = deque(maxlen=5)

    rows[frame_indices[0]] = MotionRow(
        frame_index=frame_indices[0],
        time_sec=0.0,
        dx=0.0,
        dy=0.0,
        cumulative_dx=0.0,
        cumulative_dy=0.0,
        response=1.0,
        used_fallback=False,
        method="seed",
        segment_id=0,
    )

    prev_image = first_image
    segment_id = 0
    for frame_index in frame_indices[1:]:
        curr_image = cv2.imread(str(frame_paths[frame_index]))
        if curr_image is None:
            raise SystemExit(f"Failed to read frame image: {frame_paths[frame_index]}")

        raw_dx, raw_dy, response = estimate_pair_motion(prev_image, curr_image, roi_bbox, phase_scale)
        used_fallback = False
        dx = raw_dx
        dy = raw_dy
        method = "phase"
        if response < min_response or abs(dx) > max_abs_dx or abs(dy) > max_abs_dy:
            fallback = estimate_pair_motion_akaze(prev_image, curr_image, roi_bbox)
            if fallback is not None:
                akaze_dx, akaze_dy, inlier_ratio = fallback
                if (
                    inlier_ratio >= 0.30
                    and abs(akaze_dx) <= max_abs_dx
                    and abs(akaze_dy) <= max_abs_dy
                ):
                    dx = akaze_dx
                    dy = akaze_dy
                    response = inlier_ratio
                    used_fallback = True
                    method = "akaze"
                else:
                    dx = 0.0
                    dy = 0.0
                    response = 0.0
                    used_fallback = True
                    method = "segment_break"
                    segment_id += 1
            else:
                dx = 0.0
                dy = 0.0
                response = 0.0
                used_fallback = True
                method = "segment_break"
                segment_id += 1

        if abs(dx) <= max_abs_dx:
            valid_dx_history.append(dx)
        elif valid_dx_history:
            dx = float(np.median(valid_dx_history))
        else:
            dx = 0.0

        if abs(dy) <= max_abs_dy:
            valid_dy_history.append(dy)
        elif valid_dy_history:
            dy = float(np.median(valid_dy_history))
        else:
            dy = 0.0

        cumulative_dx += dx
        cumulative_dy += dy
        rows[frame_index] = MotionRow(
            frame_index=frame_index,
            time_sec=(frame_index - frame_indices[0]) / sample_fps,
            dx=dx,
            dy=dy,
            cumulative_dx=cumulative_dx,
            cumulative_dy=cumulative_dy,
            response=response,
            used_fallback=used_fallback,
            method=method,
            segment_id=segment_id,
        )
        prev_image = curr_image

    return rows


def interval_iou(a1: float, a2: float, b1: float, b2: float) -> float:
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    if union <= 0:
        return 0.0
    return inter / union


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def safe_log_ratio(a: float, b: float) -> float:
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    return abs(math.log(a / b))


def compute_feature_for_crop(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path))
    if image is None:
        return np.zeros(3, dtype=np.float32), np.ones(16, dtype=np.float32) / 16.0
    small = cv2.resize(image, (64, 256), interpolation=cv2.INTER_AREA)
    mean_bgr = small.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum
    return mean_bgr, hist


def compute_crop_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_32F).var())


def compute_detection_quality(
    det: Detection,
    roi_bbox: list[int] | None,
    frame_width: int,
    area_norm_base: float,
    sharpness_norm_base: float,
) -> None:
    left_bound = roi_bbox[0] if roi_bbox is not None else 0
    right_bound = roi_bbox[2] if roi_bbox is not None else frame_width
    edge_margin = max(8.0, 0.01 * (right_bound - left_bound))

    area = det.width * det.height
    det.area_score = min(1.0, area / max(area_norm_base, 1.0))
    det.sharpness_score = min(1.0, compute_crop_sharpness(det.crop_abs_path) / max(sharpness_norm_base, 1.0))

    touches_left = det.x1 <= left_bound + edge_margin
    touches_right = det.x2 >= right_bound - edge_margin
    det.edge_penalty = 0.35 if (touches_left or touches_right) else 0.0
    det.completeness_score = 1.0 - det.edge_penalty

    det.quality_score = (
        0.35 * det.sharpness_score
        + 0.25 * det.score
        + 0.20 * det.area_score
        + 0.20 * det.completeness_score
        - det.edge_penalty
    )


def touches_boundary(det: Detection, left_bound: int, right_bound: int, edge_margin: float) -> bool:
    return det.x1 <= left_bound + edge_margin or det.x2 >= right_bound - edge_margin


def appearance_distance(
    det_a: Detection,
    det_b: Detection,
    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> float:
    if det_a.crop_abs_path not in feature_cache:
        feature_cache[det_a.crop_abs_path] = compute_feature_for_crop(det_a.crop_abs_path)
    if det_b.crop_abs_path not in feature_cache:
        feature_cache[det_b.crop_abs_path] = compute_feature_for_crop(det_b.crop_abs_path)

    mean_a, hist_a = feature_cache[det_a.crop_abs_path]
    mean_b, hist_b = feature_cache[det_b.crop_abs_path]

    color_dist = float(np.linalg.norm(mean_a - mean_b) / math.sqrt(3.0))
    hist_corr = float(np.minimum(hist_a, hist_b).sum())
    hist_dist = 1.0 - hist_corr
    return 0.55 * color_dist + 0.45 * hist_dist


def shifted_bbox(det: Detection, dx: float, dy: float) -> tuple[float, float, float, float]:
    return (
        det.x1 + dx,
        det.y1 + dy,
        det.x2 + dx,
        det.y2 + dy,
    )


def median_detection_values(detections: list[Detection]) -> tuple[float, float, float, float]:
    x1 = float(np.median([det.x1 for det in detections]))
    y1 = float(np.median([det.y1 for det in detections]))
    x2 = float(np.median([det.x2 for det in detections]))
    y2 = float(np.median([det.y2 for det in detections]))
    return x1, y1, x2, y2


def track_prototype_bbox(track: Track, frame_motion: dict[int, MotionRow], current_frame: int) -> tuple[float, float, float, float]:
    recent = track.observations[-3:]
    shifted = []
    for det in recent:
        dx = frame_motion[current_frame].cumulative_dx - frame_motion[det.frame_index].cumulative_dx
        dy = frame_motion[current_frame].cumulative_dy - frame_motion[det.frame_index].cumulative_dy
        shifted.append(shifted_bbox(det, dx, dy))
    x1 = float(np.median([box[0] for box in shifted]))
    y1 = float(np.median([box[1] for box in shifted]))
    x2 = float(np.median([box[2] for box in shifted]))
    y2 = float(np.median([box[3] for box in shifted]))
    return x1, y1, x2, y2


def detection_match_cost(
    track: Track,
    det: Detection,
    frame_motion: dict[int, MotionRow],
    current_frame: int,
    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    max_center_distance: float,
    max_vertical_distance: float,
    max_size_log_ratio: float,
    min_x_iou: float,
    min_box_iou: float,
) -> float | None:
    current_motion = frame_motion[current_frame]
    if det.segment_id != current_motion.segment_id:
        return None
    if frame_motion[track.last_detection.frame_index].segment_id != current_motion.segment_id:
        return None

    last_det = track.last_detection
    pred_bbox = track_prototype_bbox(track, frame_motion, current_frame)
    pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
    pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0

    center_dx = abs(det.center_x - pred_cx)
    center_dy = abs(det.center_y - pred_cy)
    prototype_width = max(1.0, pred_bbox[2] - pred_bbox[0])
    prototype_height = max(1.0, pred_bbox[3] - pred_bbox[1])
    width_ratio = safe_log_ratio(det.width, prototype_width)
    height_ratio = safe_log_ratio(det.height, prototype_height)
    x_iou = interval_iou(pred_bbox[0], pred_bbox[2], det.x1, det.x2)
    box_iou = bbox_iou(pred_bbox, det.bbox)

    if center_dx > max_center_distance:
        return None
    if center_dy > max_vertical_distance:
        return None
    if width_ratio > max_size_log_ratio or height_ratio > max_size_log_ratio:
        return None
    if x_iou < min_x_iou and box_iou < min_box_iou:
        return None

    best_ref = track.best_quality_detection if track.best_quality_detection is not None else last_det
    app_dist = 0.5 * appearance_distance(last_det, det, feature_cache) + 0.5 * appearance_distance(best_ref, det, feature_cache)
    pos_cost = center_dx / max_center_distance + 0.25 * (center_dy / max_vertical_distance)
    size_cost = 0.20 * (width_ratio / max_size_log_ratio) + 0.35 * (height_ratio / max_size_log_ratio)
    overlap_bonus = 0.35 * x_iou + 0.15 * box_iou
    quality_bonus = 0.05 * det.quality_score + 0.05 * det.score
    cost = pos_cost + size_cost + 0.40 * app_dist - overlap_bonus - quality_bonus
    return float(cost)


def assign_tracks_to_detections(cost_matrix: np.ndarray, max_match_cost: float) -> list[tuple[int, int, float]]:
    if cost_matrix.size == 0:
        return []

    assignments: list[tuple[int, int, float]] = []
    if SCIPY_LINEAR_SUM_ASSIGNMENT is not None:
        work_matrix = cost_matrix.copy()
        work_matrix[~np.isfinite(work_matrix)] = max_match_cost + 1000.0
        row_ind, col_ind = SCIPY_LINEAR_SUM_ASSIGNMENT(work_matrix)
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            cost = float(cost_matrix[row_idx, col_idx])
            if np.isfinite(cost) and cost <= max_match_cost:
                assignments.append((row_idx, col_idx, cost))
        return assignments

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = []
    for row_idx in range(cost_matrix.shape[0]):
        for col_idx in range(cost_matrix.shape[1]):
            cost = float(cost_matrix[row_idx, col_idx])
            if np.isfinite(cost):
                flat.append((cost, row_idx, col_idx))
    flat.sort()
    for cost, row_idx, col_idx in flat:
        if cost > max_match_cost:
            break
        if row_idx in used_rows or col_idx in used_cols:
            continue
        assignments.append((row_idx, col_idx, cost))
        used_rows.add(row_idx)
        used_cols.add(col_idx)
    return assignments


def link_tracks(
    detections_by_frame: dict[int, list[Detection]],
    frame_motion: dict[int, MotionRow],
    max_frame_gap: int,
    max_center_distance: float,
    max_vertical_distance: float,
    max_size_ratio: float,
    min_x_iou: float,
    min_box_iou: float,
    max_match_cost: float,
) -> list[Track]:
    tracks: list[Track] = []
    active_indices: list[int] = []
    next_track_index = 1
    feature_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    max_size_log_ratio = math.log(max_size_ratio)

    for frame_index in sorted(frame_motion):
        detections = detections_by_frame.get(frame_index, [])
        motion = frame_motion[frame_index]
        for det in detections:
            det.segment_id = motion.segment_id
            det.shelf_center_x = det.center_x - motion.cumulative_dx
            det.shelf_x1 = det.x1 - motion.cumulative_dx
            det.shelf_x2 = det.x2 - motion.cumulative_dx

        active_indices = [
            track_idx
            for track_idx in active_indices
            if frame_index - tracks[track_idx].last_frame <= max_frame_gap
        ]

        if not detections:
            continue

        candidate_track_indices = active_indices[:]
        cost_matrix = np.full((len(candidate_track_indices), len(detections)), np.inf, dtype=np.float32)
        for row_idx, track_idx in enumerate(candidate_track_indices):
            track = tracks[track_idx]
            for col_idx, det in enumerate(detections):
                cost = detection_match_cost(
                    track=track,
                    det=det,
                    frame_motion=frame_motion,
                    current_frame=frame_index,
                    feature_cache=feature_cache,
                    max_center_distance=max_center_distance,
                    max_vertical_distance=max_vertical_distance,
                    max_size_log_ratio=max_size_log_ratio,
                    min_x_iou=min_x_iou,
                    min_box_iou=min_box_iou,
                )
                if cost is not None:
                    cost_matrix[row_idx, col_idx] = cost

        assignments = assign_tracks_to_detections(cost_matrix, max_match_cost=max_match_cost)
        matched_track_rows = set()
        matched_det_cols = set()

        for row_idx, col_idx, cost in assignments:
            track_idx = candidate_track_indices[row_idx]
            det = detections[col_idx]
            det.match_cost = cost
            tracks[track_idx].add(det)
            matched_track_rows.add(row_idx)
            matched_det_cols.add(col_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_cols:
                continue
            track = Track(track_index=next_track_index, video_id=det.video_id)
            det.match_cost = None
            track.add(det)
            tracks.append(track)
            active_indices.append(len(tracks) - 1)
            next_track_index += 1

    return tracks


def finalize_track_states(
    tracks: list[Track],
    roi_bbox: list[int] | None,
    frame_width: int,
) -> None:
    left_bound = roi_bbox[0] if roi_bbox is not None else 0
    right_bound = roi_bbox[2] if roi_bbox is not None else frame_width
    edge_margin = max(8.0, 0.01 * (right_bound - left_bound))

    for track in tracks:
        best_quality = track.best_quality_detection if track.best_quality_detection is not None else track.best_detection
        track.boundary = False
        if best_quality is not None:
            track.boundary = touches_boundary(best_quality, left_bound, right_bound, edge_margin)
        best_quality = track.best_quality_detection if track.best_quality_detection is not None else track.best_detection
        if best_quality is None:
            track.status = "invalid"
            track.count_as_book = False
            continue

        if track.num_observations >= 2:
            track.status = "boundary" if track.boundary else "confirmed"
            track.count_as_book = True
        elif track.boundary:
            track.status = "boundary"
            track.count_as_book = False
        else:
            track.status = "singleton"
            track.count_as_book = False


def color_for_track(index: int) -> tuple[int, int, int]:
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
    return palette[(index - 1) % len(palette)]


def draw_roi_outline(image: np.ndarray, roi_bbox: list[int] | None) -> np.ndarray:
    if roi_bbox is None:
        return image
    vis = image.copy()
    x1, y1, x2, y2 = roi_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 215, 255), 2)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 215, 255), 1)
    return vis


def write_tracked_video(
    path: Path,
    frame_paths: dict[int, Path],
    tracks: list[Track],
    roi_bbox: list[int] | None,
    sample_fps: float,
    max_frame_gap: int,
) -> None:
    observations_by_frame: dict[int, list[tuple[int, Track, Detection]]] = defaultdict(list)
    for book_idx, track in enumerate(tracks, start=1):
        for det in track.observations:
            observations_by_frame[det.frame_index].append((book_idx, track, det))

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_indices = sorted(frame_paths)

    for frame_index in frame_indices:
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            raise SystemExit(f"Failed to read frame image: {frame_paths[frame_index]}")
        vis = draw_roi_outline(frame, roi_bbox)

        rows = sorted(observations_by_frame.get(frame_index, []), key=lambda row: row[2].x1)
        for book_idx, track, det in rows:
            color = color_for_track(book_idx)
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"book_{book_idx:04d}"
            if track.status != "confirmed":
                label = f"{label}:{track.status}"
            text_origin = (det.x1, max(18, det.y1 - 6))
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
            cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        current_detections = len(rows)
        active_tracks = sum(
            1
            for track in tracks
            if track.first_frame <= frame_index <= track.last_frame + max_frame_gap
        )
        new_tracks = sum(1 for track in tracks if track.first_frame == frame_index)
        confirmed_so_far = sum(1 for track in tracks if track.count_as_book and track.first_frame <= frame_index)
        overlay_lines = [
            f"detections: {current_detections}",
            f"active tracks: {active_tracks}",
            f"new tracks: {new_tracks}",
            f"confirmed books: {confirmed_so_far}",
        ]
        y = 28
        for line in overlay_lines:
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y += 28

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(path), fourcc, sample_fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f"Failed to create tracked video: {path}")
        writer.write(vis)

    if writer is not None:
        writer.release()


def write_motion_csv(path: Path, frame_motion: dict[int, MotionRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "time_sec",
                "dx",
                "dy",
                "cumulative_dx",
                "cumulative_dy",
                "response",
                "used_fallback",
                "method",
                "segment_id",
            ]
        )
        for frame_index in sorted(frame_motion):
            row = frame_motion[frame_index]
            writer.writerow(
                [
                    row.frame_index,
                    f"{row.time_sec:.2f}",
                    f"{row.dx:.4f}",
                    f"{row.dy:.4f}",
                    f"{row.cumulative_dx:.4f}",
                    f"{row.cumulative_dy:.4f}",
                    f"{row.response:.4f}",
                    int(row.used_fallback),
                    row.method,
                    row.segment_id,
                ]
            )


def write_observations_csv(path: Path, tracks: list[Track]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "book_id",
                "video_id",
                "frame_index",
                "time_sec",
                "instance_id",
                "segment_id",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
                "shelf_center_x",
                "shelf_x1",
                "shelf_x2",
                "match_cost",
                "quality_score",
                "crop_path",
                "mask_path",
            ]
        )
        for book_idx, track in enumerate(tracks, start=1):
            book_id = f"{track.video_id}_book_{book_idx:04d}"
            for det in track.observations:
                writer.writerow(
                    [
                        book_id,
                        det.video_id,
                        det.frame_index,
                        f"{det.time_sec:.2f}",
                        det.instance_id,
                        det.segment_id,
                        f"{det.score:.4f}",
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        f"{det.shelf_center_x:.4f}",
                        f"{det.shelf_x1:.4f}",
                        f"{det.shelf_x2:.4f}",
                        "" if det.match_cost is None else f"{det.match_cost:.4f}",
                        f"{det.quality_score:.4f}",
                        det.crop_path,
                        det.mask_path,
                    ]
                )


def write_tracks_csv(path: Path, tracks: list[Track]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "book_id",
                "video_id",
                "first_frame",
                "last_frame",
                "num_observations",
                "best_crop_path",
                "best_score",
                "best_quality_crop_path",
                "best_quality_score",
                "status",
                "count_as_book",
                "boundary",
                "segment_ids",
                "mean_shelf_center_x",
                "mean_width",
                "mean_height",
            ]
        )
        for book_idx, track in enumerate(tracks, start=1):
            if track.best_detection is None:
                continue
            book_id = f"{track.video_id}_book_{book_idx:04d}"
            mean_shelf_center_x = float(np.mean([det.shelf_center_x for det in track.observations]))
            mean_width = float(np.mean([det.width for det in track.observations]))
            mean_height = float(np.mean([det.height for det in track.observations]))
            best_quality = track.best_quality_detection if track.best_quality_detection is not None else track.best_detection
            segment_ids = sorted({det.segment_id for det in track.observations})
            writer.writerow(
                [
                    book_id,
                    track.video_id,
                    track.first_frame,
                    track.last_frame,
                    track.num_observations,
                    track.best_detection.crop_path,
                    f"{track.best_detection.score:.4f}",
                    best_quality.crop_path,
                    f"{best_quality.quality_score:.4f}",
                    track.status,
                    int(track.count_as_book),
                    int(track.boundary),
                    ",".join(str(seg_id) for seg_id in segment_ids),
                    f"{mean_shelf_center_x:.4f}",
                    f"{mean_width:.4f}",
                    f"{mean_height:.4f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True, help="Job directory such as output/jobs/000")
    parser.add_argument("--max-frame-gap", type=int, default=2, help="Allow tracks to miss this many sampled frames")
    parser.add_argument("--phase-scale", type=float, default=0.5, help="Downscale factor for phase correlation")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Sample FPS used to generate frames/")
    parser.add_argument("--min-phase-response", type=float, default=0.08, help="Minimum phase correlation response")
    parser.add_argument("--max-abs-dx", type=float, help="Reject motion spikes larger than this")
    parser.add_argument("--max-abs-dy", type=float, help="Reject vertical motion spikes larger than this")
    parser.add_argument("--max-center-distance", type=float, help="Maximum center-x delta for a match")
    parser.add_argument("--max-vertical-distance", type=float, help="Maximum center-y delta for a match")
    parser.add_argument("--max-size-ratio", type=float, default=1.9, help="Maximum width/height ratio change for a match")
    parser.add_argument("--min-x-iou", type=float, default=0.12, help="Minimum horizontal overlap for a match")
    parser.add_argument("--min-box-iou", type=float, default=0.02, help="Minimum bbox IoU fallback for a match")
    parser.add_argument("--max-match-cost", type=float, default=1.45, help="Maximum accepted assignment cost")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    detections_csv = job_dir / "detections.csv"
    frames_dir = job_dir / "frames"
    roi_json_path = job_dir / "roi.json"
    motion_csv = job_dir / "frame_motion.csv"
    observations_csv = job_dir / "book_observations.csv"
    tracks_csv = job_dir / "book_tracks.csv"
    tracked_video = job_dir / "tracked_video.mp4"

    if not detections_csv.exists():
        raise SystemExit(f"detections.csv not found: {detections_csv}")
    if not frames_dir.exists():
        raise SystemExit(f"frames directory not found: {frames_dir}")

    frame_paths = list_frame_paths(frames_dir)
    roi_json = load_roi_json(roi_json_path)
    video_id, detections_by_frame = load_detections(detections_csv, job_dir)
    first_frame = cv2.imread(str(frame_paths[sorted(frame_paths)[0]]))
    if first_frame is None:
        raise SystemExit(f"Failed to read frame image: {frame_paths[sorted(frame_paths)[0]]}")
    frame_height, frame_width = first_frame.shape[:2]
    roi_bbox = denormalize_roi_xyxy(roi_json, first_frame.shape) if roi_json is not None else None

    all_detections = [det for rows in detections_by_frame.values() for det in rows]
    median_book_width = float(np.median([det.width for det in all_detections])) if all_detections else 50.0
    roi_width = (roi_bbox[2] - roi_bbox[0]) if roi_bbox is not None else frame_width
    roi_height = (roi_bbox[3] - roi_bbox[1]) if roi_bbox is not None else frame_height
    max_abs_dx = args.max_abs_dx if args.max_abs_dx is not None else 0.25 * roi_width
    max_abs_dy = args.max_abs_dy if args.max_abs_dy is not None else 0.08 * roi_height
    max_center_distance = (
        args.max_center_distance
        if args.max_center_distance is not None
        else max(1.5 * median_book_width, 0.05 * roi_width)
    )
    max_vertical_distance = (
        args.max_vertical_distance
        if args.max_vertical_distance is not None
        else 0.08 * roi_height
    )

    frame_motion = estimate_frame_motion_series(
        frame_paths=frame_paths,
        roi_json=roi_json,
        phase_scale=args.phase_scale,
        min_response=args.min_phase_response,
        max_abs_dx=max_abs_dx,
        max_abs_dy=max_abs_dy,
        sample_fps=args.sample_fps,
    )

    area_norm_base = float(np.percentile([det.width * det.height for det in all_detections], 95)) if all_detections else 1.0
    sharpness_norm_base = float(
        np.percentile([compute_crop_sharpness(det.crop_abs_path) for det in all_detections], 95)
    ) if all_detections else 1.0
    for det in all_detections:
        compute_detection_quality(
            det=det,
            roi_bbox=roi_bbox,
            frame_width=frame_width,
            area_norm_base=area_norm_base,
            sharpness_norm_base=sharpness_norm_base,
        )

    tracks = link_tracks(
        detections_by_frame=detections_by_frame,
        frame_motion=frame_motion,
        max_frame_gap=args.max_frame_gap,
        max_center_distance=max_center_distance,
        max_vertical_distance=max_vertical_distance,
        max_size_ratio=args.max_size_ratio,
        min_x_iou=args.min_x_iou,
        min_box_iou=args.min_box_iou,
        max_match_cost=args.max_match_cost,
    )

    tracks.sort(key=lambda track: np.mean([det.shelf_center_x for det in track.observations]))
    finalize_track_states(tracks, roi_bbox=roi_bbox, frame_width=frame_width)

    write_motion_csv(motion_csv, frame_motion)
    write_observations_csv(observations_csv, tracks)
    write_tracks_csv(tracks_csv, tracks)
    write_tracked_video(
        path=tracked_video,
        frame_paths=frame_paths,
        tracks=tracks,
        roi_bbox=roi_bbox,
        sample_fps=args.sample_fps,
        max_frame_gap=args.max_frame_gap,
    )

    total_detections = sum(len(rows) for rows in detections_by_frame.values())
    merged_duplicates = total_detections - len(tracks)
    valid_tracks = [track for track in tracks if track.count_as_book]
    singleton_tracks = [track for track in tracks if track.num_observations == 1]
    boundary_tracks = [track for track in tracks if track.boundary]
    low_response_frames = sum(1 for row in frame_motion.values() if row.response < args.min_phase_response)
    fallback_frames = sum(1 for row in frame_motion.values() if row.used_fallback)
    print(f"job dir           -> {job_dir}")
    print(f"frame motion csv  -> {motion_csv}")
    print(f"observations csv  -> {observations_csv}")
    print(f"book tracks csv   -> {tracks_csv}")
    print(f"tracked video     -> {tracked_video}")
    print(f"frames processed  : {len(frame_paths)}")
    print(f"detections total  : {total_detections}")
    print(f"track count       : {len(tracks)}")
    print(f"valid books       : {len(valid_tracks)}")
    print(f"merged duplicates : {merged_duplicates}")
    print(f"singleton tracks  : {len(singleton_tracks)}")
    print(f"boundary tracks   : {len(boundary_tracks)}")
    print(f"low-response frames: {low_response_frames}")
    print(f"fallback frames   : {fallback_frames}")


if __name__ == "__main__":
    main()
