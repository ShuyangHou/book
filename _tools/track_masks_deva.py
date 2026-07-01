"""Track book-spine masks across DEVA frame sequences.

Reads MaskDINO hypotheses from export_maskdino_hypotheses.py and runs DEVA
mask propagation to produce stable object_ids.

Input:
    output/jobs/<video_id>/
      deva_frames/               — ROI frames at deva_fps
      frame_manifest.csv          — frame index mapping
      maskdino_hypotheses/        — *.npz per frame
      roi.json                    — ROI offset for full-frame mapping

Output:
    output/jobs/<video_id>/
      tracked_mask_video.mp4     — mask-only visualization with object_id
      mask_observations.csv
      unique_books.csv
      tracking_summary.json
      books/                     — best crop per confirmed object
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
# data
# ---------------------------------------------------------------------------

@dataclass
class MaskObservation:
    video_id: str
    deva_object_id: int
    book_id: str
    deva_frame_index: int
    source_frame_index: int
    time_sec: float
    source_instance_id: int
    detection_score: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask_path: str
    bbox_area: int
    mask_area: int
    mask_fill_ratio: float
    touches_left_edge: bool
    touches_right_edge: bool
    sharpness: float
    crop_area: int
    # quality
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
# helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def load_roi_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    roi = json.loads(path.read_text(encoding="utf-8"))
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(roi):
        raise SystemExit(f"ROI json missing keys: {sorted(required - set(roi))}")
    return roi


def compute_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


# ---------------------------------------------------------------------------
# DEVA
# ---------------------------------------------------------------------------

def load_deva_model(weights_path: str, device: str, max_missed: int,
                    chunk_size: int, image_size: int) -> tuple[DEVAInferenceCore, dict]:
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
        "chunk_size": chunk_size,
        "max_missed_detection_count": max_missed,
        "max_num_objects": -1,
        "size": image_size,
    }
    network = DEVA(config).to(device).eval()
    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    network.load_weights(model_weights)
    print(f"DEVA model loaded ({sum(p.numel() for p in network.parameters()) / 1e6:.1f}M params)")

    deva = DEVAInferenceCore(network, config=config)
    deva.enabled_long_id()
    return deva, config


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

def normalize_quality_scores(observations: list[MaskObservation]) -> None:
    sharpness_vals = [o.sharpness for o in observations if o.sharpness > 0]
    area_vals = [float(o.crop_area) for o in observations if o.crop_area > 0]
    height_vals = [float(o.height) for o in observations if o.height > 0]

    s_base = float(np.percentile(sharpness_vals, 95)) if sharpness_vals else 1.0
    a_base = float(np.percentile(area_vals, 95)) if area_vals else 1.0
    h_base = float(np.percentile(height_vals, 95)) if height_vals else 1.0

    for o in observations:
        o.normalized_sharpness = min(1.0, o.sharpness / max(s_base, 1.0))
        o.normalized_area = min(1.0, o.crop_area / max(a_base, 1.0))
        o.normalized_height = min(1.0, o.height / max(h_base, 1.0))
        edge = o.touches_left_edge or o.touches_right_edge
        o.edge_penalty = 0.15 if edge else 0.0
        o.completeness_score = 0.5 * (0.0 if edge else 1.0) + 0.5 * o.normalized_height
        o.quality_score = (
            0.35 * o.normalized_sharpness
            + 0.25 * o.detection_score
            + 0.15 * o.normalized_area
            + 0.15 * o.mask_fill_ratio
            + 0.10 * o.completeness_score
            - o.edge_penalty
        )


def determine_status(observations: list[MaskObservation]) -> tuple[str, int]:
    valid = [o for o in observations if o.valid]
    if not valid:
        return "invalid", 0
    if len(valid) >= 2:
        if any(not o.touches_left_edge and not o.touches_right_edge for o in valid):
            return "confirmed", 1
        return "boundary_review", 0
    if len(valid) == 1:
        return "singleton_review", 0
    return "invalid", 0


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

def object_color(obj_id: int) -> tuple[int, int, int]:
    state = np.random.RandomState(int(obj_id) % 10007)
    return (int(state.randint(60, 220)), int(state.randint(60, 220)), int(state.randint(60, 220)))


def status_color(status: str) -> tuple[int, int, int]:
    if status == "confirmed":
        return (55, 200, 90)
    if status == "boundary_review":
        return (255, 180, 60)
    if status == "singleton_review":
        return (70, 160, 255)
    return (255, 110, 110)


def write_tracked_mask_video(
    path: Path,
    deva_frames_dir: Path,
    frame_order: list[int],
    observations_by_frame: dict[int, list[MaskObservation]],
    statuses: dict[int, tuple[str, int]],
    fps: float,
    orig_w: int, orig_h: int,
) -> None:
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    obj_first_frame: dict[int, int] = {}
    for rows in observations_by_frame.values():
        for o in rows:
            if o.valid:
                obj_first_frame[o.deva_object_id] = min(
                    obj_first_frame.get(o.deva_object_id, o.deva_frame_index), o.deva_frame_index)

    for dfi in frame_order:
        frame_path = deva_frames_dir / f"{dfi:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        vis = frame.copy()

        rows = sorted(observations_by_frame.get(dfi, []), key=lambda r: r.bbox_area, reverse=True)
        for o in rows:
            color = object_color(o.deva_object_id)
            # Read mask from NPZ? No — we need per-object mask.
            # Instead, use detection mask from original MaskDINO output if available,
            # or overlay the bbox region as fallback.
            mask_img = cv2.imread(str(o.mask_path), cv2.IMREAD_GRAYSCALE) if o.mask_path else None
            if mask_img is not None and mask_img.shape[:2] == vis.shape[:2]:
                mask_bool = mask_img > 127
                vis[mask_bool] = (vis[mask_bool] * 0.45 + np.array(color[::-1]) * 0.55).astype(np.uint8)
            else:
                # Fallback: tint the bbox region lightly
                x1, y1, x2, y2 = max(0, o.x1), max(0, o.y1), min(vis.shape[1], o.x2), min(vis.shape[0], o.y2)
                if x2 > x1 and y2 > y1:
                    vis[y1:y2, x1:x2] = (vis[y1:y2, x1:x2] * 0.55 + np.array(color[::-1]) * 0.45).astype(np.uint8)

        # Labels
        for o in sorted(rows, key=lambda r: r.x1):
            status, _ = statuses.get(o.deva_object_id, ("unknown", 0))
            label = f"{o.book_id}"
            if status != "confirmed":
                label = f"{label} [{status}]"
            tx = max(2, o.x1)
            ty = max(16, o.y1 - 4)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color(status), 1)

        # HUD
        active = {o.deva_object_id for o in rows}
        started = {oid for oid, ff in obj_first_frame.items() if ff <= dfi}
        conf = sum(1 for oid in started if statuses.get(oid, ("invalid", 0))[0] == "confirmed")
        review = sum(1 for oid in started if statuses.get(oid, ("invalid", 0))[0] in {"boundary_review", "singleton_review"})

        for i, line in enumerate([
            f"frame: {dfi:06d}",
            f"active: {len(active)}",
            f"confirmed: {conf}",
            f"review: {review}",
            f"unique objects: {len(started)}",
        ]):
            cv2.putText(vis, line, (12, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(vis, line, (12, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if writer is None:
            vh, vw = vis.shape[:2]
            writer = cv2.VideoWriter(str(path), fourcc, fps, (vw, vh))
            if not writer.isOpened():
                raise SystemExit(f"Failed to create video: {path}")
        writer.write(vis)

    if writer is not None:
        writer.release()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_mask_observations_csv(p: Path, obs: list[MaskObservation]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "book_id", "video_id", "deva_object_id",
            "deva_frame_index", "source_frame_index", "time_sec",
            "source_instance_id", "score",
            "x1", "y1", "x2", "y2",
            "mask_path",
            "quality_score", "touches_left_edge", "touches_right_edge",
            "sharpness", "crop_area", "mask_fill_ratio",
        ])
        for o in sorted(obs, key=lambda r: (r.deva_frame_index, r.deva_object_id)):
            w.writerow([
                o.book_id, o.video_id, o.deva_object_id,
                o.deva_frame_index, o.source_frame_index, f"{o.time_sec:.2f}",
                o.source_instance_id, f"{o.detection_score:.4f}",
                o.x1, o.y1, o.x2, o.y2,
                o.mask_path,
                f"{o.quality_score:.4f}",
                int(o.touches_left_edge), int(o.touches_right_edge),
                f"{o.sharpness:.4f}", o.crop_area,
                f"{o.mask_fill_ratio:.4f}",
            ])


def write_unique_books_csv(
    p: Path,
    tracks: dict[int, list[MaskObservation]],
    statuses: dict[int, tuple[str, int]],
    best_obs: dict[int, MaskObservation],
) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "book_id", "video_id", "deva_object_id",
            "first_frame", "last_frame", "first_time_sec", "last_time_sec",
            "num_observations", "status", "count_as_book",
            "best_score", "mean_width", "mean_height",
        ])
        for oid in sorted(tracks):
            ol = [o for o in tracks[oid] if o.valid]
            if ol:
                ol.sort(key=lambda r: (r.deva_frame_index, r.source_instance_id))
                best = best_obs[oid]
                status, cab = statuses[oid]
                mw = float(np.mean([o.width for o in ol]))
                mh = float(np.mean([o.height for o in ol]))
                w.writerow([
                    ol[0].book_id, ol[0].video_id, oid,
                    ol[0].deva_frame_index, ol[-1].deva_frame_index,
                    f"{ol[0].time_sec:.2f}", f"{ol[-1].time_sec:.2f}",
                    len(ol), status, cab,
                    f"{best.quality_score:.4f}", f"{mw:.4f}", f"{mh:.4f}",
                ])
            else:
                w.writerow([f"unknown_{oid:04d}", "", oid, "", "", "", "", 0, "invalid", 0, "", "", ""])


def write_tracking_summary(
    p: Path, video_id: str, frame_count: int, hyp_total: int,
    objects_total: int, statuses: dict[int, tuple[str, int]],
    config: dict, suspected_merged: int,
) -> None:
    s = {
        "video_id": video_id,
        "tracker": "DEVA",
        "frames_processed": frame_count,
        "hypotheses_total": hyp_total,
        "deva_objects_total": objects_total,
        "confirmed_books": sum(1 for st, _ in statuses.values() if st == "confirmed"),
        "boundary_review": sum(1 for st, _ in statuses.values() if st == "boundary_review"),
        "singleton_review": sum(1 for st, _ in statuses.values() if st == "singleton_review"),
        "invalid_tracks": sum(1 for st, _ in statuses.values() if st == "invalid"),
        "suspected_merged_masks": suspected_merged,
        "max_missed_detection_count": config.get("max_missed_detection_count"),
    }
    p.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DEVA mask tracking for book spines")
    parser.add_argument("--job-dir", required=True, help="Job directory from export_maskdino_hypotheses.py")
    parser.add_argument("--deva-weights", default=str(DEVA_ROOT / "saves" / "DEVA-propagation.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-missed-detection-count", type=int, default=9,
                        help="Frames before undetected objects are purged")
    parser.add_argument("--image-size", type=int, default=480,
                        help="DEVA shorter-side resize; -1 for original resolution")
    parser.add_argument("--chunk-size", type=int, default=4,
                        help="Objects per forward-pass batch; reduce if OOM")
    parser.add_argument("--detection-score-threshold", type=float, default=0.0,
                        help="Only inject hypots above this score; 0 = use all saved hypots")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    deva_frames_dir = job_dir / "deva_frames"
    hypotheses_dir = job_dir / "maskdino_hypotheses"
    manifest_csv = job_dir / "frame_manifest.csv"
    roi_json_path = job_dir / "roi.json"
    tracked_video_path = job_dir / "tracked_mask_video.mp4"
    observations_csv_path = job_dir / "mask_observations.csv"
    unique_books_csv_path = job_dir / "unique_books.csv"
    summary_json_path = job_dir / "tracking_summary.json"
    books_dir = job_dir / "books"

    for p in [manifest_csv, deva_frames_dir, hypotheses_dir]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    ensure_dir(books_dir)

    # --- Load manifest ---
    video_id = None
    frame_order: list[int] = []
    frame_info: dict[int, dict] = {}
    with manifest_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video_id = video_id or row["video_id"]
            dfi = int(row["deva_frame_index"])
            frame_order.append(dfi)
            frame_info[dfi] = row
    frame_order.sort()
    if video_id is None:
        raise SystemExit("Empty manifest")

    # Read first frame for dimensions
    first_frame = cv2.imread(str(deva_frames_dir / f"{frame_order[0]:06d}.jpg"))
    if first_frame is None:
        raise SystemExit("Failed to read first DEVA frame")
    frame_h, frame_w = first_frame.shape[:2]

    # Determine processing resolution
    if args.image_size > 0:
        scale = args.image_size / min(frame_h, frame_w)
        proc_h = (int(round(frame_h * scale)) // 16) * 16
        proc_w = (int(round(frame_w * scale)) // 16) * 16
        print(f"DEVA processing res: {proc_w}x{proc_h} (scale={scale:.3f})")
    else:
        proc_h, proc_w = frame_h, frame_w
        scale = 1.0

    # ROI offset for full-frame coordinate restoration (future use)
    roi_json = load_roi_json(roi_json_path)

    # --- Load DEVA ---
    deva, config = load_deva_model(
        args.deva_weights, args.device,
        max_missed=args.max_missed_detection_count,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
    )
    deva_fps = None  # Will be inferred from frame times

    # --- Process frames ---
    all_observations: list[MaskObservation] = []
    observations_by_obj: dict[int, list[MaskObservation]] = defaultdict(list)
    observations_by_frame: dict[int, list[MaskObservation]] = defaultdict(list)

    edge_margin = max(8, int(0.01 * frame_w))
    total_hypotheses = 0
    suspected_merged_total = 0
    skipped_frames = 0

    for dfi in frame_order:
        info = frame_info[dfi]
        time_sec = float(info["time_sec"])
        source_frame_index = int(info["source_frame_index"])
        has_hypotheses = info.get("has_maskdino_hypotheses", "0") == "1"

        # Read frame
        frame = cv2.imread(str(deva_frames_dir / f"{dfi:06d}.jpg"))
        if frame is None:
            print(f"WARNING: skipping frame {dfi}, failed to read")
            skipped_frames += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for DEVA
        if scale != 1.0:
            frame_proc = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_proc = frame_rgb
        frame_tensor = torch.from_numpy(frame_proc).permute(2, 0, 1).float().to(args.device)

        # Load hypotheses if injection frame
        npz_path = hypotheses_dir / f"{dfi:06d}.npz"
        hyp_data = None
        if npz_path.exists():
            hyp_data = np.load(npz_path, allow_pickle=True)
            masks_arr = hyp_data.get("masks")
            scores_arr = hyp_data.get("scores")
            sids_arr = hyp_data.get("source_instance_ids")
            if masks_arr is not None and masks_arr.ndim == 3 and masks_arr.shape[0] > 0:
                has_hypotheses = True
            else:
                has_hypotheses = False

        if has_hypotheses and hyp_data is not None:
            masks_arr = hyp_data["masks"]
            scores_arr = hyp_data["scores"]
            sids_arr = hyp_data["source_instance_ids"]

            # Filter by score threshold
            keep = scores_arr >= args.detection_score_threshold
            if not np.all(keep):
                masks_arr = masks_arr[keep]
                scores_arr = scores_arr[keep]
                sids_arr = sids_arr[keep]

            total_hypotheses += len(scores_arr) if len(scores_arr) > 0 else 0

            if len(scores_arr) > 0:
                # Compose ID mask at processing resolution
                id_mask = np.zeros((proc_h, proc_w), dtype=np.int32)
                segments_info: list[ObjectInfo] = []
                # Sort by area descending: small masks rendered on top
                areas = [int(m.sum()) for m in masks_arr]
                order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

                for idx in order:
                    mask = masks_arr[idx].astype(bool)
                    sid = int(sids_arr[idx]) if len(sids_arr) > idx else idx + 1
                    score = float(scores_arr[idx]) if len(scores_arr) > idx else 0.0

                    if scale != 1.0:
                        mask_u8 = mask.astype(np.uint8) * 255
                        mask_u8 = cv2.resize(mask_u8, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                        mask = mask_u8 > 127

                    obj_info = ObjectInfo(id=sid, score=score)
                    segments_info.append(obj_info)
                    id_mask[mask] = sid

                id_mask_t = torch.from_numpy(id_mask).to(args.device)

                with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                    pred_prob = deva.incorporate_detection(
                        image=frame_tensor,
                        new_mask=id_mask_t,
                        segments_info=segments_info,
                        incremental=True,
                    )
            else:
                # Empty hypotheses but frame has NPZ
                if deva.memory.engaged:
                    with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                        pred_prob = deva.step(frame_tensor, mask=None, end=False)
                else:
                    continue
        else:
            # Propagation-only frame
            if deva.memory.engaged:
                with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                    pred_prob = deva.step(frame_tensor, mask=None, end=False)
            else:
                continue

        # Map DEVA objects back to frame detections
        obj_to_tmp = deva.object_manager.obj_to_tmp_id

        # Get argmax mask at processed resolution
        pred_mask_proc = torch.argmax(pred_prob, dim=0).cpu().numpy().astype(np.int32)

        # Resize back to original frame resolution for observation recording
        if scale != 1.0:
            pred_mask_full = cv2.resize(pred_mask_proc, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask_full = pred_mask_proc

        # Load original masks (at frame resolution) for IoU matching
        hyp_masks_orig = {}  # source_instance_id -> binary mask at original res
        if has_hypotheses and hyp_data is not None:
            try:
                masks_orig = hyp_data["masks"]  # N x H x W at original resolution
                sids_orig = hyp_data["scores"]  # actually scores array
                sids_ids = hyp_data["source_instance_ids"]
                for i in range(len(masks_orig)):
                    sid = int(sids_ids[i]) if i < len(sids_ids) else i + 1
                    score = float(sids_orig[i]) if i < len(sids_orig) else 0.0
                    if score >= args.detection_score_threshold:
                        hyp_masks_orig[sid] = masks_orig[i].astype(bool)
            except Exception:
                pass

        # Build observations per DEVA object this frame
        current_obs = []
        for obj_info, tmp_id in obj_to_tmp.items():
            obj_id = obj_info.id
            obj_mask = (pred_mask_full == tmp_id)
            obj_area = int(obj_mask.sum())
            if obj_area == 0:
                continue

            # Find best matching source instance
            best_iou = 0.0
            best_sid = 0
            best_score = 0.0
            best_mask_path = ""
            for sid, smask in hyp_masks_orig.items():
                inter = int((obj_mask & smask).sum())
                union = int((obj_mask | smask).sum())
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    best_sid = sid
                    best_score = float(obj_info.vote_score() or 0.0)

            if best_iou < 0.2:
                # Object propagated but no matching detection (edge case)
                continue

            bbox = mask_to_bbox(obj_mask.astype(np.uint8))
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue

            touches_left = x1 <= edge_margin
            touches_right = x2 >= frame_w - edge_margin

            # Crop from frame for sharpness
            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else np.zeros((1, 1, 3), dtype=np.uint8)
            sharpness = compute_sharpness(crop)
            crop_area = int(crop.shape[0] * crop.shape[1]) if crop.size > 0 else 0

            mask_fill = obj_area / max(1, (x2 - x1) * (y2 - y1))
            bbox_area = max(1, (x2 - x1) * (y2 - y1))

            valid = bbox_area >= 32 and crop_area > 0

            book_id = f"{video_id}_book_{obj_id:04d}"

            # Mask path: save a PNG of the object mask for video overlay
            mask_save_path = job_dir / "deva_output_masks" / f"{dfi:06d}_{obj_id:04d}.png"
            ensure_dir(mask_save_path.parent)
            cv2.imwrite(str(mask_save_path), (obj_mask.astype(np.uint8) * 255))

            obs = MaskObservation(
                video_id=video_id,
                deva_object_id=obj_id,
                book_id=book_id,
                deva_frame_index=dfi,
                source_frame_index=source_frame_index,
                time_sec=time_sec,
                source_instance_id=best_sid,
                detection_score=best_score,
                x1=x1, y1=y1, x2=x2, y2=y2,
                mask_path=str(mask_save_path),
                bbox_area=bbox_area,
                mask_area=obj_area,
                mask_fill_ratio=mask_fill,
                touches_left_edge=touches_left,
                touches_right_edge=touches_right,
                sharpness=sharpness,
                crop_area=crop_area,
                valid=valid,
            )
            current_obs.append(obs)
            observations_by_obj[obj_id].append(obs)
            observations_by_frame[dfi].append(obs)
            all_observations.append(obs)

        if hyp_data is not None:
            try:
                hyp_data.close()
            except Exception:
                pass

        print(f"deva_frame={dfi:06d} "
              f"inject={'Y' if has_hypotheses else 'N'} "
              f"tracked={len(current_obs)} "
              f"total_objects={len(observations_by_obj)}")

    # --- Post-processing ---
    normalize_quality_scores(all_observations)

    statuses = {oid: determine_status(ol) for oid, ol in observations_by_obj.items()}

    # Best crop per object
    best_observations: dict[int, MaskObservation] = {}
    for oid, ol in observations_by_obj.items():
        valid_list = [o for o in ol if o.valid]
        if valid_list:
            best = max(valid_list, key=lambda o: (o.quality_score, o.detection_score, o.crop_area))
            best_observations[oid] = best
            # Save best crop
            frame_img = cv2.imread(str(deva_frames_dir / f"{best.deva_frame_index:06d}.jpg"))
            if frame_img is not None:
                best_crop = frame_img[best.y1:best.y2, best.x1:best.x2]
                cv2.imwrite(str(books_dir / f"{best.book_id}.jpg"), best_crop)

    # --- Output ---
    write_mask_observations_csv(observations_csv_path, all_observations)
    write_unique_books_csv(unique_books_csv_path, observations_by_obj, statuses, best_observations)
    write_tracking_summary(
        summary_json_path, video_id,
        frame_count=len(frame_order) - skipped_frames,
        hyp_total=total_hypotheses,
        objects_total=len(observations_by_obj),
        statuses=statuses,
        config=config,
        suspected_merged=suspected_merged_total,
    )
    write_tracked_mask_video(
        tracked_video_path,
        deva_frames_dir,
        frame_order,
        observations_by_frame,
        statuses,
        fps=deva_fps or 3.0,
        orig_w=frame_w,
        orig_h=frame_h,
    )

    # --- Summary ---
    confirmed = sum(1 for s, _ in statuses.values() if s == "confirmed")
    boundary = sum(1 for s, _ in statuses.values() if s == "boundary_review")
    singleton = sum(1 for s, _ in statuses.values() if s == "singleton_review")
    invalid = sum(1 for s, _ in statuses.values() if s == "invalid")

    print(f"\n{'='*50}")
    print(f"job dir              -> {job_dir}")
    print(f"tracked mask video   -> {tracked_video_path}")
    print(f"mask observations    -> {observations_csv_path}")
    print(f"unique books csv     -> {unique_books_csv_path}")
    print(f"tracking summary     -> {summary_json_path}")
    print(f"books dir            -> {books_dir}")
    print(f"DEVA frames processed: {len(frame_order) - skipped_frames}")
    print(f"total hypotheses     : {total_hypotheses}")
    print(f"DEVA objects total   : {len(observations_by_obj)}")
    print(f"confirmed            : {confirmed}")
    print(f"boundary_review      : {boundary}")
    print(f"singleton_review     : {singleton}")
    print(f"invalid              : {invalid}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
