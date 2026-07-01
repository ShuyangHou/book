"""Run MaskDINO book-spine inference on a single video at fixed sample FPS.

Pipeline:
    MP4 video
    -> sample frames at fixed FPS
    -> run MaskDINO per sampled frame
    -> save raw frames, masks, crops, visualization frames
    -> write detections.csv
    -> assemble annotated_video.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def crop_with_bbox(image: np.ndarray, bbox: list[int], padding: int) -> tuple[np.ndarray, list[int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2], [x1, y1, x2, y2]


def relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def normalize_roi_xyxy(x1: int, y1: int, x2: int, y2: int, frame_shape: tuple[int, int, int]) -> dict:
    h, w = frame_shape[:2]
    return {
        "x1": max(0.0, min(1.0, x1 / w)),
        "y1": max(0.0, min(1.0, y1 / h)),
        "x2": max(0.0, min(1.0, x2 / w)),
        "y2": max(0.0, min(1.0, y2 / h)),
    }


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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok and frame is not None:
        cap.release()
        return frame

    # Some OpenCV backends, especially under WSL, can fail random frame seeks on MP4.
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
            candidates.append(
                {
                    "candidate_index": candidate_idx,
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "path": path,
                }
            )
            target_ptr += 1

        current_index += 1

    cap.release()

    if len(candidates) != len(unique_indices):
        missing = unique_indices[len(candidates):]
        raise SystemExit(
            f"Failed to export ROI candidate frames for indices {missing} from video: {video_path}"
        )
    return candidates


def choose_roi_candidate(candidates: list[dict], requested_index: int | None) -> dict:
    if not candidates:
        raise SystemExit("No ROI candidates were exported.")

    valid_indices = {row["candidate_index"] for row in candidates}
    if requested_index is not None:
        if requested_index not in valid_indices:
            raise SystemExit(
                f"Invalid --roi-candidate-index={requested_index}; "
                f"choose from {sorted(valid_indices)}"
            )
        return next(row for row in candidates if row["candidate_index"] == requested_index)

    print("ROI candidate frames:")
    for row in candidates:
        print(
            f"  [{row['candidate_index']}] frame={row['frame_index']:06d} "
            f"time={row['time_sec']:.2f}s path={row['path']}"
        )

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


def draw_roi_outline(image: np.ndarray, roi_bbox: list[int] | None) -> np.ndarray:
    if roi_bbox is None:
        return image
    vis = image.copy()
    x1, y1, x2, y2 = roi_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 215, 255), 2)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, "ROI", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 215, 255), 1)
    return vis


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


def iter_sampled_frames(video_path: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_fps <= 0:
        raise SystemExit(f"Invalid source FPS for video: {video_path}")

    duration = frame_count / src_fps if frame_count > 0 else 0.0
    print(
        f"video={video_path.name} src_fps={src_fps:.3f} "
        f"frames={frame_count} duration={duration:.2f}s sample_fps={sample_fps:.3f}"
    )

    interval = 1.0 / sample_fps
    next_sample_time = 0.0
    sample_index = 0
    source_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        time_sec = source_index / src_fps
        if time_sec + 1e-9 >= next_sample_time:
            yield sample_index, source_index, time_sec, frame
            sample_index += 1
            next_sample_time += interval

        source_index += 1

    cap.release()


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (55, 200, 90),
        (65, 140, 255),
        (255, 170, 50),
        (220, 90, 200),
        (70, 220, 220),
        (255, 110, 110),
    ]
    return palette[(index - 1) % len(palette)]


def draw_predictions(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    overlay = image.copy()
    vis = image.copy()

    for det in detections:
        color = color_for_index(det["instance_id"])
        mask = det["mask"]
        overlay[mask] = (overlay[mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

    vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0.0)

    for det in detections:
        color = color_for_index(det["instance_id"])
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        contours, _ = cv2.findContours(
            det["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 1)

        label = f"#{det['instance_id']} {det['score']:.3f}"
        text_origin = (x1, max(18, y1 - 6))
        cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="Trained model checkpoint path")
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help="MaskDINO config path",
    )
    parser.add_argument("--video-id", help="Video id used in output paths and CSV")
    parser.add_argument(
        "--job-dir",
        help="Job output directory. Defaults to output/jobs/<video_id>",
    )
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Sample video at this FPS")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Instance score threshold")
    parser.add_argument("--padding", type=int, default=6, help="Extra crop padding around predicted bbox")
    parser.add_argument("--select-roi", action="store_true", help="Interactively select a target shelf ROI")
    parser.add_argument("--roi-json", help="Path to normalized ROI json; defaults to <job_dir>/roi.json when ROI is used")
    parser.add_argument(
        "--roi-source-frame",
        choices=("first", "middle"),
        default="middle",
        help="Which frame to use for interactive ROI selection",
    )
    parser.add_argument(
        "--roi-candidate-count",
        type=int,
        default=0,
        help="Export this many evenly spaced ROI candidate frames before selection",
    )
    parser.add_argument(
        "--roi-candidate-index",
        type=int,
        help="Choose one exported ROI candidate by index and skip the terminal prompt",
    )
    parser.add_argument(
        "--roi-candidate-dir",
        help="Where to save exported ROI candidate frames; defaults to <job_dir>/roi_candidates",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="OpenCV fourcc codec for annotated video, default: mp4v",
    )
    args = parser.parse_args()

    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be > 0")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    video_id = args.video_id or video_path.stem
    job_dir = Path(args.job_dir) if args.job_dir else Path("output") / "jobs" / video_id
    frames_dir = job_dir / "frames"
    vis_dir = job_dir / "visualizations"
    masks_dir = job_dir / "masks"
    crops_dir = job_dir / "crops"
    detections_csv = job_dir / "detections.csv"
    annotated_video = job_dir / "annotated_video.mp4"
    roi_json_path = Path(args.roi_json) if args.roi_json else job_dir / "roi.json"
    roi_candidate_dir = Path(args.roi_candidate_dir) if args.roi_candidate_dir else job_dir / "roi_candidates"

    for path in (job_dir, frames_dir, vis_dir, masks_dir, crops_dir):
        ensure_dir(path)

    normalized_roi = None
    if args.select_roi:
        if args.roi_candidate_count > 0:
            candidates = export_roi_candidate_frames(video_path, roi_candidate_dir, args.roi_candidate_count)
            chosen_candidate = choose_roi_candidate(candidates, args.roi_candidate_index)
            print(
                f"selected candidate -> index={chosen_candidate['candidate_index']} "
                f"frame={chosen_candidate['frame_index']:06d} "
                f"time={chosen_candidate['time_sec']:.2f}s"
            )
            reference_frame = read_frame_at_index(video_path, chosen_candidate["frame_index"])
            print(f"roi candidates    -> {roi_candidate_dir}")
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

    predictor = build_predictor(
        weights=args.weights,
        config_file=args.config_file,
        score_threshold=args.score_threshold,
    )

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)

    with detections_csv.open("w", encoding="utf-8-sig", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
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
            ]
        )

        processed_frames = 0
        total_instances = 0

        for frame_index, source_index, time_sec, frame in iter_sampled_frames(video_path, args.sample_fps):
            frame_name = f"{frame_index:06d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            roi_bbox = denormalize_roi_xyxy(normalized_roi, frame.shape) if normalized_roi is not None else None
            roi_frame = frame
            roi_offset_x = 0
            roi_offset_y = 0
            if roi_bbox is not None:
                roi_offset_x, roi_offset_y = roi_bbox[0], roi_bbox[1]
                roi_frame = frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]]

            outputs = predictor(roi_frame)
            instances = outputs["instances"].to("cpu")
            scores = instances.scores.tolist() if instances.has("scores") else []
            boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

            frame_detections = []
            for raw_idx, score in enumerate(scores, start=1):
                if float(score) < args.score_threshold:
                    continue

                bbox = [int(round(v)) for v in boxes[raw_idx - 1]]
                mask = None
                if masks is not None:
                    mask = masks[raw_idx - 1].astype(bool)
                    bbox = mask_to_bbox(mask)
                else:
                    x1, y1, x2, y2 = bbox
                    mask = np.zeros(roi_frame.shape[:2], dtype=bool)
                    mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True

                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                bbox = [
                    bbox[0] + roi_offset_x,
                    bbox[1] + roi_offset_y,
                    bbox[2] + roi_offset_x,
                    bbox[3] + roi_offset_y,
                ]
                full_mask = np.zeros(frame.shape[:2], dtype=bool)
                full_mask[
                    roi_offset_y:roi_offset_y + mask.shape[0],
                    roi_offset_x:roi_offset_x + mask.shape[1],
                ] = mask

                frame_detections.append(
                    {
                        "score": float(score),
                        "bbox": bbox,
                        "mask": full_mask,
                    }
                )

            frame_detections.sort(key=lambda det: det["bbox"][0])

            for instance_id, det in enumerate(frame_detections, start=1):
                mask_name = f"{frame_index:06d}_{instance_id:03d}.png"
                crop_name = f"{frame_index:06d}_{instance_id:03d}.jpg"
                mask_path = masks_dir / mask_name
                crop_path = crops_dir / crop_name

                crop, _ = crop_with_bbox(frame, det["bbox"], args.padding)
                cv2.imwrite(str(mask_path), det["mask"].astype(np.uint8) * 255)
                cv2.imwrite(str(crop_path), crop)

                det["instance_id"] = instance_id
                det["mask_path"] = relative_posix(mask_path, job_dir)
                det["crop_path"] = relative_posix(crop_path, job_dir)

                csv_writer.writerow(
                    [
                        video_id,
                        frame_index,
                        f"{time_sec:.2f}",
                        instance_id,
                        f"{det['score']:.4f}",
                        det["bbox"][0],
                        det["bbox"][1],
                        det["bbox"][2],
                        det["bbox"][3],
                        det["mask_path"],
                        det["crop_path"],
                    ]
                )

            vis = draw_predictions(frame, frame_detections)
            vis = draw_roi_outline(vis, roi_bbox)
            vis_name = f"{frame_index:06d}.jpg"
            vis_path = vis_dir / vis_name
            cv2.imwrite(str(vis_path), vis)

            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(annotated_video), fourcc, args.sample_fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"Failed to create output video: {annotated_video}")

            writer.write(vis)
            processed_frames += 1
            total_instances += len(frame_detections)
            print(
                f"[{processed_frames}] sample_frame={frame_index:06d} "
                f"source_frame={source_index:06d} time={time_sec:.2f}s "
                f"detections={len(frame_detections)}"
            )

    if writer is not None:
        writer.release()

    print(f"frames dir        -> {frames_dir}")
    print(f"visualizations dir-> {vis_dir}")
    print(f"annotated video   -> {annotated_video}")
    print(f"detections csv    -> {detections_csv}")
    if normalized_roi is not None:
        print(f"roi json          -> {roi_json_path}")
    print(f"images processed: {processed_frames}")
    print(f"instances total : {total_instances}")


if __name__ == "__main__":
    main()
