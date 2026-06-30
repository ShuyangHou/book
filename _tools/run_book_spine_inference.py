"""Run book-spine segmentation inference on selected keyframes.

This script is the inference-side companion to `train_maskdino_r50.py`.
It expects a working Detectron2/MaskDINO environment plus a trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from train_maskdino_r50 import MASKDINO_ROOT, build_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask) -> list[int]:
    ys, xs = mask.nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def crop_with_bbox(image, bbox: list[int], padding: int):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2], [x1, y1, x2, y2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing selected keyframes")
    parser.add_argument("--glob", default="*.jpg", help="Keyframe glob pattern")
    parser.add_argument("--weights", required=True, help="Trained model checkpoint path")
    parser.add_argument(
        "--config-file",
        default=str(
            MASKDINO_ROOT
            / "configs"
            / "coco"
            / "instance-segmentation"
            / "maskdino_R50_bs16_50ep_3s_dowsample1_2048_bitmask.yaml"
        ),
        help="MaskDINO config path",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Instance score threshold")
    parser.add_argument("--padding", type=int, default=6, help="Extra crop padding around predicted bbox")
    parser.add_argument("--crop-dir", default="output/predicted_crops", help="Where to save per-instance crops")
    parser.add_argument("--vis-dir", default="output/predicted_vis", help="Where to save visualization images")
    parser.add_argument(
        "--detections-jsonl",
        default="output/predictions/book_spine_detections.jsonl",
        help="Prediction manifest jsonl",
    )
    args = parser.parse_args()

    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import ColorMode, Visualizer
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
        weights=args.weights,
        config_file=args.config_file,
        num_workers=2,
        allow_fallback=False,
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.score_threshold
    if hasattr(cfg.MODEL, "PANOPTIC_FPN"):
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.score_threshold
    cfg.DATASETS.TEST = ()

    predictor = DefaultPredictor(cfg)

    input_dir = Path(args.input_dir)
    crop_dir = Path(args.crop_dir)
    vis_dir = Path(args.vis_dir)
    detections_path = Path(args.detections_jsonl)
    ensure_dir(crop_dir)
    ensure_dir(vis_dir)
    ensure_dir(detections_path.parent)

    image_paths = sorted(input_dir.glob(args.glob))
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir} with glob {args.glob}")

    with detections_path.open("w", encoding="utf-8") as out:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"warning: failed to read {image_path}")
                continue

            outputs = predictor(image)
            instances = outputs["instances"].to("cpu")
            scores = instances.scores.tolist() if instances.has("scores") else []
            boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

            detections = []
            for idx, score in enumerate(scores, start=1):
                bbox = [int(round(v)) for v in boxes[idx - 1]]
                if masks is not None:
                    bbox = mask_to_bbox(masks[idx - 1])
                crop, padded_bbox = crop_with_bbox(image, bbox, padding=args.padding)
                crop_name = f"{image_path.stem}_crop_{idx:04d}.jpg"
                crop_path = crop_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                detections.append(
                    {
                        "image_path": str(image_path),
                        "crop_path": str(crop_path),
                        "score": float(score),
                        "bbox_xyxy": padded_bbox,
                    }
                )

            detections.sort(key=lambda row: row["bbox_xyxy"][0])
            for row in detections:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

            vis = Visualizer(
                image[:, :, ::-1],
                metadata=None,
                scale=1.0,
                instance_mode=ColorMode.IMAGE,
            )
            rendered = vis.draw_instance_predictions(instances).get_image()[:, :, ::-1]
            cv2.imwrite(str(vis_dir / image_path.name), rendered)

            print(f"{image_path.name} -> {len(detections)} detections")

    print(f"crop dir         -> {crop_dir}")
    print(f"visualization dir-> {vis_dir}")
    print(f"detections jsonl -> {detections_path}")


if __name__ == "__main__":
    main()
