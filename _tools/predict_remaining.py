# -*- coding: utf-8 -*-
"""Use trained MaskDINO/Mask-RCNN model to predict book spines on remaining frames.

Outputs predictions as Labelme-format JSON files for human review/correction.

Usage:
    python _tools/predict_remaining.py \
        --model-dir output/maskdino_r50_v1 \
        --frames-dir frames \
        --already-labeled frames_pick50 \
        --output-dir frames_remaining_prelabel \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import base64
from pathlib import Path

import cv2
import numpy as np


def load_predictor(model_dir: str, threshold: float):
    """Load a Detectron2 predictor from a trained model directory."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()

    # Try MaskDINO config
    try:
        from maskdino import add_maskdino_config
        add_maskdino_config(cfg)
    except ImportError:
        pass

    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    else:
        # Fallback to Mask R-CNN
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)


def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 2.0) -> list[list[float]]:
    """Convert a binary mask to a polygon (largest contour)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    if simplify_eps > 0:
        contour = cv2.approxPolyDP(contour, simplify_eps, True)
    if len(contour) < 3:
        return []
    return [[float(pt[0][0]), float(pt[0][1])] for pt in contour]


def predict_to_labelme(predictor, img_path: str, threshold: float) -> dict:
    """Run prediction and format as Labelme JSON."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    h, w = img.shape[:2]

    shapes = []
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
    scores = instances.scores.numpy() if instances.has("scores") else []

    for i in range(len(instances)):
        if len(scores) > 0 and scores[i] < threshold:
            continue

        if i < len(masks):
            polygon = mask_to_polygon(masks[i])
            if not polygon:
                continue
        else:
            continue

        shapes.append({
            "label": "book_spine",
            "points": polygon,
            "group_id": None,
            "description": f"auto_score={scores[i]:.3f}" if i < len(scores) else "auto",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })

    # Encode image as base64 for Labelme compatibility
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": img_data,
        "imageHeight": h,
        "imageWidth": w,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--frames-dir", default="frames", help="All frames directory")
    parser.add_argument("--already-labeled", default="frames_pick50", help="Already labeled dir")
    parser.add_argument("--output-dir", default="frames_remaining_prelabel", help="Output dir")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-imagedata", action="store_true",
                        help="Skip embedding imageData (smaller JSON files)")
    args = parser.parse_args()

    # Find frames not yet labeled
    labeled_stems = set()
    for jf in Path(args.already_labeled).glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        if data.get("shapes"):
            labeled_stems.add(jf.stem)

    all_frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    remaining = [f for f in all_frames if f.stem not in labeled_stems]
    print(f"Already labeled: {len(labeled_stems)}")
    print(f"Remaining frames to predict: {len(remaining)}")

    if not remaining:
        print("No remaining frames to predict!")
        return

    # Load model
    predictor = load_predictor(args.model_dir, args.threshold)

    # Predict
    os.makedirs(args.output_dir, exist_ok=True)
    total_spines = 0

    for i, img_path in enumerate(remaining):
        result = predict_to_labelme(predictor, str(img_path), args.threshold)
        if result is None:
            print(f"  [{i+1}/{len(remaining)}] SKIP {img_path.name} (read error)")
            continue

        if args.no_imagedata:
            result["imageData"] = None

        n = len(result["shapes"])
        total_spines += n

        # Save JSON
        out_json = os.path.join(args.output_dir, f"{img_path.stem}.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Also copy image
        import shutil
        out_img = os.path.join(args.output_dir, img_path.name)
        if not os.path.exists(out_img):
            shutil.copy2(img_path, out_img)

        print(f"  [{i+1}/{len(remaining)}] {img_path.name}: {n} spines predicted")

    print(f"\nTotal: {len(remaining)} frames, {total_spines} spine predictions")
    print(f"Output: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Open the output dir in Labelme")
    print("  2. Review each prediction: delete wrong ones, add missing ones, adjust boundaries")
    print("  3. Save corrected annotations")
    print("  4. Merge with existing labeled data for final training")


if __name__ == "__main__":
    main()
