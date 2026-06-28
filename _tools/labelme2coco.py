# -*- coding: utf-8 -*-
"""Convert Labelme annotations to COCO instance segmentation format.

Splits by video ID to prevent data leakage (frames from the same video
stay in the same split).

Output structure:
    book_spine_dataset/coco/
        train/
            images/
            instances_train.json
        val/
            images/
            instances_val.json

Usage:
    python _tools/labelme2coco.py [--src frames_pick50] [--dst book_spine_dataset/coco] [--val-ratio 0.2] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Labelme -> COCO")
    parser.add_argument("--src", default="frames_pick50", help="Labelme dir")
    parser.add_argument("--dst", default="book_spine_dataset/coco", help="Output COCO dir")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Approx val ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # Collect annotated files grouped by video ID
    by_video: dict[str, list[str]] = defaultdict(list)
    for jf in sorted(src.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        if not data.get("shapes"):
            continue
        vid = jf.stem.split("_")[0]
        by_video[vid].append(jf.stem)

    total = sum(len(v) for v in by_video.values())
    print(f"Annotated: {total} frames from {len(by_video)} videos")

    # Split by video ID
    video_ids = sorted(by_video.keys())
    random.seed(args.seed)
    random.shuffle(video_ids)

    target_val = max(1, int(total * args.val_ratio))
    val_frames, train_frames = [], []
    for vid in video_ids:
        if len(val_frames) < target_val:
            val_frames.extend(by_video[vid])
        else:
            train_frames.extend(by_video[vid])

    print(f"Train: {len(train_frames)} frames")
    print(f"Val:   {len(val_frames)} frames")

    for split in ["train", "val"]:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)

    for split, frame_list in [("train", train_frames), ("val", val_frames)]:
        _convert(src, dst, split, frame_list)

    print("\nDone!")


def _convert(src: Path, dst: Path, split: str, frame_list: list[str]):
    images = []
    annotations = []
    ann_id = 1

    for img_id, stem in enumerate(sorted(frame_list), start=1):
        img_path = src / f"{stem}.jpg"
        json_path = src / f"{stem}.json"

        shutil.copy2(img_path, dst / split / "images" / f"{stem}.jpg")

        with Image.open(img_path) as im:
            w, h = im.size

        images.append({
            "id": img_id,
            "file_name": f"{stem}.jpg",
            "width": w,
            "height": h,
        })

        with open(json_path) as f:
            data = json.load(f)

        for shape in data["shapes"]:
            pts = shape["points"]
            if len(pts) < 3:
                continue

            seg = []
            for p in pts:
                seg.extend([round(p[0], 2), round(p[1], 2)])

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min, y_min = min(xs), min(ys)
            bbox_w, bbox_h = max(xs) - x_min, max(ys) - y_min

            n = len(pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            area = abs(area) / 2.0

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [seg],
                "bbox": [round(x_min, 2), round(y_min, 2),
                         round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "book_spine", "supercategory": "object"}],
    }

    out_path = dst / split / f"instances_{split}.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {split}: {len(images)} images, {len(annotations)} anns -> {out_path}")


if __name__ == "__main__":
    main()
