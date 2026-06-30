"""Generate book-spine crops from LabelMe polygons.

Input:
    frames/
        001_03s.jpg
        001_03s.json

Output:
    crops/
        001_03s_crop_0001.jpg
        001_03s_crop_0002.jpg

By default this script tries to rectify each 4-point polygon into a front-facing
rectangle, which is usually friendlier for OCR than a raw bounding-box crop.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def order_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) != 4:
        raise ValueError("order_quad requires exactly 4 points")

    pts = sorted(points, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]
    return [tl, tr, br, bl]


def rectified_crop(img: Image.Image, points: list[tuple[float, float]], min_size: int) -> Image.Image:
    quad = order_quad(points)
    tl, tr, br, bl = quad
    width = max(distance(tl, tr), distance(bl, br))
    height = max(distance(tl, bl), distance(tr, br))
    out_w = max(int(round(width)), min_size)
    out_h = max(int(round(height)), min_size)

    # PIL QUAD expects source points in the order:
    # upper-left, lower-left, lower-right, upper-right.
    data = (
        tl[0],
        tl[1],
        bl[0],
        bl[1],
        br[0],
        br[1],
        tr[0],
        tr[1],
    )
    return img.transform((out_w, out_h), Image.Transform.QUAD, data, resample=Image.Resampling.BICUBIC)


def bbox_crop(img: Image.Image, points: list[tuple[float, float]], padding: int) -> Image.Image:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = max(int(math.floor(min(xs))) - padding, 0)
    top = max(int(math.floor(min(ys))) - padding, 0)
    right = min(int(math.ceil(max(xs))) + padding, img.width)
    bottom = min(int(math.ceil(max(ys))) + padding, img.height)
    return img.crop((left, top, right, bottom))


def iter_shapes(data: dict, label: str) -> list[list[tuple[float, float]]]:
    result: list[list[tuple[float, float]]] = []
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        points = shape.get("points") or []
        if len(points) < 4:
            continue
        result.append([(float(x), float(y)) for x, y in points])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="frames", help="Directory containing LabelMe jpg/json pairs")
    parser.add_argument("--dst", default="crops", help="Directory to save crops")
    parser.add_argument("--label", default="book_spine", help="LabelMe shape label to extract")
    parser.add_argument("--glob", default="*.json", help="JSON glob under src")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N json files")
    parser.add_argument("--padding", type=int, default=8, help="Padding for bbox fallback")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum rectified crop width/height")
    parser.add_argument("--mode", choices=("rectify", "bbox"), default="rectify", help="Crop strategy")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(src.glob(args.glob))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"No json files found in {src} with glob {args.glob}")

    total_frames = 0
    total_crops = 0

    for json_path in json_paths:
        stem = json_path.stem
        image_path = src / f"{stem}.jpg"
        if not image_path.exists():
            image_path = src / f"{stem}.png"
        if not image_path.exists():
            print(f"warning: image not found for {json_path.name}")
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        polygons = iter_shapes(data, args.label)
        if not polygons:
            continue

        total_frames += 1
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            for idx, points in enumerate(polygons, start=1):
                if args.mode == "rectify" and len(points) == 4:
                    crop = rectified_crop(img, points, min_size=args.min_size)
                else:
                    crop = bbox_crop(img, points, padding=args.padding)
                out_path = dst / f"{stem}_crop_{idx:04d}.jpg"
                crop.save(out_path, format="JPEG", quality=95)
                total_crops += 1

        print(f"{json_path.name} -> {len(polygons)} crops")

    print(f"frames processed: {total_frames}")
    print(f"crops generated : {total_crops}")
    print(f"output dir      : {dst}")


if __name__ == "__main__":
    main()
