# -*- coding: utf-8 -*-
"""Check and fix Labelme annotations for book spine segmentation.

Checks:
  1. Every .jpg has a corresponding .json
  2. All labels are 'book_spine'
  3. All shape_type are 'polygon'
  4. No degenerate polygons (< 3 points)
  5. No self-intersecting polygons
  6. No empty annotations

Fixes applied automatically:
  - rectangle/oriented_rectangle -> polygon
  - Self-intersecting polygons -> make_valid (keep largest)
  - Degenerate polygons (< 3 points) -> removed

Usage:
    python _tools/check_and_fix_labels.py [--dir frames_pick50] [--fix]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

def check_shapely():
    try:
        from shapely.geometry import Polygon as ShPoly
        from shapely.validation import make_valid, explain_validity
        return True
    except ImportError:
        print("WARNING: shapely not installed. Self-intersection checks skipped.")
        print("  pip install shapely")
        return False

HAS_SHAPELY = check_shapely()
if HAS_SHAPELY:
    from shapely.geometry import Polygon as ShPoly
    from shapely.validation import make_valid, explain_validity


def scan(pick_dir: str) -> list[str]:
    issues = []
    jsons = sorted(glob.glob(os.path.join(pick_dir, "*.json")))
    imgs = sorted(glob.glob(os.path.join(pick_dir, "*.jpg")))

    # Check missing JSONs
    for img in imgs:
        json_name = img.replace(".jpg", ".json")
        if not os.path.exists(json_name):
            issues.append(f"MISSING JSON: {os.path.basename(img)}")

    annotated = 0
    empty = 0
    total_shapes = 0

    for jf in jsons:
        bn = os.path.basename(jf)
        with open(jf) as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not shapes:
            empty += 1
            continue
        annotated += 1
        total_shapes += len(shapes)

        for i, s in enumerate(shapes):
            label = s.get("label", "")
            shape_type = s.get("shape_type", "")
            pts = s.get("points", [])

            if label != "book_spine":
                issues.append(f"{bn} shape#{i}: label='{label}' (expected 'book_spine')")
            if shape_type not in ("polygon",):
                issues.append(f"{bn} shape#{i}: shape_type='{shape_type}'")
            if len(pts) < 3:
                issues.append(f"{bn} shape#{i}: only {len(pts)} points (degenerate)")
            elif HAS_SHAPELY:
                try:
                    poly = ShPoly(pts)
                    if not poly.is_valid:
                        issues.append(f"{bn} shape#{i}: SELF-INTERSECT - {explain_validity(poly)}")
                except Exception as e:
                    issues.append(f"{bn} shape#{i}: ERROR - {e}")

    print(f"Images: {len(imgs)}, JSONs: {len(jsons)}")
    print(f"Annotated: {annotated}, Empty: {empty}")
    print(f"Total shapes: {total_shapes}")
    return issues


def fix(pick_dir: str) -> list[str]:
    fixes = []
    for jf in sorted(glob.glob(os.path.join(pick_dir, "*.json"))):
        bn = os.path.basename(jf)
        with open(jf) as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not shapes:
            continue

        modified = False
        new_shapes = []

        for i, s in enumerate(shapes):
            pts = s.get("points", [])
            shape_type = s.get("shape_type", "")

            # rectangle (2 points) -> polygon (4 points)
            if shape_type == "rectangle" and len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                s["points"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                s["shape_type"] = "polygon"
                fixes.append(f"{bn} shape#{i}: rectangle -> polygon")
                modified = True

            # oriented_rectangle or other -> polygon
            elif shape_type != "polygon":
                s["shape_type"] = "polygon"
                fixes.append(f"{bn} shape#{i}: {shape_type} -> polygon")
                modified = True

            # degenerate (< 3 points) -> remove
            pts = s.get("points", [])
            if len(pts) < 3:
                fixes.append(f"{bn} shape#{i}: REMOVED ({len(pts)} points)")
                modified = True
                continue

            # self-intersection -> make_valid
            if HAS_SHAPELY:
                try:
                    poly = ShPoly(pts)
                    if not poly.is_valid:
                        fixed = make_valid(poly)
                        candidates = []
                        if fixed.geom_type == "Polygon" and not fixed.is_empty:
                            candidates = [fixed]
                        elif fixed.geom_type in ("MultiPolygon", "GeometryCollection"):
                            candidates = [g for g in fixed.geoms
                                          if g.geom_type == "Polygon" and not g.is_empty]
                        if candidates:
                            largest = max(candidates, key=lambda g: g.area)
                            coords = list(largest.exterior.coords)[:-1]
                            s["points"] = [[round(x, 2), round(y, 2)] for x, y in coords]
                            fixes.append(f"{bn} shape#{i}: fixed self-intersection")
                            modified = True
                        else:
                            fixes.append(f"{bn} shape#{i}: WARNING could not fix")
                            continue
                except Exception as e:
                    fixes.append(f"{bn} shape#{i}: ERROR {e}")

            new_shapes.append(s)

        if modified:
            data["shapes"] = new_shapes
            with open(jf, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    return fixes


def main():
    parser = argparse.ArgumentParser(description="Check/fix Labelme annotations")
    parser.add_argument("--dir", default="frames_pick50", help="Labelme directory")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    args = parser.parse_args()

    print(f"=== Scanning {args.dir} ===")
    issues = scan(args.dir)
    print(f"\nIssues found: {len(issues)}")
    for iss in issues:
        print(f"  - {iss}")

    if args.fix and issues:
        print(f"\n=== Applying fixes ===")
        fixes = fix(args.dir)
        print(f"Fixes applied: {len(fixes)}")
        for f_ in fixes:
            print(f"  {f_}")
        print("\n=== Re-scanning ===")
        issues2 = scan(args.dir)
        print(f"Remaining issues: {len(issues2)}")
        for iss in issues2:
            print(f"  - {iss}")
    elif not issues:
        print("ALL CLEAN!")


if __name__ == "__main__":
    main()
