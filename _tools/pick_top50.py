# -*- coding: utf-8 -*-
"""Pick 50 high-quality frames for the first Labelme polygon pass.

Strategy:
  1. Score every frame on disk (lap_var, contrast std).
  2. Round 1: take the single sharpest frame from each video (covers all 42).
  3. Round 2: fill remaining slots from non-picked frames, ranked by lap_var
     with a contrast tiebreaker (prefers texture-dense / spine-rich frames).
  4. Hard-link picks into frames_pick50/ and emit _pick50_manifest.csv.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\15075\Desktop\课程作业\项目制实践-AI")
FRAMES = ROOT / "frames"
OUT = ROOT / "frames_pick50"
TARGET = 50

FRAME_RE = re.compile(r"^(?P<vid>\d{3})_(?P<sec>\d{2})s\.jpg$")


def score(path: Path) -> dict | None:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "lap": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "std": float(gray.std()),
        "mean": float(gray.mean()),
    }


def main() -> None:
    if OUT.exists():
        # clear any previous pick so re-runs stay clean
        for p in OUT.glob("*.jpg"):
            p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    by_video: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(FRAMES.glob("*.jpg")):
        m = FRAME_RE.match(path.name)
        if not m:
            continue
        s = score(path)
        if s is None:
            continue
        entry = {
            "path": path,
            "name": path.name,
            "video_id": m["vid"],
            "t_sec": int(m["sec"]),
            **s,
        }
        frames.append(entry)
        by_video[m["vid"]].append(entry)

    picked: list[dict] = []
    picked_names: set[str] = set()

    # Round 1: best frame per video
    for vid, items in sorted(by_video.items()):
        best = max(items, key=lambda x: x["lap"])
        best["round"] = 1
        picked.append(best)
        picked_names.add(best["name"])

    # Round 2: top up to TARGET by lap_var, contrast tiebreaker
    remaining = [f for f in frames if f["name"] not in picked_names]
    remaining.sort(key=lambda x: (-x["lap"], -x["std"]))
    for f in remaining:
        if len(picked) >= TARGET:
            break
        f["round"] = 2
        picked.append(f)
        picked_names.add(f["name"])

    # Hard-link into OUT (falls back to copy on failure)
    for f in picked:
        dst = OUT / f["name"]
        if dst.exists():
            dst.unlink()
        try:
            os.link(f["path"], dst)
        except OSError:
            dst.write_bytes(f["path"].read_bytes())

    # Write manifest, ordered: round 1 by video, then round 2 by lap desc
    manifest = OUT / "_pick50_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "video_id", "frame", "t_sec", "lap_var", "contrast_std", "brightness"])
        for f in sorted(picked, key=lambda x: (x["round"], -x["lap"])):
            w.writerow([f["round"], f["video_id"], f["name"], f["t_sec"],
                        round(f["lap"], 1), round(f["std"], 1), round(f["mean"], 1)])

    print(f"picked: {len(picked)} frames -> {OUT}")
    r1 = sum(1 for f in picked if f["round"] == 1)
    r2 = len(picked) - r1
    print(f"  round1 (1 per video): {r1}")
    print(f"  round2 (top-up)     : {r2}")
    lap_vals = [f["lap"] for f in picked]
    print(f"  lap_var min/median/max: {min(lap_vals):.0f} / {sorted(lap_vals)[len(lap_vals)//2]:.0f} / {max(lap_vals):.0f}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
