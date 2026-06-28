# -*- coding: utf-8 -*-
"""Re-extract bad keyframes at alternative timestamps.

- Reads annotations/low_quality_candidates.csv to find frames with severity >=1.
- For each, tries candidate times in CANDIDATES, in order of distance to the
  original time, skipping times that already exist for that video.
- A replacement is accepted only if it scores severity==0, OR is strictly
  better than the original on lap_var/brightness.
- Original frame file is removed and replaced with <vid>_<newt>s.jpg.
- Videos whose bad frames cannot all be rescued are listed in the run summary
  and flagged in annotations/video_manifest.csv (notes column appended with
  `suspect_invalid: ...`).
- Never deletes original .mp4 files.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\15075\Desktop\课程作业\项目制实践-AI")
DATASET = ROOT / "dataset"
FRAMES = ROOT / "frames"
ANNOTS = ROOT / "annotations"

FFMPEG = r"C:\Users\15075\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

CANDIDATES = [3, 5, 8, 10, 12, 14, 16, 17, 20, 22, 24, 27, 30]
TH_LAP_SEVERE = 60.0
TH_LAP_SUSPECT = 110.0
TH_BRIGHT_DARK = 45.0
TH_BRIGHT_BRIGHT = 215.0
TH_OVER = 0.25
TH_UNDER = 0.35
TH_CONTRAST = 28.0

FRAME_RE = re.compile(r"^(?P<vid>\d{3})_(?P<sec>\d{2})s\.jpg$")


def score(img_path: Path) -> tuple[int, dict]:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 2, {"reason": "decode_fail"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    std = float(gray.std())
    over = float((gray >= 245).mean())
    under = float((gray <= 15).mean())
    reasons = []
    sev = 0
    if lap < TH_LAP_SEVERE:
        reasons.append(f"blur(lap={lap:.0f})"); sev = max(sev, 2)
    elif lap < TH_LAP_SUSPECT:
        reasons.append(f"soft(lap={lap:.0f})"); sev = max(sev, 1)
    if mean < TH_BRIGHT_DARK:
        reasons.append(f"dark(mean={mean:.0f})"); sev = max(sev, 2 if mean < 30 else 1)
    elif mean > TH_BRIGHT_BRIGHT:
        reasons.append(f"bright(mean={mean:.0f})"); sev = max(sev, 2 if mean > 230 else 1)
    if under > TH_UNDER:
        reasons.append(f"under({under*100:.0f}%)"); sev = max(sev, 2)
    if over > TH_OVER:
        reasons.append(f"over({over*100:.0f}%)"); sev = max(sev, 2)
    if std < TH_CONTRAST:
        reasons.append(f"flat(std={std:.0f})"); sev = max(sev, 1)
    return sev, {"lap": lap, "mean": mean, "std": std, "over": over, "under": under, "reason": ",".join(reasons) or "ok"}


def probe_duration(video: Path) -> float:
    ffprobe = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe")
    out = subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video),
    ])
    try:
        return float(out.decode().strip())
    except ValueError:
        return 0.0


def extract(video: Path, t: int, dst: Path) -> bool:
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video),
           "-frames:v", "1", "-q:v", "3", str(dst)]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst.exists()
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="only accept replacements that score severity==0")
    parser.add_argument("--only-suspect", action="store_true",
                        help="ignore severe rows; only re-try suspect-only frames")
    args = parser.parse_args()

    cand_csv = ANNOTS / "low_quality_candidates.csv"
    if not cand_csv.exists():
        raise SystemExit("run quality_scan.py first")

    # parse frame-level rows
    bad_by_video: dict[str, list[dict]] = {}
    with cand_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if not row or row[0] != "frame":
                continue
            vid = row[1]; frame = row[2]; tsec = int(row[3]); sev = row[4]; reasons = row[10]
            if args.only_suspect and sev != "suspect":
                continue
            bad_by_video.setdefault(vid, []).append({"frame": frame, "t": tsec, "sev": sev, "reasons": reasons})

    report_rows = []  # (vid, old_frame, action, new_t, old_score, new_score, reasons)
    suspect_videos: dict[str, str] = {}

    for vid in sorted(bad_by_video.keys()):
        video_path = DATASET / f"{vid}.mp4"
        if not video_path.exists():
            print(f"[skip] {vid}: video missing")
            continue
        duration = probe_duration(video_path)
        existing_times = {int(p.stem.split("_")[1].rstrip("s")) for p in FRAMES.glob(f"{vid}_*.jpg")}
        unrescued = []

        for bad in sorted(bad_by_video[vid], key=lambda x: x["t"]):
            old_path = FRAMES / bad["frame"]
            if not old_path.exists():
                continue
            old_sev, old_metrics = score(old_path)
            # candidate times: avoid times already in use (other frames for this video)
            others = existing_times - {bad["t"]}
            cand = [t for t in CANDIDATES if t not in others and t + 0.2 <= duration]
            # order by distance from original timestamp, then absolute order
            cand.sort(key=lambda t: (abs(t - bad["t"]), t))
            chosen_t = None
            chosen_metrics = None
            chosen_sev = None
            for t in cand:
                if t == bad["t"]:
                    continue  # already known to be bad
                tmp = FRAMES / f"__try_{vid}_{t:02d}s.jpg"
                if tmp.exists():
                    tmp.unlink()
                if not extract(video_path, t, tmp):
                    continue
                sev, metrics = score(tmp)
                # acceptance:
                #   strict mode -> must be clean (severity == 0)
                #   otherwise   -> clean OR clearly better than original
                better = sev < old_sev and metrics["lap"] >= max(old_metrics["lap"], 50)
                if sev == 0 or (not args.strict and better):
                    chosen_t = t; chosen_sev = sev; chosen_metrics = metrics
                    # promote tmp to final name
                    final = FRAMES / f"{vid}_{t:02d}s.jpg"
                    if final.exists() and final != tmp:
                        final.unlink()
                    tmp.rename(final)
                    break
                else:
                    tmp.unlink()

            if chosen_t is None:
                unrescued.append(bad)
                report_rows.append((vid, bad["frame"], "keep_original", "-", f"sev={old_sev} lap={old_metrics['lap']:.0f}", "no candidate beat original", bad["reasons"]))
                continue

            # replaced -> remove old frame file, update existing_times
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
            existing_times.discard(bad["t"])
            existing_times.add(chosen_t)
            report_rows.append((
                vid, bad["frame"], "replaced", f"{chosen_t:02d}s",
                f"sev={old_sev} lap={old_metrics['lap']:.0f}",
                f"sev={chosen_sev} lap={chosen_metrics['lap']:.0f}",
                chosen_metrics["reason"],
            ))

        # if a "video_suspect" video still has all-bad frames, flag it
        remaining_bad = []
        for p in sorted(FRAMES.glob(f"{vid}_*.jpg")):
            sev, _ = score(p)
            if sev == 2:
                remaining_bad.append(p.name)
        good_count = len(list(FRAMES.glob(f"{vid}_*.jpg"))) - len(remaining_bad)
        if good_count <= 1 and len(remaining_bad) >= 2:
            suspect_videos[vid] = f"suspect_invalid: only {good_count} usable frame after retries"

    # write replacement report
    rep_path = ANNOTS / "reextract_report.csv"
    with rep_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "old_frame", "action", "new_t", "old_score", "new_score", "notes"])
        w.writerows(report_rows)

    # update manifest notes for suspect videos
    if suspect_videos:
        manifest_path = ANNOTS / "video_manifest.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            tag = suspect_videos.get(row["video_id"])
            if not tag:
                continue
            if "suspect_invalid" in row.get("notes", ""):
                continue
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + tag
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["video_id", "filename", "duration_sec", "width", "height", "fps", "valid", "notes"])
            w.writeheader()
            w.writerows(rows)

    # console summary
    replaced = sum(1 for r in report_rows if r[2] == "replaced")
    kept = sum(1 for r in report_rows if r[2] == "keep_original")
    print(f"replaced frames : {replaced}")
    print(f"kept (no better): {kept}")
    if suspect_videos:
        print("suspect_invalid videos (manifest notes updated):")
        for vid, why in suspect_videos.items():
            print(f"  {vid}: {why}")
    else:
        print("no video flagged suspect_invalid")
    print(f"report -> {rep_path}")


if __name__ == "__main__":
    main()
