# -*- coding: utf-8 -*-
"""Build dataset bookkeeping for 项目制实践-AI.

Steps:
  1. video_manifest.csv via ffprobe (duration/width/height/fps + valid flag)
  2. extract 4 keyframes per video at 3s/10s/17s/24s into frames/
  3. emit thumbnails/thumbnail_index.html with per-video rows
  4. seed annotations/video_gt.csv and annotations/label_status.csv stubs
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\15075\Desktop\课程作业\项目制实践-AI")
DATASET = ROOT / "dataset"
FRAMES = ROOT / "frames"
THUMBS = ROOT / "thumbnails"
ANNOTS = ROOT / "annotations"

FFMPEG = r"C:\Users\15075\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\15075\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffprobe.exe"

KEY_TIMES = (3, 10, 17, 24)
MIN_DURATION = 20.0


def probe(video: Path) -> dict:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(video),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(out.decode("utf-8", errors="ignore"))
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0.0))
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    fps_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, den = fps_raw.split("/")
        fps = round(float(num) / float(den), 2) if float(den) else 0.0
    except Exception:
        fps = 0.0
    return {"duration": duration, "width": width, "height": height, "fps": fps}


def extract_frame(video: Path, t_sec: int, dst: Path) -> bool:
    if dst.exists():
        return True
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", str(t_sec), "-i", str(video),
        "-frames:v", "1", "-q:v", "3", str(dst),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst.exists()
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    ANNOTS.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in DATASET.glob("*.mp4"))
    manifest_rows = []
    label_status_rows = []

    for video in videos:
        vid = video.stem  # e.g. 001
        try:
            info = probe(video)
        except subprocess.CalledProcessError as exc:
            print(f"[probe-fail] {video.name}: {exc}")
            manifest_rows.append({
                "video_id": vid, "filename": video.name,
                "duration_sec": "", "width": "", "height": "", "fps": "",
                "valid": "no", "notes": "ffprobe failed",
            })
            continue

        duration = info["duration"]
        valid = duration >= MIN_DURATION
        notes = "" if valid else f"too short ({duration:.1f}s < {MIN_DURATION:.0f}s)"
        manifest_rows.append({
            "video_id": vid, "filename": video.name,
            "duration_sec": f"{duration:.1f}",
            "width": info["width"], "height": info["height"], "fps": info["fps"],
            "valid": "yes" if valid else "no", "notes": notes,
        })

        extracted = []
        for t in KEY_TIMES:
            dst = FRAMES / f"{vid}_{t:02d}s.jpg"
            ok = False
            if duration >= t + 0.2:
                ok = extract_frame(video, t, dst)
            extracted.append(ok)
        label_status_rows.append({
            "video_id": vid,
            "frames_extracted": sum(extracted),
            "frames_total": len(KEY_TIMES),
            "labelme_done": 0,
            "needs_review": "yes" if not all(extracted) or not valid else "no",
        })
        print(f"[ok] {vid} dur={duration:.1f}s frames={sum(extracted)}/{len(KEY_TIMES)}")

    # write manifest
    manifest_path = ANNOTS / "video_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id", "filename", "duration_sec", "width", "height", "fps", "valid", "notes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # write label_status
    status_path = ANNOTS / "label_status.csv"
    with status_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id", "frames_extracted", "frames_total", "labelme_done", "needs_review"])
        writer.writeheader()
        writer.writerows(label_status_rows)

    # seed video_gt.csv only if absent (avoid clobbering manual edits)
    gt_path = ANNOTS / "video_gt.csv"
    if not gt_path.exists():
        with gt_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video_id", "book_title", "count", "notes"])
            for row in manifest_rows:
                writer.writerow([row["video_id"], "", "", ""])

    # thumbnail index
    build_thumbnail_index(manifest_rows)

    print(f"manifest -> {manifest_path}")
    print(f"status   -> {status_path}")
    print(f"video_gt -> {gt_path}")


def build_thumbnail_index(manifest_rows: list[dict]) -> None:
    rel_frames = "../frames"  # frames live one level up from thumbnails/
    # discover the actual frames currently sitting on disk (timestamps may have
    # been swapped by the re-extract pass), grouped by video_id.
    actual_by_vid: dict[str, list[int]] = {}
    for p in sorted(FRAMES.glob("*.jpg")):
        try:
            vid, tag = p.stem.split("_", 1)
            t = int(tag.rstrip("s"))
        except ValueError:
            continue
        actual_by_vid.setdefault(vid, []).append(t)
    for vid in actual_by_vid:
        actual_by_vid[vid].sort()
    total_frames = sum(len(v) for v in actual_by_vid.values())

    html = [
        "<!doctype html>",
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        "<title>图书盘点视频缩略图索引</title>",
        "<style>",
        "body{font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f4f5f7;color:#1f2937;margin:0;padding:24px;}",
        "h1{font-size:20px;margin:0 0 16px;}",
        "table{width:100%;border-collapse:collapse;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;vertical-align:top;text-align:left;font-size:13px;}",
        "th{background:#f9fafb;position:sticky;top:0;}",
        ".frames{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;min-width:520px;}",
        ".frames figure{margin:0;}",
        ".frames img{width:100%;height:90px;object-fit:cover;border-radius:4px;background:#e5e7eb;}",
        ".frames figcaption{font-size:11px;color:#6b7280;margin-top:2px;text-align:center;}",
        ".bad{color:#b91c1c;font-weight:600;}",
        ".ok{color:#047857;}",
        "</style></head><body>",
        "<h1>图书盘点视频缩略图索引</h1>",
        f"<p>共 {len(manifest_rows)} 段视频，{total_frames} 张关键帧。"
        f" 时间点按重抽后实际帧动态生成（可能落在 3/5/8/10/12/16/17/20/24/27s 等候选）。</p>",
        "<table><thead><tr><th>编号</th><th>时长</th><th>分辨率 / fps</th><th>关键帧</th><th>备注</th></tr></thead><tbody>",
    ]
    for row in manifest_rows:
        vid = row["video_id"]
        valid_cls = "ok" if row["valid"] == "yes" else "bad"
        frame_cells = []
        for t in actual_by_vid.get(vid, []):
            name = f"{vid}_{t:02d}s.jpg"
            frame_cells.append(
                f"<figure><img src=\"{rel_frames}/{name}\" alt=\"{name}\" loading=\"lazy\">"
                f"<figcaption>{t:02d}s</figcaption></figure>"
            )
        if not frame_cells:
            frame_cells.append("<span style=\"color:#9ca3af\">no frames</span>")
        html.append(
            "<tr>"
            f"<td><strong>{vid}</strong><br><span class=\"{valid_cls}\">{row['valid']}</span></td>"
            f"<td>{row['duration_sec']}s</td>"
            f"<td>{row['width']}x{row['height']}<br>{row['fps']} fps</td>"
            f"<td><div class=\"frames\">{''.join(frame_cells)}</div></td>"
            f"<td>{row['notes']}</td>"
            "</tr>"
        )
    html.append("</tbody></table></body></html>")
    (THUMBS / "thumbnail_index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
