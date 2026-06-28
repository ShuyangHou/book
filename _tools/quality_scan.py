# -*- coding: utf-8 -*-
"""Frame-level quality scan for frames/*.jpg.

Writes:
  annotations/low_quality_candidates.csv  (per-frame + per-video roll-up)
  thumbnails/low_quality_review.html       (sortable visual review page)

Never deletes anything; this is purely a candidate report.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\15075\Desktop\课程作业\项目制实践-AI")
FRAMES = ROOT / "frames"
ANNOTS = ROOT / "annotations"
THUMBS = ROOT / "thumbnails"

# Conservative thresholds; intentionally generous so we list more candidates.
TH_LAP_SEVERE = 60.0     # below this = badly blurred
TH_LAP_SUSPECT = 110.0   # below this = mildly soft
TH_BRIGHT_DARK = 45.0    # mean gray
TH_BRIGHT_BRIGHT = 215.0
TH_OVER = 0.25           # >25% pixels saturated bright
TH_UNDER = 0.35          # >35% pixels saturated dark
TH_CONTRAST = 28.0       # gray std below this = low-contrast

FRAME_RE = re.compile(r"^(?P<vid>\d{3})_(?P<sec>\d{2})s\.jpg$", re.IGNORECASE)


def score_frame(path: Path) -> dict | None:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    over_ratio = float((gray >= 245).mean())
    under_ratio = float((gray <= 15).mean())

    reasons = []
    severity = 0  # 0 ok, 1 suspect, 2 severe
    if lap_var < TH_LAP_SEVERE:
        reasons.append(f"blur(lap={lap_var:.0f})")
        severity = max(severity, 2)
    elif lap_var < TH_LAP_SUSPECT:
        reasons.append(f"soft(lap={lap_var:.0f})")
        severity = max(severity, 1)
    if brightness < TH_BRIGHT_DARK:
        reasons.append(f"dark(mean={brightness:.0f})")
        severity = max(severity, 2 if brightness < 30 else 1)
    elif brightness > TH_BRIGHT_BRIGHT:
        reasons.append(f"bright(mean={brightness:.0f})")
        severity = max(severity, 2 if brightness > 230 else 1)
    if under_ratio > TH_UNDER:
        reasons.append(f"under({under_ratio*100:.0f}%)")
        severity = max(severity, 2)
    if over_ratio > TH_OVER:
        reasons.append(f"over({over_ratio*100:.0f}%)")
        severity = max(severity, 2)
    if contrast < TH_CONTRAST:
        reasons.append(f"flat(std={contrast:.0f})")
        severity = max(severity, 1)

    return {
        "frame": path.name,
        "lap_var": round(lap_var, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "over_pct": round(over_ratio * 100, 1),
        "under_pct": round(under_ratio * 100, 1),
        "severity": severity,        # 0/1/2
        "reasons": ",".join(reasons),
    }


def main() -> None:
    rows = []
    per_video: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(FRAMES.glob("*.jpg")):
        m = FRAME_RE.match(path.name)
        if not m:
            continue
        info = score_frame(path)
        if info is None:
            continue
        info["video_id"] = m["vid"]
        info["t_sec"] = int(m["sec"])
        rows.append(info)
        per_video[m["vid"]].append(info)

    # video-level rollup
    video_rollup = {}
    for vid, items in per_video.items():
        sev = [it["severity"] for it in items]
        severe = sum(1 for s in sev if s == 2)
        suspect = sum(1 for s in sev if s == 1)
        ok = sum(1 for s in sev if s == 0)
        # heuristic: 3+ severe out of <=4 frames means the segment itself is suspect
        if severe >= 3:
            verdict = "video_suspect"
        elif severe + suspect >= len(items) and len(items) >= 3:
            verdict = "video_borderline"
        elif severe >= 1 or suspect >= 1:
            verdict = "frame_only"
        else:
            verdict = "ok"
        video_rollup[vid] = {
            "video_id": vid,
            "frames_scored": len(items),
            "severe": severe,
            "suspect": suspect,
            "ok": ok,
            "verdict": verdict,
        }

    # --- write per-frame candidates CSV (only frames with severity >= 1) ---
    cand_path = ANNOTS / "low_quality_candidates.csv"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_rows = sorted(
        (r for r in rows if r["severity"] >= 1),
        key=lambda r: (-r["severity"], r["lap_var"]),
    )
    with cand_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "video_id", "frame", "t_sec", "severity", "lap_var", "brightness", "contrast", "over_pct", "under_pct", "reasons", "video_verdict"])
        for r in cand_rows:
            w.writerow([
                "frame", r["video_id"], r["frame"], r["t_sec"],
                {0: "ok", 1: "suspect", 2: "severe"}[r["severity"]],
                r["lap_var"], r["brightness"], r["contrast"], r["over_pct"], r["under_pct"],
                r["reasons"], video_rollup[r["video_id"]]["verdict"],
            ])
        w.writerow([])
        w.writerow(["scope", "video_id", "frames_scored", "severe", "suspect", "ok", "verdict"])
        for vid, info in sorted(video_rollup.items()):
            if info["verdict"] == "ok":
                continue
            w.writerow(["video", vid, info["frames_scored"], info["severe"], info["suspect"], info["ok"], info["verdict"]])

    # --- write HTML review page ---
    html = [
        "<!doctype html>",
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>Low quality review</title>",
        "<style>",
        "body{font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f4f5f7;color:#1f2937;margin:0;padding:24px;}",
        "h1{font-size:20px;margin:0 0 12px;}",
        "h2{font-size:15px;margin:24px 0 8px;color:#111827;}",
        ".legend{font-size:12px;color:#4b5563;margin-bottom:8px;}",
        "table{width:100%;border-collapse:collapse;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);font-size:12px;}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;vertical-align:top;text-align:left;}",
        "th{background:#f9fafb;}",
        "img{height:120px;border-radius:4px;background:#e5e7eb;}",
        ".severe{color:#b91c1c;font-weight:600;}",
        ".suspect{color:#b45309;font-weight:600;}",
        ".ok{color:#047857;}",
        ".verdict-video_suspect{background:#fef2f2;}",
        ".verdict-video_borderline{background:#fff7ed;}",
        ".verdict-frame_only{background:#ffffff;}",
        "</style></head><body>",
        "<h1>关键帧质量筛查（仅候选，不会自动删除）</h1>",
        "<p class=\"legend\">阈值：lap_var&lt;60 严重糊 / &lt;110 偏糊；亮度 &lt;45 暗、&gt;215 亮；过曝&gt;25% 或欠曝&gt;35% 判严重；对比度 std&lt;28 偏平。video_suspect=4 张里 ≥3 张严重，建议整段重审；video_borderline=全部至少 suspect；frame_only=个别帧问题，换时间点重抽即可。</p>",
    ]

    # 1) integrally-suspect videos first
    html.append("<h2>① 整段疑似不合格（建议整段重审 / 标 invalid 等队友补）</h2>")
    sus_videos = [v for v, info in video_rollup.items() if info["verdict"] in ("video_suspect", "video_borderline")]
    if not sus_videos:
        html.append("<p>无。</p>")
    else:
        html.append("<table><thead><tr><th>video_id</th><th>verdict</th><th>severe/suspect/ok</th><th>关键帧</th></tr></thead><tbody>")
        for vid in sorted(sus_videos):
            info = video_rollup[vid]
            cells = []
            for it in sorted(per_video[vid], key=lambda x: x["t_sec"]):
                cls = {2: "severe", 1: "suspect", 0: "ok"}[it["severity"]]
                cells.append(
                    f"<figure style=\"margin:0;display:inline-block;text-align:center;margin-right:6px;\">"
                    f"<img src=\"../frames/{it['frame']}\" loading=\"lazy\">"
                    f"<figcaption class=\"{cls}\">{it['t_sec']:02d}s lap={it['lap_var']:.0f}<br>{it['reasons'] or 'ok'}</figcaption>"
                    f"</figure>"
                )
            html.append(
                f"<tr class=\"verdict-{info['verdict']}\"><td><strong>{vid}</strong></td>"
                f"<td class=\"severe\">{info['verdict']}</td>"
                f"<td>{info['severe']} / {info['suspect']} / {info['ok']}</td>"
                f"<td>{''.join(cells)}</td></tr>"
            )
        html.append("</tbody></table>")

    # 2) frame-only issues
    html.append("<h2>② 仅个别帧不行（建议换时间点重抽：试 05s / 12s / 20s / 27s）</h2>")
    frame_only_rows = [r for r in cand_rows if video_rollup[r["video_id"]]["verdict"] == "frame_only"]
    if not frame_only_rows:
        html.append("<p>无。</p>")
    else:
        html.append("<table><thead><tr><th>frame</th><th>severity</th><th>lap_var</th><th>brightness</th><th>contrast</th><th>over%</th><th>under%</th><th>reasons</th><th>预览</th></tr></thead><tbody>")
        for r in frame_only_rows:
            cls = {2: "severe", 1: "suspect"}[r["severity"]]
            html.append(
                f"<tr><td><strong>{r['frame']}</strong></td>"
                f"<td class=\"{cls}\">{cls}</td>"
                f"<td>{r['lap_var']}</td><td>{r['brightness']}</td><td>{r['contrast']}</td>"
                f"<td>{r['over_pct']}</td><td>{r['under_pct']}</td><td>{r['reasons']}</td>"
                f"<td><img src=\"../frames/{r['frame']}\" loading=\"lazy\"></td></tr>"
            )
        html.append("</tbody></table>")

    html.append("</body></html>")
    (THUMBS / "low_quality_review.html").write_text("\n".join(html), encoding="utf-8")

    # console summary
    total = len(rows)
    sev = sum(1 for r in rows if r["severity"] == 2)
    sus = sum(1 for r in rows if r["severity"] == 1)
    print(f"frames scored : {total}")
    print(f"severe frames : {sev}")
    print(f"suspect frames: {sus}")
    print("video verdicts:")
    for vid, info in sorted(video_rollup.items()):
        if info["verdict"] != "ok":
            print(f"  {vid}: {info['verdict']} (severe={info['severe']} suspect={info['suspect']} ok={info['ok']})")
    print(f"\nreport -> {cand_path}")
    print(f"review -> {THUMBS / 'low_quality_review.html'}")


if __name__ == "__main__":
    main()
