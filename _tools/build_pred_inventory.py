"""Build a draft video-level inventory CSV from OCR crop results.

Input CSV requirements:
    - one path-like column: image_path / crop_path / path / file / filename
    - one OCR text column: ocr_text / text / pred_text / raw_text / title

Optional columns:
    - matched_title: if present and non-empty, use it directly instead of fuzzy match
    - confidence
    - video_id / frame_id: if absent, parsed from filename like 001_03s_crop_0001.jpg

Outputs:
    1. per-crop matched csv
    2. draft pred_inventory csv compatible with score_video_inventory.py
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


PATH_COLUMNS = ("image_path", "crop_path", "path", "file", "filename")
TEXT_COLUMNS = ("ocr_text", "text", "pred_text", "raw_text", "title")
FRAME_RE = re.compile(r"^(?P<video>\d{3})_(?P<stamp>[^_]+?)(?:_crop_(?P<crop>\d+))?$")
SPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[\(（【〔\[][^\)）】〕\]]*[\)）】〕\]]")
NOISE_TAIL_RE = re.compile(r"\s*[／/]\s*.*$")


def normalize_text(text: str) -> str:
    text = (text or "").strip().strip("\"'“”‘’")
    text = NOISE_TAIL_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip(" .。·-—_")


def choose_first_field(row: dict[str, str], candidates: Iterable[str]) -> str:
    for name in candidates:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_frame_info(path_like: str) -> tuple[str, str]:
    stem = Path(path_like).stem
    match = FRAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Cannot parse video/frame id from '{path_like}'. "
            "Expected names like 001_03s.jpg or 001_03s_crop_0001.jpg"
        )
    video_id = match.group("video")
    frame_id = f"{video_id}_{match.group('stamp')}"
    return video_id, frame_id


class CatalogMatcher:
    def __init__(self, titles: list[str]) -> None:
        clean_titles = []
        seen = set()
        for title in titles:
            t = normalize_text(title)
            if not t or t in seen:
                continue
            seen.add(t)
            clean_titles.append(t)
        self.titles = clean_titles
        self.char_index: dict[str, list[int]] = defaultdict(list)
        for idx, title in enumerate(self.titles):
            for ch in set(title):
                if ch.strip():
                    self.char_index[ch].append(idx)

    def candidate_indices(self, text: str, limit: int = 200) -> list[int]:
        counter: Counter[int] = Counter()
        for ch in set(text):
            for idx in self.char_index.get(ch, []):
                counter[idx] += 1
        if not counter:
            return []
        return [idx for idx, _ in counter.most_common(limit)]

    def match(self, text: str, threshold: float, uncertain_threshold: float) -> tuple[str, float, str]:
        query = normalize_text(text)
        if not query:
            return "", 0.0, "empty"

        exact = query if query in self.titles else ""
        if exact:
            return exact, 1.0, "exact"

        candidates = self.candidate_indices(query)
        if not candidates:
            return "", 0.0, "unknown"

        best_title = ""
        best_score = 0.0
        for idx in candidates:
            title = self.titles[idx]
            score = difflib.SequenceMatcher(None, query, title).ratio()
            if query in title or title in query:
                score = max(score, min(len(query), len(title)) / max(len(query), len(title)))
            if score > best_score:
                best_title = title
                best_score = score

        if best_score >= threshold:
            return best_title, best_score, "matched"
        if best_score >= uncertain_threshold:
            return best_title, best_score, "uncertain"
        return "", best_score, "unknown"


def aggregate_counts(values: list[int], method: str) -> int:
    if not values:
        return 0
    if method == "max":
        return max(values)
    if method == "median":
        return int(round(median(values)))
    if method == "mode":
        counter = Counter(values)
        top_count = max(counter.values())
        top_values = sorted(v for v, c in counter.items() if c == top_count)
        return top_values[0]
    raise ValueError(f"Unsupported aggregate method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-csv", required=True, help="OCR crop result csv")
    parser.add_argument("--catalog", default="", help="Optional catalog title list txt")
    parser.add_argument("--per-crop-out", default="output/ocr_results/ocr_matched.csv", help="Matched per-crop csv")
    parser.add_argument("--inventory-out", default="output/final_inventory/pred_inventory.csv", help="Draft inventory csv")
    parser.add_argument("--frame-counts-out", default="output/final_inventory/frame_counts.csv", help="Per-frame count csv")
    parser.add_argument("--match-threshold", type=float, default=0.75, help="Accept as matched at or above this score")
    parser.add_argument("--uncertain-threshold", type=float, default=0.55, help="Keep candidate as uncertain above this score")
    parser.add_argument("--aggregate", choices=("max", "median", "mode"), default="max", help="Video-level count aggregation")
    parser.add_argument("--keep-unknown", action="store_true", help="Keep unmatched OCR texts as UNKNOWN::<text>")
    args = parser.parse_args()

    matcher = None
    if args.catalog:
        catalog_path = Path(args.catalog)
        if catalog_path.exists():
            catalog_titles = catalog_path.read_text(encoding="utf-8").splitlines()
            matcher = CatalogMatcher(catalog_titles)
        else:
            print(f"warning: catalog not found, skip fuzzy match -> {catalog_path}")

    ocr_path = Path(args.ocr_csv)
    with ocr_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_path = not any(name in fieldnames for name in PATH_COLUMNS)
        missing_text = not any(name in fieldnames for name in TEXT_COLUMNS)
        if missing_path or missing_text:
            raise ValueError(
                "OCR csv must contain one path column from "
                f"{PATH_COLUMNS} and one text column from {TEXT_COLUMNS}. Got: {fieldnames}"
            )

        matched_rows: list[dict[str, str]] = []
        frame_title_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

        for line_no, row in enumerate(reader, start=2):
            path_like = choose_first_field(row, PATH_COLUMNS)
            raw_text = choose_first_field(row, TEXT_COLUMNS)
            if not path_like:
                raise ValueError(f"{ocr_path}:{line_no} missing crop/image path")

            video_id = (row.get("video_id") or "").strip()
            frame_id = (row.get("frame_id") or "").strip()
            if not video_id or not frame_id:
                parsed_video, parsed_frame = parse_frame_info(path_like)
                video_id = video_id or parsed_video
                frame_id = frame_id or parsed_frame

            normalized_text = normalize_text(raw_text)
            matched_title = normalize_text((row.get("matched_title") or ""))
            score = 1.0
            status = "provided" if matched_title else "empty"
            if not matched_title:
                if matcher is not None:
                    matched_title, score, status = matcher.match(
                        normalized_text,
                        threshold=args.match_threshold,
                        uncertain_threshold=args.uncertain_threshold,
                    )
                else:
                    matched_title = ""
                    score = 0.0
                    status = "no_catalog"

            if not matched_title and args.keep_unknown and normalized_text:
                matched_title = f"UNKNOWN::{normalized_text}"

            confidence = (row.get("confidence") or "").strip()
            matched_rows.append(
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "image_path": path_like,
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "matched_title": matched_title,
                    "match_score": f"{score:.4f}",
                    "match_status": status,
                    "confidence": confidence,
                }
            )

            if matched_title:
                frame_title_counts[(video_id, frame_id)][matched_title] += 1

    per_crop_out = Path(args.per_crop_out)
    per_crop_out.parent.mkdir(parents=True, exist_ok=True)
    with per_crop_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_id",
                "image_path",
                "raw_text",
                "normalized_text",
                "matched_title",
                "match_score",
                "match_status",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(matched_rows)

    frame_count_rows: list[dict[str, str | int]] = []
    video_title_series: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (video_id, frame_id), counter in sorted(frame_title_counts.items()):
        for title, count in sorted(counter.items()):
            frame_count_rows.append(
                {"video_id": video_id, "frame_id": frame_id, "book_title": title, "count": count}
            )
            video_title_series[(video_id, title)].append(count)

    frame_counts_out = Path(args.frame_counts_out)
    frame_counts_out.parent.mkdir(parents=True, exist_ok=True)
    with frame_counts_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame_id", "book_title", "count"])
        writer.writeheader()
        writer.writerows(frame_count_rows)

    inventory_rows: list[dict[str, str | int]] = []
    for (video_id, title), counts in sorted(video_title_series.items()):
        final_count = aggregate_counts(counts, args.aggregate)
        notes = f"aggregate={args.aggregate}; frames={len(counts)}; series={counts}"
        inventory_rows.append(
            {"video_id": video_id, "book_title": title, "count": final_count, "notes": notes}
        )

    inventory_out = Path(args.inventory_out)
    inventory_out.parent.mkdir(parents=True, exist_ok=True)
    with inventory_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "book_title", "count", "notes"])
        writer.writeheader()
        writer.writerows(inventory_rows)

    print(f"catalog titles loaded: {len(matcher.titles)}")
    print(f"per-crop matched csv -> {per_crop_out}")
    print(f"frame count csv      -> {frame_counts_out}")
    print(f"draft inventory csv  -> {inventory_out}")
    print(f"matched rows         : {len(matched_rows)}")
    print(f"inventory rows       : {len(inventory_rows)}")


if __name__ == "__main__":
    main()
