"""Score final video-level book inventory predictions against video_gt.csv.

Expected CSV schema for both GT and prediction files:
    video_id,book_title,count,notes

Rules:
    - One row describes one normalized title count in one video.
    - Multiple rows may share the same video_id.
    - Duplicate (video_id, book_title) rows are merged by summing counts.
    - Blank title/count rows are ignored with a warning.

Recommended main metric:
    inventory_accuracy = sum(min(gt_count, pred_count)) / sum(max(gt_count, pred_count))

This penalizes both under-counting and over-counting, and stays in [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize_title(title: str) -> str:
    """Lightweight normalization for fair exact-title comparison."""
    return " ".join((title or "").strip().split())


def load_inventory_csv(path: Path) -> tuple[dict[str, Counter], list[str]]:
    inventory: dict[str, Counter] = defaultdict(Counter)
    warnings: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "book_title", "count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            title = normalize_title(row.get("book_title") or "")
            count_raw = (row.get("count") or "").strip()

            if not video_id and not title and not count_raw:
                continue

            if not video_id:
                warnings.append(f"{path}:{line_no} missing video_id; skipped")
                continue

            if not title or not count_raw:
                warnings.append(f"{path}:{line_no} blank title/count; skipped")
                continue

            try:
                count = int(count_raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} count is not an integer: {count_raw}") from exc

            if count < 0:
                raise ValueError(f"{path}:{line_no} count must be >= 0, got {count}")
            if count == 0:
                warnings.append(f"{path}:{line_no} count=0; skipped")
                continue

            inventory[video_id][title] += count

    return dict(inventory), warnings


def score_inventory(gt: dict[str, Counter], pred: dict[str, Counter]) -> tuple[dict, list[dict]]:
    all_videos = sorted(set(gt) | set(pred))
    matched_total = 0
    union_total = 0
    gt_total = 0
    pred_total = 0
    abs_error_total = 0
    exact_video_matches = 0
    mismatch_rows: list[dict] = []

    for video_id in all_videos:
        gt_counts = gt.get(video_id, Counter())
        pred_counts = pred.get(video_id, Counter())
        all_titles = sorted(set(gt_counts) | set(pred_counts))
        video_exact = True

        for title in all_titles:
            g = gt_counts.get(title, 0)
            p = pred_counts.get(title, 0)
            matched_total += min(g, p)
            union_total += max(g, p)
            gt_total += g
            pred_total += p
            abs_error_total += abs(g - p)

            if g != p:
                video_exact = False
                mismatch_rows.append(
                    {
                        "video_id": video_id,
                        "book_title": title,
                        "gt_count": g,
                        "pred_count": p,
                        "abs_error": abs(g - p),
                    }
                )

        if video_exact:
            exact_video_matches += 1

    fp_total = pred_total - matched_total
    fn_total = gt_total - matched_total
    precision = matched_total / pred_total if pred_total else 0.0
    recall = matched_total / gt_total if gt_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    inventory_accuracy = matched_total / union_total if union_total else 1.0
    exact_video_match_rate = exact_video_matches / len(all_videos) if all_videos else 1.0

    summary = {
        "videos_scored": len(all_videos),
        "gt_total_books": gt_total,
        "pred_total_books": pred_total,
        "matched_books": matched_total,
        "false_positive_books": fp_total,
        "false_negative_books": fn_total,
        "count_precision": precision,
        "count_recall": recall,
        "count_f1": f1,
        "inventory_accuracy": inventory_accuracy,
        "exact_video_match_rate": exact_video_match_rate,
        "total_absolute_count_error": abs_error_total,
        "mean_absolute_count_error_per_video": (abs_error_total / len(all_videos)) if all_videos else 0.0,
    }
    mismatch_rows.sort(key=lambda row: (-row["abs_error"], row["video_id"], row["book_title"]))
    return summary, mismatch_rows


def write_mismatches(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "book_title", "gt_count", "pred_count", "abs_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="annotations/video_gt.csv", help="Ground-truth inventory CSV")
    parser.add_argument("--pred", required=True, help="Predicted inventory CSV")
    parser.add_argument("--output-json", default=None, help="Optional summary json output path")
    parser.add_argument("--output-mismatches", default=None, help="Optional mismatch csv output path")
    parser.add_argument("--top-k", type=int, default=20, help="How many mismatch rows to print")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    gt_inventory, gt_warnings = load_inventory_csv(gt_path)
    pred_inventory, pred_warnings = load_inventory_csv(pred_path)
    summary, mismatches = score_inventory(gt_inventory, pred_inventory)

    print("=== Inventory Scoring Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    all_warnings = gt_warnings + pred_warnings
    if all_warnings:
        print("\n=== Warnings ===")
        for item in all_warnings[:50]:
            print(item)
        if len(all_warnings) > 50:
            print(f"... {len(all_warnings) - 50} more warnings omitted")

    if mismatches:
        print("\n=== Top Mismatches ===")
        for row in mismatches[: args.top_k]:
            print(
                f"{row['video_id']} | {row['book_title']} | "
                f"gt={row['gt_count']} pred={row['pred_count']} abs_err={row['abs_error']}"
            )
    else:
        print("\nAll videos matched exactly.")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary json -> {out_json}")

    if args.output_mismatches:
        out_csv = Path(args.output_mismatches)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_mismatches(out_csv, mismatches)
        print(f"mismatch csv -> {out_csv}")


if __name__ == "__main__":
    main()
