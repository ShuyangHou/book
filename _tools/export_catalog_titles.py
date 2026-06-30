"""Export normalized title dictionaries from the library inventory Excel.

The source workbook is a custom export that stores many cells as inline XML
strings, so this script reads the workbook XML directly instead of relying on a
high-level spreadsheet reader.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import iterparse


XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
SPACE_RE = re.compile(r"\s+")


def normalize_title(text: str) -> str:
    text = (text or "").strip()
    text = SPACE_RE.sub("", text)
    return text.strip(" .。·-—_")


def iter_sheet_rows(xlsx_path: Path) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    with zipfile.ZipFile(xlsx_path) as zf, zf.open("xl/worksheets/sheet1.xml") as f:
        for event, elem in iterparse(f, events=("end",)):
            if elem.tag != f"{XML_NS}row":
                continue
            row_num = int(elem.attrib.get("r", "0"))
            values = []
            for cell in elem:
                text = "".join(node.text or "" for node in cell.iter(f"{XML_NS}t"))
                values.append(text)
            rows.append((row_num, values))
            elem.clear()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        default="dataset/泰达西区库-馆藏清单-按册20260522-005.xlsx",
        help="Library inventory workbook",
    )
    parser.add_argument("--titles-out", default="output/catalog/catalog_titles.txt", help="Unique normalized titles txt")
    parser.add_argument(
        "--counts-out",
        default="output/catalog/catalog_title_counts.csv",
        help="Per-title count csv from the per-copy workbook",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    rows = iter_sheet_rows(xlsx_path)
    header_row = next((values for row_num, values in rows if row_num == 5), None)
    if not header_row:
        raise SystemExit(f"Header row 5 not found in {xlsx_path}")
    try:
        title_idx = header_row.index("题名")
    except ValueError as exc:
        raise SystemExit(f"Title column '题名' not found in row 5 of {xlsx_path}") from exc

    counts: Counter[str] = Counter()
    for row_num, values in rows:
        if row_num <= 5 or title_idx >= len(values):
            continue
        title = normalize_title(values[title_idx])
        if title:
            counts[title] += 1

    if not counts:
        raise SystemExit(f"No titles parsed from {xlsx_path}")

    titles_out = Path(args.titles_out)
    counts_out = Path(args.counts_out)
    titles_out.parent.mkdir(parents=True, exist_ok=True)
    counts_out.parent.mkdir(parents=True, exist_ok=True)

    sorted_titles = sorted(counts)
    titles_out.write_text("\n".join(sorted_titles) + "\n", encoding="utf-8")

    with counts_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "copy_count"])
        writer.writeheader()
        for title in sorted_titles:
            writer.writerow({"title": title, "copy_count": counts[title]})

    print(f"catalog title kinds -> {len(sorted_titles)}")
    print(f"catalog copy total  -> {sum(counts.values())}")
    print(f"titles txt          -> {titles_out}")
    print(f"title counts csv    -> {counts_out}")


if __name__ == "__main__":
    main()
