# -*- coding: utf-8 -*-
"""Extract a book-title dictionary from 馆藏 Excel for OCR fuzzy match."""
from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = Path(r"C:\Users\15075\Desktop\课程作业\项目制实践-AI")
SRC = ROOT / "dataset" / "泰达西区库-馆藏清单-按册20260522-005.xlsx"
OUT = ROOT / "ocr_dict"

# noisy fragments that show up tucked after a slash inside 题名 (author/编 etc.)
NOISE_TAIL = re.compile(r"\s*[／/]\s*.*$")
# bracketed annotations like 〈〉【】［］() etc.
BRACKETS = re.compile(r"[\(（【〔\[][^\)）】〕\]]*[\)）】〕\]]")
# collapse whitespace
SPACES = re.compile(r"\s+")
# keep only CJK + ASCII alnum + a few separators when cleaning
ALLOWED = re.compile(r"[\u4e00-\u9fffA-Za-z0-9·\-：. ]+")


def clean(title: str) -> str:
    if not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKC", title).strip()
    t = NOISE_TAIL.sub("", t)
    t = BRACKETS.sub("", t)
    t = SPACES.sub(" ", t).strip(" .。·-—_")
    # drop pure punctuation / very short
    if len(t) < 2:
        return ""
    # drop entries with no CJK and no letters (e.g. pure numbers)
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", t):
        return ""
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SRC, sheet_name=0, header=4, dtype=str)
    if "题名" not in df.columns:
        raise SystemExit(f"题名 column missing; got {df.columns.tolist()[:10]}")

    raw_titles = (
        df["题名"].dropna().astype(str).map(str.strip).loc[lambda s: s.ne("")].tolist()
    )
    raw_unique = sorted(set(raw_titles))

    cleaned = sorted({clean(t) for t in raw_titles} - {""})

    raw_path = OUT / "book_titles.txt"
    clean_path = OUT / "book_titles_clean.txt"
    raw_path.write_text("\n".join(raw_unique) + "\n", encoding="utf-8")
    clean_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

    print(f"raw titles  : {len(raw_titles)} rows, {len(raw_unique)} unique -> {raw_path}")
    print(f"clean titles: {len(cleaned)} unique -> {clean_path}")
    sample = cleaned[:8] + ["..."] + cleaned[-3:]
    print("sample:", " | ".join(sample))


if __name__ == "__main__":
    main()
