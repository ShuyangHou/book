"""Call Qianfan DeepSeek-OCR on local crop images and save ocr_results.csv.

Environment:
    QIANFAN_API_KEY   required
    QIANFAN_ENDPOINT  optional, default: https://qianfan.baidubce.com/v2/chat/completions

Key features:
    - local image -> base64 data URL
    - per-image hash cache to avoid repeated billing
    - on-demand rotation retry: 0 -> 90 -> 270
    - optional catalog fuzzy match for quick draft correction
    - output CSV compatible with build_pred_inventory.py

The request shape follows Qianfan's official deepseek-ocr chat completions API.
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image


DEFAULT_ENDPOINT = "https://qianfan.baidubce.com/v2/chat/completions"
DEFAULT_MODEL = "deepseek-ocr"
DEFAULT_PROMPT = "<image>\nFree OCR."
DEFAULT_ROTATIONS = (0, 90, 270)
FRAME_RE = re.compile(r"^(?P<video>\d{3})_(?P<stamp>[^_]+?)(?:_crop_(?P<crop>\d+))?$")
SPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[\(（【〔\[][^\)）】〕\]]*[\)）】〕\]]")
NOISE_TAIL_RE = re.compile(r"\s*[／/]\s*.*$")


def normalize_text(text: str) -> str:
    text = (text or "").strip().strip("\"'“”‘’")
    text = NOISE_TAIL_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = SPACE_RE.sub("", text)
    return text.strip(" .。·-—_")


def parse_frame_info(path_like: str) -> tuple[str, str]:
    stem = Path(path_like).stem
    match = FRAME_RE.match(stem)
    if not match:
        return "", ""
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
        if query in self.titles:
            return query, 1.0, "exact"

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


def image_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    cache: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skip invalid cache line {line_no} in {path}")
                continue
            image_hash = row.get("image_hash")
            rotation = row.get("rotation")
            if image_hash and isinstance(rotation, int):
                cache[(image_hash, rotation)] = row
    return cache


def append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_to_data_url(path: Path, rotation: int) -> str:
    with Image.open(path) as img:
        if rotation:
            img = img.rotate(rotation, expand=True)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def extract_text_from_response(response_json: dict) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def call_ocr(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    data_url: str,
    timeout: float,
    user: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "stream": False,
    }
    if user:
        payload["user"] = user

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    data = json.loads(raw)
    if "code" in data or "message" in data or "type" in data:
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    if "choices" not in data:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, ensure_ascii=False)}")
    return data


def build_attempt_row(
    *,
    image_path: Path,
    image_hash: str,
    rotation: int,
    raw_text: str,
    matched_title: str,
    match_score: float,
    match_status: str,
    api_status: str,
    model: str,
    prompt: str,
    confidence: str = "",
) -> dict:
    video_id, frame_id = parse_frame_info(image_path.name)
    return {
        "image_path": str(image_path),
        "video_id": video_id,
        "frame_id": frame_id,
        "image_hash": image_hash,
        "rotation": rotation,
        "ocr_text": raw_text,
        "normalized_text": normalize_text(raw_text),
        "matched_title": matched_title,
        "match_score": round(match_score, 6),
        "match_status": match_status,
        "confidence": confidence,
        "api_status": api_status,
        "model": model,
        "prompt": prompt,
        "ts": int(time.time()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="crops", help="Crop image directory")
    parser.add_argument("--glob", default="*.jpg", help="Glob pattern under input-dir")
    parser.add_argument("--catalog", default="", help="Optional catalog title txt")
    parser.add_argument("--output-csv", default="output/ocr_results/ocr_results.csv", help="Final chosen OCR csv")
    parser.add_argument("--attempts-csv", default="output/ocr_results/ocr_attempts.csv", help="All attempts csv")
    parser.add_argument("--cache-jsonl", default="output/ocr_results/ocr_cache.jsonl", help="Persistent cache jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qianfan model id, fixed to deepseek-ocr")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("QIANFAN_ENDPOINT", DEFAULT_ENDPOINT),
        help="Qianfan chat completions endpoint",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="OCR prompt")
    parser.add_argument("--user", default="", help="Optional final-user identifier passed to Qianfan")
    parser.add_argument("--match-threshold", type=float, default=0.75, help="Accept OCR if match score reaches this")
    parser.add_argument("--uncertain-threshold", type=float, default=0.55, help="Mark as uncertain above this threshold")
    parser.add_argument("--rotations", default="0,90,270", help="Rotation retry order, comma-separated")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images after sorting")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between uncached API calls")
    parser.add_argument("--overwrite", action="store_true", help="Ignore cache and force re-call API")
    parser.add_argument("--keep-unknown", action="store_true", help="Keep unmatched OCR as UNKNOWN::<text>")
    args = parser.parse_args()

    api_key = os.environ.get("QIANFAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing QIANFAN_API_KEY environment variable. Example: set QIANFAN_API_KEY=YOUR_API_KEY")

    rotations = []
    for item in args.rotations.split(","):
        item = item.strip()
        if item:
            rotations.append(int(item))
    if not rotations:
        rotations = list(DEFAULT_ROTATIONS)

    input_dir = Path(args.input_dir)
    image_paths = sorted(input_dir.glob(args.glob))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir} with glob {args.glob}")

    matcher = None
    if args.catalog:
        catalog_path = Path(args.catalog)
        if catalog_path.exists():
            catalog_titles = catalog_path.read_text(encoding="utf-8").splitlines()
            matcher = CatalogMatcher(catalog_titles)
        else:
            print(f"warning: catalog not found, skip fuzzy match -> {catalog_path}")
    cache_path = Path(args.cache_jsonl)
    cache = load_cache(cache_path)

    chosen_rows: list[dict] = []
    attempt_rows: list[dict] = []

    for idx, image_path in enumerate(image_paths, start=1):
        img_hash = image_sha256(image_path)
        best_row: dict | None = None

        print(f"[{idx}/{len(image_paths)}] {image_path.name}")
        for rotation in rotations:
            cache_key = (img_hash, rotation)
            if not args.overwrite and cache_key in cache:
                row = dict(cache[cache_key])
                row["api_status"] = "cache_hit"
            else:
                data_url = image_to_data_url(image_path, rotation)
                try:
                    response = call_ocr(
                        endpoint=args.endpoint,
                        api_key=api_key,
                        model=args.model,
                        prompt=args.prompt,
                        data_url=data_url,
                        timeout=args.timeout,
                        user=args.user,
                    )
                    raw_text = extract_text_from_response(response)
                    if matcher is not None:
                        matched_title, match_score, match_status = matcher.match(
                            raw_text,
                            threshold=args.match_threshold,
                            uncertain_threshold=args.uncertain_threshold,
                        )
                    else:
                        matched_title = ""
                        match_score = 0.0
                        match_status = "no_catalog"
                    if not matched_title and args.keep_unknown and normalize_text(raw_text):
                        matched_title = f"UNKNOWN::{normalize_text(raw_text)}"
                    row = build_attempt_row(
                        image_path=image_path,
                        image_hash=img_hash,
                        rotation=rotation,
                        raw_text=raw_text,
                        matched_title=matched_title,
                        match_score=match_score,
                        match_status=match_status,
                        api_status="ok",
                        model=args.model,
                        prompt=args.prompt,
                    )
                except Exception as exc:  # pragma: no cover - depends on network/runtime
                    row = build_attempt_row(
                        image_path=image_path,
                        image_hash=img_hash,
                        rotation=rotation,
                        raw_text="",
                        matched_title="",
                        match_score=0.0,
                        match_status="error",
                        api_status=f"error:{type(exc).__name__}:{exc}",
                        model=args.model,
                        prompt=args.prompt,
                    )
                append_cache(cache_path, row)
                cache[cache_key] = row
                if args.sleep > 0:
                    time.sleep(args.sleep)

            attempt_rows.append(row)
            if best_row is None or float(row.get("match_score", 0.0)) > float(best_row.get("match_score", 0.0)):
                best_row = row

            status = row.get("match_status", "")
            score = float(row.get("match_score", 0.0))
            print(f"  rot={rotation} status={status} score={score:.4f} text={row.get('ocr_text', '')[:40]}")
            if score >= args.match_threshold:
                break

        if best_row is None:
            continue
        chosen_rows.append(best_row)

    fieldnames = [
        "image_path",
        "video_id",
        "frame_id",
        "image_hash",
        "rotation",
        "ocr_text",
        "normalized_text",
        "matched_title",
        "match_score",
        "match_status",
        "confidence",
        "api_status",
        "model",
        "prompt",
        "ts",
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen_rows)

    attempts_csv = Path(args.attempts_csv)
    attempts_csv.parent.mkdir(parents=True, exist_ok=True)
    with attempts_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attempt_rows)

    print(f"chosen ocr csv  -> {output_csv}")
    print(f"attempts csv    -> {attempts_csv}")
    print(f"cache jsonl     -> {cache_path}")
    print(f"images processed: {len(chosen_rows)}")


if __name__ == "__main__":
    main()
