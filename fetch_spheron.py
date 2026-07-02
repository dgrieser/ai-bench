#!/usr/bin/env python3
"""
Fetch VRAM requirement estimates from the Spheron GPU recommender.

Spheron (https://www.spheron.network/tools/gpu-recommender/{org}/{model}/) is a
per-model tool: each page estimates how much VRAM a single model needs for
inference at three precisions -- FP16, INT8, INT4 -- plus recommended GPUs and
pricing. There is no leaderboard listing every model; a model is identified by
its HuggingFace-style "org/model" path (e.g. "Qwen/Qwen3-8B").

The pages are a Next.js Pages Router app, so the data is embedded in the
<script id="__NEXT_DATA__"> JSON blob (no separate API call, no self.__next_f
flight payload). The inference VRAM per precision lives at
props.pageProps.data.precisionPicks[] = [{"precision", "vramGb"}, ...], and the
canonical model id at props.pageProps.data.model.modelId.

Values are rounded to match Spheron's own display (>=10 GB -> whole number,
otherwise one decimal), e.g. Qwen3-8B -> FP16 18, INT8 8.9, INT4 4.5 GB.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


BASE_URL = "https://www.spheron.network/tools/gpu-recommender/{model}/"

# Precision keys reported by Spheron -> llm.json vram sub-keys.
PRECISIONS: dict[str, str] = {
    "fp16": "vram_fp16",
    "int8": "vram_int8",
    "int4": "vram_int4",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_next_data(html: str) -> dict:
    """Parse the __NEXT_DATA__ JSON blob out of a Spheron page."""
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError("No __NEXT_DATA__ script tag found in page")
    return json.loads(match.group(1))


def round_gb(value: float | int | None) -> float | int | None:
    """Round a VRAM figure the way Spheron displays it.

    >= 10 GB is shown as a whole number, smaller values with one decimal.
    """
    if value is None:
        return None
    num = float(value)
    return round(num) if num >= 10 else round(num, 1)


def parse_model(html: str, fallback_id: str) -> dict:
    """Extract the model id and per-precision inference VRAM from a page."""
    data = extract_next_data(html).get("props", {}).get("pageProps", {}).get("data", {})
    model_id = (data.get("model") or {}).get("modelId") or fallback_id

    by_precision: dict[str, float | int | None] = {}
    for pick in data.get("precisionPicks") or []:
        if not isinstance(pick, dict):
            continue
        precision = pick.get("precision")
        if precision in PRECISIONS:
            by_precision[precision] = round_gb(pick.get("vramGb"))

    row = {"model": model_id}
    for precision, key in PRECISIONS.items():
        row[key] = by_precision.get(precision)
    return row


def get_scores(models: list[str]) -> list[dict]:
    """Return one VRAM row per requested model path.

    Keys: model (canonical org/model id), vram_fp16, vram_int8, vram_int4 (GB).
    """
    results: list[dict] = []
    for model in models:
        url = BASE_URL.format(model=model)
        print(f"Fetching {url} ...", file=sys.stderr)
        try:
            row = parse_model(fetch_html(url), fallback_id=model)
        except Exception as exc:  # noqa: BLE001 - keep going across models
            print(f"  failed for {model}: {exc}", file=sys.stderr)
            continue
        results.append(row)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch VRAM requirement estimates from the Spheron GPU recommender."
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="ORG/NAME",
        help="HuggingFace-style model path to query (repeatable). "
        "Default: Qwen/Qwen3-8B.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.model or ["Qwen/Qwen3-8B"]
    scores = get_scores(models)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        width = max((len(e["model"]) for e in scores), default=len("MODEL"))
        width = max(width, len("MODEL"))
        fmt = f"{{:<{width}}}  {{:>6}}  {{:>6}}  {{:>6}}"
        print(fmt.format("MODEL", "FP16", "INT8", "INT4"))

        def cell(value: object) -> str:
            return "-" if value is None else str(value)

        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    cell(entry["vram_fp16"]),
                    cell(entry["vram_int8"]),
                    cell(entry["vram_int4"]),
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
