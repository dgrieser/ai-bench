#!/usr/bin/env python3
"""Update benchmark JSON scores using artificialanalysis.py."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

AA_SCRIPT = Path(__file__).resolve().with_name("artificialanalysis.py")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def to_percent(value: Any) -> int | float | None:
    if value is None:
        return None
    pct = float(value) * 100.0
    rounded = round(pct, 1)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def fmt_change_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_aa_value(value: Any) -> Any:
    # Treat zero values from Artificial Analysis as unset/null.
    if value == 0:
        return None
    return value


def _format_context_tokens(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}k"
    return str(tokens)


def normalize_context(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = str(int(value))
    elif isinstance(value, str):
        raw = value.strip().lower()
    else:
        return None

    if not raw:
        return None

    match = re.fullmatch(r"([0-9]+)\s*([kmb]?)", raw)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "b":
        tokens = amount * 1_000_000_000
    elif unit == "m":
        tokens = amount * 1_000_000
    elif unit == "k":
        tokens = amount * 1_000
    else:
        # Bare values are ambiguous; treat large values as raw tokens,
        # otherwise as shorthand for kilotokens (e.g. 262 -> 262k).
        tokens = amount if amount >= 10_000 else amount * 1_000

    # Snap close binary-window aliases from AA pages (e.g. 262k -> 256k).
    should_snap = unit in {"", "k"}
    if should_snap:
        canonical = [
            1_000,
            2_000,
            4_000,
            8_000,
            16_000,
            32_000,
            64_000,
            128_000,
            256_000,
            512_000,
            1_024_000,
            2_048_000,
        ]
        for target in canonical:
            if abs(tokens - target) / target <= 0.03:
                tokens = target
                break

    return _format_context_tokens(tokens)


def print_changes_table(changes: list[tuple[str, str, Any, Any]]) -> None:
    headers = ("MODEL", "BENCHMARK", "VALUE", "UPDATED")
    rows: list[tuple[str, str, str, str]] = [
        (model, benchmark, fmt_change_value(prev_value), fmt_change_value(new_value))
        for model, benchmark, prev_value, new_value in changes
    ]

    data = [headers, *rows]
    widths = [max(len(row[i]) for row in data) for i in range(4)]
    fmt = f"{{:<{widths[0]}}}  {{:<{widths[1]}}}  {{:<{widths[2]}}}  {{:<{widths[3]}}}"

    print(fmt.format(*headers))
    if not rows:
        print(fmt.format("-", "-", "-", "-"))
        return
    for row in rows:
        print(fmt.format(*row))


SCORE_MAPPINGS: dict[str, tuple[tuple[str, ...], Callable[[Any], Any]]] = {
    "terminal_bench_hard": (("terminalbench_hard",), to_percent),
    "tau2_bench_telecom": (("tau2",), to_percent),
    "aime_2025": (("aime_25",), to_percent),
    "mmmu_pro": (("mmmu_pro", "mmlu_pro"), to_percent),
    "gpqa_diamond": (("gpqa",), to_percent),
    "livecodebench": (("livecodebench",), to_percent),
    "scicode": (("scicode",), to_percent),
    "hle": (("hle",), to_percent),
    "aa_intelligence_index": (("artificial_analysis_intelligence_index",), lambda v: v),
    "aa_coding_index": (("artificial_analysis_coding_index",), lambda v: v),
}

def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Update benchmark JSON scores by querying artificialanalysis.py with model names as slugs."
    )
    parser.add_argument("json_file", help="Path to JSON file to read/update")
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    return parser.parse_args()


def unique_names(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def fetch_available_slugs(aa_script: Path) -> set[str]:
    cmd = [sys.executable, str(aa_script), "--list-models"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"artificialanalysis.py --list-models failed ({proc.returncode}): {proc.stderr.strip()}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def fetch_aa_data(aa_script: Path, slugs: list[str]) -> dict[str, dict[str, Any]]:
    cmd = [sys.executable, str(aa_script), "-o", "json"]
    for slug in slugs:
        cmd.extend(["-m", slug])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"artificialanalysis.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    data = payload.get("data", [])
    by_slug: dict[str, dict[str, Any]] = {}
    for row in data:
        slug = row.get("slug")
        if isinstance(slug, str) and slug:
            by_slug[slug] = row
    return by_slug


def update_scores(
    doc: dict[str, Any], by_slug: dict[str, dict[str, Any]]
) -> tuple[int, int, set[str], list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    seen_eval_keys: set[str] = set()
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        aa_model = by_slug.get(slug)
        if aa_model is None:
            continue

        matched += 1
        evaluations = aa_model.get("evaluations", {})
        if not isinstance(evaluations, dict):
            continue
        seen_eval_keys.update(k for k in evaluations.keys() if isinstance(k, str))

        old_context = model.get("context")
        new_context = normalize_context(aa_model.get("context"))
        if not (old_context is not None and new_context is None) and old_context != new_context:
            model["context"] = new_context
            updated += 1
            changes.append((slug, "context", old_context, new_context))

        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for llm_key, (aa_keys, transform) in SCORE_MAPPINGS.items():
            aa_value = None
            for aa_key in aa_keys:
                if aa_key in evaluations and evaluations.get(aa_key) is not None:
                    aa_value = evaluations.get(aa_key)
                    break
            new_value = transform(normalize_aa_value(aa_value))
            old_value = scores.get(llm_key)

            # Never overwrite an existing non-null value with null.
            if old_value is not None and new_value is None:
                continue

            if old_value != new_value:
                scores[llm_key] = new_value
                updated += 1
                changes.append((slug, llm_key, old_value, new_value))

    return matched, updated, seen_eval_keys, changes


def main() -> int:
    args = parse_args()
    llm_path = Path(args.json_file)
    aa_path = AA_SCRIPT

    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    models = doc.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Invalid JSON: models must be a list")

    slugs = unique_names(models)
    available_slugs = fetch_available_slugs(aa_path)
    existing_slugs = [slug for slug in slugs if slug in available_slugs]
    by_slug = fetch_aa_data(aa_path, existing_slugs)
    matched, _updated, seen_eval_keys, changes = update_scores(doc, by_slug)

    missing = [slug for slug in slugs if slug not in available_slugs]
    if args.write:
        llm_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"models in {llm_path}: {len(slugs)}")
    print(f"models available on artificialanalysis.py: {len(existing_slugs)}")
    print(f"models returned by artificialanalysis.py: {len(by_slug)}")
    if missing:
        print("missing models:")
        for slug in missing:
            print(f"  - {slug}")
    mapped_aa_keys = {aa_key for aa_keys, _transform in SCORE_MAPPINGS.values() for aa_key in aa_keys}
    ignored_aa_keys = sorted(seen_eval_keys - mapped_aa_keys)
    print("ignored keys:")
    if ignored_aa_keys:
        for key in ignored_aa_keys:
            print(f"  - {key}")
    else:
        print("  - (none)")
    print()
    print_changes_table(changes)
    print()
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
