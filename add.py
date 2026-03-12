#!/usr/bin/env python3
"""Add a new model entry to llm.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_nullable(value: str | None) -> str | None:
    if value is None:
        return None

    text = value.strip()
    if not text or text.lower() == "null":
        return None
    return text


def prompt_value(label: str) -> str | None:
    try:
        return parse_nullable(input(f"{label}: "))
    except EOFError:
        return None


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(description="Add a new model to llm.json.")
    parser.add_argument("json_file", nargs="?", default="llm.json", help="Path to the JSON file to update.")
    parser.add_argument("--name", help="Model name.")
    parser.add_argument("--url", help="Model URL.")
    parser.add_argument("--params", help="Model params string.")
    parser.add_argument("--context", help="Model context string.")
    parser.add_argument("--creator", help="Creator name.")
    parser.add_argument("--creator-url", help="Creator URL.")
    return parser.parse_args()


def load_doc(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        doc = json.load(handle)

    if not isinstance(doc, dict):
        raise ValueError("Top-level JSON value must be an object.")
    if not isinstance(doc.get("benchmarks"), dict):
        raise ValueError("JSON must contain a benchmarks object.")
    if not isinstance(doc.get("models"), list):
        raise ValueError("JSON must contain a models array.")
    return doc


def get_value(args_value: str | None, label: str, interactive: bool) -> str | None:
    parsed = parse_nullable(args_value)
    if parsed is not None or args_value is not None:
        return parsed
    if interactive:
        return prompt_value(label)
    return None


def build_model(args: argparse.Namespace, benchmark_keys: list[str], interactive: bool) -> dict[str, Any]:
    name = get_value(args.name, "Name", interactive)
    if not name:
        raise ValueError("Model name is required.")

    return {
        "name": name,
        "date_added": date.today().isoformat(),
        "url": get_value(args.url, "URL", interactive),
        "params": get_value(args.params, "Params", interactive),
        "context": get_value(args.context, "Context", interactive),
        "creator": {
            "name": get_value(args.creator, "Creator", interactive),
            "url": get_value(args.creator_url, "Creator URL", interactive),
        },
        "scores": {key: None for key in benchmark_keys},
    }


def ensure_unique_name(models: list[dict[str, Any]], name: str) -> None:
    for model in models:
        if model.get("name") == name:
            raise ValueError(f"Model '{name}' already exists.")


def write_doc(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    path = Path(args.json_file)
    doc = load_doc(path)

    interactive = sys.stdin.isatty()
    benchmark_keys = list(doc["benchmarks"].keys())
    model = build_model(args, benchmark_keys, interactive)

    models = doc["models"]
    ensure_unique_name(models, model["name"])
    models.append(model)
    write_doc(path, doc)

    print(f"Added model '{model['name']}' to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
