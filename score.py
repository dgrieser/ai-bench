#!/usr/bin/env python3
"""Set model scores in llm.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import termios
import tty
from pathlib import Path
from typing import Any


SCORE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


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


def infer_json_file(argv: list[str] | None = None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    json_file = "llm.json"
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--missing":
            i += 1
            continue
        if token.startswith("--"):
            if "=" in token:
                i += 1
            else:
                i += 2
            continue
        json_file = token
        i += 1
    return json_file


def parse_args(doc: dict[str, Any], argv: list[str] | None = None) -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(description="Set model scores in llm.json.")
    parser.add_argument("json_file", nargs="?", default="llm.json", help="Path to the JSON file to update.")
    parser.add_argument("--model", help="Model name to update.")
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Interactively cycle through all models with missing scores and prompt only for missing values.",
    )

    for key, benchmark in doc["benchmarks"].items():
        flag = f"--{key.replace('_', '-')}"
        parser.add_argument(flag, dest=key, help=f"Score for {benchmark.get('name', key)}. Use 'null' to clear.")

    return parser.parse_args(argv)


def find_model(models: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for model in models:
        if model.get("name") == name:
            return model
    return None


def parse_nullable(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return text


def parse_score_value(raw: str) -> int | float | None:
    text = raw.strip()
    if text.lower() == "null":
        return None
    if not SCORE_RE.fullmatch(text):
        raise ValueError("expected a decimal number like 80 or 80.8, or 'null'")
    if "." in text:
        return float(text)
    return int(text)


def format_score_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def supports_live_selector() -> bool:
    term = os.getenv("TERM", "")
    return sys.stdin.isatty() and sys.stdout.isatty() and term and term.lower() != "dumb"


def fuzzy_match(query: str, option: str) -> tuple[int, int] | None:
    haystack = option.lower()
    needle = query.lower()

    if needle in haystack:
        return (0, haystack.index(needle))

    pos = 0
    gap_score = 0
    for char in needle:
        idx = haystack.find(char, pos)
        if idx == -1:
            return None
        gap_score += idx - pos
        pos = idx + 1
    return (1, gap_score)


def find_matches(query: str, options: list[str], limit: int = 10) -> list[str]:
    if not query:
        return options[:limit]

    scored: list[tuple[tuple[int, int, int], str]] = []
    for option in options:
        match = fuzzy_match(query, option)
        if match is None:
            continue
        scored.append(((match[0], match[1], len(option)), option))
    scored.sort(key=lambda item: item[0])
    return [option for _, option in scored[:limit]]


def _render_live_selector(label: str, buffer: str, matches: list[str], lines_drawn: int) -> int:
    if lines_drawn:
        sys.stdout.write(f"\x1b[{lines_drawn}F")

    sys.stdout.write("\r\x1b[2K")
    sys.stdout.write(f"{label}: {buffer}")

    for match in matches:
        sys.stdout.write("\r\n\x1b[2K")
        sys.stdout.write(f"  {match}")

    sys.stdout.write("\x1b[J")
    if matches:
        sys.stdout.write(f"\x1b[{len(matches)}F")
        sys.stdout.write(f"\r{label}: {buffer}")
    sys.stdout.flush()
    return 1 + len(matches)


def _clear_live_selector(lines_drawn: int) -> None:
    if not lines_drawn:
        return

    sys.stdout.write("\r\x1b[2K")
    for _ in range(lines_drawn - 1):
        sys.stdout.write("\r\n\x1b[2K")
    if lines_drawn > 1:
        sys.stdout.write(f"\x1b[{lines_drawn - 1}F")
    sys.stdout.write("\r")
    sys.stdout.flush()


def prompt_existing_value(label: str, options: list[str]) -> str:
    options_lower = {option.lower(): option for option in options}

    if supports_live_selector() and options:
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        buffer = ""
        tab_index = -1
        lines_drawn = 0
        error_message: str | None = None
        last_invalid_value: str | None = None

        try:
            tty.setraw(fd)
            while True:
                matches = find_matches(buffer, options)
                if error_message is not None:
                    _clear_live_selector(lines_drawn)
                    print(error_message)
                    lines_drawn = 0
                    error_message = None
                lines_drawn = _render_live_selector(label, buffer, matches, lines_drawn)
                char = sys.stdin.read(1)

                if char in {"\r", "\n"}:
                    value = parse_nullable(buffer) or ""
                    canonical = options_lower.get(value.lower())
                    _clear_live_selector(lines_drawn)
                    if canonical is None:
                        if last_invalid_value != value:
                            error_message = f"{label} must match an existing model."
                        last_invalid_value = value
                        buffer = ""
                        tab_index = -1
                        lines_drawn = 0
                        continue
                    last_invalid_value = None
                    sys.stdout.write(f"{label}: {canonical}\r\n")
                    sys.stdout.flush()
                    return canonical

                if char == "\t":
                    if matches:
                        tab_index = (tab_index + 1) % len(matches)
                        buffer = matches[tab_index]
                    continue

                if char == "\x03":
                    raise KeyboardInterrupt

                if char == "\x04":
                    _clear_live_selector(lines_drawn)
                    raise EOFError

                if char in {"\x7f", "\b"}:
                    buffer = buffer[:-1]
                    tab_index = -1
                    last_invalid_value = None
                    continue

                if char == "\x1b":
                    next_char = sys.stdin.read(1)
                    if next_char == "[":
                        sys.stdin.read(1)
                    tab_index = -1
                    continue

                if char.isprintable():
                    buffer += char
                    tab_index = -1
                    last_invalid_value = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)

    while True:
        raw = input(f"{label}: ").strip()
        canonical = options_lower.get(raw.lower())
        if canonical is not None:
            return canonical
        print(f"{label} must match an existing model.")


def prompt_score(label: str, current: Any) -> int | float | None:
    while True:
        prompt = f"{label} ({format_score_value(current)}): " if current is not None else f"{label}: "
        try:
            raw = input(prompt)
        except EOFError:
            return current

        if raw.strip() == "":
            return current

        try:
            return parse_score_value(raw)
        except ValueError as exc:
            print(f"Invalid value for {label}: {exc}")


def get_missing_score_keys(doc: dict[str, Any], model: dict[str, Any]) -> list[str]:
    scores = model.setdefault("scores", {})
    if not isinstance(scores, dict):
        raise ValueError(f"Model '{model.get('name')}' has a non-object scores field.")
    return [key for key in doc["benchmarks"] if scores.get(key) is None]


def collect_updates(
    doc: dict[str, Any], args: argparse.Namespace, interactive: bool
) -> tuple[dict[str, Any], dict[str, int | float | None]]:
    models = doc["models"]
    model_names = [model["name"] for model in models if isinstance(model.get("name"), str)]

    if args.model is not None:
        model_name = args.model
    elif interactive:
        model_name = prompt_existing_value("Model", model_names)
    else:
        raise ValueError("--model is required in non-interactive mode.")

    model = find_model(models, model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' does not exist.")

    scores = model.setdefault("scores", {})
    if not isinstance(scores, dict):
        raise ValueError(f"Model '{model_name}' has a non-object scores field.")

    updates: dict[str, int | float | None] = {}
    for key, benchmark in doc["benchmarks"].items():
        raw_value = getattr(args, key)
        current = scores.get(key)

        if raw_value is not None:
            updates[key] = parse_score_value(raw_value)
            continue

        if interactive:
            updates[key] = prompt_score(benchmark.get("name", key), current)

    return model, updates


def collect_missing_updates(
    doc: dict[str, Any], interactive: bool
) -> list[tuple[dict[str, Any], dict[str, int | float | None]]]:
    if not interactive:
        raise ValueError("--missing requires interactive mode.")

    planned: list[tuple[dict[str, Any], dict[str, int | float | None]]] = []
    for model in doc["models"]:
        if not isinstance(model.get("name"), str):
            continue
        missing_keys = get_missing_score_keys(doc, model)
        if not missing_keys:
            continue

        print()
        print(f"Model: {model['name']}")
        updates: dict[str, int | float | None] = {}
        scores = model["scores"]
        for key in missing_keys:
            benchmark = doc["benchmarks"][key]
            updates[key] = prompt_score(benchmark.get("name", key), scores.get(key))
        planned.append((model, updates))

    return planned


def write_doc(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=True) + "\n")


def main() -> int:
    path = Path(infer_json_file())
    doc = load_doc(path)
    args = parse_args(doc)

    interactive = sys.stdin.isatty()
    if args.missing:
        score_flags = [key for key in doc["benchmarks"] if getattr(args, key) is not None]
        if args.model is not None or score_flags:
            raise ValueError("--missing cannot be combined with --model or benchmark score flags.")

        planned = collect_missing_updates(doc, interactive)
        if not planned:
            print("No models with missing scores.")
            return 0

        changed = 0
        models_changed = 0
        for model, updates in planned:
            model_changed = False
            scores = model["scores"]
            for key, value in updates.items():
                if scores.get(key) != value:
                    scores[key] = value
                    changed += 1
                    model_changed = True
            if model_changed:
                models_changed += 1

        write_doc(path, doc)
        print(f"Updated {changed} score(s) across {models_changed} model(s) in {path}")
        return 0

    model, updates = collect_updates(doc, args, interactive)
    if not updates:
        raise ValueError("No score updates provided.")

    changed = 0
    scores = model["scores"]
    for key, value in updates.items():
        if scores.get(key) != value:
            scores[key] = value
            changed += 1

    write_doc(path, doc)
    print(f"Updated {changed} score(s) for '{model['name']}' in {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
