#!/usr/bin/env python3
"""Edit model metadata and scores in llm.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import termios
import tty
from datetime import date
from pathlib import Path
from typing import Any

from _scores import stamp_score_updated


SCORE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")

METADATA_FIELDS = {"context": "Context Window", "params": "Model Size"}


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def load_doc(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
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
    short_options_with_values = {"-m", "-b"}
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            if i + 1 < len(args):
                json_file = args[i + 1]
            break
        if token == "--missing":
            i += 1
            continue
        if token in short_options_with_values:
            i += 2
            continue
        if any(token.startswith(option) and token != option for option in short_options_with_values):
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
    parser = HelpOnErrorArgumentParser(description="Edit model metadata and scores in llm.json.")
    parser.add_argument("json_file", nargs="?", default="llm.json", help="Path to the JSON file to update.")
    parser.add_argument("-m", "--model", help="Model name to update.")
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Interactively prompt only for missing values. Can be scoped with --model and/or --benchmark.",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        choices=sorted(doc["benchmarks"].keys()),
        help="Benchmark key to scope --missing to. Repeat to include multiple benchmarks.",
    )
    parser.add_argument(
        "--after",
        metavar="YYYY-MM-DD",
        help="Scope --missing to models whose date_added is after this date.",
    )
    parser.add_argument("--context", help="Context window, e.g. 256k. Use 'null' to clear.")
    parser.add_argument("--params", help="Model size, e.g. 123B or 230B-A10B. Use 'null' to clear.")

    metadata_flags = {f"--{key}" for key in METADATA_FIELDS}
    for key, benchmark in doc["benchmarks"].items():
        flag = f"--{key.replace('_', '-')}"
        if flag in metadata_flags:
            raise ValueError(f"Benchmark key '{key}' collides with the metadata flag {flag}.")
        parser.add_argument(flag, dest=key, help=f"Score for {benchmark.get('name', key)}. Use 'null' to clear.")

    return parser.parse_args(argv)


def find_model(models: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for model in models:
        if model.get("name") == name:
            return model
    return None


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"Invalid date '{value}': expected format YYYY-MM-DD.")


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


def parse_metadata_value(raw: str) -> str | None:
    text = raw.strip()
    if not text or text.lower() == "null":
        return None
    return text


def format_score_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def get_existing_values(models: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            model[key]
            for model in models
            if isinstance(model, dict) and isinstance(model.get(key), str) and model[key].strip()
        }
    )


def is_missing_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


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


def prompt_metadata_value(label: str, current: Any, options: list[str]) -> str | None:
    """Prompt for a free-text metadata value, offering existing values as completions.

    Empty input keeps the current value; 'null' clears it. Tab cycles through the
    fuzzy matches, but any value is accepted, not only existing ones.
    """
    current_text = current if isinstance(current, str) and current.strip() else None
    label_text = f"{label} ({current_text})" if current_text else label

    if supports_live_selector() and options:
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        buffer = ""
        tab_index = -1
        lines_drawn = 0

        try:
            tty.setraw(fd)
            while True:
                matches = find_matches(buffer, options)
                lines_drawn = _render_live_selector(label_text, buffer, matches, lines_drawn)
                char = sys.stdin.read(1)

                if char in {"\r", "\n"}:
                    value = current_text if not buffer.strip() else parse_metadata_value(buffer)
                    _clear_live_selector(lines_drawn)
                    sys.stdout.write(f"{label}: {value if value is not None else 'null'}\r\n")
                    sys.stdout.flush()
                    return value

                if char == "\t":
                    if matches:
                        tab_index = (tab_index + 1) % len(matches)
                        buffer = matches[tab_index]
                    continue

                if char == "\x03":
                    raise KeyboardInterrupt

                if char == "\x04":
                    _clear_live_selector(lines_drawn)
                    return current_text

                if char in {"\x7f", "\b"}:
                    buffer = buffer[:-1]
                    tab_index = -1
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
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)

    try:
        raw = input(f"{label_text}: ")
    except EOFError:
        return current_text

    if not raw.strip():
        return current_text
    return parse_metadata_value(raw)


def get_missing_score_keys(doc: dict[str, Any], model: dict[str, Any]) -> list[str]:
    scores = model.setdefault("scores", {})
    if not isinstance(scores, dict):
        raise ValueError(f"Model '{model.get('name')}' has a non-object scores field.")
    return [key for key in doc["benchmarks"] if scores.get(key) is None]


def get_missing_metadata_keys(model: dict[str, Any]) -> list[str]:
    return [key for key in METADATA_FIELDS if is_missing_text(model.get(key))]


def collect_updates(
    doc: dict[str, Any], args: argparse.Namespace, interactive: bool
) -> tuple[dict[str, Any], dict[str, int | float | None], dict[str, str | None]]:
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

    score_updates: dict[str, int | float | None] = {}
    for key, benchmark in doc["benchmarks"].items():
        raw_value = getattr(args, key)
        current = scores.get(key)

        if raw_value is not None:
            score_updates[key] = parse_score_value(raw_value)
            continue

        if interactive:
            score_updates[key] = prompt_score(benchmark.get("name", key), current)

    metadata_updates: dict[str, str | None] = {}
    for key, label in METADATA_FIELDS.items():
        raw_value = getattr(args, key)

        if raw_value is not None:
            metadata_updates[key] = parse_metadata_value(raw_value)
            continue

        if interactive:
            metadata_updates[key] = prompt_metadata_value(
                label, model.get(key), get_existing_values(models, key)
            )

    return model, score_updates, metadata_updates


def collect_missing_updates(
    doc: dict[str, Any],
    interactive: bool,
    model_name: str | None = None,
    benchmark_keys: list[str] | None = None,
    after: date | None = None,
) -> list[tuple[dict[str, Any], dict[str, int | float | None], dict[str, str | None]]]:
    if not interactive:
        raise ValueError("--missing requires interactive mode.")

    benchmark_filter = set(benchmark_keys or [])
    planned: list[tuple[dict[str, Any], dict[str, int | float | None], dict[str, str | None]]] = []
    for model in doc["models"]:
        current_model_name = model.get("name")
        if not isinstance(current_model_name, str):
            continue
        if model_name is not None and current_model_name != model_name:
            continue
        if after is not None:
            raw_date = model.get("date_added")
            try:
                model_date = date.fromisoformat(raw_date) if isinstance(raw_date, str) and raw_date.strip() else None
            except ValueError:
                model_date = None
            if model_date is None or model_date <= after:
                continue
        missing_keys = get_missing_score_keys(doc, model)
        if benchmark_filter:
            missing_keys = [key for key in missing_keys if key in benchmark_filter]
        missing_metadata_keys = get_missing_metadata_keys(model)
        if benchmark_filter:
            missing_metadata_keys = []
        if not missing_keys and not missing_metadata_keys:
            continue

        print()
        print(f"Model: {current_model_name}")
        score_updates: dict[str, int | float | None] = {}
        metadata_updates: dict[str, str | None] = {}
        scores = model["scores"]
        for key in missing_keys:
            benchmark = doc["benchmarks"][key]
            score_updates[key] = prompt_score(benchmark.get("name", key), scores.get(key))
        for key in missing_metadata_keys:
            metadata_updates[key] = prompt_metadata_value(
                METADATA_FIELDS[key], model.get(key), get_existing_values(doc["models"], key)
            )
        planned.append((model, score_updates, metadata_updates))

    return planned


def write_doc(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    path = Path(infer_json_file())
    doc = load_doc(path)
    args = parse_args(doc)

    interactive = sys.stdin.isatty()

    after: date | None = None
    if args.after is not None:
        if not args.missing:
            raise ValueError("--after can only be used with --missing.")
        after = parse_date(args.after)

    if args.missing:
        score_flags = [key for key in doc["benchmarks"] if getattr(args, key) is not None]
        if score_flags:
            raise ValueError("--missing cannot be combined with benchmark score flags.")
        metadata_flags = [key for key in METADATA_FIELDS if getattr(args, key) is not None]
        if metadata_flags:
            raise ValueError(
                "--missing cannot be combined with metadata flags: "
                + ", ".join(f"--{key}" for key in metadata_flags)
            )
        if args.model is not None and find_model(doc["models"], args.model) is None:
            raise ValueError(f"Model '{args.model}' does not exist.")

        planned = collect_missing_updates(
            doc, interactive, model_name=args.model, benchmark_keys=args.benchmark, after=after
        )
        if not planned:
            if args.model is not None and args.benchmark:
                scope = f"missing values for model '{args.model}' in benchmark(s): {', '.join(args.benchmark)}"
            elif args.model is not None:
                scope = f"missing values for model '{args.model}'"
            elif args.benchmark:
                scope = f"missing values in benchmark(s): {', '.join(args.benchmark)}"
            else:
                scope = "missing scores or metadata"
            if after is not None:
                scope += f" added after {after.isoformat()}"
            print(f"No models with {scope}.")
            return 0

        changed = 0
        models_changed = 0
        for model, score_updates, metadata_updates in planned:
            model_changed = False
            scores = model["scores"]
            for key, value in score_updates.items():
                if scores.get(key) != value:
                    scores[key] = value
                    stamp_score_updated(model, key)
                    changed += 1
                    model_changed = True
            for key, value in metadata_updates.items():
                if model.get(key) != value:
                    model[key] = value
                    changed += 1
                    model_changed = True
            if model_changed:
                models_changed += 1

        write_doc(path, doc)
        print(f"Updated {changed} field(s) across {models_changed} model(s) in {path}")
        return 0

    model, score_updates, metadata_updates = collect_updates(doc, args, interactive)
    if not score_updates and not metadata_updates:
        raise ValueError("No score or metadata updates provided.")

    changed = 0
    scores = model["scores"]
    for key, value in score_updates.items():
        if scores.get(key) != value:
            scores[key] = value
            stamp_score_updated(model, key)
            changed += 1
    for key, value in metadata_updates.items():
        if model.get(key) != value:
            model[key] = value
            changed += 1

    write_doc(path, doc)
    print(f"Updated {changed} field(s) for '{model['name']}' in {path}")
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
