#!/usr/bin/env python3
"""Edit model metadata and scores in llm.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
import termios
import tty
from datetime import date
from pathlib import Path
from typing import Any

import derive_indexes
from _scores import editable_benchmarks, round_score, stamp_score_source, stamp_score_updated
from _selector import (
    clear_selector,
    find_matches,
    render_selector,
    supports_live_selector,
    tab_completion,
)


SCORE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")

# Every metadata field a person may set on an existing entry, and its prompt
# label. The set is add.py's: what that script asks when a model is created is
# what this one can change afterwards, so nothing is enterable once and then
# frozen. vram/vram_source are absent on purpose -- Spheron writes those, and a
# hand-typed figure would be recomputed away on the next run.
METADATA_FIELDS = {
    "context": "Context Window",
    "params": "Model Size",
    "url": "Model URL",
    "creator": "Creator",
    "creator_url": "Creator URL",
    "date_added": "Date Added",
}

METADATA_HELP = {
    "context": "Context window, e.g. 256k. Use 'null' to clear.",
    "params": "Model size, e.g. 123B or 230B-A10B. Use 'null' to clear.",
    "url": "Model page, e.g. its Hugging Face repo. Use 'null' to clear.",
    "creator": "Creator name, e.g. Mistral. Use 'null' to clear.",
    "creator_url": "Creator home page. Use 'null' to clear.",
    "date_added": "Date the entry was added, YYYY-MM-DD. Use 'null' to clear.",
}

# Where a field lives on the entry. Two of them sit inside the creator object,
# so nothing may assume model[key].
FIELD_PATHS = {"creator": ("creator", "name"), "creator_url": ("creator", "url")}


def field_path(key: str) -> tuple[str, ...]:
    return FIELD_PATHS.get(key, (key,))


def read_field(model: dict[str, Any], key: str) -> Any:
    node: Any = model
    for part in field_path(key):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def write_field(model: dict[str, Any], key: str, value: str | None) -> None:
    path = field_path(key)
    node = model
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


# The provenance of the scores one run writes: the date they were read and the
# page they were read from. Not per-benchmark flags like the scores themselves,
# because the answer is the same for every score typed in from one page in one
# sitting; a second sitting is a second run.
SCORE_PROVENANCE_FLAGS = ("--score-date", "--score-url")


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
    short_options_with_values = {"-m", "-b", "-f"}
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
        help=(
            "Interactively prompt only for missing values. "
            "Can be scoped with --model, --benchmark and/or --field."
        ),
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        choices=sorted(editable_benchmarks(doc).keys()),
        help="Benchmark key to scope --missing to. Repeat to include multiple benchmarks.",
    )
    parser.add_argument(
        "-f",
        "--field",
        action="append",
        choices=sorted(METADATA_FIELDS),
        help="Metadata field to scope --missing to. Repeat to include multiple fields.",
    )
    parser.add_argument(
        "--after",
        metavar="YYYY-MM-DD",
        help="Scope --missing to models whose date_added is after this date.",
    )
    for key, label in METADATA_FIELDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, help=METADATA_HELP[key])
    parser.add_argument(
        "--score-date",
        metavar="YYYY-MM-DD",
        help="Date to stamp on the scores this run changes. Defaults to today.",
    )
    parser.add_argument(
        "--score-url",
        metavar="URL",
        help="Page the scores this run changes were read from. Use 'null', or leave it "
        "off, to credit nobody -- which is what a hand edit means by default.",
    )

    reserved_flags = {
        f"--{key.replace('_', '-')}" for key in METADATA_FIELDS
    } | set(SCORE_PROVENANCE_FLAGS)
    for key, benchmark in editable_benchmarks(doc).items():
        flag = f"--{key.replace('_', '-')}"
        if flag in reserved_flags:
            raise ValueError(f"Benchmark key '{key}' collides with the reserved flag {flag}.")
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


def parse_url_field(raw: str | None, flag: str = "--url") -> str | None:
    """A URL, or None where there is none. Blank and 'null' both clear it."""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() == "null":
        return None
    if not text.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid URL '{raw}' for {flag}: expected one starting with http:// or https://."
        )
    return text


def parse_date_field(raw: str | None, flag: str = "--date-added") -> str | None:
    """An ISO date, or None. Checked here so a typo cannot become the entry's age."""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() == "null":
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"Invalid date '{raw}' for {flag}: expected format YYYY-MM-DD.") from None


def parse_score_source(raw: str | None) -> str | None:
    """The page a hand-entered score is credited to, or None for nobody.

    None is a real answer rather than a missing one: `stamp_score_source` takes
    it to mean the score reached llm.json through a person, which is the weakest
    rung of _precedence.source_rank() and so the one every scraper may
    overwrite. Naming the page a number was actually read from is therefore not
    cosmetic -- it moves the value onto that page's rung, where only that page's
    own rung or better replaces it.
    """
    return parse_url_field(raw, "--score-url")


def parse_metadata_value(raw: str) -> str | None:
    text = raw.strip()
    if not text or text.lower() == "null":
        return None
    return text


# Fields whose value is checked rather than taken as typed. A URL that is not
# one and a date that is not one are both silent lies once they are in the file.
FIELD_PARSERS = {
    "url": parse_url_field,
    "creator_url": parse_url_field,
    "date_added": parse_date_field,
}


def parse_field_value(key: str, raw: str | None) -> str | None:
    parser = FIELD_PARSERS.get(key)
    if parser is not None:
        return parser(raw, f"--{key.replace('_', '-')}")
    return parse_metadata_value(raw) if raw is not None else None


def format_score_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)


def get_existing_values(models: list[dict[str, Any]], key: str) -> list[str]:
    values = (read_field(model, key) for model in models if isinstance(model, dict))
    return sorted({value for value in values if isinstance(value, str) and value.strip()})


def is_missing_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def prompt_existing_value(label: str, options: list[str]) -> str:
    options_lower = {option.lower(): option for option in options}

    if supports_live_selector() and options:
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        buffer = ""
        lines_drawn = 0
        error_message: str | None = None
        last_invalid_value: str | None = None

        try:
            tty.setraw(fd)
            while True:
                matches = find_matches(buffer, options)
                if error_message is not None:
                    clear_selector(lines_drawn)
                    print(error_message)
                    lines_drawn = 0
                    error_message = None
                lines_drawn = render_selector(label, buffer, matches)
                char = sys.stdin.read(1)

                if char in {"\r", "\n"}:
                    value = parse_nullable(buffer) or ""
                    canonical = options_lower.get(value.lower())
                    clear_selector(lines_drawn)
                    if canonical is None:
                        if last_invalid_value != value:
                            error_message = f"{label} must match an existing model."
                        last_invalid_value = value
                        buffer = ""
                        lines_drawn = 0
                        continue
                    last_invalid_value = None
                    sys.stdout.write(f"{label}: {canonical}\r\n")
                    sys.stdout.flush()
                    return canonical

                if char == "\t":
                    completion = tab_completion(buffer, matches)
                    if completion is not None:
                        buffer = completion
                    continue

                if char == "\x03":
                    raise KeyboardInterrupt

                if char == "\x04":
                    clear_selector(lines_drawn)
                    raise EOFError

                if char in {"\x7f", "\b"}:
                    buffer = buffer[:-1]
                    last_invalid_value = None
                    continue

                if char == "\x1b":
                    next_char = sys.stdin.read(1)
                    if next_char == "[":
                        sys.stdin.read(1)
                    continue

                if char.isprintable():
                    buffer += char
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

    Empty input keeps the current value; 'null' clears it. Tab completes as far
    as the matches agree, but any value is accepted, not only existing ones.
    """
    current_text = current if isinstance(current, str) and current.strip() else None
    label_text = f"{label} ({current_text})" if current_text else label

    if supports_live_selector() and options:
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        buffer = ""
        lines_drawn = 0

        try:
            tty.setraw(fd)
            while True:
                matches = find_matches(buffer, options)
                lines_drawn = render_selector(label_text, buffer, matches)
                char = sys.stdin.read(1)

                if char in {"\r", "\n"}:
                    value = current_text if not buffer.strip() else parse_metadata_value(buffer)
                    clear_selector(lines_drawn)
                    sys.stdout.write(f"{label}: {value if value is not None else 'null'}\r\n")
                    sys.stdout.flush()
                    return value

                if char == "\t":
                    completion = tab_completion(buffer, matches)
                    if completion is not None:
                        buffer = completion
                    continue

                if char == "\x03":
                    raise KeyboardInterrupt

                if char == "\x04":
                    clear_selector(lines_drawn)
                    return current_text

                if char in {"\x7f", "\b"}:
                    buffer = buffer[:-1]
                    continue

                if char == "\x1b":
                    next_char = sys.stdin.read(1)
                    if next_char == "[":
                        sys.stdin.read(1)
                    continue

                if char.isprintable():
                    buffer += char
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
    return [key for key in editable_benchmarks(doc) if scores.get(key) is None]


def get_missing_metadata_keys(model: dict[str, Any]) -> list[str]:
    return [key for key in METADATA_FIELDS if is_missing_text(read_field(model, key))]


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
    for key, benchmark in editable_benchmarks(doc).items():
        raw_value = getattr(args, key)
        current = scores.get(key)

        if raw_value is not None:
            score_updates[key] = round_score(doc, key, parse_score_value(raw_value))
            continue

        if interactive:
            score_updates[key] = round_score(
                doc, key, prompt_score(benchmark.get("name", key), current)
            )

    metadata_updates: dict[str, str | None] = {}
    for key, label in METADATA_FIELDS.items():
        raw_value = getattr(args, key)

        if raw_value is not None:
            metadata_updates[key] = parse_field_value(key, raw_value)
            continue

        if interactive:
            # Re-asked rather than raised on: a mistyped URL should cost one
            # line, not the whole sitting.
            while True:
                answer = prompt_metadata_value(
                    label, read_field(model, key), get_existing_values(models, key)
                )
                try:
                    metadata_updates[key] = parse_field_value(key, answer)
                    break
                except ValueError as exc:
                    print(f"  {exc}")

    return model, score_updates, metadata_updates


def collect_missing_updates(
    doc: dict[str, Any],
    interactive: bool,
    model_name: str | None = None,
    benchmark_keys: list[str] | None = None,
    metadata_keys: list[str] | None = None,
    after: date | None = None,
) -> list[tuple[dict[str, Any], dict[str, int | float | None], dict[str, str | None]]]:
    if not interactive:
        raise ValueError("--missing requires interactive mode.")

    benchmark_filter = set(benchmark_keys or [])
    metadata_filter = set(metadata_keys or [])
    scoped = bool(benchmark_filter or metadata_filter)
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
        elif scoped:
            missing_keys = []
        missing_metadata_keys = get_missing_metadata_keys(model)
        if metadata_filter:
            missing_metadata_keys = [key for key in missing_metadata_keys if key in metadata_filter]
        elif scoped:
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
            score_updates[key] = round_score(
                doc, key, prompt_score(benchmark.get("name", key), scores.get(key))
            )
        for key in missing_metadata_keys:
            while True:
                answer = prompt_metadata_value(
                    METADATA_FIELDS[key], read_field(model, key), get_existing_values(doc["models"], key)
                )
                try:
                    metadata_updates[key] = parse_field_value(key, answer)
                    break
                except ValueError as exc:
                    print(f"  {exc}")
        planned.append((model, score_updates, metadata_updates))

    return planned


def refresh_derived_scores(doc: dict[str, Any]) -> None:
    """Recompute the derived columns after a hand-edited score, before the write.

    A derived column is a function of the other scores in llm.json, so an edit
    here leaves it stale -- and because the derived indexes rank models against
    each other, changing one model's score can move other models' values.
    Recomputed into the same write so llm.json is never saved half-updated.
    """
    derive_indexes.refresh_and_report(doc)


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

    if args.field and not args.missing:
        raise ValueError("--field can only be used with --missing.")

    # Which of the two were actually passed, kept separately from their values:
    # both parse to None when cleared, and "credit nobody" has to stay
    # distinguishable from "said nothing" for the checks below.
    provenance_flags = [
        flag
        for flag, raw in zip(SCORE_PROVENANCE_FLAGS, (args.score_date, args.score_url))
        if raw is not None
    ]

    if args.missing:
        score_flags = [key for key in editable_benchmarks(doc) if getattr(args, key) is not None]
        if score_flags:
            raise ValueError("--missing cannot be combined with benchmark score flags.")
        metadata_flags = [key for key in METADATA_FIELDS if getattr(args, key) is not None]
        if metadata_flags:
            raise ValueError(
                "--missing cannot be combined with metadata flags: "
                + ", ".join(f"--{key}" for key in metadata_flags)
            )
        if provenance_flags:
            # --missing walks every model with a gap; one date and one page
            # could not honestly describe that whole sweep.
            raise ValueError(
                "--missing cannot be combined with " + " or ".join(provenance_flags) + "."
            )
        if args.model is not None and find_model(doc["models"], args.model) is None:
            raise ValueError(f"Model '{args.model}' does not exist.")

        planned = collect_missing_updates(
            doc,
            interactive,
            model_name=args.model,
            benchmark_keys=args.benchmark,
            metadata_keys=args.field,
            after=after,
        )
        if not planned:
            scope = "missing values" if args.benchmark or args.field else "missing scores or metadata"
            details: list[str] = []
            if args.model is not None:
                details.append(f"model '{args.model}'")
            if args.benchmark:
                details.append(f"benchmark(s): {', '.join(args.benchmark)}")
            if args.field:
                details.append(f"field(s): {', '.join(args.field)}")
            if details:
                scope += " for " + "; ".join(details)
            if after is not None:
                scope += f" added after {after.isoformat()}"
            print(f"No models with {scope}.")
            return 0

        changed = 0
        models_changed = 0
        scores_changed = False
        for model, score_updates, metadata_updates in planned:
            model_changed = False
            scores = model["scores"]
            for key, value in score_updates.items():
                if scores.get(key) != value:
                    scores[key] = value
                    stamp_score_updated(model, key)
                    # A hand edit has no source page; whatever attribution the
                    # previous automated write left behind is now stale.
                    stamp_score_source(model, key, None)
                    changed += 1
                    model_changed = True
                    scores_changed = True
            for key, value in metadata_updates.items():
                if read_field(model, key) != value:
                    write_field(model, key, value)
                    changed += 1
                    model_changed = True
            if model_changed:
                models_changed += 1

        # Only scores feed the derived columns; a params/context edit cannot
        # move them.
        if scores_changed:
            refresh_derived_scores(doc)

        write_doc(path, doc)
        print(f"Updated {changed} field(s) across {models_changed} model(s) in {path}")
        return 0

    # Parsed after the --missing branch has returned, so a flag that does not
    # belong there is reported as not belonging rather than as malformed.
    score_date = parse_date(args.score_date).isoformat() if args.score_date is not None else None
    score_url = parse_score_source(args.score_url)

    model, score_updates, metadata_updates = collect_updates(doc, args, interactive)
    if not score_updates and not metadata_updates:
        raise ValueError("No score or metadata updates provided.")
    if provenance_flags and not score_updates:
        raise ValueError(
            " and ".join(provenance_flags)
            + " stamps the scores this run writes, so it needs one: pass a benchmark flag too."
        )

    changed = 0
    scores_changed = False
    scores = model["scores"]
    for key, value in score_updates.items():
        if scores.get(key) != value:
            scores[key] = value
            # Today and nobody unless --score-date and --score-url say
            # otherwise: a hand edit is read today from no page anybody can
            # cite, and whatever attribution the previous automated write left
            # behind is stale either way.
            stamp_score_updated(model, key, score_date)
            stamp_score_source(model, key, score_url)
            changed += 1
            scores_changed = True
    for key, value in metadata_updates.items():
        if read_field(model, key) != value:
            write_field(model, key, value)
            changed += 1

    if scores_changed:
        refresh_derived_scores(doc)

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
