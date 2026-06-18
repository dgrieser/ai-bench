#!/usr/bin/env python3
"""Add a new model entry to llm.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import termios
import tty
from datetime import date
from pathlib import Path
from typing import Any

from _swe_rebench_mapping import (
    add_rebench_mapping,
    fetch_swe_rebench_model_names,
    load_rebench_to_slug_mapping,
)
from _osworld_mapping import (
    add_osworld_mapping,
    fetch_osworld_model_names,
    load_osworld_to_slug_mapping,
)
from _deepswe_mapping import (
    add_deepswe_mapping,
    fetch_deepswe_model_names,
    load_deepswe_to_slug_mapping,
)
from _frontierswe_mapping import (
    add_frontierswe_mapping,
    fetch_frontierswe_model_names,
    load_frontierswe_to_slug_mapping,
)
from _huggingface_mapping import (
    add_hf_mapping,
    add_hf_unmappable,
    fetch_huggingface_benchmark_names,
    load_reviewed_hf_labels,
)
from _llmstats_mapping import (
    add_llmstats_mapping,
    fetch_llmstats_model_names,
    load_llmstats_to_slug_mapping,
    add_llmstats_benchmark_mapping,
    add_llmstats_benchmark_unmappable,
    fetch_llmstats_benchmark_names,
    load_reviewed_llmstats_benchmarks,
)


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


def prompt_value_with_default(label: str, default: str | None) -> str | None:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{label}{suffix}: ")
    except EOFError:
        return default

    if raw == "":
        return default
    return parse_nullable(raw)


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(description="Add a new model to llm.json.")
    parser.add_argument("json_file", nargs="?", default="llm.json", help="Path to the JSON file to update.")
    parser.add_argument("--name", help="Model name.")
    parser.add_argument("--url", help="Model URL.")
    parser.add_argument("--params", help="Model params string.")
    parser.add_argument("--context", help="Model context string.")
    parser.add_argument("--creator", help="Creator name.")
    parser.add_argument("--creator-url", help="Creator URL.")
    parser.add_argument(
        "--skip-aa",
        action="store_true",
        help="Skip artificialanalysis.py lookups (name autocomplete and field defaults).",
    )
    parser.add_argument(
        "--skip-swe-rebench",
        action="store_true",
        help="Skip the SWE-Rebench mapping prompt.",
    )
    parser.add_argument(
        "--skip-osworld",
        action="store_true",
        help="Skip the OSWorld mapping prompt.",
    )
    parser.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="Skip the Hugging Face benchmark-label mapping review.",
    )
    parser.add_argument(
        "--skip-deepswe",
        action="store_true",
        help="Skip the DeepSWE mapping prompt.",
    )
    parser.add_argument(
        "--skip-frontierswe",
        action="store_true",
        help="Skip the FrontierSWE mapping prompt.",
    )
    parser.add_argument(
        "--skip-llmstats",
        action="store_true",
        help="Skip the llm-stats.com model and benchmark-label mapping prompts.",
    )
    return parser.parse_args()


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


def get_unique_values(models: list[dict[str, Any]], key: str) -> list[str]:
    values = {
        model.get(key)
        for model in models
        if isinstance(model, dict) and isinstance(model.get(key), str) and model.get(key)
    }
    return sorted(values)


def get_creator_names(models: list[dict[str, Any]]) -> list[str]:
    values = {
        creator.get("name")
        for model in models
        if isinstance(model, dict)
        for creator in [model.get("creator")]
        if isinstance(creator, dict) and isinstance(creator.get("name"), str) and creator.get("name")
    }
    return sorted(values)


def get_creator_urls(models: list[dict[str, Any]]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for model in models:
        creator = model.get("creator")
        if not isinstance(creator, dict):
            continue
        name = creator.get("name")
        url = creator.get("url")
        if isinstance(name, str) and name and isinstance(url, str) and url and name not in urls:
            urls[name] = url
    return urls


def fetch_aa_model_names() -> list[str]:
    aa_script = Path(__file__).resolve().with_name("artificialanalysis.py")
    if not aa_script.exists():
        return []

    proc = subprocess.run(
        [sys.executable, str(aa_script), "--list-models"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_aa_model_defaults(model_name: str) -> dict[str, str]:
    aa_script = Path(__file__).resolve().with_name("artificialanalysis.py")
    if not aa_script.exists():
        return {}

    proc = subprocess.run(
        [sys.executable, str(aa_script), "--model", model_name, "--output", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {}

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return {}

    row = data[0]
    if not isinstance(row, dict) or row.get("slug") != model_name:
        return {}

    defaults: dict[str, str] = {}
    for source_key, dest_key in (("url", "url"), ("context", "context")):
        value = row.get(source_key)
        if isinstance(value, str) and value.strip():
            defaults[dest_key] = value.strip()

    creator = row.get("model_creator")
    if isinstance(creator, dict):
        creator_name = creator.get("name")
        creator_url = creator.get("url")
        if isinstance(creator_name, str) and creator_name.strip():
            defaults["creator"] = creator_name.strip()
        if isinstance(creator_url, str) and creator_url.strip():
            defaults["creator_url"] = creator_url.strip()

    return defaults


def maybe_add_osworld_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_osworld_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        osworld_names = fetch_osworld_model_names()
    except RuntimeError as exc:
        print(f"Skipping OSWorld mapping prompt: {exc}")
        return

    if not osworld_names:
        return

    osworld_name = prompt_select_or_new("OSWorld model", osworld_names)
    if not osworld_name:
        return

    add_osworld_mapping(osworld_name, model_name)
    print(f"Added OSWorld mapping '{osworld_name}' -> '{model_name}'")


def maybe_add_swe_rebench_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_rebench_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        rebench_names = fetch_swe_rebench_model_names()
    except RuntimeError as exc:
        print(f"Skipping SWE-Rebench mapping prompt: {exc}")
        return

    if not rebench_names:
        return

    rebench_name = prompt_select_or_new("SWE-Rebench model", rebench_names)
    if not rebench_name:
        return

    add_rebench_mapping(rebench_name, model_name)
    print(f"Added SWE-Rebench mapping '{rebench_name}' -> '{model_name}'")


def maybe_add_deepswe_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_deepswe_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        deepswe_names = fetch_deepswe_model_names()
    except RuntimeError as exc:
        print(f"Skipping DeepSWE mapping prompt: {exc}")
        return

    if not deepswe_names:
        return

    deepswe_name = prompt_select_or_new("DeepSWE model", deepswe_names)
    if not deepswe_name:
        return

    add_deepswe_mapping(deepswe_name, model_name)
    print(f"Added DeepSWE mapping '{deepswe_name}' -> '{model_name}'")


def maybe_add_frontierswe_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_frontierswe_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        frontierswe_names = fetch_frontierswe_model_names()
    except RuntimeError as exc:
        print(f"Skipping FrontierSWE mapping prompt: {exc}")
        return

    if not frontierswe_names:
        return

    frontierswe_name = prompt_select_or_new("FrontierSWE model", frontierswe_names)
    if not frontierswe_name:
        return

    add_frontierswe_mapping(frontierswe_name, model_name)
    print(f"Added FrontierSWE mapping '{frontierswe_name}' -> '{model_name}'")


def maybe_add_llmstats_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_llmstats_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        llmstats_names = fetch_llmstats_model_names()
    except RuntimeError as exc:
        print(f"Skipping llm-stats mapping prompt: {exc}")
        return

    if not llmstats_names:
        return

    llmstats_name = prompt_select_or_new("llm-stats model", llmstats_names)
    if not llmstats_name:
        return

    add_llmstats_mapping(llmstats_name, model_name)
    print(f"Added llm-stats mapping '{llmstats_name}' -> '{model_name}'")


def maybe_add_llmstats_benchmark_mapping(doc: dict[str, Any], interactive: bool) -> None:
    if not interactive:
        return

    benchmark_keys = sorted(doc["benchmarks"].keys())

    try:
        llmstats_labels = fetch_llmstats_benchmark_names()
    except RuntimeError as exc:
        print(f"Skipping llm-stats benchmark mapping prompt: {exc}")
        return

    reviewed = load_reviewed_llmstats_benchmarks()
    unreviewed = [label for label in llmstats_labels if label not in reviewed]
    if not unreviewed:
        return

    for label in unreviewed:
        key = prompt_key_for_label("llm-stats benchmark", label, benchmark_keys)
        if not key:
            add_llmstats_benchmark_unmappable(label)
            print(f"Recorded llm-stats benchmark '{label}' as unmappable")
            continue
        add_llmstats_benchmark_mapping(label, key)
        print(f"Added llm-stats benchmark mapping '{label}' -> '{key}'")


def prompt_key_for_label(
    source_label: str, label: str, keys: list[str], default: str | None = None
) -> str | None:
    options_lower = {k.lower(): k for k in keys}
    while True:
        raw = prompt_select_or_new(
            f"Map {source_label} '{label}' to llm.json benchmark", keys, default=default
        )
        if raw is None:
            return None
        canonical = options_lower.get(raw.lower())
        if canonical is not None:
            return canonical
        print("Selection must match an existing llm.json benchmark key. Press Enter to skip.")


def maybe_add_huggingface_mapping(doc: dict[str, Any], interactive: bool) -> None:
    if not interactive:
        return

    benchmark_keys = sorted(doc["benchmarks"].keys())

    try:
        hf_labels = fetch_huggingface_benchmark_names()
    except RuntimeError as exc:
        print(f"Skipping Hugging Face mapping prompt: {exc}")
        return

    reviewed = load_reviewed_hf_labels()
    unreviewed = [label for label in hf_labels if label not in reviewed]
    if not unreviewed:
        return

    for label in unreviewed:
        key = prompt_key_for_label("HF label", label, benchmark_keys)
        if not key:
            add_hf_unmappable(label)
            print(f"Recorded HF label '{label}' as unmappable")
            continue
        add_hf_mapping(label, key)
        print(f"Added HF mapping '{label}' -> '{key}'")


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


def supports_live_selector() -> bool:
    term = os.getenv("TERM", "")
    return sys.stdin.isatty() and sys.stdout.isatty() and term and term.lower() != "dumb"


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


def prompt_live_select_or_new(
    label: str, options: list[str], allow_empty: bool = True, default: str | None = None
) -> str | None:
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    buffer = default or ""
    tab_index = -1
    lines_drawn = 0

    try:
        tty.setraw(fd)
        while True:
            matches = find_matches(buffer, options)
            lines_drawn = _render_live_selector(label, buffer, matches, lines_drawn)
            char = sys.stdin.read(1)

            if char in {"\r", "\n"}:
                value = parse_nullable(buffer)
                _clear_live_selector(lines_drawn)
                if value is None:
                    if allow_empty:
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        return None
                    print(f"{label} is required.")
                    buffer = ""
                    tab_index = -1
                    lines_drawn = 0
                    continue
                sys.stdout.write(f"{label}: {buffer}\r\n")
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
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

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


def prompt_select_or_new(
    label: str, options: list[str], allow_empty: bool = True, default: str | None = None
) -> str | None:
    if not options:
        if default is not None:
            return prompt_value_with_default(label, default)
        return prompt_value(label)

    if supports_live_selector():
        return prompt_live_select_or_new(label, options, allow_empty=allow_empty, default=default)

    while True:
        suffix = f" [{default}]" if default else ""
        try:
            raw = input(f"{label} (type to search or enter a new value){suffix}: ")
        except EOFError:
            return default

        if raw == "":
            if default is not None:
                return default
            if allow_empty:
                return None
            print(f"{label} is required.")
            continue

        value = parse_nullable(raw)
        if value is None:
            if allow_empty:
                return None
            print(f"{label} is required.")
            continue

        matches = find_matches(value, options)
        if not matches:
            return value
        if len(matches) == 1 and matches[0].lower() == value.lower():
            return matches[0]

        print(f"Matches for {label}:")
        for idx, match in enumerate(matches, start=1):
            print(f"  {idx}. {match}")

        try:
            choice = input("Select number or press Enter to keep typed value: ").strip()
        except EOFError:
            return value

        if choice == "":
            return value
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(matches):
                return matches[index]
        print("Invalid selection.")


def get_value(
    args_value: str | None, label: str, interactive: bool, default: str | None = None
) -> str | None:
    parsed = parse_nullable(args_value)
    if parsed is not None or args_value is not None:
        return parsed
    if interactive:
        if default is not None:
            return prompt_value_with_default(label, default)
        return prompt_value(label)
    return default


def build_model(doc: dict[str, Any], args: argparse.Namespace, interactive: bool) -> dict[str, Any]:
    models = doc["models"]
    benchmark_keys = list(doc["benchmarks"].keys())

    if interactive and args.name is None:
        aa_names = [] if args.skip_aa else fetch_aa_model_names()
        name = prompt_select_or_new("Name", aa_names, allow_empty=False)
    else:
        name = get_value(args.name, "Name", interactive)
    if not name:
        raise ValueError("Model name is required.")

    aa_defaults = {} if args.skip_aa else fetch_aa_model_defaults(name)

    url = get_value(args.url, "URL", interactive, aa_defaults.get("url"))

    if interactive and args.params is None:
        params = prompt_select_or_new("Params", get_unique_values(models, "params"))
    else:
        params = get_value(args.params, "Params", interactive)

    if interactive and args.context is None:
        context = prompt_select_or_new(
            "Context",
            get_unique_values(models, "context"),
            default=aa_defaults.get("context"),
        )
    else:
        context = get_value(args.context, "Context", interactive, aa_defaults.get("context"))

    if interactive and args.creator is None:
        creator_name = prompt_select_or_new(
            "Creator",
            get_creator_names(models),
            default=aa_defaults.get("creator"),
        )
    else:
        creator_name = get_value(args.creator, "Creator", interactive, aa_defaults.get("creator"))

    creator_urls = get_creator_urls(models)
    creator_url_default = aa_defaults.get("creator_url") or (
        creator_urls.get(creator_name) if creator_name else None
    )
    if interactive and args.creator_url is None and creator_url_default:
        creator_url = prompt_value_with_default("Creator URL", creator_url_default)
    else:
        creator_url = get_value(args.creator_url, "Creator URL", interactive, creator_url_default)

    return {
        "name": name,
        "date_added": date.today().isoformat(),
        "url": url,
        "params": params,
        "context": context,
        "creator": {
            "name": creator_name,
            "url": creator_url,
        },
        "scores": {key: None for key in benchmark_keys},
    }


def ensure_unique_name(models: list[dict[str, Any]], name: str) -> None:
    for model in models:
        if model.get("name") == name:
            raise ValueError(f"Model '{name}' already exists.")


def write_doc(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    path = Path(args.json_file)
    doc = load_doc(path)

    interactive = sys.stdin.isatty()
    model = build_model(doc, args, interactive)
    models = doc["models"]
    ensure_unique_name(models, model["name"])
    models.append(model)
    write_doc(path, doc)
    if not args.skip_swe_rebench:
        maybe_add_swe_rebench_mapping(model["name"], interactive)
    if not args.skip_osworld:
        maybe_add_osworld_mapping(model["name"], interactive)
    if not args.skip_deepswe:
        maybe_add_deepswe_mapping(model["name"], interactive)
    if not args.skip_frontierswe:
        maybe_add_frontierswe_mapping(model["name"], interactive)
    if not args.skip_llmstats:
        maybe_add_llmstats_mapping(model["name"], interactive)
        maybe_add_llmstats_benchmark_mapping(doc, interactive)
    if not args.skip_huggingface:
        maybe_add_huggingface_mapping(doc, interactive)

    print(f"Added model '{model['name']}' to {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
