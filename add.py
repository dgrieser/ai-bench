#!/usr/bin/env python3
"""Add a new model entry to llm.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import termios
import tty
from datetime import date
from pathlib import Path
from typing import Any

import _prompts
from _params import fetch_hf_params, normalize_params
from _selector import (
    clear_selector,
    find_matches,
    render_selector,
    supports_live_selector,
    tab_completion,
)
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
    fetch_all_deepswe_names,
    load_deepswe_to_slug_mapping,
)
from _frontiercode_mapping import (
    add_frontiercode_mapping,
    fetch_frontiercode_model_names,
    load_frontiercode_to_slug_mapping,
)
from _frontierswe_mapping import (
    add_frontierswe_mapping,
    fetch_frontierswe_model_names,
    load_frontierswe_to_slug_mapping,
)
from _swe_atlas_mapping import (
    add_swe_atlas_mapping,
    fetch_swe_atlas_model_names,
    load_swe_atlas_to_slug_mapping,
)
from _evals_report_mapping import (
    add_evals_report_mapping,
    fetch_evals_report_model_names,
    load_evals_report_to_slug_mapping,
)
from _swe_marathon_mapping import (
    add_swe_marathon_mapping,
    fetch_swe_marathon_model_names,
    load_swe_marathon_to_slug_mapping,
)
from _spheron_mapping import (
    add_spheron_mapping,
    hf_path_from_url,
    load_spheron_to_slug_mapping,
)
from _huggingface_mapping import (
    add_hf_mapping,
    add_hf_unmappable,
    fetch_huggingface_benchmark_names,
    load_reviewed_hf_labels,
)
from _scores import editable_benchmarks
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
        "--skip-frontiercode",
        action="store_true",
        help="Skip the FrontierCode mapping prompt.",
    )
    parser.add_argument(
        "--skip-swe-atlas",
        action="store_true",
        help="Skip the SWE Atlas mapping prompt.",
    )
    parser.add_argument(
        "--skip-evals-report",
        action="store_true",
        help="Skip the evals.report mapping prompt.",
    )
    parser.add_argument(
        "--skip-swe-marathon",
        action="store_true",
        help="Skip the SWE-Marathon mapping prompt.",
    )
    parser.add_argument(
        "--skip-spheron",
        action="store_true",
        help="Skip the Spheron mapping prompt.",
    )
    parser.add_argument(
        "--skip-llmstats",
        action="store_true",
        help="Skip the llm-stats.com model and benchmark-label mapping prompts.",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)
    return args


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


def build_name_options(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Model-slug autocomplete pool: AA names plus llm-stats-only names.

    AA names are returned untagged (display unchanged); llm-stats names not already
    covered by AA are appended and tagged so the source shows inline in the selector.
    AA wins on overlap so shared names keep their AA field auto-fill.
    """
    aa_names = [] if args.skip_aa else fetch_aa_model_names()

    llmstats_names: list[str] = []
    if not args.skip_llmstats:
        try:
            llmstats_names = fetch_llmstats_model_names()
        except RuntimeError as exc:
            print(f"Skipping llm-stats name autocomplete: {exc}")

    aa_set = set(aa_names)
    extra = sorted(name for name in llmstats_names if name not in aa_set)
    options = aa_names + extra
    sources = {name: "llm-stats" for name in extra}
    return options, sources


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
    for source_key, dest_key in (("url", "url"), ("params", "params"), ("context", "context")):
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
        deepswe_names = fetch_all_deepswe_names()
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


def maybe_add_frontiercode_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_frontiercode_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        frontiercode_names = fetch_frontiercode_model_names()
    except RuntimeError as exc:
        print(f"Skipping FrontierCode mapping prompt: {exc}")
        return

    if not frontiercode_names:
        return

    frontiercode_name = prompt_select_or_new("FrontierCode model", frontiercode_names)
    if not frontiercode_name:
        return

    add_frontiercode_mapping(frontiercode_name, model_name)
    print(f"Added FrontierCode mapping '{frontiercode_name}' -> '{model_name}'")


def maybe_add_swe_atlas_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_swe_atlas_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        swe_atlas_names = fetch_swe_atlas_model_names()
    except RuntimeError as exc:
        print(f"Skipping SWE Atlas mapping prompt: {exc}")
        return

    if not swe_atlas_names:
        return

    swe_atlas_name = prompt_select_or_new("SWE Atlas model", swe_atlas_names)
    if not swe_atlas_name:
        return

    add_swe_atlas_mapping(swe_atlas_name, model_name)
    print(f"Added SWE Atlas mapping '{swe_atlas_name}' -> '{model_name}'")


def maybe_add_evals_report_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_evals_report_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        evals_report_names = fetch_evals_report_model_names()
    except RuntimeError as exc:
        print(f"Skipping evals.report mapping prompt: {exc}")
        return

    if not evals_report_names:
        return

    evals_report_name = prompt_select_or_new("evals.report model", evals_report_names)
    if not evals_report_name:
        return

    add_evals_report_mapping(evals_report_name, model_name)
    print(f"Added evals.report mapping '{evals_report_name}' -> '{model_name}'")


def maybe_add_swe_marathon_mapping(model_name: str, interactive: bool) -> None:
    if not interactive:
        return

    existing_mapping = load_swe_marathon_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    try:
        swe_marathon_names = fetch_swe_marathon_model_names()
    except RuntimeError as exc:
        print(f"Skipping SWE-Marathon mapping prompt: {exc}")
        return

    if not swe_marathon_names:
        return

    swe_marathon_name = prompt_select_or_new("SWE-Marathon model", swe_marathon_names)
    if not swe_marathon_name:
        return

    add_swe_marathon_mapping(swe_marathon_name, model_name)
    print(f"Added SWE-Marathon mapping '{swe_marathon_name}' -> '{model_name}'")


def maybe_add_spheron_mapping(model: dict[str, Any], interactive: bool) -> None:
    if not interactive:
        return

    model_name = model.get("name")
    if not isinstance(model_name, str) or not model_name:
        return

    existing_mapping = load_spheron_to_slug_mapping()
    if model_name in existing_mapping.values():
        return

    default_path = hf_path_from_url(model.get("url"))
    spheron_path = prompt_select_or_new(
        "Spheron model path (org/model)",
        [default_path] if default_path else [],
        default=default_path,
    )
    if not spheron_path:
        return

    add_spheron_mapping(spheron_path, model_name)
    print(f"Added Spheron mapping '{spheron_path}' -> '{model_name}'")


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

    benchmark_keys = sorted(editable_benchmarks(doc).keys())

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
            f"Map {source_label} '{label}' to llm.json benchmark",
            keys,
            default=default,
            subject=label,
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

    benchmark_keys = sorted(editable_benchmarks(doc).keys())

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


def prompt_live_select_or_new(
    label: str,
    options: list[str],
    allow_empty: bool = True,
    default: str | None = None,
    sources: dict[str, str] | None = None,
) -> str | None:
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    buffer = default or ""
    lines_drawn = 0

    try:
        tty.setraw(fd)
        while True:
            matches = find_matches(buffer, options)
            lines_drawn = render_selector(label, buffer, matches, sources)
            char = sys.stdin.read(1)

            if char in {"\r", "\n"}:
                value = parse_nullable(buffer)
                clear_selector(lines_drawn)
                if value is None:
                    if allow_empty:
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        return None
                    print(f"{label} is required.")
                    buffer = ""
                    lines_drawn = 0
                    continue
                sys.stdout.write(f"{label}: {buffer}\r\n")
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
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

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


def subject_from_label(label: str) -> str:
    """The source name a prompt label is about, e.g. Map X 'foo' to Y -> foo.

    Last-resort guess only: pass subject= to prompt_select_or_new instead. A
    source name may contain an apostrophe ("Agents' Last Exam"), and a name
    recovered from the label is what propose.py writes into the mapping file, so
    reading one character too few records a decision under a name no source ever
    reports -- the real name stays unreviewed and is queued again every run.
    The quantifier is greedy so the outermost pair of quotes wins, which keeps
    the whole name for the labels this repo builds ("Map X '<name>' to Y").
    """
    quoted = re.search(r"'(.+)'", label)
    return quoted.group(1) if quoted else label


def prompt_select_or_new(
    label: str,
    options: list[str],
    allow_empty: bool = True,
    default: str | None = None,
    sources: dict[str, str] | None = None,
    subject: str | None = None,
) -> str | None:
    # Collect mode: queue the question and answer "no answer". Callers treat None
    # as "not a match", and their persist calls are frozen, so it is asked again.
    if _prompts.collecting():
        # The caller knows the exact source name; only fall back to digging it
        # back out of the label when it did not say.
        queued = subject if subject is not None else subject_from_label(label)
        candidates = ([default] if default else []) + find_matches(queued, options, limit=5)
        _prompts.record(
            kind="mapping",
            subject=queued,
            question=label,
            candidates=list(dict.fromkeys(candidates)),
            default=default,
        )
        return None

    if not options:
        if default is not None:
            return prompt_value_with_default(label, default)
        return prompt_value(label)

    if supports_live_selector():
        return prompt_live_select_or_new(
            label, options, allow_empty=allow_empty, default=default, sources=sources
        )

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
            source = sources.get(match) if sources else None
            print(f"  {idx}. {match}  ({source})" if source else f"  {idx}. {match}")

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
        options, sources = build_name_options(args)
        name = prompt_select_or_new("Name", options, allow_empty=False, sources=sources)
    else:
        name = get_value(args.name, "Name", interactive)
    if not name:
        raise ValueError("Model name is required.")

    aa_defaults = {} if args.skip_aa else fetch_aa_model_defaults(name)

    url = get_value(args.url, "URL", interactive, aa_defaults.get("url"))

    # AA's model page carries total + active counts; Hugging Face knows only the
    # total, so it fills in when AA has no page for the model.
    params_default = normalize_params(aa_defaults.get("params")) or fetch_hf_params(url)
    if interactive and args.params is None:
        if not params_default and url:
            print(f"  params not auto-filled; see {url}")
        params = prompt_select_or_new(
            "Params",
            get_unique_values(models, "params"),
            default=params_default,
        )
    else:
        params = get_value(args.params, "Params", interactive, params_default)

    if interactive and args.context is None:
        if not aa_defaults.get("context") and url:
            print(f"  context not auto-filled; see {url}")
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
        "vram": {"fp16": None, "int8": None, "int4": None},
        "creator": {
            "name": creator_name,
            "url": creator_url,
        },
        "scores": {key: None for key in benchmark_keys},
        # scores_updated mirrors the scores key set; a key's date is stamped
        # when update.py/edit.py write that score. Consumers read by key and
        # tolerate absent keys (llm.html uses optional chaining).
        "scores_updated": {key: None for key in benchmark_keys},
        # scores_source mirrors the same key set; update.py stamps the page a
        # score was read from, hand edits and derived keys stay null.
        "scores_source": {key: None for key in benchmark_keys},
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

    # Adding a model needs the name/url/params/context answered, so there is
    # nothing useful to do unattended. Queue it and leave llm.json alone.
    if _prompts.collecting():
        name = parse_nullable(args.name) or "(unnamed)"
        _prompts.record(
            kind="new-model",
            subject=name,
            question=f"Add model '{name}' to llm.json (needs metadata and mappings)",
            command="./add.py --name " + name,
        )
        print(f"collect mode: queued '{name}' for a manual ./add.py run")
        return 0

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
    if not args.skip_frontiercode:
        maybe_add_frontiercode_mapping(model["name"], interactive)
    if not args.skip_swe_atlas:
        maybe_add_swe_atlas_mapping(model["name"], interactive)
    if not args.skip_evals_report:
        maybe_add_evals_report_mapping(model["name"], interactive)
    if not args.skip_swe_marathon:
        maybe_add_swe_marathon_mapping(model["name"], interactive)
    if not args.skip_spheron:
        maybe_add_spheron_mapping(model, interactive)
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
