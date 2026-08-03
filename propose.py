#!/usr/bin/env python3
"""Turn a collected prompt queue into a reviewable proposal.

`update-all --collect-prompts` leaves a queue of questions nobody answered. This
turns that queue into changes a human can review as a diff:

  mapping names  -> a line in the source's mapping JSON. A single normalized-
                    equality match is written as the real slug; anything less
                    confident is written as __pending__, which the reviewed-set
                    loaders ignore, so merging it unanswered costs nothing and
                    the name is queued again next run.
  new models     -> a full llm.json entry via add.py, prefilled from Artificial
                    Analysis. Scores stay null; the next update.py run fills them.

Only normalized equality is ever written as a real mapping -- see _matching.py
for why substring and subset matches are not safe to commit unreviewed.

The emitted plan drives the workflow's review comments, which carry the
alternatives as clickable GitHub suggestions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import _matching
import _prompts
from _openness import PENDING

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
ADD_SCRIPT = HERE / "add.py"

MODELS = "models"
BENCHMARKS = "benchmarks"
AA_SLUGS = "aa-slugs"


@dataclass(frozen=True)
class Route:
    """Where a queued question's answer belongs."""

    module: str
    mapping_const: str
    writer: str
    universe: str


# Keyed by the script that asks, then by prompt kind for the scripts that ask
# about more than one file. test_propose.py asserts this covers every
# update_*_mapping.py in the tree, so a new source cannot be forgotten.
ROUTES: dict[str, dict[str, Route]] = {
    "update_rebench_mapping.py": {
        "*": Route("_swe_rebench_mapping", "SWE_REBENCH_MAPPING", "add_rebench_mapping", MODELS)
    },
    "update_osworld_mapping.py": {
        "*": Route("_osworld_mapping", "OSWORLD_MAPPING", "add_osworld_mapping", MODELS)
    },
    "update_deepswe_mapping.py": {
        "*": Route("_deepswe_mapping", "DEEPSWE_MAPPING", "add_deepswe_mapping", MODELS)
    },
    "update_frontierswe_mapping.py": {
        "*": Route("_frontierswe_mapping", "FRONTIERSWE_MAPPING", "add_frontierswe_mapping", MODELS)
    },
    "update_swe_atlas_mapping.py": {
        "*": Route("_swe_atlas_mapping", "SWE_ATLAS_MAPPING", "add_swe_atlas_mapping", MODELS)
    },
    "update_evals_report_mapping.py": {
        "*": Route(
            "_evals_report_mapping", "EVALS_REPORT_MAPPING", "add_evals_report_mapping", MODELS
        )
    },
    "update_swe_marathon_mapping.py": {
        "*": Route(
            "_swe_marathon_mapping", "SWE_MARATHON_MAPPING", "add_swe_marathon_mapping", MODELS
        )
    },
    "update_toolathlon_mapping.py": {
        "*": Route("_toolathlon_mapping", "TOOLATHLON_MAPPING", "add_toolathlon_mapping", MODELS)
    },
    "update_spheron_mapping.py": {
        "*": Route("_spheron_mapping", "SPHERON_MAPPING", "add_spheron_mapping", MODELS)
    },
    "update_huggingface_mapping.py": {
        "*": Route("_huggingface_mapping", "HF_MAPPING", "add_hf_mapping", BENCHMARKS)
    },
    "update_llmstats_mapping.py": {
        "llmstats-model": Route(
            "_llmstats_mapping", "LLMSTATS_MODEL_MAPPING", "add_llmstats_mapping", MODELS
        ),
        "llmstats-benchmark": Route(
            "_llmstats_mapping",
            "LLMSTATS_BENCHMARK_MAPPING",
            "add_llmstats_benchmark_mapping",
            BENCHMARKS,
        ),
    },
    "update_artificialanalysis_mapping.py": {
        "*": Route(
            "_artificialanalysis_mapping", "AA_MODEL_MAPPING", "add_aa_mapping", AA_SLUGS
        )
    },
}


@dataclass
class Proposal:
    path: Path
    key: str
    value: str
    confidence: str
    reason: str
    alternatives: list[str] = field(default_factory=list)
    line: int | None = None

    @property
    def pending(self) -> bool:
        return self.value == PENDING


def script_of(entry: dict[str, Any]) -> str:
    command = entry.get("command") or ""
    first = command.split()[0] if command.split() else ""
    return Path(first).name


def route_for(entry: dict[str, Any]) -> Route | None:
    by_kind = ROUTES.get(script_of(entry))
    if by_kind is None:
        return None
    return by_kind.get(entry.get("kind", ""), by_kind.get("*"))


def match_subject(entry: dict[str, Any], route: Route) -> str:
    """What to match on. Spheron keys are org/model paths; match the model part."""
    subject = entry.get("subject") or ""
    if route.module == "_spheron_mapping" and "/" in subject:
        return subject.rsplit("/", 1)[-1]
    return subject


def build_universes(llm_path: Path, skip_aa: bool) -> dict[str, list[str]]:
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    universes = {
        MODELS: [m["name"] for m in doc.get("models", []) if isinstance(m, dict) and m.get("name")],
        BENCHMARKS: sorted(doc.get("benchmarks", {})),
        AA_SLUGS: [],
    }
    if not skip_aa:
        try:
            from _artificialanalysis_mapping import fetch_aa_model_names

            universes[AA_SLUGS] = fetch_aa_model_names()
        except Exception as exc:  # noqa: BLE001 - a dead source must not sink the run
            print(f"warning: no Artificial Analysis slugs ({exc})", file=sys.stderr)
    return universes


def plan_mappings(
    entries: list[dict[str, Any]], universes: dict[str, list[str]]
) -> tuple[list[Proposal], list[dict[str, Any]]]:
    proposals: list[Proposal] = []
    unroutable: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("kind") == "new-model":
            continue
        route = route_for(entry)
        if route is None:
            unroutable.append(entry)
            continue

        module = importlib.import_module(route.module)
        path = getattr(module, route.mapping_const)
        options = universes.get(route.universe) or []
        # The AA prompt's own candidate generator beats a cold universe, and the
        # recorded list is all we have when the API is unreachable.
        if not options and entry.get("candidates"):
            options = list(entry["candidates"])

        match, alternatives = _matching.propose(match_subject(entry, route), options)
        proposals.append(
            Proposal(
                path=path,
                key=entry.get("subject") or "",
                value=match.option if match else PENDING,
                confidence=match.confidence if match else "none",
                reason=match.reason if match else "no confident match",
                alternatives=[a.option for a in alternatives[:4]],
            )
        )
    return proposals, unroutable


def writers_by_path() -> dict[Path, Any]:
    """Mapping file -> the add_*_mapping that owns it."""
    out: dict[Path, Any] = {}
    for by_kind in ROUTES.values():
        for route in by_kind.values():
            module = importlib.import_module(route.module)
            out[getattr(module, route.mapping_const)] = getattr(module, route.writer)
    return out


def apply_mappings(proposals: list[Proposal]) -> None:
    """Write every proposal through the source's own writer, then locate its line."""
    writers = writers_by_path()
    for proposal in proposals:
        writers[proposal.path](proposal.key, proposal.value)
    for proposal in proposals:
        proposal.line = find_line(proposal.path, proposal.key)


def find_line(path: Path, key: str) -> int | None:
    needle = json.dumps(key, ensure_ascii=False) + ":"
    for index, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text.strip().startswith(needle):
            return index
    return None


def suggestion_body(proposal: Proposal, line_text: str) -> str:
    """A GitHub review comment whose ```suggestion block is one click to apply."""
    top = proposal.alternatives[0]
    replacement = line_text.replace(json.dumps(PENDING), json.dumps(top))
    rest = proposal.alternatives[1:]

    out = [
        "No confident match, so this is parked as `__pending__`.",
        "",
        "Best guess -- commit this suggestion to accept it:",
        "",
        "```suggestion",
        replacement,
        "```",
        "",
    ]
    if rest:
        out.append("Other candidates: " + ", ".join(f"`{c}`" for c in rest))
    out.append(
        "Leave `__pending__` and it is queued again next run; "
        "set `__unmappable__` to stop being asked."
    )
    return "\n".join(out)


def plan_suggestions(proposals: list[Proposal], limit: int) -> tuple[list[dict], int]:
    """Review comments for the pending lines, most useful first."""
    candidates = [p for p in proposals if p.pending and p.alternatives and p.line]
    candidates.sort(key=lambda p: (-len(p.alternatives), p.key))

    comments = []
    for proposal in candidates[:limit]:
        lines = proposal.path.read_text(encoding="utf-8").splitlines()
        comments.append(
            {
                "path": proposal.path.name,
                "line": proposal.line,
                "side": "RIGHT",
                "body": suggestion_body(proposal, lines[proposal.line - 1]),
            }
        )
    return comments, max(0, len(candidates) - limit)


def add_models(entries: list[dict[str, Any]], llm_path: Path, limit: int) -> tuple[list[dict], int]:
    """Append a full llm.json entry per queued new model, via add.py."""
    wanted, seen = [], set()
    for entry in entries:
        if entry.get("kind") != "new-model":
            continue
        name = entry.get("subject") or ""
        if not name or name == "(unnamed)" or name in seen:
            continue
        seen.add(name)
        wanted.append(entry)

    added = []
    for entry in wanted[:limit]:
        name = entry["subject"]
        proc = subprocess.run(
            [sys.executable, str(ADD_SCRIPT), "--name", name, str(llm_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"warning: add.py failed for {name}: {proc.stderr.strip()}", file=sys.stderr)
            continue
        added.append({"name": name, "note": entry.get("note")})
    return added, max(0, len(wanted) - limit)


def render_body(plan: dict[str, Any]) -> str:
    counts = plan["counts"]
    dropped = plan["dropped"]
    mappings = plan["mappings"]
    confident = [m for m in mappings if m["value"] != PENDING]
    parked = [m for m in mappings if m["value"] == PENDING]

    out = [
        "Proposals from the daily benchmark refresh. Scores are already on `main`; "
        "this is only the part that needed a person.",
        "",
    ]

    if confident:
        out += [
            f"## Confident mappings &mdash; {len(confident)}",
            "",
            "The source name and the llm.json name are the same string once normalized, "
            "so these are safe to merge as-is.",
            "",
            "| file | source name | maps to |",
            "| --- | --- | --- |",
        ]
        for m in sorted(confident, key=lambda m: (m["file"], m["key"])):
            out.append(f"| `{m['file']}` | `{m['key']}` | `{m['value']}` |")
        out.append("")

    if parked:
        with_suggestion = len(plan["comments"])
        out += [
            f"## Parked as `__pending__` &mdash; {len(parked)}",
            "",
            "No confident match. `__pending__` is **not** a decision: the reviewed-set "
            "loaders skip it, so merging one unanswered records nothing and the name is "
            f"queued again next run. {with_suggestion} of these carry a clickable "
            "suggestion in the review comments below.",
            "",
            "| file | source name | candidates |",
            "| --- | --- | --- |",
        ]
        for m in sorted(parked, key=lambda m: (m["file"], m["key"])):
            alts = ", ".join(f"`{c}`" for c in m["alternatives"]) or "_none_"
            out.append(f"| `{m['file']}` | `{m['key']}` | {alts} |")
        out.append("")

    if plan["models"]:
        out += [
            f"## New models &mdash; {len(plan['models'])}",
            "",
            "Added to `llm.json` with metadata prefilled from Artificial Analysis and "
            "null scores; the next refresh fills the scores in. Check params and context.",
            "",
        ]
        for model in plan["models"]:
            note = f" &mdash; {model['note']}" if model.get("note") else ""
            out.append(f"- `{model['name']}`{note}")
        out.append("")

    if dropped["suggestions"] or dropped["models"] or plan["unroutable"]:
        out += ["## Left for a later run", ""]
        if dropped["suggestions"]:
            out.append(
                f"- {dropped['suggestions']} parked line(s) got no review comment "
                "(suggestion cap reached); their candidates are in the table above."
            )
        if dropped["models"]:
            out.append(f"- {dropped['models']} new model(s) not added yet (per-PR cap).")
        for entry in plan["unroutable"]:
            out.append(f"- unroutable: `{entry['command']}` &rarr; `{entry['subject']}`")
        out.append("")

    out.append(f"<sub>{counts['total']} queued question(s) in total.</sub>")
    return "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", help="Collected queue (default: $%s)" % _prompts.ENV_VAR)
    parser.add_argument("--llm-json", default=str(DEFAULT_LLM_JSON))
    parser.add_argument(
        "--apply", action="store_true", help="Write the changes (default is a dry run)."
    )
    parser.add_argument("--plan", help="Write the plan JSON here (for the workflow).")
    parser.add_argument("--body", help="Write the PR body markdown here.")
    parser.add_argument(
        "--max-suggestions",
        type=int,
        default=10,
        help="Cap review comments so a big queue does not bury the PR (default: 10).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=5,
        help="Cap new llm.json entries per PR; each needs a real look (default: 5).",
    )
    parser.add_argument("--skip-aa", action="store_true", help="Do not fetch AA slugs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # The writers are no-ops under collect mode, so applying would silently do
    # nothing. Proposing is the opposite job: it is meant to write.
    if _prompts.collecting():
        print(
            f"error: {_prompts.ENV_VAR} is set; unset it before proposing", file=sys.stderr
        )
        return 2

    llm_path = Path(args.llm_json)
    entries = _prompts.load(args.report)
    if not entries:
        print("nothing queued, nothing to propose")
        if args.plan:
            Path(args.plan).write_text(json.dumps({"counts": {"total": 0}}) + "\n", encoding="utf-8")
        return 0

    universes = build_universes(llm_path, args.skip_aa)
    proposals, unroutable = plan_mappings(entries, universes)

    exact = [p for p in proposals if not p.pending]
    pending = [p for p in proposals if p.pending]

    print(f"queued questions: {len(entries)}")
    print(f"mapping proposals: {len(proposals)} ({len(exact)} confident, {len(pending)} parked)")
    for proposal in exact:
        print(f"  {proposal.path.name}: {proposal.key!r} -> {proposal.value!r} ({proposal.reason})")
    for proposal in pending:
        alts = ", ".join(proposal.alternatives) or "no candidates"
        print(f"  {proposal.path.name}: {proposal.key!r} -> __pending__  [{alts}]")
    for entry in unroutable:
        print(f"  unroutable: {entry.get('command')} {entry.get('subject')!r}", file=sys.stderr)

    if not args.apply:
        print("\ndry run, pass --apply to write")
        return 0

    apply_mappings(proposals)
    comments, dropped_suggestions = plan_suggestions(proposals, args.max_suggestions)
    models, dropped_models = add_models(entries, llm_path, args.max_models)

    if dropped_suggestions:
        print(f"note: {dropped_suggestions} pending line(s) got no review comment (cap reached)")
    if dropped_models:
        print(f"note: {dropped_models} new model(s) left for a later run (cap reached)")

    plan = {
        "counts": {
            "total": len(entries),
            "confident": len(exact),
            "pending": len(pending),
            "models": len(models),
            "unroutable": len(unroutable),
        },
        "dropped": {"suggestions": dropped_suggestions, "models": dropped_models},
        "mappings": [
            {
                "file": p.path.name,
                "key": p.key,
                "value": p.value,
                "confidence": p.confidence,
                "reason": p.reason,
                "alternatives": p.alternatives,
                "line": p.line,
            }
            for p in proposals
        ],
        "models": models,
        "unroutable": [{"command": e.get("command"), "subject": e.get("subject")} for e in unroutable],
        "comments": comments,
    }
    if args.plan:
        Path(args.plan).write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.body:
        Path(args.body).write_text(render_body(plan), encoding="utf-8")
    print(f"\napplied: {len(exact)} confident, {len(pending)} parked, {len(models)} new model(s)")
    print(f"review comments planned: {len(comments)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
