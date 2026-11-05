#!/usr/bin/env python3
"""Preview models that have since been released: find them, and clean them up.

A lab often publishes a preview first ("Hy4 preview") and the release weeks
later ("Hy4"). The index tracks the preview under a slug that says so --
`hy4-preview` -- and when the release arrives, the boards that list it are
mapped onto that same entry rather than onto a new one: it is the same model
line, and its preview scores are the best numbers there are until the release
has been measured everywhere.

That leaves a mess behind which nothing resolves on its own, because whether a
release's number replaces a preview's is decided by the source's rung in
_precedence.py, not by which of the two is newer:

  * the name still says `preview`, and so do the model card URL and every
    preview-only page a score is credited to;
  * a **fill-only** source (Hugging Face cards, llm-stats) never replaces a
    number credited to a different page of its own, so the preview card's
    SWE-bench Pro keeps beating the release card's for good;
  * a board that maps both its preview row and its release row onto the entry
    hands `keep_best_row` two rows to choose between, so the stored number may
    be either one;
  * a release name the boards list later is asked about on the Queue, with the
    preview entry as a candidate.

Done by hand, that was a prompt and a careful diff every time. This module is
the two halves of doing it from the admin page:

  ``detect()``  reads llm.json, every mapping file, the queue and AA's model
                list, and lists each preview entry with everything a person
                needs to decide: the evidence it has been released, a rename,
                the new URL, what to do with each score and which mapping to
                park. Pure -- it only reads -- and published by the workflow to
                ``_pending/previews.json`` for the page.
  ``apply()``   carries one ``preview-cleanup`` record out (see _answers.py for
                the validation), atomically with the rest of the batch.

A case that is not a release after all -- a preview whose name simply lacks a
sibling yet, or one already cleaned up by hand -- is *checked off*: it goes in
``preview-cleanup-done.json`` and is not listed again. A cleanup that was
carried out is recorded there too, under the name it ended up with, so a
cleanup that kept the name is not offered a second time.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import _precedence
import propose
from _artificialanalysis_mapping import AA_MODEL_MAPPING, load_llm_to_aa_slugs
from fill_source_urls import canonical

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
DEFAULT_QUEUE = HERE / "_pending" / "pending.json"
DEFAULT_AA_MODELS = HERE / "_aa" / "models.json"
DEFAULT_REPORT = HERE / "_pending" / "previews.json"
DONE_FILE = HERE / "preview-cleanup-done.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# "preview" as a word of its own: "-preview", " Preview", "(Preview)",
# "Hy4-preview", "preview-" -- not inside a longer word. A source that cuts
# names short counts too: Spheron lists "internlm/Atria-Dawn-Previe", which is
# the preview, not evidence of a release.
PREVIEW_RE = re.compile(r"(?<![a-z])previ(?:ew?)?(?![a-z])", re.IGNORECASE)
# The same word with the separator in front of it, which is what has to go when
# it is taken out of a URL path: "Hy4-preview" -> "Hy4".
_URL_PREVIEW_RE = re.compile(r"[-_.]preview(?=$|[-_./?#])|(?<=/)preview[-_.]", re.IGNORECASE)

# What each score decision may be. Keep is the absence of one.
CLEAR = "clear"
RECREDIT = "recredit"
SCORE_ACTIONS = (CLEAR, RECREDIT)
KEEP = "keep"
PARK = "park"

# What a score was read off, as far as can be told.
ERA_PREVIEW = "preview"
ERA_RELEASE = "release"
ERA_EITHER = "either"
ERA_UNKNOWN = "unknown"

# The two ingests that only fill a cell (update.apply_score's fill_only): a
# number credited to another of their own pages is never replaced by them,
# which is exactly the preview card's number outliving the release card's.
FILL_ONLY_RANKS = frozenset({_precedence.RANK_AGGREGATE, _precedence.RANK_MODEL_CARD})

RANK_LABELS = {
    _precedence.RANK_AA: "Artificial Analysis",
    _precedence.RANK_AA_CODING_AGENTS: "AA Coding Agent Index",
    _precedence.RANK_BENCHMARK_SITE: "the benchmark's own board",
    _precedence.RANK_THIRD_PARTY_RUN: "a third-party run",
    _precedence.RANK_CURATED: "a curated compilation",
    _precedence.RANK_AGGREGATE: "llm-stats (fill-only)",
    _precedence.RANK_ENDPOINT_RUN: "OpenRouter endpoint runs",
    _precedence.RANK_BENCHMARKING_HUB: "Epoch AI hub",
    _precedence.RANK_MODEL_CARD: "a model card (fill-only)",
    _precedence.RANK_HAND_ENTERED: "entered by hand",
}


class CleanupError(Exception):
    """The cleanup must not go ahead. The message is shown to the sender."""


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------


def is_preview(text: Any) -> bool:
    return isinstance(text, str) and bool(PREVIEW_RE.search(text))


def strip_slug(name: str) -> str:
    """The slug without its `preview` word: hy4-preview -> hy4."""
    parts = [p for p in re.split(r"-", name) if p.lower() != "preview"]
    return "-".join(p for p in parts if p)


def strip_url(url: str) -> str:
    """The URL without its `preview` word: .../tencent/Hy4-preview -> .../tencent/Hy4."""
    return _URL_PREVIEW_RE.sub("", url)


# --------------------------------------------------------------------------
# which source wrote a score
# --------------------------------------------------------------------------


def _route_prefixes() -> dict[str, tuple[str, ...]]:
    """Mapping route -> the score pages its source credits.

    Read off _precedence's constants, which are themselves read off each
    fetcher's own, so a board that moves host moves here too. A route with no
    pages of its own (Spheron writes VRAM, not scores) is absent.
    """
    p = _precedence
    llmstats_models = canonical("https://llm-stats.com/models")
    table: dict[str, Iterable[str]] = {
        "update_rebench_mapping.py": ["https://swe-rebench.com"],
        "update_osworld_mapping.py": [p.OSWORLD_SOURCE_URL, p.OSWORLD_V2_SOURCE_URL],
        "update_deepswe_mapping.py": [p.DEEPSWE_SOURCE_URL, p.DATACURVE_SOURCE_URL],
        "update_frontierswe_mapping.py": [p.FRONTIERSWE_SOURCE_URL],
        "update_frontiercode_mapping.py": [p.FRONTIERCODE_SOURCE_URL],
        "update_swe_atlas_mapping.py": p.SWE_ATLAS_KEY_URLS.values(),
        "update_tbench_mapping.py": p.TBENCH_KEY_URLS.values(),
        "update_agents_last_exam_mapping.py": [p.AGENTS_LAST_EXAM_SOURCE_URL],
        "update_aa_coding_agents_mapping.py": [p.AA_CODING_AGENTS_SOURCE_URL],
        "update_evals_report_mapping.py": p.EVALS_REPORT_KEY_URLS.values(),
        "update_vals_mapping.py": p.VALS_KEY_URLS.values(),
        "update_swe_marathon_mapping.py": [p.SWE_MARATHON_SOURCE_URL],
        "update_toolathlon_mapping.py": [p.TOOLATHLON_SOURCE_URL],
        "update_programbench_mapping.py": [p.PROGRAMBENCH_SOURCE_URL],
        "update_kernelbench_mapping.py": [p.KERNELBENCH_SOURCE_URL],
        "update_real_swe_mapping.py": [p.REAL_SWE_SOURCE_URL],
        "update_zerobench_mapping.py": [p.ZEROBENCH_SOURCE_URL],
        "update_arc_prize_mapping.py": p.ARC_PRIZE_KEY_URLS.values(),
        "update_arena_mapping.py": p.ARENA_KEY_URLS.values(),
        "update_cybergym_mapping.py": p.CYBERGYM_KEY_URLS.values(),
        "update_sec_bench_mapping.py": p.SEC_BENCH_KEY_URLS.values(),
        "update_hle_diamond_mapping.py": [p.HLE_DIAMOND_SOURCE_URL],
        "update_openrouter_mapping.py": [p.OPENROUTER_PREFIX],
        "update_epoch_mapping.py": p.EPOCH_PAGE_URLS.values(),
        "update_mcp_atlas_mapping.py": [p.MCP_ATLAS_SOURCE_URL],
        "update_swe_bench_pro_v2_mapping.py": [p.SWE_BENCH_PRO_V2_SOURCE_URL],
        "update_bfcl_mapping.py": [p.BFCL_SOURCE_URL],
        "update_llmstats_mapping.py": [p.LLMSTATS_SOURCE_URL, llmstats_models],
        "update_artificialanalysis_mapping.py": ["https://artificialanalysis.ai"],
    }
    return {route: tuple(sorted({canonical(u) for u in urls})) for route, urls in table.items()}


ROUTE_PREFIXES = _route_prefixes()


def route_of_source(url: str | None) -> str | None:
    """The mapping route whose source credits `url`, longest page first."""
    if not url:
        return None
    candidate = canonical(url)
    best: tuple[int, str] | None = None
    for route, prefixes in ROUTE_PREFIXES.items():
        for prefix in prefixes:
            if candidate == prefix or candidate.startswith(f"{prefix}/"):
                if best is None or len(prefix) > best[0]:
                    best = (len(prefix), route)
    return best[1] if best else None


# --------------------------------------------------------------------------
# reading the repository
# --------------------------------------------------------------------------


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def model_routes() -> list[tuple[str, str, propose.Route]]:
    """(route, kind, Route) for every mapping file whose values are model names."""
    found = []
    for name, by_kind in sorted(propose.ROUTES.items()):
        for kind, route in sorted(by_kind.items()):
            if route.universe == propose.MODELS:
                found.append((name, kind, route))
    return found


def route_path(route: propose.Route) -> Path:
    return getattr(importlib.import_module(route.module), route.mapping_const)


def load_mappings() -> dict[tuple[str, str], dict[str, Any]]:
    """(route, kind) -> that mapping file's raw contents."""
    return {
        (name, kind): (_read_json(route_path(route), {}) or {})
        for name, kind, route in model_routes()
    }


def load_done(path: Path | None = None) -> dict[str, dict[str, Any]]:
    raw = _read_json(path or DONE_FILE, {})
    if not isinstance(raw, dict):
        return {}
    return {k: v if isinstance(v, dict) else {} for k, v in raw.items() if isinstance(k, str)}


def write_done(done: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    (path or DONE_FILE).write_text(
        json.dumps({k: done[k] for k in sorted(done)}, **JSON_DUMP_KWARGS) + "\n",
        encoding="utf-8",
    )


def _points_at(value: Any, name: str) -> bool:
    if isinstance(value, str):
        return value == name
    if isinstance(value, list):
        return name in value
    return False


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass
class Inputs:
    """Everything detect() reads, so a test can hand it in."""

    doc: dict[str, Any]
    mappings: dict[tuple[str, str], dict[str, Any]]
    llm_to_aa: dict[str, list[str]]
    aa_models: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    done: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_repo(
        cls,
        llm_path: Path = DEFAULT_LLM_JSON,
        queue_path: Path = DEFAULT_QUEUE,
        aa_path: Path = DEFAULT_AA_MODELS,
        done_path: Path | None = None,
    ) -> "Inputs":
        aa = _read_json(aa_path, {})
        queue = _read_json(queue_path, {})
        return cls(
            doc=json.loads(Path(llm_path).read_text(encoding="utf-8")),
            mappings=load_mappings(),
            llm_to_aa=load_llm_to_aa_slugs(AA_MODEL_MAPPING),
            aa_models=[m for m in (aa.get("models") if isinstance(aa, dict) else aa) or []
                       if isinstance(m, dict)],
            questions=[q for q in (queue.get("questions") if isinstance(queue, dict) else []) or []
                       if isinstance(q, dict)],
            done=load_done(done_path),
        )


def _derived(doc: dict[str, Any]) -> set[str]:
    return {
        key for key, bench in (doc.get("benchmarks") or {}).items()
        if isinstance(bench, dict) and bench.get("derived")
    }


def _aa_slugs_of(model: dict[str, Any], llm_to_aa: dict[str, list[str]]) -> list[str]:
    name = model.get("name") or ""
    slugs = list(llm_to_aa.get(name) or [])
    variant = (model.get("aa_variant") or {}).get("slug")
    if isinstance(variant, str) and variant and variant not in slugs:
        slugs.append(variant)
    if not slugs:
        slugs = [name]
    return slugs


def _release(model: dict[str, Any], inputs: Inputs) -> tuple[str | None, list[dict[str, str]]]:
    """(release date, evidence) -- that a preview entry now stands for a release."""
    name = model["name"]
    evidence: list[dict[str, str]] = []
    released: str | None = None

    by_slug = {m.get("slug"): m for m in inputs.aa_models}
    for slug in _aa_slugs_of(model, inputs.llm_to_aa):
        aa = by_slug.get(slug)
        if not aa:
            continue
        if is_preview(aa.get("name")):
            continue
        when = aa.get("release_date") if isinstance(aa.get("release_date"), str) else None
        released = max(filter(None, [released, when]), default=None)
        evidence.append({
            "kind": "aa",
            "text": f"Artificial Analysis lists {slug} as {aa.get('name')!r}"
                    + (f", released {when}" if when else "")
                    + " -- no 'preview' in the name",
        })

    for (route, _kind), raw in sorted(inputs.mappings.items()):
        if not isinstance(raw, dict):
            continue
        names = sorted(k for k, v in raw.items() if _points_at(v, name) and not is_preview(k))
        if names:
            evidence.append({
                "kind": "mapping",
                "text": f"{_source_label(route)} lists it as "
                        + ", ".join(repr(n) for n in names[:3])
                        + (f" and {len(names) - 3} more" if len(names) > 3 else ""),
            })

    for question in inputs.questions:
        if question.get("route") == "check_new.py":
            continue
        options = [c.get("option") for c in question.get("candidates") or [] if isinstance(c, dict)]
        if name in options and not is_preview(question.get("subject")):
            evidence.append({
                "kind": "queue",
                "text": f"{_source_label(question.get('route') or '')} asks about "
                        f"{question.get('subject')!r}, with this entry as a candidate",
            })
    return released, evidence


def _source_label(route: str) -> str:
    return route.replace("update_", "").replace("_mapping.py", "").replace("_", " ") or route


def _score_era(
    url: str | None,
    when: str | None,
    released: str | None,
    route: str | None,
    both: set[str],
) -> tuple[str, str]:
    if is_preview(url):
        return ERA_PREVIEW, "credited to a preview page"
    if released and when and when < released:
        return ERA_PREVIEW, f"read {when}, before the release on {released}"
    if route in both:
        return ERA_EITHER, (
            "this source maps both the preview and the release onto the entry, "
            "so the row it read may be either"
        )
    if released and when:
        return ERA_RELEASE, f"read {when}, on or after the release on {released}"
    return ERA_UNKNOWN, "no release date to compare against"


def _case(model: dict[str, Any], inputs: Inputs, names: set[str]) -> dict[str, Any] | None:
    name = model.get("name")
    url = model.get("url") if isinstance(model.get("url"), str) else None
    if not isinstance(name, str):
        return None
    if not (is_preview(name) or is_preview(url)):
        return None
    if name in inputs.done:
        return None

    released, evidence = _release(model, inputs)

    # The rename. The name without its preview word first, then the AA slug
    # it is mapped to, which also retires the mapping for good.
    renames: list[dict[str, Any]] = []
    if is_preview(name):
        stripped = strip_slug(name)
        if stripped and stripped != name:
            renames.append({"name": stripped, "why": "the name without 'preview'",
                            "taken": stripped in names})
        for slug in _aa_slugs_of(model, inputs.llm_to_aa):
            if slug != name and slug not in (r["name"] for r in renames) and not is_preview(slug):
                renames.append({"name": slug, "why": "the Artificial Analysis slug it is mapped to",
                                "taken": slug in names})

    new_url = strip_url(url) if is_preview(url) else None

    # Mappings: every source name pointing at the entry, by file.
    mappings: list[dict[str, Any]] = []
    both: set[str] = set()
    for (route, kind), raw in sorted(inputs.mappings.items()):
        if not isinstance(raw, dict):
            continue
        keys = sorted(k for k, v in raw.items() if _points_at(v, name))
        if not keys:
            continue
        previews = [k for k in keys if is_preview(k)]
        releases = [k for k in keys if not is_preview(k)]
        if previews and releases:
            both.add(route)
        for key in keys:
            preview = key in previews
            if preview and releases:
                suggest, why = PARK, (
                    "the release is mapped here too ("
                    + ", ".join(repr(r) for r in releases[:2])
                    + "); parking the preview row leaves only the release to read"
                )
            elif preview:
                suggest, why = KEEP, (
                    "the only row this source lists for the model; keep it until "
                    "the release appears there (the Queue will ask)"
                )
            else:
                suggest, why = KEEP, "names the release"
            mappings.append({
                "route": route, "route_kind": kind, "source": _source_label(route),
                "subject": key, "preview": preview, "suggest": suggest, "why": why,
            })

    # Scores, every non-derived one that holds a number.
    derived = _derived(inputs.doc)
    scores = model.get("scores") or {}
    updated = model.get("scores_updated") or {}
    sources = model.get("scores_source") or {}
    rows: list[dict[str, Any]] = []
    for key, value in scores.items():
        if key in derived or value is None:
            continue
        source = sources.get(key)
        when = updated.get(key)
        route = route_of_source(source)
        era, why = _score_era(source, when, released, route, both)
        rank = _precedence.source_rank(source)
        follows_url = bool(url and source and canonical(source) == canonical(url))
        target = None
        if follows_url and new_url:
            target = new_url
        elif is_preview(source):
            target = strip_url(source)
        suggest = KEEP
        note = None
        if era == ERA_PREVIEW and rank in FILL_ONLY_RANKS and target and target != source:
            suggest = RECREDIT
            note = (
                "a fill-only source never replaces a number credited to another of "
                "its pages; crediting the release's page lets the release's number "
                "replace this one when it is read"
            )
        elif era == ERA_PREVIEW and rank in FILL_ONLY_RANKS:
            note = "fill-only: the release's number replaces it only if the same page reports it"
        elif era == ERA_PREVIEW and rank == _precedence.RANK_HAND_ENTERED:
            note = "ranked as entered by hand: any source that reports the release replaces it"
        elif era == ERA_PREVIEW:
            note = "the same source re-reading the release replaces it; a lower-ranked one cannot"
        rows.append({
            "key": key, "value": value, "date": when, "source": source,
            "rank": rank, "rank_label": RANK_LABELS.get(rank, str(rank)),
            "route": route, "era": era, "why": why,
            "suggest": suggest, "note": note,
            "recredit_to": target, "follows_url": follows_url,
        })

    # Queue questions that offer this entry: the release's names, waiting.
    queue: list[dict[str, Any]] = []
    for question in inputs.questions:
        if question.get("route") == "check_new.py":
            continue
        options = [c.get("option") for c in question.get("candidates") or [] if isinstance(c, dict)]
        if name in options:
            queue.append({
                "route": question.get("route"), "route_kind": question.get("route_kind"),
                "subject": question.get("subject"), "if_previous": question.get("if_previous"),
                "source_url": question.get("source_url"),
                "preview": is_preview(question.get("subject")),
            })

    return {
        "name": name,
        "url": url,
        "new_url": new_url,
        "creator": (model.get("creator") or {}).get("name"),
        "date_added": model.get("date_added"),
        "reference": model.get("reference") is True,
        "released": released,
        "evidence": evidence,
        "renames": renames,
        "mappings": mappings,
        "scores": rows,
        "queue": queue,
    }


def detect(inputs: Inputs) -> dict[str, Any]:
    """The report the admin page renders. Pure: reads nothing but `inputs`."""
    names = {m.get("name") for m in inputs.doc.get("models") or [] if isinstance(m, dict)}
    cases = [c for m in inputs.doc.get("models") or [] if isinstance(m, dict)
             for c in [_case(m, inputs, names)] if c]
    # Released ones first; then by name, so the file only changes when a case does.
    cases.sort(key=lambda c: (not c["evidence"], c["name"]))
    return {
        "cases": cases,
        "done": [{"name": k, **v} for k, v in sorted(inputs.done.items())],
    }


def write_report(report: dict[str, Any], path: Path = DEFAULT_REPORT) -> bool:
    text = json.dumps(report, **JSON_DUMP_KWARGS, sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


# --------------------------------------------------------------------------
# applying one cleanup (validated by _answers.py first)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """One preview-cleanup record, checked and ready to carry out."""

    name: str
    action: str                                   # apply | dismiss | restore
    rename: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)   # key -> {do, url?}
    parks: tuple[tuple[str, str, str], ...] = ()  # (route, kind, subject)

    def describe(self) -> str:
        if self.action == "dismiss":
            return f"check off {self.name!r}: not a preview to clean up"
        if self.action == "restore":
            return f"list {self.name!r} as a preview to clean up again"
        bits = []
        if self.rename:
            bits.append(f"rename to {self.rename!r}")
        if self.fields:
            bits.append("set " + ", ".join(sorted(self.fields)))
        cleared = sorted(k for k, s in self.scores.items() if s["do"] == CLEAR)
        moved = sorted(k for k, s in self.scores.items() if s["do"] == RECREDIT)
        if cleared:
            bits.append(f"clear {len(cleared)} score(s)")
        if moved:
            bits.append(f"re-credit {len(moved)} score(s)")
        if self.parks:
            bits.append(f"park {len(self.parks)} mapping(s)")
        return f"clean up preview {self.name!r}: " + ("; ".join(bits) or "check it off")


def mapping_value(route: str, kind: str, subject: str) -> Any:
    by_kind = propose.ROUTES.get(route) or {}
    resolved = by_kind.get(kind, by_kind.get("*"))
    if resolved is None:
        raise CleanupError(f"unknown route {route!r}")
    raw = _read_json(route_path(resolved), {})
    return raw.get(subject) if isinstance(raw, dict) else None


def touched_paths(plan: Plan) -> list[Path]:
    paths = {DONE_FILE}
    for route, kind, _subject in plan.parks:
        by_kind = propose.ROUTES[route]
        paths.add(route_path(by_kind.get(kind, by_kind.get("*"))))
    return sorted(paths)


def carry_out(plan: Plan, llm_path: Path, *, today: str | None = None) -> list[str]:
    """Write the cleanup. The caller snapshots and rolls back on failure."""
    # Imported here: _answers imports this module, and these import a lot.
    import _history
    import _rename
    import derive_indexes
    import edit
    from _openness import UNMAPPABLE
    from _scores import stamp_score_source, stamp_score_updated

    now = today or date.today().isoformat()
    done = load_done()
    log: list[str] = []

    if plan.action == "restore":
        if done.pop(plan.name, None) is None:
            raise CleanupError(f"{plan.name!r} was never checked off")
        write_done(done)
        return [f"{DONE_FILE.name}: {plan.name!r} is listed again"]

    if plan.action == "dismiss":
        done[plan.name] = {"date": now, "how": "checked off"}
        write_done(done)
        return [f"{DONE_FILE.name}: {plan.name!r} checked off"]

    # Mappings first: they name the old slug, and the rename below carries
    # whatever still points at it.
    for route, kind, subject in plan.parks:
        by_kind = propose.ROUTES[route]
        resolved = by_kind.get(kind, by_kind.get("*"))
        getattr(importlib.import_module(resolved.module), resolved.writer)(subject, UNMAPPABLE)
        log.append(f"{route_path(resolved).name}: {subject!r} parked ({UNMAPPABLE})")

    if plan.fields or plan.scores:
        doc = json.loads(llm_path.read_text(encoding="utf-8"))
        model = next((m for m in doc.get("models") or [] if m.get("name") == plan.name), None)
        if model is None:
            raise CleanupError(f"{plan.name!r} is not a model in {llm_path.name}")
        for key, value in sorted(plan.fields.items()):
            edit.write_field(model, key, edit.parse_field_value(key, value))
        cleared = 0
        for key, decision in sorted(plan.scores.items()):
            if decision["do"] == CLEAR:
                model["scores"][key] = None
                stamp_score_updated(model, key, now)
                stamp_score_source(model, key, None)
                cleared += 1
            else:
                # The number and its date stay; only the page changes, so the
                # release's page, read later, counts as this score's own.
                stamp_score_source(model, key, canonical(decision["url"]))
        if cleared:
            derive_indexes.refresh_and_report(doc)
        _history.sync(doc)
        llm_path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
        changed = [*sorted(plan.fields),
                   *(f"{k} {'cleared' if s['do'] == CLEAR else 're-credited'}"
                     for k, s in sorted(plan.scores.items()))]
        log.append(f"{llm_path.name}: {plan.name!r}: {', '.join(changed)}")

    final = plan.name
    if plan.rename:
        log.extend(_rename.rename(plan.name, plan.rename, llm_path))
        final = plan.rename

    done.pop(plan.name, None)
    done[final] = {"date": now, "how": "cleaned up", **({"was": plan.name} if final != plan.name else {})}
    write_done(done)
    log.append(f"{DONE_FILE.name}: {final!r} recorded as cleaned up")
    return log
