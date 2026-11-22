"""Every source mapping, read the other way round: per model, not per source.

The mapping files are keyed by the name a source publishes, which is the right
shape for the queue -- one question per unrecognised name -- and the wrong one
for the question the admin page's Models tab asks: *what feeds this model?*
A preview entry cleaned up into its release, a board row parked as
``__unmappable__`` by a cleanup, a name that went to the wrong model months
ago: each is a line in some file, and finding it meant knowing which file.

This module publishes all of them in one document, ``_pending/mappings.json``,
next to the queue (see ``mapping_index.py``):

  * ``routes`` -- the source routes whose answers are model names (and the
    Artificial Analysis route, whose file runs model -> slug), with the file
    each one owns. Read off ``propose.ROUTES``, the one table the answer
    applier trusts, so a route the page can edit is a route ``answer.py`` can
    write.
  * ``files`` -- each of those files as it stands: name -> answer.
  * ``rejections`` -- the candidates ruled out per queued name ("not this
    one"), from ``rejected-candidates.json`` and the AA route's own ignore
    file (see ``_rejections.py``).
  * ``parked`` -- per model, the names parked on a sentinel
    (``__unmappable__``, ...) that look like it: the compacted name starts
    with the model's, and what follows is not a digit (``hy4preview`` for
    ``hy4``, never ``glm53`` for ``glm-5``), or is the name a preview cleanup
    renamed the model from. A cleanup parks a preview's names, and once
    parked they name nothing, so without this the model's card could not
    offer them back.
  * ``contributions`` -- per model, which columns each source currently
    supplies, read off the page each score is credited to
    (``scores_source``). A score's page says which fetcher wrote it; it does
    not say which of a source's names did, so a source listing two names that
    both map onto one model is credited once, for the model.

Reads only. The page edits through two records, ``mapping-edit`` and
``rejection-edit`` (see ``_answers.py``), validated like every other answer.
"""

from __future__ import annotations

import importlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import _matching
import _precedence as P
import _rejections
from _openness import SENTINELS
import propose
from fill_source_urls import canonical

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
PUBLISHED = HERE / "_pending" / "mappings.json"

# The universes whose answers this index covers. The benchmark-label routes
# (Hugging Face, llm-stats' benchmarks) map a source's column name onto a
# column, which is a question about a benchmark, not about a model; the
# Benchmarks tab is where those live.
COVERED = (propose.MODELS, propose.AA_SLUGS)
# The benchmark-label routes, listed too since the Benchmarks tab edits them:
# a label a source prints over a column (a Hugging Face card's "SWE-bench
# Verified (500)", an llm-stats benchmark id) onto the column it fills.
LABEL_UNIVERSES = (propose.BENCHMARKS,)
HF_LABEL_ROUTE = "update_huggingface_mapping.py *"
LLMSTATS_LABEL_ROUTE = "update_llmstats_mapping.py llmstats-benchmark"
# The model route whose pages a label route's scores are credited to: a card
# label's score cites the model's card, an llm-stats id's one of llm-stats'
# pages.
LABEL_OWNER = {
    HF_LABEL_ROUTE: "card",  # CARD, defined below
    LLMSTATS_LABEL_ROUTE: "update_llmstats_mapping.py llmstats-model",
}
# How far a stored score may sit from the card's printed number and still be
# that number: the column's rounding, at most a tenth.
VALUE_SLACK = 0.051

# Scores credited to a page no route owns. A model card is read off the
# entry's own `url` field rather than through a mapping, and anything else is
# a page cited by hand.
CARD = "card"
OTHER = "other"
# Not a score: Spheron's mapping supplies the VRAM columns, which the Models
# tab lists beside the scores so the route does not look idle.
VRAM = "vram"


def route_id(route: str, kind: str) -> str:
    """The id the page keys a route by: the script, then the prompt kind."""
    return f"{route} {kind}"


def _values(table: dict[Any, str]) -> list[str]:
    return [url for url in table.values() if url]


def route_pages() -> dict[str, list[str]]:
    """Route id -> the page prefixes its fetcher credits scores to.

    Built from the same URL constants ``_precedence`` ranks sources by, so a
    page added to a fetcher is attributed here the day it is ranked there.
    A route missing from this table supplies no score this index can see; the
    test suite lists every route on purpose so that is a choice, not an
    oversight.
    """
    pages: dict[str, list[str]] = {
        "update_rebench_mapping.py *": [],
        "update_osworld_mapping.py *": [
            P.OSWORLD_SOURCE_URL, P.OSWORLD_V2_SOURCE_URL, *_values(P.OSWORLD_KEY_URLS),
        ],
        "update_deepswe_mapping.py *": [P.DEEPSWE_SOURCE_URL, P.DATACURVE_SOURCE_URL],
        "update_frontierswe_mapping.py *": [
            P.FRONTIERSWE_SOURCE_URL, *_values(P.FRONTIERSWE_KEY_URLS),
        ],
        "update_frontiercode_mapping.py *": [P.FRONTIERCODE_SOURCE_URL],
        "update_swe_atlas_mapping.py *": _values(P.SWE_ATLAS_KEY_URLS),
        "update_tbench_mapping.py *": _values(P.TBENCH_KEY_URLS),
        "update_agents_last_exam_mapping.py *": [
            P.AGENTS_LAST_EXAM_SOURCE_URL, *_values(P.AGENTS_LAST_EXAM_KEY_URLS),
        ],
        "update_aa_coding_agents_mapping.py *": [P.AA_CODING_AGENTS_SOURCE_URL],
        "update_evals_report_mapping.py *": _values(P.EVALS_REPORT_KEY_URLS),
        "update_vals_mapping.py *": _values(P.VALS_KEY_URLS),
        "update_swe_marathon_mapping.py *": [P.SWE_MARATHON_SOURCE_URL],
        "update_toolathlon_mapping.py *": [P.TOOLATHLON_SOURCE_URL],
        "update_programbench_mapping.py *": [P.PROGRAMBENCH_SOURCE_URL],
        "update_kernelbench_mapping.py *": [P.KERNELBENCH_SOURCE_URL],
        "update_real_swe_mapping.py *": [P.REAL_SWE_SOURCE_URL],
        "update_zerobench_mapping.py *": [P.ZEROBENCH_SOURCE_URL],
        "update_arc_prize_mapping.py *": _values(P.ARC_PRIZE_KEY_URLS),
        "update_arena_mapping.py *": _values(P.ARENA_KEY_URLS),
        "update_cybergym_mapping.py *": _values(P.CYBERGYM_KEY_URLS),
        "update_sec_bench_mapping.py *": [P.SEC_BENCH_SOURCE_URL, *_values(P.SEC_BENCH_KEY_URLS)],
        "update_hle_diamond_mapping.py *": [P.HLE_DIAMOND_SOURCE_URL],
        "update_openrouter_mapping.py *": [P.OPENROUTER_PREFIX],
        "update_epoch_mapping.py *": _values(P.EPOCH_PAGE_URLS),
        "update_mcp_atlas_mapping.py *": [P.MCP_ATLAS_SOURCE_URL],
        "update_swe_bench_pro_v2_mapping.py *": [P.SWE_BENCH_PRO_V2_SOURCE_URL],
        "update_bfcl_mapping.py *": [P.BFCL_SOURCE_URL],
        # VRAM only; see VRAM above.
        "update_spheron_mapping.py *": [],
        # The leaderboard and every per-model page under the host.
        "update_llmstats_mapping.py llmstats-model": [
            P.LLMSTATS_SOURCE_URL, canonical("https://llm-stats.com"),
        ],
        # The host and the model pages; the Coding Agent Index lives under the
        # same host and wins on its longer prefix.
        "update_artificialanalysis_mapping.py *": [
            canonical("https://artificialanalysis.ai"), P.AA_MODEL_PAGE_PREFIX,
        ],
    }
    return {rid: sorted({canonical(url) for url in urls if url}) for rid, urls in pages.items()}


def _prefix_table(pages: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs = [(prefix, rid) for rid, prefixes in pages.items() for prefix in prefixes]
    pairs.append((canonical(P.HUGGING_FACE_PREFIX), CARD))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def attribute(url: str | None, table: list[tuple[str, str]]) -> str:
    """The route id a score's page belongs to, CARD, or OTHER.

    Matched on a path boundary, as ``_precedence.source_rank`` does, so a
    lookalike host cannot borrow a source's credit.
    """
    if not url:
        return OTHER
    candidate = canonical(url)
    for prefix, rid in table:
        if candidate == prefix or candidate.startswith(f"{prefix}/"):
            return rid
    return OTHER


def covered_routes(universes: tuple[str, ...] = COVERED) -> list[dict[str, str]]:
    """The routes the Models tab can edit, in propose.ROUTES' order."""
    routes = []
    for script, by_kind in propose.ROUTES.items():
        for kind, route in by_kind.items():
            if route.universe not in universes:
                continue
            module = importlib.import_module(route.module)
            path = getattr(module, route.mapping_const)
            routes.append({
                "id": route_id(script, kind),
                "route": script,
                "route_kind": kind,
                "universe": route.universe,
                "file": path.name,
            })
    return routes


def _read_file(name: str) -> dict[str, Any]:
    try:
        raw = json.loads((HERE / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str) and value:
            out[key] = value
        elif isinstance(value, list):
            items = [v for v in value if isinstance(v, str) and v]
            if items:
                out[key] = items
    return out


def contributions(doc: dict[str, Any], pages: dict[str, list[str]] | None = None) -> dict[str, dict[str, list[str]]]:
    """Model -> route id (or CARD / OTHER) -> the columns it supplies now."""
    table = _prefix_table(pages if pages is not None else route_pages())
    derived = {
        key for key, bench in (doc.get("benchmarks") or {}).items()
        if isinstance(bench, dict) and bench.get("derived")
    }
    out: dict[str, dict[str, list[str]]] = {}
    for model in doc.get("models") or []:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            continue
        scores = model.get("scores") or {}
        sources = model.get("scores_source") or {}
        by_route: dict[str, list[str]] = {}
        for key, value in scores.items():
            if value is None or key in derived:
                continue
            by_route.setdefault(attribute(sources.get(key), table), []).append(key)
        if any(v for v in (model.get("vram") or {}).values()):
            by_route.setdefault("update_spheron_mapping.py *", []).append(VRAM)
        if by_route:
            out[model["name"]] = by_route
    return out


def rejections() -> dict[str, dict[str, list[str]]]:
    """Route id -> queued name -> the candidates ruled out for it."""
    ids = {r["route"]: r["id"] for r in covered_routes((*COVERED, *LABEL_UNIVERSES))}
    out: dict[str, dict[str, list[str]]] = {}
    for route, by_subject in _rejections.load().items():
        rid = ids.get(route)
        if rid is None:
            continue
        for subject, options in by_subject.items():
            if options:
                out.setdefault(rid, {})[subject] = sorted(options)
    aa = importlib.import_module(_rejections.AA_MODULE)
    for subject, slugs in aa.load_ignored_aa_suggestions().items():
        if slugs:
            out.setdefault(ids[_rejections.AA_ROUTE], {})[subject] = sorted(slugs)
    return out


DONE_FILE = HERE / "preview-cleanup-done.json"


def _former_names() -> dict[str, list[str]]:
    """Model -> the names a preview cleanup renamed it from."""
    try:
        raw = json.loads(DONE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, list[str]] = {}
    for name, entry in (raw.items() if isinstance(raw, dict) else []):
        if isinstance(entry, dict) and isinstance(entry.get("was"), str):
            out.setdefault(name, []).append(entry["was"])
    return out


def _looks_like(name: str, model: str) -> bool:
    if not model or not name.startswith(model):
        return False
    rest = name[len(model):]
    return not rest or not rest[0].isdigit()


def parked(
    models: list[str],
    routes: list[dict[str, str]],
    files: dict[str, dict[str, Any]],
    former: dict[str, list[str]] | None = None,
) -> dict[str, list[list[str]]]:
    """Model -> [route id, name] for each parked name that looks like it."""
    former = former if former is not None else _former_names()
    model_universe = {r["id"] for r in routes if r["universe"] == propose.MODELS}
    keys = {m: _matching.compact(m) for m in models}
    olds = {m: {_matching.compact(w) for w in former.get(m, [])} for m in models}
    out: dict[str, list[list[str]]] = {}
    for rid in sorted(model_universe):
        for name, value in sorted(files.get(rid, {}).items()):
            if not (isinstance(value, str) and value in SENTINELS):
                continue
            tail = _matching.compact(name.rsplit("/", 1)[-1])
            for model in models:
                if _looks_like(tail, keys[model]) or tail in olds[model]:
                    out.setdefault(model, []).append([rid, name])
    return out


def hf_crawl(doc: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    """Model -> {card label: value}, from the crawl a refresh cached, or None.

    fetch_huggingface.crawl_all_models() caches what it read under a key of
    the (slug, repo) list and its parser; the same key is asked for here, with
    no age limit, since a card read hours ago still says which label a stored
    number came from. Never fetches: on a runner that has not crawled (an
    answers-only commit) this is None and received() falls back.
    """
    try:
        import _cache
        import fetch_huggingface as hf
    except Exception:
        return None
    key = _cache.digest([sorted(hf.iter_hf_models(doc)), hf._PARSER_DIGEST])
    rows = _cache.load("huggingface-cards", key, 10 * 365 * 86400, label="Hugging Face model cards")
    if not isinstance(rows, list):
        return None
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("model"), str) and isinstance(row.get("scores"), dict):
            out[row["model"]] = {
                k: v for k, v in row["scores"].items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
    return out


def _prior_received(path: Path = PUBLISHED) -> dict[str, dict[str, list[list[Any]]]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    got = doc.get("received") if isinstance(doc, dict) else None
    return got if isinstance(got, dict) else {}


def _owned(model: dict[str, Any], key: str, owner: str, table: list[tuple[str, str]]) -> bool:
    """Whether the stored score is still the label route's: its page is that
    route's, so no other board or AA has replaced it. A card label's score has
    to cite a model card, not a dataset page someone cited by hand."""
    url = (model.get("scores_source") or {}).get(key)
    if attribute(url, table) != owner:
        return False
    return owner != CARD or "/datasets/" not in canonical(url)


def received(
    doc: dict[str, Any],
    files: dict[str, dict[str, Any]],
    crawl: dict[str, dict[str, float]] | None = None,
    prior: dict[str, dict[str, list[list[Any]]]] | None = None,
) -> dict[str, dict[str, list[list[Any]]]]:
    """Label route id -> label -> [[model, value, shared], ...].

    The scores a label delivers right now: for each column the label maps to,
    every model whose stored score is credited to the label route's pages --
    a score another fetcher or AA has since overridden is not the label's any
    more, and is left out. When several labels of one route map onto the same
    column, which one a model's number came from is read off the card crawl
    (the label whose printed value is the stored one; a card that prints it
    under none of them delivers it under none), else off the last published
    index; failing both, the score is listed under each of them with `shared`
    set to how many it may have come from. Labels the crawl
    confirms -- a card printing one number under two headings -- each deliver
    it, and are not marked.
    """
    table = _prefix_table(route_pages())
    crawl = crawl or {}
    prior = prior or {}
    models = [m for m in doc.get("models") or [] if isinstance(m, dict) and isinstance(m.get("name"), str)]
    out: dict[str, dict[str, list[list[Any]]]] = {}
    for rid, owner in LABEL_OWNER.items():
        by_key: dict[str, list[str]] = {}
        for label, value in (files.get(rid) or {}).items():
            if isinstance(value, str) and value not in SENTINELS:
                by_key.setdefault(value, []).append(label)
        got: dict[str, list[list[Any]]] = {}
        for key, labels in by_key.items():
            for model in models:
                value = (model.get("scores") or {}).get(key)
                if not isinstance(value, (int, float)) or not _owned(model, key, owner, table):
                    continue
                name = model["name"]
                which = labels
                guessed = False
                if len(labels) > 1:
                    crawled = rid == HF_LABEL_ROUTE and name in crawl
                    if crawled:
                        # The card as read today: the labels printing the stored
                        # number. None -- a number the card no longer prints, left
                        # over from an earlier read -- is no label's.
                        printed = crawl[name]
                        which = [l for l in labels if l in printed and abs(printed[l] - value) <= VALUE_SLACK]
                    elif rid in prior:
                        # The last index that knew; it left the score out if no
                        # label delivered it, and so does this one.
                        which = [
                            l for l in labels
                            if any(e[0] == name and not e[2] and abs(e[1] - value) <= VALUE_SLACK
                                   for e in prior[rid].get(l, []) if isinstance(e, list) and len(e) == 3)
                        ]
                        if not which and not any(
                            e[0] == name for l in labels for e in prior[rid].get(l, []) if isinstance(e, list)
                        ):
                            which, guessed = labels, True
                    else:
                        which, guessed = labels, True
                    if not which:
                        continue
                # Several labels confirmed (a card printing one number under two
                # headings) all deliver it; only a guess is marked shared.
                shared = len(which) if guessed else 0
                for label in which:
                    got.setdefault(label, []).append([name, value, shared])
        if got:
            out[rid] = {label: sorted(entries) for label, entries in sorted(got.items())}
    return out


def build(llm_path: Path = DEFAULT_LLM_JSON, *, crawl: dict[str, dict[str, float]] | None = None,
          prior: dict[str, dict[str, list[list[Any]]]] | None = None) -> dict[str, Any]:
    doc = json.loads(Path(llm_path).read_text(encoding="utf-8"))
    routes = covered_routes()
    label_routes = covered_routes(LABEL_UNIVERSES)
    files = {r["id"]: _read_file(r["file"]) for r in [*routes, *label_routes]}
    models = [m["name"] for m in doc.get("models") or [] if isinstance(m, dict) and isinstance(m.get("name"), str)]
    return {
        "generated": date.today().isoformat(),
        "routes": [*routes, *label_routes],
        "files": files,
        "rejections": rejections(),
        "parked": parked(models, routes, files),
        "contributions": contributions(doc),
        "received": received(
            doc, files,
            crawl if crawl is not None else hf_crawl(doc),
            prior if prior is not None else _prior_received(),
        ),
    }


def publish(path: Path = PUBLISHED, llm_path: Path = DEFAULT_LLM_JSON) -> bool:
    """Write the index; True when the file changed."""
    text = json.dumps(build(llm_path), indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        # The date alone moving is not a change worth a commit line.
        if _without_date(old) == _without_date(text):
            return False
    path.write_text(text, encoding="utf-8")
    return True


def _without_date(text: str) -> str:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(doc, dict):
        doc.pop("generated", None)
    return json.dumps(doc, sort_keys=True)
