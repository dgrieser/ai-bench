#!/usr/bin/env python3
"""Decide whether a leaderboard model ships open weights.

Every model a source lists has one of three weight statuses:

  open       the weights are published
  upcoming   the weights are announced but not out yet ("coming soon",
             "weights to be released") -- the lab has said it will, and a
             source still calling it closed is only waiting for the upload
  closed     proprietary / API only

llm.json only tracks open-weight models, so a source model that is provably
closed can never map to one -- update_*_mapping.py records those without
prompting. An *upcoming* model is the one that must never be buried that way:
it is exactly the release llm.json will want in a few days, so it always stays
a human question.

Most sources publish the status directly, each in its own words; the fetchers
read it into an ``open_weights`` boolean (None while unknown or upcoming) and a
``weights`` status string, with the parsers below:

  llm-stats.com / zeroeval   ``license``      "proprietary" vs an open licence
  BFCL                       ``License``      the same
  LMArena                    ``license``      the same
  benchlm.ai (DeepSWE)       ``sourceType``   "Proprietary" / "Open Weight"
  Toolathlon                 model type       "Proprietary" / "Open-Weights"
  Epoch AI                   ``accessibility`` "Open weights (...)" / "API access"
  evals.report               an " Open" suffix on the model label
  OpenRouter                 ``hugging_face_id`` (set only for open models)
  Artificial Analysis        ``licensing.is_open_weights``

The rest publish only a creator, so their names are resolved against a pooled
index built from the sources above and keyed by a normalized model name. The
index is cached under ~/.cache/ai-bench/openness.json for CACHE_TTL_SECONDS.
Pooling ranks open > upcoming > closed: a name is closed only while every
source that knows it says so, which is what keeps one mislabelled source from
dropping a model.

A person has the last word: weights-overrides.json holds the statuses set by
hand (the admin page's "weights" control, or ``--set``), and beats every
source. Two guards keep a mislabelled source from silently dropping a model:

  * names that look like an llm.json model (see is_closed_weights) are never
    auto-recorded, and
  * auto-recorded entries get their own CLOSED_WEIGHTS sentinel, so
    ``--recheck-closed`` can put them back in front of a human.

The one deliberate blind spot is the reference list (see _reference.py): those
models are closed, every source says so correctly, and the index carries them
anyway. reference_model_for() is checked before any verdict, so a source
telling the truth cannot bury a row that is in the file on purpose.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from _reference import load_reference_slugs
from _timing import timed

# Sentinel stored for source names a human reviewed and deliberately left
# unmapped. Never used as a real llm.json model slug.
UNMAPPABLE = "__unmappable__"

# Sentinel stored for source names skipped automatically because the source
# says the model has closed weights. Kept distinct from UNMAPPABLE so the
# machine-made decisions can be re-reviewed.
CLOSED_WEIGHTS = "__closed_weights__"

# Sentinel written by propose.py for a source name it could not resolve with
# confidence. Unlike the two above it is NOT a decision: the reviewed-set loaders
# skip it, so a proposal PR that gets merged with one still in place leaves the
# name to be asked again. That is what makes the suggestion buttons in the PR
# safe -- ignoring one costs nothing.
PENDING = "__pending__"

# Mapping values that are markers rather than llm.json model slugs.
SENTINELS = frozenset({UNMAPPABLE, CLOSED_WEIGHTS, PENDING})

# The three weight statuses. Strings rather than an enum because they travel:
# into the fetchers' rows, the cached index, the queue JSON and the admin page.
OPEN = "open"
UPCOMING = "upcoming"
CLOSED = "closed"
STATUSES = (OPEN, UPCOMING, CLOSED)
# How pooling resolves a disagreement: the most open verdict wins. "Closed" is
# the one verdict anything acts on without asking, so it has to be unanimous;
# "open" beats "upcoming" because a source that has seen the weights is newer
# news than one that has seen the announcement.
_RANK = {CLOSED: 0, UPCOMING: 1, OPEN: 2}

HERE = Path(__file__).resolve().parent
# Statuses set by a person: {"<model name as the source spells it>": status}.
# Matched by normalize(), so one line covers every board's spelling.
OVERRIDES_FILE = HERE / "weights-overrides.json"

CACHE_PATH = os.path.expanduser("~/.cache/ai-bench/openness.json")
# Bumped when the pooled sources or the cached shape change, so an index built
# by an older run (booleans, fewer sources) is rebuilt rather than trusted.
CACHE_VERSION = 2
# Overridable because the quick run accepts a much older index: whether a
# model ships weights is a slow-moving fact, and rebuilding the index is a
# handful of sources and seconds of a run that exists to be fast.
try:
    CACHE_TTL_SECONDS = int(os.getenv("AI_BENCH_OPENNESS_TTL", str(12 * 3600)))
except ValueError:
    CACHE_TTL_SECONDS = 12 * 3600

# Trailing reasoning-effort / thinking-mode modifiers, in any of the bracket
# styles the leaderboards use ("-high", " [medium]", " (xhigh)", " thinking").
_EFFORT_RE = re.compile(
    r"[-_ \[(]+(?:xhigh|x-high|very-high|high|medium|low|minimal|max"
    r"|no[- ]?thinking|thinking|reasoning)[\])]*\s*$",
    re.IGNORECASE,
)
# Release-date suffixes: "gpt-5.2-2025-12-11" -> "gpt-5.2".
_DATE_RE = re.compile(r"[-_/](?:19|20)\d\d[-_/]\d\d[-_/]\d\d\b")
_PARENS_RE = re.compile(r"\([^)]*\)")

# Wording a source uses for weights that are announced but not published yet:
# "Open weights (coming soon)", "to be open-sourced", "weights pending",
# "TBA". Checked before the open/closed vocabularies, because the phrases
# usually carry the word "open" too. "Unreleased" is not one of them: Epoch
# labels a model nobody can use that way, with no promise attached.
_UPCOMING_RE = re.compile(
    r"coming[- ]soon|\bsoon\b|to be (?:released|published|open[- ]?sourced|opened)"
    r"|\b(?:planned|upcoming|forthcoming)\b"
    r"|weights? (?:pending|coming|release pending|tbd|tba)"
    r"|release pending|pending release|\btb[ad]\b",
    re.IGNORECASE,
)

_INDEX: dict[str, str] | None = None
_OVERRIDES: dict[str, str] | None = None
_OFFLINE: dict[str, str] | None = None


def normalize(name: str) -> str:
    """Reduce a model name to a comparison key shared across sources.

    Drops harness parentheticals, release dates and reasoning-effort suffixes,
    then flattens every separator to a single space so the ".", "-" and " "
    spellings of a version collapse together:

    "GPT-5.2-2025-12-11-xhigh"       -> "gpt 5 2"
    "Qwen3.5-27B"                    -> "qwen3 5 27b"
    "Opus 4.8 (Claude Code) xHigh"   -> "opus 4 8"
    """
    text = _PARENS_RE.sub(" ", name.strip().lower())
    text = _DATE_RE.sub(" ", text)
    while True:
        stripped = _EFFORT_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Status <-> the boolean the fetchers have always reported
# --------------------------------------------------------------------------


def status_from_open(value: Any) -> str | None:
    """A source's open-weights boolean as a status; None stays unknown."""
    if value is True:
        return OPEN
    if value is False:
        return CLOSED
    return None


def open_from_status(status: str | None) -> bool | None:
    """The ``open_weights`` boolean a fetcher reports beside ``weights``.

    Upcoming is None, not False: the weights are not out, but nothing that
    reads the boolean may treat the model as closed for good.
    """
    if status == OPEN:
        return True
    if status == CLOSED:
        return False
    return None


def is_upcoming_text(value: Any) -> bool:
    """Whether a free-text label announces weights that are not out yet."""
    return isinstance(value, str) and bool(_UPCOMING_RE.search(value))


def merge(*statuses: str | None) -> str | None:
    """Pool several verdicts on one model: the most open one wins."""
    known = [s for s in statuses if s in _RANK]
    return max(known, key=_RANK.__getitem__) if known else None


# --------------------------------------------------------------------------
# Parsers for the labels the sources publish
# --------------------------------------------------------------------------


def license_status(value: Any) -> str | None:
    """A ``license`` field as a status.

    "proprietary" (or "closed", "API only") is the only closed value; a licence
    that is announced but not applied yet ("Apache 2.0 (coming soon)", "TBA")
    is upcoming; any other licence name is open.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    if text in {"unknown", "none", "null", "-", "?"}:
        return None
    if is_upcoming_text(text):
        return UPCOMING
    if text in {"proprietary", "closed", "api", "api only", "api-only", "commercial", "n/a"}:
        return CLOSED
    if text.startswith("proprietary"):
        return CLOSED
    return OPEN


def license_open(value: Any) -> bool | None:
    """Read llm-stats' ``license`` field: "proprietary" is the only closed value."""
    return open_from_status(license_status(value))


def source_type_status(value: Any) -> str | None:
    """A source's weights-availability label as a status.

    Covers benchlm.ai's ``sourceType`` ("Open Weight" / "Proprietary"),
    Toolathlon's model type ("Open-Weights" / "Open-Source" / "Proprietary")
    and Epoch's ``accessibility`` ("Open weights (unrestricted)", "API access",
    "Hosted access (no API)"); separators are normalised so the spellings land
    on the same verdict.

    The field is also left null or "Pending" for models awaiting review, which
    carries no verdict -- a source's review queue is not a lab's release plan.
    """
    if not isinstance(value, str):
        return None
    text = re.sub(r"[-_]+", " ", value).strip().lower()
    if not text or text in {"pending", "unknown", "n/a", "none"}:
        return None
    if is_upcoming_text(text):
        return UPCOMING
    if text in {"open weight", "open weights", "open source", "open"} or text.startswith(
        ("open weight", "open source")
    ):
        return OPEN
    if text in {"proprietary", "closed", "closed weight", "closed weights", "closed source"}:
        return CLOSED
    if text.startswith(("api access", "hosted access", "unreleased", "proprietary")):
        return CLOSED
    return None


def source_type_open(value: Any) -> bool | None:
    """source_type_status() as the ``open_weights`` boolean."""
    return open_from_status(source_type_status(value))


# --------------------------------------------------------------------------
# The pooled index
# --------------------------------------------------------------------------


def _pool(statuses: dict[str, str], name: Any, status: str | None) -> None:
    """Add one verdict to the index, the most open verdict winning a key.

    A disagreement means at least one source is wrong about this name -- or
    one has caught up with a release the other has not -- so the key never
    reads closed unless every source says so.
    """
    if status not in _RANK or not isinstance(name, str):
        return
    key = normalize(name)
    if not key:
        return
    statuses[key] = merge(statuses.get(key), status) or status


def row_status(entry: dict[str, Any]) -> str | None:
    """A fetcher row's status: ``weights`` where it reports one, else its boolean."""
    status = entry.get("weights")
    if status in _RANK:
        return status
    return status_from_open(entry.get("open_weights"))


def _pool_rows(statuses: dict[str, str], rows: Iterable[dict[str, Any]], *keys: str) -> None:
    for entry in rows:
        status = row_status(entry)
        for key in keys:
            _pool(statuses, entry.get(key), status)


def _pool_llmstats(statuses: dict[str, str]) -> None:
    import fetch_llmstats

    _pool_rows(statuses, fetch_llmstats.get_scores(), "model", "name")


def _pool_deepswe(statuses: dict[str, str]) -> None:
    import fetch_deepswe

    _pool_rows(statuses, fetch_deepswe.get_scores(), "model", "slug")


def _pool_evals_report(statuses: dict[str, str]) -> None:
    import fetch_evals_report

    entries = fetch_evals_report.get_scores(
        list(fetch_evals_report.BENCHMARKS), include_unverified=True
    )
    _pool_rows(statuses, entries, "model")


def _pool_bfcl(statuses: dict[str, str]) -> None:
    import fetch_bfcl

    _pool_rows(statuses, fetch_bfcl.get_scores(), "model")


def _pool_toolathlon(statuses: dict[str, str]) -> None:
    import fetch_toolathlon

    _pool_rows(statuses, fetch_toolathlon.get_scores(include_self_reported=True), "model")


def _pool_epoch(statuses: dict[str, str]) -> None:
    import fetch_epoch

    _pool_rows(statuses, fetch_epoch.get_catalogue(), "model")


def _pool_arena(statuses: dict[str, str]) -> None:
    import fetch_arena

    # The text board alone: it lists every model the other boards do, and
    # each board is a parquet download of its own.
    _pool_rows(statuses, fetch_arena.get_scores(["arena_text"]), "model", "raw")


def _pool_sec_bench(statuses: dict[str, str]) -> None:
    import fetch_sec_bench

    _pool_rows(statuses, fetch_sec_bench.get_scores(), "model", "raw")


def _pool_openrouter(statuses: dict[str, str]) -> None:
    import fetch_openrouter

    for model_id, status in fetch_openrouter.fetch_model_weights().items():
        # "qwen/qwen3.5-27b": the bare name is how every other board spells it.
        _pool(statuses, model_id, status)
        _pool(statuses, model_id.split("/", 1)[-1], status)


# (label, pooler) for every source that publishes weight availability. A
# source added here is read by build_index() in parallel with the rest; bump
# CACHE_VERSION with it so the cached index is rebuilt.
POOLED_SOURCES: tuple[tuple[str, Any], ...] = (
    ("llm-stats", _pool_llmstats),
    ("evals.report", _pool_evals_report),
    ("DeepSWE", _pool_deepswe),
    ("BFCL", _pool_bfcl),
    ("Toolathlon", _pool_toolathlon),
    ("Epoch AI", _pool_epoch),
    ("LMArena", _pool_arena),
    ("SEC-bench", _pool_sec_bench),
    ("OpenRouter", _pool_openrouter),
)


def build_index() -> dict[str, str]:
    """Fetch every source that publishes weight availability and pool it."""
    from concurrent.futures import ThreadPoolExecutor

    sources = POOLED_SOURCES

    def read(label: str, pool: Any) -> dict[str, str]:
        # Each source pools into its own dict: the hosts are read at once
        # rather than one after another.
        own: dict[str, str] = {}
        try:
            timed(f"openness: {label}", pool, own)
        except Exception as exc:  # noqa: BLE001 - a dead source must not block review
            print(f"  openness: skipping {label} ({exc})", file=sys.stderr)
        return own

    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        per_source = list(executor.map(lambda source: read(*source), sources))

    # merge() is order-independent, so the parallel reads build the index a
    # sequential loop would.
    statuses: dict[str, str] = {}
    for own in per_source:
        for key, status in own.items():
            statuses[key] = merge(statuses.get(key), status) or status
    return statuses


def _load_cached(max_age: float | None = None) -> dict[str, str] | None:
    max_age = CACHE_TTL_SECONDS if max_age is None else max_age
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    if time.time() - payload.get("fetched", 0) > max_age:
        return None
    statuses = payload.get("statuses")
    if not isinstance(statuses, dict):
        return None
    return {k: v for k, v in statuses.items() if isinstance(k, str) and v in _RANK}


def _save_cached(statuses: dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        # Through a temporary and a rename, so a reader never sees half a
        # file -- the quick run starts twenty updaters that all read it.
        temporary = f"{CACHE_PATH}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                {"version": CACHE_VERSION, "fetched": time.time(), "statuses": statuses},
                handle,
            )
        os.replace(temporary, CACHE_PATH)
    except OSError:
        pass


def status_index(refresh: bool = False, *, offline: bool = False) -> dict[str, str]:
    """Normalized model name -> weight status, pooled across sources.

    `offline` never fetches: the index already loaded, else the cached one
    however old, else nothing. For readers that only label (the queue JSON),
    which must not turn a render into a round of downloads.
    """
    global _INDEX
    if _INDEX is not None and not refresh:
        return _INDEX
    if offline:
        global _OFFLINE
        if _OFFLINE is None:
            _OFFLINE = _load_cached(max_age=float("inf")) or {}
        return _OFFLINE
    statuses = None if refresh else _load_cached()
    if statuses is None:
        # Only the miss is timed: a hit is a file read, and a line per hit
        # would drown the timings that matter. The miss is every source over
        # the network, which every mapping updater then waits on.
        statuses = timed("_openness.build_index", build_index)
        if statuses:
            _save_cached(statuses)
    _INDEX = statuses
    return _INDEX


def open_index(refresh: bool = False) -> dict[str, bool]:
    """Normalized model name -> has open weights, for the names with a verdict.

    The older, boolean view of status_index(): upcoming names are left out,
    since their weights are neither out nor closed. Also what the updaters call
    to warm the index before their per-name output.
    """
    return {
        key: status == OPEN
        for key, status in status_index(refresh).items()
        if status != UPCOMING
    }


# --------------------------------------------------------------------------
# weights-overrides.json -- what a person said
# --------------------------------------------------------------------------


def load_overrides(path: Path | None = None) -> dict[str, str]:
    """The file as written: model name -> status."""
    path = path or OVERRIDES_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and k.strip() and v in _RANK}


def _override_index() -> dict[str, str]:
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = {}
        for name, status in load_overrides().items():
            key = normalize(name)
            if key:
                _OVERRIDES[key] = status
    return _OVERRIDES


def override_for(name: str) -> str | None:
    """The status a person set for `name` (by its normalized key), if any."""
    key = normalize(name) if isinstance(name, str) else ""
    return _override_index().get(key) if key else None


def set_override(name: str, status: str | None, path: Path | None = None) -> bool:
    """Record (or, with None, drop) a person's status for `name`.

    Any line already in the file for the same normalized name is replaced, so
    the file holds one verdict per model however many boards spell it. Returns
    True when the file changed.
    """
    global _OVERRIDES
    if status is not None and status not in _RANK:
        raise ValueError(f"status must be one of {', '.join(STATUSES)} or None, not {status!r}")
    name = name.strip()
    key = normalize(name)
    if not key:
        raise ValueError(f"{name!r} has no comparable name")
    path = path or OVERRIDES_FILE
    current = load_overrides(path)
    same = [k for k in current if normalize(k) == key]
    if status is not None and [current[k] for k in same] == [status]:
        return False  # already said, under this or another spelling
    updated = {k: v for k, v in current.items() if normalize(k) != key}
    if status is not None:
        updated[name] = status
    if updated == current:
        return False
    path.write_text(
        json.dumps({k: updated[k] for k in sorted(updated, key=str.lower)},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if path == OVERRIDES_FILE:
        _OVERRIDES = None
    return True


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def status(name: str, *, weights: str | None = None, offline: bool = False) -> str | None:
    """`name`'s weight status: a person's override, else the source's own
    verdict (`weights`, when the caller has one), else the pooled index."""
    return describe(name, weights=weights, offline=offline)[0]


def describe(
    name: str, *, weights: str | None = None, offline: bool = False
) -> tuple[str | None, str | None]:
    """(status, where it came from): "override", "source", "sources" or None."""
    override = override_for(name)
    if override is not None:
        return override, "override"
    if weights in _RANK:
        return weights, "source"
    index = status_index(offline=offline)
    pooled = index.get(normalize(name)) if isinstance(name, str) else None
    if pooled is not None:
        return pooled, "sources"
    return None, None


def reopened(name: str) -> bool:
    """Whether a person's override takes back a __closed_weights__ skip.

    The mapping files' reviewed-name loaders call this for every name recorded
    as closed: one overridden to "open" or "upcoming" is unreviewed again, so
    the next run asks about it -- the same thing --recheck-closed does for every
    such name, done for the one a person named.
    """
    return override_for(name) in (OPEN, UPCOMING)


def verdict(name: str) -> bool | None:
    """True if `name` is open, False if closed, None if unknown or upcoming."""
    return open_from_status(status(name))


def _resembles_tracked_model(name: str, tracked: Iterable[str]) -> str | None:
    """Return the llm.json name `name` looks like a shorter/longer spelling of.

    llm.json holds only open-weight models, so any resemblance to one outweighs
    a source calling the model closed -- sources do get this wrong (llm-stats
    labels Grok 2 and ERNIE 4.5 proprietary although both shipped weights).
    """
    key = normalize(name)
    if not key:
        return None
    for candidate in tracked:
        other = normalize(candidate)
        if not other:
            continue
        if other == key or other.startswith(key + " ") or key.startswith(other + " "):
            return candidate
    return None


def reference_model_for(name: str) -> str | None:
    """The reference model `name` names, if any (see _reference.py).

    These models *are* closed, so every source is right to say so and the
    verdict would be acted on -- which is exactly why the exception has to be
    checked before it. Matching is looser than anywhere else here because a
    board spells them the way a reader does ("Opus 5") while the AA slug they
    are named after carries the creator ("claude-opus-5"), so a trailing match
    counts as well as an exact one. The looseness is bounded by the list: it is
    a handful of hand-written names, not a rule let loose on the index.
    """
    key = normalize(name)
    if not key:
        return None
    for slug in load_reference_slugs():
        ref = normalize(slug)
        if not ref:
            continue
        if ref == key or ref.endswith(" " + key) or key.startswith(ref + " "):
            return slug
    return None


def is_closed_weights(
    name: str,
    *,
    open_weights: bool | None = None,
    weights: str | None = None,
    guard_names: Iterable[str] = (),
) -> bool:
    """Whether `name` can be skipped as a closed-weight model without asking.

    `weights` (a status) or `open_weights` (the older boolean) is the verdict
    the source published for this row, when it publishes one; it wins over the
    pooled index. A person's override (weights-overrides.json) wins over both,
    and is the one verdict honoured even for a name resembling an llm.json
    model: it was given about this very name. `guard_names` are the llm.json
    model names -- a name resembling one of those is never skipped on a
    source's word. An upcoming model is never skipped: its weights are coming.
    Callers should warm the index with open_index() first, so its fetches do
    not interleave with per-name output.
    """
    if reference_model_for(name):
        return False
    override = override_for(name)
    if override is not None:
        return override == CLOSED
    if _resembles_tracked_model(name, guard_names):
        return False
    if weights not in _RANK:
        weights = status_from_open(open_weights)
    return status(name, weights=weights) == CLOSED


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the pooled weight-status index used to skip closed models, "
        "or set a status by hand."
    )
    parser.add_argument("name", nargs="*", help="Model names to classify.")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild the index, ignoring the cache."
    )
    parser.add_argument(
        "--set",
        choices=[*STATUSES, "auto"],
        help="Record this status for every NAME in weights-overrides.json "
        "('auto' removes the override, so the sources decide again).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    args = parser.parse_args()

    if args.set:
        if not args.name:
            parser.error("--set needs at least one NAME")
        for name in args.name:
            changed = set_override(name, None if args.set == "auto" else args.set)
            print(f"{name}: {args.set}" + ("" if changed else " (unchanged)"))
        return 0

    index = status_index(refresh=args.refresh)

    if args.name:
        rows = [(name, *describe(name)) for name in args.name]
    else:
        rows = sorted((key, value, "sources") for key, value in index.items())
        rows += sorted((name, value, "override") for name, value in load_overrides().items())

    if args.format == "json":
        print(json.dumps({name: value for name, value, _ in rows}, ensure_ascii=False))
        return 0

    width = max((len(name) for name, _, _ in rows), default=len("MODEL"))
    print(f"{'MODEL':<{width}}  {'WEIGHTS':8s}  FROM")
    for name, value, origin in rows:
        print(f"{name:<{width}}  {value or 'unknown':8s}  {origin or ''}")
    counts = {s: sum(1 for _, v, _ in rows if v == s) for s in STATUSES}
    print(f"\n{len(rows)} names, " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
