#!/usr/bin/env python3
"""
Fetch SWE-Marathon resolution rates from https://www.swe-marathon.org/

The site is a client-rendered Vite SPA: the leaderboard is not in the served
HTML and there is no JSON endpoint. The data ships inside the hashed
/assets/*.js bundle, so this script resolves the bundle from the page's
<script src> and reads it directly.

**The site publishes two boards and they are not comparable.** It ships a
"v1.0 Archive" beside a "v1.1 Current": 1.1 updated all 20 long-horizon tasks
with tighter verification and closed-internet execution, and the site states
outright that no v1.0 score is reused for the updated tasks. Its leader sits 21
points above the archive's. llm.json therefore gives each revision its own
column (``swe_marathon_1_0``, ``swe_marathon_1_1``); see _revisions.py. Every
row here names the ``revision`` it belongs to, and --revision pins one.

The two boards are stored differently, which is why reading only the obvious
one was the bug this replaces:

  * **the archive** is a plain object-literal array carrying ``pass1``, parsed
    as a JS literal (the minifier emits backtick strings, bare keys and
    ``!0``/``!1`` booleans, so json.loads cannot be used);
  * **the current board is never stored as a leaderboard at all** -- the bundle
    ships the per-task trial log (``{version:"v1.1",tasks:JSON.parse("...")}``,
    a real JSON blob) and the page aggregates it in the browser. A literal
    scan finds only the archive, so this script reproduces the aggregation:
    pass@1 is the share of a configuration's trials whose reward is a full 1,
    which is the site's own definition, grouped over every task.

Each row is one (model, scaffold, reasoning effort) configuration -- the same
model appears more than once when it was run under several harnesses (e.g. Kimi
K2.6 under both Kimi Code CLI and Terminus 2) or efforts, and the reported score
is configuration-dependent. ``score`` is the site's headline "Resolution rate
(pass@1)" in percent (binary reward: a task counts only when every verifier test
passes).

For display the site collapses a handful of models to their max-effort row.
That filter is a hardcoded model list inside the minified bundle, so it is not
reproduced here; every configuration is reported instead and the ingest's
best-run rule picks between them, the same way the other multi-configuration
sources in this repo are read. The two agree: taking the best configuration
reproduces the published number for every model on the current board.

The Oracle (held-out solution) and NOP (no actions) rows are reference bounds,
not models, and are dropped unless --include-reference is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

from _revisions import revision_label, revision_rank


URL = "https://www.swe-marathon.org/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Field that identifies the archived leaderboard array inside the bundle.
_MARKER = "pass1:"
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+\.js)"')
# The site's own name for each board it publishes: "v1.0 Archive",
# "v1.1 Current". Which revision is archived and which is current is read from
# these rather than pinned, so a 1.2 release relabels itself here.
_BOARD_LABEL_RE = re.compile(r"label:`(v[\d.]+)\s+(Archive|Current)`")
# The per-task trial log the current board is aggregated from, with the
# revision it belongs to: {version:`v1.1`,tasks:JSON.parse(`{...}`)}.
_TRIALS_RE = re.compile(r"version:`(v[\d.]+)`,tasks:JSON\.parse\(`")

# --revision value that reports every revision instead of pinning one.
ALL_REVISIONS = "all"
DEFAULT_REVISION = ALL_REVISIONS
_KEY_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class ParseError(ValueError):
    """Raised when the bundle does not hold the expected JS literal."""


def fetch_text(url: str, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries:
                raise
            wait = delay * attempt
            print(
                f"  attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def bundle_urls(page_html: str, base: str = URL) -> list[str]:
    """Absolute URLs of every <script src> .js asset on the page."""
    urls: list[str] = []
    for src in _SCRIPT_SRC_RE.findall(page_html):
        if src.startswith("http"):
            urls.append(src)
        else:
            urls.append(base.rstrip("/") + "/" + src.lstrip("/"))
    return urls


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _parse_string(text: str, i: int) -> tuple[str, int]:
    quote = text[i]
    i += 1
    out: list[str] = []
    while i < len(text):
        c = text[i]
        if c == "\\":
            nxt = text[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
            i += 2
            continue
        if c == quote:
            return "".join(out), i + 1
        if quote == "`" and c == "$" and text[i + 1 : i + 2] == "{":
            # Interpolated template: not data, refuse rather than guess.
            raise ParseError("template interpolation in string literal")
        out.append(c)
        i += 1
    raise ParseError("unterminated string literal")


def _parse_value(text: str, i: int) -> tuple[object, int]:
    i = _skip_ws(text, i)
    if i >= len(text):
        raise ParseError("unexpected end of input")
    c = text[i]

    if c == "{":
        obj: dict[str, object] = {}
        i = _skip_ws(text, i + 1)
        if text[i] == "}":
            return obj, i + 1
        while True:
            i = _skip_ws(text, i)
            if text[i] in "\"'`":
                key, i = _parse_string(text, i)
            else:
                match = _KEY_RE.match(text, i)
                if not match:
                    raise ParseError(f"bad object key at {i}")
                key, i = match.group(0), match.end()
            i = _skip_ws(text, i)
            if text[i] != ":":
                raise ParseError(f"expected ':' at {i}")
            value, i = _parse_value(text, i + 1)
            obj[key] = value
            i = _skip_ws(text, i)
            if text[i] == ",":
                i += 1
                i = _skip_ws(text, i)
                if text[i] == "}":
                    return obj, i + 1
                continue
            if text[i] == "}":
                return obj, i + 1
            raise ParseError(f"expected ',' or '}}' at {i}")

    if c == "[":
        arr: list[object] = []
        i = _skip_ws(text, i + 1)
        if text[i] == "]":
            return arr, i + 1
        while True:
            value, i = _parse_value(text, i)
            arr.append(value)
            i = _skip_ws(text, i)
            if text[i] == ",":
                i += 1
                i = _skip_ws(text, i)
                if text[i] == "]":
                    return arr, i + 1
                continue
            if text[i] == "]":
                return arr, i + 1
            raise ParseError(f"expected ',' or ']' at {i}")

    if c in "\"'`":
        return _parse_string(text, i)

    # Minified booleans: !0 -> true, !1 -> false.
    if c == "!":
        if text[i + 1] == "0":
            return True, i + 2
        if text[i + 1] == "1":
            return False, i + 2
        raise ParseError(f"unsupported '!' expression at {i}")

    for literal, value in (("true", True), ("false", False), ("null", None), ("undefined", None)):
        if text.startswith(literal, i):
            return value, i + len(literal)

    match = _NUMBER_RE.match(text, i)
    if match:
        raw = match.group(0)
        number: object = float(raw) if ("." in raw or "e" in raw or "E" in raw) else int(raw)
        return number, match.end()

    # An identifier here means the array references a variable rather than
    # holding inline data; the shape changed and guessing would be wrong.
    raise ParseError(f"unsupported token at {i}: {text[i : i + 24]!r}")


def extract_rows(bundle: str) -> list[dict]:
    """Parse the leaderboard array (the one carrying ``pass1``) out of a bundle.

    Walks back from the marker to each preceding '[' and returns the first
    candidate that parses as an array of row objects, so an extra wrapping
    array or a reordered minification does not break extraction.
    """
    marker = bundle.find(_MARKER)
    if marker == -1:
        raise ParseError(f"{_MARKER!r} not found in bundle")

    search_from = marker
    for _ in range(10):
        start = bundle.rfind("[", 0, search_from)
        if start == -1:
            break
        try:
            value, _end = _parse_value(bundle, start)
        except (ParseError, IndexError):
            search_from = start
            continue
        if isinstance(value, list):
            rows = [r for r in value if isinstance(r, dict) and "pass1" in r]
            if rows:
                return rows
        search_from = start

    raise ParseError("could not parse a leaderboard array around the pass1 marker")


def board_revisions(bundle: str) -> dict[str, str]:
    """{"archive": label, "current": label} from the names the site gives its boards.

    Empty when the bundle names neither, which is the signal to stop rather
    than guess: a row whose revision is unknown must not land in a revision's
    column.
    """
    found = {
        role.lower(): revision_label(version)
        for version, role in _BOARD_LABEL_RE.findall(bundle)
    }
    return found


def _template_literal(bundle: str, start: int) -> tuple[str, int]:
    """The contents of the backtick string opening at ``start``, and the index after it.

    Escape-aware, so a backtick inside the payload does not end it early.
    """
    out: list[str] = []
    i = start + 1
    while i < len(bundle):
        c = bundle[i]
        if c == "\\":
            nxt = bundle[i + 1]
            # Template-literal escapes the bundler adds on top of the JSON.
            out.append(nxt if nxt in "\\`$" else "\\" + nxt)
            i += 2
            continue
        if c == "`":
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise ParseError("unterminated template literal")


def extract_trial_logs(bundle: str) -> list[tuple[str, dict]]:
    """(revision label, {task: {configs: [...]}}) for every trial log in the bundle.

    The current board is not stored as a leaderboard; this is the data the page
    builds it from, and it names its own revision.
    """
    logs: list[tuple[str, dict]] = []
    for match in _TRIALS_RE.finditer(bundle):
        raw, _end = _template_literal(bundle, match.end() - 1)
        try:
            tasks = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(f"trial log for {match.group(1)} is not JSON: {exc}") from exc
        if isinstance(tasks, dict):
            logs.append((revision_label(match.group(1)), tasks))
    return logs


def leaderboard_from_trials(tasks: dict) -> list[dict]:
    """Aggregate a trial log into one row per (model, scaffold, reasoning effort).

    pass@1 is the share of the configuration's trials that scored a full reward,
    pooled over every task -- the site's own definition. A configuration with no
    stated effort is the model's only setting; the site calls that one "max".
    """
    grouped: dict[tuple[str, str, str], dict] = {}
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        for config in task.get("configs") or []:
            if not isinstance(config, dict):
                continue
            model = config.get("model")
            scaffold = config.get("agent")
            if not isinstance(model, str) or not model:
                continue
            effort = config.get("reasoningEffort") or "max"
            entry = grouped.setdefault(
                (model, scaffold or "", effort),
                {"trials": [], "label": config.get("displayLabel")},
            )
            entry["trials"].extend(
                t for t in (config.get("trials") or []) if isinstance(t, dict)
            )

    rows: list[dict] = []
    for (model, scaffold, effort), entry in grouped.items():
        trials = entry["trials"]
        if not trials:
            continue
        resolved = sum(1 for t in trials if isinstance(t.get("reward"), (int, float))
                       and not isinstance(t.get("reward"), bool) and t["reward"] >= 1)
        partials = [t.get("partial") for t in trials
                    if isinstance(t.get("partial"), (int, float))
                    and not isinstance(t.get("partial"), bool)]
        costs = [t.get("costUsd") for t in trials
                 if isinstance(t.get("costUsd"), (int, float))
                 and not isinstance(t.get("costUsd"), bool)]
        tokens = [t.get("tokensRaw") for t in trials
                  if isinstance(t.get("tokensRaw"), (int, float))
                  and not isinstance(t.get("tokensRaw"), bool) and t["tokensRaw"] > 0]
        rows.append(
            {
                "model": model,
                "scaffold": scaffold or None,
                "effort": effort,
                "display_label": entry["label"],
                "score": round(100 * resolved / len(trials), 2),
                "partial_avg": round(100 * sum(partials) / len(partials), 2) if partials else None,
                "cost_avg": round(sum(costs) / len(costs), 2) if costs else None,
                "tok_avg": round(sum(tokens) / len(tokens) / 1e6, 2) if tokens else None,
                "trials": len(trials),
                "incomplete": False,
                "reference": False,
                "id": None,
                "site_rank": None,
            }
        )
    return rows


def archive_rows(bundle: str) -> list[dict]:
    """The archived board, as stored: one object-literal row per (model, scaffold)."""
    rows: list[dict] = []
    for row in extract_rows(bundle):
        name = row.get("name")
        pass1 = row.get("pass1")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(pass1, bool) or not isinstance(pass1, (int, float)):
            continue
        rows.append(
            {
                "model": name,
                "scaffold": row.get("scaffold"),
                "effort": None,
                "display_label": None,
                "score": round(float(pass1), 2),
                "partial_avg": row.get("partialAvg"),
                "cost_avg": row.get("costAvg"),
                "tok_avg": row.get("tokAvg"),
                "trials": row.get("nLoggedTrials"),
                "incomplete": bool(row.get("incomplete")),
                "reference": bool(row.get("ref")),
                "id": row.get("id"),
                "site_rank": row.get("rank"),
            }
        )
    return rows


def get_scores(
    include_reference: bool = False, revision: str = DEFAULT_REVISION
) -> list[dict]:
    """Return a list of score dicts, one per (revision, model, scaffold, effort).

    Keys: model, revision, scaffold, effort, score (pass@1 %), partial_avg,
    cost_avg, tok_avg, trials, incomplete, reference, id, site_rank, rank.

    revision="all" (the default) reports both published boards; any other value
    pins one by label ("1.0"). Rows are ranked within their own revision, since
    ranking across revisions would compare two different task sets.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    page = fetch_text(URL)

    bundle = None
    for asset in bundle_urls(page):
        print(f"Fetching {asset} ...", file=sys.stderr)
        text = fetch_text(asset)
        if _MARKER in text:
            bundle = text
            break
    if bundle is None:
        raise ParseError(f"no page script contained {_MARKER!r}")

    print("Parsing leaderboard data...", file=sys.stderr)
    boards = board_revisions(bundle)
    if not boards:
        raise ParseError("bundle names no Archive/Current board revisions")

    by_revision: dict[str, list[dict]] = {}
    for label, tasks in extract_trial_logs(bundle):
        by_revision.setdefault(label, []).extend(leaderboard_from_trials(tasks))

    # Whatever the site archived is the board still stored as a literal; the
    # current one is rebuilt from its trial log above. Should the archive ever
    # ship a trial log of its own, the log wins and the literal is left alone
    # rather than counted twice.
    archived = boards.get("archive")
    if archived is None:
        print("  warning: bundle names no archived board; the stored literal is skipped",
              file=sys.stderr)
    elif archived in by_revision:
        print(f"  note: revision {archived} ships a trial log; stored literal ignored",
              file=sys.stderr)
    else:
        by_revision[archived] = archive_rows(bundle)

    if revision != ALL_REVISIONS:
        wanted = revision_label(revision)
        if wanted not in by_revision:
            available = ", ".join(sorted(by_revision)) or "(none)"
            raise ValueError(
                f"Revision {revision!r} not published; available: {available}"
            )
        by_revision = {wanted: by_revision[wanted]}

    results: list[dict] = []
    for label in sorted(by_revision, key=revision_rank, reverse=True):
        rows = by_revision[label]
        dropped = 0
        kept: list[dict] = []
        for row in rows:
            if row["reference"] and not include_reference:
                dropped += 1
                continue
            kept.append({**row, "revision": label})
        kept.sort(key=lambda r: -r["score"])
        for i, r in enumerate(kept, 1):
            r["rank"] = i
        role = next((name for name, value in boards.items() if value == label), "?")
        print(
            f"  revision {label} ({role}): {len(kept)} row(s)"
            + (f", {dropped} reference row(s) dropped" if dropped else ""),
            file=sys.stderr,
        )
        results.extend(kept)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SWE-Marathon leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also keep the Oracle / NOP reference rows (not real models).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Pin one benchmark revision by label (1.0), or {ALL_REVISIONS!r} "
        f"to report every published board (default: {DEFAULT_REVISION}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(
        include_reference=args.include_reference, revision=args.revision
    )

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            4,
            max(len("SCAFFOLD"), max((len(e.get("scaffold") or "") for e in scores), default=0)),
            max(len("EFFORT"), max((len(e.get("effort") or "") for e in scores), default=0)),
            6,
        ]
        fmt = (
            f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  "
            f"{{:<{col_widths[3]}}}  {{:<{col_widths[4]}}}  {{:>{col_widths[5]}}}"
        )
        print(fmt.format("MODEL", "PASS@1", "REV", "SCAFFOLD", "EFFORT", "TRIALS"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    str(entry["score"]),
                    entry.get("revision") or "",
                    entry.get("scaffold") or "",
                    entry.get("effort") or "",
                    str(entry.get("trials") if entry.get("trials") is not None else ""),
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
