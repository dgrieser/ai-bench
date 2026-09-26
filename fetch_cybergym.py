#!/usr/bin/env python3
"""
Fetch CyberGym and ExploitGym scores from https://www.cybergym.io

Both boards belong to UC Berkeley RDI's cybersecurity observatory and are
rendered by the same script, ``/assets/js/leaderboard.js``, from one static
JSON file each. This reads those files directly: no HTML, no JavaScript.

  * ``cybergym``   ``/assets/data/cybergym.json``, the ``level1`` list. Level 1
                   is the one the board ranks: the agent gets the vulnerability
                   description and must write a crashing PoC. Levels 0, 2 and 3
                   carry one row each and are not read.
  * ``exploitgym`` ``/assets/data/exploitgym.json``, the ``results`` list: how
                   many of the 869 tasks a run exploited *on target* (the flag
                   captured through the intended vulnerability). The page
                   prints counts; the column is that count as a share of
                   ``instances.total``.

Most of what these files hold is not what the columns measure, and the rows say
so in fields rather than in the layout, so every filter below is on a field:

  * ``focus == "agent"`` marks a security vendor's pipeline around a model --
    Sangfor's, Alipay's, Tencent Xuanwu's. They top CyberGym at 0.87-0.99 with
    the same models that score 0.77-0.85 in a plain coding agent, so the number
    belongs to the harness. Dropped on both boards, and so is every
    ``Multi-model (...)`` row.
  * CyberGym's sort field, ``score_10``, is pass@10 on the 2025 CyberGym-Team
    rows, which carry their pass@1 beside it as ``score_x1`` -- up to 9x lower
    -- while still reporting ``trials: 1``. A row whose ``score_x1`` disagrees
    with ``score_10`` is dropped, as is any row run over more than one trial
    (Anthropic's ``trials: 30`` submissions). See
    docs/cybergym-coverage-2026-08.md.
  * ExploitGym mixes budgets on one board: the maintainers run 2 hours, the
    vendors report 6 (Z.ai's "rescaled 6h" is its API time rescaled by the
    model's tokens per second, and still a 6-hour budget). The column is the
    6-hour figure, the one every vendor quotes; a row whose ``eval_note`` names
    no 6-hour budget is dropped, and so is one run on a "selected subset".
    ``subRows`` carry the 2-hour cutoff of the same run and are never read.
  * ExploitGym ``hidden`` rows and ``version: "v0"`` rows (the retired 898-task
    set, a different denominator) are dropped.

Labels carry notes in a parenthetical -- "GPT-5.6 Sol (reasoning max)",
"Claude Opus 4.6 (Results obtained in collaboration with Anthropic)" -- which
never name the model, so ``model`` is the label without it and ``raw`` keeps
the original. The two boards share their labels, and one mapping file serves
both.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from _fetch_checks import check_percentages, require_rows


SITE_URL = "https://www.cybergym.io"
DATA_URL = SITE_URL + "/assets/data/{board}.json"

CYBERGYM_KEY = "cybergym"
EXPLOITGYM_KEY = "exploitgym"
KEYS = (CYBERGYM_KEY, EXPLOITGYM_KEY)

# The page a score is credited to, per column: what a reader opens, not the
# JSON file it renders from.
PAGE_URLS = {
    CYBERGYM_KEY: SITE_URL + "/cybergym/",
    EXPLOITGYM_KEY: SITE_URL + "/exploitgym/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_SIX_HOURS_RE = re.compile(r"\b6\s*h", re.IGNORECASE)


def source_url(key: str) -> str:
    return PAGE_URLS[key]


def data_url(key: str) -> str:
    return DATA_URL.format(board=key)


def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> Any:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
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


def normalize_model(raw: str) -> str:
    """The label without its notes: "GPT-5.6 Sol (reasoning max)" -> "GPT-5.6 Sol"."""
    return re.sub(r"\s+", " ", _PAREN_RE.sub(" ", raw)).strip()


def is_agent_row(row: dict) -> bool:
    model = str(row.get("model") or "")
    return row.get("focus") == "agent" or model.lower().startswith("multi-model")


def cybergym_rows(payload: Any) -> tuple[list[dict], dict[str, int]]:
    """Model-focused, single-trial, pass@1 Level 1 rows, and what was dropped."""
    if not isinstance(payload, dict) or not isinstance(payload.get("level1"), list):
        raise ValueError(
            f"{data_url(CYBERGYM_KEY)} has no level1 list -- the file changed shape; "
            "refusing to guess which level the board ranks."
        )
    dropped = {"agent": 0, "pass@k": 0, "trials": 0}
    rows: list[dict] = []
    for row in payload["level1"]:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        score = row.get("score_10")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if is_agent_row(row):
            dropped["agent"] += 1
            continue
        x1 = row.get("score_x1")
        if isinstance(x1, (int, float)) and abs(float(x1) - float(score)) > 1e-9:
            dropped["pass@k"] += 1
            continue
        trials = row.get("trials", 1)
        if trials not in (1, None):
            dropped["trials"] += 1
            continue
        rows.append(
            {
                "benchmark": CYBERGYM_KEY,
                "model": normalize_model(row["model"]),
                "raw": row["model"],
                "agent": row.get("agent"),
                "reported_by": row.get("source"),
                "date": row.get("date"),
                "score": round(float(score) * 100, 2),
            }
        )
    return rows, dropped


def exploitgym_rows(payload: Any) -> tuple[list[dict], dict[str, int]]:
    """Model-focused, full-set, 6-hour ExploitGym rows, and what was dropped."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(
            f"{data_url(EXPLOITGYM_KEY)} has no results list -- the file changed shape."
        )
    total = (payload.get("instances") or {}).get("total")
    if not isinstance(total, int) or total <= 0:
        raise ValueError(
            f"{data_url(EXPLOITGYM_KEY)} states no instances.total -- refusing to "
            "turn exploit counts into percentages without a denominator."
        )
    dropped = {"agent": 0, "hidden": 0, "v0": 0, "subset": 0, "budget": 0}
    rows: list[dict] = []
    for row in payload["results"]:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        solved = row.get("on_target")
        if not isinstance(solved, int) or isinstance(solved, bool):
            continue
        note = str(row.get("eval_note") or "")
        if row.get("hidden"):
            dropped["hidden"] += 1
            continue
        if is_agent_row(row):
            dropped["agent"] += 1
            continue
        if row.get("version") == "v0":
            dropped["v0"] += 1
            continue
        if "subset" in note.lower():
            dropped["subset"] += 1
            continue
        if not _SIX_HOURS_RE.search(note):
            dropped["budget"] += 1
            continue
        rows.append(
            {
                "benchmark": EXPLOITGYM_KEY,
                "model": normalize_model(row["model"]),
                "raw": row["model"],
                "agent": row.get("agent"),
                "reported_by": row.get("source"),
                "date": row.get("date"),
                "eval_note": note,
                "solved": solved,
                "total": total,
                "score": round(solved / total * 100, 2),
            }
        )
    return rows, dropped


READERS = {CYBERGYM_KEY: cybergym_rows, EXPLOITGYM_KEY: exploitgym_rows}


def get_scores(keys: tuple[str, ...] = KEYS, fetch=fetch_json) -> list[dict]:
    """Return one dict per kept row of each board.

    Keys: benchmark (the llm.json column), model (label without notes), raw,
    agent, reported_by, date, score (a percentage), plus eval_note, solved and
    total on ExploitGym rows.
    """
    results: list[dict] = []
    for key in keys:
        url = data_url(key)
        print(f"Fetching {url} ...", file=sys.stderr)
        rows, dropped = READERS[key](fetch(url))
        require_rows(rows, url)
        check_percentages(rows, url)
        skipped = ", ".join(f"{n} {why}" for why, n in dropped.items() if n)
        print(
            f"  {key}: kept {len(rows)} rows" + (f"; dropped {skipped}" if skipped else ""),
            file=sys.stderr,
        )
        rows.sort(key=lambda r: -r["score"])
        results.extend(rows)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CyberGym and ExploitGym scores from cybergym.io."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores()
    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        width = max((len(e["model"]) for e in scores), default=5)
        print(f"{'BOARD':<10}  {'MODEL':<{width}}  {'SCORE':>6}  AGENT")
        for e in scores:
            print(f"{e['benchmark']:<10}  {e['model']:<{width}}  {e['score']:>6}  {e.get('agent') or ''}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the cybergym.io boards: {exc}", file=sys.stderr)
        raise SystemExit(1)
