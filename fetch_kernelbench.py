#!/usr/bin/env python3
"""
Fetch KernelBench CUDA scores from https://kernelbench.com

kernelbench.com (Elliot Arledge, 2026) is an agentic GPU-kernel board: a coding
agent -- Claude Code, Codex, an OpenRouter harness, whichever the row names --
gets a reference PyTorch module and an unlimited budget to replace it with a
faster kernel on real hardware, and every published cell is audited by hand for
reward hacking (a correct-looking kernel that skips the work, reads the answer,
or times the wrong thing). It borrows its name and its problem style from
Stanford's KernelBench and says on every page that it is "not affiliated with
Stanford KernelBench". Stanford's own board
(scalingintelligence.stanford.edu/KernelBenchLeaderboard) stopped at the 2024
models -- gpt-4o, claude-3.5-sonnet, llama-3.1-405b -- so it scores nothing in
llm.json and is not read.

The site publishes decks of problems, each on one or more GPUs:

  * **Mega**  one problem, a whole-model decode megakernel, scored as speedup
              over the torch baseline.
  * **CUDA**  four problems (GLM-5.2 fused MoE, DeepSeek NSA, MegaQwen decode,
              Grid + MinGRU), each scored on its own native scale -- fraction of
              the board's roofline for most, inverse latency for NSA.
  * **Hard**  six problems (FP8 GEMM, KDA, paged attention, ...), the same way.
  * **Multi** announced, empty.

**The score is the CUDA deck on the RTX PRO 6000, as the site's own "share of
best" -- and that is a choice this file has to keep making.** The native cell
scores cannot be added across problems (35x, 10% of roofline and 0.09 ms are
three different units), so the site normalises each cell by the best audited
pass on that problem and averages over the *full* deck, a failed, hacked or
missing cell counting as zero. That per-board mean is what its bars draw, and
the /models headline is the mean of those bars over whichever decks a model
was run on. The headline is not stored: Claude Opus 5 has no Mega cell and
GPT-5.6 Sol no CUDA row, so two headlines can average different decks. One deck
on one GPU holds the task set fixed, and CUDA is the deck nearly every model
was run on -- 22 of 23 -- where Hard has 11 and Mega is a single problem.
The RTX PRO 6000 is the board the site calls canonical; the CUDA deck runs on
no other GPU today.

Recomputed here from the cells the homepage ships, this reproduces the site's
own /models headline to the tenth for every model run on Mega and CUDA alone
(Hard's normalisation does not reproduce exactly from the shipped cells, which
is a second reason it is not read).

A share of the best moves when the best moves: a new leader on a problem lowers
every other model's number on that problem. The column is read as what it is --
how close a model came to the best kernel anyone has published for these four
problems -- and a refresh that moves the whole field at once is that, not a
regression.

Rows name the harness the run used and, where it did, an effort ("claude ·
max", "codex · xhigh"); both are reported beside the published model name and
neither is part of it. The board shows one row per model.
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


SITE_URL = "https://kernelbench.com"
# The homepage carries every deck's cells, for every GPU, in its flight payload.
LEADERBOARD_URL = f"{SITE_URL}/"

# The deck the column reads, and the GPU it reads it on.
DECK = "cuda"
GPU = "rtxpro6000"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)
_DECKS_RE = re.compile(r'"decks":\s*\[')
_DECODER = json.JSONDecoder()

# The one cell kind that counts. "fail", "numerics", "no_kernel" and "hack"
# score zero, and so does a cell the board does not print a number for.
PASS = "pass"


def fetch_html(url: str = LEADERBOARD_URL, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
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


def decode_flight(html: str) -> str:
    """Concatenate the decoded self.__next_f flight chunks into one string."""
    decoded = ""
    for chunk in _PUSH_RE.findall(html):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return decoded


def extract_decks(html: str) -> list[dict]:
    """The homepage's "decks" array.

    Every `"decks":[...]` in the payload is decoded with a real JSON decoder;
    the page may repeat it, but two *different* arrays cannot be told apart and
    raise rather than letting whichever comes first win.
    """
    decoded = decode_flight(html)
    found: list[list] = []
    for match in _DECKS_RE.finditer(decoded):
        try:
            value, _ = _DECODER.raw_decode(decoded, match.end() - 1)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value not in found:
            found.append(value)
    if not found:
        raise ValueError(
            f'No "decks" array in the flight payload of {LEADERBOARD_URL} -- the page '
            "layout changed; refusing to report an empty leaderboard."
        )
    if len(found) > 1:
        raise ValueError(
            f'{LEADERBOARD_URL} carries {len(found)} different "decks" arrays; '
            "refusing to guess which one is the board."
        )
    return [deck for deck in found[0] if isinstance(deck, dict)]


def select_board(decks: list[dict], deck: str = DECK, gpu: str = GPU) -> dict:
    """One deck's board on one GPU: its problem list and its rows."""
    matches = [d for d in decks if d.get("key") == deck]
    if len(matches) != 1:
        raise ValueError(
            f"{LEADERBOARD_URL} names deck {deck!r} {len(matches)} times "
            f"(decks: {[d.get('key') for d in decks]}); refusing to guess."
        )
    by_gpu = matches[0].get("byGpu")
    board = by_gpu.get(gpu) if isinstance(by_gpu, dict) else None
    if not isinstance(board, dict):
        raise ValueError(
            f"Deck {deck!r} on {LEADERBOARD_URL} has no {gpu!r} board -- the layout "
            "or the canonical GPU changed; refusing to read another GPU's numbers."
        )
    problems = board.get("problems")
    rows = board.get("rows")
    if not isinstance(problems, list) or not problems or not isinstance(rows, list):
        raise ValueError(
            f"Deck {deck!r} on {gpu!r} at {LEADERBOARD_URL} has no problem list or "
            "no rows -- the layout changed."
        )
    return board


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _cells(row: dict) -> dict[str, dict]:
    chips = row.get("chips")
    if not isinstance(chips, list):
        return {}
    return {
        chip["problem"]: chip
        for chip in chips
        if isinstance(chip, dict) and isinstance(chip.get("problem"), str)
    }


def _passed_score(cell: dict | None) -> float | None:
    """A cell's native score if it is an audited pass, else None."""
    if not cell or cell.get("kind") != PASS:
        return None
    score = _number(cell.get("score"))
    return score if score is not None and score >= 0 else None


def best_per_problem(board: dict) -> dict[str, float]:
    """Problem id -> the best audited pass on it, the 1.00 each cell is read against."""
    best: dict[str, float] = {}
    for row in board["rows"]:
        if not isinstance(row, dict):
            continue
        for problem, cell in _cells(row).items():
            score = _passed_score(cell)
            if score is not None and score > best.get(problem, 0.0):
                best[problem] = score
    return best


def share_of_best(row: dict, problems: list[str], best: dict[str, float]) -> float:
    """Mean share of each problem's best, over the full deck, as a percentage.

    A cell that is not an audited pass counts as zero, and so does a problem
    nobody has passed -- the site's own rule, "fail/invalid/missing cells
    count as 0".
    """
    cells = _cells(row)
    total = 0.0
    for problem in problems:
        score = _passed_score(cells.get(problem))
        if score is not None and best.get(problem):
            total += score / best[problem]
    return 100.0 * total / len(problems)


def _harness_and_effort(subtitle: Any) -> tuple[str | None, str | None]:
    """'claude · max' -> ('claude', 'max'); 'or-fable' -> ('or-fable', None)."""
    if not isinstance(subtitle, str):
        return None, None
    parts = [part.strip() for part in subtitle.split("·")]
    harness = parts[0] or None
    effort = parts[1] if len(parts) > 1 and parts[1] else None
    return harness, effort


def read_rows(board: dict, deck: str = DECK, gpu: str = GPU) -> list[dict]:
    problems = [
        p["id"] for p in board["problems"] if isinstance(p, dict) and isinstance(p.get("id"), str)
    ]
    if not problems:
        raise ValueError(f"Deck {deck!r} on {gpu!r} names no problems -- the layout changed.")
    best = best_per_problem(board)

    rows: list[dict] = []
    for row in board["rows"]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        harness, effort = _harness_and_effort(row.get("subtitle"))
        cells = _cells(row)
        rows.append(
            {
                "model": name.strip(),
                "site_slug": row.get("slug") if isinstance(row.get("slug"), str) else None,
                "harness": harness,
                "effort": effort,
                "score": round(share_of_best(row, problems, best), 2),
                "passed": sum(_passed_score(cells.get(p)) is not None for p in problems),
                "problems": len(problems),
                "flagged": row.get("flagged") if isinstance(row.get("flagged"), int) else None,
                "deck": deck,
                "gpu": gpu,
            }
        )
    return rows


def get_scores(deck: str = DECK, gpu: str = GPU) -> list[dict]:
    """Return a list of score dicts for one kernelbench.com board.

    Keys: model (the board's display name), site_slug, harness, effort, score
    (mean share of each problem's best, 0-100), passed, problems, flagged
    (audited sessions the reward-hack review flagged), deck, gpu.
    """
    print(f"Fetching {LEADERBOARD_URL} ...", file=sys.stderr)
    board = select_board(extract_decks(fetch_html()), deck, gpu)
    scores = read_rows(board, deck, gpu)
    print(
        f"  parsed {len(scores)} KernelBench {deck} row(s) on {gpu} "
        f"over {len(board['problems'])} problem(s)",
        file=sys.stderr,
    )
    require_rows(scores, LEADERBOARD_URL)
    check_percentages(scores, LEADERBOARD_URL)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch kernelbench.com scores (CUDA deck, RTX PRO 6000, share of best %)."
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
        width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max([len("HARNESS"), *(len(e.get("harness") or "") for e in scores)])
        fmt = f"{{:<{width}}}  {{:>6}}  {{:>6}}  {{:<{harness_width}}}  {{}}"
        print(fmt.format("MODEL", "SHARE", "PASSED", "HARNESS", "EFFORT"))
        for entry in sorted(scores, key=lambda e: -e["score"]):
            print(
                fmt.format(
                    entry["model"],
                    f"{entry['score']:.1f}",
                    f"{entry['passed']}/{entry['problems']}",
                    entry.get("harness") or "",
                    entry.get("effort") or "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(
            f"error: could not fetch the kernelbench.com leaderboard: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
