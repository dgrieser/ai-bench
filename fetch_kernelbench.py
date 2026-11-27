#!/usr/bin/env python3
"""
Fetch KernelBench scores from https://kernelbench.com

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

The site publishes decks of problems, each on one or more GPUs. Two are read,
each into its own column, both on the RTX PRO 6000 the site calls canonical:

  * **Mega** (``kernelbench_mega``, the homepage) -- one problem today, a
    whole-model Kimi-Linear decode megakernel, every cell scored as its
    speedup over the torch baseline ("36.66x vs baseline"). The column is the
    mean speedup over the full deck, a failed, flagged or missing cell
    counting as zero. An absolute number in one unit, so it moves only when a
    model's own kernel does.
  * **Hard** (``kernelbench_hard``, /hard) -- six problems (FP8 GEMM, KDA
    CUTLASS, paged attention, top-k bitonic, Sonic MoE SwiGLU, W4A16 GEMM),
    every one scored as the fraction of the board's roofline the submitted
    kernel reaches. Retired from the homepage on 2026-11-23 ("superseded by
    KernelBench Mega and KernelBench CUDA, no longer updated for new models")
    and kept at /hard as a frozen archive, which is what is read now. The
    archive lists fewer rows than the live deck did; update.py never deletes a
    stored score, so the models it dropped keep the values read from the
    homepage before the retirement.

Not read:

  * **CUDA** -- three problems in two units: fraction of roofline for the
    fused MoE and MegaQwen decode, a latency-derived speedup for DeepSeek NSA
    ("0.072 ms geomean latency"). Cells in different units only average as
    the site's share of best, which moves every model whenever a new one
    leads, so the deck has no absolute score to store -- the reason it lost to
    Hard when the column was first admitted.
  * **Multi** -- announced, no GPU and no rows.
  * the H100 and B200 boards of any deck -- another GPU is another roofline.

**Hard's score is the mean fraction of roofline over the full deck -- an
absolute number, not the site's "share of best".** Because every Hard cell is
in the same unit, the cells can be averaged as they are and the mean reported
as a percentage of peak. That is the quantity vendors publish for "KernelBench
Hard" (MiniMax: "operator TFLOPs relative to theoretical peak of current
hardware ... average of [the] questions"), so a lab's claim lands on the
column's own scale. The share of best is reported beside every row of both
decks as ``share`` and never stored.

Every passing cell's caption is checked for its deck's unit ("% of roofline",
"x vs baseline") before its number is used, so a deck that changes unit is
refused rather than averaged.

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
from dataclasses import dataclass
from typing import Any

import _source_pages
from _fetch_checks import check_fractions, check_percentages, require_rows


SITE_URL = "https://kernelbench.com"
# The homepage carries every active deck's cells, for every GPU, in its flight
# payload; the retired Hard deck has its own archive page in the same shape.
LEADERBOARD_URL = f"{SITE_URL}/"
HARD_URL = f"{SITE_URL}/hard"

# The GPU every column reads, the one the site calls canonical.
GPU = "rtxpro6000"

# Cell units. The caption on a passing cell names the unit ("21.80% of
# roofline", "36.66× vs baseline"); a cell whose caption does not is refused.
ROOFLINE = "roofline"
SPEEDUP = "speedup"
_UNIT_CAPTIONS = {ROOFLINE: "of roofline", SPEEDUP: "× vs baseline"}

HARD_KEY = "kernelbench_hard"
MEGA_KEY = "kernelbench_mega"


@dataclass(frozen=True)
class Board:
    """One deck on one GPU, read into one llm.json column."""

    key: str
    url: str
    deck: str
    unit: str
    gpu: str = GPU


# The live deck first: a name on both boards is linked to the one still updated.
BOARDS: tuple[Board, ...] = (
    Board(MEGA_KEY, LEADERBOARD_URL, "mega", SPEEDUP),
    Board(HARD_KEY, HARD_URL, "hard", ROOFLINE),
)
BOARDS_BY_KEY = {board.key: board for board in BOARDS}
KEYS = tuple(BOARDS_BY_KEY)

# A mean speedup this large is a board that changed what it reports, not a kernel.
MAX_SPEEDUP = 1000.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)
_DECKS_RE = re.compile(r'"decks":\s*\[')
_DECODER = json.JSONDecoder()
_NO_DECKS = 'No "decks" array'

# The one cell kind that counts. "fail", "numerics", "no_kernel" and "hack"
# score zero, and so does a cell the board does not print a number for.
PASS = "pass"


def source_url(key: str) -> str:
    """The page a column's scores are read from, and cite."""
    return BOARDS_BY_KEY[key].url


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


def extract_decks(html: str, url: str = LEADERBOARD_URL) -> list[dict]:
    """The page's "decks" array.

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
            f"{_NO_DECKS} in the flight payload of {url} -- the page "
            "layout changed; refusing to report an empty leaderboard."
        )
    if len(found) > 1:
        raise ValueError(
            f'{url} carries {len(found)} different "decks" arrays; '
            "refusing to guess which one is the board."
        )
    return [deck for deck in found[0] if isinstance(deck, dict)]


def fetch_decks(url: str, attempts: int = 3, delay: float = 3.0) -> list[dict]:
    """A page's decks, re-fetched when the page came back without them.

    Vercel sometimes answers with a bot checkpoint instead of the page; that
    page has no flight payload, and a later request usually gets through. A
    page that still has no decks after the last attempt raises.
    """
    for attempt in range(1, attempts + 1):
        try:
            return extract_decks(fetch_html(url), url)
        except ValueError as exc:
            if attempt == attempts or _NO_DECKS not in str(exc):
                raise
            wait = delay * attempt
            print(
                f"  {url}: no decks on attempt {attempt}/{attempts}; retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def select_board(decks: list[dict], deck: str, gpu: str = GPU, url: str = LEADERBOARD_URL) -> dict:
    """One deck's board on one GPU: its problem list and its rows."""
    matches = [d for d in decks if d.get("key") == deck]
    if len(matches) != 1:
        raise ValueError(
            f"{url} names deck {deck!r} {len(matches)} times "
            f"(decks: {[d.get('key') for d in decks]}); refusing to guess."
        )
    by_gpu = matches[0].get("byGpu")
    board = by_gpu.get(gpu) if isinstance(by_gpu, dict) else None
    if not isinstance(board, dict):
        raise ValueError(
            f"Deck {deck!r} on {url} has no {gpu!r} board -- the layout "
            "or the canonical GPU changed; refusing to read another GPU's numbers."
        )
    problems = board.get("problems")
    rows = board.get("rows")
    if not isinstance(problems, list) or not problems or not isinstance(rows, list):
        raise ValueError(
            f"Deck {deck!r} on {gpu!r} at {url} has no problem list or "
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


def _passed_cells(board: dict) -> list[dict]:
    return [
        cell
        for row in board["rows"] if isinstance(row, dict)
        for cell in _cells(row).values() if _passed_score(cell) is not None
    ]


def check_units(board: dict, spec: Board) -> None:
    """Raise unless every passing cell's caption names the deck's unit.

    A deck that mixes units -- CUDA's NSA problem reports a latency beside two
    fractions of roofline -- cannot be averaged, and a deck that switches unit
    would otherwise be read on the old one.
    """
    caption = _UNIT_CAPTIONS[spec.unit]
    bad = sorted({
        str(cell.get("title"))
        for cell in _passed_cells(board)
        if caption not in str(cell.get("title") or "")
    })
    if bad:
        raise ValueError(
            f"Deck {spec.deck!r} on {spec.gpu!r} at {spec.url}: passing cells are expected "
            f"in {spec.unit} ({caption!r}) but read {bad[:3]} -- the deck's unit changed; "
            "refusing to average them."
        )


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


def mean_cell(row: dict, problems: list[str]) -> float:
    """Mean audited-pass score over the full deck, in the deck's own unit.

    Every cell of a read deck is in one unit (check_units), so the cells
    average as they are; a cell that is not an audited pass counts as zero.
    """
    cells = _cells(row)
    total = 0.0
    for problem in problems:
        score = _passed_score(cells.get(problem))
        if score is not None:
            total += score
    return total / len(problems)


def fraction_of_roofline(row: dict, problems: list[str]) -> float:
    """Mean fraction of the roofline over the full deck, as a percentage."""
    return 100.0 * mean_cell(row, problems)


def read_rows(board: dict, spec: Board) -> list[dict]:
    problems = [
        p["id"] for p in board["problems"] if isinstance(p, dict) and isinstance(p.get("id"), str)
    ]
    if not problems:
        raise ValueError(
            f"Deck {spec.deck!r} on {spec.gpu!r} names no problems -- the layout changed."
        )
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
        score = (
            fraction_of_roofline(row, problems) if spec.unit == ROOFLINE
            else mean_cell(row, problems)
        )
        rows.append(
            {
                "benchmark": spec.key,
                "model": name.strip(),
                "site_slug": row.get("slug") if isinstance(row.get("slug"), str) else None,
                "harness": harness,
                "effort": effort,
                "score": round(score, 2),
                "share": round(share_of_best(row, problems, best), 2),
                "passed": sum(_passed_score(cells.get(p)) is not None for p in problems),
                "problems": len(problems),
                "flagged": row.get("flagged") if isinstance(row.get("flagged"), int) else None,
                "deck": spec.deck,
                "gpu": spec.gpu,
            }
        )
    return rows


def read_board(decks: list[dict], spec: Board) -> list[dict]:
    """One board's rows, after the unit and scale checks."""
    board = select_board(decks, spec.deck, spec.gpu, spec.url)
    check_units(board, spec)
    if spec.unit == ROOFLINE:
        check_fractions(
            (cell.get("score") for cell in _passed_cells(board)),
            spec.url,
            f"{spec.deck} cell score (fraction of roofline)",
        )
    scores = read_rows(board, spec)
    print(
        f"  parsed {len(scores)} KernelBench {spec.deck} row(s) on {spec.gpu} "
        f"over {len(board['problems'])} problem(s)",
        file=sys.stderr,
    )
    require_rows(scores, spec.url)
    check_percentages(scores, spec.url, high=100.0 if spec.unit == ROOFLINE else MAX_SPEEDUP)
    return scores


def get_scores(keys: tuple[str, ...] = KEYS) -> list[dict]:
    """Return a list of score dicts, every read board's rows in BOARDS order.

    Keys: benchmark (the llm.json column), model (the board's display name),
    site_slug, harness, effort, score (Hard: mean fraction of roofline over the
    full deck, 0-100; Mega: mean speedup over the torch baseline, x), share
    (mean share of each problem's best, 0-100, the site's bars; reported, never
    stored), passed, problems, flagged (audited sessions the reward-hack review
    flagged), deck, gpu.
    """
    pages: dict[str, list[dict]] = {}
    scores: list[dict] = []
    for spec in BOARDS:
        if spec.key not in keys:
            continue
        if spec.url not in pages:
            print(f"Fetching {spec.url} ...", file=sys.stderr)
            pages[spec.url] = fetch_decks(spec.url)
        scores.extend(read_board(pages[spec.url], spec))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch kernelbench.com scores (Mega and the archived Hard deck, RTX PRO 6000)."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names", "pages"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores()

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "pages":
        _source_pages.emit(scores, lambda e: source_url(e["benchmark"]))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max([len("HARNESS"), *(len(e.get("harness") or "") for e in scores)])
        fmt = f"{{:<{width}}}  {{:>9}}  {{:>6}}  {{:>6}}  {{:<{harness_width}}}  {{}}"
        for spec in BOARDS:
            print(f"\n{spec.key} ({spec.deck} deck, {spec.gpu}, {spec.url})")
            print(fmt.format("MODEL", spec.unit.upper(), "SHARE", "PASSED", "HARNESS", "EFFORT"))
            board_rows = [e for e in scores if e["benchmark"] == spec.key]
            for entry in sorted(board_rows, key=lambda e: -e["score"]):
                value = (
                    f"{entry['score']:.1f}%" if spec.unit == ROOFLINE
                    else f"{entry['score']:.2f}x"
                )
                print(
                    fmt.format(
                        entry["model"],
                        value,
                        f"{entry['share']:.1f}",
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
