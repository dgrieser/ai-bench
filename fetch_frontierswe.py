#!/usr/bin/env python3
"""
Fetch FrontierSWE scores from https://www.frontierswe.com/

The site is a Next.js app that embeds each leaderboard as React Server Component
flight data in self.__next_f.push() script tags. This script concatenates the
decoded flight chunks and extracts the "entries" object without a headless
browser.

FrontierSWE has published two leaderboards, and they are not one benchmark
measured twice:

  * **V2** -- the root page. 34 hand-crafted tasks, five trials each under a
    20-hour budget, scored as a mean@5 task percentage. Its `entries` are keyed
    by score mode ("abs") first and by trial aggregation ("mean", "best",
    "worst") second.
  * **V1** -- /v1, "preserved as published". 17 tasks ranked by average
    per-task rank and by *dominance*, the win rate against a random opponent on
    a random task. Its `entries` are keyed by aggregation alone, the shape the
    site shipped before the score mode existed.

llm.json therefore gives each its own column (`frontierswe_2_0`,
`frontierswe_1_0`) rather than blending them, the way DeepSWE, FrontierCode and
SWE-Marathon are already split -- see _revisions.py. Every revision is reported,
newest first, and every row names the `revision` it came from; a model on both
boards appears twice, once per revision, and the models retired before V2
(Kimi K2.5, GLM 5.1, DeepSeek V4 Pro and friends) appear under V1 alone.
--revision pins a single one.

The number each revision contributes is the one its own board ranks by: V2's
`overall` percentage, and V1's `dominance` as a percentage (x100), which is the
only V1 quantity on a higher-is-better 0-100 scale. A V1 row's average per-task
rank is reported as `mean_rank` and is deliberately not a score -- it is a
position in a 17-model field, so it moves when the field moves and lower is
better. The two scales are not comparable with each other either, which is the
second half of why the columns are separate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

from _revisions import revision_label, revision_rank


URL = "https://www.frontierswe.com/"
# V1 is still served, unchanged, at its own path.
V1_URL = "https://www.frontierswe.com/v1"

# Revision label -> the page publishing it. Each board is its own source page:
# a 1.0 score cites /v1, because the root board a reader opens does not carry
# that number any more.
BOARD_URLS = {"2.0": URL, "1.0": V1_URL}

# The revision the root page serves, and the one whose payload shape is nested
# under a score mode.
CURRENT_REVISION = "2.0"

# --revision value that reports every revision instead of pinning one.
ALL_REVISIONS = "all"
DEFAULT_REVISION = ALL_REVISIONS

# V2 entries are keyed by score mode first: "abs" is the site's own percentage
# scale. V1 predates the mode and keys its views directly.
SCORE_MODE = "abs"
# ... and by trial aggregation second. The public leaderboard prints mean@5 and
# spans its whiskers from worst@5 to best@5; V1 publishes mean and best only.
GROUPS = ("mean", "best", "worst")
DEFAULT_GROUP = GROUPS[0]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_entries(html: str) -> dict:
    """Extract the leaderboard 'entries' object from the Next.js RSC flight data."""
    decoded = ""
    for chunk in _PUSH_RE.findall(html):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue

    key_idx = decoded.find('"entries":')
    if key_idx == -1:
        raise ValueError("Could not find 'entries' in page flight data")

    obj_start = decoded.index("{", key_idx)
    depth, end = 0, obj_start
    for i, c in enumerate(decoded[obj_start:], obj_start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    return json.loads(decoded[obj_start : end + 1])


def revisions_newest_first() -> list[str]:
    """Every revision this reader knows a board for, newest first."""
    return sorted(BOARD_URLS, key=revision_rank, reverse=True)


def select_group(entries: dict, group: str, revision: str = CURRENT_REVISION) -> list:
    """One leaderboard view out of one board's flight payload.

    The two levels are read separately so a failure names the one that moved,
    and each shape is read only for the board that publishes it: V2's views sit
    under a score mode, V1's do not. Reading either payload as the other would
    silently mix two scales in one column, so both directions raise -- naming
    the revision, because that is the board that has to be looked at.
    """
    board = entries
    if revision == CURRENT_REVISION:
        board = entries.get(SCORE_MODE)
        if not isinstance(board, dict):
            raise ValueError(
                f"Score mode {SCORE_MODE!r} not found in the V{revision} board; "
                f"available: {sorted(entries)}"
            )
    elif isinstance(entries.get(SCORE_MODE), dict):
        raise ValueError(
            f"The V{revision} board now nests its views under {SCORE_MODE!r}; "
            f"its scale has to be re-checked before it is read as V{revision}"
        )

    items = board.get(group)
    if not isinstance(items, list):
        raise ValueError(
            f"Group {group!r} not found in the V{revision} board; available: {sorted(board)}"
        )
    return items


def v2_row(item: dict) -> dict | None:
    """One V2 leaderboard row: the mean@5 task percentage the site prints."""
    overall = item.get("overall")
    if not isinstance(overall, (int, float)) or isinstance(overall, bool):
        return None
    return {
        "harness": item.get("harness"),
        "overall": overall,
        "score": round(float(overall), 1),
    }


def v1_row(item: dict) -> dict | None:
    """One V1 leaderboard row: dominance as a percentage.

    `overall` is V1's average per-task rank, not a percentage, so it is carried
    as `mean_rank` and never used as the score -- lower is better there, and a
    rank is a position in that board's field rather than a measurement of the
    model.
    """
    dominance = item.get("dominance")
    if not isinstance(dominance, (int, float)) or isinstance(dominance, bool):
        return None
    return {
        "harness": item.get("harness"),
        "dominance": dominance,
        "mean_rank": item.get("overall"),
        "score": round(float(dominance) * 100, 1),
    }


# Revision label -> the reader for that board's rows. A board's metric is part
# of the board, not a switch: V1 has no task percentage to report and V2 has no
# pairwise win rate.
ROW_READERS = {"2.0": v2_row, "1.0": v1_row}


def revision_rows(revision: str, entries: dict, group: str) -> list[dict]:
    """Ranked rows of one board, best first."""
    read = ROW_READERS[revision]
    rows: list[dict] = []
    for item in select_group(entries, group, revision):
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        if not isinstance(model, str) or not model:
            continue
        row = read(item)
        if row is None:
            continue
        rows.append({"model": model, "revision": revision, **row})

    rows.sort(key=lambda r: -r["score"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def resolve_revision(revision: str) -> list[str]:
    """The revisions to report, newest first, from a --revision value."""
    if revision == ALL_REVISIONS:
        return revisions_newest_first()
    label = revision_label(revision)
    if label not in BOARD_URLS:
        available = ", ".join(revisions_newest_first())
        raise ValueError(f"Revision {revision!r} has no board; available: {available}")
    return [label]


def get_scores(
    group: str = DEFAULT_GROUP, revision: str = DEFAULT_REVISION
) -> list[dict]:
    """Return a list of dicts: model, revision, harness, score, rank, ...

    revision="all" (the default) reports every revision, newest first, without
    merging them: each row carries the revision it was measured under, and a
    model published on both boards appears once per revision. Any other value
    pins that single revision, by label ("1.0") or by the site's own spelling
    ("v1"). Rows are ranked within their own revision, since ranking across
    them would compare a task percentage with a pairwise win rate.

    group selects the trial aggregation: 'mean' is the public leaderboard's
    mean@5, 'best' and 'worst' the ends of its whiskers. V1 publishes no
    'worst', so asking for one there raises rather than dropping that board.
    """
    results: list[dict] = []
    for label in resolve_revision(revision):
        url = BOARD_URLS[label]
        print(f"Fetching {url} ...", file=sys.stderr)
        entries = extract_entries(fetch_html(url))
        rows = revision_rows(label, entries, group)
        print(f"  revision {label}: {len(rows)} model(s)", file=sys.stderr)
        results.extend(rows)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch FrontierSWE leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--group",
        choices=list(GROUPS),
        default=DEFAULT_GROUP,
        help="Trial aggregation: 'mean' (the public leaderboard's mean@5), "
        f"'best' or 'worst' (default: {DEFAULT_GROUP}).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pin one benchmark revision by label (1.0) or the site's spelling "
        f"(v1), or {ALL_REVISIONS!r} to report every revision "
        f"(default: {DEFAULT_REVISION}).",
    )
    args = parser.parse_args()

    scores = get_scores(group=args.group, revision=args.revision)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for entry in scores:
            print(entry["model"])
    else:
        model_width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max([len("HARNESS"), *(len(e.get("harness") or "") for e in scores)])
        fmt = f"{{:<{model_width}}}  {{:>6}}  {{:<4}}  {{:<{harness_width}}}"
        print(fmt.format("MODEL", "SCORE", "REV", "HARNESS"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    f"{entry['score']:.1f}",
                    entry.get("revision") or "",
                    entry.get("harness") or "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
