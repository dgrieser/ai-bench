"""Shared terminal machinery for the type-to-search prompts in add.py and edit.py.

The prompt owns a block of lines: the input line, and the current matches drawn
under it. The block is redrawn on every keystroke, and it only ever draws
*downward* from the input line -- the cursor is parked back on that line at the
end of each draw, so a redraw never reaches a line above the prompt. (An earlier
version moved the cursor up by the full block height first, which walked the
block up the screen and overwrote whatever had been printed before it.) Every
line is truncated to the terminal width, because a match long enough to wrap
would occupy two rows and throw the same arithmetic off by one.

Tab completes to the longest prefix the remaining matches agree on, the way a
shell does: while several candidates are still possible it adds only the
characters all of them share, instead of jumping to one of them. Cycling through
the matches stays available for the fuzzy case, where the typed text is not a
prefix of anything and there is no shared prefix to grow.
"""

from __future__ import annotations

import os
import shutil
import sys

# Matches offered at once. The block is redrawn per keystroke, so this is also
# how tall it can get.
DEFAULT_LIMIT = 10


def supports_live_selector() -> bool:
    term = os.getenv("TERM", "")
    return sys.stdin.isatty() and sys.stdout.isatty() and term and term.lower() != "dumb"


def fuzzy_match(query: str, option: str) -> tuple[int, int] | None:
    """(tier, distance) for a match, or None. Substring beats subsequence."""
    haystack = option.lower()
    needle = query.lower()

    if needle in haystack:
        return (0, haystack.index(needle))

    pos = 0
    gap_score = 0
    for char in needle:
        idx = haystack.find(char, pos)
        if idx == -1:
            return None
        gap_score += idx - pos
        pos = idx + 1
    return (1, gap_score)


def find_matches(query: str, options: list[str], limit: int = DEFAULT_LIMIT) -> list[str]:
    if not query:
        return options[:limit]

    scored: list[tuple[tuple[int, int, int], str]] = []
    for option in options:
        match = fuzzy_match(query, option)
        if match is None:
            continue
        scored.append(((match[0], match[1], len(option)), option))
    scored.sort(key=lambda item: item[0])
    return [option for _, option in scored[:limit]]


def tab_completion(buffer: str, matches: list[str]) -> str | None:
    """What Tab should leave in the input line, or None to leave it untouched.

    While several matches remain, only the characters all of them share are
    added: Tab narrows the list, it never picks a candidate for the user (the
    matches are on screen, so the next character to type is visible). A single
    remaining match is completed in full, including the fuzzy case -- "k3" for
    "kimi-k3", where the typed text is not a prefix of the match at all.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0] if matches[0] != buffer else None
    prefix = os.path.commonprefix([match.lower() for match in matches])
    if len(prefix) <= len(buffer) or not prefix.startswith(buffer.lower()):
        return None
    # Case comes from the match, not from the lowercased prefix.
    return matches[0][: len(prefix)]


def fit(text: str) -> str:
    """Text truncated so it cannot wrap onto a second row."""
    width = max(2, shutil.get_terminal_size(fallback=(80, 24)).columns)
    return text if len(text) < width else text[: width - 1]


def render_selector(
    label: str,
    buffer: str,
    matches: list[str],
    sources: dict[str, str] | None = None,
) -> int:
    """Draw the input line plus its matches; leave the cursor on the input line.

    Must be called with the cursor already on the input line -- the first line of
    the block, which is where the previous call left it. Returns the block height
    for clear_selector().
    """
    prompt = fit(f"{label}: {buffer}")
    sys.stdout.write("\r\x1b[2K")
    sys.stdout.write(prompt)

    for match in matches:
        source = sources.get(match) if sources else None
        sys.stdout.write("\r\n\x1b[2K")
        sys.stdout.write(fit(f"  {match}  ({source})" if source else f"  {match}"))

    # Erase what a taller previous block left below, then step back onto the
    # input line so the next redraw starts where this one did.
    sys.stdout.write("\x1b[J")
    if matches:
        sys.stdout.write(f"\x1b[{len(matches)}F")
        sys.stdout.write("\r" + prompt)
    sys.stdout.flush()
    return 1 + len(matches)


def clear_selector(lines_drawn: int) -> None:
    """Blank the block and leave the cursor on its first line, ready to print."""
    if not lines_drawn:
        return

    sys.stdout.write("\r\x1b[2K")
    for _ in range(lines_drawn - 1):
        sys.stdout.write("\r\n\x1b[2K")
    if lines_drawn > 1:
        sys.stdout.write(f"\x1b[{lines_drawn - 1}F")
    sys.stdout.write("\r")
    sys.stdout.flush()
