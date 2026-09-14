#!/usr/bin/env python3
"""Tests for the padlock on a closed model's name. Run with ./test_reference_lock.py

A reference row is a closed model (see _reference.py), and llm.html marks one
two ways: its name is set in teal, and a padlock stands in front of it. The teal
is decoration — a printout, a colour-blind reader or a high-contrast theme loses
it — so the lock is the cue that actually has to be there, and it has to be there
*everywhere a model is named*, not only in the table.

Nine surfaces name a model, and they were not all marked: the table row, the
detail heading and the comparison heading carried the lock (after the name), and
the "what's new" panel, the radar legend, the two comparison tables, the
comparison picker and the table's own model filter carried nothing at all.

So llm.html composes name and lock in exactly one place, `withReferenceLock()`,
and these tests pin that: the lock leads, `referenceTag()` is never called from
anywhere else (which is what stops a surface putting it back after the name), and
every one of the nine goes through the helper. A tenth surface is a deliberate
edit here, not an oversight in the page.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

LLM_HTML = Path(__file__).resolve().parent / "llm.html"

# The renderers that put a model's name on screen. Hand-listed because there is
# no way to discover "this string is a model name" from the source -- but the
# list is checked in both directions, so a new naming surface that skips the
# helper, or a helper call that loses its surface, fails here.
NAMING_SURFACES = {
    "populateFilters",          # the table's own model filter
    "renderRecentModelName",    # the "what's new" panel
    "renderRow",                # a table row
    "renderDetailHead",         # a model's own page
    "radarLegend",              # who is who on the comparison chart
    "renderCompareHead",        # "A vs B" above a comparison
    "renderCompareMenuBody",    # the comparison's model picker
    "renderCompareSpecs",       # the comparison's details table
    "renderCompareBenchTable",  # the comparison's benchmark table
}

FUNCTION_RE = re.compile(r"\bfunction\s+([A-Za-z0-9_$]+)\s*\(")


def source() -> str:
    return LLM_HTML.read_text(encoding="utf-8")


def enclosing_function(src: str, index: int) -> str:
    """Name of the function the character at `index` sits in.

    The nearest `function NAME(` above it, which is enough here: every call
    site this file cares about is in a plain named function declaration.
    """
    names = FUNCTION_RE.findall(src[:index])
    return names[-1] if names else ""


def body_of(src: str, name: str) -> str:
    """Source of one function, signature excluded.

    The signature is dropped because it names the parameters, and a test that
    asks where `nameHtml` sits would otherwise find the argument list rather
    than the expression being checked.
    """
    start = src.index(f"function {name}(")
    return src[src.index("\n", start) : src.index("\n      }", start)]


def call_sites(src: str) -> dict[str, int]:
    """Enclosing function -> how many times it composes a name with the lock.

    The helper's own declaration matches the same text, so it is dropped: what
    is being counted is who *uses* it.
    """
    sites: dict[str, int] = {}
    for match in calls_to(src, "withReferenceLock"):
        name = enclosing_function(src, match.start())
        sites[name] = sites.get(name, 0) + 1
    return sites


def calls_to(src: str, name: str):
    """Every call of `name` that is not its own declaration."""
    for match in re.finditer(rf"\b{re.escape(name)}\(", src):
        if src[: match.start()].rstrip().endswith("function"):
            continue
        yield match


class TestTheLockLeads(unittest.TestCase):
    def test_the_helper_puts_the_lock_before_the_name(self) -> None:
        """The whole point, in the one place it is decided."""
        body = body_of(source(), "withReferenceLock")
        lock, name = body.index("referenceTag()"), body.index("nameHtml")
        self.assertLess(lock, name, "withReferenceLock() no longer leads with the lock")

    def test_nothing_else_draws_the_lock(self) -> None:
        """What stops a surface putting the mark back after the name.

        `referenceTag()` is called from the helper and from nowhere else, so no
        renderer is in a position to choose a side for it.
        """
        src = source()
        callers = {enclosing_function(src, m.start()) for m in calls_to(src, "referenceTag")}
        self.assertEqual(callers, {"withReferenceLock"})


class TestEverySurfaceIsMarked(unittest.TestCase):
    def test_every_naming_surface_goes_through_the_helper(self) -> None:
        missing = NAMING_SURFACES - set(call_sites(source()))
        self.assertEqual(missing, set(), "a surface names a model with no padlock")

    def test_no_surface_composes_a_name_unlisted(self) -> None:
        """The other direction: a helper call from somewhere this file does not
        know about means a surface was added and the list was not updated."""
        extra = set(call_sites(source())) - NAMING_SURFACES
        self.assertEqual(extra, set(), "unlisted renderer composes a model name")

    def test_the_surfaces_that_render_two_tables_call_it_once_each(self) -> None:
        """renderCompareSpecs and renderCompareBenchTable are near-identical
        column heads; a fix applied to one and not the other is the bug this
        catches."""
        sites = call_sites(source())
        for name in ("renderCompareSpecs", "renderCompareBenchTable"):
            with self.subTest(name):
                self.assertEqual(sites.get(name), 1)


class TestTheLockIsNotOnlyColour(unittest.TestCase):
    def test_the_mark_is_drawn_not_written(self) -> None:
        """An inline SVG, so it follows currentColor into both themes and needs
        no font -- and carries a label, since a lock on its own is a claim a
        screen reader is entitled to hear."""
        body = source()[source().index("function referenceTag()") :][:1200]
        self.assertIn("<svg", body)
        self.assertIn('aria-label="Closed weights"', body)

    def test_the_picker_no_longer_spells_the_word(self) -> None:
        """The comparison picker printed "closed" in the column that shows every
        other row's creator. The lock says it now, so the column says what it
        says everywhere else."""
        src = source()
        body = body_of(src, "renderCompareMenuBody")
        self.assertNotIn('? "closed"', body)
        self.assertIn('axis-menu-score">${escapeHtml(creatorNameOf(model))}', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
