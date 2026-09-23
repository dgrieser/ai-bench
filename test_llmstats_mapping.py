#!/usr/bin/env python3
"""llm-stats mapping: one release per llm.json row.

llm-stats lists a model once per release -- "deepseek-v4-flash-0423" and
"deepseek-v4-flash-0731" -- and once more for a reasoning mode of one of them
("-max", which is the 0423 release). Mapping two releases onto one row let
update.py's best-reported-run rule file the April release's numbers under the
July row, the fold #231 fixed for Vals. The April rows are parked; this keeps
it that way and catches the next pair of dated releases folded together.
"""

from __future__ import annotations

import json
import re
import unittest
from collections import defaultdict
from pathlib import Path

MAPPING = Path(__file__).resolve().with_name("model-name-mapping-llmstats-to-artificialanalysis.json")

# A trailing release date: -0423, -2512, or a 675B-instruct-2512-style tail.
_DATED_RE = re.compile(r"-(\d{4})(?:-[a-z0-9]+)?$", re.IGNORECASE)

# Rows parked because they are a different release than the llm.json row.
PARKED = (
    "deepseek-v4-flash-0423",  # the row is the 0731 release
    "deepseek-v4-flash-max",   # max effort of the 0423 release
    "mistral-large-3-2509",    # the row is the 2512 release
)


class TestLlmStatsReleases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    def test_other_releases_stay_parked(self) -> None:
        for name in PARKED:
            with self.subTest(name=name):
                self.assertEqual(self.mapping[name], "__unmappable__")

    def test_no_row_takes_two_dated_releases(self) -> None:
        dates: dict[str, set[str]] = defaultdict(set)
        for name, slug in self.mapping.items():
            if not isinstance(slug, str) or slug.startswith("__"):
                continue
            match = _DATED_RE.search(name)
            if match:
                dates[slug].add(match.group(1))
        folded = {slug: sorted(d) for slug, d in dates.items() if len(d) > 1}
        self.assertEqual(folded, {})


if __name__ == "__main__":
    unittest.main()
