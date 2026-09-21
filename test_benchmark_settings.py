#!/usr/bin/env python3
"""Every benchmark column says which run of its benchmark it holds.

`description` says what a benchmark is. `settings` tags which of its published
runs this column stores -- the metric, the subset, the tool mode, the harness --
and `excludes` names the runs that must not be filed here. The two exist
because a column and a label can carry one benchmark's name and not be the same
measurement: "HLE" and "HLE w/ tools" differ by a median 11.5 points, and
ProgramBench publishes Resolved beside the Almost Resolved this column stores.

They are read by three surfaces -- the column-header slip, the benchmark
dialog, and the admin page's mapping queue, where they are the whole of what a
human has to decide a mapping on -- so a column without them is a column that
gets mapped from its key alone.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

LLM_JSON = Path(__file__).resolve().with_name("llm.json")

# Long enough for "best of native / prompted", short enough to stay a chip.
MAX_TAG = 30


def benchmarks() -> dict:
    return json.loads(LLM_JSON.read_text(encoding="utf-8"))["benchmarks"]


class TestSettings(unittest.TestCase):
    def test_every_column_tags_its_run(self) -> None:
        for key, bench in benchmarks().items():
            with self.subTest(benchmark=key):
                settings = bench.get("settings")
                self.assertIsInstance(settings, list, "no settings recorded")
                self.assertTrue(settings, "settings is empty")

    def test_settings_are_tags_rather_than_prose(self) -> None:
        # They are drawn as chips, so each one has to stand on its own line of
        # a row: no stray whitespace, no sentence, nothing that wraps.
        for key, bench in benchmarks().items():
            with self.subTest(benchmark=key):
                tags = bench["settings"]
                for tag in tags:
                    self.assertIsInstance(tag, str)
                    self.assertTrue(tag.strip())
                    self.assertEqual(tag, tag.strip())
                    self.assertEqual(tag, " ".join(tag.split()))
                    self.assertLessEqual(len(tag), MAX_TAG, f"not a tag: {tag!r}")
                    self.assertFalse(tag.endswith("."), f"not a tag: {tag!r}")
                self.assertEqual(len(tags), len(set(tags)), "duplicate tag")

    def test_excludes_is_a_list_of_phrases(self) -> None:
        for key, bench in benchmarks().items():
            if "excludes" not in bench:
                continue
            with self.subTest(benchmark=key):
                excludes = bench["excludes"]
                self.assertIsInstance(excludes, list)
                # An empty list is not a statement; the key is left out instead.
                self.assertTrue(excludes, "excludes present but empty")
                for item in excludes:
                    self.assertIsInstance(item, str)
                    self.assertTrue(item.strip())
                    self.assertEqual(item, item.strip())
                    # Phrases, joined with a separator by the page -- a full
                    # stop here would read as the end of the list.
                    self.assertFalse(item.endswith("."), f"not a phrase: {item!r}")
                self.assertEqual(len(excludes), len(set(excludes)))

    def test_a_derived_index_says_it_is_derived(self) -> None:
        # Nothing measured them, so the tags have to say the number is not a
        # percentage before a reader compares it with one.
        for key, bench in benchmarks().items():
            if not bench.get("derived"):
                continue
            with self.subTest(benchmark=key):
                self.assertIn("index scale", bench["settings"])


if __name__ == "__main__":
    unittest.main()
