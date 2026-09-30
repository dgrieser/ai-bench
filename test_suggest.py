#!/usr/bin/env python3
"""Tests for _matching.suggest(), the candidates the queue shows a person.

Run with ./test_suggest.py. Offline: the ground truth is the mapping files
themselves -- every answer a person has already given -- replayed against the
same lists the queue offers from.

Two things are pinned. suggest() may only ever *add* to grade(): no answer a
person chose may rank lower than grade() ranked it, and nothing it adds may
claim EXACT, which is the tier propose.py commits unseen. And the spellings it
exists for -- a vendor prefix, a packaging suffix, a benchmark's display name
-- have to find their answer.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import _answers
import _matching
import pending_prompts
import propose
from _openness import SENTINELS
from _scores import editable_benchmarks

HERE = Path(__file__).resolve().parent


def ground_truth():
    """(route, universe name, subject, chosen answer, options, aliases) per mapping."""
    doc = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
    aa_file = HERE / "_aa" / "models.json"
    aa = [m["slug"] for m in json.loads(aa_file.read_text(encoding="utf-8"))["models"]] \
        if aa_file.exists() else []
    universes = {
        propose.MODELS: [m["name"] for m in doc["models"]],
        propose.BENCHMARKS: sorted(editable_benchmarks(doc)),
        propose.AA_SLUGS: aa,
    }
    aliases = {propose.BENCHMARKS: pending_prompts.benchmark_aliases(HERE / "llm.json")}
    seen = set()
    for name, by_kind in propose.ROUTES.items():
        for route in by_kind.values():
            path = _answers.mapping_path(route)
            if path in seen:
                continue
            seen.add(path)
            for key, value in json.loads(path.read_text(encoding="utf-8")).items():
                chosen = next((v for v in (value if isinstance(value, list) else [value])
                               if isinstance(v, str) and v not in SENTINELS), None)
                options = universes[route.universe]
                if chosen is None or chosen not in options:
                    continue
                subject = propose.match_subject({"subject": key}, route)
                yield name, route.universe, subject, chosen, options, aliases.get(route.universe)


def rank(matches, chosen):
    options = [m.option for m in matches]
    return options.index(chosen) + 1 if chosen in options else None


class TestAgainstTheMappingFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = []
        for name, universe, subject, chosen, options, aliases in ground_truth():
            graded = _matching.grade(subject, options)
            suggested = _matching.suggest(subject, options, aliases)
            cls.rows.append((name, universe, subject, chosen, graded, suggested))

    def test_no_answer_ranks_lower_than_grade_ranked_it(self) -> None:
        worse = [
            (name, subject, chosen, rank(graded, chosen), rank(suggested, chosen))
            for name, _u, subject, chosen, graded, suggested in self.rows
            if rank(graded, chosen) is not None
            and (rank(suggested, chosen) is None or rank(suggested, chosen) > rank(graded, chosen))
        ]
        self.assertEqual(worse, [])

    def test_nothing_added_claims_exact(self) -> None:
        for _n, _u, subject, _c, graded, suggested in self.rows:
            exact = {m.option for m in graded if m.confidence == _matching.EXACT}
            self.assertEqual({m.option for m in suggested if m.confidence == _matching.EXACT},
                             exact, subject)

    def test_most_answers_are_offered_at_all(self) -> None:
        """The reason suggest() exists: grade() missed 80 model and 106 benchmark answers."""
        for universe, floor in ((propose.MODELS, 0.98), (propose.BENCHMARKS, 0.95)):
            rows = [r for r in self.rows if r[1] == universe]
            found = sum(rank(r[5], r[3]) is not None and rank(r[5], r[3]) <= pending_prompts.MAX_CANDIDATES
                        for r in rows)
            self.assertGreaterEqual(found / len(rows), floor, f"{universe}: {found}/{len(rows)}")


class TestSpellings(unittest.TestCase):
    BENCH_ALIASES = {"aime_2025": ["AIME 2025", "AIME25"], "ifbench": ["IFBench"],
                     "hle": ["HLE"], "livecodebench": ["LiveCodeBench", "LCBench"]}

    def top(self, name, options, aliases=None):
        found = _matching.suggest(name, options, aliases)
        return found[0].option if found else None

    def test_a_vendor_prefix(self) -> None:
        self.assertEqual(self.top("anthropic/claude-opus-5.5", ["claude-opus-5", "claude-opus-5-5"]),
                         "claude-opus-5-5")

    def test_a_packaging_suffix_and_a_leading_word(self) -> None:
        self.assertEqual(self.top("prism-ml/Ternary-Bonsai-2-27B-gguf", ["bonsai-8b", "bonsai-2-27b"]),
                         "bonsai-2-27b")

    def test_punctuation(self) -> None:
        self.assertEqual(self.top("qwen-3-8-27b", ["qwen3-8-27b", "qwen3-8-max"]), "qwen3-8-27b")

    def test_a_benchmark_by_its_display_name(self) -> None:
        options = sorted(self.BENCH_ALIASES)
        self.assertEqual(self.top("AIME25", options, self.BENCH_ALIASES), "aime_2025")
        self.assertEqual(self.top("IFBench (prompt-loose)", options, self.BENCH_ALIASES), "ifbench")
        self.assertEqual(self.top("HLE no tools", options, self.BENCH_ALIASES), "hle")
        self.assertEqual(self.top("LiveCodeBenchV6", options, self.BENCH_ALIASES), "livecodebench")

    def test_generic_words_alone_suggest_nothing(self) -> None:
        options = ["ifbench", "zerobench", "exploitbench"]
        self.assertEqual(_matching.suggest("OCR Bench v2", options, {"ifbench": ["IFBench"]}), [])

    def test_a_trailing_number_is_not_a_version_for_models(self) -> None:
        # qwen3 and qwen3-8 are different models, however the numbers read.
        self.assertNotEqual(self.top("qwen3-8", ["qwen3"]), "qwen3")


if __name__ == "__main__":
    unittest.main()
