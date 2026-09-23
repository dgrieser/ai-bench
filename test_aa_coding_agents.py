#!/usr/bin/env python3
"""Tests for the AA Coding Agent Index reader. Run with ./test_aa_coding_agents.py

The index is keyed on AA's own dataset ids, and AA re-cuts it on new dataset
revisions -- "terminal-bench-v2.1" became "terminal-bench-v4", "deep-swe" became
"deep-swe-v1.1". An id this reader does not know is skipped, by design, so a
rename costs nothing louder than a line on stderr while two columns quietly stop
being fed. Nothing here can see a future rename, so what these pin is the half
that is checkable: the ids the page publishes today, and that every column the
map names is a column llm.json actually has.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import fetch_aa_coding_agents as ca

LLM_JSON = Path(__file__).resolve().with_name("llm.json")


def row(model: str, evals: dict[str, float], agent: str = "Claude Code") -> dict:
    return {
        "id": f"{abs(hash(model)):032x}"[:32],
        "agentName": agent,
        "display": {"model": model},
        "displayLabel": f"{agent} - {model}",
        "evals": [
            {"datasetIndexName": dataset, "mean": {"reward": reward}}
            for dataset, reward in evals.items()
        ],
    }


def scores(rows: list[dict]) -> list[dict]:
    # The flight payload carries no whitespace, and the row reader anchors on
    # that exact opening ('{"id":"<32 hex>"'), so the fixture has to match it.
    payload = "".join(json.dumps(r, separators=(",", ":")) for r in rows)
    with mock.patch.object(ca, "fetch_payload", return_value=payload):
        return ca.get_scores()


class TestDatasetMap(unittest.TestCase):
    def test_every_mapped_column_exists_in_llm_json(self) -> None:
        benchmarks = json.loads(LLM_JSON.read_text(encoding="utf-8"))["benchmarks"]
        for dataset, key in ca.DATASETS.items():
            with self.subTest(dataset=dataset):
                self.assertIn(key, benchmarks)

    def test_the_versioned_ids_name_the_revision_they_measure(self) -> None:
        # The reason "deep-swe" was left out until AA versioned it: llm.json
        # keeps a column per revision, and an id naming none cannot pick one.
        self.assertEqual(ca.DATASETS["deep-swe-v1.1"], "deepswe_1_1")
        self.assertEqual(ca.DATASETS["terminal-bench-v4"], "terminal_bench_4_0")


class TestGetScores(unittest.TestCase):
    def test_one_row_yields_one_entry_per_mapped_dataset(self) -> None:
        entries = scores([row("Opus 5 (max)", {
            "deep-swe-v1.1": 0.6254,
            "swe-atlas-qna": 0.621,
            "terminal-bench-v4": 0.5455,
        })])
        self.assertEqual(
            {e["key"]: e["score"] for e in entries},
            {"deepswe_1_1": 62.54, "swe_atlas_qna": 62.1, "terminal_bench_4_0": 54.55},
        )
        self.assertEqual({e["model"] for e in entries}, {"opus 5"})

    def test_an_unmapped_dataset_reaches_no_column(self) -> None:
        entries = scores([row("Opus 5 (max)", {
            "terminal-bench-v4": 0.5455,
            "terminal-bench-v5": 0.1,
        })])
        self.assertEqual([e["score"] for e in entries], [54.55])

    def test_a_row_without_a_reward_is_dropped(self) -> None:
        entries = scores([
            row("Opus 5 (max)", {"terminal-bench-v4": None}),
            row("GLM 5.2", {"terminal-bench-v4": 0.4}),
        ])
        self.assertEqual([e["model"] for e in entries], ["glm 5.2"])

    def test_a_board_with_no_scored_row_raises(self) -> None:
        # A renamed "reward" drops every row one at a time; that is a layout
        # change to report, not an empty board to write.
        with self.assertRaises(ValueError):
            scores([row("Opus 5 (max)", {"terminal-bench-v4": None})])

    def test_a_reward_on_the_percentage_scale_raises(self) -> None:
        # reward is multiplied by 100; one that is already a percentage would
        # be stored as 5,455.
        with self.assertRaises(ValueError):
            scores([row("Opus 5 (max)", {"terminal-bench-v4": 54.55})])


if __name__ == "__main__":
    unittest.main()
