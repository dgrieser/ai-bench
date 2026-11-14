#!/usr/bin/env python3
"""Tests for the AA Coding Agent Index reader. Run with ./test_aa_coding_agents.py

The index is keyed on AA's own dataset ids, and AA re-cuts it on new dataset
revisions -- "terminal-bench-v2.1" became "terminal-bench-v4", "deep-swe" became
"deep-swe-v1.1". An id this reader does not know is skipped, by design, so a
rename costs nothing louder than a line on stderr while two columns quietly stop
being fed. Nothing here can see a future rename, so what these pin is the half
that is checkable: the ids the page publishes today, and that every column the
map names is a column llm.json actually has.

One rename it can catch: an id naming no minor revision ("terminal-bench-v4",
4.0 until index v2.1, 4.1 since) is read only while the page's own label for
it is the one LABEL_GATES names, and the index only while the page names
INDEX_VERSION. Those gates are pinned below.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import fetch_aa_coding_agents as ca

LLM_JSON = Path(__file__).resolve().with_name("llm.json")


def row(
    model: str, evals: dict[str, float], agent: str = "Claude Code", index: float | None = None
) -> dict:
    return {
        "id": f"{abs(hash(model)):032x}"[:32],
        "indexScore": index,
        "indexComponentCount": 3,
        "evalCount": 3,
        "agentName": agent,
        "display": {"model": model},
        "displayLabel": f"{agent} - {model}",
        "evals": [
            {"datasetIndexName": dataset, "mean": {"reward": reward}}
            for dataset, reward in evals.items()
        ],
    }


LABELS = {
    "frontiercode-v2.0": "FrontierCode-AA 2.0",
    "terminal-bench-v4": "Terminal-Bench 4.1",
    "swe-atlas-qna": "SWE-Atlas-QnA-AA",
}
HTML = "<span>Artificial Analysis Coding Agent Index v2.1 incorporates 3 benchmarks</span>"


def label_list(labels: dict[str, str]) -> str:
    """The benchmark list as the flight payload spells it."""
    return "".join(
        json.dumps(
            ["$", "li", dataset, {"children": [["$", "span", None, {"className": "font-medium", "children": label}]]}],
            separators=(",", ":"),
        )
        for dataset, label in labels.items()
    )


def scores(rows: list[dict], labels: dict[str, str] = LABELS, html: str = HTML) -> list[dict]:
    # The flight payload carries no whitespace, and the row reader anchors on
    # that exact opening ('{"id":"<32 hex>"'), so the fixture has to match it.
    payload = label_list(labels) + "".join(json.dumps(r, separators=(",", ":")) for r in rows)

    def fetch(url: str, *args, rsc: bool = True, **kwargs) -> str:
        return payload if rsc else html

    with mock.patch.object(ca, "fetch_payload", side_effect=fetch):
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
        self.assertEqual(ca.DATASETS["frontiercode-v2.0"], "frontiercode_2_0")
        # "terminal-bench-v4" names no minor revision: index v2.1 moved it from
        # 4.0 to 4.1 under the same id, so it is held to the page's label.
        self.assertEqual(ca.DATASETS["terminal-bench-v4"], "terminal_bench_4_1")
        self.assertEqual(ca.LABEL_GATES["terminal-bench-v4"], "Terminal-Bench 4.1")

    def test_every_gate_names_a_mapped_dataset(self) -> None:
        self.assertLessEqual(set(ca.LABEL_GATES), set(ca.DATASETS))

    def test_the_index_column_exists_and_names_its_version(self) -> None:
        benchmarks = json.loads(LLM_JSON.read_text(encoding="utf-8"))["benchmarks"]
        self.assertIn(ca.INDEX_KEY, benchmarks)
        self.assertTrue(ca.INDEX_KEY.endswith("_" + ca.INDEX_VERSION.replace(".", "_")))


class TestGetScores(unittest.TestCase):
    def test_one_row_yields_one_entry_per_mapped_dataset(self) -> None:
        entries = scores([row("Opus 5 (max)", {
            "frontiercode-v2.0": 0.4397,
            "swe-atlas-qna": 0.621,
            "terminal-bench-v4": 0.5455,
        }, index=0.5354)])
        self.assertEqual(
            {e["key"]: e["score"] for e in entries},
            {
                "frontiercode_2_0": 43.97, "swe_atlas_qna": 62.1,
                "terminal_bench_4_1": 54.55, "aa_coding_agent_index_2_1": 53.54,
            },
        )
        self.assertEqual({e["model"] for e in entries}, {"opus 5"})

    def test_an_unmapped_dataset_reaches_no_column(self) -> None:
        entries = scores([row("Opus 5 (max)", {
            "terminal-bench-v4": 0.5455,
            "terminal-bench-v5": 0.1,
        })])
        self.assertEqual([e["score"] for e in entries], [54.55])

    def test_a_relabelled_dataset_is_not_read(self) -> None:
        # A 4.2 run under the same "terminal-bench-v4" id must not reach 4.1.
        for label in ("Terminal-Bench 4.2", None):
            with self.subTest(label=label):
                labels = {k: v for k, v in LABELS.items() if k != "terminal-bench-v4"}
                if label:
                    labels["terminal-bench-v4"] = label
                entries = scores(
                    [row("Opus 5 (max)", {"terminal-bench-v4": 0.5455, "swe-atlas-qna": 0.621})],
                    labels=labels,
                )
                self.assertEqual([e["key"] for e in entries], ["swe_atlas_qna"])

    def test_the_index_is_read_only_at_its_version(self) -> None:
        for html in (HTML.replace("v2.1", "v2.2"), "<span>no version</span>"):
            with self.subTest(html=html):
                entries = scores(
                    [row("Opus 5 (max)", {"swe-atlas-qna": 0.621}, index=0.5)], html=html
                )
                self.assertEqual([e["key"] for e in entries], ["swe_atlas_qna"])

    def test_an_index_over_missing_components_is_not_the_index(self) -> None:
        r = row("Opus 5 (max)", {"swe-atlas-qna": 0.621}, index=0.2)
        r["evalCount"] = 1
        self.assertEqual([e["key"] for e in scores([r])], ["swe_atlas_qna"])

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
