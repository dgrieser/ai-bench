#!/usr/bin/env python3
"""Tests for the index ranking record. Run with ./test_rankings.py

llm.json's index_rankings is what lets llm.html list a model entering an
index, moving from one place to another, or dropping out. These tests pin the
rules in _rankings.py -- places with ties, moves as changes of order, the
one-change-per-day fold, the silent start of a new index, renames -- the
writers that feed it, and the shipped llm.json.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import _rankings
import derive_indexes

DAY1 = "2026-10-01"
DAY2 = "2026-10-02"

BENCHMARKS = {
    "bench": {"name": "Bench"},
    "coding_index": {"name": "Coding", "derived": True},
    "trust_index": {"name": "Trust", "derived": True},
}


def doc_with(**values: dict) -> dict:
    """values: model name -> {index key: value}."""
    return {
        "benchmarks": dict(BENCHMARKS),
        "models": [{"name": name, "scores": dict(scores)} for name, scores in values.items()],
    }


def set_scores(doc: dict, **values: dict) -> None:
    by_name = {model["name"]: model for model in doc["models"]}
    for name, scores in values.items():
        if name not in by_name:
            by_name[name] = {"name": name, "scores": {}}
            doc["models"].append(by_name[name])
        by_name[name]["scores"].update(scores)


def logged(doc: dict) -> list[tuple]:
    return sorted(
        (e["date"], e["index"], e["model"], e["from"], e["to"])
        for e in _rankings.changes(doc)
    )


class RanksTest(unittest.TestCase):
    def test_ties_share_a_place(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 7}, c={"coding_index": 7}, d={"coding_index": 1})
        self.assertEqual(_rankings.ranks(doc, "coding_index"), {"a": 1, "b": 2, "c": 2, "d": 4})

    def test_unranked_models_are_absent(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": None}, c={"bench": 3})
        self.assertEqual(_rankings.ranks(doc, "coding_index"), {"a": 1})

    def test_only_derived_columns_are_indexes(self):
        self.assertEqual(_rankings.index_keys(doc_with()), ["coding_index", "trust_index"])


class SyncTest(unittest.TestCase):
    def test_first_sync_records_nothing(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        self.assertEqual(_rankings.sync(doc, today=DAY1), 0)
        self.assertEqual(_rankings.changes(doc), [])
        self.assertEqual(doc["index_rankings"]["ranks"]["coding_index"], {"a": 1, "b": 2})
        self.assertEqual(doc["index_rankings"]["since"], DAY1)

    def test_overtaking_is_a_move_for_both(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, b={"coding_index": 10})
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(
            logged(doc),
            [(DAY2, "coding_index", "a", 1, 2), (DAY2, "coding_index", "b", 2, 1)],
        )

    def test_a_newcomer_enters_without_moving_the_rest(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, c={"coding_index": 20})
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(logged(doc), [(DAY2, "coding_index", "c", None, 1)])
        entry = _rankings.changes(doc)[0]
        self.assertEqual(entry["of"], 3)

    def test_dropping_out_and_leaving_the_file(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5}, c={"coding_index": 1})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, a={"coding_index": None})
        doc["models"] = [m for m in doc["models"] if m["name"] != "c"]
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(
            logged(doc),
            [(DAY2, "coding_index", "a", 1, None), (DAY2, "coding_index", "c", 3, None)],
        )

    def test_one_change_per_model_per_day(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5}, c={"coding_index": 1})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, c={"coding_index": 6})
        _rankings.sync(doc, today=DAY2)
        set_scores(doc, c={"coding_index": 10})
        _rankings.sync(doc, today=DAY2)
        moves = {e["model"]: (e["from"], e["to"]) for e in _rankings.changes(doc)}
        self.assertEqual(moves, {"a": (1, 2), "b": (2, 3), "c": (3, 1)})

    def test_a_move_undone_the_same_day_leaves_nothing(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, b={"coding_index": 10})
        _rankings.sync(doc, today=DAY2)
        set_scores(doc, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(_rankings.changes(doc), [])

    def test_entered_and_left_the_same_day_leaves_nothing(self):
        doc = doc_with(a={"coding_index": 9})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY2)
        set_scores(doc, b={"coding_index": None})
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(_rankings.changes(doc), [])

    def test_a_later_shift_updates_the_days_place(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        set_scores(doc, b={"coding_index": 10})
        _rankings.sync(doc, today=DAY2)
        set_scores(doc, z={"coding_index": 50})
        _rankings.sync(doc, today=DAY2)
        moves = {e["model"]: (e["from"], e["to"]) for e in _rankings.changes(doc)}
        # b went from 2nd to 1st, then a newcomer took 1st: b ends the day 2nd,
        # where it started, so only a (overtaken) and z (entered) are news.
        self.assertEqual(moves, {"a": (1, 3), "z": (None, 1)})

    def test_a_new_index_starts_silently(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        doc["benchmarks"]["vision_index"] = {"name": "Vision", "derived": True}
        set_scores(doc, a={"vision_index": 3}, b={"vision_index": 4})
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(_rankings.changes(doc), [])
        self.assertIn("vision_index", doc["index_rankings"]["ranks"])

    def test_old_changes_expire(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today="2026-01-01")
        set_scores(doc, b={"coding_index": 10})
        _rankings.sync(doc, today="2026-01-02")
        self.assertTrue(_rankings.changes(doc))
        _rankings.sync(doc, today="2026-06-01")
        self.assertEqual(_rankings.changes(doc), [])

    def test_rename_is_not_a_departure(self):
        doc = doc_with(a={"coding_index": 9}, b={"coding_index": 5})
        _rankings.sync(doc, today=DAY1)
        doc["models"][0]["name"] = "a2"
        _rankings.rename(doc, "a", "a2")
        _rankings.sync(doc, today=DAY2)
        self.assertEqual(_rankings.changes(doc), [])
        self.assertEqual(_rankings.validate(doc), [])


class WriterTest(unittest.TestCase):
    def test_refresh_records_the_ranking(self):
        """derive_indexes.refresh() -- the path update.py, edit.py, prune.py and
        the admin answers all take -- leaves a record in step with the
        columns it just wrote."""
        doc = json.loads((Path(__file__).resolve().parent / "llm.json").read_text(encoding="utf-8"))
        doc.pop(_rankings.ROOT_FIELD, None)
        derive_indexes.refresh(doc)
        self.assertIn(_rankings.ROOT_FIELD, doc)
        self.assertEqual(doc[_rankings.ROOT_FIELD]["ranks"], _rankings.current(doc))


class ShippedTest(unittest.TestCase):
    def test_llm_json_record_is_valid(self):
        doc = json.loads((Path(__file__).resolve().parent / "llm.json").read_text(encoding="utf-8"))
        self.assertIn(_rankings.ROOT_FIELD, doc)
        self.assertEqual(_rankings.validate(doc), [])


if __name__ == "__main__":
    unittest.main()
