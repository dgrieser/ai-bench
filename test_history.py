#!/usr/bin/env python3
"""Tests for per-score change history. Run with ./test_history.py

models[].scores_history is what lets llm.html say a score *moved* rather than
only that it was written: one entry per recorded change, the last of them
mirroring the score, date and source URL the other three maps carry. These
tests pin the rules readers depend on -- the mirror, the one-entry-per-day
collapse, removals, the derived-column exclusion -- the writers that feed it,
and the shipped llm.json, which has to satisfy every one of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import _history
import update

TODAY = date.today().isoformat()
URL_A = "https://example.com/leaderboard-a"
URL_B = "https://example.com/leaderboard-b"

DOC_BENCHMARKS = {
    "bench": {"name": "Bench"},
    "other": {"name": "Other"},
    "coding_index": {"name": "Coding Index", "derived": True},
}


def doc_with(*models: dict) -> dict:
    return {"benchmarks": dict(DOC_BENCHMARKS), "models": list(models)}


def model_with(**scores) -> dict:
    """A model carrying the three maps, every key present, nothing recorded."""
    keys = list(DOC_BENCHMARKS)
    return {
        "name": "m",
        "date_added": "2026-01-01",
        "scores": {key: scores.get(key) for key in keys},
        "scores_updated": {key: None for key in keys},
        "scores_source": {key: None for key in keys},
    }


def write(model: dict, key: str, value, when: str, url: str | None = None) -> None:
    """One score write, the way every writer makes it: value, date, source."""
    model["scores"][key] = value
    model["scores_updated"][key] = when
    model["scores_source"][key] = url


def log(model: dict, key: str = "bench") -> list[dict]:
    return model.get(_history.HISTORY_FIELD, {}).get(key, [])


class TestRecording(unittest.TestCase):
    def test_first_score_is_recorded_once(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        _history.sync(doc_with(model))
        self.assertEqual(
            log(model), [{"date": "2026-02-01", "score": 45.0, "source": URL_A}]
        )

    def test_a_change_keeps_what_it_replaced(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", 51.0, "2026-03-01", URL_B)
        _history.sync(doc)
        self.assertEqual(
            log(model),
            [
                {"date": "2026-02-01", "score": 45.0, "source": URL_A},
                {"date": "2026-03-01", "score": 51.0, "source": URL_B},
            ],
        )

    def test_the_last_entry_mirrors_the_current_score(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", 51.0, "2026-03-01", URL_B)
        _history.sync(doc)
        last = log(model)[-1]
        self.assertEqual(last["score"], model["scores"]["bench"])
        self.assertEqual(last["date"], model["scores_updated"]["bench"])
        self.assertEqual(last["source"], model["scores_source"]["bench"])
        self.assertEqual(_history.validate(doc), [])

    def test_a_second_write_the_same_day_replaces_that_day(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", 47.0, "2026-02-01", URL_B)
        _history.sync(doc)
        self.assertEqual(
            log(model), [{"date": "2026-02-01", "score": 47.0, "source": URL_B}]
        )

    def test_sync_is_idempotent(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        before = json.dumps(model[_history.HISTORY_FIELD])
        self.assertEqual(_history.sync(doc), 0)
        self.assertEqual(json.dumps(model[_history.HISTORY_FIELD]), before)

    def test_a_score_with_no_date_falls_back_to_the_model_arriving(self) -> None:
        model = model_with(bench=45.0)
        _history.sync(doc_with(model))
        self.assertEqual(log(model)[0]["date"], "2026-01-01")

    def test_the_map_is_placed_after_scores_source(self) -> None:
        model = model_with(bench=45.0)
        model["vram"] = {"fp16": 10}
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        _history.sync(doc_with(model))
        fields = list(model)
        self.assertEqual(
            fields[fields.index("scores_source") + 1], _history.HISTORY_FIELD
        )
        self.assertEqual(fields[-1], "vram")


class TestProvenanceCorrections(unittest.TestCase):
    """A date or a URL arriving late is not a change; the entry is corrected."""

    def test_a_backfilled_source_lands_on_the_existing_entry(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", None)
        doc = doc_with(model)
        _history.sync(doc)
        model["scores_source"]["bench"] = URL_A
        _history.sync(doc)
        self.assertEqual(
            log(model), [{"date": "2026-02-01", "score": 45.0, "source": URL_A}]
        )

    def test_a_backfilled_date_moves_the_existing_entry(self) -> None:
        model = model_with(bench=45.0)
        _history.sync(doc_with(model))
        model["scores_updated"]["bench"] = "2026-02-01"
        _history.sync(doc_with(model))
        self.assertEqual(log(model)[0]["date"], "2026-02-01")
        self.assertEqual(len(log(model)), 1)


class TestRemovals(unittest.TestCase):
    def test_a_withdrawn_score_is_recorded_as_null(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", None, None, None)
        _history.sync(doc, today="2026-03-01")
        self.assertEqual(
            log(model),
            [
                {"date": "2026-02-01", "score": 45.0, "source": URL_A},
                {"date": "2026-03-01", "score": None, "source": None},
            ],
        )
        self.assertEqual(_history.validate(doc), [])

    def test_a_recorded_removal_is_never_restamped(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", None, None, None)
        _history.sync(doc, today="2026-03-01")
        self.assertEqual(_history.sync(doc, today="2026-04-01"), 0)
        self.assertEqual(log(model)[-1]["date"], "2026-03-01")

    def test_a_score_added_and_withdrawn_the_same_day_leaves_nothing(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", None, None, None)
        _history.sync(doc, today="2026-02-01")
        self.assertEqual(log(model), [])
        self.assertNotIn("bench", model.get(_history.HISTORY_FIELD, {}))

    def test_a_score_that_never_had_a_value_is_never_recorded(self) -> None:
        model = model_with()
        _history.sync(doc_with(model))
        self.assertEqual(model.get(_history.HISTORY_FIELD, {}), {})

    def test_a_score_that_comes_back_reads_as_an_arrival(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        write(model, "bench", None, None, None)
        _history.sync(doc, today="2026-03-01")
        write(model, "bench", 60.0, "2026-04-01", URL_B)
        _history.sync(doc)
        self.assertEqual([entry["score"] for entry in log(model)], [45.0, None, 60.0])


class TestExclusions(unittest.TestCase):
    def test_a_derived_column_is_never_recorded(self) -> None:
        model = model_with(coding_index=44000)
        write(model, "coding_index", 44000, "2026-02-01", URL_A)
        _history.sync(doc_with(model))
        self.assertEqual(model.get(_history.HISTORY_FIELD, {}), {})

    def test_history_left_on_a_derived_column_is_dropped(self) -> None:
        model = model_with(bench=45.0, coding_index=44000)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        model[_history.HISTORY_FIELD] = {
            "coding_index": [{"date": "2026-02-01", "score": 44000, "source": URL_A}]
        }
        _history.sync(doc_with(model))
        self.assertNotIn("coding_index", model[_history.HISTORY_FIELD])
        self.assertIn("bench", model[_history.HISTORY_FIELD])

    def test_a_column_that_leaves_llm_json_takes_its_history_with_it(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        del doc["benchmarks"]["bench"]
        del model["scores"]["bench"]
        _history.sync(doc)
        self.assertEqual(model.get(_history.HISTORY_FIELD, {}), {})


class TestValidate(unittest.TestCase):
    def test_a_history_that_ends_elsewhere_is_reported(self) -> None:
        model = model_with(bench=51.0)
        write(model, "bench", 51.0, "2026-03-01", URL_B)
        model[_history.HISTORY_FIELD] = {
            "bench": [{"date": "2026-02-01", "score": 45.0, "source": URL_A}]
        }
        problems = _history.validate(doc_with(model))
        self.assertEqual(len(problems), 1)
        self.assertIn("m.bench", problems[0])

    def test_an_entry_with_no_date_is_reported(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        model[_history.HISTORY_FIELD] = {"bench": [{"score": 45.0, "source": URL_A}]}
        self.assertTrue(_history.validate(doc_with(model)))

    def test_a_clean_document_reports_nothing(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        self.assertEqual(_history.validate(doc), [])


class TestWriters(unittest.TestCase):
    """The paths that actually move scores, end to end through sync()."""

    def test_an_ingest_write_is_recorded(self) -> None:
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        update.apply_score(doc, model, "m", "bench", 51.0, URL_B, [])
        _history.sync(doc)
        self.assertEqual([entry["score"] for entry in log(model)], [45.0, 51.0])
        self.assertEqual(log(model)[-1]["date"], TODAY)
        self.assertEqual(log(model)[-1]["source"], URL_B)

    def test_a_refused_write_records_nothing(self) -> None:
        """apply_score declines a lower-ranked source; so must the history."""
        model = model_with(bench=45.0)
        write(model, "bench", 45.0, "2026-02-01", URL_A)
        doc = doc_with(model)
        _history.sync(doc)
        update.apply_score(doc, model, "m", "bench", None, URL_B, [])
        self.assertEqual(_history.sync(doc), 0)
        self.assertEqual(len(log(model)), 1)

    def test_edit_py_records_a_hand_edit(self) -> None:
        doc = {
            "benchmarks": {"bench": {"name": "Bench", "decimals": 1}},
            "models": [
                {
                    "name": "m",
                    "date_added": "2026-01-01",
                    "scores": {"bench": 45.0},
                    "scores_updated": {"bench": "2026-02-01"},
                    "scores_source": {"bench": URL_A},
                }
            ],
        }
        _history.sync(doc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "edit.py"),
                    str(path),
                    "--model",
                    "m",
                    "--bench",
                    "51",
                    "--score-url",
                    URL_B,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            written = json.loads(path.read_text(encoding="utf-8"))
        entries = written["models"][0][_history.HISTORY_FIELD]["bench"]
        self.assertEqual([entry["score"] for entry in entries], [45.0, 51])
        # The page the hand edit cited, not the previous source, which never
        # produced this number.
        self.assertEqual(entries[-1]["source"], URL_B)


class TestShippedFile(unittest.TestCase):
    """llm.json as committed has to satisfy every rule above."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(
            (Path(__file__).resolve().parent / "llm.json").read_text(encoding="utf-8")
        )

    def test_history_is_consistent_with_the_scores(self) -> None:
        self.assertEqual(_history.validate(self.doc), [])

    def test_nothing_is_left_to_record(self) -> None:
        self.assertEqual(_history.sync(self.doc), 0)

    def test_history_since_is_set(self) -> None:
        since = self.doc.get(_history.SINCE_FIELD)
        self.assertIsInstance(since, str)
        date.fromisoformat(since)

    def test_entries_are_in_date_order(self) -> None:
        """Out of order, "what it was before" would name the wrong number."""
        for model in self.doc["models"]:
            for key, entries in _history.history(model).items():
                dates = [entry["date"] for entry in entries]
                self.assertEqual(
                    dates, sorted(dates), f"{model.get('name')}.{key} is out of order"
                )

    def test_every_scored_benchmark_has_a_history(self) -> None:
        derived = _history.derived_keys(self.doc)
        for model in self.doc["models"]:
            recorded = _history.history(model)
            for key, value in (model.get("scores") or {}).items():
                if value is None or key in derived:
                    continue
                self.assertIn(
                    key, recorded, f"{model.get('name')}.{key} has a score but no history"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
