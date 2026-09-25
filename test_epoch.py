#!/usr/bin/env python3
"""Tests for the Epoch AI Benchmarking Hub reader. Run with ./test_epoch.py

The hub is one archive of CSVs, a row per model configuration, with the model
each configuration belongs to in model_metadata.csv. Three of the files it is
read for mirror boards that llm.json splits by revision, and the CSVs never say
which revision they hold; the hub page does, so the page decides the column.

What these tests pin is what could silently move a number or put it in the
wrong column: that the revision is read off the page and nowhere else, that a
page naming none -- or naming one llm.json has no column for -- drops the
benchmark instead of guessing, that fractions become percentages and a
percentage scale is refused, that a configuration is credited to its model
group, and that llm.json carries the attribution the licence asks for whenever
it carries a score read from the hub.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fetch_epoch as fe
import update

LLM_JSON = Path(__file__).resolve().with_name("llm.json")

METADATA = [
    {"model_version": "glm-5.2_max", "model_group": "GLM-5.2", "accessibility": "Open weights (unrestricted)"},
    {"model_version": "glm-5.2_high", "model_group": "GLM-5.2", "accessibility": "Open weights (unrestricted)"},
    {"model_version": "gpt-5.6-sol_max", "model_group": "GPT-5.6 Sol", "accessibility": "API access"},
    {"model_version": "mystery", "model_group": "Mystery", "accessibility": ""},
]


def csv_text(rows: list[dict], fields: list[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def archive(files: dict[str, str] | None = None, metadata: list[dict] | None = None) -> zipfile.ZipFile:
    """An in-memory hub archive: model_metadata.csv plus the given files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(
            fe.METADATA_FILE,
            csv_text(metadata or METADATA, ["model_version", "model_group", "accessibility"]),
        )
        for name, text in (files or {}).items():
            zf.writestr(name, text)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def board(page: str, rows: list[tuple[str, object]]) -> dict[str, str]:
    """One benchmark's file, as (model version, score) rows."""
    bench = fe.BENCHMARKS[page]
    fields = [fe.VERSION_COLUMN, bench.column, "Notes"]
    return {
        bench.file: csv_text(
            [{fe.VERSION_COLUMN: v, bench.column: s, "Notes": "multi\nline"} for v, s in rows],
            fields,
        )
    }


def hub_page(title: str, body: str = "") -> str:
    return (
        f'<html><head><title>{title} | Epoch AI</title>'
        f'<meta property="og:title" content="{title}"></head>'
        f"<body><h1>{title}</h1><p>{body}</p></body></html>"
    )


def pages(**by_page: str):
    """A fetch_page stub serving hub pages by slug."""
    return lambda url: by_page[url.rsplit("/", 1)[-1]]


class TestPageRevision(unittest.TestCase):
    def test_the_title_names_it(self) -> None:
        self.assertEqual(fe.page_revision(hub_page("DeepSWE v1.1")), "1.1")
        self.assertEqual(fe.page_revision(hub_page("FrontierSWE (v2)")), "2")

    def test_a_bare_title_falls_back_to_the_methodology(self) -> None:
        page = hub_page(
            "FrontierCode",
            "We source results from Cognition's public leaderboard, using only "
            "the current 1.1 revision.",
        )
        self.assertEqual(fe.page_revision(page), "1.1")

    def test_a_page_naming_none_has_none(self) -> None:
        self.assertIsNone(fe.page_revision(hub_page("GPQA Diamond", "Released in 2023, 198 questions.")))

    def test_the_title_wins_over_the_text(self) -> None:
        page = hub_page("DeepSWE v1.1", "the current 1.0 revision")
        self.assertEqual(fe.page_revision(page), "1.1")


class TestResolveKey(unittest.TestCase):
    def test_each_mirror_lands_in_the_revision_its_page_names(self) -> None:
        fetch = pages(
            deepswe=hub_page("DeepSWE v1.1"),
            frontierswe=hub_page("FrontierSWE (v2)"),
            frontiercode=hub_page("FrontierCode", "using only the current 1.1 revision"),
        )
        for page, key in (
            ("deepswe", "deepswe_1_1"),
            ("frontierswe", "frontierswe_2_0"),
            # Epoch charts the Main board, which keeps the plain base.
            ("frontiercode", "frontiercode_1_1"),
        ):
            with self.subTest(page=page):
                self.assertEqual(fe.resolve_key(page, fe.BENCHMARKS[page], fetch)[0], key)

    def test_an_unknown_or_missing_revision_resolves_to_no_column(self) -> None:
        for title in ("DeepSWE v1.2", "DeepSWE"):
            with self.subTest(title=title):
                key, _ = fe.resolve_key("deepswe", fe.BENCHMARKS["deepswe"], pages(deepswe=hub_page(title)))
                self.assertIsNone(key)

    def test_a_fixed_column_reads_no_page(self) -> None:
        def no_fetch(url: str) -> str:
            raise AssertionError(f"fetched {url}")

        self.assertEqual(
            fe.resolve_key("gpqa-diamond", fe.BENCHMARKS["gpqa-diamond"], no_fetch),
            ("gpqa_diamond", None),
        )

    def test_every_key_it_can_resolve_is_a_column(self) -> None:
        benchmarks = json.loads(LLM_JSON.read_text(encoding="utf-8"))["benchmarks"]
        self.assertTrue(fe.possible_keys() <= set(benchmarks), fe.possible_keys() - set(benchmarks))


class TestRead(unittest.TestCase):
    def test_rows_are_percentages_credited_to_the_model_group(self) -> None:
        zf = archive(board("gpqa-diamond", [("glm-5.2_max", 0.9186), ("glm-5.2_high", 0.9), ("gone", 0.5)]))
        rows = fe.get_scores(["gpqa-diamond"], archive=zf)
        self.assertEqual([r["score"] for r in rows], [91.86, 90.0, 50.0])
        self.assertEqual([r["model"] for r in rows], ["GLM-5.2", "GLM-5.2", "gone"])
        first = rows[0]
        self.assertEqual(first["key"], "gpqa_diamond")
        self.assertEqual(first["version"], "glm-5.2_max")
        self.assertIs(first["open_weights"], True)
        self.assertEqual(first["source"], "https://epoch.ai/benchmarks/gpqa-diamond")

    def test_rows_without_a_model_version_or_score_are_skipped(self) -> None:
        zf = archive(board("frontiercode", [("", 0.42), ("glm-5.2_max", ""), ("glm-5.2_high", 0.245)]))
        fetch = pages(frontiercode=hub_page("FrontierCode", "the current 1.1 revision"))
        rows = fe.get_scores(["frontiercode"], archive=zf, fetch_page=fetch)
        self.assertEqual([(r["version"], r["score"]) for r in rows], [("glm-5.2_high", 24.5)])
        self.assertEqual(rows[0]["revision"], "1.1")

    def test_a_benchmark_whose_revision_is_unknown_contributes_nothing(self) -> None:
        zf = archive({**board("deepswe", [("glm-5.2_max", 0.44)]), **board("gpqa-diamond", [("glm-5.2_max", 0.9)])})
        rows = fe.get_scores(["deepswe", "gpqa-diamond"], archive=zf, fetch_page=pages(deepswe=hub_page("DeepSWE v1.2")))
        self.assertEqual({r["key"] for r in rows}, {"gpqa_diamond"})

    def test_a_percentage_scale_is_refused(self) -> None:
        zf = archive(board("gpqa-diamond", [("glm-5.2_max", 91.86)]))
        with self.assertRaisesRegex(ValueError, "fraction"):
            fe.get_scores(["gpqa-diamond"], archive=zf)

    def test_a_renamed_column_or_a_missing_file_stops_the_read(self) -> None:
        renamed = {"gpqa_diamond.csv": csv_text([{fe.VERSION_COLUMN: "glm-5.2_max", "Score": 0.9}], [fe.VERSION_COLUMN, "Score"])}
        with self.assertRaisesRegex(ValueError, "missing the column"):
            fe.get_scores(["gpqa-diamond"], archive=archive(renamed))
        with self.assertRaisesRegex(ValueError, "no longer carries"):
            fe.get_scores(["gpqa-diamond"], archive=archive())

    def test_accessibility_reads_as_open_weights(self) -> None:
        self.assertIs(fe.open_weights("Open weights (non-commercial)"), True)
        self.assertIs(fe.open_weights("API access"), False)
        self.assertIs(fe.open_weights("Hosted access (no API)"), False)
        self.assertIsNone(fe.open_weights(""))

    def test_the_catalogue_is_the_scored_groups_with_their_verdict(self) -> None:
        files = {}
        for page in fe.BENCHMARKS:
            files.update(board(page, []))
        files.update(board("gpqa-diamond", [("glm-5.2_max", 0.9), ("gpt-5.6-sol_max", 0.93), ("mystery", 0.3)]))
        self.assertEqual(
            fe.get_catalogue(archive(files)),
            [
                {"model": "GLM-5.2", "open_weights": True},
                {"model": "GPT-5.6 Sol", "open_weights": False},
                {"model": "Mystery", "open_weights": None},
            ],
        )


class TestIngest(unittest.TestCase):
    def test_scores_land_with_the_hub_page(self) -> None:
        doc = {
            "benchmarks": {},
            "models": [{"name": "a", "scores": {"frontierswe_2_0": None, "gpqa_diamond": None}}],
        }
        by_slug = {
            "a": {
                "frontierswe_2_0": {"score": 32.2, "source": fe.page_url("frontierswe")},
                "gpqa_diamond": {"score": 91.86, "source": fe.page_url("gpqa-diamond")},
            }
        }
        matched, updated, _ = update.update_epoch_scores(doc, by_slug)
        self.assertEqual((matched, updated), (1, 2))
        model = doc["models"][0]
        self.assertEqual(model["scores"]["gpqa_diamond"], 91.9)
        self.assertEqual(model["scores_source"]["frontierswe_2_0"], "https://epoch.ai/benchmarks/frontierswe")

    def test_only_mapped_groups_are_read(self) -> None:
        path = Path(tempfile.mkdtemp()) / "mapping.json"
        path.write_text(json.dumps({"GLM-5.2": "glm-5-2", "GPT-5.6 Sol": "__closed_weights__"}), encoding="utf-8")
        payload = [
            {"model": "GLM-5.2", "key": "gpqa_diamond", "score": 91.86, "source": fe.page_url("gpqa-diamond")},
            {"model": "GPT-5.6 Sol", "key": "gpqa_diamond", "score": 93.5, "source": fe.page_url("gpqa-diamond")},
        ]
        proc = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch.object(update, "run_fetch", return_value=proc):
            by_slug = update.fetch_epoch_data(Path("fetch_epoch.py"), path)
        self.assertEqual(list(by_slug), ["glm-5-2"])


class TestCredit(unittest.TestCase):
    """CC BY 4.0 asks for the source and authors to be credited wherever the
    data is used, and llm.html prints llm.json's credits under the benchmarks."""

    def setUp(self) -> None:
        self.doc = json.loads(LLM_JSON.read_text(encoding="utf-8"))

    def test_llm_json_carries_epochs_credit_as_the_fetcher_states_it(self) -> None:
        self.assertIn(fe.EPOCH_CREDIT, self.doc.get("credits", []))

    def test_every_credit_is_complete(self) -> None:
        for credit in self.doc.get("credits", []):
            with self.subTest(credit=credit.get("name")):
                for field in ("name", "url", "license", "license_url", "citation"):
                    self.assertTrue(isinstance(credit.get(field), str) and credit[field].strip(), field)

    def test_no_score_cites_the_hub_without_its_credit(self) -> None:
        roots = [c["url"].rstrip("/") for c in self.doc.get("credits", [])]
        for model in self.doc["models"]:
            for key, url in (model.get("scores_source") or {}).items():
                if isinstance(url, str) and url.startswith(fe.SITE_URL):
                    with self.subTest(model=model["name"], key=key):
                        self.assertTrue(any(url == r or url.startswith(r + "/") for r in roots), url)


if __name__ == "__main__":
    unittest.main()
