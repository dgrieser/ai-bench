#!/usr/bin/env python3
"""Tests for the HLE column's one meaning. Run with ./test_hle_no_tools.py

`hle` is the no-tools column. Artificial Analysis supplies 93% of it from one
run -- 2,158 text-only questions, no tools, pass@1 -- and tools are worth a
median +11.5 points on this benchmark, so a with-tools number landing in the
same column is not a slightly generous reading of the same thing, it is a
different measurement a third of the scale away. Five of them did land, and
they were the column's entire top five (docs/hle-tool-mode-audit-2026-09.md).

They arrived through three doors, and these tests hold all three shut:

  * label aliasing. 23 spellings mapped onto `hle`, six of them explicitly
    with-tools, and the HF ingest resolves an alias collision by keeping the
    *best* number -- which on this benchmark is always the tools run.
  * the Hub's `notes`, which states the mode verbatim and used to be dropped,
    leaving two runs of one card colliding under `cais/hle`.
  * llm-stats' flat `hle_score`, one field over a population its own
    `analysis_method` splits into with-tools, without, and never says.

And the fourth test class holds the *other* half of what makes the column one
thing: AA's numbers win over every other source, and AA's next run replaces
AA's last one rather than being turned away by its own earlier value. A column
that could not be refreshed in place would freeze at whatever AA's
implementation said the day a model was added.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import fetch_huggingface
import fetch_llmstats
import update
from _precedence import HUGGING_FACE_PREFIX, LLMSTATS_SOURCE_URL, may_overwrite

HERE = Path(__file__).resolve().parent
HF_MAPPING = HERE / "huggingface-benchmark-name-mapping.json"
LLMSTATS_MAPPING = HERE / "llmstats-benchmark-name-mapping.json"

AA_PAGE = "https://artificialanalysis.ai/models/glm-5-3"
AA_OTHER_PAGE = "https://artificialanalysis.ai/models/glm-5-3-flash"
HF_CARD = f"{HUGGING_FACE_PREFIX}/deepseek-ai/DeepSeek-V4.1-Flash"
VENDOR_PAGE = "https://www.anthropic.com/news/claude-sonnet-5"

# Any of these in a label means the run had tools. Deliberately broader than
# the mapping needs: a spelling this catches that nobody has written yet is a
# spelling that cannot be mapped onto `hle` by accident later.
WITH_TOOLS_IN_LABEL = re.compile(
    r"w/\s*tools?|with\s+tools?|with\s+search|search\s+agent|tool[- ]augmented",
    re.IGNORECASE,
)


def load(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestNoWithToolsLabelIsMapped(unittest.TestCase):
    """Door one: which source labels are allowed to become an `hle` value."""

    def test_hugging_face_labels(self) -> None:
        for label, key in load(HF_MAPPING).items():
            if key != "hle":
                continue
            with self.subTest(label=label):
                self.assertIsNone(
                    WITH_TOOLS_IN_LABEL.search(label),
                    f"{label!r} names a with-tools run and must not map to hle",
                )

    def test_llmstats_labels(self) -> None:
        for label, key in load(LLMSTATS_MAPPING).items():
            if key != "hle":
                continue
            with self.subTest(label=label):
                self.assertIsNone(WITH_TOOLS_IN_LABEL.search(label))

    def test_the_known_offenders_stay_parked(self) -> None:
        # The six spellings that were mapped, plus HLE-Verified, which is not a
        # tool mode at all but a different question set (llm-stats runs it as
        # its own board) scoring about 9 points above the full set.
        mapping = load(HF_MAPPING)
        for label in (
            "HLE (w/ Tools)",
            "HLE (with tools)",
            "HLE w/ Tools",
            "HLE w/ tool",
            "HLE with search",
            "HLE-Full (w/ tools)",
            "HLE-Verified¹",
            "cais/hle (with tools)",
            # Parked in the provenance pass: each had produced a stored value
            # that was not this column's measurement.
            "ZeroBench (Pass@5)",
            "BrowseComp Top100",
            "BrowseComp+ (OpenClaw)",
            "BrowseComp (\u226410 searches)",
            "open-agent-leaderboard/results (browsecomp_plus)",
            "AIME 25 (w/ Tools)",
            "AIME25 (with tools)",
            "AIME 2026 I",
            "MMLU-Pro (EM)",
            "IFBench (prompt)",
            "IFBench (loose)",
        ):
            with self.subTest(label=label):
                self.assertEqual(mapping.get(label), "__unmappable__")

    def test_the_no_tools_spellings_are_still_mapped(self) -> None:
        # Parking must not have taken the column's own labels with it.
        mapping = load(HF_MAPPING)
        for label in ("HLE (no tools)", "cais/hle", "cais/hle (no tools)"):
            with self.subTest(label=label):
                self.assertEqual(mapping.get(label), "hle")
        self.assertEqual(load(LLMSTATS_MAPPING).get("hle (no tools)"), "hle")

    def test_llmstats_bare_label_is_parked(self) -> None:
        # fetch_llmstats never emits it any more; parked so that a shape change
        # upstream reinstating it is refused rather than ingested.
        self.assertEqual(load(LLMSTATS_MAPPING).get("hle"), "__unmappable__")


class TestHuggingFaceNotes(unittest.TestCase):
    """Door two: one card, two runs, one dataset id, and only `notes` to tell
    them apart."""

    def test_tool_mode_reads_the_phrasings_cards_use(self) -> None:
        for notes, expected in (
            ("With tools", "with tools"),
            ("With tools; harness not specified in the model card.", "with tools"),
            ("w/ tools", "with tools"),
            ("With search", "with tools"),
            ("No tools.", "no tools"),
            ("Text-only, no tools.", "no tools"),
            ("Without tools", "no tools"),
            ("w/o tools", "no tools"),
            ("Pass@1", None),
            ("", None),
            (None, None),
            (42, None),
        ):
            with self.subTest(notes=notes):
                self.assertEqual(fetch_huggingface._tool_mode(notes), expected)

    def test_a_note_describing_both_runs_claims_neither(self) -> None:
        # "With tools: 57.4%. Without tools: 43.2%" is a comparison, not a
        # statement about the value it is attached to, so the entry stays
        # unqualified rather than being filed under a mode it may not be.
        self.assertIsNone(
            fetch_huggingface._tool_mode("With tools: 57.4%. Without tools: 43.2%.")
        )

    def test_eval_results_splits_hle_by_mode(self) -> None:
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "cais/hle", "task_id": "hle"},
                          "value": 36.8, "notes": "No tools."}},
                {"data": {"dataset": {"id": "cais/hle", "task_id": "hle"},
                          "value": 63.9, "notes": "With tools"}},
            ]
        }
        self.assertEqual(
            fetch_huggingface.extract_eval_results(payload),
            {"cais/hle (no tools)": 36.8, "cais/hle (with tools)": 63.9},
        )

    def test_an_unqualified_entry_keeps_the_plain_label(self) -> None:
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "cais/hle", "task_id": "hle"}, "value": 36.8}}
            ]
        }
        self.assertEqual(
            fetch_huggingface.extract_eval_results(payload), {"cais/hle": 36.8}
        )

    def test_agentic_datasets_are_left_alone(self) -> None:
        # Tools are the point of Toolathlon, so qualifying its label would mint
        # a label per phrasing and queue a mapping question for each.
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "hkust-nlp/Toolathlon",
                                      "task_id": "toolathlon_verified"},
                          "value": 71.2, "notes": "With tools"}}
            ]
        }
        self.assertEqual(
            fetch_huggingface.extract_eval_results(payload),
            {"hkust-nlp/Toolathlon (toolathlon_verified)": 71.2},
        )

    def test_only_hle_is_tool_mode_sensitive(self) -> None:
        self.assertEqual(fetch_huggingface.TOOL_MODE_SENSITIVE_DATASETS, {"cais/hle"})


class TestLlmStatsResolution(unittest.TestCase):
    """Door three: one `hle_score` field over three different measurements."""

    @staticmethod
    def _run(payload, detail_for):
        def fake_fetch_json(url, timeout=60):
            if url == fetch_llmstats.URL:
                return payload
            return detail_for(url.rsplit("/", 1)[-1])
        original = fetch_llmstats.fetch_json
        fetch_llmstats.fetch_json = fake_fetch_json
        try:
            return {r["model"]: r["scores"] for r in fetch_llmstats.get_scores()}
        finally:
            fetch_llmstats.fetch_json = original

    def test_a_no_tools_note_keeps_the_headline(self) -> None:
        self.assertEqual(
            fetch_llmstats.hle_no_tools_score(0.252, "text-only, without tools"), 0.252
        )

    def test_a_with_tools_note_drops_the_score(self) -> None:
        for method in (
            "Multidisciplinary reasoning with tools.",
            "With tools; max reasoning effort.",
            "With search and code execution",
            "Search Agent (with tools)",
        ):
            with self.subTest(method=method):
                self.assertIsNone(fetch_llmstats.hle_no_tools_score(0.647, method))

    def test_an_unstated_note_drops_the_score(self) -> None:
        # An unverified number is not a no-tools number, whichever way it leans.
        for method in ("Pass@1", "accuracy", "Text-only", "", None):
            with self.subTest(method=method):
                self.assertIsNone(fetch_llmstats.hle_no_tools_score(0.4, method))

    def test_a_both_modes_note_yields_the_without_tools_figure(self) -> None:
        method = (
            "Multidisciplinary reasoning (2,500 questions). With tools (web search, "
            "web fetch, code execution): 57.4%. Without tools: 43.2%."
        )
        self.assertAlmostEqual(
            fetch_llmstats.hle_no_tools_score(0.574, method), 0.432, places=6
        )

    def test_a_both_modes_note_with_no_figure_drops_the_score(self) -> None:
        # The headline is the tools run, and the note names no other number.
        self.assertIsNone(
            fetch_llmstats.hle_no_tools_score(0.65, "With tools, unlike the no tools run")
        )

    def test_get_scores_never_publishes_the_bare_label(self) -> None:
        payload = [
            {"model_id": "with-tools-model", "license": "proprietary",
             "hle_score": 0.647, "gpqa_score": 0.9},
            {"model_id": "no-tools-model", "license": "apache-2.0", "hle_score": 0.252},
            {"model_id": "both-modes-model", "license": "proprietary", "hle_score": 0.574},
            {"model_id": "unstated-model", "license": "proprietary", "hle_score": 0.4},
        ]
        details = {
            "with-tools-model": "Multidisciplinary reasoning with tools.",
            "no-tools-model": "text-only, without tools",
            "both-modes-model": "With tools: 57.4%. Without tools: 43.2%.",
            "unstated-model": "Pass@1",
        }

        def fake_fetch_json(url, timeout=60):
            if url == fetch_llmstats.URL:
                return payload
            model_id = url.rsplit("/", 1)[-1]
            return {"benchmarks": [{"benchmark_id": fetch_llmstats.HLE_BENCHMARK_ID,
                                    "analysis_method": details[model_id]}]}

        original = fetch_llmstats.fetch_json
        fetch_llmstats.fetch_json = fake_fetch_json
        try:
            results = fetch_llmstats.get_scores()
        finally:
            fetch_llmstats.fetch_json = original

        by_model = {r["model"]: r["scores"] for r in results}
        self.assertNotIn("hle", {label for s in by_model.values() for label in s})
        self.assertNotIn("hle (no tools)", by_model["with-tools-model"])
        self.assertEqual(by_model["no-tools-model"]["hle (no tools)"], 0.252)
        self.assertAlmostEqual(by_model["both-modes-model"]["hle (no tools)"], 0.432)
        # Nothing left to report once its one score is refused.
        self.assertNotIn("unstated-model", by_model)

    def test_the_exact_board_wins_over_the_flat_field(self) -> None:
        # llm-stats runs a board that IS this column -- no tools, text-only --
        # and the flat field is the full multimodal set even when its note says
        # no tools, which is a different question set worth 2-3 points.
        payload = [{"model_id": "m", "license": "mit", "hle_score": 0.368}]
        detail = {"benchmarks": [
            {"benchmark_id": fetch_llmstats.HLE_BENCHMARK_ID,
             "analysis_method": "full multimodal evaluation, no tools", "score": 0.368},
            {"benchmark_id": fetch_llmstats.HLE_TEXT_ONLY_BENCHMARK_ID,
             "analysis_method": "text-only subset, no tools", "score": 0.391},
        ]}
        got = self._run(payload, lambda _mid: detail)
        self.assertAlmostEqual(got["m"]["hle (no tools)"], 0.391)

    def test_the_exact_board_is_published_without_a_flat_score(self) -> None:
        # Free: the detail request has already been made for this model.
        payload = [{"model_id": "m", "license": "mit", "gpqa_score": 0.9}]
        detail = {"benchmarks": [
            {"benchmark_id": fetch_llmstats.HLE_TEXT_ONLY_BENCHMARK_ID,
             "analysis_method": "No tools, text-only", "score": 0.434},
        ]}
        got = self._run(payload, lambda _mid: detail)
        self.assertAlmostEqual(got["m"]["hle (no tools)"], 0.434)

    def test_the_other_no_tools_columns_reject_a_tools_run(self) -> None:
        # Same defect as HLE, smaller: three AIME 2025 entries and one MMMU-Pro
        # on the live board say tools were used.
        payload = [{"model_id": "m", "license": "mit",
                    "aime_2025_score": 0.967, "mmmu_pro_score": 0.801,
                    "scicode_score": 0.5, "gpqa_score": 0.9}]
        detail = {"benchmarks": [
            {"benchmark_id": "aime-2025", "analysis_method": "With tools"},
            {"benchmark_id": "mmmu-pro", "analysis_method": "w/ python"},
            {"benchmark_id": "scicode", "analysis_method": "no tools"},
            {"benchmark_id": "gpqa-diamond", "analysis_method": "Accuracy"},
        ]}
        got = self._run(payload, lambda _mid: detail)
        self.assertNotIn("aime_2025", got["m"])
        self.assertNotIn("mmmu_pro", got["m"])
        # A silent note is not a tools note: these columns keep their coverage.
        self.assertEqual(got["m"]["scicode"], 0.5)
        self.assertEqual(got["m"]["gpqa"], 0.9)

    def test_used_tools_reads_a_code_interpreter_as_a_tool(self) -> None:
        for method in ("w/ python", "with python", "code execution", "code interpreter"):
            with self.subTest(method=method):
                self.assertTrue(fetch_llmstats.used_tools(method))
        for method in ("no tools", "Pass@1", "", None):
            with self.subTest(method=method):
                self.assertFalse(fetch_llmstats.used_tools(method))

    def test_skipping_the_lookup_drops_the_gated_columns(self) -> None:
        # The cheap path must not become a second door for an unverified number.
        payload = [{"model_id": "m", "license": "mit", "hle_score": 0.6,
                    "aime_2025_score": 0.9, "swe_bench_verified_score": 0.7}]
        original = fetch_llmstats.fetch_json
        fetch_llmstats.fetch_json = lambda url, timeout=60: payload
        try:
            results = fetch_llmstats.get_scores(resolve_hle=False)
        finally:
            fetch_llmstats.fetch_json = original
        scores = results[0]["scores"]
        self.assertNotIn("hle", scores)
        self.assertNotIn("aime_2025", scores)
        self.assertEqual(scores["swe_bench_verified"], 0.7)

    def test_the_benchmark_id_is_the_full_board(self) -> None:
        # Its sibling "hle-verified" is a different question set; reading it
        # here would answer the tool-mode question about the wrong run.
        self.assertEqual(fetch_llmstats.HLE_BENCHMARK_ID, "humanity's-last-exam")


class TestArtificialAnalysisKeepsTheColumn(unittest.TestCase):
    """AA's run is the column's definition, so it both wins and refreshes."""

    @staticmethod
    def model_with(score, source):
        return {
            "name": "m",
            "scores": {"hle": score},
            "scores_updated": {"hle": "2026-02-13"},
            "scores_source": {"hle": source},
        }

    def test_nothing_outranks_an_aa_hle_value(self) -> None:
        for url in (HF_CARD, LLMSTATS_SOURCE_URL, VENDOR_PAGE, None):
            with self.subTest(url=url):
                self.assertFalse(may_overwrite(url, AA_PAGE))
        self.assertTrue(may_overwrite(AA_PAGE, HF_CARD))

    def test_a_later_aa_run_replaces_an_earlier_one(self) -> None:
        # The requirement AA's rank alone does not give: rank blocks a write
        # only when the stored value came from a *strictly* better source, so
        # AA over AA is allowed and a regrade lands on the next refresh.
        self.assertTrue(may_overwrite(AA_PAGE, AA_PAGE))
        self.assertTrue(may_overwrite(AA_PAGE, AA_OTHER_PAGE))

        doc = {"benchmarks": {"hle": {"decimals": 1}}}
        model = self.model_with(42.3, AA_PAGE)
        changes: list = []
        written = update.apply_score(
            doc, model, "m", "hle", 44.8, AA_PAGE, changes
        )
        self.assertEqual(written, 1)
        self.assertEqual(model["scores"]["hle"], 44.8)
        self.assertNotEqual(model["scores_updated"]["hle"], "2026-02-13")

    def test_an_unchanged_aa_refresh_does_not_churn_the_date(self) -> None:
        # Why a pre-regrade stamp does not imply a pre-regrade number: the date
        # records the last *change*, so it cannot be read as a vintage.
        doc = {"benchmarks": {"hle": {"decimals": 1}}}
        model = self.model_with(42.3, AA_PAGE)
        changes: list = []
        self.assertEqual(
            update.apply_score(doc, model, "m", "hle", 42.3, AA_PAGE, changes), 0
        )
        self.assertEqual(model["scores_updated"]["hle"], "2026-02-13")

    def test_the_aggregates_cannot_overwrite_anything(self) -> None:
        # Belt and braces on top of rank: both aggregate ingests are fill-only,
        # so a with-tools number that slipped past the labels still could not
        # displace a stored value.
        doc = {"benchmarks": {"hle": {"decimals": 1}}}
        model = self.model_with(42.3, AA_PAGE)
        changes: list = []
        for url in (HF_CARD, LLMSTATS_SOURCE_URL):
            with self.subTest(url=url):
                self.assertEqual(
                    update.apply_score(
                        doc, model, "m", "hle", 62.5, url, changes, fill_only=True
                    ),
                    0,
                )
        self.assertEqual(model["scores"]["hle"], 42.3)


class TestTheColumnSaysWhatItIs(unittest.TestCase):
    def test_description_states_the_tool_mode(self) -> None:
        doc = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
        description = doc["benchmarks"]["hle"]["description"]
        self.assertIn("no-tools", description)
        self.assertIn("without tools", description)


if __name__ == "__main__":
    unittest.main(verbosity=2)
