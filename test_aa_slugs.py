#!/usr/bin/env python3
"""Tests for reading a model from several Artificial Analysis slugs.

Run with ./test_aa_slugs.py

AA sometimes tracks one model under more than one slug, each carrying a
different slice of the benchmarks. A list value in the llm -> AA mapping reads
them all; these tests pin the two halves of that: the priority order the list
resolves to, and the per-benchmark merge that fills gaps without ever letting a
later slug override a measured value.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _artificialanalysis_mapping as aa_mapping
import update
from _matching import normalize_slug


class MappingFileTestCase(unittest.TestCase):
    def write_mapping(self, mapping: dict) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "mapping.json"
        tmp.write_text(json.dumps(mapping), encoding="utf-8")
        return tmp


class TestMappingFile(MappingFileTestCase):
    def test_reads_a_string_and_a_list_alike(self) -> None:
        path = self.write_mapping({"one": "one-aa", "two": ["two-new", "two-old"]})
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path),
            {"one": ["one-aa"], "two": ["two-new", "two-old"]},
        )

    def test_leading_slug_stays_the_single_slug_view(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        self.assertEqual(aa_mapping.load_llm_to_aa_mapping(path), {"two": "two-new"})

    def test_mapped_slugs_include_the_trailing_ones(self) -> None:
        path = self.write_mapping({"one": "one-aa", "two": ["two-new", "two-old"]})
        self.assertEqual(
            aa_mapping.mapped_aa_slugs(path), {"one-aa", "two-new", "two-old"}
        )

    def test_junk_values_are_dropped(self) -> None:
        path = self.write_mapping({"a": None, "b": [], "c": [""], "d": ["ok", 7]})
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"d": ["ok"]})

    def test_writing_a_new_pick_keeps_the_rest_of_the_list(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        aa_mapping.add_aa_mapping("two", "two-newest", path)
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path),
            {"two": ["two-newest", "two-new", "two-old"]},
        )

    def test_writing_a_slug_already_listed_changes_nothing(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        aa_mapping.add_aa_mapping("two", "two-old", path)
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path), {"two": ["two-new", "two-old"]}
        )


class TestSentinels(MappingFileTestCase):
    """A sentinel is a marker, never a slug -- and only one of them is an answer.

    propose.py parks a name it could not resolve as __pending__. Read back as a
    slug, that line posed as a mapping: update_artificialanalysis_mapping.py saw
    a truthy value, printed "mapped already" and skipped the model, so a parked
    name was never asked about again. Every other source drops sentinels in its
    load_*_to_slug_mapping and excludes only PENDING from its reviewed set;
    these pin the same two halves here.
    """

    def test_sentinels_are_not_slugs(self) -> None:
        path = self.write_mapping(
            {
                "parked": "__pending__",
                "declined": "__unmappable__",
                "closed": "__closed_weights__",
                "real": "real-aa",
            }
        )
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"real": ["real-aa"]})
        self.assertEqual(aa_mapping.mapped_aa_slugs(path), {"real-aa"})

    def test_a_sentinel_beside_a_slug_leaves_the_slug(self) -> None:
        path = self.write_mapping({"two": ["__pending__", "two-old"]})
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"two": ["two-old"]})

    def test_a_parked_name_is_not_reviewed(self) -> None:
        path = self.write_mapping(
            {
                "parked": "__pending__",
                "declined": "__unmappable__",
                "closed": "__closed_weights__",
                "real": "real-aa",
            }
        )
        self.assertEqual(
            aa_mapping.load_reviewed_llm_names(path),
            {"declined", "closed", "real"},
        )

    def test_closed_weights_can_be_rechecked(self) -> None:
        path = self.write_mapping({"declined": "__unmappable__", "closed": "__closed_weights__"})
        self.assertEqual(
            aa_mapping.load_reviewed_llm_names(path, include_closed=False),
            {"declined"},
        )


class TestResolveAaSlugs(MappingFileTestCase):
    def test_unmapped_model_reads_its_own_slug(self) -> None:
        path = self.write_mapping({})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m"}, path), {"m": ["m"]}
        )

    def test_model_absent_from_aa_is_left_out(self) -> None:
        path = self.write_mapping({})
        self.assertEqual(update.resolve_aa_slugs(["m"], set(), path), {})

    def test_list_resolves_in_its_own_order(self) -> None:
        path = self.write_mapping({"m": ["m-2", "m-1"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m-1", "m-2"}, path), {"m": ["m-2", "m-1"]}
        )

    def test_slugs_missing_from_aa_are_skipped(self) -> None:
        path = self.write_mapping({"m": ["m-2", "m-1"]})
        self.assertEqual(update.resolve_aa_slugs(["m"], {"m-1"}, path), {"m": ["m-1"]})

    def test_own_slug_leads_unless_the_list_places_it(self) -> None:
        path = self.write_mapping({"m": ["m-old"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m", "m-old"}, path), {"m": ["m", "m-old"]}
        )

        path = self.write_mapping({"m": ["m-old", "m"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m", "m-old"}, path), {"m": ["m-old", "m"]}
        )


class TestVariantPolicy(MappingFileTestCase):
    """An unmapped row reads its model's highest-effort reasoning run."""

    NAMES = {
        "glm": "GLM (Non-reasoning)",
        "glm-reasoning": "GLM (Reasoning)",
        "gpt": "GPT (max)",
        "gpt-xhigh": "GPT (xhigh)",
        "gpt-non-reasoning": "GPT (Non-reasoning)",
        "claude": "Claude (Adaptive Reasoning, Max Effort)",
        "claude-non-reasoning": "Claude (Non-reasoning, High Effort)",
        "kimi": "Kimi",
        "kimi-non-reasoning": "Kimi (Non-reasoning)",
        "ds": "DS 0731 (Reasoning, Max Effort)",
        "ds-0420": "DS 0420 (Reasoning, Max Effort)",
        "ds-0420-high": "DS 0420 (Reasoning, High Effort)",
    }

    def resolve(self, slugs, available=None, names=None, mapping=None):
        return update.resolve_aa_slugs(
            slugs,
            set(self.NAMES) if available is None else available,
            self.write_mapping(mapping or {}),
            self.NAMES if names is None else names,
        )

    def test_non_reasoning_bare_slug_yields_to_the_reasoning_run(self) -> None:
        self.assertEqual(self.resolve(["glm"]), {"glm": ["glm-reasoning"]})

    def test_bare_slug_that_is_already_the_top_run_stays(self) -> None:
        self.assertEqual(self.resolve(["gpt"]), {"gpt": ["gpt"]})
        self.assertEqual(self.resolve(["claude"]), {"claude": ["claude"]})
        self.assertEqual(self.resolve(["kimi"]), {"kimi": ["kimi"]})

    def test_another_checkpoint_is_never_a_variant(self) -> None:
        self.assertEqual(self.resolve(["ds"]), {"ds": ["ds"]})

    def test_a_sibling_with_its_own_row_is_left_to_it(self) -> None:
        self.assertEqual(
            self.resolve(["glm", "glm-reasoning"]),
            {"glm": ["glm"], "glm-reasoning": ["glm-reasoning"]},
        )

    def test_a_mapping_entry_is_read_as_written(self) -> None:
        self.assertEqual(self.resolve(["glm"], mapping={"glm": "glm"}), {"glm": ["glm"]})

    def test_suffixes_decide_without_published_names(self) -> None:
        self.assertEqual(self.resolve(["glm"], names={}), {"glm": ["glm-reasoning"]})
        # An unnamed bare slug is AA's default run, never displaced by effort
        # guesses from a suffix alone.
        self.assertEqual(self.resolve(["gpt"], names={}), {"gpt": ["gpt"]})

    def test_name_effort(self) -> None:
        self.assertEqual(update.aa_name_effort("GPT (Non-Reasoning)"), 0)
        self.assertEqual(update.aa_name_effort("X (Non-reasoning, High Effort)"), 0)
        self.assertEqual(update.aa_name_effort("X (low)"), 2)
        self.assertEqual(update.aa_name_effort("X (Reasoning, High Effort)"), 4)
        self.assertEqual(update.aa_name_effort("X (xhigh)"), 5)
        self.assertEqual(update.aa_name_effort("X (Adaptive Reasoning, Max Effort)"), 6)
        self.assertEqual(update.aa_name_effort("X (Reasoning)"), update.AA_DEFAULT_EFFORT)
        self.assertEqual(update.aa_name_effort("X"), update.AA_DEFAULT_EFFORT)


class TestUpdateScoresVariants(unittest.TestCase):
    PAGE = "https://artificialanalysis.ai/models/{}"

    def doc(self, scores, sources):
        return {
            "benchmarks": {},
            "models": [
                {
                    "name": "m",
                    "scores": dict(scores),
                    "scores_updated": {k: "2026-01-01" for k in scores},
                    "scores_source": dict(sources),
                }
            ],
        }

    def run_update(self, doc, records):
        by_slug = {"m": update.merge_aa_models(records)}
        return update.update_scores(doc, by_slug)

    def test_a_zero_is_written(self) -> None:
        doc = self.doc(
            {"critpt": 1.2}, {"critpt": "https://huggingface.co/some/model"}
        )
        self.run_update(doc, [{"slug": "m", "name": "M", "evaluations": {"critpt": 0}}])
        model = doc["models"][0]
        self.assertEqual(model["scores"]["critpt"], 0)
        self.assertEqual(model["scores_source"]["critpt"], self.PAGE.format("m"))

    def test_the_run_read_is_stored(self) -> None:
        doc = self.doc({}, {})
        self.run_update(doc, [{"slug": "m-reasoning", "name": "M (Reasoning)", "evaluations": {}}])
        self.assertEqual(
            doc["models"][0]["aa_variant"], {"slug": "m-reasoning", "name": "M (Reasoning)"}
        )

    def test_a_score_from_a_run_no_longer_read_is_cleared(self) -> None:
        doc = self.doc(
            {"critpt": 3.0, "ifbench": 40.0, "hle": 20.0},
            {
                "critpt": self.PAGE.format("m"),
                "ifbench": self.PAGE.format("m-reasoning"),
                "hle": "https://huggingface.co/some/model",
            },
        )
        self.run_update(doc, [{"slug": "m-reasoning", "evaluations": {}}])
        scores = doc["models"][0]["scores"]
        # Credited to the old variant's page: not this row's run any more.
        self.assertIsNone(scores["critpt"])
        self.assertIsNone(doc["models"][0]["scores_source"]["critpt"])
        # Credited to the page still read: AA has no number today, keep it.
        self.assertEqual(scores["ifbench"], 40.0)
        # Credited to another source altogether: not AA's to clear.
        self.assertEqual(scores["hle"], 20.0)

    def test_a_gap_filled_from_a_second_mapped_slug_is_kept(self) -> None:
        doc = self.doc({"critpt": 3.0}, {"critpt": self.PAGE.format("m-old")})
        self.run_update(doc, [{"slug": "m", "evaluations": {}}, {"slug": "m-old", "evaluations": {}}])
        self.assertEqual(doc["models"][0]["scores"]["critpt"], 3.0)


class TestHandAddedModelMeetsAa(MappingFileTestCase):
    """A model added before Artificial Analysis tracked it.

    The entry exists with no AA mapping at all. What happens next depends on the
    slug AA eventually publishes, and only one of the two cases resolves itself.
    """

    def test_a_slug_aa_does_not_have_yet_reads_nothing(self) -> None:
        path = self.write_mapping({})
        self.assertEqual(update.resolve_aa_slugs(["acme-model-1"], {"other"}, path), {})

    def test_the_same_slug_appearing_on_aa_maps_itself(self) -> None:
        """No mapping entry, no answer to give: the name is the match."""
        path = self.write_mapping({})
        self.assertEqual(
            update.resolve_aa_slugs(["acme-model-1"], {"acme-model-1", "other"}, path),
            {"acme-model-1": ["acme-model-1"]},
        )

    def test_a_declined_name_still_maps_itself_once_aa_has_it(self) -> None:
        """__unmappable__ said AA had no slug for it; AA now says otherwise."""
        path = self.write_mapping({"acme-model-1": "__unmappable__"})
        self.assertEqual(
            update.resolve_aa_slugs(["acme-model-1"], {"acme-model-1"}, path),
            {"acme-model-1": ["acme-model-1"]},
        )

    def test_a_slug_that_differs_only_in_punctuation_does_not_map_itself(self) -> None:
        """Which is why the entry has to be mapped or renamed: this comparison
        is byte for byte, however alike the two names look."""
        path = self.write_mapping({})
        self.assertEqual(update.resolve_aa_slugs(["acme-model-1.5"], {"acme-model-1-5"}, path), {})
        # Either remedy connects them. Renaming leaves nothing to maintain.
        mapped = self.write_mapping({"acme-model-1.5": "acme-model-1-5"})
        self.assertEqual(
            update.resolve_aa_slugs(["acme-model-1.5"], {"acme-model-1-5"}, mapped),
            {"acme-model-1.5": ["acme-model-1-5"]},
        )
        self.assertEqual(
            update.resolve_aa_slugs(["acme-model-1-5"], {"acme-model-1-5"}, self.write_mapping({})),
            {"acme-model-1-5": ["acme-model-1-5"]},
        )

    def test_the_reviewer_is_told_which_case_it_is(self) -> None:
        """update_artificialanalysis_mapping.py used to promise the punctuation
        case would be fetched directly, and then it was not."""
        import update_artificialanalysis_mapping as aa_review

        aa_by_norm = {normalize_slug(s): s for s in ["acme-model-1-5", "other"]}
        self.assertEqual(
            aa_review.exact_aa_slug({"name": "acme-model-1.5"}, aa_by_norm), "acme-model-1-5"
        )
        self.assertIsNone(aa_review.exact_aa_slug({"name": "acme-model-9"}, aa_by_norm))


class TestMergeAaModels(unittest.TestCase):
    def test_leading_record_wins_every_value_it_has(self) -> None:
        merged = update.merge_aa_models(
            [
                {"slug": "m-2", "evaluations": {"hle": 0.4}},
                {"slug": "m-1", "evaluations": {"hle": 0.9}},
            ]
        )
        self.assertEqual(merged["slug"], "m-2")
        self.assertEqual(merged["evaluations"]["hle"], 0.4)

    def test_later_records_fill_gaps_per_benchmark(self) -> None:
        merged = update.merge_aa_models(
            [
                {"evaluations": {"hle": 0.4, "gpqa": None}},
                {"evaluations": {"gpqa": 0.7, "tau2": 0.5}},
            ]
        )
        self.assertEqual(
            merged["evaluations"], {"hle": 0.4, "gpqa": 0.7, "tau2": 0.5}
        )

    def test_zero_is_a_measurement(self) -> None:
        # A 0% is a real floor (CritPt, ZeroBench), so a later record cannot
        # fill it the way it fills a null.
        merged = update.merge_aa_models(
            [{"evaluations": {"hle": 0}}, {"evaluations": {"hle": 0.6}}]
        )
        self.assertEqual(merged["evaluations"]["hle"], 0)

    def test_zero_model_field_is_still_a_gap(self) -> None:
        merged = update.merge_aa_models([{"context": 0}, {"context": "128k"}])
        self.assertEqual(merged["context"], "128k")

    def test_merged_record_lists_the_slugs_read(self) -> None:
        merged = update.merge_aa_models([{"slug": "m"}, {"slug": "m-old"}])
        self.assertEqual(merged["_aa_slugs"], ["m", "m-old"])

    def test_false_is_a_value_not_a_gap(self) -> None:
        merged = update.merge_aa_models(
            [{"microevals_enabled": False}, {"microevals_enabled": True}]
        )
        self.assertIs(merged["microevals_enabled"], False)

    def test_top_level_fields_fill_too(self) -> None:
        merged = update.merge_aa_models(
            [
                {"context": "", "params": "70B"},
                {"context": "128k", "params": "8B"},
            ]
        )
        self.assertEqual(merged["context"], "128k")
        self.assertEqual(merged["params"], "70B")

    def test_records_are_not_mutated(self) -> None:
        first = {"evaluations": {"hle": None}}
        update.merge_aa_models([first, {"evaluations": {"hle": 0.6}}])
        self.assertIsNone(first["evaluations"]["hle"])


if __name__ == "__main__":
    unittest.main()
