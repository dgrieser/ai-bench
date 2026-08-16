#!/usr/bin/env python3
"""Tests for collect mode. Run with ./test_prompts.py (stdlib unittest only).

The load-bearing guarantee is that collect mode records no answers: a regression
there would silently bury hundreds of source names as __unmappable__ on the next
scheduled run, and nobody would be prompted for them again. test_freeze_invariant
drives every mapping module generically so a newly added source cannot quietly
escape the guard.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import _prompts

HERE = Path(__file__).resolve().parent

# Discovered, not listed. A hand-maintained table silently missed
# _toolathlon_mapping.py when that source was added, which defeated the point of
# the freeze test -- so the modules and their writers are found on disk instead.
IGNORED_WRITERS = {
    # Takes a list of slugs rather than one value; covered separately.
    "add_ignored_aa_suggestions",
}


def mapping_modules() -> list[str]:
    return sorted(p.stem for p in HERE.glob("_*_mapping.py"))


def mapping_writers():
    """(label, func, arity, path_kwarg) for every add_* writer in every module."""
    for module_name in mapping_modules():
        module = importlib.import_module(module_name)
        for attr in sorted(vars(module)):
            if not attr.startswith("add_") or attr in IGNORED_WRITERS:
                continue
            func = getattr(module, attr)
            if not callable(func) or getattr(func, "__module__", None) != module_name:
                continue
            params = list(inspect.signature(func).parameters)
            # add_x_mapping(key, value, path=...) vs add_x_unmappable(key, path=...)
            arity = len(params) - 1
            if arity not in (1, 2):
                continue
            yield f"{module_name}.{attr}", func, arity, params[-1]


class CollectModeBase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.get(_prompts.ENV_VAR)
        self.tmp = Path(tempfile.mkdtemp())
        self.report = self.tmp / "pending.jsonl"
        os.environ.pop(_prompts.ENV_VAR, None)
        _prompts._seen.clear()

    def tearDown(self) -> None:
        os.environ.pop(_prompts.ENV_VAR, None)
        if self._saved_env is not None:
            os.environ[_prompts.ENV_VAR] = self._saved_env
        _prompts._seen.clear()

    def collect(self) -> None:
        _prompts.enable(self.report)
        _prompts.reset(self.report)


class TestActivation(CollectModeBase):
    def test_off_by_default(self) -> None:
        self.assertIsNone(_prompts.collecting())
        self.assertFalse(_prompts.freeze_decisions())

    def test_enable_sets_env_for_subprocesses(self) -> None:
        _prompts.enable(self.report)
        self.assertEqual(os.environ[_prompts.ENV_VAR], str(self.report))
        self.assertEqual(_prompts.collecting(), self.report)
        self.assertTrue(_prompts.freeze_decisions())

    def test_blank_env_is_off(self) -> None:
        os.environ[_prompts.ENV_VAR] = "   "
        self.assertIsNone(_prompts.collecting())

    def test_reset_truncates(self) -> None:
        self.report.write_text("stale\n", encoding="utf-8")
        _prompts.reset(self.report)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "")


class TestRecord(CollectModeBase):
    def test_noop_outside_collect_mode(self) -> None:
        _prompts.record(kind="mapping", subject="x", question="q?")
        self.assertFalse(self.report.exists())

    def test_appends_entry(self) -> None:
        self.collect()
        _prompts.record(
            kind="mapping",
            subject="Qwen3",
            question="Map Qwen3?",
            candidates=["qwen3-32b"],
            default="qwen3-32b",
            note="hi",
            command="./x.py -w",
        )
        entry = json.loads(self.report.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["subject"], "Qwen3")
        self.assertEqual(entry["candidates"], ["qwen3-32b"])
        self.assertEqual(entry["default"], "qwen3-32b")
        self.assertEqual(entry["command"], "./x.py -w")

    def test_dedupes_on_command_kind_subject(self) -> None:
        self.collect()
        for _ in range(3):
            _prompts.record(kind="mapping", subject="X", question="q?", command="./a -w")
        _prompts.record(kind="mapping", subject="X", question="q?", command="./b -w")
        _prompts.record(kind="new-model", subject="X", question="q?", command="./a -w")
        self.assertEqual(len(self.report.read_text(encoding="utf-8").splitlines()), 3)


class TestLoad(CollectModeBase):
    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(_prompts.load(self.tmp / "nope.jsonl"), [])

    def test_skips_malformed_and_non_object_lines(self) -> None:
        self.report.write_text(
            '{"command":"./a","kind":"mapping","subject":"ok"}\n'
            "not json at all\n"
            "\n"
            "[1,2,3]\n",
            encoding="utf-8",
        )
        entries = _prompts.load(self.report)
        self.assertEqual([e["subject"] for e in entries], ["ok"])

    def test_dedupes_across_appends(self) -> None:
        # Two processes appending the same question: _seen is per-process, so the
        # cross-run dedupe has to happen on read.
        line = '{"command":"./a","kind":"mapping","subject":"dup"}\n'
        self.report.write_text(line + line, encoding="utf-8")
        self.assertEqual(len(_prompts.load(self.report)), 1)


class TestFreezeInvariant(CollectModeBase):
    def writers(self):
        return mapping_writers()

    def test_every_module_contributes_a_writer(self) -> None:
        # Guards the discovery itself: if a module stops matching the add_* shape
        # its writers would silently drop out of both tests below.
        covered = {label.split(".")[0] for label, *_ in self.writers()}
        self.assertEqual(covered, set(mapping_modules()))
        self.assertGreaterEqual(len(list(self.writers())), 2 * len(mapping_modules()))

    def test_writers_persist_outside_collect_mode(self) -> None:
        for label, func, arity, kw in self.writers():
            with self.subTest(label):
                target = self.tmp / "off.json"
                target.write_text("{}\n", encoding="utf-8")
                args = ("Source Name", "llm-slug")[:arity]
                func(*args, **{kw: target})
                self.assertNotEqual(
                    json.loads(target.read_text(encoding="utf-8")), {}, f"{label} did not write"
                )

    def test_writers_frozen_in_collect_mode(self) -> None:
        self.collect()
        for label, func, arity, kw in self.writers():
            with self.subTest(label):
                target = self.tmp / "on.json"
                target.write_text("{}\n", encoding="utf-8")
                args = ("Source Name", "llm-slug")[:arity]
                func(*args, **{kw: target})
                self.assertEqual(
                    json.loads(target.read_text(encoding="utf-8")), {}, f"{label} still wrote!"
                )

    def test_aa_ignored_suggestions_frozen(self) -> None:
        import _artificialanalysis_mapping as aa

        target = self.tmp / "ignored.json"
        target.write_text("{}\n", encoding="utf-8")
        aa.add_ignored_aa_suggestions("model", ["slug"], path=target)
        self.assertNotEqual(json.loads(target.read_text(encoding="utf-8")), {})

        target.write_text("{}\n", encoding="utf-8")
        self.collect()
        aa.add_ignored_aa_suggestions("model", ["slug"], path=target)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {})

    def test_check_new_dismiss_frozen(self) -> None:
        import check_new

        self.collect()
        before = check_new.DISMISSED_FILE.read_text(encoding="utf-8")
        check_new.dismiss("some-brand-new-model")
        self.assertEqual(check_new.DISMISSED_FILE.read_text(encoding="utf-8"), before)


class TestPromptHooks(CollectModeBase):
    def test_select_returns_none_and_records(self) -> None:
        from add import prompt_select_or_new

        self.collect()
        answer = prompt_select_or_new(
            "Map SWE-Rebench model 'Qwen3-Next-80B' to llm.json model",
            ["qwen3-next-80b-a3b", "glm-5-air"],
        )
        self.assertIsNone(answer)
        entries = _prompts.load(self.report)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["subject"], "Qwen3-Next-80B")
        self.assertIn("qwen3-next-80b-a3b", entries[0]["candidates"])

    def test_benchmark_key_prompt_returns_none(self) -> None:
        from add import prompt_key_for_label

        self.collect()
        self.assertIsNone(
            prompt_key_for_label("HF label", "SWE-bench Verified", ["swe_bench_verified", "hle"])
        )
        self.assertEqual(_prompts.load(self.report)[0]["subject"], "SWE-bench Verified")

    def test_subject_falls_back_to_label_without_quotes(self) -> None:
        from add import subject_from_label

        self.assertEqual(subject_from_label("Map X 'foo' to Y"), "foo")
        self.assertEqual(subject_from_label("OSWorld model"), "OSWorld model")

    def test_subject_keeps_an_apostrophe_in_the_name(self) -> None:
        # Truncating here queues a name no source reports: the answer is recorded
        # against the stub, the real name stays unreviewed, and it comes back
        # every run for good.
        from add import subject_from_label

        self.assertEqual(
            subject_from_label("Map HF label 'Frontier agentic tasks Agents' Last Exam' to X"),
            "Frontier agentic tasks Agents' Last Exam",
        )

    def test_caller_supplied_subject_wins_over_the_label(self) -> None:
        from add import prompt_select_or_new, subject_from_label

        self.collect()
        name = "Agents' Last Exam"
        # A label the fallback cannot read back correctly, so this only passes if
        # the caller's name is the one recorded.
        label = f"Map HF label '{name}' to llm.json benchmark 'hle'"
        self.assertNotEqual(subject_from_label(label), name)
        self.assertIsNone(prompt_select_or_new(label, ["hle"], subject=name))
        self.assertEqual(_prompts.load(self.report)[0]["subject"], name)

    def test_every_mapping_script_queues_the_whole_name(self) -> None:
        """Each source's own prompt, replayed with an apostrophe in the name."""
        import importlib

        name = "Qwen's Own Model"
        cases = [
            ("update_huggingface_mapping", "prompt_key_for_label", (name, ["hle"])),
            ("update_rebench_mapping", "prompt_slug_for_rebench_name", (name, [])),
            ("update_osworld_mapping", "prompt_slug_for_osworld_name", (name, [])),
            ("update_deepswe_mapping", "prompt_slug_for_deepswe_name", (name, [])),
            ("update_frontierswe_mapping", "prompt_slug_for_frontierswe_name", (name, [])),
            ("update_frontiercode_mapping", "prompt_slug_for_frontiercode_name", (name, [])),
            ("update_swe_atlas_mapping", "prompt_slug_for_swe_atlas_name", (name, [])),
            ("update_evals_report_mapping", "prompt_slug_for_evals_report_name", (name, [])),
            ("update_swe_marathon_mapping", "prompt_slug_for_swe_marathon_name", (name, [])),
            ("update_toolathlon_mapping", "prompt_slug_for_toolathlon_name", (name, [])),
            ("update_spheron_mapping", "prompt_slug_for_spheron_path", (name, [], None)),
        ]
        for module_name, func_name, args in cases:
            with self.subTest(module_name):
                self.collect()
                prompt = getattr(importlib.import_module(module_name), func_name)
                self.assertIsNone(prompt(*args))
                entries = _prompts.load(self.report)
                self.assertEqual([e["subject"] for e in entries], [name])


class TestRendering(CollectModeBase):
    def entries(self) -> list[dict]:
        return [
            {
                "command": "./update_rebench_mapping.py -w",
                "kind": "mapping",
                "subject": "Qwen3-Next-80B",
                "question": "Map it",
                "candidates": ["qwen3-next-80b-a3b", "glm-5"],
                "default": "qwen3-next-80b-a3b",
                "note": None,
            },
            {
                "command": "./check_new.py",
                "kind": "new-model",
                "subject": "deepseek-v4",
                "question": "Add it",
                "candidates": [],
                "default": None,
                "note": "released 2026-07-28 - DeepSeek",
            },
        ]

    def test_empty_queue(self) -> None:
        import pending_prompts

        self.assertIn("Nothing pending", pending_prompts.render_markdown([]))
        self.assertEqual(pending_prompts.render_text([]), "nothing pending\n")

    def test_markdown_groups_by_command(self) -> None:
        import pending_prompts

        out = pending_prompts.render_markdown(self.entries())
        self.assertIn("### `./check_new.py`", out)
        self.assertIn("### `./update_rebench_mapping.py -w`", out)
        self.assertIn("suggested: `qwen3-next-80b-a3b`", out)
        # the default must not be repeated in the candidate list
        self.assertIn("candidates: `glm-5`", out)
        self.assertIn("released 2026-07-28 - DeepSeek", out)
        self.assertIn("**2** question(s)", out)

    def test_markdown_is_stable_across_input_order(self) -> None:
        # The workflow diffs this against the live issue body to decide whether
        # to notify, so ordering must not depend on how the report was appended.
        import pending_prompts

        entries = self.entries()
        self.assertEqual(
            pending_prompts.render_markdown(entries),
            pending_prompts.render_markdown(list(reversed(entries))),
        )

    def test_text_lists_every_subject(self) -> None:
        import pending_prompts

        out = pending_prompts.render_text(self.entries())
        self.assertIn("Qwen3-Next-80B", out)
        self.assertIn("deepseek-v4", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
