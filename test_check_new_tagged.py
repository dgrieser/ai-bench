#!/usr/bin/env python3
"""check_new.py reads AA once, tagged, instead of an --open and a --closed run.
Run with ./test_check_new_tagged.py

The one run has to come back as the two did: the same open models, the same
closed ones, in the same order, with the same verdict per model -- and the
page reads that verdict needs are done once, side by side, not twice in turn.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import artificialanalysis as aa
import check_new

MODELS = [
    {"slug": "open-a", "release_date": "2026-11-01"},
    {"slug": "closed-b", "release_date": "2026-11-02"},
    {"slug": "unknown-c", "release_date": "2026-11-03"},
    {"slug": "open-d", "release_date": "2026-11-04"},
    {"slug": "closed-e", "release_date": "2026-11-05", "licensing": {"is_open_weights": False}},
]
OPEN = {"open-a", "open-d"}


def fake_page(slug: str, creator_name: str = "") -> dict:
    url = f"https://huggingface.co/x/{slug}" if slug in OPEN else ""
    return {"hugging_face_url": url, "page_read": slug != "unknown-c"}


def run(*flags: str) -> list[dict]:
    """artificialanalysis.py's main() on MODELS, --output json, no network."""
    pages: list[str] = []

    cache: dict[str, dict] = {}

    def page(slug: str, creator_name: str = "") -> dict:
        # Cached per slug like the real _fetch_page_metrics, so a second
        # lookup of one page is free and only first reads are counted.
        if slug not in cache:
            pages.append(slug)
            cache[slug] = fake_page(slug, creator_name)
        return cache[slug]

    out = io.StringIO()
    with mock.patch.dict(os.environ, {"ARTIFICIAL_ANALYSIS_API_KEY": "k"}), \
            mock.patch.object(aa, "_fetch_models",
                              return_value=({"data": [dict(m) for m in MODELS]}, None)), \
            mock.patch.object(aa, "_save_cache"), \
            mock.patch.object(aa, "_fetch_page_metrics", side_effect=page), \
            mock.patch.object(aa, "_enrich_structured_metrics"), \
            mock.patch.object(sys, "argv", ["artificialanalysis.py", "-o", "json", *flags]), \
            redirect_stdout(out):
        aa.main()
    run.pages = pages  # type: ignore[attr-defined]
    return json.loads(out.getvalue())["data"]


class TestTaggedRun(unittest.TestCase):
    def test_one_tagged_run_is_the_open_and_closed_runs(self) -> None:
        open_run = run("--open")
        closed_run = run("--closed")
        tagged = run("--tag-weights")
        split = check_new.split_tagged(tagged)
        self.assertEqual(split, open_run + [{**m, "_closed": True} for m in closed_run])
        self.assertEqual([m["slug"] for m in split], ["open-a", "open-d", "closed-b", "unknown-c", "closed-e"])

    def test_the_pages_are_read_once_each(self) -> None:
        """Prefetched side by side, then served from memory to the filter."""
        prefetched: list[list[str]] = []
        real = aa._prefetch_pages
        with mock.patch.object(aa, "_prefetch_pages",
                               side_effect=lambda ms: prefetched.append([m["slug"] for m in ms])
                               or real(ms)):
            run("--tag-weights")
        free_tier = [m["slug"] for m in MODELS if "licensing" not in m]
        # Every page the verdicts need, and only those, in one prefetch.
        self.assertEqual(prefetched, [free_tier])
        self.assertEqual(sorted(run.pages), sorted(free_tier))  # type: ignore[attr-defined]

    def test_a_model_neither_open_nor_closed_is_dropped(self) -> None:
        with mock.patch.object(aa, "_is_open_source",
                               side_effect=lambda m: None if m["slug"] == "unknown-c"
                               else m["slug"] in OPEN):
            slugs = [m["slug"] for m in run("--tag-weights")]
        self.assertNotIn("unknown-c", slugs)

    def test_split_handles_untagged_rows(self) -> None:
        self.assertEqual(check_new.split_tagged([{"slug": "x"}]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
