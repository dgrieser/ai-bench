#!/usr/bin/env python3
"""Live checks that the aggregators still serve the revisions their columns hold.

Run with ./test_versions_aggregators.py. A companion to test_source_versions.py
(whose fetch helpers it reuses, and whose Terminal-Bench, coding-agents,
tbench.ai and Vals checks it does not repeat): these reach the network because
what they guard is which revision a third party serves this week.

Covered here, one or two requests a test:

* Artificial Analysis -- the Intelligence Index version on the methodology
  page's Version History against update.AA_VERSION_GATES (and against the API
  envelope the gate actually reads, when a key is set); the changelog's latest
  entry, pinned below, so a regrade-in-place is noticed; every other gate's
  evaluation page; every _PAGE_FLOAT_FIELDS name on the model pages.
* Epoch AI -- every fetch_epoch.BENCHMARKS file in the archive; every
  revision-routed page still naming a revision llm.json has a column for;
  Mercor's APEX-Agents board, which apex_agents_1_1 follows, still on 1.1.
* llm-stats -- every fetch_llmstats.BOARD_FIELDS board answering, and no
  APEX-Agents note naming a revision newer than the ones it splits into.
* OpenRouter -- the model pages still carrying fetch_openrouter.BENCHMARK.

Each test's docstring says why it exists and what to do when it fails. A host
that cannot be reached skips; an HTTP error fails, since a page that moved is
as much a reason to look as one that changed.
"""

from __future__ import annotations

import io
import json
import os
import re
import socket
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import artificialanalysis
import fetch_epoch
import fetch_llmstats
import fetch_openrouter
import update
from _revisions import KNOWN_REVISIONS, known_revision_key, subset_base
from test_source_versions import HEADERS, LiveTestCase, Unreachable, as_tuple, newer_than

HERE = Path(__file__).resolve().parent

# The latest entry in AA's Intelligence Index Version History that this
# repository has caught up with. 5.0.1 regraded CritPt v2 in place and 5.0.2
# moved Terminal-Bench 4.0 -> 4.1 and regraded FrontierCode-AA 2.0, all under
# the 5.0 the gate holds. Bump it once a new patch's regrades are dealt with.
AA_INDEX_CHANGELOG_LATEST = "5.0.2"
AA_METHODOLOGY_URL = "https://artificialanalysis.ai/methodology/intelligence-benchmarking"

# "<strong>Version 5.0.2</strong></p><p ...>November 2026 to current</p>"
_CHANGELOG_ENTRY_RE = re.compile(
    r">\s*Version\s+(\d+(?:\.\d+)*)\s*</strong>\s*</p>\s*<p[^>]*>([^<]*)</p>", re.IGNORECASE
)

MERCOR_APEX_URL = "https://www.mercor.com/apex/apex-agents-leaderboard/"
# "dataLink":"https://huggingface.co/datasets/mercor/apex-agents-v1.1", plain
# or escaped inside the flight payload.
_DATALINK_RE = re.compile(r'dataLink\\?"\s*:\s*\\?"([^"\\]+)')
# A revision named next to the benchmark's name: "APEX-Agents v1.1",
# "APEX-Agents 1.1 64.2", "apex-agents-v1.1". A bare "APEX-Agents;" names none.
_APEX_NAMED_RE = re.compile(r"apex[- ]?agents[\s-]*(?:v\s*)?(\d+(?:\.\d+)*)(?![\d])", re.IGNORECASE)

# OpenRouter model pages that fed gpqa_diamond, all in the mapping: open-weight
# models, which are the ones OpenRouter benchmarks per endpoint.
OPENROUTER_PROBES = ("xiaomi/mimo-v2.6-flash", "minimax/minimax-m3.1-flash")


def major_minor(version: str) -> str:
    """"5.0.2" -> "5.0": the part of a version a column is named for."""
    parts = version.split(".") + ["0"]
    return f"{int(parts[0])}.{int(parts[1])}"


def changelog_entries(page_html: str) -> list[tuple[str, str]]:
    """[(version, dates), ...] off the Version History, newest first."""
    start = page_html.find('id="version-history"')
    return [(v, d.strip()) for v, d in _CHANGELOG_ENTRY_RE.findall(page_html[max(start, 0):])]


def apex_revisions_named(text: str) -> set[str]:
    return {update.version_label(m) for m in _APEX_NAMED_RE.findall(text or "")}


def _get(url: str, headers: dict[str, str], tries: int = 3) -> str:
    """A page's body. Retries AA's Vercel checkpoint (429/403), raises the rest."""
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) and attempt + 1 < tries:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
            raise Unreachable(f"{url}: {exc}") from exc
    raise AssertionError("unreachable")


class Live(LiveTestCase):
    def body(self, url: str, *, rsc: bool = False, headers: dict | None = None) -> str:
        """The page, or a skip when offline, or a failure naming the HTTP status."""
        merged = {**HEADERS, **({"RSC": "1"} if rsc else {}), **(headers or {})}
        try:
            return _get(url, merged)
        except Unreachable as exc:
            self.skipTest(f"offline: {exc}")
        except urllib.error.HTTPError as exc:
            self.fail(f"{url}: HTTP {exc.code}")


class TestOffline(unittest.TestCase):
    """The parsers on their own, so a failure below is about the source."""

    def test_changelog_entries(self):
        page = ('<h2 id="version-history">x</h2><p><strong class="a">Version 5.0.2</strong></p>'
                '<p class="b">November 2026 to current</p><p><strong>Version 5.0</strong></p>'
                '<p>October 2026</p>')
        self.assertEqual(changelog_entries(page), [("5.0.2", "November 2026 to current"),
                                                   ("5.0", "October 2026")])
        self.assertEqual(major_minor("5.0.2"), "5.0")
        self.assertEqual(major_minor("6"), "6.0")

    def test_apex_revisions_named(self):
        self.assertEqual(apex_revisions_named("APEX-Agents v1.1 score 72.1, apex-agents-v1.2"), {"1.1", "1.2"})
        self.assertEqual(apex_revisions_named("APEX-Agents 2 pass@1"), {"2.0"})
        self.assertEqual(apex_revisions_named("APEX-Agents; Grok 4.6 High."), set())


class TestArtificialAnalysis(Live):
    gate_version = update.AA_VERSION_GATES["intelligence_index"][0]

    def changelog(self) -> list[tuple[str, str]]:
        entries = changelog_entries(self.body(AA_METHODOLOGY_URL))
        if not entries:
            self.fail(f"{AA_METHODOLOGY_URL} carries no Version History entries this test can read")
        return entries

    def test_current_intelligence_index_version_is_the_gated_one(self):
        """update.py writes aa_intelligence_index_5_0 only while
        version_label(the API envelope's intelligence_index_version) equals
        AA_VERSION_GATES["intelligence_index"] ("5.0"). The methodology page's
        Version History marks AA's current version "... to current". A patch
        (5.0.2) is a regrade within the version the column is named for and is
        the next test's business; a new major.minor (5.1, 6.0) is a new index
        that needs a column of its own. On failure: add the
        aa_intelligence_index_X_Y column, move SCORE_MAPPINGS and the gate to
        it, archive the old column as it stands."""
        entries = self.changelog()
        current = next((v for v, dates in entries if "current" in dates.lower()), entries[0][0])
        self.assertEqual(
            major_minor(current), self.gate_version,
            f"{AA_METHODOLOGY_URL} calls Intelligence Index v{current} current; "
            f"update.AA_VERSION_GATES holds aa_intelligence_index at {self.gate_version}",
        )

    def test_changelog_latest_entry_is_the_one_caught_up_with(self):
        """AA regrades evaluations in place under a patch version (CritPt v2 in
        5.0.1, Terminal-Bench 4.1 and FrontierCode-AA 2.0 in 5.0.2) and the gate,
        which holds only 5.0, lets the new numbers straight into the old
        columns. This pins the latest Version History entry so the next patch
        is read before it is absorbed. On failure: read the new entry, check
        which columns' numbers moved and whether any needs a column of its own
        (as terminal_bench_4_1 did), then bump AA_INDEX_CHANGELOG_LATEST."""
        latest, dates = self.changelog()[0]
        self.assertEqual(
            latest, AA_INDEX_CHANGELOG_LATEST,
            f"{AA_METHODOLOGY_URL}: latest Version History entry is v{latest} ({dates}); "
            f"this test was last caught up with v{AA_INDEX_CHANGELOG_LATEST}",
        )

    def test_api_envelope_states_the_gated_version(self):
        """The comparison the gate itself makes: version_label() of the API
        envelope's intelligence_index_version against the gate. version_label
        keeps every part, so an envelope saying 5.0.2 is "5.0.2" and blocks
        the column outright. Needs ARTIFICIAL_ANALYSIS_API_KEY (one request of
        the free tier's budget); skipped without one. On failure: as for the
        changelog test above, or, if AA only started stating patches, decide
        whether the gate should compare major.minor."""
        key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
        if not key:
            self.skipTest("ARTIFICIAL_ANALYSIS_API_KEY is not set")
        url = f"{artificialanalysis.MODELS_FREE_URL}?page=1"
        payload = json.loads(self.body(url, headers={"x-api-key": key}))
        stated = update.version_label(payload.get("intelligence_index_version"))
        self.assertEqual(
            stated, self.gate_version,
            f"{url} states intelligence_index_version "
            f"{payload.get('intelligence_index_version')!r} (-> {stated!r}); "
            f"the gate holds {self.gate_version}",
        )

    def test_gated_evaluation_pages_name_their_version_and_nothing_newer(self):
        """Each other AA_VERSION_GATES entry is read off its evaluation page
        (update.AA_PAGE_VERSIONS) at refresh time, and a mismatch silently
        stops its columns. This puts it on the Tests tab, and is stricter: any
        newer version the page's pattern finds fails, not just the most cited.
        On failure: the evaluation was re-versioned in place; give the new
        version a column, move SCORE_MAPPINGS and the gate's version to it."""
        for gate, (expected, columns) in update.AA_VERSION_GATES.items():
            if gate == "intelligence_index":
                continue  # the API envelope and the changelog tests above
            with self.subTest(gate=gate):
                self.assertIn(gate, update.AA_PAGE_VERSIONS,
                              f"{gate} has no page to read its version from, so it never opens")
                url, pattern = update.AA_PAGE_VERSIONS[gate]
                found = {update.version_label(v) for v in pattern.findall(self.body(url, rsc=True))}
                self.assertIn(expected, found,
                              f"{url} no longer names {gate} {expected} "
                              f"(found {sorted(found, key=as_tuple) or 'none'}); {', '.join(columns)}")
                self.assertEqual(newer_than(found, expected), [],
                                 f"{url} names {gate} {', '.join(newer_than(found, expected))}; "
                                 f"{', '.join(columns)} hold {expected}")

    def test_every_page_float_field_is_still_on_the_model_pages(self):
        """artificialanalysis._PAGE_FLOAT_FIELDS names the payload keys the
        model pages are read by, and a renamed key matches nothing without a
        word -- the column just stops being fed. The same check as
        `artificialanalysis.py --audit-page-fields`, on two of its AUDIT_SLUGS
        (a key counts if either page carries it). On failure: run that audit
        for the full picture and update the stale names."""
        seen: set[str] = set()
        for slug in artificialanalysis.AUDIT_SLUGS[:2]:
            url = artificialanalysis.MODEL_PAGE_URL.format(slug)
            text = artificialanalysis._normalize_page_text(self.body(url, rsc=True))
            chunk = artificialanalysis._metrics_chunk(text, slug)
            self.assertTrue(chunk, f"{url} has no metrics record for {slug}")
            seen.update(key for key, _ in artificialanalysis._PAGE_FLOAT_FIELDS if f'"{key}"' in chunk)
        missing = [f"{key} -> {internal}" for key, internal in artificialanalysis._PAGE_FLOAT_FIELDS
                   if key not in seen]
        self.assertEqual(missing, [], f"no AA model page ({', '.join(artificialanalysis.AUDIT_SLUGS[:2])}) "
                                      f"carries: {', '.join(missing)}")


class TestEpoch(Live):
    def test_every_benchmark_file_is_still_in_the_archive(self):
        """fetch_epoch reads each BENCHMARKS file out of benchmark_data.zip and
        refuses the benchmark when one is gone or lost its score column. Also
        fails when Epoch's own benchmark_metadata.csv marks a file read here as
        superseded (FrontierMath v1 -> v2 was one). On failure: find the file's
        new name or successor; a successor is a new revision and needs a column."""
        try:
            raw = fetch_epoch.fetch_bytes(fetch_epoch.DATA_URL, retries=2)
        except urllib.error.HTTPError as exc:
            self.fail(f"{fetch_epoch.DATA_URL}: HTTP {exc.code}")
        except (urllib.error.URLError, OSError) as exc:
            self.skipTest(f"offline: {fetch_epoch.DATA_URL}: {exc}")
        archive = zipfile.ZipFile(io.BytesIO(raw))
        for page, bench in fetch_epoch.BENCHMARKS.items():
            with self.subTest(page=page):
                try:
                    fetch_epoch.read_csv(archive, bench.file, (fetch_epoch.VERSION_COLUMN, bench.column))
                except ValueError as exc:
                    self.fail(f"{page}: {exc}")
        superseded = fetch_epoch.report_superseded(archive, list(fetch_epoch.BENCHMARKS))
        self.assertEqual(superseded, [], f"{fetch_epoch.DATA_URL}: Epoch marks {superseded} superseded")

    def test_revision_routed_pages_still_name_a_revision_with_a_column(self):
        """deepswe, frontierswe and frontiercode are filed under the revision
        fetch_epoch.page_revision() reads off Epoch's page; a page naming none,
        or one KNOWN_REVISIONS has no column for, drops the benchmark at refresh
        with only a stderr line. On failure: if the page changed wording, teach
        page_revision() the new one; if it moved to a new revision, add the
        column and its KNOWN_REVISIONS entry."""
        for page, bench in fetch_epoch.BENCHMARKS.items():
            if not bench.base:
                continue
            with self.subTest(page=page):
                url = fetch_epoch.page_url(page)
                revision = fetch_epoch.page_revision(self.body(url))
                base = subset_base(bench.base, bench.subset)
                self.assertIsNotNone(
                    known_revision_key(base, revision),
                    f"{url} names revision {revision!r}; fetch_epoch reads {base} only at "
                    f"{', '.join(KNOWN_REVISIONS.get(base, ()))}",
                )

    def test_mercor_apex_agents_board_is_still_the_revision_read(self):
        """Epoch's apex-agents file feeds apex_agents_1_1 with no revision on
        the row or the page title: it follows Mercor's board, whose dataLink
        names the dataset (mercor/apex-agents-v1.1). On failure: Mercor moved
        to a new revision; give it a column and point fetch_epoch.BENCHMARKS
        ["apex-agents"] (and llm-stats' split) at it."""
        expected = update.version_label(fetch_epoch.BENCHMARKS["apex-agents"].key.rsplit("agents_", 1)[1])
        links = [link for link in _DATALINK_RE.findall(self.body(MERCOR_APEX_URL)) if "apex-agents" in link]
        self.assertTrue(links, f"{MERCOR_APEX_URL} carries no APEX-Agents dataLink")
        found = set().union(*(apex_revisions_named(link) for link in links))
        self.assertIn(expected, found, f"{MERCOR_APEX_URL} dataLink(s) {links}; expected v{expected}")
        self.assertEqual(newer_than(found, expected), [],
                         f"{MERCOR_APEX_URL} dataLink(s) {links} are newer than the v{expected} read")


class TestLlmStats(Live):
    def board(self, benchmark_id: str) -> list:
        url = fetch_llmstats.BOARD_URL.format(benchmark_id=urllib.parse.quote(benchmark_id, safe=""))
        payload = json.loads(self.body(url))
        models = payload.get("models") if isinstance(payload, dict) else None
        self.assertTrue(isinstance(models, list) and models, f"{url} lists no models")
        return models

    def test_every_board_still_answers(self):
        """fetch_llmstats.board_entries() turns a dead board into an empty one
        with a warning, so a renamed benchmark_id simply stops feeding its
        column. On failure: find the board's new id on llm-stats.com and
        update BOARD_FIELDS."""
        for benchmark_id in sorted(set(fetch_llmstats.BOARD_FIELDS.values())):
            with self.subTest(board=benchmark_id):
                self.board(benchmark_id)

    def test_apex_agents_notes_name_no_newer_revision(self):
        """The apex-agents board is split into apex_agents_1_0 / _1_1 by
        apex_revision(), which knows only those two and files anything after
        APEX_AGENTS_1_1_DATE as 1.1 -- so a "v1.2" note would land in 1.1.
        On failure: add the revision's column, label and apex_revision rule."""
        newest = max(fetch_llmstats._APEX_LABEL_REVISIONS.values(), key=as_tuple)
        url = fetch_llmstats.BOARD_URL.format(benchmark_id=fetch_llmstats.BOARD_FIELDS["apex_agents_1_1"])
        newer = {}
        for entry in self.board(fetch_llmstats.BOARD_FIELDS["apex_agents_1_1"]):
            ahead = newer_than(apex_revisions_named(entry.get("analysis_method")), newest)
            if ahead:
                newer[entry.get("model_id")] = ahead
        self.assertEqual(newer, {}, f"{url}: notes name APEX-Agents revisions newer than {newest}: {newer}")


class TestOpenRouter(Live):
    def test_model_pages_still_carry_the_benchmark_read(self):
        """fetch_openrouter reads rows whose benchmark_type is BENCHMARK
        ("gpqa_diamond"); rows of any other type are skipped, so a rename
        leaves every model with no score and nothing failing. On failure: see
        what the pages carry now (the message lists it); a different question
        set or grading (e.g. "gpqa_diamond_unsat") is a different measurement,
        so decide whether gpqa_diamond may take it before renaming BENCHMARK."""
        carried: dict[str, list[str]] = {}
        for model in OPENROUTER_PROBES:
            url = fetch_openrouter.MODEL_PAGE_URL.format(model=model)
            query = fetch_openrouter.benchmark_query(self.body(url))
            self.assertIsNotNone(query, f"{url} has no benchmarkScores query")
            rows = ((query.get("state") or {}).get("data") or {}).get("scores") or []
            types = sorted({row.get("benchmark_type") for row in rows if isinstance(row, dict)}, key=str)
            if fetch_openrouter.BENCHMARK in types:
                return
            carried[url] = types
        self.fail(f"no probed OpenRouter page carries benchmark_type "
                  f"{fetch_openrouter.BENCHMARK!r}; they carry {carried}")


if __name__ == "__main__":
    unittest.main()
