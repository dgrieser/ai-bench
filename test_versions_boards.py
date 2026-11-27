#!/usr/bin/env python3
"""Live checks that each leaderboard still serves the revision its fetcher reads.

Run with ./test_versions_boards.py (or ./run_tests.py test_versions_boards.py).
The sibling of test_source_versions.py, built the same way and on its helpers:
those tests watch Artificial Analysis, tbench.ai and Vals; these watch the
first-party boards that publish more than one revision of themselves, or that
could start to.

A fetcher that meets a revision it does not know mostly does the right thing at
refresh time -- skips it and says so on stderr, or files it under no column --
which is exactly why nobody notices. Each test here asks the source what it
publishes *now* and compares that with what the fetcher reads, pinned to the
fetcher's own constants or to _revisions.KNOWN_REVISIONS wherever one exists,
so a code bump that adds a revision moves the expectation with it. Where the
fetcher keeps no constant, the test pins one of its own next to the class and
says which docstring it mirrors.

A failure is a prompt to look, not proof of a switch. Every test docstring says
why it exists and what to change when it fails. In general: a newer revision
needs a column in llm.json, a KNOWN_REVISIONS entry and the fetcher's constant
bumped; a revision that stopped being served needs the fetcher pointed at
wherever it went (or its column frozen).

Not covered, because neither has a versioned column to drift out of:
swe_rebench (one rolling board) and spheron (a price list, not a benchmark).

A host that cannot be reached at all skips; an HTTP error fails. Some hosts
(kernelbench.com, frontierswe.com, lastexam.ai, agents-last-exam.org) sometimes
answer with a Vercel bot checkpoint instead of the page; that is retried a few
times before it counts as a failure.
"""

from __future__ import annotations

import json
import re
import time
import unittest
import urllib.error

import fetch_agents_last_exam
import fetch_arc_prize
import fetch_arena
import fetch_cybergym
import fetch_datacurve
import fetch_deepswe
import fetch_frontiercode
import fetch_frontierswe
import fetch_hle_diamond
import fetch_kernelbench
import fetch_osworld
import fetch_programbench
import fetch_real_swe
import fetch_sec_bench
import fetch_swe_marathon
import fetch_zerobench
from _revisions import KNOWN_REVISIONS, revision_label, revision_rank
from test_source_versions import LiveTestCase, Unreachable, as_tuple, fetch

# fetch_swe_rebench / fetch_spheron: no versioned column, nothing to drift from.

_CHECKPOINT_MARKERS = ("Vercel Security Checkpoint", "vercel-challenge", "_vercel/challenge")


def newest(labels) -> str:
    return max(labels, key=revision_rank)


class BoardTestCase(LiveTestCase):
    """LiveTestCase plus: HTTP errors fail naming the URL, and a Vercel bot
    checkpoint (a 429/403 page, or a 200 carrying the challenge) is retried."""

    def get(self, url: str, attempts: int = 3) -> str:
        for attempt in range(1, attempts + 1):
            try:
                _final, body = fetch(url)
            except Unreachable as exc:
                # A reset mid-transfer is usually transient; a host that stays
                # unreachable is offline, which skips.
                if attempt < attempts:
                    time.sleep(3 * attempt)
                    continue
                self.skipTest(f"offline: {exc}")
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429) and attempt < attempts:
                    time.sleep(3 * attempt)
                    continue
                self.fail(f"{url}: HTTP {exc.code}")
            if any(marker in body for marker in _CHECKPOINT_MARKERS):
                if attempt < attempts:
                    time.sleep(3 * attempt)
                    continue
                self.fail(f"{url}: still a Vercel checkpoint after {attempts} attempts")
            return body
        raise AssertionError("unreachable")

    def get_json(self, url: str):
        body = self.get(url)
        try:
            return json.loads(body)
        except ValueError as exc:
            self.fail(f"{url}: not JSON any more ({exc}); starts {body[:120]!r}")


# --------------------------------------------------------------- SWE-Marathon


class TestSweMarathon(BoardTestCase):
    _cached: dict[str, str] | None = None

    def _boards(self) -> dict[str, str]:
        # Both tests read the same ~1 MB bundle; fetch it once.
        if TestSweMarathon._cached is not None:
            return TestSweMarathon._cached
        page = self.get(fetch_swe_marathon.URL)
        for asset in fetch_swe_marathon.bundle_urls(page):
            bundle = self.get(asset)
            if fetch_swe_marathon._MARKER in bundle:
                TestSweMarathon._cached = fetch_swe_marathon.board_revisions(bundle)
                return TestSweMarathon._cached
        self.fail(f"{fetch_swe_marathon.URL}: no page script carries "
                  f"{fetch_swe_marathon._MARKER!r} any more")

    def test_current_board_is_the_newest_known_revision(self):
        """swe-marathon.org names one board "Current" and the rest "Archive";
        the fetcher reads every one into swe_marathon_<rev>. A new Current
        (v1.3) is read but skipped by update.py until it has a column, and the
        old Current quietly becomes an archive. Fails when the Current board is
        not the newest KNOWN_REVISIONS["swe_marathon"]: add the revision's
        column to llm.json and _revisions.KNOWN_REVISIONS."""
        boards = self._boards()
        current = sorted(label for label, role in boards.items() if role == "current")
        expected = newest(KNOWN_REVISIONS["swe_marathon"])
        self.assertEqual(
            current, [expected],
            f"{fetch_swe_marathon.URL} names {current or 'no'} board(s) Current "
            f"(all: {boards}); expected {expected}, the newest KNOWN_REVISIONS['swe_marathon']",
        )

    def test_every_listed_board_has_a_column(self):
        """Every board the bundle names (current and archive) must map to a
        column, or its rows are dropped at ingest. Fails naming the revision
        without one: add it to KNOWN_REVISIONS and llm.json."""
        boards = self._boards()
        self.assertTrue(boards, f"{fetch_swe_marathon.URL}: the bundle names no boards at all")
        missing = sorted(set(boards) - set(KNOWN_REVISIONS["swe_marathon"]), key=revision_rank)
        self.assertEqual(
            missing, [],
            f"{fetch_swe_marathon.URL} lists boards {missing} with no column; "
            f"KNOWN_REVISIONS['swe_marathon'] = {KNOWN_REVISIONS['swe_marathon']}",
        )


# ---------------------------------------------------------------- KernelBench

# Decks on the homepage that fetch_kernelbench deliberately does not read (its
# docstring's "Not read": CUDA mixes units, Multi has no rows). A deck outside
# both this and BOARDS is new.
KERNELBENCH_IGNORED_DECKS = frozenset({"cuda", "multi"})


class TestKernelBench(BoardTestCase):
    def _decks(self, url: str) -> list[dict]:
        html = self.get(url)
        try:
            return fetch_kernelbench.extract_decks(html, url)
        except ValueError as exc:
            self.fail(f"{url}: {exc}")

    def test_homepage_decks_are_the_ones_known(self):
        """The homepage carries every live deck. The fetcher reads the ones in
        BOARDS at "/" (Mega today) and ignores CUDA and Multi on purpose. Fails
        when a deck appears that is in neither (a new deck to judge, as Hard
        and CUDA were) or a read deck disappears: decide whether it gets a
        column, then update BOARDS or KERNELBENCH_IGNORED_DECKS here."""
        url = fetch_kernelbench.LEADERBOARD_URL
        found = {d.get("key") for d in self._decks(url)}
        read = {b.deck for b in fetch_kernelbench.BOARDS if b.url == url}
        expected = read | KERNELBENCH_IGNORED_DECKS
        self.assertEqual(
            found, expected,
            f"{url} decks: found {sorted(map(str, found))}, expected {sorted(expected)} "
            f"(read {sorted(read)}, ignored {sorted(KERNELBENCH_IGNORED_DECKS)})",
        )

    def test_hard_is_still_the_frozen_archive(self):
        """kernelbench_hard is read from /hard, the archive Hard moved to when
        it was retired on 2026-11-23. Fails when /hard stops serving the hard
        deck on the canonical GPU, or stops calling it archived (it was revived
        and may change under the same column): re-check fetch_kernelbench's
        HARD_URL and docstring."""
        spec = fetch_kernelbench.BOARDS_BY_KEY[fetch_kernelbench.HARD_KEY]
        decks = self._decks(spec.url)
        try:
            fetch_kernelbench.select_board(decks, spec.deck, spec.gpu, spec.url)
        except ValueError as exc:
            self.fail(f"{spec.url}: {exc}")
        (deck,) = [d for d in decks if d.get("key") == spec.deck]
        self.assertIn(
            "archived", str(deck.get("title", "")).lower(),
            f"{spec.url}: deck {spec.deck!r} is titled {deck.get('title')!r}, expected 'archived'",
        )


# --------------------------------------------------------------- FrontierCode

_FRONTIERCODE_PROSE_RE = re.compile(r"FrontierCode\s+v?(\d+\.\d+)\b")


class _FetchFailed(Exception):
    """An HTTP error, raised as something split_file_versions does not swallow."""


class TestFrontierCode(BoardTestCase):
    def test_no_split_file_newer_than_the_known_ones(self):
        """From 2.0 on, each FrontierCode revision ships in a file of its own,
        named in the page's JavaScript; split_file_versions() finds them there.
        Its result is merged with KNOWN_SPLIT_VERSIONS, so a 2.1 would be read
        but skipped by update.py for want of a column. This runs the fetcher's
        own discovery (the page plus its ~20 chunks) and fails on a version not
        in KNOWN_SPLIT_VERSIONS: add it there, to KNOWN_REVISIONS and llm.json."""

        def strict_fetch(url: str) -> str:
            try:
                return fetch(url)[1]
            except urllib.error.HTTPError as exc:
                raise _FetchFailed(f"{url}: HTTP {exc.code}") from exc

        try:
            found = fetch_frontiercode.split_file_versions(fetch=strict_fetch)
        except Unreachable as exc:
            self.skipTest(f"offline: {exc}")
        except _FetchFailed as exc:
            self.fail(str(exc))
        unknown = [v for v in found if v not in fetch_frontiercode.KNOWN_SPLIT_VERSIONS]
        self.assertEqual(
            unknown, [],
            f"{fetch_frontiercode.LEADERBOARD_URL} loads split files {unknown}; "
            f"KNOWN_SPLIT_VERSIONS = {fetch_frontiercode.KNOWN_SPLIT_VERSIONS}",
        )

    def test_page_names_the_newest_known_revision(self):
        """The leaderboard page names its revisions in prose ("FrontierCode
        2.0"). Fails when it names one newer than the newest
        KNOWN_REVISIONS["frontiercode"], or stops naming that one (renamed or
        withdrawn): check the payload and split files, then the column."""
        url = fetch_frontiercode.LEADERBOARD_URL
        named = set(_FRONTIERCODE_PROSE_RE.findall(self.get(url)))
        expected = newest(KNOWN_REVISIONS["frontiercode"])
        newer = sorted((v for v in named if as_tuple(v) > as_tuple(expected)), key=as_tuple)
        self.assertEqual(newer, [], f"{url} names FrontierCode {newer}; newest known is {expected}")
        self.assertIn(expected, named, f"{url} no longer names FrontierCode {expected} (names {sorted(named)})")


# ---------------------------------------------------------------- FrontierSWE

_FRONTIERSWE_VERSION_RE = re.compile(
    r"""(?:href=["']/v(\d+)\b|frontier-?swe[- ]v(\d+)\b)""", re.IGNORECASE
)


class TestFrontierSwe(BoardTestCase):
    def _rows(self, url: str, label: str) -> tuple[list[dict], str]:
        html = self.get(url)
        try:
            entries = fetch_frontierswe.extract_entries(html)
            return fetch_frontierswe.revision_rows(label, entries, fetch_frontierswe.DEFAULT_GROUP), html
        except ValueError as exc:
            self.fail(f"{url} is not the V{label} board any more: {exc}")

    def test_root_is_still_v2(self):
        """frontierswe_2_0 is read from the site root in V2's shape (views
        nested under score mode 'abs', scored by 'overall'). Fails when the
        root's board no longer reads as CURRENT_REVISION -- most likely a V3
        moved in: give it a column and point BOARD_URLS at it."""
        label = fetch_frontierswe.CURRENT_REVISION
        url = fetch_frontierswe.BOARD_URLS[label]
        rows, _html = self._rows(url, label)
        self.assertTrue(rows, f"{url}: the V{label} board parsed to no rows")

    def test_v1_is_still_1_0(self):
        """frontierswe_1_0 is read from /v1, "preserved as published". Fails
        when /v1 is gone or no longer reads as V1 (dominance, no score mode)."""
        url = fetch_frontierswe.BOARD_URLS["1.0"]
        rows, html = self._rows(url, "1.0")
        self.assertTrue(rows, f"{url}: the V1 board parsed to no rows")
        self.assertRegex(html, r"FrontierSWE V1\b", f"{url} no longer calls itself FrontierSWE V1")

    def test_nothing_newer_than_the_known_revisions(self):
        """The root links its archived boards (/v1) and its task repository
        (frontier-swe-v2). Fails when either names a major newer than the
        newest KNOWN_REVISIONS["frontierswe"] (a /v3 page, a frontier-swe-v3
        repo): add the revision to BOARD_URLS, ROW_READERS and the columns."""
        url = fetch_frontierswe.URL
        html = self.get(url)
        majors = {int(a or b) for a, b in _FRONTIERSWE_VERSION_RE.findall(html)}
        ceiling = as_tuple(newest(KNOWN_REVISIONS["frontierswe"]))[0]
        self.assertTrue(majors, f"{url} links no /vN board and no frontier-swe-vN repo any more")
        self.assertLessEqual(
            max(majors), ceiling,
            f"{url} names FrontierSWE V{max(majors)}; newest known is V{ceiling} "
            f"(KNOWN_REVISIONS['frontierswe'] = {KNOWN_REVISIONS['frontierswe']})",
        )


# -------------------------------------------------------------------- DeepSWE


class TestDeepSwe(BoardTestCase):
    def test_version_buttons_are_the_known_revisions(self):
        """deepswe.datacurve.ai toggles between its revisions with one button
        each and opens on the newest artifact; fetch_datacurve reads them all,
        and update.py skips any without a column. Fails when the toggle
        (plus the live artifact) is not exactly KNOWN_REVISIONS["deepswe"]: a
        new one needs a column; a vanished one means its column is frozen."""
        url = fetch_datacurve.SITE_URL
        html = self.get(url)
        dirs = set(fetch_datacurve._VERSION_BUTTON_RE.findall(html))
        dirs |= {d for d, _f in fetch_datacurve._ARTIFACT_RE.findall(html)}
        found = {revision_label(d) for d in dirs}
        self.assertEqual(
            sorted(found, key=revision_rank), list(KNOWN_REVISIONS["deepswe"]),
            f"{url} offers DeepSWE {sorted(found, key=revision_rank)}; "
            f"KNOWN_REVISIONS['deepswe'] = {KNOWN_REVISIONS['deepswe']}",
        )

    def test_benchlm_still_mirrors_the_newest_artifact(self):
        """fetch_deepswe files benchlm.ai's DeepSWE rows under the revision of
        the Datacurve artifact the page links. Fails when it links none (rows
        would be skipped), one with no column, or not the newest known one
        (benchlm lags, or moved past us): check the mirror, then the column."""
        url = fetch_deepswe.URL
        found = fetch_deepswe.extract_revision(self.get(url))
        expected = newest(KNOWN_REVISIONS["deepswe"])
        self.assertEqual(found, expected, f"{url} mirrors DeepSWE {found!r}; expected {expected}")


# ------------------------------------------------------------------- Real-SWE

# What fetch_real_swe's docstring and its "95% CI" column assume the page says.
# Bump with that docstring (and the whisker/harness parsing) when the page moves.
REAL_SWE_RUNS_PER_TASK = "eight"
REAL_SWE_INTERVAL = "95% confidence interval"

_REAL_SWE_RUNS_RE = re.compile(r"averaged over\s+(\w+)\s+independent runs per task", re.IGNORECASE)
_REAL_SWE_INTERVALS = {
    "95% confidence interval": re.compile(r"95\s*%\s*(?:confidence|CI)", re.IGNORECASE),
    "one standard error": re.compile(r"one standard error", re.IGNORECASE),
}


class TestRealSwe(BoardTestCase):
    def test_title_revision_has_a_column(self):
        """realswe.withspecific.com shows one release at a time and names it
        in the title; 1.1 replaced 1.0 at the same address without keeping it.
        Fails when the title names a release that is not the newest
        KNOWN_REVISIONS["real_swe"]: give it a column (the previous one keeps
        what was read before the switch)."""
        url = fetch_real_swe.URL
        try:
            found = fetch_real_swe.page_revision(self.get(url))
        except ValueError as exc:
            self.fail(str(exc))
        expected = newest(KNOWN_REVISIONS["real_swe"])
        self.assertEqual(found, expected, f"{url} title names Real-SWE {found}; expected {expected}")

    def test_parsed_rows_still_carry_harness_and_interval(self):
        """fetch_real_swe reads each row's harness and its interval whisker
        beside the score. Both are decoration, so a markup change empties them
        without any error -- which is how a methodology change (fewer runs, a
        different interval) slips past. Fails when no row carries a harness or
        an interval: re-read the table's columns and the error-bar markup."""
        url = fetch_real_swe.URL
        html = self.get(url)
        try:
            rows = fetch_real_swe.parse_rows(fetch_real_swe.select_leaderboard_table(html))
        except ValueError as exc:
            self.fail(str(exc))
        self.assertTrue(rows, f"{url}: leaderboard table has no rows")
        header = sorted(rows[0])
        with self.subTest(field="harness"):
            self.assertTrue(
                any(fetch_real_swe._text(r.get("harness", "")) for r in rows),
                f"{url}: no row has a harness; table columns are now {header}",
            )
        with self.subTest(field="interval"):
            intervals = [fetch_real_swe._confidence_interval(r.get("resolution rate", "")) for r in rows]
            self.assertTrue(
                any(low is not None for low, _high in intervals),
                f"{url}: no row has a readable interval whisker "
                f"(data-confidence-whisker found {html.count('data-confidence-whisker')} times, "
                f"data-error-bar {html.count('data-error-bar')} times)",
            )

    def test_runs_per_task_and_interval_are_what_the_fetcher_assumes(self):
        """The score is pass@1 averaged over N runs per task, with an interval
        drawn around it; fetch_real_swe documents eight runs and a 95% CI. A
        change there is a change of methodology under the same column. Fails
        naming what the page states now: decide whether the column still holds
        one measurement, then update the fetcher's docstring and the constants
        above."""
        url = fetch_real_swe.URL
        html = self.get(url)
        text = re.sub(r"<[^>]+>", " ", html)
        runs = sorted({m.lower() for m in _REAL_SWE_RUNS_RE.findall(text)})
        kinds = sorted(k for k, rx in _REAL_SWE_INTERVALS.items() if rx.search(text))
        with self.subTest(what="runs per task"):
            self.assertEqual(
                runs, [REAL_SWE_RUNS_PER_TASK],
                f"{url} states runs per task {runs or 'nowhere'}; expected {REAL_SWE_RUNS_PER_TASK!r}",
            )
        with self.subTest(what="interval"):
            self.assertIn(
                REAL_SWE_INTERVAL, kinds,
                f"{url} draws its interval as {kinds or 'nothing recognised'}; expected {REAL_SWE_INTERVAL!r}",
            )


# --------------------------------------------------------- Agents' Last Exam


class TestAgentsLastExam(BoardTestCase):
    def test_no_release_newer_than_known(self):
        """agents-last-exam.org's leaderboard names its release ("ALE v1.1");
        the fetcher reads whatever it names but update.py files only
        KNOWN_REVISIONS["agents_last_exam"]. Fails when the page names a newer
        release, or none at all: add the column, and KNOWN_RELEASES."""
        url = fetch_agents_last_exam.LEADERBOARD_URL
        named = {revision_label(v) for v in
                 fetch_agents_last_exam._RELEASE_LABEL_RE.findall(self.get(url))}
        self.assertTrue(named, f"{url} names no 'ALE vX.Y' release any more")
        expected = newest(KNOWN_REVISIONS["agents_last_exam"])
        unknown = sorted(named - set(KNOWN_REVISIONS["agents_last_exam"]), key=revision_rank)
        self.assertEqual(
            unknown, [],
            f"{url} names ALE {unknown}; KNOWN_REVISIONS['agents_last_exam'] = "
            f"{KNOWN_REVISIONS['agents_last_exam']}",
        )
        self.assertIn(expected, named, f"{url} no longer names ALE v{expected} (names {sorted(named)})")


# -------------------------------------------------------------------- OSWorld


class TestOsworld(BoardTestCase):
    def test_release_versions_are_the_ones_read(self):
        """OSWorld 2.0's results file lists its task-set releases; each feeds
        its own column through fetch_osworld.RELEASES, and an unlisted one is
        skipped with a stderr warning. Fails when releaseVersions differs from
        RELEASES (a new release to give a column, or one withdrawn), or the
        file stops passing check_v2_payload (another benchmark, task count or
        metric)."""
        url = fetch_osworld.OSWORLD_V2_JSON_URL
        payload = self.get_json(url)
        try:
            fetch_osworld.check_v2_payload(payload)
        except ValueError as exc:
            self.fail(str(exc))
        found = payload.get("releaseVersions")
        self.assertEqual(
            sorted(found or []), sorted(fetch_osworld.RELEASES),
            f"{url} releaseVersions = {found}; fetch_osworld.RELEASES reads {sorted(fetch_osworld.RELEASES)}",
        )


# ------------------------------------------------------------------ SEC-bench

_SEC_OVERALL_RE = re.compile(r'"instances":\s*(\d+),\s*"is_overall":\s*true')
_META_REFRESH_RE = re.compile(r'<meta[^>]+http-equiv="refresh"[^>]+url=([^"\s>]+)', re.IGNORECASE)


class TestSecBench(BoardTestCase):
    def test_results_json_carries_exactly_the_tracked_snapshots(self):
        """results.json carries every SEC-bench Pro snapshot; the fetcher reads
        those in SNAPSHOTS and only warns about the rest. Fails when the file
        lists another snapshot (new task set: give it a column) or drops one."""
        payload = self.get_json(fetch_sec_bench.DATA_URL)
        found = sorted((payload.get("snapshots") or {}) if isinstance(payload, dict) else [])
        self.assertEqual(
            found, sorted(fetch_sec_bench.SNAPSHOTS),
            f"{fetch_sec_bench.DATA_URL} carries snapshots {found}; SNAPSHOTS = {sorted(fetch_sec_bench.SNAPSHOTS)}",
        )

    def test_default_version_is_the_one_credited_to_the_root(self):
        """SNAPSHOTS credits one snapshot's scores to the site root, on the
        premise that the root shows the default snapshot. Fails when
        results.json's default_version is another one: swap the page URLs in
        SNAPSHOTS so each score cites the page that shows it."""
        payload = self.get_json(fetch_sec_bench.DATA_URL)
        root = f"{fetch_sec_bench.SITE_URL}/"
        credited = [s for s, (_k, _n, page) in fetch_sec_bench.SNAPSHOTS.items() if page == root]
        found = payload.get("default_version") if isinstance(payload, dict) else None
        self.assertEqual(
            [found], credited,
            f"{fetch_sec_bench.DATA_URL} default_version = {found!r}; SNAPSHOTS credits {credited} to {root}",
        )

    def test_each_snapshot_page_shows_its_snapshot(self):
        """Each snapshot's scores cite the page in SNAPSHOTS. Fails when that
        page shows another snapshot's overall board (a different task count)
        or is only a redirect stub: point SNAPSHOTS' page URL at the page that
        shows it (today 260505 is the root, 260617 has /260617/)."""
        for snapshot, (_key, instances, page) in fetch_sec_bench.SNAPSHOTS.items():
            with self.subTest(snapshot=snapshot):
                html = self.get(page)
                refresh = _META_REFRESH_RE.search(html)
                found = sorted({int(n) for n in _SEC_OVERALL_RE.findall(html)})
                self.assertEqual(
                    found, [instances],
                    f"{page} (credited for snapshot {snapshot}, {instances} tasks) shows overall "
                    f"board(s) of {found or 'no'} tasks"
                    + (f"; it is a redirect to {refresh.group(1)!r}" if refresh else ""),
                )


# ------------------------------------------------------------------ ZeroBench

# The newest question-set version on zerobench.github.io's changelog when the
# board was last checked. fetch_zerobench reads the board without a version
# (zerobench is one column), so a v4 that rewrites questions would be filed
# into it silently.
ZEROBENCH_VERSION = 3

_ZEROBENCH_ENTRY_RE = re.compile(r'class="timeline-head">\s*v(\d+)\b')


class TestZeroBench(BoardTestCase):
    def test_changelog_names_no_newer_question_set(self):
        """The page's updates timeline lists question-set versions ("v3 --
        question text clarified"). Fails when the newest is not
        ZEROBENCH_VERSION: check whether the change alters scores (v3 only
        reworded two questions), decide whether the column needs splitting,
        then bump ZEROBENCH_VERSION."""
        url = fetch_zerobench.URL
        versions = sorted({int(v) for v in _ZEROBENCH_ENTRY_RE.findall(self.get(url))})
        self.assertTrue(versions, f"{url} has no 'vN' changelog entries any more")
        self.assertEqual(
            versions[-1], ZEROBENCH_VERSION,
            f"{url} changelog's newest question set is v{versions[-1]}; pinned v{ZEROBENCH_VERSION}",
        )


# ------------------------------------------------------------------ ARC Prize

_ARC_VERSION_RE = re.compile(r"^v(\d+)_")


class TestArcPrize(BoardTestCase):
    def test_datasets_are_the_known_ones(self):
        """evaluations.json files each row under a datasetId; the fetcher reads
        the semi-private sets in DATASETS and warns about other semi-private
        ones. Fails when a semi-private set is not in DATASETS, one in DATASETS
        is gone, or any dataset names a version newer than DATASETS reads (an
        ARC-AGI-4): give it a column and a DATASETS entry."""
        url = fetch_arc_prize.EVALUATIONS_URL
        payload = self.get_json(url)
        ids = {e.get("datasetId") for e in payload if isinstance(e, dict)} if isinstance(payload, list) else set()
        ids = {i for i in ids if isinstance(i, str)}
        known = set(fetch_arc_prize.DATASETS)
        with self.subTest(check="semi-private sets"):
            semi = {i for i in ids if i.endswith("_Semi_Private")}
            self.assertEqual(sorted(semi), sorted(known), f"{url} semi-private datasets {sorted(semi)}; DATASETS {sorted(known)}")
        with self.subTest(check="no newer version"):
            ceiling = max(int(_ARC_VERSION_RE.match(d).group(1)) for d in known)
            newer = sorted(i for i in ids if (m := _ARC_VERSION_RE.match(i)) and int(m.group(1)) > ceiling)
            self.assertEqual(newer, [], f"{url} has datasets {newer}, newer than ARC-AGI-{ceiling}")


# ---------------------------------------------------------------------- Arena

# Agent Arena signals Arena has retired, with where it said so. arena.ai's
# changelog is kept by hand here (verified 2026-11-27: the page payload served
# to a browser UA carries the entry's text under "November 18, 2026").
# The dataset subset may keep publishing after retirement (it still did on
# 2026-11-27), which is why nothing else notices.
ARENA_RETIRED_SUBSETS = {
    "agent_tool_hallucination": (
        "https://arena.ai/blog/leaderboard-changelog/, 2026-11-18: "
        "'We are retiring the Tool Hallucination signal'"
    ),
}


class TestArena(BoardTestCase):
    def test_each_subset_read_is_still_published(self):
        """fetch_arena reads one 'latest' parquet per subset of
        lmarena-ai/leaderboard-dataset. Fails when a subset's tree no longer
        holds exactly one (renamed, split, or dropped): check the dataset card
        and BOARDS."""
        for subset in sorted({b.subset for b in fetch_arena.BOARDS.values()}):
            with self.subTest(subset=subset):
                url = fetch_arena.TREE_URL.format(subset=subset)
                listing = self.get_json(url)
                try:
                    fetch_arena.latest_path(subset, fetch_json=lambda _u, _l=listing: _l)
                except ValueError as exc:
                    self.fail(f"{url}: {exc}")

    def test_no_column_reads_a_retired_signal(self):
        """A retired Agent Arena signal stops meaning what its column says even
        if the file keeps coming. Fails while BOARDS reads a subset in
        ARENA_RETIRED_SUBSETS: drop the column from BOARDS (keeping stored
        values), or remove the entry here if Arena reinstates it."""
        reading = {key: b.subset for key, b in fetch_arena.BOARDS.items()}
        retired = sorted(k for k, s in reading.items() if s in ARENA_RETIRED_SUBSETS)
        self.assertEqual(
            retired, [],
            "fetch_arena.BOARDS still reads retired signal(s): "
            + "; ".join(f"{k} <- {reading[k]} ({ARENA_RETIRED_SUBSETS[reading[k]]})" for k in retired)
            + f"; dataset: {fetch_arena.DATASET_URL}",
        )


# ------------------------------------------------------------------- CyberGym

# Task counts the two boards are read as, from fetch_cybergym's docstring.
CYBERGYM_INSTANCES = 1507
EXPLOITGYM_TASKS = 869
# ExploitGym row versions: v0 is the retired 898-task set the fetcher drops,
# v1 (or none) the current one. Anything else is a new task set.
EXPLOITGYM_VERSIONS = {None, "v0", "v1"}


class TestCyberGym(BoardTestCase):
    def test_boards_read_exist_with_their_task_counts(self):
        """Both boards are read from one JSON each, and each column assumes
        one task set: CyberGym's level 1 over 1,507 instances, ExploitGym's
        869. Fails when a file loses the list it is read from, ExploitGym's
        instances.total moves, rows name an unknown version, or the CyberGym
        page states another instance count: re-check the task set, then the
        fetcher and the constants above."""
        for key in fetch_cybergym.KEYS:
            with self.subTest(board=key):
                url = fetch_cybergym.data_url(key)
                payload = self.get_json(url)
                try:
                    rows, _dropped = fetch_cybergym.READERS[key](payload)
                except ValueError as exc:
                    self.fail(str(exc))
                self.assertTrue(rows, f"{url}: no rows survive the fetcher's filters")
                if key == fetch_cybergym.EXPLOITGYM_KEY:
                    total = payload["instances"]["total"]
                    self.assertEqual(total, EXPLOITGYM_TASKS, f"{url}: instances.total = {total}")
                    versions = {r.get("version") for r in payload["results"] if isinstance(r, dict)}
                    self.assertLessEqual(versions, EXPLOITGYM_VERSIONS, f"{url}: row versions {versions}")
        with self.subTest(board="cybergym page"):
            page = fetch_cybergym.PAGE_URLS[fetch_cybergym.CYBERGYM_KEY]
            text = re.sub(r"<[^>]+>", " ", self.get(page))
            counts = {int(n.replace(",", "")) for n in re.findall(r"([\d,]+)\s+benchmark instances", text)}
            self.assertEqual(counts, {CYBERGYM_INSTANCES}, f"{page} states {sorted(counts)} benchmark instances")


# ---------------------------------------------------------------- HLE-Diamond


class TestHleDiamond(BoardTestCase):
    def test_results_table_is_present(self):
        """hle_diamond is the post's Model/Accuracy table (highest effort, no
        tools). Fails when that table is missing or duplicated, or its rows no
        longer parse as percentages -- the post was revised: re-read which view
        the column holds before changing fetch_hle_diamond."""
        url = fetch_hle_diamond.URL
        found = fetch_hle_diamond.tables(self.get(url))
        try:
            rows = fetch_hle_diamond._one_table(found, fetch_hle_diamond.RESULTS_HEADERS, required=True)
        except ValueError as exc:
            self.fail(f"{exc} (tables now: {[h for h, _r in found]})")
        scored = [r for r in rows if isinstance(r, list) and len(r) == 2 and fetch_hle_diamond.percent(r[1]) is not None]
        self.assertTrue(scored, f"{url}: the {fetch_hle_diamond.RESULTS_HEADERS} table has no percentage rows")


# --------------------------------------------------------------- ProgramBench

PROGRAMBENCH_TASKS = 200  # fetch_programbench's docstring: "The 200 tasks"


class TestProgramBench(BoardTestCase):
    def test_task_count_and_extended_is_the_full_table(self):
        """fetch_programbench reads /extended/ as the complete table of the
        same 200-task runs the root shows the top of. Fails when either page
        states another task count (a new task set) or the root lists a model
        /extended/ does not (it stopped being the full table): check whether
        "Extended" became a separate split, as FrontierCode's is."""
        models = {}
        for url in (f"{fetch_programbench.SITE_URL}/", fetch_programbench.LEADERBOARD_URL):
            html = self.get(url)
            with self.subTest(page=url):
                self.assertEqual(
                    fetch_programbench.task_count(html), PROGRAMBENCH_TASKS,
                    f"{url} states {fetch_programbench.task_count(html)} tasks; expected {PROGRAMBENCH_TASKS}",
                )
            try:
                rows = fetch_programbench.parse_rows(fetch_programbench.select_leaderboard_table(html))
            except ValueError as exc:
                self.fail(f"{url}: {exc}")
            models[url] = {m for m in (fetch_programbench._text(r.get("model", "")) for r in rows) if m}
        root, extended = models.values()
        self.assertEqual(sorted(root - extended), [],
                         f"{fetch_programbench.LEADERBOARD_URL} lacks models the root lists")


if __name__ == "__main__":
    unittest.main()
