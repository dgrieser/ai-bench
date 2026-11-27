#!/usr/bin/env python3
"""Live checks that the Scale Labs, Toolathlon, BFCL, Vals and evals.report
sources still serve the benchmark revision their fetchers read.

Run with ./test_versions_scale_vals.py. Like test_source_versions.py, whose
helpers these reuse, these reach the network: what they guard is which
revision a third party is publishing this week, which no file here records.

Every fetcher below already refuses a page it can *tell* is wrong -- another
board's slug, a missing table, a version stamp it does not pin. What none of
them notices is a source moving on beside the page it reads: a "v1.2" board
next to SWE Atlas 1.1, a Vibe Code Bench v1.1.1 that archives the v1.1 page
the fetcher still happily reads, a BFCL-v5 link above a CSV that still has
V4's columns. Nothing fails then; the column simply stops tracking the
benchmark it is named for. These fail as soon as a source names a revision
newer than the one its column measures, or stops naming that one.

Expectations are pinned to the fetchers' own constants (fetch_*.VARIANTS,
fetch_bfcl.SERIES, fetch_vals.VERSIONS, _revisions.KNOWN_REVISIONS) or to
llm.json's own description of the column, so bumping the code to the new
revision moves the test with it.

A failure is a prompt to look, not proof of a switch: each test's docstring
says what to check and what to change. A host that cannot be reached at all
skips; an HTTP error fails, because a page that moved is as much a reason to
look as a page that changed. A Vercel bot checkpoint is retried, then skipped.
"""

from __future__ import annotations

import html
import json
import re
import time
import unittest
import urllib.error
from pathlib import Path

import fetch_bfcl
import fetch_evals_report
import fetch_mcp_atlas
import fetch_swe_atlas
import fetch_swe_bench_pro_v2
import fetch_toolathlon
import fetch_vals
import test_source_versions as tsv
from _revisions import KNOWN_REVISIONS, SUBSET_BASES, revision_label, revision_rank
from _scale_labs import NoRowsError, decode_flight

HERE = Path(__file__).resolve().parent

# Vercel answers some requests with a bot checkpoint instead of the page; a
# later request usually gets through.
CHECKPOINT_ATTEMPTS = 3
CHECKPOINT_DELAY = 3.0
_CHECKPOINT_RE = re.compile(r"Vercel Security Checkpoint|x-vercel-challenge", re.IGNORECASE)


def llm_benchmark(key: str) -> dict:
    """llm.json's entry for one column -- what the repository says it holds."""
    data = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
    return data["benchmarks"][key]


def column_revision(key: str) -> str | None:
    """"swe_bench_pro_2_0" -> "2.0", "frontiercode_1_1" -> "1.1"; None for an
    unversioned key. The inverse of _revisions.revision_key's suffix."""
    match = re.search(r"_(\d+)_(\d+)$", key)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def newest(labels) -> str:
    return max(labels, key=revision_rank)


def page_text(markup: str) -> str:
    """Visible text of an HTML fragment, scripts and styles dropped."""
    markup = re.sub(r"<(script|style)\b.*?</\1>", " ", markup, flags=re.DOTALL | re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Scale Labs: every page carries the site's full board list in its flight
# payload, as {"name":...,"id":...,"airtableTableName":...,"deprecated":...}.
# A new revision of a board shows up there first, under a new id or table.

_BOARD_RE = re.compile(
    r'\{"name":"((?:[^"\\]|\\.)*)","id":"([^"]*)","airtableTableName":"([^"]*)",'
    r'"deprecated":(true|false)'
)


def scale_boards(decoded: str) -> list[dict]:
    """Every board in the site's board list: name, id, table, deprecated."""
    seen: dict[str, dict] = {}
    for name, board_id, table, deprecated in _BOARD_RE.findall(decoded):
        seen[board_id] = {
            "name": json.loads(f'"{name}"'),
            "id": board_id,
            "table": table,
            "deprecated": deprecated == "true",
        }
    return list(seen.values())


# "SWE-Bench Pro V2", "swe_bench_pro_public_v2", "SWE-Bench Pro V2.1",
# "swe-bench-pro-v3". Only a "v" followed by digits is a revision.
_SWE_PRO_VERSION_RE = re.compile(
    r"swe[-_ ]?bench[-_ ]?pro[-_ ]?(?:public[-_ ]?)?v(\d+)(?:[._](\d+))?(?!\d)", re.IGNORECASE
)
# "MCP-Atlas v2", "MCP Atlas 2.0", "mcp_atlas_v2". A task count in
# parentheses ("MCP-Atlas (1,000 tasks)") is not a revision.
_MCP_VERSION_RE = re.compile(
    r"mcp[-_ ]?atlas[-_ ]?(?:v|version\s*)(\d+(?:\.\d+)*)|mcp[-_ ]?atlas[-_ ](\d+\.\d+)",
    re.IGNORECASE,
)
# "sweatlas_1.1 Overall", "SWE Atlas v1.2", "swe-atlas-v1-2".
_SWE_ATLAS_VERSION_RE = re.compile(
    r"swe[-_ ]?atlas[-_ ]?(?:v|version\s*)?(\d+)[._-](\d+)(?!\d)", re.IGNORECASE
)


class ScaleTestCase(tsv.LiveTestCase):
    def get(self, url: str) -> str:
        """Body of ``url``: skip when unreachable or walled, fail on HTTP errors."""
        for attempt in range(1, CHECKPOINT_ATTEMPTS + 1):
            try:
                _final, body = self.page(url)
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < CHECKPOINT_ATTEMPTS:
                    time.sleep(CHECKPOINT_DELAY * attempt)
                    continue
                self.fail(f"{url}: HTTP {exc.code}")
            if not _CHECKPOINT_RE.search(body[:20000]):
                return body
            if attempt < CHECKPOINT_ATTEMPTS:
                time.sleep(CHECKPOINT_DELAY * attempt)
        self.skipTest(f"{url}: still behind a bot checkpoint after {CHECKPOINT_ATTEMPTS} attempts")

    def get_with_rows(self, url: str, read):
        """read(body) over a fresh page, fetched once more if Scale served it
        without rows for a moment (see _scale_labs.fetch_board_rows)."""
        for attempt in (1, 2):
            body = self.get(url)
            try:
                return body, read(body)
            except NoRowsError as exc:
                if attempt == 2:
                    self.fail(f"{url}: {exc}")
                time.sleep(10)
            except ValueError as exc:
                self.fail(f"{url}: {exc}")
        raise AssertionError("unreachable")


class TestOffline(unittest.TestCase):
    """Offline: the helpers themselves, so a failure below is about the source."""

    def test_column_revision_reads_the_suffix(self):
        self.assertEqual(column_revision("swe_bench_pro_2_0"), "2.0")
        self.assertEqual(column_revision("frontiercode_1_1"), "1.1")
        self.assertIsNone(column_revision("mcp_atlas"))
        self.assertIsNone(column_revision("bfcl_v4"))

    def test_board_list_parses(self):
        decoded = (
            '[{"name":"SWE Atlas","id":"swe-atlas","airtableTableName":"sweatlas_1.1 Overall",'
            '"deprecated":false,"icon":"x"},{"name":"MCP Atlas","id":"mcp_atlas",'
            '"airtableTableName":"mcp_atlas","deprecated":true}]'
        )
        boards = scale_boards(decoded)
        self.assertEqual([b["id"] for b in boards], ["swe-atlas", "mcp_atlas"])
        self.assertEqual(boards[0]["table"], "sweatlas_1.1 Overall")
        self.assertTrue(boards[1]["deprecated"])

    def test_version_patterns(self):
        self.assertEqual(
            {m.group(1) for m in _SWE_PRO_VERSION_RE.finditer("SWE-Bench Pro V2 swe_bench_pro_public_v2_hard swe-bench-pro-v3")},
            {"2", "3"},
        )
        self.assertIsNone(_MCP_VERSION_RE.search("MCP-Atlas (1,000 tasks) MCP Atlas public set"))
        self.assertIsNotNone(_MCP_VERSION_RE.search("MCP-Atlas v2"))
        self.assertEqual(
            {f"{a}.{b}" for a, b in _SWE_ATLAS_VERSION_RE.findall("sweatlas_1.1 Overall, SWE Atlas v1.2")},
            {"1.1", "1.2"},
        )

    def test_every_evals_report_table_is_classified(self):
        """Every evals.report table fetch_evals_report reads is either checked
        below (EVALS_REPORT_CHECKS) or listed as naming no revision
        (EVALS_REPORT_UNVERSIONED). A new BENCHMARKS entry fails this until
        someone decides which -- add a check when its column is versioned or
        a subset."""
        classified = set(EVALS_REPORT_CHECKS) | set(EVALS_REPORT_UNVERSIONED)
        self.assertEqual(
            set(fetch_evals_report.BENCHMARKS) - classified, set(),
            "fetch_evals_report.BENCHMARKS has tables no version check covers",
        )


# ---------------------------------------------------------------------------


class TestSweAtlas(ScaleTestCase):
    def test_current_page_names_the_newest_known_revision(self):
        """fetch_swe_atlas reads the current page's revision off the board
        list ("sweatlas_1.1 Overall") and files rows only under a revision
        _revisions.KNOWN_REVISIONS has a column for; anything else is fetched,
        warned about on stderr, and not filed. This puts that on the Tests
        tab, and also fails when the site lists a newer SWE Atlas board beside
        the one read, or the page lost one of CURRENT_VARIANTS' tables.

        When it fails: if Scale published a new revision, add it to
        KNOWN_REVISIONS for swe_atlas and every swe_atlas_* track, add the
        matching llm.json columns, and move the old revision's per-track
        boards into LEGACY_* if Scale keeps them as v1.1 was kept."""
        url = fetch_swe_atlas.CURRENT_URL
        expected = newest(KNOWN_REVISIONS["swe_atlas"])
        body, (revision, tables) = self.get_with_rows(url, fetch_swe_atlas.read_current)
        self.assertEqual(
            revision, expected,
            f"{url} names SWE Atlas revision {revision!r}; the newest in "
            f"_revisions.KNOWN_REVISIONS['swe_atlas'] is {expected!r}",
        )
        for variant, base in fetch_swe_atlas.CURRENT_VARIANTS.items():
            with self.subTest(variant=variant):
                self.assertTrue(tables[variant], f"{url}: variant {variant!r} has no rows")
                self.assertIn(
                    expected, KNOWN_REVISIONS.get(base, ()),
                    f"the current page's {variant} table feeds {base}, which has no "
                    f"{expected} column in KNOWN_REVISIONS",
                )

        decoded = decode_flight(body)
        named = {f"{int(a)}.{int(b)}" for a, b in _SWE_ATLAS_VERSION_RE.findall(decoded)}
        newer = tsv.newer_than(named, expected)
        self.assertEqual(
            newer, [],
            f"{url} names SWE Atlas {', '.join(newer)}; the newest revision read is {expected}",
        )
        known_ids = {fetch_swe_atlas.CURRENT_SLUG} | {
            f"sweatlas-{track}" for track in fetch_swe_atlas.LEGACY_TRACKS
        }
        listed = {
            b["id"] for b in scale_boards(decoded)
            if re.search(r"swe[-_ ]?atlas", b["name"] + " " + b["id"], re.IGNORECASE)
        }
        self.assertTrue(listed, f"{url}: no SWE Atlas boards in the site's board list -- layout changed")
        self.assertEqual(
            listed - known_ids, set(),
            f"{url} lists SWE Atlas boards {sorted(listed - known_ids)} that "
            f"fetch_swe_atlas does not read (it reads {sorted(known_ids)})",
        )

    def test_legacy_boards_still_answer(self):
        """The v1.0 per-track boards (LEGACY_URL) are frozen but still read
        into swe_atlas_*_1_0; Scale marks them "Legacy". This fails when one
        stops serving its rows -- a 404, or a redirect to another board.

        When it fails: check the board in a browser. Scale has 404'd these
        for hours and recovered, so re-run first. If it is gone for good, the
        1.0 columns stop moving (scores are applied, never cleared) and the
        track can be dropped from LEGACY_TRACKS."""
        for track in fetch_swe_atlas.LEGACY_TRACKS:
            with self.subTest(track=track):
                url = fetch_swe_atlas.LEGACY_URL.format(track=track)
                _body, rows = self.get_with_rows(
                    url, lambda body, t=track: fetch_swe_atlas.extract_rows(body, t)
                )
                self.assertTrue(rows, f"{url}: no rows")


class TestSweBenchProV2(ScaleTestCase):
    def test_page_carries_its_variants_and_names_nothing_newer(self):
        """fetch_swe_bench_pro_v2 reads VARIANTS (full, hard) off one page into
        columns spelled at revision 2.0. Scale numbers SWE-Bench Pro revisions
        in the board name and slug ("SWE-Bench Pro V2",
        swe_bench_pro_public_v2); a V2.1 or V3 could replace this board or
        sit beside it unread. Fails when the page loses a variant, names a
        newer SWE-Bench Pro revision, or the site lists another live
        SWE-Bench Pro board.

        When it fails: if Scale shipped a new revision, give it columns
        (swe_bench_pro_<x>_<y> and the hard subset) and a fetcher of its own
        or a new URL/VARIANTS here; never point VARIANTS' 2.0 columns at it."""
        url = fetch_swe_bench_pro_v2.URL
        revisions = {column_revision(key) for key in fetch_swe_bench_pro_v2.VARIANTS.values()}
        self.assertEqual(len(revisions), 1, f"VARIANTS spell several revisions: {revisions}")
        expected = revisions.pop()

        def read(body):
            return {v: fetch_swe_bench_pro_v2.extract_rows(body, v) for v in fetch_swe_bench_pro_v2.VARIANTS}

        body, tables = self.get_with_rows(url, read)
        for variant, rows in tables.items():
            with self.subTest(variant=variant):
                self.assertTrue(rows, f"{url}: variant {variant!r} has no rows")

        decoded = decode_flight(body)
        named = {revision_label(f"{m.group(1)}.{m.group(2) or 0}") for m in _SWE_PRO_VERSION_RE.finditer(decoded)}
        self.assertIn(expected, named, f"{url} no longer names SWE-Bench Pro V{expected} (found {sorted(named)})")
        newer = tsv.newer_than(named, expected)
        self.assertEqual(
            newer, [],
            f"{url} names SWE-Bench Pro {', '.join(newer)}; VARIANTS read {expected}",
        )
        live = {
            b["id"] for b in scale_boards(decoded)
            if not b["deprecated"] and re.search(r"swe[-_ ]?bench[-_ ]?pro", b["name"] + " " + b["id"], re.IGNORECASE)
        }
        self.assertEqual(
            live, {fetch_swe_bench_pro_v2.BOARD_SLUG},
            f"{url}: the site's live SWE-Bench Pro boards are {sorted(live)}; "
            f"fetch_swe_bench_pro_v2 reads {fetch_swe_bench_pro_v2.BOARD_SLUG!r}",
        )


class TestMcpAtlas(ScaleTestCase):
    """mcp_atlas names no revision; llm.json's description pins the
    configuration it holds instead (task set, tool-call budget, the April
    2026 re-score). These fail when Scale's page stops describing that."""

    def expected_config(self) -> dict[str, str]:
        entry = llm_benchmark("mcp_atlas")
        settings = " | ".join(entry.get("settings", []))
        subset = re.search(r"public (\d+)-task subset", settings)
        budget = re.search(r"(\d+) tool calls/task", settings)
        since = re.search(r"under the (\w+ \d{4}) configuration", entry["description"])
        self.assertTrue(
            subset and budget and since,
            "llm.json's mcp_atlas no longer states its task subset, tool-call budget and "
            "configuration date in the form this test reads; update expected_config()",
        )
        return {"subset": subset.group(1), "budget": budget.group(1), "since": since.group(1)}

    def test_page_still_describes_the_configuration_read(self):
        """Scale re-scores MCP-Atlas in place under the same URL and slug: in
        April 2026 it swapped the 20-turn cap for a 100-tool-call budget and a
        new judge, and said so in a dated paragraph. A second such paragraph,
        a changed budget, a dropped public-subset description, a revision
        number, or a second live MCP Atlas board would all leave mcp_atlas
        quietly blending two configurations.

        When it fails: read the page. A new configuration needs either a new
        column or a decision to replace mcp_atlas wholesale (with llm.json's
        description and settings updated, which re-pins this test)."""
        url = fetch_mcp_atlas.URL
        expected = self.expected_config()
        body, _rows = self.get_with_rows(url, fetch_mcp_atlas.extract_rows)
        decoded = decode_flight(body)
        text = " ".join(json.loads(f'"{t}"') for t in re.findall(r'"text":"((?:[^"\\]|\\.)*)"', decoded))
        text = re.sub(r"\s+", " ", text)

        updates = set(re.findall(r"In (\w+ \d{4}), Scale updated the MCP-Atlas evaluation", text))
        self.assertEqual(
            updates, {expected["since"]},
            f"{url} describes MCP-Atlas evaluation updates of {sorted(updates) or 'none'}; "
            f"llm.json's mcp_atlas holds the {expected['since']} configuration",
        )
        budgets = set(re.findall(r"(\d+) max tool calls per task", text))
        self.assertEqual(
            budgets, {expected["budget"]},
            f"{url} states a tool-call budget of {sorted(budgets) or 'none'}; "
            f"mcp_atlas holds {expected['budget']} tool calls/task",
        )
        subsets = set(re.findall(r"public leaderboard subset contains (\d+) tasks", text))
        self.assertEqual(
            subsets, {expected["subset"]},
            f"{url} describes a public subset of {sorted(subsets) or 'no stated size'}; "
            f"mcp_atlas holds the public {expected['subset']}-task subset",
        )
        versions = [m.group(0) for m in _MCP_VERSION_RE.finditer(decoded)]
        self.assertEqual(versions, [], f"{url} names an MCP-Atlas revision: {sorted(set(versions))}")
        live = {
            b["id"] for b in scale_boards(decoded)
            if not b["deprecated"] and re.search(r"mcp[-_ ]?atlas", b["name"] + " " + b["id"], re.IGNORECASE)
        }
        self.assertEqual(
            live, {fetch_mcp_atlas.BOARD_SLUG},
            f"{url}: the site's live MCP Atlas boards are {sorted(live)}; "
            f"fetch_mcp_atlas reads {fetch_mcp_atlas.BOARD_SLUG!r}",
        )

    def test_leaderboard_ranks_the_task_set_the_column_claims(self):
        """The page carries a comparison table, "Pass Rate % (All 1000)"
        beside "Pass Rate % (Public <n>)", for a dozen models. Whichever
        column the leaderboard's own scores reproduce is the task set the
        leaderboard -- and so mcp_atlas -- actually ranks. Fails when that is
        not the public subset llm.json says mcp_atlas holds.

        When it fails: either Scale moved the leaderboard to another task set
        (a new configuration: see the test above), or llm.json's mcp_atlas
        description and settings were wrong about which set it holds and
        should be corrected -- the scores are what they are."""
        url = fetch_mcp_atlas.URL
        expected = self.expected_config()["subset"]
        body, rows = self.get_with_rows(url, fetch_mcp_atlas.extract_rows)
        decoded = decode_flight(body)
        scores = {round(float(r["score"]), 1) for r in rows}

        header_re = re.compile(r'"cells":\[("Model",(?:"[^"]*",?)+)\],"isHeader":true')
        header = header_re.search(decoded)
        if header is None:
            self.fail(f"{url}: the All/Public comparison table is gone; cannot tell which task set is ranked")
        columns = json.loads(f"[{header.group(1)}]")
        table = decoded[header.end():]
        table = table[: table.find('"_type":"leaderboardTable"')] if '"_type":"leaderboardTable"' in table else table[:20000]
        cells = [json.loads(f"[{c}]") for c in re.findall(r'"cells":\[((?:"[^"]*",?)+)\]', table)]
        cells = [c for c in cells if len(c) == len(columns)]
        matches = {}
        for i, name in enumerate(columns[1:], 1):
            values = [float(c[i].rstrip("%")) for c in cells if re.fullmatch(r"\d+(?:\.\d+)?%?", c[i])]
            matches[name] = sum(round(v, 1) in scores for v in values)
        ranked = max(matches, key=matches.get)
        self.assertRegex(
            ranked, rf"\bPublic {expected}\b",
            f"{url}: the leaderboard's scores reproduce the {ranked!r} column "
            f"(matches per column: {matches}); llm.json's mcp_atlas says it holds "
            f"the public {expected}-task subset",
        )


# ---------------------------------------------------------------------------


class TestToolathlon(ScaleTestCase):
    # "Toolathlon v2", "Toolathlon 2.0", "Toolathlon-Verified v1.1",
    # "Toolathlon-Verified 2".
    _VERSION_RE = re.compile(
        r"Toolathlon(?:-Verified)?[\s-]*(?:v|version\s*)\d+(?:\.\d+)?|"
        r"Toolathlon(?:-Verified)?[\s-]+\d+(?:\.\d+)?\b(?!\s*(?:tasks|models|%))",
        re.IGNORECASE,
    )

    def test_current_table_is_still_the_series_the_column_holds(self):
        """fetch_toolathlon reads the table classed CURRENT_TABLE_CLASS and
        nothing else, trusting the site to keep that class on the series
        llm.json's toolathlon column is named for (Toolathlon-Verified). A
        newer series would most likely move into the same slot under the
        same class, and be read as Verified. Fails when the text introducing
        the current table stops naming that series, or the page names any
        other Toolathlon-<Name> series or a numbered revision.

        When it fails: if a new series replaced Verified, it needs a column of
        its own (Verified's numbers are not comparable across a re-grade); add
        it to llm.json and either point a new fetcher at it or teach this one
        which column each table feeds."""
        url = fetch_toolathlon.URL
        series = llm_benchmark("toolathlon")["name"]
        body = self.get(url)
        try:
            fetch_toolathlon.select_current_table(body)
        except ValueError as exc:
            self.fail(f"{url}: {exc}")
        start = body.find(fetch_toolathlon.CURRENT_TABLE_CLASS)
        intro = page_text(body[:start])[-1200:]
        self.assertIn(
            series, intro,
            f"{url}: the text introducing the {fetch_toolathlon.CURRENT_TABLE_CLASS} table "
            f"no longer names {series!r}: ...{intro[-400:]!r}",
        )
        names = {n.rstrip(".") for n in re.findall(r"Toolathlon-[A-Z][\w.]*", body)}
        self.assertEqual(
            names - {series}, set(),
            f"{url} names {sorted(names - {series})}; the toolathlon column holds {series}",
        )
        versions = sorted({m.group(0) for m in self._VERSION_RE.finditer(page_text(body))})
        self.assertEqual(versions, [], f"{url} names a numbered Toolathlon revision: {versions}")


class TestBfcl(ScaleTestCase):
    _SERIES_ANY_RE = re.compile(r"bfcl[-_ ]?v(\d+)", re.IGNORECASE)

    def test_page_and_csv_name_nothing_newer_than_the_series_read(self):
        """fetch_bfcl.check_series refuses a leaderboard page whose newest
        "BFCL-vN" link is not SERIES, and parse_rows refuses a CSV without
        V4's own columns; both only run at refresh time. This puts them on
        the Tests tab, and also catches a newer series spelled another way
        ("BFCL V5", bfcl_v5 in a blog slug) or named in the CSV's header.

        When it fails: a V5 is a new score series (its Overall Acc averages
        other categories). Add a bfcl_v5 column and move fetch_bfcl.SERIES /
        V4_COLUMNS / the mappings to it; bfcl_v4 keeps what it has."""
        page_url, csv_url = fetch_bfcl.LEADERBOARD_URL, fetch_bfcl.CSV_URL
        page = self.get(page_url)
        try:
            fetch_bfcl.check_series(page)
        except ValueError as exc:
            self.fail(str(exc))
        named = {int(n) for n in self._SERIES_ANY_RE.findall(page)}
        newer = sorted(n for n in named if n > fetch_bfcl.SERIES)
        self.assertEqual(
            newer, [],
            f"{page_url} names BFCL v{', v'.join(map(str, newer))}; fetch_bfcl.SERIES is {fetch_bfcl.SERIES}",
        )

        csv_text = self.get(csv_url)
        try:
            fetch_bfcl.parse_rows(csv_text)
        except ValueError as exc:
            self.fail(str(exc))
        header = csv_text.splitlines()[0] if csv_text else ""
        in_header = sorted({int(n) for n in re.findall(r"(?i)\bv(\d+)\b", header)} - {fetch_bfcl.SERIES})
        self.assertEqual(
            in_header, [],
            f"{csv_url}'s header names series v{in_header}; fetch_bfcl.SERIES is {fetch_bfcl.SERIES}: {header!r}",
        )


# ---------------------------------------------------------------------------


class TestValsSuperseded(ScaleTestCase):
    _BANNER_RE = re.compile(r"Archived Benchmark|has been superseded|superseded by", re.IGNORECASE)

    def test_no_pinned_board_is_superseded(self):
        """fetch_vals.VERSIONS pins boards whose slug names no revision, and
        test_source_versions checks the stamped version still matches. That
        passes on a board Vals has *archived* in favour of a newer revision at
        another slug: the old page keeps its stamp, so the pin holds while the
        column stops tracking the current benchmark. Vals says so on the page
        -- an "Archived Benchmark" banner, and a version selector whose
        newest entry is another slug. Fails on either.

        When it fails: the new revision lives at the slug the selector names.
        Give it a column (vibe_code_bench_1_1_1, say), add that slug to
        fetch_vals.BENCHMARKS (and VERSIONS if its slug names no revision
        either), and keep the old slug only if its column should keep filling
        -- as cyber-v1 does for CyberBench 1.0 and vibe-code for Vibe Code
        Bench 1.1. A board kept that way goes in fetch_vals.ARCHIVED: its
        banner is then expected, and what is checked instead is that the
        revision superseding it is read too, under the version it is labelled
        with."""
        for slug, pinned in fetch_vals.VERSIONS.items():
            with self.subTest(slug=slug):
                url = fetch_vals.benchmark_url(slug)
                body = self.get(url)
                archived = slug in fetch_vals.ARCHIVED
                banner = self._BANNER_RE.search(page_text(body))
                if banner and not archived:
                    text = page_text(body)
                    at = text.find(banner.group(0))
                    self.fail(
                        f"{url} (pinned {pinned}) says it is archived: "
                        f"{text[max(0, at - 40):at + 260]!r}"
                    )
                props = fetch_vals.island_props(body, "BenchmarkVersionSelector")
                if not props or not isinstance(props.get("versions"), list):
                    continue  # a board with one revision may carry no selector
                versions = {
                    v["slug"]: revision_label(v.get("label", ""))
                    for v in props["versions"]
                    if isinstance(v, dict) and isinstance(v.get("slug"), str)
                }
                self.assertEqual(
                    versions.get(slug), pinned,
                    f"{url}: the version selector labels {slug!r} {versions.get(slug)!r}; "
                    f"fetch_vals.VERSIONS pins {pinned!r}",
                )
                latest = max(versions, key=lambda s: revision_rank(versions[s]))
                if archived:
                    self.assertIn(
                        latest, fetch_vals.BENCHMARKS,
                        f"{url} is archived in favour of {versions[latest]} at {latest!r}, "
                        "which fetch_vals.BENCHMARKS does not read",
                    )
                    self.assertEqual(
                        fetch_vals.VERSIONS.get(latest, versions[latest]), versions[latest],
                        f"{latest!r} is labelled {versions[latest]} but fetch_vals.VERSIONS "
                        f"pins {fetch_vals.VERSIONS.get(latest)!r}",
                    )
                    continue
                self.assertEqual(
                    latest, slug,
                    f"{url}: the version selector's newest revision is "
                    f"{versions[latest]} at {latest!r}, newer than the {pinned} pinned at {slug!r}",
                )


# ---------------------------------------------------------------------------
# evals.report serializes every score with a free-text benchmarkVersion
# ("FrontierCode 1.1 Main (100 tasks)") and a verifiedStatus, and the
# benchmark itself with a caveat. Only Official/Verified rows are read.

_EVALS_VERSION_RE = re.compile(r'benchmarkVersion:"((?:[^"\\]|\\.)*)"')
_EVALS_STATUS_RE = re.compile(r'verifiedStatus:"(\w+)"')
_EVALS_CAVEAT_RE = re.compile(r'[,{]caveat:"((?:[^"\\]|\\.)*)"')


def evals_report_versions(page: str) -> list[tuple[str, str | None]]:
    """(benchmarkVersion, verifiedStatus) of every score on the page."""
    found = list(_EVALS_VERSION_RE.finditer(page))
    out = []
    for i, match in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(page)
        status = _EVALS_STATUS_RE.search(page, match.end(), end)
        out.append((match.group(1), status.group(1).lower() if status else None))
    return out


def _frontiercode_check(slug: str, key: str):
    revision = column_revision(key)
    subset = next(
        name for name, base in SUBSET_BASES["frontiercode"].items() if base == key.rsplit("_", 2)[0]
    ).capitalize()
    label = f"FrontierCode {revision} {subset}"

    def check(case, url, page, trusted, caveat):
        for version in trusted:
            case.assertIn(label, version, f"{url}: a trusted row is labelled {version!r}; {key} holds {label}")
        case.assertIn(label, caveat, f"{url}: the table's caveat no longer says {label}: {caveat!r}")

    return check


def _bfcl_check(slug: str, key: str):
    def check(case, url, page, trusted, caveat):
        named = {int(n) for v in trusted for n in re.findall(r"(?i)BFCL[-_ ]?V(\d+)", v)}
        case.assertEqual(
            named, {fetch_bfcl.SERIES},
            f"{url}: trusted rows name BFCL series {sorted(named) or 'none'}; {key} holds V{fetch_bfcl.SERIES}",
        )

    return check


def _no_number_check(name_re: str):
    def factory(slug: str, key: str):
        def check(case, url, page, trusted, caveat):
            bad = sorted({v for v in trusted if re.search(name_re + r"[-_ ]?(?:v|version\s*)?\d", v, re.IGNORECASE)})
            case.assertEqual(bad, [], f"{url}: trusted rows name a revision {key} does not hold: {bad}")

        return check

    return factory


def _phrase_check(required: str, forbidden: str | None = None):
    def factory(slug: str, key: str):
        def check(case, url, page, trusted, caveat):
            case.assertRegex(
                page_text(page), required,
                f"{url} no longer describes {required!r}, the split {key} holds",
            )
            if forbidden:
                bad = sorted({v for v in trusted if re.search(forbidden, v, re.IGNORECASE)})
                case.assertEqual(bad, [], f"{url}: trusted rows are labelled with another split than {key}'s: {bad}")

        return check

    return factory


# evals.report slug -> a check of its page against the column it feeds.
EVALS_REPORT_CHECKS = {
    "frontiercode": _frontiercode_check,
    "bfcl": _bfcl_check,
    "osworld": _no_number_check(r"OSWorld"),
    "mcp-atlas": _no_number_check(r"MCP[-_ ]?Atlas"),
    "mathvista": _phrase_check(r"testmini"),
    "charxiv": _phrase_check(r"CharXiv-R|reasoning split", forbidden=r"descriptive|CharXiv-D\b"),
}
# Tables whose column names no revision or subset, so there is nothing to
# hold their label to.
EVALS_REPORT_UNVERSIONED = {
    "swe-bench-multimodal", "swe-bench-multilingual", "ifbench", "mmlu-pro", "zerobench",
}


class TestEvalsReport(ScaleTestCase):
    """evals.report publishes no revision in its URLs, so fetch_evals_report
    files each table under the column its numbers identified when it was
    added (BENCHMARKS' comments say how). Each row's label and the table's
    caveat say which revision evals.report means; these fail when that stops
    matching the column the table feeds.

    When one fails: read the table. If evals.report moved to a newer
    revision (FrontierCode 2.0, say), point BENCHMARKS at the matching column
    -- or drop the slug if llm.json has none -- rather than let new-revision
    rows fill the old column."""

    def check(self, slug: str):
        key = fetch_evals_report.BENCHMARKS.get(slug)
        if key is None:
            self.skipTest(f"fetch_evals_report no longer reads {slug!r}")
        url = fetch_evals_report.BASE_URL.format(slug=slug)
        page = self.get(url)
        rows = evals_report_versions(page)
        self.assertTrue(rows, f"{url}: no benchmarkVersion labels on the page -- layout changed")
        trusted = [html.unescape(v) for v, status in rows if status in fetch_evals_report.TRUSTED_STATUSES]
        self.assertTrue(trusted, f"{url}: no Official/Verified rows")
        caveats = _EVALS_CAVEAT_RE.findall(page)
        caveat = html.unescape(caveats[0]) if caveats else ""
        EVALS_REPORT_CHECKS[slug](slug, key)(self, url, page, trusted, caveat)

    def test_frontiercode_is_still_the_revision_and_subset_its_column_holds(self):
        """frontiercode feeds frontiercode_1_1 (Main): every trusted row and
        the caveat must still say "FrontierCode 1.1 Main"."""
        self.check("frontiercode")

    def test_bfcl_rows_are_still_v4(self):
        """bfcl feeds bfcl_v4: every trusted row naming a series names SERIES."""
        self.check("bfcl")

    def test_osworld_rows_name_no_numbered_revision(self):
        """osworld feeds osworld_verified: a row labelled OSWorld 2.x belongs
        in an osworld_2_* column, not here."""
        self.check("osworld")

    def test_mcp_atlas_rows_name_no_numbered_revision(self):
        """mcp-atlas feeds the unversioned mcp_atlas: fails on an MCP-Atlas v2."""
        self.check("mcp-atlas")

    def test_mathvista_is_still_testmini(self):
        """mathvista feeds mathvista_mini: the page must still say testmini."""
        self.check("mathvista")

    def test_charxiv_is_still_the_reasoning_split(self):
        """charxiv feeds charxiv_reasoning: the page must still describe the
        reasoning split, and no trusted row be labelled descriptive."""
        self.check("charxiv")


if __name__ == "__main__":
    unittest.main()
