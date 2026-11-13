#!/usr/bin/env python3
"""Live checks that each source still runs the benchmark revision its column is named for.

Run with ./test_source_versions.py. Unlike the other suites these reach the
network, because what they guard is not in the repository: it is which revision
of a benchmark a third party is running this week.

A column such as terminal_bench_4_0 says which revision it holds, and most
sources say so too, in a URL or a dataset id the fetcher checks. Not all of
them. Artificial Analysis files its Terminal-Bench runs under
"terminal-bench-v4" on the coding-agents page and under "terminalbench_4_0" in
its model records, and a switch to 4.1 could arrive under either without a
fetch noticing -- the 4.0 column would quietly fill with 4.1 numbers. The same
goes for a newer tbench.ai board than fetch_tbench.BOARDS reads: nothing fails,
it simply goes unread.

So these read what each source says about itself, in prose and slugs, and fail
when it names a revision newer than the one the column measures (or stops
naming the one it does). A failure here is a prompt to look, not proof of a
switch -- AA mentioning 4.1 in a footnote fails it too, and that is fine. It
lands on the admin page's Tests tab through the daily full-tests run.

What to do when one fails:

* AA names a newer Terminal-Bench: check whether its runs moved. If they did,
  give the new revision a column (or point AA's extractor at the existing one,
  e.g. terminal_bench_4_1), let that column take 4.0's weight in
  derive_indexes.py with a REVISION_FALLBACKS entry from 4.0, and bump
  AA_TERMINAL_BENCH below. If it is only a mention, bump nothing and note why.
* tbench.ai names a board newer than BOARDS: add it to fetch_tbench.BOARDS and
  llm.json, as 4.1 was.
* A Vals pin no longer matches: fetch_vals.VERSIONS explains the choice; the
  board was revised in place and needs a column of its own.

A host that cannot be reached at all (no network, DNS) skips rather than
fails, so the suite stays usable offline. An HTTP error does fail: a page that
moved is as much a reason to look as a page that changed.
"""

from __future__ import annotations

import re
import socket
import unittest
import urllib.error
import urllib.parse
import urllib.request

import fetch_tbench
import fetch_vals


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# The Terminal-Bench revisions this repository reads from Artificial Analysis:
# update.py's terminal_bench_4_0 (model records' terminalbench_4_0 and the
# coding-agents page's terminal-bench-v4) and terminal_bench_science_0_1.
# terminal_bench_2_1 is read too but is older, so it can never be "newer".
AA_TERMINAL_BENCH = "4.0"
AA_TERMINAL_BENCH_SCIENCE = "0.1"

# Pages where AA describes the evaluations it runs. Each names the revision in
# prose ("Terminal-Bench 4.0") or in a slug ("terminal-bench-v4-0").
AA_PAGES = {
    "evaluations": "https://artificialanalysis.ai/evaluations",
    "methodology": "https://artificialanalysis.ai/methodology/intelligence-benchmarking",
    "coding agents": "https://artificialanalysis.ai/agents/coding-agents",
}

TBENCH_HOME = "https://www.tbench.ai/"
TBENCH_SCIENCE = "https://www.tbench.ai/leaderboard/terminal-bench-science"

# "Terminal-Bench 4.0", "terminal-bench-v4-0", "terminalbench-4-1",
# "Terminal Bench 2.1", "Terminal-Bench-Science 0.1". A bare major
# ("terminal-bench-v4") names no revision and is not matched.
_VERSION_RE = re.compile(
    r"terminal[- ]?bench(?P<science>[- ]science)?[- ]?(?:v|version\s*)?"
    r"(?P<major>\d+)[.-](?P<minor>\d+)(?![\d.])",
    re.IGNORECASE,
)


def as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def versions_named(text: str) -> dict[str, set[str]]:
    """{"terminal-bench": {...}, "science": {...}}: every revision ``text`` names."""
    found: dict[str, set[str]] = {"terminal-bench": set(), "science": set()}
    for m in _VERSION_RE.finditer(text):
        family = "science" if m.group("science") else "terminal-bench"
        found[family].add(f"{int(m.group('major'))}.{int(m.group('minor'))}")
    return found


def newer_than(versions: set[str], ceiling: str) -> list[str]:
    return sorted((v for v in versions if as_tuple(v) > as_tuple(ceiling)), key=as_tuple)


class Unreachable(Exception):
    """The host could not be reached at all -- offline, not changed."""


def fetch(url: str, *, follow: bool = True) -> tuple[str, str]:
    """(final URL, body). Raises Unreachable for a connection that never formed."""
    request = urllib.request.Request(url, headers=HEADERS)
    opener = (
        urllib.request.build_opener()
        if follow
        else urllib.request.build_opener(_NoRedirect)
    )
    try:
        with opener.open(request, timeout=30) as response:
            return response.geturl(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if not follow and exc.code in (301, 302, 303, 307, 308):
            return exc.headers.get("Location", ""), ""
        raise
    except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
        raise Unreachable(f"{url}: {exc}") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 -- stdlib signature
        return None


class LiveTestCase(unittest.TestCase):
    def page(self, url: str, **kwargs) -> tuple[str, str]:
        try:
            return fetch(url, **kwargs)
        except Unreachable as exc:
            self.skipTest(f"offline: {exc}")


class TestVersionParsing(unittest.TestCase):
    """Offline: the matcher itself, so a failure below is about the source."""

    def test_reads_prose_and_slugs(self):
        got = versions_named(
            "Terminal-Bench 4.0, terminal-bench-v2-1, terminalbench-4-1, "
            "Terminal-Bench-Science 0.1, terminal-bench-v4"
        )
        self.assertEqual(got["terminal-bench"], {"4.0", "2.1", "4.1"})
        self.assertEqual(got["science"], {"0.1"})

    def test_a_bare_major_names_no_revision(self):
        self.assertEqual(versions_named("terminal-bench-v4 terminal-bench/x")["terminal-bench"], set())

    def test_newer_than_compares_numerically(self):
        self.assertEqual(newer_than({"4.0", "4.10", "2.1", "5.0"}, "4.1"), ["4.10", "5.0"])


class TestArtificialAnalysis(LiveTestCase):
    def test_aa_still_runs_the_terminal_bench_revision_its_column_holds(self):
        """AA feeds terminal_bench_4_0 (107 values) under ids that name no minor
        revision, so a move to 4.1 would land in the 4.0 column unannounced.
        This fails as soon as an AA page names a Terminal-Bench newer than
        AA_TERMINAL_BENCH, or stops naming that one: check AA's runs, then see
        this file's docstring for what to change."""
        named_anywhere: set[str] = set()
        for label, url in AA_PAGES.items():
            with self.subTest(page=label):
                _final, body = self.page(url)
                found = versions_named(body)["terminal-bench"]
                named_anywhere |= found
                self.assertEqual(
                    newer_than(found, AA_TERMINAL_BENCH), [],
                    f"AA's {label} page ({url}) names Terminal-Bench "
                    f"{', '.join(newer_than(found, AA_TERMINAL_BENCH))}, newer than the "
                    f"{AA_TERMINAL_BENCH} that terminal_bench_4_0 holds",
                )
        self.assertIn(
            AA_TERMINAL_BENCH, named_anywhere,
            f"No AA page names Terminal-Bench {AA_TERMINAL_BENCH} any more",
        )

    def test_aa_still_runs_terminal_bench_science_0_1(self):
        """terminal_bench_science_0_1 is read from AA's unversioned
        "terminalbench_science" field; fails when AA names a newer Science."""
        for label, url in AA_PAGES.items():
            with self.subTest(page=label):
                _final, body = self.page(url)
                found = versions_named(body)["science"]
                self.assertEqual(
                    newer_than(found, AA_TERMINAL_BENCH_SCIENCE), [],
                    f"AA's {label} page names Terminal-Bench-Science "
                    f"{', '.join(newer_than(found, AA_TERMINAL_BENCH_SCIENCE))}",
                )


class TestTbench(LiveTestCase):
    def newest_read(self) -> str:
        versions = [
            spec["version"]
            for key, spec in fetch_tbench.BOARDS.items()
            if spec["package"].startswith("terminal-bench/")
        ]
        return max(versions, key=as_tuple)

    def test_no_newer_terminal_bench_board_than_the_ones_read(self):
        """tbench.ai renders its current board on the home page. A newer one than
        fetch_tbench.BOARDS reads (4.2, 5.0, ...) would simply go unread; this
        names it so it can be added the way 4.1 was."""
        _final, body = self.page(TBENCH_HOME)
        found = versions_named(body)["terminal-bench"]
        self.assertTrue(found, f"{TBENCH_HOME} names no Terminal-Bench revision at all")
        self.assertEqual(
            newer_than(found, self.newest_read()), [],
            f"tbench.ai names Terminal-Bench {', '.join(newer_than(found, self.newest_read()))}; "
            f"fetch_tbench.BOARDS reads up to {self.newest_read()}",
        )

    def test_terminal_bench_science_still_points_at_the_board_read(self):
        """tbench.ai's Science link redirects to the hub board it calls current,
        naming it in ``leaderboard=``. Fails when that is no longer v0-1-eval."""
        location, _body = self.page(TBENCH_SCIENCE, follow=False)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
        current = (query.get("leaderboard") or [None])[0]
        expected = fetch_tbench.BOARDS["terminal_bench_science_0_1"]["name"]
        self.assertEqual(
            current, expected,
            f"{TBENCH_SCIENCE} now redirects to {location!r}; "
            f"fetch_tbench reads {expected!r}",
        )


class TestVals(LiveTestCase):
    def test_every_pinned_board_still_carries_its_pinned_version(self):
        """fetch_vals.VERSIONS pins the boards whose slug names no revision
        (terminal-bench-4 -> 4.0, terminal-bench-science -> 0.1, vibe-code,
        cyber). The fetcher refuses a mismatch at refresh time, which surfaces
        as a fetcher warning; this puts the same check on the Tests tab. Read
        past the board cache, so it is today's page."""
        for slug, pinned in fetch_vals.VERSIONS.items():
            with self.subTest(slug=slug):
                url = fetch_vals.benchmark_url(slug)
                try:
                    page = fetch_vals.fetch_html(url, retries=2)
                except (urllib.error.HTTPError,) as exc:
                    self.fail(f"{url}: HTTP {exc.code}")
                except (urllib.error.URLError, OSError) as exc:
                    self.skipTest(f"offline: {url}: {exc}")
                try:
                    fetch_vals.parse_board(page, slug)
                except ValueError as exc:
                    self.fail(f"{slug} (pinned {pinned}): {exc}")


if __name__ == "__main__":
    unittest.main()
