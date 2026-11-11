#!/usr/bin/env python3
"""Run every test_*.py in the repository and report what failed.

The refresh workflow runs only the handful of suites that guard what it is
about to write, because it runs every three hours and has to stay quick. The
rest -- every fetcher's parser, every mapping's checks against llm.json -- used
to run only when someone remembered to. full-tests.yml runs this once a day
instead, and the admin page's Tests tab shows what it found.

    ./run_tests.py                      every test_*.py, a line per file
    ./run_tests.py test_vals.py ...     just these
    ./run_tests.py --report _tests/report.json --summary "$GITHUB_STEP_SUMMARY"

Each file runs in a process of its own, so one that hangs, exits or leaves the
environment changed cannot take the others with it. Within a file, unittest
runs as usual with its output buffered, so each failure carries what its test
printed.

Only failures are reported in detail; a passing test is a count. Per failure
the report keeps everything known about it: the test's id, file and line, its
docstring (the tests here say *why* they exist, which is most of what a reader
needs), the exception's last line, the traceback, what the test printed, and
``since`` -- the first daily run that saw it fail, carried over from the
previous report while it keeps failing.

The report is rewritten only when what it says changes -- a new failure, a
fixed one, a different traceback, a test added -- not when only the timestamp
would, so a day on which nothing changed leaves the file alone and the workflow
has nothing to commit. When the tests ran is still visible: the Tests tab reads
the workflow's runs from GitHub as well.

Exit status: 0 when every test passed, 1 when any failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_VERSION = 1
DEFAULT_TIMEOUT = 900  # seconds per file; test_index_math.py, the slowest, takes about a minute
TEXT_LIMIT = 20_000    # characters kept of one traceback or one captured stream

# Fields that differ from run to run without the failure being any different.
# They are written, but not compared when deciding whether to rewrite.
VOLATILE = ("checked_at", "commit", "run_url", "duration_s", "python")
VOLATILE_FAILURE = ("duration_s",)


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    """Keep the end of a long text: a traceback's exception is at the bottom."""
    text = text or ""
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} characters cut ...]\n" + text[-limit:]


def relative(text: str, root: Path) -> str:
    """Repository paths as the repository spells them, not the runner's checkout."""
    return (text or "").replace(f"{root}{os.sep}", "")


# ------------------------------------------------------------------- child


class CollectingResult(unittest.TestResult):
    """A TestResult that keeps every failure as a dict, with what the test printed."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.buffer = True
        self.root = root
        self.failed: list[dict] = []
        self.failed_ids: set[str] = set()  # real tests only, for the passed count
        self._started: dict[str, float] = {}

    def startTest(self, test: unittest.TestCase) -> None:
        self._started[test.id()] = time.monotonic()
        super().startTest(test)

    def _record(self, test, err, kind: str, test_id: str | None = None) -> None:
        stdout = self._stdout_buffer.getvalue() if self._stdout_buffer else ""
        stderr = self._stderr_buffer.getvalue() if self._stderr_buffer else ""
        if err is None:
            tb, message = "", "the test was expected to fail, and passed"
        else:
            # The traceback without the captured output appended: that is kept
            # in its own fields.
            self.buffer = False
            try:
                tb = self._exc_info_to_string(err, test)
            finally:
                self.buffer = True
            lines = traceback.format_exception_only(err[0], err[1])
            message = lines[-1].strip() if lines else kind
        started = self._started.get(getattr(test, "id", lambda: "")())
        entry = {
            "test": test_id or test.id(),
            "kind": kind,
            "message": clip(relative(message, self.root), 2_000),
            "traceback": clip(relative(tb, self.root)),
            "stdout": clip(relative(stdout, self.root)),
            "stderr": clip(relative(stderr, self.root)),
            "duration_s": round(time.monotonic() - started, 3) if started else None,
        }
        entry.update(describe(test, self.root))
        self.failed.append(entry)
        if isinstance(test, unittest.TestCase):
            self.failed_ids.add(test.id())

    def addFailure(self, test, err) -> None:
        self._record(test, err, "failure")
        super().addFailure(test, err)

    def addError(self, test, err) -> None:
        # setUpClass, setUpModule and a module the loader could not read arrive
        # here as stand-in "tests" with no method of their own.
        self._record(test, err, "error")
        super().addError(test, err)

    def addSubTest(self, test, subtest, err) -> None:
        if err is not None:
            kind = "failure" if issubclass(err[0], test.failureException) else "error"
            self._record(test, err, kind, test_id=subtest.id())
        super().addSubTest(test, subtest, err)

    def addUnexpectedSuccess(self, test) -> None:
        self._record(test, None, "unexpected-success")
        super().addUnexpectedSuccess(test)


def describe(test, root: Path) -> dict:
    """The test method's docstring and where it is defined."""
    method = getattr(test, "_testMethodName", None)
    func = getattr(type(test), method, None) if method else None
    if func is None:
        return {"doc": "", "line": None}
    doc = inspect.cleandoc(func.__doc__) if func.__doc__ else ""
    try:
        line = inspect.getsourcelines(func)[1]
    except (OSError, TypeError):
        line = None
    return {"doc": doc, "line": line}


def run_child(path: Path, out: Path) -> int:
    """Run one test file in this process and write what it found to `out`."""
    root = path.parent
    sys.path.insert(0, str(root))
    os.chdir(root)
    result = CollectingResult(root)
    body: dict = {"file": path.name, "tests": 0, "skipped": 0, "failures": []}
    started = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 -- a SystemExit at import is a failure too
        body["failures"].append({
            "test": path.stem,
            "kind": "import",
            "message": clip(relative("".join(traceback.format_exception_only(type(exc), exc)).strip(), root), 2_000),
            "traceback": clip(relative(traceback.format_exc(), root)),
            "stdout": "", "stderr": "", "doc": "", "line": None, "duration_s": None,
        })
    else:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite.run(result)
        body["tests"] = result.testsRun
        body["skipped"] = len(result.skipped)
        body["failures"] = result.failed
        body["failed_tests"] = len(result.failed_ids)
    body["duration_s"] = round(time.monotonic() - started, 3)
    out.write_text(json.dumps(body), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ parent


def run_file(path: Path, timeout: float) -> dict:
    """Run one test file in a process of its own."""
    started = time.monotonic()
    body = _run_file(path, timeout)
    body["duration_s"] = round(time.monotonic() - started, 3)
    for failure in body["failures"]:
        failure["file"] = path.name
    return body


def _run_file(path: Path, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(path), "--out", str(out)]
        try:
            proc = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            def text(stream) -> str:
                return stream.decode(errors="replace") if isinstance(stream, bytes) else (stream or "")
            return {
                "file": path.name, "tests": 0, "skipped": 0,
                "failures": [{
                    "test": path.stem, "kind": "timeout",
                    "message": f"{path.name} did not finish within {timeout:g} seconds",
                    "traceback": "", "doc": "", "line": None, "duration_s": None,
                    "stdout": clip(relative(text(exc.stdout), path.parent)),
                    "stderr": clip(relative(text(exc.stderr), path.parent)),
                }],
            }
        if out.exists():
            body = json.loads(out.read_text(encoding="utf-8"))
        else:
            body = {"file": path.name, "tests": 0, "skipped": 0, "failures": [{
                "test": path.stem, "kind": "crash",
                "message": f"{path.name} exited with status {proc.returncode} before reporting",
                "traceback": "", "doc": "", "line": None, "duration_s": None,
                "stdout": clip(relative(proc.stdout, path.parent)),
                "stderr": clip(relative(proc.stderr, path.parent)),
            }]}
    return body


def carry_since(failures: list[dict], previous: dict | None, now: str, commit: str) -> None:
    """Keep the first run that saw each failure, for as long as it keeps failing."""
    seen = {}
    for old in (previous or {}).get("failures", []):
        seen[(old.get("file"), old.get("test"))] = old
    for failure in failures:
        old = seen.get((failure["file"], failure["test"]))
        if old and old.get("since"):
            failure["since"] = old["since"]
            failure["since_commit"] = old.get("since_commit", "")
        else:
            failure["since"] = now
            failure["since_commit"] = commit


def stable(report: dict | None) -> dict | None:
    """What a report says, without what differs between two identical runs."""
    if not report:
        return None
    view = {k: v for k, v in report.items() if k not in VOLATILE}
    view["failures"] = [
        {k: v for k, v in f.items() if k not in VOLATILE_FAILURE}
        for f in report.get("failures", [])
    ]
    return view


def build_report(results: list[dict], *, previous: dict | None, now: str, commit: str,
                 run_url: str, duration: float) -> dict:
    failures = sorted((f for r in results for f in r["failures"]),
                      key=lambda f: (f["file"], f["test"]))
    carry_since(failures, previous, now, commit)
    broken = {f["file"] for f in failures if f["kind"] in ("import", "timeout", "crash")}
    tests = sum(r["tests"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    # A subtest failure is one failure of one test, however many subtests fail;
    # a setUpClass error is one failure that stands for the tests it stopped.
    failed = len({(f["file"], f["test"].split(" ", 1)[0]) for f in failures})
    failed_tests = sum(r.get("failed_tests", 0) for r in results)
    return {
        "version": REPORT_VERSION,
        "checked_at": now,
        "commit": commit,
        "run_url": run_url,
        "python": platform.python_version(),
        "duration_s": round(duration, 1),
        "totals": {
            "files": len(results),
            "files_broken": len(broken),
            "tests": tests,
            "failed": failed,
            "skipped": skipped,
            "passed": max(tests - failed_tests - skipped, 0),
        },
        "failures": failures,
    }


def summary_markdown(report: dict) -> str:
    t = report["totals"]
    lines = [f"## Full test run\n",
             f"{t['tests']} tests in {t['files']} files, {t['passed']} passed, "
             f"{t['failed']} failed, {t['skipped']} skipped ({report['duration_s']:g}s).\n"]
    if report["failures"]:
        lines.append("| File | Test | Kind | Since | Message |")
        lines.append("|---|---|---|---|---|")
        for f in report["failures"]:
            msg = f["message"].replace("|", "\\|").replace("\n", " ")[:300]
            lines.append(f"| `{f['file']}` | `{f['test']}` | {f['kind']} | {f['since'][:10]} | {msg} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def discover(root: Path, names: list[str]) -> list[Path]:
    if names:
        paths = [(root / n).resolve() if not Path(n).is_absolute() else Path(n) for n in names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"no such test file: {', '.join(missing)}")
        return paths
    return sorted(root.glob("test_*.py"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every test_*.py and report the failures.")
    parser.add_argument("files", nargs="*", help="test files to run (default: every test_*.py)")
    parser.add_argument("--report", type=Path, help="write the JSON report here (only when it changed)")
    parser.add_argument("--summary", type=Path, help="append a markdown summary here, e.g. $GITHUB_STEP_SUMMARY")
    parser.add_argument("--run-url", default="", help="the CI run, linked from the report")
    parser.add_argument("--jobs", type=int, default=1,
                        help="files to run side by side (default 1: the suites were written to run alone)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per file")
    parser.add_argument("--root", type=Path, default=HERE, help=argparse.SUPPRESS)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child:
        return run_child(args.child.resolve(), args.out)

    root = args.root.resolve()
    paths = discover(root, args.files)
    started = time.monotonic()

    def one(path: Path) -> dict:
        result = run_file(path, args.timeout)
        bad = len(result["failures"])
        status = "ok" if not bad else f"{bad} FAILED"
        print(f"{status:>10}  {result['duration_s']:7.1f}s  {path.name}  ({result['tests']} tests)", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        results = list(pool.map(one, paths))

    previous = None
    if args.report and args.report.exists():
        try:
            previous = json.loads(args.report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    report = build_report(results, previous=previous, now=now, commit=git_commit(root),
                          run_url=args.run_url, duration=time.monotonic() - started)

    t = report["totals"]
    print(f"\n{t['tests']} tests in {t['files']} files: {t['passed']} passed, "
          f"{t['failed']} failed, {t['skipped']} skipped")
    for f in report["failures"]:
        print(f"  {f['kind']:>18}  {f['file']}  {f['test']}\n{'':22}{f['message'][:200]}")

    if args.report:
        if stable(report) != stable(previous):
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.report}")
        else:
            print(f"{args.report} unchanged")
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(summary_markdown(report))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
