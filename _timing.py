"""One stopwatch, shared by everything a refresh runs.

`update-all` times the steps it launches, but a step is a whole script, and
the two slowest of them spend their time in subprocesses of their own -- the
llm-stats mapping updater scrapes llm-stats three times before it matches a
single name. Reporting those from here keeps one format across the run, so a
log can be read top to bottom as a single list of timings rather than as
several scripts' separate ideas of how to say how long something took.

Everything goes to stderr. Python keeps stderr line-buffered whether or not
it is a terminal, so a timing arrives as its step finishes; stdout is block-
buffered against a pipe and would deliver the lot at exit, which is precisely
when they stop being useful. It also leaves stdout as each script's report
proper, for a run that pipes it somewhere.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

__all__ = ["report", "timed"]


def report(label: str, started: float) -> None:
    """Report what the work since `started` (a time.monotonic() reading) cost."""
    print(f"  {label}: {time.monotonic() - started:.1f}s", file=sys.stderr)


def timed(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call `fn`, reporting what it cost.

    In a finally, so a call that raised is timed too: how long a dead source
    took before giving up -- a timeout, a retry loop -- is exactly what a
    reader is after when a run that normally takes four minutes took twenty.
    """
    started = time.monotonic()
    try:
        return fn(*args, **kwargs)
    finally:
        report(label, started)
