"""A small TTL'd response cache under ~/.cache/ai-bench, shared by the fetchers.

A refresh reads several sources more than once: the mapping updater scrapes a
board to learn its names and the fetcher scrapes it again for the scores, and
the two runs are separate processes minutes apart, so nothing in memory can
help. Against llm-stats that was four reads of the same flat leaderboard; the
Hugging Face crawl was a hundred and forty model cards, walked twice.

The design copies artificialanalysis.py's response cache, which has carried
that job for AA since the request budget became a concern -- see "The request
budget" in README.md -- rather than inventing a second convention:

  * The TTL is what decides freshness, not the caller. The cron is three hours
    apart and the TTL is one, so a scheduled refresh always re-reads; only the
    second read *inside* one run, and a merge- or dispatch-triggered run
    landing inside the window, are served from here.
  * A `key` names what was asked for -- the set of models being crawled, say --
    and a changed key misses. That is what stops a model added to llm.json
    being answered from a crawl that predates it.
  * Every failure is a miss. A cache that raises is worse than no cache, so a
    corrupt file, an unwritable directory and a full disk all end as "fetch it
    again".

The directory is the one the workflow already carries between runs (see the
`ai-bench-openness-` cache step), so a file written here survives to the next
run without anything being added to the workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any

CACHE_DIR = os.path.expanduser("~/.cache/ai-bench")

# One hour, matching artificialanalysis.py's response cache. Long enough that
# the two reads inside one refresh share a fetch, short enough that the
# three-hourly cron never sees a cached answer.
DEFAULT_TTL_SECONDS = 3600


def ttl_seconds(env_var: str) -> int:
    """The TTL for one cache, overridable per source.

    0 or negative disables that cache entirely, which is how a run that must
    not be served anything stored asks for it. A value that is not a number is
    a typo rather than an instruction, so it falls back to the default instead
    of silently disabling the cache.
    """
    try:
        return int(os.getenv(env_var, str(DEFAULT_TTL_SECONDS)))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def digest(value: Any) -> str:
    """A short, stable key for whatever identifies a request.

    json.dumps with sorted keys so a dict's iteration order cannot produce two
    keys for one request, and so a list's order still can -- callers that do
    not care about order sort before calling.
    """
    raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def load(name: str, key: str, ttl: int, label: str | None = None) -> Any | None:
    """The cached payload for `key`, or None on any miss.

    A hit is announced on stderr with its age, because a run that is faster
    than usual should say why rather than leave a reader wondering whether the
    source got quicker.
    """
    if ttl <= 0:
        return None
    try:
        with open(_path(name), encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(stored, dict) or stored.get("key") != key:
        return None
    age = time.time() - stored.get("fetched", 0)
    if age < 0 or age > ttl:
        return None
    if "payload" not in stored:
        return None
    print(
        f"< {label or name} from the response cache ({int(age)}s old)",
        file=sys.stderr,
    )
    return stored["payload"]


def store(name: str, key: str, payload: Any, ttl: int) -> None:
    """Cache `payload` under `key`. Never raises: a failed write is a miss next
    time, which is exactly what would have happened without a cache at all."""
    if ttl <= 0:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Written beside the target and moved into place, so a run cancelled
        # mid-write leaves the previous good file rather than a truncated one
        # that every later run has to fail to parse.
        temporary = _path(name) + f".{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"key": key, "fetched": time.time(), "payload": payload}, handle)
        os.replace(temporary, _path(name))
    except (OSError, TypeError, ValueError):
        return
