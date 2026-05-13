#!/usr/bin/env python3
"""Fetch benchmark scores from Hugging Face model card READMEs (lenient parse)."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")
HF_BASE = "https://huggingface.co"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


def normalize_repo(arg: str) -> str:
    """Accept an HF URL or 'owner/name'; return 'owner/name'."""
    if arg.startswith("http://") or arg.startswith("https://"):
        match = re.match(r"^https?://huggingface\.co/([^/]+/[^/?#]+)", arg)
        if not match:
            raise ValueError(f"Not a Hugging Face URL: {arg}")
        return match.group(1)
    return arg.strip("/")


def fetch_readme(repo: str, timeout: int = 30) -> str:
    """Fetch the raw README.md for repo. Tries main, then master."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain, text/markdown, */*"}
    last_error: Exception | None = None
    for branch in ("main", "master"):
        url = f"{HF_BASE}/{repo}/raw/{branch}/README.md"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_error = exc
                continue
            raise
    raise FileNotFoundError(f"No README.md found for {repo} (tried main, master): {last_error}")


_FOOTNOTE_CHARS = "*†‡§¶♦♣♠♥"


def _strip_emphasis(s: str) -> str:
    while True:
        prev = s
        for wrap in ("**", "__", "*", "_", "`"):
            if s.startswith(wrap) and s.endswith(wrap) and len(s) > 2 * len(wrap):
                s = s[len(wrap) : -len(wrap)].strip()
        if s == prev:
            return s


_INVISIBLE_SPACES = ("\xa0", " ", " ", " ", "​")


def clean_cell(text: str) -> str:
    """Decode HTML entities, strip markdown decoration, footnote markers, numeric-only parens."""
    s = html.unescape(text)
    for ch in _INVISIBLE_SPACES:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = _strip_emphasis(s)
    s = s.rstrip(_FOOTNOTE_CHARS).rstrip()
    # Drop trailing footnote refs like " (1)" or " (12,3)" but preserve informative parens.
    s = re.sub(r"\s*\(\s*[\d,]+\s*\)\s*$", "", s)
    return s.strip()


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PLACEHOLDERS = {"", "-", "—", "–", "n/a", "na", "/", "?", "—%", "tbd"}


def parse_score(text: str) -> float | None:
    """Extract first numeric value from cell. Returns None for placeholders."""
    cleaned = clean_cell(text)
    if cleaned.lower() in _PLACEHOLDERS:
        return None
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_PIPE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_markdown_tables(md: str) -> list[Table]:
    """Find pipe-delimited tables in a markdown document."""
    tables: list[Table] = []
    lines = md.splitlines()
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        if _PIPE_LINE_RE.match(line) and i + 1 < len(lines) and _SEPARATOR_RE.match(lines[i + 1]):
            headers = _split_row(line)
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and _PIPE_LINE_RE.match(lines[j]):
                rows.append(_split_row(lines[j]))
                j += 1
            tables.append(Table(headers=headers, rows=rows))
            i = j
        else:
            i += 1
    return tables


def _strip_tags(markup: str) -> str:
    text = re.sub(r"<[^>]+>", " ", markup)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_html_tables(html: str) -> list[Table]:
    """Lenient regex parser for <table> blocks; no bs4 dependency."""
    tables: list[Table] = []
    for table_match in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.IGNORECASE | re.DOTALL):
        body = table_match.group(1)
        rows: list[list[str]] = []
        for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", body, re.IGNORECASE | re.DOTALL):
            cells = re.findall(
                r"<t[hd][^>]*>(.*?)</t[hd]>", tr_match.group(1), re.IGNORECASE | re.DOTALL
            )
            cleaned = [_strip_tags(c) for c in cells]
            if any(cleaned):
                rows.append(cleaned)
        if len(rows) >= 2:
            tables.append(Table(headers=rows[0], rows=rows[1:]))
    return tables


def _norm_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _is_category_row(row: list[str], header_count: int) -> bool:
    nonempty = [c for c in row if c.strip()]
    if len(nonempty) <= 1:
        return True
    distinct = {c.strip() for c in row[:header_count]}
    return len(distinct) <= 1


def select_column(table: Table, repo: str) -> int | None:
    """Pick the column matching the HF repo's model name (last path segment)."""
    model_name = repo.split("/")[-1]
    target = _norm_for_match(model_name)
    if not target:
        return None

    best: tuple[int, int] | None = None  # (length_distance, col_idx)
    for idx, header in enumerate(table.headers):
        h_norm = _norm_for_match(clean_cell(header))
        if not h_norm:
            continue
        if h_norm == target:
            return idx
        # Allow loose substring matches if length difference is modest.
        if target in h_norm and len(h_norm) - len(target) <= 24:
            score = len(h_norm) - len(target)
        elif h_norm in target and len(target) - len(h_norm) <= 24 and len(h_norm) >= 6:
            score = (len(target) - len(h_norm)) + 100  # weaker than "header contains target"
        else:
            continue
        if best is None or score < best[0]:
            best = (score, idx)
    return best[1] if best else None


def extract_scores_from_tables(tables: list[Table], repo: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for table in tables:
        if not table.headers or not table.rows:
            continue
        col = select_column(table, repo)
        if col is None or col >= len(table.headers):
            continue
        for row in table.rows:
            if not row or _is_category_row(row, len(table.headers)):
                continue
            if col >= len(row):
                continue
            label = clean_cell(row[0])
            if not label:
                continue
            value = parse_score(row[col])
            if value is None:
                continue
            out.setdefault(label, value)
    return out


def extract_scores(repo: str) -> dict[str, float]:
    md = fetch_readme(repo)
    return extract_scores_from_tables(parse_markdown_tables(md) + parse_html_tables(md), repo)


def iter_hf_models(doc: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (slug, repo) for every model in llm.json whose url is on huggingface.co."""
    out: list[tuple[str, str]] = []
    for model in doc.get("models", []):
        if not isinstance(model, dict):
            continue
        slug = model.get("name")
        url = model.get("url")
        if not isinstance(slug, str) or not slug:
            continue
        if not isinstance(url, str):
            continue
        if not (url.startswith("https://huggingface.co/") or url.startswith("http://huggingface.co/")):
            continue
        try:
            repo = normalize_repo(url)
        except ValueError:
            continue
        out.append((slug, repo))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark scores from Hugging Face model card READMEs."
    )
    parser.add_argument(
        "repo_or_url",
        nargs="?",
        help="Hugging Face repo 'owner/name' or full URL.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Iterate every model in llm.json with a Hugging Face URL.",
    )
    parser.add_argument(
        "--json-file",
        default=str(DEFAULT_LLM_JSON),
        help="Path to llm.json (used with --all-models).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "table", "names"],
        default="json",
        help="Output format (default: json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.all_models:
        doc = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        results: list[dict[str, Any]] = []
        for slug, repo in iter_hf_models(doc):
            try:
                scores = extract_scores(repo)
            except Exception as exc:
                print(f"warning: {slug} ({repo}): {exc}", file=sys.stderr)
                continue
            results.append({"model": slug, "repo": repo, "scores": scores})
    else:
        if not args.repo_or_url:
            print("error: repo_or_url is required when --all-models is not set", file=sys.stderr)
            return 2
        repo = normalize_repo(args.repo_or_url)
        scores = extract_scores(repo)
        results = [{"model": repo.split("/")[-1], "repo": repo, "scores": scores}]

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.format == "names":
        labels: set[str] = set()
        for entry in results:
            labels.update(entry["scores"].keys())
        for label in sorted(labels):
            print(label)
    else:
        for entry in results:
            print(f"\n## {entry['model']} ({entry['repo']})")
            if not entry["scores"]:
                print("  (no scores)")
                continue
            width = max(len(k) for k in entry["scores"])
            for label, value in sorted(entry["scores"].items()):
                print(f"  {label:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
