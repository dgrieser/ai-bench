#!/usr/bin/env python3
"""Fetch benchmark scores from Hugging Face model cards.

Two sources per repo, merged with structured data taking precedence:

  * the Hub's structured eval metadata (/api/models/<repo>?expand[]=evalResults
    &expand[]=model-index), which is what the model page renders as its
    'Evaluation results' section — evalResults covers .eval_results/*.yaml
    files (including ones still pending on open Hub PRs), model-index the
    classic card frontmatter;
  * markdown/HTML tables in the README body (lenient parse).
"""

from __future__ import annotations

import argparse
import html
import json
import os
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


def _auth_headers() -> dict[str, str]:
    """Request headers, with a Bearer token from the environment for gated repos."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain, text/markdown, */*"}
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_readme(repo: str, timeout: int = 30) -> str:
    """Fetch the raw README.md for repo. Tries main, then master."""
    headers = _auth_headers()
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
    # Markdown backslash escapes ("𝛕3\-Banking", "Avg\*") from doc exports.
    s = re.sub(r"\\([\\`*_{}\[\]()#+.!|~-])", r"\1", s)
    s = _strip_emphasis(s)
    s = s.rstrip(_FOOTNOTE_CHARS).rstrip()
    # Drop trailing footnote refs like " (1)" or " (12,3)" but preserve informative parens.
    s = re.sub(r"\s*\(\s*[\d,]+\s*\)\s*$", "", s)
    return s.strip()


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PLACEHOLDERS = {"", "-", "—", "–", "n/a", "na", "/", "?", "—%", "tbd"}


def parse_score(text: str) -> float | None:
    """Extract first numeric value from cell. Returns None for placeholders."""
    # Lower-is-better markers ("Violation (↓): 26.4 <br> Coverage: 64.8") flag
    # cells whose first number is not a benchmark score in the usual sense.
    if "↓" in text:
        return None
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


# Soft line breaks that word-processor exports leave inside table cells;
# str.splitlines() treats them as row boundaries, which shreds the table.
_SOFT_LINE_BREAKS = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def _logical_lines(md: str) -> list[str]:
    for ch in _SOFT_LINE_BREAKS:
        md = md.replace(ch, " ")
    return md.splitlines()


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
    lines = _logical_lines(md)
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


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def _is_category_row(row: list[str], header_count: int) -> bool:
    nonempty = [c for c in row if c.strip()]
    if len(nonempty) <= 1:
        return True
    distinct = {c.strip() for c in row[:header_count]}
    return len(distinct) <= 1


_LABEL_DENY_EXACT = {
    "type", "link", "precision", "model", "modelname", "metric", "size",
    "license", "date", "releasedate", "quantization", "description", "download",
    "contextlength", "blackwell", "hopper", "huggingface", "modelscope",
    "gitcode", "github", "gitee", "weights", "checkpoint", "activatedtotal",
}
_LABEL_DENY_SUB = (
    "parameter", "totalparam", "activatedparam", "activeparam", "contextlength",
    "embedding", "attentionhead", "headsize", "kvhead", "mamba", "expert",
    "activation", "sequencelength", "positionembedding", "hiddensize", "vocab",
    "statesize", "numberoflayers", "numlayers", "layers",
)
# Context-length column labels like "4k", "128k", "1000k", "2m".
_CONTEXT_LEN_RE = re.compile(r"^\d+(?:\.\d+)?[km]$")


def _is_metadata_label(label: str) -> bool:
    """True for non-benchmark labels (model specs/metadata) that should not be scored."""
    n = _norm_for_match(label)
    if not n or n in _LABEL_DENY_EXACT or _CONTEXT_LEN_RE.match(n):
        return True
    return any(sub in n for sub in _LABEL_DENY_SUB)


def _name_match_score(candidate: str, model_name: str) -> int | None:
    """Score how well `candidate` names `model_name` (lower = better; None = no match)."""
    target = _norm_for_match(model_name)
    c_norm = _norm_for_match(candidate)
    if not target or not c_norm:
        return None
    if c_norm == target:
        return 0
    # Allow loose substring matches if length difference is modest.
    if target in c_norm and len(c_norm) - len(target) <= 24:
        return len(c_norm) - len(target)
    if c_norm in target and len(target) - len(c_norm) <= 24 and len(c_norm) >= 6:
        return (len(target) - len(c_norm)) + 100  # weaker than "candidate contains target"
    # Token-subset match: names often drop size/param tokens the repo name carries
    # (e.g. "Mellum2 Thinking" vs "Mellum2-12B-A2.5B-Thinking"), which breaks
    # contiguous-substring matching.
    c_tokens = _tokens(candidate)
    t_tokens = _tokens(model_name)
    shared = c_tokens & t_tokens
    subset = c_tokens <= t_tokens or t_tokens <= c_tokens
    if subset and len("".join(shared)) >= 6:
        return 200 + len(c_tokens ^ t_tokens)  # weaker than substring matches
    return None


_SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?b$")


def _select_size_variant_column(table: Table, model_name: str) -> int | None:
    """Match size-variant headers like '8B Dense' for repos like 'granite-4.1-8b'.

    Fires only when several headers start with a size token and share the same
    remainder (one model line, varying parameter count), so a lone competitor
    column in a comparison table is not mistaken for the model.
    """
    model_sizes = {t for t in _tokens(model_name) if _SIZE_TOKEN_RE.match(t)}
    if not model_sizes:
        return None
    family: list[tuple[int, str, frozenset[str]]] = []  # (idx, size, remainder)
    for idx, header in enumerate(table.headers):
        toks = [t for t in re.split(r"[^a-z0-9]+", clean_cell(header).lower()) if t]
        if toks and _SIZE_TOKEN_RE.match(toks[0]):
            family.append((idx, toks[0], frozenset(toks[1:])))
    if len(family) < 2:
        return None
    if len({rem for _, _, rem in family}) != 1:
        return None
    matches = [idx for idx, size, _ in family if size in model_sizes]
    return matches[0] if len(matches) == 1 else None


_LABEL_COLUMN_HEADERS = {
    "benchmark", "benchmarks", "task", "tasks", "dataset", "datasets",
    "eval", "evals", "evaluation", "evaluations",
}


def _find_label_column(table: Table) -> int:
    """Column holding benchmark names. Usually 0, but tables like
    `| Category | Benchmark | Model-X | ... |` put a grouping column first."""
    for idx, header in enumerate(table.headers):
        if _norm_for_match(clean_cell(header)) in _LABEL_COLUMN_HEADERS:
            return idx
    return 0


# Match score assigned to the size-variant fallback: weaker than any direct
# name match so exact-name tables win when scores are merged across tables.
_SIZE_VARIANT_SCORE = 400


def _select_column_scored(table: Table, repo: str) -> tuple[int, int] | None:
    """(match_score, col_idx) for the column naming this model; lower is better."""
    model_name = repo.split("/")[-1]
    best: tuple[int, int] | None = None
    for idx, header in enumerate(table.headers):
        score = _name_match_score(clean_cell(header), model_name)
        if score is None:
            continue
        if best is None or score < best[0]:
            best = (score, idx)
    if best is not None:
        return best
    size_col = _select_size_variant_column(table, model_name)
    if size_col is not None:
        return (_SIZE_VARIANT_SCORE, size_col)
    return None


def select_column(table: Table, repo: str) -> int | None:
    """Pick the column matching the HF repo's model name (last path segment)."""
    scored = _select_column_scored(table, repo)
    return scored[1] if scored else None


def _select_row_scored(table: Table, repo: str) -> tuple[int, int] | None:
    """(match_score, row_idx) for transposed tables (models in rows)."""
    model_name = repo.split("/")[-1]
    best: tuple[int, int] | None = None
    for idx, row in enumerate(table.rows):
        if not row:
            continue
        score = _name_match_score(clean_cell(row[0]), model_name)
        if score is None:
            continue
        if best is None or score < best[0]:
            best = (score, idx)
    return best


def select_row(table: Table, repo: str) -> int | None:
    """For transposed tables (models in rows): pick the row whose first cell names the model."""
    scored = _select_row_scored(table, repo)
    return scored[1] if scored else None


def extract_scores_from_tables(tables: list[Table], repo: str) -> dict[str, float]:
    # Each table is extracted independently, then merged best-name-match first:
    # a table naming the model exactly must beat one that only matched loosely
    # (e.g. the "MiMo-V2-Flash Base" pretrain column when the repo is the
    # post-trained MiMo-V2-Flash), regardless of document order.
    extracts: list[tuple[int, dict[str, float]]] = []  # (match_score, scores)
    for table in tables:
        if not table.headers or not table.rows:
            continue
        scored_col = _select_column_scored(table, repo)
        label_col = _find_label_column(table)
        if scored_col is not None and scored_col[1] != label_col and scored_col[1] < len(table.headers):
            # Column orientation: benchmarks in `label_col`, this model in `col`.
            match_score, col = scored_col
            found: dict[str, float] = {}
            for row in table.rows:
                if label_col > 0 and len(row) == len(table.headers) - 1:
                    # Category tables often omit the leading category cell on
                    # continuation rows, shifting everything left by one.
                    row = [""] + row
                if not row or _is_category_row(row, len(table.headers)):
                    continue
                if col >= len(row) or label_col >= len(row):
                    continue
                label = clean_cell(row[label_col])
                if not label or _is_metadata_label(label):
                    continue
                value = parse_score(row[col])
                if value is None:
                    continue
                found.setdefault(label, value)
            extracts.append((match_score, found))
            continue
        # Transposed orientation: this model is a row, benchmarks are the headers.
        scored_row = _select_row_scored(table, repo)
        if scored_row is None:
            continue
        match_score, row_idx = scored_row
        row = table.rows[row_idx]
        found = {}
        for j in range(1, len(table.headers)):
            if j >= len(row):
                break
            label = clean_cell(table.headers[j])
            if not label or _is_metadata_label(label):
                continue
            value = parse_score(row[j])
            if value is None:
                continue
            found.setdefault(label, value)
        extracts.append((match_score, found))
    # Better-matching tables win outright; between equally-matched tables the
    # best reported run wins (same policy update.py applies to label aliases).
    out: dict[str, float] = {}
    tier: dict[str, int] = {}
    for match_score, found in sorted(extracts, key=lambda pair: pair[0]):
        for label, value in found.items():
            if label not in out:
                out[label] = value
                tier[label] = match_score
            elif tier[label] == match_score and value > out[label]:
                out[label] = value
    return out


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return parse_score(value)
    return None


def fetch_api_eval_data(repo: str, timeout: int = 30) -> dict[str, Any]:
    """Structured eval metadata from the Hub API: the 'Evaluation results'
    widget (.eval_results/*.yaml files, including ones pending on open PRs)
    plus the classic model-index card metadata."""
    url = f"{HF_BASE}/api/models/{repo}?expand[]=evalResults&expand[]=model-index"
    req = urllib.request.Request(url, headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _eval_results_label(dataset: dict[str, Any]) -> str | None:
    """'Idavidrein/gpqa (diamond)'; the task suffix is dropped when it just
    repeats the dataset name ('cais/hle' + task 'hle')."""
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        return None
    task_id = dataset.get("task_id")
    if isinstance(task_id, str) and task_id:
        task_norm = _norm_for_match(task_id)
        if task_norm not in (
            _norm_for_match(dataset_id),
            _norm_for_match(dataset_id.split("/")[-1]),
        ):
            return f"{dataset_id} ({task_id})"
    return dataset_id


def extract_eval_results(payload: dict[str, Any]) -> dict[str, float]:
    """Scores from the Hub's 'Evaluation results' widget (evalResults)."""
    out: dict[str, float] = {}
    entries = payload.get("evalResults")
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        dataset = data.get("dataset")
        if not isinstance(dataset, dict):
            continue
        label = _eval_results_label(dataset)
        value = _coerce_score(data.get("value"))
        if label is None or value is None:
            continue
        out.setdefault(label, value)
    return out


def extract_model_index(payload: dict[str, Any]) -> dict[str, float]:
    """Scores from classic model-index card metadata (the pre-evalResults
    source of the 'Evaluation results' widget)."""
    out: dict[str, float] = {}
    index = payload.get("model-index")
    if not isinstance(index, list):
        return out
    for model_entry in index:
        if not isinstance(model_entry, dict):
            continue
        for result in model_entry.get("results") or []:
            if not isinstance(result, dict):
                continue
            dataset = result.get("dataset")
            if not isinstance(dataset, dict):
                continue
            base = dataset.get("name") or dataset.get("type")
            if not isinstance(base, str) or not base:
                continue
            metrics = [m for m in result.get("metrics") or [] if isinstance(m, dict)]
            scored = [
                (m, v) for m in metrics if (v := _coerce_score(m.get("value"))) is not None
            ]
            for metric, value in scored:
                metric_name = metric.get("name") or metric.get("type")
                label = base
                if len(scored) > 1 and isinstance(metric_name, str) and metric_name:
                    label = f"{base} ({metric_name})"
                out.setdefault(label, value)
    return out


def extract_scores(repo: str) -> dict[str, float]:
    # Structured metadata first: it is what the Hub renders as 'Evaluation
    # results' and is unambiguous. README tables then fill remaining labels.
    out: dict[str, float] = {}
    try:
        payload = fetch_api_eval_data(repo)
    except Exception as exc:
        print(f"warning: {repo}: eval metadata fetch failed: {exc}", file=sys.stderr)
        payload = {}
    out.update(extract_eval_results(payload))
    for label, value in extract_model_index(payload).items():
        out.setdefault(label, value)
    try:
        md = fetch_readme(repo)
    except FileNotFoundError:
        if not out:
            raise
        return out
    tables = parse_markdown_tables(md) + parse_html_tables(md)
    for label, value in extract_scores_from_tables(tables, repo).items():
        out.setdefault(label, value)
    return out


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
