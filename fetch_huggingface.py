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
from functools import lru_cache
from pathlib import Path
from typing import Any

import _cache

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")
HF_BASE = "https://huggingface.co"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"

# Seconds a model-card crawl may be served from ~/.cache/ai-bench; 0 turns the
# cache off, which is what a test replacing the reader wants.
CRAWL_CACHE_TTL_VAR = "AI_BENCH_HF_CACHE_TTL"

# How much of the crawl has to have come back before it is worth storing.
_MIN_CACHEABLE_SHARE = 0.9


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


# Inline markup a card writes *inside* a cell. Read literally it lands in the
# benchmark label -- the name mapping carries keys like
# "AIME24<sub>(Mean@32)</sub>" for exactly that reason -- so every tag becomes a
# space and the spaces are collapsed after. A space rather than nothing,
# because `<br>` is a line break: "Terminal-Bench<br>4.0 (Pass@1)" has to read
# as one line, not as one word.
_INLINE_TAG_RE = re.compile(r"<[^<>]{0,400}?>")
# "[Qwen3-Next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B)": the link text
# is the name and the URL is noise the model-name match would otherwise have to
# see past -- and a link to a *sibling* repo is how one variant's row gets read
# as another's.
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def clean_cell(text: str) -> str:
    """One line of plain text for a table cell.

    Decodes HTML entities, drops inline markup (`<br>`, `<sub>`, `<b>`, ...),
    unwraps markdown links, flattens every kind of line break and run of
    whitespace to a single space, and strips markdown decoration, footnote
    markers and numeric-only parens. What comes out is what a reader of the
    rendered card sees, and it is what both the benchmark label and the
    model-name match are read from.
    """
    s = html.unescape(text)
    for ch in _INVISIBLE_SPACES:
        s = s.replace(ch, " ")
    # After unescape, so an entity that decoded into a tag is dropped too.
    s = _INLINE_TAG_RE.sub(" ", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Markdown backslash escapes ("𝛕3\-Banking", "Avg\*") from doc exports.
    s = re.sub(r"\\([\\`*_{}\[\]()#+.!|~-])", r"\1", s)
    s = _strip_emphasis(s)
    # Emphasis a partial wrap leaves behind ("**Solar Open 2** 250B-A15B"),
    # which _strip_emphasis only removes when it wraps the whole cell.
    s = re.sub(r"(?<!\w)\*\*(?=\S)|(?<=\S)\*\*(?!\w)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(_FOOTNOTE_CHARS).rstrip()
    # Drop trailing footnote refs like " (1)" or " (12,3)" but preserve informative parens.
    s = re.sub(r"\s*\(\s*[\d,]+\s*\)\s*$", "", s)
    return s.strip()


# The grouped form has to come first: against "1,441" the bare alternative
# matches at the same position and stops at the "1", so an Elo-scale column
# (GDPval-AA, AA-Briefcase) would store a value three orders of magnitude off
# rather than miss it. Only strict three-digit groups qualify, so a card
# writing a decimal comma ("63,1") falls to the bare branch, where
# _DECIMAL_COMMA_RE reads it as 63.1 rather than as 63.
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
# A comma with one or two digits after it and no third digit is a decimal
# separator, which is how a German- or French-language card writes a score.
# Three digits after it is the thousands group _NUMBER_RE already reads.
_DECIMAL_COMMA_RE = re.compile(r"(?<![\d,])(\d{1,3}),(\d{1,2})(?![\d,])")
_PLACEHOLDERS = {"", "-", "—", "–", "n/a", "na", "/", "?", "—%", "tbd", "x", "✗", "✓"}

# A rank is not a score. "1st" in a leaderboard-position column used to store a
# 1, which on a percentage benchmark is indistinguishable from a real score and
# lands as the model's worst result anywhere.
_ORDINAL_RE = re.compile(r"^\s*[-+]?\d+\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
# "10/50 steps: 40.1", "Avg: 63.2": the number that belongs to the cell is the
# one after the colon, and the ones before it count something else.
_LABELLED_VALUE_RE = re.compile(r":\s*([-+]?[\d.,]+)\s*%?\s*$")
# A score stands on its own in the cell. Qwen's cards write "Without CI 83.7
# With CI 90.2" where both settings are published, so a word in front of the
# number is allowed -- but a number *wedged into* a word is not a score, and
# that is the common case and the expensive one: the first number in
# "NVIDIA-Nemotron-3-Nano-Omni-30B" is "-3", which stored as a score is a
# negative percentage nobody can trace back to a typo, and "8B / 16B" is a
# parameter count, not an 8.
_STANDALONE_NUMBER_RE = re.compile(
    r"(?:^|(?<=[\s(\[~≈<>=±/|]))"
    r"([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)"
    r"(?=$|[\s%)\]/|,;:±" + _FOOTNOTE_CHARS.replace("*", "\\*") + r"])"
)
# "43.2 (no tools) / 57.4 (with tools)": two runs in one cell, each with its own
# qualifier and neither of them the cell's value. Taking the first number here
# is taking whichever run the card happened to print first, so the cell is
# skipped instead and the two runs are left to the labels that name them.
_ALTERNATIVES_RE = re.compile(
    r"[\d.,]+\s*%?\s*\([^)]*[A-Za-z][^)]*\)\s*[/|]\s*[\d.,]+\s*%?\s*\([^)]*[A-Za-z][^)]*\)"
)


def parse_score(text: str) -> float | None:
    """The numeric value a table cell reports, or None when it reports none.

    None covers the placeholders a card writes for "not run" as well as the
    cells whose first number is not the cell's value: a lower-is-better pair, a
    leaderboard rank, and two qualified runs printed side by side.
    """
    # Lower-is-better markers ("Violation (↓): 26.4 <br> Coverage: 64.8") flag
    # cells whose first number is not a benchmark score in the usual sense.
    if "↓" in text:
        return None
    cleaned = clean_cell(text)
    if cleaned.lower() in _PLACEHOLDERS:
        return None
    if _ORDINAL_RE.match(cleaned):
        return None
    if _ALTERNATIVES_RE.search(cleaned):
        return None
    labelled = _LABELLED_VALUE_RE.search(cleaned)
    if labelled:
        cleaned = labelled.group(1)
    cleaned = _DECIMAL_COMMA_RE.sub(r"\1.\2", cleaned)
    match = _STANDALONE_NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


# Soft line breaks that word-processor exports leave inside table cells;
# str.splitlines() treats them as row boundaries, which shreds the table.
_SOFT_LINE_BREAKS = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def _logical_lines(md: str) -> list[str]:
    for ch in _SOFT_LINE_BREAKS:
        md = md.replace(ch, " ")
    return md.splitlines()


# A table row: opens with a pipe and carries at least one more. The closing
# pipe is optional, because a card that leaves it off is still a table to every
# renderer -- DeepSeek-R1-0528's whole comparison table is written that way, and
# requiring the trailing pipe is why not one of its benchmarks was read here.
# The separator line underneath is what actually says "this is a table", so
# dropping the requirement costs nothing: prose starting with a pipe does not
# have a row of dashes beneath it.
_PIPE_LINE_RE = re.compile(r"^\s*\|[^|]*\|")
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
    "dataset", "datasets", "seeddataset", "datasetsize", "samples", "modality",
}
_LABEL_DENY_SUB = (
    "parameter", "totalparam", "activatedparam", "activeparam", "contextlength",
    "embedding", "attentionhead", "headsize", "kvhead", "mamba",
    "activation", "sequencelength", "positionembedding", "hiddensize", "vocab",
    "statesize", "numberoflayers", "numlayers", "layers",
    # Serving and packaging rows that sit in the same tables as the scores:
    # "Prefill (tok/s)", "Decode (tok/s)", "Size (GB)", "Effective bpw".
    "toks", "tokenspersecond", "sizegb", "bpw", "diskspace", "memoryfootprint",
)
# "expert" is an architecture word ("# Experts", "Experts per Token") and also
# the opening of a benchmark's own caption ("HLE — Expert-level reasoning"), so
# it is matched as a whole word rather than as a substring: the spec rows write
# it on its own, a caption hyphenates it into the next one. As a substring it
# ate every HLE row on IFM's cards, which caption each benchmark inside its
# label cell.
_LABEL_DENY_WORDS = {"expert", "experts"}
# Context-length column labels like "4k", "128k", "1000k", "2m".
_CONTEXT_LEN_RE = re.compile(r"^\d+(?:\.\d+)?[km]$")


def _is_metadata_label(label: str) -> bool:
    """True for non-benchmark labels (model specs/metadata) that should not be scored."""
    # "HF-ASR (WER↓)", "Latency (s, ↓)": the arrow says the column is scored
    # the other way up, and a value read off it means the opposite of what
    # every benchmark column here holds.
    if "↓" in label:
        return True
    n = _norm_for_match(label)
    if not n or n in _LABEL_DENY_EXACT or _CONTEXT_LEN_RE.match(n):
        return True
    if any(word.strip(".:,#()") in _LABEL_DENY_WORDS for word in label.lower().split()):
        return True
    return any(sub in n for sub in _LABEL_DENY_SUB)


# --------------------------------------------------------------- model names
#
# The whole ingest turns on one question: which column of a comparison table is
# *this* model? A card answers it in whatever shorthand its author had in mind.
# DeepSeek's own card heads its frontier table "DS-V4.1-Flash", Upstage writes
# "Solar Open (102B)" for a repo called Solar-Open-100B, NVIDIA writes
# "N-3-Ultra <br> 550B-A55B". Read literally none of those name the repo, and a
# table nothing matches is dropped whole -- which is why every agentic number
# on DeepSeek-V4.1-Flash's card was invisible here while its *base* table, the
# one column whose header spells the repo out, was read.
#
# So the repo name is expanded into the spellings a card is likely to use, and
# a header is matched against all of them. Three kinds, weakest last, because
# the penalty each carries is what keeps a loose reading from beating a literal
# one when both are on the page:
#
#   * the name as the repo writes it;
#   * with each camel-cased word replaced by its initials (DeepSeek -> DS,
#     LongCat -> LC), which is the shorthand a lab uses for its own family;
#   * with one word replaced by its first letter (Nemotron -> N), which is
#     weaker still and only ever matches when everything else lines up.
#
# Anything those miss is a mapping entry rather than a cleverer rule: see
# load_model_column_aliases() below.

_MODEL_COLUMN_MAPPING = Path(__file__).resolve().with_name(
    "huggingface-model-column-mapping.json"
)


@lru_cache(maxsize=4)
def load_model_column_aliases(path: Path = _MODEL_COLUMN_MAPPING) -> dict[str, list[str]]:
    """Hand-written 'this repo is this column' entries, keyed by HF repo.

    The heuristics below cover the shorthands that follow a rule. This file is
    for the ones that do not -- a codename, an internal name, a family name the
    card never spells out -- and is read rather than guessed at, because a
    wrong column is a wrong score under the right name, which is worse than no
    score at all.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for repo, names in raw.items():
        if not isinstance(repo, str) or repo.startswith("#"):
            continue
        if isinstance(names, str):
            names = [names]
        if isinstance(names, list):
            picked = [n for n in names if isinstance(n, str) and n.strip()]
            if picked:
                out[repo] = picked
    return out


def _norm_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# "Olmo3" and "Olmo 3" are one name written two ways, and a card uses both on
# one page. Splitting a word from the number glued to its end makes the two
# tokenise alike; the boundary needs two letters in front of it so a version
# prefix ("V2", "R1") and a parameter count ("7b", "a3b") stay whole.
_GLUED_NUMBER_RE = re.compile(r"(?<=[a-z]{2})(?=\d)")


def _tokens(s: str) -> set[str]:
    out: set[str] = set()
    for raw in re.split(r"[^a-z0-9]+", s.lower()):
        if not raw:
            continue
        out.update(part for part in _GLUED_NUMBER_RE.split(raw) if part)
    return out


# "30b", "a3b", "1t", "900m": a parameter count, which two spellings of one
# model disagree about more often than they disagree about its name.
_PARAM_TOKEN_RE = re.compile(r"^a?\d+(?:\.\d+)?[bmt]$")
_PARAM_SCALE = {"m": 0.001, "b": 1.0, "t": 1000.0}


def _param_size(token: str) -> float | None:
    if not _PARAM_TOKEN_RE.match(token):
        return None
    body = token[1:] if token.startswith("a") else token
    try:
        return float(body[:-1]) * _PARAM_SCALE[body[-1]]
    except (ValueError, KeyError):
        return None


def _split_param_tokens(tokens: set[str]) -> tuple[set[str], set[str]]:
    sizes = {t for t in tokens if _PARAM_TOKEN_RE.match(t)}
    return tokens - sizes, sizes


def _camel_parts(word: str) -> list[str]:
    return re.findall(r"[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+", word)


def _name_spellings(name: str) -> list[tuple[str, int]]:
    """(spelling, how far it is from the literal name); the literal one first."""
    spellings: list[tuple[str, int]] = [(name, 0)]
    words = re.split(r"([^A-Za-z0-9]+)", name)
    # Every camel-cased word at once: "DeepSeek-V4.1-Flash" -> "DS-V4.1-Flash".
    initialled = list(words)
    changed = False
    for idx, word in enumerate(words):
        parts = _camel_parts(word)
        if len(parts) >= 2 and word.isalpha():
            initialled[idx] = "".join(part[0] for part in parts)
            changed = True
    if changed:
        spellings.append(("".join(initialled), 1))
    # One word at a time down to its first letter: "Nemotron-3-Ultra" ->
    # "N-3-Ultra". Weakest, and only ever reached when the rest of the header
    # already lines up.
    for idx, word in enumerate(words):
        if word.isalpha() and len(word) >= 4:
            shortened = list(words)
            shortened[idx] = word[0]
            spellings.append(("".join(shortened), 2))
    return spellings


# Words that name a *variant* rather than a model: the same card often carries
# several columns whose names differ only by one of these, and picking the
# wrong one stores a number the model never produced under its name.
_BASE_WORDS = {"base", "pretrain", "pretrained", "pretraining"}
_EFFORT_WORDS = {
    "high", "max", "maximum", "xhigh", "low", "medium", "mid", "minimal",
    "effort", "budget",
}
# Words already weighed by _variant_penalty, so _foreign_word_penalty leaves
# them alone rather than charging for them twice.
_MODE_TOKENS = {
    "thinking", "think", "reasoning", "reason", "cot", "non", "nonthinking",
    "instruct", "instructed", "chat", "it", "tools", "tool",
}
_TOOLS_RE = re.compile(r"\bw/\s*tools?\b|\bwith\s+tools?\b|\btool[- ]augmented\b", re.IGNORECASE)
_THINKING_RE = re.compile(r"\bthinking\b|\bthink\b|\breasoning\b|\bcot\b|\bhigh reasoning\b", re.IGNORECASE)
_NON_THINKING_RE = re.compile(
    r"\bnon[- ]?thinking\b|\bno[- ]?think(?:ing)?\b|\bnon[- ]?reasoning\b|\bwithout thinking\b"
    r"|\binstruct(?:ed)?\b|\bnon[- ]?think\b",
    re.IGNORECASE,
)


def _reasoning_mode(text: str) -> str | None:
    """'thinking', 'non-thinking', or None when the name does not say.

    Checked as phrases rather than tokens because "Non-thinking" tokenises to
    {non, thinking} -- the two modes share every word that matters.
    """
    if _NON_THINKING_RE.search(text):
        return "non-thinking"
    if _THINKING_RE.search(text):
        return "thinking"
    return None


@dataclass(frozen=True)
class ModelNames:
    """Every spelling of one repo's model, and the variant it is."""

    display: str
    spellings: tuple[tuple[str, frozenset[str], int], ...]
    is_base: bool
    mode: str | None

    @property
    def sizes(self) -> set[str]:
        return {t for _, tokens, _ in self.spellings for t in tokens if _PARAM_TOKEN_RE.match(t)}


def model_names(repo: str, slug: str | None = None) -> ModelNames:
    """The names a card may use for the model `repo` holds.

    `slug` is llm.json's own name for the row being filled, which is the only
    thing that says which variant of a hybrid model the row wants: the repo
    `Qwen/Qwen3-8B` serves both modes and the row asking for it is
    `qwen3-8b-instruct`.
    """
    display = repo.split("/")[-1]
    sources = [display]
    sources.extend(load_model_column_aliases().get(repo, []))
    # A mapped alias is a statement about this repo, so it is expanded and
    # ranked exactly as the repo's own name is.
    seen: dict[str, tuple[frozenset[str], int]] = {}
    for source in sources:
        for spelling, rank in _name_spellings(source):
            norm = _norm_for_match(spelling)
            if not norm:
                continue
            previous = seen.get(norm)
            if previous is None or rank < previous[1]:
                seen[norm] = (frozenset(_tokens(spelling)), rank)
    named = " ".join(sources)
    hint = f"{named} {slug or ''}"
    return ModelNames(
        display=display,
        spellings=tuple((norm, tokens, rank) for norm, (tokens, rank) in seen.items()),
        is_base=bool(_tokens(named) & _BASE_WORDS),
        mode=_reasoning_mode(hint.replace("-", " ")),
    )


# How much worse a header reading has to be before a better one outranks it.
_BASE_PENALTY = 60          # a base-model column when the repo is post-trained
_MODE_PENALTY = 40          # "Thinking" where the row wants "Instruct"
_UNASKED_MODE_PENALTY = 6   # a mode qualifier where the row names none
_TOOLS_PENALTY = 30         # a with-tools column beside a plain one
_EFFORT_PENALTY = 4         # "(max)", "(high)": a tiebreak, not a verdict
_SIZE_MISMATCH_PENALTY = 10
# Two parameter counts this far apart are two models, not two spellings of one.
_SIZE_TOLERANCE = 0.1


def _variant_penalty(candidate: str, model: ModelNames) -> int:
    """How much this header disagrees with the variant the repo names."""
    tokens = _tokens(candidate)
    spaced = candidate.replace("-", " ")
    penalty = 0
    if (tokens & _BASE_WORDS) and not model.is_base:
        penalty += _BASE_PENALTY
    if model.is_base and not (tokens & _BASE_WORDS):
        penalty += _BASE_PENALTY
    candidate_mode = _reasoning_mode(spaced)
    if candidate_mode is not None:
        if model.mode is None:
            penalty += _UNASKED_MODE_PENALTY
        elif candidate_mode != model.mode:
            penalty += _MODE_PENALTY
    if _TOOLS_RE.search(candidate):
        penalty += _TOOLS_PENALTY
    penalty += _EFFORT_PENALTY * len(tokens & _EFFORT_WORDS)
    return penalty


def _sizes_agree(candidate_sizes: set[str], model_sizes: set[str]) -> int | None:
    """Penalty for the parameter counts two names carry, or None if they differ.

    One side naming no size at all is the ordinary case -- a card writes
    "Nemotron 3 Super" for a repo called NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    -- and costs nothing. Two sides naming *different* counts is two models,
    unless they are the same count rounded differently: Upstage ships
    Solar-Open-100B and heads its column "Solar Open (102B)".
    """
    if not candidate_sizes or not model_sizes:
        return 0
    if candidate_sizes == model_sizes:
        return 0
    # "122B-A10B" is a 122B model activating 10B per token, so its "A10B" says
    # nothing about a dense 9B: the counts are compared within their own kind,
    # and a name that gives only one kind against a name that gives only the
    # other is not a match at all.
    compared = False
    for active in (False, True):
        c_sizes = [_param_size(t) for t in candidate_sizes if t.startswith("a") == active]
        m_sizes = [_param_size(t) for t in model_sizes if t.startswith("a") == active]
        c_sizes = [s for s in c_sizes if s]
        m_sizes = [s for s in m_sizes if s]
        if not c_sizes or not m_sizes:
            continue
        compared = True
        if not any(
            abs(c - m) / max(c, m) <= _SIZE_TOLERANCE for c in c_sizes for m in m_sizes
        ):
            return None
    return _SIZE_MISMATCH_PENALTY if compared else None


def _is_derived_model(c_tokens: frozenset[str], t_tokens: frozenset[str]) -> bool:
    """True when the candidate is a model *made from* the target, not the target.

    "DeepSeek-R1-0528-Qwen3-8B" is R1-0528 distilled into Qwen3-8B, and it sits
    in a table on R1-0528's own card. It contains the repo's whole name, so
    every substring reading matches it, and what lands is the 8B distill's
    numbers under the 685B model's name. What says "derived" rather than
    "spelled differently" is the pair: a parameter count the repo does not
    carry *and* a family name from somewhere else.
    """
    extra = c_tokens - t_tokens
    has_size = any(_PARAM_TOKEN_RE.match(token) for token in extra)
    foreign = any(
        token.isalpha() and len(token) > 1
        and token not in _BASE_WORDS and token not in _EFFORT_WORDS
        and token not in _MODE_TOKENS
        for token in extra
    )
    return has_size and foreign


def _spelling_score(
    c_norm: str, c_tokens: frozenset[str], t_norm: str, t_tokens: frozenset[str]
) -> int | None:
    if c_norm == t_norm:
        return 0
    # Allow loose substring matches if length difference is modest.
    if t_norm in c_norm and len(c_norm) - len(t_norm) <= 24:
        if _is_derived_model(c_tokens, t_tokens):
            return None
        return len(c_norm) - len(t_norm)
    if c_norm in t_norm and len(t_norm) - len(c_norm) <= 24 and len(c_norm) >= 6:
        return (len(t_norm) - len(c_norm)) + 100  # weaker than "candidate contains target"
    # Token-subset match: names often drop size/param tokens the repo name
    # carries (e.g. "Mellum2 Thinking" vs "Mellum2-12B-A2.5B-Thinking"), which
    # breaks contiguous-substring matching.
    # The same words in another order or with the spaces moved: "Olmo3
    # Instruct 7B" is Olmo-3-7B-Instruct, which no substring reading sees.
    if c_tokens == t_tokens:
        return 2
    shared = c_tokens & t_tokens
    if (c_tokens <= t_tokens or t_tokens <= c_tokens) and len("".join(shared)) >= 6:
        return 200 + len(c_tokens ^ t_tokens)  # weaker than substring matches
    # The same reading with the parameter counts set aside, which is how
    # "Solar Open (102B)" names Solar-Open-100B. Weaker again, and only when
    # the counts are close enough to be the same model.
    c_core, c_sizes = _split_param_tokens(set(c_tokens))
    t_core, t_sizes = _split_param_tokens(set(t_tokens))
    if c_core and t_core and (c_core <= t_core or t_core <= c_core):
        if len("".join(c_core & t_core)) >= 6:
            size_penalty = _sizes_agree(c_sizes, t_sizes)
            if size_penalty is not None:
                return 300 + len(c_core ^ t_core) + size_penalty
    return None


# A word the header adds that names something else entirely. The case that
# costs a score is a derived model sitting in the same table as the one it was
# derived from: "DeepSeek-R1-0528-Qwen3-8B" contains "DeepSeek-R1-0528", so a
# substring reading matches it, and the 8B distill's numbers land on the 685B
# model's row. Weighted to lose to any reading of the model's own name while
# still matching when it is the only reading on the page.
_FOREIGN_WORD_PENALTY = 50


# A release date or build number, which is the whole difference between two
# models a card lists side by side: Qwen's long-context table carries
# "Qwen3-235B-A22B (Thinking)" beside "Qwen3-235B-A22B-Thinking-2507", and only
# the "2507" says which of them the repo is.
_VERSION_TOKEN_RE = re.compile(r"^\d{3,}$")
_MISSING_VERSION_PENALTY = 60


def _missing_version_penalty(c_tokens: frozenset[str], model: ModelNames) -> int:
    wanted = {
        token
        for _, tokens, rank in model.spellings
        if rank == 0
        for token in tokens
        if _VERSION_TOKEN_RE.match(token)
    }
    return _MISSING_VERSION_PENALTY * len(wanted - c_tokens)


def _foreign_word_penalty(c_tokens: frozenset[str], model: ModelNames) -> int:
    known: set[str] = set()
    for _, tokens, _ in model.spellings:
        known |= tokens
    extra = {
        token
        for token in c_tokens - known
        if token.isalpha()
        and len(token) > 1
        and not _PARAM_TOKEN_RE.match(token)
        and token not in _BASE_WORDS
        and token not in _EFFORT_WORDS
        and token not in _MODE_TOKENS
    }
    return _FOREIGN_WORD_PENALTY * len(extra)


def _tier_floor(score: int) -> int:
    """The tier a score sits in, without the detail that separates names inside it."""
    return (score // 100) * 100


def _name_match_score(candidate: str, model: ModelNames) -> int | None:
    """Score how well `candidate` names the model (lower = better; None = no match)."""
    c_norm = _norm_for_match(candidate)
    if not c_norm:
        return None
    c_tokens = frozenset(_tokens(candidate))
    best: int | None = None
    for t_norm, t_tokens, rank in model.spellings:
        if not t_norm:
            continue
        score = _spelling_score(c_norm, c_tokens, t_norm, t_tokens)
        if score is None:
            continue
        score += rank
        if best is None or score < best:
            best = score
    if best is None:
        return None
    if _reasoning_mode(candidate.replace("-", " ")) is not None and model.mode is None:
        # "(Thinking)" and "(Non-thinking)" are the same name with a mode
        # attached, and the second is three characters longer. Without this the
        # length of the qualifier picks the mode, which is how a hybrid model's
        # row quietly took its thinking numbers.
        best = _tier_floor(best)
    return (
        best
        + _variant_penalty(candidate, model)
        + _foreign_word_penalty(c_tokens, model)
        + _missing_version_penalty(c_tokens, model)
    )


_SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?b$")


def _select_size_variant_column(table: Table, model: ModelNames) -> int | None:
    """Match size-variant headers like '8B Dense' for repos like 'granite-4.1-8b'.

    Fires only when several headers start with a size token and share the same
    remainder (one model line, varying parameter count), so a lone competitor
    column in a comparison table is not mistaken for the model.
    """
    model_sizes = {t for t in model.sizes if _SIZE_TOKEN_RE.match(t)}
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
# Headers over a column of model names, which is what a transposed table --
# benchmarks across the top, one row per model -- puts in front of its rows.
# Usually the first column; "| Benchmark | ... | Model |" puts it last.
_MODEL_COLUMN_HEADERS = {
    "model", "models", "modelname", "name", "system", "method", "llm",
    "variant", "checkpoint",
}

# Match score assigned to the size-variant fallback: weaker than any direct
# name match so exact-name tables win when scores are merged across tables.
_SIZE_VARIANT_SCORE = 400


# "Benchmark", "Benchmark (Metric)", "Task", "Dataset": a header that opens
# with one of these words names the column of benchmark names. The word has to
# end there -- "Taskmaster-3B" is a model, not a column of tasks.
_LABEL_COLUMN_RE = re.compile(
    r"^(?:benchmarks?|tasks?|datasets?|evals?|evaluations?)\b", re.IGNORECASE
)


def _find_label_column(table: Table) -> int:
    """Column holding benchmark names. Usually 0, but tables like
    `| Category | Benchmark (Metric) | Model-X | ... |` put a grouping column
    first, and reading the grouping column as the labels loses the table."""
    for idx, header in enumerate(table.headers):
        if _LABEL_COLUMN_RE.match(clean_cell(header)):
            return idx
    return 0


def _best_scored(
    scored: list[tuple[int, int, str]], what: str, model: ModelNames
) -> tuple[int, int] | None:
    """The single best candidate, or None when two *different* names tie.

    A tie between two names the card spells differently -- "Thinking" and
    "Non-thinking" on a hybrid model whose row names neither -- is a question
    this parser cannot answer. Picking either stores a number under a name that
    does not describe it, and picking the leftmost is picking whichever the
    author typed first, so the table is left to whichever other table names the
    model outright.

    A tie between two cells carrying the *same* name is not that: a download
    table lists one model once per precision, and the first of them is as good
    as the last.
    """
    if not scored:
        return None
    best = min(scored)
    tied = [entry for entry in scored if entry[0] == best[0]]
    if len({_norm_for_match(entry[2]) for entry in tied}) > 1:
        names = ", ".join(sorted({entry[2] for entry in tied})[:4])
        print(
            f"warning: {model.display}: {len(tied)} {what}s match equally well "
            f"({names}); skipping the table",
            file=sys.stderr,
        )
        return None
    return (best[0], best[1])


def _select_column_scored(table: Table, model: ModelNames) -> tuple[int, int] | None:
    """(match_score, col_idx) for the column naming this model; lower is better."""
    label_col = _find_label_column(table)
    scored: list[tuple[int, int, str]] = []
    for idx, header in enumerate(table.headers):
        if idx == label_col:
            continue
        name = clean_cell(header)
        score = _name_match_score(name, model)
        if score is not None:
            scored.append((score, idx, name))
    best = _best_scored(scored, "column", model)
    if best is not None:
        return best
    size_col = _select_size_variant_column(table, model)
    if size_col is not None:
        return (_SIZE_VARIANT_SCORE, size_col)
    return None


def select_column(table: Table, repo: str) -> int | None:
    """Pick the column matching the HF repo's model name (last path segment)."""
    scored = _select_column_scored(table, model_names(repo))
    return scored[1] if scored else None


def _name_columns(table: Table) -> list[int]:
    """Columns that could hold model names in a transposed table.

    The first column normally, the last where a card puts its legend on the
    right, and any column whose header says so outright.
    """
    width = max((len(row) for row in table.rows), default=0)
    width = max(width, len(table.headers))
    if width <= 1:
        return [0]
    candidates = [0, width - 1]
    for idx, header in enumerate(table.headers):
        if _norm_for_match(clean_cell(header)) in _MODEL_COLUMN_HEADERS:
            candidates.append(idx)
    return list(dict.fromkeys(candidates))


def _select_row_scored(table: Table, model: ModelNames) -> tuple[int, int, int] | None:
    """(match_score, row_idx, name_col) for transposed tables (models in rows)."""
    scored: list[tuple[int, int, str]] = []
    placement: dict[int, int] = {}  # row -> name column it matched in
    for name_col in _name_columns(table):
        for idx, row in enumerate(table.rows):
            if name_col >= len(row):
                continue
            name = clean_cell(row[name_col])
            score = _name_match_score(name, model)
            if score is None:
                continue
            previous = [entry for entry in scored if entry[1] == idx]
            if previous and previous[0][0] <= score:
                continue
            scored = [entry for entry in scored if entry[1] != idx]
            scored.append((score, idx, name))
            placement[idx] = name_col
    best = _best_scored(scored, "row", model)
    if best is None:
        return None
    return (best[0], best[1], placement[best[1]])


def select_row(table: Table, repo: str) -> int | None:
    """For transposed tables (models in rows): pick the row whose name cell names the model."""
    scored = _select_row_scored(table, model_names(repo))
    return scored[1] if scored else None


def _promote_header_band(table: Table, model: ModelNames) -> Table:
    """Re-read a table whose header row is a caption band over the real header.

    Cards increasingly group their comparison columns under a caption first --
    an empty cell or two, then one `colspan` cell reading "Reference models" or
    "Open-weight models" -- and put the model names in the row below it. Read
    literally the caption becomes the header, nothing names the model, and the
    table is dropped whole: that is why every benchmark on IFM's K2-Horizon
    cards was invisible here.

    Which row is the header cannot be read off the markup, because the shape
    that says "caption band" and the shape that says "two-row header" are the
    same colspan. MiniCPM5's card leads with `rowspan` header cells and puts
    the *rest* of its comparison columns in the second row -- promote there and
    the model's own column is lost. So the rows decide: a table is re-read one
    row down only when nothing in it names the model as it stands, and the row
    below does name it, as a column beside the labels rather than as a row of
    its own. Anything this parser already matched is returned untouched.
    """
    if _select_column_scored(table, model) is not None or _select_row_scored(table, model) is not None:
        return table
    if len(table.rows) < 2:
        return table
    # A caption band labels groups of columns, so it names fewer than half of
    # them; a header row that fills its own width is a header, however little
    # it matched. Without this an artifact table ("Model card | Hugging Face
    # <link> | Available") promotes too, because the link cell carries the repo
    # name and the row above it never mentions the model either.
    captions = [cell for cell in table.headers if cell.strip()]
    if len(captions) * 2 > len(table.rows[0]):
        return table
    candidate = Table(headers=table.rows[0], rows=table.rows[1:])
    scored = _select_column_scored(candidate, model)
    if scored is None or scored[1] == _find_label_column(candidate):
        return table
    return candidate


# A column of values that are all at most 1 is a column of fractions, whatever
# the header says: Mistral's Ministral cards report "MMLU 5-shot" as 0.794
# where every other card reports 79.4, and stored as-is that lands in llm.json
# as a sub-1% score. Three values are asked for before rescaling, because one
# or two near-zero numbers are what a genuinely hard benchmark looks like --
# ZeroBench's whole published field is 0.0 to 12.0 -- and because a card that
# reports a single fraction gives nothing to tell the two apart.
_MIN_FRACTION_VALUES = 3
# A fraction column has to reach far enough up its own range to be one. Under
# this, "all values <= 1" is just a hard benchmark.
_FRACTION_FLOOR = 0.2


# A benchmark whose splits are rows of their own, indented under its name:
#
#   | TauBench V3            |      |      |
#   | &nbsp;&nbsp;Airline    | 81.5 | 75.3 |
#   | &nbsp;&nbsp;Telecom    | 92.9 | 89.6 |
#
# Read one row at a time those labels are "Airline" and "Telecom", which name
# nothing and can only ever be parked. Read with the row they hang off they are
# "TauBench V3 Airline" and "TauBench V3 Telecom", which are columns this site
# carries. The indent is the whole signal, and a card writes it with a
# non-breaking space or a bullet rather than with plain spaces, because the
# table syntax eats those. Emphasis is deliberately not an indent: "**Safety**"
# is a row in bold, not a row nested under the one above it.
_SUBROW_RE = re.compile(
    "^(?:[\\s\u00a0\u2002\u2003\u2007\u2009]+|[•↳└├─→>]+[\\s\u00a0]*|[-–—]\\s+|\\|_\\s*)(?=\\S)"
)
_SUBROW_STRIP = " \t\u00a0\u2002\u2003\u2007\u2009•↳└├─→>-–—|_"


def _subrow_label(raw: str) -> str | None:
    """The label of a row indented under the benchmark above it, or None."""
    text = html.unescape(raw)
    if not _SUBROW_RE.match(text):
        return None
    stripped = text.lstrip(_SUBROW_STRIP)
    # "- 12.4" is a value, not a nested label.
    if not stripped or not any(ch.isalpha() for ch in stripped):
        return None
    return clean_cell(stripped)


def _rescale_fractions(found: dict[str, float]) -> dict[str, float]:
    values = [v for v in found.values() if v is not None]
    if len(values) < _MIN_FRACTION_VALUES:
        return found
    if any(v < 0 or v > 1 for v in values):
        return found
    if max(values) < _FRACTION_FLOOR or len({round(v, 4) for v in values}) < 2:
        return found
    return {label: round(value * 100, 4) for label, value in found.items()}


def extract_scores_from_tables(
    tables: list[Table], repo: str, slug: str | None = None
) -> dict[str, float]:
    # Each table is extracted independently, then merged best-name-match first:
    # a table naming the model exactly must beat one that only matched loosely
    # (e.g. the "MiMo-V2-Flash Base" pretrain column when the repo is the
    # post-trained MiMo-V2-Flash), regardless of document order.
    model = repo if isinstance(repo, ModelNames) else model_names(repo, slug)
    extracts: list[tuple[int, dict[str, float]]] = []  # (match_score, scores)
    for table in tables:
        if not table.headers or not table.rows:
            continue
        table = _promote_header_band(table, model)
        scored_col = _select_column_scored(table, model)
        label_col = _find_label_column(table)
        if scored_col is not None and scored_col[1] != label_col and scored_col[1] < len(table.headers):
            # Column orientation: benchmarks in `label_col`, this model in `col`.
            match_score, col = scored_col
            found: dict[str, float] = {}
            heading = ""  # the last value-less row, which nested rows hang off
            for row in table.rows:
                if label_col > 0 and len(row) == len(table.headers) - 1:
                    # Category tables often omit the leading category cell on
                    # continuation rows, shifting everything left by one.
                    row = [""] + row
                if not row or _is_category_row(row, len(table.headers)):
                    if row and label_col < len(row):
                        heading = clean_cell(row[label_col])
                    continue
                if col >= len(row) or label_col >= len(row):
                    continue
                nested = _subrow_label(row[label_col])
                label = nested if nested else clean_cell(row[label_col])
                if nested and heading:
                    label = f"{heading} {nested}"
                if not label or _is_metadata_label(label):
                    continue
                value = parse_score(row[col])
                if value is None:
                    continue
                found.setdefault(label, value)
            extracts.append((match_score, _rescale_fractions(found)))
            continue
        # Transposed orientation: this model is a row, benchmarks are the headers.
        scored_row = _select_row_scored(table, model)
        if scored_row is None:
            continue
        match_score, row_idx, name_col = scored_row
        row = table.rows[row_idx]
        found = {}
        for j in range(len(table.headers)):
            if j == name_col:
                continue
            if j >= len(row):
                break
            label = clean_cell(table.headers[j])
            if not label or _is_metadata_label(label):
                continue
            value = parse_score(row[j])
            if value is None:
                continue
            found.setdefault(label, value)
        extracts.append((match_score, _rescale_fractions(found)))
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


# Datasets whose score means a different thing with tools than without, where
# one card routinely carries both runs under the same dataset id. Only the
# entry's free-text ``notes`` says which is which, so for these -- and only for
# these -- the mode is folded into the label, splitting one ambiguous key into
# ``cais/hle (no tools)`` and ``cais/hle (with tools)`` that the benchmark-name
# mapping can answer separately. Kept to a named set rather than applied to
# every dataset because a note is free text: a general rule would mint a new
# label for every phrasing a card author invents, and flood the mapping queue
# with questions nobody needs to answer. Tool use is the *point* of an agentic
# benchmark, so nothing agentic belongs here.
#
# HLE qualifies on the numbers: across the 25 models where one publisher states
# both modes, tools are worth a median +11.5 points and up to +27.1. See
# docs/hle-tool-mode-audit-2026-09.md.
TOOL_MODE_SENSITIVE_DATASETS = {"cais/hle"}

_WITH_TOOLS_RE = re.compile(
    r"\bwith(?:\s+|-)(?:tools?|search|browsing|retrieval)\b|\bw/\s*tools?\b|\btool[- ]augmented\b",
    re.IGNORECASE,
)
_NO_TOOLS_RE = re.compile(
    r"\bno\s+tools?\b|\bwithout\s+tools?\b|\bw/o\s+tools?\b|\btool[- ]free\b",
    re.IGNORECASE,
)


def _tool_mode(notes: Any) -> str | None:
    """'no tools', 'with tools', or None when the note does not say.

    A note carrying both phrasings ("With tools: 57.4%. Without tools: 43.2%")
    describes two runs and names neither as its own value, so it says nothing
    about *this* entry and returns None.
    """
    if not isinstance(notes, str) or not notes:
        return None
    with_tools = bool(_WITH_TOOLS_RE.search(notes))
    no_tools = bool(_NO_TOOLS_RE.search(notes))
    if with_tools and no_tools:
        return None
    if no_tools:
        return "no tools"
    if with_tools:
        return "with tools"
    return None


# A note that names the *run* rather than the model: which harness drove it,
# which agent scaffold, which reasoning effort, how many video frames, whether
# tools were allowed. One card routinely files the same dataset several times,
# once per setting, and every one of them arrives under the same dataset id --
# the label cannot tell them apart, so the entries themselves have to.
_QUALIFIED_RUN_RE = re.compile(
    r"\bharness\b|\bscaffold\b|\bagent\s*[:=]|\breasoning\s*[:=]|\beffort\b"
    r"|\b\d+\s*frames?\b|\bpass@\d+\b|\bavg@\d+\b|\bmajority\b|\bbest[- ]of\b"
    r"|\bwith\s+tools?\b|\bw/\s*tools?\b|\bno\s+tools?\b|\bwithout\s+tools?\b"
    r"|\bthinking\s+(?:on|off|enabled|disabled)\b|\bnon[- ]thinking\b",
    re.IGNORECASE,
)


def _is_qualified_run(notes: Any) -> bool:
    return isinstance(notes, str) and bool(_QUALIFIED_RUN_RE.search(notes))


def _pick_eval_entry(entries: list[dict[str, Any]], label: str) -> float | None:
    """The one value a label's entries report, or None when they do not agree.

    A card files one dataset several times and the widget shows every one of
    them. Where the runs differ, an unqualified entry -- one whose note names
    no harness, scaffold, effort or tool mode -- is the card's own headline and
    wins outright. Where *every* entry names a run and they disagree, the card
    is reporting a spread rather than a score: NVIDIA files SWE-bench Verified
    three times, at 60.47 on OpenHands, 59.2 on OpenCode and 53.73 on Codex,
    and nothing on the page says which of the three the benchmark's column
    should carry. Taking whichever the Hub returned first is how a harness
    number lands under a plain benchmark's name, so nothing is taken.

    Between entries that all qualify equally, one merged into the card beats
    one still pending on an open Hub PR, and a later date beats an earlier one.
    """
    plain = [entry for entry in entries if not _is_qualified_run(entry.get("notes"))]
    considered = plain or entries
    values = {round(entry["value"], 6) for entry in considered}
    if len(values) > 1 and not plain:
        print(
            f"warning: {label}: {len(values)} qualified runs and no plain one "
            f"({', '.join(str(v) for v in sorted(values))}); skipping",
            file=sys.stderr,
        )
        return None
    # Two stable passes rather than one key: latest date first, then merged
    # entries ahead of ones still pending on an open Hub PR.
    considered.sort(key=lambda entry: (entry["date"] or "", -entry["order"]), reverse=True)
    considered.sort(key=lambda entry: entry["pull_request"] is not None)
    return considered[0]["value"]


def extract_eval_results(payload: dict[str, Any]) -> dict[str, float]:
    """Scores from the Hub's 'Evaluation results' widget (evalResults).

    This is the section a model page renders under "Evaluation results", and it
    is the most trustworthy thing on a card: every entry names its dataset by
    Hub id, so nothing has to be guessed from a heading. What it does not carry
    is which *run* an entry is, beyond a free-text note -- see
    _pick_eval_entry() and TOOL_MODE_SENSITIVE_DATASETS.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    entries = payload.get("evalResults")
    if not isinstance(entries, list):
        return {}
    for order, entry in enumerate(entries):
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
        if dataset.get("id") in TOOL_MODE_SENSITIVE_DATASETS:
            mode = _tool_mode(data.get("notes"))
            if mode is not None:
                label = f"{label} ({mode})"
        grouped.setdefault(label, []).append(
            {
                "value": value,
                "notes": data.get("notes"),
                "date": data.get("date") if isinstance(data.get("date"), str) else "",
                "pull_request": entry.get("pullRequest"),
                "dataset": dataset.get("id"),
                "order": order,
            }
        )
    out: dict[str, float] = {}
    for label, group in grouped.items():
        value = _pick_eval_entry(group, label)
        if value is not None:
            out[label] = value
    return _rescale_eval_fractions(out, {
        label: group[0]["dataset"] for label, group in grouped.items() if group
    })


def _rescale_eval_fractions(
    scores: dict[str, float], datasets: dict[str, Any]
) -> dict[str, float]:
    """Put a dataset reported as 0-1 back on the scale every other card uses.

    Per dataset rather than per card, and by the same three-value rule the
    README tables go by: open-agent-leaderboard files a whole board as
    fractions (0.5204 for SWE-bench), while a single ParseBench chart score of
    0.4 beside siblings in the eighties is a small model scoring 0.4%.
    """
    by_dataset: dict[Any, dict[str, float]] = {}
    for label, value in scores.items():
        by_dataset.setdefault(datasets.get(label), {})[label] = value
    out = dict(scores)
    for group in by_dataset.values():
        out.update(_rescale_fractions(group))
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


def extract_scores(repo: str, slug: str | None = None) -> dict[str, float]:
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
    for label, value in extract_scores_from_tables(tables, repo, slug).items():
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


def crawl_all_models(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Read every Hugging Face model card llm.json points at, cached per crawl.

    One `update-all` walks this twice in two processes minutes apart -- the
    mapping updater for the benchmark labels, then update.py's fetcher for the
    scores -- and it is a hundred and forty model cards each time, the second
    most expensive thing in a refresh. The cache lets the second walk reuse the
    first; nothing else about either changes.

    The key is the (slug, repo) list, so a model added to or renamed in
    llm.json misses rather than being answered from a crawl that never saw it.
    A model whose *card* changed inside the TTL is served the older read, which
    is the same bargain artificialanalysis.py's response cache makes: the cron
    is three hours apart and the TTL is one, so a scheduled refresh always
    re-reads.
    """
    pairs = iter_hf_models(doc)
    ttl = _cache.ttl_seconds(CRAWL_CACHE_TTL_VAR)
    key = _cache.digest(sorted(pairs))
    cached = _cache.load("huggingface-cards", key, ttl, label="Hugging Face model cards")
    if isinstance(cached, list):
        return cached

    results: list[dict[str, Any]] = []
    for slug, repo in pairs:
        try:
            scores = extract_scores(repo, slug)
        except Exception as exc:
            print(f"warning: {slug} ({repo}): {exc}", file=sys.stderr)
            continue
        results.append({"model": slug, "repo": repo, "scores": scores})
    # A short crawl is not stored, the same way artificialanalysis.py refuses
    # to publish a short model list: a run that lost its network partway would
    # otherwise serve a stub to the fetcher that follows it, for an hour.
    #
    # Not "complete", though, because a complete crawl is not the normal case:
    # a gated or withdrawn repo answers 401 or 404 on every run, and a
    # threshold that those models can never meet is a cache that is never
    # written. A card that is missing for that reason is missing from the live
    # crawl too, and the ingest reading it fills gaps rather than overwriting,
    # so it costs a fill, not a score.
    if pairs and len(results) >= _MIN_CACHEABLE_SHARE * len(pairs):
        _cache.store("huggingface-cards", key, results, ttl)
    elif pairs:
        print(
            f"not caching a crawl of {len(results)}/{len(pairs)} model card(s)",
            file=sys.stderr,
        )
    return results


def main() -> int:
    args = parse_args()

    if args.all_models:
        doc = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        results = crawl_all_models(doc)
    else:
        if not args.repo_or_url:
            print("error: repo_or_url is required when --all-models is not set", file=sys.stderr)
            return 2
        repo = normalize_repo(args.repo_or_url)
        scores = extract_scores(repo)
        results: list[dict[str, Any]] = [
            {"model": repo.split("/")[-1], "repo": repo, "scores": scores}
        ]

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
