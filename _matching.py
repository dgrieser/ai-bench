#!/usr/bin/env python3
"""Shared model/benchmark name matching, with a confidence grade.

The primitives here (normalize_slug .. token_match_score) were factored out of
update_artificialanalysis_mapping.py unchanged; that script still uses them for
its candidate list. propose.py uses grade()/propose() on top to decide what may
be written into a proposal PR without a human looking first.

Only normalized equality (MATCH_EXACT at variant index 0) is safe to propose.
Measured against the mapping files' own ground truth, that tier proposed the
right slug 208 times and the wrong one never; admitting substring/subset matches
added 25 correct proposals but also 2 wrong ones, because a shorter name is a
prefix of the next version -- DeepSeek-V3 scores a confident match against
deepseek-v3-2-0925, and hermes-3-70b against hermes-4-llama-3-1-70b. Anything
below exact is offered as a suggestion for a human to click, never committed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DESCRIPTOR_TOKENS = {
    "base",
    "chat",
    "instruct",
    "inst",
    "it",
    "reasoning",
    "thinking",
    "vision",
    "vl",
    "preview",
    "exp",
    "experimental",
    "mini",
    "small",
    "lite",
}
SIZE_WITH_UNIT_RE = re.compile(r"^a?\d+[bkm]$")
SHORT_CODE_RE = re.compile(r"^[a-z]{1,3}\d+$")
MATCH_LABELS = {
    0: "exact match",
    1: "substring match",
    2: "all components match",
    3: "most components match",
}


def normalize_slug(value: str) -> str:
    text = value.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def components(value: str) -> list[str]:
    return [part for part in normalize_slug(value).split("-") if part]


def join_components(parts: list[str]) -> str:
    return "-".join(parts)


def strip_descriptor_suffix(parts: list[str]) -> list[str]:
    out = list(parts)
    while out and out[-1] in DESCRIPTOR_TOKENS:
        out.pop()
    return out


def size_group_indices(parts: list[str]) -> set[int]:
    drop: set[int] = set()
    for idx, part in enumerate(parts):
        match = SIZE_WITH_UNIT_RE.match(part)
        if not match:
            continue
        drop.add(idx)

        prev = idx - 1
        size_digits = part.rstrip("bkm").lstrip("a")
        if (
            prev >= 0
            and parts[prev].isdigit()
            and len(size_digits) == 1
            and (prev == 0 or not parts[prev - 1].isdigit())
        ):
            drop.add(prev)
            prev -= 1
        if prev >= 0 and re.fullmatch(r"a\d+", parts[prev]):
            drop.add(prev)
    return drop


def strip_parameter_sizes(parts: list[str]) -> list[str]:
    drop = size_group_indices(parts)
    return [part for idx, part in enumerate(parts) if idx not in drop]


def strip_descriptors(parts: list[str]) -> list[str]:
    return [part for part in parts if part not in DESCRIPTOR_TOKENS]


def strip_short_codes(parts: list[str]) -> list[str]:
    out = []
    for idx, part in enumerate(parts):
        if idx > 0 and SHORT_CODE_RE.match(part):
            continue
        out.append(part)
    return out


def significant_parts(parts: list[str]) -> list[str]:
    out = strip_parameter_sizes(parts)
    out = strip_descriptors(out)
    out = strip_short_codes(out)
    return out


def add_variant(
    variants: list[tuple[str, str]], seen: set[str], parts: list[str], reason: str
) -> None:
    value = join_components(parts)
    if not value or value in seen:
        return
    seen.add(value)
    variants.append((value, reason))


def query_variants(query: str) -> list[tuple[str, str]]:
    parts = components(query)
    variants: list[tuple[str, str]] = []
    seen: set[str] = set()

    add_variant(variants, seen, parts, "exact")

    for end in range(len(parts) - 1, 1, -1):
        add_variant(variants, seen, parts[:end], "drop suffix")

    no_descriptor_suffix = strip_descriptor_suffix(parts)
    add_variant(variants, seen, no_descriptor_suffix, "drop descriptor suffix")
    add_variant(variants, seen, strip_parameter_sizes(parts), "drop size")
    add_variant(
        variants,
        seen,
        strip_parameter_sizes(no_descriptor_suffix),
        "drop descriptor suffix and size",
    )
    add_variant(variants, seen, strip_descriptors(parts), "drop descriptors")
    add_variant(variants, seen, strip_short_codes(parts), "drop short code")
    add_variant(variants, seen, significant_parts(parts), "significant components")

    return variants


def token_match_score(variant: str, aa_slug: str) -> tuple[int, int] | None:
    variant_norm = normalize_slug(variant)
    aa_norm = normalize_slug(aa_slug)
    if not variant_norm or not aa_norm:
        return None
    if aa_norm == variant_norm:
        return (0, 0)
    if aa_norm.startswith(f"{variant_norm}-") or variant_norm in aa_norm:
        return (1, len(aa_norm) - len(variant_norm))

    ordered_variant_parts = components(variant)
    variant_parts = set(ordered_variant_parts)
    aa_parts = set(components(aa_slug))
    if not variant_parts:
        return None

    # Avoid size-only/version-only false positives such as mellum2-12b -> gemma-3-12b.
    anchor = ordered_variant_parts[0]
    if anchor not in aa_parts and anchor not in aa_norm:
        return None

    shared = variant_parts & aa_parts
    if variant_parts <= aa_parts:
        return (2, len(aa_parts - variant_parts))
    if len(variant_parts) > 1 and len(shared) >= max(2, len(variant_parts) - 1):
        return (3, len(variant_parts - shared) + len(aa_parts - shared))
    return None


# --- confidence grading -------------------------------------------------------

EXACT = "exact"
WEAK = "weak"

# token_match_score kinds that mean "the names are the same string once
# normalized". Higher kinds allow the option to carry extra tokens, which is
# exactly how a version bump sneaks in, so they never reach EXACT.
MATCH_EXACT = 0


@dataclass(frozen=True)
class Match:
    option: str
    confidence: str
    reason: str


def grade(name: str, options: list[str]) -> list[Match]:
    """Every option that matches name at all, best first."""
    best: dict[str, tuple[tuple, int, int, str]] = {}
    for variant_index, (variant, variant_reason) in enumerate(query_variants(name)):
        for option in options:
            scored = token_match_score(variant, option)
            if scored is None:
                continue
            kind, distance = scored
            key = (variant_index, kind, distance, len(option), option)
            if option not in best or key < best[option][0]:
                best[option] = (key, variant_index, kind, variant_reason)

    matches: list[Match] = []
    for option, (_key, variant_index, kind, variant_reason) in sorted(
        best.items(), key=lambda item: item[1][0]
    ):
        exact = variant_index == 0 and kind == MATCH_EXACT
        label = MATCH_LABELS.get(kind, "match")
        matches.append(
            Match(
                option=option,
                confidence=EXACT if exact else WEAK,
                reason=f"{variant_reason} to {variant_index and 'a variant' or name}, {label}",
            )
        )
    return matches


def propose(name: str, options: list[str]) -> tuple[Match | None, list[Match]]:
    """(proposal, alternatives).

    The proposal is returned only when a single option matches by normalized
    equality; two options normalizing to the same string is ambiguous and gets
    no proposal. Everything else comes back as alternatives to suggest.
    """
    matches = grade(name, options)
    exact = [m for m in matches if m.confidence == EXACT]
    if len(exact) == 1:
        return exact[0], [m for m in matches if m.option != exact[0].option]
    return None, matches


# --- suggestions for a person -------------------------------------------------
#
# grade() is deliberately narrow: its EXACT tier is what propose.py may commit
# unseen, and test_propose.py holds it to never contradicting a mapping file.
# That narrowness costs the queue its suggestions, though. Measured against the
# mapping files, grade() offered the answer a person eventually chose not at
# all for 80 of 825 model mappings (a vendor prefix, "anthropic/claude-opus-
# 5.5"; a packaging suffix, "Ternary-Bonsai-2-27B-gguf"; "qwen-3" for "qwen3")
# and for 106 of 274 Hugging Face labels, whose answers are benchmark keys a
# label rarely spells ("AIME25" for aime_2025, "HLE (no tools)" for hle).
#
# suggest() is for the candidates a person is shown, and nothing else: it keeps
# every grade() result, in grade()'s order, and adds what those spellings find.
# Nothing it adds is ever EXACT, so nothing it adds can be proposed unseen. The
# order is
#
#   0  grade()'s exact matches
#   1  the same name once a vendor prefix or a packaging suffix is dropped, or
#      spelled without punctuation, or one of the option's own display names
#   2  grade()'s other matches, as before
#   3  the same, once a parenthetical or leading words are dropped -- "HLE (no
#      tools)", "Agentic coding SWE-bench Pro" -- and component matches on any
#      of the spellings above
#
# so what grade() already ranked first stays first unless a tier-1 spelling is
# literally the same name, which is the case the extra tiers exist for.

PACKAGING_TOKENS = {
    "it", "gguf", "hf", "mlx", "awq", "gptq", "fp8", "fp16", "bf16", "int4", "int8",
    "instruct", "chat",
}
# Words that name no benchmark or model on their own. A spelling cut down to
# nothing but these ("Bench v2" out of "OCR Bench v2") matches every option
# that says "bench", so it is not offered.
GENERIC_TOKENS = {
    "bench", "benchmark", "test", "eval", "evals", "score", "avg", "average", "mean",
    "pass", "acc", "accuracy", "pro", "plus", "verified", "hard", "full", "text", "only",
    "no", "with", "without", "wo", "tools", "tool", "the", "of", "and", "for", "gb", "vs",
}
_BRACKETS_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")


def compact(value: str) -> str:
    """A name with every separator gone: qwen-3-8 and qwen3.8 are one string."""
    return re.sub(r"[^a-z0-9]+", "", value.lower().replace("'", ""))


def _spellings(name: str, labels: bool = False) -> list[tuple[str, str, int]]:
    """(spelling, why, tier) for the ways a source may have spelled `name`."""
    out: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    def add(text: str, why: str, tier: int) -> None:
        text = text.strip(" *-_/")
        if tier > 1 and all(p in GENERIC_TOKENS or re.fullmatch(r"v?\d+", p) for p in components(text)):
            return
        key = compact(text)
        if key and key not in seen:
            seen.add(key)
            out.append((text, why, tier))

    add(name, "the same name spelled without punctuation", 1)
    tail = name.rsplit("/", 1)[-1]
    if tail != name:
        add(tail, "the same name without the vendor prefix", 1)
    for base in list(dict.fromkeys([name, tail])):
        parts = components(base)
        trimmed = list(parts)
        while trimmed and trimmed[-1] in PACKAGING_TOKENS:
            trimmed.pop()
        if trimmed != parts:
            add("-".join(trimmed), "the same name without the packaging suffix", 1)
        unbracketed = _BRACKETS_RE.sub(" ", base)
        if unbracketed != base:
            add(unbracketed, "the same name without the parenthetical", 3)
        # Leading words: a vendor spelled as a word ("Ternary-Bonsai"), or a
        # category the card puts before the benchmark ("Agentic coding ...").
        words = components(unbracketed)
        for drop in range(1, min(3, len(words) - 1) + 1):
            add("-".join(words[drop:]), f"the same name without its first {drop} word(s)", 3)
        if labels:
            # A benchmark label trails words as often as it leads with them
            # ("BrowseComp Web browsing", "HLE no tools"). Not for model names,
            # where the words after the first are what tells two models apart.
            for keep in range(1, min(3, len(words) - 1) + 1):
                add("-".join(words[:keep]), f"the first {keep} word(s) of the label", 3)
    return out


# A version tail on a benchmark label: LiveCodeBenchV6, BrowseComp 6.
_VERSION_TAIL_RE = re.compile(r"^v?\d+$")


def suggest(
    name: str, options: list[str], aliases: dict[str, list[str]] | None = None
) -> list[Match]:
    """Candidates worth showing a person for `name`, best first. See above."""
    aliases = aliases or {}
    best: dict[str, tuple[int, int, Match]] = {}
    order = 0

    def offer(option: str, tier: int, confidence: str, reason: str) -> None:
        nonlocal order
        order += 1
        if option not in best or (tier, order) < best[option][:2]:
            best[option] = (tier, order, Match(option=option, confidence=confidence, reason=reason))

    for match in grade(name, options):
        offer(match.option, 0 if match.confidence == EXACT else 2, match.confidence, match.reason)

    # Every name an option answers to, compacted, for the equality tiers.
    names: dict[str, list[tuple[str, str]]] = {}
    for option in options:
        for label in [option, *aliases.get(option, [])]:
            names.setdefault(compact(label), []).append((option, label))

    spellings = _spellings(name, labels=bool(aliases))
    for spelling, why, tier in spellings:
        for option, label in names.get(compact(spelling), []):
            said = why if label == option else f"{why}, as {label!r}"
            offer(option, tier, WEAK, said)
        if aliases:
            # Only where the options are labels too: for a model, a trailing
            # number is a different model (qwen3 and qwen3-8).
            key = compact(spelling)
            for known, pairs in names.items():
                if known and key.startswith(known) and _VERSION_TAIL_RE.match(key[len(known):]):
                    for option, label in pairs:
                        offer(option, 3, WEAK, f"{why}, plus a version, as {label!r}")
        if spelling == name:
            continue
        for match in grade(spelling, options):
            offer(match.option, 1 if match.confidence == EXACT and tier == 1 else 3, WEAK,
                  f"{why}: {match.reason}")
    if aliases:
        labels = {label: option for option in options for label in aliases.get(option, [])}
        for spelling, why, tier in spellings:
            for match in grade(spelling, list(labels)):
                offer(labels[match.option], 3, WEAK, f"{why}: {match.reason}, as {match.option!r}")

    return [m for _t, _o, m in sorted(best.values(), key=lambda item: item[:2])]
