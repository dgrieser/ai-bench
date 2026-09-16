#!/usr/bin/env python3
"""Compute the derived index columns in llm.json from the benchmarks they
aggregate: the Coding index from the coding benchmarks, the Tooling index from
the agentic tool-use benchmarks, the Knowledge index from the knowledge and
reasoning benchmarks, the Vision index from the multimodal ones, the Trust
index from the honesty, grounding and instruction-compliance ones.

Unlike every other column, a derived index is not scraped: it is computed from
scores already in llm.json, so it has to be recomputed whenever any of them
changes (update-all does this after the scrapers have run).

The math is a weighted Bradley-Terry model fitted on head-to-head comparisons,
because the thing being estimated is a single latent ability per model and the
evidence for it is ragged: no two models are measured on the same set of
benchmarks.

  * Every benchmark contributes comparisons, not scores. For each benchmark,
    every pair of models that both carry a score is compared once, and the
    comparison is worth that benchmark's reliability weight from INDEXES
    below. A pair whose values are equal to within nearly_equal() splits the
    comparison. A lower-is-better column compares the other way round.
  * That is the whole use a raw score is put to, and it is what makes the
    columns commensurable. "Model A beat model B on SciCode" needs no common
    unit; a percentile does. The older method averaged percentiles taken from
    twelve different fields, which silently equated them: last of the five
    models on Real-SWE -- a field of near-frontier systems -- scored the same
    0.0 as last of LiveCodeBench's 118, and the two are nothing alike.
  * A missing score produces no comparison at all. Nothing is imputed, no
    stand-in value is invented, and a benchmark a model never ran cannot move
    it in either direction. This is the property the median fill could not
    have: filling a gap at the median beat actually being measured and placing
    last, so a model was better off never being run on a hard benchmark.
  * Comparisons a model never had are still answered, by the fit rather than
    by a guess. Two models measured on disjoint benchmarks are linked through
    the opponents they share -- the same anchoring that lets two candidates
    who sat different exam papers be placed on one scale -- so the estimate
    uses every model in the field as evidence, not only the ones a model was
    measured beside.
  * The fit carries a small symmetric prior (BT_PRIOR) pulling each model
    toward the middle of the field. It keeps a model that won or lost
    everything finite, keeps a sparsely compared model from being flung to an
    extreme on two comparisons, and shrinks in proportion to how little
    evidence there is. Unlike the median fill it is symmetric: it never helps
    a weak model more than a strong one.
  * A benchmark nobody can be ranked on (fewer than two scored models) yields
    no comparisons and is left out of the total weight so it dilutes nobody.
  * A model measured on less than MIN_SCORED_FRACTION of the total weight is
    left unranked (null) rather than scored. The bar is about how much of the
    construct was covered, not about statistical confidence: a model measured
    on one benchmark against a hundred opponents is precisely estimated on
    that benchmark and still says little about coding.

The fitted ability is a log-odds and means nothing on its own, so what the
column reports is what that ability predicts: the share of head-to-head
comparisons the model would be expected to win against the ranked field, drawn
one opponent at a time. That is a real quantity with a scale of its own -- 0.5
is "as good as the middle of the field", 0.9 is "wins nine of ten" -- and it
is the same quantity in every index, so Coding and Tooling can be read side by
side. It is multiplied by SCALE and rounded, so a model scores something like
89136 rather than 89.1, which keeps neighbouring models apart: two of them can
sit millionths of a win rate apart and still need distinct integers. Models do
still share one, either because the fit gave them the same ability to machine
precision or because they are closer than SCALE can separate -- a difference of
one part in a hundred thousand of a win rate is noise, and reporting an order
there would be inventing precision rather than measuring it.

Because the win rate is taken against the current field, every model's value
can move when a model or a score is added. That also means null is a real
result here, so -- unlike the scrapers, which never overwrite a value with null
because a source can drop out for a day -- this script does clear a value (and
its date) when a model no longer qualifies.

The field being ranked is every model in the file, the closed reference rows
(see _reference.py) included. They are hidden from the table by default but
they are not hidden from the arithmetic: an index is "how good is this model
among the ones we have measured", and a reader comparing an open model with the
frontier is asking about one field, not two. So a reference row moves the
ranking exactly as any other model does, and an open model's value can move
when one is added -- which is already true whenever any model is added, and is
the paragraph above.

No leaderboard publishes these columns, so a value is attributed to the page
that documents how it is derived: the first URL the column declares in llm.json
(a section of this repository's README).

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

import _history
from _reference import apply_reference_flags
from _scores import stamp_score_updated

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


class IndexDef(NamedTuple):
    """One derived column: the key it fills, the page a value cites when
    llm.json declares no URL for the column (there is no leaderboard behind a
    derived score, so the source is the README section documenting the method),
    the contributing benchmarks with their reliability weights, and how far one
    of those benchmarks generalises to the rest. Weights are relative, so only
    their ratios matter; scale them all and the ranking does not move.

    `transfer_ratio` is how far a unit weight's worth of one benchmark misses
    the rest of the index, over the variance between models -- how much of what
    a benchmark tells you is about this model rather than about this benchmark.
    It is in units of weight, so it is read against the group's total: 0.053
    against the coding group's 7.3 is mild, 0.693 against Trust's 2.35 is not.
    It sets how hard a thinly covered model is shrunk toward the middle (see
    coverage_reliability), and it is measured rather than chosen: run
    ./derive_indexes.py --calibrate to re-derive it. 0.0 means never calibrated
    and shrinks nobody.
    """

    key: str
    fallback_source_url: str
    contributing: list[tuple[str, float]]
    transfer_ratio: float = 0.0


# The derived columns this script fills, in the order their score keys are
# kept in each model's maps. Label, description and icon live in llm.json like
# every other benchmark's.
INDEXES: list[IndexDef] = [
    IndexDef(
        key="coding_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#coding-index",
        contributing=[
            # DeepSWE, FrontierSWE, FrontierCode and SWE-Marathon each keep a
            # column per published revision, and only the current one is
            # aggregated -- the same treatment terminal_bench_2_0 already gets
            # beside terminal_bench_2_1. A superseded revision measured a
            # different task set, so it compares a model against a
            # field that no longer exists; aggregating both would also count
            # the benchmark twice for whoever was re-run and once for everyone
            # else. The archived columns stay visible in the table.
            ("real_swe", 1.0),
            ("deepswe_1_1", 0.9),
            ("frontierswe_2_0", 0.9),
            ("frontiercode_1_1", 0.9),
            ("swe_marathon_1_1", 0.9),
            ("terminal_bench_2_1", 0.85),
            ("swe_bench_pro", 0.4),
            ("livecodebench", 0.4),
            ("scicode", 0.35),
            ("swe_bench_multilingual", 0.3),
            # SWE Atlas contributes its Codebase Q&A track only: every model
            # scored on Refactoring or Test Writing is also scored on Q&A, so
            # the other two tracks add no coverage, correlate 0.94 with each
            # other and 0.77-0.89 with Q&A, and their weight only raised the
            # MIN_SCORED_FRACTION bar. One track at 0.25 instead of three at
            # 0.17 -- see README, "Why SWE Atlas contributes one track".
            ("swe_atlas_qna", 0.25),
            ("swe_bench_verified", 0.15),
        ],
        # Coding benchmarks largely agree: hold one out and the rest predict
        # it to within a twentieth of a unit weight's worth of the spread
        # between models. Against a group weight of 7.3 that is mild -- a model
        # on 18% of the weight keeps 96% of its distance from the middle, one
        # measured throughout 99% -- which is the data's verdict, not a
        # preference: a model that places top-decile on three coding benchmarks
        # really is unlikely to be mid-field on the rest.
        transfer_ratio=0.053,
    ),
    IndexDef(
        key="tooling_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#tooling-index",
        contributing=[
            ("tau3_bench_banking", 1.0),
            ("toolathlon", 0.9),
            ("mcp_atlas", 0.85),
            ("terminal_bench_2_1", 0.8),
            ("gdpval_aa", 0.7),
            ("itbench_aa", 0.6),
            ("bfcl_v4", 0.5),
            ("tau2_bench_telecom", 0.3),
            ("terminal_bench_hard", 0.3),
            ("ifbench", 0.2),
        ],
        # Tool-use benchmarks agree more closely still, over a wider group
        # weight, so this shrinks almost nobody.
        transfer_ratio=0.022,
    ),
    IndexDef(
        key="knowledge_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#knowledge-index",
        contributing=[
            ("aa_omniscience", 1.0),
            ("hle", 0.9),
            ("critpt", 0.6),
            ("mmlu_pro", 0.5),
            ("aime_2025", 0.4),
            ("gpqa_diamond", 0.3),
            # Deliberately not aggregated, though they are knowledge columns in
            # llm.json: aa_omniscience_hallucination (the same AA run as
            # aa_omniscience, whose index already prices confident errors and
            # abstentions in, and it adds 6 models of coverage), aime_2026
            # (the same exam one year on, scored on 28 models, 19 of them at or
            # above 90), mmmu_pro (0.96 with GPQA and only multimodal models can
            # be scored at all), aa_lcr and browsecomp (comprehension of a
            # supplied document and retrieval with a browser, not what the model
            # knows), aa_omniscience_accuracy (the correct share of the same run
            # aa_omniscience already scores, the Index simply before the
            # confident errors are netted off), aa_intelligence_index (a
            # composite over benchmarks in this table, coding and tool use among
            # them) -- see README, "Knowledge index". The first and the fourth
            # of those are the Trust index's two anchors: what they measure is
            # honesty, which is why they are worth aggregating there and not
            # here.
        ],
        # A knowledge benchmark predicts its siblings well -- the group's
        # members correlate 0.61 to 0.83 with each other, and it shows.
        transfer_ratio=0.029,
    ),
    IndexDef(
        key="vision_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#vision-index",
        contributing=[
            ("mmmu_pro", 1.0),
            ("osworld_verified", 0.7),
            ("mathvista_mini", 0.35),
            ("zerobench", 0.35),
            ("charxiv_reasoning", 0.35),
            # Deliberately not aggregated, though it is the widest multimodal
            # column in llm.json: gdpval_aa scores visual *output* (documents,
            # slides, diagrams) produced in AA's agentic harness, so what it
            # separates is shell agency -- 0.95 with the Tooling index that
            # already prices it at 0.7, 0.97 with AA Intelligence. And 38
            # models carry it as their only score here: at 0.7 its share is
            # 0.226, over the evidence bar, so it would rank all 38 on no
            # visual evidence at all -- see README, "Why GDPval-AA is left
            # out".
        ],
        # The lowest of the five: the vision members track each other almost
        # exactly -- MMMU Pro correlates 0.96 with MathVista-mini and 0.93 with
        # ZeroBench -- which is the same redundancy their weights are already
        # discounted for. The consequence is worth stating plainly: this column
        # barely penalises a model measured on MMMU Pro alone, because on this
        # evidence MMMU Pro predicts the rest.
        transfer_ratio=0.010,
    ),
    IndexDef(
        key="trust_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#trust-index",
        contributing=[
            ("aa_omniscience_hallucination", 1.0),
            # The brake on the anchor, and the reason the anchor can carry 1.0.
            # A hallucination rate is incorrect / (incorrect + partial + not
            # attempted), so a model that abstains on everything scores
            # perfectly: without the accuracy of the same run beside it this
            # column would rank a 3B model that answers nothing at the top --
            # see README, "Why the anchor cannot stand alone".
            ("aa_omniscience_accuracy", 0.6),
            ("aa_lcr", 0.45),
            ("ifbench", 0.3),
            # Deliberately not aggregated, though both are trust-adjacent
            # columns in llm.json: aa_omniscience is a function of the two
            # members above (accuracy netted against confident error) over the
            # same AA run, so it would spend a third slot on one run, and it is
            # already the 1.0 anchor of the Knowledge index; browsecomp has no
            # first-party run at all -- all 43 of its values are third-party
            # self-reports, which is disqualifying provenance for a column
            # whose whole subject is trustworthiness -- see README, "What the
            # Trust index leaves out".
        ],
        # Thirteen times the coding group's, seventy times Vision's, and the
        # reason this parameter is per-index rather than global. A
        # hallucination rate, an accuracy, a long-context recall and an
        # instruction-following score are nearly separate constructs, and the
        # models sit close together on all of them (the fitted spread here is
        # an eighth of the coding group's), so one of these benchmarks says
        # little about the next. Against a group weight of 2.35 that bites
        # hard: a model carrying the 1.0 anchor alone keeps 59% of its
        # distance from the middle, one measured throughout 77%.
        transfer_ratio=0.693,
    ),
]

# A benchmark that has re-run itself keeps a column per revision, because the
# two are not comparable as published (see _revisions.py). For the *index* that
# leaves a hole: a model measured only on the retired board contributes nothing
# to the benchmark it was actually measured on, and is imputed at the median
# instead -- which flatters a model that scored near zero there.
#
# The fix is a scale conversion, not a second column. The re-run's own overlap
# -- the open-weight models published on both boards -- gives the factor that
# carries an old score onto the current board's scale; applying it lets those
# models join the current revision's population and be ranked against today's
# field like everyone else. The converted value is used *here only*: llm.json's
# columns keep exactly what each board published, so nothing in the table ever
# shows a number its leaderboard did not.
#
# current revision key -> (older revision key, multiplier onto the current scale)
REVISION_FALLBACKS: dict[str, tuple[str, float]] = {
    # DeepSWE 1.1 reads lower than 1.0 for the same model.
    "deepswe_1_1": ("deepswe_1_0", 1 / 1.069),
    # FrontierCode 1.1 reads higher: the mean of the two open-weight models
    # published on both boards (GLM 5.2 19.2 -> 24.5, Kimi K2.7 22.0 -> 30.06).
    "frontiercode_1_1": ("frontiercode_1_0", 1.32),
    # FrontierSWE 2.0 deliberately has no fallback. A conversion carries a
    # score from one scale onto another, and 1.0 published no score on 2.0's
    # scale to carry: its column is a pairwise win rate over a 17-model field,
    # 2.0's a mean@5 task percentage. A factor fitted on the three models on
    # both boards would be fitting the two metrics' relationship to each other,
    # not a revision's drift, and would rank a model on the shape of a field it
    # was never measured against. A model on 1.0 alone is imputed instead.
}


# Below this share of an index's total weight a model is left unranked. 0.18
# rather than a round 0.2 because model coverage clusters rather than spreading
# evenly: the coding group has 11 models measured on 19% of its weight and none
# between 19% and 20%, so 0.2 was cutting a natural block of two-benchmark
# models in half. The next cluster sits at 13-14%, and admitting it would make
# 90% of the ranked field less than half measured -- see README, "Why the
# evidence bar is 18%".
MIN_SCORED_FRACTION = 0.18

# The Bradley-Terry prior: pseudo-comparisons, half won and half lost, against
# an opponent fixed at the centre of the field. It is what keeps the fit
# finite, because a model that won every comparison it ever had has no
# maximum-likelihood ability short of infinity, and llm.json holds several.
# It also shrinks a thinly compared model toward the middle in proportion to
# how little evidence stands behind it -- 0.3 against a model carrying
# hundreds of units of comparison weight is nothing, against a model carrying
# two it is most of the story.
#
# 0.3 rather than a round 1.0 because the ranking has to be the data's and not
# the prior's: at 0.3 the ordering is indistinguishable from an unregularised
# fit (Spearman 1.00000 against 0.1, no model moving a single rank), and it
# still holds at 1.0 (0.99997) and 3.0 (0.99974). The value is chosen from the
# bottom of that plateau rather than the top, so the prior does the two jobs
# above and no more.
BT_PRIOR = 0.3

# Convergence of the MM iteration, on the largest change in log-ability across
# one sweep. 1e-10 is far tighter than the output needs: every index in
# llm.json rounds to exactly the same integers at 1e-10 as at 1e-13, and the
# looser bound saves roughly a third of the run time.
BT_TOLERANCE = 1e-10

# A ceiling so a pathological field cannot hang update-all. The real fits
# converge in a few hundred to ~13,000 sweeps; this is well clear of that, and
# hitting it warns rather than failing, because a slightly under-converged
# ranking is still a ranking.
BT_MAX_ITERATIONS = 100_000

# Calibration only (--calibrate). A model is asked about a held-out benchmark
# only once the benchmarks left behind cover this share of their weight: the
# question is whether a benchmark agrees with an estimate worth disagreeing
# with. 0.30 keeps a few hundred comparisons per index, which is enough for a
# variance and strict enough to keep two-benchmark models out of it.
CALIBRATION_MIN_COVERAGE = 0.30

# The bisection's range, in log-odds. Wide enough that no real model reaches
# it: touching it means a clean sweep, which no finite ability explains.
CALIBRATION_ABILITY_BOUND = 25.0

# Index points per full percentile: a model that tops every contributing
# benchmark scores SCALE, the median model half of it. Sized for headroom, and
# the headroom is real: the densest column is Knowledge, whose 133 ranked models
# still leave its closest pair 1 point apart with no two values colliding, where
# a 0-100 scale would have rounded a good part of its mid-field into ties.
SCALE = 100_000


def to_number(value: Any) -> float:
    """Numeric value of a score, or NaN for anything unusable (None, a bool, a
    non-numeric string)."""
    if isinstance(value, bool) or value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else math.nan
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def nearly_equal(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def source_url(doc: dict[str, Any], index: IndexDef) -> str:
    """The page every derived value is attributed to: the column's first URL in
    llm.json, so the link is spelled in one place and the benchmark card, the
    score tooltip and the Sources panel all point at the same page."""
    urls = ((doc.get("benchmarks") or {}).get(index.key) or {}).get("urls")
    if isinstance(urls, list):
        for url in urls:
            if isinstance(url, str) and url.strip():
                return url.strip()
    return index.fallback_source_url


def is_lower_better(doc: dict[str, Any], key: str) -> bool:
    benchmark = (doc.get("benchmarks") or {}).get(key) or {}
    return benchmark.get("lower_is_better") is True


def index_score(model: dict[str, Any], key: str) -> float:
    """The value this benchmark contributes for one model, NaN when it has none.

    Normally the stored score. For a benchmark with a REVISION_FALLBACKS entry,
    a model absent from the current revision falls back to its older-revision
    score converted onto the current scale, so it is ranked against the current
    field rather than imputed at the median. A model published on both keeps
    the current revision's own number -- the conversion only ever fills a hole.
    """
    scores = model.get("scores") or {}
    value = to_number(scores.get(key))
    if math.isfinite(value):
        return value
    fallback = REVISION_FALLBACKS.get(key)
    if fallback is None:
        return value
    older_key, factor = fallback
    older = to_number(scores.get(older_key))
    return older * factor if math.isfinite(older) else older


def scored_on(
    models: list[dict[str, Any]], key: str
) -> list[tuple[str, float]]:
    """(name, value) for every model carrying a score on one benchmark, revision
    fallbacks included, in llm.json order."""
    entries = [
        (model["name"], index_score(model, key))
        for model in models
        if isinstance(model.get("name"), str)
    ]
    return [(name, value) for name, value in entries if math.isfinite(value)]


class Comparisons(NamedTuple):
    """The head-to-head record one index is fitted on.

    `wins[(a, b)]` is the comparison weight a took off b, `pairs[(a, b)]` the
    weight they were compared over; both are symmetric in the sense that
    wins[(a, b)] + wins[(b, a)] == pairs[(a, b)] == pairs[(b, a)]. `weight` is
    the per-benchmark weight actually in play -- benchmarks with fewer than two
    scored models produce no comparisons and are absent, which is what keeps
    them out of the coverage denominator.
    """

    wins: dict[tuple[str, str], float]
    pairs: dict[tuple[str, str], float]
    weight: dict[str, float]


def comparisons(
    models: list[dict[str, Any]], doc: dict[str, Any], index: IndexDef
) -> Comparisons:
    """Every model-versus-model comparison the contributing benchmarks support.

    One comparison per benchmark per pair of models that both carry a score on
    it, worth that benchmark's weight. Two values equal to within nearly_equal()
    split it, and a lower-is-better column is read the other way round. A model
    missing a score simply takes part in no comparison there: that is how a gap
    is handled, and the whole of how it is handled.
    """
    wins: dict[tuple[str, str], float] = {}
    pairs: dict[tuple[str, str], float] = {}
    weight: dict[str, float] = {}

    for key, benchmark_weight in index.contributing:
        entries = scored_on(models, key)
        if len(entries) < 2:
            # Nobody can be compared here, so the benchmark ranks nobody and is
            # left out of the total weight rather than diluting it.
            continue
        weight[key] = benchmark_weight
        lower = is_lower_better(doc, key)
        for a in range(len(entries)):
            name_a, value_a = entries[a]
            for b in range(a + 1, len(entries)):
                name_b, value_b = entries[b]
                if nearly_equal(value_a, value_b):
                    share = 0.5
                elif lower:
                    share = 1.0 if value_a < value_b else 0.0
                else:
                    share = 1.0 if value_a > value_b else 0.0
                forward = (name_a, name_b)
                backward = (name_b, name_a)
                wins[forward] = wins.get(forward, 0.0) + benchmark_weight * share
                wins[backward] = (
                    wins.get(backward, 0.0) + benchmark_weight * (1.0 - share)
                )
                pairs[forward] = pairs.get(forward, 0.0) + benchmark_weight
                pairs[backward] = pairs.get(backward, 0.0) + benchmark_weight

    return Comparisons(wins, pairs, weight)


def bradley_terry(record: Comparisons) -> dict[str, float]:
    """Model name -> fitted ability, as a log-odds centred on the field.

    Bradley-Terry says the odds of a beating b are exp(ability_a - ability_b),
    and the fit is the abilities that best explain the comparisons actually
    observed. It is solved by the standard MM (minorise-maximise) iteration,
    which is monotone -- every sweep raises the likelihood -- and needs no
    derivatives or matrix algebra.

    The sweep is the fixed point of

        strength_i <- (wins_i + prior) / (sum_j pairs_ij / (strength_i + strength_j)
                                          + 2 * prior / (strength_i + 1))

    with the abilities renormalised to a geometric mean of 1 each time, so the
    prior's notional opponent sits at the centre of the field and the scale is
    pinned (only differences are identified, so some anchor is needed).

    Models are visited in sorted order and the iteration starts from each
    model's own weighted win rate, so the result is deterministic and does not
    depend on llm.json's row order.
    """
    opponents: dict[str, list[str]] = {}
    for a, b in record.pairs:
        opponents.setdefault(a, []).append(b)
    if not opponents:
        return {}

    names = sorted(opponents)
    position = {name: i for i, name in enumerate(names)}
    # Flat, index-addressed copies: the inner loop runs tens of millions of
    # times over the larger indexes and dict lookups dominate it otherwise.
    neighbours = [[position[b] for b in sorted(opponents[a])] for a in names]
    pair_weight = [
        [record.pairs[(a, b)] for b in sorted(opponents[a])] for a in names
    ]
    won = [sum(record.wins.get((a, b), 0.0) for b in opponents[a]) for a in names]
    played = [sum(weights) for weights in pair_weight]

    # Warm start from the smoothed win rate. It costs nothing and lands close
    # enough to cut the sweep count substantially on the wider indexes.
    strength = []
    for i in range(len(names)):
        rate = (won[i] + BT_PRIOR) / (played[i] + 2 * BT_PRIOR)
        rate = min(max(rate, 1e-4), 1.0 - 1e-4)
        strength.append(rate / (1.0 - rate))
    strength = _recentre(strength)

    for sweep in range(BT_MAX_ITERATIONS):
        updated = [0.0] * len(names)
        for i in range(len(names)):
            own = strength[i]
            denominator = 2 * BT_PRIOR / (own + 1.0)
            neighbours_i = neighbours[i]
            weights_i = pair_weight[i]
            for t in range(len(neighbours_i)):
                denominator += weights_i[t] / (own + strength[neighbours_i[t]])
            updated[i] = (won[i] + BT_PRIOR) / denominator
        updated = _recentre(updated)
        shift = max(
            abs(math.log(updated[i] / strength[i])) for i in range(len(names))
        )
        strength = updated
        if shift < BT_TOLERANCE:
            break
    else:
        print(
            f"Warning: the Bradley-Terry fit did not settle within "
            f"{BT_MAX_ITERATIONS} sweeps (last shift {shift:.2e}); the ranking "
            "is reported anyway",
            file=sys.stderr,
        )

    return {name: math.log(strength[i]) for i, name in enumerate(names)}


def _recentre(strength: list[float]) -> list[float]:
    """The same abilities with a geometric mean of 1. Only differences are
    identified, so this pins the scale and puts the prior's notional opponent
    at the centre of the field."""
    centre = math.exp(sum(math.log(value) for value in strength) / len(strength))
    return [value / centre for value in strength]


def coverage_reliability(weights: list[float], transfer_ratio: float) -> float:
    """How much of a model's fitted ability to believe, given only these
    benchmarks measured it. 1.0 keeps it whole, 0.5 halves its distance from
    the middle of the field.

    A benchmark is treated as one draw from the construct the index is named
    after: it reads a model's ability plus something specific to that benchmark
    (a harness, a task mix, a scoring rule). A weight says how much that draw
    can be trusted -- that is what "reliability weight" means, and it is why the
    fit already counts a comparison by its weight -- so a weight is a precision,
    and the weights a model was measured on simply add up:

        variance of the measured ability = transfer_ratio / sum(w)

    in units of the between-model variance. Kelley's formula then says how much
    of an observed deviation from the middle is real, which reduces to the
    covered weight against itself plus a constant:

        reliability = sum(w) / (sum(w) + transfer_ratio)

    So the thing that buys reliability is total weight measured, the same
    quantity MIN_SCORED_FRACTION is a bar on -- one heavy benchmark can be
    worth more than three light ones, exactly as the weights claim.

    This is what stops a model being ranked on the benchmarks it happens to
    have run. It is not a guess at a missing score -- nothing is imputed, and a
    model's own numbers are never touched. It only says how far a fitted
    ability may be carried from the middle of the field before it is claiming
    more than the measurements support.

    How much that bites is the data's decision, not a preference: at the coding
    group's calibrated ratio a model on three of the twelve benchmarks keeps
    97% of its distance, because coding benchmarks agree with each other. At
    the Trust group's it keeps 72%, because they do not.
    """
    total = sum(weights)
    if total <= 0 or transfer_ratio <= 0:
        return 1.0
    return total / (total + transfer_ratio)


def expected_win_rate(ability: float, field: list[float]) -> float:
    """The share of head-to-head comparisons a model of this ability would win
    against `field`, one opponent drawn at a time.

    This is what the column reports. It is bounded, it is monotone in ability,
    and it means the same thing in every index, which an ability in log-odds
    does not: 0.5 is the middle of the field and 0.9 is nine wins in ten.
    Averaging over the whole field rather than against a single median opponent
    is what keeps the top of the table legible -- a model that beats the weak
    models with probability ~1 is still separated from its actual rivals.
    """
    if not field:
        # One ranked model and no field to be ranked against. There is no
        # comparison to report, so it reports the midpoint rather than a win
        # rate it never earned.
        return 0.5
    return sum(
        1.0 / (1.0 + math.exp(-(ability - other))) for other in field
    ) / len(field)


def compute_index(
    models: list[dict[str, Any]], doc: dict[str, Any], index: IndexDef
) -> dict[str, int | None]:
    """Model name -> index value in points (0..SCALE), or None when unranked.

    Three steps: collect the head-to-head record, fit one ability per model to
    it, and report what that ability predicts against the field the column
    displays.

    The ability is fitted on every comparison there is, including comparisons
    with models that will not themselves be ranked -- a model too thinly
    measured to carry a value of its own is still a perfectly good opponent,
    and dropping it would throw away the links that place everyone else. The
    win rate is then taken against the ranked field only, because that is the
    field a reader sees, and it is what puts the median of each column near
    half of SCALE.
    """
    record = comparisons(models, doc, index)
    total_weight = sum(record.weight.values())

    # Coverage first: it decides who is ranked and how far their ability is
    # believed, and it reads only the scores, so nothing here depends on the
    # fit.
    measured: dict[str, list[float]] = {}
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue
        measured[name] = [
            weight
            for key, weight in record.weight.items()
            if math.isfinite(index_score(model, key))
        ]

    min_scored_weight = MIN_SCORED_FRACTION * total_weight
    ability = bradley_terry(record)

    # bradley_terry centres the abilities on the field, so the middle this
    # shrinks toward is 0 and the reliability is a plain multiplier.
    believed = {
        name: ability[name]
        * coverage_reliability(measured.get(name, []), index.transfer_ratio)
        for name in ability
    }

    ranked = sorted(
        name
        for name, weights in measured.items()
        if total_weight > 0
        and sum(weights) > 0
        and sum(weights) >= min_scored_weight
        and name in believed
    )
    field = [believed[name] for name in ranked]
    ranked_set = set(ranked)

    result: dict[str, int | None] = {}
    for name in measured:
        if name not in ranked_set:
            result[name] = None
            continue
        # A model is not one of its own opponents.
        others = [value for other, value in zip(ranked, field) if other != name]
        result[name] = round(expected_win_rate(believed[name], others) * SCALE)
    return result


def apparent_ability(
    name: str, key: str, models: list[dict[str, Any]], field: dict[str, float],
    record: Comparisons,
) -> float | None:
    """The ability that explains one model's record on one benchmark alone,
    given everyone else's ability. None when it has no opponents there, and
    None when it won or lost every comparison and no finite ability fits.

    A model's expected win rate rises monotonically with its ability, so this
    is a bisection rather than an optimisation.
    """
    others = [
        other for other, _ in scored_on(models, key)
        if other != name and other in field
    ]
    if not others:
        return None
    won = sum(record.wins.get((name, other), 0.0) for other in others)
    played = sum(record.pairs.get((name, other), 0.0) for other in others)
    if played <= 0:
        return None
    target = won / played
    abilities = [field[other] for other in others]

    low, high = -CALIBRATION_ABILITY_BOUND, CALIBRATION_ABILITY_BOUND
    for _ in range(80):
        middle = (low + high) / 2
        rate = sum(
            1.0 / (1.0 + math.exp(-(middle - other))) for other in abilities
        ) / len(abilities)
        if rate < target:
            low = middle
        else:
            high = middle
    fitted = (low + high) / 2
    # Pinned against a bound means a clean sweep in one direction, which no
    # finite ability explains; it carries no information about how far off the
    # rest of the index was.
    if abs(fitted) >= CALIBRATION_ABILITY_BOUND - 1.0:
        return None
    return fitted


def calibrate(
    models: list[dict[str, Any]], doc: dict[str, Any], index: IndexDef
) -> tuple[float, int] | None:
    """Measure `index.transfer_ratio` -- how much a benchmark disagrees with the
    rest of its index, against how much models disagree with each other.

    Leave-one-benchmark-out: fit the abilities without one benchmark, then ask
    what that benchmark alone would have said about each model. The gap between
    the two is what could not have been predicted from the rest of the index,
    which is exactly what a model with gaps is exposed to. Held out rather than
    measured in place because the fit has already minimised the in-place
    residual -- on llm.json's coding group that reads 0.311 against a true
    1.322, a fourfold understatement, and it would have shrunk nobody.

    Only models already measured on a decent share of the remaining benchmarks
    are asked, because the question is whether a benchmark agrees with a
    trustworthy estimate, not whether two thin estimates agree.

    Returns (ratio, sample size), or None when the index is too small to hold a
    benchmark out.
    """
    full = comparisons(models, doc, index)
    keys = list(full.weight)
    if len(keys) < 3:
        return None

    ability = bradley_terry(full)
    if len(ability) < 2:
        return None
    centre = sum(ability.values()) / len(ability)
    between_models = sum(
        (value - centre) ** 2 for value in ability.values()
    ) / len(ability)
    if between_models <= 0:
        return None

    # (weight of the held-out benchmark, how far it missed). The weight is
    # carried because a benchmark is trusted in proportion to it: under
    # variance = transfer_ratio / weight, a heavy benchmark is expected to miss
    # by less, so each gap is scaled back to what a unit-weight benchmark would
    # have missed by before they are pooled.
    gaps: list[tuple[float, float]] = []
    for held_out in keys:
        rest = IndexDef(
            key=index.key,
            fallback_source_url=index.fallback_source_url,
            contributing=[
                (key, weight)
                for key, weight in index.contributing
                if key != held_out
            ],
        )
        record = comparisons(models, doc, rest)
        without = bradley_terry(record)
        rest_weight = sum(record.weight.values())
        for model in models:
            name = model.get("name")
            if not isinstance(name, str) or name not in without:
                continue
            covered = sum(
                weight
                for key, weight in record.weight.items()
                if math.isfinite(index_score(model, key))
            )
            if covered < CALIBRATION_MIN_COVERAGE * rest_weight:
                continue
            fitted = apparent_ability(name, held_out, models, without, full)
            if fitted is None:
                continue
            gaps.append((full.weight[held_out], fitted - without[name]))

    if len(gaps) < 2:
        return None
    mean = sum(gap for _, gap in gaps) / len(gaps)
    # E[(gap)^2] = transfer_ratio / weight, so weight * gap^2 estimates the
    # unit-weight variance directly and the estimates pool by averaging.
    between_benchmarks = sum(
        weight * (gap - mean) ** 2 for weight, gap in gaps
    ) / len(gaps)
    return between_benchmarks / between_models, len(gaps)


def scored_count(model: dict[str, Any], index: IndexDef) -> int:
    """How many contributing benchmarks this model actually has a score on.

    Counts a revision fallback, because the model was measured on that
    benchmark -- on its retired board -- and the index ranks it accordingly.
    """
    return sum(
        1
        for key, _ in index.contributing
        if math.isfinite(index_score(model, key))
    )


def put_first(mapping: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Same mapping with `key` set and moved to the front, so the score keys
    stay in the benchmark order llm.json declares (the derived indexes are the
    leading columns)."""
    rest = {k: v for k, v in mapping.items() if k != key}
    return {key: value, **rest}


def validate(doc: dict[str, Any]) -> list[str]:
    problems = []
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return ['"benchmarks" is missing or not an object']
    index_keys = {index.key for index in INDEXES}
    for index in INDEXES:
        if index.key not in benchmarks:
            problems.append(
                f'benchmark "{index.key}" is not declared in "benchmarks"; '
                "add its label, description and icon there first"
            )
        missing = [key for key, _ in index.contributing if key not in benchmarks]
        if missing:
            problems.append(
                f'benchmarks contributing to "{index.key}" absent from '
                '"benchmarks": ' + ", ".join(missing)
            )
        derived = index_keys.intersection(key for key, _ in index.contributing)
        if derived:
            problems.append(
                f'"{index.key}" cannot aggregate a derived column: '
                + ", ".join(sorted(derived))
            )
    return problems


def apply_index(
    doc: dict[str, Any], index: IndexDef, values: dict[str, int | None]
) -> list[tuple[str, int | None, int | None]]:
    """Write one computed index into the models of `doc`, in memory only, and
    return the (model, old, new) triples that moved.

    Nothing reaches disk here, so a caller can decide whether to save (main()
    saves only with -w) and a rewritten score plus its refreshed indexes land
    in one write instead of two.
    """
    url = source_url(doc, index)
    changes: list[tuple[str, int | None, int | None]] = []
    for model in doc.get("models") or []:
        name = model.get("name")
        if not isinstance(name, str) or name not in values:
            continue

        scores = model.get("scores")
        if not isinstance(scores, dict):
            scores = {}
        old = scores.get(index.key)
        new = values[name]
        model["scores"] = put_first(scores, index.key, new)

        updated = model.get("scores_updated")
        if not isinstance(updated, dict):
            updated = {}
        model["scores_updated"] = put_first(
            updated, index.key, updated.get(index.key)
        )

        # The index is computed, not read off a leaderboard, so it cites the
        # page that documents how it is computed rather than nothing at all --
        # a reader who clicks the value gets the method. An unranked model
        # reports no source, the same way it reports no date.
        sources = model.get("scores_source")
        if not isinstance(sources, dict):
            sources = {}
        model["scores_source"] = put_first(
            sources, index.key, url if new is not None else None
        )

        if old != new:
            changes.append((name, old, new))
            if new is None:
                # No value, no date: an unranked model reports neither.
                model["scores_updated"][index.key] = None
            else:
                stamp_score_updated(model, index.key)
    return changes


def refresh(doc: dict[str, Any]) -> list[tuple[str, str, int | None, int | None]]:
    """Recompute the derived columns in `doc` in memory; returns the changes as
    (index key, model, old, new) tuples.

    For the tools that write llm.json themselves -- edit.py -- so a hand-edited
    score cannot leave a stale index behind. Raises ValueError when llm.json is
    shaped wrong, which is a configuration error rather than something to paper
    over.
    """
    problems = validate(doc)
    if problems:
        raise ValueError("; ".join(problems))
    models = doc.get("models")
    if not isinstance(models, list):
        raise ValueError('"models" is missing or not a list')
    # The reference rows decide which field each index is computed over, so the
    # flag is brought in step with reference-models.json before anything is
    # ranked -- a slug added to that list takes effect on the next refresh.
    apply_reference_flags(doc)
    changes: list[tuple[str, str, int | None, int | None]] = []
    # Applied back to front because put_first prepends: the last index applied
    # ends up leading each model's maps, so the keys sit in INDEXES order.
    for index in reversed(INDEXES):
        changes.extend(
            (index.key, name, old, new)
            for name, old, new in apply_index(
                doc, index, compute_index(models, doc, index)
            )
        )
    return changes


def refresh_and_report(
    doc: dict[str, Any],
) -> list[tuple[str, str, int | None, int | None]]:
    """refresh() for the tools that write llm.json for their own reasons, with
    the reporting they all want: a line per index naming how many models moved,
    and a warning instead of an exception when llm.json is shaped wrong.

    Every writer of "scores" or "models" has to call this before saving --
    update.py after a fetch, edit.py after a hand edit, prune.py after dropping
    a model -- because an index is a function of the whole table: a score that
    changes, or a model that leaves, re-ranks everyone else. Not reporting a
    problem loudly here is deliberate: the caller's own write is what the user
    asked for, and ./derive_indexes.py can repair the columns afterwards.
    """
    try:
        changes = refresh(doc)
    except ValueError as exc:
        print(
            f"Warning: could not recompute the derived indexes ({exc}); "
            "run ./derive_indexes.py once llm.json is fixed",
            file=sys.stderr,
        )
        return []
    for index in INDEXES:
        moved = sum(1 for key, *_ in changes if key == index.key)
        if moved:
            print(f"Recomputed {index.key} for {moved} model(s)")
    return changes


def fmt(value: int | None) -> str:
    # Grouped only in this script's output; llm.json keeps a plain integer.
    return "—" if value is None else f"{value:,}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=15,
        help="How many top-ranked models to print per index (default: 15, 0 for none).",
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Re-measure each index's transfer_ratio by leave-one-benchmark-out "
        "and print the values to paste into INDEXES. Writes nothing, and takes "
        "a few minutes: it refits every index once per contributing benchmark.",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    doc = json.loads(path.read_text(encoding="utf-8"))

    problems = validate(doc)
    if problems:
        for problem in problems:
            print(f"Error: {problem}", file=sys.stderr)
        return 1

    models = doc.get("models")
    if not isinstance(models, list):
        print(f'Error: "models" in {path} is missing or not a list', file=sys.stderr)
        return 1

    by_name = {
        model["name"]: model for model in models if isinstance(model.get("name"), str)
    }

    # Same step refresh() takes, for the same reason: which rows are reference
    # rows decides the field each index is ranked over, so the flag is brought
    # in step with reference-models.json before anything is computed.
    for name, marked in apply_reference_flags(doc):
        print(f"{name}: {'now' if marked else 'no longer'} a reference model")

    if args.calibrate:
        print(
            "Leave-one-benchmark-out calibration. Paste a changed ratio into "
            "INDEXES; nothing is written here.\n"
        )
        for index in INDEXES:
            measured = calibrate(models, doc, index)
            if measured is None:
                print(f"  {index.key:16} too few benchmarks to hold one out")
                continue
            ratio, sample = measured
            print(
                f"  {index.key:16} transfer_ratio={ratio:.3f} "
                f"(was {index.transfer_ratio:.3f}, n={sample})"
            )
        return 0

    all_changes: list[tuple[str, str, int | None, int | None]] = []
    restamped = 0
    for index in reversed(INDEXES):
        values = compute_index(models, doc, index)

        ranked = sum(1 for value in values.values() if value is not None)
        print(
            f"{index.key}: {len(values)} model(s): {ranked} ranked, "
            f"{len(values) - ranked} unranked "
            f"(< {MIN_SCORED_FRACTION:.0%} of the weight of "
            f"{len(index.contributing)} benchmarks)"
        )

        # Counted before the write, because apply_index leaves nothing to
        # compare against afterwards. A run whose values all hold still can
        # carry a source restamp -- the column's URL changed in llm.json, or
        # the value predates the source being recorded at all -- and that is
        # worth writing.
        url = source_url(doc, index)
        restamped += sum(
            1
            for name, value in values.items()
            if (by_name[name].get("scores_source") or {}).get(index.key)
            != (url if value is not None else None)
        )

        # In memory only; the file is written further down, and only with -w.
        changes = apply_index(doc, index, values)
        all_changes.extend((index.key, name, old, new) for name, old, new in changes)

        if args.top:
            best = sorted(
                ((value, name) for name, value in values.items() if value is not None),
                reverse=True,
            )[: args.top]
            if best:
                print(f"\nTop {len(best)} by {index.key}:")
                for rank, (value, name) in enumerate(best, start=1):
                    measured = scored_count(by_name[name], index)
                    print(
                        f"  {rank:2d}. {fmt(value):>9s}  {name:40s} "
                        f"{measured}/{len(index.contributing)} measured"
                    )
        print()

    if not all_changes and not restamped:
        print("The derived indexes are up to date. Nothing to do.")
        return 0

    if all_changes:
        print(f"{len(all_changes)} value(s) change:")
        for key, name, old, new in sorted(all_changes, key=lambda c: (c[0], c[1])):
            print(f"  {key:14s} {name:40s} {fmt(old):>9s} -> {fmt(new):>9s}")
    if restamped:
        print(f"\n{restamped} source URL(s) restamped")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    _history.sync(doc)
    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
