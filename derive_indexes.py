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
    every pair of models that both carry a score is compared once. A pair whose
    values are within one step of the column's rounding grid (tie_tolerance)
    splits the comparison, and a lower-is-better column compares the other way
    round.
  * A benchmark's reliability weight from INDEXES is what those comparisons add
    up to, not what each one is worth: the weight is divided by the total and
    again by the opponents available, so a model collects exactly that
    benchmark's share of the index from it however wide the board. Weighting
    each pair by the raw weight instead would hand the vote to field size --
    measured on the coding group, Real-SWE at 1.0 supplied 1.4% of the
    likelihood and SciCode at 0.35 supplied 17.9%. Dividing by the total is
    also what keeps the weights relative, so scaling every one of them leaves
    the ranking alone.
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
    last, so a model was better off never being run on a hard benchmark. The
    one exception is a benchmark that re-ran itself: a model measured on the
    retired board only may have that score carried onto the current board's
    scale (REVISION_FALLBACKS), and only once enough models sit on both boards
    to fit the conversion. That is a measurement moved between scales rather
    than a value made up, and every such value is named in the model's
    index_coverage record, so the page says so wherever it is used.
  * Not every score in a column is the same instrument. A comparison involving
    a second-hand score -- a model card, an aggregator, a compilation, a hand
    entry -- counts SECOND_HAND_WEIGHT of one between two scores this
    repository read off the benchmark's own run or a third party's re-run.
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
    no comparisons and is left out of the total before the shares are taken, so
    it dilutes nobody.
  * A model measured on less than MIN_SCORED_FRACTION of the index's weight is
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
import time
from pathlib import Path
from typing import Any, NamedTuple

import _history
from _reference import apply_reference_flags
from _precedence import RANK_CURATED, source_rank
from _scores import score_step, stamp_score_updated
# The five indexes are the slowest thing this repository does that is not a
# network call -- about 41 seconds of a 279-second refresh, spent twice over,
# and nothing in either log said which of them it went to, because the only
# per-index line printed is a count of models.
from _timing import report

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


class IndexDef(NamedTuple):
    """One derived column: the key it fills, the page a value cites when
    llm.json declares no URL for the column (there is no leaderboard behind a
    derived score, so the source is the README section documenting the method),
    the contributing benchmarks with their reliability weights, and how far one
    of those benchmarks generalises to the rest. Weights are relative, so only
    their ratios matter; scale them all and the ranking does not move, because
    comparisons() divides them by their own total before anything reads them.

    `transfer_ratio` is how far a whole index's worth of one benchmark misses
    the rest of the index, over the variance between models -- how much of what
    a benchmark tells you is about this model rather than about this benchmark.
    It is in shares of the group, so the five are directly comparable with each
    other and with MIN_SCORED_FRACTION. It sets how hard a thinly covered model
    is pulled toward the middle (see coverage_reliability), and it is measured
    rather than chosen: ./derive_indexes.py --calibrate -w re-derives it into
    CALIBRATION_JSON, which a nightly workflow does, and INDEXES picks the
    stored value up at import. 0.0 means never calibrated and shrinks nobody.
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
            # beside terminal_bench_2_1 and terminal_bench_4_0. A superseded
            # revision measured a
            # different task set, so it compares a model against a
            # field that no longer exists; aggregating both would also count
            # the benchmark twice for whoever was re-run and once for everyone
            # else. The archived columns stay visible in the table.
            ("real_swe", 1.0),
            # 0.6: the group's narrowest head (4.6 points between the best
            # model and the fifth, against Real-SWE's 15.0) on a board the rest
            # of the group predicts well -- 0.85 Spearman with SWE-bench
            # Verified, 0.82 with FrontierCode 1.1, 0.81 with Terminal-Bench
            # 4.0 -- and 6 of its 18 values are not the maintainers' own runs.
            ("deepswe_1_1", 0.6),
            ("frontierswe_2_0", 0.9),
            # FrontierCode enters through its Extended board, the full
            # 150-task set, and not through Main, the 100 hardest of those same
            # tasks. The two are one set of runs scored twice and they rank the
            # same field almost identically (Spearman 0.99 over the 18 models on
            # both, 3 discordant pairs in 153), so aggregating both would spend
            # two slots on one measurement; Extended is the one kept because it
            # is the larger sample of the two. frontiercode_1_1 stays a column
            # in llm.json, scraped and rendered like any other -- it is simply
            # not voted on here, the way frontiercode_1_0 is not. See README,
            # "FrontierCode Extended in the coding group".
            ("frontiercode_extended_1_1", 0.9),
            # Long-horizon execution adds useful coverage, but 20 distinct
            # tasks warrant less influence than the larger frontier boards.
            ("swe_marathon_1_1", 0.65),
            # Terminal-Bench is the one family aggregated twice over. 4.0 is
            # the current release and carries the weight 2.1 held; 2.1 stays in
            # at 0.4 because it is not a superseded revision in the sense above
            # -- it is a separate series, scored on 105 models against 4.0's
            # 32, and dropping it would take the group's coverage backbone out
            # with it. The lower weight is what keeps the pair from voting
            # twice on one construct.
            ("terminal_bench_4_0", 0.85),
            ("terminal_bench_2_1", 0.4),
            # The only member that measures building rather than maintaining:
            # every other board here sets its tasks inside a repository that
            # already works, and this one hands the model a specification for a
            # web application and has a browser agent drive whatever it
            # deploys. 0.75 prices it level with the SWE Atlas tracks and a
            # rung under Terminal-Bench 4.0: the group's provenance-and-
            # coverage argument, minus a head discount.
            #
            # For it: Vals runs all 96 rows itself -- 88 in OpenHands, 8 in the
            # vendor harness the row names -- publishes a standard error per
            # cell and holds out 50 of the 100 specifications, which is the
            # provenance Real-SWE is trusted for; 35 of those rows are models
            # in this file, every one an OpenHands row, which is the widest
            # field of any frontier agentic board here -- FrontierCode Extended
            # and DeepSWE have 18, SWE-Marathon 12, FrontierSWE 8, Real-SWE 5
            # -- and it spreads them over 0 to 90.3, where the boards below
            # saturate. Against it: 5.6 points separate the best model from the
            # fifth, the same narrow head that holds DeepSWE at 0.6, and one
            # harness for nearly everyone measures a model-in-OpenHands rather
            # than a model.
            #
            # Redundancy argues neither way. Its highest Spearman against a
            # member with power is 0.908 with SWE-bench Verified over 30 models
            # and 0.882 with SciCode over 34, both of which sit at the bottom
            # of this list precisely because they are saturated, and it reads
            # 0.858 against FrontierCode Extended and 0.815 against
            # Terminal-Bench 4.0. The 0.93s against the SWE Atlas tracks are 12
            # and 9 models wide and carry no weight. Nothing here is the
            # near-copy that cut Terminal-Bench 2.1 to 0.4 (0.91 against 4.0)
            # or kept FrontierCode Main out (0.99 against Extended).
            #
            # The index is insensitive to the exact number: anywhere in
            # 0.55-1.0 the same 4 models join the ranked field and the same 5
            # -- Command A+, both Devstrals, GLM 4.7 and MiMo V2 Flash, none of
            # them scored here -- fall under the evidence bar the new weight
            # lifts. See README, "Why Vibe Code Bench enters at 0.75".
            ("vibe_code_bench_1_1", 0.75),
            ("swe_bench_pro", 0.4),
            ("livecodebench", 0.4),
            ("scicode", 0.35),
            ("swe_bench_multilingual", 0.3),
            # SWE Atlas contributes two of its three tracks, at 0.75 each.
            # Test Writing is the one left out, and the correlations pick it:
            # it ranks the field 0.955 with Codebase Q&A over the 11 models on
            # both, which is a second copy of a column already here, where
            # Refactoring sits at 0.755 -- the loosest pair in the family -- and
            # is the only track scoring a model Q&A does not (MiniMax M3). So
            # the family votes 1.5 on two tracks that disagree rather than 0.51
            # on three that do not, and Test Writing stays a column nothing
            # aggregates. See README, "Why SWE Atlas contributes two tracks".
            ("swe_atlas_qna", 0.75),
            ("swe_atlas_rf", 0.75),
            # ProgramBench rebuilt from its binary: the agent gets a
            # reference executable and its documentation, nothing else, and has
            # to produce a codebase whose behaviour matches across 200 tasks
            # from compact CLI tools up to FFmpeg, SQLite and the PHP
            # interpreter. The column stores Almost Resolved, at least 95% of a
            # task's behavioural tests passing, which is the looser of the two
            # readings the board publishes; the authors' primary metric,
            # Resolved, needs every test to pass and is close to the floor --
            # 14 of the same 22 models at zero -- so it ranks the top few and
            # calls everything below them equal. Almost separates the same
            # field over 0 to 53.5, and the column is named for the reading it
            # stores so the two are never confused.
            #
            # For it: the widest head in the group, 37.0 points between the
            # best model and the fifth against FrontierSWE 2.0's 39.6 and
            # Terminal-Bench 4.0's 20.2, on a board nothing is near saturating;
            # two sources, the benchmark's own leaderboard at rank 2 and Vals'
            # wider re-run beneath it; and a tie share of 3.9% overall and 7.6%
            # among the open-weight models, which is better than Terminal-Bench
            # 4.0 manages at 0.85.
            #
            # Against it: 22 scored models is mid-field here, and it agrees
            # with Vibe Code Bench at 0.922 over all 22 -- the highest
            # well-powered correlation between any two members of this group
            # outside the Terminal-Bench family. Both ask a model to build
            # rather than maintain, so that is one construct measured twice,
            # and the second member takes the discount the same way
            # terminal_bench_2_1 does beside 4.0. 0.5 against Vibe Code's 0.75
            # gives the pair 1.25 of 9.65, the same share the Terminal-Bench
            # pair carries, which is the most any single construct is allowed
            # here.
            #
            # The band is wide: every weight from 0.3 to 0.57 produces exactly
            # the same ranked field, because the evidence bar crosses 1.70 at
            # 0.294 -- taking Gemma 4 31B, gpt-oss-120b, MiniCPM5 2B and Qwen3
            # Coder Next, none of them scored here -- and does not reach the
            # next group until 0.572. So 0.5 is chosen on merit inside a band
            # where merit is the only thing left to choose on. See README,
            # "Why ProgramBench enters at 0.50".
            ("programbench_almost", 0.5),
            ("swe_bench_verified", 0.15),
        ],
    ),
    IndexDef(
        key="tooling_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#tooling-index",
        contributing=[
            ("tau3_bench_banking", 1.0),
            ("toolathlon", 0.9),
            ("mcp_atlas", 0.85),
            # Same swap as in the coding group, one rung down: 4.0 takes the
            # weight 2.1 held, 2.1 stays as coverage at 0.3.
            ("terminal_bench_4_0", 0.8),
            ("terminal_bench_2_1", 0.3),
            ("gdpval_aa", 0.7),
            ("itbench_aa", 0.6),
            ("bfcl_v4", 0.5),
            ("tau2_bench_telecom", 0.3),
            ("terminal_bench_hard", 0.3),
            ("ifbench", 0.2),
        ],
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
            # 0.15, the floor, and a sentinel rather than a discriminator today.
            # OSWorld 2.0 is the one long-horizon computer-use run in the table
            # -- 108 multi-hour workflows where the best official run completes
            # a fifth and open-weight models a twentieth -- so it is the member
            # with the most headroom. Only its 2026.06.24 release is aggregated
            # (2026.08.08 scores two closed models; the vendor column mixes
            # binary and partial scores), and two models this file tracks are
            # scored on it, tied at 4.6: one comparison, a tie between two
            # models MMMU Pro and OSWorld-Verified already compare. At 0.15 it moves nobody a rank,
            # and it stays inert up to 0.7. Worth re-pricing once enough tracked
            # models are scored on one release to rank -- see README, "Vision
            # Index".
            ("osworld_2_0_2026_06_24", 0.15),
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
    ),
]

# Where each index's calibrated transfer_ratio is kept. It is data rather than a
# constant in INDEXES because it is a measurement of llm.json and moves when
# llm.json does: typed into the source it went stale between hand re-runs --
# Knowledge carried 0.072 while the file itself calibrated to 0.046. The
# nightly "calibrate indexes" workflow re-measures it with
# `--calibrate --write` and commits this file together with the llm.json the
# new values produce.
CALIBRATION_JSON = Path(__file__).resolve().parent / "index-calibration.json"


def load_calibration(path: Path = CALIBRATION_JSON) -> dict[str, float]:
    """index key -> stored transfer_ratio. A missing or malformed file, or
    entry, yields nothing for that index, which leaves it uncalibrated (0.0,
    shrinking nobody) rather than failing the whole refresh."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    ratios: dict[str, float] = {}
    for key, entry in stored.items():
        ratio = entry.get("transfer_ratio") if isinstance(entry, dict) else None
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio >= 0:
            ratios[key] = float(ratio)
    return ratios


def calibrated(
    indexes: list[IndexDef], ratios: dict[str, float]
) -> list[IndexDef]:
    """`indexes` with each transfer_ratio taken from `ratios` where it has one."""
    return [
        index._replace(transfer_ratio=ratios[index.key])
        if index.key in ratios
        else index
        for index in indexes
    ]


INDEXES = calibrated(INDEXES, load_calibration())


def write_calibration(
    results: dict[str, tuple[float, int]], path: Path = CALIBRATION_JSON
) -> bool:
    """Store freshly measured ratios, keeping the previous entry for an index
    that could not be measured this time; returns whether the file changed.

    Rounded to three places, the precision the ratios were always quoted at:
    finer than that is calibration noise, and writing it would churn the file
    and llm.json every night for no change anyone could read. An entry whose
    rounded ratio holds is kept as it was, sample and date included, so
    "calibrated" is the date the ratio last moved.
    """
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    stored: dict[str, Any] = {}
    for index in INDEXES:
        if index.key in results:
            ratio, sample = results[index.key]
            old = previous.get(index.key)
            # Only a change in the ratio itself is news: the sample grows with
            # every model added, and recording that alone would commit every
            # night for a file whose effect on the ranking had not moved.
            if isinstance(old, dict) and old.get("transfer_ratio") == round(ratio, 3):
                stored[index.key] = old
                continue
            stored[index.key] = {
                "transfer_ratio": round(ratio, 3),
                "sample": sample,
                "calibrated": time.strftime("%Y-%m-%d", time.gmtime()),
            }
        elif index.key in previous:
            stored[index.key] = previous[index.key]
    if stored == previous:
        return False
    path.write_text(json.dumps(stored, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    return True


# A benchmark that has re-run itself keeps a column per revision, because the
# two are not comparable as published (see _revisions.py). For the *index* that
# leaves a hole: a model measured only on the retired board contributes nothing
# to the benchmark it was actually measured on, so it takes part in no
# comparison there at all -- which flatters a model that scored near zero.
#
# The fix is a scale conversion, not a second column. The re-run's own overlap
# -- the models published on both boards -- gives the factor that carries an
# old score onto the current board's scale; applying it lets those models join
# the current revision's population and be ranked against today's field like
# everyone else. llm.json's columns keep exactly what each board published, and
# every converted value is named in the model's index_coverage record, so the
# page says which of a model's index inputs is a conversion rather than a
# measurement.
#
# The factor is fitted from llm.json on every run, not typed in here, and it is
# only used once MIN_FALLBACK_OVERLAP models carry both columns. A constant
# fitted once goes stale the moment either board moves: DeepSWE's used to be
# 1 / 1.069, and the two models now on both boards say x1.06 and x7.6, which is
# not a revision drift at all but two models saying opposite things. A factor
# is a claim about how two scales relate, and two points cannot tell a scale
# from a coincidence -- below the bar the hole stays open, which is the same
# answer FrontierSWE gets below, and the reason is the README's: a fitted
# factor that has not earned its keep is inventing a number, which is worse
# than admitting one is missing.
#
# current revision key -> older revision key it may fall back to
REVISION_FALLBACKS: dict[str, str] = {
    "deepswe_1_1": "deepswe_1_0",
    # FrontierCode is aggregated through Extended, so the conversion targets
    # that column. It crosses a revision and a subset at once (1.0 Main onto
    # 1.1 Extended), which is one more gap than the others cross, but it is
    # fitted the same way and on the same kind of overlap.
    #
    # frontiercode_1_1 needs no entry of its own now that it is not aggregated;
    # a fallback only ever fills a hole in a column the index reads.
    "frontiercode_extended_1_1": "frontiercode_1_0",
    # FrontierSWE 2.0 deliberately has no fallback. A conversion carries a
    # score from one scale onto another, and 1.0 published no score on 2.0's
    # scale to carry: its column is a pairwise win rate over a 17-model field,
    # 2.0's a mean@5 task percentage. A factor fitted on the models on both
    # boards would be fitting the two metrics' relationship to each other,
    # not a revision's drift, and would rank a model on the shape of a field it
    # was never measured against. A model on 1.0 alone simply supports no
    # comparison on this benchmark.
}

# How many models must carry both revisions before their overlap is trusted
# to convert one scale onto the other. Three is the fewest that can disagree
# with a pair: with two, one odd model is half the evidence.
MIN_FALLBACK_OVERLAP = 3


class RevisionFactor(NamedTuple):
    """A fitted conversion: the older column, the multiplier that carries its
    scores onto the current column's scale, and how many models it was fitted
    on."""

    older: str
    factor: float
    overlap: int


def revision_factors(models: list[dict[str, Any]]) -> dict[str, RevisionFactor]:
    """current key -> RevisionFactor, for every REVISION_FALLBACKS entry whose
    overlap clears MIN_FALLBACK_OVERLAP.

    The factor is the geometric mean of current / older over the models on
    both boards, because a revision drift is a ratio: averaging the ratios
    arithmetically would let one model that doubled outweigh one that halved.
    A zero on either board says nothing about a ratio and is left out of it.
    """
    factors: dict[str, RevisionFactor] = {}
    for key, older in REVISION_FALLBACKS.items():
        logs = []
        for model in models:
            scores = model.get("scores") or {}
            current = to_number(scores.get(key))
            previous = to_number(scores.get(older))
            if current > 0 and previous > 0:
                logs.append(math.log(current / previous))
        if len(logs) >= MIN_FALLBACK_OVERLAP:
            factors[key] = RevisionFactor(
                older, math.exp(sum(logs) / len(logs)), len(logs)
            )
    return factors


# A comparison is only as good as the two numbers in it, and a column in
# llm.json is not one instrument: mmlu_pro holds Artificial Analysis' own run
# beside Vals' re-run beside a lab's model-card claim, and a pair across them
# compares harnesses as much as models. A score whose source ranks at or below
# SECOND_HAND_RANK (see _precedence: curated compilations, aggregators, model
# cards and hand entries -- a number someone else produced under their own
# prompts and settings, which this repository only read) is second-hand, and
# any comparison it takes part in counts SECOND_HAND_WEIGHT of a first-hand
# one. It still counts: dropping it would unrank most of the open field on
# benchmarks only the labs report, and a lab's claim is evidence, just weaker
# evidence than a run somebody else can reproduce.
#
# Halved rather than cut harder because the fit already shrinks a thin
# measurement toward the middle through BT_PRIOR, and a model whose whole
# record is second-hand is thin in exactly that sense after this.
SECOND_HAND_RANK = RANK_CURATED
SECOND_HAND_WEIGHT = 0.5


# Below this share of an index's total weight a model is left unranked. 0.18
# rather than a round 0.2 because model coverage clusters rather than spreading
# evenly: the coding group has 7 models measured on 18.8-19.5% of its weight and
# nothing between 16.9% and 20.8%, so 0.2 was cutting a natural block in half.
# The next cluster sits at 15.6-16.9%, and going down far enough to admit the
# 12-15% one would make 90% of the ranked field less than half measured -- see
# README, "Why the evidence bar is 18%".
MIN_SCORED_FRACTION = 0.18

# The Bradley-Terry prior: pseudo-comparisons, half won and half lost, against
# an opponent fixed at the centre of the field. It is what keeps the fit
# finite, because a model that won every comparison it ever had has no
# maximum-likelihood ability short of infinity, and llm.json holds several.
# It also pulls a thinly compared model toward the middle in proportion to how
# little evidence stands behind it.
#
# It is read against a model's comparison mass, which comparisons() normalises
# to the share of the index that model was measured on -- so at most 1, and
# 0.212 for a model on 21% of the coding group. 0.001 is therefore about a
# thousandth of a well-covered model's evidence, and the ceiling it puts on an
# unbeaten model is around log(1/0.001), near 7 log-odds.
#
# Chosen from a sweep rather than picked: the ranking keeps moving while the
# prior is heavy (16, 14, 11 and 13 places of worst-case movement stepping down
# 0.02 -> 0.01 -> 0.005 -> 0.002) and settles once it is not (4 places to
# 0.001, then 3 and 3). Below that it buys nothing and costs convergence --
# at 0.0001 the MM iteration no longer meets BT_TOLERANCE inside
# BT_MAX_ITERATIONS. 0.001 is the top of the flat stretch.
BT_PRIOR = 0.001

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

# The scale block's bisection (see _set_scale). The bound is in log-odds and is
# far outside any real field; 60 halvings of a 120-wide interval land well below
# what a float can represent, so the block is solved exactly as far as the
# arithmetic is concerned.
_SCALE_SEARCH_BOUND = 60.0
_SCALE_SEARCH_STEPS = 60

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


def tie_tolerance(doc: dict[str, Any], key: str) -> float:
    """How close two scores on one benchmark must be to split a comparison
    rather than decide it: one step of the grid the column is stored on.

    llm.json rounds every score to that grid (_scores.score_step), so two
    values one step apart may come from true scores anywhere from nearly equal
    to two steps apart, and which side of a rounding boundary each landed on is
    the whole of what separates them. Deciding a comparison on that would be
    letting the stored precision rank the models. Two steps apart is a real
    difference and is counted as one.

    What is not attempted is weighting a comparison by the size of the gap.
    The index uses the order of the scores and nothing else, which is what
    lets an Elo and a percentage sit in one group without a common unit; a
    0.3-point edge on a saturated board still counts the same as a 30-point one
    on an open board. The weights in INDEXES are where a saturated board is
    priced down.
    """
    # The factor absorbs the float error in a difference such as 88.4 - 88.3.
    return float(score_step(doc, key)) * (1 + 1e-6)


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


def index_score(
    model: dict[str, Any],
    key: str,
    factors: dict[str, RevisionFactor] | None = None,
) -> float:
    """The value this benchmark contributes for one model, NaN when it has none.

    Normally the stored score. For a benchmark with a fitted revision factor
    (`factors`, from revision_factors()), a model absent from the current
    revision falls back to its older-revision score converted onto the current
    scale, so it is compared against the current field rather than sitting the
    benchmark out. A model published on both keeps the current revision's own
    number -- the conversion only ever fills a hole.
    """
    scores = model.get("scores") or {}
    value = to_number(scores.get(key))
    if math.isfinite(value):
        return value
    fallback = (factors or {}).get(key)
    if fallback is None:
        return value
    older = to_number(scores.get(fallback.older))
    return older * fallback.factor if math.isfinite(older) else older


def is_converted(
    model: dict[str, Any], key: str, factors: dict[str, RevisionFactor] | None
) -> bool:
    """Whether index_score() answers for this model with a converted
    older-revision score rather than a number the current board published."""
    if key not in (factors or {}):
        return False
    scores = model.get("scores") or {}
    return not math.isfinite(to_number(scores.get(key))) and math.isfinite(
        index_score(model, key, factors)
    )


def is_second_hand(
    model: dict[str, Any], key: str, factors: dict[str, RevisionFactor] | None
) -> bool:
    """Whether the number index_score() uses came from a second-hand source
    (see SECOND_HAND_RANK). A converted value is judged by the source of the
    archived score it was converted from, since that is the number behind it."""
    if is_converted(model, key, factors):
        key = factors[key].older
    url = (model.get("scores_source") or {}).get(key)
    return source_rank(url) >= SECOND_HAND_RANK


def scored_on(
    models: list[dict[str, Any]],
    key: str,
    factors: dict[str, RevisionFactor] | None = None,
) -> list[tuple[str, float]]:
    """(name, value) for every model carrying a score on one benchmark, revision
    fallbacks included, in llm.json order."""
    entries = [
        (model["name"], index_score(model, key, factors))
        for model in models
        if isinstance(model.get("name"), str)
    ]
    return [(name, value) for name, value in entries if math.isfinite(value)]


class Comparisons(NamedTuple):
    """The head-to-head record one index is fitted on.

    `wins[(a, b)]` is the comparison weight a took off b, `pairs[(a, b)]` the
    weight they were compared over; both are symmetric in the sense that
    wins[(a, b)] + wins[(b, a)] == pairs[(a, b)] == pairs[(b, a)]. `weight` is
    the per-benchmark *share* actually in play, summing to 1 over the
    benchmarks that rank anybody -- benchmarks with fewer than two scored
    models produce no comparisons and are absent, which is what keeps them out
    of the coverage denominator.

    The shares are what a model's comparisons add up to: every model scored on
    a benchmark first-hand, against opponents scored first-hand, collects
    exactly `weight[key]` of comparison mass from it, so a model's total mass
    is the share of the index it was measured on, and MIN_SCORED_FRACTION,
    coverage_reliability and the fit are all reading the same quantity. A
    comparison with a second-hand score in it carries SECOND_HAND_WEIGHT of
    that, so a model measured mostly second-hand brings less evidence to the
    fit than its coverage says, and the prior leans on it accordingly.
    """

    wins: dict[tuple[str, str], float]
    pairs: dict[tuple[str, str], float]
    weight: dict[str, float]


def live_benchmarks(
    models: list[dict[str, Any]],
    index: IndexDef,
    factors: dict[str, RevisionFactor],
) -> list[tuple[str, float, list[tuple[str, float]]]]:
    """(key, share, scored entries) for every contributing benchmark that ranks
    anybody, the shares summing to 1.

    Only benchmarks two models can be compared on rank anybody, so the rest
    are left out of the total rather than diluting it.
    """
    live = [
        (key, benchmark_weight, scored_on(models, key, factors))
        for key, benchmark_weight in index.contributing
    ]
    live = [entry for entry in live if len(entry[2]) >= 2]
    declared = sum(benchmark_weight for _, benchmark_weight, _ in live)
    if declared <= 0:
        return []
    return [
        (key, benchmark_weight / declared, entries)
        for key, benchmark_weight, entries in live
    ]


def comparisons(
    models: list[dict[str, Any]],
    doc: dict[str, Any],
    index: IndexDef,
    factors: dict[str, RevisionFactor] | None = None,
) -> Comparisons:
    """Every model-versus-model comparison the contributing benchmarks support.

    One comparison per benchmark per pair of models that both carry a score on
    it, worth that benchmark's weight -- or SECOND_HAND_WEIGHT of it when
    either score is second-hand. Two values within tie_tolerance() split it,
    and a lower-is-better column is read the other way round. A model missing
    a score simply takes part in no comparison there: that is how a gap is
    handled, and the whole of how it is handled.

    `factors` defaults to the revision factors fitted from `models` itself.
    """
    if factors is None:
        factors = revision_factors(models)
    wins: dict[tuple[str, str], float] = {}
    pairs: dict[tuple[str, str], float] = {}
    weight: dict[str, float] = {}
    by_name = {
        model["name"]: model for model in models if isinstance(model.get("name"), str)
    }

    for key, share, entries in live_benchmarks(models, index, factors):
        # Two normalisations, and the index's stated contract needs both.
        #
        # By the total, so only the ratios between weights matter: doubling
        # every weight in INDEXES must leave the ranking exactly where it was,
        # which it cannot do while BT_PRIOR and transfer_ratio are fixed
        # amounts measured against them. After this a share is what those
        # constants are read against, and MIN_SCORED_FRACTION -- already a
        # share -- is on the same footing.
        #
        # By the field size, so a benchmark's weight is the influence it
        # actually has. Every pair on a benchmark carrying the same weight
        # sounds right and is not: a model on a 118-model board collects 117
        # comparisons where one on a 5-model board collects 4, so an unweighted
        # pair makes width, not reliability, decide the fit. Measured on the
        # coding group before this was fixed, Real-SWE at 1.0 supplied 1.4% of
        # the likelihood mass and SciCode at 0.35 supplied 17.9% -- the weights
        # were being set by the boards' sizes rather than by INDEXES. Dividing
        # by the opponents available hands each model exactly `share` of
        # evidence from a benchmark it ran, whatever the size of the field it
        # ran against, which is what the weight is documented to mean and what
        # coverage_reliability already assumes.
        weight[key] = share
        per_pair = share / (len(entries) - 1)
        lower = is_lower_better(doc, key)
        tolerance = tie_tolerance(doc, key)
        trust = [
            SECOND_HAND_WEIGHT if is_second_hand(by_name[name], key, factors) else 1.0
            for name, _ in entries
        ]
        for a in range(len(entries)):
            name_a, value_a = entries[a]
            for b in range(a + 1, len(entries)):
                name_b, value_b = entries[b]
                if abs(value_a - value_b) <= tolerance:
                    won = 0.5
                elif lower:
                    won = 1.0 if value_a < value_b else 0.0
                else:
                    won = 1.0 if value_a > value_b else 0.0
                mass = per_pair * min(trust[a], trust[b])
                forward = (name_a, name_b)
                backward = (name_b, name_a)
                wins[forward] = wins.get(forward, 0.0) + mass * won
                wins[backward] = wins.get(backward, 0.0) + mass * (1.0 - won)
                pairs[forward] = pairs.get(forward, 0.0) + mass
                pairs[backward] = pairs.get(backward, 0.0) + mass

    return Comparisons(wins, pairs, weight)


def bradley_terry(record: Comparisons) -> dict[str, float]:
    """Model name -> fitted ability, as a log-odds with zero at the middle of
    the field.

    Bradley-Terry says the odds of a beating b are exp(ability_a - ability_b),
    and the fit maximises the likelihood of the comparisons actually observed,
    penalised by BT_PRIOR's pseudo-comparisons against an opponent pinned at
    strength 1. That penalised objective is strictly concave in the abilities,
    so it has one maximum and the only question is getting there.

    Two blocks, alternated, each of which can only raise the objective:

      * the shape, by the standard MM (minorise-maximise) step, which is
        monotone and needs no derivatives or matrix algebra --

            strength_i <- (wins_i + prior)
                          / (sum_j pairs_ij / (strength_i + strength_j)
                             + 2 * prior / (strength_i + 1))

      * the overall level, solved outright by _set_scale(), because the
        comparisons say nothing about it and the prior says everything.

    At a fixed point of the pair the MM step can only be multiplying every
    strength by one constant, and the scale block forces that constant to 1, so
    every model's gradient is zero and the result is the maximum rather than
    merely somewhere the iteration stopped moving. test_index_math.py checks
    that gradient directly rather than trusting this paragraph.

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
    strength = _set_scale(strength)

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
        updated = _set_scale(updated)
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


def _set_scale(strength: list[float]) -> list[float]:
    """The same abilities shifted along the one direction the comparisons say
    nothing about, to the level the prior most prefers.

    Bradley-Terry itself is scale-free -- only differences are identified -- so
    the overall level is fixed by the prior alone, and fixing it is a
    one-parameter problem that can be solved outright rather than iterated
    toward. Writing the shift as c, the prior contributes

        sum_i [ log(c*s_i / (c*s_i + 1)) + log(1 / (c*s_i + 1)) ]

    whose derivative vanishes exactly when the shifted abilities average a
    win probability of one half against the prior's opponent:

        sum_i  c*s_i / (c*s_i + 1)  ==  n / 2

    The left side rises monotonically with c, so a bisection finds it. Two
    things follow, and the fit depends on both.

    It makes the sweep a block coordinate ascent whose fixed point is the real
    optimum. Each MM sweep raises the penalised likelihood and so does this, so
    the pair still only ever climbs; and at a fixed point the sweep can only be
    rescaling everything by one constant, which this step's own stationarity
    then forces to be 1 -- leaving the gradient zero for every model, which is
    what a solution means. Simply renormalising to a geometric mean of 1
    instead looks like the same housekeeping and is not: the prior's opponent
    is pinned at 1, so sliding every model past it changes the penalty, and the
    iteration settles where the sweep scales everything by some c != 1 and no
    model's gradient is zero. Against llm.json that left the coding group at
    c = 1.000503 and gradients of 4.5e-4 rather than 1e-10, and moved models
    across each other.

    And it puts zero where the column's own scale says the middle is. The
    condition above says a model at ability zero wins half its comparisons
    against this field on average -- which is expected_win_rate()'s definition
    of the midpoint, to the digit. So coverage_reliability can shrink a thin
    measurement toward zero and be shrinking it toward the middle of the field,
    with no separate centring step to disagree with.
    """
    target = len(strength) / 2.0
    low, high = -_SCALE_SEARCH_BOUND, _SCALE_SEARCH_BOUND
    for _ in range(_SCALE_SEARCH_STEPS):
        middle = (low + high) / 2
        shift = math.exp(middle)
        total = sum(value * shift / (value * shift + 1.0) for value in strength)
        if total < target:
            low = middle
        else:
            high = middle
    shift = math.exp((low + high) / 2)
    return [value * shift for value in strength]


def coverage_reliability(weights: list[float], transfer_ratio: float) -> float:
    """How much of a model's fitted ability to believe, given only these
    benchmarks measured it. 1.0 keeps it whole, 0.5 halves its distance from
    the middle of the field.

    A benchmark is treated as one draw from the construct the index is named
    after: it reads a model's ability plus something specific to that benchmark
    (a harness, a task mix, a scoring rule). A weight says how much that draw
    can be trusted -- that is what "reliability weight" means, and it is why a
    benchmark hands each model that ran it exactly its share of comparison mass
    -- so a weight is a precision, and the shares a model was measured on
    simply add up:

        variance of the measured ability = transfer_ratio / sum(share)

    in units of the between-model variance. Kelley's formula then says how much
    of an observed deviation from the middle is real, which reduces to the
    covered share against itself plus a constant:

        reliability = sum(share) / (sum(share) + transfer_ratio)

    `weights` holds the shares comparisons() computed, summing to 1 for a fully
    measured model, so what buys reliability is the share of the index
    measured: the same quantity MIN_SCORED_FRACTION is a bar on, and the same
    quantity the fit weighs a model's comparisons by. One heavy benchmark can
    be worth more than three light ones, exactly as the weights claim.

    This is what stops a model being ranked on the benchmarks it happens to
    have run. It is not a guess at a missing score -- nothing is imputed, and a
    model's own numbers are never touched. It only says how far a fitted
    ability may be carried from the middle of the field before it is claiming
    more than the measurements support.

    How much that bites is the data's decision, not a preference, and the
    calibrated ratios differ by two orders of magnitude across the five groups
    -- see the table in INDEXES and the README section they link to.
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
    models: list[dict[str, Any]],
    doc: dict[str, Any],
    index: IndexDef,
    factors: dict[str, RevisionFactor] | None = None,
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
    if factors is None:
        factors = revision_factors(models)
    record = comparisons(models, doc, index, factors)
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
            if math.isfinite(index_score(model, key, factors))
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


def index_coverage(
    models: list[dict[str, Any]],
    index: IndexDef,
    factors: dict[str, RevisionFactor] | None = None,
) -> dict[str, dict[str, Any]]:
    """Model name -> what one index value rests on, for every model measured
    on at least one of the benchmarks that rank anybody.

      * "measured": the ones the model carries a score on, in INDEXES order;
      * "of": how many there are;
      * "share": the share of the index's weight those cover -- the quantity
        MIN_SCORED_FRACTION is a bar on, rounded to three places;
      * "converted": the benchmarks answered by a revision conversion rather
        than by a number the current board published (omitted when none);
      * "second_hand": the benchmarks answered by a second-hand score, whose
        comparisons count SECOND_HAND_WEIGHT (omitted when none).

    This is what the page shows beside an index value, so a reader can see
    that a Vision score rests on MMMU Pro alone, or that a Coding score leans
    on a converted archive, without re-deriving it.
    """
    if factors is None:
        factors = revision_factors(models)
    live = live_benchmarks(models, index, factors)
    result: dict[str, dict[str, Any]] = {}
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue
        keys = [
            (key, share)
            for key, share, _ in live
            if math.isfinite(index_score(model, key, factors))
        ]
        if not keys:
            continue
        record: dict[str, Any] = {
            "measured": [key for key, _ in keys],
            "of": len(live),
            "share": round(sum(share for _, share in keys), 3),
        }
        converted = [key for key, _ in keys if is_converted(model, key, factors)]
        if converted:
            record["converted"] = converted
        second_hand = [
            key for key, _ in keys if is_second_hand(model, key, factors)
        ]
        if second_hand:
            record["second_hand"] = second_hand
        result[name] = record
    return result


def apparent_ability(
    name: str, key: str, models: list[dict[str, Any]], field: dict[str, float],
    record: Comparisons, factors: dict[str, RevisionFactor] | None = None,
) -> float | None:
    """The ability that explains one model's record on one benchmark alone,
    given everyone else's ability. None when it has no opponents there, and
    None when it won or lost every comparison and no finite ability fits.

    A model's expected win rate rises monotonically with its ability, so this
    is a bisection rather than an optimisation.

    `record` must hold the comparisons from `key` and nothing else, which is
    the whole point of the function and easy to get wrong: a record covering
    the rest of the index too would answer with how the model did *overall*
    against the opponents that happen to have run this benchmark, because
    wins[(name, other)] pools every benchmark the pair share. Calibration would
    then be scoring the rest of the index against a target it had already been
    folded into, and would read the benchmarks as far more mutually predictive
    than they are.
    """
    others = [
        other for other, _ in scored_on(models, key, factors)
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
    factors = revision_factors(models)
    full = comparisons(models, doc, index, factors)
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
        record = comparisons(models, doc, rest, factors)
        without = bradley_terry(record)
        rest_weight = sum(record.weight.values())
        # The held-out benchmark on its own, so apparent_ability sees only what
        # it said. Passing the whole index here would feed `without` back into
        # the target it is being scored against -- see apparent_ability.
        alone = comparisons(
            models,
            doc,
            IndexDef(
                key=index.key,
                fallback_source_url=index.fallback_source_url,
                contributing=[
                    (key, weight)
                    for key, weight in index.contributing
                    if key == held_out
                ],
            ),
            factors,
        )
        for model in models:
            name = model.get("name")
            if not isinstance(name, str) or name not in without:
                continue
            covered = sum(
                weight
                for key, weight in record.weight.items()
                if math.isfinite(index_score(model, key, factors))
            )
            if covered < CALIBRATION_MIN_COVERAGE * rest_weight:
                continue
            fitted = apparent_ability(
                name, held_out, models, without, alone, factors
            )
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


def scored_count(
    model: dict[str, Any],
    index: IndexDef,
    factors: dict[str, RevisionFactor] | None = None,
) -> int:
    """How many contributing benchmarks this model actually has a score on.

    Counts a revision fallback in `factors`, because the model was measured on
    that benchmark -- on its retired board -- and the index ranks it
    accordingly.
    """
    return sum(
        1
        for key, _ in index.contributing
        if math.isfinite(index_score(model, key, factors))
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


def apply_coverage(
    doc: dict[str, Any],
    per_index: dict[str, tuple[dict[str, int | None], dict[str, dict[str, Any]]]],
) -> int:
    """Write each model's index_coverage record -- index key -> what that value
    rests on (see index_coverage()) -- for every index the model is ranked on,
    in INDEXES order, and return how many models' records changed.

    `per_index` maps an index key to (its computed values, its coverage). An
    unranked model reports no coverage for that index, the same way it reports
    no source or date, and a model ranked on none carries no record at all.
    """
    changed = 0
    for model in doc.get("models") or []:
        name = model.get("name")
        if not isinstance(name, str):
            continue
        record = {
            index.key: per_index[index.key][1][name]
            for index in INDEXES
            if index.key in per_index
            and per_index[index.key][0].get(name) is not None
            and name in per_index[index.key][1]
        }
        if record != model.get("index_coverage"):
            changed += 1
        if record:
            model["index_coverage"] = record
        else:
            model.pop("index_coverage", None)
    return changed


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
    factors = revision_factors(models)
    per_index: dict[str, tuple[dict[str, int | None], dict[str, dict[str, Any]]]] = {}
    # Applied back to front because put_first prepends: the last index applied
    # ends up leading each model's maps, so the keys sit in INDEXES order.
    for index in reversed(INDEXES):
        started = time.monotonic()
        values = compute_index(models, doc, index, factors)
        per_index[index.key] = (values, index_coverage(models, index, factors))
        changes.extend(
            (index.key, name, old, new)
            for name, old, new in apply_index(doc, index, values)
        )
        report(index.key, started)
    apply_coverage(doc, per_index)
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


def report_stored(
    doc: dict[str, Any], by_name: dict[str, dict[str, Any]], top: int
) -> int:
    """Print the ranking llm.json already holds, computing nothing.

    update.py refreshes every index in memory before it writes, so on the
    update-all path the columns in the file are this run's, and recomputing
    them to look at them cost a second full fit -- 41 seconds of a 439-second
    refresh, whose usual verdict is "up to date. Nothing to do." This prints
    the same tables from the stored values instead.

    It is a report, not a check: a file whose columns are stale reports its
    stale columns without complaint. `./derive_indexes.py -w` is still the
    thing that recomputes and repairs, and still what to run on a file edited
    by hand -- the one cheap check made here is that the columns exist at all,
    since a missing key means no writer ever got to them and the tables below
    would quietly be a report about nothing.
    """
    # Where apply_index put them: alongside the measured scores, so the derived
    # columns read like any other column to every consumer.
    stored = {
        name: model["scores"] if isinstance(model.get("scores"), dict) else {}
        for name, model in by_name.items()
    }
    # The factors read only the stored scores, so this costs nothing and keeps
    # the report line for line what a refit prints.
    print_conversions(revision_factors(list(by_name.values())))
    # Reversed like main()'s own loop, so the report reads in the order every
    # log of this script has ever printed it in.
    for index in reversed(INDEXES):
        values = {name: scores.get(index.key) for name, scores in stored.items()}
        missing = [name for name, scores in stored.items() if index.key not in scores]
        ranked = sum(1 for value in values.values() if isinstance(value, int))
        print(
            f"{index.key}: {len(values)} model(s): {ranked} ranked, "
            f"{len(values) - ranked} unranked "
            f"(< {MIN_SCORED_FRACTION:.0%} of the weight of "
            f"{len(index.contributing)} benchmarks)"
        )
        if missing:
            print(
                f"Warning: {len(missing)} model(s) carry no {index.key} at all "
                f"(first: {missing[0]}); run ./derive_indexes.py -w",
                file=sys.stderr,
            )

        if top:
            best = sorted(
                ((value, name) for name, value in values.items() if isinstance(value, int)),
                reverse=True,
            )[:top]
            if best:
                print(f"\nTop {len(best)} by {index.key}:")
                for rank, (value, name) in enumerate(best, start=1):
                    coverage = (by_name[name].get("index_coverage") or {}).get(
                        index.key
                    )
                    print(f"  {rank:2d}. {fmt(value):>9s}  {name:40s} {describe(coverage)}")
        print()

    print("Reported from the stored values; nothing recomputed.")
    return 0


def print_conversions(factors: dict[str, RevisionFactor]) -> None:
    """Which revision fallbacks are in use this run, and on what overlap."""
    for key, older in REVISION_FALLBACKS.items():
        fitted = factors.get(key)
        if fitted is None:
            print(
                f"{key}: no conversion from {older} (fewer than "
                f"{MIN_FALLBACK_OVERLAP} models on both boards)"
            )
        else:
            print(
                f"{key}: {older} converted x{fitted.factor:.3f}, fitted on "
                f"{fitted.overlap} models on both boards"
            )
    print()


def describe(coverage: dict[str, Any] | None) -> str:
    """One model's index_coverage record as the CLI prints it beside a value:
    "3/5 measured (62%)", plus how many of those are converted or second-hand."""
    if not isinstance(coverage, dict):
        return "coverage not recorded"
    text = (
        f"{len(coverage.get('measured') or [])}/{coverage.get('of', 0)} measured "
        f"({coverage.get('share', 0):.0%})"
    )
    notes = []
    if coverage.get("converted"):
        notes.append(f"{len(coverage['converted'])} converted")
    if coverage.get("second_hand"):
        notes.append(f"{len(coverage['second_hand'])} second-hand")
    return f"{text}, {', '.join(notes)}" if notes else text


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
        "--report-only",
        action="store_true",
        help="Print the ranking already stored in the file instead of fitting "
        "it again, for a caller that has just refreshed it -- update-all after "
        "update.py. Computes nothing and writes nothing.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Re-measure each index's transfer_ratio by leave-one-benchmark-out "
        "and print it; with --write, store it in index-calibration.json. Takes "
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

    if args.report_only:
        return report_stored(doc, by_name, args.top)

    if args.calibrate:
        print(
            "Leave-one-benchmark-out calibration"
            + (f", written to {CALIBRATION_JSON.name}" if args.write else " (dry-run)")
            + ".\n"
        )
        results: dict[str, tuple[float, int]] = {}
        for index in INDEXES:
            measured = calibrate(models, doc, index)
            if measured is None:
                print(f"  {index.key:16} too few benchmarks to hold one out")
                continue
            ratio, sample = measured
            results[index.key] = measured
            print(
                f"  {index.key:16} transfer_ratio={ratio:.3f} "
                f"(was {index.transfer_ratio:.3f}, n={sample})"
            )
        if not args.write:
            print("\ndry-run only, pass --write to store the ratios")
        elif write_calibration(results):
            print(
                f"\nWrote {CALIBRATION_JSON}; run ./derive_indexes.py -w to "
                "refit llm.json with the new ratios"
            )
        else:
            print(f"\n{CALIBRATION_JSON.name} is up to date")
        return 0

    all_changes: list[tuple[str, str, int | None, int | None]] = []
    restamped = 0
    started_all = time.monotonic()
    factors = revision_factors(models)
    print_conversions(factors)
    per_index: dict[str, tuple[dict[str, int | None], dict[str, dict[str, Any]]]] = {}
    for index in reversed(INDEXES):
        started = time.monotonic()
        values = compute_index(models, doc, index, factors)
        coverage = index_coverage(models, index, factors)
        per_index[index.key] = (values, coverage)
        report(index.key, started)

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
                    print(
                        f"  {rank:2d}. {fmt(value):>9s}  {name:40s} "
                        f"{describe(coverage.get(name))}"
                    )
        print()

    report("all indexes", started_all)
    recovered = apply_coverage(doc, per_index)

    if not all_changes and not restamped and not recovered:
        print("The derived indexes are up to date. Nothing to do.")
        return 0

    if all_changes:
        print(f"{len(all_changes)} value(s) change:")
        for key, name, old, new in sorted(all_changes, key=lambda c: (c[0], c[1])):
            print(f"  {key:14s} {name:40s} {fmt(old):>9s} -> {fmt(new):>9s}")
    if restamped:
        print(f"\n{restamped} source URL(s) restamped")
    if recovered:
        print(f"\n{recovered} model(s) with a changed index_coverage record")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    started_write = time.monotonic()
    _history.sync(doc)
    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    report(f"_history.sync + write {path.name}", started_write)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
