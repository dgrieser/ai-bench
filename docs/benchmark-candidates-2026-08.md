# Benchmark candidate evaluation — August 2026

Assessment of 19 candidate benchmarks against three criteria: **coverage** on the 122
models in `llm.json`, **scrapeability**, and **trustworthiness**. Coding candidates also
get a proposed weight for the `coding` sort group.

All coverage figures were measured, not estimated from documentation. Method and caveats
are in [Appendix A](#appendix-a--how-coverage-was-measured). Snapshot: 2026-08-01.

> **Revision 2** — redone against the live Artificial Analysis API (v2) rather than
> page-scraping alone. Six conclusions changed; see
> [§8 Corrections](#8-corrections-from-revision-1). The AA index audit in
> [§0](#0-are-we-using-the-current-aa-indices) is new, and it recommends **removing**
> `aa_coding_index` from the coding sort group.

---

## 0. Are we using the current AA indices?

Short answer: **the Intelligence Index is current and correct. The Coding Index is current
too — but it is defined as the average of two benchmarks already in the coding group, so it
should be dropped from that group as a double-count. The Agentic Index we don't use at all,
and shouldn't.**

The API is *not* returning stale values. Spot-checked 12 models three ways — API,
live model page, `llm.json` — and they agree to rounding:

```
slug                  API intel  PAGE intel  json intel |  API code  PAGE code  json code
glm-5-2                   51.10       51.09       51.10 |     68.80      68.76      68.80
kimi-k3                   57.10       57.11       57.10 |     76.20      76.24      76.20
deepseek-v4-pro           44.30       44.27       44.30 |     59.40      59.36      59.40
qwen3-5-397b-a17b         33.70       33.68       33.70 |     48.20      48.21      48.20
```

Only one value drifts anywhere in the dataset: `qwen3-5-0-8b` has `aa_coding_index=15` in
`llm.json` against `0` in the API.

### `aa_intelligence_index` → current, v4.1 ✅

**Intelligence Index v4.1**, released **2026-06-15**. All 115 of our values are stamped
`2026-06` or later, so there is **no pre/post-v4.1 mixing** in that column. Composition:

| category | weight | evaluations | in `llm.json` as its own column? |
|---|---|---|---|
| Agents | 34% | GDPval-AA v2 (20%), **τ³-Banking (14%)** | ❌ neither |
| Coding | 24% | Terminal-Bench v2.1 (16%), SciCode (8%) | ✅ both |
| General | 18% | AA-LCR (6%), AA-Omniscience (12%) | ❌ neither |
| Sci. reasoning | 24% | HLE (12%), GPQA Diamond (6%), CritPt (6%) | ✅ HLE, GPQA-D · ❌ CritPt |

We already carry **42%** of the index as separate columns *and* the composite on top. The
missing 58% is exactly `GDPval-AA v2` (20) + `τ³-Banking` (14) + `AA-Omniscience` (12) +
`AA-LCR` (6) + `CritPt` (6) — which is, independently, the recommendation list in §1. Worth
noting the implication: **once those five are added, `aa_intelligence_index` is fully
redundant** and should be weighted near zero in any composite, or dropped.

What v4.1 changed on 2026-06-15, verbatim from AA:

- Terminal-Bench Hard → **Terminal-Bench 2.1**
- τ²-Bench Telecom → **τ³-Bench Banking** — "move to newer, more robust task sets with
  harder, more realistic agentic scenarios"
- GDPval-AA → **GDPval-AA v2** — "re-baselines Elo to human performance at 1000, introduces
  a rotating panel of frontier-model judges, and raises the turn limit from 100 to 250"
- **Removed: IFBench** — "The benchmark no longer distinguishes frontier models
  sufficiently"

### `aa_coding_index` → **drop it from the coding group. It is a duplicate, not a benchmark** ⚠️

Not a staleness problem — a redundancy problem. AA is **actively maintaining** it: 92% of
models added in 2026-H2 carry it (75% in 2026-H1), including `kimi-k3` and `inkling-small`.
An earlier draft called it "legacy, being wound down"; that was wrong.

The actual defect is its definition. Per AA's coding-capability page, the Coding Index is
**an equal-weighted average of Terminal-Bench v2.1 and SciCode** — both already in
`sortGroups[coding]`. Fitting it over the 60 of our models that have all three:

```
aa_coding_index = 1.131 * mean(terminal_bench_2_1, scicode) - 4.86
R² = 0.9926        mean |residual| = 1.3 points        max = 5.0
```

It is a linear function of two group members. Keeping it at weight 0.2 therefore does not
add a signal, it silently reweights two existing ones:

| | nominal | effective | change |
|---|---|---|---|
| `terminal_bench_2_1` | 0.85 | **0.95** | +12% |
| `scicode` | 0.35 | **0.45** | **+29%** |

SciCode is the narrowest benchmark in the group and was deliberately parked at 0.35. It
takes the largest silent boost. That inverts what the weight ladder is trying to express.

**Information content of its 75 values:**

| | n | |
|---|---|---|
| both components present | **60** | pure duplication |
| one component missing | 15 | *looks* unique — but **11 of the 15 are pre-v4.1 withdrawn values**, whose index was computed from Terminal-Bench **Hard**, not v2.1. Obsolete definition. |
| — of which current | 4 | `minicpm-v4-6-1-3b`, `gemma-3-12b`, `ministral-3-3b`, `qwen3-5-0-8b`; AA has no TB2.1 for any of them either, so no backfill exists |
| only coding signal for a model | **0** | |

The 11 orphans (AA no longer serves them; 9 carry pre-v4.1 dates):

```
glm-4-7-flash        25.9  (2026-02-13)   sarvam-30b            7.9  (2026-03-27)
minimax-m2-5         37.4  (2026-02-13)   sarvam-105b           9.8  (2026-03-27)
qwen3-coder-30b      19.4  (2026-02-13)   step-3-5-flash       34.6  (2026-04-14)
qwen3-5-27b          34.9  (2026-03-03)   lfm2-5-8b-a1b         5.6  (2026-06-19)
qwen3-5-35b-a3b      30.3  (2026-03-03)   lfm2-5-1-2b-thinking  1.4  (2026-06-19)
longcat-flash-lite   16.5  (2026-03-11)
```

**Cost of dropping it, measured.** Three models leave the coding ranking:
`mistral-medium-3-5`, `nvidia-nemotron-3-ultra-550b-a55b`, `trinity-large-thinking`. Each
holds exactly `terminal_bench_2_1` + `scicode` + `swe_bench_verified` = **1.35** scored
weight against a **1.502** bar; with the duplicate column they cleared 1.542 by 0.008. They
were passing the evidence threshold on *duplicated* evidence, which is the exact failure
`minScoredFraction` exists to prevent. If you want them shown, lower `minScoredFraction` —
do not buy it with a duplicate column.

Overall rank impact of removal: mean **2.1** places, **28 of 69** models move ≥3, top-15
order shifts from position 7 down. That is the double-count unwinding, not information lost.

**Recommendation.** Remove it from `sortGroups[coding]`. Whether to keep the display column
is a softer call — it adds nothing beside two columns already shown, and 11 of 75 values are
from a retired definition; if you keep it for familiarity, null those 11.

**The same trap is queued for `aa_intelligence_index`**, which is a weighted average of 9
evaluations, 4 of them already displayed (42%). Harmless today because it is in no sort
group — but if the five recommended components are added it becomes 100% redundant. It must
never share a group with them.

### AA's *other* coding metric, the agent-level one — also do not add it

AA maintains two coding metrics in parallel. The model-level Coding Index above
(TB2.1 + SciCode), and the agent-level **Coding Agent Index v1.3** (July 2026):
`DeepSWE + Terminal-Bench v2 + SWE-Atlas-QnA`, **equal weights**, 321 tasks × 3 attempts.
Both are live; both are averages of benchmarks you already run.

**All three components are already in the repo's coding group** at weights 1.0, 0.85/0.35
and 0.17. Adding it would be a 100% double-count of benchmarks we run individually, at
lower resolution. It is also reported **per harness+model pair** (the same model appears
under Cursor, Claude Code, OpenCode…), which does not fit the one-row-per-model schema.
Skip it.

### Agentic Index → not used, and shouldn't be

`agenticIndex` is page-only (not in the API). But per v4.1, "Agents (34%)" is a *category
inside* the Intelligence Index — GDPval-AA v2 + τ³-Banking. So `agenticIndex` is a
sub-score of a composite we already carry, built from two benchmarks §1 recommends adding
individually. **Add the two components, not the sub-index.**

### Also unused, also in the API

`artificial_analysis_math_index` (55 models), `math_500` (28), `aime` (28). `update.py`
already prints an "ignored keys" report listing exactly these — that report was the fastest
path to this whole finding and is worth reading after each run.

---

## 1. Executive summary

**Add now (6).** Ordered by value per unit of work. Coverage is exact (AA API) except where
noted.

| # | Benchmark | Coverage of our 122 | Work | Why |
|---|---|---|---|---|
| 1 | **AA-Omniscience** (+ hallucination rate) | **105 (86%)** / **111 (91%)** | 1 line ea. | Already fetched. v4.1 component (12%). **Hallucination rate is measured nowhere in `llm.json`** and has the best coverage of any unused field. ⚠️ inverted + signed, see §6. |
| 2 | **AA-LCR** (long-context reasoning) | **95 (78%)** | 1 line | API field `lcr`. Long context is also measured **at zero** today. v4.1 component. |
| 3 | **τ³-Banking** | **61 (50%)** | 1 line | API field `tau_banking` — *no page scraping*. **Zero** models ≥0.90 and an 18.1pp top-10 spread, against τ²'s 20-model pileup at ≥0.90 and 3.8pp. v4.1 component (14%). |
| 4 | **Toolathlon** | **22** (18 lab-independent) | new fetcher, easy | Execution-verified (final system state, no LLM judge). Better coverage than `deepswe`/`frontierswe`/`swe_marathon`/`osworld`. |
| 5 | **HMMT Feb 2026** (MathArena) | 15 + new source | new fetcher | Post-cutoff exam, independently run, per-question data + CIs. Unlocks MathArena (APEX 2025: 18 models, USAMO 2026, AIME backfill). |
| 6 | **GDPval-AA v2** | 59 raw / **40 normalized** | 1 line | Already fetched. Largest v4.1 component (20%). Thinner than rev 1 claimed — pick raw Elo (59) or normalized 0–1 (40). |

**Add with low weight / eyes open (4).** `MMLU-Pro` (**75 (61%) union** — upgraded, see §8),
`MCP-Atlas` (~8–10; `fetch_swe_atlas.py` handles the URL unmodified), `AA Math Index`
(54, API, unused), `SWE-bench Multilingual` (coding, w≈0.3).

**Skip (14).** `BFCL V4`, `CursorBench`, `ProgramBench`, `Codeforces`, `Aider Polyglot`,
`IMO-AnswerBench` (marginal), plus three that revision 1 got wrong or that the API settles:

- **`IFBench`** — 83% coverage, but AA removed it from v4.1 for saturation (§8).
- **`CritPt`** — 44 models, **median 1.4%**, max 23%. Floor-compressed to uselessness on
  open-weight models. Revision 1 recommended it on a bad sample (§8).
- **`APEX-Agents`** — good benchmark, **12 models (10%)**. Not enough to rank.

And the whole 2023–24 tool-use cohort: `ToolBench`, `ToolLLM`, `APIBench`, `WildToolBench`,
`HammerBench` — **zero** coverage in every channel checked.

**`llm.json` is fully in sync with AA.** A dry `./update.py` against the live API reports
`score values updated from artificialanalysis.py: 0` across all 115 matched models. Nothing
mapped is missing. (An earlier draft claimed ~26 missing ingests; that was an artefact of
counting AA's literal `0` as a score — `normalize_aa_value()` correctly treats `0` as unset.
See §8.)

**Also: fix `aa_coding_index`.** It holds 11 values AA no longer serves, 9 of them
pre-v4.1. See [§0](#aa_coding_index--legacy-and-mixing-two-definitions-).

---

## 2. Your tau3 question, answered directly

### Which variant is listed most?

**`tau3-bench` — the cross-domain aggregate.** But every variant is too thin to use.
Model counts on llm-stats.com (all models, open + closed):

| variant | models listed |
|---|---|
| **τ³-Bench (aggregate)** | **5** |
| τ³-Banking | 2 |
| τ³-Telecom | 1 |
| τ³-Retail | 1 |
| τ³-Airline | 1 |
| *τ²-Telecom (what we track today)* | *35* |
| *τ²-Retail* | *26* |
| *τ²-Airline* | *23* |

Hugging Face model cards across our 112 HF-hosted models agree: exactly **2** models
report any τ³ number (`TAU3-Bench` on kimi-k3, `τ³-Banking` on qwen3-6-35b-a3b). Neither
evals.report nor Scale Labs lists τ³ at all.

**Conclusion: do not add τ³-Bench from the public aggregators. It is not measurable yet.**

### Which variant is provided by a fetch tool we have? — τ³-Banking, straight from the API

This is the useful answer, and it is cheaper than the first draft assumed. **The AA v2 API
returns `tau_banking` inside `evaluations`** — no page scraping involved. `update.py`
already sees it: its own "ignored keys" report lists `tau_banking` among 19 AA fields it
fetches and then discards. So the whole change is one line:

```python
# update.py — that's it. No artificialanalysis.py change needed.
SCORE_MAPPINGS = {..., "tau3_banking": (("tau_banking",), to_percent)}
```

**Coverage: 61 of 122 models (50%)**, counted the way the repo counts — AA's literal `0`
treated as unset, per `normalize_aa_value()`.

### The case for adding it: τ² has run out of headroom, τ³ has not

Distributions over our own models, from the live API:

| | n | min | p25 | median | p75 | max | **n ≥ 0.90** | **top-10 spread** | n ≤ 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| `tau2` (tracked) | 95 | 0.041 | 0.246 | 0.436 | 0.874 | 0.991 | **20** | **0.038** | 2 |
| `tau_banking` | 61 | 0.004 | 0.052 | 0.082 | 0.135 | 0.334 | **0** | **0.181** | 14 |

τ²-Telecom has **20 models bunched at ≥0.90** and separates its top ten by **3.8 percentage
points** — inside run-to-run noise. τ³-Banking has nothing at the ceiling and **4.8× the
top-10 spread**. For a table whose job is ranking, that is the whole argument.

**But do not replace τ² with τ³ — add it.** They fail at opposite ends. τ³ is
*floor*-compressed: 14 of 61 models score ≤0.05, so it cannot separate the small-model tail,
which is exactly where τ² still resolves cleanly (2 of 95 at ≤0.05). Their coverage is
complementary too:

```
tau2 only: 42    both: 53    tau3 only: 8    union: 103 of 122 (84%)
```

Keep both columns. If you build an agentic composite later, weight τ³ high and τ² low — the
same logic that put `swe_bench_verified` at 0.15 in the coding group.

### Corroboration that AA reached the same conclusion

Intelligence Index **v4.1** (2026-06-15) swapped τ²-Bench Telecom → τ³-Bench Banking, stated
reason: "move to newer, more robust task sets with harder, more realistic agentic
scenarios." Full v4.1 composition is in [§0](#0-are-we-using-the-current-aa-indices).

Note the other implication: v4.1 also dropped LiveCodeBench, AIME-2025 and MMMU-Pro as index
components — consistent with `livecodebench: null`, `aime25: null`, `mmmuPro: null` on
GLM-5.2. **Our `livecodebench` (90 models), `aime_2025` (75), `mmmu_pro` (42) and
`tau2_bench_telecom` (100) columns will gain new models from AA more slowly than they used
to**, worth planning for independently of anything on this list.

## 3. Coverage / scrapeability / trust — full table

Coverage = number of the 122 models in `llm.json` obtainable from the **best single
channel**. All AA figures are exact full-population counts using the repo convention
(`0` == unset). Non-AA figures come from the channel named; see Appendix A.

| Candidate | Coverage | Best channel | Scrapeable? | Trust | Verdict |
|---|---|---|---|---|---|
| **AA-Omniscience** (+ halluc. rate) | **105 (86%)** / **111 (91%)** | AA page `omniscience` | ✅ already fetched | **High** — AA-run, v4.1 component (12%) | **Add** |
| **AA-LCR** | **95 (78%)** | AA API `lcr` | ✅ already fetched | **High** — AA-run, v4.1 component | **Add** |
| **τ³-Banking** | **61 (50%)** | AA API `tau_banking` | ✅ already fetched | **High** — AA-run, no self-report | **Add** |
| **GDPval-AA v2** | 59 raw / **40 norm.** | AA page `gdpval` / `gdpval_normalized` | ✅ already fetched | **High** — AA-run, human-baselined Elo | **Add** |
| CritPt | **44 (36%)** | AA page `critpt` | ✅ already fetched | High provenance, but **median 1.4%** — floor-compressed | **Skip**, see §6 |
| **Toolathlon** | **22** (18 verified) | `toolathlon.xyz/docs/leaderboard` | ✅ plain HTML, open-weight + verified flags | **High** — execution-based state verification | **Add** |
| **HMMT Feb 2026** | **15** + 3 HF | MathArena `/competition_tables/hmmt--hmmt_feb_2026` | ✅ JSON endpoint, `Open` column | **High** — independent, post-cutoff, CIs | **Add** |
| MMLU-Pro | **75 (61%) union**: 38 AA API + 43 HF, overlap 6 | AA API `mmlu_pro` + HF cards | ✅ | **Medium** — half the coverage is AA-run; saturated 83–90%, documented label errors | Add, low weight |
| AA Agentic Index | 57 (47%) | AA page `agentic_index` | ✅ already fetched | n/a — sub-score of `aa_intelligence_index` | **Skip**, see §0 |
| MCP-Atlas | ~8–10 | Scale Labs `mcp_atlas` (6) + llm-stats (8) | ✅ **existing code works unmodified** | Med-High — Scale AI, 1000 human-verified tasks; LLM judge changed Apr 2026 | Add |
| AA Math Index | 54 (44%) | AA API `artificial_analysis_math_index` | ✅ already fetched | Medium — composite, overlaps AIME | Add, low weight |
| SWE-bench Multilingual | 17 HF / 6 evals.report / 3 official | swebench.com embedded JSON (`os_model` flag) | ✅ | **Medium** — official board has only 4 open models; the 17 are self-reports | Add, w≈0.3 |
| APEX-Agents-AA | **12 (10%)** | AA page `apex_agents` | ✅ already fetched | **High** — AA-run, 452 pro-services tasks | **Skip** — too thin to rank |
| APEX (MathArena math) | **18** | MathArena `apex--apex_2025` | ✅ | High — independent | Add if you take MathArena |
| **IFBench** | 101 (83%) | AA API `ifbench` | ✅ already fetched | **Low** — AA *removed* it from v4.1: "no longer distinguishes frontier models sufficiently" | **Skip** despite coverage |
| IMO-AnswerBench | 11 HF / 19 listed | HF, llm-stats | ⚠️ no canonical leaderboard | Medium — DeepMind, 400 problems; top open models 90–92% → saturating | Skip / defer |
| BFCL V4 | 14 official + 11 HF, **not poolable** | `gorilla.cs.berkeley.edu/data_overall.csv` | ✅ CSV, has `License` col | **Low-Med** — see §5 | **Skip** |
| Codeforces | 9 HF | none standardized | ⚠️ | **Low** — Elo units, every lab a different harness/pass@k | **Skip** |
| Aider Polyglot | 3 HF / 4 open | evals.report (1 line) | ✅ trivial | **Low** — saturated, effectively frozen | **Skip** |
| CursorBench | 2 open | cursor.com HTML | ⚠️ | Med — contamination-proof by construction, but vendor-run, no method paper | **Skip** (coverage) |
| ProgramBench | 3 HF / 0 open on evals.report | vals.ai, evals.report | ✅ | Med-High — novel construction, Meta/Stanford/Harvard | **Skip** (coverage), revisit |
| AA Coding Agent Index | n/a | AA | ✅ | n/a — **100% double-count** of `deepswe` + `terminal_bench_2_0` + `swe_atlas_qna`; per-harness rows | **Skip**, see §0 |
| τ³-Bench (aggregate) | 5 listed, 1–2 open | llm-stats only | ⚠️ | n/a — too thin to rank | **Skip**, see §2 |
| GDPval (raw OpenAI) | 1 HF / 3 listed | — | — | — | Use AA variant instead |
| ToolBench | **0** | — | — | — | **Skip** |
| ToolLLM | **0** | — | — | — | **Skip** |
| APIBench (Gorilla) | **0** (3 listed on llm-stats) | — | — | — | **Skip** |
| WildToolBench | **0** | — | — | — | **Skip** |
| HammerBench | **0** | — | — | — | **Skip** |

### On the 2023–24 tool-use cohort

`ToolBench`, `ToolLLM`, `APIBench`, `WildToolBench`, `HammerBench` are all still cited in
academic work, so "deprecated" is the wrong word. But for a *leaderboard aggregator* they
are dead:

- **0 occurrences** across 629 distinct benchmark labels scraped from 112 HF model cards.
- **0** of them appear among llm-stats.com's 626 benchmark slugs (except `api-bank` and
  `gorilla-benchmark-api-bench`, at 3 models each — both closed-model-only).
- **0** on evals.report's 82 benchmarks.

No open-weight frontier model reports them. There is nothing to aggregate. Skip all five.

---

## 4. Coding candidates and how to weight them

### Reading the existing weight scale

Weights in `sortGroups[coding]` are consumed by `computeGroupComposite()` in `llm.html`,
which percentile-normalizes each benchmark and then weights by *reliability*. The existing
ladder decodes cleanly as **contamination resistance × unsaturation × harness control ×
task realism × non-redundancy**:

| w | benchmarks | what earns that tier |
|---|---|---|
| 1.00 | `deepswe` | original hand-written long-horizon tasks, no public answers |
| 0.90 | `frontierswe`, `frontiercode`, `swe_marathon` | frontier hand-crafted, unsaturated, independent |
| 0.85 | `terminal_bench_2_1` | real agentic terminal work, controlled harness, current version |
| 0.80 | `swe_rebench` | continuously refreshed + decontaminated, standardized harness |
| 0.40 | `swe_bench_pro`, `livecodebench` | contamination-resistant-ish, but LLM-judged or algorithm-only |
| 0.35 | `terminal_bench_2_0`, `scicode` | superseded version / narrow domain |
| 0.20 | `aa_coding_index` | **remove** — it *is* `mean(terminal_bench_2_1, scicode)`, R²=0.993 |
| 0.17 | `swe_atlas_{rf,tw,qna}` | small n (70–124), rubric-graded, narrow slice; 3 × 0.17 ≈ 0.5 combined |
| 0.15 | `swe_bench_verified` | saturated, contaminated, harness-confounded |

Current group total weight **7.71**; `minScoredFraction` 0.2 ⇒ a model needs **1.542**
scored weight to be ranked at all. **72 of 122 models clear that bar.**

**Before adding anything, remove `aa_coding_index` from this group.** AA defines it as the
equal-weighted average of Terminal-Bench v2.1 and SciCode — both already here — and that
reconstructs at **R² = 0.9926** on our own models. Its 0.2 weight is therefore not a signal;
it silently lifts `terminal_bench_2_1` to 0.95 effective (+12%) and `scicode` to 0.45
(+29%), handing the largest boost to the narrowest benchmark in the group. Removal costs
3 models their ranking, all of which were clearing the evidence bar on duplicated evidence.
Full working in [§0](#aa_coding_index--drop-it-from-the-coding-group-it-is-a-duplicate-not-a-benchmark-).

An important structural fact: the four 0.9–1.0 benchmarks carry **47.9% of the group
weight but cover only 5–9 models each**. For the other ~113 models the composite is
effectively `terminal_bench_2_1 + swe_rebench + livecodebench + scicode +
aa_coding_index`. So a mid-coverage coding benchmark has more real influence on the
visible ranking than its nominal weight suggests — argues for being conservative.

### Proposed weights

**`swe_bench_multilingual` → 0.30**

Real GitHub PRs across 9 languages, 300 tasks, official leaderboard runs a standardized
mini-SWE-agent. Genuinely adds language coverage the group lacks (everything else is
Python-heavy). Discounts:

- Same construction pipeline and public since 2025 → the contamination profile of
  `swe_bench_pro` (0.4), not of `deepswe`.
- Heavily overlaps `swe_bench_verified`/`swe_bench_pro`.
- **The coverage is the problem.** The official leaderboard has only **4** open-weight
  entries (GLM-5, MiniMax 2.5, Kimi K2.5, DeepSeek V3.2 — 3 of ours). The 17-model figure
  comes from HF model cards, i.e. self-reported, harness-of-choice numbers. Mixing
  self-reports and official runs in one percentile-normalized column is exactly the trust
  hazard the weight scale exists to price.

0.30 sits it just below `swe_bench_pro` — right, given the mixed provenance.

**`cursorbench` → do not add.** On the *design* axis it would earn ~0.6: tasks are private
and drawn from live IDE sessions, so it is contamination-proof, and it discriminates well
(37.6–70.5% across 53 configurations). But it is vendor-run with no published methodology,
not reproducible, and only **2 open-weight models** appear on any aggregator. At 2/122 it
would be imputation for 98% of the table.

**`programbench` → do not add yet, at w≈0.5 when it grows.** The construction (rebuild a
program from a binary + spec, hidden fuzz-generated tests) is the most
contamination-resistant thing on this list, and it comes from Meta/Stanford/Harvard. But
the public leaderboard is 7 models, and evals.report shows **0** open-weight entries. Worth
a calendar reminder, not a column.

**`codeforces` → do not add.** Two independent blockers. (a) Units: labs publish Elo,
percentile, *and* pass-rate under that one name — the percentile normalization in
`computeGroupComposite` would happily rank incommensurable numbers against each other. (b)
There is no independent standardized leaderboard; the 9 models come from self-reports at
different pass@k. Capability-wise it is `livecodebench` (0.4) again, algorithm-only.

**`aider_polyglot` → do not add.** 225 fixed Exercism exercises, saturated, leaderboard
effectively frozen, 3 HF self-reports and 4 open models on evals.report. It would be a
0.10 column with 3% coverage. It is the one candidate here I would call plainly dead.

### Non-coding candidates, for reference

`toolathlon`, `mcp_atlas` and `tau3_banking` are all agentic **tool use**, not coding —
they belong beside `tau2_bench_telecom`, not in the coding group. If you ever add an
"Agentic Tool Use (grouped)" sort group, the analogous weights would be roughly:
`tau3_banking` 0.9 (AA-run, unsaturated, best coverage), `toolathlon` 0.85
(execution-verified, no judge), `mcp_atlas` 0.4 (LLM judge, judge changed mid-2026 so the
time series is not internally comparable), `tau2_bench_telecom` 0.15 (saturated — same
reasoning that put `swe_bench_verified` at 0.15).

---

## 5. Why BFCL V4 fails despite looking good on paper

It is the most tempting skip on the list, so the reasoning in full:

**In its favour.** Official, academic, peer-reviewed (ICML 2025), and the data is a clean
public CSV at `https://gorilla.cs.berkeley.edu/data_overall.csv` — 109 rows, 36 columns,
including a `License` column that identifies open weights for free. V4 adds web search,
memory, and a 26-variation format-sensitivity probe.

**Why it still fails.** The official leaderboard covers the *tail*, not the *head*. The 14
of our models it contains are `gemma-3-*`, `llama-4-*`, `phi-4`, original `qwen3-*`,
`command-a`, `glm-4.6`. It has **no** Kimi K2/K3, **no** DeepSeek V4, **no** Qwen3.5/3.6,
**no** GLM-5.x — i.e. none of the models whose relative ranking anyone is looking at. The
11 models that *do* report BFCL V4 in their HF cards are exactly the frontier ones the
official board lacks, so the two channels are disjoint, and V3 and V4 numbers are not
comparable.

You would be building a column that is precise about 2025 models and silent about 2026
ones. For the same capability, τ³-Banking gives 61 models from a source we already query,
and Toolathlon gives 22 execution-verified ones. Secondary concerns (AST-matching brittleness, format sensitivity
being a sub-track rather than the headline number, `IFEval-FC` and `MCP-AgentBench`
critiques) only reinforce it.

---

## 6. Bonus — 19 AA fields you already fetch and discard

`update.py`'s own report, run against the live API, lists exactly what is being thrown away:

```
ignored keys:
  agentic_index                     gdpval_normalized
  aime                              harvey_lab_criteria_pass
  apex_agents                       ifbench
  artificial_analysis_math_index    it_bench_sre
  automation_bench_partial_score    lcr
  critpt                            math_500
  enterprise_ops_gym                mmlu_pro
  gdpval                            omniscience
                                    omniscience_accuracy
                                    omniscience_hallucination_rate
```

Exact coverage over our 122 models, from a full `./artificialanalysis.py --open` run
(111 of our slugs matched; `step-3-5-flash`, `north-mini-code`, `glm-4-6` and
`nvidia-nemotron-3-ultra-550b-a55b` are dropped by the `--open` filter, so these are lower
bounds), plus the value range so you can judge discrimination:

| AA field | n | %122 | source | range (median) | verdict |
|---|---|---|---|---|---|
| `omniscience_hallucination_rate` | **111** | **91%** | page | 0.121 – 0.988 (0.844) | **Add.** Best coverage of anything unused. Hallucination is measured *nowhere* in `llm.json`. ⚠️ **lower = better** — inverts the sort/percentile assumption. |
| `omniscience` | **105** | **86%** | page | −84.7 – 18.4 (−48.1) | **Add.** v4.1 component (12%). ⚠️ **signed** — use an identity transform, not `to_percent`. |
| `ifbench` | 101 | 83% | API | — | **Skip** — AA retired it from v4.1 for saturation. |
| `lcr` | **91** | **75%** | API | — | **Add.** Long-context reasoning, measured nowhere in `llm.json`. |
| `tau_banking` | 59 | 48% | API | 0.004 – 0.334 (0.082) | **Add.** See §2. |
| `gdpval` (raw Elo) | 59 | 48% | page | −120 – 1687 (718) | Add *or* use normalized. Elo, can be negative. |
| `agentic_index` | 57 | 47% | page | 0.3 – 50.1 (10.5) | Skip — sub-score of `aa_intelligence_index`. |
| `artificial_analysis_math_index` | 54 | 44% | API | — | Add, low weight (overlaps AIME). |
| `critpt` | 44 | 36% | page | 0.003 – **0.234** (**0.014**) | **Skip.** Median 1.4% — floor-compressed to uselessness on open-weight models. |
| `gdpval_normalized` | 40 | 33% | page | 0.021 – 0.594 (0.236) | Add — cleanest scale, but 19 fewer models than raw `gdpval`. |
| `mmlu_pro` | 37 | 30% | API | — | Add, low weight; union with HF self-reports = 75 (61%). |
| `math_500` | 28 | 23% | API | — | Skip — saturated. |
| `aime` | 27 | 22% | API | — | Skip — `aime_2025`/`aime_2026` already cover this better. |
| `it_bench_sre` | 14 | 11% | page | — | Skip — too thin. |
| `harvey_lab_criteria_pass` | 13 | 11% | page | — | Skip — too thin, legal-domain-specific. |
| `apex_agents` | **12** | **10%** | page | — | **Skip** — this is the APEX-Agents on your candidate list. High-quality benchmark, 12 models. |
| `enterprise_ops_gym` | 12 | 10% | page | — | Skip — too thin. |
| `automation_bench_partial_score` | 11 | 9% | page | — | Skip — too thin. |
| `omniscience_accuracy` | 111 | 91% | page | — | Redundant with `omniscience`; pick one. |

**The headline is AA-Omniscience, not CritPt.** Revision 1's 35-model sample had CritPt at
91% and Omniscience at 26%; the exact figures are the reverse (36% and 86%). AA's per-benchmark
backfill is not monotone in model recency, so a recency-skewed sample is unreliable in
*both* directions — see §8.

Two of the three best-covered unused fields (`omniscience_hallucination_rate`, `omniscience`)
need transform care: one is inverted, one is signed. Neither is a `to_percent` field.

---

## 7. Suggested order of work

Steps 1–2 are one line of code each and cover 4–6 new columns.

1. **`tau3_banking` + `aa_lcr` + `mmlu_pro` + `aa_math_index`** — one `SCORE_MAPPINGS` entry
   each in `update.py`, plus a `benchmarks` entry in `llm.json`. All four come from the AA
   **API**, already fetched, already listed in `update.py`'s own "ignored keys" report:

   ```python
   SCORE_MAPPINGS = {
       ...,
       "tau3_banking":   (("tau_banking",),                     to_percent),
       "aa_lcr":         (("lcr",),                             to_percent),
       "mmlu_pro":       (("mmlu_pro",),                        to_percent),
       "aa_math_index":  (("artificial_analysis_math_index",),  lambda v: v),
   }
   ```

2. **`omniscience` (105), `omniscience_hallucination_rate` (111), `gdpval` (59)** — same
   one-liner each, but page-only (`_PAGE_ONLY_EVALS`), so they cost a model-page fetch per
   model on every run — already paid for today, just not stored. Mind the transforms:
   `omniscience` is **signed** (−84.7 … 18.4) so it needs identity, not `to_percent`; and
   `omniscience_hallucination_rate` is **inverted** (lower is better), which the sort and
   `computeGroupComposite()` do not expect.

3. **Remove `aa_coding_index` from `sortGroups[coding]`** — it is `mean(terminal_bench_2_1,
   scicode)` at R²=0.993, so it double-counts two group members. Optionally drop the display
   column too, or at least null its 11 orphaned values. See
   [§0](#aa_coding_index--drop-it-from-the-coding-group-it-is-a-duplicate-not-a-benchmark-).
   This is a correctness fix, not an addition.

4. **`mcp_atlas`** — add `"mcp_atlas"` to `TRACKS` in `fetch_swe_atlas.py`; the
   flight-payload parser already handles that URL unmodified (verified). Plus mapping wiring.

5. **`toolathlon`** — new `fetch_toolathlon.py`. Plain HTML table; the page exposes
   open-weight and "evaluated by us" flags, so provenance filtering is free.

6. **`hmmt_feb_2026`** — new `fetch_matharena.py` against
   `/competition_tables/<competition-id>`. Also unlocks APEX 2025 (18 models), USAMO 2026,
   and a backfill for `aime_2025`/`aime_2026`.

7. **`swe_bench_multilingual` @ w=0.30** — swebench.com embeds
   `<script type="application/json" id="leaderboard-data">` with an `os_model` flag. Decide
   first whether to pool HF self-reports with official runs; I would take official-only and
   accept 3–4 models until the board fills in.

Deliberately **not** in this list: `IFBench` (86% coverage but AA dropped it for
saturation), the `AA Coding Agent Index` (pure double-count), `agenticIndex` (sub-score of a
composite we carry).

---

## 8. Corrections from revision 1

Revision 1 relied on scraping AA model pages. With the API key, four things changed. Each is
a case where the earlier number was measured but measured wrongly.

**1. τ³-Banking coverage: ~85 → 61 (50%).** Rev 1 sampled 28 newest + 12 oldest models, which
over-weighted exactly the models AA has run τ³ on. The exact count over all 122 is 61. Also
the implementation is cheaper than stated: it is in the **API** as `tau_banking`, so no
`artificialanalysis.py` change is needed at all — one `SCORE_MAPPINGS` line.

**2. "AA dropped MMLU-Pro" — wrong, and MMLU-Pro is upgraded.** Rev 1 reported 0/35 from AA.
That was an artefact of probing a page key (`mmluPro`) that **does not exist**; the model
page only carries `mmmuPro` (MMMU-Pro, a different benchmark). The API does serve `mmlu_pro`,
for **38** of our models. Those 38 are AA-run, i.e. trustworthy, and they skew *older* while
the 43 HF self-reports skew *newer* (Qwen3.5/3.6, gemma-4, granite-4.1) — overlap is only 6.
**Union: 75 of 122 (61%)**, the best-covered candidate on the list. It is still saturated at
the top; the argument for it remains resolution at the small-model tail, but the channel is
better than rev 1 claimed.

**3. "IFBench: add it" — retracted.** 105 of 122 (86%) coverage, which is why rev 1 liked it.
But AA **removed IFBench from Intelligence Index v4.1** on 2026-06-15, stated reason: "The
benchmark no longer distinguishes frontier models sufficiently." Adding a column its own
maintainer just retired for saturation is the mistake this report exists to avoid.

**4. "~26 missing ingests" — retracted entirely.** Rev 1 claimed the API held scores
`llm.json` lacked (`terminal_bench_hard` +14, `tau2` +7, …). Those API values are literal
`0`, and `normalize_aa_value()` in `update.py` correctly treats `0` as unset. A dry
`./update.py` against the live API confirms: **`score values updated from
artificialanalysis.py: 0`** across all 115 matched models. `llm.json` is fully in sync.

That last one also means every "0-as-null" coverage count in rev 1 was inflated. Corrected
figures: `lcr` 112→**95**, `tau2` 102→**95**, `terminalbench_hard` 102→**88**,
`terminalbench_v2_1` 64→**60**, `tau_banking` 63→**61**. `ifbench` (101) and `mmlu_pro` (38)
contain no zeros and are unchanged.

**5. CritPt and AA-Omniscience swapped places, and CritPt is now a skip.** Rev 1's 35-model
sample put CritPt at 91% and Omniscience at 26%. The exact figures from a full
`./artificialanalysis.py --open` run are the reverse: **CritPt 44 (36%)**, **Omniscience 105
(86%)**. Worse for CritPt, its distribution on open-weight models is 0.003 – 0.234 with
**median 0.014** — one and a half percent. It cannot separate models in this index at all.
AA-Omniscience is now a top-two recommendation, and its companion field
`omniscience_hallucination_rate` (111, 91%) has the best coverage of any unused field, in a
dimension `llm.json` does not measure.

**Methodological lesson.** The rev-1 sample was 28 newest + 12 oldest models by `date_added`,
on the assumption that AA coverage decreases with model age. It does not, uniformly: AA
backfills some benchmarks onto old models and not others, so a recency-skewed sample errs in
*both* directions — CritPt was overestimated 2.5×, Omniscience underestimated 3.3×. Every
coverage figure sourced from AA in this revision is an exact full-population count. The
remaining sampled or fuzzy-matched figures are flagged in Appendix A.

**6. `aa_coding_index` is not "legacy, being wound down" — it is current, and a duplicate.**
Rev 1 read its absence from the v4.1 intelligence-benchmarking page as deprecation. In fact
AA documents it separately and still produces it for 92% of models added in 2026-H2. The real
defect is that AA defines it as `mean(Terminal-Bench v2.1, SciCode)`, which reconstructs at
**R² = 0.9926** on our models — both components are already in the coding group. The verdict
changes from "clean up its 11 stale values" to "remove it from the group": see
[§0](#aa_coding_index--drop-it-from-the-coding-group-it-is-a-duplicate-not-a-benchmark-).

**Unchanged from rev 1:** the Toolathlon (22), MCP-Atlas (~8–10), HMMT Feb 2026 (15),
MathArena APEX (18), SWE-bench Multilingual (17/6/3) and BFCL V4 (14 official + 11 HF)
figures, all the skip verdicts on the 2023–24 tool-use cohort, and every coding weight in §4
for benchmarks being *added*.

---

## Appendix A — how coverage was measured

Four independent channels, each measured directly rather than read off documentation.

**1. Hugging Face model cards.** Ran `./fetch_huggingface.py --all-models`; 112 of 122
models have an HF URL and parsed successfully. Yielded 629 distinct benchmark labels, then
counted distinct `llm.json` models per candidate across all label spellings (e.g. HMMT
appears as 19 different strings). *Caveat: self-reported. Labs pick the harness, the
pass@k, and whether to publish at all. High coverage, low trust — the two are inversely
correlated across this whole table.*

**2. Artificial Analysis — exact, via the v2 API and a full page run.** Two paths:

- `GET /api/v2/data/llms/models` with a key → 590 records, 115 of our 122 slugs matched
  exactly. Serves 17 evaluation fields including `tau_banking`, `lcr`, `ifbench`, `mmlu_pro`,
  `artificial_analysis_math_index`, `math_500`, `aime`.
- A full `./artificialanalysis.py --open --output json` run → 332 open-weight models, 111 of
  ours (the `--open` filter drops `step-3-5-flash`, `north-mini-code`, `glm-4-6`,
  `nvidia-nemotron-3-ultra-550b-a55b`), which adds the page-only fields: `gdpval`,
  `gdpval_normalized`, `critpt`, `omniscience*`, `apex_agents`, `agentic_index`,
  `it_bench_sre`, `harvey_lab_criteria_pass`, `automation_bench_partial_score`,
  `enterprise_ops_gym`.

**All AA counts use the repo's own convention: a literal `0` counts as unset**, matching
`normalize_aa_value()` in `update.py`. Ignoring that inflates every count — see §8. Cross-checks:
`./update.py` dry-run reports 0 score changes across 115 matched models, and llm.json's own
per-benchmark counts are ≥ the API counts everywhere (it also ingests from HF/llm-stats).
The 7 models AA does not carry at all: `longcat-flash-chat`, `longcat-flash-thinking`,
`ornith-1-0-{9b,35b,397B}`, `agents-a1`, `laguna-xs-2`.

**3. Aggregators.** llm-stats.com (626 benchmark slugs; per-benchmark model counts via the
Next.js flight payload), evals.report (82 benchmarks; scraped 18 candidate pages through
`fetch_evals_report.py` with the `BENCHMARKS` dict extended at runtime — its `" Open"`
suffix convention gives open-weight status), benchlm.ai (369 slugs).

**4. Official leaderboards.** Scale Labs (`mcp_atlas`, via `fetch_swe_atlas.py`
unmodified), toolathlon.xyz, swebench.com (embedded `leaderboard-data` JSON),
matharena.ai (`/competition_tables/<id>`), gorilla.cs.berkeley.edu (`data_overall.csv`),
cursor.com, vals.ai.

Model-name matching to `llm.json` slugs used normalization plus substring fallback, so
counts are ±1–2 at the margins (e.g. Toolathlon's "Qwen-3-Coder" is ambiguous between
`qwen3-coder-next` and `qwen3-coder-480b-a35b-instruct`). Where a candidate's verdict
hinged on the number, the matches are listed in full above.

**Still not measured.** Whether Toolathlon and MCP-Atlas keep adding open-weight models at
their current rate; the size of the disagreement between HF-self-reported and officially-run
scores for the same model/benchmark pair (worth checking before pooling them in
`swe_bench_multilingual`, and now checkable for MMLU-Pro on the 6 models that appear in both
the AA API and HF cards); and whether AA's `omniscience` sign convention is stable enough to
build a column on.
