# Benchmark candidate evaluation — August 2026

Assessment of 19 candidate benchmarks against three criteria: **coverage** on the 122
models in `llm.json`, **scrapeability**, and **trustworthiness**. Coding candidates also
get a proposed weight for the `coding` sort group.

All coverage figures were measured, not estimated from documentation. Method and caveats
are in [Appendix A](#appendix-a--how-coverage-was-measured). Snapshot: 2026-08-01.

---

## 1. Executive summary

**Add now (5).** Ordered by value per unit of work:

| # | Benchmark | Coverage of our 122 | Work | Why |
|---|---|---|---|---|
| 1 | **τ³-Banking (AA implementation)** | ~85 (69% sampled) | 3 lines | Replaces `tau2_bench_telecom`, which is **saturated** (GLM-5.2 = 99.1%). AA already dropped τ² from its index in favour of this. |
| 2 | **GDPval-AA v2 (normalized)** | ~85 (69% sampled) | 2 lines | Already fetched by `artificialanalysis.py`; just not stored. Real-work tasks, AA-graded. |
| 3 | **HMMT Feb 2026 (MathArena)** | 15 direct + new source | new fetcher | Post-cutoff exam, independently run, per-question data + CIs. Unlocks MathArena as a source (APEX 2025: 18 models, USAMO 2026, AIME backfill). |
| 4 | **Toolathlon** | 22 (18 lab-independent) | new fetcher, easy | Execution-verified (final system state, no LLM judge). Better coverage than `deepswe`/`frontierswe`/`swe_marathon`/`osworld`. |
| 5 | **MCP-Atlas** | ~8–10 | ~10 lines | `fetch_swe_atlas.py` works on `labs.scale.com/leaderboard/mcp_atlas` **unmodified** (verified). |

**Add with low weight / eyes open (3).** `SWE-bench Multilingual` (coding, w≈0.3),
`APEX-Agents-AA` (~25 models, free — already fetched), `MMLU-Pro` (43 models but
saturated; only worth it for the small-model tail).

**Skip (11).** `BFCL V4`, `CursorBench`, `ProgramBench`, `Codeforces`, `Aider Polyglot`,
`IMO-AnswerBench` (marginal), and the whole 2023–24 tool-use cohort: `ToolBench`,
`ToolLLM`, `APIBench`, `WildToolBench`, `HammerBench` — **zero** coverage in every
channel checked.

**Bonus finding: four AA fields are already being fetched and thrown away.** `lcr`
(AA-LCR, 97% coverage), `critpt` (91%), `ifbench` (83%), `agenticIndex` (66%). Each is a
1-line `SCORE_MAPPINGS` entry plus a `benchmarks` entry. AA-LCR in particular covers a
capability the index currently measures **not at all** (long-context reasoning) at higher
coverage than any benchmark we track except `gpqa_diamond`/`hle`. See [§6](#6-bonus--four-benchmarks-you-are-already-fetching-and-discarding).

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

### Which variant is provided by a fetch tool we have? — τ³-Banking, via AA

This is the useful answer. `artificialanalysis.py` already scrapes AA model pages; those
pages carry a **`tauBanking`** field that `_PAGE_FLOAT_FIELDS` does not list. AA runs
τ³-Banking themselves — 18 models on their leaderboard, and in a 35-model sample of our
own slugs, **24 had a `tauBanking` score**.

Sampled values (AA, 0–1 scale):

```
kimi-k3            0.334      glm-5-2      0.268      deepseek-v4-pro   0.258
inkling            0.237      gemma-4-31b  0.151      qwen3-5-397b      0.134
minimax-m3         0.130
```

Compare the same models on τ²-Telecom, which we *do* track:

```
glm-5-2  0.991    deepseek-v4-pro  0.962    qwen3-5-397b  0.956    minimax-m3  0.889
```

**τ²-Bench Telecom is saturated for frontier open-weight models.** It contributes ~zero
discrimination at the top of the table while still occupying a column at 82% coverage.
τ³-Banking spreads the same models across 13–33%.

Corroborating evidence that AA made the same call: the `intelligenceIndex` cost breakdown
on an AA model page enumerates its component evaluations as

```
gdpval-aa, tau3-banking, terminalbench-v2-1, scicode, humanitys-last-exam,
gpqa-diamond, critpt, omniscience, artificial-analysis-long-context-reasoning
```

τ²-Bench is gone. So are LiveCodeBench, AIME-2025 and MMMU-Pro — consistent with
`livecodebench: null`, `aime25: null`, `mmmuPro: null` on GLM-5.2's page. **Our
`livecodebench` (74%), `aime_2025` (61%), `mmmu_pro` (34%) and `tau2_bench_telecom` (82%)
columns will quietly stop gaining new models**, which is worth planning for independently
of anything on this list.

**Recommendation:** add `tau3_banking`. Keep `tau2_bench_telecom` as a historical column
but consider dropping it out of any composite you build later.

Implementation (verified against the live payload):

```python
# artificialanalysis.py
_PAGE_FLOAT_FIELDS = [..., ("tauBanking", "tau3_banking")]
_PAGE_ONLY_EVALS   = [..., "tau3_banking"]

# update.py
SCORE_MAPPINGS = {..., "tau3_banking": (("tau3_banking",), to_percent)}
```

---

## 3. Coverage / scrapeability / trust — full table

Coverage = number of the 122 models in `llm.json` obtainable from the **best single
channel**. `AA(s)` = sampled estimate, see Appendix A.

| Candidate | Coverage | Best channel | Scrapeable? | Trust | Verdict |
|---|---|---|---|---|---|
| **τ³-Banking (AA)** | **~85** `AA(s) 24/35` | AA model page `tauBanking` | ✅ already wired | **High** — AA runs it, no self-report | **Add** |
| **GDPval-AA v2 norm.** | **~85** `AA(s) 24/35` | AA `gdpvalNormalized` | ✅ already fetched | **High** — AA-run, automated grading | **Add** |
| **HMMT Feb 2026** | **15** + 3 HF | MathArena `/competition_tables/hmmt--hmmt_feb_2026` | ✅ JSON endpoint, `Open` column | **High** — independent, post-cutoff, CIs | **Add** |
| **Toolathlon** | **22** (18 verified) | `toolathlon.xyz/docs/leaderboard` | ✅ plain HTML, has open-weight + verified flags | **High** — execution-based state verification | **Add** |
| **MCP-Atlas** | ~8–10 | Scale Labs `mcp_atlas` (6) + llm-stats (8) | ✅ **existing code works unmodified** | Med-High — Scale AI, 1000 human-verified tasks; LLM judge, changed Apr 2026 | **Add** |
| SWE-bench Multilingual | 17 HF / 6 evals.report / 3 official | swebench.com embedded JSON (`os_model` flag) | ✅ | **Medium** — official leaderboard has only 4 open models; the 17 are self-reports | Add, w≈0.3 |
| APEX-Agents-AA | ~25 `AA(s) 7/35` | AA `apexAgents` | ✅ already fetched | **High** — AA-run, 452 pro-services tasks | Add (free) |
| MMLU-Pro | **43** HF / 127 listed | HF cards, llm-stats | ✅ | **Low-Med** — saturated 83–90%, documented label errors, **AA dropped it (0/35)** | Add only for small-model tail |
| IMO-AnswerBench | 11 HF / 19 listed | HF, llm-stats | ⚠️ no canonical leaderboard | Medium — DeepMind, 400 problems; top open models 90–92% → saturating | Skip / defer |
| BFCL V4 | 14 official + 11 HF, **not poolable** | `gorilla.cs.berkeley.edu/data_overall.csv` | ✅ CSV, has `License` col | **Low-Med** — see §5 | **Skip** |
| Codeforces | 9 HF | none standardized | ⚠️ | **Low** — Elo units, every lab a different harness/pass@k | **Skip** |
| Aider Polyglot | 3 HF / 4 open | evals.report (1 line) | ✅ trivial | **Low** — saturated, effectively frozen | **Skip** |
| CursorBench | 2 open | cursor.com HTML | ⚠️ | Med — contamination-proof by construction, but vendor-run, no method paper | **Skip** (coverage) |
| ProgramBench | 3 HF / 0 open on evals.report | vals.ai, evals.report | ✅ | Med-High — novel construction, Meta/Stanford/Harvard | **Skip** (coverage), revisit |
| GDPval (raw OpenAI) | 1 HF / 3 listed | — | — | — | Use AA variant instead |
| APEX (MathArena math) | **18** | MathArena `apex--apex_2025` | ✅ | High — independent | Add if you take MathArena |
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
| 0.20 | `aa_coding_index` | composite, overlaps everything else in the group |
| 0.17 | `swe_atlas_{rf,tw,qna}` | small n (70–124), rubric-graded, narrow slice; 3 × 0.17 ≈ 0.5 combined |
| 0.15 | `swe_bench_verified` | saturated, contaminated, harness-confounded |

Current group total weight **7.71**; `minScoredFraction` 0.2 ⇒ a model needs **1.542**
scored weight to be ranked at all. **72 of 122 models clear that bar.**

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
ones. For the same capability, τ³-Banking gives ~85 models and Toolathlon gives 22
execution-verified ones. Secondary concerns (AST-matching brittleness, format sensitivity
being a sub-track rather than the headline number, `IFEval-FC` and `MCP-AgentBench`
critiques) only reinforce it.

---

## 6. Bonus — four benchmarks you are already fetching and discarding

`_PAGE_ONLY_EVALS` in `artificialanalysis.py` already pulls these on every run. They are
absent from `SCORE_MAPPINGS` and from `llm.json`, so they are fetched and dropped. Coverage
from the same 35-model sample:

| AA field | benchmark | sampled coverage | note |
|---|---|---|---|
| `lcr` | AA Long Context Reasoning | **34/35 (97%)** | Index component. **No benchmark in `llm.json` measures long context at all.** |
| `critpt` | CritPt (physics research) | 32/35 (91%) | Index component, unsaturated (~20%) |
| `ifbench` | IFBench (instruction following) | 29/35 (83%) | No IF benchmark currently tracked |
| `agenticIndex` | AA Agentic Index | 23/35 (66%) | Composite; would want low weight like `aa_coding_index` |

Also present but thin: `harveyLabCriteriaPass` 12/35, `automationBenchPartialScore` 10/35,
`enterpriseOpsGym` 9/35, `omniscience` 9/35, `itBenchSre` 6/35.

**AA-LCR is, by coverage × novelty, the single best available addition** — better coverage
than everything in `llm.json` except `gpqa_diamond` (97%) and `hle` (96%), in a capability
dimension currently at zero. It just was not on the candidate list.

---

## 7. Suggested order of work

1. **`tau3_banking`** — 3 lines (`artificialanalysis.py` ×2, `update.py` ×1) + `llm.json`
   `benchmarks` entry. ~85 models. Fixes the saturated-τ² problem.
2. **`aa_lcr`, `critpt`, `ifbench`, `gdpval_aa`** — 1 `SCORE_MAPPINGS` line each, already
   fetched. Adds 4 columns at 69–97% coverage for near-zero work.
3. **`mcp_atlas`** — add `"mcp_atlas"` to `TRACKS` in `fetch_swe_atlas.py` (the flight-payload
   parser already works on that URL, verified) + mapping wiring.
4. **`toolathlon`** — new `fetch_toolathlon.py`. Plain HTML table; the page exposes
   open-weight and "evaluated by us" flags, so provenance filtering is free.
5. **`hmmt_feb_2026`** — new `fetch_matharena.py` against
   `/competition_tables/<competition-id>`. Also unlocks APEX 2025 (18 models), USAMO 2026,
   and a backfill for `aime_2025`/`aime_2026`.
6. **`swe_bench_multilingual` @ w=0.30** — swebench.com embedded
   `<script id="leaderboard-data">` carries an `os_model` flag. Decide first whether you
   want HF self-reports pooled with official runs; I would take official-only and accept
   3–4 models until the board fills in.
7. **`mmlu_pro`** — only if you want better resolution at the small end of the table
   (`gemma-4-e2b`, `qwen3-5-0-8b`, `lfm2-*`), where `gpqa_diamond` and `hle` are
   floor-saturated and therefore noise. It is genuinely the best-covered candidate (43
   models) and genuinely the least informative at the top. That trade is a judgement call,
   not a technical one.

---

## Appendix A — how coverage was measured

Four independent channels, each measured directly rather than read off documentation.

**1. Hugging Face model cards.** Ran `./fetch_huggingface.py --all-models`; 112 of 122
models have an HF URL and parsed successfully. Yielded 629 distinct benchmark labels, then
counted distinct `llm.json` models per candidate across all label spellings (e.g. HMMT
appears as 19 different strings). *Caveat: self-reported. Labs pick the harness, the
pass@k, and whether to publish at all. High coverage, low trust — the two are inversely
correlated across this whole table.*

**2. Artificial Analysis model pages.** Fetched the RSC flight payload for a 35-model
sample (the 28 most recently added + 12 oldest by `date_added`; 5 of 40 requests failed)
and read fields out of the `currentModel` object. Percentages are of 35 and carry roughly
±8pp; extrapolations to 122 are stated as `~N`. *Method validated against `llm.json`:
33/35 agreement on `tau2` presence and 34/35 on `terminalbenchV21`.* Note `ARTIFICIAL_ANALYSIS_API_KEY`
is not set in this environment, so `artificialanalysis.py` itself could not be run — the
page-scraping path it uses was reproduced directly.

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

**Not measured.** AA coverage for models outside the 35-model sample; whether Toolathlon
and MCP-Atlas will keep adding open-weight models at their current rate; the size of the
disagreement between HF-self-reported and officially-run scores for the same
model/benchmark pair (worth checking before pooling them in `swe_bench_multilingual`).
