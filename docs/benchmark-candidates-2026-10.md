# Benchmark candidates — October 2026

Follow-up to [`benchmark-candidates-2026-08.md`](benchmark-candidates-2026-08.md) and
[`hf-ignored-benchmark-overlap-2026-08.md`](hf-ignored-benchmark-overlap-2026-08.md).
The question is the same: which benchmarks that `llm.json` does not carry would be worth
adding? This time the answer is measured against the **198 models** now in the table (180
open-weight, 18 closed reference rows), and it looks at sources the earlier audits did not
read: the full Epoch AI hub archive, Arena's leaderboard dataset, and every benchmark
llm-stats lists for each mapped model rather than only its flat leaderboard.

Snapshot 2026-10-30. Method and caveats are in [Appendix A](#appendix-a--method).

> **Outcome (2026-10-30).** Everything recommended was added as columns: the six
> Epoch files plus FrontierMath Tiers 1-3, the Arena boards (Text, Text
> non-English, Code/WebDev, Agent, Agent tool hallucination), τ²-Bench Retail and
> Airline as two columns rather than one pooled column, HMMT February 2026,
> MMMLU, Claw-Eval (its official General-split score only) and NL2Repo. Twelve
> joined an index: ALE-Bench and NL2Repo in Coding; APEX-Agents, Vending-Bench 2,
> Claw-Eval and both τ² domains in Tooling; both FrontierMath columns and HMMT in
> Knowledge; SimpleQA Verified and Arena tool hallucination in Trust. See
> README, "OpenBench Index", for the six that did not.
>
> **Correction.** Arena's `agent_tool_hallucination` score is *higher*-is-better
> (rank 1 has the highest score), not lower-is-better as written in §1–§2. The
> correlations below were computed with the sign flipped, so they read
> +0.52 / +0.31 rather than −0.52 / −0.31: the same modest independence, in the
> expected direction.

---

## 0. Headline

- **The table measures no human preference at all.** Arena (formerly LMArena) publishes
  its whole leaderboard as a CC BY 4.0 parquet dataset on Hugging Face, refreshed daily.
  Exact name normalisation alone matches **73 of our models (58 open)**, and a reviewed
  mapping would add about ten more. It is also the **most independent signal of anything
  tested**: Spearman 0.63 against the Coding index and 0.65 against the OpenBench index,
  where every Epoch candidate sits at 0.79–0.96.
- **Epoch's hub already holds several strong benchmarks we don't read.** `fetch_epoch.py`
  downloads `benchmark_data.zip` and then reads 12 of the ~85 files inside it. Among the
  files it skips are **APEX-Agents (33 models)**, **FrontierMath Tier 4 v2 (30)**,
  **SimpleQA Verified (32)**, **ARC-AGI-2 (29)**, **ALE-Bench (51)** and **Vending-Bench 2
  (27)**. Most of these are Epoch's own runs, so each needs only a `Benchmark(...)` entry
  and a column.
- **APEX-Agents has outgrown its August "skip".** In August it was skipped at 12 models.
  The union is now **46 (31 open)**: Epoch 33, llm-stats 9, model cards 13.
- **Math has run out of headroom.** 27 of the 37 `aime_2026` values are ≥ 90, and
  `aime_2025` has a 2.3-point gap between the top model and the fifth. FrontierMath (Epoch's
  own runs, private holdout) is the obvious replacement. HMMT Feb 2026 was recommended in
  August and has still not been added.
- **Three capability areas have no column at all:** multilingual ability, human
  preference, and model safety (refusal or jailbreak robustness). The first two can now be
  filled. Safety still cannot: no channel that was read reports a safety benchmark for more
  than 3 of our models.
- **Already recommended but never added:** τ²-Bench Retail/Airline (now 41 models, all
  open), HMMT Feb 2026 (19), Claw-Eval (24) and WildClawBench (18).

---

## 1. Summary table

`union` = distinct `llm.json` models with a value from at least one channel; `open` /
`closed` splits it by `reference-models.json`. `ρ` is Spearman's rank correlation against
the existing index named; lower means more new information. `top-5` is the best value
minus the fifth best, which shows whether the benchmark can still separate the strongest
models.

| # | Candidate | Measures | union (open / closed) | Best channel | Provenance | top-5 | ρ vs nearest index | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Arena Text (overall)** | Human pairwise preference on real prompts | **73+ (58 / 15)** | HF dataset `lmarena-ai/leaderboard-dataset` | Independent, crowd votes, CIs | 41 Elo | **0.63** Coding, 0.81 Knowledge | **Add**, new category |
| 2 | **APEX-Agents** | Long-horizon professional-services agent tasks (banking, consulting, law) | **46 (31 / 15)** | Epoch `apex_agents_external.csv` + llm-stats + HF | Mercor board via Epoch; self-reports fill | 13.8 pt | 0.82 Tooling | **Add** → Tooling / Professional Work |
| 3 | **FrontierMath T1-3 v2 / Tier 4 v2** | Unpublished research-level math | **33 (18 / 15) / 30 (15 / 15)** | Epoch `frontiermath_*_v2.csv` | **Epoch's own runs**, private set | 3.5 / **10** pt | 0.84 / 0.80 Knowledge | **Add** (T4 first) → Knowledge |
| 4 | **SimpleQA Verified** | Short-form factual recall, abstention | **32 (18 / 14)** | Epoch `simpleqa_verified.csv` | **Epoch's own runs** | 5.2 pt | **0.45** Trust | **Add** → Trust / Knowledge |
| 5 | **ALE-Bench** | Long-horizon algorithm engineering (AtCoder Heuristic, scored not pass/fail) | **51 (38 / 13)** | Epoch `ale_bench_external.csv` | Sakana board via Epoch | 774 rating | 0.88 Coding | **Add** → Coding, low-mid weight |
| 6 | **τ²-Bench Retail + Airline** | Tool-calling customer-service dialogue | **41 (41 / 0)** | HF cards + llm-stats | Self-reported | 3.1 pt | 0.86 Tooling | **Add**, low weight (2nd time recommended) |
| 7 | **Vending-Bench 2** | Year-long agent coherence running a business ($) | **27 (13 / 14)** | Epoch `vending_bench_2_external.csv` | Andon Labs board via Epoch | wide | 0.81 Tooling, **0.77** Trust | **Add** → Tooling |
| 8 | **ARC-AGI-2** | Abstract fluid reasoning on novel grid puzzles | **29 (16 / 13)** | Epoch `arc_agi_2_external.csv` | ARC Prize, verified runs | **3.8 pt (max 95%)** | 0.88 Knowledge | **Add as column, no index** — head saturating; watch ARC-AGI-3 |
| 9 | **Arena Agent / Agent tool-hallucination** | Live agent-session success; rate of tool calls the model hallucinates | **35 (19 / 16)** | same HF dataset | Independent, telemetry-based | — | **+0.52** OpenBench | **Add hallucination to Trust**, agent score as column |
| 10 | **HMMT Feb 2026** | Post-cutoff olympiad math | **19 (19 / 0)** | MathArena board + HF/llm-stats | Independent (MathArena) | 10.9 pt | 0.88 Knowledge (n=8) | **Add** (2nd time recommended) |
| 11 | **MMMLU** + Arena non-English | Multilingual knowledge (14 human-translated languages) | **24 (24 / 0)** | HF cards + llm-stats | Self-reported | 2.6 pt | 0.81 Knowledge | **Add, low weight** — no multilingual column exists |
| 12 | **Claw-Eval / WildClawBench** | Agentic tasks in the "claw" harness family | **24 / 18 (all open)** | HF eval-results + llm-stats | Self-reported, public boards | 17.7 / 8.6 pt | — | **Add Claw-Eval**, defer WildClaw |
| 13 | **NL2Repo** | Build a whole repository from a natural-language spec | **25 (25 / 0)** | llm-stats + HF | Self-reported | 6.0 pt | 0.87 Coding, **−0.22** Trust | Add, low weight → Coding |
| 14 | **MRCR v2 (8-needle)** | Long-context multi-round coreference | **14 (11 / 3)** | llm-stats + HF | Self-reported, lengths differ | **48.8 pt** | 0.84–0.89 | Defer (variant mixing) |
| 15 | **OmniDocBench 1.5** | Document parsing / OCR | **22 (22 / 0)** | HF cards | Self-reported, metric varies | 2.3 pt | — | Defer (metric direction mixed) |

The skips are in [§8](#8-skip-and-why).

---

## 2. Arena (human preference) — the biggest single gap

**What it tests.** Arena collects anonymous pairwise votes from users on their own prompts
and fits a Bradley-Terry rating, reported with 95% CIs and vote counts. The `text` board
splits into 29 categories. The ones that matter here are `overall`, `coding`,
`hard_prompts`, `instruction_following`, `multi_turn`, `creative_writing`, `math`,
`longer_query`, and per-language boards for `german`, `french`, `spanish`, `japanese` and
others. There are separate boards for `webdev`, `vision`, `search` and `document`, plus a
new `agent` family: `agent`, `agent_tool_hallucination`, `agent_bash_reliability` and
`agent_steerability`.

**Why it is worth having.** Every column in `llm.json` is graded by an automatic checker or
an LLM judge. None asks a person which answer they prefer. Arena is also the least
redundant candidate measured:

| Arena board | matched (open) | ρ OpenBench | ρ Coding | ρ Tooling | ρ Knowledge | ρ Trust |
|---|---|---|---|---|---|---|
| Text overall | 73 (58) | 0.65 | **0.63** | 0.79 | 0.81 | 0.75 |
| Text factuality | 62 (47) | 0.74 | 0.73 | 0.82 | 0.83 | 0.70 |
| WebDev | 56 (42) | 0.95 | 0.94 | 0.94 | 0.94 | 0.81 |
| Agent | 35 (19) | 0.89 | 0.90 | 0.78 | 0.91 | 0.59 |
| Agent tool-hallucination | 35 (19) | 0.52 | 0.51 | 0.54 | 0.62 | 0.31 |
| Vision | 29 (18) | 0.55 | 0.51 | 0.63 | 0.64 | 0.47 |

**Coverage.** These matches use plain name normalisation only. Rows such as `GLM-5.1`,
`DeepSeek-V3.2`, `Beam 1.0 501B A23B`, `NVIDIA-Nemotron-3-Ultra-550B-A55B` and
`MiniMax-M2.1` need a mapping file. The usual `update_*_mapping.py` flow would supply it and
should take Text to roughly 85 models. The `german` board alone rates 320 models, including
GLM-5.3, Kimi K3, DeepSeek-V4.1-Flash and Mistral Medium 3.5.

**Scraping.** `arena.ai` returns 403 to scripted requests, but the dataset does not:
`https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/<board>/latest-00000-of-00001.parquet`.
The last publish date was 2026-10-28. Reading parquet requires `pyarrow`, which is a new
dependency (`pandas` is not required). Columns: `model_name, organization, license,
rating, rating_lower, rating_upper, vote_count, rank, category, leaderboard_publish_date`.
The agent boards use `score`/`score_ci_*` in place of `rating`.

**Caveats to encode.**
- Arena lists one row per effort setting (`(Max)`, `(High)`). Use the same "best
  configuration" rule as `deepswe_1_1`.
- Low-vote rows have wide CIs; several open models have under 300 German votes. Use a
  minimum-votes gate, e.g. `vote_count ≥ 1000` for overall.
- Prefer the **style-controlled** board (`text_style_control`). The raw Elo rewards length
  and markdown.
- The score is an Elo, not a percentage. Store it with `range` and `decimals: 0`, as
  `gdpval_aa` already does.
- Lab pre-release testing (the 2025 private-variant controversy) means a released
  checkpoint and the tested row can differ. Map only names that match a released model.
- **`agent_tool_hallucination` is higher-is-better**: a higher score means fewer hallucinated tool calls.

**Placement.** Add a new category, "Human Preference", holding Text (style-controlled) and
WebDev. `agent_tool_hallucination` is a Trust-index candidate: it measures a failure mode
nothing else in Trust covers, and it is only weakly correlated with the rest
(ρ 0.31 against Trust). Start it at about 0.3 and recalibrate.

---

## 3. Epoch hub files we download and discard

`fetch_epoch.py` already pulls `benchmark_data.zip` and reads 12 of its files: GPQA,
SWE-bench Verified, DeepSWE, FrontierSWE, FrontierCode, Terminal-Bench, OSWorld,
OSWorld 2.0, HLE, CritPt, SciCode and ExploitBench. Adding a file is a `Benchmark(...)`
entry plus a column. Coverage over all ~85 files:

| File | models (open) | Epoch-run? | median / max | top-5 | Notes |
|---|---|---|---|---|---|
| `ale_bench_external` | **51 (38)** | external (Sakana) | 1047 / 2951 | 774 | Scored, not binary. Best open coverage of any new file |
| `lmca_external` | 51 (37) | external | 37 / 68 | 4.6 | **Skip**: ρ 0.95–0.98 against every index, so it adds nothing |
| `dtbench_external` | 51 (37) | external | 0.88 / 0.99 | 1.6 | **Skip**: decision-theory multiple choice, saturated, ρ 0.92–0.96 |
| `webdev_arena_external` | 45 (33) | Arena mirror | — | — | Use Arena directly (§2) |
| `proofbench_external` | 39 (25) | external | 0.54 / **1.00** | **1.0** | Lean 4 formal proofs, no judge, but the head is saturated |
| `mystery_game_puzzles` | 35 (20) | **Epoch** | 0.30 / 0.84 | 19 | Niche game puzzles, details hidden on purpose. Column at most |
| `apex_agents_external` | 33 (18) | external (Mercor) | 0.53 / 0.82 | 13.8 | See §4 |
| `frontiermath_tiers_1_3_v2` | 33 (18) | **Epoch** | 0.71 / 0.94 | 3.5 | Head compressing |
| `simpleqa_verified` | 32 (18) | **Epoch** | 0.43 / 0.76 | 5.2 | See §5 |
| `weirdml_external` (v2) | 31 (26) | external | 0.44 / 0.94 | 15.3 | Last new model 2026-09-03; v3 replaces it |
| `frontiermath_tier_4_v2` | 30 (15) | **Epoch** | 0.46 / 1.00 | **10** | Hardest math anywhere |
| `arc_agi_2_external` | 29 (16) | external (ARC Prize) | 0.61 / **0.95** | 3.8 | Head saturating; ARC-AGI-3 is live |
| `chess_puzzles` | 28 (15) | **Epoch** | 0.45 / 0.76 | 17 | Niche |
| `vending_bench_2_external` | 27 (13) | external (Andon) | $6.2k / $16.5k | wide | Dollar scale; log-transform it |
| `weirdml_v3_external` | 19 (5) | external | 0.15 / 0.42 | — | Mostly closed for now |
| `blueprint_bench_2`, `mirrorcode`, `cursorbench` | 16 / 14 / 13 | mixed | — | — | **Skip**: 2–3 open models each |
| `metr_time_horizons_external` | **3** | METR | — | — | **Skip**: important, but METR rarely runs open models |
| `fictionlivebench_external` | **4** | external | — | — | **Skip**: stale since January |

**Recommendation.** Add in this order: **FrontierMath Tier 4 v2, APEX-Agents, SimpleQA
Verified, ALE-Bench, Vending-Bench 2, ARC-AGI-2.** Optionally add FrontierMath T1-3 v2 as a
second track, as SWE Atlas has. All six already pass through the same CSV parsing,
`_fetch_checks` and Epoch credit as the files read today.

---

## 4. Agentic and professional work

- **APEX-Agents (Mercor).** Multi-hour tasks set by investment-banking, consulting and law
  professionals, run inside realistic file and app environments. It reports pass@1 plus a
  rubric mean. Union coverage is 46 (31 open). The top-5 gap is 13.8 points, so the head
  is far from saturated, and ρ against Tooling is 0.82, close to τ³-Banking's
  independence. It fits "Real-world Professional Work" beside GDPval-AA and AA-Briefcase,
  and belongs in Tooling at about 0.5. AA also runs `apex_agents` (the August audit
  counted 12). Rank Epoch's copy (Mercor's own board) above vendor cards, and keep AA's
  run in a separate column or drop it.
- **Vending-Bench 2 (Andon Labs).** A simulated year of running a vending business. It
  tests whether an agent stays coherent over very long horizons, which nothing else in
  Tooling measures: Terminal-Bench and τ tasks last minutes. Its relatively high ρ with
  Trust (0.77) is consistent with that. It covers 27 models, but only 13 open. Weight it
  at about 0.4 because the score is in dollars and noisy.
- **τ²-Bench Retail + Airline.** These are the τ² domains that still have room
  (`tau2_bench_telecom` has 25 of 112 models at ≥ 90). Coverage is 41 open models, all
  self-reported. This is the second time the recommendation has been made. Pool only
  retail and airline. Do not pool the τ² "average", which includes telecom.
- **Claw-Eval** (24 open, 9 vendors) and **WildClawBench** (18). These are agentic boards
  in the claw harness family that open-model cards have adopted quickly. Claw-Eval is the
  one ready to rank. WildClawBench also publishes time and cost per task, which no column
  here records.
- **Arena Agent** (35). Success on real user agent sessions. The score is a model rating
  rather than a task pass rate. Add it as a column and keep it out of any index until it
  has 6 months of history.

---

## 5. Knowledge, math and factuality

- **FrontierMath (Epoch).** Unpublished, expert-written problems with automatically
  checkable answers, run by Epoch itself. That makes it about as clean as provenance gets
  in this table. Tier 4 v2 has a top-5 gap of 10 points against AIME 2026's 2.9 with 27
  models ≥ 90. Put it in Knowledge at about 0.6 and lower `aime_2025` toward 0.2.
- **HMMT Feb 2026 (MathArena).** Independently run and post-cutoff. It covers 19 models,
  all open. MathArena's per-competition JSON is the clean channel; the August doc gives
  the endpoint. This would also open APEX 2025 and USAMO 2026 from the same source.
- **SimpleQA Verified (Epoch-run).** Short factual questions whose labels were cleaned
  after errors in the original SimpleQA; it rewards abstaining over guessing. It covers 32
  models (18 open). It correlates only 0.45 with the Trust index, which suggests it
  measures recall honesty differently from AA-Omniscience. It fits Trust at about 0.4.
  **Do not pool it with the 22 original-SimpleQA self-reports from model cards.** Those
  are a different, noisier question set.
- **ARC-AGI-2.** Few-shot abstraction on novel visual grid puzzles, built to resist
  memorisation. It is the only test of its kind available. The best model reaches 95% and
  the top-5 gap is 3.8 points, so add it as a column outside any index. ARC Prize has
  launched **ARC-AGI-3**, which uses interactive games and where open models score near 0.
  Only 4 of our models have ARC-AGI-3 numbers on llm-stats so far, but it should replace
  ARC-AGI-2 once Epoch or ARC Prize publish more.
- **ProofBench.** Formal Lean 4 proofs, so a machine checks the answer and no judge is
  involved. It covers 39 models (25 open), but the head is at 100%. Add it as a column at
  most.

---

## 6. Multilingual — currently zero columns

No column in `llm.json` measures anything except English and code. Candidates:

| Benchmark | union | What it is | Issue |
|---|---|---|---|
| **MMMLU** | 24 open | OpenAI's human translation of MMLU into 14 languages | Saturated head (top-5 gap 2.6) |
| WMT24++ | 18 open | Machine translation, 55 languages | Only 4 vendors; metric (xCOMET vs chrF) varies by card |
| INCLUDE / Global-MMLU / MMLU-ProX | 32 combined | Regional-knowledge MCQ | Three different tests; Qwen template-heavy |
| **Arena `german` / `non_english`** | ~70 | Human preference by language | See §2 |

**Recommendation.** Add MMMLU as a low-weight column, and add an Arena `non_english` board
(or `german`, given this table's audience) next to the Arena overall column. Together they
cover the gap with one self-reported knowledge test and one independent preference
signal.

---

## 7. Long context, vision and coding extras

- **Long context.** `aa_lcr` is the only long-context column, and its top-5 gap is 3.4
  points. **MRCR v2 8-needle** separates models far better (48.8-point top-5 gap) but
  covers only 14 models, and cards report it at 128k, 256k or 1M. It needs a pinned
  length before it can be a column. LongBench v2 (17 open) is head-compressed. RULER,
  HELMET and Graphwalks each cover 3 models or fewer. Defer all of them.
- **Vision.** The vision categories with no column yet are document parsing
  (**OmniDocBench 1.5**, 22 open), video (**VideoMME / VideoMMMU**, 16) and GUI grounding
  (**ScreenSpot-Pro**, 11). OmniDocBench is the furthest along, but cards mix the overall
  score with "edit distance, lower is better". It needs exclusion rules first.
- **Coding.** **ALE-Bench** (§3) is the best addition: optimisation problems scored on a
  rating, so it does not saturate the way pass/fail suites do. Start it at about 0.4 in
  Coding. **NL2Repo** (25 open) is the only test of building a repository from scratch;
  add it at about 0.3. SWE-rebench is already fetchable (`fetch_swe_rebench.py`; 39 of
  our models are mapped). It was unwired deliberately, because each model is scored on its
  own monthly task window, and nothing has changed that.

---

## 8. Skip, and why

| Benchmark | Reason |
|---|---|
| LMCA, DTBench | 51 models, but ρ 0.92–0.98 against every index, so they are redundant; DTBench is also saturated |
| IFEval, MMLU, MMLU-Redux, MATH-500, GSM8K, HumanEval, MBPP, BBH | Saturated legacy tests (the August Tier B list) |
| Arena-Hard v2, MultiChallenge, WritingBench | Use Arena's human votes instead of these LLM-judged proxies |
| HealthBench (all variants) | 15 models, 11 of them closed |
| METR time horizons | 3 models; METR rarely evaluates open models |
| Fiction.LiveBench | 4 models, nothing new since January |
| CursorBench, Blueprint-Bench 2, MirrorCode | 2–3 open models each |
| HLE with tools | 47 models, but harnesses vary widely ([HLE audit](hle-tool-mode-audit-2026-09.md)) |
| LEXam (17), yc-bench (10), Finance Agent v2 (10), PostTrainBench (9), Terminal-Bench Science (15, 7 open) | Too thin or too narrow for now. Revisit in Q1 |
| **Safety / refusal / jailbreak** (HarmBench, XSTest, StrongREJECT, AgentHarm…) | **No channel has more than 3 of our models.** This is a real gap with no source yet. AA does not publish one, and labs do not put these on cards |

---

## 9. Suggested order of work

1. **Epoch files** (§3): FrontierMath T4 v2, APEX-Agents, SimpleQA Verified, ALE-Bench,
   Vending-Bench 2, ARC-AGI-2. Each is a `Benchmark` entry, a column, and a test case in
   `test_epoch.py`. This is the most coverage for the least work.
2. **Arena fetcher** (§2): a new `fetch_arena.py` reading parquet, a mapping updater, and
   a new "Human Preference" category with Text (style-controlled), WebDev and a
   `non_english` board. Then put `agent_tool_hallucination` into Trust.
3. **Model-card / llm-stats labels** (§4–6): un-park τ²-Bench Retail/Airline, HMMT Feb 2026,
   MMMLU, Claw-Eval and NL2Repo in `huggingface-benchmark-name-mapping.json` and
   `llmstats-benchmark-name-mapping.json`, and add their columns.
4. **Recalibrate**: run `derive_indexes.py --calibrate` with the new members at the
   starting weights given above, and let the nightly workflow refit them.

---

## Appendix A — method

- **llm-stats:** for each of the 130 llm-stats ids mapped to a model in `llm.json`, read
  `api.zeroeval.com/leaderboard/models/<id>`, which lists every benchmark for that model
  (2 fetches failed). Counts are models / distinct organisations.
- **Hugging Face:** a fresh `fetch_huggingface.py --all-models --format json` run read 167
  cards. For the summary, labels were folded with a case- and punctuation-insensitive
  regex per benchmark. For the Tier list, only labels currently mapped to
  `__unmappable__` or absent from the mapping were counted.
- **Epoch:** `epoch.ai/data/benchmark_data.zip`, model versions joined to model groups
  through `model_metadata.csv` and then through
  `model-name-mapping-epoch-to-artificialanalysis.json`. The score is the file's
  `score_column` according to `benchmark_metadata.csv`.
- **Arena:** the `latest` parquet of each board in `lmarena-ai/leaderboard-dataset`,
  matched by lower-case, punctuation-folded name with effort suffixes stripped. No mapping
  file exists yet, so these counts are lower bounds.
- **Spearman ρ:** computed on the models that have both the candidate and the index value.
  It is shown only when at least 6 models overlap; the overlap count is in parentheses in
  the working data.
- **Not measured:** Artificial Analysis fields that are fetched but not stored
  (`apex_agents`, `harvey_lab_criteria_pass`, `enterprise_ops_gym`, the math index). No
  AA API key was available in this session. The August counts in
  `benchmark-candidates-2026-08.md` §6 are the latest available for these.
