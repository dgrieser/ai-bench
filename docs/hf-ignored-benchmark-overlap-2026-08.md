# What we throw away from Hugging Face model cards — August 2026

`huggingface-benchmark-name-mapping.json` holds 923 reviewed labels. **734 of them are
`__unmappable__`**, i.e. every ingest reads them and drops them. This is an audit of that
discard pile: which of those labels describe a benchmark that enough of our models actually
report to be usable, what each one measures, and whether the numbers still separate models.

Snapshot 2026-08-23. Method and caveats in [Appendix A](#appendix-a--method).
Companion to [`benchmark-candidates-2026-08.md`](benchmark-candidates-2026-08.md), which
asked the same questions of *candidate sources*; this one asks them of *labels we already
fetch and bin*.

---

## 0. Headline

- Fresh run of `fetch_huggingface.py --all-models`: **125 of 136** HF-hosted models parsed
  (11 gated repos return 401), **811 distinct labels**, up from 629 labels / 112 models on
  2026-08-01.
- After folding alias spellings together, **17 discarded benchmarks clear a real
  multi-vendor overlap bar**; a further 11 have large row counts but are **saturated legacy**
  tests, and roughly 30 more have row counts of 8–12 that come from **a single vendor's card
  template** replicated across sibling model sizes.
- The single biggest miss is **MMLU-Pro**: 60 of 125 models, 23 distinct vendors. It was
  already recommended ("add, low weight") on 2026-08-01 and is still not a column.
- The most interesting *new* find is the **Claw family** — `Claw-Eval` (19 models / 8
  vendors) and `WildClawBench` (14 / 8, with a public leaderboard, an arXiv paper, released
  trajectories, and per-task **time and cost** figures we measure nowhere).
- **τ²-Bench outside Telecom** — aggregate/airline/retail together cover **35 models / 14
  vendors** and are *not* at the ceiling, while our own `tau2_bench_telecom` column has
  **23 of 101 models ≥ 90**.

---

## 1. The counting rule: vendors, not rows

HF card numbers are self-reports, and a lab publishes one benchmark table across its whole
release family. Qwen3.5 ships 8 sibling sizes from one template, so any label in that
template instantly "covers" 8 models without a second opinion existing anywhere. Row count
alone therefore overstates overlap badly:

| label | models | vendors | what the row count really is |
|---|---|---|---|
| `PolyMATH` | 12 | **1** | one Qwen table |
| `MMLU-ProX` | 14 | **2** | Qwen + nvidia |
| `MedXpertQA-MM` | 13 | **2** | Qwen + google |
| `MMLU-Pro` | 60 | **23** | genuine cross-lab convention |

Every count below is therefore reported as **models / vendors**, where vendor = the HF org
owning the repo. A benchmark needs both to be rankable: vendors for independence, models for
resolution.

For calibration, `llm.json` already ships columns at `mcp_atlas` 17, `toolathlon` 21,
`bfcl_v4` 24, `swe_marathon` 6. A 14-model / 8-vendor HF benchmark is not thin by the
standards of this table — it is mid-pack.

---

## 2. Tier A — discarded labels with real overlap

Saturation columns: `med` / `max` are over the best value per model; `≥90` counts models at
or above 90; `top-5 spread` is max minus 5th-best — the repo's usual "can this thing still
rank the head of the field" test.

| Benchmark | models / vendors | med | max | ≥90 | top-5 spread | Trust | Saturated? |
|---|---|---|---|---|---|---|---|
| **MMLU-Pro** | **60 / 23** | 81.8 | 87.8 | 0 | **1.1** | Med | head yes, tail no |
| **τ²-Bench (agg + airline + retail)** | **35 / 14** | 70.4 | 88.2 | 0 | 7.0 | Med-Low | no |
| **SWE-bench Multilingual** | **24 / 11** | 69.3 | 79.6 | 0 | 3.1 | Med | no |
| **Claw-Eval** | **19 / 8** | 65.8 | 81.4 | 0 | 8.9 | Med | no |
| SuperGPQA | 17 / 4 | 60.8 | 70.4 | 0 | 5.5 | Med | no |
| **WildClawBench** | **14 / 8** | 43.5 | 56.2 | 0 | 8.6 | **Med-High** | no |
| **HMMT Feb 2026** | **14 / 7** | 85.6 | 92.7 | 3 | 5.6 | **High** | starting |
| IFStruct (LiquidAI) | 13 / 8 | 79.0 | 95.9 | 3 | 10.4 | **High** (method) | by design at frontier |
| MultiChallenge | 13 / 5 | 52.2 | 67.6 | 0 | **12.4** | Med | no |
| BFCL v3 | 13 / 7 | 68.3 | 74.4 | 0 | 2.5 | Low-Med | superseded by v4 (a column already) |
| LongBench v2 | 12 / 5 | 59.6 | 63.2 | 0 | 2.6 | Med | head compressed |
| OJBench | 12 / 3 | 37.8 | 60.6 | 0 | **20.5** | Med | no |
| NL2Repo | 11 / 4 | 42.3 | 59.5 | 0 | 13.3 | Low-Med | no |
| IMO-AnswerBench | 11 / 6 | 81.8 | 91.0 | 1 | 9.2 | Med | starting |
| WideSearch | 10 / 3 | 64.5 | 80.8 | 0 | 13.0 | Med | no |
| BrowseComp-ZH | 9 / 6 | 66.6 | 73.7 | 0 | 7.1 | Med | no |
| yc-bench | 9 / 6 | — (funds, $) | — | — | — | **High** (deterministic sim) | no |
| ParseBench (llamaindex) | 9 / 3 | 40.5 | 62.4 | 0 | 21.9 | Med-High | no |
| LEXam | 8 / 4 | 46.6 | 56.5 | 0 | 11.2 | Med-High | no |
| SkillsBench | 7 / 5 | 48.2 | 54.0 | 0 | 9.7 | Med-High | no |
| ResearchClawBench | 6 / 5 | 17.0 | 20.7 | 0 | 6.5 | Med | floor-ish |
| Long-Horizon Terminal-Bench | 6 / 4 | 3.0 | 38.5 | 0 | 37.5 | Med-High | **floor-compressed** |

Notes on each, in that order.

### MMLU-Pro — 60 / 23
Twelve-thousand reasoning-heavy questions with ten answer options each, rebuilt by TIGER-Lab
from MMLU across 14 subjects to undo MMLU's saturation and its noisy-answer problem. The most
widely reported benchmark we discard, by a factor of two, and the only one here that is a
cross-lab *convention* rather than a cluster of habits.

*Trust:* medium. 42 of the 60 arrive through structured Hub eval metadata under the dataset id
`TIGER-Lab/MMLU-Pro`, 45 through README tables, and on the 27 models carrying both the two
channels agree **exactly in 25**. The exception is loud: `deepseek-v4-pro` reports **87.5**
in metadata against **73.5** in its own table (`qwen3-5-0-8b` differs by 4.9). Against that,
the AA API serves `mmlu_pro` for 37 models — an independently-run channel to pool or prefer,
with only ~6 models of overlap, as the August 1 doc found.

*Saturation:* split. The head is done — top-5 spread **1.1 point**, nothing above 87.8 — but
the floor reaches 29.7, so it still resolves the small-model tail, which is where our coding
and agentic columns all collapse. Also carries MMLU's inherited hazard: public since 2024,
so contamination exposure is total, and the label-error critiques of the dataset are real.

**Verdict: add, low weight, pooled with the AA API and preferring the API value.** Same
conclusion as three weeks ago; the coverage has only grown.

### τ²-Bench outside Telecom — 35 / 14
Sierra's dual-control agent benchmark: the model works a customer-service task against a
simulated user who also holds tools, so it tests coordination, not just tool syntax. We track
only `tau2_bench_telecom`. The discard pile holds the cross-domain aggregate (21 / 8), Airline
(14 / 7), Retail (13 / 6) and a few τ¹ leftovers (5 / 2), under **28 different spellings** —
`TAU2-Bench`, `\(\tau^2\)-Bench`, `TauBench V2 (Average)`, `Tau2 (average over 3)`, bare
`Airline`/`Retail` column heads, and so on.

*Trust:* medium-low, and this is the reason they were parked rather than an oversight. An
LLM-driven user simulator plus judged outcomes, self-reported at whatever avg@k the lab chose;
the bare `Airline`/`Retail` heads are not even reliably τ². Mapping any of it onto the telecom
column would be wrong — different domains, different difficulty.

*Saturation:* the interesting part. Airline median **58.0**, Retail **66.7**, aggregate 76.9,
**nothing ≥ 90** anywhere, top-5 spread 7.0 — against our own telecom column at 23 of 101
models ≥ 90 and the ceiling pileup that made AA replace τ²-Telecom with τ³-Banking in
Intelligence Index v4.1.

**Verdict: worth one new column** (`tau2_bench_retail` or an explicit "τ² average over
domains"), *not* a merge into telecom. It measures the same capability our most saturated
agentic column measures, with headroom left.

### SWE-bench Multilingual — 24 / 11
300 real GitHub PR-resolution tasks in nine non-Python languages; the official board runs a
standardized mini-SWE-agent. Recommended at w≈0.30 on 2026-08-01 with 17 HF models; now **24
models across 11 vendors**, so the trend the earlier doc wanted to watch is up.

*Trust:* medium. All 13 models present in both channels agree to ≤0.02, so the numbers are at
least internally consistent — but they are still lab-run at each lab's harness of choice, and
the official leaderboard's open-weight section remains tiny. Pooling official runs with
self-reports in one percentile-normalized column is the hazard, not the count.

*Saturation:* none. Median 69.3, **no model at 80**, max 79.6 — though the top-5 spread is
only 3.1, so it separates the middle better than the head.

**Verdict: the earlier w=0.30 recommendation is now better supported. Official-only if you
can stand 3–4 rows; otherwise pool and flag provenance.**

### Claw-Eval — 19 / 8
`claw-eval/Claw-Eval` (MIT): 300 agent tasks in three splits — 161 `general`, 101
`multimodal`, 38 `multi_turn` — English and Chinese, fixture-based, aimed at the
OpenClaw-style CLI-agent ecosystem. Reported by 8 vendors including Qwen, DeepSeek, Moonshot,
MiniMax, LiquidAI, nvidia and ornith-ai — unusually broad for a benchmark this young.

*Trust:* medium. 16 of 19 come through structured metadata, but the splits are reported
inconsistently (`general` alone, `Claw-Eval Avg`, `Claw Eval (pass^3)`, `Claw-Eval average
(EN)` — 10 spellings), and `kimi-k2-6` shows **65.8 in metadata against 80.9 in its README**,
a 15-point gap. Pick one split (`general` is the one 15 vendors-worth of models actually
report) and ignore the rest.

*Saturation:* none. Median 65.8, max 81.4, top-5 spread 8.9.

### SuperGPQA — 17 / 4
26k graduate-level questions spanning 285 disciplines, built to keep resolving where GPQA
Diamond flattens. Median 60.8, max 70.4, no model near the ceiling — cleanly unsaturated, and
`gpqa_diamond` (our 132-model column) already has 9 models ≥ 90.

*Trust:* medium — self-reported, and the four vendors are ByteDance, Qwen, Xiaomi and
DeepSeek, i.e. one regional cluster. Good complement to GPQA-D on paper; too vendor-narrow to
rank on today.

### WildClawBench — 14 / 8 — the strongest new candidate
`internlm/WildClawBench`: 60 wild agent tasks run across **4 harnesses**, with a public
leaderboard (31 models), an arXiv report, a Harbor-format task package and released
trajectories. 14 of the 14 values reach us through structured eval metadata rather than
README tables, and the vendor spread (Qwen, DeepSeek, Moonshot, MiniMax, Xiaomi, StepFun,
zai, meta-models) is the widest of any benchmark this young.

*Trust:* medium-high — an independent leaderboard exists to check self-reports against, which
is exactly what the other Claw-family entries lack. Small n (60 tasks) is the main discount.

*Saturation:* none — median 43.5, max 56.2, top-5 spread 8.6.

It also publishes `avg_time` (13 models) and `avg_cost` (10 models) under the same dataset id.
**`llm.json` measures neither latency nor cost anywhere**; those two fields are currently
discarded along with the accuracy.

### HMMT Feb 2026 — 14 / 7
MathArena's run of the February 2026 HMMT exam. 13 of 14 values arrive as
`MathArena/hmmt_feb_2026` structured metadata, i.e. **independently run and post-cutoff** —
the highest-provenance benchmark in this entire discard pile, and the one the earlier doc
already recommended building `fetch_matharena.py` for.

*Saturation:* starting. Median 85.6, 3 of 14 ≥ 90, min 71.2 — one more model generation and
it is `aime_2026` (19 of 28 ≥ 90). The 2025 exams are further gone: **Feb 2025 is 27 models /
11 vendors** — the second-broadest thing we discard — but median 83.9 with 9 of 27 ≥ 90 and a
98.4 max; Nov 2025 (15 / 5) is at median 89.2. Take Feb 2026, leave 2025 in the bin.

### IFStruct — 13 / 8
LiquidAI's structured-output compliance test: 2,000 prompts, each asking for JSON/YAML against
a sampled schema plus presentation constraints, scored **binary by a deterministic validator
with no constrained decoding and no LLM judge**. Method trust is the highest here after
MathArena — there is nothing to grade subjectively.

*Saturation:* by design. The dataset card states difficulty is calibrated "to be
discriminative for low-to-mid-ability models and to saturate near 100% at the frontier", and
our sample already shows 3 of 13 ≥ 90 with a 95.9 max. It is a tail instrument. Note we
already carry `ifbench` (112 models) whose AA original was retired from Intelligence Index
v4.1 for exactly this failure mode.

### MultiChallenge — 13 / 5
Scale AI's multi-turn instruction-following suite (instruction retention, self-coherence,
inference memory over a conversation). Median 52.2, max 67.6, **top-5 spread 12.4** — the best
discrimination of anything in the instruction-following family, against `ifbench` at median
44.2 and IFEval at median 88.4. Judge-based, so medium trust, and five vendors is thin.

### BFCL v3 — 13 / 7
The Berkeley function-calling leaderboard, previous generation. We already ship `bfcl_v4`
(24 models) from the official CSV; v3 and v4 are not comparable, and the earlier doc
documented why the official V4 board and the HF self-reports are disjoint populations.
**Correctly ignored** — the useful move is not a v3 column but noticing that 13 models
self-report a BFCL generation the official board does not cover.

### LongBench v2 — 12 / 5
503 long-context multiple-choice questions, 8k–2M words. Median 59.6 but max 63.2 and top-5
spread **2.6** — the head is a dead heat, partly because the multiple-choice format floors at
25%. We now have `aa_lcr` (106 models, AA-run) covering the dimension far better; this stays
in the bin.

### OJBench — 12 / 3
Competitive-programming judge problems (NOI/ICPC-level, execution-verified). Median 37.8,
max 60.6, **top-5 spread 20.5** — the widest head spread of any coding label we discard, i.e.
genuinely unsaturated. Three vendors (Qwen, Moonshot, LongCat) and heavy capability overlap
with `livecodebench` are the reasons to hold.

### NL2Repo — 11 / 4
Repo-level code generation from a natural-language spec — build a whole project, not a patch.
Four vendors (Qwen, DeepSeek, ornith-ai, zai-org), median 42.3, top-5 spread 13.3, nothing
near saturation. Attractive on capability: our coding group is patch-shaped and this is not.
*But* no canonical public leaderboard turned up under that name, so every number is a
self-report against the reporting lab's own harness — low-medium trust, and the label appears
both bare and as "Repo-level code generation NL2Repo-Bench".

### IMO-AnswerBench — 11 / 6
400 olympiad problems with checkable short answers (DeepMind). Median 81.8, max 91.0, one
model ≥ 90 — the "skip / defer" call from three weeks ago still holds, and it is closing the
same way `aime_2026` did.

### WideSearch (10 / 3) and BrowseComp-ZH (9 / 6)
Broad-coverage information-gathering (WideSearch: assemble a complete table from the open web,
item-F1 graded) and the Chinese-web BrowseComp variant. We ship `browsecomp` (43 models), so
BrowseComp-ZH is a locale complement across six vendors rather than a new capability; both
are unsaturated (medians 64.5 / 66.6). WideSearch's three vendors are the blocker.

### yc-bench — 9 / 6
Collinear AI's long-horizon coherence sim: the model plays CEO of an AI startup for a
simulated year over hundreds of CLI turns against a **deterministic discrete-event
simulation**, scored on average final funds over three seeds. High method trust — no judge,
no grader, reproducible — and all nine values arrive as structured metadata.

The catch is the unit: **dollars, 0 to 2,100,000**, not a percentage. It cannot share a
formatter with anything else in `llm.json`, and one bankrupt model at 0 anchors any linear
scale. Percentile normalization in `computeGroupComposite()` would cope; the display column
would need its own formatting rule.

### ParseBench (9 / 3), LEXam (8 / 4), SkillsBench (7 / 5), ResearchClawBench (6 / 5), LHTB (6 / 4)
The structured-metadata tail — all reach us as dataset ids, so all are checkable, and each is
at or slightly below the coverage of columns we already ship:

- **ParseBench** (llamaindex): PDF→markdown fidelity over five splits (table, chart, layout,
  text content, text formatting) against expected markdown. Median 40.5, spread 21.9 —
  unsaturated. Document parsing is outside what this table measures today.
- **LEXam**: Swiss law-exam questions, MCQ and open-question splits, 8 models / 4 vendors
  including microsoft and openai. Median 46.6. A legal-domain column beside
  `harvey_lab_criteria_pass` (which we already skip at 13 models).
- **SkillsBench** (benchflow, arXiv 2602.12670): does giving an agent *skills* actually help —
  87 active task packages, audited leaderboard archive. Median 48.2, 5 vendors. Watch.
- **ResearchClawBench** (InternScience): deep-research agent tasks; median 17.0, max 20.7 —
  near the floor for open weights, same failure mode as `critpt`.
- **Long-Horizon Terminal-Bench**: long-horizon terminal work. Median **3.0** with one
  outlier at 38.5 — floor-compressed to uselessness on this population, exactly the reason
  `critpt` was skipped.

---

## 3. Tier B — big overlap, dead signal

These have the row counts to qualify and are still correctly discarded, because they no longer
rank anything. Ordered by models / vendors:

| Benchmark | models / vendors | med | max | ≥90 | top-5 spread | Why it stays out |
|---|---|---|---|---|---|---|
| IFEval | 29 / 9 | 88.4 | 95.0 | 6 | 3.5 | 2023 instruction-following; head saturated. `ifbench` already covers this and AA retired *its* version for the same reason. |
| MMMLU (multilingual MMLU) | 26 / 9 | 76.6 | 90.3 | 1 | 4.0 | Translated MMLU; inherits MMLU's contamination wholesale. |
| MMLU-Redux | 24 / 7 | 91.8 | 94.9 | **16 of 24** | 1.4 | The de-noised MMLU relabel — the single most saturated thing in the pile. |
| MMLU (original) | 21 / 11 | 84.8 | 90.6 | 2 | 2.9 | Broadest vendor spread of anything discarded, and completely spent. |
| BBH | 16 / 6 | 82.1 | 89.8 | 0 | 3.2 | BIG-Bench Hard; the "hard" subset stopped being hard. |
| AIME 2024 | 15 / 8 | 71.9 | 96.7 | 3 | 10.7 | Superseded by our own `aime_2025` / `aime_2026`; contaminated. |
| C-Eval | 15 / 4 | 90.2 | 93.1 | **9 of 15** | 1.7 | Chinese MMLU analogue, saturated. |
| HumanEval(+) | 14 / 8 | 84.0 | 92.1 | 2 | 3.7 | 164 toy functions. Dead for frontier ranking. |
| HMMT Feb 2025 | 27 / 11 | 83.9 | 98.4 | 9 | 3.7 | See HMMT above — take Feb 2026 instead. |
| MATH-500 | 12 / 7 | 96.6 | 99.2 | **9 of 12** | 2.4 | AA also fetches `math_500` and we skip that too, consistently. |
| MBPP(+) | 11 / 5 | 71.4 | 92.7 | 1 | 18.6 | Same era as HumanEval; the spread is scale-mixing, not signal. |
| GSM8K | 8 / 4 | 92.4 | 99.6 | 5 of 8 | 7.3 | Grade-school math. Reported now only by small-model cards. |

`SimpleQA` (12 / 9) belongs here too but for a different reason — see the hazards below.

---

## 4. Tier C — the overlap that isn't: single-vendor clusters

About 30 labels sit at 5–12 models with **one or two vendors**, and every one of them is one
lab's evaluation table copied across its own release family. Adding any would rank Qwen
against Qwen:

| vendors | labels (models each) |
|---|---|
| **Qwen only** | `PolyMATH` (12), `RealWorldQA` (11), `MMBench EN v1.1` (10), `MVBench` (10), `MLVU` (10), `ERQA` (10), `RefCOCO` (10), `EmbSpatialBench` (10), `RefSpatialBench` (10), `CC-OCR` (10), `HallusionBench` (9), `DynaMath` (9), `CountBench` (9), `MMLongBench-Doc` (8), `AndroidWorld` (8), `Global PIQA` (8), `NOVA-63` (8), `MAXIFE` (8), `Hypersim` (8), `Nuscene` (8), `SLAKE` (8), `PMC-VQA` (8), `VlmsAreBlind` (8), `DeepPlanning` (7, and the dataset is `Qwen/DeepPlanning`), `ODInW13` (7), `LingoQA` (6), `SUNRGBD` (6), `TIR-Bench` (5) |
| Qwen + one other | `MMLU-ProX` (14), `MedXpertQA-MM` (13), `VideoMME` (12), `VideoMMMU` (11), `SimpleVQA` (11), `ZeroBench` (11), `MMStar` (10), `OCRBench` (10), `AI2D` (10), `LVBench` (10), `MMVU` (10), `ScreenSpot-Pro` (9), `BabyVision` (9), `V*` (8), `VitaBench` (9 — and the benchmark is `meituan-longcat/VitaBench`, i.e. one of the two reporters authored it) |
| 3–4 vendors, one region | `MathVision` (16 / 4), `CharXiv` (16 / 4), `OmniDocBench 1.5` (18 / 4), `INCLUDE` (16 / 3), `MathVista` (12 / 3), `WMT24++` (12 / 3) |

Two structural notes. First, this whole block is **multimodal/vision**, where `llm.json`
carries exactly one column (`mmmu_pro`, 45 models) — so the gap is real, but the evidence for
filling it is one vendor deep. Second, a benchmark reported mainly by the lab that *wrote* it
(`VitaBench`, `DeepPlanning`) is a self-report in the strongest sense.

---

## 5. Hazards to encode, not columns to add

Four things in the discard pile are not "benchmarks we're missing" but traps that would have
corrupted a column had the labels been mapped naively. Worth keeping mapped to
`__unmappable__` deliberately, and worth a check before anything here is promoted.

**Scale mixing (0–1 vs 0–100).** Mistral's cards report fractions where everyone else reports
percent: `ministral-3-14b` has `Multilingual MMLU` **0.742**, `MMLU 5-shot` **0.794**,
`MMLU Redux 5-shot` **0.82**. Any pooled MMLU-family column that does not normalize units
would rank three Ministral models below every 2B model in the table. This is the source of the
`min = 0.7` in the MMLU-family distributions.

**One name, two metrics.**
- `OmniDocBench 1.5`: Qwen/Moonshot report accuracy 61.0–91.1, while gemma-4 cards report
  "average edit distance, **lower is better**" 0.131–0.319. Same label, inverted metric,
  incompatible scale.
- `SimpleQA`: ten models sit at 3.0–20.6 (plausible accuracy), then `olmo-3-7b-instruct`
  reports 74.2 and `deepseek-v3-2-0925` reports 97.1 — those are a different quantity
  (attempted-rate or F1), not accuracy. A pooled column would put a small Olmo above every
  frontier model.
- `APEX`: `mercor/apex-agents` runs 3.1–41.0 while `a-x-k2` reports `Apex` 45.8 *and*
  `Apex-shortlist` **88.6** — the shortlist is a different population.
- `Codeforces`: 633–2150. Elo, not a percentage — the blocker the earlier doc already named.

**Non-score fields in the same namespace.** `internlm/WildClawBench (avg_time)` (13 models)
and `(avg_cost)` (10) are latency and money; `yc-bench` is dollars; `# Total Params`,
`H200(GPUs)`, `Batch Size (per GPU)`, `Download Link`, `Base Model`, `Release Note` are card
furniture the table parser picks up. All correctly binned, and the params/GPU ones are why the
`__unmappable__` list is 734 long rather than ~400.

**The two channels can disagree.** `extract_scores()` merges structured Hub eval metadata
(dataset-id labels) with README tables, metadata winning via `setdefault`. On the 27 models
reporting MMLU-Pro both ways, 25 agree exactly — but `deepseek-v4-pro` differs by **14.0
points** (87.5 vs 73.5) and `kimi-k2-6`'s Claw-Eval differs by **15.1** (65.8 vs 80.9).
Metadata-wins is the right default (it is the channel a benchmark owner can also write to via
a Hub PR), but a 14-point disagreement inside one model card is worth a warning rather than a
silent pick.

---

## 6. If you act on one thing

Ranked by value per unit of work, and consistent with the August 1 recommendations rather than
re-deciding them:

1. **`mmlu_pro`** — 60 HF models / 23 vendors here, plus 37 from the AA API for ~75 union.
   One `SCORE_MAPPINGS` line for the API half, and the HF half needs only a mapping-file edit
   (six alias spellings → `mmlu_pro`) because `update_huggingface_scores()` already writes
   into nulls without overwriting. Low weight; it is a tail instrument.
2. **`wildclawbench`** — 14 / 8, unsaturated, independent leaderboard to validate against,
   and its `avg_time` / `avg_cost` fields open a dimension the table does not have.
3. **τ² beyond Telecom** — one column (Retail, or the explicit 3-domain average), *never*
   merged into `tau2_bench_telecom`. Our most-saturated agentic column has an unsaturated
   sibling sitting in the bin at 35 models.
4. **`swe_bench_multilingual` @ w=0.30** — the call was already made; coverage has gone 17 → 24
   and 11 vendors in three weeks.
5. **`claw_eval` (general split only)** — 19 / 8 and unsaturated, but resolve the 15-point
   channel disagreement and pick one split first.

Explicitly **not** worth it despite qualifying on rows: IFEval, MMLU-Redux, MMMLU, MMLU,
C-Eval, MATH-500, HMMT 2025, AIME 2024, GSM8K, HumanEval, MBPP, BBH (Tier B — saturated), the
whole Tier C vision cluster (single-vendor), BFCL v3 (we ship v4), and LongBench v2 (`aa_lcr`
covers it better).

---

## Appendix A — method

```bash
./fetch_huggingface.py --all-models --format json > hf_all.json
```

125 of 136 HF-hosted models parsed; 11 fail with HTTP 401 (gated: gemma-3-*, llama-4-*,
jamba-1-7-*, dbrx, glm-5-3), so every count here is a **lower bound**. 811 distinct labels;
734 of the 923 labels in `huggingface-benchmark-name-mapping.json` are `__unmappable__`.

Labels were folded into benchmarks by hand-written pattern per benchmark, not by fuzzy
matching — τ² alone needed 28 spellings and HMMT 11, and normalization heuristics silently
merged AIME 24 with AIME 25 in a first pass. For each group: distinct `llm.json` models,
distinct HF org owners ("vendors"), and the distribution of the **best** value per model
(a model reporting `MathVision` and `MathVision (w/ python)` counts once, at the higher).

Saturation is judged the way `benchmark-candidates-2026-08.md` §2 judges it: count at ≥90 and
the max-minus-5th-best spread, not the median alone. Trust ratings are the same five-part
scale that document's §4 decodes from the coding weight ladder — contamination resistance ×
unsaturation × harness control × task realism × non-redundancy — plus one HF-specific axis:
whether the value arrives as **structured Hub eval metadata** (a dataset id, writable by the
benchmark owner through a Hub PR, and checkable against a public leaderboard) or as a **README
table cell** (the lab's own claim, harness unstated). That split is reported per candidate
where it matters.

Descriptions of the newer benchmarks (Claw-Eval, WildClawBench, IFStruct, yc-bench,
ParseBench, SkillsBench, LHTB) come from their Hugging Face dataset cards, read directly.
`NL2Repo`, `MAXIFE`, `NOVA-63` and `BabyVision` have no canonical dataset or leaderboard that
turned up under those names; they are described from the reporting cards alone and rated
accordingly.

**Not measured here.** Whether the AA API and HF self-reports agree on MMLU-Pro for the ~6
models that appear in both (the August 1 doc flagged the same gap); whether WildClawBench's
public leaderboard agrees with the 14 self-reported values; and the 11 gated repos, which need
`HF_TOKEN` in the environment to count at all.
