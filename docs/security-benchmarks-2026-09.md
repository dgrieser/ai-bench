# Security benchmarks — CyberGym, ExploitGym, ExploitBench, SEC-bench Pro, September 2026

Follow-up to [`cybergym-coverage-2026-08.md`](cybergym-coverage-2026-08.md), which deferred
CyberGym on comparability. Since then, four security boards have started appearing side by
side on open-weight model cards (GLM-5.3, DeepSeek-V4.1-Flash, MiMo-V2.6 Pro/Flash). All four
labels are currently pinned `__unmappable__` in `huggingface-benchmark-name-mapping.json`.
Coverage below is measured against the **166 models in `llm.json`** (156 open + 10 closed
reference rows). Snapshot: 2026-09-25.

**Verdict.**

- **Add ExploitGym and CyberGym as columns now.** Both come from one new fetcher reading the
  cybergym.io static JSON, with llm-stats and HF model cards as fill-only sources.
- **Add SEC-bench Pro as a column pinned to the 260505 snapshot.** That is the snapshot the
  vendors report on.
- **Skip ExploitBench.** It has 41 tasks, the top is already claimed at 100%, and the official
  board is stale.
- **None of the four belongs in the Coding or Tooling index.**
- **A Security index is viable but premature.** Ship the columns at weight 0 first, and revisit
  once about 20 models are covered and there is at least one independent run per benchmark.
  Vals' CyberBench is the missing independent anchor; see §6.

---

## 1. Summary

| | CyberGym | ExploitGym | ExploitBench | SEC-bench Pro |
|---|---|---|---|---|
| Maintainer | UC Berkeley RDI (Sunblaze) | UC Berkeley RDI, co-authored with Anthropic / OpenAI / Google | CMU (Brumley) | UIUC (Lingming Zhang) |
| Paper | arXiv 2506.02548 | arXiv 2605.11086 | arXiv 2605.14153 | arXiv 2605.26548 (successor of SEC-bench 2506.11791) |
| Tasks | 1,507 OSS-Fuzz vulns, 188 projects | 869 (502 userspace, 181 V8, 186 kernel) | **41** V8 N-days | 344 (V8, SpiderMonkey, Linux); vendors report the 183-task 260505 snapshot |
| Agent must | Write a PoC crashing input from a description (Level 1) | Turn a crashing PoV into a working exploit that captures a flag | Climb a 5-tier capability ladder toward a V8 exploit | Write a PoC from a vague report, no crash trace |
| Verification | Execution (sanitizer crash) | Execution (flag) **plus** a model judge for "on target" | Execution (per-tier capability checks) | Execution on 3 images **plus** an LLM judge (99.2% precision) |
| Headline metric | Success rate, but `score_10` mixes pass@1 and pass@10 on old rows | Count of on-target exploits out of 869, at a 2h **or** 6h budget | Capability coverage % (plain / AutoNudge regimes) | Success % over all instances (timeouts count as failures) |
| **Top score** | 0.985 (agent) / 95.1 (model, self-reported) | 42.4 (GPT-6 Astra, self-reported) / 33.7 official | **100** (GPT-6 Astra, self-reported) / 78 official | 85.4 (GPT-6 Astra, self-reported) / 58.4 official |
| **Saturation** | **Saturating.** Agent rows sit at 0.91–0.99; our top-5 models span 84.5–95.1 | **Plenty of headroom**; the floor is compressed (weak models score 0–5%) | **Saturated at the top** by self-report | Headroom; floor compressed (open models score 0.6–3.8% on the official board) |
| Official-board matches (ours) | **6** (all open) | **4** (2 open, 2 ref) | **2** (open) | **2** (open) |
| Union across all sources | **15** (12 open, 3 ref) | **16** (9 open, 7 ref) | **~12** (5 open, 6–7 ref) | **14** (8 open, 6 ref) |
| Board maintained? | Yes; 75 L1 rows, last row 2026-09-13 | Yes; JSON modified 2026-09-21 | **No**; data generated 2026-06-29 | Harness yes; **board data unchanged since 2026-06-23** |
| Scrape | Static JSON ✅ | Static JSON ✅ (same site, same schema family) | Embedded in a hashed Next.js chunk ⚠️; the Epoch AI CSV is clean | Embedded JSON in static HTML, or `results.json` on GitHub ✅ |
| **Recommendation** | Add, weight 0 | **Add, weight 0** (strongest candidate) | Skip | Add pinned to 260505, weight 0 |

## 2. Coverage per benchmark

Matching used `_openness.normalize()` plus review by hand. Dated DeepSeek checkpoints
(`-0731`, `-0813`) fold into the undated slug, as every mapping file already does. Rejected
near-misses are listed under each benchmark.

### 2.1 CyberGym

**Official, `focus == "model"`, Level 1** (`https://www.cybergym.io/assets/data/cybergym.json`):

| slug | score | agent | reported by |
|---|---|---|---|
| `glm-5-3` | 84.5 | Claude Code | Zhipu (vendor) |
| `deepseek-v4-pro` | 83.3 | DeepSeek Agent (`-0813`) | DeepSeek (vendor) |
| `glm-5-2` | 77.2 | Claude Code | Zhipu (vendor), **only source for this value** |
| `deepseek-v4-flash` | 76.7 | DeepSeek Agent (`-0731`) | DeepSeek (vendor) |
| `kimi-k2-5` | 41.3 | Kimi Agent | Moonshot (vendor) |
| `glm-4-7` | 23.5 | Claude Code | Zhipu (vendor) |

`minimax-m3` appears only as an agent row (MopMonk, 0.731). The board has grown from 47 to 75
Level-1 rows since August. **Every new row above 0.85 except XekRung is `focus: agent`**:
GLM-5.3, DeepSeek-V4-Pro/Flash and GLM-5.2 running inside security vendors' pipelines
(Sangfor, Alipay, Huawei, NSFOCUS, DARKNAVY, …) at 0.87–0.97. The August filter,
`focus == "model"` with multi-model rows dropped, still keeps the column clean. The
pass@10 marker (`score_x1` ≠ `score_10`) now only appears on 2025 CyberGym-Team rows, none of
which are our models.

**Fill-only extras** (llm-stats `cybergym` board, HF cards, benchlm):

- Open models: `deepseek-v4-1-flash` 88.1, `atria-dawn-preview` 86.5, `mimo-v2-6-pro` 94.0,
  `mimo-v2-6-flash` 95.1, `hy4-preview` 78.4, `deepseek-v4-flash-vision` 75.3.
- Closed, benchlm only: `gpt-5-6-sol` 84.5, `gpt-5-6-terra` 81.8, `gpt-5-6-luna` 77.9.
- `kimi-k3` 80.0 / 78.7 appears only as a competitor column on other vendors' cards, so it is
  **not** first-party. Reject it.
- The source for `hy4-preview` is unconfirmed: llm-stats cites the HF card, and the card has
  no cyber content today.

Rejected near-misses: `GLM-5` (43.2) and `GLM-5.1` (68.7) → `glm-5-2`/`glm-5-3`;
`DeepSeek-V4-Pro Preview` (57.7) → `deepseek-v4-pro`; `DeepSeek-V3` → `deepseek-v3-2-0925`;
`Qwen3-235B-A22B` (2025-05) → the 2507 slugs.

### 2.2 ExploitGym

**Official** (`https://www.cybergym.io/assets/data/exploitgym.json`, score = `on_target / 869`):

| slug | on-target | % | budget | reported by |
|---|---|---|---|---|
| `gpt-5-6-sol` (ref) | 293 (2h: 216) | 33.7 (24.9) | 6h, Codex CLI | "OpenAI & ExploitGym Team" |
| `claude-opus-5` (ref) | 191 (2h: 171) | 22.0 (19.7) | 6h | Anthropic |
| `glm-5-3` | 130 (2h: 105) | 15.0 (12.1) | "rescaled 6h", Claude Code | Z.ai |
| `glm-5-2` | 39 (2h: 29) | 4.5 (3.3) | "rescaled 6h", Claude Code | Z.ai |

Reject: the DoGNAVY `glm-5-2` row (agent-focused, "selected subset"); `Claude Mythos 5` (this
is Fable 5, not `claude-fable-5-1`); GLM-5.1; GPT-5.4/5.5; Opus 4.6–4.8.

**Fill-only extras:**

- Open models: `deepseek-v4-1-flash` 15.3, `mimo-v2-6-pro` 17.8, `mimo-v2-6-flash` 6.0.
- Cross-reported only: `deepseek-v4-pro` 5.4, `deepseek-v4-flash` 1.8 (DeepSeek card);
  `mimo-v2-5-pro` 0.2 (MiMo card); `kimi-k3` 70/869 = 8.1 (Z.ai card).
- Closed, self-reported, via llm-stats/benchlm: `gpt-6-astra` 42.4, `gpt-5-6-terra` 23.2,
  `gpt-6-sol` 22.1, `gpt-5-6-luna` 12.4, `gpt-6-luna` 11.6.

**Unit trap.** The GLM-5.3 card prints raw task counts, `ExploitGym (2h / 6h) | 105 / 130`.
`fetch_huggingface.py` already parses that cell as **105.0**. It is harmless today only because
the label is `__unmappable__`, and it must stay unmappable. `ExploitGym (Pass@1)` (DeepSeek)
and plain `ExploitGym` (MiMo) are percentages. **Budget trap:** the MiMo card gives GPT-5.6 Sol
30.3 where the board has 33.7 (6h), and budgets are not stated on most vendor rows.
**Denominator trap:** benchlm divides some rows by 898 (the v0 task count) instead of 869.

### 2.3 ExploitBench

**Official** (exploitbench.ai, data generated 2026-06-29): 20 rows, mostly closed.

| slug | coverage % (plain / AutoNudge) | run by |
|---|---|---|
| `kimi-k2-6` | 16 / 18 | maintainers |
| `minimax-m2-7` | 13 / 13 | maintainers |

GLM 5.1 (16/18) is not ours.

**Fill-only extras:**

- Open models: `glm-5-3` 54.4, `mimo-v2-6-pro` 47.9, `mimo-v2-6-flash` 25.3.
- Cross-reported: `glm-5-2` 24.4, `kimi-k3` 32.2 (Z.ai card).
- Closed, self-reported: `gpt-6-astra` **100.0**, `gpt-5-6-sol` 73.5 / 76.5 / 78.5 (three
  sources, three values), `gpt-5-6-terra` 52.9, `gpt-5-6-luna` 33.2, `claude-opus-5` 70.0.

**Name collisions:** `NexLM/Oden-1-Preview` ("ExploitBench HARD*", a split that does not
exist); `namenotfoundai/Nightlight` (promotional 92%); `shirman/exploitbench-answers` (claims to
hold the answer set, which is a contamination red flag for the whole board).

### 2.4 SEC-bench Pro

**Official** (`sec-bench.github.io`, maintainer-run, every row dated 2026-06-17):

| slug | 260617 (344 tasks) | 260505 (183 tasks) | agent |
|---|---|---|---|
| `kimi-k2-5` | 2.3 | 2.2 | OpenCode on Bedrock |
| `minimax-m2-5` | 0.6 | 0.0 | OpenCode on Bedrock |

Reject: GLM-5 (3.8) → `glm-5-*`; GPT-5.4/5.5; Opus 4.6. The historical V8-only Kimi K2.6 row
(11.7) is gone from the site.

**Vendor reports** are all on the **260505 / 183-task** snapshot (OpenAI's system cards say
so explicitly):

- Open models: `deepseek-v4-1-flash` 62.8 (Claude Code harness), `deepseek-v4-pro` 56.4,
  `deepseek-v4-flash` 30.9, `mimo-v2-6-pro` 66.3, `mimo-v2-6-flash` 47.5, `mimo-v2-5-pro` 17.7.
- Closed models: `gpt-6-astra` 85.4, `gpt-5-6-sol` **71.2 / 74.3 / 79.1**, `gpt-5-6-terra`
  57.7, `gpt-5-6-luna` 48.9, `gpt-6-sol` 66.3, `gpt-6-luna` 34.2.
- The three GPT-5.6 Sol values come from OpenAI changing the grader between system cards (the
  GPT-6 card adds a root-cause-analysis agent) and from the token budget chosen.

## 3. Quality and saturation

**ExploitGym is the best of the four.**

- It is large (869 tasks) and spread across three targets.
- Nobody is near the ceiling (33.7 official, 42.4 claimed).
- Its top-5 spread is 18.9 points.
- It is maintained by the CyberGym team, with frontier labs as co-authors.

Its weaknesses are fixable at ingest:

- Mixed budgets. Take the 6h figure, which is what vendors quote, and record `eval_note`.
- Counts versus percentages. Always divide by `instances.total`.
- An LLM judge for "on target". This is a documented, audited step, not a free-form grade.

Floor compression is real: most open models below the frontier would score 0–5%. That makes
it a frontier discriminator, not a full-field one.

**CyberGym is saturating and harness-bound.**

- Level 1 hands the agent the vulnerability description.
- Security vendors' pipelines already reach 0.95–0.99.
- Among model-focused numbers our top five span 84.5–95.1, and the 95.1 is a MiMo self-report
  that names no harness. MiMo's card also describes cyber tasks being in its RL mix.
- Every open-model value is a vendor self-report run in the vendor's own or Claude Code
  harness. There are no maintainer runs of any current open model.

It still has the widest open-weight coverage, and it resolves the mid-field (23.5 → 77.2)
well. That makes it worth a column, not a vote.

**SEC-bench Pro is a good benchmark with a bad data situation.** The design is strong: large
real targets, withheld traces, an audited judge, and a "self-evolving" refresh. The problems:

- The official board has 6 rows, frozen since June, and its only open-model runs were made in
  OpenCode on Bedrock, where every open model sits near zero.
- Vendors report a different snapshot (183 vs 344 tasks) under at least two graders.
- Web-lookup cheating has been documented (fetching fix commits), and the harness hardening is
  from late September.

A column is only safe if it is pinned to snapshot 260505, the one vendors report and the
official board also publishes.

**ExploitBench is not worth it.**

- 41 tasks from one target (V8).
- Two scoring regimes (plain / AutoNudge) at 300 or 3000 turns.
- An official board untouched since June.
- A self-reported 100% at the top.
- A third-party repo claiming to hold the answer set.

Four of its five tiers are already bunched at the bottom for open models (13–18%), and the
only open values above that are vendor self-reports.

## 4. Do any of them belong in an existing index?

Rank correlation against our indexes, over the models that carry both. These are small
samples, and vendor and official values are mixed:

| | n | vs `coding_index` | vs `tooling_index` | vs `terminal_bench_4_0` |
|---|---|---|---|---|
| CyberGym | 10–12 | 0.76 | 0.78 | 0.63 |
| ExploitGym | 9 | 0.90 | 0.93 | 0.83 |
| ExploitBench | 7 | 0.86 | 0.96 | 0.94 |
| SEC-bench Pro | 7–8 | 0.95 | 0.96 | 0.70 |

**Coding index — no.**

- These boards are correlated with coding because exploit writing *is* agentic coding in a
  terminal, but they measure a narrow, dual-use specialty that vendors now train for directly.
  MiMo's cyber RL mix is one example; a CyberGym 95 on a model at `coding_index` 85.7 is
  another.
- Voting them into the coding index would reward that specialisation as general coding
  ability.
- Every open-model number is a self-report in the vendor's chosen harness. The coding group's
  weight ladder is built on maintainer or third-party runs (Real-SWE, Vibe Code Bench, DeepSWE).
- CyberGym, the only one with coverage, is also the most saturated.

**Tooling index — no.** The tooling group measures breadth of tool and API use across domains
(τ³, Toolathlon, MCP-Atlas, BFCL). A security harness is one terminal and one target. The high
Spearman is the frontier-versus-rest split that every agentic board shows, not shared
construct.

**A new Security index — yes, but not yet.**

- **Coverage.** Union coverage is ~15 models per benchmark, clustered on the same frontier set
  (GLM-5.x, DeepSeek-V4.x, MiMo-V2.6, the closed references). The Vision index ranks 63 models;
  a Security index would rank roughly 15–18.
- **Provenance.** Today it would rank almost entirely on vendor self-reports.
- **Conditions to build it:**
  1. The columns exist and have been refreshed for a month without mapping incidents.
  2. At least one independent run source per model. Candidates: Vals CyberBench (§6),
     ExploitGym maintainer runs, SEC-bench maintainer runs.
  3. About 20 ranked models under `MIN_SCORED_FRACTION`.
- **Proposed starting weights, when it ships:** `exploitgym` 1.0 (headroom, size, provenance
  trajectory); `vals_cyberbench` 0.9 (independent run, private set, PoC + patch);
  `sec_bench_pro_260505` 0.6; `cybergym` 0.4 (saturating, harness-bound). No `exploitbench`.

## 5. Where to fetch

| Source | Endpoint | Carries | Rank (`_precedence.py`) | Work |
|---|---|---|---|---|
| **cybergym.io (official)** | `/assets/data/cybergym.json`, `/assets/data/exploitgym.json` | CyberGym, ExploitGym | 2 (first-party) | **New fetcher `fetch_cybergym.py`** handling both files. Same site, same `leaderboard.js` renderer, plain `json.load`, CORS `*`. |
| **SEC-bench (official)** | `https://raw.githubusercontent.com/SEC-bench/sec-bench.github.io/main/data/results.json` (all snapshots, `snapshots.<ver>.leaderboards[]`), or `<script id="leaderboard-data">` on sec-bench.github.io | SEC-bench Pro 260505 + 260617, legacy SEC-bench | 2 | New small fetcher, or a second target in the same security fetcher. Only 2 of our models today. |
| **llm-stats** | `api.zeroeval.com/leaderboard/benchmarks/{id}/details`, ids `cybergym`, `exploitgym`, `exploitbench`, `sec-bench-pro` | All four, 7–20 rows each, all `self_reported` | 5 (fill-only) | **One line each in `fetch_llmstats.BOARD_FIELDS`**; the per-board path already exists for OSWorld 2.0. Add the `mimo-v2.6-*` ids to the llm-stats mapping. |
| **HF model cards** | existing `fetch_huggingface.py` | All four | 5 (fill-only) | Remap labels (§7). Nothing to fetch. |
| benchlm.ai | `/benchmarks/{cybergym,exploitgym,secbenchpro}`, `__NEXT_DATA__` | Adds `gpt-5-6-*` / `gpt-6-*` closed rows | 4 | **Not worth it.** Every row is a vendor self-report, some ExploitGym rows use the /898 denominator, and it adds only closed rows. |
| **Epoch AI Benchmarking Hub** | `https://epoch.ai/data/benchmark_data.zip` (CSV per benchmark, CC-BY) | ExploitBench, Cybench (not CyberGym, ExploitGym or SEC-bench Pro) | would be 3–4 | **New-collector candidate in general, not for this task.** It also carries DeepSWE, FrontierSWE, FrontierCode, Terminal-Bench, OSWorld 2, SciCode, HLE, GDPval and more, so it is a potential second source for many existing columns. It is the only clean ExploitBench source. |
| exploitbench.ai | Data inlined in a content-hashed Next.js chunk | ExploitBench | 2 | Fragile; skip together with the benchmark. |
| evals.report, Artificial Analysis, Scale Labs, Datacurve | — | none | — | Absent. AA shows a "Cyber Indexes" nav label only; `/cyber` returns 404. |

No existing fetcher needs a new site. The official boards need one new fetcher (two if
SEC-bench is done separately); everything else is configuration.

## 6. Related: Vals CyberBench

This was not in the question, but it matters for any Security index.

- `fetch_vals.py` already fetches Vals, and its docstring names CyberBench as deliberately
  unmapped.
- **What it is:** a Vals-owned benchmark with a private dataset, Vals-run (rank 2 for Vals' own
  boards, like Vibe Code Bench). It scores PoC generation and patching on OSS-Fuzz
  vulnerabilities, version 1.1, updated 2026-09-23.
- **Coverage:** 6 of our models.
  - Open: `mimo-v2-6-flash` 75.4, `deepseek-v4-1-flash` 73.7, `mimo-v2-6-pro` 72.9.
  - Closed: `claude-fable-5-1` 70.4, `claude-opus-5` 65.4, `gpt-6-astra` 41.1 (0% PoC,
    presumably refusals).
- **Why it matters:** it is the only security number here that nobody self-reported, and all
  six models are already in `model-name-mapping-vals-to-artificialanalysis.json`.
- **Caveat:** the patch sub-score is saturated (82–88% for everyone), so the PoC sub-score
  (0–65) carries the signal.

## 7. Implementation checklist (when approved)

1. **`fetch_cybergym.py`**, modelled on `fetch_toolathlon.py`:
   - `cybergym`: `level1` only. Keep `focus != "agent"`, drop `Multi-model*`, and refuse rows
     where `score_x1` is present and differs from `score_10`. `to_percent`. Ignore `subRows`.
   - `exploitgym`: take `on_target / instances.total × 100` from the top-level row (the 6h
     number where both exist). Refuse `focus == "agent"`, `hidden`, and any `eval_note`
     containing "subset". Strip parenthesised notes from `model`. Record `eval_note` in the
     output.
2. **SEC-bench Pro:** read `results.json`, snapshot `260505` only, `overall` board,
   `score_modes.headline.score`. Keep a `sec_bench_pro_260617` column only if a second
   revision column is wanted, following the DeepSWE / FrontierSWE revision convention.
3. **`fetch_llmstats.BOARD_FIELDS`**: `cybergym`, `exploitgym`, `sec-bench-pro`.
4. **`huggingface-benchmark-name-mapping.json`**:
   - `CyberGym`, `Cybergym`, `CyberGym (Pass@1)` → `cybergym`.
   - `ExploitGym`, `ExploitGym (Pass@1)` → `exploitgym`.
   - `SEC Bench Pro`, `SEC-Bench Pro (Pass@1)` → `sec_bench_pro_260505`.
   - **Keep `ExploitGym (2h / 6h)` `__unmappable__`**, because that cell holds task counts.
   - Keep `ExploitBench` and `MiMo Cyber Bench` unmappable.
5. **`llm.json`**: new category `Security`, weight 0, no sort group. `settings` must name the
   harness dependence, and `excludes` must name agent-focused rows, pass@10, the 2h budget
   (ExploitGym) and the 344-task snapshot (SEC-bench Pro).
6. Optionally map Vals `cyber` → `vals_cyberbench`, taking the overall score or the PoC
   sub-score (§6).

---

### Method

- Official data: `curl` of `cybergym.json` (75 L1 rows), `exploitgym.json` (15 rows, 1
  hidden), SEC-bench `results.json` (snapshots 260505 and 260617), and exploitbench.ai data
  extracted from its JS bundle (20 rows, generated 2026-06-29).
- Cross-sources: llm-stats `/leaderboard/benchmarks/{id}/details` (live), benchlm.ai page JSON,
  Epoch `benchmark_data.zip`, the Vals `cyber` board, and the evals.report / AA / Scale Labs
  indexes.
- `./fetch_huggingface.py --all-models` over 146 parseable cards. Raw cards read for GLM-5.3,
  DeepSeek-V4.1-Flash, MiMo-V2.6-Pro/Flash-RL, Atria-Dawn-Preview and Kimi-K3 to confirm
  harness footnotes and competitor columns.
- Spearman correlations computed against `llm.json` as of commit `64173ce`.
