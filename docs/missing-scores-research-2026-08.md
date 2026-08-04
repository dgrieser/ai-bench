# Missing benchmark scores — research report (2026-08-04)

Deep research into every missing (null) benchmark cell for the four newest
models in `llm.json`: `deepseek-v4-flash`, `glm-5-2`, `kimi-k3` and
`inkling-small`, restricted to the benchmark columns tracked by the index.
Sources checked per cell: the official leaderboard (fetched live, raw
HTML/embedded JSON parsed in full), the vendor's release blog + HuggingFace
model card (raw README), the benchmark's GitHub repo/paper, aggregators
(Artificial Analysis, BenchLM, llm-stats, rankedagi, BenchmarkList, vals.ai,
CodingFleet), and community/press coverage (Reddit, X, VentureBeat, etc.).

## Bottom line

**Not a single missing cell can be filled with a legitimate number as of
2026-08-04.** Every gap is a confirmed absence — the model is verifiably not
on the official leaderboard and the vendor did not self-report the benchmark —
not a failed lookup. No `llm.json` changes are warranted.

Two patterns explain nearly all gaps:

1. **Release timing.** DeepSeek V4 Flash (0731, released 2026-07-31) and
   Inkling Small (released 2026-07-30) postdate the latest update of almost
   every slow-moving leaderboard (SWE Atlas: 2026-07-28, SWE-Rebench window:
   2026-07-01, FrontierCode/evals.report snapshot: 2026-07-25, SWE-Marathon
   v1.1: 7 models only).
2. **Benchmark churn.** Mid-2026 vendor launch tables have standardized on
   newer suites (SWE-bench Pro, Terminal-Bench 2.1, τ³-Banking, FrontierSWE,
   DeepSWE, SWE-Marathon) and dropped SWE-bench Verified, LiveCodeBench,
   τ²-Telecom and BrowseComp. Those older leaderboards contain no post-July
   2026 frontier entries from these labs at all.

## Per-cell findings

### deepseek-v4-flash (DeepSeek-V4-Flash-0731, released 2026-07-31)

| Benchmark | Finding |
|---|---|
| swe_atlas_rf / _tw / _qna | Not on any of the three Scale leaderboards (last updated 2026-07-28; only DeepSeek V4 Pro listed). Not vendor-reported. |
| frontierswe | Not on frontierswe.com (15 models enumerated; only V4 Pro at 27.3 — do not proxy). Not in DeepSeek's 9-benchmark launch table. |
| frontiercode | Not on evals.report (snapshot 2026-07-25 predates release; only V4 Pro at 17.6). Not vendor-reported. |
| swe_marathon | Not in swe-marathon.org v1.1 data (JS bundle parsed: 7 models, no V4 Flash). Not on llm-stats. Not vendor-reported. |

### glm-5-2 (Z.ai, released 2026-06-13)

| Benchmark | Finding |
|---|---|
| swe_bench_verified | Z.ai deliberately reports SWE-bench Pro (62.1) instead for this release — no Verified number in blog, model card, or docs. Not on swebench.com; not on vals.ai (updated 2026-08-03). |
| livecodebench | Not vendor-reported (full benchmark list extracted from the blog's JS bundle). LiveCodeBench official repos are stale (last submissions 2025-12; zero "GLM" hits in the org). CodingFleet explicitly notes GLM hasn't published LCB scores. |
| browsecomp | Dropped entirely from the 5.2 release (no browsing section at all). GLM-5 scored 75.9 and GLM-5.1 68.0 — different models, do not attribute. |

### kimi-k3 (Moonshot, released ~2026-07-16)

| Benchmark | Finding |
|---|---|
| swe_bench_pro | Not in Moonshot's launch table; not on Scale's 33-model leaderboard; CodingFleet lists K3 as "—" and notes Moonshot didn't publish one. |
| swe_atlas_rf / _tw | Not on Scale leaderboards (only Kimi K2.5: RF 20.95, TW 25.77). Not in the K3 tech report (which reports MCP-Atlas — a different benchmark). |
| frontiercode | Not on evals.report (snapshot 2026-07-25, after K3's release — K3 was skipped, not just too new). See trap #1 below. |
| swe_rebench | Not on swe-rebench.com (all 111 entries parsed; newest Moonshot model is K2.6). Leaderboard window ends 2026-07-01, before release. |
| terminal_bench_2_0 | Not on tbench.ai 2.0 (142 rows parsed; best Moonshot: K2.5 + Terminus 2 = 43.2). See trap #2 below. |
| livecodebench | Not vendor-reported; K3 not registered in the official LCB harness (zero GitHub hits); not on llm-stats LCB v6 (newest: K2.6). |
| tau2_bench_telecom | Not vendor-reported — Moonshot reports the successor τ³-Banking (33.4) instead. Absent from AA τ²-Telecom, BenchLM (143 models) and llm-stats (35 models). |

### inkling-small (Thinking Machines, released 2026-07-30)

| Benchmark | Finding |
|---|---|
| swe_atlas_rf / _tw / _qna | No "Inkling" entry of any kind on any of the three Scale leaderboards. Not vendor-reported. |
| deepswe | Not on deepswe.datacurve.ai (19 entries parsed, updated 2026-08-04) nor in the DeepSWE v1.1 additions. Not vendor-reported. |
| frontierswe | Not on frontierswe.com; not vendor-reported. |
| frontiercode | Not on evals.report. See trap #3 below. |
| swe_marathon | Not in swe-marathon.org v1.1 data; not on evals.report or llm-stats. |
| swe_rebench | Not on swe-rebench.com (zero Inkling/Thinking Machines matches). |
| livecodebench | Not in model card, launch post, official LCB harness, llm-stats, or BenchmarkList (29 benchmarks matched, no LCB). |
| tau2_bench_telecom | Vendor reports τ³-Banking (15.5) instead; absent from all τ²-Telecom leaderboards. |

## Trap numbers — do NOT use

1. **Kimi K3 "FrontierCode 59.6"** — from Cognition's devin.ai/blog/kimi-k3,
   but measured on "FrontierCode 1.1", a different scale sitting ~14 points
   above the evals.report metric this index uses (e.g. Opus 4.8: 60.6 there
   vs 46.5 on evals.report). Incomparable.
2. **Kimi K3 "Terminal-Bench 2.0 = 88.3"** — shown on benchlm.ai/models/kimi-3,
   but Moonshot's own model card labels that exact figure Terminal-Bench
   **2.1** (own Kimi Code harness). A mislabel; benchlm's own TB2.0
   leaderboard page doesn't list K3 either.
3. **Inkling-Small "FrontierCode 14.0"** — evals.report's 14.0 belongs to the
   larger, earlier **Inkling** model (2026-07-15), not Inkling-Small.
4. **GLM-5.2 "SWE-bench Verified ~62%"** — EdenAI blog table; almost certainly
   the SWE-bench **Pro** score (62.1) mislabeled as Verified. No corroboration
   anywhere.

## Audit flags on existing data (side findings, not changed)

- **`kimi-k3.swe_atlas_qna = 23`** could not be corroborated anywhere: the
  official Scale Q&A leaderboard lists only Kimi K2.5 (13.10), and neither
  Moonshot's tech report nor its model card reports SWE Atlas. Source worth
  re-checking.
- **`inkling-small.terminal_bench_2_0 = 64.7`** is suspect: Thinking Machines
  labels 64.7 as its **Terminal Bench 2.1** (best harness) score, and
  benchlm's TB2.0 column (which matches our 2.0 values for GLM-5.2, DeepSeek
  V4 Flash and Inkling exactly) appears to absorb vendor-reported 2.1 numbers.
  None of those three models appear on the official tbench.ai 2.0 board.
- **`kimi-k3.frontierswe = 81.2`** is a Moonshot vendor-run number (official
  frontierswe.com does not list K3), whereas `glm-5-2.frontierswe = 72.5`
  matches the official leaderboard exactly — the column mixes official and
  vendor-reported values.
- Vendor tables occasionally disagree with the official boards we track
  (e.g. Kimi K3 DeepSWE: model card 67.5 vs official leaderboard 68.5;
  GLM-5.2 DeepSWE: model card 46.2 vs official 43.8; GLM-5.2 FrontierSWE:
  model card 74.4 vs official 72.5). Current values match the official
  leaderboards, which is consistent.

## Re-check schedule

All four models are recent; most gaps should close as leaderboards refresh.
Worth re-running the scrapers / re-checking in 1–2 weeks, in particular:
Scale SWE Atlas (updates ~weekly), swe-rebench.com (monthly window),
evals.report FrontierCode, deepswe.datacurve.ai, frontierswe.com and
swe-marathon.org.
