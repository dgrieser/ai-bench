# Score-gap and fetcher audit, October 2026

Run on 2026-10-08 over the 36 models added since 2026-09-01. Three questions:
which published scores for those models `llm.json` was missing, which of them
a fetcher could have collected, and whether every fetcher collects what its
source offers for the columns and models tracked here. Each source cited below
was read on that date.

## The first finding: a broken Artificial Analysis scraper

The 10:37 UTC refresh on 2026-10-08 wrote the **same** numbers onto eleven
models: SciCode 58.9, HLE 46.5, AA-LCR 88.3, CritPt 20.9, GDPval-AA 1585,
Briefcase 1425, ITBench 55.6, AutomationBench 51, MMMU-Pro 76.4,
Terminal-Bench 4.0 33.3, and Omniscience 16.4 / 43 / 41.5. Those are Step 5
Preview's values. AA's new page layout dropped the `"currentModel"` wrapper,
and `_metrics_chunk()` read on from the first bare mention of the slug (now a
`"currentRelease"` summary) into the next record, which on most pages is a
pinned Step 5 Preview. The models affected were kimi-k3, glm-5-3, qwen3-8-27b,
deepseek-v4-1-flash, claude-fable-5-1, gpt-6-astra, claude-opus-5-5,
gpt-6-luna, claude-sonnet-5-5, gpt-6-1-sol and claude-haiku-5-5.

Fixed in `artificialanalysis.py` (`_current_model_start`). The 154 cells were
restored from b006230 and the indexes recomputed. The metrics window was also
widened from 5,000 to 12,000 characters, because `briefcaseBreakdown` had
drifted to about 4,600 characters from the start.

## New scores, and how they arrived

| Model | Column | Value | Route |
| --- | --- | --- | --- |
| claude-opus-5-5 | swe_bench_multilingual / swe_bench_multimodal / toolathlon | 93.9 / 61.4 / 77.8 | llm-stats boards, now read (system card) |
| claude-sonnet-5-5 | swe_bench_multilingual / swe_bench_multimodal / toolathlon / osworld_2_0_vendor | 90.3 / 54.3 / 77.8 / 80.1 | llm-stats boards (system card) |
| claude-haiku-5-5 | swe_bench_multilingual / swe_bench_multimodal / osworld_2_0_vendor | 83.7 / 30.7 / 72.4 | llm-stats boards (system card) |
| claude-haiku-5-5 | swe_marathon_1_1 | 50.0 | swe-marathon.org, once the queued mapping was answered |
| claude-haiku-5-5 | exploitbench / exploitgym | 38 / 9.4 | hand, system card (§3.3.1 text; figure 3.3.4.A, 82 of 869 at 6h) |
| claude-sonnet-5-5 | exploitgym | 26.6 | hand, system card (figure 3.2.4.A, 231 of 869 at 6h) |
| ling-3-1-flash | swe_atlas_qna | 55.9 | llm-stats `swe-atlas-codebase-qna` board, now read |
| k2-horizon-3-7b | aime_2025 / aime_2026 / livecodebench / bfcl_v4 | 89.2 / 90.8 / 64.5 / 64.6 | model card's "after" column (the released checkpoint), now read |
| k2-horizon-7b | aime_2025 / aime_2026 / livecodebench / bfcl_v4 / swe_bench_verified | 90.3 / 90.2 / 72.7 / 67.0 / 72.4 | same |
| dots3-note-prev | livecodebench / charxiv_reasoning | 91.5 / 83.1 | hand, from the card's table images (no fetcher reads images) |
| gemini-4-argon | agents_last_exam | 39.5 | hand, Google's evaluation methodology (binary pass rate, ALE-Claw harness) |

Models with nothing new to find:

- **MiniMax-M3.1-Flash-Preview:** its Hugging Face repo does not exist (401). It
  is a closed preview inside MiniMax Code, and no benchmark table has been
  published. Figures circulating on SEO sites belong to MiniMax-M3.
- **Apodex-1.1-mini:** its card's scores are PNG charts of APEX-Agent and
  FrontierFinance only. The numbers in the card's prose are for the 397B
  flagship.
- **Kolibri-1, Bonsai-2-27B, MiniCPM5-2B, K2-Horizon-0.9B/36B/375B,
  Ling-3.0-flash-VL/-Fin, Mistral Large 4, MiMo-V2.6:** no board lists them
  that is not already read, and their cards add nothing in a tracked variant.

## Values that were wrong, and what changed so they stay fixed

| Cells | Problem | Fix |
| --- | --- | --- |
| gpt-5-6-luna/terra/sol toolathlon (53.4 / 53.1 / 58.0) | Pre-Verified Toolathlon series, from llm-stats' flat field | `fetch_llmstats` reads Toolathlon off its boards and keeps only rows whose note says "Verified"; cells cleared |
| mistral-large-4 cybergym 82 | AA's CyberGym-E2E (reproduce + patch), filed under CyberGym by llm-stats | llm-stats board filter drops `E2E` notes; cell cleared |
| deepseek-v3-2-0925 swe_bench_verified 73.1, browsecomp 51.4, toolathlon 35.2 | llm-stats' `deepseek-v3.2` (the final V3.2) mapped onto the V3.2-Exp entry | mapping swapped to `deepseek-v3.2-exp`; now 67.8 / 40.1; toolathlon cleared |
| ling-3-0-flash-fin gpqa_diamond 39.6 | OpenRouter median over a broken endpoint (15.9%, below chance) | runs at or under 25% dropped; now 63.2 |
| claude-fable-5-1 / glm-5-3-flash / qwen3-8-2-4t-a95b mmmu_pro | Vals' "MMMU Pro" is the 4-option set; the column is 10-option | Vals `mmmu` board no longer read; cells cleared (qwen re-filled at 82.3 from llm-stats) |
| kolibri-1 aa_omniscience_hallucination 44 | Card row is the *non*-hallucination rate | label set unmappable; value set to 56 |
| nex-n2-5-pro/mini/max browsecomp, terminal_bench_2_1 | Footnotes: Summary context compaction; NexAU harness, not Terminus-2 | `huggingface-card-exclusions.json`; cells cleared |
| k2-horizon-7b browsecomp 59 | Discard-all@95k context management | card exclusion; cleared |
| step-5-preview programbench_almost 80.5, swe_marathon_1_1 72.7 | On neither board's scale (the card has Opus 5 at 82.3 / 85.6) | card exclusion, and a bare "ProgramBench" label is unmappable everywhere; cleared |
| deepseek-v4-1-flash agents_last_exam 31.8 | ALE-CLI tier (tech report §5.3.1) | card exclusion; cleared |
| atria-dawn-preview terminal_bench_2_1 78.3 | Claude Code harness (tech report Appendix B) | card exclusion; cleared |
| nex-n2-5-max vram | Spheron answers `Nex-N2.5-Max` for a request for `Nex-N2.5-max` | case-insensitive lookup; filled |
| devstral-2 / devstral-small-2 on Vals | Vals paths wrongly `__unmappable__` (a hand clear on 09-19 was re-inferred away) | mapped; devstral-2 SWE-bench Verified now Vals' 62.8 over its card's 72.2 |
| Step 5 Preview on Epoch | `__closed_weights__`, set before the model was tracked | mapped |
| gpt-6-astra on Datacurve | All five effort rows `__unmappable__` | mapped |

The Terminal-Bench 2.1 ingest now keeps only Terminus 2 rows
(`fetch_tbench.REQUIRED_AGENT`). No cell changes today, but the board's best
row per model was a Codex or Claude Code run.

## Flagged, not changed

These need a decision rather than a fix:

- **Step 5 Preview has no AA mapping.** AA's slug `step-5`, a record named
  "Step 5 Preview" with release `step-5-preview` and 600B-A27B, is on
  `model-name-mapping-llm-to-artificialanalysis-ignored.json` for
  step-5-preview. If that was not deliberate,
  `./answer.py update_artificialanalysis_mapping.py step-5-preview step-5 -w`
  brings in AA's own runs.
- **Four `swe_atlas_rf` values Scale no longer lists:** deepseek-v4-pro 53.8,
  glm-5-3 51.4, kimi-k3 46.2 and minimax-m3 28.1. They are credited to the
  refactoring board, which has none of them today. They stay because a dropped
  score is never nulled.
- **k2-horizon-375b-a23b swe_bench_pro 42.6** comes from a label now parked
  unmappable ("SWE Bench Pro (strict)"). The card defines strict as "no
  internet", not strict grading.
- **deepseek-v4-1-flash swe_bench_pro 56.8 and ling-3-1-flash deepswe_1_1
  59.7** (hand entries citing threatfrontier.com): the first is Ant Group's
  run of DeepSeek, from Ling's launch table. The second's revision was
  inferred.
- **atria-dawn-preview browsecomp 92.5** (llm-stats): the tech report says
  "BrowseComp additionally uses the discard-all context-management strategy".
  llm-stats' note does not say so, so the scaffold filter cannot see it.
- **gpt-6-astra exploitbench 100 / exploitgym 42.4 / sec_bench_pro 85.4**
  (llm-stats, from OpenAI's launch table): OpenAI's system card describes an
  updated ExploitBench metric, no 6-hour cap on ExploitGym and its own
  SEC-bench grader. GPT-6 Sol, GPT-6 Luna and GPT-6.1 Sol publish the same
  variants (e.g. Sol 81.7 / 22.1 / 66.3) and were **not** added for that
  reason.
- **evals.report and Vals "DeepSeek V3.2"** rows map to deepseek-v3-2-0925
  (V3.2-Exp) and may be the final V3.2. The cells involved are
  swe_bench_multilingual 59 and bfcl_v4 56.7 (evals.report), and
  terminal_bench_2_0 34.8 (Vals).
- **dots3-note-prev SWE-bench Verified/Multilingual/Pro** were run with
  live-swe-agent. That is borderline under the "runs named for a harness
  (SWE-agent)" exclusion.
- **gpt-5-6-terra terminal_bench_4_0 35.4 (AA)** against 21.5 on tbench.ai and
  22.7 on Vals. Its terminal_bench_2_1 88.0 equals GPT-5.6 Sol's exactly.
  Worth a look once the AA refresh runs with the fixed parser.
- **The `terminal_bench_4_0` definition** excludes "runs named for a harness
  (Claude Code, Terminus-2)", but every row on tbench.ai's own 4.0 board is a
  vendor-harness run. Nothing lands from it today, because AA leads, but the
  definition and the board disagree.
- **Stale slugs** `claude-opus-5` / `claude-sonnet-5` are still mapped in
  several files (epoch, zerobench, hle-diamond, osworld, agents-last-exam,
  llmstats, openrouter, aa-coding-agents). They are harmless, since there is
  nothing to write to.

## Sources with something tracked that no fetcher reads

| Source | Columns | Format | Worth it? |
| --- | --- | --- | --- |
| Scale `humanitys_last_exam_text_only` | hle | Next.js `entries` JSON (the `_scale_labs.py` shape) | Low: AA leads HLE and covers nearly everyone. Would fill models AA has not run |
| Scale `swe_bench_pro` (public/private) and `swe_bench_pro_public_v2` (full/hard) | swe_bench_pro | flight payload, two variants per page | Needs a column decision first: the column excludes the public split, yet several stored values equal Scale's public numbers, and V2 is a new near-saturated set |
| swebench.com `#leaderboard-data` | swe_bench_verified / multilingual / multimodal | inline JSON | Low: no entries after 2026-02; would fill llama-4-maverick (21.0) and llama-4-scout (9.1) |
| AA `/evaluations/<benchmark>` pages | every AA column | full per-model records | A fallback if the model pages break again |
| benchlm.ai `/md/models/<slug>.md` | mixed | markdown with per-row sources | No: it mislabels variants (HLE with tools as HLE, ZeroBench@5 as ZeroBench, ALE-CLI as ALE) |
| Anthropic / OpenAI system cards, OpenAI launch pages | cyber, SWE-bench sets, OSWorld 2.x | PDF / Contentful CSV | Not as fetchers: llm-stats' boards now relay the system-card numbers that matter |

The first-party boards for LiveCodeBench, MMMU-Pro, CharXiv and MathVista are
machine-readable but frozen (last data 2025-04 to 2026-07) and list none of the
new models.
