# CyberGym — coverage and scrapeability assessment, August 2026

Follow-up to [`benchmark-candidates-2026-08.md`](benchmark-candidates-2026-08.md), which
covered 19 candidates but no security benchmark. Same method: coverage measured against the
**122 models in `llm.json`**, not estimated from documentation. Snapshot: 2026-08-01.

**Verdict: coverage is fine by this repo's own bar (6 models — more than `frontierswe` or
`swe_marathon` carry today). The blocker is comparability, not thinness. Defer until the
leaderboard separates pass@k, or add it with a hard `focus == "model"` filter and accept a
4-model column.**

---

## 1. What CyberGym is

[CyberGym](https://www.cybergym.io/cybergym/) (UC Berkeley Sunblaze / RDI,
[arXiv:2506.02548](https://arxiv.org/abs/2506.02548)) is a real-world vulnerability-analysis
benchmark: 1,507 instances drawn from historical OSS-Fuzz vulnerabilities across 188
projects. The public leaderboard ranks **Level 1** only — the agent gets a vulnerability
description plus the unpatched codebase and must produce a working PoC that reproduces the
crash. Metric is **Success Rate**: % of instances where any trial produces a working PoC.
Levels 0/2/3 vary the input richness (0 = no description, 2 = + stack trace, 3 = + ground
truth patch) and carry exactly **one row each** (GPT-4.1/OpenHands), so they are not usable.

Execution-verified, not LLM-judged, and post-cutoff for most of our models — the provenance
is as good as Toolathlon's. The problem is downstream of that.

## 2. Coverage: 6 of 122 (4.9%)

Matched with `_openness.normalize()` plus a prefix fallback, same as Appendix A of the
candidates doc.

| `llm.json` slug | CyberGym label | Score | Agent / scaffold | `focus` | Reported by |
|---|---|---|---|---|---|
| `deepseek-v4-flash` | DeepSeek-V4-Flash | 76.7% | DeepSeek Agent | model | DeepSeek (vendor) |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 57.7% | Claude Code | model | XDxAI (third party) |
| `kimi-k2-5` | Kimi K2.5 | 41.3% | Kimi Agent | model | Kimi (vendor) |
| `glm-4-7` | GLM-4.7 | 23.5% | Claude Code | model | Zhipu AI (vendor) |
| `glm-5-2` | GLM-5.2 | 86.3% / 84.8% | Sangfor AI / Xuanwu Atuin AI | **agent** | Sangfor, Tencent Xuanwu |
| `minimax-m3` | MiniMax M3 | 73.1% | MopMonk Agent | **agent** | MopMonk AI |

All six are `trials: 1`, so at least among *our* models the column is internally pass@1.

**Only 4 have a `focus: "model"` row.** GLM-5.2 and MiniMax M3 appear solely as the engine
inside someone else's competition agent — GLM-5.2's 86.3% is Sangfor's scaffold, not a
GLM-5.2 capability number. A `focus == "model"` filter (the defensible choice) drops the
column to **4 models (3.3%)**.

### Three near-matches that must be rejected

The prefix fallback proposes these; all three are wrong and would silently corrupt a column:

| CyberGym label | Proposed slug | Why it's wrong |
|---|---|---|
| `GLM-5` (43.2%) | `glm-5-2` | GLM-5 and GLM-5.1 are distinct releases; neither is in `llm.json`. A 43.2% would land on the model whose real row is 86.3%. |
| `DeepSeek-V3` (3.6%) | `deepseek-v3-2-0925` | Original V3 (2024-12), not V3.2. |
| `Qwen3-235B-A22B` (1.9%) | `qwen3-235b-a22b-instruct-2507{,-reasoning}` | The OpenHands run is dated 2025-05-15 — it predates the 2507 refresh, and matches both slugs ambiguously. |

Also present but not in `llm.json`: GLM-5, GLM-5.1, GLM-5.1-FP8, Muse Spark / Muse Spark 1.1
(Meta, closed), R2E-Gym-32B, OpenHands-LM-32B, SWE-Gym-32B (SWE fine-tunes we don't carry).
The remaining ~30 rows are closed models (GPT-5.x, Claude, Gemini) or multi-model agents.

### Context — 6 is not the problem

`llm.json` already ships columns at this depth: `frontierswe` 5, `swe_marathon` 5,
`swe_atlas_rf` 6, `swe_atlas_qna` 7, `frontiercode` 8, `deepswe` 9, `toolathlon` 16. On
coverage alone CyberGym clears the bar that `deepswe` — weight 1.0 in the coding group —
clears.

## 3. Where it can be scraped

| Source | Endpoint | Open-weight rows | Verdict |
|---|---|---|---|
| **cybergym.io (official)** | `https://www.cybergym.io/assets/data/cybergym.json` | **6 of ours** | **Primary.** Static JSON, no auth, no JS execution. Config URL is declared in `/assets/js/leaderboard.js` (`dataUrl`). |
| **Hugging Face model cards** | `moonshotai/Kimi-K2.5` README | 1 of ours | **Cross-check.** Card reports `CyberGym = 41.3`, exactly the official 0.413. Already reachable via `fetch_huggingface.py`; the label is currently pinned `__unmappable__` in `huggingface-benchmark-name-mapping.json:175`. |
| **benchlm.ai** | `/benchmarks/cybergym` (16 models, `Open weight` flag) | 3, **0 new** | **Not worth it.** Mirrors official numbers where they overlap, but only carries `focus: model` rows *and* drops GLM-4.7, Kimi K2.5, DeepSeek-V4-Pro and MiniMax M3. It adds 5 closed models absent from the official board (Sakana Fugu Cyber, GPT-5.6 Sol/Terra/Luna, Gemini 3.5 Flash Cyber) and relabels `DeepSeek-V4-Flash` as `DeepSeek V4 Flash (Max)` — see §4. HTML-only, no flight/JSON endpoint. |
| llm-stats.com | `api.zeroeval.com/leaderboard/models/full` | — | **Absent.** 335 records, 22 `*_score` keys, no cybergym slug. |
| evals.report | `/benchmarks/cybergym` | — | **404.** Not among its 82 benchmarks. |
| Artificial Analysis | v2 API + page | — | **Absent.** No security field in the 17 API or 10 page-only fields; nothing in `update.py`'s ignored-keys list. |
| `sunblaze-ucb/cybergym` (GitHub) | — | — | Code + `SUBMISSION.md` only; results live on the website. |
| **CyberGym-E2E** | `/assets/data/cybergym-e2e.json` | **0** | **Useless for us.** 9 rows, all closed (Claude, GPT-5.x, Gemini), all CyberGym-E2E-Team-run. Nice schema (`patch_only`, `s1`–`s4`, budget), zero open-weight models. |

So: one good source, one one-row corroboration, and nothing else. "Multiple sources" is not
really available for this benchmark yet.

## 4. Why to hold off — the `score_10` field mixes pass@1 and pass@10

The single column the site sorts on is `score_10`, and it does **not** mean one thing:

- The 2025 CyberGym-Team rows carry both `score_10` and `score_x1`, and they differ by up to
  **9×** (`o4-mini`: 2.46% vs 0.07%; `Claude Sonnet 4`: 17.85% vs 1.99%). `score_10` is the
  paper's pass@10; `score_x1` is pass@1. Yet every one of those rows also reports
  `trials: 1`, so the `trials` field cannot be used to tell them apart.
- Anthropic submitted the same four models twice at `trials: 30` and `trials: 1`, in the same
  `score_10` field: Claude Sonnet 4.5 is **66.7% at 30 trials and 28.9% at 1**. Both rows are
  ranked in the same list.
- Scaffold dominates the number. GPT-4.1 ranges 7.2%–9.4% across ENiGMA / Codex CLI / Cybench
  / OpenHands. GLM-5.1 is 68.7% under Claude Code and 84.0% under Xuanwu Atuin.
- 8 of the ~14 top rows are `focus: agent` — competition systems (Wiz Atlas 90.9%, MDASH
  88.5%, Crystalline 89.6%) whose scores belong to the harness, some of them multi-model.

A naive `score_10` scrape therefore produces a column where a vendor's pass@1 sits next to a
research team's pass@10 next to a security firm's tuned multi-stage pipeline. That is a
different failure mode from thin coverage, and worse: it looks fine.

Secondary label risk (resolved): benchlm calls our 76.7% row `DeepSeek V4 Flash (Max)`, the
official `source_url` points at `deepseek-ai/DeepSeek-V4-Flash-0731`, and llm-stats carries
`deepseek-v4-flash-max` as a separate model id. `llm.json` now keeps only the final variant, on
the undated slug `deepseek-v4-flash` (the name Artificial Analysis itself gives that release),
and every mapping file folds those labels into it — so that score can be assigned to the single
row without picking a variant.

## 5. If we add it anyway

Cheap, and the failure modes are all avoidable:

1. `fetch_cybergym.py` reading `/assets/data/cybergym.json` — a `json.load`, no HTML parsing.
   Model it on `fetch_toolathlon.py`, which already solves the same "two series in one place,
   don't mix them" problem by selecting explicitly and raising if the shape changes.
2. Take `level1` only. Ignore levels 0/2/3 (one row each).
3. **Filter `focus == "model"`** and skip rows whose `model` starts with `Multi-model`. Report
   the dropped count, as `fetch_toolathlon.py` does for self-reported rows.
4. **Refuse rows where `score_x1` is present and differs from `score_10`** — that is the
   pass@10 marker. Prefer `score_x1` when both exist, or drop the row.
5. Expose `trials`, `focus`, `agent`, `source` and `source_url` in the fetcher output so
   `add.py`/`update.py` can see what scaffold produced a number. Consider refusing
   `trials > 1`.
6. Benchmark key `cybergym`, category `Security` (new category), `to_percent` transform
   (values are 0–1 fractions). **Weight 0 / no sort group** — it is not a coding benchmark and
   the scaffold variance makes it unfit for a composite.
7. Map `CyberGym` → `cybergym` in `huggingface-benchmark-name-mapping.json` (currently
   `__unmappable__`) to pick up Kimi K2.5's self-report as a corroborating second source.

Steps 3–4 are the whole point. Without them the column is not measuring models.

## 6. Recommendation

**Defer.** Not on coverage — 6 models beats four columns already shipping — but because the
leaderboard's primary field conflates pass@1, pass@10 and agent-harness scores with no field
that reliably separates them, and only 4 of the 6 have a model-focused row at all. Revisit if
the site splits pass@k into its own column, or if AA/evals.report picks up the benchmark and
runs it themselves.

If it ships sooner, ship it as a zero-weight `Security` column with the §5 filters, and note
in its `description` that scores are scaffold-dependent and pass@1-only.

---

### Method

- `curl` on `https://www.cybergym.io/assets/data/cybergym.json` (17,961 bytes; 47 `level1`
  rows + 1 `subRows` entry + 1 row each for levels 0/2/3) and `/cybergym-e2e.json`.
- Matching: `_openness.normalize()` on both sides, exact then prefix fallback; all 9 raw hits
  reviewed by hand, 3 rejected (§2).
- `./fetch_huggingface.py --all-models` — 112 of 122 models parsed (10 gated/401), 1 CyberGym
  label found.
- llm-stats via `api.zeroeval.com/leaderboard/models/full` (the endpoint `fetch_llmstats.py`
  uses); evals.report `/benchmarks` index + direct slug probe; benchlm.ai page scrape; AA
  field list from §6 of the candidates doc.
