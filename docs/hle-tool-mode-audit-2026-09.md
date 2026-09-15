# Is the HLE column one measurement? — September 2026

`hle` carries **144 values across 158 models**, the second-broadest column in the table and
the 0.9-weight member of the [Knowledge index](../README.md#knowledge-index). This is an
audit of what those 144 numbers actually measure: whether each was run **with tools** or
**without**, and where the run differs from its neighbours on some other axis entirely.

Snapshot 2026-09-15, against `llm.json` at `6cff0ea`. Method and sources in
[Appendix A](#appendix-a--method).

---

## 0. Headline

- **134 of 144 values (93%) are clean and identical in configuration**: Artificial Analysis'
  own run — 2,158 text-only questions, **no tools**, pass@1, GPT-5.6 Luna (medium) grader.
  Verified against AA's published methodology and spot-checked live against AA's model
  pages. There is no stale grader vintage hiding in this block.
- **5 values are not that measurement, and 5 of those 5 are wrong**: `agents-a1`,
  `deepseek-v4-1-flash`, `claude-opus-5`, `claude-sonnet-5` and `claude-fable-5-1` all store
  a **with-tools** number in a no-tools column.
- Those 5 are **exactly the top 5 of the column**. Every model the HLE leaderboard currently
  shows above `kimi-k3` (46.9) is there because it was measured with web search and a code
  interpreter and its neighbours were not.
- The remaining 5 non-AA values (`ornith-1-5-*` ×3, `hy4-preview`, `longcat-flash-thinking`)
  are in the right **mode** — no tools — but come from each vendor's own harness, judge and
  question set, not AA's.
- Where a publisher reports both modes for the same model, **tools are worth a median
  +11.5 points**, up to **+27.1** (DeepSeek-V4.1-Flash: 36.8 → 63.9). That is roughly twice
  the CharXiv tool-mode gap this repo already parks labels over.
- The mechanism is not one bug. **23 label spellings map onto `hle`**, six of them explicitly
  with-tools, and `update.fetch_huggingface_data()` resolves label collisions by *keeping the
  best number*. The no-tools values that survived did so by ingest ordering, not by rule.

---

## 1. What "an HLE score" can mean

HLE is published five different ways, and the column currently accepts all five:

| axis | variants seen in our sources | size of the effect |
|---|---|---|
| **Tools** | no tools · with tools · with search · with search + code execution · search agent | **median +11.5 pp**, max +27.1 |
| **Question set** | full 2,500 (multimodal) · 2,158 text-only subset · "HLE-Full" · "HLE-Verified" | text-only ≈ +2–3 pp; HLE-Verified ≈ +9 pp |
| **Grader** | GPT-5.6 Luna (AA, GLM) · Claude Opus 4.6 (Ornith) · GPT-4o (Qwen3.8) | unquantified; Anthropic re-stated Sonnet 4.6 by −5.6 pp on a grader change alone |
| **Budget** | pass@1 · "heavy" · max reasoning effort · effort-vs-cost curves | kimi-k2-thinking: 23.8 (AA pass@1) vs 51.0 ("heavy") |
| **Safeguards** | production safeguards on/off | Anthropic: Fable 5.1 vs Mythos 5.1 differ only here |

Only the first is what the question asked about, and only the first is currently doing real
damage — but the others are the reason a "fix the tool mode" patch is not sufficient on its
own to make the column comparable end to end.

---

## 2. The reference implementation

134 of 144 values come from Artificial Analysis, and AA's
[methodology page](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
pins the configuration exactly:

> 2,158 text-only questions across mathematics, humanities and the natural sciences (from the
> May 2025 revision which contains 2,500 total questions — we use the text-only subset for
> maximum comparability across models)

with `Tool Usage: ✗` in the Intelligence Index table, pass@1 scoring, and an equality-checker
LLM on GPT-5.6 Luna (medium). AA sits at `RANK_AA` in `_precedence.py`, so nothing but a later
AA number can displace one of these.

**Grader vintage is not a problem here.** AA v4.1.1 (Aug 2026) replaced the HLE grader —
GPT-4o (Aug '24) → GPT-5.6 Luna — and 11 of our AA values still carry a `scores_updated`
stamp from before that switch. `apply_score()` only re-stamps on change, so the date cannot
distinguish "never regraded" from "regraded, unchanged". Fetching all 11 live from AA's model
pages settles it: every one matches the stored value to the rounding grid
(`devstral-2` 0.03568 → 3.6, `gemma-4-e2b` 0.04773 → 4.8, `granite-4-1-3b` 0.03429 → 3.4, …).
The AA block is uniformly current.

**This is the definition the column should carry**, and it is not the definition the column
currently advertises. `llm.json`'s `benchmarks.hle.description` says "multi-modal benchmark
with 2,500 expert-level questions", which describes the dataset, not the run — and is wrong
about 93% of the values under it.

---

## 3. Verdict on every value

### 3.1 The AA block — 134 values

No tools, text-only 2,158, pass@1, GPT-5.6 Luna. Internally consistent; comparable to each
other and to nothing else on this page. No action.

### 3.2 The 10 non-AA values

| model | stored | mode | question set | evidence | verdict |
|---|---:|---|---|---|---|
| `ornith-1-5-397b` | 44.6 | **no tools** | full, Claude Opus 4.6 judge | card eval metadata `notes: "No tools."`; card row `HLE (no tools)` | mode OK, foreign harness |
| `ornith-1-5-35b-a3b` | 25.6 | **no tools** | full, Claude Opus 4.6 judge | same | mode OK, foreign harness |
| `ornith-1-5-9b` | 20.2 | **no tools** | full, Claude Opus 4.6 judge | same | mode OK, foreign harness |
| `hy4-preview` | 43.4 | **no tools** | text-only | card eval metadata `notes: "Text-only, no tools."` | mode OK, foreign harness |
| `longcat-flash-thinking` | 25.2 | **no tools** | text-only | llm-stats `analysis_method: "text-only, without tools"` | mode OK, foreign harness |
| `agents-a1` | 47.6 | **WITH TOOLS** | full | card eval metadata `notes: "With tools"`; card prose "HLE with tools (47.6)" | **wrong** |
| `deepseek-v4-1-flash` | 63.9 | **WITH TOOLS** | full | card metadata `notes: "With tools; harness not specified"`; card table row `HLE w/ tools (Pass@1)` = **63.9**, listed under *Agentic* | **wrong** — no-tools value is 36.8 |
| `claude-opus-5` | 64.7 | **WITH TOOLS** | full, multimodal | llm-stats `analysis_method: "Multidisciplinary reasoning with tools."` | **wrong** |
| `claude-sonnet-5` | 57.4 | **WITH TOOLS** | full, multimodal | llm-stats: "With tools (web search, web fetch, code execution): 57.4%. **Without tools: 43.2%**" | **wrong** — no-tools value is 43.2 |
| `claude-fable-5-1` | 65.0 | **WITH TOOLS** | full, multimodal | vendor table: "65.0% with tools" / "**60.9% no tools**" | **wrong** — no-tools value is 60.9 |

Replacement values exist for three of the five. For `agents-a1` **no no-tools number is
published anywhere** — the card reports HLE only in its agentic table — so the honest value
is `null`. For `claude-opus-5` the with-tools figure is the only one on the cited
announcement page; a no-tools counterpart, if it exists, is in the system-card PDF and was
not extractable here.

---

## 4. How big the error is

The column ranks by percentile inside the Knowledge index, so the damage is positional:

| model | stored | pctile | corrected | pctile | shift |
|---|---:|---:|---:|---:|---|
| `claude-fable-5-1` | 65.0 | 100.0 | 60.9 | 98.6 | −4.1 pp |
| `claude-opus-5` | 64.7 | 99.3 | *unknown* | — | — |
| `deepseek-v4-1-flash` | 63.9 | 98.6 | 36.8 | 88.8 | **−27.1 pp** |
| `claude-sonnet-5` | 57.4 | 97.9 | 43.2 | 95.1 | −14.2 pp |
| `agents-a1` | 47.6 | 97.2 | *none published* | — | → null |

`deepseek-v4-1-flash` currently outranks `kimi-k3`, `qwen3-8-2-4t-a95b` and `glm-5-3` on this
column. On the same footing as those three it sits behind all of them.

The paired evidence for the size of the tool effect, everywhere a publisher gives both modes
for one model (25 pairs, HF cards + vendor blogs):

```
min +0.9   median +11.5   mean +12.7   max +27.1   (median ratio 1.35x)
```

For calibration, `_huggingface_mapping.py` already parks `CharXiv (RQ) (w/ python)` and
llm-stats' `charxiv_r` over a gap the module itself measures at "about 6 points". HLE's gap
is nearly twice that, on a column with nine times the weight.

---

## 5. How they got in

Three separate mechanisms, none of which is a one-off:

**5.1 — 23 spellings, one key, best-number-wins.**
`huggingface-benchmark-name-mapping.json` maps 23 labels onto `hle`, six of them explicitly
with-tools: `HLE (w/ Tools)`, `HLE (with tools)`, `HLE w/ Tools`, `HLE w/ tool`,
`HLE with search`, `HLE-Full (w/ tools)`. A seventh, `HLE-Verified¹`, is a different question
set (llm-stats runs it as its own benchmark, 3 models). When several labels alias one key,
`update.fetch_huggingface_data()` resolves the collision by keeping the **largest** value:

```python
# Several leaderboard labels can alias one llm.json benchmark; the
# best reported run wins rather than whichever label came first.
```

On a benchmark where tools are worth a median +11.5 points, "best run wins" is a rule that
systematically selects the with-tools number on every card that prints both. 17 cards in the
current corpus print both.

**5.2 — the structured `notes` field is read and discarded.**
Every one of these cards *says* which mode it ran. The Hub's eval metadata carries it
verbatim:

```json
{"dataset": {"id": "cais/hle"}, "value": 63.9,
 "notes": "With tools; harness not specified in the model card."}
{"dataset": {"id": "cais/hle"}, "value": 47.6, "notes": "With tools"}
{"dataset": {"id": "cais/hle"}, "value": 43.4, "notes": "Text-only, no tools."}
```

`fetch_huggingface.extract_eval_results()` reads `dataset` and `value` and drops `notes`, so
two entries that differ only in tool mode collapse to one label and are then separated by
`setdefault()` — i.e. by whichever order the Hub API happens to return them in.

**5.3 — llm-stats publishes one `hle_score` field over a mixed population.**
`fetch_llmstats.py` reads the flat `/leaderboard/models/full` endpoint, where HLE is a single
`hle_score` column. Behind it, the per-model endpoint carries an `analysis_method` string,
and across the 104 models that have an HLE score there it reads:

| `analysis_method` says | models |
|---|---:|
| with tools / with search | 31 |
| both modes, headline number is **with tools** | 7 |
| no tools | 29 |
| nothing about tools | 37 |

The field we ingest is a 38/29/37 blend of with-tools, no-tools, and unstated. This is the
same defect `llmstats-benchmark-name-mapping.json` already parks `charxiv_r` for — the note
there says llm-stats "serves the with-tools number for those models" — but `hle` is mapped.

**Why the five that are right are right is luck, not design.** The Ornith trio's cards print
`HLE (with tools)` values 7.8–11.5 points above what we store; they survived only because
`apply_score(fill_only=True)` had already seen the no-tools number on an earlier pass. A
first ingest landing after the card gained its second row would have taken the tools number.

---

## 6. What is exposed going forward

- **AA-covered models are safe.** `RANK_AA` is locked at the top of `_precedence.py` and both
  aggregates are fill-only, so no HF card or llm-stats row can displace an AA HLE value. 13
  cards currently carry a with-tools number above the stored AA value (up to +26.4 on
  `qwen3-5-35b-a3b`) and all 13 bounce off precedence.
- **Everything AA does not cover is exposed**, which is where all 5 bad values live. 14 models
  in the table have a null `hle` today; none is currently reachable from llm-stats, but the
  mapping is live and this table adds models weekly.
- The `hle` column description in `llm.json` documents the dataset, not the run, so nothing on
  the page tells a reader which of the five axes in §1 a given cell was measured on.

---

## 7. Options

Ordered by how much they buy:

1. **Park the six with-tools labels and `HLE-Verified¹`** in
   `huggingface-benchmark-name-mapping.json` (`__unmappable__`), exactly as the Zerobench and
   CharXiv tool-mode variants are parked. Stops mechanism 5.1 at the source and needs no code
   change. This is the minimum.
2. **Correct the five values.** `deepseek-v4-1-flash` → 36.8, `claude-sonnet-5` → 43.2,
   `claude-fable-5-1` → 60.9, `agents-a1` → `null`. `claude-opus-5` needs the system-card
   figure or `null`. Hand-entered values sit at `RANK_HAND_ENTERED`, so a later AA run
   overwrites them cleanly.
3. **Park llm-stats' `hle`**, on the `charxiv_r` precedent: one field over a 38/29/37 blend
   cannot be read as a no-tools number. Costs the `longcat-flash-thinking` value (which
   happens to be right) and closes the channel that produced three of the five bad ones.
4. **Teach the HF ingest to read `notes`** — carry it as a label qualifier so
   `cais/hle` + "With tools" and `cais/hle` + "No tools." stop colliding. This is the only
   option that fixes the general case rather than the labels we have seen; it also unblocks
   the same problem on AIME, GPQA and CharXiv.
5. **Restate the column.** Change `benchmarks.hle.description` to say what the values are —
   text-only subset, no tools, pass@1 — and note the foreign-harness minority, the way the
   README already does for CharXiv ("the reasoning split, without tools").

A second column (`hle_tools`) is *not* recommended: only 5 of 158 models would have a value
AA does not also publish no-tools, and AA does not run HLE with tools at all, so the column
would be a self-report board with no spine.

---

## 8. Resolution — what was changed

All of §7 was applied except the `hle_tools` column, which stays unbuilt for the
reason given there.

**The five values (§3.2).** Each now stores its publisher's no-tools figure, or nothing:

| model | was | now | source of the new value |
|---|---:|---:|---|
| `claude-fable-5-1` | 65.0 | **60.9** | vendor table, "60.9% no tools" |
| `claude-opus-5` | 64.7 | **56.6** | same table, Opus 5 column — see below |
| `claude-sonnet-5` | 57.4 | **43.2** | Sonnet 5 launch page, "Without tools: 43.2%" |
| `deepseek-v4-1-flash` | 63.9 | **36.8** | card's own `HLE (Pass@1)` row |
| `agents-a1` | 47.6 | **null** | no no-tools run is published anywhere |

`claude-opus-5`'s figure was recovered from the Fable 5.1 announcement's
four-column comparison table (`Fable 5.1 | Fable 5 | Opus 5 | GPT-5.6 Sol`),
whose column order is confirmed independently by the prose above it quoting the
Terminal-Bench-Science leaderboard for columns 2 and 3. That table also restates
Fable 5 and Opus 5 below the figures llm-stats still carries from their own
launch pages (63.8 / 63.6 with tools against 64.5 / 64.7), which is Anthropic
re-reporting after a grader change — the same note they published for Sonnet 4.6.

The column's top five is no longer the five with-tools values:

```
before: fable-5-1 65.0 | opus-5 64.7 | deepseek-v4-1-flash 63.9 | sonnet-5 57.4 | agents-a1 47.6
after:  fable-5-1 60.9 | opus-5 56.6 | kimi-k3 46.9 | ornith-1-5-397b 44.6 | hy4-preview 43.4
```

**Door one — labels (§5.1).** The six with-tools spellings and `HLE-Verified¹`
are `__unmappable__` in `huggingface-benchmark-name-mapping.json`; the bare
`hle` is parked in `llmstats-benchmark-name-mapping.json`. `HLE w/ CoT` stays
mapped — chain of thought is not a tool.

**Door two — the Hub's `notes` (§5.2).** `fetch_huggingface.py` now folds the
tool mode into the label for the datasets in `TOOL_MODE_SENSITIVE_DATASETS`
(currently `cais/hle` alone), splitting the ambiguous id into
`cais/hle (no tools)` (mapped) and `cais/hle (with tools)` (parked). Restricted
to a named set on purpose: a note is free text, so a general rule would mint a
label per phrasing and flood the mapping queue, and tools are the *point* of an
agentic benchmark. A note describing both runs ("With tools: 57.4%. Without
tools: 43.2%") claims neither mode, so the entry stays unqualified.

Re-running the full corpus through the new ingest moves eleven cards and adds
exactly two labels, both already answered, so nothing is queued:

```
agents-a1 47.6→none   glm-5-3 62.5→none      glm-5-3-flash 55.3→none
step-3-7-flash 48.1→none   mimo-v2-5-pro 48.0→none
kimi-k2-thinking 44.9→23.9   kimi-k2-5 50.2→30.1   gemma-4-31b 26.5→19.5
gemma-4-26b-a4b 17.2→8.7   gpt-oss-120b 11.3→8.6   gpt-oss-20b 8.8→7.0
```

**Door three — llm-stats (§5.3).** `fetch_llmstats.py` no longer publishes the
bare `hle` field. It re-reads HLE per model from the detail endpoint, where
`analysis_method` states the mode, and emits `hle (no tools)` only where it can
verify one: a both-modes note has its without-tools figure parsed out and used,
a no-tools note keeps the headline, and a with-tools or silent note is dropped.
On the live board that keeps **36 of 104** — including `claude-sonnet-5` at
0.432, recovered from a both-modes note, and `deepseek-v4.1-flash` at 0.368 —
and drops 68. The lookup costs one request per scored model, so callers that
only need ids or licences pass `--no-hle-detail`.

**What this does not close.** An unqualified `cais/hle` stays mapped, and one
card in the corpus abuses that: Kimi K3 publishes 56.0 under the bare id with no
note, which llm-stats' own `analysis_method` identifies as "HLE-Full with
tools", against the 43.5 the same card prints as `HLE-Full`. Nothing in the HF
payload distinguishes the two, so the best-value rule still picks 56.0. It is
inert — AA scores kimi-k3 at 46.9 and the HF ingest is fill-only — and the
alternative, refusing every `cais/hle` that does not state a mode, would throw
away some 50 correct values to catch it. Recorded here rather than fixed: if a
model ever depends on the bare id for its HLE value, this is the case to check.

**AA (§2, and the requirement that it keeps the column).** No change was needed
and none was made — AA is `RANK_AA`, the ingest is not fill-only, and rank
blocks a write only from a *strictly* better source, so AA over AA is allowed
and a regrade lands on the next refresh. What was missing was anything holding
that true. `test_hle_no_tools.py` now pins it, along with all three doors, and
runs in the update workflow's gate.

No version number is pinned anywhere. The column tracks whatever AA's current
HLE implementation says, which is what let the v4.1.1 grader change land without
anyone doing anything, and the description says so rather than naming a version
that would go stale.

**The column description.** Now states the tool mode, the size of the effect,
and that with-tools runs are refused rather than blended — and cites AA's
methodology page alongside lastexam.ai.

---

## 9. Provenance pass — does the cited page say what the column measures?

The tool-mode work above asks what a *label* means. This pass asks the prior
question of every score in the file: **does the page it is cited to still carry
that number, under a label the column's main source would recognise?** Method:
match each of the 341 scores cited to a Hugging Face card back to the label on
that exact card carrying that value, then read the column's main-source
methodology (AA's page, or the benchmark's board) and compare.

28 failed. Two corrections to the tool-mode list came out of it first:

- **SciCode was backwards.** AA "test[s] with scientist-annotated background
  information included in the prompt" and "report[s] sub-problem level scoring",
  so `SciCode (subtask)` and `SciCode (wbg)` *match* the column and the plain
  `SciCode` label is the loose one. Nothing was parked.
- **`avg@k` is not a different measurement.** AA scores AIME 2025 as "pass@1
  with 10 repeats" and τ²-Telecom as "pass@1 as the average of 3 attempts", so
  `AIME25(avg@32)` and `τ²-Bench (telecom) avg@4` are the same estimator at a
  different *n*.

### 9.1 Corrected (5)

| model | column | was | now | why |
|---|---|---:|---:|---|
| `qwen3-5-397b-a17b` | `browsecomp` | 78.6 | **69.0** | 78.6 was the with-context half of the card's `69.0/78.6` pair |
| `apriel-v1-6-15b-thinker` | `swe_bench_verified` | 16.0 | **23.0** | 16 is the **Apriel-1.5** column; the card's header puts Apriel-1.6 first |
| `deepseek-v4-1-flash` | `hle` | 36.8 | **39.1** | the card's `†` figure: the text-only subset, which is AA's question set |
| `hy4-preview` | `gdpval_aa`, `critpt`, `mcp_atlas` | — | — | values right, re-cited to the llm-stats page that carries them |
| `glm-5-3-flash` / `deepseek-v4-flash` | `deepswe_1_1` / `terminal_bench_2_0` | — | — | same: re-cited from a card that does not carry the number |

### 9.2 Nulled (23)

No valid measurement exists for these, so the cell is empty rather than wrong:

- **A different question set.** `minicpm5-2b` browsecomp (`Top100`, 100 of
  1,266), `lfm2-5-2-6b` browsecomp (`BrowseComp+`, another dataset, reported
  0-1), `a-x-k2` browsecomp (`≤10 searches`), `nanbeige4-1-3b` aime_2026
  (`AIME 2026 I`, one of the two papers — and the stored 81.5 was a competitor
  column; the model's own figure is 87.4).
- **A different budget.** `deepseek-v4-flash-vision` and `deepseek-v4-1-flash`
  zerobench, both pass@5 where the column is main-set pass@1 — and the DeepSeek
  row is `ZeroBench-main w/ tools (Pass@5)`, run on the Claude Code harness.
- **Tools on.** `sarvam-30b` and `sarvam-105b` aime_2025 (`AIME 25 (w/ Tools)`).
- **A different metric.** `deepseek-v4-1-flash` mmlu_pro (`(EM)` against AA's
  10-option regex extraction); `granite-4-2-3b/8b/30b` and
  `nemotron-3-5-lightning` ifbench (`(prompt)` / `(loose)`, which put an 8B
  granite above Qwen3.5-397B — not the score AA reports).
- **A different harness entirely.** `mimo-v2-5-pro` aime_2025 (`AIME 24&25`,
  2-shot, off a base-model pretraining table); `agents-a1` scicode (the card's
  own agentic framework).
- **Nowhere on the cited page, nor on any board.** `agents-a1`
  tau2_bench_telecom (the card has no τ² row at all), `hy4-preview`
  swe_atlas_rf/tw/qna, `hy3` livecodebench, `qwen3-coder-30b-a3b-instruct`
  swe_bench_verified and terminal_bench_2_0.
- **Right number, wrong version.** `nvidia-nemotron-3-super-120b-a12b`
  livecodebench: llm-stats has 81.2, noted "v5 2024-07 to 2024-12"; the column
  is v6.

Coverage cost is 23 values across 14 columns — one or two each, except
zerobench (13 → 11).

### 9.3 Kept, with the reason stated

Not every difference is an error. A **harness** difference on SWE-bench or
Terminal-Bench has no neutral alternative — every published number is
harness-conditioned, vals.ai's own included — so `nemotron-3-super` and
`seed-oss-36b-instruct` keep their OpenHands numbers. Likewise BrowseComp
context management (`inkling`, `inkling-small`, `agents-a1`): the whole column
is scaffold-varied, and scaffolding is not a different question set. These stay
as the column-wide caveats §1 records, not as per-value fixes.

### 9.4 What was changed so the fixes hold

The HF ingest is fill-only, so a nulled cell is refilled on the next refresh
unless the label that produced it is parked. Eleven were:
`ZeroBench (Pass@5)` (its twin `ZeroBench (pass@5)` was already parked — a
capitalisation escape), `BrowseComp Top100`, `BrowseComp+ (OpenClaw)`,
`open-agent-leaderboard/results (browsecomp_plus)`, `BrowseComp (≤10 searches)`,
`AIME 25 (w/ Tools)`, `AIME25 (with tools)`, `AIME 2026 I`, `MMLU-Pro (EM)`,
`IFBench (prompt)`, `IFBench (loose)`.

Two `fetch_llmstats.py` changes were needed for the same reason:

- **The tool-mode gate now covers every no-tools column llm-stats feeds**, not
  just HLE. AIME 2025, GPQA, MMMU-Pro and SciCode lose only the entries whose
  `analysis_method` says tools were used — proportionate, because on those the
  no-tools run is the default and tools are the labelled exception, unlike HLE
  where the tools headline is the norm. Today that rejects 9, including the
  96.7 llm-stats would otherwise have put back on both Sarvam models. A code
  interpreter counts as a tool, which is what catches MMMU-Pro "w/ python".
- **llm-stats' own exact board now wins.** It runs
  `humanity's-last-exam-(no-tools,-text-only)` — no tools, text-only subset,
  which *is* this column — so where a model is on it that score is used instead
  of the flat field, which is the full multimodal set even when its note says
  no tools. Published HLE coverage goes from 36 to 42 models, and
  `deepseek-v4-1-flash` resolves to 39.1, matching the hand fix above. The
  board is read even where the flat field is null, which costs nothing: the
  detail request has already been made.

`--no-hle-detail` now drops the gated columns rather than passing them through
ungated, so the cheap path cannot become a second door.

---

## Appendix A — method

- Values, sources and dates read from `llm.json` at `6cff0ea` (144 non-null `hle` values,
  `scores_source` resolved through the `sources` index).
- AA configuration from `https://artificialanalysis.ai/methodology/intelligence-benchmarking`
  (Intelligence Index v4.3.1, "HLE (Humanity's Last Exam)" and Version History sections).
  Live AA values spot-checked for all 11 pre-August-2026 stamps via the embedded JSON on
  `https://artificialanalysis.ai/models/<slug>`.
- HF card values from a fresh `fetch_huggingface.py --all-models --format json` run
  (155/158 repos parsed; 3 gated repos return 401). Tool mode read from the Hub's
  `evalResults[].data.notes` via `/api/models/<repo>?expand[]=evalResults`, and from the card
  README where the metadata is silent.
- llm-stats tool mode from `https://api.zeroeval.com/leaderboard/models/<id>`, field
  `benchmarks[].analysis_method`, pulled for all 104 models carrying an HLE score. The
  ingested `/leaderboard/models/full` endpoint does not expose it.
- Vendor figures from the pages llm-stats cites as `self_reported_source`.
  `claude-opus-5`'s no-tools figure could not be recovered: its announcement page carries
  only the with-tools number and the system card is a PDF no tooling here could read.
- The 25 paired with/without observations in §4 are every model in the corpus for which one
  publisher states both modes.
