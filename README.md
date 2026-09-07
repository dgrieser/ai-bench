# AI Benchmark Aggregator (`ai-bench`)

A comprehensive system for collecting, normalizing, and aggregating LLM benchmark scores across 15+ benchmark sources. This tool creates a unified dataset of AI model performance metrics from diverse evaluation platforms.

## Project Overview

`ai-bench` solves the problem of fragmented AI model benchmarking: different organizations run benchmarks independently, each with their own model naming conventions, scoring formats, and update schedules. This project aggregates all these scores into a single, normalized JSON file (`llm.json`) where:

- **Each model** is represented once with a canonical slug identifier
- **All benchmark scores** from all sources are unified under that model
- **Model name mappings** handle the fact that different benchmarks call the same model by different names
- **Metadata** about sources, timestamps, and weights is preserved

## Benchmark Sources Integrated

| Source | Type | Data Format |
|--------|------|-------------|
| **Artificial Analysis** | Commercial API | HTTP endpoint |
| **AA Coding Agent Index** | Commercial (AA agents leaderboard) | RSC page payload |
| **Hugging Face** | Community | Model card READMEs + Hub eval metadata (evalResults / model-index) |
| **DeepSWE** | Research (benchlm.ai mirror) | JSON API |
| **FrontierSWE** | Research | RSC flight payload |
| **SWE Atlas** | Research | JSON API |
| **MCP-Atlas (Scale Labs)** | Research (benchmark's own leaderboard) | RSC flight payload |
| **SWE Marathon** | Research | JS bundle (leaderboard literal + trial log) |
| **OSWorld** | Research | JSON API |
| **Spheron** | Infrastructure | JSON API |
| **LLMStats** | Community Aggregator | JSON API |
| **Evals Report** | Research | JSON API |
| **FrontierCode** | Research (Cognition) | Static leaderboard JSON |
| **DeepSWE (Datacurve)** | Research (benchmark's own site) | One versioned JSON artifact per revision |
| **BFCL (Berkeley/Gorilla)** | Research (benchmark's own leaderboard) | CSV the page hydrates from |
| **Terminal-Bench** | Research (benchmark's own leaderboard) | RSC flight payload |
| **Agents' Last Exam (Berkeley RDI)** | Research (benchmark's own leaderboard) | JSON API |

## Core Data Structure

### `llm.json` Format

```json
{
  "benchmarks": { "deepswe": { "name": "DeepSWE", "urls": [ ... ], ... }, ... },
  "models": [
    {
      "name": "devstral-2",
      "date_added": "2026-02-03",
      "url": "https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512",
      "params": "123B",
      "context": "256k",
      "creator": { "name": "Mistral", "url": "https://mistral.ai/..." },
      "scores":         { "deepswe": 92.3, "swe_bench_verified": 72.2, ... },
      "scores_updated": { "deepswe": "2026-07-25", "swe_bench_verified": "2026-02-13", ... },
      "scores_source":  { "deepswe": "https://benchlm.ai/benchmarks/deepSwe", ... },
      "vram": { "fp16": 273, "int8": 136, "int4": 68 }
    }
  ],
  "sources": [ "https://...", ... ]
}
```

Each model object contains:
- **`name`**: Canonical slug (used across the system)
- **`scores`**: Flat map, benchmark key → score (null until a source reports one)
- **`scores_updated`**: Same key set → ISO date the score last changed. `llm.html`
  sets a score stamped within the last seven days in red, the same red a new
  model's name gets. Two cases are left unmarked: a model added inside the same
  window, which carries the `NEW` badge instead because every one of its scores
  arrived with it, and a derived column, whose date moves when its inputs are
  recomputed rather than when anything new is measured
- **`scores_source`**: Same key set → URL of the page the score was read from
  (null for hand edits; a derived index column cites this repository, which is
  where it is computed)
- **`date_added`**: ISO date the index first listed the model. It drives the `NEW`
  badge and the **Recently Added** panel, and it is what `llm.html`'s **Date
  Added** filter narrows on — its relative windows count calendar days back from
  today, so "Last 7 days" holds exactly the models the table badges as new
- **`params`** / **`context`** / **`vram`** / **`creator`**: model metadata

The three score maps carry the full benchmark key set with null placeholders;
`update.py` stamps date and source URL together whenever it writes a score.
Values no scraper wrote (hand edits, rows that have since moved) keep a null
date or source; `fill_missing_source_urls.py` asks for those, plus a missing
`vram_source`, model or creator URL.

Underneath the table, `llm.html` writes those same seven days out as a
**Recently Added** panel: a timeline of the days something landed, each holding
the models added that day and the individual scores stamped that day on models
already in the index. Either way the scores are spelled out the same, one chip
per benchmark carrying its name, its value and a link to the source it was read
from; a new model's chips follow its arrival count, since every one of them
arrived with the model. Membership is decided by the same tests as the marks in
the table, over the models and columns the table is currently showing, so the two
cannot disagree — including the one thing they both leave out, a derived column,
whose date moves when its inputs are recomputed rather than when anything new is
measured. A day longer than twelve models folds its tail behind a disclosure,
since one fetch can restamp a hundred rows at once. Everything reads as *added*:
a date records only the last write, so a score measured a second time cannot be
told from a first one.

### Score Precision

Every score is rounded onto a per-benchmark grid before it is stored, by
`_scores.round_score()`, called from the one place each writer funnels through:
`apply_score()` in `update.py` for the scrapers, `collect_updates()` in
`edit.py` for hand edits. Two fields on the benchmark set the grid:

| Field | Default | Meaning |
| --- | --- | --- |
| `decimals` | `1` | Digits `llm.html` and `llm-cli` print |
| `round_to` | `10 ** -decimals` | Step a stored value is snapped to |

The default exists because sources disagree about precision: one leaderboard
reports `48.77` where another reports `48.8`, and the site prints
both as `48.8`. Stored raw, that disagreement rewrites the score and restamps
its date on every refresh for a change no reader can see. Rounding in one place
means a score means the same thing whichever source produced it, and a refresh
is a no-op unless the printed number actually moved.

`round_to` is the escape hatch for a benchmark that moves on its own.
**GDPval-AA** is the one that needs it: it is an Elo (1000 = human expert)
spanning roughly -120 to 1740, and Artificial Analysis re-anchors the whole
field by a point or two whenever it re-runs the pairwise judging — in the
history of this file, refreshes that moved all 62 scored models by a mean
-1.5 Elo and then straight back by +1.5. At `"decimals": 0, "round_to": 5` a
value is recorded in 5-Elo steps, below the movement the metric shows on its
own and far below the 21-Elo median gap between neighbouring models, which
takes about three quarters of that churn out of the file.

Rounding is quantization, not a filter on how much a score has to move to be
worth writing: it is idempotent and has no memory, so re-running the pipeline
over any starting file yields the same numbers. Halves round away from zero
rather than to even, and an integral result is stored as an int (`34`, not
`34.0`) — the shape the rest of `llm.json` already uses.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              CLI Commands & Entry Points                 │
├─────────────────────────────────────────────────────────┤
│  add.py  │ edit.py  │ update.py  │ prune.py  │ etc.     │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│           Mapping & Data Transformation Layer            │
├─────────────────────────────────────────────────────────┤
│  _*_mapping.py files: normalize model names             │
│  update_*_mapping.py: sync mappings from sources        │
│  _scores.py: score rounding, timestamps, write logic    │
│  _openness.py: model openness classification            │
│  _params.py / _context.py: model size & window fields   │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│          Data Fetchers for Each Benchmark               │
├─────────────────────────────────────────────────────────┤
│  fetch_huggingface.py     │ fetch_deepswe.py            │
│  fetch_frontierswe.py     │ fetch_osworld.py            │
│  fetch_swe_atlas.py       │ fetch_swe_marathon.py       │
│  fetch_spheron.py         │                             │
│  fetch_llmstats.py        │ fetch_evals_report.py       │
│  fetch_frontiercode.py    │ fetch_datacurve.py          │
│  fetch_mcp_atlas.py       │ fetch_bfcl.py               │
│  artificialanalysis.py                                  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│               External Benchmark APIs                    │
├─────────────────────────────────────────────────────────┤
│  huggingface.co  │ Various research APIs  │ Others      │
└─────────────────────────────────────────────────────────┘

Output: llm.json (unified dataset)
        + model-name-mapping-*.json files (slug mappings)
        + benchmark-name-mapping.json files (benchmark aliases)
```

## Command Reference

### Data Collection & Synchronization

```bash
# Update all scores from all configured sources
./update.py llm.json

# One-time backfill of models[].scores_source: attribute stored scores to the
# first source (in the usual update order) whose current value matches, where
# no URL is stored yet. Dry-run first, then persist with -w.
./update.py --fill-source-urls
./update.py --fill-source-urls -w

# Fetch from specific benchmarks
./fetch_aa_coding_agents.py              # SWE-Atlas-QnA, Terminal-Bench 2.1 as run by AA
./fetch_huggingface.py --repo owner/model-name
./fetch_deepswe.py                      # DeepSWE, mirrored by benchlm.ai
./fetch_frontierswe.py
./fetch_osworld.py
./fetch_spheron.py
./fetch_swe_atlas.py
./fetch_swe_marathon.py                 # both published boards, 1.0 and 1.1
./fetch_swe_marathon.py --revision 1.1  # or pin one
./fetch_mcp_atlas.py                    # MCP-Atlas, from Scale's own leaderboard
./fetch_bfcl.py                         # BFCL v4 Overall Accuracy, from the Gorilla team
./fetch_llmstats.py
./fetch_evals_report.py
./fetch_datacurve.py                    # DeepSWE, from the benchmark's own site
./fetch_datacurve.py --all-configs      # every harness/effort row, not the best
./fetch_datacurve.py --revision 1.0     # one revision instead of every published one
./fetch_frontiercode.py                 # every revision, each row labelled
./fetch_frontiercode.py --revision 1.0  # or pin one revision
./fetch_tbench.py                       # Terminal-Bench 4.0, from the benchmark's own board
./fetch_agents_last_exam.py             # Agents' Last Exam, Overall Pass Rate
./fetch_agents_last_exam.py --split full/last-exam   # or another tier, on its own scale

# Update model name mappings from source APIs
./update_aa_coding_agents_mapping.py
./update_artificialanalysis_mapping.py
./update_bfcl_mapping.py
./update_deepswe_mapping.py
./update_frontierswe_mapping.py
./update_tbench_mapping.py
./update_agents_last_exam_mapping.py
./update_frontiercode_mapping.py
./update_huggingface_mapping.py
./update_llmstats_mapping.py
./update_mcp_atlas_mapping.py
./update_osworld_mapping.py
./update_spheron_mapping.py
./update_swe_marathon_mapping.py
```

### Model Management

```bash
# Add a new model to llm.json (interactive CLI)
./add.py --json llm.json

# Edit existing model entries
./edit.py --json llm.json [model-slug]

# Remove models or prune invalid entries
./prune.py llm.json

# Synchronize score update timestamps and per-score source URLs
./sync_score_dates.py llm.json
```

### Utilities

```bash
# Recompute the derived index columns, Coding, Tooling, Knowledge, Vision and
# Trust
# (dry-run; -w to persist)
./derive_indexes.py llm.json
./derive_indexes.py llm.json -w

# Fill in missing source URLs for benchmark records
./fill_source_urls.py llm.json

# Ask for the dates and source URLs nothing can derive (dry-run; -w to persist)
./fill_missing_source_urls.py llm.json
./fill_missing_source_urls.py llm.json -w
./fill_missing_source_urls.py llm.json --list                  # report the gaps, ask nothing
./fill_missing_source_urls.py llm.json -m kimi-k3 -w           # one model only
./fill_missing_source_urls.py llm.json --only vram-source -w   # one kind of gap only

# Answer the queued questions directly, no proposal PR
#   Dry-run by default; -w to persist. All or nothing per batch.
./answer.py tbench "Fable 5.1" __unmappable__ -w
./answer.py --stdin -w < answers.json

# Render the queue the last unattended run left behind
#   markdown for the proposal PR, json for answer.py and the admin page
./pending_prompts.py pending-prompts.jsonl --format markdown
./pending_prompts.py pending-prompts.jsonl --format json --out _pending/pending.json

# Check for newly added/dismissed models
#   [y] add it, [n] never ask again, [q] stop asking. Every run first carries
#   out whatever check_new-decisions.json already says -- see below.
./check_new.py

# Assess model openness (open-source vs closed-weight)
python3 -c "from _openness import open_index; open_index('llm.json')"
```

## Key Workflows

### 1. Adding a New Benchmark Score to an Existing Model

1. **Fetch** the data from the benchmark source (`fetch_*.py`)
2. **Map** the benchmark's model name to canonical slug (via `*_mapping.py`)
3. **Merge** into the model's `scores` dict (handled by fetch scripts)
4. **Update timestamp** (automatic via `stamp_score_updated()`)

### 2. Adding a New Model

```bash
./add.py --json llm.json
```

Interactive prompt guides you through:
- Model name and aliases
- Weights/licensing info
- Openness classification
- Optional manual score entries

### 2b. Deciding a New Model Unattended

`check_new.py` asks one question per newly released Artificial Analysis model:
add it, or never offer it again. The scheduled GitHub Actions run cannot answer
either, so it queues the question and `propose.py` turns it into a PR that
carries **both** answers:

- the model itself, added to `llm.json` with metadata prefilled from AA and null
  scores — merge as-is and that is the "add";
- a line in `check_new-decisions.json`:

  ```json
  { "some-new-model": "__added__" }
  ```

  Flip it to `__ignored__` — the PR attaches a clickable suggestion for exactly
  that — and the next run removes the entry from `llm.json` again and records
  the slug in `check_new-dismissed.json`, so AA never offers it.

Both are applied by `apply_decisions()` (`_new_models.py`) at the top of the
next `check_new.py` run, *before* any score is fetched, so an ignored model is
gone before a mapping or a score can attach to it. The entry is dropped once
acted on; a value that is neither sentinel is left in place and reported rather
than guessed at.

### 2c. Answering Without a Pull Request

The proposal PR is one way to answer the queue. `answer.py` is the other: it
applies a batch of answers directly, through each source's own writer, with no
PR anywhere in it.

```bash
# One answer, by hand
./answer.py tbench "Fable 5.1" __unmappable__ -w

# A batch, as JSON
./answer.py --stdin -w < answers.json
```

Dry run by default, like everything else here, and a batch is all or nothing —
one bad record and nothing is written, because half an answered queue is the
outcome nobody could reconstruct afterwards.

The record shapes live in `_answers.py`, which reuses `propose.ROUTES` and
`build_universes()` so the answer space is exactly the proposal space. It
assumes its input came from outside the repository and refuses rather than
sanitises: a route is a key of the hard-coded table and never a module or a
path, `__pending__` is refused everywhere (it is a parking marker, and writing
it over a decision re-queues that name for good), an empty universe refuses
rather than admits, and only a question the queue actually asked can be
answered at all.

Two shapes are worth knowing because the obvious version is wrong:

- **Adding a model is `model-add`, not `{"answer": "__added__"}`.**
  `apply_decisions()` keeps `__added__` only for an entry already in `llm.json`;
  for a slug that is not, it clears the line without dismissing it, and
  `check_new.py` offers the model again on the next run and every run after
  that. `model-add` runs `add.py` and *then* records `__added__`, which is the
  pair `propose.py` already uses.
- **The Artificial Analysis mapping runs the other way round.** Its keys are
  `llm.json` model names and its values are AA slugs, one or a list, so both
  ends are checked against different universes.

### 2d. The Admin Page

`_admin/` is the same queue as a web page, for answering from somewhere that is
not a terminal. It renders `_pending/pending.json`, you tap an answer, and it
dispatches `update-benchmarks.yml` with the batch as an input; the run applies
it with `answer.py` and pushes to `main`.

It is not on the published site. Pages runs Jekyll here (there is no
`.nojekyll`), and Jekyll does not copy `_`-prefixed paths into the built site —
already why `_matching.py` and its siblings 404 there. `test_answers.py` asserts
the `.nojekyll` stays absent, since adding one would publish both `_admin/` and
`_pending/`.

Deployment, the token's scope and why it is scoped that way are in
`_admin/README.md`.

### 3. Updating All Benchmarks

```bash
./update.py llm.json
```

This orchestrates:
1. Runs all `fetch_*.py` scripts
2. Runs all `update_*_mapping.py` scripts
3. Merges results into `llm.json`
4. Updates timestamps

### Source precedence

Most columns have more than one publisher — 32 of them carry values from two or
more sources — so every refresh has to decide whose reading lands. Each source
is declared with a **rank** in `_precedence.py`, and `apply_score()` refuses a
write whose source is outranked by the one already credited with the stored
value. Rank is matched against the page in `scores_source`, so the decision is
made from what is in the file rather than from what ran, and the result does not
depend on which ingests ran or in what order — the property `keep_best_row()`
already gives row collisions inside a single source.

| Rank | Source | Why there |
| --- | --- | --- |
| 1 | **Artificial Analysis** (`artificialanalysis.py`, API + model pages) | First-party runs of one harness across the whole field, and the leading source for 21 columns. Locked: only a later AA number replaces an AA number. |
| 2 | **The benchmark's own leaderboard** — Toolathlon, Scale (MCP-Atlas, SWE-Atlas), Gorilla BFCL, OSWorld, DeepSWE/Datacurve, FrontierSWE, Cognition FrontierCode, SWE-Marathon, Terminal-Bench, Agents' Last Exam | First-party for the column it publishes. No two members publish the same column, so their relative order is unobservable and none is declared. A board publishing several revisions of itself is first-party for each of their columns. |
| 3 | **Curated third parties** — evals.report, benchlm.ai | evals.report keeps only Official and Verified rows (`TRUSTED_STATUSES`); benchlm.ai has no status of its own but is a compiler of results rather than a lab reporting on itself. |
| 4 | **AA Coding Agent Index** (`fetch_aa_coding_agents.py`) | AA-published, but AA's *own harness* over someone else's benchmark, and it disagrees systematically with that benchmark's board — so it does not inherit rank 1. Fill-only, so it reaches a column only where it is still null. Its DeepSWE rows are not ingested at all: see [Benchmarks that publish more than one revision](#benchmarks-that-publish-more-than-one-revision). |
| 5 | **Cross-benchmark aggregates** — llm-stats, Hugging Face model cards | Republished numbers nobody in the chain ran. Both fill-only; where they overlap, llm-stats runs first and so claims the gap. |
| 6 | **Hand entries** (`add.py`, `edit.py`) | Whatever page the entry cited, or null where a hand edit cleared the attribution. A hand entry seeds a column until something measures it, and any scraper may overwrite it. |

Two sources on the same rank may still overwrite each other, which is what lets
a source refresh its own value: rank blocks a write only when the stored value
came from a *strictly* better-ranked source.

The rungs are where they are because of what actually disagrees. Where both a
first-party run and a self-report exist for one model they differ by a point or
two, and `mcp_atlas`, `bfcl_v4`, `frontiercode_1_1`, `swe_marathon_1_1` and
`deepswe_1_1` each carry both, so the run has to be the one that lands. IFBench and MMLU-Pro
are the two columns where AA and evals.report both measure and the ranking
therefore changed an outcome: before it, precedence was the order `main()`
happened to call the ingests, so a run that skipped evals.report left AA's
numbers and the next full refresh replaced them — a 17-value round trip
(commits `2ab9ab0`…`0c24cc9`). SWE-bench Multimodal, ZeroBench and MathVista-mini
sit the other way round and need no exception: nothing first-party is scraped
for them, so evals.report leads at rank 3 unopposed, with the model cards
filling gaps beneath it.

### Tool Use and Instruction Following

Three columns cover function calling, MCP tool use and instruction following.
None has a single publisher, so each is assembled from the sources that measure
it, in the precedence order above:

| Column | Leading source | Gap fillers |
| --- | --- | --- |
| **BFCL v4** (`bfcl_v4`) | `fetch_bfcl.py`, the Gorilla team's own leaderboard (`data_overall.csv`, Overall Accuracy) | evals.report's `bfcl` table, then Hugging Face model cards |
| **MCP-Atlas** (`mcp_atlas`) | `fetch_mcp_atlas.py`, Scale's own leaderboard runs | evals.report's `mcp-atlas` table, llm-stats' `mcp_atlas` column, then Hugging Face model cards |
| **IFBench** (`ifbench`) | Artificial Analysis, which runs Ai2's benchmark itself (`ifbench` in the API's evaluations, with the model page as fallback) | evals.report's `ifbench` table, then Hugging Face model cards |

Two version traps are handled in the benchmark-name mappings rather than by
hoping the labels agree:

- **BFCL v4 is its own series.** V4 grew the benchmark from tool calls to
  agentic evaluation, so `huggingface-benchmark-name-mapping.json` maps only the
  v4-labelled aliases (`BFCL v4`, `BFCL-V4`, `BFCLv4`) onto `bfcl_v4` and leaves
  `BFCL v3`, `BFCL(avg v1&v2)` and a bare `BFCL` unmapped — the same rule
  Toolathlon-Verified gets.
- **MCP-Atlas is not MCPMark.** The `MCP Atlas` / `MCP-Atlas (Public Set)`
  spellings map onto `mcp_atlas`; `MCPMark` and `MCP Mark Verified` are a
  different benchmark and stay unmapped.
- **Terminal-Bench is four columns, and only a versioned label picks one.**
  `terminal_bench_4_0`, `terminal_bench_2_1`, `terminal_bench_2_0` and
  `terminal_bench_hard` are separate series — 4.0 alone removed 8 tasks and
  changed the resource budgets — so the alias file maps only labels that name a
  version (`Terminal-Bench 4.0`, `harborframework/terminal-bench-4.0
  (terminalbench_4)`, `Terminal Bench 2.1 (Terminus-2)`, …) and leaves a bare
  `Terminal-Bench`, `TerminalBench<sub>(acc)</sub>` and the whole 3.0 family
  unmapped. llm-stats' `terminal_bench` column is unmapped for the same reason
  and one more: its field tops out at Claude Sonnet 4.5, so whatever series it
  is, it is not the current one.
- **Agents' Last Exam is one tier of several.** The stored column is the Overall
  Pass Rate, so `Agents' Last Exam` maps onto `agents_last_exam` and
  `Agents' Last Exam (ALE-CLI)` — the Linux-only subset — does not.

All three feed the [Tooling index](#tooling-index); the weight each carries, and
why, is in that section's table.

### Benchmarks that publish more than one revision

Three benchmarks in this table have re-run themselves, and in every case the
re-run changed the task set, the verification or the scoring — so a 1.0 number
and a 1.1 number are two different measurements that happen to share a name.
Each revision gets its own column, the way `terminal_bench_2_0` and
`terminal_bench_2_1` are already separate. `_revisions.py` owns the naming
(`<benchmark>_<major>_<minor>`) so the scrapers, the ingest and the mappings
cannot disagree about which column a row belongs in.

| Benchmark | Columns | What changed between them |
| --- | --- | --- |
| **DeepSWE** | `deepswe_1_1`, `deepswe_1_0` | Datacurve serves one JSON artifact per revision and toggles between them. The re-run moved DeepSeek V4 Pro from 7.5 to 62.8, and 1.0 is the only revision that ever scored the dozen models retired before it. |
| **FrontierCode** | `frontiercode_1_1`, `frontiercode_1_0` | Cognition's payload carries a block per revision; the current one covers only the models it re-ran. GLM 5.2 scores 19.2 at 1.0 and 24.5 at 1.1. |
| **SWE-Marathon** | `swe_marathon_1_1`, `swe_marathon_1_0` | 1.1 updated all 20 tasks with tighter verification and closed-internet execution. The site states it reuses no 1.0 score for the updated tasks, and its leader sits 21 points above the archive's. |

Only the current revision feeds the [Coding index](#coding-index), as one
member at one weight. Admitting both would count the benchmark twice for
whoever was re-run and once for everyone else, and an archived revision's
percentile ranks a model against a field that no longer exists.

That would leave a hole, though: a model measured **only** on the retired board
contributes nothing to the benchmark it was actually measured on, so the index
imputes it at the median — which flatters a model that scored near zero there.
`REVISION_FALLBACKS` in `derive_indexes.py` closes it with a scale conversion
rather than a second index member:

| Column | Falls back to | Factor | Derivation |
| --- | --- | --- | --- |
| `deepswe_1_1` | `deepswe_1_0` | ÷ 1.069 | 1.1 reads lower than 1.0 for the same model |
| `frontiercode_1_1` | `frontiercode_1_0` | × 1.32 | mean of the two open-weight models published on both boards (GLM 5.2 19.2 → 24.5, Kimi K2.7 22.0 → 30.06) |

A model absent from the current revision has its archived score carried onto
the current scale and joins that revision's population, so it is ranked against
today's field like everyone else instead of being imputed. A model published on
both keeps the current board's own number — the conversion only ever fills a
hole, it never displaces a measurement. `swe_marathon` needs no factor: both
models on its archive alone scored 0.0, which converts to 0.0 either way.

**The conversion lives in the index and nowhere else.** `llm.json`'s columns
keep exactly what each board published, so nothing in the table ever shows a
number its leaderboard did not — the archived columns stay visible and literal,
which is what they are for.

**A source that does not say which revision it measured does not write to a
revision column.** This is the rule a bare `BFCL` and a bare `Toolathlon`
already get, applied consistently:

- `fetch_datacurve.py` reads the revision from the artifact directory, and
  discovers the older revisions from the page's own toggle — their paths never
  appear in the served HTML, because the client requests them only on click.
- `fetch_deepswe.py` (benchlm.ai) mirrors one artifact and names it in the
  page's metadata; that path is what its rows are labelled with.
- `fetch_swe_marathon.py` reads both boards the bundle ships. Only the archive
  is stored as a leaderboard; the current board exists solely as the per-task
  trial log the page aggregates in the browser, so the scraper reproduces that
  aggregation (pass@1 is the share of a configuration's trials scoring a full
  reward).
- evals.report's `frontiercode` table is ingested into `frontiercode_1_1`,
  because every row in it matches Cognition's 1.1 block. Its `swe-marathon`
  table is **not** ingested: seven rows are the 1.0 archive verbatim, beside a
  Kimi K3 that is on neither published board, with nothing saying which is
  which.
- The AA Coding Agent Index's `deep-swe` rows are **not** ingested, because its
  dataset id carries no revision the way `terminal-bench-v2.1` does. Versioning
  the id upstream would make re-enabling it a one-line change.
- In `huggingface-benchmark-name-mapping.json`, `DeepSWE (v1.1)` and
  `Agentic coding DeepSWE 1.1` map onto `deepswe_1_1` and a bare `DeepSWE` is
  unmapped; `SWE-Marathon (v1.1)` maps onto `swe_marathon_1_1`. llm-stats'
  `swe_marathon` column is unmapped for the same reason.

### Vision

Three columns are multimodal: **MMMU Pro** (`mmmu_pro`), which Artificial
Analysis runs itself, plus two assembled the same way the tool-use columns are.

| Column | Leading source | Gap fillers |
| --- | --- | --- |
| **ZeroBench** (`zerobench`) | evals.report's `zerobench` table | Hugging Face model cards |
| **MathVista-mini** (`mathvista_mini`) | evals.report's `mathvista` table | Hugging Face model cards |

Neither leads from its own leaderboard, which is unusual here and worth saying
why. MathVista's board stopped at the 2024 field — nothing on it is in this
index. ZeroBench's is alive and first-party, but its *official* table is almost
entirely closed-weight (Llama 4 Maverick and Scout are the only open rows we
carry), and the open-weight numbers it does publish sit on its *externally
reported* board, which is the same lab self-reports the model cards give us. So
evals.report leads both and the cards fill gaps ahead of it. Scraping
zerobench.github.io directly is the obvious next step, and it would pay off
immediately: the official board has Maverick at 0.4 and Scout at 1.6 where
evals.report reports 0.0 for both.

Two traps are handled in the benchmark-name mapping rather than by hoping the
labels agree, and one of them is sharper than the tool-use pair above, because
`update.py`'s Hugging Face ingest keeps the **best** value when several labels
alias one key:

- **ZeroBench is pass@1 on the 100 main questions.** `ZEROBench` and
  `ZeroBench` map onto `zerobench`. `ZEROBench_sub` (the 334 subquestions) is a
  different question set, three to four times higher on the same card;
  `ZeroBench (pass@5)` and `ZeroBench (w/ tools)` are the same questions on a
  larger budget, which on a near-zero benchmark roughly doubles the number —
  kimi-k2-5 reports 9.0 plain against 11.0 with tools. All three stay unmapped,
  or best-wins would hand the column the flattered variant.
- **MathVista is always testmini.** Every card reporting it reports the
  1,000-example split whether or not the label says so, so `MathVista`,
  `MathVista (mini)`, `MathVista mini`, `MathVista_MINI` and `Mathvista(mini)`
  all map onto `mathvista_mini`. MathVerse and MathVision are different
  benchmarks and stay unmapped.

All three feed the [Vision index](#vision-index), and only that one. The
modality gate — only a multimodal model can be scored at all — is what keeps
them out of the Knowledge index, where the coverage they add would be a slice of
the field rather than a measurement the rest of it is missing (see [What the
Knowledge index leaves out](#what-the-knowledge-index-leaves-out)). In a column
whose whole subject is that modality it is not a defect but the definition, so
the same three columns that are wrong for Knowledge are right there.

## Mapping System

The project uses a **multi-layer mapping strategy** to handle model name fragmentation:

### Layer 1: Canonical Slugs

Every model has a `slug` (e.g., `gpt-4-turbo`) used throughout the system.

### Layer 2: Benchmark-Specific Mappings

Each benchmark has a mapping file:
- `model-name-mapping-artificialanalysis.json`
- `model-name-mapping-deepswe-to-artificialanalysis.json` (shared by both
  DeepSWE readers: `fetch_deepswe.py` and `fetch_datacurve.py` label a run the
  same way, `glm-5-2[max]`, so one review covers both)
- `model-name-mapping-huggingface-to-artificialanalysis.json`
- etc.

Maps benchmark-specific names → canonical model slugs.

### Layer 2b: Several Artificial Analysis Slugs per Model

Artificial Analysis sometimes tracks one model under more than one slug, each
carrying a different slice of the benchmarks. A value in
`model-name-mapping-llm-to-artificialanalysis.json` may therefore be a list
instead of a single slug:

```json
{
  "some-model": "some-model-on-aa",
  "other-model": ["other-model-v2", "other-model-v1"]
}
```

`update.py` reads every slug in the list and merges their evaluations per
benchmark: the first slug decides every value it measured, later ones only fill
the gaps it leaves. The model's own slug leads unless the list places it
somewhere else. The extra slugs are an ingestion detail — they never reach
`llm.json`, `llm.html` or `llm-cli`, and `check_new.py` does not offer them as
new models.

### Many Source Rows, One Slug

Mapping files fold a model's variants onto a single slug on purpose: a base row
and its `[high]` sibling, one label spelled two ways, a dated re-release. Every
ingest resolves such a collision the same way — **best reported run wins**, per
benchmark. Source row ordering never decides a published number.

Two deliberate exceptions:
- **VRAM** (spheron): the *largest* estimate wins. It is a requirement, not a
  score, so the conservative direction is the safe one.
- **Artificial Analysis**: several slugs per model merge by the priority order in
  the mapping list, not by score (see above) — the leading slug is the model's
  current release, and a higher number from an older one must not displace it.

### Layer 3: Benchmark Name Aliases

Some benchmarks have internal name variations:
- `huggingface-benchmark-name-mapping.json` (benchmark name aliases)
- `llmstats-benchmark-name-mapping.json`

### Update Process

```
API Source
    ↓
fetch_*.py (pulls model names)
    ↓
update_*_mapping.py (syncs mapping files with fresh API data)
    ↓
*_mapping.py (applies mappings during score ingestion)
    ↓
llm.json (unified, deduplicated dataset)
```

## Coding Index

The first benchmark column, **Coding**, is not scraped: `derive_indexes.py` computes
it from the coding benchmarks already in the file and writes it to each model's
`scores.coding_index`. It is also the table's default sort. The
[Tooling](#tooling-index), [Knowledge](#knowledge-index) and
[Vision](#vision-index) indexes are its siblings — same script, same math,
different contributing benchmarks.

How a value is produced:

1. **Rank, don't average raw numbers.** Every contributing benchmark is turned into
   a tie-averaged percentile rank across the models scored on it, so a pass rate and
   an index score can be compared at all. A `lower_is_better` benchmark is inverted,
   so a percentile always means "how good".
2. **Weight by reliability.** The ranks are averaged with the per-benchmark weights
   the index declares in `INDEXES` (1.0 for DeepSWE 1.1, the highest, down to 0.15
   for SWE-bench Verified, the lowest), so
   the benchmarks worth trusting lead and the weaker ones fill gaps and break ties.
   Weights are relative — scaling them all leaves the ranking unchanged.
3. **Impute blanks instead of zeroing them.** A missing score is filled between the
   median (50) and the level the model has actually demonstrated, trusting the
   latter in proportion to the weight it was measured on — but never *above* the
   median. A gap counts as an unknown opponent, and an unknown opponent is never
   assumed better than the median model: a blank can hold a strong model back or
   drag a weak one down, yet it can never lift anyone, so a sparsely measured
   model cannot outrank a well-tested one on imputed strength alone.
4. **Refuse to guess.** A benchmark with fewer than two scored models carries no
   rank and is dropped from the total weight. A model measured on less than
   `MIN_SCORED_FRACTION` (18%, see [below](#why-the-evidence-bar-is-18)) of that
   weight is left unranked (`null`) rather than reported as a mostly-imputed number.

The result is reported as **whole index points**, `SCALE` (100,000) per full
percentile, so the median model sits near 50,000 and the current field spans roughly
29,000–89,000. Points rather than a percentage for two reasons: these are ranks, not
a share of tasks solved, so a number approaching 100 would read as a saturated score
it isn't; and the wide scale is what keeps the ranking strict — neighbouring models
can sit thousandths of a percentile apart (the closest pair in the current field is
15 points), which a 0–100 value would round into a tie. The column carries
`"decimals": 0` in `llm.json`, which is how `llm.html` and `llm-cli` know to print it
without the decimal every other benchmark gets.

No leaderboard publishes this column, so every ranked value is attributed to this
repository instead of to a scraped page: `derive_indexes.py` stamps
`scores_source.coding_index` with the first URL the column declares in
`llm.json` (`https://github.com/dgrieser/ai-bench#coding-index`, this section), so
clicking the score in `llm.html` opens the method rather than nothing. Change that
URL in `llm.json` and the next run restamps every value. An unranked model reports
no source, the same way it reports no date.

Two consequences worth knowing (they hold for every derived index):

- Ranks are relative to the models currently in the file, so **adding a model or a
  score moves other models' values**. That is also why `derive_indexes.py`
  clears a value back to `null` when a model stops qualifying, unlike the scrapers,
  which never overwrite a value with `null`. Adding a *benchmark* can do the same
  from the other direction: its weight joins the denominator as soon as two models
  are scored on it, which lifts the `MIN_SCORED_FRACTION` bar every model has to
  clear, so a column measured on almost nobody can cost thinly measured models
  their rank. That is why SWE-bench Multimodal, scored on two of 136 models, is
  carried as a column but left out of `INDEXES` for now — admitting it at 0.15
  dropped four models from ranked to `null` and bought no discrimination in
  return. Worth adding once its coverage grows. SWE-bench Multilingual, admitted
  at 0.30 on 26 scored models, unseated the same four — and they came back when
  the SWE Atlas trio was collapsed to one track, which took more weight out of
  the denominator than the new column put in. Both moves are worked through
  below: [the weight](#why-swe-bench-multilingual-sits-at-030), [the trio](#why-swe-atlas-contributes-one-track).
- **Terminal-Bench 4.0 and Agents' Last Exam are out of `INDEXES` for the same
  reason**, and they are the clearest illustration of it. Terminal-Bench 4.0 is
  the better-run board of the Terminal-Bench pair — 4.0 removed the saturated
  tasks and the ones with public solutions and calibrated every task's resource
  budget — but it is scored on one model of 144, so it carries no rank at all and
  any weight for it would be an assertion from release notes rather than a
  measurement. Agents' Last Exam is scored on eight, and that was enough: admitted
  to both groups it moved *every* ranked value, all 92 Coding and 93 Tooling. Eight
  overlapping models is too thin a base to re-rank the whole table on, and the
  weight it could carry was being chosen around the bar rather than around what it
  measures — 0.30 dropped twelve models to `null`, so only 0.25 fit. Both stay
  columns, scraped and rendered like any other; both are worth admitting once
  their coverage grows.
- The column has to be recomputed after every change to `scores` **or** to the set of
  models, and every writer that can cause one does it for you, in the same write, via
  `derive_indexes.refresh_and_report()`:
  - `update.py -w` — after the scrapers have merged their scores (so a direct run is
    self-sufficient; `update-all` additionally runs the script as its last step).
  - `edit.py` — after a hand-edited score. Skipped for a params/context-only edit,
    which cannot move a rank.
  - `prune.py -w` — after dropping models, because removing one that carried a
    contributing score re-ranks the survivors even though none of their own scores
    moved.

  `add.py` needs no refresh: a new model arrives with all-null scores, and a model
  with no score in a benchmark is not part of that benchmark's population, so nothing
  is re-ranked. `fill_source_urls.py`, `fill_missing_source_urls.py` and `sync_score_dates.py`
  touch neither scores nor models.
- Derived columns are never a mapping target. The interactive prompts
  (`add.py`, `edit.py`, `update_*_mapping.py`) and the unattended proposal builder
  (`propose.py`, via `editable_benchmarks()`) all exclude them, so a fetched source
  score cannot be routed into a column the next derivation would overwrite.

The math is the one `llm.html` and `llm-cli` implement for a sort group (`sortGroups`
in `llm.json`), which is what this column replaced — that machinery is still in
place, just with no group configured.

### Why SWE-bench Multilingual sits at 0.30

The newest member of the coding group, and the worked example of how a weight on the
ladder gets chosen. Measured on the current file (26 scored models):

| Axis | Measurement | Pull |
| --- | --- | --- |
| What it tests | 300 real issue/PR pairs from 42 repositories in nine languages other than Python, graded by running the repository's own fail-to-pass and pass-to-pass tests. No judge, no algorithmic toy problems. The rest of the group is Python-heavy, so this is the only column that can tell a model that only writes Python from one that ships Go and Rust. | **up** |
| Saturation | Median 68.0, max 79.6, **nothing above 80** — no ceiling problem, unlike SWE-bench Verified (93.4 max, saturated) or LiveCodeBench (4 models ≥ 90). | **up** |
| Head resolution | Top model minus fifth is **3.1 points**, the tightest in the group (SWE-bench Verified 12.8, SWE-bench Pro 18.2). It separates the mid-field well and the leaders barely at all. | down |
| Redundancy | Spearman **0.92** with SWE-bench Verified on the 23 models that have both — the strongest overlap of any pair here, which is what the shared collection pipeline predicts. Also 0.85 with Terminal-Bench 2.1 and 0.77 with SWE-bench Pro. | down |
| Trust | The official leaderboard runs one standardized mini-SWE-agent, but only **3** of our 26 values come from a run we can check (evals.report, Official/Verified). The other 23 are Hugging Face card self-reports at each lab's harness of choice. Public since 2025 and built by the SWE-bench pipeline, so its contamination profile is SWE-bench Pro's, not DeepSWE's. | down |
| Coverage | 26 of 136 models (19%), from 11 vendors — mid-pack for this group: ahead of DeepSWE and the SWE-Atlas tracks (6–14) and behind Terminal-Bench 2.1 (78), LiveCodeBench (94), SciCode (125). | — |

0.30 is where those pull: below `swe_bench_pro` (0.4) because most values are
self-reported rather than harness-controlled, above `swe_bench_verified` (0.15)
because it is unsaturated and it is the only non-Python signal in the group. The
ranking barely moves — Spearman 0.994 against the pre-addition index, mean 2.3
places, max 7.

**What it cost on admission, measured.** Four models — `glm-4-6`, `glm-4-7-flash`,
`qwen3-5-27b`, `qwen3-coder-480b-a35b-instruct` — went from ranked to `null`. None of
them lost a score; the bar moved. Each carries the same four cheap members (SWE-bench
Pro + LiveCodeBench + SciCode + SWE-bench Verified = **1.30** scored weight), and under
the 20% threshold in force at the time that bar went from 1.272 to 1.332 — they had
been clearing it by 0.028, so *any* new coding column above 0.14 would have unseated
them. Not a fact about SWE-bench Multilingual. Both follow-ups have since removed the
problem rather than papered over it: [the SWE Atlas
trio](#why-swe-atlas-contributes-one-track) gave 0.26 back to the denominator, and
[the threshold](#why-the-evidence-bar-is-18) moved to 18%, which puts the bar at 1.152
and leaves them 0.15 of margin instead of 0.02.

One knob deliberately not turned: weighting it **0.5** would also have cleared the
bar for two models (at w ≥ 0.465 `deepseek-v3-2-0925` and `kimi-k2-thinking` qualify
on their own SWE-bench Multilingual score). Buying coverage with a weight the trust
evidence does not support is the mistake the ladder exists to prevent; the weight
describes the benchmark, not the roster.

### Why SWE Atlas contributes one track

Scale AI's SWE Atlas ships three tracks and the index carried all three at 0.17 each,
so the family voted 0.51 — deliberately, as "three narrow rubric-graded slices, half a
benchmark between them". Measured on the current file, that reasoning does not hold:

- **The three tracks are not three populations.** Every model scored on Refactoring
  (6) or Test Writing (7) is also scored on Codebase Q&A (13). Dropping the first two
  removes **no model** from the index, and Q&A alone covers the family's whole roster.
- **They are barely three measurements.** Spearman **0.94** between Refactoring and
  Test Writing, 0.77 and 0.89 against Q&A. Three near-duplicate ranks over one
  ≤13-model population is the same triple-count that got `aa_coding_index` removed
  from this group, one order of magnitude smaller.
- **The redundant weight was not free.** Weight in the denominator raises the
  `MIN_SCORED_FRACTION` bar for *every* model, including the ones the extra tracks
  never measured.

So the family now contributes **Codebase Q&A at 0.25**: one track, weighted a little
above the 0.17 it held as one third of a trio, because it now carries the family's
whole vote — and well below the 0.51 the trio held, because it is one small
rubric-graded track. It stays under `swe_bench_multilingual` (0.30) and every
execution-graded column above it, which is the honest place for it: Q&A is the one
track in the trio that requires **no code changes** at all (124 comprehension tasks —
tracing execution paths, multi-file reasoning), so on task shape it is the weakest of
the three for a coding index. It is kept over the other two anyway because the
correlations say all three rank models alike and only Q&A has the coverage.
Refactoring and Test Writing remain columns in `llm.json`; they are simply not
aggregated.

### Why the evidence bar is 18%

`MIN_SCORED_FRACTION` is the one number all five derived indexes share: sum the
weights of the contributing benchmarks a model actually has a score on, and if that is
below `MIN_SCORED_FRACTION × total group weight`, the model is `null` instead of
ranked. It is a share of *weight*, not a count of benchmarks — three cheap columns can
be worth less evidence than one expensive one.

At **0.18** the bars are 1.152 of 6.40 (Coding, 92 of 144 models ranked), 1.107 of
6.15 (Tooling, 93 ranked), 0.666 of 3.70 (Knowledge, 141 ranked), 0.432 of 2.40
(Vision, 47 ranked) and 0.423 of 2.35 (Trust, 133 ranked, now that
[AA-Omniscience Accuracy](#why-the-anchor-cannot-stand-alone) is fetched).

Coverage does not spread evenly across models, it clusters, and the threshold should
fall between clusters rather than through one. Measured over the current file:

| fraction | Coding ranked | Tooling ranked | Knowledge ranked | Vision ranked | Trust ranked | what the cut admits |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 75 | 90 | 140 | 47 | 133 | |
| 0.20 | 80 | 93 | 141 | 47 | 133 | the 20%-measured block, cut in half |
| **0.18** | **92** | **93** | **141** | **47** | **133** | the rest of that block: 12 models at 19% |
| 0.15 | 92 | 95 | 141 | 47 | 133 | nothing in Coding |
| 0.12 | 105 | 122 | 141 | 47 | 133 | the 13–16% cluster |
| 0.10 | 135 | 123 | 141 | 47 | 133 | the ~12% cluster |
| 0.05 | 143 | 135 | 141 | 47 | 133 | models measured on a twentieth of the weight |

0.20 was splitting a natural block: eleven models sat at 19% and none between 19% and
20% (twelve today), so `kimi-k2-thinking` and `deepseek-v3-2-0925` (both scored on LiveCodeBench,
SciCode, SWE-bench Multilingual and SWE-bench Verified) were unranked while models with
the same amount of evidence were not. 0.18 takes the whole block and stops before the
next one. Lowering the bar never changes a ranked model's *value* — the total weight is
the same either way — it only decides who appears; previously ranked models move a mean
of 3 places as the new arrivals slot in, and Tooling does not move at all.

The exact fractions drift as members join and leave — every admission changes the
denominator, so the same absolute evidence is a smaller share afterwards — but the
shape does not, and 0.18 has stayed inside the gap through each of them. Today the
Coding ladder jumps 14.1% → 18.8% and the Tooling ladder 16.3% → 21.1%, so the bar
still falls between clusters rather than through one. That is a live constraint on
what may join, not just a historical note: it is one of the reasons Terminal-Bench
4.0 and Agents' Last Exam are [carried as columns and left out of
`INDEXES`](#coding-index) for now.

Going further would change what the column means rather than how much of it is filled
in. At 0.12, **90%** of ranked models would be measured on less than half the weight;
at 0.10, 92%. The head stays safe at any setting — the imputation cap means a model
measured on 12% of the weight cannot score above 56,000 even by topping every benchmark
it has, and no model under 25% measured reaches the visible top 20 at any fraction I
tested — but the mid-field crowds: the ten models measured on at least half the weight
fall a mean 5 places at 0.12 and 11 places at 0.10, below models whose numbers are
mostly imputation. 18% keeps the ranked field majority-measured; 12% does not.

The setting is global to all five indexes today. They do not want the same thing —
Tooling barely moves between 0.20 and 0.15 where Coding gains 11 models, Knowledge is
flat across almost the whole range, and Vision and Trust are flat across all of it — so
if they ever need to diverge, the place for it is a per-index override on `IndexDef`,
not a compromise value.

Vision and Trust are the two columns the bar could never bite, at any setting in the
table above. Vision's cheapest member, MMMU Pro, is 42% of the group weight on its own
and every model it ranks has that score, so the floor of its coverage ladder sits at
0.417 and the threshold would have to clear 0.40 to cut anyone. Trust is the same shape
for the same reason — its anchor is 57% of today's effective weight and carried by every
model it ranks. What filters both columns is availability of the underlying run: the
modality gate for one, whether Artificial Analysis has run the model at all for the
other — see [Why the evidence bar is inert
here](#why-the-evidence-bar-is-inert-here) and [The evidence bar is inert here
too](#the-evidence-bar-is-inert-here-too).

Knowledge is the column the threshold does the least for, at any setting: its members
cover 40–97% of the table each, so 124 of the 133 models it ranks are measured on at
least half its weight and the three it leaves out have no score in any of its six
benchmarks. Nothing between 0.05 and 0.20 changes that — see
[Knowledge index](#knowledge-index).

Effects, measured: total group weight 6.66 → 6.40, and with it the bar (1.332 → 1.280
at the 20% threshold in force at the time), which put the four models the SWE-bench
Multilingual admission had unseated back in the ranking. Ranking agreement with the
pre-change index is Spearman 0.999 (mean 390 index points across the table); the seven
models that lose Refactoring/Test Writing evidence move by a mean of 1,103 points —
`ornith-1-0-9b` most, +3,347, because its remaining evidence ranks it better than the
two tracks it lost did.

Worth being clear about what carries what: at today's 18% threshold those four clear
the bar with or without this change, so the redundancy argument above is the whole
justification for it, not the four models. Removing near-duplicate weight is right on
its own; rescuing a rank is a lever, and [the threshold](#why-the-evidence-bar-is-18)
is the honest one.

## Tooling Index

The second derived column, **Tooling**, is the Coding index's sibling for agentic
tool use: same script (`derive_indexes.py`), same percentile-rank math, imputation
and coverage rules as [above](#coding-index), computed over the tool-use
benchmarks instead — plus one instruction-following column, see below — and
written to each model's `scores.tooling_index`. Ranked values cite this
section (`https://github.com/dgrieser/ai-bench#tooling-index`) the same way the
Coding index cites its own.

Contributing benchmarks and why they carry the weight they do:

| Benchmark | Weight | Rationale |
| --- | --- | --- |
| τ³-Bench Banking | 1.0 | The most reliable measurement of the set: every score run independently by Artificial Analysis, execution-graded against backend state, far from saturation, and it tests tool *discovery* (tools hidden in KB documents, unlocked via meta-tools) — a signal the other benchmarks don't carry. |
| Toolathlon-Verified | 0.9 | The purest tool-use benchmark available: long-horizon tasks over real MCP servers, execution-graded and unsaturated. Below 1.0 because the leaderboard is run by the benchmark's own team with a mix of verified and self-reported entries, and it is young — two incompatible score series in under a year. |
| MCP-Atlas | 0.85 | The purest *MCP* signal in the set: production-like servers, hundreds of tools, judged on end-task success, and it correlates 0.72 with Toolathlon — close enough to be the same capability, far enough to still add information. Above Terminal-Bench 2.1 because it is far more tool-shaped; below Toolathlon because only 6 of its 17 stored scores are Scale's own runs and the rest are lab-reported, where the Toolathlon ingest drops self-reported rows outright. |
| Terminal-Bench 2.1 | 0.8 | Broad, widely trusted, mostly AA-run in this file — but the least tool-shaped of the set (terminal/CLI agency rather than structured tool calling, overlapping the Coding index), nearing its ceiling, with fully public tasks and documented harness variance. |
| GDPval-AA v2 | 0.7 | Tool use is how the work gets done here, not a side effect: AA runs the model in its Stirrup agentic harness with shell access to a sandbox filesystem and web browsing, and the deliverable — a document, spreadsheet, slide deck, diagram — is the output of that trajectory. AA tags the evaluation `agentic` and `tool-use`, the same pair Terminal-Bench 2.1, τ³ Banking and ITBench-AA carry. It has the best provenance of the set (70 of 71 scores AA-run) and the sharpest discrimination (210 Elo between the best model and the fifth against a ~21-Elo median gap between neighbours), over 220 tasks spanning 44 occupations, which is why it outranks the narrower ITBench-AA. Below Terminal-Bench 2.1 because the two largely measure the same shell-agency axis — they correlate 0.92 — and Terminal-Bench scores task success directly where this Elo is mediated by pairwise judging of deliverable quality, so a polished artifact can be rewarded over a clean trajectory. |
| ITBench-AA | 0.6 | High trust per measurement (AA-run end to end, a third of the tasks held privately by IBM, unsaturated) but the smallest task set of the ten and domain-narrow: diagnosing Kubernetes incidents from an offline snapshot. |
| BFCL v4 | 0.5 | High trust per measurement — first-party runs, published model responses, reproducible at a pinned commit — but it correlates 0.91 with τ³ Banking and 0.93 with Terminal-Bench Hard, so it buys coverage and stability rather than information. Its Overall Accuracy is an unweighted average dominated by AST-checked single-call categories, and the board refreshes slowly, so most frontier open-weight scores arrive as card self-reports. |
| τ²-Bench Telecom | 0.3 | Effectively saturated — the leaders sit within noise of each other — so it can no longer separate frontier models. Kept as a coverage backbone — 101 scored models, second only to IFBench — so it fills gaps and breaks mid-field ties without leading anything. |
| Terminal-Bench Hard | 0.3 | Correlates ~0.94 with Terminal-Bench 2.1, so it adds coverage and stability rather than information: it is AA-run, unsaturated and broadly scored, which keeps thinly measured models from floating up on imputation alone. |
| IFBench | 0.2 | A tool call has to be well-formed before it can be right, which is the whole of its claim here — it is not a tool-use benchmark, and it is weighted last accordingly. It correlates 0.73-0.74 with the Terminal-Bench columns and 0.60 with ITBench-AA, but only 0.13 with Toolathlon, the weakest link to the purest tool-use member of any contributor. It is also the second most saturated column after τ²-Telecom, 3.4 points between the best model and the fifth. What it brings is reach: 112 scored models, the widest of the ten and almost all AA-run, which keeps thinly measured models from floating up on imputation alone. |

## Knowledge Index

The third derived column, **Knowledge**, is the Coding and Tooling indexes' sibling
for what a model knows unaided (the fourth, [Vision](#vision-index), came later): same script (`derive_indexes.py`), same
percentile-rank math, imputation and coverage rules as
[above](#coding-index), computed over the knowledge, science and math benchmarks
instead and written to each model's `scores.knowledge_index`. Ranked values cite
this section (`https://github.com/dgrieser/ai-bench#knowledge-index`) the same way
the other two cite their own.

The axis is deliberately narrow: **recall and unaided reasoning over subject
matter** — general and obscure facts, graduate science, research physics,
competition math. What the model can *look up* is a different capability and is
left out, which is why BrowseComp (a browser in the loop) and AA-LCR (questions
over a document supplied in the prompt) are not members even though both are
question-answering columns. See [what it leaves
out](#what-the-knowledge-index-leaves-out).

It is also not the other two indexes in a different hat. Against the models it
shares with them, the Knowledge ranking agrees with Coding at Spearman **0.79** (81
models) and with Tooling at **0.84** (85) — related, as one would expect of columns
over the same field, but far from the 0.9x an index would show if it were
re-measuring them.

Contributing benchmarks and why they carry the weight they do:

| Benchmark | Weight | Rationale |
| --- | --- | --- |
| AA-Omniscience | 1.0 | The only member built to measure knowledge as such rather than to measure reasoning and read knowledge off it: 6,000 questions over 42 economically relevant topics in six domains, and a bounded −100…100 index that rewards correct recall, penalises confident wrong answers and credits abstention. Every one of its 120 values is an Artificial Analysis run — the best provenance in the set alongside CritPt — it is nowhere near its ceiling (best 19.7, median −48.0 on a scale that goes to 100), and it has by far the sharpest head resolution: 15.4 points between the best model and the fifth. It is also among the least redundant members: mean Spearman **0.61** against the other five (0.30 with AIME 2025 at the low end, 0.74 with GPQA Diamond at the high), against 0.83 for GPQA Diamond, so it leads on information as well as on trust. |
| Humanity's Last Exam | 0.9 | The obscure-knowledge column: 2,500 expert-written questions across many subjects, built to resist retrieval, and the broadest coverage of any member (131 of 136 models, 43 creators, 126 values AA-run). Unsaturated with room to spare — best 47.6, median 10.1. Below AA-Omniscience because it is the second most redundant member of the set — mean Spearman **0.81** against the other five (0.91 with GPQA Diamond, 0.86 with CritPt, 0.81 with AIME 2025, 0.78 with MMLU-Pro), behind only GPQA Diamond itself — so a good part of its vote is already cast by the members below it, and because a single hard exam mixes knowledge with reasoning in a ratio nobody can read off the score. |
| CritPt | 0.6 | The hardest science in the table and the cleanest measurement of it: 70 research-level physics problems, all 55 values AA-run, and the least saturated column anywhere in this index — best 23.4, median 1.7. High trust per measurement, and it separates the leaders (5.4 points between first and fifth). Held to 0.6 by what it cannot do: 70 problems in one discipline is the smallest and narrowest task set of the six, and the floor is crowded — 42 of its 55 values sit in a tie, 13 of them at 0.3 — so below the frontier it ranks almost nobody. Same profile, same weight as ITBench-AA in the [Tooling index](#tooling-index). |
| MMLU-Pro | 0.5 | The breadth column: ~12,000 questions across 14 disciplines, ten options instead of four, the trivial and mislabelled items MMLU had accumulated filtered out. Nothing else here covers ordinary academic knowledge across that many fields, and at 95 scored models from 33 creators it is the widest member after HLE and GPQA. Below CritPt on two counts: only 51 of its 95 values are AA-run, the rest Hugging Face card self-reports at each lab's harness of choice; and the head is flat — 1.1 points between the best model and the fifth against a median of 77.6, so it sorts the mid-field and barely touches the leaders. |
| AIME 2025 | 0.4 | The math column, and the member that overlaps the others least: mean Spearman **0.58**, the lowest of the six, and it owns the two weakest links in the set — 0.30 with AA-Omniscience and 0.27 with CritPt — because working out a competition problem is not recalling a fact. 77 scored models, 30 creators, 59 values AA-run. Weighted below MMLU-Pro because 15 integer-answer problems is a narrow instrument, the head is saturated (12 models at 90 or above, 2.6 points between first and fifth), and a 2025 exam has had a year of public exposure — the contamination risk the newer paper is written for. |
| GPQA Diamond | 0.3 | The coverage backbone: 132 of 136 models, 42 creators, 126 of those values AA-run — the widest column in the index. It is here for reach and tie-breaking rather than for information, on both of the usual counts. Saturated: 9 models at 90 or above and **0.7 points** between the best model and the fifth, so it cannot separate the frontier at all. And redundant: mean Spearman **0.83** against the other five, the highest of the six, including 0.91 with HLE and 0.92 with MMLU-Pro — the two strongest links anywhere in this set — which is what a graduate-science multiple-choice test shares with a broad multiple-choice test and a hard mixed exam. What it buys is that almost nobody in the table is unmeasured, which keeps thinly measured models from floating up on imputation alone. |

### What the Knowledge index leaves out

Six columns in `llm.json` answer knowledge-shaped questions and are still not
members. Each is kept as a column in the table, none is aggregated, and the reasons
differ:

- **AA-Omniscience Hallucination Rate** — the same Artificial Analysis run as
  AA-Omniscience, and the Omniscience Index is already defined over the behaviour it
  reports: correct recall rewarded, confident wrong answers penalised, abstention
  credited. Admitting it means one 6,000-question run casting **33%** of the index's
  weight, and it buys 6 models of coverage (126 scored against 120). Measured, it is
  not a second opinion either: Spearman **−0.79** against the column it duplicates,
  and of the 20 best hallucination rates in the file, **none** sits below the
  Omniscience median — the two rank the field alike. It is the same triple-count
  argument as [SWE Atlas](#why-swe-atlas-contributes-one-track), and it is not free:
  at 0.3 it moved the ranked field a mean of 3 places and pushed
  `deepseek-v4-pro` from first to sixth on the strength of one re-counted run.
- **AIME 2026** — the same exam one year on, and the fresher contamination profile is
  real. Everything else about it is not ready: 28 scored models (21% of the table),
  **19 of them at 90 or above** with a median of 93.0, so it cannot rank even the
  field it covers, and 23 of its 28 values are Hugging Face card self-reports.
  Admitted at 0.2 it reordered four of the top six, on imputed weight charged to the
  108 models it does not measure — the failure mode the Coding index documents for
  [SWE-bench Multimodal](#coding-index). Worth revisiting once AA's coverage arrives,
  and then as AIME 2025's *replacement* rather than beside it.
- **MMMU Pro** — the most redundant candidate anywhere in the set: Spearman **0.96**
  with GPQA Diamond on the 45 models that have both. It is also gated on modality —
  only a multimodal model can be scored at all — so admitting it would charge
  text-only models imputation weight for something other than knowledge. At 0.3 it
  moved ranks a mean of 1.2 places and left the head untouched: nothing gained, a
  modality tax paid. It is not unaggregated, though — it anchors the [Vision
  index](#vision-index) at 1.0, where the modality gate is the subject rather
  than a tax.
- **AA-LCR and BrowseComp** — the retrieval axis this index is defined against.
  AA-LCR answers open questions over 10k–100k-token documents *supplied in the
  prompt*, and BrowseComp is scored with a browser in the loop, so a strong score can
  belong to a strong search agent rather than to a knowledgeable model. Both measure
  something worth measuring; neither measures what the model knows. (BrowseComp is
  also the thinnest-provenance candidate: 43 values, 32 from llm-stats and 11 from
  model cards, none from a run we control.)
- **AA Intelligence Index** — Artificial Analysis' own composite, over benchmarks in
  this very table with coding and agentic tool use among them. Including it would let
  one vendor's weighting vote a second time, and against the six members it would sit
  beside it agrees at Spearman **0.93** (126 models) — close agreement being the
  problem, not the reassurance. Exactly why `aa_coding_index` was dropped from the
  [Coding index](#coding-index).

### Why the coverage rules barely bite here

The Knowledge group is the densely measured one. Its members cover 40–97% of the
table each (GPQA Diamond 132, HLE 131, AA-Omniscience 120, MMLU-Pro 95, AIME 2025 77,
CritPt 55), against a Coding group whose heaviest member is scored on 14 models. Three
consequences, all different from the other two indexes:

- **133 of 136 models are ranked**, and the three that are not — the `ornith-1-0`
  family — have no score in any of the six benchmarks, so no threshold rescues them.
  The next-thinnest model, `agents-a1`, is measured on 24% of the weight. The
  [18% evidence bar](#why-the-evidence-bar-is-18) is therefore doing almost nothing
  for this column: it ranks 133 models at every setting from 0.05 to 0.20, and 132 at
  0.25.
- **The ranked field is measured, not imputed.** 124 of the 133 ranked models carry
  at least half the group's weight in real scores, and 18 carry all six. That is the
  property the evidence bar exists to protect elsewhere and gets for free here.
- **The scale earns its width.** 133 ranked values inside one percentile range put the
  closest pair **1 index point** apart (Coding's is 5, Tooling's 7) with no two values
  colliding. On a 0–100 scale a good part of this mid-field would have rounded into
  ties, which is the argument for `SCALE` made visible.

The flip side of that density is that the imputation rule does real work at the top.
A model that leads the columns it is measured on can still finish behind one measured
on more of them — `kimi-k3` is first on AA-Omniscience and CritPt, joint first on GPQA
Diamond and second on HLE, and still places behind `deepseek-v4-pro`, which leads none
of the six but is scored on five of them against `kimi-k3`'s four. That is the imputation
cap behaving as designed (a blank is an unknown opponent, never a better one), and with
this group's coverage it is a visible effect rather than a footnote.


## Vision Index

The fourth derived column, **Vision**, is the Coding, Tooling and Knowledge
indexes' sibling for what a model can do with an image: same script
(`derive_indexes.py`), same percentile-rank math, imputation and coverage rules
as [above](#coding-index), computed over the multimodal benchmarks instead and
written to each model's `scores.vision_index`. Ranked values cite this section
(`https://github.com/dgrieser/ai-bench#vision-index`) the same way the other
three cite their own.

The axis is **anything that starts with pixels**: college-level question
answering over figures and diagrams, mathematical reasoning in visual contexts,
deliberately-hard multi-step visual puzzles, and driving a real desktop GUI from
screenshots. That last one is why the column is not called "vision-language" —
OSWorld measures visual *agency*, not visual question answering, and it is the
member the other three cannot stand in for.

**It ranks 47 of 143 models, and the 96 blanks are the point.** Only a
multimodal model can be scored on any of these benchmarks, so a blank here says
"this model has never been pointed at an image", which is exactly the question
the column exists to answer. That modality gate is the reason MMMU Pro is
[kept out of the Knowledge index](#what-the-knowledge-index-leaves-out) — there
it would charge text-only models imputation weight for something other than
knowledge. Here it is the subject.

It is the least independent of the five, and the number is worth stating rather
than burying. Against the models it shares with them:

| | full field | top 20 |
| --- | --- | --- |
| vs [Coding](#coding-index) | 0.83 (41 models) | **0.17** |
| vs [Tooling](#tooling-index) | 0.88 (45) | **0.38** |
| vs [Knowledge](#knowledge-index) | 0.94 (47) | **0.57** |
| vs AA Intelligence Index | 0.93 (47) | 0.50 |
| vs GPQA Diamond | 0.93 (47) | 0.60 |

Those full-field figures sit above the 0.79/0.84 the [Knowledge
index](#knowledge-index) holds up as proof it is not re-measuring its siblings,
and no amount of reweighting fixes that, because it is mostly a range effect:
the 47 ranked models run from `qwen3-5-0-8b` to 400B-plus mixtures of experts,
and across a spread that wide nearly every capability column agrees with nearly
every other. Inside the top 20 — where a reader actually chooses between models
— it comes apart, to **0.17** against Coding and 0.38 against Tooling.

So this column does not earn its place by being orthogonal to the other three.
It earns it by covering a **modality** none of them touches, by saying so about
the two thirds of the table it leaves blank, and by separating the frontier once
general capability stops explaining the ranking.

Contributing benchmarks and why they carry the weight they do (total group
weight **2.40**):

| Benchmark | Weight | Rationale |
| --- | --- | --- |
| MMMU Pro | 1.0 | The anchor, and the only member that ranks the field rather than a corner of it: **47 scored models across 15 creators**, against 11-14 and 3-4 creators for the rest. It also has the best provenance in the group by a distance — **46 of its 47 values are Artificial Analysis runs**, where the other three are dominated by Hugging Face card self-reports at each lab's harness of choice. Vision-centric by construction: MMMU Pro filters out the questions a text-only model could answer without the image, augments the candidate set so a lucky guess is worth less, and adds a vision-only setting where the question itself is embedded in the picture. Peer-reviewed (ACL 2025) and unsaturated, 25.8-82.3 with a median of 69.2. What it does *not* lead on is discrimination: 3.7 points between the best model and the fifth is a flatter head than OSWorld's, and it correlates 0.96 with MathVista-mini and 0.93 with ZeroBench, so it is the group's centre of gravity rather than its most independent voice. |
| OSWorld-Verified | 0.7 | The highest-value *measurement* here and the least redundant member: mean Spearman **0.87** against the other three, including the group's weakest link at 0.78 with ZeroBench. A GUI agent driving a real Ubuntu desktop from screenshots is the only visual **agency** in the table, the Verified revision exists to repair task graders that were mis-scoring the original, and it is the least saturated member with by far the sharpest head — **13.0 points** between first and fifth against a 63.3 median, where the next best is ZeroBench's 4.0 on a benchmark whose entire observed range is 12 points. On design it would earn 0.85-0.90. It is discounted a full tier for two things it cannot currently do: **13 scored models from 4 creators**, the thinnest coverage of the four; and provenance no better than the coverage — only 3 of those 13 values come from the official board, 9 are card self-reports, and on a benchmark whose score moves with the step budget (this column takes the Foundation E2E GUI subset at a 100-step cap) mixing harnesses inside one percentile-normalized column is the trust hazard the weight scale exists to price. The weight is not load-bearing: moving it between 0.6 and 0.8 shifts the ranked field a mean of under one place and never touches the top three. |
| MathVista-mini | 0.35 | The saturated member, and the most redundant. Median **86.0**, p75 87.4, best 90.3 — the entire top of the field is packed inside three points, and **2.9 points** separate first from fifth, the flattest head anywhere in this index. It is also mean Spearman **0.95** against the other three, including **0.96 with MMMU Pro** and 0.97 with ZeroBench, so most of its vote is already cast by members that measure more. Public since 2023, so it carries the contamination profile three years of exposure buys, and 12 of its 14 values are card self-reports. Kept because it is the second-widest member and the mid-field is where it still separates models — the coverage-backbone role GPQA Diamond plays at 0.30 in the [Knowledge index](#knowledge-index). |
| ZeroBench | 0.35 | The opposite failure mode, which is why it lands on the same rung rather than above it. Its *design* is the best in the group: 100 hand-crafted multi-step questions built so that nothing solves them, which makes it the one member structurally immune to the saturation MathVista is already suffering. Its *measurement* is the weakest. Median **3.0**, best 12.0, three models tied at 0.0 — and at 100 questions the binomial standard error near p = 0.1 is about 3 points, so the whole observed 0-12 range is a few standard errors wide and a single question moves a rank. 11 scored models from 3 creators, 9 of them card self-reports. It is a headroom sentinel that will earn weight as models climb, not a discriminator today. Its exposure to a flattered variant slipping in through `update.py`'s best-value-wins Hugging Face ingest is handled in the benchmark-name mapping — see [Vision](#vision) above. |

### Why GDPval-AA is left out

`gdpval_aa` is the obvious fifth member and is deliberately not one. It has the
best coverage in the table (79 models, 25 creators, 78 of them AA-run), and its
deliverables are documents, slide decks, diagrams and spreadsheets, so it is the
only column in `llm.json` scoring visual *output* rather than visual input. Two
measurements keep it out:

- **It is not a visual signal.** Spearman **0.95** with the Tooling index and
  **0.97** with AA Intelligence Index. Artificial Analysis runs it in the
  Stirrup agentic harness with shell access and a browser, so what the Elo
  separates is shell agency mediated by pairwise judging of deliverable
  quality — which is precisely what the [Tooling index](#tooling-index) already
  prices it for at 0.70. Admitting it here would let one run vote twice on two
  different axes.
- **38 models carry it as their only score in this group** — models nothing has
  ever measured on an image. At its Tooling weight of 0.7 its share of a
  five-member group would be 0.226, over the [18% evidence
  bar](#why-the-evidence-bar-is-18), so it would not merely contribute: it would
  *rank* all 38 on no visual evidence whatsoever, taking the column from 47
  ranked models to 85 and turning two thirds of a vision ranking into a restated
  GDPval ranking.

Holding it below the bar instead of dropping it was the other option — at 0.5 of
a 2.9 total its share is 0.172, just under — and it was rejected as the wrong
kind of clever: a weight chosen to sit under a threshold is one edit away from
silently admitting 38 models, and the first objection above stands whatever the
weight. GDPval-AA remains a column in the table and a member of the Tooling
index; it is simply not evidence about vision.

### Why the evidence bar is inert here

Coverage does not spread across this group, it **steps**, because every one of
the 47 ranked models has MMMU Pro and nothing else is scored on a model MMMU Pro
is not:

| share of weight | models | what they have |
| --- | --- | --- |
| 1.000 | 7 | all four |
| 0.708 | 10 | MMMU Pro + OSWorld, or + both small members |
| 0.563 | 3 | MMMU Pro + MathVista + ZeroBench |
| 0.417 | 27 | MMMU Pro alone |

The floor of that ladder is 0.417, so `MIN_SCORED_FRACTION` would have to exceed
**0.40** to cut anybody: Vision ranks the same 47 models at every setting from
0.05 to 0.30. The [18% bar](#why-the-evidence-bar-is-18) is completely inert
here and **no per-index override is needed** — the modality gate is already
doing the filtering the bar does elsewhere, and doing it on better evidence.

The honest weakness that leaves is the mirror image of Knowledge's: **27 of the
47 ranked models are measured on MMMU Pro alone**, and only 20 of 47 carry at
least half the group's weight, where the Knowledge index gets that property for
free from its own density. For most of this
column's field, the ranking *is* MMMU Pro plus imputation, and it should be read
that way — `--top` prints an `N/4 measured` column next to every value for
exactly this reason.

What keeps that safe is the imputation cap, and here it is legible in the
numbers rather than a footnote. A model measured on MMMU Pro alone cannot score
above **70,833** however well it does, because the 58% of the weight it is
missing is filled at the median and never above it. So the head of the column is
reserved for models measured on more of it, by construction. The leader,
`qwen3-8-2-4t-a95b`, sits at **85,417** — exactly its own cap at 70.8% coverage
— because it tops both MMMU Pro (82.3) and OSWorld-Verified (86.1) and has no
score on either small member.

Two other properties of the current field:

- **47 ranked, 46 distinct values.** The one collision is not a rounding
  artifact and no weighting can remove it: `devstral-small-2` and `gemma-4-e2b`
  both score 44.6 on MMMU Pro and have no other score in the group, so their
  evidence is byte-identical and the index is right to tie them at 26,336. Apart
  from that pair the closest neighbours sit **27 index points** apart, the widest
  margin of the five — a small ranked field spread over the full scale, where the
  other four pack many more values into the same range.
- **The range is the widest of the five**, 6,137 to 85,417, because the
  multimodal field in this table runs from 0.8B models to frontier mixtures of
  experts with very little in between.

## Trust Index

The fifth derived column, **Trust**, is the odd one out of the five, and
deliberately so. Coding, Tooling, Knowledge and Vision all rank **capability** —
what a model can do. This one ranks whether the answer can be believed. Same
script (`derive_indexes.py`), same percentile-rank math, imputation and coverage
rules as [above](#coding-index), computed over the honesty, grounding and
instruction-compliance benchmarks and written to each model's
`scores.trust_index`. Ranked values cite this section
(`https://github.com/dgrieser/ai-bench#trust-index`) the same way the other four
cite their own.

The axis is two failure modes that have nothing to do with how much a model
knows:

- **Confident fabrication** — what it does when it does *not* know. Answer
  anyway, or say so? (AA-Omniscience Hallucination Rate and Accuracy.)
- **Departing from what it was given** — answering from memory instead of the
  document in front of it, or quietly ignoring the constraint you set. (AA-LCR,
  IFBench.)

The single result that justifies the column: **`deepseek-v4-pro` ranks 1st on
[Knowledge](#knowledge-index) and 80th on Trust.** It knows more than anything
else in the table and hallucinates on 94.8% of what it gets wrong. `glm-4-7`
goes 22nd to 89th, `deepseek-v4-flash` 13th to 63rd, `step-3-5-flash` 27th to
87th. No other column in this file says that about a model, because no other
column is measuring it.

**It ranks 132 of 143 models** — second only to Knowledge, because its anchor is
one of the widest columns in the table. The 11 blanks are models Artificial
Analysis has never run at all.

### It is the most independent of the five

The [Vision index](#vision-index) had to argue its way around a 0.94 correlation
with Knowledge. This one has the opposite problem — none. Against the models it
shares with each:

| | full field | top 20 |
| --- | --- | --- |
| vs [Coding](#coding-index) | **0.44** (84 models) | **−0.17** |
| vs [Tooling](#tooling-index) | 0.66 (88) | 0.13 |
| vs [Knowledge](#knowledge-index) | 0.78 (132) | 0.05 |
| vs [Vision](#vision-index) | 0.77 (47) | 0.13 |
| vs AA Intelligence Index | 0.73 (132) | 0.19 |
| vs GPQA Diamond | 0.67 (132) | 0.21 |
| vs HLE | 0.64 (132) | 0.18 |

Every one of those sits below the 0.79/0.84 the [Knowledge
index](#knowledge-index) holds up as proof it is not re-measuring its siblings,
and 0.44 against Coding is the lowest figure any two of these five columns
produce. Inside the top 20 the relationship is gone entirely — **0.05** against
Knowledge, **−0.17** against Coding. Among models a reader would actually choose
between, knowing which one is more capable tells you nothing about which one
will make something up.

That is not a reweighting trick. It is one benchmark doing the work: the
hallucination rate is the most orthogonal column in this file by a wide margin
(0.53 with the Knowledge index, 0.38 with Tooling, 0.18 with Coding, 0.09 with
BrowseComp), and it is aggregated nowhere else.

### Why the anchor cannot stand alone

A hallucination rate is `incorrect / (incorrect + partial + not attempted)` — of
everything the model got wrong, the share it got *confidently* wrong. A model
that answers nothing scores zero. Perfectly.

That is not a hypothetical weakness. The current head of that column on its own
is `g9v3-3b` (a 3B model, 11.7) and `lfm2-5-2-6b` (16.0, which scores 5.3 on
AA-LCR); `gemma-3-270m` posts a creditable 30.4 while answering under 1% of the
questions correctly. Ranked on honesty alone, the most trustworthy model in the
table is one that has nothing to say.

**AA-Omniscience Accuracy is the answer to that**, and it is why the member list
is not just the one orthogonal column. It is the other half of the same
6,000-question run — how much the model actually got right, before the Omniscience
Index nets the confident errors off against it. Read together the pair says the
thing the column exists to say: *knows a lot, and admits it when it doesn't.*
Neither half means much alone.

It is weighted at **0.60 rather than parity** on purpose. Accuracy is the
*knowledge* half, and knowledge is what the other four columns already rank —
every point of weight it carries buys resistance to abstention-gaming and costs
independence. Sweeping it across 0.40 / 0.50 / 0.60 / 0.70 / 0.85 showed a
smooth monotone trade with no natural knee, so the choice is a judgement rather
than a discovery: 0.60 is the point where the abstainers are clear of the top 40
and the column still ranks trust rather than knowledge.

> **The Accuracy column is declared but not yet fetched.** `update.py` maps it
> and `artificialanalysis.py` already scrapes it, so it fills on the next AA run
> (the `update-benchmarks` workflow, every three hours). Until then
> `percentile_map()` returns `None` for it, `compute_index()` drops its weight
> from the group total, and the index computes over the other three — the same
> 132 models, at an effective group weight of 1.75 against the declared 2.35.
> The independence figures above are measured in that state and will tighten
> somewhat once Accuracy lands; the ranked field will not change.

Contributing benchmarks and why they carry the weight they do (declared group
weight **2.35**):

| Benchmark | Weight | Rationale |
| --- | --- | --- |
| AA-Omniscience Hallucination Rate | 1.0 | The anchor, and the reason this column exists. It is the **widest member — 132 models across 41 creators — and every one of those 132 values is a first-party Artificial Analysis run**, with no self-reports mixed in; nothing else in this file combines that reach with that provenance. Unsaturated across essentially its whole definable range (11.7–98.2, median 84.3) with a live head, 6.7 points between first and fifth. And it is the least redundant member by a distance: mean Spearman **0.40** against the other two scored members, against 0.59 and 0.61 for them. `lower_is_better: true` is declared on the column, so `percentile_map()` inverts it and a low rate ranks at the 1.0 end. Its one weakness — that it can be gamed by abstaining — is what the accuracy below is for, not a reason to weight it lower. |
| AA-Omniscience Accuracy | 0.6 | Not a discriminator in its own right so much as the **brake that lets the anchor carry 1.0**, for the reason argued above. Same run, same first-party provenance, and the [candidate audit](docs/benchmark-candidates-2026-08.md) measured its coverage at 91% of the table, level with the hallucination rate. Held well below parity because it is the knowledge half of a knowledge-and-honesty pair, and knowledge is already priced by the [Knowledge index](#knowledge-index) at 1.0 through `aa_omniscience` — the composite of this and the rate. Weighted for what it *prevents*, not what it measures. |
| AA-LCR | 0.45 | What it uniquely tests is the second failure mode and nothing else here covers it: 100 open-answer questions over real 10k–100k-token documents, each needing facts synthesised from scattered parts of the text rather than looked up — grounding in a supplied source rather than recall from weights. 112 models across 34 creators, all Artificial Analysis runs, and genuinely unsaturated (3.0–82.7, median 50.2). **Discounted hard for redundancy, which is the honest reason it is not higher**: 0.94 with the AA Intelligence Index, 0.91 with GPQA Diamond, 0.90 with the Knowledge index, 0.89 with HLE. On coverage and provenance it would earn 0.7–0.8; as measured it ranks general capability nearly as much as it ranks grounding, and importing that at full weight would turn this column into the sibling of the four it is supposed to be independent of. Its head is also the second-flattest here, 4.7 points between first and fifth. |
| IFBench | 0.3 | The second reliability axis, and mostly its own: **0.41 against the anchor**, the lowest pair in the group. It is also the only member here graded without a model in the loop — 58 output constraints held out from the small set models have overfit to, each checked by a **verification function**, so a response either satisfies the constraint or does not and there is no judge to charm. That matters more than usual in a column about trustworthiness. 116 models across 36 creators. Discounted for two things: **the weakest provenance in the group** — 95 Artificial Analysis runs, 12 from evals.report and 9 Hugging Face card self-reports, three harnesses inside one percentile-normalised column — and the flattest head anywhere here, 3.1 points between first and fifth, consistent with AA having retired it from Intelligence Index v4.1 for saturation. It is also carried at 0.20 in the [Tooling index](#tooling-index), where the same score is priced as agent competence rather than as reliability. |

### What the Trust index leaves out

- **AA-Omniscience Index** (`aa_omniscience`) — the obvious candidate, and left
  out precisely because it is too close to the members. It *is* accuracy netted
  against confident error over the same 6,000 questions; with both halves already
  aggregated here, adding the composite would spend a third of the member slots on
  one AA run and re-price the same evidence twice. It is also the 1.0 anchor of the
  [Knowledge index](#knowledge-index), which is the right home for it: the Index
  answers "how much does this model reliably know", where this column answers
  "what does it do when it doesn't". Keeping the halves separate is what lets
  them be weighted 1.0 and 0.6 rather than averaged at birth.
- **BrowseComp** — 43 models and a plausible reading (finding a verifiable answer
  rather than inventing one), but **not one of those 43 values is a first-party
  run**: 32 come from llm-stats.com and 11 from Hugging Face model cards. Weak
  provenance is a discount in the other four indexes; in a column whose entire
  subject is whether a number can be believed, it is disqualifying. Its 0.09
  correlation with the anchor is interesting enough to revisit if OpenAI ever
  publishes a board.
- **AA-Briefcase, GDPval-AA** — professional deliverables judged in part by
  pairwise LLM comparison. What they reward is quality of output, not honesty
  about its limits, and both already carry weight in
  [Tooling](#tooling-index).

### The evidence bar is inert here too

At 18% the bar is 0.315 of the effective 1.75 (0.423 of 2.35 once Accuracy
lands), and it cuts nobody at any setting in the [table
above](#why-the-evidence-bar-is-18): Trust ranks the same 132 models from 0.25
all the way down to 0.05. The anchor alone is 57% of today's effective weight
(42.6% of the declared total), every model the column ranks has that score, and
the 11 it leaves out have no score in any member. As with
[Vision](#why-the-evidence-bar-is-inert-here), what filters this column is the
availability of the underlying run, not the threshold.

## Openness Classification

The `_openness.py` module classifies models as:

- **`open`**: Open-source weights available
- **`closed`**: Closed-weight proprietary model

This information is stored in each model's `weights` metadata and affects aggregation logic (some analyses exclude closed models).

## Model Size and Context Fields

`params` and `context` come from Artificial Analysis' model pages, parsed out of the
`currentModel` payload (`parameters`, `inferenceParametersActiveBillions`,
`contextWindowTokens`) by `artificialanalysis.py`. They are handled differently on
purpose:

- **`context` is refreshed** on every `update.py` run. Sources report raw token counts,
  so `_context.py` snaps them to the advertised size (`262144` → `256k`, `131072` →
  `128k`, `1048576` → `1m`) before comparing.
- **`params` is only filled when missing**, never refreshed. AA reports *measured*
  counts, which sit just off the size a creator advertises (Qwen3-32B measures 32.8B,
  Gemma 4 E2B measures 5.1B/A2.3B). Refreshing would overwrite the advertised names
  the site displays, so a filled value is a starting point for `edit.py`, not a
  maintained one.
- **Hugging Face is the `params` fallback** (`_params.py`), used for models AA has no
  page for. Its API carries only `safetensors.total`, so those models get a total
  ("562B") and the `-A…` active half stays a manual edit.

As with scores, neither field is ever overwritten with `null`.

## GPU Fit Filter and Hosting Presets

`gpu.json` is the card catalogue behind `llm.html`'s GPU filter: pick a card, set a
card count, and the page turns that into a total VRAM budget (minus the **Reserve**
share) that shades the VRAM columns and drops models that do not fit.

Leading that panel is a compact preset strip — one chip per mittwald
[Managed Dedicated AI Hosting](https://www.mittwald.de/mstudio/ai-dedicated-hosting)
tier, which is a shortcut into that same filter and nothing more: a chip fills in the
tier's card and card count, and everything downstream behaves exactly as if the fields
had been set by hand. Clicking the armed chip again clears the filter, and editing the
fields unarms it. The strip is a full-width row of the filters grid, so it collapses
with the panel like every other control.

| Tier | Cards | Total VRAM |
|---|---|---|
| **M** | 1 × NVIDIA RTX PRO 6000 Blackwell (Server) | 96 GB |
| **L** | 2 × | 192 GB |
| **XL** | 4 × | 384 GB |

The product name is set in the hosting brand's own headline face and weight
(Proxima Nova Semibold, tracked at `-0.016em`), which `--font-hosting` leads with for
anyone who has that licensed face installed; Figtree 600 — the closest free match on
width and x-height — is what everyone else gets, and `--font-ui` backs both up.

The chips are rendered from `gpu.json`, not from hard-coded totals, so the card's
VRAM there is the single source of truth for the labels. Tiers live in
`HOSTING_PRESETS` in `llm.html` (tier name + card count); the card they point at is
`HOSTING_PRESET_VENDOR`/`HOSTING_PRESET_GPU`. If that card is missing from
`gpu.json`, the strip stays hidden rather than offering a preset it cannot apply.

## Data Quality Features

- **Model deduplication**: Same model across multiple benchmarks merged under one slug
- **Timestamp tracking**: Score update date stored for cache validation
- **Score precision**: every writer rounds onto the benchmark's grid, so a source
  reporting more digits than the site prints cannot restamp a score's date for an
  invisible change (see [Score Precision](#score-precision))
- **Derived columns**: benchmarks flagged `"derived": true` in `llm.json` are computed
  from other columns, so `add.py`/`edit.py` neither prompt for them nor offer them as
  a mapping target (see [Coding Index](#coding-index))
- **Source attribution**: Each score records its origin benchmark
- **Ignore lists**: Models or mappings can be explicitly ignored (`*-ignored.json` files)
- **Pruning**: Remove invalid or duplicate entries

## Development & Extension

### Adding a New Benchmark Source

1. Create `fetch_<benchmark>.py`:
   - Implement API client or web scraper
   - Output normalized score format
   - Map model names to canonical slugs

2. Create mapping files:
   - `model-name-mapping-<benchmark>-to-artificialanalysis.json`
   - May also need `<benchmark>-benchmark-name-mapping.json`

3. Create `update_<benchmark>_mapping.py`:
   - Fetches fresh model names from benchmark API
   - Updates mapping file

4. Create `_<benchmark>_mapping.py`, the loader the ingest reads the mapping
   through. `test_prompts.py` finds it by glob and drives it, so it has to expose
   the whole contract: `<SOURCE>_MAPPING`, `fetch_*_model_names()`,
   `load_*_to_slug_mapping()` (sentinels filtered out), `load_reviewed_*_names()`,
   `write_*_to_slug_mapping()` (a no-op under `freeze_decisions()`),
   `add_*_mapping()`, `add_*_unmappable()` and, if the source publishes weight
   availability, `add_*_closed_weights()`. Copy `_frontierswe_mapping.py`.

5. Register in `update.py`:
   - Add fetch command builder
   - Add skip flags
   - Add to orchestration flow

6. Rank the source in `_precedence.py`. Import the module and add
   `<SOURCE>_SOURCE_URL = canonical(fetch_<benchmark>.LEADERBOARD_URL)` to
   `_ranked_prefixes()` at the rung it belongs on. Read the URL off the scraper's
   own constant, never spell it out again, and prefer the most specific page: a
   bare host prefix-matches every other board the site serves.

7. List the pages it reads in `fill_source_urls.build_inventory()`, so the
   leaderboard reaches `benchmarks[].urls` and the Sources panel. An API host with
   a human-facing equivalent goes in `COVERED_BY` instead, which reports it without
   inserting it.

8. Route its reviewer in `propose.ROUTES` — `test_propose.py` fails on an unrouted
   `update_*_mapping.py` — and add the ingest to the `CASES` list in
   `test_source_collisions.py`, which pins that row order cannot change the score.

9. Decide whether the new column joins a derived index. If it does, three places
   have to agree: `derive_indexes.INDEXES`, the `description` of the index entry in
   `llm.json` (it restates the weight list), and the rationale table in this file
   (see [Coding Index](#coding-index) and [Tooling Index](#tooling-index)). Check
   the ranked counts before and after: a new member's weight joins the denominator
   and lifts the [evidence bar](#why-the-evidence-bar-is-18) for every model that
   does not have the new score.

`update-all` needs no change — it globs `update_*.py`.

### Key Dependencies

- **Python 3.7+**: Core language
- **urllib**: HTTP requests (no external network library)
- **argparse**: CLI argument parsing
- **json**: Data serialization
- **datetime**: Timestamp handling
- **pathlib**: File operations

## File Organization

```
ai-bench/
├── llm.json                    # Main unified dataset
├── llm.html                    # Web visualization
│
├── add.py                      # Add new model (interactive CLI)
├── edit.py                     # Edit model metadata
├── update.py                   # Master orchestrator (fetch all)
├── prune.py                    # Remove invalid entries
│
├── fetch_*.py                  # Benchmark data fetchers (18 files)
├── update_*_mapping.py         # Mapping sync scripts (18 files)
├── _*_mapping.py               # Mapping application modules (18 files)
│
├── derive_indexes.py           # Derived Coding, Tooling, Knowledge, Vision & Trust index columns (see above)
│
├── _scores.py                  # Score rounding grid, timestamps, derived-column helper
├── _revisions.py               # Benchmark revisions: labels, order, and the column each feeds
├── _selector.py                # Type-to-search prompt: drawing + Tab completion
├── _openness.py                # Model openness classification
├── _params.py                  # params field: AA counts, HF fallback (see below)
├── _context.py                 # context field: token counts → advertised sizes
├── check_new.py                # Detect new/dismissed models
├── _new_models.py              # New-model decisions: add vs. ignore (see below)
├── fill_source_urls.py         # Utility for URLs
├── fill_missing_source_urls.py  # Interactive backfill of missing dates/source URLs
├── sync_score_dates.py         # Timestamp synchronization
├── make_favicons.py            # Render the icon set from the logo (see below)
│
├── model-name-mapping-*.json   # Benchmark → canonical slug mappings
├── answer.py                   # Apply queued answers directly (no PR)
├── _answers.py                 # Validation and writing behind answer.py
├── _admin/                     # The admin page and its dispatch endpoint
│                               #   (leading _, so Jekyll never publishes it)
├── _pending/pending.json       # The queue as data, for answer.py and the page
├── check_new-decisions.json    # Open add/ignore questions for the proposal PR
├── check_new-dismissed.json    # AA slugs never to offer again
├── huggingface-benchmark-name-mapping.json
├── llmstats-benchmark-name-mapping.json
├── gpu.json                    # GPU configuration reference
├── model-names-*.txt           # Cached model name lists
│
├── icons/mittwald.svg             # mittwald wordmark (source of truth; inlined into llm.html)
├── icons/openbenchindex_logo.svg  # Master logo (hand-edited source of truth)
├── icons/openbenchindex_mark.svg  # Generated: logo minus the "INDEX" wordmark
├── favicon.ico / favicon_*.png    # Generated icon set
│
└── index.html / site.webmanifest  # Web assets
```

### Logo and Favicons

`icons/openbenchindex_logo.svg` is the only file to edit by hand. Three shapes
are rendered from it:

| Shape | What it is | Where it lands |
|---|---|---|
| **full** | the master as-is: dark tile, monogram, `INDEX` wordmark | `favicon_120/152/180/192/512.png`, `icons/openbenchindex_logo.png` |
| **mark** | tile cropped square around the monogram, wordmark dropped | `icons/openbenchindex_mark.svg`, `favicon_16/32/48.png`, `favicon.ico` |
| **flat** | monogram only — no tile, no shadow, B painted in `currentColor` | injected into `llm.html` between its `brand-mark` markers |

```bash
python3 -m pip install cairosvg pillow   # build-only, not in requirements.txt
./make_favicons.py                       # rewrite whatever is stale
./make_favicons.py --check               # exit 1 if a file is out of date
```

Why three: the wordmark is unreadable much below 120px, so small icons take the
crop. And the page header needs the monogram to sit directly on the page in the
current text colour — an `<img>` cannot follow the theme toggle, so that variant
is inlined into the HTML instead of linked. The crops are measured off a render
rather than hand-typed, so moving the artwork does not leave a stale bounding
box behind.

## Example Usage Patterns

### Get all scores for a model:
```bash
python3 -c "
import json
from pathlib import Path
doc = json.loads(Path('llm.json').read_text())
models = {m['slug']: m for m in doc['models']}
print(json.dumps(models['gpt-4']['scores'], indent=2))
"
```

### Find models by openness:
```bash
python3 -c "
import json
from pathlib import Path
from _openness import open_index
oi = open_index('llm.json')
print('Open models:', [k for k,v in oi.items() if v])
"
```

### Export scores for analysis:
```bash
python3 -c "
import json, csv
from pathlib import Path
doc = json.loads(Path('llm.json').read_text())
rows = []
for m in doc['models']:
    for source, score_data in m['scores'].items():
        rows.append({
            'model': m['slug'],
            'source': source,
            'score': score_data.get('score'),
            'date': score_data.get('date')
        })
json.dump(rows, open('export.json','w'), indent=2)
"
```

## Performance Characteristics

- **Index size**: ~2,300 nodes, ~6,100 edges (code graph)
- **Data size**: llm.json ~100KB+ (variable with model count)
- **Update time**: ~10-30 seconds (depending on number of active benchmarks)
- **Memory**: Minimal (loads entire llm.json into memory)

## Notes

- **No external dependencies**: Uses only Python stdlib (urllib, json, argparse)
- **Stateless design**: All state stored in JSON files and mapping files
- **Idempotent operations**: Safe to re-run update scripts
- **Lenient parsing**: Handles missing fields, malformed data gracefully

---

## Live Demo

See https://dgrieser.github.io/ai-bench/ for OpenBench Index, aggregated benchmarks of open-weight LLMs.

