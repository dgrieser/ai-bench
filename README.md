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
| **DeepSWE** | Research | JSON API |
| **FrontierSWE** | Research | JSON API |
| **SWE Atlas** | Research | JSON API |
| **MCP-Atlas (Scale Labs)** | Research (benchmark's own leaderboard) | RSC flight payload |
| **SWE Marathon** | Research | JSON API |
| **OSWorld** | Research | JSON API |
| **Spheron** | Infrastructure | JSON API |
| **LLMStats** | Community Aggregator | JSON API |
| **Evals Report** | Research | JSON API |
| **FrontierCode** | Research (Cognition) | Static leaderboard JSON |
| **DeepSWE (Datacurve)** | Research (benchmark's own site) | Versioned JSON artifact |
| **BFCL (Berkeley/Gorilla)** | Research (benchmark's own leaderboard) | CSV the page hydrates from |

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
  (null for hand edits; the derived Coding index cites this repository, which is
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
./fetch_aa_coding_agents.py              # DeepSWE, SWE-Atlas-QnA, Terminal-Bench 2.1 as run by AA
./fetch_huggingface.py --repo owner/model-name
./fetch_deepswe.py
./fetch_frontierswe.py
./fetch_osworld.py
./fetch_spheron.py
./fetch_swe_atlas.py
./fetch_swe_marathon.py
./fetch_mcp_atlas.py                    # MCP-Atlas, from Scale's own leaderboard
./fetch_bfcl.py                         # BFCL v4 Overall Accuracy, from the Gorilla team
./fetch_llmstats.py
./fetch_evals_report.py
./fetch_datacurve.py                    # DeepSWE, from the benchmark's own site
./fetch_datacurve.py --all-configs      # every harness/effort row, not the best
./fetch_frontiercode.py                 # every revision, newest wins per model
./fetch_frontiercode.py --revision 1.0  # or pin one revision

# Update model name mappings from source APIs
./update_aa_coding_agents_mapping.py
./update_artificialanalysis_mapping.py
./update_bfcl_mapping.py
./update_deepswe_mapping.py
./update_frontierswe_mapping.py
./update_frontiercode_mapping.py
./update_huggingface_mapping.py
./update_llmstats_mapping.py
./update_mcp_atlas_mapping.py
./update_osworld_mapping.py
./update_spheron_mapping.py
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
# Recompute the derived index columns, Coding and Tooling (dry-run; -w to persist)
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

### 3. Updating All Benchmarks

```bash
./update.py llm.json
```

This orchestrates:
1. Runs all `fetch_*.py` scripts
2. Runs all `update_*_mapping.py` scripts
3. Merges results into `llm.json`
4. Updates timestamps

Sources are applied in a fixed order and a later one overwrites an earlier
value, so the order encodes precedence: a benchmark's own site runs *after* the
aggregator that republishes it. `fetch_swe_marathon.py` and
`fetch_frontiercode.py` (Cognition's leaderboard) therefore run after
`fetch_evals_report.py`, and evals.report only supplies models the benchmark's
own leaderboard does not list. SWE-bench Multimodal is the one column where
that is reversed: nothing scrapes swebench.com, so evals.report *is* the
primary source and does overwrite, with the model cards filling gaps ahead of
it. `fetch_datacurve.py` (DeepSWE's own leaderboard) stands in the same
relation to `fetch_deepswe.py`, which reads benchlm.ai. `fetch_mcp_atlas.py`
(Scale's own MCP-Atlas board) and `fetch_bfcl.py` (the Gorilla team's BFCL
leaderboard) run last for the same reason: evals.report, llm-stats and the
model cards all republish self-reported numbers for those two benchmarks, and
where both a first-party run and a self-report exist they disagree by a point
or two, so the first-party run has to be the one that lands.
`fetch_aa_coding_agents.py` (Artificial Analysis' Coding Agent Index, its own
agent-harness runs of DeepSWE, SWE-Atlas-QnA and Terminal-Bench 2.1) is
*fill-only*, like the Hugging Face and llm-stats aggregates: AA's runs disagree
systematically with the benchmarks' own leaderboards, so overwriting would flip
a score within one run and restamp its date every refresh. It runs ahead of the
other gap-fillers so a gap AA measured directly is filled by that measurement
rather than a self-report; the leading sources overwrite either way.

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

All three feed the [Tooling index](#tooling-index); the weight each carries, and
why, is in that section's table.

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
[Tooling index](#tooling-index) is its sibling — same script, same math, different
contributing benchmarks.

How a value is produced:

1. **Rank, don't average raw numbers.** Every contributing benchmark is turned into
   a tie-averaged percentile rank across the models scored on it, so a pass rate and
   an index score can be compared at all. A `lower_is_better` benchmark is inverted,
   so a percentile always means "how good".
2. **Weight by reliability.** The ranks are averaged with the per-benchmark weights
   the index declares in `INDEXES` (DeepSWE 1.0 down to SWE-bench Verified 0.15), so
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

`MIN_SCORED_FRACTION` is the one number both derived indexes share: sum the weights of
the contributing benchmarks a model actually has a score on, and if that is below
`MIN_SCORED_FRACTION × total group weight`, the model is `null` instead of ranked. It
is a share of *weight*, not a count of benchmarks — three cheap columns can be worth
less evidence than one expensive one.

At **0.18** the bars are 1.152 of 6.40 (Coding, 84 of 136 models ranked) and 1.107 of
6.15 (Tooling, 85 ranked).

Coverage does not spread evenly across models, it clusters, and the threshold should
fall between clusters rather than through one. Measured over the current file:

| fraction | Coding ranked | Tooling ranked | what the cut admits |
| --- | --- | --- | --- |
| 0.25 | 68 | 82 | |
| 0.20 | 73 | 85 | the 20%-measured block, cut in half |
| **0.18** | **84** | **85** | the rest of that block: 11 models at 19% |
| 0.15 | 84 | 87 | nothing in Coding |
| 0.12 | 97 | 114 | the 13–14% cluster |
| 0.10 | 127 | 115 | the ~12% cluster |
| 0.05 | 135 | 127 | models measured on a twentieth of the weight |

0.20 was splitting a natural block: eleven models sit at 19% and none between 19% and
20%, so `kimi-k2-thinking` and `deepseek-v3-2-0925` (both scored on LiveCodeBench,
SciCode, SWE-bench Multilingual and SWE-bench Verified) were unranked while models with
the same amount of evidence were not. 0.18 takes the whole block and stops before the
next one. Lowering the bar never changes a ranked model's *value* — the total weight is
the same either way — it only decides who appears; previously ranked models move a mean
of 3 places as the new arrivals slot in, and Tooling does not move at all.

Going further would change what the column means rather than how much of it is filled
in. At 0.12, **90%** of ranked models would be measured on less than half the weight;
at 0.10, 92%. The head stays safe at any setting — the imputation cap means a model
measured on 12% of the weight cannot score above 56,000 even by topping every benchmark
it has, and no model under 25% measured reaches the visible top 20 at any fraction I
tested — but the mid-field crowds: the ten models measured on at least half the weight
fall a mean 5 places at 0.12 and 11 places at 0.10, below models whose numbers are
mostly imputation. 18% keeps the ranked field majority-measured; 12% does not.

The setting is global to both indexes today. They do not want the same thing — Tooling
is flat between 0.20 and 0.15 where Coding gains 11 models — so if the two ever need to
diverge, the place for it is a per-index override on `IndexDef`, not a compromise
value.

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

4. Register in `update.py`:
   - Add fetch command builder
   - Add skip flags
   - Add to orchestration flow

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
├── fetch_*.py                  # Benchmark data fetchers (16 files)
├── update_*_mapping.py         # Mapping sync scripts (16 files)
├── _*_mapping.py               # Mapping application modules (16 files)
│
├── derive_indexes.py           # Derived Coding & Tooling index columns (see above)
│
├── _scores.py                  # Score rounding grid, timestamps, derived-column helper
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

