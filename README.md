# AI Benchmark Aggregator (`ai-bench`)

A comprehensive system for collecting, normalizing, and aggregating LLM benchmark scores across 11+ benchmark sources. This tool creates a unified dataset of AI model performance metrics from diverse evaluation platforms.

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
| **Hugging Face** | Community | Model card READMEs + Hub eval metadata (evalResults / model-index) |
| **DeepSWE** | Research | JSON API |
| **FrontierSWE** | Research | JSON API |
| **SWE ReBench** | Research | JSON API |
| **SWE Atlas** | Research | JSON API |
| **SWE Marathon** | Research | JSON API |
| **OSWorld** | Research | JSON API |
| **Spheron** | Infrastructure | JSON API |
| **LLMStats** | Community Aggregator | JSON API |
| **Evals Report** | Research | JSON API |
| **FrontierCode** | Research (Cognition) | Static leaderboard JSON |
| **DeepSWE (Datacurve)** | Research (benchmark's own site) | Versioned JSON artifact |

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
  underscores a score stamped within the last seven days in red, the stroke fading as
  the score ages. Two cases are left unmarked: a model added inside the same window,
  which carries the `NEW` badge instead because every one of its scores arrived with
  it, and a derived column, whose date moves when its inputs are recomputed rather
  than when anything new is measured
- **`scores_source`**: Same key set → URL of the page the score was read from
  (null for hand edits; the derived Coding index cites this repository, which is
  where it is computed)
- **`params`** / **`context`** / **`vram`** / **`creator`**: model metadata

The three score maps carry the full benchmark key set with null placeholders;
`update.py` stamps date and source URL together whenever it writes a score.
Values no scraper wrote (hand edits, rows that have since moved) keep a null
date or source; `fill_missing_source_urls.py` asks for those, plus a missing
`vram_source`, model or creator URL.

Underneath the table, `llm.html` writes those same seven days out as a
**Recently Added** panel: a timeline of the days something landed, each holding
the models added that day and, for models already in the index, the individual
scores stamped that day with their value and a link to the source they were read
from. Membership is decided by the same tests as the marks in the table, over the
models and columns the table is currently showing, so the two cannot disagree —
including what they leave out, a derived column and the scores a brand-new model
arrived with (counted beside the model instead). A day longer than twelve models
folds its tail behind a disclosure, since one fetch can restamp a hundred rows at
once. Everything reads as *added*: a date records only the last write, so a score
measured a second time cannot be told from a first one.

### Score Precision

Every score is rounded onto a per-benchmark grid before it is stored, by
`_scores.round_score()`, called from the one place each writer funnels through:
`apply_score()` in `update.py` for the scrapers, `collect_updates()` in
`edit.py` for hand edits. Two fields on the benchmark set the grid:

| Field | Default | Meaning |
| --- | --- | --- |
| `decimals` | `1` | Digits `llm.html` and `llm-cli` print |
| `round_to` | `10 ** -decimals` | Step a stored value is snapped to |

The default exists because sources disagree about precision: swe-rebench
reports `48.77` where Artificial Analysis reports `48.8`, and the site prints
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
│  fetch_swe_rebench.py     │ fetch_spheron.py            │
│  fetch_llmstats.py        │ fetch_evals_report.py       │
│  fetch_frontiercode.py    │ fetch_datacurve.py          │
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
./fetch_huggingface.py --repo owner/model-name
./fetch_deepswe.py
./fetch_frontierswe.py
./fetch_osworld.py
./fetch_spheron.py
./fetch_swe_atlas.py
./fetch_swe_marathon.py
./fetch_swe_rebench.py
./fetch_llmstats.py
./fetch_evals_report.py
./fetch_datacurve.py                    # DeepSWE, from the benchmark's own site
./fetch_datacurve.py --all-configs      # every harness/effort row, not the best
./fetch_frontiercode.py                 # every revision, newest wins per model
./fetch_frontiercode.py --revision 1.0  # or pin one revision

# Update model name mappings from source APIs
./update_artificialanalysis_mapping.py
./update_deepswe_mapping.py
./update_frontierswe_mapping.py
./update_frontiercode_mapping.py
./update_huggingface_mapping.py
./update_llmstats_mapping.py
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
# Recompute the derived Coding index column (dry-run; -w to persist)
./derive_coding_index.py llm.json
./derive_coding_index.py llm.json -w

# Fill in missing source URLs for benchmark records
./fill_source_urls.py llm.json

# Ask for the dates and source URLs nothing can derive (dry-run; -w to persist)
./fill_missing_source_urls.py llm.json
./fill_missing_source_urls.py llm.json -w
./fill_missing_source_urls.py llm.json --list                  # report the gaps, ask nothing
./fill_missing_source_urls.py llm.json -m kimi-k3 -w           # one model only
./fill_missing_source_urls.py llm.json --only vram-source -w   # one kind of gap only

# Check for newly added/dismissed models
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
own leaderboard does not list. `fetch_datacurve.py` (DeepSWE's own leaderboard)
stands in the same relation to `fetch_deepswe.py`, which reads benchlm.ai.

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

The first benchmark column, **Coding**, is the only score in `llm.json` that is not
scraped: `derive_coding_index.py` computes it from the coding benchmarks already in
the file and writes it to each model's `scores.coding_index`. It is also the table's
default sort.

How a value is produced:

1. **Rank, don't average raw numbers.** Every contributing benchmark is turned into
   a tie-averaged percentile rank across the models scored on it, so a pass rate and
   an index score can be compared at all. A `lower_is_better` benchmark is inverted,
   so a percentile always means "how good".
2. **Weight by reliability.** The ranks are averaged with the per-benchmark weights
   in `CONTRIBUTING` (DeepSWE 1.0 down to SWE-bench Verified 0.15), so the
   benchmarks worth trusting lead and the weaker ones fill gaps and break ties.
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
   `MIN_SCORED_FRACTION` (20%) of that weight is left unranked (`null`) rather than
   reported as a mostly-imputed number.

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
repository instead of to a scraped page: `derive_coding_index.py` stamps
`scores_source.coding_index` with the first URL the column declares in
`llm.json` (`https://github.com/dgrieser/ai-bench#coding-index`, this section), so
clicking the score in `llm.html` opens the method rather than nothing. Change that
URL in `llm.json` and the next run restamps every value. An unranked model reports
no source, the same way it reports no date.

Two consequences worth knowing:

- Ranks are relative to the models currently in the file, so **adding a model or a
  score moves other models' values**. That is also why `derive_coding_index.py`
  clears a value back to `null` when a model stops qualifying, unlike the scrapers,
  which never overwrite a value with `null`.
- The column has to be recomputed after every change to `scores` **or** to the set of
  models, and every writer that can cause one does it for you, in the same write, via
  `derive_coding_index.refresh_and_report()`:
  - `update.py -w` — after the scrapers have merged their scores (so a direct run is
    self-sufficient; `update-all` additionally runs the script as its last step).
  - `edit.py` — after a hand-edited score. Skipped for a params/context-only edit,
    which cannot move a rank.
  - `prune.py -w` — after dropping models, because removing one that carried a coding
    score re-ranks the survivors even though none of their own scores moved.

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
├── fetch_*.py                  # Benchmark data fetchers (13 files)
├── update_*_mapping.py         # Mapping sync scripts (13 files)
├── _*_mapping.py               # Mapping application modules (13 files)
│
├── derive_coding_index.py      # Derived Coding index column (see above)
│
├── _scores.py                  # Score rounding grid, timestamps, derived-column helper
├── _selector.py                # Type-to-search prompt: drawing + Tab completion
├── _openness.py                # Model openness classification
├── _params.py                  # params field: AA counts, HF fallback (see below)
├── _context.py                 # context field: token counts → advertised sizes
├── check_new.py                # Detect new/dismissed models
├── fill_source_urls.py         # Utility for URLs
├── fill_missing_source_urls.py  # Interactive backfill of missing dates/source URLs
├── sync_score_dates.py         # Timestamp synchronization
│
├── model-name-mapping-*.json   # Benchmark → canonical slug mappings
├── huggingface-benchmark-name-mapping.json
├── llmstats-benchmark-name-mapping.json
├── gpu.json                    # GPU configuration reference
├── model-names-*.txt           # Cached model name lists
│
└── index.html / site.webmanifest  # Web assets
```

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

