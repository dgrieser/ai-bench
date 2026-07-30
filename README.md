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
| **Hugging Face** | Community | Model card READMEs |
| **DeepSWE** | Research | JSON API |
| **FrontierSWE** | Research | JSON API |
| **SWE ReBench** | Research | JSON API |
| **SWE Atlas** | Research | JSON API |
| **SWE Marathon** | Research | JSON API |
| **OSWorld** | Research | JSON API |
| **Spheron** | Infrastructure | JSON API |
| **LLMStats** | Community Aggregator | JSON API |
| **Evals Report** | Research | JSON API |

## Core Data Structure

### `llm.json` Format

```json
{
  "models": [
    {
      "slug": "gpt-4-turbo",
      "names": ["gpt-4-turbo-preview", "GPT-4 Turbo"],
      "created": "2024-01-15",
      "openness": "closed",
      "weights": { ... },
      "scores": {
        "artificial-analysis": { "score": 95.2, "date": "2024-07-29" },
        "huggingface": { "score": 94.1, "date": "2024-07-28" },
        "deepswe": { "score": 92.3, "date": "2024-07-25" },
        ...
      }
    },
    ...
  ]
}
```

Each model object contains:
- **`slug`**: Canonical identifier (used across system)
- **`names`**: Aliases this model is known by
- **`created`**: Initial entry date
- **`openness`**: "open" or "closed" (weights/licensing information)
- **`weights`**: Model size, quantization, and licensing details
- **`scores`**: Nested dict mapping benchmark source → score record

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
│  _scores.py: score update logic & timestamps            │
│  _openness.py: model openness classification            │
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

# Update model name mappings from source APIs
./update_artificialanalysis_mapping.py
./update_deepswe_mapping.py
./update_frontierswe_mapping.py
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

# Synchronize score update timestamps
./sync_score_dates.py llm.json
```

### Utilities

```bash
# Fill in missing source URLs for benchmark records
./fill_source_urls.py llm.json

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

## Mapping System

The project uses a **multi-layer mapping strategy** to handle model name fragmentation:

### Layer 1: Canonical Slugs

Every model has a `slug` (e.g., `gpt-4-turbo`) used throughout the system.

### Layer 2: Benchmark-Specific Mappings

Each benchmark has a mapping file:
- `model-name-mapping-artificialanalysis.json`
- `model-name-mapping-deepswe-to-artificialanalysis.json`
- `model-name-mapping-huggingface-to-artificialanalysis.json`
- etc.

Maps benchmark-specific names → canonical model slugs.

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

## Openness Classification

The `_openness.py` module classifies models as:

- **`open`**: Open-source weights available
- **`closed`**: Closed-weight proprietary model

This information is stored in each model's `weights` metadata and affects aggregation logic (some analyses exclude closed models).

## Data Quality Features

- **Model deduplication**: Same model across multiple benchmarks merged under one slug
- **Timestamp tracking**: Score update date stored for cache validation
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
├── fetch_*.py                  # Benchmark data fetchers (11 files)
├── update_*_mapping.py         # Mapping sync scripts (8 files)
├── _*_mapping.py               # Mapping application modules (11 files)
│
├── _scores.py                  # Score update timestamp logic
├── _openness.py                # Model openness classification
├── check_new.py                # Detect new/dismissed models
├── fill_source_urls.py         # Utility for URLs
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

