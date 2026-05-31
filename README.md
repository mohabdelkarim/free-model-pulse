# Free Model Pulse

Automated benchmark and observability system for all currently available FREE LLMs on OpenRouter.

## Overview

Free Model Pulse automatically discovers free models from OpenRouter's Models API, benchmarks them using stable prompts, stores historical data, detects model catalog changes, and generates clean derived metrics over time.

### Key Features

- **Auto-discovery**: Dynamically discovers free models from OpenRouter on every run
- **No hardcoded lists**: Model discovery happens entirely from the OpenRouter API
- **Historical tracking**: Stores catalog snapshots (JSON), catalog history (JSONL), raw benchmarks (CSV), and aggregated metrics (CSV)
- **Resilient**: Handles rate limits, timeouts, and transient upstream failures
- **Event-driven**: GitHub Actions workflows trigger benchmarks when new free models appear
- **Clean architecture**: Separate concerns for discovery, benchmarking, aggregation, and orchestration

## Architecture

```
free-model-pulse/
├── common.py              # Shared utilities (paths, CSV, JSON helpers)
├── watch_models.py        # Model discovery and watcher
├── benchmark.py           # Benchmarking module
├── analyze.py             # Aggregation and analysis module
├── prompts.json           # Benchmark prompt definitions
├── data/
│   ├── catalog/
│   │   ├── current_free_models.json     # Current free model snapshot
│   │   └── free_models_history.jsonl   # Historical catalog log
│   ├── raw/
│   │   └── benchmark_runs.csv           # Raw benchmark results (append-only)
│   └── derived/
│       └── model_index.csv             # Aggregated model metrics
├── .github/workflows/
│   ├── model-watch.yml   # Model discovery workflow
│   └── benchmark.yml     # Benchmark and analysis workflow
└── .env.example          # Environment configuration
```

### Data Layers

1. **Catalog Snapshots** (`data/catalog/`):
   - `current_free_models.json` - Current free model snapshot
   - `free_models_history.jsonl` - Historical catalog changes (append-only)

2. **Raw Benchmark Records** (`data/raw/benchmark_runs.csv`):
   - One row per model per run (append-only CSV)
   - Fields: run_id, benchmark_reason, timestamp_utc, model_id, latency_sec, status, etc.

3. **Derived Index** (`data/derived/model_index.csv`):
   - Aggregated metrics per model
   - Fields: total_runs, success_rate, latency_sec_avg/median/p95, tokens_per_sec_avg, etc.

## Setup

### Prerequisites

- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Installation

```bash
git clone https://github.com/yourusername/free-model-pulse.git
cd free-model-pulse
pip install requests
cp .env.example .env
# Edit .env with your API key
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Yes |
| `OPENROUTER_SITE_URL` | Your site URL for OpenRouter requests | No |
| `OPENROUTER_SITE_EMAIL` | Your email for OpenRouter requests | No |
| `BENCHMARK_TIMEOUT_SEC` | Request timeout in seconds (default: 120) | No |
| `BENCHMARK_MAX_RETRIES` | Max retries on failure (default: 3) | No |
| `BENCHMARK_PROMPT_FILE` | Path to prompts.json (default: prompts.json) | No |

## Usage

### Discover Models

```bash
python watch_models.py
```

Options:
- `--watch` - Run continuously in watch mode
- `--interval N` - Check every N minutes (default: 60)
- `--force` - Force re-fetch even if unchanged

### Run Benchmarks

```bash
python benchmark.py
```

Options:
- `--reason` - Benchmark reason: `manual`, `scheduled`, or `new_model_detected`
- `--prompt` - Prompt ID to use
- `--new-only` - Benchmark only current models

### Analyze Results

```bash
python analyze.py --min-runs 3
```

Options:
- `--min-runs N` - Minimum runs for aggregation (default: 3)
- `--stats` - Show summary statistics

## Benchmark Log Fields

Each benchmark run records:

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `benchmark_reason` | scheduled / new_model_detected / manual |
| `timestamp_utc` | Run timestamp |
| `prompt_version` | Prompt version from prompts.json |
| `model_id` | OpenRouter model ID |
| `canonical_family` | Provider/family name |
| `display_name` | Human-readable model name |
| `context_length` | Maximum context length |
| `latency_sec` | Response latency in seconds |
| `status` | success / error / timeout |
| `error_message` | Error details if failed |
| `response_id` | OpenRouter response ID |
| `finish_reason` | Completion finish reason |
| `prompt_tokens` | Tokens in prompt |
| `completion_tokens` | Tokens in completion |
| `total_tokens` | Total tokens used |
| `cached_tokens` | Cached tokens (if available) |
| `cache_write_tokens` | Cache write tokens (if available) |
| `reasoning_tokens` | Reasoning tokens (if available) |
| `cost` | Calculated cost |

## Derived Index Fields

Aggregated per model:

| Field | Description |
|-------|-------------|
| `model_id` | OpenRouter model ID |
| `canonical_family` | Provider/family name |
| `display_name` | Human-readable name |
| `context_length` | Maximum context length |
| `total_runs` | Total benchmark runs |
| `successful_runs` | Successful runs |
| `failed_runs` | Failed runs |
| `success_rate` | Success ratio |
| `error_rate` | Error ratio |
| `latency_sec_avg` | Average latency |
| `latency_sec_median` | Median latency |
| `latency_sec_p95` | 95th percentile latency |
| `latency_sec_min` | Minimum latency |
| `latency_sec_max` | Maximum latency |
| `total_tokens_avg` | Average total tokens |
| `completion_tokens_avg` | Average completion tokens |
| `tokens_per_sec_avg` | Average tokens per second |
| `cost_avg` | Average cost per run |
| `cost_total` | Total cost |
| `first_seen` | First benchmark timestamp |
| `last_seen` | Most recent benchmark timestamp |

## GitHub Actions

### Required Secrets

- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Required Variables

- `OPENROUTER_SITE_URL`: Your site URL (optional but recommended)
- `OPENROUTER_SITE_EMAIL`: Your contact email (optional)

### Workflows

1. **Model Watch** (every 6 hours):
   - Polls OpenRouter for model catalog changes
   - Detects new/removed free models
   - Stores catalog snapshot and history

2. **Benchmark & Analysis** (daily + on-demand):
   - Discovers current free models
   - Benchmarks each model
   - Generates aggregated model index
   - Commits updated data back to repo

## Design Principles

1. **No hardcoded model lists**: All model discovery is dynamic from the OpenRouter API
2. **No `openrouter/free`**: Each model is benchmarked individually for historical accuracy
3. **Separation of concerns**: Discovery, benchmarking, and analysis are separate modules
4. **Raw over derived**: Raw data is preserved; derived data is always regeneratable
5. **Fail gracefully**: One model failure doesn't crash the entire benchmark run
6. **Deterministic files**: CSV/JSON files over opaque database state
7. **Append-only logs**: Historical data is never overwritten
8. **Secrets management**: API keys never committed; `.env.example` for configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests if available
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This project is not affiliated with OpenRouter. It's an independent observability tool for tracking free LLM availability and performance over time.
