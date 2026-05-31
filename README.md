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

### Workflow Overview

The project uses two workflows that work together:

```
┌─────────────────────┐
│   Model Discovery   │  Every 6 hours (schedule)
│   (model-watch.yml) │  Manual trigger (workflow_dispatch)
└─────────┬───────────┘
          │ Changes detected?
          ▼
    ┌───────────────┐
    │  New models   │──────► Manual trigger of benchmark
    │  found?      │         via workflow_dispatch
    └───────────────┘
          │
          │ No changes
          ▼
    ┌───────────────┐
    │  Exit cleanly  │
    └───────────────┘

┌─────────────────────┐
│   Benchmark &      │  Daily (schedule)
│   Analysis          │  Manual trigger (workflow_dispatch)
│   (benchmark.yml)  │  External trigger (repository_dispatch)
└─────────┬───────────┘
          │
          ▼
    ┌───────────────┐
    │  Unit Tests    │
    └───────────────┘
          │
          ▼
    ┌───────────────┐
    │  Run Benchmark │  All free models or new only
    └───────────────┘
          │
          ▼
    ┌───────────────┐
    │  Analyze Data  │  Generate model_index.csv
    └───────────────┘
          │
          ▼
    ┌───────────────┐
    │  Commit Data   │  Push updated data to repo
    └───────────────┘
```

### Trigger Matrix

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| Model Discovery | `schedule: "0 */6 * * *"` | Every 6 hours |
| Model Discovery | `workflow_dispatch` | Manual catalog refresh |
| Benchmark | `schedule: "0 0 * * *"` | Daily at midnight |
| Benchmark | `workflow_dispatch` | Manual benchmark run |
| Benchmark | `repository_dispatch` | Triggered by external systems |

### Required Secrets

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |

### Required Variables

| Variable | Required | Description |
|---------|----------|-------------|
| `OPENROUTER_SITE_URL` | No | Site URL for OpenRouter |
| `OPENROUTER_SITE_EMAIL` | No | Contact email for OpenRouter |

### Anti-Loop Protections

1. **Concurrency Groups**: Each workflow has its own concurrency group
   - `model-watch`: `cancel-in-progress: true` - Cancels overlapping watch runs
   - `benchmark`: `cancel-in-progress: false` - Allows benchmark to complete even if new triggers arrive

2. **Conditional Triggers**: The model watch workflow only commits and could trigger benchmark if actual new models are detected

3. **Idempotent Data**: Benchmark data is append-only (CSV), catalog snapshots are content-addressed by hash

4. **Git History Check**: Commit step checks `git diff --cached --quiet` before committing to avoid empty commits

### Event Flow: New Model Detection

```
1. model-watch.yml runs (schedule or manual)
2. Python script fetches OpenRouter models
3. Script compares with previous catalog (from free_models_history.jsonl)
4. If new models found:
   a. Commits updated catalog to repo
   b. Workflow completes with "changes detected" status
5. User manually triggers benchmark.yml or uses workflow_dispatch
   - Or external system sends repository_dispatch event
6. benchmark.yml runs:
   a. Unit tests pass
   b. Benchmarks all current free models
   c. Generates model_index.csv
   d. Commits updated data to repo
```

### Architecture Notes

- **`GITHUB_TOKEN` is sufficient**: No PAT needed since workflows dispatch within the same repo
- **Workflows exist on main branch**: Required for repository_dispatch to work
- **Data commits use conditional**: Empty diffs don't create commits

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

## Limitations and Interpretation Caveats

**Important: Free Model Pulse tracks operational behavior, not model intelligence.**

### What This Tool Does

- Measures latency, token throughput, and availability of *currently free* OpenRouter models
- Records historical observations over time
- Detects when new free models appear in the catalog
- Produces operational metrics useful for monitoring and planning

### What This Tool Does NOT Claim

- This is NOT a ranking of model intelligence or quality
- Results should NOT be interpreted as "which model is best"
- Latency differences do not necessarily reflect model capability

### Known Limitations

1. **Rate Limits**: Free models are frequently rate-limited. A failed benchmark may reflect OpenRouter's load, not the model's quality or availability.

2. **Provider Routing**: OpenRouter routes requests to underlying providers. Queue conditions, provider hiccups, and regional routing can significantly affect latency measurements.

3. **Dynamic Catalog Membership**: Free models can be added or removed without notice. A model absent from today's results may appear tomorrow, or vice versa.

4. **Single Prompt Benchmarking**: Default benchmarks use one simple prompt. Model performance may vary significantly on different task types.

5. **Network Variability**: GitHub Actions runners have variable network conditions. Latency measurements include this variability.

6. **Cost Field Accuracy**: The `cost` field comes from OpenRouter's response. Actual costs may vary based on provider pricing changes.

### Interpreting Results

- Use success rate to gauge overall availability
- Use latency percentiles (median, p95) to understand typical vs worst-case performance
- Use token throughput (tokens/sec) to understand generation speed
- Track changes over time, not absolute values
- Always verify with your own workloads

## FAQ

### Why doesn't this benchmark use `openrouter/free`?

The `openrouter/free` router routes requests randomly among free models and updates its routing strategy without notice. This makes it impossible to attribute results to a specific model or track historical performance per model.

### How often does the catalog change?

OpenRouter occasionally adds new free models and may retire or modify existing ones. The model watch workflow runs every 6 hours to detect these changes.

### Why did my benchmark fail?

Common reasons:
- Rate limit (429) - try again later
- Timeout - the model is slow or unavailable
- Network issue - temporary connectivity problem
- Model removed - no longer in the free catalog

### Can I run benchmarks locally?

Yes. Set `OPENROUTER_API_KEY` in your environment and run:
```bash
python watch_models.py  # Discover current models
python benchmark.py     # Run benchmarks
python analyze.py       # Generate analysis
```

### How is data committed back to the repo?

The benchmark workflow runs as a GitHub Actions job. It commits updated data files using `GITHUB_TOKEN`. This happens in the `commit` job after analysis completes.

### Do I need a PAT for repository dispatch?

No. `GITHUB_TOKEN` is sufficient for repository dispatch within the same repo.

## Roadmap / Future Work

### Near-term

- [ ] Add more benchmark prompts (coding, reasoning, creative writing)
- [ ] Implement multiple prompt variants per model for more robust metrics
- [ ] Add latency breakdown (time to first token vs total time)
- [ ] Create visualization dashboard for historical trends

### Medium-term

- [ ] Add model capability metadata (vision, function calling, etc.)
- [ ] Implement alerts for model availability drops
- [ ] Add comparison mode vs previous benchmark runs
- [ ] Support custom benchmark prompts via PR workflow

### Long-term

- [ ] Explore cross-provider free model benchmarking
- [ ] Add response quality assessment (beyond operational metrics)
- [ ] Create public API for accessing benchmark data
- [ ] Consider model "health scores" based on availability trends

## License

MIT License - see LICENSE file for details.

## Disclaimer

The benchmark results represent operational observations under specific conditions and should not be interpreted as endorsements or rankings of any model. Model availability and performance can change without notice.

This project is not affiliated with OpenRouter. It's an independent observability tool for tracking free LLM availability and performance over time.
