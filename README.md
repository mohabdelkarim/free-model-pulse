# Free Model Pulse

Automated benchmark and observability system for all currently available FREE LLMs on OpenRouter.

## Overview

Free Model Pulse automatically discovers free models from OpenRouter's Models API, benchmarks them using stable prompts, stores historical data, detects model catalog changes, and generates clean derived metrics over time.

### Key Features

- **Auto-discovery**: Dynamically discovers free models from OpenRouter on every run
- **No hardcoded lists**: Model discovery happens entirely from the OpenRouter API
- **Historical tracking**: Stores catalog snapshots, raw benchmark results, and aggregated metrics
- **Resilient**: Handles rate limits, timeouts, and transient upstream failures
- **Event-driven**: GitHub Actions workflows trigger benchmarks when new free models appear
- **Clean architecture**: Separate concerns for discovery, benchmarking, aggregation, and orchestration

## Architecture

```
free-model-pulse/
├── watch_models.py      # Model discovery module
├── benchmark.py         # Benchmarking module
├── analyze.py           # Aggregation and analysis module
├── prompts.json         # Benchmark prompt definitions
├── data/
│   ├── catalogs/        # Model catalog snapshots
│   ├── runs/            # Raw benchmark results
│   └── derived/         # Aggregated metrics and reports
├── .github/workflows/   # CI/CD automation
└── .env.example         # Environment configuration
```

### Data Layers

1. **Catalog Snapshots** (`data/catalogs/`): Current free models with historical log
2. **Raw Benchmark Records** (`data/runs/`): One row per model per run
3. **Derived Index** (`data/derived/`): Aggregated metrics per model

## Setup

### Prerequisites

- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/free-model-pulse.git
cd free-model-pulse

# Install dependencies
pip install requests

# Copy environment template
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

## Usage

### Discover Models

```bash
python watch_models.py
```

Add `--watch` for continuous monitoring:
```bash
python watch_models.py --watch --interval 60
```

### Run Benchmarks

```bash
# Benchmark all current free models
python benchmark.py

# Only benchmark newly discovered models
python benchmark.py --new-only

# Use specific prompts
python benchmark.py --prompts reasoning_simple code_simple
```

### Analyze Results

```bash
# Generate aggregated metrics
python analyze.py --window 30 --min-runs 3

# Generate summary report
python analyze.py --report

# List all benchmark runs
python analyze.py --list-runs
```

## Expected Metrics

Each benchmark result captures:

| Field | Description |
|-------|-------------|
| `timestamp` | Run timestamp |
| `run_id` | Unique run identifier |
| `model_id` | OpenRouter model ID |
| `canonical_family` | Provider/family name |
| `display_name` | Human-readable model name |
| `context_length` | Maximum context length |
| `latency_sec` | Response latency in seconds |
| `prompt_tokens` | Tokens in prompt |
| `completion_tokens` | Tokens in completion |
| `total_tokens` | Total tokens used |
| `cached_tokens` | Cached tokens (if available) |
| `cache_write_tokens` | Cache write tokens (if available) |
| `reasoning_tokens` | Reasoning tokens (if available) |
| `cost` | Calculated cost |
| `finish_reason` | Completion finish reason |
| `status` | success/error/timeout |
| `error_message` | Error details if failed |

## GitHub Actions

The project includes automated workflows:

### Model Discovery Workflow
- Runs every 6 hours (configurable)
- Detects new/removed free models
- Triggers benchmark workflow when changes detected

### Benchmark Workflow
- Runs daily (configurable)
- Discovers current free models dynamically
- Benchmarks each model with configured prompts
- Generates aggregated analysis

### Required Secrets

- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Required Variables

- `OPENROUTER_SITE_URL`: Your site URL (optional but recommended)
- `OPENROUTER_SITE_EMAIL`: Your contact email (optional)

## Design Principles

1. **No hardcoded model lists**: All model discovery is dynamic from the OpenRouter API
2. **No `openrouter/free`**: Each model is benchmarked individually for historical accuracy
3. **Separation of concerns**: Discovery, benchmarking, and analysis are separate modules
4. **Raw over derived**: Raw data is preserved; derived data is always regeneratable
5. **Fail gracefully**: One model failure doesn't crash the entire benchmark run
6. **Deterministic files**: JSON files over opaque database state
7. **Secrets management**: API keys never committed; `.env.example` for configuration

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
