#!/usr/bin/env python3
"""
Shared utilities for free-model-pulse.

Provides:
- Path management for data directories
- CSV append helpers
- JSON/JL file helpers
- Environment variable access
"""

import os
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CATALOG_DIR = DATA_DIR / "catalog"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"

CATALOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_MODELS_FILE = CATALOG_DIR / "current_free_models.json"
HISTORY_FILE = CATALOG_DIR / "free_models_history.jsonl"
BENCHMARK_RUNS_FILE = RAW_DIR / "benchmark_runs.csv"
MODEL_INDEX_FILE = DERIVED_DIR / "model_index.csv"

BENCHMARK_CSV_COLUMNS = [
    "run_id",
    "benchmark_reason",
    "timestamp_utc",
    "prompt_version",
    "model_id",
    "canonical_family",
    "display_name",
    "context_length",
    "latency_sec",
    "status",
    "error_message",
    "response_id",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost",
]

INDEX_COLUMNS = [
    "model_id",
    "canonical_family",
    "display_name",
    "context_length",
    "total_runs",
    "successful_runs",
    "failed_runs",
    "success_rate",
    "error_rate",
    "latency_sec_avg",
    "latency_sec_median",
    "latency_sec_p95",
    "latency_sec_min",
    "latency_sec_max",
    "total_tokens_avg",
    "completion_tokens_avg",
    "tokens_per_sec_avg",
    "cost_avg",
    "cost_total",
    "last_seen",
    "first_seen",
]


def get_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def ensure_csv(path: Path, columns: list[str]) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    ensure_csv(path, list(row.keys()))
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def load_prompts() -> dict:
    prompt_file = os.getenv("BENCHMARK_PROMPT_FILE", "prompts.json")
    path = Path(prompt_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "unknown", "prompts": [], "default_prompt_id": None}
