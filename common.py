#!/usr/bin/env python3
"""
Shared utilities for free-model-pulse.

Provides:
- Path management for data directories
- Atomic file operations (write to temp, then rename)
- CSV append helpers with header safety
- JSON/JL file helpers
- Environment variable access
- Structured logging helpers
"""

import os
import sys
import csv
import json
import uuid
import logging
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from functools import wraps


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


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger("free_model_pulse")
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str = "free_model_pulse") -> logging.Logger:
    return logging.getLogger(name)


LOG = get_logger()


def get_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json_write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        shutil.move(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def atomic_csv_write(path: Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

        shutil.move(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_json(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_json(path: Path, data: dict) -> None:
    atomic_json_write(path, data)


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, ensure_ascii=False) + "\n"

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

        os.unlink(temp_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


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


def ensure_csv_header(path: Path, columns: list[str]) -> bool:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header == columns:
                    return False
        except (csv.Error, IOError):
            pass

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
    return True


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(row.keys())
    is_new = ensure_csv_header(path, columns)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_prompts() -> dict:
    prompt_file = os.getenv("BENCHMARK_PROMPT_FILE", "prompts.json")
    path = Path(prompt_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "unknown", "prompts": [], "default_prompt_id": None}


ensure_csv = ensure_csv_header
