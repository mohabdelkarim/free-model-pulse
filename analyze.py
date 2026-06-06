#!/usr/bin/env python3
"""
Analysis Module.

Aggregates raw benchmark results into a summary index with all required metrics:
  success_rate, error_rate, avg/median/p95 latency, prompt/completion/total tokens avg,
  tokens_per_sec, avg_cost, availability, last_seen, total_runs.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from common import (
    BENCHMARK_RUNS_FILE,
    MODEL_INDEX_FILE,
    read_csv,
    ensure_csv,
    safe_float,
    safe_int,
    now_iso,
    setup_logging,
    get_logger,
)

LOG = get_logger("analyze")

MODEL_INDEX_FIELDS = [
    "model_id",
    "display_name",
    "canonical_family",
    "context_length",
    "total_runs",
    "successful_runs",
    "failed_runs",
    "success_rate",
    "error_rate",
    "availability",
    "latency_sec_avg",
    "latency_sec_median",
    "latency_sec_p95",
    "latency_sec_min",
    "latency_sec_max",
    "prompt_tokens_avg",
    "completion_tokens_avg",
    "total_tokens_avg",
    "tokens_per_sec_avg",
    "cost_avg",
    "cost_total",
    "first_seen",
    "last_seen",
]


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = min(int(len(sorted_vals) * percentile / 100), len(sorted_vals) - 1)
    return sorted_vals[index]


def aggregate_model_metrics(rows: list[dict], min_runs: int = 1) -> list[dict]:
    model_data: dict = defaultdict(lambda: {
        "runs": set(),
        "successes": 0,
        "failures": 0,
        "latencies": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "costs": [],
        "timestamps": [],
        "run_ids_seen": set(),   # unique run_ids to compute availability
    })

    all_run_ids: set = set()

    for row in rows:
        model_id = row.get("model_id", "").strip()
        if not model_id:
            continue

        run_id = row.get("run_id", "").strip()
        if run_id:
            all_run_ids.add(run_id)

        entry = model_data[model_id]
        entry["model_id"] = model_id
        entry["canonical_family"] = row.get("canonical_family", "unknown")
        entry["display_name"] = row.get("display_name", model_id)
        entry["context_length"] = row.get("context_length")

        if run_id:
            entry["run_ids_seen"].add(run_id)

        status = row.get("status", "")
        timestamp = row.get("timestamp_utc", "")

        if status == "success":
            entry["successes"] += 1

            latency = safe_float(row.get("latency_sec"))
            if latency > 0:
                entry["latencies"].append(latency)

            prompt_toks = safe_int(row.get("prompt_tokens"))
            completion_toks = safe_int(row.get("completion_tokens"))
            total_toks = safe_int(row.get("total_tokens"))

            if prompt_toks > 0:
                entry["prompt_tokens"].append(prompt_toks)
            if completion_toks > 0:
                entry["completion_tokens"].append(completion_toks)
            if total_toks > 0:
                entry["total_tokens"].append(total_toks)

            cost = safe_float(row.get("cost"))
            if cost >= 0:
                entry["costs"].append(cost)

            if timestamp:
                entry["timestamps"].append(timestamp)
        else:
            entry["failures"] += 1

    total_unique_runs = len(all_run_ids) if all_run_ids else 1

    aggregated = []
    for model_id, data in model_data.items():
        total_runs = data["successes"] + data["failures"]
        if total_runs < min_runs:
            continue

        latencies = data["latencies"]
        costs = data["costs"]

        tokens_per_sec = [
            total_toks / latency
            for latency, total_toks in zip(latencies, data["total_tokens"])
            if latency > 0 and total_toks > 0
        ]

        timestamps = sorted(data["timestamps"])
        first_seen = timestamps[0] if timestamps else None
        last_seen = timestamps[-1] if timestamps else None

        # availability = fraction of total runs where this model was tested
        runs_seen = len(data["run_ids_seen"])
        availability = round(runs_seen / total_unique_runs, 4) if total_unique_runs > 0 else None

        agg = {
            "model_id": model_id,
            "display_name": data["display_name"],
            "canonical_family": data["canonical_family"],
            "context_length": data.get("context_length"),
            "total_runs": total_runs,
            "successful_runs": data["successes"],
            "failed_runs": data["failures"],
            "success_rate": round(data["successes"] / total_runs, 4) if total_runs > 0 else 0.0,
            "error_rate": round(data["failures"] / total_runs, 4) if total_runs > 0 else 0.0,
            "availability": availability,
            "latency_sec_avg": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_sec_median": round(statistics.median(latencies), 3) if latencies else None,
            "latency_sec_p95": round(compute_percentile(latencies, 95), 3) if latencies else None,
            "latency_sec_min": round(min(latencies), 3) if latencies else None,
            "latency_sec_max": round(max(latencies), 3) if latencies else None,
            "prompt_tokens_avg": round(statistics.mean(data["prompt_tokens"])) if data["prompt_tokens"] else None,
            "completion_tokens_avg": round(statistics.mean(data["completion_tokens"])) if data["completion_tokens"] else None,
            "total_tokens_avg": round(statistics.mean(data["total_tokens"])) if data["total_tokens"] else None,
            "tokens_per_sec_avg": round(statistics.mean(tokens_per_sec), 2) if tokens_per_sec else None,
            "cost_avg": round(statistics.mean(costs), 8) if costs else 0.0,
            "cost_total": round(sum(costs), 8) if costs else 0.0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }
        aggregated.append(agg)

    aggregated.sort(key=lambda x: (-x["success_rate"], x.get("latency_sec_median") or 9999))
    return aggregated


def generate_model_index(min_runs: int = 1) -> dict:
    LOG.info("Loading benchmark data...")
    rows = read_csv(BENCHMARK_RUNS_FILE)
    LOG.info("Loaded %d benchmark records", len(rows))

    if not rows:
        LOG.warning("No benchmark data found")
        return {"success": False, "error": "No benchmark data found"}

    LOG.info("Aggregating metrics (min_runs=%d)...", min_runs)
    aggregated = aggregate_model_metrics(rows, min_runs=min_runs)
    LOG.info("Aggregated %d models", len(aggregated))

    if not aggregated:
        LOG.warning("No models met minimum run threshold of %d", min_runs)
        return {"success": False, "error": "No models met minimum run threshold"}

    MODEL_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_INDEX_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_INDEX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregated)

    LOG.info("Model index saved: %s (%d models)", MODEL_INDEX_FILE, len(aggregated))

    return {
        "success": True,
        "index_path": str(MODEL_INDEX_FILE),
        "models_aggregated": len(aggregated),
        "total_records_analyzed": len(rows),
        "generated_at": now_iso(),
    }


def get_summary_stats() -> dict:
    rows = read_csv(BENCHMARK_RUNS_FILE)
    if not rows:
        return {"success": False, "error": "No data"}

    model_ids = {r.get("model_id") for r in rows if r.get("model_id")}
    run_ids = {r.get("run_id") for r in rows if r.get("run_id")}
    successful = [r for r in rows if r.get("status") == "success"]
    latencies = [safe_float(r.get("latency_sec")) for r in successful if r.get("latency_sec")]

    return {
        "total_records": len(rows),
        "total_models": len(model_ids),
        "total_runs": len(run_ids),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": round(len(successful) / len(rows), 4) if rows else 0,
        "avg_latency_sec": round(statistics.mean(latencies), 3) if latencies else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--min-runs", type=int, default=1, help="Minimum runs for aggregation (default: 1)")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics only")
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        print(json.dumps(get_summary_stats(), indent=2))
        return

    print(json.dumps(generate_model_index(min_runs=args.min_runs), indent=2))


if __name__ == "__main__":
    main()
