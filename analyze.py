#!/usr/bin/env python3
"""
Analysis Module.

Aggregates raw benchmark results and generates derived metrics.
Produces per-model summaries with median, p95 latency, tokens/sec, etc.
"""

import json
import statistics
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

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


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(len(sorted_vals) * percentile / 100)
    index = min(index, len(sorted_vals) - 1)
    return sorted_vals[index]


def aggregate_model_metrics(rows: list[dict], min_runs: int = 3) -> list[dict]:
    model_data = defaultdict(lambda: {
        "runs": [],
        "successes": 0,
        "failures": 0,
        "latencies": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "costs": [],
        "timestamps": [],
    })

    for row in rows:
        model_id = row.get("model_id", "unknown")
        if not model_id:
            continue

        entry = model_data[model_id]
        entry["model_id"] = model_id
        entry["canonical_family"] = row.get("canonical_family", "unknown")
        entry["display_name"] = row.get("display_name", model_id)
        entry["context_length"] = row.get("context_length")
        entry["runs"].append(row.get("run_id", "unknown"))

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

    aggregated = []
    for model_id, data in model_data.items():
        total_runs = data["successes"] + data["failures"]

        latencies = data["latencies"]
        costs = data["costs"]

        tokens_per_sec = []
        for latency, total_toks in zip(latencies, data["total_tokens"]):
            if latency > 0 and total_toks > 0:
                tokens_per_sec.append(total_toks / latency)

        timestamps = sorted(data["timestamps"])
        first_seen = timestamps[0] if timestamps else None
        last_seen = timestamps[-1] if timestamps else None

        agg = {
            "model_id": model_id,
            "canonical_family": data["canonical_family"],
            "display_name": data["display_name"],
            "context_length": data.get("context_length"),
            "total_runs": total_runs,
            "successful_runs": data["successes"],
            "failed_runs": data["failures"],
            "success_rate": round(data["successes"] / total_runs, 4) if total_runs > 0 else 0.0,
            "error_rate": round(data["failures"] / total_runs, 4) if total_runs > 0 else 0.0,
            "latency_sec_avg": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_sec_median": round(statistics.median(latencies), 3) if latencies else None,
            "latency_sec_p95": round(compute_percentile(latencies, 95), 3) if latencies else None,
            "latency_sec_min": round(min(latencies), 3) if latencies else None,
            "latency_sec_max": round(max(latencies), 3) if latencies else None,
            "total_tokens_avg": round(statistics.mean(data["total_tokens"])) if data["total_tokens"] else None,
            "completion_tokens_avg": round(statistics.mean(data["completion_tokens"])) if data["completion_tokens"] else None,
            "tokens_per_sec_avg": round(statistics.mean(tokens_per_sec), 2) if tokens_per_sec else None,
            "cost_avg": round(statistics.mean(costs), 8) if costs else 0.0,
            "cost_total": round(sum(costs), 8) if costs else 0.0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }

        if data["successes"] >= min_runs:
            aggregated.append(agg)

    aggregated.sort(key=lambda x: (-x["success_rate"], x.get("latency_sec_avg") or 9999))
    return aggregated


def generate_model_index(min_runs: int = 3) -> dict:
    LOG.info("Loading benchmark data...")
    rows = read_csv(BENCHMARK_RUNS_FILE)
    LOG.info("Loaded %d benchmark records", len(rows))

    if not rows:
        LOG.warning("No benchmark data found")
        return {"success": False, "error": "No benchmark data found"}

    LOG.info("Aggregating metrics (min runs: %d)...", min_runs)
    aggregated = aggregate_model_metrics(rows, min_runs=min_runs)
    LOG.info("Aggregated metrics for %d models", len(aggregated))

    if not aggregated:
        LOG.warning("No models met minimum run threshold of %d", min_runs)
        return {"success": False, "error": "No models met minimum run threshold"}

    timestamp = now_iso()

    ensure_csv(MODEL_INDEX_FILE, list(aggregated[0].keys()))

    import csv
    with open(MODEL_INDEX_FILE, "w", newline="", encoding="utf-8") as f:
        if aggregated:
            writer = csv.DictWriter(f, fieldnames=list(aggregated[0].keys()))
            writer.writeheader()
            writer.writerows(aggregated)

    LOG.info("Model index saved to: %s", MODEL_INDEX_FILE)

    return {
        "success": True,
        "index_path": str(MODEL_INDEX_FILE),
        "models_aggregated": len(aggregated),
        "total_records_analyzed": len(rows),
        "generated_at": timestamp,
    }


def get_summary_stats() -> dict:
    LOG.info("Computing summary statistics...")
    rows = read_csv(BENCHMARK_RUNS_FILE)
    if not rows:
        LOG.warning("No data available")
        return {"success": False, "error": "No data"}

    model_ids = set(r.get("model_id") for r in rows if r.get("model_id"))
    total_runs = set(r.get("run_id") for r in rows if r.get("run_id"))

    successful = len([r for r in rows if r.get("status") == "success"])
    failed = len([r for r in rows if r.get("status") != "success"])

    latencies = [safe_float(r.get("latency_sec")) for r in rows if r.get("latency_sec")]
    avg_latency = round(statistics.mean(latencies), 3) if latencies else 0

    LOG.info("Summary: %d records, %d models, %d runs",
             len(rows), len(model_ids), len(total_runs))

    return {
        "total_records": len(rows),
        "total_models": len(model_ids),
        "total_runs": len(total_runs),
        "successful": successful,
        "failed": failed,
        "success_rate": round(successful / len(rows), 4) if rows else 0,
        "avg_latency_sec": avg_latency,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--min-runs", type=int, default=3, help="Minimum runs for aggregation")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics")
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        stats = get_summary_stats()
        print(json.dumps(stats, indent=2))
        return

    result = generate_model_index(min_runs=args.min_runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
