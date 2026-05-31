#!/usr/bin/env python3
"""
Analysis Module

Aggregates raw benchmark results and generates derived metrics.
Produces one row per model summarizing aggregated metrics.
"""

import os
import json
import statistics
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from collections import defaultdict


def get_default_dirs() -> tuple[Path, Path, Path]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    catalogs_dir = data_dir / "catalogs"
    runs_dir = data_dir / "runs"
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, catalogs_dir, runs_dir, derived_dir


def load_all_runs(runs_dir: Path, days_limit: Optional[int] = None) -> list[dict]:
    all_results = []
    cutoff_time = None
    if days_limit:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_limit)

    for run_file in sorted(runs_dir.glob("run_*.json"), reverse=True):
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                timestamp_str = data.get("timestamp", "")
                if cutoff_time and timestamp_str:
                    try:
                        run_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if run_time < cutoff_time:
                            continue
                    except ValueError:
                        pass
                all_results.extend(data.get("results", []))
        except (json.JSONDecodeError, IOError):
            continue

    return all_results


def aggregate_model_metrics(results: list[dict], min_runs: int = 3) -> dict:
    model_data = defaultdict(lambda: {
        "runs": [],
        "latencies": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "costs": [],
        "successes": 0,
        "failures": 0,
    })

    for result in results:
        model_id = result.get("model_id", "unknown")
        entry = model_data[model_id]

        entry["model_id"] = model_id
        entry["display_name"] = result.get("display_name", model_id)
        entry["canonical_family"] = result.get("canonical_family", "unknown")
        entry["context_length"] = result.get("context_length")
        entry["runs"].append(result.get("run_id", "unknown"))
        entry["prompt_id"] = result.get("prompt_id", "unknown")

        if result.get("status") == "success":
            entry["successes"] += 1
            entry["latencies"].append(result.get("latency_sec", 0))
            entry["prompt_tokens"].append(result.get("prompt_tokens", 0))
            entry["completion_tokens"].append(result.get("completion_tokens", 0))
            entry["total_tokens"].append(result.get("total_tokens", 0))
            cost = result.get("cost")
            if cost is not None:
                entry["costs"].append(cost)
        else:
            entry["failures"] += 1

    aggregated = []
    for model_id, data in model_data.items():
        total_runs = data["successes"] + data["failures"]
        if data["successes"] < min_runs:
            continue

        latencies = data["latencies"]
        costs = data["costs"]

        agg = {
            "model_id": model_id,
            "display_name": data["display_name"],
            "canonical_family": data["canonical_family"],
            "context_length": data["context_length"],
            "total_runs": total_runs,
            "successful_runs": data["successes"],
            "failed_runs": data["failures"],
            "success_rate": round(data["successes"] / total_runs, 4) if total_runs > 0 else 0,
            "latency_sec_avg": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_sec_median": round(statistics.median(latencies), 3) if latencies else None,
            "latency_sec_min": round(min(latencies), 3) if latencies else None,
            "latency_sec_max": round(max(latencies), 3) if latencies else None,
            "latency_sec_stddev": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
            "prompt_tokens_avg": round(statistics.mean(data["prompt_tokens"])) if data["prompt_tokens"] else None,
            "completion_tokens_avg": round(statistics.mean(data["completion_tokens"])) if data["completion_tokens"] else None,
            "total_tokens_avg": round(statistics.mean(data["total_tokens"])) if data["total_tokens"] else None,
            "cost_avg": round(statistics.mean(costs), 8) if costs else 0,
            "cost_total": round(sum(costs), 8) if costs else 0,
            "last_run": max((r.get("timestamp") for r in results if r.get("model_id") == model_id), default=None),
        }

        aggregated.append(agg)

    aggregated.sort(key=lambda x: (x["success_rate"], x["latency_sec_avg"] or 9999), reverse=True)
    return aggregated


def generate_derived_index(
    runs_dir: Optional[Path] = None,
    derived_dir: Optional[Path] = None,
    window_days: Optional[int] = None,
    min_runs: int = 3,
) -> dict:
    if runs_dir is None:
        _, _, runs_dir, derived_dir = get_default_dirs()

    window = window_days or int(os.getenv("ANALYSIS_WINDOW_DAYS", "30"))
    min_required = min_runs or int(os.getenv("MIN_RUNS_FOR_AGGREGATION", "3"))

    print(f"Analyzing benchmark results (window: {window} days, min runs: {min_required})...")

    all_results = load_all_runs(runs_dir, days_limit=window)
    print(f"Loaded {len(all_results)} benchmark results")

    if not all_results:
        return {"success": False, "error": "No benchmark results found"}

    aggregated = aggregate_model_metrics(all_results, min_runs=min_required)
    print(f"Aggregated metrics for {len(aggregated)} models")

    timestamp = datetime.now(timezone.utc).isoformat()
    derived_record = {
        "index_type": "model_aggregated_metrics",
        "generated_at": timestamp,
        "window_days": window,
        "min_runs_for_aggregation": min_required,
        "total_results_analyzed": len(all_results),
        "models_aggregated": len(aggregated),
        "models": aggregated,
    }

    index_filename = f"index_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    index_path = derived_dir / index_filename

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(derived_record, f, indent=2, ensure_ascii=False)

    latest_link = derived_dir / "latest.json"
    with open(latest_link, "w", encoding="utf-8") as f:
        json.dump(derived_record, f, indent=2, ensure_ascii=False)

    print(f"Derived index saved to: {index_path}")

    return {
        "success": True,
        "index_path": str(index_path),
        "models_aggregated": len(aggregated),
        "total_results_analyzed": len(all_results),
    }


def generate_summary_report(derived_dir: Path) -> dict:
    latest_path = derived_dir / "latest.json"
    if not latest_path.exists():
        return {"success": False, "error": "No derived index found"}

    with open(latest_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    models = index_data.get("models", [])
    if not models:
        return {"success": False, "error": "No models in index"}

    report_lines = []
    report_lines.append("# Free Model Pulse - Benchmark Summary\n")
    report_lines.append(f"Generated: {index_data.get('generated_at', 'unknown')}")
    report_lines.append(f"Window: {index_data.get('window_days', 'unknown')} days")
    report_lines.append(f"Models analyzed: {len(models)}\n")

    report_lines.append("## Top Performing Models (by success rate + latency)\n")
    report_lines.append("| Model | Family | Success Rate | Avg Latency | Avg Tokens | Cost/RUN |")
    report_lines.append("|-------|--------|--------------|-------------|------------|----------|")

    for m in models[:10]:
        latency = f"{m['latency_sec_avg']:.3f}s" if m.get("latency_sec_avg") else "N/A"
        tokens = m.get("total_tokens_avg", "N/A")
        cost = f"${m.get('cost_avg', 0):.6f}" if m.get("cost_avg") else "$0.000000"
        report_lines.append(f"| {m['model_id']} | {m.get('canonical_family', 'N/A')} | "
                            f"{m.get('success_rate', 0)*100:.1f}% | {latency} | {tokens} | {cost} |")

    report_lines.append("\n## Models by Success Rate\n")
    by_success = sorted(models, key=lambda x: x.get("success_rate", 0), reverse=True)
    for m in by_success:
        report_lines.append(f"- {m['model_id']}: {m.get('success_rate', 0)*100:.1f}% success")

    report_text = "\n".join(report_lines)

    report_path = derived_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "success": True,
        "report_path": str(report_path),
        "text": report_text,
    }


def list_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for run_file in sorted(runs_dir.glob("run_*.json"), reverse=True):
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs.append({
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("timestamp"),
                    "total_tests": data.get("total_tests"),
                    "successful": data.get("successful"),
                    "failed": data.get("failed"),
                    "file": str(run_file.name),
                })
        except (json.JSONDecodeError, IOError):
            continue
    return runs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--window", type=int, help="Analysis window in days")
    parser.add_argument("--min-runs", type=int, help="Minimum runs for aggregation")
    parser.add_argument("--report", action="store_true", help="Generate markdown summary report")
    parser.add_argument("--list-runs", action="store_true", help="List all benchmark runs")
    args = parser.parse_args()

    _, _, runs_dir, derived_dir = get_default_dirs()

    if args.list_runs:
        runs = list_runs(runs_dir)
        print(json.dumps(runs, indent=2))
        return

    if args.report:
        result = generate_summary_report(derived_dir)
        print(json.dumps(result, indent=2))
        return

    result = generate_derived_index(
        runs_dir=runs_dir,
        derived_dir=derived_dir,
        window_days=args.window,
        min_runs=args.min_runs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
