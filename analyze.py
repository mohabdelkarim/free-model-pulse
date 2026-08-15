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
    get_logger,
    now_iso,
    read_csv,
    safe_float,
    safe_int,
    setup_logging,
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


def _load_catalog() -> dict:
    from common import CURRENT_MODELS_FILE

    catalog = {}
    if not CURRENT_MODELS_FILE.exists():
        return catalog
    try:
        raw = json.loads(CURRENT_MODELS_FILE.read_text(encoding="utf-8"))
        for m in raw.get("models", []):
            mid = m.get("model_id", "")
            if mid:
                catalog[mid] = m
    except Exception as e:
        LOG.warning("Catalog parse error: %s", e)
    return catalog


def _sort_leaderboard_rows(rows: list[dict]) -> list[dict]:
    def sort_key(r):
        try:
            sr = float(r.get("success_rate") or 0)
        except (TypeError, ValueError):
            sr = 0.0
        try:
            lat = float(r.get("latency_sec_median") or 999)
        except (TypeError, ValueError):
            lat = 999.0
        return (-sr, lat)

    return sorted(rows, key=sort_key)


def _fmt_context(r: dict, cat: dict) -> str:
    ctx_raw = r.get("context_length") or cat.get("context_length")
    if not ctx_raw:
        return "?"
    try:
        ctx_val = int(float(ctx_raw))
        return f"{ctx_val // 1000}k" if ctx_val >= 1000 else str(ctx_val)
    except (TypeError, ValueError):
        return "?"


def _fmt_latency(raw) -> str:
    if not raw:
        return "?"
    try:
        return f"{float(raw):.2f}s"
    except (TypeError, ValueError):
        return "?"


def _fmt_range(min_raw, max_raw) -> str:
    if not min_raw or not max_raw:
        return "?"
    try:
        return f"{float(min_raw):.1f}s to {float(max_raw):.1f}s"
    except (TypeError, ValueError):
        return "?"


def _fmt_tps(raw) -> str:
    if not raw:
        return "?"
    try:
        return f"{float(raw):.0f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_released(cat: dict) -> str:
    from datetime import datetime, timezone

    created_raw = cat.get("created")
    if not created_raw:
        return "?"
    try:
        dt = datetime.fromtimestamp(int(created_raw), tz=timezone.utc)
        return dt.strftime("%b %Y")
    except (TypeError, ValueError, OSError):
        return "?"


def _fmt_status(r: dict) -> str:
    try:
        sr = float(r.get("success_rate") or 0)
        return "🟢" if sr >= 0.5 else "🔴"
    except (TypeError, ValueError):
        return "⚪"


def _updated_at_label() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y %b %d %H:%M UTC")


def _leaderboard_entries(limit: int = 30) -> list[dict]:
    if not MODEL_INDEX_FILE.exists():
        return []
    rows = read_csv(MODEL_INDEX_FILE)
    if not rows:
        return []
    catalog = _load_catalog()
    medals = ["🥇", "🥈", "🥉"]
    entries = []
    for i, r in enumerate(_sort_leaderboard_rows(rows)[:limit]):
        model_id = r.get("model_id", "?")
        cat = catalog.get(model_id, {})
        entries.append({
            "rank": medals[i] if i < 3 else str(i + 1),
            "name": r.get("display_name") or model_id,
            "context": _fmt_context(r, cat),
            "latency": _fmt_latency(r.get("latency_sec_median")),
            "range": _fmt_range(r.get("latency_sec_min"), r.get("latency_sec_max")),
            "tps": _fmt_tps(r.get("tokens_per_sec_avg")),
            "released": _fmt_released(cat),
            "status": _fmt_status(r),
        })
    return entries


def build_readme_leaderboard_block() -> str | None:
    entries = _leaderboard_entries()
    if not entries:
        return None

    lines = [
        "| # | Model | Context | Latency | Range | Tokens/s | Released | Status |",
        "|---|-------|---------|---------|-------|----------|----------|--------|",
    ]
    for e in entries:
        lines.append(
            f"| {e['rank']} | `{e['name']}` | {e['context']} | {e['latency']} | "
            f"{e['range']} | {e['tps']} | {e['released']} | {e['status']} |"
        )

    updated_at = _updated_at_label()
    return "\n".join([
        "<!-- LEADERBOARD_START -->",
        f"> 🕒 Last updated: **{updated_at}** &nbsp;|&nbsp; Auto generated by GitHub Actions",
        "",
        *lines,
        "",
        "<!-- LEADERBOARD_END -->",
    ])


def update_readme_leaderboard(readme_path: Path | None = None) -> dict:
    import re

    readme_path = readme_path or Path("README.md")
    block = build_readme_leaderboard_block()
    if block is None:
        LOG.warning("No model index rows; skipping README update")
        return {"success": False, "error": "No leaderboard rows"}

    content = readme_path.read_text(encoding="utf-8")
    pattern = r"<!-- LEADERBOARD_START -->.*?<!-- LEADERBOARD_END -->"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, block, content, flags=re.DOTALL)
    else:
        new_content = content + "\n\n## 📊 Live Leaderboard\n\n" + block + "\n"

    readme_path.write_text(new_content, encoding="utf-8")
    LOG.info("README leaderboard updated")
    return {"success": True, "path": str(readme_path)}


def write_pages_leaderboard(docs_dir: Path | None = None) -> dict:
    docs_dir = docs_dir or Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    entries = _leaderboard_entries()
    updated_at = _updated_at_label()

    rows_html = []
    for e in entries:
        rows_html.append(
            "<tr>"
            f"<td>{e['rank']}</td>"
            f"<td><code>{e['name']}</code></td>"
            f"<td>{e['context']}</td>"
            f"<td>{e['latency']}</td>"
            f"<td>{e['range']}</td>"
            f"<td>{e['tps']}</td>"
            f"<td>{e['released']}</td>"
            f"<td>{e['status']}</td>"
            "</tr>"
        )

    body_rows = "\n".join(rows_html) if rows_html else (
        "<tr><td colspan='8'>No benchmark data yet.</td></tr>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Free Model Pulse</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a222c;
      --text: #e7ecf1;
      --muted: #9aa7b5;
      --accent: #3dd6c6;
      --line: #2a3542;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, #1d3a3a 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #243049 0%, transparent 50%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 48px 20px 72px;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3rem);
      margin: 0 0 8px;
      letter-spacing: 0.02em;
    }}
    .tagline {{
      color: var(--muted);
      font-size: 1.05rem;
      margin: 0 0 28px;
      max-width: 42rem;
    }}
    .meta {{
      color: var(--accent);
      font-size: 0.95rem;
      margin-bottom: 18px;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 92%, black);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    th {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: #141b23;
      position: sticky;
      top: 0;
    }}
    tr:hover td {{ background: #202935; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92rem;
    }}
    footer {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <h1>🫀 Free Model Pulse</h1>
    <p class="tagline">Live latency leaderboard for free OpenRouter LLMs. Updated automatically from GitHub Actions.</p>
    <p class="meta">Last updated: {updated_at}</p>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Model</th>
            <th>Context</th>
            <th>Latency</th>
            <th>Range</th>
            <th>Tokens/s</th>
            <th>Released</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
{body_rows}
        </tbody>
      </table>
    </div>
    <footer>
      Source repo:
      <a href="https://github.com/mohabdelkarim/free-model-pulse">github.com/mohabdelkarim/free-model-pulse</a>
      · Not affiliated with OpenRouter.
    </footer>
  </main>
</body>
</html>
"""
    out = docs_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    LOG.info("Pages leaderboard written: %s", out)
    return {"success": True, "path": str(out), "models": len(entries)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--min-runs", type=int, default=1, help="Minimum runs for aggregation (default: 1)")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics only")
    parser.add_argument("--update-readme", action="store_true", help="Refresh README leaderboard markers")
    parser.add_argument("--write-pages", action="store_true", help="Write docs/index.html for GitHub Pages")
    parser.add_argument("--skip-index", action="store_true", help="Skip regenerating model_index.csv")
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        print(json.dumps(get_summary_stats(), indent=2))
        return

    result = {"index": None, "readme": None, "pages": None}
    if not args.skip_index:
        result["index"] = generate_model_index(min_runs=args.min_runs)

    if args.update_readme:
        result["readme"] = update_readme_leaderboard()

    if args.write_pages:
        result["pages"] = write_pages_leaderboard()

    print(json.dumps(result if (args.update_readme or args.write_pages) else result["index"], indent=2))


if __name__ == "__main__":
    main()
