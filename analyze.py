#!/usr/bin/env python3
"""
Analysis Module.

Aggregates raw benchmark results into a summary index with all required metrics:
  success_rate, error_rate, avg/median/p95 latency, prompt/completion/total tokens avg,
  tokens_per_sec, avg_cost, availability, last_seen, total_runs.

Leaderboard and Pages show ONLY models present in the current free catalog.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import (
    BENCHMARK_CSV_COLUMNS,
    BENCHMARK_RUNS_FILE,
    INDEX_COLUMNS,
    INFRA_STATUSES,
    MODEL_FAILURE_STATUSES,
    MODEL_INDEX_FILE,
    get_logger,
    html_escape,
    now_iso,
    read_csv,
    safe_float,
    safe_int,
    sanitize_error_message,
    setup_logging,
)

LOG = get_logger("analyze")

MODEL_INDEX_FIELDS = list(INDEX_COLUMNS)


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = min(int(len(sorted_vals) * percentile / 100), len(sorted_vals) - 1)
    return sorted_vals[index]


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def filter_rows_by_window(rows: list[dict], window_days: int | None) -> list[dict]:
    if not window_days or window_days <= 0:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    kept = []
    for row in rows:
        ts = _parse_timestamp(row.get("timestamp_utc", ""))
        if ts is None or ts >= cutoff:
            kept.append(row)
    return kept


def aggregate_model_metrics(
    rows: list[dict],
    min_runs: int = 1,
) -> list[dict]:
    model_data: dict = defaultdict(lambda: {
        "successes": 0,
        "failures": 0,
        "infra": 0,
        "latencies": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "total_tokens": [],
        "tokens_per_sec": [],
        "costs": [],
        "timestamps": [],
        "run_ids_seen": set(),
    })

    all_run_ids: set = set()

    for row in rows:
        model_id = (row.get("model_id") or "").strip()
        if not model_id:
            continue

        run_id = (row.get("run_id") or "").strip()
        if run_id:
            all_run_ids.add(run_id)

        entry = model_data[model_id]
        entry["model_id"] = model_id
        entry["canonical_family"] = row.get("canonical_family", "unknown")
        entry["display_name"] = row.get("display_name", model_id)
        entry["context_length"] = row.get("context_length")

        if run_id:
            entry["run_ids_seen"].add(run_id)

        status = (row.get("status") or "").strip()
        timestamp = row.get("timestamp_utc", "")

        if status in INFRA_STATUSES:
            entry["infra"] += 1
            if timestamp:
                entry["timestamps"].append(timestamp)
            continue

        if status == "success":
            entry["successes"] += 1

            latency = safe_float(row.get("latency_sec"))
            prompt_toks = safe_int(row.get("prompt_tokens"))
            completion_toks = safe_int(row.get("completion_tokens"))
            total_toks = safe_int(row.get("total_tokens"))

            if latency > 0:
                entry["latencies"].append(latency)
                if total_toks > 0:
                    entry["tokens_per_sec"].append(total_toks / latency)

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
        elif status in MODEL_FAILURE_STATUSES or status:
            # Unknown statuses count as model failures so they stay visible.
            entry["failures"] += 1
            if timestamp:
                entry["timestamps"].append(timestamp)

    total_unique_runs = len(all_run_ids) if all_run_ids else 1

    aggregated = []
    for model_id, data in model_data.items():
        scored_runs = data["successes"] + data["failures"]
        total_runs = scored_runs + data["infra"]
        if total_runs < min_runs:
            continue

        latencies = data["latencies"]
        costs = data["costs"]

        timestamps = sorted(data["timestamps"])
        first_seen = timestamps[0] if timestamps else None
        last_seen = timestamps[-1] if timestamps else None

        runs_seen = len(data["run_ids_seen"])
        availability = round(runs_seen / total_unique_runs, 4) if total_unique_runs > 0 else None

        if scored_runs > 0:
            success_rate = round(data["successes"] / scored_runs, 4)
            error_rate = round(data["failures"] / scored_runs, 4)
        else:
            success_rate = None
            error_rate = None

        agg = {
            "model_id": model_id,
            "display_name": data["display_name"],
            "canonical_family": data["canonical_family"],
            "context_length": data.get("context_length"),
            "total_runs": total_runs,
            "successful_runs": data["successes"],
            "failed_runs": data["failures"],
            "infra_runs": data["infra"],
            "success_rate": success_rate,
            "error_rate": error_rate,
            "availability": availability,
            "latency_sec_avg": round(statistics.mean(latencies), 3) if latencies else None,
            "latency_sec_median": round(statistics.median(latencies), 3) if latencies else None,
            "latency_sec_p95": round(compute_percentile(latencies, 95), 3) if latencies else None,
            "latency_sec_min": round(min(latencies), 3) if latencies else None,
            "latency_sec_max": round(max(latencies), 3) if latencies else None,
            "prompt_tokens_avg": round(statistics.mean(data["prompt_tokens"])) if data["prompt_tokens"] else None,
            "completion_tokens_avg": round(statistics.mean(data["completion_tokens"])) if data["completion_tokens"] else None,
            "total_tokens_avg": round(statistics.mean(data["total_tokens"])) if data["total_tokens"] else None,
            "tokens_per_sec_avg": round(statistics.mean(data["tokens_per_sec"]), 2) if data["tokens_per_sec"] else None,
            "cost_avg": round(statistics.mean(costs), 8) if costs else 0.0,
            "cost_total": round(sum(costs), 8) if costs else 0.0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }
        aggregated.append(agg)

    aggregated.sort(key=_sort_key)
    return aggregated


def _sort_key(row: dict) -> tuple:
    try:
        sr = row.get("success_rate")
        sr_val = float(sr) if sr is not None and sr != "" else -1.0
    except (TypeError, ValueError):
        sr_val = -1.0
    try:
        lat = row.get("latency_sec_median")
        lat_val = float(lat) if lat is not None and lat != "" else 9999.0
    except (TypeError, ValueError):
        lat_val = 9999.0
    try:
        runs = int(float(row.get("successful_runs") or 0))
    except (TypeError, ValueError):
        runs = 0
    return (-sr_val, lat_val, -runs)


def generate_model_index(min_runs: int = 1, window_days: int | None = None) -> dict:
    LOG.info("Loading benchmark data...")
    rows = read_csv(BENCHMARK_RUNS_FILE)
    LOG.info("Loaded %d benchmark records", len(rows))

    if not rows:
        LOG.warning("No benchmark data found")
        return {"success": False, "error": "No benchmark data found"}

    if window_days:
        before = len(rows)
        rows = filter_rows_by_window(rows, window_days)
        LOG.info("Window filter %d days: %d -> %d rows", window_days, before, len(rows))

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
        "window_days": window_days,
        "generated_at": now_iso(),
    }


def get_summary_stats() -> dict:
    rows = read_csv(BENCHMARK_RUNS_FILE)
    if not rows:
        return {"success": False, "error": "No data"}

    model_ids = {r.get("model_id") for r in rows if r.get("model_id")}
    run_ids = {r.get("run_id") for r in rows if r.get("run_id")}
    successful = [r for r in rows if r.get("status") == "success"]
    infra = [r for r in rows if r.get("status") in INFRA_STATUSES]
    latencies = [safe_float(r.get("latency_sec")) for r in successful if r.get("latency_sec")]

    return {
        "total_records": len(rows),
        "total_models": len(model_ids),
        "total_runs": len(run_ids),
        "successful": len(successful),
        "failed": len(rows) - len(successful) - len(infra),
        "infra": len(infra),
        "success_rate": round(len(successful) / len(rows), 4) if rows else 0,
        "avg_latency_sec": round(statistics.mean(latencies), 3) if latencies else 0,
    }


def _catalog_model_is_live(model: dict) -> bool:
    """Re-apply discovery filters to persisted catalog rows (defense in depth)."""
    from watch_models import is_benchmarkable

    probe = {
        "id": model.get("model_id") or model.get("id", ""),
        "name": model.get("display_name") or model.get("name", ""),
        "pricing": model.get("pricing") or {"prompt": "0", "completion": "0"},
        "architecture": model.get("architecture") or {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
        "supported_parameters": model.get("supported_parameters"),
        "disabled": model.get("disabled", False),
        "hidden": model.get("hidden", False),
    }
    ok, _ = is_benchmarkable(probe)
    return ok


def _load_catalog() -> dict:
    from common import CURRENT_MODELS_FILE

    catalog = {}
    if not CURRENT_MODELS_FILE.exists():
        return catalog
    try:
        raw = json.loads(CURRENT_MODELS_FILE.read_text(encoding="utf-8"))
        for m in raw.get("models", []):
            mid = m.get("model_id", "")
            if mid and _catalog_model_is_live(m):
                catalog[mid] = m
    except Exception as e:
        LOG.warning("Catalog parse error: %s", e)
    return catalog


def _sort_leaderboard_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_sort_key)


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
    if raw is None or raw == "":
        return "?"
    try:
        return f"{float(raw):.2f}s"
    except (TypeError, ValueError):
        return "?"


def _fmt_range(min_raw, max_raw) -> str:
    if min_raw in (None, "") or max_raw in (None, ""):
        return "?"
    try:
        return f"{float(min_raw):.1f}s to {float(max_raw):.1f}s"
    except (TypeError, ValueError):
        return "?"


def _fmt_tps(raw) -> str:
    if raw is None or raw == "":
        return "?"
    try:
        return f"{float(raw):.0f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_success(raw) -> str:
    if raw is None or raw == "":
        return "?"
    try:
        return f"{float(raw) * 100:.0f}%"
    except (TypeError, ValueError):
        return "?"


def _fmt_runs(r: dict) -> str:
    try:
        return str(int(float(r.get("successful_runs") or 0)))
    except (TypeError, ValueError):
        return "0"


def _fmt_released(cat: dict) -> str:
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
        sr = r.get("success_rate")
        if sr is None or sr == "":
            return "⚪"
        sr_f = float(sr)
        if sr_f >= 0.8:
            return "🟢"
        if sr_f >= 0.5:
            return "🟡"
        return "🔴"
    except (TypeError, ValueError):
        return "⚪"


def _updated_at_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y %b %d %H:%M UTC")


def _leaderboard_entries(limit: int = 30, min_runs: int = 1) -> list[dict]:
    if not MODEL_INDEX_FILE.exists():
        return []
    rows = read_csv(MODEL_INDEX_FILE)
    if not rows:
        return []

    catalog = _load_catalog()
    if not catalog:
        LOG.warning("No current catalog; refusing to publish a historical leaderboard")
        return []

    live_rows = []
    for r in rows:
        model_id = r.get("model_id", "")
        if model_id not in catalog:
            continue
        try:
            total = int(float(r.get("total_runs") or 0))
        except (TypeError, ValueError):
            total = 0
        if total < min_runs:
            continue
        live_rows.append(r)

    medals = ["🥇", "🥈", "🥉"]
    entries = []
    for i, r in enumerate(_sort_leaderboard_rows(live_rows)[:limit]):
        model_id = r.get("model_id", "?")
        cat = catalog.get(model_id, {})
        entries.append({
            "rank": medals[i] if i < 3 else str(i + 1),
            "name": r.get("display_name") or model_id,
            "context": _fmt_context(r, cat),
            "latency": _fmt_latency(r.get("latency_sec_median")),
            "range": _fmt_range(r.get("latency_sec_min"), r.get("latency_sec_max")),
            "tps": _fmt_tps(r.get("tokens_per_sec_avg")),
            "success": _fmt_success(r.get("success_rate")),
            "runs": _fmt_runs(r),
            "released": _fmt_released(cat),
            "status": _fmt_status(r),
        })
    return entries


def build_readme_leaderboard_block(min_runs: int = 1) -> str | None:
    entries = _leaderboard_entries(min_runs=min_runs)
    if not entries:
        return None

    lines = [
        "| # | Model | Context | Latency | Range | Tokens/s | Success | Runs | Released | Status |",
        "|---|-------|---------|---------|-------|----------|---------|------|----------|--------|",
    ]
    for e in entries:
        lines.append(
            f"| {e['rank']} | `{e['name']}` | {e['context']} | {e['latency']} | "
            f"{e['range']} | {e['tps']} | {e['success']} | {e['runs']} | "
            f"{e['released']} | {e['status']} |"
        )

    updated_at = _updated_at_label()
    return "\n".join([
        "<!-- LEADERBOARD_START -->",
        f"> 🕒 Last updated: **{updated_at}** &nbsp;|&nbsp; Current free catalog only &nbsp;|&nbsp; Auto generated by GitHub Actions",
        "",
        *lines,
        "",
        "<!-- LEADERBOARD_END -->",
    ])


def update_readme_leaderboard(readme_path: Path | None = None, min_runs: int = 1) -> dict:
    readme_path = readme_path or Path("README.md")
    block = build_readme_leaderboard_block(min_runs=min_runs)
    if block is None:
        LOG.warning("No live leaderboard rows; skipping README update")
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


def write_pages_leaderboard(docs_dir: Path | None = None, min_runs: int = 1) -> dict:
    docs_dir = docs_dir or Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    entries = _leaderboard_entries(min_runs=min_runs)
    updated_at = _updated_at_label()

    rows_html = []
    for e in entries:
        rows_html.append(
            "<tr>"
            f"<td>{html_escape(e['rank'])}</td>"
            f"<td><code>{html_escape(e['name'])}</code></td>"
            f"<td>{html_escape(e['context'])}</td>"
            f"<td>{html_escape(e['latency'])}</td>"
            f"<td>{html_escape(e['range'])}</td>"
            f"<td>{html_escape(e['tps'])}</td>"
            f"<td>{html_escape(e['success'])}</td>"
            f"<td>{html_escape(e['runs'])}</td>"
            f"<td>{html_escape(e['released'])}</td>"
            f"<td>{html_escape(e['status'])}</td>"
            "</tr>"
        )

    body_rows = "\n".join(rows_html) if rows_html else (
        "<tr><td colspan='10'>No live free-model benchmark data yet.</td></tr>"
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
      min-width: 860px;
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
    <h1>Free Model Pulse</h1>
    <p class="tagline">Live latency leaderboard for models currently free on OpenRouter. Historical models are excluded.</p>
    <p class="meta">Last updated: {html_escape(updated_at)}</p>
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
            <th>Success</th>
            <th>Runs</th>
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


def _reclassify_historical_status(status: str, error_message: str) -> str:
    """Map old error rows that were really quota/auth into infra statuses."""
    if status in INFRA_STATUSES or status == "success":
        return status
    text = (error_message or "").lower()
    if "429" in text:
        return status or "error"
    if any(token in text for token in (
        "key limit exceeded",
        "limit exceeded",
        "insufficient credits",
        "quota",
        "credit",
        "usage limit",
        "[redacted-key-url]",
    )):
        return "quota"
    if "http 401" in text or "http 403" in text:
        return "auth"
    return status


def redact_raw_error_messages(path: Path | None = None) -> dict:
    """Sanitize secrets in error_message and reclassify infra failures. Metrics columns untouched."""
    path = path or BENCHMARK_RUNS_FILE
    if not path.exists():
        return {"success": False, "error": "raw file missing"}

    rows = read_csv(path)
    changed = 0
    reclassified = 0
    for row in rows:
        original = row.get("error_message", "") or ""
        cleaned = sanitize_error_message(original, max_len=max(200, len(original)))
        if cleaned != original:
            row["error_message"] = cleaned
            changed += 1

        old_status = row.get("status", "") or ""
        new_status = _reclassify_historical_status(old_status, cleaned or original)
        if new_status != old_status:
            row["status"] = new_status
            reclassified += 1

    fieldnames = list(BENCHMARK_CSV_COLUMNS)
    # Preserve unknown extra columns if present historically.
    if rows:
        for key in rows[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    LOG.info(
        "Redacted %d error_message cells and reclassified %d statuses in %s",
        changed, reclassified, path,
    )
    return {
        "success": True,
        "redacted": changed,
        "reclassified": reclassified,
        "rows": len(rows),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--min-runs", type=int, default=1, help="Minimum runs for aggregation (default: 1)")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Only aggregate rows from the last N days (default: all history)",
    )
    parser.add_argument("--stats", action="store_true", help="Show summary statistics only")
    parser.add_argument("--update-readme", action="store_true", help="Refresh README leaderboard markers")
    parser.add_argument("--write-pages", action="store_true", help="Write docs/index.html for GitHub Pages")
    parser.add_argument("--skip-index", action="store_true", help="Skip regenerating model_index.csv")
    parser.add_argument(
        "--redact-raw-errors",
        action="store_true",
        help="Sanitize secrets in raw error_message column",
    )
    parser.add_argument(
        "--leaderboard-min-runs",
        type=int,
        default=1,
        help="Minimum total runs for a live model to appear on README/Pages",
    )
    args = parser.parse_args()

    setup_logging()

    if args.stats:
        print(json.dumps(get_summary_stats(), indent=2))
        return

    result = {"index": None, "readme": None, "pages": None, "redact": None}
    if args.redact_raw_errors:
        result["redact"] = redact_raw_error_messages()

    if not args.skip_index:
        result["index"] = generate_model_index(min_runs=args.min_runs, window_days=args.window_days)

    if args.update_readme:
        result["readme"] = update_readme_leaderboard(min_runs=args.leaderboard_min_runs)

    if args.write_pages:
        result["pages"] = write_pages_leaderboard(min_runs=args.leaderboard_min_runs)

    if args.update_readme or args.write_pages or args.redact_raw_errors:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result["index"], indent=2))


if __name__ == "__main__":
    main()
