#!/usr/bin/env python3
"""
Benchmarking Module

Benchmarks free models using stable prompts from prompts.json.
Stores raw results in data/runs/ with one record per model per run.
"""

import os
import json
import time
import uuid
import hashlib
import requests
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = int(os.getenv("BENCHMARK_TIMEOUT_SEC", "120"))
MAX_RETRIES = int(os.getenv("BENCHMARK_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("BENCHMARK_RETRY_DELAY_SEC", "5"))


def get_default_dirs() -> tuple[Path, Path, Path]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    catalogs_dir = data_dir / "catalogs"
    runs_dir = data_dir / "runs"
    derived_dir = data_dir / "derived"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, catalogs_dir, runs_dir


def load_prompts() -> dict:
    prompt_file = os.getenv("BENCHMARK_PROMPT_FILE", "prompts.json")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_api_headers() -> dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_email = os.getenv("OPENROUTER_SITE_EMAIL")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_email:
        headers["X-Title"] = "Free Model Pulse"
    return headers


def load_latest_catalog(catalogs_dir: Path) -> Optional[dict]:
    latest_path = catalogs_dir / "latest.json"
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_run_history(runs_dir: Path, model_id: str, limit: int = 10) -> list[dict]:
    history = []
    for run_file in sorted(runs_dir.glob("run_*.json"), reverse=True):
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for result in data.get("results", []):
                    if result.get("model_id") == model_id:
                        history.append(result)
                        if len(history) >= limit:
                            return history
        except (json.JSONDecodeError, IOError):
            continue
    return history


def benchmark_single_model(
    model_id: str,
    prompt_text: str,
    prompt_id: str,
    timeout: int = REQUEST_TIMEOUT
) -> dict:
    headers = get_api_headers()

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 512,
    }

    start_time = time.time()

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", RETRY_DELAY * (attempt + 1)))
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"HTTP {response.status_code}: {response.text[:200]}",
                    "latency_sec": elapsed_time,
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                }

            data = response.json()
            usage = data.get("usage", {})
            choice = data.get("choices", [{}])[0]

            finish_reason = choice.get("finish_reason", "unknown")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("reason", "unknown")

            return {
                "status": "success",
                "latency_sec": round(elapsed_time, 3),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cached_tokens": usage.get("cached_tokens"),
                "cache_write_tokens": usage.get("cache_write_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "cost": data.get("cost"),
                "finish_reason": finish_reason,
                "response_id": data.get("id"),
                "model_id": model_id,
                "prompt_id": prompt_id,
                "raw_response": data,
            }

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                print(f"  Timeout. Retrying...")
                time.sleep(RETRY_DELAY)
                continue
            return {
                "status": "timeout",
                "error_message": f"Request timed out after {timeout}s",
                "latency_sec": time.time() - start_time,
                "model_id": model_id,
                "prompt_id": prompt_id,
            }

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Request failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
                continue
            return {
                "status": "error",
                "error_message": str(e)[:200],
                "latency_sec": time.time() - start_time,
                "model_id": model_id,
                "prompt_id": prompt_id,
            }

    return {
        "status": "error",
        "error_message": "Max retries exceeded",
        "latency_sec": time.time() - start_time,
        "model_id": model_id,
        "prompt_id": prompt_id,
    }


def run_benchmark(
    models: Optional[list[dict]] = None,
    prompt_ids: Optional[list[str]] = None,
    run_id: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    catalogs_dir: Optional[Path] = None,
) -> dict:
    if runs_dir is None:
        _, catalogs_dir, runs_dir = get_default_dirs()
    else:
        if catalogs_dir is None:
            catalogs_dir = runs_dir.parent / "catalogs"

    prompts_data = load_prompts()
    prompts_list = prompts_data.get("prompts", [])
    if prompt_ids:
        selected_prompts = [p for p in prompts_list if p["id"] in prompt_ids]
    else:
        default_id = prompts_data.get("default_prompt_id", "reasoning_simple")
        selected_prompts = [p for p in prompts_list if p["id"] == default_id]
        if not selected_prompts:
            selected_prompts = [prompts_list[0]] if prompts_list else []

    if not models:
        catalog = load_latest_catalog(catalogs_dir)
        if not catalog:
            return {"success": False, "error": "No catalog available. Run watch_models.py first."}
        models = catalog.get("models", [])

    if not run_id:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    timestamp = datetime.now(timezone.utc).isoformat()
    catalog_id = None
    latest_catalog = load_latest_catalog(catalogs_dir)
    if latest_catalog:
        catalog_id = latest_catalog.get("catalog_id")

    results = []
    total_tests = len(models) * len(selected_prompts)
    completed = 0

    print(f"\nStarting benchmark run: {run_id}")
    print(f"Models to test: {len(models)}")
    print(f"Prompts per model: {len(selected_prompts)}")
    print(f"Total benchmark tests: {total_tests}\n")

    for model in models:
        model_id = model.get("model_id")
        if not model_id:
            continue

        if model.get("disabled") or model.get("hidden"):
            print(f"Skipping disabled/hidden model: {model_id}")
            continue

        for prompt in selected_prompts:
            prompt_id = prompt["id"]
            prompt_text = prompt["text"]

            print(f"[{completed + 1}/{total_tests}] Testing {model_id} with {prompt_id}...")

            result = benchmark_single_model(model_id, prompt_text, prompt_id)
            result["canonical_family"] = model.get("canonical_family", "unknown")
            result["display_name"] = model.get("display_name", model_id)
            result["context_length"] = model.get("context_length")
            result["run_id"] = run_id
            result["timestamp"] = timestamp

            results.append(result)
            completed += 1

            if result["status"] == "success":
                print(f"  OK - Latency: {result['latency_sec']:.3f}s, "
                      f"Tokens: {result['total_tokens']}, "
                      f"Cost: ${result.get('cost', 0):.6f}")
            else:
                print(f"  FAILED - {result.get('error_message', 'Unknown error')}")

            time.sleep(1)

    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "catalog_id": catalog_id,
        "total_models": len(models),
        "total_tests": total_tests,
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] != "success"]),
        "prompts_used": [p["id"] for p in selected_prompts],
        "results": results,
    }

    run_file = runs_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{run_id}.json"
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark run complete: {run_id}")
    print(f"Results saved to: {run_file}")
    print(f"Successful: {run_record['successful']}/{total_tests}")

    return {
        "success": True,
        "run_id": run_id,
        "run_file": str(run_file),
        "total_tests": total_tests,
        "successful": run_record["successful"],
        "failed": run_record["failed"],
    }


def benchmark_new_models(runs_dir: Optional[Path] = None, catalogs_dir: Optional[Path] = None) -> dict:
    if runs_dir is None:
        _, catalogs_dir, runs_dir = get_default_dirs()

    latest_catalog = load_latest_catalog(catalogs_dir)
    if not latest_catalog:
        return {"success": False, "error": "No catalog available"}

    new_models = latest_catalog.get("models", [])

    return run_benchmark(models=new_models, runs_dir=runs_dir, catalogs_dir=catalogs_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark free models from OpenRouter")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to benchmark")
    parser.add_argument("--prompts", nargs="+", help="Specific prompt IDs to use")
    parser.add_argument("--run-id", help="Custom run ID")
    parser.add_argument("--new-only", action="store_true", help="Only benchmark newly discovered models")
    args = parser.parse_args()

    models = None
    if args.models:
        models = [{"model_id": m, "display_name": m, "canonical_family": "manual"} for m in args.models]

    if args.new_only:
        result = benchmark_new_models()
    else:
        result = run_benchmark(models=models, prompt_ids=args.prompts, run_id=args.run_id)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
