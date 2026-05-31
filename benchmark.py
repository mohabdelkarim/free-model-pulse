#!/usr/bin/env python3
"""
Benchmarking Module.

Benchmarks free models using stable prompts from prompts.json.
Records results to append-only CSV raw datasets.
"""

import os
import json
import time
import requests
from datetime import datetime, timezone
from typing import Optional

from common import (
    CURRENT_MODELS_FILE,
    BENCHMARK_RUNS_FILE,
    load_json,
    load_prompts,
    get_run_id,
    now_iso,
    append_csv_row,
    safe_float,
    safe_int,
)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = int(os.getenv("BENCHMARK_TIMEOUT_SEC", "120"))
MAX_RETRIES = int(os.getenv("BENCHMARK_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("BENCHMARK_RETRY_DELAY_SEC", "5"))


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


def load_current_catalog() -> Optional[dict]:
    return load_json(CURRENT_MODELS_FILE)


def benchmark_single_model(
    model_id: str,
    prompt_text: str,
    prompt_version: str,
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
                    "latency_sec": round(elapsed_time, 3),
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
                "prompt_tokens": safe_int(usage.get("prompt_tokens")),
                "completion_tokens": safe_int(usage.get("completion_tokens")),
                "total_tokens": safe_int(usage.get("total_tokens")),
                "cached_tokens": usage.get("cached_tokens"),
                "cache_write_tokens": usage.get("cache_write_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "cost": safe_float(data.get("cost")),
                "finish_reason": finish_reason,
                "response_id": data.get("id"),
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
            }

    return {
        "status": "error",
        "error_message": "Max retries exceeded",
        "latency_sec": time.time() - start_time,
    }


def run_benchmark(
    models: Optional[list[dict]] = None,
    prompt_id: Optional[str] = None,
    benchmark_reason: str = "manual",
    run_id: Optional[str] = None,
) -> dict:
    prompts_data = load_prompts()
    prompts_list = prompts_data.get("prompts", [])

    if prompt_id:
        selected_prompts = [p for p in prompts_list if p["id"] == prompt_id]
    else:
        default_id = prompts_data.get("default_prompt_id", "reasoning_simple")
        selected_prompts = [p for p in prompts_list if p["id"] == default_id]
        if not selected_prompts and prompts_list:
            selected_prompts = [prompts_list[0]]

    if not selected_prompts:
        return {"success": False, "error": "No prompts found in prompts.json"}

    prompt = selected_prompts[0]
    prompt_id = prompt["id"]
    prompt_text = prompt["text"]
    prompt_version = prompts_data.get("version", "1.0.0")

    if not models:
        catalog = load_current_catalog()
        if not catalog:
            return {"success": False, "error": "No catalog available. Run watch_models.py first."}
        models = catalog.get("models", [])

    if not run_id:
        run_id = get_run_id()

    timestamp = now_iso()
    catalog_id = None
    catalog = load_current_catalog()
    if catalog:
        catalog_id = catalog.get("catalog_id")

    total_tests = len(models)
    completed = 0
    successful = 0
    failed = 0

    print(f"\nBenchmark run: {run_id}")
    print(f"Reason: {benchmark_reason}")
    print(f"Prompt: {prompt_id} (v{prompt_version})")
    print(f"Models to test: {total_tests}\n")

    for model in models:
        model_id = model.get("model_id")
        if not model_id:
            continue

        print(f"[{completed + 1}/{total_tests}] Testing {model_id}...")

        result = benchmark_single_model(model_id, prompt_text, prompt_version)

        row = {
            "run_id": run_id,
            "benchmark_reason": benchmark_reason,
            "timestamp_utc": timestamp,
            "prompt_version": prompt_version,
            "model_id": model_id,
            "canonical_family": model.get("canonical_family", "unknown"),
            "display_name": model.get("display_name", model_id),
            "context_length": model.get("context_length"),
            "latency_sec": result.get("latency_sec"),
            "status": result.get("status"),
            "error_message": result.get("error_message", ""),
            "response_id": result.get("response_id", ""),
            "finish_reason": result.get("finish_reason", ""),
            "prompt_tokens": result.get("prompt_tokens", ""),
            "completion_tokens": result.get("completion_tokens", ""),
            "total_tokens": result.get("total_tokens", ""),
            "cached_tokens": result.get("cached_tokens", ""),
            "cache_write_tokens": result.get("cache_write_tokens", ""),
            "reasoning_tokens": result.get("reasoning_tokens", ""),
            "cost": result.get("cost", ""),
        }

        append_csv_row(BENCHMARK_RUNS_FILE, row)

        completed += 1
        if result.get("status") == "success":
            successful += 1
            print(f"  OK - Latency: {result['latency_sec']:.3f}s, "
                  f"Tokens: {result.get('total_tokens')}, "
                  f"Cost: ${result.get('cost', 0):.6f}")
        else:
            failed += 1
            print(f"  FAILED - {result.get('error_message', 'Unknown error')}")

        time.sleep(1)

    print(f"\nBenchmark complete: {run_id}")
    print(f"Successful: {successful}/{total_tests}, Failed: {failed}/{total_tests}")

    return {
        "success": True,
        "run_id": run_id,
        "benchmark_reason": benchmark_reason,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "total_tests": total_tests,
        "successful": successful,
        "failed": failed,
        "catalog_id": catalog_id,
    }


def benchmark_new_models(new_model_ids: list[str], benchmark_reason: str = "new_model_detected") -> dict:
    catalog = load_current_catalog()
    if not catalog:
        return {"success": False, "error": "No catalog available"}

    all_models = catalog.get("models", [])
    models_to_test = [m for m in all_models if m.get("model_id") in new_model_ids]

    if not models_to_test:
        return {"success": False, "error": "None of the new model IDs found in catalog"}

    return run_benchmark(models=models_to_test, benchmark_reason=benchmark_reason)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark free models from OpenRouter")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to benchmark")
    parser.add_argument("--prompt", help="Prompt ID to use")
    parser.add_argument("--reason", default="manual", choices=["manual", "scheduled", "new_model_detected"],
                        help="Benchmark reason")
    parser.add_argument("--run-id", help="Custom run ID")
    parser.add_argument("--new-only", action="store_true", help="Benchmark only newly discovered models")
    args = parser.parse_args()

    models = None
    if args.models:
        models = [{"model_id": m, "display_name": m, "canonical_family": "manual"} for m in args.models]

    if args.new_only:
        catalog = load_current_catalog()
        if catalog:
            models = catalog.get("models", [])
        result = run_benchmark(models=models, prompt_id=args.prompt,
                              benchmark_reason="new_model_detected", run_id=args.run_id)
    else:
        result = run_benchmark(models=models, prompt_id=args.prompt,
                              benchmark_reason=args.reason, run_id=args.run_id)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
