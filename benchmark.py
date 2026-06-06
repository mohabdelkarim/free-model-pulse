#!/usr/bin/env python3
"""
Benchmarking Module.

Benchmarks free models using stable prompts from prompts.json.
Records results to append-only CSV raw datasets.
"""

import os
import json
import time
import hashlib
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
    setup_logging,
    get_logger,
)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = int(os.getenv("BENCHMARK_TIMEOUT_SEC", "120"))
MAX_RETRIES = int(os.getenv("BENCHMARK_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = 2
MAX_RETRY_DELAY = 60
RETRY_BUDGET_SEC = 300

# Delay between models inside a batch
INTER_MODEL_DELAY = float(os.getenv("BENCHMARK_INTER_MODEL_DELAY", "8"))

# Batching: test N models, then pause before next batch
BATCH_SIZE = int(os.getenv("BENCHMARK_BATCH_SIZE", "5"))
BATCH_PAUSE_SEC = float(os.getenv("BENCHMARK_BATCH_PAUSE_SEC", "60"))

# Circuit breaker: pause entire benchmark after N consecutive 429s
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("BENCHMARK_CIRCUIT_BREAKER_THRESHOLD", "3"))
CIRCUIT_BREAKER_PAUSE = float(os.getenv("BENCHMARK_CIRCUIT_BREAKER_PAUSE", "120"))

# HTTP status codes that are never worth retrying
NON_RETRYABLE_STATUSES = {400, 401, 403, 404, 422}

LOG = get_logger("benchmark")


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


def is_retryable_status(status_code: int) -> bool:
    if status_code in NON_RETRYABLE_STATUSES:
        return False
    return status_code == 429 or (500 <= status_code < 600)


def compute_backoff_delay(attempt: int, retry_after: Optional[int] = None) -> float:
    if retry_after and retry_after > 0:
        return min(retry_after, MAX_RETRY_DELAY)
    delay = RETRY_DELAY_BASE * (2 ** attempt)
    jitter = delay * 0.1 * (int(hashlib.md5(str(time.time()).encode()).hexdigest()[:2], 16) % 10)
    return min(delay + jitter, MAX_RETRY_DELAY)


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
    last_error = None

    for attempt in range(MAX_RETRIES):
        elapsed = time.time() - start_time
        if elapsed >= RETRY_BUDGET_SEC:
            LOG.warning("Retry budget exhausted for %s after %.1fs", model_id, elapsed)
            return {
                "status": "error",
                "error_message": f"Retry budget exhausted after {elapsed:.1f}s",
                "latency_sec": round(elapsed, 3),
            }

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            elapsed_time = time.time() - start_time

            # Non-retryable errors (403, 401, 404, etc.) — skip immediately, no retry
            if response.status_code in NON_RETRYABLE_STATUSES:
                LOG.warning("[%s] Non-retryable HTTP %d — skipping. Body: %s",
                            model_id, response.status_code, response.text[:200])
                return {
                    "status": "error",
                    "error_message": f"HTTP {response.status_code} (non-retryable): {response.text[:200]}",
                    "latency_sec": round(elapsed_time, 3),
                }

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 0))
                delay = compute_backoff_delay(attempt, retry_after)
                LOG.warning("[%s] Rate limited. Attempt %d/%d. Waiting %.1fs",
                            model_id, attempt + 1, MAX_RETRIES, delay)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    continue
                # Return with a clear 429 marker so the circuit breaker can detect it
                return {
                    "status": "error",
                    "error_message": f"429: Rate limited after {MAX_RETRIES} attempts",
                    "latency_sec": round(elapsed_time, 3),
                }

            if is_retryable_status(response.status_code):
                delay = compute_backoff_delay(attempt)
                LOG.warning("[%s] Server error %d. Attempt %d/%d. Waiting %.1fs",
                            model_id, response.status_code, attempt + 1, MAX_RETRIES, delay)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    continue
                return {
                    "status": "error",
                    "error_message": f"Server error {response.status_code} after {MAX_RETRIES} attempts",
                    "latency_sec": round(elapsed_time, 3),
                }

            if response.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"HTTP {response.status_code}: {response.text[:200]}",
                    "latency_sec": round(elapsed_time, 3),
                }

            data = response.json()

            if "choices" not in data or not data["choices"]:
                return {
                    "status": "error",
                    "error_message": "Empty or malformed response: no choices",
                    "latency_sec": round(elapsed_time, 3),
                }

            usage = data.get("usage", {})
            choice = data.get("choices", [{}])[0]

            finish_reason = choice.get("finish_reason", "unknown")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("reason", "unknown")

            LOG.debug("[%s] Success - Latency: %.3fs, Tokens: %s",
                      model_id, elapsed_time, usage.get("total_tokens", "N/A"))

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
            elapsed_time = time.time() - start_time
            LOG.warning("[%s] Timeout after %.1fs. Attempt %d/%d",
                        model_id, elapsed_time, attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                delay = compute_backoff_delay(attempt)
                time.sleep(delay)
                continue
            return {
                "status": "timeout",
                "error_message": f"Request timed out after {timeout}s",
                "latency_sec": round(elapsed_time, 3),
            }

        except requests.exceptions.ConnectionError as e:
            elapsed_time = time.time() - start_time
            LOG.warning("[%s] Connection error: %s. Attempt %d/%d",
                        model_id, str(e)[:100], attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                delay = compute_backoff_delay(attempt)
                time.sleep(delay)
                continue
            return {
                "status": "error",
                "error_message": f"Connection error: {str(e)[:200]}",
                "latency_sec": round(elapsed_time, 3),
            }

        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            LOG.error("[%s] Request failed: %s", model_id, str(e)[:200])
            return {
                "status": "error",
                "error_message": str(e)[:200],
                "latency_sec": round(elapsed_time, 3),
            }

    return {
        "status": "error",
        "error_message": last_error or "Max retries exceeded",
        "latency_sec": time.time() - start_time,
    }


def run_benchmark(
    models: Optional[list[dict]] = None,
    prompt_id: Optional[str] = None,
    benchmark_reason: str = "manual",
    run_id: Optional[str] = None,
) -> dict:
    LOG.info("Starting benchmark run")

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
        LOG.error("No prompts found in prompts.json")
        return {"success": False, "error": "No prompts found in prompts.json"}

    prompt = selected_prompts[0]
    prompt_id = prompt["id"]
    prompt_text = prompt["text"]
    prompt_version = prompts_data.get("version", "1.0.0")

    if not models:
        catalog = load_current_catalog()
        if not catalog:
            LOG.error("No catalog available. Run watch_models.py first.")
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
    skipped = 0

    LOG.info("Run ID: %s", run_id)
    LOG.info("Reason: %s", benchmark_reason)
    LOG.info("Prompt: %s (v%s)", prompt_id, prompt_version)
    LOG.info("Models to test: %d", total_tests)
    LOG.info("Batch size: %d, pause between batches: %.0fs", BATCH_SIZE, BATCH_PAUSE_SEC)
    LOG.info("Inter-model delay: %.1fs", INTER_MODEL_DELAY)
    LOG.info("Circuit breaker: %d consecutive 429s → pause %.0fs",
             CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_PAUSE)

    start_time = time.time()
    consecutive_429s = 0

    for i, model in enumerate(models):
        model_id = model.get("model_id")
        if not model_id:
            skipped += 1
            LOG.warning("Skipping model with empty model_id")
            continue

        completed += 1
        LOG.info("[%d/%d] Testing %s...", completed, total_tests, model_id)

        result = benchmark_single_model(model_id, prompt_text, prompt_version)

        # --- Circuit breaker: detect account-level 429 throttle ---
        error_msg = result.get("error_message", "")
        if result.get("status") in ("error", "timeout") and "429" in error_msg:
            consecutive_429s += 1
            LOG.warning("Consecutive 429s: %d/%d", consecutive_429s, CIRCUIT_BREAKER_THRESHOLD)
            if consecutive_429s >= CIRCUIT_BREAKER_THRESHOLD:
                LOG.warning(
                    "Circuit breaker triggered: %d consecutive 429s — pausing %.0fs before continuing",
                    consecutive_429s, CIRCUIT_BREAKER_PAUSE
                )
                time.sleep(CIRCUIT_BREAKER_PAUSE)
                consecutive_429s = 0
        else:
            consecutive_429s = 0

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

        if result.get("status") == "success":
            successful += 1
            LOG.info("[%s] SUCCESS - Latency: %.3fs, Tokens: %s, Cost: $%.6f",
                     model_id,
                     result.get("latency_sec", 0),
                     result.get("total_tokens", "N/A"),
                     result.get("cost", 0))
        else:
            failed += 1
            LOG.error("[%s] FAILED - %s", model_id, result.get("error_message", "Unknown error"))

        # Inter-model delay (skip after last model)
        if i < total_tests - 1:
            time.sleep(INTER_MODEL_DELAY)

        # Batch pause: after every BATCH_SIZE models, pause to let rate limits recover
        batch_position = (i + 1) % BATCH_SIZE
        is_last_model = i == total_tests - 1
        if batch_position == 0 and not is_last_model:
            LOG.info("--- Batch of %d complete. Pausing %.0fs before next batch... ---",
                     BATCH_SIZE, BATCH_PAUSE_SEC)
            time.sleep(BATCH_PAUSE_SEC)

    total_duration = time.time() - start_time

    LOG.info("=" * 60)
    LOG.info("Benchmark complete: %s", run_id)
    LOG.info("Duration: %.1fs", total_duration)
    LOG.info("Results: %d/%d successful, %d failed, %d skipped",
             successful, total_tests, failed, skipped)
    LOG.info("=" * 60)

    return {
        "success": True,
        "run_id": run_id,
        "benchmark_reason": benchmark_reason,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "total_tests": total_tests,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "duration_sec": round(total_duration, 1),
        "catalog_id": catalog_id,
    }


def benchmark_new_models(new_model_ids: list[str], benchmark_reason: str = "new_model_detected") -> dict:
    LOG.info("Benchmarking %d new models", len(new_model_ids))

    catalog = load_current_catalog()
    if not catalog:
        LOG.error("No catalog available")
        return {"success": False, "error": "No catalog available"}

    all_models = catalog.get("models", [])
    models_to_test = [m for m in all_models if m.get("model_id") in new_model_ids]

    if not models_to_test:
        LOG.error("None of the new model IDs found in catalog")
        return {"success": False, "error": "None of the new model IDs found in catalog"}

    return run_benchmark(models=models_to_test, benchmark_reason=benchmark_reason)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark free models from OpenRouter")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to benchmark")
    parser.add_argument("--prompt", help="Prompt ID to use")
    parser.add_argument("--reason", default="manual",
                        choices=["manual", "scheduled", "new_model_detected"],
                        help="Benchmark reason")
    parser.add_argument("--run-id", help="Custom run ID")
    parser.add_argument("--new-only", action="store_true",
                        help="Benchmark only newly discovered models")
    args = parser.parse_args()

    setup_logging()

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
