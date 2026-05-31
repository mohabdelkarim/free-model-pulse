#!/usr/bin/env python3
"""
Model Discovery and Watcher Module.

Dynamically discovers free models from OpenRouter API.
Handles rate limits, transient failures, and model churn.
Detects new/removed models and stores catalog history.
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from common import (
    CATALOG_DIR,
    CURRENT_MODELS_FILE,
    HISTORY_FILE,
    load_json,
    save_json,
    append_jsonl,
    now_iso,
    setup_logging,
    get_logger,
)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
MAX_RETRY_DELAY = 60
RETRY_BUDGET_SEC = 120


LOG = get_logger("watch_models")


def get_api_headers() -> dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_email = os.getenv("OPENROUTER_SITE_EMAIL")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_email:
        headers["X-Title"] = "Free Model Pulse"
    return headers


def is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or (500 <= status_code < 600)


def compute_backoff_delay(attempt: int, retry_after: Optional[int] = None) -> float:
    if retry_after and retry_after > 0:
        return min(retry_after, MAX_RETRY_DELAY)

    delay = RETRY_DELAY_BASE * (2 ** attempt)
    jitter = delay * 0.1 * (hashlib.md5(str(time.time()).encode()).hexdigest()[0:2], int.from_bytes) % 10)
    return min(delay + jitter, MAX_RETRY_DELAY)


def fetch_models_with_retry(retries: int = MAX_RETRIES) -> Optional[dict]:
    headers = get_api_headers()
    start_time = time.time()

    for attempt in range(retries):
        elapsed = time.time() - start_time
        if elapsed >= RETRY_BUDGET_SEC:
            LOG.warning("Retry budget exhausted after %.1fs", elapsed)
            return None

        try:
            LOG.debug("Fetching models (attempt %d/%d)", attempt + 1, retries)
            response = requests.get(
                OPENROUTER_API_URL,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 0))
                delay = compute_backoff_delay(attempt, retry_after)
                LOG.warning("Rate limited. Attempt %d/%d. Waiting %.1fs (budget: %.1fs remaining)",
                           attempt + 1, retries, delay, RETRY_BUDGET_SEC - elapsed)
                time.sleep(delay)
                continue

            if is_retryable_status(response.status_code):
                delay = compute_backoff_delay(attempt)
                LOG.warning("Server error %d. Attempt %d/%d. Waiting %.1fs",
                           response.status_code, attempt + 1, retries, delay)
                time.sleep(delay)
                continue

            if response.status_code != 200:
                LOG.error("API error %d: %s", response.status_code, response.text[:200])
                return None

            LOG.info("Successfully fetched models from OpenRouter")
            return response.json()

        except requests.exceptions.Timeout:
            delay = compute_backoff_delay(attempt)
            LOG.warning("Request timeout. Attempt %d/%d. Waiting %.1fs",
                       attempt + 1, retries, delay)
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            LOG.error("All retry attempts exhausted due to timeout")

        except requests.exceptions.ConnectionError as e:
            delay = compute_backoff_delay(attempt)
            LOG.warning("Connection failed: %s. Attempt %d/%d. Waiting %.1fs",
                       str(e)[:100], attempt + 1, retries, delay)
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            LOG.error("All retry attempts exhausted due to connection error")

        except requests.exceptions.RequestException as e:
            LOG.error("Request failed: %s", str(e)[:200])
            return None

    LOG.error("Failed to fetch models after %d attempts", retries)
    return None


def fetch_models() -> Optional[dict]:
    return fetch_models_with_retry()


def is_benchmarkable(model: dict) -> tuple[bool, str]:
    model_id = model.get("id", "")

    if not model_id:
        return False, "empty model id"

    if "free" in model_id.lower() and "/" not in model_id:
        return False, "id contains 'free' without provider prefix"

    if model.get("disabled", False):
        return False, "model is disabled"

    if model.get("hidden", False):
        return False, "model is hidden"

    pricing = model.get("pricing", {})
    try:
        prompt_price = float(pricing.get("prompt", 0))
        completion_price = float(pricing.get("completion", 0))
        if prompt_price > 0 or completion_price > 0:
            return False, "model is not free"
    except (ValueError, TypeError):
        return False, "invalid pricing data"

    supported_params = model.get("supported_parameters", [])
    if supported_params and "messages" not in supported_params:
        return False, "model does not support messages API"

    return True, ""


def filter_free_models(models_data: dict) -> list[dict]:
    all_models = models_data.get("data", [])
    LOG.info("Filtering %d total models for benchmarkable free models", len(all_models))

    free_models = []
    excluded = []

    for model in all_models:
        is_free, reason = is_benchmarkable(model)
        if is_free:
            free_models.append(model)
        else:
            excluded.append((model.get("id", "unknown"), reason))

    if excluded:
        LOG.debug("Excluded models: %d", len(excluded))
        for model_id, reason in excluded[:5]:
            LOG.debug("  - %s: %s", model_id, reason)
        if len(excluded) > 5:
            LOG.debug("  ... and %d more", len(excluded) - 5)

    LOG.info("Found %d benchmarkable free models out of %d total",
             len(free_models), len(all_models))
    return free_models


def normalize_model(model: dict) -> dict:
    model_id = model.get("id", "")
    parts = model_id.split("/")
    canonical_family = parts[0] if len(parts) > 1 else "unknown"

    return {
        "model_id": model_id,
        "display_name": model.get("name", model_id),
        "canonical_family": canonical_family,
        "context_length": model.get("context_length"),
        "description": (model.get("description") or "")[:500],
        "pricing": model.get("pricing", {}),
        "top_provider": model.get("top_provider", {}),
        "created": model.get("created"),
        "supported_parameters": model.get("supported_parameters", []),
    }


def generate_catalog_id(models: list[dict]) -> str:
    model_ids = sorted([m["model_id"] for m in models])
    content = json.dumps(model_ids, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_current_catalog() -> Optional[dict]:
    return load_json(CURRENT_MODELS_FILE)


def save_current_catalog(models: list[dict], catalog_id: str) -> None:
    timestamp = now_iso()
    normalized = [normalize_model(m) for m in models]

    catalog = {
        "catalog_id": catalog_id,
        "timestamp": timestamp,
        "total_models": len(normalized),
        "models": normalized,
    }

    save_json(CURRENT_MODELS_FILE, catalog)

    history_entry = {
        "catalog_id": catalog_id,
        "timestamp": timestamp,
        "action": "snapshot",
        "total_models": len(normalized),
        "model_ids": sorted([m["model_id"] for m in normalized]),
    }
    append_jsonl(HISTORY_FILE, history_entry)


def detect_changes(old_catalog: Optional[dict], new_models: list[dict]) -> dict:
    if old_catalog is None:
        return {
            "has_changes": True,
            "new_models": [m["model_id"] for m in new_models],
            "removed_models": [],
            "total_new": len(new_models),
            "total_removed": 0,
        }

    old_models = old_catalog.get("models", [])
    old_ids = set(m["model_id"] for m in old_models if isinstance(m, dict))
    new_ids = set(m["model_id"] for m in new_models if isinstance(m, dict))

    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)

    return {
        "has_changes": bool(added or removed),
        "new_models": added,
        "removed_models": removed,
        "total_new": len(added),
        "total_removed": len(removed),
    }


def discover_models(force: bool = False) -> dict:
    LOG.info("Starting model discovery")

    models_data = fetch_models()
    if not models_data:
        LOG.error("Failed to fetch models from OpenRouter")
        return {"success": False, "error": "API fetch failed"}

    all_models = models_data.get("data", [])
    LOG.info("Received %d models from API", len(all_models))

    free_models = filter_free_models(models_data)

    if len(free_models) == 0:
        LOG.warning("No free models found. OpenRouter catalog may have changed.")

    catalog_id = generate_catalog_id(free_models)

    old_catalog = load_current_catalog()
    old_catalog_id = old_catalog.get("catalog_id") if old_catalog else None

    if catalog_id == old_catalog_id and not force:
        LOG.info("Catalog unchanged (ID: %s). No action needed.", catalog_id)
        return {
            "success": True,
            "catalog_id": catalog_id,
            "has_changes": False,
            "new_models": [],
            "removed_models": [],
            "total_models": len(free_models),
        }

    changes = detect_changes(old_catalog, free_models)

    history_entry = {
        "catalog_id": catalog_id,
        "timestamp": now_iso(),
        "action": "update" if old_catalog else "initial",
        "changes": changes,
        "total_models": len(free_models),
    }
    append_jsonl(HISTORY_FILE, history_entry)

    save_current_catalog(free_models, catalog_id)

    LOG.info("Catalog saved: %s", catalog_id)
    if changes["has_changes"]:
        if changes["total_new"] > 0:
            LOG.info("New models detected: %d", changes["total_new"])
            for m in changes["new_models"]:
                LOG.info("  + %s", m)
        if changes["total_removed"] > 0:
            LOG.info("Removed models: %d", changes["total_removed"])
            for m in changes["removed_models"]:
                LOG.info("  - %s", m)

    return {
        "success": True,
        "catalog_id": catalog_id,
        "has_changes": changes["has_changes"],
        "new_models": changes["new_models"],
        "removed_models": changes["removed_models"],
        "total_models": len(free_models),
    }


def watch_models(interval_minutes: int = 60) -> None:
    LOG.info("Starting model watcher (checking every %d minutes)", interval_minutes)
    LOG.info("Press Ctrl+C to stop")

    try:
        while True:
            result = discover_models()
            if result["success"]:
                if result["has_changes"] and result["new_models"]:
                    LOG.info("NEW MODELS DETECTED: %d", len(result["new_models"]))
                    LOG.info("Benchmark trigger available. Use --trigger-benchmark flag.")
                else:
                    LOG.info("No model changes detected.")
            else:
                LOG.error("Watcher error: %s", result.get("error"))

            LOG.info("Waiting %d minutes before next check...", interval_minutes)
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        LOG.info("Watcher stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover free models from OpenRouter")
    parser.add_argument("--watch", action="store_true", help="Run continuously in watch mode")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in minutes")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if unchanged")
    args = parser.parse_args()

    setup_logging()

    if args.watch:
        watch_models(args.interval)
    else:
        setup_logging()
        result = discover_models(force=args.force)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
