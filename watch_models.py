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
)


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5


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


def fetch_models(retries: int = MAX_RETRIES) -> Optional[dict]:
    headers = get_api_headers()
    for attempt in range(retries):
        try:
            response = requests.get(
                OPENROUTER_API_URL,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", RETRY_DELAY * (attempt + 1)))
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                print(f"API error {response.status_code}: {response.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Request timeout on attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
    return None


def is_benchmarkable(model: dict) -> tuple[bool, str]:
    model_id = model.get("id", "")

    if "free" in model_id.lower() and "/" not in model_id:
        return False, "id contains 'free' without provider prefix"

    if model.get("disabled", False):
        return False, "model is disabled"

    if model.get("hidden", False):
        return False, "model is hidden"

    pricing = model.get("pricing", {})
    prompt_price = float(pricing.get("prompt", 0))
    completion_price = float(pricing.get("completion", 0))
    if prompt_price > 0 or completion_price > 0:
        return False, "model is not free"

    supported_types = model.get("supported_parameters", [])
    if "messages" not in supported_types and "prompt" not in str(model.get("capabilities", {})):
        return False, "model does not support messages API"

    return True, ""


def filter_free_models(models_data: dict) -> list[dict]:
    free_models = []
    for model in models_data.get("data", []):
        is_free, reason = is_benchmarkable(model)
        if is_free:
            free_models.append(model)
        else:
            print(f"  Excluded {model.get('id')}: {reason}")
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
    print("Fetching models from OpenRouter...")
    models_data = fetch_models()
    if not models_data:
        return {"success": False, "error": "API fetch failed"}

    all_models = models_data.get("data", [])
    free_models = filter_free_models(all_models)

    print(f"Found {len(free_models)} benchmarkable free models out of {len(all_models)} total")

    if len(free_models) == 0:
        print("WARNING: No free models found. OpenRouter catalog may have changed.")

    catalog_id = generate_catalog_id(free_models)

    old_catalog = load_current_catalog()
    old_catalog_id = old_catalog.get("catalog_id") if old_catalog else None

    if catalog_id == old_catalog_id and not force:
        print("Catalog unchanged. No action needed.")
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

    print(f"Catalog saved: {catalog_id}")
    if changes["has_changes"]:
        if changes["total_new"] > 0:
            print(f"New models: {changes['total_new']}")
            for m in changes["new_models"]:
                print(f"  + {m}")
        if changes["total_removed"] > 0:
            print(f"Removed models: {changes['total_removed']}")
            for m in changes["removed_models"]:
                print(f"  - {m}")

    return {
        "success": True,
        "catalog_id": catalog_id,
        "has_changes": changes["has_changes"],
        "new_models": changes["new_models"],
        "removed_models": changes["removed_models"],
        "total_models": len(free_models),
    }


def watch_models(interval_minutes: int = 60) -> None:
    print(f"Starting model watcher (checking every {interval_minutes} minutes)...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            result = discover_models()
            if result["success"]:
                if result["has_changes"] and result["new_models"]:
                    print(f"\nNEW MODELS DETECTED: {len(result['new_models'])}")
                    print("Benchmark trigger available. Use --trigger-benchmark flag.")
            else:
                print(f"Watcher error: {result.get('error')}")

            print(f"\nWaiting {interval_minutes} minutes before next check...")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\nWatcher stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Discover free models from OpenRouter")
    parser.add_argument("--watch", action="store_true", help="Run continuously in watch mode")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in minutes")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if unchanged")
    args = parser.parse_args()

    if args.watch:
        watch_models(args.interval)
    else:
        result = discover_models(force=args.force)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
