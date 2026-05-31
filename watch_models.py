#!/usr/bin/env python3
"""
Model Discovery Module

Dynamically discovers free models from OpenRouter API.
Handles rate limits, transient failures, and model churn.
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5


def get_default_dirs() -> tuple[Path, Path]:
    catalogs_dir = Path(os.getenv("CATALOGS_DIR", "data/catalogs"))
    catalogs_dir.mkdir(parents=True, exist_ok=True)
    return catalogs_dir


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


def filter_free_models(models_data: dict) -> list[dict]:
    free_models = []
    for model in models_data.get("data", []):
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0))
        completion_price = float(pricing.get("completion", 0))
        if prompt_price == 0 and completion_price == 0:
            free_models.append(model)
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
        "description": model.get("description", "")[:500],
        "pricing": model.get("pricing", {}),
        "top_provider": model.get("top_provider", {}),
        "created": model.get("created"),
        "disabled": model.get("disabled", False),
        "hidden": model.get("hidden", False),
    }


def generate_catalog_id(models: list[dict]) -> str:
    model_ids = sorted([m["model_id"] for m in models])
    content = json.dumps(model_ids, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def save_catalog_snapshot(models: list[dict], catalogs_dir: Path) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    catalog_id = generate_catalog_id(models)

    snapshot = {
        "catalog_id": catalog_id,
        "timestamp": timestamp,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "total_models": len(models),
        "models": [normalize_model(m) for m in models],
    }

    filename = f"catalog_{timestamp}_{catalog_id}.json"
    filepath = catalogs_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    latest_link = catalogs_dir / "latest.json"
    with open(latest_link, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"Saved catalog snapshot: {filename}")
    return catalog_id, filepath


def detect_model_changes(old_models: list[dict], new_models: list[dict]) -> dict:
    old_ids = set(m["model_id"] for m in old_models)
    new_ids = set(m["model_id"] for m in new_models)

    added = list(new_ids - old_ids)
    removed = list(old_ids - new_ids)
    unchanged = list(new_ids & old_ids)

    return {
        "added_models": added,
        "removed_models": removed,
        "unchanged_models": unchanged,
        "total_added": len(added),
        "total_removed": len(removed),
    }


def load_latest_catalog(catalogs_dir: Path) -> Optional[dict]:
    latest_path = catalogs_dir / "latest.json"
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def discover_models(catalogs_dir: Optional[Path] = None, force: bool = False) -> dict:
    if catalogs_dir is None:
        catalogs_dir = get_default_dirs()[0]

    catalogs_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching models from OpenRouter...")
    models_data = fetch_models()
    if not models_data:
        print("Failed to fetch models from OpenRouter")
        return {"success": False, "error": "API fetch failed"}

    free_models = filter_free_models(models_data)
    print(f"Found {len(free_models)} free models out of {len(models_data.get('data', []))} total")

    if len(free_models) == 0:
        print("WARNING: No free models found. OpenRouter catalog may have changed.")

    new_catalog_id, snapshot_path = save_catalog_snapshot(free_models, catalogs_dir)

    old_catalog = load_latest_catalog(catalogs_dir)
    old_models = old_catalog.get("models", []) if old_catalog else []
    changes = detect_model_changes(old_models, free_models)

    return {
        "success": True,
        "catalog_id": new_catalog_id,
        "snapshot_path": str(snapshot_path),
        "total_models": len(free_models),
        "changes": changes,
    }


def watch_models(interval_minutes: int = 60) -> None:
    catalogs_dir = get_default_dirs()[0]

    print(f"Starting model watcher (checking every {interval_minutes} minutes)...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            result = discover_models(catalogs_dir)
            if result["success"]:
                changes = result["changes"]
                if changes["total_added"] > 0:
                    print(f"\nNEW MODELS DETECTED: {changes['total_added']}")
                    for model_id in changes["added_models"]:
                        print(f"  + {model_id}")
                if changes["total_removed"] > 0:
                    print(f"\nMODELS REMOVED: {changes['total_removed']}")
                    for model_id in changes["removed_models"]:
                        print(f"  - {model_id}")
                if changes["total_added"] == 0 and changes["total_removed"] == 0:
                    print("No model changes detected.")
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
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in minutes (default: 60)")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if unchanged")
    args = parser.parse_args()

    if args.watch:
        watch_models(args.interval)
    else:
        result = discover_models(force=args.force)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
