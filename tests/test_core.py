#!/usr/bin/env python3
"""
Unit tests for free-model-pulse core functions.

Tests:
- Model filtering (is_benchmarkable, is_router_or_helper)
- Model normalization (normalize_model)
- Diff detection (detect_changes)
- Aggregate calculations on mixed success/error rows

Run with: pytest tests/ -v
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import (
    safe_float,
    safe_int,
    load_json,
    save_json,
    read_jsonl,
    append_jsonl,
    read_csv,
    append_csv_row,
    ensure_csv_header,
)

import watch_models
import analyze


class TestSafeConversions(unittest.TestCase):
    def test_safe_float_with_valid_value(self):
        self.assertEqual(safe_float(1.5), 1.5)
        self.assertEqual(safe_float("1.5"), 1.5)
        self.assertEqual(safe_float(0), 0.0)

    def test_safe_float_with_none(self):
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float(None, 99.0), 99.0)

    def test_safe_float_with_invalid(self):
        self.assertEqual(safe_float("invalid"), 0.0)
        self.assertEqual(safe_float([1, 2]), 0.0)
        self.assertEqual(safe_float("invalid", 42.0), 42.0)

    def test_safe_int_with_valid_value(self):
        self.assertEqual(safe_int(5), 5)
        self.assertEqual(safe_int(5.9), 5)
        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(0), 0)

    def test_safe_int_with_none(self):
        self.assertEqual(safe_int(None), 0)
        self.assertEqual(safe_int(None, 99), 99)

    def test_safe_int_with_invalid(self):
        self.assertEqual(safe_int("invalid"), 0)
        self.assertEqual(safe_int("invalid", 42), 42)


class TestRouterExclusion(unittest.TestCase):
    def test_excludes_openrouter_free(self):
        is_router, reason = watch_models.is_router_or_helper("openrouter/free")
        self.assertTrue(is_router)
        self.assertIn("router", reason.lower())

    def test_excludes_openrouter_auto(self):
        is_router, reason = watch_models.is_router_or_helper("openrouter/auto")
        self.assertTrue(is_router)
        self.assertIn("router", reason.lower())

    def test_excludes_pareto_code(self):
        is_router, reason = watch_models.is_router_or_helper("openrouter/pareto-code")
        self.assertTrue(is_router)
        self.assertIn("pareto-code", reason)

    def test_excludes_bodybuilder(self):
        is_router, reason = watch_models.is_router_or_helper("openrouter/bodybuilder")
        self.assertTrue(is_router)
        self.assertIn("bodybuilder", reason)

    def test_includes_normal_free_model(self):
        is_router, reason = watch_models.is_router_or_helper("nvidia/nemotron-3-nano-30b-a3b:free")
        self.assertFalse(is_router)

    def test_includes_google_gemma(self):
        is_router, reason = watch_models.is_router_or_helper("google/gemma-4-26b-a4b-it:free")
        self.assertFalse(is_router)


class TestModelFiltering(unittest.TestCase):
    def test_includes_free_model_with_zero_pricing(self):
        model = {
            "id": "nvidia/nemotron-3-nano-30b-a3b:free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertTrue(is_free)
        self.assertEqual(reason, "")

    def test_includes_free_model_with_float_zero_pricing(self):
        model = {
            "id": "google/gemma-4-26b-a4b-it:free",
            "pricing": {"prompt": 0.0, "completion": 0.0},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertTrue(is_free)

    def test_excludes_openrouter_free_router(self):
        model = {
            "id": "openrouter/free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("router", reason.lower())

    def test_excludes_openrouter_auto(self):
        model = {
            "id": "openrouter/auto",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("router", reason.lower())

    def test_excludes_pareto_code(self):
        model = {
            "id": "openrouter/pareto-code",
            "pricing": {"prompt": "-1", "completion": "-1"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("pareto-code", reason)

    def test_excludes_bodybuilder(self):
        model = {
            "id": "openrouter/bodybuilder",
            "pricing": {"prompt": "-1", "completion": "-1"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("bodybuilder", reason)

    def test_excludes_model_with_price(self):
        model = {
            "id": "openai/gpt-4",
            "pricing": {"prompt": 0.00001, "completion": 0.00002},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("not free", reason)

    def test_excludes_disabled_model(self):
        model = {
            "id": "test/model",
            "pricing": {"prompt": 0, "completion": 0},
            "disabled": True,
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("disabled", reason)

    def test_excludes_hidden_model(self):
        model = {
            "id": "test/model",
            "pricing": {"prompt": 0, "completion": 0},
            "hidden": True,
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("hidden", reason)

    def test_excludes_empty_model_id(self):
        model = {
            "id": "",
            "pricing": {"prompt": 0, "completion": 0},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("empty", reason)

    def test_model_missing_optional_fields_does_not_crash(self):
        model = {
            "id": "test/model",
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertTrue(is_free)

    def test_model_with_missing_supported_parameters_does_not_crash(self):
        model = {
            "id": "liquid/lfm-2.5-1.2b-instruct:free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertTrue(is_free)

    def test_excludes_whisper_model(self):
        model = {
            "id": "openai/whisper-1",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)

    def test_excludes_audio_model_in_name(self):
        model = {
            "id": "some/audio-model:free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)

    def test_excludes_vl_model(self):
        model = {
            "id": "nvidia/nemotron-nano-12b-v2-vl:free",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)

    def test_excludes_clip_model(self):
        model = {
            "id": "google/lyria-3-clip-preview",
            "pricing": {"prompt": "0", "completion": "0"},
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)

    def test_excludes_model_with_audio_description(self):
        model = {
            "id": "test/audio-model:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "description": "An audio transcription model",
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)

    def test_excludes_model_with_video_description(self):
        model = {
            "id": "test/video-model:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "description": "A video understanding model",
        }
        is_free, reason = watch_models.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("non-text", reason)


class TestModelNormalization(unittest.TestCase):
    def test_normalize_model_with_provider(self):
        model = {
            "id": "google/gemini-pro",
            "name": "Gemini Pro",
            "context_length": 32000,
            "description": "A great model",
        }
        result = watch_models.normalize_model(model)
        self.assertEqual(result["model_id"], "google/gemini-pro")
        self.assertEqual(result["display_name"], "Gemini Pro")
        self.assertEqual(result["canonical_family"], "google")
        self.assertEqual(result["context_length"], 32000)

    def test_normalize_model_without_provider(self):
        model = {
            "id": "unknown-model",
            "name": "Unknown Model",
        }
        result = watch_models.normalize_model(model)
        self.assertEqual(result["canonical_family"], "unknown")

    def test_normalize_model_truncates_description(self):
        model = {
            "id": "test/model",
            "description": "x" * 1000,
        }
        result = watch_models.normalize_model(model)
        self.assertEqual(len(result["description"]), 500)

    def test_normalize_model_handles_none_description(self):
        model = {
            "id": "test/model",
            "description": None,
        }
        result = watch_models.normalize_model(model)
        self.assertEqual(result["description"], "")


class TestDiffDetection(unittest.TestCase):
    def test_detect_changes_with_no_old_catalog(self):
        new_models = [
            {"model_id": "model/1"},
            {"model_id": "model/2"},
        ]
        result = watch_models.detect_changes(None, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(result["new_models"], ["model/1", "model/2"])
        self.assertEqual(result["removed_models"], [])
        self.assertEqual(result["total_new"], 2)

    def test_detect_changes_with_added_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}]}
        new_models = [
            {"model_id": "model/1"},
            {"model_id": "model/2"},
            {"model_id": "model/3"},
        ]
        result = watch_models.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(sorted(result["new_models"]), ["model/2", "model/3"])
        self.assertEqual(result["removed_models"], [])
        self.assertEqual(result["total_new"], 2)

    def test_detect_changes_with_removed_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}, {"model_id": "model/2"}]}
        new_models = [{"model_id": "model/1"}]
        result = watch_models.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(result["new_models"], [])
        self.assertEqual(result["removed_models"], ["model/2"])
        self.assertEqual(result["total_removed"], 1)

    def test_detect_changes_with_no_changes(self):
        old_catalog = {"models": [{"model_id": "model/1"}, {"model_id": "model/2"}]}
        new_models = [{"model_id": "model/1"}, {"model_id": "model/2"}]
        result = watch_models.detect_changes(old_catalog, new_models)
        self.assertFalse(result["has_changes"])
        self.assertEqual(result["new_models"], [])
        self.assertEqual(result["removed_models"], [])

    def test_detect_changes_handles_non_dict_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}, None, "invalid"]}
        new_models = [{"model_id": "model/1"}, {"model_id": "model/3"}]
        result = watch_models.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(result["new_models"], ["model/3"])
        self.assertEqual(result["removed_models"], [])


class TestAggregateCalculations(unittest.TestCase):
    def test_aggregate_with_all_success(self):
        rows = [
            {"model_id": "m1", "status": "success", "latency_sec": "1.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-01T00:00:00Z"},
            {"model_id": "m1", "status": "success", "latency_sec": "2.0",
             "total_tokens": "150", "cost": "0.0", "run_id": "r2",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-02T00:00:00Z"},
        ]
        result = analyze.aggregate_model_metrics(rows, min_runs=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_id"], "m1")
        self.assertEqual(result[0]["total_runs"], 2)
        self.assertEqual(result[0]["successful_runs"], 2)
        self.assertEqual(result[0]["failed_runs"], 0)
        self.assertEqual(result[0]["success_rate"], 1.0)
        self.assertEqual(result[0]["latency_sec_avg"], 1.5)

    def test_aggregate_with_mixed_success_error(self):
        rows = [
            {"model_id": "m1", "status": "success", "latency_sec": "1.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-01T00:00:00Z"},
            {"model_id": "m1", "status": "error", "latency_sec": "0.5",
             "total_tokens": "0", "cost": "0.0", "run_id": "r2",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-02T00:00:00Z"},
        ]
        result = analyze.aggregate_model_metrics(rows, min_runs=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["total_runs"], 2)
        self.assertEqual(result[0]["successful_runs"], 1)
        self.assertEqual(result[0]["failed_runs"], 1)
        self.assertEqual(result[0]["success_rate"], 0.5)

    def test_aggregate_excludes_below_min_runs(self):
        rows = [
            {"model_id": "m1", "status": "success", "latency_sec": "1.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-01T00:00:00Z"},
        ]
        result = analyze.aggregate_model_metrics(rows, min_runs=3)
        self.assertEqual(len(result), 0)

    def test_aggregate_handles_empty_rows(self):
        result = analyze.aggregate_model_metrics([], min_runs=1)
        self.assertEqual(result, [])

    def test_aggregate_sorts_by_success_rate_and_latency(self):
        rows = [
            {"model_id": "m1", "status": "success", "latency_sec": "5.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M1", "timestamp_utc": "2024-01-01T00:00:00Z"},
            {"model_id": "m2", "status": "success", "latency_sec": "1.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M2", "timestamp_utc": "2024-01-01T00:00:00Z"},
            {"model_id": "m3", "status": "success", "latency_sec": "2.0",
             "total_tokens": "100", "cost": "0.0", "run_id": "r1",
             "canonical_family": "test", "display_name": "M3", "timestamp_utc": "2024-01-01T00:00:00Z"},
        ]
        result = analyze.aggregate_model_metrics(rows, min_runs=1)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["model_id"], "m1")
        self.assertEqual(result[1]["model_id"], "m3")
        self.assertEqual(result[2]["model_id"], "m2")


class TestCSVOperations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test.csv"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_append_csv_row_creates_header(self):
        append_csv_row(self.csv_path, {"col1": "val1", "col2": "val2"})
        self.assertTrue(self.csv_path.exists())

        with open(self.csv_path, "r") as f:
            content = f.read()
        self.assertIn("col1,col2", content)
        self.assertIn("val1,val2", content)

    def test_append_csv_row_appends_without_duplicate_header(self):
        append_csv_row(self.csv_path, {"col1": "val1", "col2": "val2"})
        append_csv_row(self.csv_path, {"col1": "val3", "col2": "val4"})

        with open(self.csv_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].strip(), "col1,col2")

    def test_ensure_csv_header_preserves_existing(self):
        append_csv_row(self.csv_path, {"col1": "val1", "col2": "val2"})
        result = ensure_csv_header(self.csv_path, ["col1", "col2"])
        self.assertFalse(result)

    def test_ensure_csv_header_overwrites_mismatch(self):
        append_csv_row(self.csv_path, {"col1": "val1", "col2": "val2"})
        result = ensure_csv_header(self.csv_path, ["col1", "col3"])
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
