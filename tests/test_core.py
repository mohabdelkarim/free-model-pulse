#!/usr/bin/env python3
"""
Unit tests for free-model-pulse core functions.

Tests:
- Model filtering (is_benchmarkable)
- Model normalization (normalize_model)
- Diff detection (detect_changes)
- Aggregate calculations on mixed success/error rows
- CSV / JSONL file operations
- benchmark_single_model HTTP layer (mocked)
- Circuit breaker logic in run_benchmark (mocked)

Run with: pytest tests/ -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import (
    append_csv_row,
    ensure_csv_header,
    safe_float,
    safe_int,
)

# ---------------------------------------------------------------------------
# Safe conversions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model filtering
# ---------------------------------------------------------------------------

class TestModelFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from watch_models import is_benchmarkable
        cls.is_benchmarkable = staticmethod(is_benchmarkable)

    def _free_model(self, **overrides):
        base = {
            "id": "test/model",
            "pricing": {"prompt": 0, "completion": 0},
        }
        base.update(overrides)
        return base

    def test_free_model_with_provider_prefix(self):
        is_free, _ = self.is_benchmarkable(self._free_model(id="google/gemini-pro"))
        self.assertTrue(is_free)

    def test_excludes_router_free(self):
        is_free, reason = self.is_benchmarkable(self._free_model(id="openrouter/free"))
        self.assertFalse(is_free)
        self.assertIn("router", reason)

    def test_excludes_router_fusion(self):
        """openrouter/fusion must be blocked (returns 403)."""
        is_free, reason = self.is_benchmarkable(self._free_model(id="openrouter/fusion"))
        self.assertFalse(is_free)
        self.assertIn("router", reason)

    def test_excludes_model_with_price(self):
        is_free, reason = self.is_benchmarkable(
            self._free_model(id="openai/gpt-4", pricing={"prompt": 0.00001, "completion": 0.00002})
        )
        self.assertFalse(is_free)
        self.assertIn("not free", reason)

    def test_excludes_disabled_model(self):
        is_free, reason = self.is_benchmarkable(self._free_model(disabled=True))
        self.assertFalse(is_free)
        self.assertIn("disabled", reason)

    def test_excludes_hidden_model(self):
        is_free, reason = self.is_benchmarkable(self._free_model(hidden=True))
        self.assertFalse(is_free)
        self.assertIn("hidden", reason)

    def test_excludes_empty_model_id(self):
        is_free, reason = self.is_benchmarkable(self._free_model(id=""))
        self.assertFalse(is_free)
        self.assertIn("empty", reason)

    def test_excludes_image_output_modality(self):
        model = self._free_model(architecture={
            "input_modalities": ["text"],
            "output_modalities": ["text", "image"],
        })
        is_free, reason = self.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("modality", reason)

    def test_excludes_blocklist_keyword_in_id(self):
        is_free, reason = self.is_benchmarkable(self._free_model(id="test/whisper-large"))
        self.assertFalse(is_free)
        self.assertIn("whisper", reason)

    def test_excludes_blocklist_keyword_in_name(self):
        model = self._free_model(id="test/model", name="Some TTS Model")
        is_free, reason = self.is_benchmarkable(model)
        self.assertFalse(is_free)
        self.assertIn("tts", reason)

    def test_supported_parameters_not_checked(self):
        """is_benchmarkable does NOT filter on supported_parameters — test reflects reality."""
        # A model with only "prompt" in supported_parameters is still benchmarkable
        # because the code doesn't enforce messages support.
        model = self._free_model(supported_parameters=["prompt"])
        is_free, _ = self.is_benchmarkable(model)
        self.assertTrue(is_free)


# ---------------------------------------------------------------------------
# Model normalization
# ---------------------------------------------------------------------------

class TestModelNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from watch_models import normalize_model
        cls.normalize_model = staticmethod(normalize_model)

    def test_normalize_model_with_provider(self):
        model = {
            "id": "google/gemini-pro",
            "name": "Gemini Pro",
            "context_length": 32000,
            "created": 1700000000,
            "description": "A great model",
        }
        result = self.normalize_model(model)
        self.assertEqual(result["model_id"], "google/gemini-pro")
        self.assertEqual(result["display_name"], "Gemini Pro")
        self.assertEqual(result["canonical_family"], "google")
        self.assertEqual(result["context_length"], 32000)
        self.assertEqual(result["created"], 1700000000)

    def test_normalize_model_without_provider(self):
        model = {"id": "unknown-model", "name": "Unknown Model"}
        result = self.normalize_model(model)
        self.assertEqual(result["canonical_family"], "unknown")

    def test_normalize_model_truncates_description(self):
        model = {"id": "test/model", "description": "x" * 1000}
        result = self.normalize_model(model)
        self.assertEqual(len(result["description"]), 500)

    def test_normalize_model_handles_none_description(self):
        model = {"id": "test/model", "description": None}
        result = self.normalize_model(model)
        self.assertEqual(result["description"], "")

    def test_normalize_model_stores_architecture(self):
        """architecture must be stored so leaderboard can read input_modalities."""
        arch = {"input_modalities": ["text", "image"], "output_modalities": ["text"]}
        model = {"id": "test/model", "architecture": arch}
        result = self.normalize_model(model)
        self.assertEqual(result["architecture"], arch)

    def test_normalize_model_created_none_when_missing(self):
        model = {"id": "test/model"}
        result = self.normalize_model(model)
        self.assertIsNone(result.get("created"))


# ---------------------------------------------------------------------------
# Diff detection
# ---------------------------------------------------------------------------

class TestDiffDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from watch_models import detect_changes
        cls.detect_changes = staticmethod(detect_changes)

    def test_detect_changes_with_no_old_catalog(self):
        new_models = [{"model_id": "model/1"}, {"model_id": "model/2"}]
        result = self.detect_changes(None, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(sorted(result["new_models"]), ["model/1", "model/2"])
        self.assertEqual(result["removed_models"], [])
        self.assertEqual(result["total_new"], 2)

    def test_detect_changes_with_added_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}]}
        new_models = [{"model_id": "model/1"}, {"model_id": "model/2"}, {"model_id": "model/3"}]
        result = self.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(sorted(result["new_models"]), ["model/2", "model/3"])
        self.assertEqual(result["removed_models"], [])
        self.assertEqual(result["total_new"], 2)

    def test_detect_changes_with_removed_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}, {"model_id": "model/2"}]}
        new_models = [{"model_id": "model/1"}]
        result = self.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(result["new_models"], [])
        self.assertEqual(result["removed_models"], ["model/2"])
        self.assertEqual(result["total_removed"], 1)

    def test_detect_changes_with_no_changes(self):
        old_catalog = {"models": [{"model_id": "model/1"}, {"model_id": "model/2"}]}
        new_models = [{"model_id": "model/1"}, {"model_id": "model/2"}]
        result = self.detect_changes(old_catalog, new_models)
        self.assertFalse(result["has_changes"])
        self.assertEqual(result["new_models"], [])
        self.assertEqual(result["removed_models"], [])

    def test_detect_changes_handles_non_dict_models(self):
        old_catalog = {"models": [{"model_id": "model/1"}, None, "invalid"]}
        new_models = [{"model_id": "model/1"}, {"model_id": "model/3"}]
        result = self.detect_changes(old_catalog, new_models)
        self.assertTrue(result["has_changes"])
        self.assertEqual(result["new_models"], ["model/3"])
        self.assertEqual(result["removed_models"], [])


# ---------------------------------------------------------------------------
# Aggregate calculations
# ---------------------------------------------------------------------------

class TestAggregateCalculations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from analyze import aggregate_model_metrics
        cls.aggregate = staticmethod(aggregate_model_metrics)

    def _row(self, model_id, status, latency="1.0", tokens="100", run_id="r1", ts="2024-01-01T00:00:00Z"):
        return {
            "model_id": model_id, "status": status,
            "latency_sec": latency, "total_tokens": tokens,
            "cost": "0.0", "run_id": run_id,
            "canonical_family": "test", "display_name": model_id.upper(),
            "timestamp_utc": ts,
        }

    def test_aggregate_with_all_success(self):
        rows = [
            self._row("m1", "success", latency="1.0", run_id="r1"),
            self._row("m1", "success", latency="2.0", run_id="r2", ts="2024-01-02T00:00:00Z"),
        ]
        result = self.aggregate(rows, min_runs=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["total_runs"], 2)
        self.assertEqual(result[0]["successful_runs"], 2)
        self.assertEqual(result[0]["failed_runs"], 0)
        self.assertEqual(result[0]["success_rate"], 1.0)
        self.assertEqual(result[0]["latency_sec_avg"], 1.5)

    def test_aggregate_with_mixed_success_error(self):
        rows = [
            self._row("m1", "success", run_id="r1"),
            self._row("m1", "error", run_id="r2", ts="2024-01-02T00:00:00Z"),
        ]
        result = self.aggregate(rows, min_runs=1)
        self.assertEqual(result[0]["total_runs"], 2)
        self.assertEqual(result[0]["successful_runs"], 1)
        self.assertEqual(result[0]["failed_runs"], 1)
        self.assertEqual(result[0]["success_rate"], 0.5)

    def test_aggregate_excludes_below_min_runs(self):
        rows = [self._row("m1", "success")]
        result = self.aggregate(rows, min_runs=3)
        self.assertEqual(len(result), 0)

    def test_aggregate_handles_empty_rows(self):
        self.assertEqual(self.aggregate([], min_runs=1), [])

    def test_aggregate_sorts_by_success_rate_and_latency(self):
        rows = [
            self._row("m1", "success", latency="5.0"),
            self._row("m2", "success", latency="1.0"),
            self._row("m3", "success", latency="2.0"),
        ]
        result = self.aggregate(rows, min_runs=1)
        self.assertEqual(result[0]["model_id"], "m2")
        self.assertEqual(result[1]["model_id"], "m3")
        self.assertEqual(result[2]["model_id"], "m1")


# ---------------------------------------------------------------------------
# CSV operations
# ---------------------------------------------------------------------------

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
        content = self.csv_path.read_text()
        self.assertIn("col1,col2", content)
        self.assertIn("val1,val2", content)

    def test_append_csv_row_appends_without_duplicate_header(self):
        append_csv_row(self.csv_path, {"col1": "val1", "col2": "val2"})
        append_csv_row(self.csv_path, {"col1": "val3", "col2": "val4"})
        lines = self.csv_path.read_text().splitlines()
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


# ---------------------------------------------------------------------------
# benchmark_single_model — HTTP layer (mocked)
# ---------------------------------------------------------------------------

class TestBenchmarkSingleModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmark import benchmark_single_model
        cls.benchmark_single_model = staticmethod(benchmark_single_model)

    def _mock_response(self, status_code, body=None, headers=None):
        mock = MagicMock()
        mock.status_code = status_code
        mock.headers = headers or {}
        mock.text = json.dumps(body or {})
        mock.json.return_value = body or {}
        return mock

    def _success_body(self):
        return {
            "id": "test-response-id",
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost": 0.0,
        }

    def test_success_response(self):
        with patch("requests.post", return_value=self._mock_response(200, self._success_body())):
            result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_tokens"], 15)
        self.assertEqual(result["finish_reason"], "stop")

    def test_handles_429_returns_error_with_429_in_message(self):
        """After MAX_RETRIES of 429, status must be 'error' and '429' must appear in error_message."""
        mock_resp = self._mock_response(429, headers={"Retry-After": "1"})
        with patch("requests.post", return_value=mock_resp):
            with patch("time.sleep"):  # don't actually sleep
                result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertIn("429", result["error_message"])

    def test_handles_403_non_retryable(self):
        """403 must return immediately without retrying."""
        mock_resp = self._mock_response(403, {"error": "forbidden"})
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertIn("403", result["error_message"])
        self.assertEqual(mock_post.call_count, 1)  # no retries

    def test_handles_401_non_retryable(self):
        mock_resp = self._mock_response(401, {"error": "unauthorized"})
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertEqual(mock_post.call_count, 1)

    def test_handles_500_retries_then_fails(self):
        mock_resp = self._mock_response(500, {"error": "server error"})
        with patch("requests.post", return_value=mock_resp):
            with patch("time.sleep"):
                result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertIn("500", result["error_message"])

    def test_handles_timeout(self):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.Timeout()):
            with patch("time.sleep"):
                result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "timeout")
        self.assertIn("timed out", result["error_message"].lower())

    def test_handles_connection_error(self):
        import requests as req
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("conn refused")):
            with patch("time.sleep"):
                result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertIn("connection", result["error_message"].lower())

    def test_handles_empty_choices(self):
        body = {"choices": [], "usage": {}}
        with patch("requests.post", return_value=self._mock_response(200, body)):
            result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "error")
        self.assertIn("choices", result["error_message"].lower())

    def test_success_after_one_retry(self):
        """First call returns 500, second returns 200 — should succeed."""
        fail_resp = self._mock_response(500)
        ok_resp = self._mock_response(200, self._success_body())
        with patch("requests.post", side_effect=[fail_resp, ok_resp]):
            with patch("time.sleep"):
                result = self.benchmark_single_model("test/model", "prompt", "1.0")
        self.assertEqual(result["status"], "success")


# ---------------------------------------------------------------------------
# Circuit breaker in run_benchmark (mocked)
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmark import CIRCUIT_BREAKER_PAUSE, CIRCUIT_BREAKER_THRESHOLD, run_benchmark
        cls.run_benchmark = staticmethod(run_benchmark)
        cls.threshold = CIRCUIT_BREAKER_THRESHOLD
        cls.base_pause = CIRCUIT_BREAKER_PAUSE  # pause for the first trip

    def _make_models(self, n):
        return [
            {"model_id": f"test/model-{i}", "display_name": f"Model {i}", "canonical_family": "test"}
            for i in range(n)
        ]

    def _429_result(self):
        return {"status": "error", "error_message": "429: Rate limited after 3 attempts", "latency_sec": 1.0}

    def _ok_result(self):
        return {
            "status": "success", "latency_sec": 0.5,
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "cost": 0.0, "finish_reason": "stop", "response_id": "r1",
        }

    def _mock_env(self):
        return {
            "prompts": [{"id": "p1", "text": "What is 2+2?"}],
            "default_prompt_id": "p1",
            "version": "1.0",
        }

    def test_circuit_breaker_triggers_after_threshold(self):
        """After THRESHOLD consecutive 429s, time.sleep must be called with base_pause (first trip)."""
        models = self._make_models(self.threshold)

        with patch("benchmark.benchmark_single_model", return_value=self._429_result()), \
             patch("benchmark.append_csv_row"), \
             patch("benchmark.load_current_catalog", return_value={"catalog_id": "test", "models": []}), \
             patch("benchmark.load_prompts", return_value=self._mock_env()), \
             patch("time.sleep") as mock_sleep:

            self.run_benchmark(models=models, run_id="test-run")

        # First trip fires with base_pause (120s)
        pause_calls = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertIn(
            self.base_pause, pause_calls,
            f"Expected circuit breaker sleep({self.base_pause}s) but got: {pause_calls}",
        )

    def test_circuit_breaker_resets_on_success(self):
        """A successful result resets the consecutive counter — breaker must NOT fire."""
        # Pattern: 2 x 429, then 1 success, then 2 x 429 → never hits threshold of 3 in a row
        results = (
            [self._429_result()] * 2
            + [self._ok_result()]
            + [self._429_result()] * 2
        )
        models = self._make_models(len(results))

        with patch("benchmark.benchmark_single_model", side_effect=results), \
             patch("benchmark.append_csv_row"), \
             patch("benchmark.load_current_catalog", return_value={"catalog_id": "test", "models": []}), \
             patch("benchmark.load_prompts", return_value=self._mock_env()), \
             patch("time.sleep") as mock_sleep:

            self.run_benchmark(models=models, run_id="test-run")

        # No sleep call >= base_pause should have been made (circuit breaker didn't fire)
        breaker_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= self.base_pause]
        self.assertEqual(
            len(breaker_calls), 0,
            f"Circuit breaker must NOT fire when successes break the streak; got sleeps: {breaker_calls}",
        )

    def test_circuit_breaker_does_not_trigger_on_non_429_errors(self):
        """Non-429 errors (e.g. 500, timeout) must NOT increment the 429 counter."""
        non_429 = {"status": "error", "error_message": "HTTP 500: server error", "latency_sec": 1.0}
        models = self._make_models(self.threshold + 1)

        with patch("benchmark.benchmark_single_model", return_value=non_429), \
             patch("benchmark.append_csv_row"), \
             patch("benchmark.load_current_catalog", return_value={"catalog_id": "test", "models": []}), \
             patch("benchmark.load_prompts", return_value=self._mock_env()), \
             patch("time.sleep") as mock_sleep:

            self.run_benchmark(models=models, run_id="test-run")

        breaker_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= self.base_pause]
        self.assertEqual(
            len(breaker_calls), 0,
            f"Circuit breaker must only fire on 429 errors, not generic errors; got: {breaker_calls}",
        )


if __name__ == "__main__":
    unittest.main()
