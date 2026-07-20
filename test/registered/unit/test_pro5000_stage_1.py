from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
PRO5000_SCRIPTS = REPO_ROOT / "scripts" / "pro5000"


@contextmanager
def pro5000_scripts_on_path():
    sys.path.insert(0, str(PRO5000_SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(PRO5000_SCRIPTS))


def load_script(filename: str):
    path = PRO5000_SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pro5000_scripts_on_path():
        spec.loader.exec_module(module)
    return module


class TestPro5000Stage1(unittest.TestCase):
    def test_flashinfer_scale_copy_plan_preserves_empty_experts(self) -> None:
        smoke = load_script("flashinfer_sm120_fp8_smoke.py")
        self.assertEqual(
            smoke.build_scale_copy_plan([0, 0, 1, 9, 9, 12]),
            [
                (0, 0, 0),
                (0, 1, 0),
                (1, 9, 4),
                (9, 9, 16),
                (9, 12, 20),
            ],
        )

    def test_uniform_profile_has_exact_total(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        rows = bench.build_uniform_rows(num_experts=256, cum_m=65536)
        self.assertEqual(rows, [256] * 256)

    def test_synthetic_skew_is_deterministic_and_stressful(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        first = bench.build_synthetic_skew_rows(
            256, 65536, seed=42, block_size=64
        )
        second = bench.build_synthetic_skew_rows(
            256, 65536, seed=42, block_size=64
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual(sum(first), 65536)
        self.assertEqual(min(first), 0)
        self.assertTrue(any(0 < rows < 64 for rows in first))
        self.assertGreater(max(first), 4 * (65536 / 256))

    def test_rows_json_accepts_array_or_object(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        rows = [1] * 256
        with tempfile.TemporaryDirectory() as temp_dir:
            array_path = Path(temp_dir) / "array.json"
            object_path = Path(temp_dir) / "object.json"
            array_path.write_text(json.dumps(rows))
            object_path.write_text(json.dumps({"rows_per_expert": rows}))
            self.assertEqual(
                bench.load_rows_per_expert(array_path, 256, 256), rows
            )
            self.assertEqual(
                bench.load_rows_per_expert(object_path, 256, 256), rows
            )

    def test_rows_json_rejects_invalid_inputs(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.json"
            invalid_rows = (
                [1] * 255,
                [-1] + [1] * 255,
                [1] * 256,
            )
            expected_totals = (256, 255, 257)
            for rows, expected_total in zip(invalid_rows, expected_totals):
                with self.subTest(rows=len(rows), expected_total=expected_total):
                    path.write_text(json.dumps(rows))
                    with self.assertRaises(ValueError):
                        bench.load_rows_per_expert(path, 256, expected_total)

    def test_latency_summary_uses_median_and_relative_spread(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        summary = bench.summarize_latencies([1.0, 1.1, 0.9, 1.0, 1.0])
        self.assertEqual(summary["min_ms"], 0.9)
        self.assertEqual(summary["median_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 1.1)
        self.assertAlmostEqual(summary["relative_spread"], 0.2)

    def test_default_boost_decision_boundaries(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        cases = (
            (0.31, 0.20, 0.04, 0.04, "GO"),
            (0.30, 0.20, 0.05, 0.05, "GO"),
            (0.30, 0.20, 0.06, 0.04, "NEEDS_LOCKED_RERUN"),
            (0.31, 0.19, 0.02, 0.02, "NEEDS_LOCKED_RERUN"),
            (0.20, 0.20, 0.02, 0.02, "NEEDS_LOCKED_RERUN"),
            (0.15, 0.20, 0.02, 0.02, "NEEDS_LOCKED_RERUN"),
            (0.149, 0.20, 0.02, 0.02, "NO_GO"),
        )
        for speedup, minimum, triton_spread, flashinfer_spread, expected in cases:
            with self.subTest(speedup=speedup, expected=expected):
                decision = bench.decide_default_boost(
                    speedup=speedup,
                    min_trial_speedup=minimum,
                    triton_relative_spread=triton_spread,
                    flashinfer_relative_spread=flashinfer_spread,
                    environment_stable=True,
                    correctness_passed=True,
                )
                self.assertEqual(decision["status"], expected)

    def test_decision_rejects_bad_correctness_or_environment(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        bad_correctness = bench.decide_default_boost(
            speedup=0.50,
            min_trial_speedup=0.50,
            triton_relative_spread=0.01,
            flashinfer_relative_spread=0.01,
            environment_stable=True,
            correctness_passed=False,
        )
        unstable_environment = bench.decide_default_boost(
            speedup=0.50,
            min_trial_speedup=0.50,
            triton_relative_spread=0.01,
            flashinfer_relative_spread=0.01,
            environment_stable=False,
            correctness_passed=True,
        )
        self.assertEqual(bad_correctness["status"], "ERROR")
        self.assertEqual(
            unstable_environment["status"], "NEEDS_LOCKED_RERUN"
        )

    def test_locked_decision_uses_twenty_percent_gate(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        for speedup, expected in ((0.20, "GO"), (0.199, "NO_GO")):
            with self.subTest(speedup=speedup):
                decision = bench.decide_locked(
                    speedup=speedup,
                    triton_relative_spread=0.05,
                    flashinfer_relative_spread=0.05,
                    correctness_passed=True,
                )
                self.assertEqual(decision["status"], expected)
        unstable = bench.decide_locked(
            speedup=0.30,
            triton_relative_spread=0.051,
            flashinfer_relative_spread=0.01,
            correctness_passed=True,
        )
        self.assertEqual(unstable["status"], "NEEDS_LOCKED_RERUN")

    def test_cli_defaults_match_approved_design(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        args = bench.parse_args([])
        self.assertEqual(args.operations, ["gemm1", "gemm2"])
        self.assertEqual(args.profiles, ["uniform", "synthetic-skew"])
        self.assertEqual(args.cum_m, [65536, 131072])
        self.assertEqual(args.warmup, 20)
        self.assertEqual(args.iterations, 100)
        self.assertEqual(args.trials, 5)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.clock_mode, "default")

    def test_cli_rejects_json_and_generated_profiles_from_sys_argv(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        argv = [
            "benchmark_flashinfer_sm120_fp8_moe.py",
            "--rows-per-expert-json",
            "rows.json",
            "--cum-m",
            "256",
            "--profiles",
            "uniform",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            bench.parse_args()


if __name__ == "__main__":
    unittest.main()
