from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
    previous = sys.modules.get(path.stem)
    sys.modules[path.stem] = module
    try:
        with pro5000_scripts_on_path():
            spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(path.stem, None)
        else:
            sys.modules[path.stem] = previous
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
                    environment_stable=True,
                    correctness_passed=True,
                )
                self.assertEqual(decision["status"], expected)
        unstable = bench.decide_locked(
            speedup=0.30,
            triton_relative_spread=0.051,
            flashinfer_relative_spread=0.01,
            environment_stable=True,
            correctness_passed=True,
        )
        self.assertEqual(unstable["status"], "NEEDS_LOCKED_RERUN")

    def test_locked_decision_rejects_missing_gpu_evidence(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        target = {
            "operation": "gemm1",
            "profile": "uniform",
            "cum_m": 65536,
            "speedup": 0.40,
            "min_trial_speedup": 0.40,
            "correctness": {"passed": True},
            "latency": {
                "triton": {"relative_spread": 0.01},
                "flashinfer": {"relative_spread": 0.01},
            },
            "gpu_stability": {"stable": False},
        }
        decision = bench._select_decision([target], "locked")
        self.assertEqual(decision["status"], "NEEDS_LOCKED_RERUN")

    def test_nvidia_smi_gpu_id_is_resolved_from_current_process(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        response = {
            "returncode": 0,
            "stdout": "999, GPU-other\n1234, GPU-current\n",
            "stderr": "",
        }
        with mock.patch.object(bench, "_run_command", return_value=response):
            self.assertEqual(
                bench.resolve_nvidia_smi_gpu_id(pid=1234), "GPU-current"
            )

    def test_unresolved_gpu_identity_does_not_guess(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        with mock.patch.object(bench, "_run_command") as run_command:
            sample = bench.sample_gpu_state(
                None, identity_error="no GPU UUID for current process"
            )
        self.assertFalse(sample["ok"])
        self.assertIn("no GPU UUID", sample["identity_error"])
        run_command.assert_not_called()

    def test_environment_contract_rejects_version_or_checkout_drift(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        environment = {
            "python_version_info": [3, 12, 3],
            "torch_version": "2.11.0",
            "torch_cuda": "13.0",
            "gpu": "NVIDIA RTX PRO 5000 72GB Blackwell",
            "compute_capability": [12, 0],
            "packages": {
                "flashinfer-python": "0.6.15.dev20260716",
                "flashinfer-jit-cache": None,
                "nvidia-cutlass-dsl": "4.5.2",
                "sglang-kernel": "0.4.4",
            },
            "nvcc": {
                "returncode": 0,
                "stdout": "Cuda compilation tools, release 13.0, V13.0.48",
            },
            "sglang_file": "/repo/python/sglang/__init__.py",
        }
        bench.validate_environment_contract(
            environment, expected_repo=Path("/repo")
        )

        environment["packages"]["flashinfer-jit-cache"] = "0.6.15"
        with self.assertRaisesRegex(RuntimeError, "flashinfer-jit-cache"):
            bench.validate_environment_contract(
                environment, expected_repo=Path("/repo")
            )

        environment["packages"]["flashinfer-jit-cache"] = None
        environment["sglang_file"] = "/somewhere/site-packages/sglang/__init__.py"
        with self.assertRaisesRegex(RuntimeError, "checkout"):
            bench.validate_environment_contract(
                environment, expected_repo=Path("/repo")
            )

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
        profile_forms = (("--profiles", "uniform"), ("--profiles=uniform",))
        for profile_args in profile_forms:
            argv = [
                "benchmark_flashinfer_sm120_fp8_moe.py",
                "--rows-per-expert-json",
                "rows.json",
                "--cum-m",
                "256",
                *profile_args,
            ]
            with (
                self.subTest(profile_args=profile_args),
                mock.patch.object(sys, "argv", argv),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                bench.parse_args()

    def test_no_output_error_path_emits_structured_json(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                bench, "collect_environment", side_effect=RuntimeError("boom")
            ),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            status = bench.main(
                [
                    "--operations",
                    "gemm1",
                    "--profiles",
                    "uniform",
                    "--cum-m",
                    "4096",
                    "--warmup",
                    "1",
                    "--iterations",
                    "1",
                    "--trials",
                    "1",
                ]
            )
        self.assertEqual(status, 1)
        output = stdout.getvalue()
        self.assertIn("STAGE_1_JSON_BEGIN", output)
        self.assertIn('"status": "error"', output)
        self.assertIn("STAGE_1_JSON_END", output)

    def test_benchmark_uses_low_level_kernels_without_quant_wrapper(self) -> None:
        source = (
            PRO5000_SCRIPTS / "benchmark_flashinfer_sm120_fp8_moe.py"
        ).read_text()
        self.assertIn("moe_gemm_fp8_nt_groupwise(", source)
        self.assertIn("fused_moe_kernel[grid](", source)
        self.assertNotIn("invoke_fused_moe_kernel(", source)
        self.assertIn("out=case.flashinfer_output", source)
        self.assertNotIn("FLASHINFER_DISABLE_JIT=1", source)

    def test_benchmark_records_both_scale_layouts(self) -> None:
        source = (
            PRO5000_SCRIPTS / "benchmark_flashinfer_sm120_fp8_moe.py"
        ).read_text()
        self.assertIn("a_scale_row_major", source)
        self.assertIn("a_scale_flashinfer", source)
        self.assertIn("b_scale_triton", source)
        self.assertIn("b_scale_flashinfer", source)

    def test_trial_order_alternates_backends(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        self.assertEqual(bench.trial_backend_order(0), ("triton", "flashinfer"))
        self.assertEqual(bench.trial_backend_order(1), ("flashinfer", "triton"))

    def test_result_schema_has_reproducibility_fields(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
        payload = bench.empty_result_payload(["benchmark.py"])
        self.assertGreaterEqual(
            set(payload),
            {
                "schema_version",
                "timestamp_utc",
                "command",
                "git",
                "environment",
                "parameters",
                "cases",
                "decision",
                "status",
            },
        )

    def test_stage_1_wrapper_is_safe_and_uses_existing_venv(self) -> None:
        script = PRO5000_SCRIPTS / "run_stage_1_benchmark.sh"
        completed = subprocess.run(
            ["bash", "-n", str(script)], text=True, capture_output=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        content = script.read_text()
        self.assertIn('${PRO5000_ROOT}/.venv/bin/python3', content)
        self.assertIn('uv pip check --python "${PYTHON}"', content)
        self.assertIn('uv pip freeze --python "${PYTHON}"', content)
        self.assertIn("benchmark_flashinfer_sm120_fp8_moe.py", content)
        self.assertIn("FLASHINFER_WORKSPACE_BASE", content)
        self.assertNotIn("FLASHINFER_DISABLE_JIT=1", content)
        self.assertNotIn("nvidia-smi -lgc", content)
        self.assertNotIn("nvidia-smi -rgc", content)
        self.assertIsNone(re.search(r"(?m)^\s*pip\s", content))
        self.assertIsNone(re.search(r"python3?\s+-m\s+pip", content))
        self.assertNotIn("rm -rf", content)

    def test_readme_documents_stage_1_without_mandatory_clock_lock(self) -> None:
        content = (PRO5000_SCRIPTS / "README.md").read_text()
        self.assertIn("## Stage 1：FP8 MoE kernel 微基准", content)
        self.assertIn("bash scripts/pro5000/run_stage_1_benchmark.sh", content)
        self.assertIn("默认 boost 首测", content)
        self.assertIn("NEEDS_LOCKED_RERUN", content)
        self.assertIn("1732 MHz", content)
        self.assertIn("条件锁频", content)
        self.assertIn("uv pip --python", content)
        stage_1 = content.split("## Stage 1：FP8 MoE kernel 微基准", 1)[1]
        self.assertNotIn("FLASHINFER_DISABLE_JIT=1", stage_1)


if __name__ == "__main__":
    unittest.main()
