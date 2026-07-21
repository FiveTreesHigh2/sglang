from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path


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


class TestPro5000Stage2(unittest.TestCase):
    def test_stage_2_decision_contract(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        self.assertEqual(
            bench.decide(
                correct=True,
                graph=True,
                prefill_speedup=0.12,
                decode_regression=0.03,
            ),
            "GO",
        )
        self.assertEqual(
            bench.decide(
                correct=True,
                graph=True,
                prefill_speedup=0.05,
                decode_regression=0.08,
            ),
            "FUNCTIONAL_ONLY",
        )
        self.assertEqual(
            bench.decide(
                correct=False,
                graph=True,
                prefill_speedup=0.30,
                decode_regression=0.0,
            ),
            "NO_GO",
        )
        self.assertEqual(
            bench.decide(
                correct=True,
                graph=False,
                prefill_speedup=0.30,
                decode_regression=0.0,
            ),
            "NO_GO",
        )

    def test_cli_defaults_match_approved_stage_2_workload(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        args = bench.parse_args([])
        self.assertEqual(args.tokens, [1, 8, 128, 8192, 16384])
        self.assertEqual(args.top_k, 8)
        self.assertEqual(args.profiles, ["uniform", "synthetic-skew"])
        self.assertEqual(args.warmup, 10)
        self.assertEqual(args.trials, 5)
        self.assertEqual(args.iterations, 100)
        self.assertIsNone(args.output_json)
        self.assertFalse(args.check_cuda_graph)
        self.assertFalse(args.cutlass_preflight)

        checked = bench.parse_args(
            ["--check-cuda-graph", "--cutlass-preflight"]
        )
        self.assertTrue(checked.check_cuda_graph)
        self.assertTrue(checked.cutlass_preflight)

    def test_uniform_routing_is_balanced_and_valid_topk(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        rows = bench.build_routing_rows(
            tokens=257,
            top_k=8,
            num_experts=256,
            profile="uniform",
        )
        counts = [0] * 256
        for row in rows:
            self.assertEqual(len(row), 8)
            self.assertEqual(len(set(row)), 8)
            for expert in row:
                counts[expert] += 1
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_synthetic_skew_routing_is_deterministic_and_valid(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        first = bench.build_routing_rows(
            tokens=8192,
            top_k=8,
            num_experts=256,
            profile="synthetic-skew",
        )
        second = bench.build_routing_rows(
            tokens=8192,
            top_k=8,
            num_experts=256,
            profile="synthetic-skew",
        )
        self.assertEqual(first, second)
        self.assertTrue(all(len(row) == len(set(row)) == 8 for row in first))

        counts = [0] * 256
        for row in first:
            for expert in row:
                counts[expert] += 1
        self.assertGreater(max(counts), 4 * (sum(counts) / len(counts)))

    def test_select_decision_uses_fixed_prefill_and_decode_cases(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        def case(tokens, profile, triton_ms, flashinfer_ms, status="PASS"):
            return {
                "tokens": tokens,
                "profile": profile,
                "correctness": {"status": status},
                "triton": {"median_ms": triton_ms},
                "flashinfer_sm120_fp8": {"median_ms": flashinfer_ms},
                "speedup_percent": (triton_ms / flashinfer_ms - 1.0) * 100,
            }

        cases = [
            case(1, "uniform", 1.0, 1.04),
            case(8, "synthetic-skew", 1.0, 1.05),
            case(8192, "uniform", 10.0, 8.9),
            case(16384, "uniform", 20.0, 21.0),
        ]
        decision = bench.select_decision(cases, cuda_graph_passed=True)
        self.assertEqual(decision["status"], "GO")
        self.assertAlmostEqual(decision["decode_regression"], 0.05)
        self.assertGreater(decision["prefill_speedup"], 0.10)

        cases[1] = case(8, "synthetic-skew", 1.0, 1.051)
        self.assertEqual(
            bench.select_decision(cases, cuda_graph_passed=True)["status"],
            "FUNCTIONAL_ONLY",
        )

        cases[0] = case(1, "uniform", 1.0, 1.0, status="FAIL")
        self.assertEqual(
            bench.select_decision(cases, cuda_graph_passed=True)["status"],
            "NO_GO",
        )

    def test_select_decision_requires_the_fixed_main_case(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        with self.assertRaisesRegex(ValueError, "tokens=8192.*uniform"):
            bench.select_decision([], cuda_graph_passed=True)

    def test_case_result_preserves_trials_and_component_schema(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        components = {
            "routing_quant_pack": 0.1,
            "gemm1": 0.2,
            "swiglu_quant": 0.3,
            "scale_layout_gemm2": 0.4,
            "gemm2": 0.5,
            "unpermute_combine": 0.6,
        }

        result = bench.build_case_result(
            tokens=8192,
            top_k=8,
            profile="uniform",
            correctness={"status": "PASS", "calc_diff": 0.001},
            triton_trials=[1.1, 1.0, 0.9],
            flashinfer_trials=[0.8, 0.9, 0.7],
            components_ms=components,
            cutlass_trials=[0.95, 0.85, 0.9],
        )

        self.assertEqual(result["routed_rows"], 65536)
        self.assertEqual(result["triton"]["median_ms"], 1.0)
        self.assertEqual(result["triton"]["trials_ms"], [1.1, 1.0, 0.9])
        self.assertEqual(
            result["flashinfer_sm120_fp8"]["median_ms"], 0.8
        )
        self.assertEqual(result["cutlass"]["median_ms"], 0.9)
        self.assertAlmostEqual(result["speedup_percent"], 25.0)
        self.assertEqual(result["components_ms"], components)

    def test_empty_payload_has_reproducibility_fields(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        payload = bench.empty_result_payload(["benchmark.py", "--tokens", "8"])
        self.assertGreaterEqual(
            set(payload),
            {
                "schema_version",
                "timestamp_utc",
                "command",
                "git",
                "environment",
                "parameters",
                "cutlass_preflight",
                "cuda_graph",
                "cases",
                "decision",
                "status",
            },
        )
        self.assertEqual(payload["command"], ["benchmark.py", "--tokens", "8"])
        self.assertEqual(payload["status"], "running")

    def test_case_result_distinguishes_skipped_cutlass(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        components = {key: 0.1 for key in bench.COMPONENT_KEYS}

        result = bench.build_case_result(
            tokens=8,
            top_k=8,
            profile="uniform",
            correctness={"status": "PASS"},
            triton_trials=[1.0],
            flashinfer_trials=[0.9],
            components_ms=components,
            cutlass_status="SKIPPED",
        )

        self.assertEqual(result["cutlass"], {"status": "SKIPPED"})

    def test_cli_rejects_non_positive_work_sizes(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        for argv in (
            ["--tokens", "0"],
            ["--top-k", "0"],
            ["--warmup", "0"],
            ["--trials", "0"],
            ["--iterations", "0"],
        ):
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                bench.parse_args(argv)

    def test_benchmark_uses_full_production_runners(self) -> None:
        source = (
            PRO5000_SCRIPTS / "benchmark_flashinfer_sm120_fp8_runner.py"
        ).read_text()
        self.assertIn("fused_experts_none_to_flashinfer_sm120_fp8", source)
        self.assertIn("triton_fused_moe.fused_experts", source)
        self.assertIn("cutlass_fused_experts_fp8", source)
        self.assertNotIn("FLASHINFER_DISABLE_JIT=1", source)

    def test_wrapper_is_non_mutating_and_enables_required_checks(self) -> None:
        source = (
            PRO5000_SCRIPTS / "run_stage_2_benchmark.sh"
        ).read_text()
        self.assertIn("benchmark_flashinfer_sm120_fp8_runner.py", source)
        self.assertIn("--check-cuda-graph", source)
        self.assertIn("--cutlass-preflight", source)
        self.assertNotIn("uv pip install", source)
        self.assertNotIn("python -m pip", source)
        self.assertNotIn("nvidia-smi -lgc", source)
        self.assertNotIn("nvidia-smi --lock-gpu-clocks", source)


if __name__ == "__main__":
    unittest.main()
