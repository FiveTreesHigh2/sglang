from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

        def case(
            tokens,
            profile,
            triton_eager,
            flashinfer_eager,
            *,
            triton_graph=None,
            flashinfer_graph=None,
            status="PASS",
        ):
            result = {
                "tokens": tokens,
                "profile": profile,
                "correctness": {"status": status},
                "triton": {"median_ms": triton_eager},
                "flashinfer_sm120_fp8": {"median_ms": flashinfer_eager},
                "speedup_percent": (
                    triton_eager / flashinfer_eager - 1.0
                )
                * 100,
            }
            result["cuda_graph"] = (
                {
                    "triton": {"median_ms": triton_graph},
                    "flashinfer_sm120_fp8": {
                        "median_ms": flashinfer_graph
                    },
                }
                if triton_graph is not None and flashinfer_graph is not None
                else {"status": "NOT_RUN"}
            )
            return result

        cases = [
            case(
                1,
                "uniform",
                1.0,
                2.0,
                triton_graph=1.0,
                flashinfer_graph=1.04,
            ),
            case(
                8,
                "synthetic-skew",
                1.0,
                2.0,
                triton_graph=1.0,
                flashinfer_graph=1.05,
            ),
            case(8192, "uniform", 10.0, 8.9),
            case(16384, "uniform", 20.0, 21.0),
        ]
        decision = bench.select_decision(cases, cuda_graph_passed=True)
        self.assertEqual(decision["status"], "GO")
        self.assertAlmostEqual(decision["decode_regression"], 0.05)
        self.assertGreater(decision["prefill_speedup"], 0.10)

        cases[1] = case(
            8,
            "synthetic-skew",
            1.0,
            2.0,
            triton_graph=1.0,
            flashinfer_graph=1.051,
        )
        self.assertEqual(
            bench.select_decision(cases, cuda_graph_passed=True)["status"],
            "FUNCTIONAL_ONLY",
        )

        cases[0] = case(
            1,
            "uniform",
            1.0,
            2.0,
            triton_graph=1.0,
            flashinfer_graph=1.0,
            status="FAIL",
        )
        self.assertEqual(
            bench.select_decision(cases, cuda_graph_passed=True)["status"],
            "NO_GO",
        )

        with self.assertRaisesRegex(ValueError, "decode CUDA Graph"):
            bench.select_decision(
                [
                    case(1, "uniform", 1.0, 1.0),
                    case(8192, "uniform", 10.0, 8.9),
                ],
                cuda_graph_passed=True,
            )

    def test_select_decision_requires_the_fixed_main_case(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        with self.assertRaisesRegex(ValueError, "tokens=8192.*uniform"):
            bench.select_decision([], cuda_graph_passed=True)

    def test_component_profiles_keep_stable_rollup_schema(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        legacy_detail = {
            "quant1": 0.01,
            "moe_permute": 0.02,
            "scale_pack_gemm1": 0.03,
            "gemm1": 0.10,
            "silu": 0.04,
            "quant2": 0.05,
            "scale_pack_gemm2": 0.06,
            "gemm2": 0.20,
            "unpermute_combine": 0.07,
        }
        fused_detail = {
            "quant1": 0.01,
            "moe_permute": 0.02,
            "scale_pack_gemm1": 0.03,
            "gemm1": 0.10,
            "fused_swiglu_quant_pack_gemm2": 0.08,
            "gemm2": 0.20,
            "unpermute_combine": 0.07,
        }

        legacy = bench.build_component_profile(legacy_detail)
        fused = bench.build_component_profile(fused_detail)

        self.assertEqual(legacy["path"], "legacy")
        self.assertEqual(fused["path"], "fused")
        self.assertAlmostEqual(
            legacy["rollup_ms"]["gemm1_input_prepare"], 0.06
        )
        self.assertAlmostEqual(
            legacy["rollup_ms"]["gemm2_input_prepare"], 0.15
        )
        self.assertAlmostEqual(
            fused["rollup_ms"]["gemm2_input_prepare"], 0.08
        )
        self.assertEqual(
            set(legacy["rollup_ms"]), set(fused["rollup_ms"])
        )

        missing_stage = dict(legacy_detail)
        missing_stage.pop("quant2")
        with self.assertRaisesRegex(ValueError, "legacy/fused schema"):
            bench.build_component_profile(missing_stage)

        mixed_paths = dict(legacy_detail)
        mixed_paths["fused_swiglu_quant_pack_gemm2"] = 0.08
        with self.assertRaisesRegex(ValueError, "legacy/fused schema"):
            bench.build_component_profile(mixed_paths)

    def test_component_trace_requires_exact_order_and_call_counts(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        legacy_trace = [
            "quant1",
            "moe_permute",
            "scale_pack_gemm1",
            "gemm1",
            "silu",
            "quant2",
            "scale_pack_gemm2",
            "gemm2",
            "unpermute_combine",
        ]
        legacy_counts = {
            "quant": 2,
            "pack": 2,
            "gemm": 2,
            "moe_permute": 1,
            "unpermute_combine": 1,
            "silu": 1,
            "fused": 0,
        }
        fused_trace = [
            "quant1",
            "moe_permute",
            "scale_pack_gemm1",
            "gemm1",
            "fused_swiglu_quant_pack_gemm2",
            "gemm2",
            "unpermute_combine",
        ]
        fused_counts = {
            "quant": 1,
            "pack": 1,
            "gemm": 2,
            "moe_permute": 1,
            "unpermute_combine": 1,
            "silu": 0,
            "fused": 1,
        }

        self.assertEqual(
            bench.validate_component_trace(legacy_trace, legacy_counts),
            "legacy",
        )
        self.assertEqual(
            bench.validate_component_trace(fused_trace, fused_counts),
            "fused",
        )

        with self.assertRaisesRegex(ValueError, "component call trace"):
            bench.validate_component_trace(
                [*legacy_trace, "quant1"],
                {**legacy_counts, "quant": 3},
            )
        with self.assertRaisesRegex(ValueError, "component call trace"):
            bench.validate_component_trace(
                [legacy_trace[1], legacy_trace[0], *legacy_trace[2:]],
                legacy_counts,
            )
        with self.assertRaisesRegex(ValueError, "component call counts"):
            bench.validate_component_trace(
                legacy_trace,
                {**legacy_counts, "gemm": 3},
            )

    def test_cuda_graph_helpers_warm_capture_and_time_replay_only(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")

        class FakeStream:
            def __init__(self):
                self.waited_for = []

            def wait_stream(self, stream):
                self.waited_for.append(stream)

        class FakeGraph:
            def __init__(self):
                self.replays = 0

            def replay(self):
                self.replays += 1

        class FakeEvent:
            def __init__(self, enable_timing):
                self.enable_timing = enable_timing

            def record(self):
                pass

            def elapsed_time(self, other):
                return 12.0

        @contextmanager
        def passthrough(_):
            yield

        current_stream = FakeStream()
        fake_cuda = SimpleNamespace(
            Stream=FakeStream,
            current_stream=lambda: current_stream,
            stream=passthrough,
            CUDAGraph=FakeGraph,
            graph=passthrough,
            Event=FakeEvent,
            synchronize=lambda: None,
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        launches = []

        with patch.dict(sys.modules, {"torch": fake_torch}):
            captured = bench.capture_backend_graph(
                lambda: launches.append(len(launches)) or launches[-1]
            )
            latency = bench.time_graph_replay(captured, iterations=4)

        self.assertEqual(launches, [0, 1, 2])
        self.assertEqual(captured.output, 2)
        self.assertEqual(captured.graph.replays, 4)
        self.assertEqual(latency, 3.0)
        self.assertEqual(len(current_stream.waited_for), 1)

    def test_case_result_preserves_eager_graph_and_components(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        component_profile = bench.build_component_profile(
            {
                "quant1": 0.01,
                "moe_permute": 0.02,
                "scale_pack_gemm1": 0.03,
                "gemm1": 0.10,
                "silu": 0.04,
                "quant2": 0.05,
                "scale_pack_gemm2": 0.06,
                "gemm2": 0.20,
                "unpermute_combine": 0.07,
            }
        )

        graph_trials = {
            "triton": [0.7, 0.6, 0.5],
            "flashinfer_sm120_fp8": [0.55, 0.50, 0.45],
        }
        result = bench.build_case_result(
            tokens=8,
            top_k=8,
            profile="uniform",
            correctness={"status": "PASS", "calc_diff": 0.001},
            triton_trials=[1.1, 1.0, 0.9],
            flashinfer_trials=[0.8, 0.9, 0.7],
            component_profile=component_profile,
            cuda_graph_trials=graph_trials,
            cutlass_trials=[0.95, 0.85, 0.9],
        )

        self.assertEqual(result["routed_rows"], 64)
        self.assertEqual(result["triton"]["median_ms"], 1.0)
        self.assertEqual(result["triton"]["trials_ms"], [1.1, 1.0, 0.9])
        self.assertEqual(
            result["flashinfer_sm120_fp8"]["median_ms"], 0.8
        )
        self.assertEqual(result["cutlass"]["median_ms"], 0.9)
        self.assertAlmostEqual(result["speedup_percent"], 25.0)
        self.assertEqual(result["components"], component_profile)
        self.assertEqual(result["cuda_graph"]["triton"]["median_ms"], 0.6)
        self.assertEqual(
            result["cuda_graph"]["flashinfer_sm120_fp8"]["median_ms"],
            0.50,
        )

        prefill = bench.build_case_result(
            tokens=8192,
            top_k=8,
            profile="uniform",
            correctness={"status": "PASS"},
            triton_trials=[1.0],
            flashinfer_trials=[0.9],
            component_profile=component_profile,
        )
        self.assertEqual(prefill["cuda_graph"], {"status": "NOT_RUN"})

        with self.assertRaisesRegex(ValueError, "decode CUDA Graph"):
            bench.build_case_result(
                tokens=1,
                top_k=8,
                profile="uniform",
                correctness={"status": "PASS"},
                triton_trials=[1.0],
                flashinfer_trials=[0.9],
                component_profile=component_profile,
            )
        with self.assertRaisesRegex(ValueError, "prefill.*CUDA Graph"):
            bench.build_case_result(
                tokens=8192,
                top_k=8,
                profile="uniform",
                correctness={"status": "PASS"},
                triton_trials=[1.0],
                flashinfer_trials=[0.9],
                component_profile=component_profile,
                cuda_graph_trials=graph_trials,
            )

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
        self.assertEqual(payload["schema_version"], 3)

    def test_case_result_distinguishes_skipped_cutlass(self) -> None:
        bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
        component_profile = bench.build_component_profile(
            {
                key: 0.1
                for key in (
                    bench.COMMON_COMPONENT_DETAIL_KEYS
                    | bench.LEGACY_COMPONENT_EXTRA_KEYS
                )
            }
        )

        result = bench.build_case_result(
            tokens=128,
            top_k=8,
            profile="uniform",
            correctness={"status": "PASS"},
            triton_trials=[1.0],
            flashinfer_trials=[0.9],
            component_profile=component_profile,
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
