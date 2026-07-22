import argparse
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py"
INPUT_LENGTHS = (4096, 6144, 14336, 30720, 63488)
SEEDS = (17, 29, 43)
NUM_PROMPTS = 100
FIXED_SERVER_FIELDS = (
    "model_path",
    "served_model_name",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "tp_size",
    "dp_size",
    "ep_size",
    "pp_size",
    "disable_radix_cache",
    "mem_fraction_static",
    "attention_backend",
    "prefill_attention_backend",
    "reasoning_parser",
    "tool_call_parser",
    "chunked_prefill_size",
    "fp8_gemm_runner_backend",
    "moe_runner_backend",
    "enable_metrics",
)


def load_bench():
    spec = importlib.util.spec_from_file_location("pro5000_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench():
    return load_bench()


def server_info(backend="triton"):
    return {
        "served_model_name": "qwen35-fp8",
        "enable_metrics": True,
        "moe_runner_backend": backend,
        "fp8_gemm_runner_backend": "flashinfer_cutlass",
        "chunked_prefill_size": 8192,
        "tp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "pp_size": 1,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.9,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "attention_backend": "flashinfer",
        "prefill_attention_backend": "flashinfer",
        "model_path": "/models/qwen",
        "dtype": "auto",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "version": "0.5.14",
    }


def run_key_args():
    return {
        "backend": "triton",
        "commit": "a" * 40,
        "input_length": 4096,
        "num_prompts": NUM_PROMPTS,
        "seed": 17,
        "server_args_hash": "b" * 64,
        "a1_mode": "not_applicable",
        "flashinfer_artifact_sha256": "f" * 64,
        "gpu_uuid": "GPU-test",
        "gpu_frequency_strategy": "default-unlocked",
        "runtime_fingerprint": {
            "flashinfer_version": "0.6.15.dev20260716",
            "torch_version": "2.11.0",
            "torch_cuda_version": "13.0",
            "sglang_kernel_version": "0.4.4",
            "python_executable": "/venv/bin/python3",
            "dense_fp8_backend": "flashinfer_cutlass",
            "chunked_prefill_size": 8192,
        },
    }


@pytest.mark.parametrize(
    ("dimension", "changed_value"),
    (
        ("backend", "flashinfer_sm120_fp8"),
        ("commit", "c" * 40),
        ("input_length", 6144),
        ("num_prompts", 99),
        ("seed", 29),
        ("server_args_hash", "d" * 64),
        ("a1_mode", "fused"),
        ("flashinfer_artifact_sha256", "e" * 64),
        ("gpu_uuid", "GPU-other"),
        ("gpu_frequency_strategy", "application-clocks-locked"),
        (
            "runtime_fingerprint",
            {
                **run_key_args()["runtime_fingerprint"],
                "torch_cuda_version": "13.1",
            },
        ),
    ),
)
def test_build_run_key_changes_for_each_contract_dimension(
    bench, dimension, changed_value
):
    baseline = run_key_args()
    changed = {**baseline, dimension: changed_value}

    assert bench.build_run_key(**baseline) != bench.build_run_key(**changed)


def test_defaults_and_server_snapshot_include_all_fixed_fields(bench):
    assert bench.INPUT_LENGTHS == INPUT_LENGTHS
    assert bench.SEEDS == SEEDS
    assert bench.FIXED_SERVER_FIELDS == FIXED_SERVER_FIELDS

    info = server_info()
    snapshot = bench.verify_server_info(info, "triton")
    assert snapshot == {field: info[field] for field in FIXED_SERVER_FIELDS}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("moe_runner_backend", "flashinfer_sm120_fp8"),
        ("fp8_gemm_runner_backend", "triton"),
        ("chunked_prefill_size", 4096),
        ("tp_size", 2),
        ("dp_size", 2),
        ("ep_size", 2),
        ("pp_size", 2),
        ("disable_radix_cache", False),
        ("enable_metrics", False),
        ("mem_fraction_static", 0.8),
        ("reasoning_parser", "none"),
        ("tool_call_parser", "none"),
        ("model_path", ""),
        ("served_model_name", ""),
    ),
)
def test_server_info_rejects_each_required_fixed_value(bench, field, value):
    bad = server_info()
    bad[field] = value

    with pytest.raises(ValueError, match=field):
        bench.verify_server_info(bad, "triton")


def manifest(backend, ratios):
    cases = []
    for input_length in INPUT_LENGTHS:
        for seed, ratio in zip(SEEDS, ratios[input_length]):
            base = 1000.0
            cases.append(
                {
                    "input_length": input_length,
                    "seed": seed,
                    "completed": NUM_PROMPTS,
                    "total_input_tokens": input_length * NUM_PROMPTS,
                    "input_throughput": base if backend == "triton" else base * ratio,
                    "median_ttft_ms": 1.0,
                }
            )
    return {
        "schema_version": 1,
        "metadata": {
            "moe_backend": backend,
            "pairing_hash": "c" * 64,
            "sglang_commit": "d" * 40,
            "flashinfer_artifact_sha256": "f" * 64,
            "flashinfer_version": "0.6.15.dev20260716",
            "torch_version": "2.11.0",
            "torch_cuda_version": "13.0",
            "sglang_kernel_version": "0.4.4",
            "python_executable": "/venv/bin/python3",
            "gpu_uuid": "GPU-test",
            "gpu_frequency_strategy": "default-unlocked",
            "dense_fp8_backend": "flashinfer_cutlass",
            "num_prompts": NUM_PROMPTS,
        },
        "cases": cases,
    }


def passing_ratios():
    return {
        4096: (1.10, 1.10, 1.10),
        **{length: (1.01, 1.02, 1.03) for length in INPUT_LENGTHS if length != 4096},
    }


def compare(bench, ratios):
    return bench.compare_manifests(
        manifest("triton", ratios),
        manifest("flashinfer_sm120_fp8", ratios),
    )


@pytest.mark.parametrize(
    ("speedups", "accepted"),
    (
        ((0.10, 0.10, 0.10), True),
        ((0.099999, 0.099999, 0.20), False),
    ),
)
def test_4096_speedup_gate_accepts_exact_boundary_and_rejects_lower_median(
    bench, speedups, accepted
):
    assert bench.evaluate_speedup_gate(4096, speedups) is accepted


def test_compare_uses_speedup_gate_for_each_input_length(bench, monkeypatch):
    calls = []

    def gate(input_length, speedups):
        calls.append((input_length, tuple(speedups)))
        return input_length != 4096

    monkeypatch.setattr(bench, "evaluate_speedup_gate", gate)
    result = compare(bench, passing_ratios())

    assert result["decision"] == "FUNCTIONAL_ONLY"
    assert [input_length for input_length, _ in calls] == list(INPUT_LENGTHS)
    assert all(len(speedups) == len(SEEDS) for _, speedups in calls)


def test_compare_rejects_4096_median_below_ten_percent(bench):
    ratios = passing_ratios()
    ratios[4096] = (1.099, 1.099, 1.11)

    assert compare(bench, ratios)["decision"] == "FUNCTIONAL_ONLY"


@pytest.mark.parametrize("seed_index", range(len(SEEDS)))
def test_compare_rejects_4096_when_any_seed_regresses(bench, seed_index):
    ratios = passing_ratios()
    samples = list(ratios[4096])
    samples[seed_index] = 0.99
    ratios[4096] = tuple(samples)

    assert compare(bench, ratios)["decision"] == "FUNCTIONAL_ONLY"


@pytest.mark.parametrize(
    "input_length, seed_index, ratio",
    product(INPUT_LENGTHS[1:], range(len(SEEDS)), (1.0, 0.99)),
)
def test_compare_rejects_zero_or_negative_improvement_for_every_non4096_seed(
    bench, input_length, seed_index, ratio
):
    ratios = passing_ratios()
    samples = list(ratios[input_length])
    samples[seed_index] = ratio
    ratios[input_length] = tuple(samples)

    assert compare(bench, ratios)["decision"] == "FUNCTIONAL_ONLY"


@pytest.mark.parametrize("side", ("triton", "flashinfer"))
def test_compare_rejects_incomplete_cases_on_either_manifest_side(bench, side):
    ratios = passing_ratios()
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    target = triton if side == "triton" else flashinfer
    target["cases"][0]["completed"] = NUM_PROMPTS - 1

    with pytest.raises(ValueError, match="completed"):
        bench.compare_manifests(triton, flashinfer)


@pytest.mark.parametrize("side", ("triton", "flashinfer"))
def test_compare_rejects_missing_input_length_seed_pair_on_either_side(bench, side):
    ratios = passing_ratios()
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    target = triton if side == "triton" else flashinfer
    target["cases"].pop()

    with pytest.raises(ValueError, match="every input-length/seed pair"):
        bench.compare_manifests(triton, flashinfer)


@pytest.mark.parametrize("side", ("triton", "flashinfer"))
def test_compare_rejects_duplicate_input_length_seed_pair_on_either_side(bench, side):
    ratios = passing_ratios()
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    target = triton if side == "triton" else flashinfer
    target["cases"].append(dict(target["cases"][0]))

    with pytest.raises(ValueError, match="duplicate input-length/seed pair"):
        bench.compare_manifests(triton, flashinfer)


def test_compare_rejects_token_count_mismatch_between_paired_cases(bench):
    ratios = passing_ratios()
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    flashinfer["cases"][0]["total_input_tokens"] -= 1

    with pytest.raises(ValueError, match="total_input_tokens"):
        bench.compare_manifests(triton, flashinfer)


@pytest.mark.parametrize("side", ("triton", "flashinfer"))
def test_compare_rejects_unsupported_schema_on_either_manifest_side(bench, side):
    ratios = passing_ratios()
    valid = manifest("triton", ratios)
    invalid = {"schema_version": 2}
    triton, flashinfer = (
        (invalid, valid) if side == "triton" else (valid, invalid)
    )

    with pytest.raises(ValueError, match="schema_version"):
        bench.compare_manifests(triton, flashinfer)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("pairing_hash", "e" * 64),
        ("sglang_commit", "e" * 40),
        ("flashinfer_artifact_sha256", "e" * 64),
        ("flashinfer_version", "0.6.16"),
        ("torch_version", "2.11.1"),
        ("torch_cuda_version", "13.1"),
        ("sglang_kernel_version", "0.4.5"),
        ("python_executable", "/other-venv/bin/python3"),
        ("gpu_uuid", "GPU-other"),
        ("gpu_frequency_strategy", "application-clocks-locked"),
        ("dense_fp8_backend", "triton"),
        ("num_prompts", NUM_PROMPTS - 1),
    ),
)
def test_compare_rejects_each_paired_metadata_drift(bench, field, changed_value):
    ratios = passing_ratios()
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    flashinfer["metadata"][field] = changed_value

    with pytest.raises(ValueError, match=field):
        bench.compare_manifests(triton, flashinfer)


def capture_args(tmp_path, *, expected_backend="triton", a1_mode="not_applicable"):
    return argparse.Namespace(
        host="127.0.0.1",
        port=30000,
        expected_backend=expected_backend,
        a1_mode=a1_mode,
        server_log=None,
        repo=tmp_path,
        dataset_path=tmp_path / "random.json",
        flashinfer_artifact_sha256="f" * 64,
        gpu_frequency_strategy="default-unlocked",
        output=tmp_path / "manifest.json",
    )


def install_capture_environment(bench, monkeypatch, commands):
    commit = "d" * 40
    monkeypatch.setattr(bench, "fetch_server_info", lambda host, port: server_info())
    monkeypatch.setattr(bench, "git_commit", lambda repo: commit)
    monkeypatch.setattr(bench, "query_single_gpu_uuid", lambda: "GPU-test")
    monkeypatch.setattr(
        bench.importlib.metadata,
        "version",
        lambda package: {
            "flashinfer-python": "0.6.15.dev20260716",
            "sglang-kernel": "0.4.4",
        }[package],
    )
    monkeypatch.setattr(bench.torch, "__version__", "2.11.0")
    monkeypatch.setattr(bench.torch.version, "cuda", "13.0")

    def fake_run(command, check):
        commands.append(command)
        output = Path(command[command.index("--output-file") + 1])
        num_prompts = int(command[command.index("--num-prompts") + 1])
        output.write_text(
            json.dumps(
                {
                    "input_throughput": 1000.0,
                    "median_ttft_ms": 1.0,
                    "completed": num_prompts,
                    "total_input_tokens": int(
                        command[command.index("--random-input-len") + 1]
                    )
                    * num_prompts,
                }
            )
            + "\n"
        )

    monkeypatch.setattr(bench.subprocess, "run", fake_run)


def test_capture_runs_warmup_and_all_formal_cases_with_fixed_bench_contract(
    bench, monkeypatch, tmp_path
):
    commands = []
    args = capture_args(tmp_path)
    install_capture_environment(bench, monkeypatch, commands)
    result = bench.run_capture(args)

    assert len(result["cases"]) == len(INPUT_LENGTHS) * len(SEEDS)
    assert len(commands) == 1 + len(INPUT_LENGTHS) * len(SEEDS)
    for command in commands:
        assert command[:3] == [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
        ]
        assert command[command.index("--backend") + 1] == "sglang"
        assert command[command.index("--dataset-name") + 1] == "random"
        assert command[command.index("--model") + 1] == "/models/qwen"
        assert (
            command[command.index("--served-model-name") + 1]
            == "qwen35-fp8"
        )
        assert command[command.index("--tokenizer") + 1] == "/models/qwen"
        assert command[command.index("--random-output-len") + 1] == "1"
        assert command[command.index("--random-range-ratio") + 1] == "1"
        assert command[command.index("--warmup-requests") + 1] == "0"
        assert "--flush-cache" in command

    warmup = next(
        command
        for command in commands
        if command[command.index("--num-prompts") + 1] == "1"
    )
    assert warmup[warmup.index("--random-input-len") + 1] == "4096"
    assert warmup[warmup.index("--seed") + 1] == "0"

    formal_cases = [command for command in commands if command is not warmup]
    assert {
        (
            int(command[command.index("--random-input-len") + 1]),
            int(command[command.index("--num-prompts") + 1]),
            int(command[command.index("--seed") + 1]),
        )
        for command in formal_cases
    } == {(length, NUM_PROMPTS, seed) for length in INPUT_LENGTHS for seed in SEEDS}


def test_capture_repeats_explicit_warmup_but_reuses_formal_results(
    bench, monkeypatch, tmp_path
):
    commands = []
    args = capture_args(tmp_path)
    install_capture_environment(bench, monkeypatch, commands)

    first = bench.run_capture(args)
    second = bench.run_capture(args)

    warmups = [
        command
        for command in commands
        if command[command.index("--num-prompts") + 1] == "1"
    ]
    formal_cases = [
        command
        for command in commands
        if command[command.index("--num-prompts") + 1] == str(NUM_PROMPTS)
    ]
    assert len(warmups) == 2
    assert len(formal_cases) == len(INPUT_LENGTHS) * len(SEEDS)
    assert len(second["cases"]) == len(INPUT_LENGTHS) * len(SEEDS)
    assert [case["run_key"] for case in second["cases"]] == [
        case["run_key"] for case in first["cases"]
    ]
    result_fields = (
        "input_throughput",
        "median_ttft_ms",
        "completed",
        "total_input_tokens",
    )
    assert [
        {field: case[field] for field in result_fields} for case in second["cases"]
    ] == [{field: case[field] for field in result_fields} for case in first["cases"]]


def test_capture_validates_flashinfer_marker_before_benchmark_or_manifest(
    bench, monkeypatch, tmp_path
):
    args = capture_args(
        tmp_path,
        expected_backend="flashinfer_sm120_fp8",
        a1_mode="fused",
    )
    args.server_log = tmp_path / "server.log"
    args.server_log.write_text("server started without the A1 marker\n")

    monkeypatch.setattr(
        bench,
        "fetch_server_info",
        lambda host, port: server_info("flashinfer_sm120_fp8"),
    )
    monkeypatch.setattr(
        bench.importlib.metadata,
        "version",
        lambda package: {
            "flashinfer-python": "0.6.15.dev20260716",
            "sglang-kernel": "0.4.4",
        }[package],
    )
    monkeypatch.setattr(bench.torch, "__version__", "2.11.0")
    monkeypatch.setattr(bench.torch.version, "cuda", "13.0")
    run = Mock(side_effect=AssertionError("subprocess ran before A1 marker validation"))
    monkeypatch.setattr(bench.subprocess, "run", run)

    with pytest.raises(ValueError, match="server log missing A1 marker"):
        bench.run_capture(args)

    run.assert_not_called()
    assert not args.output.exists()
