import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("pro5000_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "model_path": "/models/qwen",
        "dtype": "auto",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "version": "0.5.14",
    }


def test_defaults_and_run_key_include_backend_commit_seed():
    bench = load_bench()
    assert bench.INPUT_LENGTHS == (4096, 6144, 14336, 30720, 63488)
    assert bench.SEEDS == (17, 29, 43)
    runtime = {
        "flashinfer_version": "0.6.15.dev20260716",
        "torch_version": "2.11.0",
        "torch_cuda_version": "13.0",
        "sglang_kernel_version": "0.4.4",
        "python_executable": "/venv/bin/python3",
        "dense_fp8_backend": "flashinfer_cutlass",
        "chunked_prefill_size": 8192,
    }
    first = bench.build_run_key(
        backend="triton",
        commit="a" * 40,
        input_length=4096,
        num_prompts=100,
        seed=17,
        server_args_hash="b" * 64,
        a1_mode="not_applicable",
        flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test",
        gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    second = bench.build_run_key(
        backend="flashinfer_sm120_fp8",
        commit="a" * 40,
        input_length=4096,
        num_prompts=100,
        seed=17,
        server_args_hash="b" * 64,
        a1_mode="fused",
        flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test",
        gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    assert first != second
    legacy = bench.build_run_key(
        backend="flashinfer_sm120_fp8",
        commit="a" * 40,
        input_length=4096,
        num_prompts=100,
        seed=17,
        server_args_hash="b" * 64,
        a1_mode="legacy",
        flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test",
        gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    assert legacy != second


def test_server_info_requires_exact_backend_dense_backend_and_chunk():
    bench = load_bench()
    snapshot = bench.verify_server_info(server_info(), "triton")
    assert snapshot["moe_runner_backend"] == "triton"
    for field, value in (
        ("moe_runner_backend", "flashinfer_sm120_fp8"),
        ("fp8_gemm_runner_backend", "triton"),
        ("chunked_prefill_size", 4096),
        ("tp_size", 2),
        ("enable_metrics", False),
        ("mem_fraction_static", 0.8),
    ):
        bad = server_info()
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            bench.verify_server_info(bad, "triton")


def manifest(backend, ratios):
    cases = []
    for input_length in (4096, 6144, 14336, 30720, 63488):
        for seed, ratio in zip((17, 29, 43), ratios[input_length]):
            base = 1000.0
            cases.append(
                {
                    "input_length": input_length,
                    "seed": seed,
                    "completed": 100,
                    "total_input_tokens": input_length * 100,
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
            "num_prompts": 100,
        },
        "cases": cases,
    }


def test_compare_requires_4096_ten_percent_and_no_seed_regression():
    bench = load_bench()
    ratios = {
        4096: (1.10, 1.11, 1.12),
        6144: (1.01, 1.02, 1.03),
        14336: (1.01, 1.02, 1.03),
        30720: (1.01, 1.02, 1.03),
        63488: (1.01, 1.02, 1.03),
    }
    result = bench.compare_manifests(
        manifest("triton", ratios),
        manifest("flashinfer_sm120_fp8", ratios),
    )
    assert result["decision"] == "GO"
    assert result["by_input_length"]["4096"]["median_speedup"] >= 0.10

    ratios[4096] = (0.99, 1.11, 1.12)
    assert (
        bench.compare_manifests(
            manifest("triton", ratios),
            manifest("flashinfer_sm120_fp8", ratios),
        )["decision"]
        == "FUNCTIONAL_ONLY"
    )


def test_compare_rejects_config_drift_or_incomplete_case():
    bench = load_bench()
    ratios = {length: (1.2, 1.2, 1.2) for length in bench.INPUT_LENGTHS}
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    flashinfer["metadata"]["pairing_hash"] = "e" * 64
    with pytest.raises(ValueError, match="pairing_hash"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["pairing_hash"] = "c" * 64
    flashinfer["metadata"]["sglang_commit"] = "e" * 40
    with pytest.raises(ValueError, match="sglang_commit"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["sglang_commit"] = "d" * 40
    flashinfer["metadata"]["gpu_frequency_strategy"] = "application-clocks-locked"
    with pytest.raises(ValueError, match="gpu_frequency_strategy"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["gpu_frequency_strategy"] = "default-unlocked"
    flashinfer["cases"][0]["completed"] = 99
    with pytest.raises(ValueError, match="completed"):
        bench.compare_manifests(triton, flashinfer)
