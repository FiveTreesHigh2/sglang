#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence
from unittest.mock import patch


PREFILL_MAIN_TOKENS = 8192
PREFILL_MAIN_PROFILE = "uniform"
DECODE_TOKENS = frozenset((1, 8))
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
BLOCK_SHAPE = (128, 128)
COMMON_COMPONENT_DETAIL_KEYS = frozenset(
    (
        "quant1",
        "moe_permute",
        "scale_pack_gemm1",
        "gemm1",
        "gemm2",
        "unpermute_combine",
    )
)
LEGACY_COMPONENT_EXTRA_KEYS = frozenset(
    ("silu", "quant2", "scale_pack_gemm2")
)
FUSED_COMPONENT_EXTRA_KEYS = frozenset(
    ("fused_swiglu_quant_pack_gemm2",)
)
FUSED_A1_A2_COMPONENT_DETAIL_KEYS = frozenset(
    (
        "moe_permute_prepare",
        "fused_quant_scatter_pack_gemm1",
        "gemm1",
        "fused_swiglu_quant_pack_gemm2",
        "gemm2",
        "unpermute_combine",
    )
)
COMPONENT_ROLLUP_KEYS = (
    "gemm1_input_prepare",
    "gemm1",
    "gemm2_input_prepare",
    "gemm2",
    "unpermute_combine",
)
LEGACY_COMPONENT_TRACE = (
    "quant1",
    "moe_permute",
    "scale_pack_gemm1",
    "gemm1",
    "silu",
    "quant2",
    "scale_pack_gemm2",
    "gemm2",
    "unpermute_combine",
)
A1_LEGACY_A2_FUSED_COMPONENT_TRACE = (
    "quant1",
    "moe_permute",
    "scale_pack_gemm1",
    "gemm1",
    "fused_swiglu_quant_pack_gemm2",
    "gemm2",
    "unpermute_combine",
)
A1_FUSED_A2_FUSED_COMPONENT_TRACE = (
    "moe_permute_prepare",
    "fused_quant_scatter_pack_gemm1",
    "gemm1",
    "fused_swiglu_quant_pack_gemm2",
    "gemm2",
    "unpermute_combine",
)
A1_LEGACY_A2_LEGACY_COMPONENT_CALL_COUNTS = {
    "quant": 2,
    "pack": 2,
    "gemm": 2,
    "moe_permute": 1,
    "prepare": 0,
    "unpermute": 1,
    "silu": 1,
    "fused_a1": 0,
    "fused_a2": 0,
}
A1_LEGACY_A2_FUSED_COMPONENT_CALL_COUNTS = {
    "quant": 1,
    "pack": 1,
    "gemm": 2,
    "moe_permute": 1,
    "prepare": 0,
    "unpermute": 1,
    "silu": 0,
    "fused_a1": 0,
    "fused_a2": 1,
}
A1_FUSED_A2_FUSED_COMPONENT_CALL_COUNTS = {
    "quant": 0,
    "pack": 0,
    "gemm": 2,
    "moe_permute": 0,
    "prepare": 1,
    "unpermute": 1,
    "silu": 0,
    "fused_a1": 1,
    "fused_a2": 1,
}
FULL_MEAN_ABS_REL_TOL = 5e-3
FULL_SYMMETRIC_DIFF_TOL = 1e-4
FULL_NORMALIZED_RMSE_TOL = 1e-2
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SharedWeights:
    w13_weight: Any
    w2_weight: Any
    w13_scale: Any
    w2_scale: Any
    w13_scale_fi: Any
    w2_scale_fi: Any
    # Gated (up-first) FI variants; None when SGLANG_FLASHINFER_SM120_FP8_GATED
    # is off. The Triton reference always consumes the gate-first originals.
    w13_weight_gated: Any = None
    w13_scale_fi_gated: Any = None


@dataclass
class RunnerCase:
    tokens: int
    top_k: int
    profile: str
    dispatch: Any
    config: Any
    quant_info: Any
    shared_weights: SharedWeights


@dataclass
class CapturedGraph:
    graph: Any
    output: Any


@dataclass
class CutlassState:
    ab_strides1: Any
    c_strides1: Any
    ab_strides2: Any
    c_strides2: Any
    workspace: Any
    a_ptr: Any
    b_ptr: Any
    out_ptr: Any
    a_scales_ptr: Any
    b_scales_ptr: Any
    expert_offsets: Any
    problem_sizes1: Any
    problem_sizes2: Any


def decide(
    *,
    correct: bool,
    graph: bool,
    prefill_speedup: float,
    decode_regression: float,
) -> str:
    if not correct or not graph:
        return "NO_GO"
    if prefill_speedup >= 0.10 and decode_regression <= 0.05 + 1e-12:
        return "GO"
    return "FUNCTIONAL_ONLY"


def build_routing_rows(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    profile: str,
) -> list[list[int]]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be in [1, num_experts]")

    if profile == "uniform":
        return [
            [
                (token * top_k + rank) % num_experts
                for rank in range(top_k)
            ]
            for token in range(tokens)
        ]
    if profile != "synthetic-skew":
        raise ValueError(f"unsupported routing profile: {profile}")

    hot_count = max(1, top_k // 2)
    cold_count = top_k - hot_count
    hot_experts = list(range(hot_count))
    cold_pool = num_experts - hot_count
    return [
        hot_experts
        + [
            hot_count + (token * cold_count + rank) % cold_pool
            for rank in range(cold_count)
        ]
        for token in range(tokens)
    ]


def select_decision(
    cases: Sequence[dict[str, Any]], *, cuda_graph_passed: bool
) -> dict[str, Any]:
    main_cases = [
        case
        for case in cases
        if case.get("tokens") == PREFILL_MAIN_TOKENS
        and case.get("profile") == PREFILL_MAIN_PROFILE
    ]
    if len(main_cases) != 1:
        raise ValueError(
            "Stage 2 decision requires exactly one tokens=8192 uniform case"
        )

    decode_cases = [case for case in cases if case.get("tokens") in DECODE_TOKENS]
    if not decode_cases:
        raise ValueError("Stage 2 decision requires tokens=1 or tokens=8 decode cases")

    main_case = main_cases[0]
    triton_main_ms = float(main_case["triton"]["median_ms"])
    flashinfer_main_ms = float(
        main_case["flashinfer_sm120_fp8"]["median_ms"]
    )
    prefill_speedup = triton_main_ms / flashinfer_main_ms - 1.0

    decode_graph_pairs: list[tuple[float, float]] = []
    for case in decode_cases:
        graph = case.get("cuda_graph")
        try:
            triton_graph_ms = float(graph["triton"]["median_ms"])
            flashinfer_graph_ms = float(
                graph["flashinfer_sm120_fp8"]["median_ms"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Stage 2 decision requires decode CUDA Graph latencies "
                "for Triton and flashinfer_sm120_fp8"
            ) from error
        decode_graph_pairs.append((triton_graph_ms, flashinfer_graph_ms))
    decode_regression = max(
        0.0,
        *(
            flashinfer_ms / triton_ms - 1.0
            for triton_ms, flashinfer_ms in decode_graph_pairs
        ),
    )
    correct = all(
        case.get("correctness", {}).get("status") == "PASS" for case in cases
    )
    status = decide(
        correct=correct,
        graph=cuda_graph_passed,
        prefill_speedup=prefill_speedup,
        decode_regression=decode_regression,
    )
    return {
        "status": status,
        "correct": correct,
        "cuda_graph_passed": cuda_graph_passed,
        "prefill_main_case": {
            "tokens": PREFILL_MAIN_TOKENS,
            "routed_rows": PREFILL_MAIN_TOKENS * int(main_case["top_k"])
            if "top_k" in main_case
            else None,
            "profile": PREFILL_MAIN_PROFILE,
        },
        "prefill_speedup": prefill_speedup,
        "decode_regression": decode_regression,
    }


def summarize_latencies(values_ms: Sequence[float]) -> dict[str, Any]:
    values = list(values_ms)
    if not values:
        raise ValueError("at least one latency value is required")
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("latencies must be finite and positive")
    return {
        "median_ms": statistics.median(values),
        "trials_ms": values,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def build_component_profile(detail_ms: dict[str, float]) -> dict[str, Any]:
    keys = frozenset(detail_ms)
    legacy_keys = (
        COMMON_COMPONENT_DETAIL_KEYS | LEGACY_COMPONENT_EXTRA_KEYS
    )
    fused_keys = COMMON_COMPONENT_DETAIL_KEYS | FUSED_COMPONENT_EXTRA_KEYS
    if keys == legacy_keys:
        path = "a1_legacy_a2_legacy"
        gemm1_input_prepare = sum(
            detail_ms[key]
            for key in ("quant1", "moe_permute", "scale_pack_gemm1")
        )
        gemm2_input_prepare = sum(
            detail_ms[key]
            for key in ("silu", "quant2", "scale_pack_gemm2")
        )
    elif keys == fused_keys:
        path = "a1_legacy_a2_fused"
        gemm1_input_prepare = sum(
            detail_ms[key]
            for key in ("quant1", "moe_permute", "scale_pack_gemm1")
        )
        gemm2_input_prepare = detail_ms[
            "fused_swiglu_quant_pack_gemm2"
        ]
    elif keys == FUSED_A1_A2_COMPONENT_DETAIL_KEYS:
        path = "a1_fused_a2_fused"
        gemm1_input_prepare = sum(
            detail_ms[key]
            for key in (
                "moe_permute_prepare",
                "fused_quant_scatter_pack_gemm1",
            )
        )
        gemm2_input_prepare = detail_ms[
            "fused_swiglu_quant_pack_gemm2"
        ]
    else:
        raise ValueError(
            "component detail must match exactly one legacy/fused schema; "
            f"got {sorted(keys)}"
        )

    rollup = {
        "gemm1_input_prepare": gemm1_input_prepare,
        "gemm1": detail_ms["gemm1"],
        "gemm2_input_prepare": gemm2_input_prepare,
        "gemm2": detail_ms["gemm2"],
        "unpermute_combine": detail_ms["unpermute_combine"],
    }
    return {
        "path": path,
        "detail_ms": dict(detail_ms),
        "rollup_ms": rollup,
    }


def validate_component_trace(
    trace: Sequence[str], call_counts: dict[str, int]
) -> str:
    normalized_trace = tuple(trace)
    if normalized_trace == LEGACY_COMPONENT_TRACE:
        path = "a1_legacy_a2_legacy"
        expected_counts = A1_LEGACY_A2_LEGACY_COMPONENT_CALL_COUNTS
    elif normalized_trace == A1_LEGACY_A2_FUSED_COMPONENT_TRACE:
        path = "a1_legacy_a2_fused"
        expected_counts = A1_LEGACY_A2_FUSED_COMPONENT_CALL_COUNTS
    elif normalized_trace == A1_FUSED_A2_FUSED_COMPONENT_TRACE:
        path = "a1_fused_a2_fused"
        expected_counts = A1_FUSED_A2_FUSED_COMPONENT_CALL_COUNTS
    else:
        raise ValueError(
            "component call trace does not match a supported runner path: "
            f"{list(normalized_trace)}"
        )
    if call_counts != expected_counts:
        raise ValueError(
            f"component call counts for {path} must be {expected_counts}, "
            f"got {call_counts}"
        )
    return path


def build_case_result(
    *,
    tokens: int,
    top_k: int,
    profile: str,
    correctness: dict[str, Any],
    triton_trials: Sequence[float],
    flashinfer_trials: Sequence[float],
    component_profile: dict[str, Any],
    cuda_graph_trials: dict[str, Sequence[float]] | None = None,
    cutlass_trials: Sequence[float] | None = None,
    cutlass_status: str = "CUTLASS_UNAVAILABLE",
) -> dict[str, Any]:
    if set(component_profile) != {"path", "detail_ms", "rollup_ms"}:
        raise ValueError(
            "component_profile must contain path, detail_ms, and rollup_ms"
        )
    normalized_profile = build_component_profile(
        component_profile["detail_ms"]
    )
    if normalized_profile != component_profile:
        raise ValueError("component_profile is inconsistent with detail_ms")
    triton = summarize_latencies(triton_trials)
    flashinfer = summarize_latencies(flashinfer_trials)
    result = {
        "tokens": tokens,
        "routed_rows": tokens * top_k,
        "top_k": top_k,
        "profile": profile,
        "correctness": correctness,
        "triton": triton,
        "flashinfer_sm120_fp8": flashinfer,
        "speedup_percent": (
            triton["median_ms"] / flashinfer["median_ms"] - 1.0
        )
        * 100.0,
        "components": normalized_profile,
    }
    is_decode = tokens in DECODE_TOKENS
    if is_decode and cuda_graph_trials is None:
        raise ValueError("decode CUDA Graph trials are required")
    if not is_decode and cuda_graph_trials is not None:
        raise ValueError("prefill cases must not contain CUDA Graph trials")
    if cuda_graph_trials is None:
        result["cuda_graph"] = {"status": "NOT_RUN"}
    else:
        expected_graph_backends = {"triton", "flashinfer_sm120_fp8"}
        if set(cuda_graph_trials) != expected_graph_backends:
            raise ValueError(
                "cuda_graph_trials must contain exactly Triton and "
                "flashinfer_sm120_fp8"
            )
        result["cuda_graph"] = {
            name: summarize_latencies(values)
            for name, values in cuda_graph_trials.items()
        }
    result["cutlass"] = (
        summarize_latencies(cutlass_trials)
        if cutlass_trials is not None
        else {"status": cutlass_status}
    )
    return result


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def empty_result_payload(command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "git": {},
        "environment": {},
        "parameters": {},
        "cutlass_preflight": {"status": "NOT_RUN"},
        "cuda_graph": {"status": "NOT_RUN"},
        "cases": [],
        "decision": {"status": "NOT_EVALUATED"},
        "status": "running",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def collect_environment() -> dict[str, Any]:
    from benchmark_flashinfer_sm120_fp8_moe import (
        collect_environment as collect_stage_1_environment,
    )

    environment = collect_stage_1_environment()
    environment["stage_2_packages"] = {
        package: _package_version(package)
        for package in ("torch", "sglang", "flashinfer-python", "sglang-kernel")
    }
    return environment


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_environment(environment: dict[str, Any]) -> None:
    from benchmark_flashinfer_sm120_fp8_moe import validate_environment_contract

    validate_environment_contract(environment, expected_repo=REPO_ROOT)
    if "FLASHINFER_DISABLE_JIT" in os.environ:
        raise RuntimeError(
            "FLASHINFER_DISABLE_JIT must be unset for the Stage 2 benchmark"
        )


def _calc_diff(actual: Any, expected: Any) -> float:
    numerator = (actual.float() - expected.float()).abs().mean()
    denominator = expected.float().abs().mean().clamp_min(1e-12)
    return float((numerator / denominator).item())


def _calc_symmetric_diff(actual: Any, expected: Any) -> float:
    actual = actual.double()
    expected = expected.double()
    denominator = (actual.square() + expected.square()).sum().clamp_min(1e-24)
    return float((1.0 - 2.0 * (actual * expected).sum() / denominator).item())


def _calc_normalized_rmse(actual: Any, expected: Any) -> float:
    error_rms = (actual.float() - expected.float()).square().mean().sqrt()
    expected_rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error_rms / expected_rms).item())


def compare_outputs(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(actual).all()) and bool(
        torch.isfinite(expected).all()
    )
    calc_diff = _calc_diff(actual, expected)
    symmetric_diff = _calc_symmetric_diff(actual, expected)
    normalized_rmse = _calc_normalized_rmse(actual, expected)
    passed = (
        finite
        and calc_diff < FULL_MEAN_ABS_REL_TOL
        and symmetric_diff < FULL_SYMMETRIC_DIFF_TOL
        and normalized_rmse < FULL_NORMALIZED_RMSE_TOL
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "finite": finite,
        "calc_diff": calc_diff,
        "symmetric_diff": symmetric_diff,
        "normalized_rmse": normalized_rmse,
        "thresholds": {
            "calc_diff": FULL_MEAN_ABS_REL_TOL,
            "symmetric_diff": FULL_SYMMETRIC_DIFF_TOL,
            "normalized_rmse": FULL_NORMALIZED_RMSE_TOL,
        },
    }


def _quantize_weight_tensor(
    shape: tuple[int, int, int], scale: float
) -> tuple[Any, Any]:
    import torch
    from flashinfer.testing.utils import per_block_cast_to_fp8

    experts, rows, columns = shape
    quantized = torch.empty(
        shape,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    block_scales = torch.empty(
        (experts, rows // 128, columns // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for expert in range(experts):
        source = (
            torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
            * scale
        )
        expert_weight, expert_scale = per_block_cast_to_fp8(source)
        quantized[expert].copy_(expert_weight)
        block_scales[expert].copy_(expert_scale)
        if (expert + 1) % 32 == 0 or expert + 1 == experts:
            print(
                f"[prepare] quantized weight experts={expert + 1}/{experts} "
                f"shape=({rows},{columns})",
                flush=True,
            )
    return quantized.contiguous(), block_scales.contiguous()


def create_shared_weights(seed: int) -> SharedWeights:
    import torch
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        prepare_flashinfer_sm120_fp8_weight_scales,
    )

    torch.manual_seed(seed)
    w13_weight, w13_scale = _quantize_weight_tensor(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        HIDDEN_SIZE**-0.5,
    )
    w2_weight, w2_scale = _quantize_weight_tensor(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        INTERMEDIATE_SIZE**-0.5,
    )
    w13_scale_fi, w2_scale_fi = prepare_flashinfer_sm120_fp8_weight_scales(
        w13_scale,
        w2_scale,
    )
    w13_weight_gated = None
    w13_scale_fi_gated = None
    from sglang.srt.environ import envs as _envs

    if _envs.SGLANG_FLASHINFER_SM120_FP8_GATED.get():
        half = w13_weight.shape[1] // 2
        scale_half = w13_scale.shape[1] // 2
        w13_weight_gated = torch.cat(
            [w13_weight[:, half:], w13_weight[:, :half]], dim=1
        ).contiguous()
        w13_scale_gated = torch.cat(
            [w13_scale[:, scale_half:], w13_scale[:, :scale_half]], dim=1
        ).contiguous()
        w13_scale_fi_gated, _ = prepare_flashinfer_sm120_fp8_weight_scales(
            w13_scale_gated,
            w2_scale,
        )
    return SharedWeights(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_fi=w13_scale_fi,
        w2_scale_fi=w2_scale_fi,
        w13_weight_gated=w13_weight_gated,
        w13_scale_fi_gated=w13_scale_fi_gated,
    )


def make_runner_case(
    *,
    tokens: int,
    top_k: int,
    profile: str,
    shared_weights: SharedWeights,
    seed: int,
) -> RunnerCase:
    import torch
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        FlashInferSm120Fp8MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    torch.manual_seed(seed)
    hidden_states = (
        torch.randn(
            tokens,
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 8
    )
    routing_rows = build_routing_rows(
        tokens=tokens,
        top_k=top_k,
        num_experts=NUM_EXPERTS,
        profile=profile,
    )
    topk_ids = torch.tensor(routing_rows, device="cuda", dtype=torch.int32)
    topk_weights = torch.rand(
        tokens,
        top_k,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_output = StandardTopKOutput(
        topk_weights,
        topk_ids,
        torch.empty(0, device="cuda"),
    )
    dispatch = StandardDispatchOutput(hidden_states, None, topk_output)
    config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
        routed_scaling_factor=1.0,
    )
    if shared_weights.w13_weight_gated is not None:
        quant_info = FlashInferSm120Fp8MoeQuantInfo(
            shared_weights.w13_weight_gated,
            shared_weights.w2_weight,
            shared_weights.w13_scale_fi_gated,
            shared_weights.w2_scale_fi,
            BLOCK_SHAPE,
            w13_up_first=True,
        )
    else:
        quant_info = FlashInferSm120Fp8MoeQuantInfo(
            shared_weights.w13_weight,
            shared_weights.w2_weight,
            shared_weights.w13_scale_fi,
            shared_weights.w2_scale_fi,
            BLOCK_SHAPE,
        )
    return RunnerCase(
        tokens=tokens,
        top_k=top_k,
        profile=profile,
        dispatch=dispatch,
        config=config,
        quant_info=quant_info,
        shared_weights=shared_weights,
    )


def launch_flashinfer(case: RunnerCase) -> Any:
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        fused_experts_none_to_flashinfer_sm120_fp8,
    )

    return fused_experts_none_to_flashinfer_sm120_fp8(
        case.dispatch,
        case.quant_info,
        case.config,
    ).hidden_states


@contextmanager
def triton_single_rank_context() -> Iterator[Any]:
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe as triton_fused_moe,
    )
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(
        enable_deterministic_inference=False
    ), patch.object(
        triton_fused_moe,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    ):
        yield triton_fused_moe


def launch_triton(case: RunnerCase, triton_fused_moe: Any) -> Any:
    return triton_fused_moe.fused_experts(
        case.dispatch.hidden_states,
        case.shared_weights.w13_weight,
        case.shared_weights.w2_weight,
        case.dispatch.topk_output,
        case.config,
        use_fp8_w8a8=True,
        w1_scale=case.shared_weights.w13_scale,
        w2_scale=case.shared_weights.w2_scale,
        block_shape=list(BLOCK_SHAPE),
    )


def create_cutlass_state() -> CutlassState:
    import torch

    device = "cuda"
    return CutlassState(
        ab_strides1=torch.full(
            (NUM_EXPERTS,), HIDDEN_SIZE, device=device, dtype=torch.int64
        ),
        c_strides1=torch.full(
            (NUM_EXPERTS,),
            2 * INTERMEDIATE_SIZE,
            device=device,
            dtype=torch.int64,
        ),
        ab_strides2=torch.full(
            (NUM_EXPERTS,),
            INTERMEDIATE_SIZE,
            device=device,
            dtype=torch.int64,
        ),
        c_strides2=torch.full(
            (NUM_EXPERTS,), HIDDEN_SIZE, device=device, dtype=torch.int64
        ),
        workspace=torch.empty(90000, device=device, dtype=torch.uint8),
        a_ptr=torch.empty(NUM_EXPERTS, device=device, dtype=torch.int64),
        b_ptr=torch.empty(NUM_EXPERTS, device=device, dtype=torch.int64),
        out_ptr=torch.empty(NUM_EXPERTS, device=device, dtype=torch.int64),
        a_scales_ptr=torch.empty(
            NUM_EXPERTS, device=device, dtype=torch.int64
        ),
        b_scales_ptr=torch.empty(
            NUM_EXPERTS, device=device, dtype=torch.int64
        ),
        expert_offsets=torch.empty(
            NUM_EXPERTS + 1, device=device, dtype=torch.int32
        ),
        problem_sizes1=torch.empty(
            NUM_EXPERTS, 3, device=device, dtype=torch.int32
        ),
        problem_sizes2=torch.empty(
            NUM_EXPERTS, 3, device=device, dtype=torch.int32
        ),
    )


def launch_cutlass(case: RunnerCase, state: CutlassState) -> Any:
    from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

    weights = case.shared_weights
    return cutlass_fused_experts_fp8(
        case.dispatch.hidden_states,
        weights.w13_weight.transpose(1, 2),
        weights.w2_weight.transpose(1, 2),
        weights.w13_scale.transpose(1, 2),
        weights.w2_scale.transpose(1, 2),
        case.dispatch.topk_output.topk_weights,
        case.dispatch.topk_output.topk_ids,
        state.ab_strides1,
        state.c_strides1,
        state.ab_strides2,
        state.c_strides2,
        state.workspace,
        state.a_ptr,
        state.b_ptr,
        state.out_ptr,
        state.a_scales_ptr,
        state.b_scales_ptr,
        state.expert_offsets,
        state.problem_sizes1,
        state.problem_sizes2,
        use_fp8_blockscale=True,
    )


def run_cutlass_preflight(
    case: RunnerCase, state: CutlassState
) -> dict[str, Any]:
    import torch
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

    try:
        if not cutlass_fp8_supported():
            raise RuntimeError("cutlass_fp8_supported() returned False")
        with triton_single_rank_context() as triton_fused_moe:
            expected = launch_triton(case, triton_fused_moe)
        actual = launch_cutlass(case, state)
        torch.cuda.synchronize()
        correctness = compare_outputs(actual, expected)
        return {
            "status": "CUTLASS_AVAILABLE",
            "tokens": case.tokens,
            "routed_rows": case.tokens * case.top_k,
            "profile": case.profile,
            "correctness": correctness,
        }
    except Exception:
        return {
            "status": "CUTLASS_UNAVAILABLE",
            "tokens": case.tokens,
            "routed_rows": case.tokens * case.top_k,
            "profile": case.profile,
            "error": traceback.format_exc(),
        }


def warmup_backend(fn: Callable[[], Any], warmup: int) -> Any:
    import torch

    retained = None
    for _ in range(warmup):
        retained = fn()
    torch.cuda.synchronize()
    return retained


def time_backend(fn: Callable[[], Any], iterations: int) -> tuple[float, Any]:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    retained = None
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        retained = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations, retained


def capture_backend_graph(fn: Callable[[], Any]) -> CapturedGraph:
    import torch

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return CapturedGraph(graph=graph, output=output)


def time_graph_replay(state: CapturedGraph, iterations: int) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        state.graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def trial_backend_order(
    trial: int, *, include_cutlass: bool
) -> tuple[str, ...]:
    backends = ["triton", "flashinfer_sm120_fp8"]
    if include_cutlass:
        backends.append("cutlass")
    shift = trial % len(backends)
    return tuple(backends[shift:] + backends[:shift])


def profile_flashinfer_components(
    case: RunnerCase, *, iterations: int
) -> dict[str, Any]:
    import torch
    from sglang.srt.layers.moe.moe_runner import (
        flashinfer_sm120_fp8 as flashinfer_runner,
    )

    events: dict[str, list[tuple[Any, Any]]] = {}
    call_counts = {
        "quant": 0,
        "pack": 0,
        "gemm": 0,
        "moe_permute": 0,
        "prepare": 0,
        "unpermute": 0,
        "silu": 0,
        "fused_a1": 0,
        "fused_a2": 0,
    }
    iteration_trace: list[str] = []
    observed_path: str | None = None

    def recorded(
        label: str, category: str, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_counts[category] += 1
            iteration_trace.append(label)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            events.setdefault(label, []).append((start, end))
            return result

        return wrapper

    def sequential(
        category: str,
        labels: tuple[str, ...],
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            index = call_counts[category]
            label = (
                labels[index]
                if index < len(labels)
                else f"unexpected_{category}_{index + 1}"
            )
            return recorded(label, category, fn)(*args, **kwargs)

        return wrapper

    def quant_wrapper(*args: Any, **kwargs: Any) -> Any:
        return sequential(
            "quant",
            ("quant1", "quant2"),
            originals["quant"],
        )(*args, **kwargs)

    def pack_wrapper(*args: Any, **kwargs: Any) -> Any:
        return sequential(
            "pack",
            ("scale_pack_gemm1", "scale_pack_gemm2"),
            originals["pack"],
        )(*args, **kwargs)

    def gemm_wrapper(*args: Any, **kwargs: Any) -> Any:
        return sequential(
            "gemm",
            ("gemm1", "gemm2"),
            originals["gemm"],
        )(*args, **kwargs)

    originals = {
        "quant": flashinfer_runner.sglang_per_token_group_quant_fp8,
        "pack": flashinfer_runner.pack_flashinfer_sm120_fp8_scale,
        "gemm": flashinfer_runner._run_grouped_gemm,
        "moe_permute": flashinfer_runner.moe_permute,
        "moe_permute_prepare": getattr(
            flashinfer_runner,
            "moe_permute_prepare",
            None,
        ),
        "unpermute_combine": flashinfer_runner.moe_unpermute,
        "silu": getattr(flashinfer_runner, "silu_and_mul", None),
        "fused_a1": getattr(
            flashinfer_runner,
            "fused_quant_scatter_pack_flashinfer_sm120_fp8",
            None,
        ),
        "fused_a2": getattr(
            flashinfer_runner,
            "fused_swiglu_quant_pack_flashinfer_sm120_fp8",
            None,
        ),
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "sglang_per_token_group_quant_fp8",
                quant_wrapper,
            )
        )
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "moe_permute",
                recorded(
                    "moe_permute",
                    "moe_permute",
                    originals["moe_permute"],
                ),
            )
        )
        if originals["moe_permute_prepare"] is not None:
            stack.enter_context(
                patch.object(
                    flashinfer_runner,
                    "moe_permute_prepare",
                    recorded(
                        "moe_permute_prepare",
                        "prepare",
                        originals["moe_permute_prepare"],
                    ),
                )
            )
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "pack_flashinfer_sm120_fp8_scale",
                pack_wrapper,
            )
        )
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "_run_grouped_gemm",
                gemm_wrapper,
            )
        )
        if originals["silu"] is not None:
            stack.enter_context(
                patch.object(
                    flashinfer_runner,
                    "silu_and_mul",
                    recorded("silu", "silu", originals["silu"]),
                )
            )
        if originals["fused_a1"] is not None:
            stack.enter_context(
                patch.object(
                    flashinfer_runner,
                    "fused_quant_scatter_pack_flashinfer_sm120_fp8",
                    recorded(
                        "fused_quant_scatter_pack_gemm1",
                        "fused_a1",
                        originals["fused_a1"],
                    ),
                )
            )
        if originals["fused_a2"] is not None:
            stack.enter_context(
                patch.object(
                    flashinfer_runner,
                    "fused_swiglu_quant_pack_flashinfer_sm120_fp8",
                    recorded(
                        "fused_swiglu_quant_pack_gemm2",
                        "fused_a2",
                        originals["fused_a2"],
                    ),
                )
            )
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "moe_unpermute",
                recorded(
                    "unpermute_combine",
                    "unpermute",
                    originals["unpermute_combine"],
                ),
            )
        )
        for _ in range(iterations):
            iteration_trace.clear()
            for category in call_counts:
                call_counts[category] = 0
            flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
                case.dispatch,
                case.quant_info,
                case.config,
            )
            iteration_path = validate_component_trace(
                iteration_trace,
                call_counts,
            )
            if observed_path is None:
                observed_path = iteration_path
            elif observed_path != iteration_path:
                raise RuntimeError(
                    "FlashInfer component path changed during profiling: "
                    f"{observed_path} -> {iteration_path}"
                )
    torch.cuda.synchronize()

    detail_ms = {
        label: sum(start.elapsed_time(end) for start, end in label_events)
        / iterations
        for label, label_events in events.items()
    }
    profile = build_component_profile(detail_ms)
    if profile["path"] != observed_path:
        raise RuntimeError(
            "component detail path does not match observed call trace: "
            f"{profile['path']} != {observed_path}"
        )
    return profile


def run_cuda_graph_check(case: RunnerCase) -> dict[str, Any]:
    import torch
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    static_x = case.dispatch.hidden_states
    static_ids = case.dispatch.topk_output.topk_ids
    static_weights = case.dispatch.topk_output.topk_weights

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            launch_flashinfer(case)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = launch_flashinfer(case)
    output_ptr = graph_output.data_ptr()

    alternate_profile = (
        "synthetic-skew" if case.profile == "uniform" else "uniform"
    )
    route_sets = (
        build_routing_rows(
            tokens=case.tokens,
            top_k=case.top_k,
            num_experts=NUM_EXPERTS,
            profile=case.profile,
        ),
        build_routing_rows(
            tokens=case.tokens,
            top_k=case.top_k,
            num_experts=NUM_EXPERTS,
            profile=alternate_profile,
        ),
    )
    for route_rows in route_sets:
        new_x = torch.randn_like(static_x) / 8
        new_ids = torch.tensor(route_rows, device="cuda", dtype=torch.int32)
        static_x.copy_(new_x)
        static_ids.copy_(new_ids)
        static_weights.fill_(1.0 / case.top_k)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_output.clone()

        eager_dispatch = StandardDispatchOutput(
            new_x,
            None,
            StandardTopKOutput(
                static_weights.clone(),
                new_ids,
                torch.empty(0, device="cuda"),
            ),
        )
        eager_case = RunnerCase(
            tokens=case.tokens,
            top_k=case.top_k,
            profile=alternate_profile,
            dispatch=eager_dispatch,
            config=case.config,
            quant_info=case.quant_info,
            shared_weights=case.shared_weights,
        )
        eager = launch_flashinfer(eager_case)
        torch.cuda.synchronize()
        if graph_output.data_ptr() != output_ptr:
            raise RuntimeError("CUDA Graph output address changed across replay")
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)

    allocated_before = torch.cuda.memory_allocated()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after != allocated_before:
        raise RuntimeError(
            "CUDA Graph replay changed allocated memory: "
            f"before={allocated_before} after={allocated_after}"
        )
    return {
        "status": "PASS",
        "tokens": case.tokens,
        "top_k": case.top_k,
        "profiles": [case.profile, alternate_profile],
        "output_ptr": output_ptr,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "replays": 20,
    }


def run_benchmark_case(
    case: RunnerCase,
    *,
    warmup: int,
    trials: int,
    iterations: int,
    cutlass_state: CutlassState | None,
    cutlass_status: str,
) -> dict[str, Any]:
    import torch

    with triton_single_rank_context() as triton_fused_moe:
        triton_output = launch_triton(case, triton_fused_moe)
        flashinfer_output = launch_flashinfer(case)
        torch.cuda.synchronize()
        correctness = compare_outputs(flashinfer_output, triton_output)

        launchers: dict[str, Callable[[], Any]] = {
            "triton": lambda: launch_triton(case, triton_fused_moe),
            "flashinfer_sm120_fp8": lambda: launch_flashinfer(case),
        }
        if cutlass_state is not None:
            launchers["cutlass"] = lambda: launch_cutlass(case, cutlass_state)

        for name, launcher in launchers.items():
            print(
                f"[warmup] backend={name} tokens={case.tokens} "
                f"profile={case.profile}",
                flush=True,
            )
            warmup_backend(launcher, warmup)

        latencies: dict[str, list[float]] = {
            name: [] for name in launchers
        }
        retained: dict[str, Any] = {}
        for trial in range(trials):
            for name in trial_backend_order(
                trial,
                include_cutlass=cutlass_state is not None,
            ):
                latency, retained[name] = time_backend(
                    launchers[name], iterations
                )
                latencies[name].append(latency)
                print(
                    f"[trial {trial + 1}/{trials}] backend={name} "
                    f"tokens={case.tokens} profile={case.profile} "
                    f"latency_ms={latency:.6f}",
                    flush=True,
                )

        graph_latencies: dict[str, list[float]] | None = None
        graph_states: dict[str, CapturedGraph] = {}
        if case.tokens in DECODE_TOKENS:
            graph_latencies = {
                "triton": [],
                "flashinfer_sm120_fp8": [],
            }
            for name in graph_latencies:
                print(
                    f"[graph capture] backend={name} tokens={case.tokens} "
                    f"profile={case.profile}",
                    flush=True,
                )
                graph_states[name] = capture_backend_graph(launchers[name])
            for trial in range(trials):
                for name in trial_backend_order(
                    trial,
                    include_cutlass=False,
                ):
                    latency = time_graph_replay(
                        graph_states[name],
                        iterations,
                    )
                    graph_latencies[name].append(latency)
                    print(
                        f"[graph trial {trial + 1}/{trials}] "
                        f"backend={name} tokens={case.tokens} "
                        f"profile={case.profile} latency_ms={latency:.6f}",
                        flush=True,
                    )

    component_iterations = min(iterations, 20)
    component_profile = profile_flashinfer_components(
        case,
        iterations=component_iterations,
    )
    result = build_case_result(
        tokens=case.tokens,
        top_k=case.top_k,
        profile=case.profile,
        correctness=correctness,
        triton_trials=latencies["triton"],
        flashinfer_trials=latencies["flashinfer_sm120_fp8"],
        component_profile=component_profile,
        cuda_graph_trials=graph_latencies,
        cutlass_trials=latencies.get("cutlass"),
        cutlass_status=cutlass_status,
    )
    result["component_profile_iterations"] = component_iterations
    result["component_scope"] = (
        "CUDA kernel time inside the production FlashInfer runner; "
        "Python and allocator host time are excluded"
    )
    del graph_states, retained
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 8, 128, 8192, 16384],
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("uniform", "synthetic-skew"),
        default=["uniform", "synthetic-skew"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--check-cuda-graph", action="store_true")
    parser.add_argument("--cutlass-preflight", action="store_true")
    args = parser.parse_args(argv)
    if any(tokens <= 0 for tokens in args.tokens):
        parser.error("--tokens values must be positive")
    if args.top_k <= 0 or args.top_k > NUM_EXPERTS:
        parser.error(f"--top-k must be in [1, {NUM_EXPERTS}]")
    for name in ("warmup", "trials", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def _print_case_summary(result: dict[str, Any]) -> None:
    cutlass = result["cutlass"]
    cutlass_text = (
        f"{cutlass['median_ms']:.6f}"
        if "median_ms" in cutlass
        else cutlass["status"]
    )
    graph = result["cuda_graph"]
    graph_text = (
        " triton_graph_ms="
        f"{graph['triton']['median_ms']:.6f}"
        " flashinfer_graph_ms="
        f"{graph['flashinfer_sm120_fp8']['median_ms']:.6f}"
        if graph.get("status") != "NOT_RUN"
        else ""
    )
    print(
        f"RESULT tokens={result['tokens']} routed_rows={result['routed_rows']} "
        f"profile={result['profile']} "
        f"triton_ms={result['triton']['median_ms']:.6f} "
        "flashinfer_ms="
        f"{result['flashinfer_sm120_fp8']['median_ms']:.6f} "
        f"cutlass={cutlass_text} "
        f"speedup={result['speedup_percent']:.2f}% "
        f"correctness={result['correctness']['status']}"
        f"{graph_text}",
        flush=True,
    )


def _emit_stdout_json(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        return
    print("STAGE_2_JSON_BEGIN", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("STAGE_2_JSON_END", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    payload = empty_result_payload(command)
    payload["git"] = _git_snapshot()
    payload["parameters"] = {
        "tokens": args.tokens,
        "top_k": args.top_k,
        "profiles": args.profiles,
        "warmup": args.warmup,
        "trials": args.trials,
        "iterations": args.iterations,
        "seed": args.seed,
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "block_shape": list(BLOCK_SHAPE),
        "check_cuda_graph": args.check_cuda_graph,
        "cutlass_preflight": args.cutlass_preflight,
    }
    current_stage = "environment"

    def persist() -> None:
        if args.output_json is not None:
            write_json_atomic(args.output_json, payload)

    try:
        payload["environment"] = collect_environment()
        validate_environment(payload["environment"])
        persist()

        current_stage = "weight-preparation"
        print("[prepare] creating shared Qwen3.5-A3B MoE weights", flush=True)
        shared_weights = create_shared_weights(args.seed)
        persist()

        cutlass_state: CutlassState | None = None
        cutlass_status = "SKIPPED"
        if args.cutlass_preflight:
            current_stage = "cutlass-preflight"
            print(
                "[preflight] CUTLASS tokens=8192 routed_rows=65536 "
                "profile=uniform",
                flush=True,
            )
            preflight_case = make_runner_case(
                tokens=PREFILL_MAIN_TOKENS,
                top_k=args.top_k,
                profile=PREFILL_MAIN_PROFILE,
                shared_weights=shared_weights,
                seed=args.seed + 9000,
            )
            candidate_state = create_cutlass_state()
            payload["cutlass_preflight"] = run_cutlass_preflight(
                preflight_case,
                candidate_state,
            )
            cutlass_status = payload["cutlass_preflight"]["status"]
            if payload["cutlass_preflight"]["status"] == "CUTLASS_AVAILABLE":
                cutlass_state = candidate_state
            print(
                "CUTLASS_PREFLIGHT_STATUS="
                f"{payload['cutlass_preflight']['status']}",
                flush=True,
            )
            del preflight_case
            persist()
        else:
            payload["cutlass_preflight"] = {"status": "SKIPPED"}

        if args.check_cuda_graph:
            current_stage = "cuda-graph"
            print(
                "[cuda-graph] capture/replay tokens=8 top_k="
                f"{args.top_k}",
                flush=True,
            )
            graph_case = make_runner_case(
                tokens=8,
                top_k=args.top_k,
                profile="uniform",
                shared_weights=shared_weights,
                seed=args.seed + 8000,
            )
            try:
                payload["cuda_graph"] = run_cuda_graph_check(graph_case)
            except Exception:
                payload["cuda_graph"] = {
                    "status": "FAIL",
                    "error": traceback.format_exc(),
                }
            print(
                f"CUDA_GRAPH_STATUS={payload['cuda_graph']['status']}",
                flush=True,
            )
            del graph_case
            persist()
        else:
            payload["cuda_graph"] = {"status": "SKIPPED"}

        current_stage = "benchmark"
        for token_index, tokens in enumerate(args.tokens):
            for profile_index, profile in enumerate(args.profiles):
                print(
                    f"[case] tokens={tokens} routed_rows={tokens * args.top_k} "
                    f"profile={profile}",
                    flush=True,
                )
                case = make_runner_case(
                    tokens=tokens,
                    top_k=args.top_k,
                    profile=profile,
                    shared_weights=shared_weights,
                    seed=args.seed + token_index * 100 + profile_index,
                )
                result = run_benchmark_case(
                    case,
                    warmup=args.warmup,
                    trials=args.trials,
                    iterations=args.iterations,
                    cutlass_state=cutlass_state,
                    cutlass_status=cutlass_status,
                )
                payload["cases"].append(result)
                _print_case_summary(result)
                persist()
                del case

        current_stage = "decision"
        try:
            payload["decision"] = select_decision(
                payload["cases"],
                cuda_graph_passed=(
                    payload["cuda_graph"].get("status") == "PASS"
                ),
            )
        except ValueError as error:
            payload["decision"] = {
                "status": "NOT_EVALUATED",
                "reason": str(error),
            }
        payload["status"] = "completed"
        persist()
        _emit_stdout_json(payload, args.output_json)
        print(
            f"STAGE_2_DECISION={payload['decision']['status']}",
            flush=True,
        )
        print(
            "STAGE_2_RESULT_JSON="
            f"{args.output_json.resolve() if args.output_json else 'stdout-only'}",
            flush=True,
        )
        return 0
    except Exception:
        payload["status"] = "error"
        payload["error"] = {
            "stage": current_stage,
            "traceback": traceback.format_exc(),
        }
        persist()
        _emit_stdout_json(payload, args.output_json)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
