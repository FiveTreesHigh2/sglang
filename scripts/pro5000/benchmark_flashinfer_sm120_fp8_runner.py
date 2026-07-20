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
COMPONENT_KEYS = (
    "routing_quant_pack",
    "gemm1",
    "swiglu_quant",
    "scale_layout_gemm2",
    "gemm2",
    "unpermute_combine",
)
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
    decode_regression = max(
        0.0,
        *(
            float(case["flashinfer_sm120_fp8"]["median_ms"])
            / float(case["triton"]["median_ms"])
            - 1.0
            for case in decode_cases
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


def build_case_result(
    *,
    tokens: int,
    top_k: int,
    profile: str,
    correctness: dict[str, Any],
    triton_trials: Sequence[float],
    flashinfer_trials: Sequence[float],
    components_ms: dict[str, float],
    cutlass_trials: Sequence[float] | None = None,
) -> dict[str, Any]:
    if set(components_ms) != set(COMPONENT_KEYS):
        raise ValueError(
            f"components_ms must contain exactly {sorted(COMPONENT_KEYS)}"
        )
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
        "components_ms": dict(components_ms),
    }
    result["cutlass"] = (
        summarize_latencies(cutlass_trials)
        if cutlass_trials is not None
        else {"status": "CUTLASS_UNAVAILABLE"}
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
        "schema_version": 2,
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
    return SharedWeights(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_scale_fi=w13_scale_fi,
        w2_scale_fi=w2_scale_fi,
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
) -> dict[str, float]:
    import torch
    from sglang.srt.layers.moe.moe_runner import (
        flashinfer_sm120_fp8 as flashinfer_runner,
    )

    events: dict[str, list[tuple[Any, Any]]] = {
        key: [] for key in COMPONENT_KEYS
    }
    counters = {"quant": 0, "pack": 0, "gemm": 0}

    def recorded(label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            events[label].append((start, end))
            return result

        return wrapper

    def quant_wrapper(*args: Any, **kwargs: Any) -> Any:
        label = (
            "routing_quant_pack"
            if counters["quant"] % 2 == 0
            else "swiglu_quant"
        )
        counters["quant"] += 1
        return recorded(
            label,
            originals["quant"],
        )(*args, **kwargs)

    def pack_wrapper(*args: Any, **kwargs: Any) -> Any:
        label = (
            "routing_quant_pack"
            if counters["pack"] % 2 == 0
            else "scale_layout_gemm2"
        )
        counters["pack"] += 1
        return recorded(
            label,
            originals["pack"],
        )(*args, **kwargs)

    def gemm_wrapper(*args: Any, **kwargs: Any) -> Any:
        label = "gemm1" if counters["gemm"] % 2 == 0 else "gemm2"
        counters["gemm"] += 1
        return recorded(label, originals["gemm"])(*args, **kwargs)

    originals = {
        "quant": flashinfer_runner.sglang_per_token_group_quant_fp8,
        "pack": flashinfer_runner.pack_flashinfer_sm120_fp8_scale,
        "gemm": flashinfer_runner._run_grouped_gemm,
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
                recorded("routing_quant_pack", flashinfer_runner.moe_permute),
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
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "silu_and_mul",
                recorded("swiglu_quant", flashinfer_runner.silu_and_mul),
            )
        )
        stack.enter_context(
            patch.object(
                flashinfer_runner,
                "moe_unpermute",
                recorded(
                    "unpermute_combine", flashinfer_runner.moe_unpermute
                ),
            )
        )
        for _ in range(iterations):
            flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
                case.dispatch,
                case.quant_info,
                case.config,
            )
    torch.cuda.synchronize()

    del originals
    return {
        label: sum(start.elapsed_time(end) for start, end in label_events)
        / iterations
        for label, label_events in events.items()
    }


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

    component_iterations = min(iterations, 20)
    components = profile_flashinfer_components(
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
        components_ms=components,
        cutlass_trials=latencies.get("cutlass"),
    )
    result["component_profile_iterations"] = component_iterations
    result["component_scope"] = (
        "CUDA kernel time inside the production FlashInfer runner; "
        "Python and allocator host time are excluded"
    )
    del retained
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
    print(
        f"RESULT tokens={result['tokens']} routed_rows={result['routed_rows']} "
        f"profile={result['profile']} "
        f"triton_ms={result['triton']['median_ms']:.6f} "
        "flashinfer_ms="
        f"{result['flashinfer_sm120_fp8']['median_ms']:.6f} "
        f"cutlass={cutlass_text} "
        f"speedup={result['speedup_percent']:.2f}% "
        f"correctness={result['correctness']['status']}",
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
