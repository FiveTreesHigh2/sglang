#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from flashinfer_sm120_fp8_smoke import (
    build_offsets,
    build_scale_copy_plan,
    calc_diff,
    compute_padded_offset,
)


NUM_EXPERTS = 256
TOP_K = 8
BLOCK_SHAPE = (128, 128)
DIFF_THRESHOLD = 2e-3
TARGET_CASE = ("gemm1", "uniform", 65536)
OP_SHAPES = {
    "gemm1": {"n": 1024, "k": 2048, "historical_ms": 1.267},
    "gemm2": {"n": 2048, "k": 512, "historical_ms": 0.794},
}


@dataclass
class QuantizedCase:
    operation: str
    profile: str
    rows_per_expert: list[int]
    offsets: list[int]
    n: int
    k: int
    a_fp8: Any
    b_fp8: Any
    a_scale_row_major: Any
    a_scale_flashinfer: Any
    b_scale_triton: Any
    b_scale_flashinfer: Any
    m_indptr: Any
    topk_ids: Any
    topk_weights: Any
    sorted_token_ids: Any
    expert_ids: Any
    num_tokens_post_padded: Any
    triton_config: dict[str, Any]
    triton_output: Any
    flashinfer_output: Any
    reference: Any


def validate_rows_per_expert(
    rows: Sequence[int], num_experts: int, cum_m: int
) -> list[int]:
    normalized = list(rows)
    if len(normalized) != num_experts:
        raise ValueError(
            f"expected {num_experts} rows_per_expert values, got {len(normalized)}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized):
        raise ValueError("rows_per_expert values must be integers")
    if any(value < 0 for value in normalized):
        raise ValueError("rows_per_expert values must be non-negative")
    if sum(normalized) != cum_m:
        raise ValueError(
            f"rows_per_expert sums to {sum(normalized)}, expected cum_m={cum_m}"
        )
    return normalized


def build_uniform_rows(num_experts: int, cum_m: int) -> list[int]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if cum_m < 0:
        raise ValueError("cum_m must be non-negative")
    rows, remainder = divmod(cum_m, num_experts)
    return [rows + (expert_id < remainder) for expert_id in range(num_experts)]


def build_synthetic_skew_rows(
    num_experts: int, cum_m: int, *, seed: int, block_size: int
) -> list[int]:
    if num_experts < 4:
        raise ValueError("synthetic-skew requires at least four experts")
    if cum_m <= 0:
        raise ValueError("synthetic-skew requires cum_m > 0")
    if block_size <= 1:
        raise ValueError("block_size must be greater than one")

    rng = random.Random(seed)
    expert_ids = list(range(num_experts))
    rng.shuffle(expert_ids)

    zero_count = max(1, num_experts // 16)
    small_count = max(1, num_experts // 16)
    small_ids = expert_ids[zero_count : zero_count + small_count]
    active_ids = expert_ids[zero_count + small_count :]
    rows = [0] * num_experts

    remaining = cum_m
    for rank, expert_id in enumerate(small_ids, start=1):
        if remaining == 0:
            break
        count = min(rank, block_size - 1, remaining)
        rows[expert_id] = count
        remaining -= count

    if remaining:
        if not active_ids:
            rows[small_ids[-1]] += remaining
        else:
            weights = [1.0 / rank for rank in range(1, len(active_ids) + 1)]
            weight_sum = sum(weights)
            exact = [remaining * weight / weight_sum for weight in weights]
            allocated = [math.floor(value) for value in exact]
            leftovers = remaining - sum(allocated)
            by_fraction = sorted(
                range(len(active_ids)),
                key=lambda index: (exact[index] - allocated[index], -index),
                reverse=True,
            )
            for index in by_fraction[:leftovers]:
                allocated[index] += 1
            for expert_id, count in zip(active_ids, allocated):
                rows[expert_id] = count

    return validate_rows_per_expert(rows, num_experts, cum_m)


def load_rows_per_expert(
    path: Path, num_experts: int, cum_m: int
) -> list[int]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "rows_per_expert" not in payload:
            raise ValueError("rows JSON object must contain rows_per_expert")
        payload = payload["rows_per_expert"]
    if not isinstance(payload, list):
        raise ValueError("rows JSON must be an array or a rows_per_expert object")
    return validate_rows_per_expert(payload, num_experts, cum_m)


def summarize_latencies(values_ms: Sequence[float]) -> dict[str, Any]:
    values = list(values_ms)
    if not values:
        raise ValueError("at least one latency value is required")
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("latencies must be finite and positive")
    minimum = min(values)
    median = statistics.median(values)
    maximum = max(values)
    return {
        "trials_ms": values,
        "min_ms": minimum,
        "median_ms": median,
        "max_ms": maximum,
        "relative_spread": (maximum - minimum) / median,
    }


def decide_default_boost(
    *,
    speedup: float,
    min_trial_speedup: float,
    triton_relative_spread: float,
    flashinfer_relative_spread: float,
    environment_stable: bool,
    correctness_passed: bool,
) -> dict[str, Any]:
    if not correctness_passed:
        return {"status": "ERROR", "reasons": ["correctness failed"]}
    if not environment_stable:
        return {
            "status": "NEEDS_LOCKED_RERUN",
            "reasons": ["GPU environment was not stable"],
        }
    if max(triton_relative_spread, flashinfer_relative_spread) > 0.05:
        return {
            "status": "NEEDS_LOCKED_RERUN",
            "reasons": ["latency relative spread exceeded 5%"],
        }
    if speedup >= 0.30 and min_trial_speedup >= 0.20:
        return {"status": "GO", "reasons": ["stable speedup reached 30%"]}
    if speedup < 0.15:
        return {"status": "NO_GO", "reasons": ["stable speedup was below 15%"]}
    return {
        "status": "NEEDS_LOCKED_RERUN",
        "reasons": ["speedup was near the decision boundary"],
    }


def decide_locked(
    *,
    speedup: float,
    triton_relative_spread: float,
    flashinfer_relative_spread: float,
    correctness_passed: bool,
) -> dict[str, Any]:
    if not correctness_passed:
        return {"status": "ERROR", "reasons": ["correctness failed"]}
    if max(triton_relative_spread, flashinfer_relative_spread) > 0.05:
        return {
            "status": "NEEDS_LOCKED_RERUN",
            "reasons": ["locked latency relative spread exceeded 5%"],
        }
    if speedup >= 0.20:
        return {"status": "GO", "reasons": ["locked speedup reached 20%"]}
    return {"status": "NO_GO", "reasons": ["locked speedup was below 20%"]}


def resolve_triton_config(operation: str, cum_m: int) -> dict[str, Any]:
    if operation not in OP_SHAPES:
        raise ValueError(f"unsupported operation: {operation}")
    if cum_m % TOP_K != 0:
        raise ValueError(f"cum_m={cum_m} must be divisible by top_k={TOP_K}")

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        try_get_optimal_moe_config,
    )
    from sglang.srt.runtime_context import get_context, get_server_args

    try:
        get_server_args()
    except ValueError:
        get_context().set_server_args(
            SimpleNamespace(enable_deterministic_inference=False)
        )

    up_config, (down_config, _) = try_get_optimal_moe_config(
        (NUM_EXPERTS, 1024, 2048),
        (NUM_EXPERTS, 2048, 512),
        TOP_K,
        "fp8_w8a8",
        cum_m // TOP_K,
        block_shape=list(BLOCK_SHAPE),
        return_down_config=True,
    )
    selected = up_config if operation == "gemm1" else (down_config or up_config)
    config = dict(selected)
    use_tma = bool(config.pop("USE_TMA", False))
    if use_tma:
        raise RuntimeError(
            "Stage 1 packed-A comparison does not support a USE_TMA=True Triton "
            "config; provide a production-equivalent non-TMA config"
        )
    required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"}
    missing = required - set(config)
    if missing:
        raise RuntimeError(f"Triton config is missing fields: {sorted(missing)}")
    return config


def _pack_flashinfer_a_scale(a_scale_row_major, offsets: list[int]):
    import torch

    num_experts = len(offsets) - 1
    k_blocks = a_scale_row_major.shape[1]
    m_padded = compute_padded_offset(offsets[-1], num_experts)
    a_scale_flashinfer = torch.zeros(
        (k_blocks, m_padded),
        dtype=torch.float32,
        device=a_scale_row_major.device,
    )
    for start, end, packed_start in build_scale_copy_plan(offsets):
        if start != end:
            a_scale_flashinfer[:, packed_start : packed_start + end - start] = (
                a_scale_row_major[start:end].T
            )
    return a_scale_flashinfer.contiguous()


def build_fp32_reference(
    *,
    a_fp8,
    b_fp8,
    a_scale_row_major,
    b_scale_triton,
    offsets: list[int],
    n: int,
    k: int,
):
    import torch

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a_dequant = a_fp8.float() * a_scale_row_major.repeat_interleave(
            BLOCK_SHAPE[1], dim=1
        )[:, :k]
        reference = torch.empty(
            (offsets[-1], n), dtype=torch.float32, device=a_fp8.device
        )
        for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:])):
            if start == end:
                continue
            scale = b_scale_triton[expert_id]
            b_dequant = b_fp8[expert_id].float() * scale.repeat_interleave(
                BLOCK_SHAPE[0], dim=0
            ).repeat_interleave(BLOCK_SHAPE[1], dim=1)[:n, :k]
            reference[start:end] = a_dequant[start:end] @ b_dequant.T
        return reference
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32


def make_quantized_case(
    operation: str,
    profile: str,
    rows_per_expert: Sequence[int],
    *,
    seed: int,
) -> QuantizedCase:
    import torch
    from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in {(12, 0), (12, 1)}:
        raise RuntimeError(
            f"FlashInfer FP8 grouped GEMM requires compute capability 12.0 or "
            f"12.1, got {capability}"
        )

    shape = OP_SHAPES[operation]
    n = int(shape["n"])
    k = int(shape["k"])
    rows = validate_rows_per_expert(
        rows_per_expert, NUM_EXPERTS, sum(rows_per_expert)
    )
    offsets = build_offsets(rows)
    cum_m = offsets[-1]
    config = resolve_triton_config(operation, cum_m)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    a_bf16 = torch.randn((cum_m, k), dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(
        (NUM_EXPERTS, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    a_fp8, a_scale_row_major = per_token_cast_to_fp8(a_bf16)
    a_scale_flashinfer = _pack_flashinfer_a_scale(a_scale_row_major, offsets)

    b_fp8_parts = []
    b_scale_parts = []
    for expert_id in range(NUM_EXPERTS):
        b_fp8_expert, b_scale_expert = per_block_cast_to_fp8(b_bf16[expert_id])
        b_fp8_parts.append(b_fp8_expert)
        b_scale_parts.append(b_scale_expert)
    b_fp8 = torch.stack(b_fp8_parts, dim=0).contiguous()
    b_scale_triton = torch.stack(b_scale_parts, dim=0).contiguous()
    b_scale_flashinfer = b_scale_triton.transpose(-1, -2).contiguous()
    del a_bf16, b_bf16, b_fp8_parts, b_scale_parts

    expected_a_scale_shape = (cum_m, k // BLOCK_SHAPE[1])
    expected_b_scale_shape = (
        NUM_EXPERTS,
        n // BLOCK_SHAPE[0],
        k // BLOCK_SHAPE[1],
    )
    if tuple(a_scale_row_major.shape) != expected_a_scale_shape:
        raise RuntimeError(
            f"unexpected A scale shape {tuple(a_scale_row_major.shape)}, "
            f"expected {expected_a_scale_shape}"
        )
    if tuple(b_scale_triton.shape) != expected_b_scale_shape:
        raise RuntimeError(
            f"unexpected B scale shape {tuple(b_scale_triton.shape)}, "
            f"expected {expected_b_scale_shape}"
        )
    if a_scale_flashinfer.data_ptr() % 16 != 0:
        raise RuntimeError("FlashInfer A scale pointer is not 16-byte aligned")

    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    counts = torch.tensor(rows, dtype=torch.int64, device="cuda")
    packed_expert_ids = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, dtype=torch.int32, device="cuda"), counts
    )
    topk_ids = packed_expert_ids.view(cum_m, 1)
    topk_weights = torch.ones((cum_m, 1), dtype=torch.float32, device="cuda")
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], NUM_EXPERTS
    )

    reference = build_fp32_reference(
        a_fp8=a_fp8,
        b_fp8=b_fp8,
        a_scale_row_major=a_scale_row_major,
        b_scale_triton=b_scale_triton,
        offsets=offsets,
        n=n,
        k=k,
    )

    return QuantizedCase(
        operation=operation,
        profile=profile,
        rows_per_expert=rows,
        offsets=offsets,
        n=n,
        k=k,
        a_fp8=a_fp8,
        b_fp8=b_fp8,
        a_scale_row_major=a_scale_row_major,
        a_scale_flashinfer=a_scale_flashinfer,
        b_scale_triton=b_scale_triton,
        b_scale_flashinfer=b_scale_flashinfer,
        m_indptr=m_indptr,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        triton_config=config,
        triton_output=torch.empty(
            (cum_m, n), dtype=torch.bfloat16, device="cuda"
        ),
        flashinfer_output=torch.empty(
            (cum_m, n), dtype=torch.bfloat16, device="cuda"
        ),
        reference=reference,
    )


def launch_flashinfer(case: QuantizedCase) -> None:
    import torch
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    result = moe_gemm_fp8_nt_groupwise(
        case.a_fp8,
        case.b_fp8,
        case.a_scale_flashinfer,
        case.b_scale_flashinfer,
        case.m_indptr,
        out=case.flashinfer_output,
        out_dtype=torch.bfloat16,
    )
    if result.data_ptr() != case.flashinfer_output.data_ptr():
        raise RuntimeError("FlashInfer did not reuse the supplied output tensor")


def launch_triton(case: QuantizedCase) -> None:
    import triton
    import triton.language as tl
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        fused_moe_kernel,
        should_enable_swap_ab,
    )

    config = case.triton_config
    grid = lambda meta: (
        triton.cdiv(case.sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
        * triton.cdiv(case.n, meta["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel[grid](
        case.a_fp8,
        None,
        case.b_fp8,
        None,
        None,
        case.triton_output,
        case.a_scale_row_major,
        case.b_scale_triton,
        case.topk_weights,
        case.sorted_token_ids,
        case.expert_ids,
        case.num_tokens_post_padded,
        None,
        case.n,
        case.k,
        case.sorted_token_ids.shape[0],
        case.topk_ids.numel(),
        case.a_fp8.stride(0),
        case.a_fp8.stride(1),
        case.b_fp8.stride(0),
        case.b_fp8.stride(2),
        case.b_fp8.stride(1),
        0,
        0,
        case.triton_output.stride(0),
        case.triton_output.stride(1),
        case.a_scale_row_major.stride(0),
        case.a_scale_row_major.stride(1),
        case.b_scale_triton.stride(0),
        case.b_scale_triton.stride(2),
        case.b_scale_triton.stride(1),
        BLOCK_SHAPE[0],
        BLOCK_SHAPE[1],
        MUL_ROUTED_WEIGHT=False,
        top_k=1,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=(case.k % config["BLOCK_SIZE_K"] == 0),
        c_sorted=False,
        filter_expert=False,
        swap_ab=should_enable_swap_ab(
            config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
        ),
        FUSE_ADD_TO_OUTPUT=False,
        MASK_OUTPUT=False,
        LORA_PRESERVE_BASE=False,
        FUSE_SUM_ALL_REDUCE=False,
        ROUTER_TOPK=1,
        **config,
    )


def _validate_output(case: QuantizedCase, backend: str, output) -> float:
    import torch

    expected_shape = (case.offsets[-1], case.n)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"{case.operation}/{case.profile}/{backend}: output shape "
            f"{tuple(output.shape)} != {expected_shape}"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{case.operation}/{case.profile}/{backend}: output dtype "
            f"{output.dtype} != torch.bfloat16"
        )
    if not torch.isfinite(output).all().item():
        raise RuntimeError(
            f"{case.operation}/{case.profile}/{backend}: output is not finite"
        )
    difference = calc_diff(output.float(), case.reference)
    if difference > DIFF_THRESHOLD:
        raise RuntimeError(
            f"{case.operation}/{case.profile}/{backend}: calc_diff="
            f"{difference:.6e} exceeds {DIFF_THRESHOLD:.1e}"
        )
    return difference


def validate_correctness(case: QuantizedCase) -> dict[str, float]:
    return {
        "triton_vs_reference": _validate_output(
            case, "triton", case.triton_output
        ),
        "flashinfer_vs_reference": _validate_output(
            case, "flashinfer", case.flashinfer_output
        ),
        "triton_vs_flashinfer": calc_diff(
            case.triton_output.float(), case.flashinfer_output.float()
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=tuple(OP_SHAPES),
        default=["gemm1", "gemm2"],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("uniform", "synthetic-skew"),
        default=["uniform", "synthetic-skew"],
    )
    parser.add_argument("--cum-m", nargs="+", type=int, default=[65536, 131072])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clock-mode", choices=("default", "locked"), default="default"
    )
    parser.add_argument("--rows-per-expert-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(raw_argv)

    if args.rows_per_expert_json is not None:
        if len(args.cum_m) != 1:
            parser.error("--rows-per-expert-json requires exactly one --cum-m")
        if "--profiles" in raw_argv:
            parser.error("--rows-per-expert-json cannot be combined with --profiles")
    for name in ("warmup", "iterations", "trials"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(cum_m <= 0 for cum_m in args.cum_m):
        parser.error("--cum-m values must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    raise RuntimeError("GPU benchmark implementation is added in Stage 1 Task 3")


if __name__ == "__main__":
    raise SystemExit(main())
