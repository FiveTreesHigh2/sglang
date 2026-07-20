#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from typing import Any


DIFF_THRESHOLD = 1e-3
SUPPORTED_CC = {(12, 0), (12, 1)}


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def build_offsets(rows_per_expert: list[int]) -> list[int]:
    offsets = [0]
    for rows in rows_per_expert:
        if rows < 0:
            raise ValueError(f"rows_per_expert must be non-negative, got {rows}")
        offsets.append(offsets[-1] + rows)
    return offsets


def build_scale_copy_plan(offsets: list[int]) -> list[tuple[int, int, int]]:
    if not offsets or offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("offsets must be non-decreasing")
    return [
        (start, end, compute_padded_offset(start, expert_id))
        for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:]))
    ]


def calc_diff(x, y) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum().item()
    if denominator == 0:
        return 0.0
    return 1.0 - 2.0 * (x * y).sum().item() / denominator


def quantize_and_pack_a(x, m_indptr):
    import torch
    from flashinfer.testing.utils import per_token_cast_to_fp8

    x_fp8, scale_row_major = per_token_cast_to_fp8(x)
    num_experts = m_indptr.numel() - 1
    k_blocks = scale_row_major.shape[1]
    m_padded = compute_padded_offset(x.shape[0], num_experts)
    packed = torch.zeros(
        (k_blocks, m_padded), dtype=torch.float32, device=x.device
    )
    for start, end, packed_start in build_scale_copy_plan(m_indptr.tolist()):
        if start == end:
            continue
        packed[:, packed_start : packed_start + end - start] = (
            scale_row_major[start:end].t()
        )
    return x_fp8, packed


def make_inputs(rows_per_expert: list[int], n: int, k: int):
    import torch
    from flashinfer.testing.utils import per_block_cast_to_fp8

    torch.manual_seed(0)
    offsets = build_offsets(rows_per_expert)
    total_rows = offsets[-1]
    num_experts = len(rows_per_expert)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    a_bf16 = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    reference = torch.zeros(
        (total_rows, n), dtype=torch.bfloat16, device="cuda"
    )
    for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start < end:
            reference[start:end] = a_bf16[start:end] @ b_bf16[expert_id].t()

    a_fp8, a_scale = quantize_and_pack_a(a_bf16, m_indptr)
    b_fp8_parts = []
    b_scale_parts = []
    for expert_id in range(num_experts):
        b_fp8, b_scale = per_block_cast_to_fp8(b_bf16[expert_id])
        b_fp8_parts.append(b_fp8)
        b_scale_parts.append(b_scale)
    b_fp8 = torch.stack(b_fp8_parts, dim=0)
    b_scale = torch.stack(b_scale_parts, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr, reference


def run_case(name: str, rows_per_expert: list[int], n: int, k: int) -> dict[str, Any]:
    import torch
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    a, b, a_scale, b_scale, m_indptr, reference = make_inputs(
        rows_per_expert, n, k
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    output = moe_gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        out_dtype=torch.bfloat16,
    )
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    kernel_ms = start_event.elapsed_time(end_event)
    difference = calc_diff(output.float(), reference.float())
    if difference >= DIFF_THRESHOLD:
        raise RuntimeError(
            f"{name}: calc_diff={difference:.6e} exceeds {DIFF_THRESHOLD:.1e}"
        )
    return {
        "name": name,
        "num_experts": len(rows_per_expert),
        "cum_m": sum(rows_per_expert),
        "n": n,
        "k": k,
        "calc_diff": difference,
        "kernel_ms": kernel_ms,
        "first_call_wall_seconds": wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-shapes",
        action="store_true",
        help="Also validate the target 256-expert GEMM1 and GEMM2 N/K shapes.",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_CC:
        raise RuntimeError(
            f"Expected compute capability 12.0 or 12.1, got {capability}"
        )

    cases = [("small_irregular_empty", [0, 1, 8, 0, 16], 256, 512)]
    if args.real_shapes:
        one_row_per_expert = [1] * 256
        cases.extend(
            [
                ("qwen_gemm1_shape", one_row_per_expert, 1024, 2048),
                ("qwen_gemm2_shape", one_row_per_expert, 2048, 512),
            ]
        )

    results = [run_case(name, rows, n, k) for name, rows, n, k in cases]
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
