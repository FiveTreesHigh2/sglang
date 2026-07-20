#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

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
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in normalized
    ):
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
    environment_stable: bool,
    correctness_passed: bool,
) -> dict[str, Any]:
    if not correctness_passed:
        return {"status": "ERROR", "reasons": ["correctness failed"]}
    if not environment_stable:
        return {
            "status": "NEEDS_LOCKED_RERUN",
            "reasons": ["locked GPU environment evidence was not stable"],
        }
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


def trial_backend_order(trial_index: int) -> tuple[str, str]:
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    if trial_index % 2 == 0:
        return ("triton", "flashinfer")
    return ("flashinfer", "triton")


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as error:
        return {"command": list(command), "error": repr(error)}


def _git_snapshot() -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"])
    branch = _run_command(["git", "branch", "--show-current"])
    status = _run_command(["git", "status", "--porcelain"])
    return {
        "commit": commit.get("stdout"),
        "branch": branch.get("stdout"),
        "dirty": bool(status.get("stdout")),
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "torch",
        "triton",
        "sglang",
        "sglang-kernel",
        "flashinfer-python",
        "flashinfer-jit-cache",
        "nvidia-cutlass-dsl",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def empty_result_payload(command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "git": {},
        "environment": {},
        "parameters": {},
        "cases": [],
        "decision": {"status": "NOT_EVALUATED", "reasons": []},
        "status": "running",
    }


def collect_environment() -> dict[str, Any]:
    import flashinfer
    import sglang
    import torch
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    del moe_gemm_fp8_nt_groupwise
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "packages": _package_versions(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name() if cuda_available else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability()) if cuda_available else None
        ),
        "flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        "flashinfer_disable_jit": os.environ.get("FLASHINFER_DISABLE_JIT"),
        "flashinfer_file": str(Path(flashinfer.__file__).resolve()),
        "sglang_file": str(Path(sglang.__file__).resolve()),
        "nvcc": _run_command(["nvcc", "--version"]),
    }


def validate_environment_contract(
    environment: dict[str, Any], *, expected_repo: Path = REPO_ROOT
) -> None:
    errors: list[str] = []
    version_info = environment.get("python_version_info") or []
    if list(version_info[:2]) != [3, 12]:
        errors.append(f"Python must be 3.12, got {version_info}")

    torch_version = str(environment.get("torch_version") or "")
    if torch_version.split("+", 1)[0] != "2.11.0":
        errors.append(f"torch must be 2.11.0, got {torch_version or None}")
    if environment.get("torch_cuda") != "13.0":
        errors.append(
            f"torch CUDA runtime must be 13.0, got {environment.get('torch_cuda')}"
        )

    packages = environment.get("packages") or {}
    required_packages = {
        "flashinfer-python": "0.6.15.dev20260716",
        "nvidia-cutlass-dsl": "4.5.2",
        "sglang-kernel": "0.4.4",
    }
    for package, expected_version in required_packages.items():
        actual_version = packages.get(package)
        if actual_version != expected_version:
            errors.append(
                f"{package} must be {expected_version}, got {actual_version}"
            )
    if packages.get("flashinfer-jit-cache") is not None:
        errors.append(
            "flashinfer-jit-cache must be absent for the runtime-JIT Stage 1 run"
        )

    gpu_name = str(environment.get("gpu") or "")
    if "RTX PRO 5000" not in gpu_name:
        errors.append(f"GPU must be RTX PRO 5000, got {gpu_name or None}")
    if tuple(environment.get("compute_capability") or ()) not in {
        (12, 0),
        (12, 1),
    }:
        errors.append(
            "compute capability must be 12.0 or 12.1, got "
            f"{environment.get('compute_capability')}"
        )

    nvcc = environment.get("nvcc") or {}
    nvcc_output = f"{nvcc.get('stdout', '')}\n{nvcc.get('stderr', '')}"
    if nvcc.get("returncode") != 0:
        errors.append("nvcc --version did not complete successfully")
    elif "release 13.0" not in nvcc_output or "V13.0.48" not in nvcc_output:
        errors.append("NVCC must report CUDA 13.0 V13.0.48")

    sglang_file = environment.get("sglang_file")
    expected_sglang_root = (expected_repo / "python" / "sglang").resolve()
    try:
        Path(str(sglang_file)).resolve().relative_to(expected_sglang_root)
    except (TypeError, ValueError):
        errors.append(
            "sglang must import from the current checkout at "
            f"{expected_sglang_root}, got {sglang_file}"
        )

    if errors:
        raise RuntimeError("Stage 1 environment contract failed: " + "; ".join(errors))


def resolve_nvidia_smi_gpu_id(*, pid: int | None = None) -> str:
    target_pid = os.getpid() if pid is None else pid
    completed = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.get("returncode") != 0:
        raise RuntimeError(
            "failed to map the CUDA process to an nvidia-smi GPU: "
            f"{completed.get('stderr') or completed}"
        )
    matches: set[str] = set()
    for line in completed.get("stdout", "").splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2:
            continue
        try:
            row_pid = int(values[0])
        except ValueError:
            continue
        if row_pid == target_pid and values[1]:
            matches.add(values[1])
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one nvidia-smi GPU UUID for pid {target_pid}, "
            f"found {sorted(matches)}"
        )
    return next(iter(matches))


def sample_gpu_state(
    gpu_id: str | None, *, identity_error: str | None = None
) -> dict[str, Any]:
    if gpu_id is None:
        return {
            "ok": False,
            "identity_error": identity_error or "nvidia-smi GPU identity unresolved",
        }
    fields = (
        "uuid",
        "index",
        "pstate",
        "clocks.current.sm",
        "temperature.gpu",
        "power.draw",
        "utilization.gpu",
        "memory.used",
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
        f"--id={gpu_id}",
    ]
    completed = _run_command(command)
    if completed.get("returncode") != 0:
        return {"ok": False, **completed}
    lines = [line.strip() for line in completed.get("stdout", "").splitlines()]
    if len(lines) != 1:
        return {
            "ok": False,
            **completed,
            "parse_error": f"expected one GPU row, got {len(lines)}",
        }
    values = [value.strip() for value in lines[0].split(",")]
    if len(values) != len(fields):
        return {
            "ok": False,
            **completed,
            "parse_error": f"expected {len(fields)} values, got {len(values)}",
        }

    def parse_number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "ok": True,
        "gpu_uuid": values[0],
        "expected_gpu_uuid": gpu_id,
        "identity_matches": values[0] == gpu_id,
        "index": values[1],
        "pstate": values[2],
        "sm_clock_mhz": parse_number(values[3]),
        "temperature_c": parse_number(values[4]),
        "power_draw_w": parse_number(values[5]),
        "utilization_percent": parse_number(values[6]),
        "memory_used_mib": parse_number(values[7]),
        "raw": lines[0],
    }


def evaluate_gpu_stability(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples or any(
        not sample.get("ok") or not sample.get("identity_matches")
        for sample in samples
    ):
        return {
            "stable": False,
            "reasons": ["one or more nvidia-smi samples failed identity validation"],
            "sm_clock_relative_spread": None,
        }
    pstates = {sample.get("pstate") for sample in samples}
    clocks = [sample.get("sm_clock_mhz") for sample in samples]
    if any(clock is None or clock <= 0 for clock in clocks):
        return {
            "stable": False,
            "reasons": ["one or more SM clock samples were invalid"],
            "sm_clock_relative_spread": None,
        }
    median_clock = statistics.median(clocks)
    clock_spread = (max(clocks) - min(clocks)) / median_clock
    reasons = []
    if len(pstates) != 1:
        reasons.append("P-state changed during the benchmark")
    if clock_spread > 0.05:
        reasons.append("SM clock relative spread exceeded 5%")
    return {
        "stable": not reasons,
        "reasons": reasons,
        "pstates": sorted(pstates),
        "sm_clock_relative_spread": clock_spread,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def emit_stdout_json_if_needed(
    payload: dict[str, Any], output_path: Path | None
) -> None:
    if output_path is not None:
        return
    print("STAGE_1_JSON_BEGIN", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("STAGE_1_JSON_END", flush=True)


def prepare_backend(fn: Callable[[], None]) -> float:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def warmup_backend(fn: Callable[[], None], warmup: int) -> None:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()


def time_backend(fn: Callable[[], None], iterations: int) -> float:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _rows_summary(rows: Sequence[int], block_size_m: int) -> dict[str, Any]:
    return {
        "min": min(rows),
        "max": max(rows),
        "mean": sum(rows) / len(rows),
        "zero_experts": sum(value == 0 for value in rows),
        "small_experts": sum(0 < value < block_size_m for value in rows),
    }


def run_benchmark_case(
    operation: str,
    profile: str,
    rows: Sequence[int],
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    import torch

    cum_m = sum(rows)
    print(
        f"[prepare] operation={operation} profile={profile} cum_m={cum_m}",
        flush=True,
    )
    case = make_quantized_case(operation, profile, rows, seed=seed)
    gpu_identity_error = None
    try:
        nvidia_smi_gpu_id = resolve_nvidia_smi_gpu_id()
        print(f"[gpu] nvidia_smi_id={nvidia_smi_gpu_id}", flush=True)
    except RuntimeError as error:
        nvidia_smi_gpu_id = None
        gpu_identity_error = str(error)
        print(f"[gpu] identity unresolved: {gpu_identity_error}", flush=True)
    functions = {
        "triton": lambda: launch_triton(case),
        "flashinfer": lambda: launch_flashinfer(case),
    }

    jit_prepare_seconds: dict[str, float] = {}
    for backend in ("triton", "flashinfer"):
        print(f"[jit] {backend} {operation}/{profile}/{cum_m}", flush=True)
        jit_prepare_seconds[backend] = prepare_backend(functions[backend])

    correctness = validate_correctness(case)
    case.reference = None
    print(
        f"[correctness] PASS {operation}/{profile}/{cum_m} "
        f"triton={correctness['triton_vs_reference']:.3e} "
        f"flashinfer={correctness['flashinfer_vs_reference']:.3e}",
        flush=True,
    )

    for backend in ("triton", "flashinfer"):
        warmup_backend(functions[backend], args.warmup)

    latencies = {"triton": [], "flashinfer": []}
    gpu_samples: list[dict[str, Any]] = []
    for trial_index in range(args.trials):
        for backend in trial_backend_order(trial_index):
            before = sample_gpu_state(
                nvidia_smi_gpu_id, identity_error=gpu_identity_error
            )
            latency_ms = time_backend(functions[backend], args.iterations)
            after = sample_gpu_state(
                nvidia_smi_gpu_id, identity_error=gpu_identity_error
            )
            latencies[backend].append(latency_ms)
            gpu_samples.extend(
                [
                    {
                        "trial": trial_index,
                        "backend": backend,
                        "phase": "before",
                        **before,
                    },
                    {
                        "trial": trial_index,
                        "backend": backend,
                        "phase": "after",
                        **after,
                    },
                ]
            )
            print(
                f"[trial {trial_index + 1}/{args.trials}] {backend}="
                f"{latency_ms:.6f} ms",
                flush=True,
            )

    triton_summary = summarize_latencies(latencies["triton"])
    flashinfer_summary = summarize_latencies(latencies["flashinfer"])
    speedup = (
        triton_summary["median_ms"] / flashinfer_summary["median_ms"] - 1.0
    )
    paired_speedups = [
        triton_ms / flashinfer_ms - 1.0
        for triton_ms, flashinfer_ms in zip(
            latencies["triton"], latencies["flashinfer"]
        )
    ]
    stability = evaluate_gpu_stability(gpu_samples)
    config = case.triton_config
    triton_grid_size = math.ceil(
        case.sorted_token_ids.shape[0] / config["BLOCK_SIZE_M"]
    ) * math.ceil(case.n / config["BLOCK_SIZE_N"])
    actual_padded = int(case.num_tokens_post_padded.item())

    result = {
        "operation": operation,
        "profile": profile,
        "cum_m": cum_m,
        "n": case.n,
        "k": case.k,
        "rows_per_expert": list(rows),
        "rows_summary": _rows_summary(rows, config["BLOCK_SIZE_M"]),
        "triton_config": config,
        "triton_grid_size": triton_grid_size,
        "sorted_token_ids_capacity": case.sorted_token_ids.shape[0],
        "num_tokens_post_padded": actual_padded,
        "jit_prepare_seconds": jit_prepare_seconds,
        "correctness": {"passed": True, **correctness},
        "latency": {
            "triton": triton_summary,
            "flashinfer": flashinfer_summary,
        },
        "paired_trial_speedups": paired_speedups,
        "min_trial_speedup": min(paired_speedups),
        "speedup": speedup,
        "historical_reference_ms": OP_SHAPES[operation]["historical_ms"],
        "gpu_samples": gpu_samples,
        "gpu_stability": stability,
        "nvidia_smi_gpu_id": nvidia_smi_gpu_id,
        "gpu_identity_error": gpu_identity_error,
    }
    del case, functions
    torch.cuda.empty_cache()
    return result


def _case_rows(
    profile: str,
    cum_m: int,
    args: argparse.Namespace,
) -> list[int]:
    if args.rows_per_expert_json is not None:
        return load_rows_per_expert(
            args.rows_per_expert_json, NUM_EXPERTS, cum_m
        )
    if profile == "uniform":
        return build_uniform_rows(NUM_EXPERTS, cum_m)
    if profile == "synthetic-skew":
        return build_synthetic_skew_rows(
            NUM_EXPERTS, cum_m, seed=args.seed, block_size=64
        )
    raise ValueError(f"unsupported profile: {profile}")


def _select_decision(
    cases: Sequence[dict[str, Any]], clock_mode: str
) -> dict[str, Any]:
    target = next(
        (
            case
            for case in cases
            if (case["operation"], case["profile"], case["cum_m"])
            == TARGET_CASE
        ),
        None,
    )
    if target is None:
        return {
            "status": "NOT_EVALUATED",
            "reasons": ["the decisive GEMM1/uniform/cum_m=65536 case was not run"],
        }
    common = {
        "speedup": target["speedup"],
        "triton_relative_spread": target["latency"]["triton"][
            "relative_spread"
        ],
        "flashinfer_relative_spread": target["latency"]["flashinfer"][
            "relative_spread"
        ],
        "correctness_passed": target["correctness"]["passed"],
        "environment_stable": target["gpu_stability"]["stable"],
    }
    if clock_mode == "locked":
        decision = decide_locked(**common)
    else:
        decision = decide_default_boost(
            **common,
            min_trial_speedup=target["min_trial_speedup"],
        )
    return {
        **decision,
        "target": {
            "operation": target["operation"],
            "profile": target["profile"],
            "cum_m": target["cum_m"],
            "speedup": target["speedup"],
        },
    }


def _print_case_summary(result: dict[str, Any]) -> None:
    triton_ms = result["latency"]["triton"]["median_ms"]
    flashinfer_ms = result["latency"]["flashinfer"]["median_ms"]
    print(
        f"RESULT {result['operation']} {result['profile']} "
        f"cum_m={result['cum_m']} triton_ms={triton_ms:.6f} "
        f"flashinfer_ms={flashinfer_ms:.6f} "
        f"speedup={result['speedup'] * 100:.2f}% correctness=PASS",
        flush=True,
    )


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
        if any(
            value == "--profiles" or value.startswith("--profiles=")
            for value in raw_argv
        ):
            parser.error("--rows-per-expert-json cannot be combined with --profiles")
    for name in ("warmup", "iterations", "trials"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(cum_m <= 0 for cum_m in args.cum_m):
        parser.error("--cum-m values must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    payload = empty_result_payload(command)
    payload["git"] = _git_snapshot()
    payload["parameters"] = {
        "operations": args.operations,
        "profiles": args.profiles,
        "cum_m": args.cum_m,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trials": args.trials,
        "seed": args.seed,
        "clock_mode": args.clock_mode,
        "rows_per_expert_json": (
            str(args.rows_per_expert_json)
            if args.rows_per_expert_json is not None
            else None
        ),
    }
    current_stage = "environment"
    current_case: dict[str, Any] | None = None

    def persist() -> None:
        if args.output is not None:
            write_json_atomic(args.output, payload)

    try:
        if "FLASHINFER_DISABLE_JIT" in os.environ:
            raise RuntimeError(
                "FLASHINFER_DISABLE_JIT must be unset for the Stage 1 benchmark"
            )
        payload["environment"] = collect_environment()
        validate_environment_contract(payload["environment"])
        if not payload["environment"]["cuda_available"]:
            raise RuntimeError("CUDA is not available")
        if tuple(payload["environment"]["compute_capability"]) not in {
            (12, 0),
            (12, 1),
        }:
            raise RuntimeError(
                "Stage 1 requires compute capability 12.0 or 12.1; got "
                f"{payload['environment']['compute_capability']}"
            )
        persist()

        profiles = (
            ["real-replay"]
            if args.rows_per_expert_json is not None
            else args.profiles
        )
        current_stage = "benchmark"
        for operation_index, operation in enumerate(args.operations):
            for cum_m_index, cum_m in enumerate(args.cum_m):
                for profile in profiles:
                    current_case = {
                        "operation": operation,
                        "profile": profile,
                        "cum_m": cum_m,
                    }
                    rows = _case_rows(profile, cum_m, args)
                    case_seed = args.seed + operation_index * 100 + cum_m_index * 10
                    result = run_benchmark_case(
                        operation, profile, rows, args, seed=case_seed
                    )
                    payload["cases"].append(result)
                    _print_case_summary(result)
                    persist()

        payload["decision"] = _select_decision(
            payload["cases"], args.clock_mode
        )
        payload["status"] = "completed"
        persist()
        emit_stdout_json_if_needed(payload, args.output)
        print(f"STAGE_1_DECISION={payload['decision']['status']}", flush=True)
        print(
            "STAGE_1_RESULT_JSON="
            f"{args.output.resolve() if args.output is not None else 'stdout-only'}",
            flush=True,
        )
        return 0
    except Exception as error:
        payload["status"] = "error"
        payload["decision"] = {
            "status": "ERROR",
            "reasons": [f"{type(error).__name__}: {error}"],
        }
        payload["error"] = {
            "stage": current_stage,
            "case": current_case,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            persist()
        except Exception as persist_error:
            print(
                f"ERROR: failed to write partial JSON: {persist_error}",
                file=sys.stderr,
                flush=True,
            )
        traceback.print_exc()
        emit_stdout_json_if_needed(payload, args.output)
        print("STAGE_1_DECISION=ERROR", flush=True)
        print(
            "STAGE_1_RESULT_JSON="
            f"{args.output.resolve() if args.output is not None else 'stdout-only'}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
