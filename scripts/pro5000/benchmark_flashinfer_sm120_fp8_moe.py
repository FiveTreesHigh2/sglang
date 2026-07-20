#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
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
