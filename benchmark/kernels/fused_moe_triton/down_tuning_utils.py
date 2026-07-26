from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch


ROUTE_PROFILES = ("uniform", "synthetic-skew")


def candidate_key(config: Mapping[str, Any], use_tma: bool) -> str:
    normalized = {
        key: value for key, value in config.items() if key != "USE_TMA"
    }
    return json.dumps(
        {"config": normalized, "use_tma": bool(use_tma)},
        sort_keys=True,
        separators=(",", ":"),
    )


def select_robust_candidate(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    samples = defaultdict(lambda: defaultdict(list))
    all_workloads = set()
    for record in records:
        candidate = str(record["candidate"])
        workload = (str(record["profile"]), int(record["seed"]))
        latency = float(record["median_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(
                f"median_ms must be finite and positive, got {latency}"
            )
        samples[candidate][workload].append(latency)
        all_workloads.add(workload)

    complete = {
        candidate: {
            workload: statistics.median(latencies)
            for workload, latencies in workloads.items()
        }
        for candidate, workloads in samples.items()
        if set(workloads) == all_workloads
    }
    if not complete:
        raise ValueError(
            "no candidate has complete coverage of every profile and seed"
        )

    workload_best = {
        workload: min(latencies[workload] for latencies in complete.values())
        for workload in all_workloads
    }
    ranked = []
    for candidate, latencies in complete.items():
        regrets = [
            latencies[workload] / workload_best[workload] - 1.0
            for workload in sorted(all_workloads)
        ]
        values = list(latencies.values())
        ranked.append(
            (
                max(regrets),
                statistics.median(regrets),
                statistics.median(values),
                candidate,
                regrets,
                latencies,
            )
        )

    (
        max_regret,
        median_regret,
        median_latency,
        selected,
        regrets,
        latencies,
    ) = min(ranked, key=lambda item: item[:4])
    return {
        "candidate": selected,
        "max_regret": max_regret,
        "median_regret": median_regret,
        "median_ms": median_latency,
        "workload_count": len(all_workloads),
        "regrets": regrets,
        "latencies_ms": {
            f"{profile}/seed-{seed}": latency
            for (profile, seed), latency in sorted(latencies.items())
        },
    }


def shortlist_candidate_keys(
    records: Sequence[Mapping[str, Any]],
    per_workload: int,
) -> Tuple[str, ...]:
    if per_workload <= 0:
        raise ValueError(f"per_workload must be positive, got {per_workload}")
    if not records:
        raise ValueError("records must not be empty")

    samples = defaultdict(lambda: defaultdict(list))
    for record in records:
        workload = (str(record["profile"]), int(record["seed"]))
        candidate = str(record["candidate"])
        latency = float(record["median_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(
                f"median_ms must be finite and positive, got {latency}"
            )
        samples[workload][candidate].append(latency)

    selected = set()
    for candidates in samples.values():
        ranked = sorted(
            (
                statistics.median(latencies),
                candidate,
            )
            for candidate, latencies in candidates.items()
        )
        selected.update(candidate for _, candidate in ranked[:per_workload])
    return tuple(sorted(selected))


def shortlist_joint_candidate_keys(
    records: Sequence[Mapping[str, Any]],
    per_workload: int,
) -> Tuple[str, ...]:
    selected = set()
    for operation in ("up", "down"):
        operation_records = [
            record for record in records if record.get("operation") == operation
        ]
        if not operation_records:
            raise ValueError(f"missing {operation} timing records")
        selected.update(
            shortlist_candidate_keys(
                operation_records,
                per_workload=per_workload,
            )
        )
    return tuple(sorted(selected))


def select_robust_joint_pair(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    samples = {
        "up": defaultdict(lambda: defaultdict(list)),
        "down": defaultdict(lambda: defaultdict(list)),
    }
    block_m_by_candidate = {}
    all_workloads = set()
    for record in records:
        operation = str(record.get("operation"))
        if operation not in samples:
            raise ValueError(f"invalid operation {operation!r}")
        candidate = str(record["candidate"])
        workload = (str(record["profile"]), int(record["seed"]))
        latency = float(record["median_ms"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(
                f"median_ms must be finite and positive, got {latency}"
            )
        block_m = int(record["config"]["BLOCK_SIZE_M"])
        previous_block_m = block_m_by_candidate.setdefault(candidate, block_m)
        if previous_block_m != block_m:
            raise ValueError(
                f"candidate {candidate!r} has inconsistent BLOCK_SIZE_M"
            )
        samples[operation][candidate][workload].append(latency)
        all_workloads.add(workload)

    complete = {}
    for operation, candidates in samples.items():
        complete[operation] = {
            candidate: {
                workload: statistics.median(latencies)
                for workload, latencies in workloads.items()
            }
            for candidate, workloads in candidates.items()
            if set(workloads) == all_workloads
        }
        if not complete[operation]:
            raise ValueError(
                f"no {operation} candidate covers every profile and seed"
            )

    pairs = []
    for up_candidate, up_latencies in complete["up"].items():
        for down_candidate, down_latencies in complete["down"].items():
            block_m = block_m_by_candidate[up_candidate]
            if block_m != block_m_by_candidate[down_candidate]:
                continue
            latencies = {
                workload: up_latencies[workload] + down_latencies[workload]
                for workload in all_workloads
            }
            pairs.append(
                {
                    "up_candidate": up_candidate,
                    "down_candidate": down_candidate,
                    "block_m": block_m,
                    "latencies": latencies,
                }
            )
    if not pairs:
        raise ValueError(
            "no complete up/down candidate pair shares BLOCK_SIZE_M"
        )

    workload_best = {
        workload: min(pair["latencies"][workload] for pair in pairs)
        for workload in all_workloads
    }
    ranked = []
    for pair in pairs:
        regrets = [
            pair["latencies"][workload] / workload_best[workload] - 1.0
            for workload in sorted(all_workloads)
        ]
        values = list(pair["latencies"].values())
        ranked.append(
            (
                max(regrets),
                statistics.median(regrets),
                statistics.median(values),
                pair["block_m"],
                pair["up_candidate"],
                pair["down_candidate"],
                regrets,
                pair,
            )
        )

    (
        max_regret,
        median_regret,
        median_latency,
        _,
        _,
        _,
        regrets,
        selected,
    ) = min(ranked, key=lambda item: item[:6])
    return {
        "up_candidate": selected["up_candidate"],
        "down_candidate": selected["down_candidate"],
        "block_m": selected["block_m"],
        "max_regret": max_regret,
        "median_regret": median_regret,
        "median_ms": median_latency,
        "workload_count": len(all_workloads),
        "regrets": regrets,
        "latencies_ms": {
            f"{profile}/seed-{seed}": latency
            for (profile, seed), latency in sorted(
                selected["latencies"].items()
            )
        },
    }


def validate_anchor_sizes(
    batch_sizes: Sequence[int], full_search_size: int
) -> Tuple[int, ...]:
    if full_search_size <= 0:
        raise ValueError(
            f"full search size must be positive, got {full_search_size}"
        )
    normalized = tuple(sorted({int(size) for size in batch_sizes}))
    if not normalized or normalized[0] <= 0:
        raise ValueError("batch sizes must be positive")
    if full_search_size not in normalized:
        raise ValueError(
            f"full search size {full_search_size} must be present in batch sizes"
        )
    return normalized


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_route_contract(
    num_tokens: int,
    num_experts: int,
    topk: int,
    profile: str,
) -> None:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if not 0 < topk <= num_experts:
        raise ValueError(
            f"topk must be in [1, num_experts], got {topk=} and {num_experts=}"
        )
    if profile not in ROUTE_PROFILES:
        raise ValueError(
            f"profile must be one of {ROUTE_PROFILES}, got {profile!r}"
        )


def _expert_permutation(num_experts: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(
        num_experts,
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )


def _uniform_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    flat_indices = torch.arange(
        num_tokens * topk,
        dtype=torch.int64,
        device="cpu",
    )
    return permutation[flat_indices.remainder(num_experts)].view(num_tokens, topk)


def _synthetic_skew_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if topk == 1 and num_experts > 1:
        token_ids = torch.arange(
            num_tokens,
            dtype=torch.int64,
            device="cpu",
        )
        hot_pool_size = max(1, min(num_experts - 1, num_experts // 16))
        cold_pool_size = num_experts - hot_pool_size
        use_cold = token_ids.remainder(16).eq(0)
        hot_indices = token_ids.remainder(hot_pool_size)
        cold_indices = hot_pool_size + token_ids.div(16, rounding_mode="floor").remainder(
            cold_pool_size
        )
        indices = torch.where(use_cold, cold_indices, hot_indices)
        return permutation[indices].view(num_tokens, 1)

    cold_slots = max(1, topk // 4)
    hot_slots = topk - cold_slots
    hot_pool_size = min(
        max(hot_slots, num_experts // 16),
        num_experts - cold_slots,
    )
    cold_pool_size = num_experts - hot_pool_size

    token_ids = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device="cpu",
    ).unsqueeze(1)
    hot_offsets = torch.arange(
        hot_slots,
        dtype=torch.int64,
        device="cpu",
    ).unsqueeze(0)
    cold_offsets = torch.arange(
        cold_slots,
        dtype=torch.int64,
        device="cpu",
    ).unsqueeze(0)
    hot_indices = (token_ids * hot_slots + hot_offsets).remainder(hot_pool_size)
    cold_indices = hot_pool_size + (
        token_ids * cold_slots + cold_offsets
    ).remainder(cold_pool_size)
    return permutation[torch.cat((hot_indices, cold_indices), dim=1)]


def generate_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    profile: str,
    seed: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build deterministic generic routing for Triton MoE tuning.

    Expert IDs within one token are unique. IDs may repeat across tokens, as
    they do in real MoE routing. Generation happens on CPU so a given seed is
    reproducible independently of the target CUDA device.
    """

    _validate_route_contract(num_tokens, num_experts, topk, profile)
    permutation = _expert_permutation(num_experts, seed)
    if profile == "uniform":
        topk_ids = _uniform_topk_ids(
            num_tokens,
            num_experts,
            topk,
            permutation,
        )
    else:
        topk_ids = _synthetic_skew_topk_ids(
            num_tokens,
            num_experts,
            topk,
            permutation,
        )
    return topk_ids.to(device=device, dtype=torch.int32).contiguous()
