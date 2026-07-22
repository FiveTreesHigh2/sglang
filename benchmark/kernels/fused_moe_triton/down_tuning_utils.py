from __future__ import annotations

from typing import Union

import torch


ROUTE_PROFILES = ("uniform", "synthetic-skew")


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
    return torch.randperm(num_experts, generator=generator, dtype=torch.int64)


def _uniform_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    flat_indices = torch.arange(num_tokens * topk, dtype=torch.int64)
    return permutation[flat_indices.remainder(num_experts)].view(num_tokens, topk)


def _synthetic_skew_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if topk == 1 and num_experts > 1:
        token_ids = torch.arange(num_tokens, dtype=torch.int64)
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

    token_ids = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(1)
    hot_offsets = torch.arange(hot_slots, dtype=torch.int64).unsqueeze(0)
    cold_offsets = torch.arange(cold_slots, dtype=torch.int64).unsqueeze(0)
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
