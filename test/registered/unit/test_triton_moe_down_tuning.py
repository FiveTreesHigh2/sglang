from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TUNER_DIR = REPO_ROOT / "benchmark" / "kernels" / "fused_moe_triton"


def load_down_tuning_utils():
    path = TUNER_DIR / "down_tuning_utils.py"
    spec = importlib.util.spec_from_file_location("down_tuning_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def assert_unique_experts_per_token(topk_ids: torch.Tensor) -> None:
    ordered = torch.sort(topk_ids, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1])


def test_uniform_route_is_deterministic_balanced_and_unique_per_token() -> None:
    utils = load_down_tuning_utils()

    first = utils.generate_topk_ids(8192, 256, 8, "uniform", 7)
    second = utils.generate_topk_ids(8192, 256, 8, "uniform", 7)

    assert torch.equal(first, second)
    assert first.shape == (8192, 8)
    assert first.dtype == torch.int32
    assert first.is_contiguous()
    assert_unique_experts_per_token(first)
    counts = torch.bincount(first.to(torch.int64).flatten(), minlength=256)
    assert int(counts.max() - counts.min()) <= 1


def test_synthetic_skew_route_is_deterministic_skewed_and_unique() -> None:
    utils = load_down_tuning_utils()

    routes = utils.generate_topk_ids(8192, 256, 8, "synthetic-skew", 11)
    repeated = utils.generate_topk_ids(8192, 256, 8, "synthetic-skew", 11)

    assert torch.equal(routes, repeated)
    assert routes.shape == (8192, 8)
    assert routes.dtype == torch.int32
    assert routes.is_contiguous()
    assert_unique_experts_per_token(routes)
    counts = torch.bincount(routes.to(torch.int64).flatten(), minlength=256)
    assert counts.max().item() > 4 * counts.float().mean().item()
    assert torch.count_nonzero(counts).item() == 256


@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "topk", "profile", "message"),
    [
        (0, 256, 8, "uniform", "num_tokens"),
        (8, 0, 8, "uniform", "num_experts"),
        (8, 256, 0, "uniform", "topk"),
        (8, 4, 8, "uniform", "topk"),
        (8, 256, 8, "request-capture", "profile"),
    ],
)
def test_route_generation_rejects_invalid_contract(
    num_tokens: int,
    num_experts: int,
    topk: int,
    profile: str,
    message: str,
) -> None:
    utils = load_down_tuning_utils()

    with pytest.raises(ValueError, match=message):
        utils.generate_topk_ids(
            num_tokens,
            num_experts,
            topk,
            profile,
            seed=0,
        )
