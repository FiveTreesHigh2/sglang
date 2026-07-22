from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

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


@contextmanager
def tuner_directory_on_path():
    sys.path.insert(0, str(TUNER_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(TUNER_DIR))


def load_sep_tuner():
    path = TUNER_DIR / "tuning_fused_moe_triton_sep.py"
    spec = importlib.util.spec_from_file_location("tuning_fused_moe_triton_sep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    ray = types.ModuleType("ray")
    ray_experimental = types.ModuleType("ray.experimental")
    ray_tqdm = types.ModuleType("ray.experimental.tqdm_ray")
    ray_tqdm.tqdm = lambda iterable: iterable
    ray.experimental = ray_experimental
    ray_experimental.tqdm_ray = ray_tqdm
    optional_modules = {
        "ray": ray,
        "ray.experimental": ray_experimental,
        "ray.experimental.tqdm_ray": ray_tqdm,
    }
    try:
        with (
            tuner_directory_on_path(),
            mock.patch.dict(sys.modules, optional_modules),
        ):
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


def test_candidate_key_is_canonical_and_includes_tma_mode() -> None:
    utils = load_down_tuning_utils()
    left = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    }
    right = dict(reversed(tuple(left.items())))

    assert utils.candidate_key(left, False) == utils.candidate_key(right, False)
    assert utils.candidate_key(left, False) != utils.candidate_key(left, True)


def test_robust_selection_minimizes_worst_profile_regret() -> None:
    utils = load_down_tuning_utils()
    records = [
        {
            "candidate": "fast-uniform",
            "profile": "uniform",
            "seed": 0,
            "median_ms": 1.00,
        },
        {
            "candidate": "fast-uniform",
            "profile": "synthetic-skew",
            "seed": 0,
            "median_ms": 1.50,
        },
        {
            "candidate": "robust",
            "profile": "uniform",
            "seed": 0,
            "median_ms": 1.05,
        },
        {
            "candidate": "robust",
            "profile": "synthetic-skew",
            "seed": 0,
            "median_ms": 1.08,
        },
    ]

    selection = utils.select_robust_candidate(records)

    assert selection["candidate"] == "robust"
    assert selection["max_regret"] == pytest.approx(0.05)
    assert selection["workload_count"] == 2


def test_robust_selection_rejects_incomplete_candidates() -> None:
    utils = load_down_tuning_utils()
    records = [
        {
            "candidate": "complete",
            "profile": "uniform",
            "seed": 0,
            "median_ms": 1.0,
        },
        {
            "candidate": "complete",
            "profile": "synthetic-skew",
            "seed": 0,
            "median_ms": 1.1,
        },
        {
            "candidate": "incomplete",
            "profile": "uniform",
            "seed": 0,
            "median_ms": 0.1,
        },
    ]

    assert utils.select_robust_candidate(records)["candidate"] == "complete"


def test_anchor_validation_sorts_deduplicates_and_requires_full_search() -> None:
    utils = load_down_tuning_utils()

    assert utils.validate_anchor_sizes(
        [8192, 1, 8, 4096, 8192],
        8192,
    ) == (1, 8, 4096, 8192)
    with pytest.raises(ValueError, match="full search"):
        utils.validate_anchor_sizes([2048, 4096], 8192)
    with pytest.raises(ValueError, match="positive"):
        utils.validate_anchor_sizes([0, 8192], 8192)


def test_atomic_json_writer_creates_parent_and_complete_document() -> None:
    utils = load_down_tuning_utils()
    payload = {
        "8192": {
            "BLOCK_SIZE_M": 64,
            "USE_TMA": True,
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "nested" / "down.json"
        utils.write_json_atomic(output, payload)

        assert json.loads(output.read_text()) == payload
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_cli_accepts_approved_down_tuning_contract() -> None:
    sep = load_sep_tuner()

    args = sep.parse_args(
        [
            "--model",
            "/model",
            "--tp-size",
            "1",
            "--ep-size",
            "1",
            "--dtype",
            "fp8_w8a8",
            "--kernel",
            "down",
            "--batch-sizes",
            "1",
            "8",
            "32",
            "128",
            "512",
            "2048",
            "4096",
            "6144",
            "8192",
            "--route-profiles",
            "uniform",
            "synthetic-skew",
            "--route-seeds",
            "0",
            "1",
            "2",
            "--full-search-size",
            "8192",
            "--shortlist-size",
            "16",
            "--output",
            "/tmp/down.json",
            "--tune",
        ]
    )

    assert args.kernel == "down"
    assert args.batch_sizes == [1, 8, 32, 128, 512, 2048, 4096, 6144, 8192]
    assert args.route_profiles == ["uniform", "synthetic-skew"]
    assert args.route_seeds == [0, 1, 2]
    assert args.full_search_size == 8192
    assert args.shortlist_size == 16
    assert args.output == "/tmp/down.json"
    assert args.topk_ids_dir is None


def test_cli_rejects_conflicting_batch_size_forms() -> None:
    sep = load_sep_tuner()

    with pytest.raises(SystemExit):
        sep.parse_args(
            [
                "--batch-size",
                "8192",
                "--batch-sizes",
                "4096",
                "8192",
            ]
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--kernel", "down", "--tune", "--batch-sizes", "8192"],
        [
            "--kernel",
            "down",
            "--tune",
            "--batch-sizes",
            "4096",
            "--full-search-size",
            "8192",
            "--output",
            "/tmp/down.json",
        ],
    ],
)
def test_cli_rejects_incomplete_down_tuning_contract(argv: list[str]) -> None:
    sep = load_sep_tuner()

    with pytest.raises(SystemExit):
        sep.parse_args(argv)
