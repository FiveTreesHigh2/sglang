from __future__ import annotations

import functools
import importlib.util
import inspect
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


@functools.lru_cache(maxsize=1)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_route_generation_ignores_global_cuda_default_device() -> None:
    utils = load_down_tuning_utils()

    torch.set_default_device("cuda")
    try:
        routes = utils.generate_topk_ids(
            64,
            256,
            8,
            "uniform",
            seed=3,
            device="cpu",
        )
    finally:
        torch.set_default_device("cpu")

    assert routes.device.type == "cpu"
    assert routes.shape == (64, 8)


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


def test_synthetic_route_source_builds_reproducible_profile_seed_workloads() -> None:
    sep = load_sep_tuner()

    first = sep.build_topk_ids_list(
        num_tokens=64,
        num_experts=256,
        topk=8,
        topk_ids_dir=None,
        route_profiles=["uniform", "synthetic-skew"],
        route_seeds=[0, 1],
        num_samples=3,
    )
    second = sep.build_topk_ids_list(
        num_tokens=64,
        num_experts=256,
        topk=8,
        topk_ids_dir=None,
        route_profiles=["uniform", "synthetic-skew"],
        route_seeds=[0, 1],
        num_samples=3,
    )

    assert tuple(first) == (
        "uniform/seed-0",
        "uniform/seed-1",
        "synthetic-skew/seed-0",
        "synthetic-skew/seed-1",
    )
    for workload, samples in first.items():
        assert len(samples) == 3
        assert all(sample.shape == (64, 8) for sample in samples)
        assert all(sample.dtype == torch.int32 for sample in samples)
        assert all(
            torch.equal(left, right)
            for left, right in zip(samples, second[workload])
        )
    assert not torch.equal(
        first["uniform/seed-0"][0],
        first["uniform/seed-1"][0],
    )


def test_captured_route_source_preserves_legacy_loader(monkeypatch) -> None:
    sep = load_sep_tuner()
    captured = [
        torch.full((64, 8), index, dtype=torch.int32)
        for index in range(3)
    ]
    calls = []

    def fake_load(directory: str, index: int) -> torch.Tensor:
        calls.append((directory, index))
        return captured[index]

    monkeypatch.setattr(sep, "load_topk_ids", fake_load)
    sources = sep.build_topk_ids_list(
        num_tokens=64,
        num_experts=256,
        topk=8,
        topk_ids_dir="/routes",
        route_profiles=["uniform"],
        route_seeds=[0],
        num_samples=3,
    )

    assert tuple(sources) == ("captured",)
    assert all(
        actual is expected
        for actual, expected in zip(sources["captured"], captured)
    )
    assert calls == [("/routes", 0), ("/routes", 1), ("/routes", 2)]


def test_down_only_wrapper_selection_constructs_and_times_down_variants() -> None:
    sep = load_sep_tuner()
    construction_calls = []

    class FakeWrapper:
        def __init__(self, cost: float):
            self.cost = cost
            self.forward_calls = 0

        def forward_cost(self) -> float:
            self.forward_calls += 1
            return self.cost

    costs = {
        ("up", False): 2.0,
        ("up", True): 1.8,
        ("down", False): 1.0,
        ("down", True): 0.8,
    }

    def factory(operation: str, use_tma: bool) -> FakeWrapper:
        construction_calls.append((operation, use_tma))
        return FakeWrapper(costs[(operation, use_tma)])

    wrappers = sep.build_selected_kernel_wrappers("down", factory)
    prepare_calls = []
    timings = sep.benchmark_kernel_wrappers(
        wrappers,
        prepare=lambda index, inner_iter: prepare_calls.append(
            (index, inner_iter)
        ),
        num_iters=20,
        inner_iter=10,
        warmup=True,
    )

    assert construction_calls == [("down", False), ("down", True)]
    assert tuple(wrappers) == ("down", "down_tma")
    assert timings == pytest.approx({"down": 100.0, "down_tma": 80.0})
    assert prepare_calls == [(0, 10), (1, 10)]
    assert all(wrapper.forward_calls == 3 for wrapper in wrappers.values())


def test_wrapper_benchmark_reports_median_instead_of_mean() -> None:
    sep = load_sep_tuner()

    class VariableWrapper:
        def __init__(self):
            self.costs = iter((1.0, 9.0, 1.0))

        def forward_cost(self) -> float:
            return next(self.costs)

    timings = sep.benchmark_kernel_wrappers(
        {"down": VariableWrapper()},
        prepare=lambda index, inner_iter: None,
        num_iters=30,
        inner_iter=10,
        warmup=False,
    )

    assert timings == pytest.approx({"down": 100.0})


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ("up", (("up", False), ("up", True))),
        ("both", (("up", False), ("up", True), ("down", False), ("down", True))),
    ],
)
def test_wrapper_selection_preserves_up_and_legacy_both_modes(
    kernel: str,
    expected: tuple[tuple[str, bool], ...],
) -> None:
    sep = load_sep_tuner()
    calls = []

    def factory(operation: str, use_tma: bool):
        calls.append((operation, use_tma))
        return object()

    sep.build_selected_kernel_wrappers(kernel, factory)

    assert tuple(calls) == expected


def test_benchmark_config_routes_construction_and_timing_through_kernel_selection() -> None:
    sep = load_sep_tuner()

    signature = inspect.signature(sep.benchmark_config)
    assert signature.parameters["kernel"].default == "both"
    source = inspect.getsource(sep.benchmark_config)
    assert "build_selected_kernel_wrappers(kernel, wrapper_factory)" in source
    assert "benchmark_kernel_wrappers(" in source
    assert "kernel0, kernel1 = get_kernel_wrapper" not in source


def test_down_timing_records_name_tma_candidates_and_convert_us_to_ms() -> None:
    sep = load_sep_tuner()
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    }

    records = sep.down_timing_records(
        config,
        workload="uniform/seed-7",
        timings_us={"down": 100.0, "down_tma": 80.0},
    )

    assert [record["use_tma"] for record in records] == [False, True]
    assert [record["median_ms"] for record in records] == pytest.approx(
        [0.1, 0.08]
    )
    assert all(record["profile"] == "uniform" for record in records)
    assert all(record["seed"] == 7 for record in records)
    assert records[0]["candidate"] != records[1]["candidate"]


def test_down_candidate_worker_uses_generated_workloads_and_down_kernel() -> None:
    sep = load_sep_tuner()

    source = inspect.getsource(sep.BenchmarkWorker.benchmark_down_candidate)
    assert "build_topk_ids_list(" in source
    assert 'kernel="down"' in source
    assert "down_timing_records(" in source


def test_single_gpu_tuner_does_not_require_ray_at_module_import() -> None:
    source = (TUNER_DIR / "tuning_fused_moe_triton_sep.py").read_text()
    module_preamble = source.split("def main", maxsplit=1)[0]

    assert "\nimport ray\n" not in module_preamble
    assert "from tqdm import tqdm" in module_preamble


def test_profile_shortlist_keeps_each_workload_winner() -> None:
    utils = load_down_tuning_utils()
    records = [
        {"candidate": "uniform-best", "profile": "uniform", "seed": 0, "median_ms": 1.0},
        {"candidate": "uniform-best", "profile": "synthetic-skew", "seed": 0, "median_ms": 2.0},
        {"candidate": "skew-best", "profile": "uniform", "seed": 0, "median_ms": 1.5},
        {"candidate": "skew-best", "profile": "synthetic-skew", "seed": 0, "median_ms": 0.9},
    ]

    assert utils.shortlist_candidate_keys(records, per_workload=1) == (
        "skew-best",
        "uniform-best",
    )


def test_staged_search_runs_full_space_then_reuses_shortlist_for_anchors() -> None:
    sep = load_sep_tuner()
    utils = load_down_tuning_utils()
    config_a = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    config_b = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    }
    default_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }
    calls = []

    def benchmark(
        num_tokens,
        config,
        profiles,
        seeds,
        num_iters,
        phase,
    ):
        calls.append(
            (
                phase,
                num_tokens,
                config["BLOCK_SIZE_M"],
                tuple(profiles),
                tuple(seeds),
                num_iters,
            )
        )
        records = []
        for profile in profiles:
            for seed in seeds:
                if profile == "uniform":
                    base = 1.0 if config == config_a else 1.2
                else:
                    base = 0.9 if config == config_b else 1.3
                for use_tma, factor in ((False, 1.0), (True, 0.95)):
                    records.append(
                        {
                            "candidate": utils.candidate_key(config, use_tma),
                            "config": dict(config),
                            "use_tma": use_tma,
                            "profile": profile,
                            "seed": seed,
                            "median_ms": base * factor,
                        }
                    )
        return records

    result = sep.run_staged_down_search(
        batch_sizes=[1, 8192],
        full_search_size=8192,
        search_space=[config_a, config_b],
        default_configs={1: default_config, 8192: default_config},
        route_profiles=["uniform", "synthetic-skew"],
        route_seeds=[0, 1],
        shortlist_size=1,
        coarse_iters=20,
        stable_iters=100,
        benchmark=benchmark,
    )

    assert set(result["configs"]) == {"1", "8192"}
    assert all("USE_TMA" in config for config in result["configs"].values())
    assert len(result["raw_timings"]["coarse"]) == 8
    assert any(call[0] == "stable-full" for call in calls)
    anchor_calls = [call for call in calls if call[0] == "stable-anchor"]
    assert anchor_calls
    assert all(call[1] == 1 for call in anchor_calls)
    assert all(call[3] == ("uniform", "synthetic-skew") for call in calls)
    assert all(call[4] == (0,) for call in calls if call[0] == "coarse")
    assert all(call[4] == (0, 1) for call in calls if call[0] != "coarse")


def test_cli_accepts_staged_search_runtime_controls() -> None:
    sep = load_sep_tuner()

    args = sep.parse_args(
        [
            "--kernel",
            "down",
            "--tune",
            "--batch-sizes",
            "1",
            "8192",
            "--output",
            "/tmp/down.json",
            "--coarse-iters",
            "20",
            "--stable-iters",
            "100",
            "--max-configs",
            "8",
        ]
    )

    assert args.coarse_iters == 20
    assert args.stable_iters == 100
    assert args.max_configs == 8


def test_joint_selection_uses_one_block_m_and_minimizes_total_latency() -> None:
    utils = load_down_tuning_utils()

    def record(operation, block_m, latency):
        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        }
        return {
            "operation": operation,
            "candidate": utils.candidate_key(config, operation == "down"),
            "config": config,
            "use_tma": operation == "down",
            "profile": "uniform",
            "seed": 0,
            "median_ms": latency,
        }

    selection = utils.select_robust_joint_pair(
        [
            record("up", 32, 1.0),
            record("down", 32, 1.2),
            record("up", 64, 1.4),
            record("down", 64, 0.7),
        ]
    )

    assert selection["block_m"] == 64
    assert selection["median_ms"] == pytest.approx(2.1)


def test_cli_accepts_single_gpu_joint_tuning() -> None:
    sep = load_sep_tuner()

    args = sep.parse_args(
        [
            "--kernel",
            "joint",
            "--tune",
            "--batch-sizes",
            "1",
            "8192",
            "--output-dir",
            "/tmp/joint",
        ]
    )

    assert args.kernel == "joint"
    assert args.output_dir == "/tmp/joint"
