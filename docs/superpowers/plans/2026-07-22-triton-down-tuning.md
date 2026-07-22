# Pro5000 Triton MoE Down/GEMM2 Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend SGLang's separate Triton MoE tuner with a reproducible down-only workflow that generates a robust `_down.json` for saturated 8192-token prefill batches without relying on captured request routes.

**Architecture:** Keep GPU allocation and kernel invocation in `tuning_fused_moe_triton_sep.py`, but place deterministic route generation, candidate identity, robust scoring, anchor validation, and atomic JSON writing in a small testable helper module. Run a full down/TMA search at 8192, retain a shortlist, and evaluate that shortlist plus defaults at secondary and small anchor sizes.

**Tech Stack:** Python 3.12, PyTorch 2.11, Triton 3.6, SGLang fused MoE Triton kernels, `unittest`/pytest, JSON.

## Global Constraints

- Production Triton kernels and FlashInfer code must not change.
- Target runtime contract is SM120, `E=256`, `top_k=8`, `fp8_w8a8`, `block_shape=[128, 128]`, GEMM2 K=512 and N=2048.
- `chunked_prefill_size=8192`; the full search target is `num_tokens=8192`, or 65536 routed rows.
- Route selection uses deterministic `uniform` and `synthetic-skew` profiles, not captured request data.
- Large anchors are exactly `2048, 4096, 6144, 8192`; small protection anchors are exactly `1, 8, 32, 128, 512`.
- Output keys are original token counts, never routed-row counts.
- A generated `_down.json` is written to an explicit output path before installation.
- Existing CLI behavior remains available when `--kernel both` and `--topk-ids-dir` are used.

---

### Task 1: Pure down-tuning utilities

**Files:**
- Create: `benchmark/kernels/fused_moe_triton/down_tuning_utils.py`
- Create: `test/registered/unit/test_triton_moe_down_tuning.py`

**Interfaces:**
- Produces: `generate_topk_ids(num_tokens: int, num_experts: int, topk: int, profile: str, seed: int, device: str | torch.device = "cpu") -> torch.Tensor`
- Produces: `candidate_key(config: Mapping[str, Any], use_tma: bool) -> str`
- Produces: `select_robust_candidate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]`
- Produces: `validate_anchor_sizes(batch_sizes: Sequence[int], full_search_size: int) -> Tuple[int, ...]`
- Produces: `write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`

- [ ] **Step 1: Write route-generation failing tests**

Add tests that load `down_tuning_utils.py` from the benchmark directory and assert:

```python
def test_uniform_routes_are_deterministic_balanced_and_unique_per_token():
    first = utils.generate_topk_ids(8192, 256, 8, "uniform", 7)
    second = utils.generate_topk_ids(8192, 256, 8, "uniform", 7)
    assert torch.equal(first, second)
    assert first.shape == (8192, 8)
    assert first.dtype == torch.int32
    assert torch.all(torch.sort(first, dim=1).values[:, 1:] != torch.sort(first, dim=1).values[:, :-1])
    counts = torch.bincount(first.to(torch.int64).flatten(), minlength=256)
    assert int(counts.max() - counts.min()) <= 1


def test_synthetic_skew_routes_are_deterministic_skewed_and_unique_per_token():
    routes = utils.generate_topk_ids(8192, 256, 8, "synthetic-skew", 11)
    repeated = utils.generate_topk_ids(8192, 256, 8, "synthetic-skew", 11)
    assert torch.equal(routes, repeated)
    assert torch.all(torch.sort(routes, dim=1).values[:, 1:] != torch.sort(routes, dim=1).values[:, :-1])
    counts = torch.bincount(routes.to(torch.int64).flatten(), minlength=256)
    assert counts.max().item() > 4 * counts.float().mean().item()
    assert torch.count_nonzero(counts).item() == 256
```

- [ ] **Step 2: Run route tests to verify RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k route -q
```

Expected: FAIL because `down_tuning_utils.py` does not exist.

- [ ] **Step 3: Implement deterministic route generation**

Implement validation for `num_tokens > 0`, `num_experts > 0`, and `0 < topk <= num_experts`. Use a seed-specific expert permutation. Uniform routing advances through the permutation in groups of `topk`; skew routing selects six slots from a hot pool and two from the cold pool for `topk=8`, generalized so both pools remain valid for other `topk` values. Return contiguous `torch.int32` on the requested device.

- [ ] **Step 4: Write scoring and anchor failing tests**

Add:

```python
def test_robust_selection_minimizes_worst_profile_regret():
    records = [
        {"candidate": "fast-uniform", "profile": "uniform", "seed": 0, "median_ms": 1.00},
        {"candidate": "fast-uniform", "profile": "synthetic-skew", "seed": 0, "median_ms": 1.50},
        {"candidate": "robust", "profile": "uniform", "seed": 0, "median_ms": 1.05},
        {"candidate": "robust", "profile": "synthetic-skew", "seed": 0, "median_ms": 1.08},
    ]
    assert utils.select_robust_candidate(records)["candidate"] == "robust"


def test_anchor_validation_requires_full_search_size_and_unique_positive_values():
    assert utils.validate_anchor_sizes([1, 8, 8192, 4096, 8192], 8192) == (1, 8, 4096, 8192)
    with pytest.raises(ValueError, match="full search"):
        utils.validate_anchor_sizes([2048, 4096], 8192)
```

- [ ] **Step 5: Implement robust selection and atomic output**

Normalize each candidate latency by the best latency for the same `(profile, seed)` workload. Rank candidates by `(maximum_regret, median_regret, median_latency, candidate_key)`. Reject incomplete candidates that do not cover the same workload set. Write JSON through a sibling temporary file followed by `Path.replace`.

- [ ] **Step 6: Run Task 1 tests**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add benchmark/kernels/fused_moe_triton/down_tuning_utils.py test/registered/unit/test_triton_moe_down_tuning.py
git commit -m "test: define robust Triton down tuning utilities"
```

---

### Task 2: CLI contract and synthetic route source

**Files:**
- Modify: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- Modify: `test/registered/unit/test_triton_moe_down_tuning.py`

**Interfaces:**
- Consumes: `generate_topk_ids`, `validate_anchor_sizes` from Task 1.
- Produces: `parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace`
- Produces: `build_topk_ids_list(..., topk_ids_dir: Optional[str], route_profiles: Sequence[str], route_seeds: Sequence[int], num_samples: int) -> Dict[str, List[torch.Tensor]]`

- [ ] **Step 1: Write failing CLI tests**

Load `tuning_fused_moe_triton_sep.py` with GPU-heavy modules mocked and assert:

```python
args = sep.parse_args([
    "--model", "/model",
    "--tp-size", "1",
    "--ep-size", "1",
    "--dtype", "fp8_w8a8",
    "--kernel", "down",
    "--batch-sizes", "1", "8", "32", "128", "512", "2048", "4096", "6144", "8192",
    "--route-profiles", "uniform", "synthetic-skew",
    "--route-seeds", "0", "1", "2",
    "--full-search-size", "8192",
    "--shortlist-size", "16",
    "--output", "/tmp/down.json",
    "--tune",
])
assert args.kernel == "down"
assert args.batch_sizes[-1] == 8192
assert args.topk_ids_dir is None
```

Also assert `--batch-size` and `--batch-sizes` are mutually exclusive, and synthetic mode rejects an unknown profile.

- [ ] **Step 2: Run CLI tests to verify RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k cli -q
```

Expected: FAIL because the new parser options are absent.

- [ ] **Step 3: Move parser construction into `parse_args` and add options**

Keep every old argument. Add:

```python
parser.add_argument("--kernel", choices=("up", "down", "both"), default="both")
batch_group = parser.add_mutually_exclusive_group()
batch_group.add_argument("--batch-size", type=int)
batch_group.add_argument("--batch-sizes", type=int, nargs="+")
parser.add_argument("--route-profiles", nargs="+", choices=("uniform", "synthetic-skew"), default=("uniform", "synthetic-skew"))
parser.add_argument("--route-seeds", type=int, nargs="+", default=(0, 1, 2))
parser.add_argument("--full-search-size", type=int)
parser.add_argument("--shortlist-size", type=int, default=16)
parser.add_argument("--output", type=str)
parser.add_argument("--topk-ids-dir")
```

For `--kernel down --tune`, require `--batch-sizes`, `--output`, and a full-search size present in the batch-size list. Preserve the legacy requirement for `--topk-ids-dir` only when the user chooses captured-route mode explicitly.

- [ ] **Step 4: Replace hard-coded route loading at the call boundary**

Add `build_topk_ids_list`. If `topk_ids_dir` is present, call the existing `load_topk_ids`; otherwise generate enough deterministic synthetic tensors for each requested `(profile, seed)`. Keep the legacy loader unchanged for backward compatibility.

- [ ] **Step 5: Run CLI and route-source tests**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k 'cli or source' -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py test/registered/unit/test_triton_moe_down_tuning.py
git commit -m "feat: add generic Triton down tuning CLI"
```

---

### Task 3: Down-only GPU benchmark and independent TMA selection

**Files:**
- Modify: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- Modify: `test/registered/unit/test_triton_moe_down_tuning.py`

**Interfaces:**
- Consumes: route workloads built in Task 2.
- Produces: `benchmark_config(..., kernel: str = "both") -> Dict[str, float]` with keys selected from `up`, `up_tma`, `down`, `down_tma`.
- Produces: `BenchmarkWorker.benchmark_down_candidate(...) -> List[Dict[str, Any]]`.

- [ ] **Step 1: Write failing source-contract tests**

Use injected fake `KernelWrapper` instances and assert that `kernel="down"` invokes only the down non-TMA and down TMA wrappers. Assert the returned timing record contains both variants and no up timing.

- [ ] **Step 2: Run the down-only test to verify RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k down_only -q
```

Expected: FAIL because all four wrappers are currently created and timed.

- [ ] **Step 3: Refactor `benchmark_config` around selected wrappers**

Build only the requested wrappers:

```python
wrappers = {}
if kernel in ("up", "both"):
    wrappers["up"] = make_up(use_tma=False)
    wrappers["up_tma"] = make_up(use_tma=True)
if kernel in ("down", "both"):
    wrappers["down"] = make_down(use_tma=False)
    wrappers["down_tma"] = make_down(use_tma=True)
```

Warm and time only `wrappers.values()`. Return microseconds per invocation in a dictionary. Keep a compatibility adapter for legacy callers that expect the four-value tuple.

- [ ] **Step 4: Remove GEMM1 coupling from down selection**

For down-only tuning, rank every valid `(config, USE_TMA)` globally. Do not choose a shared `BLOCK_SIZE_M` by summing GEMM1 and GEMM2 costs. Catch `OutOfResources` and candidate-local `RuntimeError`, record the rejection, and continue; fail the run if no candidate completes.

- [ ] **Step 5: Add a GPU preflight command to the README test fixture**

The command must evaluate one configuration at `num_tokens=32` with both TMA states and print structured timing keys. It must not write a production config.

- [ ] **Step 6: Run CPU tests and compile checks**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -q
python3 -m py_compile benchmark/kernels/fused_moe_triton/down_tuning_utils.py benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py
```

Expected: PASS and no syntax errors.

- [ ] **Step 7: Commit Task 3**

```bash
git add benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py test/registered/unit/test_triton_moe_down_tuning.py
git commit -m "feat: benchmark Triton MoE down independently"
```

---

### Task 4: Staged search, shortlist reuse, and `_down.json` output

**Files:**
- Modify: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- Modify: `benchmark/kernels/fused_moe_triton/down_tuning_utils.py`
- Modify: `test/registered/unit/test_triton_moe_down_tuning.py`

**Interfaces:**
- Consumes: timing records and robust selector from Tasks 1–3.
- Produces: `run_down_tuning(...) -> Dict[str, Any]` containing `configs`, `raw_timings`, `rejections`, and `metadata`.
- Produces: runtime `_down.json` whose values contain normal Triton config keys plus `USE_TMA`.

- [ ] **Step 1: Write failing staged-search tests with a fake benchmark callback**

Assert that:

- every full-search candidate is evaluated at 8192;
- only the shortlist/default/neighborhood union is evaluated at other anchors;
- every output M has exactly one selected config;
- the output includes `1, 8, 32, 128, 512, 2048, 4096, 6144, 8192`;
- raw timing metadata is retained separately from the runtime config.

- [ ] **Step 2: Run staged-search tests to verify RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k staged -q
```

Expected: FAIL because no staged orchestrator exists.

- [ ] **Step 3: Implement coarse search and shortlist construction**

At 8192, evaluate the full valid search space with a reduced iteration count. Keep the best `shortlist_size` candidates for each route profile and take the union by canonical candidate key. Include TMA state in the key.

- [ ] **Step 4: Implement stable rerun and robust selection**

Rerun the union across all profiles and route seeds with the stable iteration count. Use `select_robust_candidate` to choose the 8192 winner. For secondary and small anchors, evaluate only the shortlist, the SGLang default heuristic config, and one-step legal variations of `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps`, and `num_stages` already present in the canonical search space.

- [ ] **Step 5: Write runtime and evidence JSON files**

Write the runtime mapping to `--output`. Write the evidence payload next to it as `<stem>.timings.json`. Runtime values must pass `sort_config` and include `USE_TMA`. Metadata must include model path, device name, torch/triton versions, route profiles, route seeds, batch sizes, full-search size, search-space size, rejected candidate count, and command line.

- [ ] **Step 6: Run all tuner unit tests**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add benchmark/kernels/fused_moe_triton/down_tuning_utils.py benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py test/registered/unit/test_triton_moe_down_tuning.py
git commit -m "feat: stage robust Triton down config search"
```

---

### Task 5: Documentation and end-to-end verification handoff

**Files:**
- Modify: `benchmark/kernels/fused_moe_triton/README.md`
- Modify: `test/registered/unit/test_triton_moe_down_tuning.py`

**Interfaces:**
- Consumes: final CLI from Tasks 2–4.
- Produces: exact server preflight, formal tuning, config-install, and A/B verification commands.

- [ ] **Step 1: Add README command-contract test**

Assert the README contains `--kernel down`, all nine anchors, both route profiles, `--full-search-size 8192`, an explicit `--output`, and `SGLANG_MOE_CONFIG_DIR` A/B instructions.

- [ ] **Step 2: Run documentation test to verify RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -k readme -q
```

Expected: FAIL because the new workflow is undocumented.

- [ ] **Step 3: Document the server workflow**

Use the server venv and local model path. The formal command must be equivalent to:

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
MODEL_PATH=/home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/

"${VENV_PY}" benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
  --model "${MODEL_PATH}" \
  --tp-size 1 \
  --ep-size 1 \
  --dtype fp8_w8a8 \
  --kernel down \
  --batch-sizes 1 8 32 128 512 2048 4096 6144 8192 \
  --route-profiles uniform synthetic-skew \
  --route-seeds 0 1 2 \
  --full-search-size 8192 \
  --shortlist-size 16 \
  --output /home/logs/sennian/pro5000-fi-moe/runs/triton-down/down.json \
  --tune
```

Document how to point `SGLANG_MOE_CONFIG_DIR` at an isolated A/B directory before copying the accepted file into `configs/triton_3_6_0/`.

- [ ] **Step 4: Run the complete local test set**

Run:

```bash
python3 -m pytest test/registered/unit/test_triton_moe_down_tuning.py -q
python3 -m py_compile benchmark/kernels/fused_moe_triton/down_tuning_utils.py benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py
git diff --check
```

Expected: all tests pass, compile succeeds, and `git diff --check` is silent.

- [ ] **Step 5: Commit Task 5**

```bash
git add benchmark/kernels/fused_moe_triton/README.md test/registered/unit/test_triton_moe_down_tuning.py
git commit -m "docs: add Pro5000 Triton down tuning workflow"
```

- [ ] **Step 6: Server verification checkpoint**

The user runs the documented preflight first. Proceed to the formal search only after preflight reports timings for both `down` and `down_tma`, the detected config filename contains `E=256,N=512`, and the output path is writable. Treat GPU OOM, all-candidate rejection, missing TMA support, or a model-shape mismatch as a blocking error rather than silently writing a partial config.
