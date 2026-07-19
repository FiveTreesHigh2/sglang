# FlashInfer SM120 FP8 routed-MoE integration design

**Status:** Approved for implementation planning
**Date:** 2026-07-19
**Target branch:** `feat/flashinfer-sm120-fp8-moe`
**SGLang base:** `8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587`
**Target model:** Qwen3.5-35B-A3B-FP8
**Target GPU:** NVIDIA RTX PRO 5000 72GB Blackwell, compute capability 12.0

## 1. Outcome

Add an experimental SGLang routed-MoE backend that invokes FlashInfer's existing
`flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise` SM120 CuTe kernel for both expert
GEMMs. The integration must preserve SGLang's existing model loading, routing, shared
expert, and fallback paths. It must not copy, fork, or reimplement the FlashInfer GEMM
kernel.

The initial backend is explicitly selected with:

```text
--moe-runner-backend flashinfer_sm120_fp8
```

`auto` continues to choose the existing backend until the new path has passed the
correctness and performance gates in this document.

## 2. Motivation and baseline

The target model has 40 MoE layers, 256 routed experts, top-k 8, hidden size 2048,
expert intermediate size 512, and one shared expert. Its checkpoint uses FP8 E4M3
weights with float32 `[128, 128]` block scales and dynamic per-token activation
quantization with K-group 128.

The measured Triton baseline for a chunked-prefill workload is:

| Operation | Effective grouped shape | Latency |
| --- | --- | ---: |
| GEMM1 / w13 | M=65536, N=1024, K=2048 | 1.267 ms |
| GEMM2 / w2 | M=65536, N=2048, K=512 | 0.794 ms |
| Routed MoE across 40 layers | chunk size 8192 | about 82 ms |

FlashInfer's target entry consumes the same weight quantization granularity and uses
token-packed zero-padding mode, avoiding Triton's expert `BLOCK_M` activation padding.

## 3. Fixed versions and repository policy

### 3.1 Git layout

- `origin`: `git@github.com:FiveTreesHigh2/sglang.git`
- `upstream`: `https://github.com/sgl-project/sglang.git`
- `origin/main` remains an upstream mirror.
- All work happens on `feat/flashinfer-sm120-fp8-moe` from the fixed SGLang base.
- The server never receives manual code edits. It fetches the fork and checks out an
  exact commit SHA in detached-HEAD mode for each experiment.
- Experiment manifests record the Git SHA, dependency versions, wheel hashes, launch
  command, relevant environment variables, and benchmark inputs.

### 3.2 Local and server layout

Local development clone:

```text
/Users/bsy/Desktop/workspace/Pro5000-Optimize/sglang/
```

Persistent server root:

```text
/home/logs/sennian/pro5000-fi-moe/
  sglang/
  .venv/
  wheelhouse/
  cache/
  runs/
```

The repository and virtual environment are siblings. Wheels, caches, model files,
profiles, and run artifacts are not committed.

### 3.3 Old environment

The existing environment at `/home/logs/sennian/py-venv/sglang5.14` is preserved and
is not upgraded in place. It remains the immediate operational rollback.

## 4. Dependency design

The target API first appears in the selected FlashInfer nightly after PR #3891. The
experiment pins matching core and JIT-cache builds:

| Package | Version / artifact | SHA256 |
| --- | --- | --- |
| `flashinfer-python` | `0.6.15.dev20260716` | `ed0634d9c32f069dafe7583addf74de7a4f366ae07d3093250109bd315b4ba26` |
| `flashinfer-jit-cache` | `0.6.15.dev20260716+cu130` | `86a0944b4cadde0a4227f249e5a0fe466207d7c25e8eb7dee4c3f75fdd5f9bbf` |

The feature branch updates SGLang's stable `flashinfer_python[cu13]==0.6.15` pin to the
exact development version required by the new API. The JIT-cache version must match the
core version prefix.

The corrected server toolchain facts are:

- Python 3.12
- PyTorch 2.11.0 with CUDA 13.0
- NVIDIA driver 580.126.09
- CUDA compiler 13.0, NVCC 13.0.48
- compute capability `(12, 0)`

The official JIT-cache is preferred to avoid compilation latency. Production does not
set `FLASHINFER_DISABLE_JIT`. If the AOT artifact is absent, FlashInfer may use NVCC and
Ninja as a runtime-JIT fallback. Bootstrap explicitly prewarms the target kernel and
keeps the compilation cache under the persistent experiment root.

`FLASHINFER_DISABLE_JIT=1` is only an optional diagnostic: success proves the official
JIT-cache wheel contains the target module; failure does not block the environment if
the same smoke test succeeds after unsetting it and compiling with NVCC.

## 5. SGLang architecture decision

Three integration shapes were considered:

1. Add a direct conditional to `Fp8MoEMethod.apply`. This is small but further couples
   quantization, routing, and execution.
2. Register one fused `none -> FlashInfer` function. This matches mature fused backends
   but makes the experimental pipeline harder to isolate and profile.
3. Add a first-class `MoeRunnerCore` with registered standard pre- and post-permute
   adapters.

Option 3 is selected. It follows the existing runner architecture and makes routing,
quantization, GEMM, activation, and combine independently testable. The intended code
touch points are:

- `python/sglang/srt/layers/moe/utils.py`: backend enum and predicate;
- `python/sglang/srt/layers/moe/moe_runner/runner.py`: instantiate the new core;
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`: quant info, runner
  input/output, pre-permute, two GEMM calls, and post-permute;
- `python/sglang/srt/layers/quantization/fp8.py`: select the runner, prepare quant info,
  and cache the FlashInfer weight-scale layout;
- focused unit, CUDA correctness, and benchmark files.

## 6. Initial support boundary

The first implementation supports only:

- CUDA compute capability 12.0 or 12.1;
- serialized FP8 E4M3 routed-expert weights;
- `[128, 128]` blockwise float32 weight scales;
- dynamic activation quantization;
- `moe-a2a-backend=none`;
- TP=1 and EP=1;
- the target Qwen gated-SiLU semantics, with `gemm1_alpha`, GEMM1 clamp, and
  `swiglu_limit` unset;
- standard combine, top-k 8 for the target workload;
- no expert bias, LoRA, or `no_combine` mode.

The shared expert remains on SGLang's existing dense path. It is not appended as a 257th
group in this change.

Unsupported hardware or model semantics fail during model/runner initialization when
the experimental backend is explicitly selected. Runtime exceptions are not silently
converted into Triton calls. The existing `auto` and explicit Triton paths remain
unchanged and provide deterministic rollback.

## 7. FlashInfer kernel contract

Both expert matrix multiplications invoke the same official function:

```python
from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
```

The fixed contract is:

- `a`: token-packed FP8 `[cum_m, K]`;
- `b`: FP8 `[num_experts, N, K]`;
- `a_scale`: float32 `[ceil(K/128), m_padded]`;
- `b_scale`: float32 `[num_experts, ceil(K/128), ceil(N/128)]`;
- `m_indptr`: CUDA int32 `[num_experts + 1]` CSR prefix sum;
- scale granularity `(1, 128, 128)`;
- `scale_major_mode="MN"`, `backend="cute"`;
- BF16 output.

For expert `i`, the A-scale destination starts at:

```text
floor((m_indptr[i] + 3*i) / 4) * 4
```

and:

```text
m_padded = floor((cum_m + 3*num_experts) / 4) * 4
```

## 8. Forward data flow

### 8.1 Routing and packing

The standard dispatcher supplies BF16 hidden states and SGLang-computed top-k ids and
weights. The pre-permute adapter:

1. flattens token/top-k assignments;
2. counts assignments per expert;
3. creates the CUDA int32 `m_indptr`;
4. writes token rows into expert-major, token-packed order without expert `BLOCK_M`
   padding;
5. stores the inverse assignment-to-packed-row map for combine.

Existing Triton and DeepGEMM preprocess outputs cannot be passed through unchanged:
they use block padding or masked expert tensors, while the target API requires compact
rows and only pads the A-scale columns. Their routing semantics and helper operations
should be reused where their contracts match; incompatible tensor layouts must not be
reused merely to reduce code size.

### 8.2 Activation quantization and A-scale layout

The correctness-first implementation reuses SGLang's per-token-group FP8 quantization
for group size 128, then performs the minimum required scale-layout adapter into the
FlashInfer zero-padding layout. The adapter follows FlashInfer's official reference
transformation exactly.

If profiling proves the separate repack launch materially consumes the GEMM gain, a
later optimization may make the SGLang quantization helper write the target layout
directly. This is an adapter optimization, not a FlashInfer GEMM modification.

### 8.3 GEMM1

For the target model:

```text
A:   [cum_m, 2048]
W13: [256, 1024, 2048]
out: [cum_m, 1024] BF16
```

The output follows the same contiguous gate/up-half interpretation used by SGLang's
ordinary Qwen `silu_and_mul` path and computes `SiLU(gate) * up`, producing BF16
`[cum_m, 512]`. The implementation must test this against the existing Triton result
instead of inferring it from the `gate_up_interleaved` flag, which is only consulted by
special clamped/alpha activation variants in the current runner.

### 8.4 GEMM2

The activation result is dynamically quantized again with group size 128 and uses the
same `m_indptr`:

```text
A:  [cum_m, 512]
W2: [256, 2048, 512]
out:[cum_m, 2048] BF16
```

### 8.5 Combine

The post-permute adapter uses the saved inverse map to gather expert outputs for each
original token, applies the original top-k weights and routed scaling factor, and sums
the eight routed contributions into `[num_tokens, 2048]`.

## 9. Weight handling

The FP8 weights remain contiguous in SGLang's canonical `[E, N, K]` form. They are not
requantized or transposed for the new kernel.

Checkpoint block scales are loaded as `[E, N_blocks, K_blocks]`. After loading, the new
backend creates non-persistent cached tensors:

```text
[E, N_blocks, K_blocks] -> transpose(1, 2).contiguous()
                         -> [E, K_blocks, N_blocks]
```

The original scale parameters are retained unchanged so that the explicit Triton path,
hot reload behavior, and A/B comparison remain available.

## 10. Correctness gates

Validation is sequential. A failing stage blocks the next stage.

### 10.1 Environment and standalone API

- import the exact FlashInfer API;
- confirm CUDA availability and compute capability `(12, 0)`;
- record NVCC, Torch CUDA, package versions, and file locations;
- run balanced, irregular, and empty-expert inputs;
- run the real GEMM1 and GEMM2 shapes;
- verify the second call reuses the installed or compiled cache.

The optional no-JIT diagnostic is run first. If it fails, the failure is recorded and
the test is repeated with runtime JIT enabled.

### 10.2 Adapter tests

- `m_indptr` exactly equals the expert histogram prefix sum;
- packed rows and inverse mapping round-trip exactly;
- A-scale locations match the official reference element by element;
- cover empty experts, concentrated routing, uneven routing, and zero-token input.

### 10.3 Kernel and runner numerical tests

- standalone FlashInfer GEMMs use its normalized error metric and require
  `calc_diff < 1e-3` against a BF16 reference;
- the integrated FlashInfer and Triton paths use identical hidden states, weights,
  top-k ids, and top-k weights;
- both are compared to the same BF16/dequantized reference;
- FlashInfer's routed-MoE error must not exceed Triton's error by more than 10%;
- assert finite outputs and validate gate/up order, SiLU, top-k weighting, routed
  scaling, and shared-expert composition.

### 10.4 Model-level tests

Use fixed prompts, seeds, and sampling settings. Compare logits, greedy tokens, and
generated text for prefill and decode separately. FP8 results are not required to be
bitwise identical, but all material deviations are recorded and investigated.

## 11. Performance gates

Measure with CUDA events after warmup and report kernel-only, adapter, complete routed
MoE, and model-level timings separately. Use uniform routing and captured real routing
distributions.

| Gate | Requirement |
| --- | --- |
| GEMM1 | at most 1.014 ms, at least 20% faster than 1.267 ms |
| GEMM2 | at most 0.834 ms, no more than 5% slower than 0.794 ms |
| Complete routed MoE | at least 10% faster including packing, quantization, activation, and combine |

If kernel-only performance passes but the adapter erases the gain, perform one focused
adapter-optimization round and repeat the full gate. A failure to produce meaningful
complete-MoE improvement stops production integration and leaves the experimental
branch available for analysis.

The initial backend is functionally available to both prefill and decode because the
official zero-padding entry specifically targets small expert-M decode while also
supporting large M. Performance and CUDA-graph compatibility are measured separately
before any automatic or forward-mode-specific selection is designed.

## 12. Stage B: reproducible server environment

Stage B is completed before runner implementation:

1. Commit the approved design, exact dependency pin, bootstrap, smoke test, and
   environment collector on the feature branch.
2. Push the feature branch to the fork and record its exact SHA.
3. On the server, clone the fork into the persistent root and check out that SHA.
4. Create `.venv` with `uv` and Python 3.12; do not alter the old venv.
5. Download wheels with resume support and verify SHA256 before installation.
6. Install the fork and exact dependencies, then run `pip check`.
7. Run the optional no-JIT diagnostic and the required normal prewarm/smoke test.
8. Save the manifest and output under `runs/` and return the output for review.

Stage B passes when the new venv successfully invokes the target kernel on the RTX PRO
5000 with correct numerical output. Either official AOT-cache loading or a recorded,
prewarmed CUDA 13.0 runtime-JIT build is acceptable.

## 13. Synchronization and rollback

For every later experiment, local changes are committed and pushed first. The server
uses only:

```bash
git fetch origin
git switch --detach <exact-commit-sha>
```

Rollback requires no deletion: exit the new venv, return to the old environment, or
check out a previously recorded commit. Large wheels and caches are retained to avoid
unnecessary downloads but remain outside Git.

## 14. Deferred work

The following are outside the first integration:

- editing or tuning FlashInfer CuTe/CUDA kernel code;
- TP>1, EP>1, or an A2A backend;
- shared-expert fusion into the grouped GEMMs;
- LoRA, expert bias, or no-combine support;
- silent runtime fallback;
- automatic backend selection;
- production containerization and rollout, which occur only after the performance gate.

## 15. Authoritative references

- [FlashInfer FP8 SM120 zero-padding API](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/flashinfer/grouped_mm/cute_sm120_fp8_groupwise/core.py)
- [FlashInfer FP8 SM120 tests](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/tests/grouped_mm/test_cute_sm120_fp8.py)
- [FlashInfer AOT registry](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/flashinfer/aot.py)
- [SGLang MoE runner](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/moe/moe_runner/runner.py)
- [SGLang DeepGEMM runner adapter pattern](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py)
- [SGLang FP8 MoE method](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/quantization/fp8.py)
