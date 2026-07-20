# FlashInfer SM120 FP8 MoE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SGLang 中增加显式 `flashinfer_sm120_fp8` MoE runner backend，在单卡
SM120/SM121 上让 blockwise FP8 MoE 的 GEMM1 和 GEMM2 调用 FlashInfer
`moe_gemm_fp8_nt_groupwise`，并保持标准 SGLang routing、SwiGLU、combine 与 CUDA
Graph 数据流。

**Architecture:** 新 backend 使用标准 dispatcher 的 fused-func 接口，先对原 token
量化，再用 SGLang `moe_permute` 生成 packed FP8 activation、`src2dst` 和
`m_indptr`。新增一个 Triton scale-layout op，利用
`topk_ids_flat/src2dst/m_indptr` 为 GEMM1 和 GEMM2 生成 FlashInfer MN-major
4-row-aligned A-scale；两次 grouped GEMM 之间复用 SGLang 的 fused
SwiGLU+per-token-group FP8 quant，最终复用 `moe_unpermute`。

**Tech Stack:** Python 3.12、PyTorch 2.11、Triton、SGLang MoE runner、FlashInfer
0.6.15 nightly、CUDA 13.0、pytest/unittest、CUDA Graph。

## Global Constraints

- 目标设备仅为 CUDA compute capability `(12, 0)` 或 `(12, 1)`。
- 目标 FlashInfer 版本为 `flashinfer-python==0.6.15.dev20260716`，使用 runtime JIT；
  不设置 `FLASHINFER_DISABLE_JIT=1`。
- 权重和 activation dtype 分别为 `torch.float8_e4m3fn` 和
  `torch.bfloat16`，输出为 `torch.bfloat16`。
- 权重 block shape 必须严格等于 `(128, 128)`，activation 按 K 维每 128 元素动态
  per-token-group 量化。
- 只支持 gated SiLU/SwiGLU、无 expert bias、无 GPT-OSS alpha/limit、无额外
  `swiglu_limit`。
- 初始范围严格为 `tp=1`、`ep=1`、`--moe-a2a-backend none`、LoRA/TBO/SBO
  关闭。
- backend 必须通过 `--moe-runner-backend flashinfer_sm120_fp8` 显式选择；不得按
  token 数、batch 或错误类型静默回退 Triton/CUTLASS。
- 第一阶段 GEMM1 和 GEMM2 都调用 FlashInfer；decode 性能未达标时结果归类为
  `FUNCTIONAL_ONLY`，不在本计划内加入 hybrid threshold。
- `topk_ids` 在 runner 边界为 contiguous `int32`；`m_indptr` 为 CUDA contiguous
  `int32 [E+1]`。
- A-scale data pointer 必须 16 字节对齐；padding 每次调用清零，CUDA Graph replay
  也必须重新执行清零和 layout。
- 服务器虚拟环境为 `/home/logs/sennian/pro5000-fi-moe/.venv`；所有包安装、卸载和
  依赖检查只使用 `uv pip --python ...`，不得使用 `pip` 或 `python -m pip`。
- 代码修改开始前必须得到用户批准；服务器命令由用户执行。

## File Map

- Create: `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`
  - 唯一职责：计算 `M_padded`，并把两类 row-major activation scale 写入
    FlashInfer A-scale 布局。
- Create: `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`
  - 唯一职责：feature detection、量化信息、能力检查、两次 FlashInfer GEMM 和
    标准 dispatcher fused-func 数据流。
- Modify: `python/sglang/srt/layers/moe/utils.py`
  - 注册 backend enum 和 predicate。
- Modify: `python/sglang/srt/server_args.py`
  - 暴露 CLI choice。
- Modify: `python/sglang/srt/arg_groups/overrides.py`
  - 在启动配置解析阶段拒绝不支持的量化、并行和 overlap/LoRA 配置。
- Modify: `python/sglang/srt/layers/moe/moe_runner/runner.py`
  - 把新 backend 声明为 fused-only runner，并确保注册模块已导入。
- Modify: `python/sglang/srt/layers/quantization/fp8.py`
  - 仅为匹配的 blockwise FP8 创建新 runner；加载阶段转换 weight scale；forward
    阶段构造新 quant-info。
- Create: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`
  - CPU 可运行的 enum、配置、scale 转换和失败策略测试。
- Create: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`
  - SM120 上的 FP8 bitwise permute、layout、完整 runner、CUDA Graph 测试。
- Create: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
  - 完整 runner correctness/component/end-to-end microbenchmark 和 JSON 输出。
- Create: `scripts/pro5000/run_stage_2_benchmark.sh`
  - 可恢复的服务器执行 wrapper 和结果目录管理。
- Create: `test/registered/unit/test_pro5000_stage_2.py`
  - benchmark 参数、结果判定和 shell 安全性测试。
- Modify: `scripts/pro5000/README.md`
  - Stage 2 同步、预热、运行、打包和恢复频率步骤。

---

### Task 1: 注册显式 backend 并在启动阶段拒绝越界配置

**Files:**
- Modify: `python/sglang/srt/layers/moe/utils.py:90-157`
- Modify: `python/sglang/srt/server_args.py:251-274`
- Modify: `python/sglang/srt/arg_groups/overrides.py:1907-1960`
- Create: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`

**Interfaces:**
- Consumes: `MoeRunnerBackend`、`MOE_RUNNER_BACKEND_CHOICES`、
  `_moe_runner_backend_quant_constraints(view)`。
- Produces: `MoeRunnerBackend.FLASHINFER_SM120_FP8`、
  `MoeRunnerBackend.is_flashinfer_sm120_fp8()`、CLI value
  `flashinfer_sm120_fp8`。

- [ ] **Step 1: 写 backend 与配置失败测试**

```python
from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.overrides import (
    _moe_runner_backend_quant_constraints,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _view(**overrides):
    values = dict(
        quantization="fp8",
        moe_runner_backend="flashinfer_sm120_fp8",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        enable_lora=False,
        lora_paths=[],
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_backend_is_registered():
    backend = MoeRunnerBackend("flashinfer_sm120_fp8")
    assert backend is MoeRunnerBackend.FLASHINFER_SM120_FP8
    assert backend.is_flashinfer_sm120_fp8()
    assert "flashinfer_sm120_fp8" in MOE_RUNNER_BACKEND_CHOICES


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"quantization": "modelopt_fp8"}, "blockwise FP8"),
        ({"tp_size": 2}, "tp_size=1"),
        ({"ep_size": 2}, "ep_size=1"),
        ({"moe_a2a_backend": "deepep"}, "moe_a2a_backend=none"),
        ({"enable_lora": True}, "LoRA"),
        ({"lora_paths": ["adapter"]}, "LoRA"),
        ({"enable_two_batch_overlap": True}, "TBO"),
        ({"enable_single_batch_overlap": True}, "SBO"),
    ],
)
def test_backend_rejects_out_of_scope_server_config(override, message):
    with pytest.raises(ValueError, match=message):
        _moe_runner_backend_quant_constraints(_view(**override))


def test_backend_accepts_fp8_or_autodetected_quantization():
    assert _moe_runner_backend_quant_constraints(_view()) == {}
    assert _moe_runner_backend_quant_constraints(_view(quantization=None)) == {}
```

- [ ] **Step 2: 运行测试并确认因为 backend 未注册而失败**

Run:

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
```

Expected: FAIL，首个失败包含 `flashinfer_sm120_fp8 is not a valid MoeRunnerBackend` 或
缺少 enum attribute。

- [ ] **Step 3: 增加 enum、predicate 和 CLI choice**

在 `MoeRunnerBackend` 中加入：

```python
FLASHINFER_SM120_FP8 = "flashinfer_sm120_fp8"

def is_flashinfer_sm120_fp8(self):
    return self == MoeRunnerBackend.FLASHINFER_SM120_FP8
```

在 `MOE_RUNNER_BACKEND_CHOICES` 中加入：

```python
"flashinfer_sm120_fp8",
```

- [ ] **Step 4: 增加启动配置验证**

在 `_moe_runner_backend_quant_constraints` 返回前加入：

```python
if moe_runner_backend == "flashinfer_sm120_fp8":
    if view.quantization not in (None, "fp8"):
        raise ValueError(
            "flashinfer_sm120_fp8 requires an autodetected or explicit "
            "blockwise FP8 checkpoint (--quantization fp8)."
        )
    if view.tp_size != 1:
        raise ValueError("flashinfer_sm120_fp8 requires tp_size=1.")
    if view.ep_size != 1:
        raise ValueError("flashinfer_sm120_fp8 requires ep_size=1.")
    if view.moe_a2a_backend != "none":
        raise ValueError("flashinfer_sm120_fp8 requires moe_a2a_backend=none.")
    if bool(view.enable_lora) or bool(view.lora_paths):
        raise ValueError("flashinfer_sm120_fp8 does not support LoRA.")
    if view.enable_two_batch_overlap:
        raise ValueError("flashinfer_sm120_fp8 does not support TBO.")
    if view.enable_single_batch_overlap:
        raise ValueError("flashinfer_sm120_fp8 does not support SBO.")
```

- [ ] **Step 5: 运行测试并提交**

Run:

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
```

Expected: `10 passed`。

Commit:

```bash
git add python/sglang/srt/layers/moe/utils.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/arg_groups/overrides.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py
git commit -m "feat: register FlashInfer SM120 FP8 MoE backend"
```

### Task 2: 验证 FP8 permute 并实现两种 source-row 的 scale layout

**Files:**
- Create: `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`
- Create: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: `moe_permute(q_hidden, topk_ids, num_experts)` 返回的
  `(packed_hidden, src2dst, m_indptr)`。
- Produces:
  - `flashinfer_sm120_m_padded(cum_m: int, num_experts: int) -> int`
  - `pack_flashinfer_sm120_fp8_scale(source_scale, topk_ids, src2dst,
    m_indptr, *, source_is_packed, out=None) -> torch.Tensor`

- [ ] **Step 1: 写现有 permute 的 FP8 bitwise 测试**

```python
import importlib.util
import unittest

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() in {
    (12, 0),
    (12, 1),
}
_HAS_TARGET_API = (
    importlib.util.find_spec("flashinfer.grouped_mm.cute_sm120_fp8_groupwise")
    is not None
)


@unittest.skipUnless(_IS_SM120, "SM120/SM121 required")
class TestFlashInferSm120Fp8Packing(unittest.TestCase):
    def test_existing_moe_permute_copies_fp8_bits(self):
        torch.manual_seed(7)
        tokens, hidden, experts, top_k = 8, 256, 8, 2
        q_hidden = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        topk_ids = torch.tensor(
            [[0, 7], [1, 1], [3, 0], [7, 2], [2, 4], [4, 0], [6, 6], [5, 3]],
            device="cuda",
            dtype=torch.int32,
        )

        packed, src2dst, m_indptr = moe_permute(q_hidden, topk_ids, experts)
        expected = torch.empty_like(packed)
        src2dst_cpu = src2dst.cpu().tolist()
        for route in range(tokens * top_k):
            expected[src2dst_cpu[route]].copy_(q_hidden[route // top_k])

        torch.testing.assert_close(
            packed.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        self.assertEqual(m_indptr.dtype, torch.int32)
        self.assertEqual(m_indptr.shape, (experts + 1,))
```

- [ ] **Step 2: 在 PRO 5000 环境运行 FP8 permute gate**

Run on server:

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py::TestFlashInferSm120Fp8Packing::test_existing_moe_permute_copies_fp8_bits \
  -q
```

Expected: PASS。

如果该测试 FAIL，停止后续 runner 工作并保存完整 Triton compile/runtime error。只有
此 gate 失败时，才在新 kernel 文件中增加下面的 bit-preserving 专用 gather，并让
runner 调用它：

```python
@triton.jit
def _pack_fp8_routes_kernel(
    source_ptr,
    output_ptr,
    src2dst_ptr,
    hidden_size,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    dst = tl.load(src2dst_ptr + route)
    token = route // top_k
    values = tl.load(
        source_ptr + token * hidden_size + cols,
        mask=cols < hidden_size,
    )
    tl.store(
        output_ptr + dst * hidden_size + cols,
        values,
        mask=cols < hidden_size,
    )
```

- [ ] **Step 3: 添加 layout reference 与失败测试**

向同一测试文件增加：

```python
def _layout_reference(source, topk_ids, src2dst, m_indptr, source_is_packed):
    routes = topk_ids.numel()
    experts = m_indptr.numel() - 1
    top_k = topk_ids.shape[1]
    m_padded = ((routes + 3 * experts) // 4) * 4
    result = torch.zeros(
        source.shape[1], m_padded, dtype=torch.float32, device=source.device
    )
    flat_ids = topk_ids.flatten()
    for route in range(routes):
        expert = int(flat_ids[route].item())
        dst = int(src2dst[route].item())
        expert_start = int(m_indptr[expert].item())
        aligned_start = ((expert_start + 3 * expert) // 4) * 4
        source_row = dst if source_is_packed else route // top_k
        result[:, aligned_start + dst - expert_start] = source[source_row]
    return result


def test_layout_token_and_packed_source_rows(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        pack_flashinfer_sm120_fp8_scale,
    )

    topk_ids = torch.tensor(
        [[3, 0], [3, 3], [1, 0], [7, 3]], device="cuda", dtype=torch.int32
    )
    _, src2dst, m_indptr = moe_permute(
        torch.zeros((4, 128), device="cuda", dtype=torch.float8_e4m3fn),
        topk_ids,
        8,
    )
    token_scale = torch.arange(8, device="cuda", dtype=torch.float32).view(4, 2)
    packed_scale = torch.arange(16, device="cuda", dtype=torch.float32).view(8, 2)

    actual_gemm1 = pack_flashinfer_sm120_fp8_scale(
        token_scale, topk_ids, src2dst, m_indptr, source_is_packed=False
    )
    actual_gemm2 = pack_flashinfer_sm120_fp8_scale(
        packed_scale, topk_ids, src2dst, m_indptr, source_is_packed=True
    )
    torch.testing.assert_close(
        actual_gemm1,
        _layout_reference(token_scale, topk_ids, src2dst, m_indptr, False),
    )
    torch.testing.assert_close(
        actual_gemm2,
        _layout_reference(packed_scale, topk_ids, src2dst, m_indptr, True),
    )
    self.assertEqual(actual_gemm1.data_ptr() % 16, 0)
```

- [ ] **Step 4: 运行 layout 测试并确认缺少模块**

Run:

```bash
python3 -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py::TestFlashInferSm120Fp8Packing::test_layout_token_and_packed_source_rows \
  -q
```

Expected: FAIL with
`No module named 'sglang.kernels.ops.moe.flashinfer_sm120_fp8'`。

- [ ] **Step 5: 实现最小 scale-layout op**

创建 `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`：

```python
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def flashinfer_sm120_m_padded(cum_m: int, num_experts: int) -> int:
    if cum_m < 0 or num_experts <= 0:
        raise ValueError(
            f"expected cum_m >= 0 and num_experts > 0, got {cum_m=} {num_experts=}"
        )
    return ((cum_m + 3 * num_experts) // 4) * 4


@triton.jit
def _pack_flashinfer_sm120_fp8_scale_kernel(
    source_ptr,
    topk_ids_ptr,
    src2dst_ptr,
    m_indptr_ptr,
    output_ptr,
    source_stride_m,
    source_stride_k,
    output_stride_k,
    output_stride_m,
    num_k_blocks,
    top_k: tl.constexpr,
    source_is_packed: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    k_blocks = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    expert = tl.load(topk_ids_ptr + route)
    dst = tl.load(src2dst_ptr + route)
    expert_start = tl.load(m_indptr_ptr + expert)
    aligned_start = ((expert_start + 3 * expert) // 4) * 4
    output_col = aligned_start + dst - expert_start
    source_row = dst if source_is_packed else route // top_k
    values = tl.load(
        source_ptr + source_row * source_stride_m + k_blocks * source_stride_k,
        mask=k_blocks < num_k_blocks,
    )
    tl.store(
        output_ptr + k_blocks * output_stride_k + output_col * output_stride_m,
        values,
        mask=k_blocks < num_k_blocks,
    )


def pack_flashinfer_sm120_fp8_scale(
    source_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    source_is_packed: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if source_scale.dtype != torch.float32 or source_scale.ndim != 2:
        raise TypeError("source_scale must be a 2D float32 tensor")
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2:
        raise TypeError("topk_ids must be a 2D int32 tensor")
    if src2dst.dtype != torch.int32 or src2dst.numel() != topk_ids.numel():
        raise TypeError("src2dst must be int32 with one entry per routed slot")
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    routes = topk_ids.numel()
    experts = m_indptr.numel() - 1
    k_blocks = source_scale.shape[1]
    expected_rows = routes if source_is_packed else topk_ids.shape[0]
    if source_scale.shape[0] != expected_rows:
        raise ValueError(
            f"source_scale rows must be {expected_rows}, got {source_scale.shape[0]}"
        )
    expected_shape = (k_blocks, flashinfer_sm120_m_padded(routes, experts))
    if out is None:
        out = torch.empty(expected_shape, device=source_scale.device, dtype=torch.float32)
    if out.shape != expected_shape or out.dtype != torch.float32 or not out.is_contiguous():
        raise ValueError(f"out must be contiguous float32 with shape {expected_shape}")
    if out.data_ptr() % 16 != 0:
        raise ValueError("FlashInfer A-scale output must be 16-byte aligned")
    out.zero_()
    if routes:
        _pack_flashinfer_sm120_fp8_scale_kernel[
            (routes, triton.cdiv(k_blocks, 32))
        ](
            source_scale,
            topk_ids,
            src2dst,
            m_indptr,
            out,
            source_scale.stride(0),
            source_scale.stride(1),
            out.stride(0),
            out.stride(1),
            k_blocks,
            top_k=topk_ids.shape[1],
            source_is_packed=source_is_packed,
            BLOCK_K=32,
        )
    return out
```

- [ ] **Step 6: 扩展空 expert、skew、padding 重写测试并运行**

向 `TestFlashInferSm120Fp8Packing` 增加以下两个测试：

```python
def test_layout_covers_topk_and_empty_expert_profiles(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        pack_flashinfer_sm120_fp8_scale,
    )

    experts, tokens = 8, 8
    for top_k in (1, 2, 8):
        with self.subTest(top_k=top_k):
            topk_ids = (
                torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
                .remainder(experts - 1)
                .view(tokens, top_k)
            )
            # expert 7 remains empty; experts 0..6 are uniform enough to cover
            # non-aligned starts and duplicate expert assignments.
            _, src2dst, m_indptr = moe_permute(
                torch.zeros(
                    tokens, 128, device="cuda", dtype=torch.float8_e4m3fn
                ),
                topk_ids,
                experts,
            )
            source = torch.arange(
                tokens * 2, device="cuda", dtype=torch.float32
            ).view(tokens, 2)
            actual = pack_flashinfer_sm120_fp8_scale(
                source,
                topk_ids,
                src2dst,
                m_indptr,
                source_is_packed=False,
            )
            expected = _layout_reference(
                source, topk_ids, src2dst, m_indptr, False
            )
            torch.testing.assert_close(actual, expected)

def test_reused_out_clears_padding_after_route_change(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        pack_flashinfer_sm120_fp8_scale,
    )

    experts = 8
    route_a = torch.tensor(
        [[0, 0], [0, 1], [1, 1], [1, 1]], device="cuda", dtype=torch.int32
    )
    route_b = torch.tensor(
        [[7, 7], [6, 7], [5, 6], [4, 7]], device="cuda", dtype=torch.int32
    )
    source = torch.arange(8, device="cuda", dtype=torch.float32).view(4, 2)
    out = None
    for topk_ids in (route_a, route_b):
        _, src2dst, m_indptr = moe_permute(
            torch.zeros((4, 128), device="cuda", dtype=torch.float8_e4m3fn),
            topk_ids,
            experts,
        )
        if out is None:
            out = pack_flashinfer_sm120_fp8_scale(
                source,
                topk_ids,
                src2dst,
                m_indptr,
                source_is_packed=False,
            )
        else:
            pack_flashinfer_sm120_fp8_scale(
                source,
                topk_ids,
                src2dst,
                m_indptr,
                source_is_packed=False,
                out=out,
            )
    expected = _layout_reference(source, route_b, src2dst, m_indptr, False)
    torch.testing.assert_close(out, expected)
```

然后运行：

```bash
python3 -m pytest test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k "Packing" -q
```

Expected: 所有 packing/layout case PASS；非 SM120 机器 SKIP。

- [ ] **Step 7: 提交**

```bash
git add python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "feat: add FlashInfer SM120 FP8 scale layout"
```

### Task 3: 实现 fused-only FlashInfer runner 数据流

**Files:**
- Create: `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`
- Modify: `python/sglang/srt/layers/moe/moe_runner/runner.py:27-105`
- Modify: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`

**Interfaces:**
- Consumes:
  - `pack_flashinfer_sm120_fp8_scale(...)`
  - `moe_permute(...)` / `moe_unpermute(...)`
  - `sglang_per_token_group_quant_fp8(..., group_size=128)`
  - FlashInfer `moe_gemm_fp8_nt_groupwise(..., out=..., out_dtype=bf16)`
- Produces:
  - `FlashInferSm120Fp8MoeQuantInfo`
  - `prepare_flashinfer_sm120_fp8_weight_scales(...)`
  - `fused_experts_none_to_flashinfer_sm120_fp8(...)`

- [ ] **Step 1: 写 weight-scale 转换与 fused-func 注册测试**

```python
import torch


def test_weight_scale_conversion_is_transpose_contiguous():
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        prepare_flashinfer_sm120_fp8_weight_scales,
    )

    w13 = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
    w2 = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    w13_fi, w2_fi = prepare_flashinfer_sm120_fp8_weight_scales(w13, w2)
    assert w13_fi.shape == (2, 4, 8)
    assert w2_fi.shape == (2, 8, 4)
    assert w13_fi.is_contiguous() and w2_fi.is_contiguous()
    torch.testing.assert_close(w13_fi, w13.transpose(1, 2))
    torch.testing.assert_close(w2_fi, w2.transpose(1, 2))


def test_fused_func_is_registered_for_standard_dispatch():
    import sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8  # noqa: F401
    from sglang.srt.layers.moe.moe_runner.base import FusedOpPool

    assert FusedOpPool.get_fused_func("none", "flashinfer_sm120_fp8") is not None
```

- [ ] **Step 2: 运行并确认缺少 runner 模块**

Run:

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
```

Expected: FAIL with missing `flashinfer_sm120_fp8` runner module。

- [ ] **Step 3: 创建 quant-info、feature detection 和 scale 转换**

新 runner 文件先实现：

```python
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute, moe_unpermute
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
    pack_flashinfer_sm120_fp8_scale,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.topk import TopKOutputChecker


@dataclass
class FlashInferSm120Fp8MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale_fi: torch.Tensor
    w2_weight_scale_fi: torch.Tensor
    block_shape: tuple[int, int]


def prepare_flashinfer_sm120_fp8_weight_scales(
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("FlashInfer SM120 FP8 weight scales must be float32")
    if w13_scale.ndim != 3 or w2_scale.ndim != 3:
        raise ValueError("FlashInfer SM120 FP8 weight scales must be rank 3")
    return (
        w13_scale.transpose(1, 2).contiguous(),
        w2_scale.transpose(1, 2).contiguous(),
    )


@functools.lru_cache(maxsize=1)
def _target_grouped_gemm():
    try:
        from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
    except ImportError as error:
        raise RuntimeError(
            "flashinfer_sm120_fp8 requires "
            "flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise"
        ) from error
    return moe_gemm_fp8_nt_groupwise
```

- [ ] **Step 4: 实现 fail-fast contract validator**

```python
def _validate_contract(
    dispatch_output,
    quant_info: FlashInferSm120Fp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    capability = torch.cuda.get_device_capability(dispatch_output.hidden_states.device)
    if capability not in {(12, 0), (12, 1)}:
        raise RuntimeError(
            f"flashinfer_sm120_fp8 requires capability 12.0 or 12.1, got {capability}"
        )
    if dispatch_output.hidden_states.dtype != torch.bfloat16:
        raise TypeError("flashinfer_sm120_fp8 activation input must be bfloat16")
    if quant_info.w13_weight.dtype != torch.float8_e4m3fn or quant_info.w2_weight.dtype != torch.float8_e4m3fn:
        raise TypeError("flashinfer_sm120_fp8 weights must be float8_e4m3fn")
    if not quant_info.w13_weight.is_contiguous() or not quant_info.w2_weight.is_contiguous():
        raise ValueError("flashinfer_sm120_fp8 weights must be contiguous [E, n, k]")
    if tuple(quant_info.block_shape) != (128, 128):
        raise ValueError("flashinfer_sm120_fp8 block_shape must be (128, 128)")
    for name, weight, scale in (
        ("w13", quant_info.w13_weight, quant_info.w13_weight_scale_fi),
        ("w2", quant_info.w2_weight, quant_info.w2_weight_scale_fi),
    ):
        if weight.shape[1] % 128 or weight.shape[2] % 128:
            raise ValueError(f"{name} n/k dimensions must be divisible by 128")
        expected_scale = (
            weight.shape[0],
            weight.shape[2] // 128,
            weight.shape[1] // 128,
        )
        if scale.dtype != torch.float32 or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"{name} scale must be float32 with shape {expected_scale}, "
                f"got dtype={scale.dtype} shape={tuple(scale.shape)}"
            )
        if not scale.is_contiguous():
            raise ValueError(f"{name} scale must be contiguous")
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError("flashinfer_sm120_fp8 supports gated SiLU/SwiGLU only")
    if runner_config.apply_router_weight_on_input:
        raise ValueError("apply_router_weight_on_input is not supported")
    if runner_config.no_combine:
        raise ValueError("no_combine is not supported")
    if runner_config.gemm1_alpha is not None or runner_config.gemm1_clamp_limit is not None:
        raise ValueError("GPT-OSS alpha/limit is not supported")
    if runner_config.swiglu_limit is not None:
        raise ValueError("swiglu_limit is not supported")
```

- [ ] **Step 5: 实现完整 fused 数据流**

```python
@register_fused_func("none", "flashinfer_sm120_fp8")
def fused_experts_none_to_flashinfer_sm120_fp8(
    dispatch_output,
    quant_info: FlashInferSm120Fp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if not isinstance(quant_info, FlashInferSm120Fp8MoeQuantInfo):
        raise TypeError(f"unexpected quant_info type: {type(quant_info)}")
    if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
        raise TypeError("flashinfer_sm120_fp8 requires StandardTopKOutput")
    _validate_contract(dispatch_output, quant_info, runner_config)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_output.topk_ids
    topk_weights = dispatch_output.topk_output.topk_weights
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()

    q_hidden, q_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
    packed_hidden, src2dst, m_indptr = moe_permute(
        q_hidden, topk_ids, quant_info.w13_weight.shape[0]
    )
    a1_scale_fi = pack_flashinfer_sm120_fp8_scale(
        q_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=False,
    )

    gemm = _target_grouped_gemm()
    gate_up = torch.empty(
        packed_hidden.shape[0],
        quant_info.w13_weight.shape[1],
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    gemm(
        packed_hidden,
        quant_info.w13_weight,
        a1_scale_fi,
        quant_info.w13_weight_scale_fi,
        m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        backend="cute",
        out=gate_up,
        out_dtype=torch.bfloat16,
    )

    down_input, down_scale = sglang_per_token_group_quant_fp8(
        gate_up, 128, fuse_silu_and_mul=True
    )
    a2_scale_fi = pack_flashinfer_sm120_fp8_scale(
        down_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=True,
    )
    down_output = torch.empty(
        packed_hidden.shape[0],
        quant_info.w2_weight.shape[1],
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    gemm(
        down_input,
        quant_info.w2_weight,
        a2_scale_fi,
        quant_info.w2_weight_scale_fi,
        m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        backend="cute",
        out=down_output,
        out_dtype=torch.bfloat16,
    )

    output = moe_unpermute(
        down_output,
        src2dst,
        topk_ids,
        topk_weights,
        routed_scaling_factor=runner_config.routed_scaling_factor,
    )
    return StandardCombineInput(hidden_states=output)
```

- [ ] **Step 6: 在 MoeRunner 中注册 fused-only backend**

在 `runner.py` 的 fused-only 分支加入：

```python
elif runner_backend.is_flashinfer_sm120_fp8():
    self.runner_core = None
    from sglang.srt.layers.moe.moe_runner import flashinfer_sm120_fp8  # noqa: F401
```

- [ ] **Step 7: 运行 CPU 测试并提交**

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
git add python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py \
  python/sglang/srt/layers/moe/moe_runner/runner.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py
git commit -m "feat: add FlashInfer SM120 FP8 MoE runner"
```

Expected: 全部 CPU 测试 PASS，测试过程中不需要 CUDA 或导入目标 JIT module。

### Task 4: 将 blockwise Fp8MoEMethod 接到新 runner

**Files:**
- Modify: `python/sglang/srt/layers/quantization/fp8.py:998-1034`
- Modify: `python/sglang/srt/layers/quantization/fp8.py:1891-1900`
- Modify: `python/sglang/srt/layers/quantization/fp8.py:2097-2150`
- Modify: `python/sglang/srt/layers/quantization/fp8.py:2149-2380`
- Modify: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`

**Interfaces:**
- Consumes: `prepare_flashinfer_sm120_fp8_weight_scales` 和
  `FlashInferSm120Fp8MoeQuantInfo`。
- Produces: layer parameters `w13_weight_scale_fi`、`w2_weight_scale_fi`，以及
  `Fp8MoEMethod.apply()` 到新 fused func 的标准调用。

- [ ] **Step 1: 写量化契约和 layer scale 注册测试**

```python
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def _method_for_blockwise_test(block_shape=(128, 128)):
    method = Fp8MoEMethod.__new__(Fp8MoEMethod)
    method.block_quant = True
    method.use_mxfp8 = False
    method.is_fp4_expert = False
    method.weight_block_size = list(block_shape)
    method.quant_config = SimpleNamespace(weight_block_size=list(block_shape))
    return method


def test_prepare_layer_registers_flashinfer_weight_scales():
    method = _method_for_blockwise_test()
    layer = torch.nn.Module()
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8),
        requires_grad=False,
    )
    with patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_SM120_FP8,
    ), patch.object(method, "process_weights_after_loading_block_quant"):
        method.process_weights_after_loading(layer)
    assert layer.w13_weight_scale_fi.shape == (2, 4, 8)
    assert layer.w2_weight_scale_fi.shape == (2, 8, 4)
    assert layer.w13_weight_scale_inv.shape == (2, 8, 4)
    assert layer.w2_weight_scale_inv.shape == (2, 4, 8)
```

同时增加以下量化配置测试：

```python
def _quant_config(block_shape=(128, 128), *, use_mxfp8=False, fp4=False):
    return SimpleNamespace(
        use_mxfp8=use_mxfp8,
        weight_block_size=list(block_shape),
        is_fp4_experts=fp4,
        dequant_fp4_to_fp8=False,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_quant_config((64, 128)), "block_shape must be \\(128, 128\\)"),
        (_quant_config(use_mxfp8=True), "MXFP8"),
        (_quant_config(fp4=True), "FP4 expert"),
    ],
)
def test_fp8_method_rejects_unsupported_flashinfer_sm120_quant(config, message):
    with patch(
        "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_SM120_FP8,
    ), pytest.raises(ValueError, match=message):
        Fp8MoEMethod(config)
```

expert bias 在 `apply` 分支中通过 `self.with_bias` fail fast；Task 5 的真实 quant-info
测试再确认该错误发生在 FlashInfer API 调用之前。

- [ ] **Step 2: 运行并确认新 layer 参数尚未生成**

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
```

Expected: FAIL，缺少 `w13_weight_scale_fi`。

- [ ] **Step 3: 在 Fp8MoEMethod 初始化和加载后增加严格检查及 scale 参数**

仅当 `get_moe_runner_backend().is_flashinfer_sm120_fp8()` 时执行：

```python
if not self.block_quant or tuple(self.weight_block_size or ()) != (128, 128):
    raise ValueError("flashinfer_sm120_fp8 block_shape must be (128, 128)")
if self.use_mxfp8:
    raise ValueError("flashinfer_sm120_fp8 does not support MXFP8")
if self.is_fp4_expert:
    raise ValueError("flashinfer_sm120_fp8 does not support FP4 experts")
```

在 blockwise weight processing 完成后注册：

```python
if get_moe_runner_backend().is_flashinfer_sm120_fp8():
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        prepare_flashinfer_sm120_fp8_weight_scales,
    )

    w13_scale_fi, w2_scale_fi = prepare_flashinfer_sm120_fp8_weight_scales(
        layer.w13_weight_scale_inv.data,
        layer.w2_weight_scale_inv.data,
    )
    layer.register_parameter(
        "w13_weight_scale_fi",
        torch.nn.Parameter(w13_scale_fi, requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_scale_fi",
        torch.nn.Parameter(w2_scale_fi, requires_grad=False),
    )
```

- [ ] **Step 4: 创建 runner 并构造 quant-info**

在 `create_moe_runner` 的支持列表加入新 backend，并在构造前导入注册模块：

```python
if moe_runner_backend.is_flashinfer_sm120_fp8():
    import sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8  # noqa: F401

self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
```

在 `apply` 的 CUDA backend 分支中加入：

```python
if self.runner.runner_backend.is_flashinfer_sm120_fp8():
    if self.with_bias:
        raise ValueError("flashinfer_sm120_fp8 does not support expert bias")
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        FlashInferSm120Fp8MoeQuantInfo,
    )

    quant_info = FlashInferSm120Fp8MoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale_fi=layer.w13_weight_scale_fi,
        w2_weight_scale_fi=layer.w2_weight_scale_fi,
        block_shape=tuple(self.weight_block_size),
    )
    return self.runner.run(dispatch_output, quant_info)
```

- [ ] **Step 5: 运行单元测试并提交**

```bash
python3 -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
git add python/sglang/srt/layers/quantization/fp8.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py
git commit -m "feat: route blockwise FP8 MoE to FlashInfer SM120"
```

Expected: config、scale 转换、原始 scale 保留和失败策略测试全部 PASS。

### Task 5: 完整 runner 正确性与 CUDA Graph

**Files:**
- Modify: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: 完整 production fused func 和 Triton `fused_experts` reference。
- Produces: uniform/skew/empty-expert、decode/prefill、CUDA Graph replay 的硬回归测试。

- [ ] **Step 1: 添加可复现的 blockwise FP8 case builder**

使用 `flashinfer.testing.utils.per_block_cast_to_fp8` 对每个 expert 的 BF16 权重量化，
固定 `E=8`、`K=N=256`，生成：

```python
def _make_runner_case(tokens, top_k, topk_ids, seed=11):
    from flashinfer.testing.utils import per_block_cast_to_fp8
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        FlashInferSm120Fp8MoeQuantInfo,
        prepare_flashinfer_sm120_fp8_weight_scales,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    torch.manual_seed(seed)
    experts, hidden, intermediate = 8, 256, 256
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 8
    w13_bf16 = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
    ) / hidden**0.5
    w2_bf16 = torch.randn(
        experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
    ) / intermediate**0.5

    def quantize(weight):
        q_parts, s_parts = [], []
        for expert in range(weight.shape[0]):
            q, s = per_block_cast_to_fp8(weight[expert])
            q_parts.append(q)
            s_parts.append(s)
        return torch.stack(q_parts).contiguous(), torch.stack(s_parts).contiguous()

    w13, w13_scale = quantize(w13_bf16)
    w2, w2_scale = quantize(w2_bf16)
    w13_scale_fi, w2_scale_fi = prepare_flashinfer_sm120_fp8_weight_scales(
        w13_scale, w2_scale
    )
    topk_weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_output = StandardTopKOutput(topk_weights, topk_ids, torch.empty(0, device="cuda"))
    dispatch = StandardDispatchOutput(x, None, topk_output)
    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
        routed_scaling_factor=1.0,
    )
    quant_info = FlashInferSm120Fp8MoeQuantInfo(
        w13, w2, w13_scale_fi, w2_scale_fi, (128, 128)
    )
    return dispatch, config, quant_info, w13_scale, w2_scale
```

- [ ] **Step 2: 添加 Triton full-runner 数值对照**

对下列路由执行新 fused func 和 Triton `fused_experts`：

```text
T=1, top_k=1: 单 expert decode
T=8, top_k=8: 大量空 expert + 重复热点 expert
T=128, top_k=2: uniform
T=1024, top_k=8: synthetic skew
```

向测试类增加：

```python
def _calc_diff(actual, expected):
    return float(
        ((actual.float() - expected.float()).abs().mean() / expected.float().abs().mean().clamp_min(1e-12)).item()
    )

def _run_triton_reference(dispatch, config, quant_info, w13_scale, w2_scale):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts

    return fused_experts(
        dispatch.hidden_states.clone(),
        quant_info.w13_weight,
        quant_info.w2_weight,
        dispatch.topk_output,
        config,
        use_fp8_w8a8=True,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )


def test_full_runner_correctness(self):
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        fused_experts_none_to_flashinfer_sm120_fp8,
    )

    cases = [
        (1, 1, torch.tensor([[0]], device="cuda", dtype=torch.int32)),
        (
            8,
            8,
            torch.tensor(
                [
                    [0, 0, 0, 1, 1, 2, 2, 7],
                    [0, 0, 1, 1, 1, 2, 6, 7],
                    [0, 1, 1, 2, 2, 2, 5, 7],
                    [0, 0, 0, 0, 3, 3, 4, 7],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [7, 7, 7, 6, 6, 5, 5, 4],
                    [0, 0, 0, 0, 0, 0, 0, 7],
                    [1, 2, 3, 4, 5, 6, 7, 7],
                ],
                device="cuda",
                dtype=torch.int32,
            ),
        ),
        (
            128,
            2,
            torch.arange(256, device="cuda", dtype=torch.int32).remainder(8).view(128, 2),
        ),
        (
            1024,
            8,
            torch.arange(8192, device="cuda", dtype=torch.int32)
            .square()
            .remainder(8)
            .view(1024, 8),
        ),
    ]
    for tokens, top_k, topk_ids in cases:
        with self.subTest(tokens=tokens, top_k=top_k):
            dispatch, config, quant_info, w13_scale, w2_scale = _make_runner_case(
                tokens, top_k, topk_ids
            )
            actual = fused_experts_none_to_flashinfer_sm120_fp8(
                dispatch, quant_info, config
            ).hidden_states
            expected = _run_triton_reference(
                dispatch, config, quant_info, w13_scale, w2_scale
            )
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertLess(_calc_diff(actual, expected), 1e-3)
```

- [ ] **Step 3: 运行 correctness 并修正最小实现直到通过**

Run on server:

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k "correctness" -q -s
```

Expected: 所有 case PASS；日志不得出现 fallback backend。

- [ ] **Step 4: 添加 CUDA Graph 动态路由 replay 测试**

增加以下测试；第二套路由让原本为空的 expert 变为非空，从而强制 scale 的有效列和
padding 列发生切换：

```python
def test_cuda_graph_replays_new_hidden_and_routing(self):
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        fused_experts_none_to_flashinfer_sm120_fp8,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    tokens, top_k = 8, 8
    route_a = torch.arange(64, device="cuda", dtype=torch.int32).remainder(4).view(8, 8)
    route_b = torch.arange(64, device="cuda", dtype=torch.int32).remainder(4).add(4).view(8, 8)
    dispatch, config, quant_info, _, _ = _make_runner_case(
        tokens, top_k, route_a
    )
    static_x = dispatch.hidden_states
    static_ids = dispatch.topk_output.topk_ids
    static_weights = dispatch.topk_output.topk_weights
    static_dispatch = StandardDispatchOutput(
        static_x,
        None,
        StandardTopKOutput(
            static_weights, static_ids, dispatch.topk_output.router_logits
        ),
    )

    # Warm all SGLang JIT, Triton and the shared FlashInfer .so before capture.
    fused_experts_none_to_flashinfer_sm120_fp8(static_dispatch, quant_info, config)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused_experts_none_to_flashinfer_sm120_fp8(
            static_dispatch, quant_info, config
        ).hidden_states
    captured_output_ptr = graph_output.data_ptr()

    inputs = [
        (torch.randn_like(static_x), route_a),
        (torch.randn_like(static_x), route_b),
    ]
    for new_x, new_ids in inputs:
        static_x.copy_(new_x)
        static_ids.copy_(new_ids)
        static_weights.fill_(1.0 / top_k)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_output.clone()
        eager_dispatch = StandardDispatchOutput(
            new_x,
            None,
            StandardTopKOutput(
                static_weights.clone(), new_ids, torch.empty(0, device="cuda")
            ),
        )
        eager = fused_experts_none_to_flashinfer_sm120_fp8(
            eager_dispatch, quant_info, config
        ).hidden_states
        self.assertEqual(graph_output.data_ptr(), captured_output_ptr)
        self.assertTrue(bool(torch.isfinite(replayed).all()))
        self.assertLess(_calc_diff(replayed, eager), 1e-3)

    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    self.assertEqual(allocated_after, allocated_before)
```

连续 replay 20 次，并比较前后 `torch.cuda.memory_allocated()`；允许 caching allocator
一次性保留，但不得每次 replay 单调增长。

- [ ] **Step 5: 运行完整 GPU 测试并提交**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest test/registered/moe/test_flashinfer_sm120_fp8_moe.py -q -s
git add test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py
git commit -m "test: cover FlashInfer SM120 FP8 MoE runner"
```

Expected: packing、layout、correctness、CUDA Graph 全部 PASS。

### Task 6: Stage 2 benchmark、CUTLASS preflight 与结果包装

**Files:**
- Create: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
- Create: `scripts/pro5000/run_stage_2_benchmark.sh`
- Create: `test/registered/unit/test_pro5000_stage_2.py`
- Modify: `scripts/pro5000/README.md`

**Interfaces:**
- Consumes: production fused func、Triton full runner、可用时的 CUTLASS block-FP8
  runner。
- Produces: `benchmark.json`、`stdout.txt`、`stderr.txt`、环境信息和
  `STAGE_2_DECISION=GO|FUNCTIONAL_ONLY|NO_GO`。

- [ ] **Step 1: 写 benchmark CLI 和判定的 CPU 测试**

测试通过 importlib 加载脚本，固定以下行为：

```python
def test_stage_2_decision_contract():
    bench = load_script("benchmark_flashinfer_sm120_fp8_runner.py")
    assert bench.decide(correct=True, graph=True, prefill_speedup=0.12, decode_regression=0.03) == "GO"
    assert bench.decide(correct=True, graph=True, prefill_speedup=0.05, decode_regression=0.08) == "FUNCTIONAL_ONLY"
    assert bench.decide(correct=False, graph=True, prefill_speedup=0.30, decode_regression=0.0) == "NO_GO"
    assert bench.decide(correct=True, graph=False, prefill_speedup=0.30, decode_regression=0.0) == "NO_GO"
```

同时断言 parser 包含：

```text
--tokens 1 8 128 8192 16384
--top-k 8
--profiles uniform synthetic-skew
--warmup 10
--trials 5
--iterations 100
--output-json
--check-cuda-graph
--cutlass-preflight
```

- [ ] **Step 2: 运行并确认脚本缺失**

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_2.py -q
```

Expected: FAIL with missing benchmark script。

- [ ] **Step 3: 实现 benchmark 的固定结果 schema**

每个 case 输出：

```python
{
    "tokens": tokens,
    "routed_rows": tokens * top_k,
    "top_k": top_k,
    "profile": profile,
    "correctness": {"status": "PASS", "calc_diff": calc_diff},
    "triton": {"median_ms": triton_ms, "trials_ms": triton_trials},
    "flashinfer_sm120_fp8": {
        "median_ms": flashinfer_ms,
        "trials_ms": flashinfer_trials,
    },
    "speedup_percent": (triton_ms / flashinfer_ms - 1.0) * 100.0,
    "components_ms": {
        "routing_quant_pack": routing_quant_pack_ms,
        "gemm1": gemm1_ms,
        "swiglu_quant": swiglu_quant_ms,
        "scale_layout_gemm2": scale_layout_gemm2_ms,
        "gemm2": gemm2_ms,
        "unpermute_combine": unpermute_combine_ms,
    },
}
```

`decide` 使用：

```python
def decide(*, correct, graph, prefill_speedup, decode_regression):
    if not correct or not graph:
        return "NO_GO"
    if prefill_speedup >= 0.10 and decode_regression <= 0.05:
        return "GO"
    return "FUNCTIONAL_ONLY"
```

CUTLASS preflight 必须实际执行 `cutlass_fused_experts_fp8` 的一个目标 shape；失败时
写入：

```python
{"status": "CUTLASS_UNAVAILABLE", "error": traceback.format_exc()}
```

不得因此跳过 FlashInfer correctness，也不得把 CUTLASS 失败记为零毫秒。

- [ ] **Step 4: 实现可恢复的 Stage 2 shell wrapper**

`run_stage_2_benchmark.sh` 使用：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/logs/sennian/pro5000-fi-moe
PYTHON="${ROOT}/.venv/bin/python3"
REPO="${ROOT}/sglang"
RUN_DIR="${ROOT}/runs/stage-2-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "${REPO}" rev-parse --short=12 HEAD)"
mkdir -p "${RUN_DIR}"

"${PYTHON}" "${REPO}/scripts/pro5000/collect_stage_b_env.py" \
  >"${RUN_DIR}/environment.json"

set +e
"${PYTHON}" "${REPO}/scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py" \
  --tokens 1 8 128 8192 16384 \
  --top-k 8 \
  --profiles uniform synthetic-skew \
  --warmup 10 \
  --trials 5 \
  --iterations 100 \
  --check-cuda-graph \
  --cutlass-preflight \
  --output-json "${RUN_DIR}/benchmark.json" \
  >"${RUN_DIR}/stdout.txt" 2>"${RUN_DIR}/stderr.txt"
STATUS=$?
set -e

echo "STAGE_2_STATUS=${STATUS}"
echo "STAGE_2_RUN_DIR=${RUN_DIR}"
exit "${STATUS}"
```

脚本不得锁频、不得删除目录、不得安装包；锁频只在用户明确批准的最终性能复测中
单独执行。

- [ ] **Step 5: 更新中文 README 的服务器步骤**

写明：fetch + detached checkout、editable install 无需重复执行时可直接跑 wrapper、
JIT warm-up 可能耗时、结果目录、tar 命令、GPU 频率恢复检查。结果打包命令固定为：

```bash
RUN_DIR=/home/logs/sennian/pro5000-fi-moe/runs/stage-2-<timestamp>-<commit>
tar -C "$(dirname "${RUN_DIR}")" -czf \
  /tmp/pro5000-stage2.tar.gz "$(basename "${RUN_DIR}")"
```

- [ ] **Step 6: 运行 CPU 测试和 shell syntax check 并提交**

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_2.py -q
bash -n scripts/pro5000/run_stage_2_benchmark.sh
git add scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  scripts/pro5000/run_stage_2_benchmark.sh \
  scripts/pro5000/README.md \
  test/registered/unit/test_pro5000_stage_2.py
git commit -m "bench: add FlashInfer SM120 FP8 MoE stage 2"
```

### Task 7: 真实模型服务与最终验收

**Files:**
- Modify only if a reproducible defect is found: files owned by Tasks 1-6
- Record artifacts under server path:
  `/home/logs/sennian/pro5000-fi-moe/runs/model-stage-2-*`

**Interfaces:**
- Consumes: 已通过 Task 5/6 的 backend 和 benchmark。
- Produces: 真实 Qwen3.5 服务日志、请求输出、显存和性能记录，以及最终
  `GO/FUNCTIONAL_ONLY/NO_GO` 结论。

- [ ] **Step 1: 同步到已提交 commit 并确认环境**

用户在服务器执行：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe

VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" - <<'PY'
import torch
from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
from sglang.srt.layers.moe.utils import MoeRunnerBackend
print("torch", torch.__version__, torch.version.cuda)
print("capability", torch.cuda.get_device_capability())
print("backend", MoeRunnerBackend("flashinfer_sm120_fp8"))
print("api", moe_gemm_fp8_nt_groupwise.__name__)
PY
```

Expected: capability `(12, 0)`、backend 枚举成功、目标 API 可导入。

- [ ] **Step 2: 运行完整测试与 Stage 2 benchmark**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  test/registered/unit/test_pro5000_stage_2.py \
  -q -s
bash scripts/pro5000/run_stage_2_benchmark.sh
```

Expected: pytest PASS，wrapper `STAGE_2_STATUS=0`，JSON 中所有 correctness PASS 且
CUDA Graph PASS。

- [ ] **Step 3: 启动目标模型服务**

运行前由用户设置真实模型路径；命令用 shell 参数检查拒绝未设置值，不在脚本中保留
占位路径：

```bash
: "${MODEL_PATH:?请先 export MODEL_PATH 为 Qwen3.5-35B-A3B-FP8 的真实路径}"
export SERVER_LOG=/home/logs/sennian/pro5000-fi-moe/runs/model-stage-2-server.log

/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size 1 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --moe-runner-backend flashinfer_sm120_fp8 \
  --chunked-prefill-size 8192 \
  2>&1 | tee "${SERVER_LOG}"
```

Expected: 启动日志只声明一次 `flashinfer_sm120_fp8`；不存在 Triton/CUTLASS fallback
日志；FlashInfer JIT 在 capture 前完成。

- [ ] **Step 4: 覆盖真实请求矩阵**

分别运行：短 prompt + 256 decode tokens、长 prompt、batch decode、8192
chunked-prefill、多轮连续请求和并发请求。每组保存请求 JSON、响应 JSON、服务器日志
片段和 `nvidia-smi` 显存记录。硬条件：无 NaN/Inf、乱码、illegal memory access、
JIT-in-capture、显存持续增长。

- [ ] **Step 5: 仅在用户批准 GPU 独占和锁频后做最终复测**

复用 Stage 1 已验证的锁频/恢复流程；先记录默认频率，再对 Triton、CUTLASS（若
preflight 可用）和 FlashInfer 分别重启进程测试，确保完全相同模型、请求、clock、
warmup、trial 数。测试结束立即恢复默认频率并运行：

```bash
nvidia-smi -q -d CLOCK | sed -n '1,220p'
```

- [ ] **Step 6: 根据固定门槛给出结论**

```text
GO:
  correctness/CUDA Graph/真实服务全部通过；
  M=65536 完整 runner 相对 Triton >=10%；
  decode 稳定回退 <=5%。

FUNCTIONAL_ONLY:
  功能全部正确，但 prefill 收益不足或 decode 回退 >5%；
  保持显式实验 backend，不增加本计划范围外的自动回退。

NO_GO:
  correctness、CUDA Graph 或真实服务稳定性任一失败。
```

发现失败时只修复具有可复现测试的根因；修复前先补失败测试，修复后重跑 Task 5、
Task 6 和相应真实请求，不用提高 tolerance 或跳过失败 case 来获得 PASS。

## Final Verification

- [ ] CPU/unit：

```bash
python3 -m pytest \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py \
  test/registered/unit/test_pro5000_stage_2.py -q
```

- [ ] SM120 GPU：

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest test/registered/moe/test_flashinfer_sm120_fp8_moe.py -q -s
```

- [ ] 静态检查：

```bash
git diff --check
bash -n scripts/pro5000/run_stage_2_benchmark.sh
python3 -m compileall -q \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py
```

- [ ] 工作树和 commit：

```bash
git status --short --branch
git log --oneline --decorate -8
```

Expected: 无未解释的修改；每个 task 为独立 commit；没有把服务器结果、模型文件或
wheel 提交进 Git。
