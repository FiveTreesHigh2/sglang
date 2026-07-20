# RTX PRO 5000 Stage 1 FP8 MoE Kernel 微基准 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不修改 SGLang MoE runner 和 Python 依赖的前提下，新增可复现的
FlashInfer `moe_gemm_fp8_nt_groupwise` 与 SGLang Triton
`fused_moe_kernel` 低层微基准，并由用户在 RTX PRO 5000 服务器上完成 Stage 1
go/no-go 实验。

**Architecture:** 一个 CPU-safe 的 Python 入口负责构造路由 profile、共享 FP8
数据、两种 scale layout、FP32 参考、低层 kernel 调用、交替计时和 JSON 决策；
一个 Bash wrapper 只负责固定 venv、run 目录、环境/包/GPU 快照与退出码留档。
现有 Stage B smoke 只做小范围 helper 提取，保持服务器已经验证过的调用语义。

**Tech Stack:** Python 3.12、PyTorch 2.11.0+cu130、Triton、FlashInfer
0.6.15.dev20260716、Bash、pytest、uv、Git。

**Approved Design:**
`docs/superpowers/specs/2026-07-20-pro5000-stage-1-kernel-benchmark-design.md`

---

## 全局约束

- 开始实现前，目标分支必须为 `feat/flashinfer-sm120-fp8-moe`，工作树必须干净。
- Stage 1 不修改 `python/pyproject.toml`、MoE runner、后端枚举或 FlashInfer 源码。
- 不执行包安装；包检查只能使用 `uv pip --python`，禁止裸 `pip` 和
  `python -m pip`。
- 正常 benchmark 不得设置 `FLASHINFER_DISABLE_JIT=1`。
- wrapper 不得自动执行 `nvidia-smi -lgc` 或 `nvidia-smi -rgc`。
- 本地没有 RTX PRO 5000；本地只声称静态/CPU 测试通过，GPU 验收必须由用户
  在服务器执行并返回产物。
- 每个任务遵循：写失败测试 -> 确认失败原因 -> 最小实现 -> 重新验证 -> 独立
  提交。
- 不把 synthetic-skew 描述为真实 Qwen3.5 路由。
- 性能 `NO_GO` / `NEEDS_LOCKED_RERUN` 是有效结果，进程退出码仍为 0；环境、
  JIT 或正确性失败才非零退出。

---

### Task 1：为 Stage 1 建立共享 scale 对齐 helper

**Files:**

- Create: `test/registered/unit/test_pro5000_stage_1.py`
- Modify: `scripts/pro5000/flashinfer_sm120_fp8_smoke.py`

**Purpose:** 在不改变 Stage B smoke 输出契约的情况下，把 FlashInfer A-scale
4 行对齐的 copy plan 暴露为可复用纯函数，避免 benchmark 复制并逐渐偏离已验证
逻辑。

- [ ] **Step 1：先写 copy-plan 失败测试**

创建 `test/registered/unit/test_pro5000_stage_1.py`：

```python
from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRO5000_SCRIPTS = REPO_ROOT / "scripts" / "pro5000"


@contextmanager
def pro5000_scripts_on_path():
    sys.path.insert(0, str(PRO5000_SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(PRO5000_SCRIPTS))


def load_script(filename: str):
    path = PRO5000_SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pro5000_scripts_on_path():
        spec.loader.exec_module(module)
    return module


def test_flashinfer_scale_copy_plan_preserves_empty_experts() -> None:
    smoke = load_script("flashinfer_sm120_fp8_smoke.py")
    assert smoke.build_scale_copy_plan([0, 0, 1, 9, 9, 12]) == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 9, 4),
        (9, 9, 16),
        (9, 12, 20),
    ]
```

这里的 tuple 为 `(source_start, source_end, packed_start)`。

- [ ] **Step 2：运行测试并确认 helper 尚不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_1.py::test_flashinfer_scale_copy_plan_preserves_empty_experts \
  -q
```

Expected: FAIL with `AttributeError: ... build_scale_copy_plan`。

- [ ] **Step 3：在 smoke 中实现纯 copy plan 并复用**

在 `build_offsets` 后增加：

```python
def build_scale_copy_plan(offsets: list[int]) -> list[tuple[int, int, int]]:
    if not offsets or offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("offsets must be non-decreasing")
    return [
        (start, end, compute_padded_offset(start, expert_id))
        for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:]))
    ]
```

将 `quantize_and_pack_a` 中的循环改为消费该 plan；空 expert 继续跳过，最终
`m_padded` 仍使用：

```python
m_padded = compute_padded_offset(x.shape[0], num_experts)
for start, end, packed_start in build_scale_copy_plan(m_indptr.tolist()):
    if start == end:
        continue
    packed[:, packed_start : packed_start + end - start] = (
        scale_row_major[start:end].t()
    )
```

实现时不要在 `build_scale_copy_plan` 中 import torch，确保它是 CPU 纯逻辑。
若 `m_indptr.tolist()` 引入 GPU 同步，只允许出现在当前 smoke/benchmark 的
untimed 准备阶段。

- [ ] **Step 4：回归 Stage B 与新测试**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py \
  test/registered/unit/test_pro5000_stage_1.py \
  -q
```

Expected: 原 Stage B 11 项和新测试全部 PASS。

- [ ] **Step 5：提交 helper**

```bash
git add \
  scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  test/registered/unit/test_pro5000_stage_1.py
git commit -m "test: share FlashInfer FP8 scale packing contract"
```

---

### Task 2：实现 profile、外部回放、统计与决策纯逻辑

**Files:**

- Create: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py`
- Modify: `test/registered/unit/test_pro5000_stage_1.py`

**Purpose:** 先完成不 import torch/flashinfer/triton 的可测试控制面；GPU 实现随后
填入同一入口。

- [ ] **Step 1：写 profile 和 JSON 校验失败测试**

追加：

```python
import json


def test_uniform_profile_has_exact_total() -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    rows = bench.build_uniform_rows(num_experts=256, cum_m=65536)
    assert rows == [256] * 256


def test_synthetic_skew_is_deterministic_and_stressful() -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    first = bench.build_synthetic_skew_rows(256, 65536, seed=42, block_size=64)
    second = bench.build_synthetic_skew_rows(256, 65536, seed=42, block_size=64)
    assert first == second
    assert len(first) == 256
    assert sum(first) == 65536
    assert min(first) == 0
    assert any(0 < rows < 64 for rows in first)
    assert max(first) > 4 * (65536 / 256)


def test_rows_json_accepts_array_or_object(tmp_path: Path) -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    rows = [1] * 256
    array_path = tmp_path / "array.json"
    object_path = tmp_path / "object.json"
    array_path.write_text(json.dumps(rows))
    object_path.write_text(json.dumps({"rows_per_expert": rows}))
    assert bench.load_rows_per_expert(array_path, 256, 256) == rows
    assert bench.load_rows_per_expert(object_path, 256, 256) == rows
```

另加 invalid length、负数和 sum 不匹配的 `pytest.raises(ValueError)` case。

- [ ] **Step 2：写统计和边界判定失败测试**

追加：

```python
def test_latency_summary_uses_median_and_relative_spread() -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    summary = bench.summarize_latencies([1.0, 1.1, 0.9, 1.0, 1.0])
    assert summary["min_ms"] == 0.9
    assert summary["median_ms"] == 1.0
    assert summary["max_ms"] == 1.1
    assert summary["relative_spread"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("speedup", "triton_spread", "flashinfer_spread", "expected"),
    [
        (0.31, 0.04, 0.04, "GO"),
        (0.30, 0.05, 0.05, "GO"),
        (0.30, 0.06, 0.04, "NEEDS_LOCKED_RERUN"),
        (0.20, 0.02, 0.02, "NEEDS_LOCKED_RERUN"),
        (0.15, 0.02, 0.02, "NEEDS_LOCKED_RERUN"),
        (0.149, 0.02, 0.02, "NO_GO"),
    ],
)
def test_default_boost_decision_boundaries(
    speedup: float,
    triton_spread: float,
    flashinfer_spread: float,
    expected: str,
) -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    decision = bench.decide_default_boost(
        speedup=speedup,
        min_trial_speedup=0.20,
        triton_relative_spread=triton_spread,
        flashinfer_relative_spread=flashinfer_spread,
        environment_stable=True,
        correctness_passed=True,
    )
    assert decision["status"] == expected
```

再覆盖 `correctness_passed=False -> ERROR`、`environment_stable=False ->
NEEDS_LOCKED_RERUN`、总 speedup 大于 30% 但任一 paired trial 小于 20%时必须
`NEEDS_LOCKED_RERUN`，以及锁频 `speedup >= 0.20 -> GO` 的边界。

- [ ] **Step 3：运行测试并确认 benchmark 文件尚不存在**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_1.py -q
```

Expected: 新增测试 FAIL with `FileNotFoundError`。

- [ ] **Step 4：创建 CPU-safe benchmark 骨架**

文件顶层只 import 标准库和已 CPU-safe 的 smoke helpers：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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
```

实现以下纯函数：

```python
def build_uniform_rows(num_experts: int, cum_m: int) -> list[int]: ...
def build_synthetic_skew_rows(
    num_experts: int, cum_m: int, *, seed: int, block_size: int
) -> list[int]: ...
def validate_rows_per_expert(
    rows: Sequence[int], num_experts: int, cum_m: int
) -> list[int]: ...
def load_rows_per_expert(
    path: Path, num_experts: int, cum_m: int
) -> list[int]: ...
def summarize_latencies(values_ms: Sequence[float]) -> dict[str, Any]: ...
def decide_default_boost(...) -> dict[str, Any]: ...
def decide_locked(...) -> dict[str, Any]: ...
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...
```

synthetic-skew 使用 `random.Random(seed)` 打乱 expert ID：固定一组 empty experts、
固定一组 `1..BLOCK_SIZE_M-1` 的 small experts，再用可重复权重和 largest-remainder
方法分配其余行。最后必须通过 `validate_rows_per_expert`；不要调用
`torch.multinomial`，避免跨 PyTorch/CUDA 版本变化。

CLI 默认值必须为：

```text
--operations gemm1 gemm2
--profiles uniform synthetic-skew
--cum-m 65536 131072
--warmup 20
--iterations 100
--trials 5
--seed 42
--clock-mode default
--output <required by wrapper; direct run may omit>
```

允许 `--rows-per-expert-json`，但它与 `--profiles` 中的生成 profile 互斥，并要求
命令中只有一个 `--cum-m`。

- [ ] **Step 5：运行纯逻辑测试**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_1.py -q
```

Expected: Task 1/2 全部 PASS；导入 benchmark 不初始化 CUDA。

- [ ] **Step 6：提交纯控制面**

```bash
git add \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  test/registered/unit/test_pro5000_stage_1.py
git commit -m "bench: add Pro5000 Stage 1 benchmark controls"
```

---

### Task 3：实现共享 FP8 case、FP32 参考和两种低层 backend

**Files:**

- Modify: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py`
- Modify: `test/registered/unit/test_pro5000_stage_1.py`

**Purpose:** 在计时外构造同一份 FP8 A/B 与 scale 数值，并确保 FlashInfer 和
Triton 只执行 GEMM。

- [ ] **Step 1：写 backend 契约静态失败测试**

追加：

```python
def test_benchmark_uses_low_level_kernels_without_quant_wrapper() -> None:
    source = (
        PRO5000_SCRIPTS / "benchmark_flashinfer_sm120_fp8_moe.py"
    ).read_text()
    assert "moe_gemm_fp8_nt_groupwise(" in source
    assert "fused_moe_kernel[grid](" in source
    assert "invoke_fused_moe_kernel(" not in source
    assert "out=case.flashinfer_output" in source
    assert "FLASHINFER_DISABLE_JIT=1" not in source


def test_benchmark_records_both_scale_layouts() -> None:
    source = (
        PRO5000_SCRIPTS / "benchmark_flashinfer_sm120_fp8_moe.py"
    ).read_text()
    assert "a_scale_row_major" in source
    assert "a_scale_flashinfer" in source
    assert "b_scale_triton" in source
    assert "b_scale_flashinfer" in source
```

- [ ] **Step 2：运行并确认低层 backend 尚未实现**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_1.py::test_benchmark_uses_low_level_kernels_without_quant_wrapper \
  test/registered/unit/test_pro5000_stage_1.py::test_benchmark_records_both_scale_layouts \
  -q
```

Expected: FAIL on missing source strings。

- [ ] **Step 3：定义 case 数据结构与 shape 表**

在 GPU 函数内部再 import torch；文件顶层保持 CPU-safe。增加：

```python
OP_SHAPES = {
    "gemm1": {"n": 1024, "k": 2048, "historical_ms": 1.267},
    "gemm2": {"n": 2048, "k": 512, "historical_ms": 0.794},
}


@dataclass
class QuantizedCase:
    operation: str
    profile: str
    rows_per_expert: list[int]
    offsets: list[int]
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
```

`dataclass` 需从标准库导入；Tensor 类型保持 `Any`，避免模块导入时加载 torch。

- [ ] **Step 4：一次量化生成两个 A-scale 与两个 B-scale layout**

实现 `make_quantized_case(...)`：

```python
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

a_fp8, a_scale_row_major = per_token_cast_to_fp8(a_bf16)

a_scale_flashinfer = torch.zeros(
    (k // 128, compute_padded_offset(cum_m, num_experts)),
    dtype=torch.float32,
    device="cuda",
)
for start, end, packed_start in build_scale_copy_plan(offsets):
    if start != end:
        a_scale_flashinfer[:, packed_start : packed_start + end - start] = (
            a_scale_row_major[start:end].T
        )

# per_block_cast_to_fp8 returns the Triton semantic [N_blocks, K_blocks].
b_scale_triton = torch.stack(b_scale_parts).contiguous()
b_scale_flashinfer = b_scale_triton.transpose(-1, -2).contiguous()
```

紧接量化后 assert dtype、shape、contiguous 和 16-byte alignment。A/B 原始 BF16
tensor 在 reference 构造完成后释放；每个 case 顺序执行，避免同时保留 8 组大
tensor。

- [ ] **Step 5：构造 Triton production-like config 与 padding metadata**

复用：

```python
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
```

config selection 使用完整 Qwen shape pair、`top_k=8` 和原始 token 数
`cum_m // 8`：

```python
up_config, (down_config, _) = try_get_optimal_moe_config(
    (256, 1024, 2048),
    (256, 2048, 512),
    8,
    "fp8_w8a8",
    cum_m // 8,
    block_shape=[128, 128],
    return_down_config=True,
)
config = dict(up_config if operation == "gemm1" else (down_config or up_config))
```

如果 config 包含 `USE_TMA=True`，Stage 1 首版必须明确报错并要求显式 non-TMA
config；不得静默改变生产 config。`USE_TMA=False` 则先从传给 Triton kernel 的
config 中移除，避免把未知 meta-parameter 传入 kernel。目标 Pro5000 当前没有
专用配置文件，预期使用
blockwise default：`BM=64, BN=128, BK=128, 4 warps, 3 stages`。

packed A 已按 expert 排序，因此构造 `topk_ids` shape `(cum_m, 1)`、
`topk_weights=1`，再调用：

```python
sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    topk_ids, config["BLOCK_SIZE_M"], 256
)
```

保留 `sorted_token_ids.shape[0]` 的 worst-case grid allocation；不要裁剪到
`num_tokens_post_padded`。这正是当前 Triton grid 为 GEMM1 产生 10216 CTA 的
关键行为。

- [ ] **Step 6：实现 FP32 反量化参考**

实现 `build_fp32_reference(case)`，按 128-K/128-N block 扩展 scale：

```python
a_dequant = case.a_fp8.float() * case.a_scale_row_major.repeat_interleave(
    128, dim=1
)[:, :k]

for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:])):
    if start == end:
        continue
    b_scale = case.b_scale_triton[expert_id]
    b_dequant = case.b_fp8[expert_id].float() * b_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)[:n, :k]
    reference[start:end] = a_dequant[start:end] @ b_dequant.T
```

reference dtype 为 FP32。构造 reference 时临时禁用 TF32，结束后恢复用户原值；
不要永久修改全局 matmul 设置。

- [ ] **Step 7：实现 FlashInfer 低层调用并复用 out**

```python
def launch_flashinfer(case: QuantizedCase) -> None:
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
    assert result.data_ptr() == case.flashinfer_output.data_ptr()
```

- [ ] **Step 8：直接调用 Triton `fused_moe_kernel`**

实现 `launch_triton(case)`，不要调用会重新量化 A 的
`invoke_fused_moe_kernel`。核心 launch 必须显式传入已经量化的数据：

```python
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_moe_kernel,
    should_enable_swap_ab,
)

grid = lambda meta: (
    triton.cdiv(case.sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
    * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
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
    n,
    k,
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
    128,
    128,
    MUL_ROUTED_WEIGHT=False,
    top_k=1,
    compute_type=tl.bfloat16,
    use_fp8_w8a8=True,
    use_int8_w8a8=False,
    use_int8_w8a16=False,
    per_channel_quant=False,
    even_Ks=(k % case.triton_config["BLOCK_SIZE_K"] == 0),
    c_sorted=False,
    filter_expert=False,
    swap_ab=should_enable_swap_ab(
        case.triton_config["BLOCK_SIZE_M"],
        case.triton_config["BLOCK_SIZE_N"],
    ),
    FUSE_ADD_TO_OUTPUT=False,
    MASK_OUTPUT=False,
    LORA_PRESERVE_BASE=False,
    FUSE_SUM_ALL_REDUCE=False,
    ROUTER_TOPK=1,
    **case.triton_config,
)
```

在实现时逐项对照当前 commit 的 `invoke_fused_moe_kernel` 参数顺序；若 upstream
签名与本计划不同，以当前源码为准并在 plan execution 记录差异，不能凭猜测
调整。

- [ ] **Step 9：实现 correctness gate**

`validate_correctness(case)` 必须检查：

```python
assert output.shape == (cum_m, n)
assert output.dtype == torch.bfloat16
assert torch.isfinite(output).all().item()
assert calc_diff(output.float(), reference) <= 2e-3
```

分别记录 Triton-vs-reference、FlashInfer-vs-reference 和两 backend 之间的
`calc_diff`。任一失败抛出带 operation/profile/cum_m/backend 的异常。

- [ ] **Step 10：运行本地契约测试并提交**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py \
  test/registered/unit/test_pro5000_stage_1.py \
  -q
python3 -m py_compile \
  scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py
```

Expected: 全部 PASS；本地不执行 GPU case。

```bash
git add \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  test/registered/unit/test_pro5000_stage_1.py
git commit -m "bench: compare FlashInfer and Triton FP8 MoE kernels"
```

---

### Task 4：实现 JIT prepare、交替计时、GPU 采样与 JSON

**Files:**

- Modify: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py`
- Modify: `test/registered/unit/test_pro5000_stage_1.py`

**Purpose:** 完成可执行 benchmark，同时严格区分 prepare/correctness/timing。

- [ ] **Step 1：写 trial order 与输出 schema 失败测试**

追加：

```python
def test_trial_order_alternates_backends() -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    assert bench.trial_backend_order(0) == ("triton", "flashinfer")
    assert bench.trial_backend_order(1) == ("flashinfer", "triton")


def test_result_schema_has_reproducibility_fields() -> None:
    bench = load_script("benchmark_flashinfer_sm120_fp8_moe.py")
    payload = bench.empty_result_payload(["benchmark.py"])
    assert set(payload) >= {
        "schema_version",
        "timestamp_utc",
        "command",
        "git",
        "environment",
        "parameters",
        "cases",
        "decision",
        "status",
    }
```

- [ ] **Step 2：运行并确认 helper 不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_1.py::test_trial_order_alternates_backends \
  test/registered/unit/test_pro5000_stage_1.py::test_result_schema_has_reproducibility_fields \
  -q
```

Expected: FAIL with `AttributeError`。

- [ ] **Step 3：实现 JIT prepare 与 CUDA Event 计时**

```python
def prepare_backend(name: str, fn: Callable[[], None]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def time_backend(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations
```

每个 backend 每个 case 只做一次 prepare 和一次 warmup；5 个 trial 不重复 warmup。
如果为了函数复用把 warmup 与 trial 分开，实现应反映该语义并相应调整签名。

- [ ] **Step 4：实现 trial 外 GPU snapshot**

使用 `subprocess.run` 调用：

```text
nvidia-smi --query-gpu=index,pstate,clocks.current.sm,temperature.gpu,power.draw,
utilization.gpu,memory.used --format=csv,noheader,nounits
```

在每个 backend trial 前后采样，但绝不放入 CUDA Event 区间。查询失败不隐藏；
记录 returncode/stderr，并把环境稳定性设为 unknown，从而使主结果至少进入
`NEEDS_LOCKED_RERUN`，而不是错误地 `GO`。

首版把 `environment_stable` 精确定义为：所有采样成功、P-state 全程不变，且
观测到的 SM clock relative spread 不超过 5%。温度、功耗、utilization 和显存
只记录，不设置未经硬件验证的绝对阈值。

- [ ] **Step 5：实现 case result 与总决策**

每个 case JSON 至少包括：

```python
{
    "operation": operation,
    "profile": profile,
    "cum_m": cum_m,
    "n": n,
    "k": k,
    "rows_per_expert": rows,
    "rows_summary": {...},
    "triton_config": config,
    "triton_grid_size": grid_size,
    "num_tokens_post_padded": actual_padded,
    "jit_prepare_seconds": {"triton": ..., "flashinfer": ...},
    "correctness": {...},
    "latency": {
        "triton": {"trials_ms": [...], ...},
        "flashinfer": {"trials_ms": [...], ...},
    },
    "speedup": ...,
    "historical_reference_ms": ...,
    "gpu_samples": [...],
}
```

总决策只读取 `("gemm1", "uniform", 65536)`。非默认 case 集合（例如服务器
preflight）返回 `NOT_EVALUATED`，退出码仍为 0。

默认 boost 的 `GO` 还必须满足每个 paired trial 的 speedup 都不低于 20%；总
median speedup 虽达到 30%，但有单轮低于 20%时，应返回
`NEEDS_LOCKED_RERUN`。锁频模式若 relative spread 仍超过 5%，同样不能给出
最终 `GO`。

- [ ] **Step 6：实现异常时的 partial JSON**

`main()` 先创建 `status="running"` 的 payload。任何异常都：

1. 设置 `status="error"`；
2. 设置 `decision.status="ERROR"`；
3. 写入异常阶段、case、type 和 message；
4. 尽力写到 `--output`；
5. 将 traceback 输出到 stderr；
6. 返回非零退出。

正常性能结果设置 `status="completed"` 并返回 0。JSON 写入使用同目录临时文件
再 `replace`，避免中断留下看似完整的半个 JSON。

- [ ] **Step 7：输出人类可读表格**

stdout 每个 case 打印一行：

```text
operation profile cum_m triton_ms flashinfer_ms speedup correctness decision
```

最后单独打印：

```text
STAGE_1_DECISION=<...>
STAGE_1_RESULT_JSON=<absolute path or stdout-only>
```

不要把完整大 JSON 再打印到 stdout；wrapper 已保存 `benchmark.json`。

- [ ] **Step 8：运行 CPU 测试并提交**

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py \
  test/registered/unit/test_pro5000_stage_1.py \
  -q
python3 -m py_compile scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py
git diff --check
```

Expected: 全部 PASS。

```bash
git add \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  test/registered/unit/test_pro5000_stage_1.py
git commit -m "bench: record Pro5000 FP8 MoE performance gate"
```

---

### Task 5：增加幂等运行 wrapper 与产物留档

**Files:**

- Create: `scripts/pro5000/run_stage_1_benchmark.sh`
- Modify: `test/registered/unit/test_pro5000_stage_1.py`

**Purpose:** 用户只运行一条命令即可使用固定 Stage B venv，并无论成功失败都
保留可诊断产物。

- [ ] **Step 1：写 wrapper 静态失败测试**

追加：

```python
import re
import subprocess


def test_stage_1_wrapper_is_safe_and_uses_existing_venv() -> None:
    script = PRO5000_SCRIPTS / "run_stage_1_benchmark.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    content = script.read_text()
    assert '${PRO5000_ROOT}/.venv/bin/python3' in content
    assert "uv pip check --python" in content
    assert "uv pip freeze --python" in content
    assert "benchmark_flashinfer_sm120_fp8_moe.py" in content
    assert "FLASHINFER_WORKSPACE_BASE" in content
    assert "FLASHINFER_DISABLE_JIT=1" not in content
    assert "nvidia-smi -lgc" not in content
    assert "nvidia-smi -rgc" not in content
    assert re.search(r"(?m)^\s*pip\s", content) is None
    assert re.search(r"python3?\s+-m\s+pip", content) is None
```

- [ ] **Step 2：运行并确认脚本不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_1.py::test_stage_1_wrapper_is_safe_and_uses_existing_venv \
  -q
```

Expected: FAIL because file does not exist / `bash -n` nonzero。

- [ ] **Step 3：创建 wrapper**

脚本骨架：

```bash
#!/usr/bin/env bash
set -euo pipefail

PRO5000_ROOT="${PRO5000_ROOT:-/home/logs/sennian/pro5000-fi-moe}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON="${PRO5000_ROOT}/.venv/bin/python3"
RUNS_DIR="${PRO5000_ROOT}/runs"
EXPECTED_REPO="${PRO5000_ROOT}/sglang"

if [[ -n "${FLASHINFER_DISABLE_JIT:-}" ]]; then
  echo "ERROR: unset FLASHINFER_DISABLE_JIT for Stage 1" >&2
  exit 1
fi

export FLASHINFER_WORKSPACE_BASE="${PRO5000_ROOT}/cache/flashinfer-workspace-base"
```

必须验证：Linux x86_64、repo 路径、工作树 clean、venv Python 可执行、Python
3.12、`uv`、`nvidia-smi`、`nvcc`、`git` 均存在。

创建：

```text
stage-1-<UTC timestamp>-<12-char git sha>
```

然后按顺序执行：

```bash
uv pip check --python "${PYTHON}" >"${RUN_DIR}/pip-check-before.txt"
uv pip freeze --python "${PYTHON}" >"${RUN_DIR}/packages-before.txt"
"${PYTHON}" scripts/pro5000/collect_stage_b_env.py >environment.json
nvidia-smi >nvidia-smi-before.txt
```

运行 benchmark 时 `set +e` 捕获状态，stdout/stderr 分开留档；随后无条件采集
after 环境、package freeze 和 `nvidia-smi`。比较 before/after package 文件，若
发生变化则把 wrapper 视为环境错误。

使用 Bash process substitution 和 `tee` 同时实时显示并保存输出：

```bash
"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py" \
  ... \
  > >(tee "${RUN_DIR}/benchmark.stdout.txt") \
  2> >(tee "${RUN_DIR}/benchmark.stderr.txt" >&2)
BENCHMARK_STATUS=$?
```

这样首次 JIT 即使持续约 100 秒，用户仍能看到脚本已经进入哪个 case；同时不得
用 `tee` 的状态覆盖 Python 原始退出码。wrapper 的 required commands 需包含
`tee` 和 `cmp`。

wrapper 默认不传 shape/profile 覆盖，只传：

```bash
--clock-mode "${STAGE1_CLOCK_MODE:-default}"
--output "${RUN_DIR}/benchmark.json"
```

最后打印：

```text
STAGE_1_STATUS=<python exit status>
STAGE_1_RUN_DIR=<absolute run dir>
```

并返回 Python 原始非零状态；若 Python 成功而环境 after 检查失败，则返回 wrapper
错误状态。

- [ ] **Step 4：运行 shell/静态测试**

```bash
bash -n scripts/pro5000/run_stage_1_benchmark.sh
python3 -m pytest test/registered/unit/test_pro5000_stage_1.py -q
git diff --check
```

Expected: PASS。

- [ ] **Step 5：提交 wrapper**

```bash
git add \
  scripts/pro5000/run_stage_1_benchmark.sh \
  test/registered/unit/test_pro5000_stage_1.py
git commit -m "bench: add reproducible Pro5000 Stage 1 runner"
```

---

### Task 6：补充中文服务器运行手册

**Files:**

- Modify: `scripts/pro5000/README.md`
- Modify: `test/registered/unit/test_pro5000_stage_1.py`

**Purpose:** 把默认 boost、结果解读和条件锁频拆成清晰步骤，避免用户误把 CUDA
版本、SM compute capability 和 GPU clock 混为一谈。

- [ ] **Step 1：写 README 内容失败测试**

追加：

```python
def test_readme_documents_stage_1_without_mandatory_clock_lock() -> None:
    content = (PRO5000_SCRIPTS / "README.md").read_text()
    assert "## Stage 1：FP8 MoE kernel 微基准" in content
    assert "bash scripts/pro5000/run_stage_1_benchmark.sh" in content
    assert "默认 boost 首测" in content
    assert "NEEDS_LOCKED_RERUN" in content
    assert "1732 MHz" in content
    assert "条件锁频" in content
    assert "uv pip --python" in content
    assert "FLASHINFER_DISABLE_JIT=1" not in content.split(
        "## Stage 1：FP8 MoE kernel 微基准", 1
    )[1]
```

- [ ] **Step 2：运行并确认 Stage 1 章节不存在**

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_1.py::test_readme_documents_stage_1_without_mandatory_clock_lock \
  -q
```

Expected: FAIL。

- [ ] **Step 3：追加 Stage 1 中文章节**

章节至少包含以下小节和命令。

**更新到精确 feature commit：**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
```

**只读依赖检查：**

```bash
uv pip check \
  --python /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
```

明确说明不要运行 `pip` 或 `python -m pip`，Stage 1 不需要安装任何包。

**默认 boost 首测：**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/run_stage_1_benchmark.sh
```

**查看产物：**

```bash
RUN_DIR="$(ls -1dt /home/logs/sennian/pro5000-fi-moe/runs/stage-1-* | head -n1)"
sed -n '1,240p' "${RUN_DIR}/benchmark.stdout.txt"
sed -n '1,320p' "${RUN_DIR}/benchmark.stderr.txt"
sed -n '1,360p' "${RUN_DIR}/benchmark.json"
```

**仅当结果为 `NEEDS_LOCKED_RERUN` 或最终验收时条件锁频：**

先说明必须确认 GPU 独占和权限，再给出可恢复的 subshell：

```bash
(
  set -e
  nvidia-smi -i 0 -lgc 1732,1732
  trap 'nvidia-smi -i 0 -rgc' EXIT
  STAGE1_CLOCK_MODE=locked \
    bash scripts/pro5000/run_stage_1_benchmark.sh
)
```

明确说明：`13.0` 是 CUDA toolkit 版本，`12.0` 是 GPU compute capability，
`1732 MHz` 是运行时 SM clock，三者不是同一个概念。

**返回给本地分析的文件：** `benchmark.json`、stdout/stderr、environment、
before/after `nvidia-smi` 和 package snapshots。

- [ ] **Step 4：运行测试并提交文档**

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py \
  test/registered/unit/test_pro5000_stage_1.py \
  -q
git diff --check
```

Expected: PASS。

```bash
git add scripts/pro5000/README.md test/registered/unit/test_pro5000_stage_1.py
git commit -m "docs: add Pro5000 Stage 1 benchmark runbook"
```

---

### Task 7：本地总验证、推送 fork 与服务器分层验收

**Files:**

- Verify only: all Stage 1 files
- External state: push `feat/flashinfer-sm120-fp8-moe` after local verification

- [ ] **Step 1：运行本地总验证**

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py \
  test/registered/unit/test_pro5000_stage_1.py \
  -q
bash -n scripts/pro5000/bootstrap_stage_b.sh
bash -n scripts/pro5000/run_stage_1_benchmark.sh
python3 -m py_compile \
  scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  scripts/pro5000/collect_stage_b_env.py
git diff --check
git status --short --branch
```

Expected:

- 所有 pytest PASS；
- 两个 Bash 脚本 syntax PASS；
- Python compile PASS；
- `git diff --check` 无输出；
- 工作树 clean。

- [ ] **Step 2：审计范围与危险行为**

```bash
git diff d09d91bdb..HEAD --stat
git diff d09d91bdb..HEAD --name-only
rg -n "FLASHINFER_DISABLE_JIT=1|nvidia-smi -(l|r)gc|python3? -m pip|(^|[^a-z])pip install" \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  scripts/pro5000/run_stage_1_benchmark.sh \
  scripts/pro5000/README.md
```

Expected:

- 只包含本计划声明的脚本、测试和 README；
- benchmark/wrapper 中没有 disable-JIT、自动锁频或裸 pip；
- README 只在人工“条件锁频”小节出现 `-lgc/-rgc`。

- [ ] **Step 3：推送 feature branch**

```bash
git push origin feat/flashinfer-sm120-fp8-moe
```

记录远端 HEAD，确认与本地一致：

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 4：用户在服务器先执行小规模 preflight**

更新 detached checkout 后：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe

export FLASHINFER_WORKSPACE_BASE=/home/logs/sennian/pro5000-fi-moe/cache/flashinfer-workspace-base
unset FLASHINFER_DISABLE_JIT

/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  --operations gemm1 gemm2 \
  --profiles synthetic-skew \
  --cum-m 4096 \
  --warmup 2 \
  --iterations 5 \
  --trials 1 \
  --clock-mode default \
  --output /tmp/pro5000-stage1-preflight.json
```

Expected:

- 两种 backend 都完成首次 JIT；
- GEMM1/GEMM2 correctness PASS；
- decision 为 `NOT_EVALUATED`；
- 退出码 0。

若失败，停止，不运行全量 benchmark；用户返回 `/tmp/pro5000-stage1-preflight.json`
和完整 stderr。

- [ ] **Step 5：用户运行默认 boost 全量 benchmark**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/run_stage_1_benchmark.sh
```

Expected: 最后打印 `STAGE_1_STATUS=0` 和 `STAGE_1_RUN_DIR=...`。正确性失败则
状态非零，并保留 partial JSON。

- [ ] **Step 6：按 decision 分支处理**

- `GO`：保留默认 boost 证据，进入 Stage 2 设计；最终报告前仍可补锁频。
- `NEEDS_LOCKED_RERUN`：确认 GPU 独占/权限后，按 README 条件锁频命令重跑。
- `NO_GO`：停止 runner 接入，复核数据后转向 Triton 调优。
- `ERROR`：先修正确性/环境，不作性能结论。

锁频结果的最终 gate：决定性 GEMM1 case `speedup >= 20%` 才进入 Stage 2。

- [ ] **Step 7：收集用户返回证据**

用户返回最新 `STAGE_1_RUN_DIR` 下：

```text
benchmark.json
benchmark.stdout.txt
benchmark.stderr.txt
environment.json
environment-after.json
nvidia-smi-before.txt
nvidia-smi-after.txt
pip-check-before.txt
pip-check-after.txt
packages-before.txt
packages-after.txt
```

分析时先核对 git SHA、dirty=false、package before/after 相同、correctness 全通过，
再解释 speedup；不能只依据 stdout 的单个延迟数字作结论。
