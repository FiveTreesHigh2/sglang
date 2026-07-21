# FlashInfer SM120 FP8 MoE A2 融合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 FlashInfer 源码的前提下，为 SGLang `flashinfer_sm120_fp8` runner 增加生产口径 benchmark，并把 GEMM2 前的 `SwiGLU + FP8 quant + A2 scale pack` 融合成一个 CUDA JIT kernel。

**Architecture:** 第一提交只升级 Stage 2 benchmark，在旧 runner 上生成 decode CUDA Graph baseline；第二阶段先写并运行 GPU RED 测试，再新增独立 CUDA JIT 变体、Python adapter 和 runner 接线。同一 benchmark 通过 call-trace 兼容 legacy/fused 两条路径，prefill 使用 eager、decode 使用 graph replay 做最终判定。

**Tech Stack:** Python 3.12、PyTorch 2.11/CUDA 13.0、SGLang JIT `load_jit`、CUDA C++、TVM-FFI、FlashInfer `0.6.15.dev20260716`、pytest/unittest。

## Global Constraints

- 目标硬件为单卡 NVIDIA RTX PRO 5000 72GB Blackwell，CUDA capability `(12, 0)`。
- Python 必须使用 `/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3`；安装命令只能使用 `uv pip`，本计划不新增依赖。
- FlashInfer API 固定为 `flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise`；GEMM1/GEMM2 都不得替换。
- Activation/weight dtype 固定为 `torch.float8_e4m3fn`，A/B scale 固定为 FP32，scale granularity 固定为 `(1, 128, 128)`。
- 不修改 FlashInfer 源码，不支持 DeepEP、多卡 EP、TP 大于 1，不增加静默 fallback。
- decode（tokens=1、8）正式性能使用 CUDA Graph replay；prefill 正式性能使用 eager。
- GO 仍要求 tokens=8192 uniform prefill speedup `>= 10%`、decode graph 最大回退 `<= 5%`、正确性和 CUDA Graph 全部通过。
- 完整 runner 阈值保持 `calc_diff < 0.005`、`symmetric_diff < 1e-4`、`normalized_rmse < 0.01`。
- 保留用户未跟踪文件 `test/registered/moe/debug_flashinfer_sm120_fp8_stagewise.py`，不得修改、删除或提交。
- 所有 production code 使用 TDD：先看到目标测试以预期原因失败，再写实现。

---

### Task 1: Stage 2 benchmark v3 契约与 decode graph 计时

**Files:**
- Modify: `test/registered/unit/test_pro5000_stage_2.py`
- Modify: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`

**Interfaces:**
- Consumes: 现有 `RunnerCase`、`launch_triton`、`launch_flashinfer`、`time_backend`、`run_cuda_graph_check`。
- Produces: `build_component_profile(detail_ms)`、decode case 的 `cuda_graph` latency、schema version 3、按 graph decode 判定的 `select_decision`。

- [ ] **Step 1: 写 graph/eager 判定的失败测试**

在 `TestPro5000Stage2` 中把 case fixture 改为同时含 eager 和 graph：

```python
def case(
    tokens,
    profile,
    triton_eager,
    flashinfer_eager,
    *,
    triton_graph=None,
    flashinfer_graph=None,
    status="PASS",
):
    result = {
        "tokens": tokens,
        "top_k": 8,
        "profile": profile,
        "correctness": {"status": status},
        "triton": {"median_ms": triton_eager},
        "flashinfer_sm120_fp8": {"median_ms": flashinfer_eager},
    }
    result["cuda_graph"] = (
        {
            "triton": {"median_ms": triton_graph},
            "flashinfer_sm120_fp8": {"median_ms": flashinfer_graph},
        }
        if triton_graph is not None and flashinfer_graph is not None
        else {"status": "NOT_RUN"}
    )
    return result
```

用以下数据证明 decode 判定忽略 eager、读取 graph：

```python
cases = [
    case(1, "uniform", 1.0, 2.0, triton_graph=1.0, flashinfer_graph=1.04),
    case(8, "synthetic-skew", 1.0, 2.0, triton_graph=1.0, flashinfer_graph=1.05),
    case(8192, "uniform", 10.0, 8.9),
]
decision = bench.select_decision(cases, cuda_graph_passed=True)
self.assertEqual(decision["status"], "GO")
self.assertAlmostEqual(decision["decode_regression"], 0.05)
```

另加 decode graph 缺失时的断言：

```python
with self.assertRaisesRegex(ValueError, "decode CUDA Graph"):
    bench.select_decision(
        [case(1, "uniform", 1.0, 1.0), case(8192, "uniform", 10.0, 8.9)],
        cuda_graph_passed=True,
    )
```

- [ ] **Step 2: 写 legacy/fused component fixture 的失败测试**

加入两套 detail：

```python
legacy_detail = {
    "quant1": 0.01,
    "moe_permute": 0.02,
    "scale_pack_gemm1": 0.03,
    "gemm1": 0.10,
    "silu": 0.04,
    "quant2": 0.05,
    "scale_pack_gemm2": 0.06,
    "gemm2": 0.20,
    "unpermute_combine": 0.07,
}
fused_detail = {
    "quant1": 0.01,
    "moe_permute": 0.02,
    "scale_pack_gemm1": 0.03,
    "gemm1": 0.10,
    "fused_swiglu_quant_pack_gemm2": 0.08,
    "gemm2": 0.20,
    "unpermute_combine": 0.07,
}
legacy = bench.build_component_profile(legacy_detail)
fused = bench.build_component_profile(fused_detail)
self.assertEqual(legacy["path"], "legacy")
self.assertEqual(fused["path"], "fused")
self.assertAlmostEqual(legacy["rollup_ms"]["gemm1_input_prepare"], 0.06)
self.assertAlmostEqual(legacy["rollup_ms"]["gemm2_input_prepare"], 0.15)
self.assertAlmostEqual(fused["rollup_ms"]["gemm2_input_prepare"], 0.08)
self.assertEqual(set(legacy["rollup_ms"]), set(fused["rollup_ms"]))
```

再用缺 stage 和混合 legacy/fused stage 的字典断言 `ValueError`。

- [ ] **Step 3: 运行 CPU 测试确认 RED**

Run:

```bash
python3 -m unittest test.registered.unit.test_pro5000_stage_2
```

Expected: FAIL，原因包括 `build_component_profile` 不存在、`select_decision` 仍从 eager 读取 decode。

- [ ] **Step 4: 实现稳定组件 schema**

在 benchmark 顶部替换旧 `COMPONENT_KEYS`：

```python
COMMON_COMPONENT_DETAIL_KEYS = frozenset(
    (
        "quant1",
        "moe_permute",
        "scale_pack_gemm1",
        "gemm1",
        "gemm2",
        "unpermute_combine",
    )
)
LEGACY_COMPONENT_EXTRA_KEYS = frozenset(("silu", "quant2", "scale_pack_gemm2"))
FUSED_COMPONENT_EXTRA_KEYS = frozenset(("fused_swiglu_quant_pack_gemm2",))
COMPONENT_ROLLUP_KEYS = (
    "gemm1_input_prepare",
    "gemm1",
    "gemm2_input_prepare",
    "gemm2",
    "unpermute_combine",
)
```

新增纯函数：

```python
def build_component_profile(detail_ms: dict[str, float]) -> dict[str, Any]:
    keys = frozenset(detail_ms)
    legacy_keys = COMMON_COMPONENT_DETAIL_KEYS | LEGACY_COMPONENT_EXTRA_KEYS
    fused_keys = COMMON_COMPONENT_DETAIL_KEYS | FUSED_COMPONENT_EXTRA_KEYS
    if keys == legacy_keys:
        path = "legacy"
        gemm2_input_prepare = sum(
            detail_ms[key] for key in ("silu", "quant2", "scale_pack_gemm2")
        )
    elif keys == fused_keys:
        path = "fused"
        gemm2_input_prepare = detail_ms["fused_swiglu_quant_pack_gemm2"]
    else:
        raise ValueError(
            "component detail must match exactly one legacy/fused schema; "
            f"got {sorted(keys)}"
        )
    rollup = {
        "gemm1_input_prepare": sum(
            detail_ms[key]
            for key in ("quant1", "moe_permute", "scale_pack_gemm1")
        ),
        "gemm1": detail_ms["gemm1"],
        "gemm2_input_prepare": gemm2_input_prepare,
        "gemm2": detail_ms["gemm2"],
        "unpermute_combine": detail_ms["unpermute_combine"],
    }
    return {"path": path, "detail_ms": dict(detail_ms), "rollup_ms": rollup}
```

- [ ] **Step 5: 重写组件 hook，使同一代码兼容旧/新 runner**

在 `profile_flashinfer_components` 中每次顶层 runner 调用前重置：

```python
call_counts = {
    "quant": 0,
    "pack": 0,
    "gemm": 0,
    "moe_permute": 0,
    "unpermute_combine": 0,
    "silu": 0,
    "fused": 0,
}
iteration_trace: list[str] = []
```

映射规则固定为：

```python
quant_labels = ("quant1", "quant2")
pack_labels = ("scale_pack_gemm1", "scale_pack_gemm2")
gemm_labels = ("gemm1", "gemm2")
```

每轮结束不仅校验 detail key，还要校验每个 symbol 的**精确调用次数**，防止重复调用被
`set` 吞掉：

```python
legacy_counts = {
    "quant": 2, "pack": 2, "gemm": 2,
    "moe_permute": 1, "unpermute_combine": 1,
    "silu": 1, "fused": 0,
}
fused_counts = {
    "quant": 1, "pack": 1, "gemm": 2,
    "moe_permute": 1, "unpermute_combine": 1,
    "silu": 0, "fused": 1,
}
```

调用顺序也必须分别等于：

```python
legacy_trace = [
    "quant1", "moe_permute", "scale_pack_gemm1", "gemm1",
    "silu", "quant2", "scale_pack_gemm2", "gemm2",
    "unpermute_combine",
]
fused_trace = [
    "quant1", "moe_permute", "scale_pack_gemm1", "gemm1",
    "fused_swiglu_quant_pack_gemm2", "gemm2", "unpermute_combine",
]
```

新 adapter hook 仅在：

```python
fused = getattr(
    flashinfer_runner,
    "fused_swiglu_quant_pack_flashinfer_sm120_fp8",
    None,
)
```

不为 `None` 时注册。不要根据通用 quant/pack symbol 存在与否判断路径。每次被 hook 的
函数都先追加 label、增加对应 count，再在 CUDA Event 区间内执行原函数；每轮结束把
`dict(component_samples)` 送入 `build_component_profile`，并按 profile 的 `path` 选择上述
count/trace 契约。

- [ ] **Step 6: 增加 graph capture/replay 计时**

新增：

```python
@dataclass
class CapturedGraph:
    graph: Any
    output: Any


def capture_backend_graph(fn: Callable[[], Any]) -> CapturedGraph:
    import torch

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return CapturedGraph(graph=graph, output=output)


def time_graph_replay(state: CapturedGraph, iterations: int) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        state.graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations
```

在 `run_benchmark_case` 中仅当 `case.tokens in DECODE_TOKENS` 时分别 capture
Triton/FlashInfer，保留两个 `CapturedGraph` 到 case 完成；按 trial 交替顺序填充：

```python
graph_latencies = {
    "triton": [],
    "flashinfer_sm120_fp8": [],
}
```

计时区间不得包含 input copy、capture 或 output clone。

- [ ] **Step 7: 把 graph latency 写入 case，升级判定**

case 继续把现有顶层 `triton`/`flashinfer_sm120_fp8` 解释为 eager；新增：

```python
"cuda_graph": {
    "triton": summarize_latencies(graph_triton_trials),
    "flashinfer_sm120_fp8": summarize_latencies(graph_flashinfer_trials),
}
```

非 decode case 写 `{"status": "NOT_RUN"}`。`select_decision` 的 prefill 仍读取顶层
eager，decode 必须读取 `case["cuda_graph"]`；缺失时抛出包含 `decode CUDA Graph`
的 `ValueError`。把 `empty_result_payload()["schema_version"]` 改为 `3`。

把 `build_case_result` 的 `components_ms` 参数改为 `component_profile`，case 写入：

```python
"components": {
    "path": component_profile["path"],
    "detail_ms": component_profile["detail_ms"],
    "rollup_ms": component_profile["rollup_ms"],
}
```

`_print_case_summary` 对 decode 追加 graph median；prefill 不打印伪造 graph 数字。

- [ ] **Step 8: 运行 CPU 回归并检查格式**

Run:

```bash
python3 -m unittest test.registered.unit.test_pro5000_stage_2
python3 -m compileall -q scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py
git diff --check
```

Expected: 全部 PASS，`compileall` 和 `git diff --check` 无输出。

- [ ] **Step 9: 提交 benchmark-only 变更**

```bash
git add scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  test/registered/unit/test_pro5000_stage_2.py
git commit -m "bench: measure SM120 FP8 decode graph replay"
```

---

### Task 2: 服务器检查点 A——旧 runner 基线

**Files:**
- Read: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
- Read: `scripts/pro5000/run_stage_2_benchmark.sh`
- Produce on server: `${PRO5000_ROOT}/runs/stage-2-*/benchmark.json`

**Interfaces:**
- Consumes: Task 1 commit，尚未融合的 production runner。
- Produces: 可归档的旧 runner eager/graph baseline；这是进入 Task 3 的硬门槛。

- [ ] **Step 1: 推送 Task 1 提交**

```bash
git push origin feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 2: 服务器同步到远端精确提交**

在服务器执行：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git status --short --branch
```

Expected: detached at Task 1 commit，工作区无修改。

- [ ] **Step 3: 服务器运行 CPU 契约测试**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest test/registered/unit/test_pro5000_stage_2.py -q
```

Expected: PASS。

- [ ] **Step 4: 运行旧 runner 低成本 graph preflight**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  --tokens 1 8 8192 \
  --top-k 8 \
  --profiles uniform synthetic-skew \
  --warmup 2 \
  --trials 2 \
  --iterations 20 \
  --check-cuda-graph \
  --output-json /tmp/pro5000-a2-legacy-preflight.json
```

Expected: exit 0；tokens=1、8 含 `cuda_graph.triton` 和
`cuda_graph.flashinfer_sm120_fp8`，tokens=8192 的 graph 状态为 `NOT_RUN`。

- [ ] **Step 5: 运行正式旧 runner baseline**

```bash
bash scripts/pro5000/run_stage_2_benchmark.sh
```

Expected: 输出 `STAGE_2_STATUS=0`、`STAGE_2_WRAPPER_STATUS=0` 和
`STAGE_2_RUN_DIR=...`。决策允许是 `FUNCTIONAL_ONLY`，但 correctness/graph 必须 PASS。

- [ ] **Step 6: 记录基线提交和关键数字**

```bash
RUN_DIR="$(ls -1dt /home/logs/sennian/pro5000-fi-moe/runs/stage-2-* | head -n1)"
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" - "${RUN_DIR}/benchmark.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("commit:", p["git"]["commit"])
for c in p["cases"]:
    if c["tokens"] in (1, 8, 8192):
        print(c["tokens"], c["profile"], "eager", c["triton"]["median_ms"], c["flashinfer_sm120_fp8"]["median_ms"], "graph", c["cuda_graph"])
print("decision:", p["decision"])
PY
```

把完整 `RUN_DIR` 压缩并传回本地。未保存该 artifact 前停止执行本计划。

---

### Task 3: 用 GPU RED 测试锁定 A2 adapter 契约

**Files:**
- Modify: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: `moe_permute`、`pack_flashinfer_sm120_fp8_scale`、现有 `silu_and_mul_contig_post_quant`。
- Produces: 新函数 `fused_swiglu_quant_pack_flashinfer_sm120_fp8` 的行为规范；本任务不创建该函数。

- [ ] **Step 1: 写非 identity `src2dst` 的 RED 测试**

在 SM120 test class 中增加：

```python
def test_fused_swiglu_quant_pack_matches_contig_reference(self):
    from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        fused_swiglu_quant_pack_flashinfer_sm120_fp8,
        pack_flashinfer_sm120_fp8_scale,
    )

    torch.manual_seed(17)
    tokens, top_k, experts, hidden = 8, 2, 8, 512
    topk_ids = torch.tensor(
        [[3, 0], [1, 1], [7, 2], [0, 6], [5, 3], [4, 0], [6, 6], [2, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    _, src2dst, m_indptr = moe_permute(
        torch.zeros((tokens, 128), device="cuda", dtype=torch.float8_e4m3fn),
        topk_ids,
        experts,
    )
    self.assertFalse(torch.equal(src2dst, torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)))
    gate_up = torch.randn(
        (tokens * top_k, hidden * 2), device="cuda", dtype=torch.bfloat16
    )
    ref_q = torch.empty(
        (tokens * top_k, hidden), device="cuda", dtype=torch.float8_e4m3fn
    )
    ref_scale = torch.empty(
        (tokens * top_k, hidden // 128), device="cuda", dtype=torch.float32
    )
    silu_and_mul_contig_post_quant(gate_up, ref_q, ref_scale, 128)
    expected_scale = pack_flashinfer_sm120_fp8_scale(
        ref_scale, topk_ids, src2dst, m_indptr, source_is_packed=True
    )

    actual_q, actual_scale = fused_swiglu_quant_pack_flashinfer_sm120_fp8(
        gate_up, topk_ids, src2dst, m_indptr
    )
    torch.testing.assert_close(actual_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-6, atol=0)
```

- [ ] **Step 2: 写动态 padding/output reuse 的 RED 测试**

新增 `test_fused_swiglu_quant_pack_reused_outputs_clear_padding`。复用低编号 expert 热、
高编号 expert 热两组 route；预分配同一 `out/out_scale`，每一轮都从现有 contig kernel
和 pack op 重新生成 reference：

```python
from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
    fused_swiglu_quant_pack_flashinfer_sm120_fp8,
    pack_flashinfer_sm120_fp8_scale,
)

experts, tokens, top_k, hidden = 8, 4, 2, 512
routes = (
    torch.tensor([[0, 0], [0, 1], [1, 1], [1, 1]], device="cuda", dtype=torch.int32),
    torch.tensor([[7, 7], [6, 7], [5, 6], [4, 7]], device="cuda", dtype=torch.int32),
)
actual_q = torch.empty(
    (tokens * top_k, hidden), device="cuda", dtype=torch.float8_e4m3fn
)
actual_scale = torch.empty(
    (hidden // 128, ((tokens * top_k + 3 * experts) // 4) * 4),
    device="cuda",
    dtype=torch.float32,
)
q_ptr, scale_ptr = actual_q.data_ptr(), actual_scale.data_ptr()

for topk_ids in routes:
    _, src2dst, m_indptr = moe_permute(
        torch.zeros((tokens, 128), device="cuda", dtype=torch.float8_e4m3fn),
        topk_ids,
        experts,
    )
    gate_up = torch.randn(
        (tokens * top_k, hidden * 2), device="cuda", dtype=torch.bfloat16
    )
    ref_q = torch.empty_like(actual_q)
    ref_scale = torch.empty(
        (tokens * top_k, hidden // 128), device="cuda", dtype=torch.float32
    )
    silu_and_mul_contig_post_quant(gate_up, ref_q, ref_scale, 128)
    expected_scale = pack_flashinfer_sm120_fp8_scale(
        ref_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=True,
    )
    returned_q, returned_scale = fused_swiglu_quant_pack_flashinfer_sm120_fp8(
        gate_up,
        topk_ids,
        src2dst,
        m_indptr,
        out=actual_q,
        out_scale=actual_scale,
    )
    self.assertEqual(returned_q.data_ptr(), q_ptr)
    self.assertEqual(returned_scale.data_ptr(), scale_ptr)
    torch.testing.assert_close(
        returned_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(returned_scale, expected_scale, rtol=1e-6, atol=0)
    self.assertTrue(torch.equal(returned_scale[expected_scale == 0], torch.zeros_like(returned_scale[expected_scale == 0])))
```

第二轮会覆盖第一轮不同的 padding 分布，因此可证明 replay/reuse 时不是依赖一次性初值。

- [ ] **Step 3: 写 top-k/空 expert/重复 expert 的 adapter RED 测试**

新增 `test_fused_swiglu_quant_pack_route_profiles`，对以下 `topk_ids` 逐个执行与 Step 1
相同的 contig payload + packed scale reference 比较：

```python
profiles = (
    torch.tensor([[0], [0], [7], [3]], device="cuda", dtype=torch.int32),
    torch.tensor(
        [[3, 3], [0, 7], [3, 0], [7, 7]],
        device="cuda",
        dtype=torch.int32,
    ),
    torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 7, 6, 5, 4, 3, 2, 0],
        ],
        device="cuda",
        dtype=torch.int32,
    ),
)
```

每个 subtest 使用 `experts=8`，因此同时覆盖 top-k 1/2/8、空 expert、skew、跨 token
重复 expert 和同一 token 内重复 expert。除数值比较外，断言 output 为 contiguous
E4M3、scale 为 contiguous FP32、shape 为 `[N/128,m_padded]` 且地址 16-byte 对齐。

- [ ] **Step 4: 提交并推送 RED 契约**

```bash
git add test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "test: specify SM120 FP8 fused A2 packing"
git push origin feat/flashinfer-sm120-fp8-moe
```

该提交在 SM120 上预期失败；它用于让无 SSH 的服务器取得 RED 测试，不代表功能完成。

- [ ] **Step 5: 服务器运行目标测试确认 RED**

服务器 fetch/switch 到该提交后执行：

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'fused_swiglu_quant_pack' -q -s
```

Expected: FAIL，原因是无法导入
`fused_swiglu_quant_pack_flashinfer_sm120_fp8`。若测试因其他错误失败，先修测试直到只剩
缺少 production symbol。

---

### Task 4: 实现 CUDA JIT A2 adapter 并使专项测试 GREEN

**Files:**
- Create: `python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_swiglu_quant.cuh`
- Create: `python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py`
- Modify: `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`
- Test: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: BF16 `gate_up [M,2N]`、int32 `topk_ids/src2dst/m_indptr`。
- Produces: `fused_swiglu_quant_pack_flashinfer_sm120_fp8(...) -> (down_input FP8 [M,N], a2_scale FP32 [N/128,m_padded])`。

- [ ] **Step 1: 新增 Python JIT module loader**

创建 `python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py`：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_flashinfer_sm120_fp8_moe_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "flashinfer_sm120_fp8_moe",
        *args,
        cuda_files=["moe/flashinfer_sm120_fp8_swiglu_quant.cuh"],
        cuda_wrappers=[
            (
                "silu_quant_pack",
                f"FlashInferSm120Fp8SiluQuantPackKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


def flashinfer_sm120_fp8_silu_quant_pack(
    gate_up: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
) -> None:
    module = _jit_flashinfer_sm120_fp8_moe_module(is_arch_support_pdl())
    module.silu_quant_pack(
        gate_up, output, output_scale, topk_ids, src2dst, m_indptr
    )
```

C++ header 的 include/namespace 基线直接对齐
`python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh`：使用
`sgl_kernel/tensor.h`、`utils.h`、`math.cuh`、`type.cuh`、`utils.cuh`、`vec.cuh`、
`warp.cuh` 和 `deepseek_v4/fp8_utils.cuh`，并复用其中的 `silu_and_mul` 数学实现与
`deepseek_v4::fp8::pack_fp8`。只复制所需 helper，不 include 另一个 `.cuh`，避免匿名
namespace 和额外 kernel wrapper 一并进入编译单元。

- [ ] **Step 2: 实现 CUDA 参数和 routed-row 分支**

C++ 参数必须包含：

```cpp
struct FlashInferSm120Fp8SiluQuantPackParams {
  const bf16_t* input;
  fp8_e4m3_t* output;
  float* output_scale;
  const int32_t* topk_ids;
  const int32_t* src2dst;
  const int32_t* m_indptr;
  int64_t hidden_dim;
  int64_t m_padded;
  uint32_t num_routes;
  uint32_t num_experts;
};
```

`blockIdx.x < num_routes` 分支必须先做：

```cpp
const uint32_t route = blockIdx.x;
const uint32_t dst = params.src2dst[route];
const uint32_t expert = params.topk_ids[route];
const uint32_t expert_start = params.m_indptr[expert];
const uint32_t aligned_start = ((expert_start + 3u * expert) / 4u) * 4u;
const uint32_t scale_col = aligned_start + dst - expert_start;
const auto input = params.input + int64_t(dst) * params.hidden_dim * 2;
const auto output = params.output + int64_t(dst) * params.hidden_dim;
```

之后逐项移植现有 contig kernel 的 `AlignedVector` load、`silu_and_mul`、16-thread group
`warp::reduce_max`、`absmax / FP8_E4M3_MAX`、`pack_fp8` 和 vector store；唯一 scale
store 为：

```cpp
params.output_scale[int64_t(work_id) * params.m_padded + scale_col] = scale;
```

不能照抄原 kernel 以 `blockDim.x` 作为 gate/up 半区跨度的写法：`num_groups` 为奇数时，
32-thread 对齐会使 `blockDim.x > hidden_dim / 8`。有效 group 内固定使用：

```cpp
const uint32_t lane_in_work = threadIdx.x % kWorkThreads;
const uint32_t vector_id = work_id * kWorkThreads + lane_in_work;
const uint32_t vectors_per_half = params.hidden_dim / 8u;
gate_vec.load(input, vector_id);
up_vec.load(input, vector_id + vectors_per_half);
out_vec.store(output, vector_id);
```

这样 `N=128、384` 等奇数 group 数也不会越界；测试使用的 `N=512` 仍走同一寻址。

每个 CTA 进入 route/padding 分支前先调用一次 `PDLWaitPrimary<kUsePDL>()`。route block
先计算 `num_groups = params.hidden_dim / 128u`，之后只有
`work_id < num_groups` 的 16-thread group 执行完整 load、SwiGLU、reduction、quant、
payload store 和上述 FP32 scale store。`PDLTriggerSecondary<kUsePDL>()` 放在计算完成、
两个 store 之前，并由整个 CTA 共同执行；padding block 也在 zero store 前调用一次
trigger。`num_threads` 向 32 对齐后产生的多余 16-thread group 不得访问任何 tensor。
不得实例化或放开原有 `kTransposed` 路径。

- [ ] **Step 3: 在同一 CUDA launch 实现 expert padding 分支**

`blockIdx.x >= num_routes` 时：

```cpp
const uint32_t expert = blockIdx.x - params.num_routes;
const uint32_t start = params.m_indptr[expert];
const uint32_t end = params.m_indptr[expert + 1];
const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
const uint32_t valid_end = aligned + end - start;
const uint32_t next = expert + 1 == params.num_experts
    ? params.m_padded
    : ((end + 3u * (expert + 1)) / 4u) * 4u;
const uint32_t gap = next - valid_end;
const uint32_t num_groups = params.hidden_dim / 128u;
if (gap != 0) {
  for (uint32_t i = threadIdx.x; i < num_groups * gap; i += blockDim.x) {
    const uint32_t group = i / gap;
    const uint32_t column = valid_end + i % gap;
    params.output_scale[int64_t(group) * params.m_padded + column] = 0.0f;
  }
}
```

`gap == 0` 时不得生成除零表达式。route block 和 padding block 的 store 区间不得重叠。

- [ ] **Step 4: 实现 C++ host wrapper 检查和 launch**

`FlashInferSm120Fp8SiluQuantPackKernel<kUsePDL>::run` 使用 `TensorMatcher` 验证：

```text
gate_up      [M,2N] BF16 contiguous
output       [M,N]  FP8 E4M3 contiguous
output_scale [N/128,m_padded] FP32 contiguous
topk_ids     [T,top_k] int32 contiguous，numel=M
src2dst      [M] int32 contiguous
m_indptr     [E+1] int32 contiguous
```

host 只从 tensor metadata 计算：

```cpp
num_groups = N / 128;
num_experts = m_indptr.size(0) - 1;
expected_m_padded = ((M + 3 * num_experts) / 4) * 4;
num_threads = ((num_groups * 16 + 31) / 32) * 32;
grid = M + num_experts;
```

用 `expected_m_padded` 绑定 `output_scale` 的第二维并写入 params；同时验证
`topk_ids.numel() == M`、`src2dst.numel() == M`。要求 `N % 128 == 0`、
`num_threads <= 1024`。不能读取 `m_indptr` 内容。launch 使用
`LaunchKernel(grid, num_threads, device).enable_pdl(kUsePDL)`。

- [ ] **Step 5: 新增经过验证的 Python adapter**

在现有 MoE ops 文件中新增：

```python
def fused_swiglu_quant_pack_flashinfer_sm120_fp8(
    gate_up: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

检查 dtype、rank、contiguous、device、`gate_up.shape[1] % 2 == 0`、
`gate_up.shape[0] == topk_ids.numel()`、`topk_ids.shape[1] > 0`、
`src2dst.numel() == topk_ids.numel()`、`m_indptr.numel() >= 2`、`N % 128 == 0`。
输出固定为：

```python
expected_out_shape = (topk_ids.numel(), gate_up.shape[1] // 2)
expected_scale_shape = (
    expected_out_shape[1] // 128,
    flashinfer_sm120_m_padded(topk_ids.numel(), m_indptr.numel() - 1),
)
```

`out` 为 E4M3 FP8，`out_scale` 为 FP32 且 16-byte aligned。分配使用
`torch.empty`，不得调用 `zero_()`。最后只调用一次低层 JIT wrapper并返回两个 output。

- [ ] **Step 6: 创建并推送 adapter 候选提交**

本地先运行：

```bash
python3 -m compileall -q \
  python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py
git diff --check
```

然后提交并推送服务器可取得的候选实现：

```bash
git add \
  python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_swiglu_quant.cuh \
  python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py
git commit -m "feat: fuse SM120 FP8 SwiGLU quant packing"
git push origin feat/flashinfer-sm120-fp8-moe
```

该 commit 只有在后续服务器 GREEN 后才视为已验证。

- [ ] **Step 7: 服务器运行专项测试确认 GREEN**

服务器 fetch/switch 到候选提交后：

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'fused_swiglu_quant_pack' -q -s
```

Expected: 三个专项测试 PASS；首次运行允许出现一次 JIT 编译。

- [ ] **Step 8: 运行 packing 回归**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'Packing and not full_runner and not cuda_graph' -q -s
```

Expected: 现有 scale pack 与新 adapter 测试全部 PASS。

- [ ] **Step 9: 若服务器失败，逐个修复并重新验证**

每次只针对当前编译或断言失败做最小修复，提交 `fix:` commit，push、服务器
fetch/switch，再重复 Steps 7-8。不得因实现差异放宽 payload/scale/padding 契约。

---

### Task 5: 接入 production runner 并验证完整正确性/CUDA Graph

**Files:**
- Modify: `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`
- Modify: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`
- Read: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`

**Interfaces:**
- Consumes: Task 4 的 `fused_swiglu_quant_pack_flashinfer_sm120_fp8`。
- Produces: production runner 的单-launch GEMM2 input prepare；benchmark 自动识别 fused call-trace。

- [ ] **Step 1: 写 production runner 调用次数的 RED 测试**

新增 `test_full_runner_uses_single_fused_a2_prepare`，沿用 `_make_runner_case` 生成
`tokens=8/top_k=2` 的合法完整输入，并 patch runner module 实际绑定的四个 symbol：

```python
from sglang.srt.layers.moe.moe_runner import (
    flashinfer_sm120_fp8 as flashinfer_runner,
)
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
    fused_swiglu_quant_pack_flashinfer_sm120_fp8 as fused_adapter,
)

legacy_silu = getattr(flashinfer_runner, "silu_and_mul", None)

topk_ids = (
    torch.arange(16, device="cuda", dtype=torch.int32)
    .remainder(16)
    .view(8, 2)
)
dispatch, config, quant_info, _, _ = _make_runner_case(8, 2, topk_ids)
with patch.object(
    flashinfer_runner,
    "fused_swiglu_quant_pack_flashinfer_sm120_fp8",
    wraps=fused_adapter,
    create=True,
) as fused, patch.object(
    flashinfer_runner,
    "sglang_per_token_group_quant_fp8",
    wraps=flashinfer_runner.sglang_per_token_group_quant_fp8,
) as quant, patch.object(
    flashinfer_runner,
    "pack_flashinfer_sm120_fp8_scale",
    wraps=flashinfer_runner.pack_flashinfer_sm120_fp8_scale,
) as pack_scale, patch.object(
    flashinfer_runner,
    "silu_and_mul",
    wraps=legacy_silu,
    create=True,
) as silu:
    flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
        dispatch, quant_info, config
    )

self.assertEqual(fused.call_count, 1)
self.assertEqual(quant.call_count, 1)
self.assertEqual(pack_scale.call_count, 1)
self.assertEqual(silu.call_count, 0)
```

因为 RED 时 runner module 还没有新 symbol，测试先从 public ops module 取新 adapter，
再用 `patch.object(..., create=True)` 注入 runner module；这能让当前 runner 正常运行并以
`fused.call_count == 0`、旧 quant/pack 各 2 次的预期原因失败。GREEN 后同一 patch 会覆盖
runner 已导入的 symbol，无需分叉测试代码。旧 `silu_and_mul` 必须从 runner 当前绑定
取得：RED 时 `wraps` 的就是实际生产调用，GREEN 删除 import 后则为 `None`，此时
`patch.object(..., wraps=None, create=True)` 只注入不会被正确路径调用的 mock。目标断言：

```text
new fused adapter: 1 次
generic quant:      1 次（仅 GEMM1）
generic pack:       1 次（仅 GEMM1）
old silu_and_mul:   0 次
```

Expected RED: runner 完整执行成功，然后仅在调用次数断言处失败：当前 runner 不调用新
adapter，通用 quant/pack 各调用两次、旧 `silu_and_mul` 一次。若提前出现 `TypeError`、
数值错误或其他异常，视为 RED 测试本身错误，先修测试，不能进入 production 接线。

- [ ] **Step 2: 替换 runner 的四步 A2 路径**

导入：

```python
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
    fused_swiglu_quant_pack_flashinfer_sm120_fp8,
    pack_flashinfer_sm120_fp8_scale,
)
```

删除 `silu_and_mul` import，并把当前 lines 278-297 替换为：

```python
down_input, a2_scale_fi = fused_swiglu_quant_pack_flashinfer_sm120_fp8(
    gate_up,
    topk_ids,
    src2dst,
    m_indptr,
)
```

保留 GEMM1 quant/pack 和两个 `_run_grouped_gemm` 不变。

- [ ] **Step 3: 创建并推送 runner 候选提交**

```bash
git add \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "feat: use fused A2 packing in SM120 FP8 MoE"
git push origin feat/flashinfer-sm120-fp8-moe
```

服务器 fetch/switch 到该提交；只有 Steps 4-6 GREEN 后才视为接线完成。

- [ ] **Step 4: 服务器运行完整 runner correctness**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py::TestFlashInferSm120Fp8Packing::test_full_runner_correctness \
  -q -s
```

Expected: PASS，三个数值指标均低于既有阈值。

- [ ] **Step 5: 服务器运行 CUDA Graph 动态路由测试**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py::TestFlashInferSm120Fp8Packing::test_cuda_graph_replays_new_hidden_and_routing \
  -q -s
```

Expected: PASS，output pointer 不变，20 次 replay 后 allocated memory 不增长。

- [ ] **Step 6: 运行完整 GPU/CPU 回归**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest test/registered/moe/test_flashinfer_sm120_fp8_moe.py -q -s
"${VENV_PY}" -m pytest test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
"${VENV_PY}" -m pytest test/registered/unit/test_pro5000_stage_2.py -q
```

Expected: 全部 PASS。pytest 的 `asyncio_mode` 和 torch.jit deprecation warning 允许存在。

- [ ] **Step 7: 若回归失败，最小修复并重跑全部三组测试**

修复使用独立 `fix:` commit；不改变既定数值阈值，不恢复旧四步 fallback。

---

### Task 6: 服务器检查点 B/C——融合 preflight、正式矩阵与结果归档

**Files:**
- Read: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
- Read: `scripts/pro5000/run_stage_2_benchmark.sh`
- Produce on server: 新 `stage-2-*` artifact。

**Interfaces:**
- Consumes: Tasks 4-5 的融合 production runner、Task 2 baseline artifact。
- Produces: 同口径 before/after 比较和最终 `GO`/`FUNCTIONAL_ONLY`/`NO_GO`。

- [ ] **Step 1: 推送融合提交并同步服务器**

```bash
git push origin feat/flashinfer-sm120-fp8-moe
```

服务器：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git status --short --branch
```

- [ ] **Step 2: 运行融合 preflight**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  --tokens 1 8 8192 \
  --top-k 8 \
  --profiles uniform synthetic-skew \
  --warmup 2 \
  --trials 2 \
  --iterations 20 \
  --check-cuda-graph \
  --output-json /tmp/pro5000-a2-fused-preflight.json
```

Expected: exit 0，correctness/graph PASS；组件 `path` 为 `fused`，不存在 legacy 的
`silu/quant2/scale_pack_gemm2` detail。

- [ ] **Step 3: 与检查点 A preflight 做同口径比较**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" - /tmp/pro5000-a2-legacy-preflight.json /tmp/pro5000-a2-fused-preflight.json <<'PY'
import json, sys
before, after = (json.load(open(path)) for path in sys.argv[1:])
def index(payload):
    return {(c["tokens"], c["profile"]): c for c in payload["cases"]}
b, a = index(before), index(after)
for key in sorted(a):
    old, new = b[key], a[key]
    eager = old["flashinfer_sm120_fp8"]["median_ms"] / new["flashinfer_sm120_fp8"]["median_ms"] - 1
    print(key, "FI eager improvement", f"{eager:.2%}")
    if new["cuda_graph"].get("status") != "NOT_RUN":
        graph = old["cuda_graph"]["flashinfer_sm120_fp8"]["median_ms"] / new["cuda_graph"]["flashinfer_sm120_fp8"]["median_ms"] - 1
        print(key, "FI graph improvement", f"{graph:.2%}")
PY
```

- [ ] **Step 4: 运行正式 Stage 2 矩阵**

```bash
bash scripts/pro5000/run_stage_2_benchmark.sh
```

Expected: wrapper/process exit 0，correctness/graph PASS。决策含义：

```text
GO              两个性能门槛均通过
FUNCTIONAL_ONLY 正确性通过，但 prefill 或 decode 未过线
NO_GO           正确性或 CUDA Graph 失败，停止性能结论
```

- [ ] **Step 5: 归档并传回结果**

```bash
RUN_DIR="$(ls -1dt /home/logs/sennian/pro5000-fi-moe/runs/stage-2-* | head -n1)"
ARCHIVE="/tmp/$(basename "${RUN_DIR}").tar.gz"
tar -C "$(dirname "${RUN_DIR}")" -czf "${ARCHIVE}" "$(basename "${RUN_DIR}")"
echo "${ARCHIVE}"
```

将 archive 传回本地，保存到：

```text
/Users/bsy/Desktop/workspace/Pro5000-Optimize/ServerDownload/
```

- [ ] **Step 6: 更新结果文档并提交**

把 baseline/fused commit、eager/graph 主 case、组件 rollup 和最终 decision 写入当前
中文设计文档的“实验结果”附录。只陈述实测，不改变阈值。

```bash
git add docs/superpowers/specs/2026-07-21-flashinfer-sm120-fp8-a2-fusion-design.md
git commit -m "docs: record SM120 FP8 A2 fusion results"
```

---

### Task 7: 最终验证与分支交付

**Files:**
- Verify: 所有本计划修改文件
- Preserve: `test/registered/moe/debug_flashinfer_sm120_fp8_stagewise.py`

**Interfaces:**
- Consumes: Task 6 正式 artifact。
- Produces: 可同步到本地/服务器的已验证分支状态。

- [ ] **Step 1: 本地静态与 CPU 验证**

```bash
python3 -m unittest test.registered.unit.test_pro5000_stage_2
python3 -m compileall -q \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py
git diff --check
```

- [ ] **Step 2: 核对提交范围和用户文件**

```bash
git status --short --branch
git log --oneline --decorate -8
git diff origin/feat/flashinfer-sm120-fp8-moe...HEAD --stat
```

Expected: 调试脚本仍为未跟踪且未提交；无意外依赖、wheel 或 artifact 进入 git。

- [ ] **Step 3: 使用 verification-before-completion 核对服务器证据**

逐项确认最新 artifact 中：

```text
status == completed
all correctness == PASS
cuda_graph.status == PASS
decode cases contain graph trials
tokens=8192 uniform contains eager trials
component path == fused
git.commit == server detached HEAD
```

- [ ] **Step 4: 推送最终文档并报告结果**

```bash
git push origin feat/flashinfer-sm120-fp8-moe
```

最终报告必须分别给出：prefill eager、decode graph、正确性、CUDA Graph、组件收益和
最终 decision；不得用单一 speedup 概括所有 workload。
