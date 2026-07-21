# RTX PRO 5000：FlashInfer SM120 FP8 MoE A2 融合与生产口径基准设计

**状态：** 对话设计已批准，待书面规范审阅

**日期：** 2026-07-21

**目标分支：** `feat/flashinfer-sm120-fp8-moe`

## 1. 背景

SGLang 已在单卡 NVIDIA RTX PRO 5000 72GB Blackwell（SM120）上接入
FlashInfer：

```python
flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise
```

当前 backend 为 `flashinfer_sm120_fp8`。GEMM1 和 GEMM2 均消费 E4M3 FP8
activation/weight、FP32 scale，并输出 BF16。Stage 2 已验证完整 runner 正确性和
CUDA Graph 动态路由 replay，但正式性能结论为 `FUNCTIONAL_ONLY`：

- tokens=8192、uniform：Triton `2.06355 ms`，FlashInfer `1.93307 ms`，加速
  `6.75%`；
- tokens=8192、synthetic-skew：FlashInfer 相对 Triton 回退 `3.33%`；
- tokens=8、uniform eager：FlashInfer 相对 Triton 回退 `21.05%`。

tokens=8192、uniform 的 FlashInfer 组件测量中：

```text
routing_quant_pack   0.24594 ms
gemm1                0.77712 ms
swiglu_quant         0.16588 ms
scale_layout_gemm2   0.01289 ms
gemm2                0.50540 ms
unpermute_combine    0.23274 ms
```

当前 GEMM2 输入准备依次发射：

```text
silu_and_mul
  -> per-token/per-128-group FP8 quant
  -> a2_scale_fi.zero_()
  -> A2 scale pack
```

这四步重复读写中间张量并产生四个 kernel launch。本轮在不修改 FlashInfer 源码的
前提下，将其替换为一个 SGLang CUDA JIT adapter。同时修正 Stage 2 的性能口径：
decode 使用生产中的 CUDA Graph replay 延迟，prefill 使用 eager 延迟。

## 2. 目标与非目标

### 2.1 目标

1. 增加 decode CUDA Graph replay 的 Triton/FlashInfer 对等计时。
2. 保留所有 case 的 eager 计时，便于诊断和历史数据比较。
3. 以 graph replay 判定 tokens=1、8，以 eager 判定 prefill。
4. 把 `SwiGLU + quant#2 + A2 scale zero/pack` 融合为一个 CUDA kernel。
5. 直接产生 FlashInfer 要求的 FP32、MN-major、4-row-padded A2 scale。
6. 保持 E4M3 FP8、每 128 元素一组的现有量化数学语义。
7. 支持空 expert、不同 token 重复选择同一 expert，并且不依赖同一 token 内的
   expert ID 唯一性。
8. 支持 CUDA Graph 中 hidden state 和路由在 replay 之间变化。
9. 用独立提交和服务器基准把“测量口径变化”与“融合优化收益”分开归因。

### 2.2 非目标

本轮不做以下工作：

- 不修改 FlashInfer kernel、scheduler、tile 选择或 L2 swizzle；
- 不修改 GEMM1、GEMM2、路由、SGLang combine 或 router weight 语义；
- 不优化 GEMM1 的 A-scale 全量 `zero_()`；
- 不引入 runtime backend fallback 或按 batch 大小切换 backend；
- 不实现 workspace 池或缓存 `_validate_contract`；
- 不支持 DeepEP、多卡 EP、TP 大于 1 或新的量化格式；
- 不调整现有完整 runner 正确性阈值；
- 不把 CUDA Event 组件分解结果作为 GO/NO-GO 判定依据；
- 不硬编码模型名或 Qwen 专用 `N=512`。

## 3. 方案选择

### 3.1 采用：独立 CUDA JIT 三合一 adapter

新增专用 CUDA JIT kernel 变体，借用 SGLang 现有
`silu_mul_quant_contig_kernel` 已验证的计算和向量化基础逻辑：

- BF16 gate/up 向量化加载；
- SwiGLU 计算；
- 128 元素 group 的 absmax reduction；
- E4M3 FP8 scale 和 payload 生成；
- FP8 向量化写回。

现有 `kTransposed` 分支不能直接复用：它由 static assert 限制为 UE8M0，并按
int32/byte packing 写出 4 个 exponent，不支持普通 FP32 scale。新变体不得尝试放开
该分支，而是独立实现以下三项 FlashInfer 专用逻辑：

1. 对源 routed slot `r` 执行 `dst = src2dst[r]`，从 `gate_up[dst]` 读取并写回
   `down_input[dst]`；
2. 使用 `topk_ids/m_indptr/dst` 计算 per-expert 4-row-aligned 列，直接写出
   `[G, m_padded]` FP32 scale；
3. 在同一 launch 中重写动态 padding 为零。

真正复用的是 SwiGLU、128-group reduction、E4M3 conversion、向量化 load/store 和
一行一个 block 的调度骨架；scale 写出和 routed-row 寻址属于新实现。

该实现位于 SGLang，不改变或 fork FlashInfer 源码。

### 3.2 未采用：新 Triton 三合一 kernel

Triton 实现更容易迭代，但必须重新实现 SwiGLU、分组 reduction、E4M3 conversion
和向量化策略。当前目标首先是减少 glue 开销而不是重新调优量化 kernel，因此复用
现有 CUDA JIT 路径的风险更低，也更有机会保留其 SM120 性能。

### 3.3 未采用：现有二合一 kernel 后继续 pack

直接使用 `silu_and_mul_contig_post_quant` 可以消除 BF16 中间张量，但仍需
`zero_ + scale pack`，总计三个 launch，并保留一次完整 scale 读写。该方案只能
作为诊断参照，不作为生产实现。

## 4. 组件与文件边界

### 4.1 CUDA JIT 实现

新增专用 header 和对应的 JIT module loader：

```text
python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_swiglu_quant.cuh
python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py
```

它只负责：

- 读取 packed GEMM1 输出；
- 执行 SwiGLU 和 FP8 group quant；
- 写出 packed FP8 GEMM2 input；
- 写出 FlashInfer A2 scale 和 padding。

它不调用 FlashInfer GEMM，不排序路由，也不 combine 输出。Python JIT module
loader 只负责 `load_jit` 缓存和低层 launch，不承担上层数据契约判断。

### 4.2 Python op

扩展现有文件：

```text
python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py
```

新增一个具名 adapter，负责输入输出 shape/dtype/stride 检查、输出分配，并调用
JIT module loader 发射单次 kernel。JIT 必须在 CUDA Graph capture 前通过正常
warmup 完成。

### 4.3 Runner

修改：

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py
```

只把：

```text
silu_and_mul -> quant#2 -> zero_ -> pack#2
```

替换为一次新 adapter 调用。GEMM1/GEMM2 的 `_run_grouped_gemm`、权重、B-scale、
`m_indptr` 和 `out=` 契约保持不变。

### 4.4 Benchmark 与测试

修改：

```text
scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py
test/registered/unit/test_pro5000_stage_2.py
test/registered/moe/test_flashinfer_sm120_fp8_moe.py
```

benchmark 负责生产口径计时和结构化输出；GPU 测试负责 adapter、完整 runner 和
CUDA Graph 正确性。

## 5. Adapter 数据契约

定义：

```text
T      token 数
top_k 每个 token 的 routed expert 数
M      T * top_k
E      本地 expert 数
N      GEMM1 gate/up 各自的宽度，即 GEMM2 输入宽度
G      N / 128
```

### 5.1 输入

```text
gate_up  [M, 2N]     BF16，contiguous，按 expert packed
topk_ids [T, top_k]  int32，contiguous
src2dst  [M]         int32，contiguous
m_indptr [E+1]       int32，contiguous
```

`src2dst[r]` 是源 routed slot `r` 在 packed row 中的位置。重复 expert ID 合法；
每个 routed slot 仍对应唯一 `dst`。

### 5.2 输出

```text
down_input [M, N]          torch.float8_e4m3fn，contiguous
a2_scale   [G, m_padded]   float32，contiguous，16-byte aligned

m_padded = ((M + 3 * E) // 4) * 4
```

`down_input` 按 packed row 排列，可直接作为 FlashInfer GEMM2 的 `a`。
`a2_scale` 可直接作为 FlashInfer GEMM2 的 `a_scale`。

### 5.3 有效 scale 列映射

对源 routed slot `r`：

```text
expert       = topk_ids.flatten()[r]
dst          = src2dst[r]
expert_start = m_indptr[expert]
aligned      = ((expert_start + 3 * expert) // 4) * 4
scale_col    = aligned + dst - expert_start
```

kernel 从 `gate_up[dst]` 读取数据，并把量化 scale 写到
`a2_scale[group, scale_col]`。不在 device 或 host 上搜索 `m_indptr`。

### 5.4 数学语义

对每行、每个连续 128 元素 group：

```text
value     = SiLU(gate) * up
absmax    = max(max(abs(value)), 1e-10)
scale     = absmax / 448.0
quantized = E4M3(clamp(value / scale, -448.0, 448.0))
```

输出 scale 为普通 FP32，不是 UE8M0，不使用 MXFP8 scale packing。

## 6. Kernel 调度与 padding

### 6.1 有效 routed row

沿用现有 CUDA JIT kernel 的“一行一个 block”模式。一个 block 处理某个源 routed
slot 对应的完整 packed row，内部以 128 元素为 quant group 完成 reduction 和写回。
该模式避免为 `N/128` 个 group 分别发射独立 program。

### 6.2 Padding block

同一 grid 额外包含 E 个 expert block。expert `e` 的有效 scale 结束列和下一个 expert
起始列为：

```text
valid_end = aligned_start(e) + m_indptr[e + 1] - m_indptr[e]
next      = aligned_start(e + 1)
```

其中最后一个 expert 的 `next` 使用 `m_padded`。padding block 把：

```text
a2_scale[:, valid_end:next]
```

全部写零。该区间允许为空；空 expert 和连续空 expert 也必须正确处理。

padding 每次 forward/replay 都重写，不能依赖 capture 前一次初始化。这保证路由改变
后不会把旧 scale 暴露给后续调用，也删除了独立的全量 `a2_scale.zero_()`。

### 6.3 CUDA Graph 要求

热路径不得出现：

- `.item()`、`.cpu()` 或 GPU 到 CPU 同步；
- route-dependent host 分支；
- capture 期间首次 JIT；
- 基于 `m_indptr` 数值的 host 侧 shape 或 launch 计算。

grid 只依赖静态的 M、E、N；同一 graph bucket 内路由内容可以变化。

## 7. 错误处理和兼容性

adapter 必须明确拒绝：

- 非 CUDA tensor 或跨 device tensor；
- `gate_up` 非二维 BF16 contiguous；
- `gate_up.shape[1]` 不是 `2N`，或 N 不是 128 的倍数；
- 非 E4M3 FP8 输出 buffer；
- 非 FP32、非目标 shape 或未对齐的 scale buffer；
- 非 int32/contiguous 的 `topk_ids/src2dst/m_indptr`；
- `src2dst.numel() != topk_ids.numel()`；
- 当前 CUDA JIT launch 的线程上限不支持的 N。

这些条件失败时抛出明确异常。生产 runner 不回退到旧四步路径或 Triton backend。

## 8. Benchmark 设计

### 8.1 两种计时口径

每个 case 保留 eager latency。对 tokens=1、8，另外分别 capture Triton 和
FlashInfer runner，记录 CUDA Graph replay latency。

decode graph 计时：

1. 正常 warmup，完成所有 JIT 和 lazy initialization；
2. 为每个 backend 单独 capture；
3. 计时区间只包含重复 `graph.replay()`；
4. 不包含输入 `copy_`、路由生成、graph capture 和结果 clone；
5. trial 顺序在 Triton 和 FlashInfer 之间交替；
6. 输出每个 trial 和 min/median/max。

prefill 不 capture graph；tokens≥128 的正式数据采用 eager latency。

### 8.2 组件统计

结构化输出提供稳定 rollup：

```text
gemm1_input_prepare
gemm1
gemm2_input_prepare
gemm2
unpermute_combine
```

旧路径详细项为：

```text
quant1
moe_permute
scale_pack_gemm1
silu
quant2
scale_pack_gemm2
```

新路径后三项替换为：

```text
fused_swiglu_quant_pack_gemm2
```

`gemm2_input_prepare` 在两个版本中都表示 GEMM1 输出到 GEMM2 输入就绪的总 GPU
时间。逐组件 CUDA Event 会改变小 kernel 调度，只用于瓶颈定位，不参与性能 gate。

检查点 A 的 profiler 必须同时理解尚未出现 fused symbol 的旧 runner，以及检查点 B
调用新 adapter 的 runner。实现不得继续假设 quant/pack 一定各调用两次，也不能只因
通用 quant/pack symbol 存在就判断为旧路径；新路径仍会在 GEMM1 使用这些函数。
profiler 应为所有当前可用 stage 注册 hook，根据实际调用轨迹生成且仅生成以下之一：

```text
legacy detail: quant1, moe_permute, scale_pack_gemm1, silu, quant2,
               scale_pack_gemm2
fused detail:  quant1, moe_permute, scale_pack_gemm1,
               fused_swiglu_quant_pack_gemm2
```

任何缺失、重复或两套 detail 混合都应使组件诊断显式失败，而不是输出误标数据。

### 8.3 判定口径

正确性和 CUDA Graph 均通过后：

```text
prefill_speedup =
  triton_eager(tokens=8192, uniform)
  / flashinfer_eager(tokens=8192, uniform) - 1

decode_regression = max(
  flashinfer_graph(tokens=1 or 8, any required profile)
  / triton_graph(tokens=1 or 8, same profile) - 1,
  0
)
```

正式状态：

```text
GO:
  prefill_speedup >= 10%
  decode_regression <= 5%
  correctness PASS
  CUDA Graph correctness PASS

FUNCTIONAL_ONLY:
  correctness和graph通过，但任一性能门槛未通过

NO_GO:
  correctness或CUDA Graph失败
```

## 9. 测试设计

### 9.1 CPU/unit 测试

1. 结果 schema 同时表达 eager 和可选 graph latency。
2. GO 判定只从 decode graph 字段读取 decode regression。
3. tokens=8192、uniform 的 prefill 判定继续读取 eager 字段。
4. decode graph 缺失时不能产生 `GO`。
5. 组件 detail 不同，但 rollup keys 在旧/新路径间稳定。
6. 用两个 fake runner/call-trace fixture 分别覆盖 legacy detail 和 fused detail，确保
   检查点 A 的同一份 benchmark 代码在检查点 B 无需改变统计语义。

### 9.2 GPU adapter 测试

1. 将 adapter 的 FP8 payload 和重排前 FP32 scale 与现有
   `silu_and_mul_contig_post_quant` 参照比较。
2. 把 adapter scale 从 FlashInfer 布局还原为 packed row 后比较数值。
3. 检查所有 padding 精确为零。
4. 覆盖 top-k 1、2、8，空 expert 和 skew 路由。
5. 覆盖不同 token 重复选择同一 expert，以及同一 token 内重复 expert ID 的边界
   输入。
6. 构造明确非 identity 的 `src2dst`，分别检查 input 从 `gate_up[dst]` 读取、payload
   写到 `down_input[dst]`，而不是错误使用 `blockIdx.x` 对应的源行。
7. 复用同一输出 buffer，在路由改变后再次调用，检查不存在旧 padding 残留。
8. 检查输出 dtype、shape、contiguous 和 16-byte scale 对齐。

常规 top-k router 通常为同一 token 返回不同 expert，但 adapter 不把这一点作为输入
契约。同一 token 内出现重复 expert ID 时，每个 routed slot 仍由 `src2dst` 映射到
唯一 packed row，因此不会发生 payload 或有效 scale 写冲突。

### 9.3 完整 runner 与 CUDA Graph

保留现有 Triton 完整 runner 参照及阈值：

```text
calc_diff          < 0.005
symmetric_diff     < 1e-4
normalized_rmse    < 0.01
```

CUDA Graph 测试必须在同一 capture 上 replay 至少两组不同 hidden state 和不同
路由，检查：

- replay 与同输入 eager 输出一致；
- 输出地址不变；
- replay 后 allocated memory 不增长；
- padding 随新路由正确更新。

## 10. 分阶段实施与服务器检查点

### 10.1 检查点 A：只升级 benchmark

第一个提交只修改 benchmark 和 CPU 测试。服务器在旧 runner 上运行：

- tokens=1、8 的 eager + graph baseline；
- tokens=8192、uniform eager baseline；
- CUDA Graph 动态路由正确性。

该提交同时包含 legacy/fused 两套组件 call-trace 的 CPU 契约测试和可选 fused hook；
旧 runner 上 fused hook 未被调用属于正常情况。检查点 B 不得通过修改组件定义来制造
不可比较的结果。

结果留档后才进入融合提交。这样 decode 的 graph 性能变化不会与 benchmark 口径
变化混在一起。

### 10.2 检查点 B：实现融合 adapter

第二个提交按 TDD 增加 adapter、runner 接线和 GPU 测试。服务器先运行 adapter 和
完整 runner 正确性，再运行与检查点 A 完全相同的性能命令。

### 10.3 检查点 C：正式矩阵

正确性和 preflight 通过后，运行 Stage 2 正式 tokens/profile 矩阵并归档 JSON、
stdout、stderr 和环境信息。若融合仍未达到 GO 门槛，保留测量结果，后续单独评估：

1. GEMM1 padding-only zero；
2. 静态契约缓存；
3. FlashInfer tile/scheduler 和 L2 swizzle。

不得通过放宽阈值或静默切换 backend 把 `FUNCTIONAL_ONLY` 改写为 `GO`。

## 11. 风险与控制

### 11.1 Prefill 性能预算

tokens=8192、top-k=8 时，`M=65536, N=512` 的 BF16 中间张量大小为 64 MiB；写入
一次再由 quant 读取一次产生约 128 MiB（134 MB）流量。按 1.3 TB/s 粗略估计，完全
消除这部分 HBM 流量的上界收益约为 `0.10 ms`：

```text
1.933 - 0.10 = 1.833 ms
2.0636 / 1.833 - 1 = 约 12.6%
```

但该估算不是验收预言：中间张量可能部分命中 L2，新 kernel 的 gather/scatter 和
padding 写出也会消耗指令与带宽。达到 10% 所需的 FlashInfer 延迟为：

```text
2.06355 / 1.10 = 1.87595 ms
```

即相对当前 `1.93307 ms` 至少节省约 `0.0571 ms`。本轮具备越过 prefill 门槛的
可能，但默认预期应是测量后在 `GO` 与 `FUNCTIONAL_ONLY` 之间作出判断；若结果为
9.x%，按既定纪律进入后续 GEMM1 优化，不放宽阈值。

现有 `routing_quant_pack=0.24594 ms` 是 quant1、路由排序/permute、A1 zero/pack 的
合计，不能在细分测量前把全部时间归因于 A1。

### 11.2 数值差异

CUDA JIT 的 fast-math 与现有独立 activation 路径可能产生细微差异。控制方式是先
比较 adapter 与现有 fused quant 参照，再验证完整 runner 三个指标；不要求不同
实现的 BF16 中间值逐 bit 相同。

### 11.3 小 decode 的 kernel 下限

融合可以减少 glue launch，但不能消除 FlashInfer GEMM 本身在小 M 下的固定开销。
即使 prefill 达到 10%，decode graph 仍可能超过 5% 门槛。该结果应如实保留为
`FUNCTIONAL_ONLY`，而不是在本轮增加未设计的双路径。

当前 tokens=8、uniform 的组件诊断中，FlashInfer GEMM1/GEMM2 合计约
`0.1305 + 0.0704 = 0.2009 ms`，已经接近 Triton 完整 eager runner 的约
`0.197 ms`。组件 CUDA Event 会扰动小 kernel，因此这不是最终 graph 结论，但足以
说明 decode 过线不应作为默认预期。CUDA Graph 能减少 CPU 提交间隙，却不会把多个
GPU kernel 合并，也不会消除 FlashInfer GEMM 的 device 侧执行时间。

现有数据不能证明 `0.1305 ms` 主要由“扫描全部 256 个 expert”导致：tokens=8 的
synthetic-skew 有更多空 expert，却把 GEMM1 降至约 `0.0836 ms`。当前对照更符合
active expert/tile 数和权重工作集共同变化；scheduler 扫描、L2 和 tile 贡献需要
独立 profiler 或 A/B patch 才能定因。

### 11.4 JIT 和部署

新 adapter 增加一个 SGLang CUDA JIT module。服务器已有 CUDA 13.0 NVCC，运行
环境允许 JIT。部署启动或 graph capture 前必须完成 warmup；JIT 构建失败时显式
终止，不能切换旧实现。

### 11.5 性能归因

已有 Stage 2 artifact 使用 eager decode，不能作为新 graph decode 的直接基线。
必须先在 benchmark-only commit 上生成检查点 A，再与融合 commit 比较。
