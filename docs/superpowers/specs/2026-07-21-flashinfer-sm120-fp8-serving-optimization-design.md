# RTX PRO 5000：FlashInfer SM120 FP8 MoE 服务 Prefill 优化设计

**状态：** 对话设计已批准，待书面规范审阅

**日期：** 2026-07-21

**目标分支：** `feat/flashinfer-sm120-fp8-moe`

## 1. 背景与当前结论

SGLang 已在单卡 NVIDIA RTX PRO 5000 72GB Blackwell（SM120）上接入
FlashInfer：

```python
flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise
```

当前 `flashinfer_sm120_fp8` MoE backend 的 GEMM1 和 GEMM2 均调用该接口，消费
E4M3 FP8 activation/weight、FP32 block scale，并输出 BF16。已完成的验证证明：

- 完整 runner 数值正确；
- CUDA Graph 可在 replay 间更新 hidden state 和路由；
- Nsight 中出现了该 backend 独有的融合准备 kernel 和 FlashInfer SM120 CuTe GEMM，
  因此实际服务确实进入了新 backend；
- Stage 1 在预打包输入上测得 GEMM1/GEMM2 相对 Triton 有约 13%～26% 的收益；
- 完整 runner 的收益显著缩小，且 skew 路由下可能回退；
- 实际 `bench_serving` 吞吐尚未获得目标中的 10% 提升。

根因不是 FlashInfer GEMM 没有使用 SM120 能力，也不是 host/device 同步，而是两类
开销抵消了 GEMM 收益：

1. SGLang 在 GEMM1 前需要量化、物理 permute 和 scale layout，产生中间张量、额外
   显存流量和多个 kernel launch；
2. FlashInfer ZeroPadding scheduler 会遍历大量空 expert，且 skew 场景下缺少足够的
   L2 局部性和混合 expert 行数适配。

此前 A2 融合已将 `SwiGLU + quant#2 + A2 scale pack` 合并。本设计继续优化 GEMM1
输入准备，并在严格服务 A/B 仍未达标时，才进入 FlashInfer scheduler 优化。

## 2. 目标与非目标

### 2.1 正式目标

唯一端到端 GO 指标是生产形态 `sglang.bench_serving` 的 `input_throughput`，不是
单请求延迟，也不是独立 GEMM 或 runner microbenchmark。

在相同机器、模型、SGLang commit、Python 环境、prompt 集合和服务参数下，以当前
生产使用的 Triton MoE backend 为配对基线：

| input length | 当前 Triton 参考值（tok/s） | 新 backend 正式要求 |
| ---: | ---: | --- |
| 4096 | 32371.59 | 至少 35608.75，即相对同口径 Triton 提升至少 10% |
| 6144 | 31147.57 | 高于同口径 Triton |
| 14336 | 28271.76 | 高于同口径 Triton |
| 30720 | 23142.89 | 高于同口径 Triton |
| 63488 | 17151.96 | 高于同口径 Triton |

表中的数值是历史参考值。最终判定必须在同一 commit 和环境下重新生成 Triton 配对
基线，不能直接把历史结果与新的 FlashInfer 结果比较。

### 2.2 正确性与兼容性目标

1. 保持现有 FP8 模型权重、量化数学、路由权重和 combine 语义。
2. 保持完整 runner 数值阈值和 CUDA Graph 动态输入能力。
3. 不影响 Triton、CUTLASS、非量化和 BF16 路径。
4. backend 只在用户显式选择 `flashinfer_sm120_fp8` 时启用，不静默回退 Triton。
5. 实现对模型、expert 数、`top_k` 和输入长度保持通用，不为 Qwen3.5 或某几个长度
   写特化分支。

### 2.3 非目标

本设计不包含：

- 多卡 TP/EP、DeepEP 或跨卡通信优化；
- dense FP8 GEMM backend 的替换或调优；
- 先调优 Triton baseline；
- 按 input length 硬编码 backend 或 tile；
- 自动切换到 Triton 的双路径策略；
- 把 `bench_one_batch`、单请求 BS=1 延迟或 Nsight 报告作为正式 GO 门槛；
- 在没有新 profile 证据前直接实现 FlashInfer GEMM epilogue 融合。

## 3. 测量口径与实验契约

### 3.1 服务 workload

正式 A/B 沿用现有服务脚本的语义：

```text
python -m sglang.bench_serving
--backend sglang
--dataset-name random
--random-input <input_length>
--random-output 1
--random-range-ratio 1
--num-prompts 100
--flush-cache
```

不设置 request rate，因此测量的是 100 个并发请求产生的聚合 input throughput。服务
端显式固定 `chunked_prefill_size=8192`；在 72GB GPU 上这也与当前默认值一致。长输入
会被切成多个 prefill chunk，调度器还可能把不同请求的 chunk 合并，所以 8192 routed
token 的 runner case 具有生产归因价值，但不能代替服务吞吐判定。

每个 backend 使用 3 个固定随机种子，Triton 和 FlashInfer 使用完全相同的 prompt
集合并采用配对顺序。每个长度必须 `completed == 100`；失败、重试或 token 数不一致的
run 不进入统计。

对每个 seed 单独计算配对吞吐提升：

```text
paired_speedup(seed) = flashinfer_input_throughput(seed)
                       / triton_input_throughput(seed) - 1
```

正式结果取 3 个 `paired_speedup` 的中位数。4096 case 的中位数必须至少为 10%，且
三个 seed 均不得回退；其他四个长度的中位数和每个 seed 都必须大于 0。原始样本全部
保留，不允许只挑最好的一次。

### 3.2 必须固定的变量

除 MoE backend 外，以下变量必须一致：

- GPU 和频率策略；
- 模型目录与 served model name；
- SGLang commit、venv 和 Python executable；
- FlashInfer、Torch、CUDA 和 `sglang-kernel` 版本；
- dense GEMM backend，固定为 `flashinfer_cutlass`；
- attention backend、内存比例、radix cache、解析器和其他 server args；
- `chunked_prefill_size=8192`；
- prompt token IDs、随机种子、请求数和输出长度；
- server warmup、缓存清理与客户端执行顺序。

服务启动后，实验脚本必须读取 `/server_info`，核验实际生效的 MoE backend、dense FP8
backend 和 chunked prefill 配置；不能只相信启动命令文本。

### 3.3 结果元数据与防误复用

当前脚本的进度 key 只有模型、GPU、input length 和 prompt 数，可能把 Triton 结果误当
成 FlashInfer 结果复用。新 harness 的缓存 key 和每条结果必须至少包含：

```text
moe_backend
dense_fp8_backend
sglang_commit
flashinfer_version
flashinfer_artifact_sha256
python_executable
server_args_hash
chunked_prefill_size
input_length
num_prompts
random_seed
timestamp
```

Triton 与 FlashInfer 使用分离的输出目录。CSV/JSON 同时记录 `input_throughput`、
`median_ttft_ms`、`completed`、`total_input_tokens` 和服务器核验结果。TTFT 是诊断指标，
不替代 throughput 主指标。

### 3.4 三层性能证据

1. `bench_serving`：唯一正式端到端门槛；
2. 完整 MoE runner component benchmark：解释收益或回退来自 input prepare、GEMM、
   A2 prepare 还是 unpermute；
3. `bench_one_batch + Nsight`：仅在结果异常、组件归因不充分时临时使用，不是每次
   验证必跑项。

### 3.5 路由分布采样

性能判断不能只依赖 `uniform` 和手工 `synthetic-skew`。另开一个不计入吞吐的 profiling
run，复用 SGLang 现有 expert distribution recorder：

```text
/start_expert_distribution_record
/stop_expert_distribution_record
/dump_expert_distribution_record
```

记录每个真实输入长度、每个 prefill chunk 的 active expert 数、每 expert routed rows
分布、最大值、分位数和空 expert 比例。不得在正式 timed run 的热路径中增加同步、文件
写入或 Python 统计。该数据用于构造可复现的真实路由 profile，并为 scheduler 的通用
cost model 提供依据，不用于识别模型名或输入长度。

## 4. 分阶段方案

采用“先消除 SGLang glue，再按证据修改 FlashInfer scheduler”的分阶段共设计方案：

```text
阶段 A：SGLang A1 融合
  -> 完整正确性和 CUDA Graph
  -> 五个长度的严格服务 A/B
       -> 达标：停止，继续使用官方固定版本 FlashInfer wheel
       -> 未达标：阶段 B

阶段 B：FlashInfer device-side scheduler + L2 swizzle
  -> 同一组测试与服务 A/B
       -> 达标：固定自建 wheel 和 commit
       -> 未达标：重新 profile，再单独设计更深的 GEMM/epilogue 改造
```

阶段 A 和阶段 B 使用独立提交与独立基准产物，确保收益可以归因。不得为了得到更好数字
同时修改 benchmark workload、量化语义和 scheduler。

## 5. 阶段 A：融合 GEMM1 输入准备

### 5.1 当前数据流

```text
hidden BF16 [T, K]
  -> per-token/per-128-group quant
     q_hidden FP8 [T, K]
     q_scale  FP32 [T, K/128]
  -> 根据 src2dst 物理 permute
     packed_hidden FP8 [M, K]
  -> scale gather + transpose + per-expert 4-row padding
     a1_scale_fi FP32 [K/128, m_padded]
  -> FlashInfer GEMM1
```

这里 `M = T * top_k`。`q_hidden` 和 `q_scale` 是只为后续 permute/pack 存在的中间张量，
hidden 被量化一次后，FP8 payload 又被按每条 route 读写一次。

### 5.2 新数据流

保留现有 GPU 路由准备：

```text
topk_ids
  -> moe_permute_prepare
     src2dst int32 [M]
     m_indptr int32 [E+1]
```

将其后的量化、payload scatter 和 scale layout 合并：

```text
hidden BF16 [T, K] + topk_ids + src2dst + m_indptr
  -> fused quant/scatter/scale-pack
     packed_hidden FP8 [M, K]
     a1_scale_fi FP32 [K/128, m_padded]
  -> FlashInfer GEMM1
```

每个源 token/group 只读取一次 BF16、计算一次 FP8 payload 和 scale，再把同一 payload
scatter 到该 token 的各个 routed destination。这样删除 `q_hidden`、`q_scale` 中间张量，
并把原来的 quant、FP8 permute、scale pack 多次 launch 合为一次主 launch。

### 5.3 输入输出契约

定义：

```text
T       token 数
top_k   每个 token 的 routed expert 数
M       T * top_k
E       本地 expert 数
K       hidden width
G       K / 128
```

输入：

```text
hidden    [T, K]       BF16，contiguous
topk_ids  [T, top_k]   int32，contiguous
src2dst   [M]          int32，contiguous
m_indptr  [E+1]        int32，contiguous，位于 GPU
```

输出：

```text
packed_hidden  [M, K]          torch.float8_e4m3fn，contiguous
a1_scale_fi    [G, m_padded]   float32，contiguous，16-byte aligned
m_padded = ((M + 3 * E) // 4) * 4
```

FlashInfer GEMM1 的 `m_indptr`、weight、weight scale、`out=` 和输出 dtype 契约不变。

### 5.4 量化语义

对每个 token 的每个连续 128 元素 group：

```text
absmax    = max(max(abs(value)), 1e-10)
scale     = absmax / 448.0
quantized = E4M3(clamp(value / scale, -448.0, 448.0))
```

必须与现有 `sglang_per_token_group_quant_fp8` 的非 fast-math 结果保持一致。一个 token
选择多个 expert 时，量化结果可复用，但每个 routed slot 都必须写到自己的 `dst`；同一
token 内出现重复 expert ID 也不得破坏映射或覆盖其他 routed slot。

### 5.5 Scale 列映射与 padding

对源 routed slot `r`：

```text
token         = r // top_k
expert        = topk_ids.flatten()[r]
dst           = src2dst[r]
expert_start  = m_indptr[expert]
aligned_start = ((expert_start + 3 * expert) // 4) * 4
scale_col     = aligned_start + dst - expert_start
```

将 token/group 的 FP32 scale 写到 `a1_scale_fi[group, scale_col]`。所有动态 padding
列必须在同一次调用中被写零；CUDA Graph replay 时路由可能改变，不能依赖 capture 前的
一次性初始化，也不能额外调用整张量 `zero_()`。

### 5.6 调度和通用性

首版使用二维逻辑调度：token/group 负责量化，内部沿 `top_k` scatter payload 和 scale。
具体 block 形状由实现阶段的 microbenchmark 选择，但只能依赖 `T/E/top_k/K` 等通用
shape，不得读取模型名或对 4096、6144 等输入长度分支。

如果直接沿 `top_k` scatter 导致写合并差，允许用同一 launch 中的分阶段 block 组织或
增加通用的 destination 排序利用，但不得重新引入完整 `q_hidden/q_scale` 中间张量。所有
索引计算留在 GPU，不允许 `.item()`、`.cpu()` 或 host 侧读取 `m_indptr`。

### 5.7 阶段 A 开关

实验期使用：

```text
SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0  # legacy A1 prepare
SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=1  # fused A1 prepare
```

服务启动时只记录一次选择结果。用户显式启用后若 contract 不满足，必须报出清晰错误，
不得静默切回 legacy 或 Triton。正确性和性能验证完成后，新路径可成为该 backend 默认值，
legacy 开关暂时保留作为可恢复的回滚手段。

## 6. 阶段 B：FlashInfer scheduler 优化（条件执行）

只有阶段 A 完成严格服务 A/B 后仍未达到目标，才创建 FlashInfer fork 并实施本阶段。

### 6.1 现有问题

ZeroPadding scheduler 的 persistent CTA 会从 expert 0 开始顺序扫描 group offsets。
在 256 expert、小 batch 或大量空 expert 下，扫描固定成本可能超过实际 GEMM 工作；在
长尾 skew 下，全局单一 tile 和缺少 L2 swizzle 又会降低热 expert 的 B-weight 复用。

### 6.2 Device-side tile 任务表

在 GEMM 前增加一个 GPU prepare kernel：

```text
rows[e]         = m_indptr[e+1] - m_indptr[e]
tile_count[e]   = ceil_div(rows[e], TILE_M)
tile_indptr     = exclusive_scan(tile_count)   # [E+1]
total_tiles     = tile_indptr[E]
work_counter    = 0
```

所有 workspace 都位于 GPU 并可复用。不得把 `total_tiles` 或 expert rows 复制回 host。
GEMM persistent CTA 通过 device atomic counter 取得逻辑 tile id，再在 `tile_indptr` 上
做 GPU binary search 得到 expert 和该 expert 内的 tile，跳过空 expert 的线性遍历。

该 prepare 成本必须计入完整 runner 和服务时间；不能只测修改后的 GEMM kernel。

### 6.3 L2 swizzle

对同一 expert 内的逻辑 tile 采用通用 L2 swizzle，使相邻 CTA 更可能复用该 expert 的
B-weight tiles。swizzle 只改变 tile 访问顺序，不改变输出布局、数值或 `m_indptr` 契约。

### 6.4 通用 cost model

保留 legacy scheduler 和新 scheduler，由只依赖 `M/E/N/K`、tile 数和 workspace 成本的
通用 cost model 选择。该选择发生在 FlashInfer grouped GEMM 内部，两个分支仍是同一个
FlashInfer backend，不是 Triton fallback。

cost model 的阈值必须来自多组 uniform、empty-heavy 和长尾路由 profile，不能使用模型
名或服务 input length。若 host 侧 shape 足以选择，可以在 capture 前固定；若依赖动态
路由，则必须在 device 上完成且保持 CUDA Graph 安全。

### 6.5 可选的多 TILE_M 调度

只有 prefix-task + L2 swizzle 仍不足，并且 profile 证明全局统一 `TILE_M` 是主要瓶颈时，
才增加 device-side small/medium/large expert 分类。多类 kernel launch 的成本必须纳入
端到端测量；没有净收益就不保留该复杂度。

### 6.6 API 与数值契约

以下接口和语义保持不变：

```python
moe_gemm_fp8_nt_groupwise(
    a,
    b,
    a_scale,
    b_scale,
    m_indptr,
    scale_granularity_mnk=(1, 128, 128),
    scale_major_mode="MN",
    backend="cute",
    out=output,
    out_dtype=torch.bfloat16,
)
```

不改变 E4M3 payload、FP32 scale、MN-major A-scale、NT weight、4-row padding 或 BF16
输出。prepare 和 GEMM 必须支持 CUDA Graph replay 时 `m_indptr` 内容变化。

## 7. 正确性与性能验证

### 7.1 阶段 A 测试顺序

1. CPU contract：backend 选择、量化限制、环境开关、错误信息；
2. GPU adapter：`top_k=1/2/8`、uniform、skew、空 expert、重复 expert ID；
3. 不同 `T/E/K`，其中 `K % 128 == 0`；
4. FP8 payload bitwise 对齐 legacy，FP32 scale 采用严格误差检查；
5. 重用输出 buffer 时 padding 每次都正确清零；
6. 完整 runner 与 Triton 的现有数值阈值；
7. CUDA Graph 多次 replay，hidden、路由和空 expert 分布均变化；
8. runner component benchmark；
9. 五个正式 input length 的三种子服务 A/B。

任何正确性或 CUDA Graph 失败都阻止性能 GO。

### 7.2 阶段 B 补充测试

除复跑阶段 A 全套测试外，增加：

- 全 uniform；
- 大量空 expert；
- 单一 hot expert；
- 长尾 skew；
- 每 expert 只有 1～3 行；
- 全部 expert 非空；
- legacy/new scheduler 逐 case 数值一致；
- replay 间改变 `m_indptr`；
- scheduler prepare、GEMM 与 workspace 的组件时间；
- 五个真实服务长度对应的离线路由 profile。

### 7.3 GO、FUNCTIONAL_ONLY 与停止条件

只有同时满足以下条件才标记 `GO`：

1. 正确性和 CUDA Graph 全部通过；
2. 4096 的三种子配对 `input_throughput` 提升中位数至少为 10%，且每个 seed 均不
   回退；
3. 6144、14336、30720、63488 的配对提升中位数和每个 seed 均大于 0；
4. 没有请求失败、完成数下降或配置漂移；
5. 提升在重复 run 中稳定，不由单个异常样本决定。

若功能正确但未达到吞吐门槛，保持 `FUNCTIONAL_ONLY`，不把 microbenchmark 的 GEMM
收益解释成服务 GO。阶段 A 已达到 GO 就停止，不创建 FlashInfer fork。阶段 B 仍未达到
GO 时停止继续堆改动，重新采集服务/runner/Nsight 证据；任何 GEMM epilogue 或输出融合
必须另写设计。

## 8. FlashInfer fork、制品与回滚

### 8.1 不需要 fork 的情况

若阶段 A 达标，继续使用固定的官方 FlashInfer wheel。这意味着仍然依赖和调用
FlashInfer，只是不维护其源代码分叉。环境清单记录精确版本、下载 URL 和 SHA256。

### 8.2 需要 fork 的情况

若进入阶段 B：

1. 创建 `FiveTreesHigh2/flashinfer`；
2. 从已验证的官方 commit 建立专用分支；
3. scheduler 修改、测试和 benchmark 独立提交；
4. 构建与 CUDA 13.0、Python 3.12 ABI 和 SM120 匹配的 wheel；
5. 把 wheel 放到有长期写权限的 GitHub Release；
6. 记录 fork commit、wheel 文件名、SHA256、Torch/CUDA/CUTLASS DSL 版本；
7. bootstrap 只使用 `uv pip` 安装，不修改原有 venv。

### 8.3 回滚

- A1 回滚：设置 `SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0`；
- scheduler 回滚：重新安装记录在 manifest 中的官方 FlashInfer wheel；
- backend 回滚：显式选择 Triton MoE backend；
- 不删除服务器上的旧 venv、旧 wheel 或历史 benchmark 产物。

## 9. 实施提交边界

建议按以下边界提交，便于服务器同步和二分：

1. 服务 benchmark harness 元数据、缓存 key 和 `/server_info` 核验；
2. A1 RED 测试；
3. A1 fused kernel 与 Python adapter；
4. runner 接线、CUDA Graph 和完整回归；
5. 阶段 A 性能结果与结论；
6. 仅在需要时：FlashInfer fork scheduler RED/实现/结果；
7. 最终 manifest、部署说明和回滚说明。

实现期间保留用户未跟踪文件：

```text
test/registered/moe/debug_flashinfer_sm120_fp8_stagewise.py
```

不得把它误加入提交，也不得覆盖或删除。

## 10. 设计原则总结

本轮不再把“更快的单个 GEMM”直接等同于“更快的服务”。先通过 A1 融合减少 SGLang
为 FlashInfer 准备输入所支付的代价，再以真实服务 A/B 决定是否需要维护 FlashInfer
scheduler fork。所有选择均由通用 shape 和 GPU 路由分布驱动，不特化模型或输入长度；
所有性能收益必须包含 prepare、GEMM、调度和服务框架的完整成本。
