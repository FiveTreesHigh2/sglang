# RTX PRO 5000：FlashInfer SM120 Groupwise FP8 MoE 集成设计

**状态：** 对话设计已批准，待书面规范审阅

**日期：** 2026-07-20

**目标分支：** `feat/flashinfer-sm120-fp8-moe`

## 1. 背景

目标模型为 Qwen3.5-35B-A3B-FP8，部署设备为单卡 NVIDIA RTX PRO 5000
72GB Blackwell，CUDA 计算能力为 12.0。现有 SGLang routed MoE 使用 Triton
blockwise FP8 路径，chunked-prefill 性能受到 MoE GEMM 限制。

待接入的 FlashInfer API 为：

```python
flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise
```

Stage B 已验证以下运行环境：

- PyTorch `2.11.0+cu130`；
- FlashInfer `0.6.15.dev20260716`；
- CUDA toolkit / NVCC `13.0.48`；
- `nvidia-cutlass-dsl 4.5.2`；
- GPU 计算能力 `(12, 0)`；
- FlashInfer runtime JIT 可用；
- GEMM1 和 GEMM2 smoke test 通过。

Stage 1 已在锁定 GPU 时钟的条件下证明目标 kernel 具备继续集成的性能价值。
决定性 case 为 GEMM1、uniform、`cum_m=65536`：

```text
Triton      1.119975 ms
FlashInfer  0.891171 ms
speedup     25.67%
```

全部 8 个 Stage 1 case 的正确性和 GPU 稳定性检查均通过。Stage 2 的任务是把
该 GEMM-only API 接入 SGLang 的真实单卡 MoE 数据流，并验证完整 runner、CUDA
Graph 和模型服务，而不是修改 FlashInfer kernel 本身。

## 2. 目标与非目标

### 2.1 目标

1. 新增显式 MoE runner backend：`flashinfer_sm120_fp8`。
2. 在 SGLang 标准单卡 dispatcher 后消费真实 `hidden_states`、`topk_ids` 和
   `topk_weights`，不生成或修改路由结果。
3. GEMM1 和 GEMM2 都调用
   `flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise`。
4. 保留 SGLang 现有路由、SwiGLU、shared expert 处理和最终 combine 语义。
5. 原生消费 `(128, 128)` blockwise FP8 checkpoint，不转换成 per-tensor FP8。
6. 覆盖 prefill、decode 和 CUDA Graph replay。
7. 不静默回退到 Triton；环境或模型不兼容时在启动阶段明确失败。
8. 用 Triton 和 SGLang CUTLASS 作为完整 runner 性能对照。

### 2.2 非目标

本阶段不做以下工作：

- 不修改 FlashInfer kernel 源码；
- 不支持 DeepEP、多卡 EP、Mooncake、NIXL 或 FlashInfer A2A；
- 不支持 TP/EP 大于 1；
- 不把新 backend 加入 `auto` 自动选择；
- 不重写 SGLang router、top-k、shared expert 或 combine 算法；
- 不把 blockwise FP8 权重重新量化为 per-tensor FP8；
- 不支持 NVFP4、MXFP4、INT8、FP8 FNUZ 或非 `(128, 128)` block shape；
- 不实现 expert bias、非 gated MoE 或非 SwiGLU activation；
- 不支持 LoRA、TBO、SBO 或其他未列入验收矩阵的组合；显式启用时必须
  fail fast；
- 不用模型名或 Qwen 固定 shape 硬编码 kernel 选择。

这些限制描述首个生产验证范围，不表示目标 FlashInfer GEMM 在数学上只能用于
Qwen。后续扩展必须基于独立正确性和性能证据逐项放开。

## 3. 方案选择

### 3.1 采用方案：独立 RunnerCore

新增独立的 `FlashInferSm120Fp8RunnerCore`，采用 SGLang 已有的
`MoeRunnerCore + pre/post permute` 边界：

```text
Fp8MoEMethod
  -> MoeRunner
  -> StandardDispatchOutput
  -> FlashInferSm120Fp8RunnerCore
  -> StandardCombineInput
```

该方案与 Humming grouped-contiguous 和 DeepGEMM runner 的仓库结构一致：runner
负责组织两次 grouped GEMM，路由与 combine 仍属于 SGLang。

### 3.2 未采用：改造 Triton runner 为 GEMM provider

Triton 当前主要使用 indexed/sorted-token 数据布局，目标 FlashInfer API 需要
expert-contiguous packed A 和 CSR `m_indptr`。把 Triton runner 抽象为可替换 GEMM
provider 会扩大改动面，并把两种不相同的数据契约强行塞进同一个实现。

### 3.3 未采用：复用现有 fused FlashInfer backend

现有 `flashinfer_cutlass` 调用完整 `cutlass_fused_moe`，其 SGLang FP8 接入主要为
ModelOpt per-tensor scale。现有 `flashinfer_cutedsl` 主要面向 NVFP4，
`flashinfer_trtllm` 使用另一套 fused API 和权重布局。它们都不等价于本项目的
GEMM-only、`(1, 128, 128)` groupwise FP8 API。

独立 backend 能确保实验确实调用目标 kernel，也符合“显式选择、失败即报错、
不静默换 kernel”的要求。

## 4. 组件边界

### 4.1 Backend 注册与选择

新增枚举值和 CLI 选择：

```bash
--moe-a2a-backend none \
--moe-runner-backend flashinfer_sm120_fp8
```

该 backend 不进入 `auto`。显式选择后，任何不兼容条件都必须终止启动。

### 4.2 QuantInfo 与权重准备

新增只携带目标 kernel 所需内容的 quant info：

- `w13_weight`、`w2_weight`；
- FlashInfer 布局的 `w13_scale`、`w2_scale`；
- block shape 和 dtype 元数据；
- 必要的 expert、hidden 和 intermediate shape。

权重 FP8 payload 不重排、不重新量化。只在模型加载完成后，为 weight scale 建立
FlashInfer 专用 contiguous 布局，并保留原始 SGLang scale。

### 4.3 RunnerCore

RunnerCore 只负责：

1. 生成 expert packing 元数据；
2. 量化并打包 GEMM1 输入；
3. 调用 FlashInfer GEMM1；
4. 执行 SGLang SwiGLU 和 GEMM2 输入量化；
5. 调用 FlashInfer GEMM2；
6. 调用 SGLang unpermute/combine；
7. 返回标准 combine input。

RunnerCore 不决定 token 去哪个 expert，也不修改 `topk_ids/topk_weights`。

### 4.4 Scale layout op

新增一个小型 GPU layout op，把普通 row-major activation scale 写成 FlashInfer
要求的 MN-major、per-expert 4-row-padded 布局。该 op 只搬运 scale，不重新计算
scale，不访问 CPU。

## 5. 数据契约

定义：

```text
T      当前 token 数
top_k 每个 token 选择的 expert 数
M      T * top_k，即有效 routed rows
E      本地 expert 数；本阶段等于全局 expert 数
K      hidden size
N      intermediate size
```

### 5.1 标准 dispatcher 输入

```text
hidden_states  [T, K]       BF16
topk_ids       [T, top_k]   integer
topk_weights   [T, top_k]   floating point
```

`topk_ids/topk_weights` 完全来自现有 SGLang router。

### 5.2 权重与 scale

Checkpoint 权重：

```text
w13_weight [E, 2N, K] FP8 E4M3FN
w2_weight  [E, K, N]  FP8 E4M3FN
```

Checkpoint scale：

```text
w13_scale [E, 2N/128, K/128]
w2_scale  [E, K/128, N/128]
```

FlashInfer GEMM1/GEMM2 的 B-scale 布局：

```text
w13_scale_fi [E, K/128, 2N/128]
w2_scale_fi  [E, N/128, K/128]
```

转换仅为 `transpose + contiguous`，在加载阶段执行一次。

### 5.3 `m_indptr`

SGLang packing metadata 生成：

```text
m_indptr[e]     expert e 在 packed A 中的第一行
m_indptr[e + 1] expert e 在 packed A 中的结束行
```

必须满足：

- shape 为 `[E + 1]`；
- dtype 为 `int32`；
- 第一个元素为 0；
- 单调不下降；
- 最后一个元素为 `M`；
- 允许 `m_indptr[e] == m_indptr[e + 1]`，即空 expert。

## 6. 前向数据流

### 6.1 路由元数据

复用 SGLang `moe_permute_prepare` 的 GPU 元数据生成逻辑，得到：

- `src2dst`：原 token/top-k slot 到 expert-contiguous 行的映射；
- `expert_offsets`：语义上直接作为 `m_indptr`。

本阶段 `M = T * top_k`，不存在多卡 EP 过滤后的无效 expert 行。

### 6.2 GEMM1 输入：先量化一次，再打包

不照抄 Humming grouped-contiguous 的“先复制 BF16，再量化”顺序。对于较大的
`top_k`，该顺序会重复量化同一个原 token。

采用：

```text
hidden_states [T, K] BF16
  -> per-token/per-128-K-group quant
q_hidden      [T, K] FP8
q_scale       [T, K/128] FP32
  -> 使用同一 src2dst 分别打包 FP8 数据和 scale
packed_hidden [M, K] FP8
a1_scale_fi   [K/128, M_padded] FP32
```

FP8 数值和 scale 只改变排列，不进行第二次量化。

### 6.3 FlashInfer GEMM1

调用：

```python
moe_gemm_fp8_nt_groupwise(
    packed_hidden,
    w13_weight,
    a1_scale_fi,
    w13_scale_fi,
    m_indptr,
    scale_granularity_mnk=(1, 128, 128),
    scale_major_mode="MN",
    backend="cute",
    out=gate_up_output,
    out_dtype=torch.bfloat16,
)
```

输出：

```text
gate_up_output [M, 2N] BF16
```

### 6.4 SwiGLU 与 GEMM2 输入

复用 SGLang activation/quantization 实现，优先使用 fused
`silu_and_mul + per-token-group FP8 quant`：

```text
gate_up_output [M, 2N] BF16
  -> fused SwiGLU + per-128 quant
down_input     [M, N] FP8
down_scale     [M, N/128] FP32
  -> scale layout op
a2_scale_fi    [N/128, M_padded] FP32
```

GEMM1 的每条 routed 行已经属于不同 expert，GEMM2 输入必须逐 routed 行量化，
这里不存在可复用的原 token 量化结果。

### 6.5 FlashInfer GEMM2

调用相同 API，仅替换 A/B、scale 和输出：

```python
moe_gemm_fp8_nt_groupwise(
    down_input,
    w2_weight,
    a2_scale_fi,
    w2_scale_fi,
    m_indptr,
    scale_granularity_mnk=(1, 128, 128),
    scale_major_mode="MN",
    backend="cute",
    out=down_output,
    out_dtype=torch.bfloat16,
)
```

输出：

```text
down_output [M, K] BF16
```

### 6.6 Unpermute 与 combine

复用 SGLang `moe_unpermute`：

- 根据 `src2dst` 恢复原 token 顺序；
- 应用 `topk_weights`；
- 汇总一个 token 的多个 expert 输出；
- 应用现有 `routed_scaling_factor`；
- 得到 `[T, K]` BF16。

shared expert 仍走 SGLang 现有上层路径，不进入目标 FlashInfer GEMM。

## 7. Activation scale 的 FlashInfer 布局

FlashInfer A-scale 不是普通 `[M, K_blocks]` 转置。每个 expert 的 scale 起始位置
需要 4 行对齐。对 expert `e`：

```text
packed_scale_start[e] = ((m_indptr[e] + 3 * e) // 4) * 4
```

总 scale buffer 的第二维为：

```text
M_padded = ((M + 3 * E) // 4) * 4
```

layout op 根据 `src2dst/m_indptr` 在 GPU 上完成：

```text
row-major scales [source_rows, K_blocks]
  -> MN-major padded scales [K_blocks, M_padded]
```

必须满足：

- 不使用 `.item()`、`.tolist()` 或 Python expert 循环；
- 不发生 CPU/GPU 同步；
- 支持空 expert 和极端 skew；
- padding 区域写零；
- GEMM1 和 GEMM2 共用实现；
- 同一份 scale 数值只搬运，不重新计算。

## 8. CUDA Graph 与显存生命周期

### 8.1 输出管理

目标 FlashInfer API 原生支持 `out=`。SGLang 在调用前创建输出 tensor，并显式
传入：

```text
GEMM1 -> gate_up_output
GEMM2 -> down_output
```

这与现有 `flashinfer_cutlass` 的 output 管理原则一致，并避免
`flashinfer_trtllm` 部分路径中的“临时输出后再 copy 到固定输出”开销。

### 8.2 Graph 地址稳定

“固定 buffer”指 graph replay 生命周期内地址固定，不表示为每个 MoE 层永久
保留最大 prefill workspace：

- eager 模式使用 PyTorch CUDA caching allocator；
- capture 时临时 tensor 进入 CUDA Graph memory pool；
- replay 使用 capture 固定下来的地址；
- FlashInfer 接收 SGLang tensor 的 `out` 地址，不在 API 内隐藏输出分配。

### 8.3 防止按层放大显存

每层只永久保存权重引用、转换后的 weight scale 和少量配置。不得为所有 MoE
层分别永久保存 `M=65536/131072` 的大型 activation workspace。大型中间结果应
为调用期临时 tensor、graph pool tensor，或由经验证可安全复用的共享 workspace
管理。

### 8.4 Capture 条件

- FlashInfer JIT 必须在 capture 前完成；
- capture/replay 路径不得有 host sync；
- graph 对应的 tensor shape 和地址固定；
- `m_indptr`、top-k 和 activation 内容允许变化；
- output 和中间 buffer 内容每次 replay 正确覆盖；
- 动态 prefill shape 在 eager 路径按需分配，固定 decode/piecewise graph shape
  由现有 SGLang graph 机制管理。

## 9. 能力检查与失败策略

### 9.1 必须满足的条件

| 项目 | 要求 |
| --- | --- |
| 设备 | CUDA，计算能力 12.0 或 12.1 |
| FlashInfer | 可导入目标 grouped-mm API |
| 权重 | `torch.float8_e4m3fn` |
| 量化 | blockwise FP8 |
| block shape | `(128, 128)` |
| activation input | BF16，动态 per-token-group 量化 |
| GEMM output | BF16 |
| shape | GEMM 的 N/K 满足 128 对齐 |
| MoE | gated SwiGLU，无 expert bias |
| 并行 | `tp=1`、`ep=1`、A2A `none` |
| 其他执行模式 | LoRA、TBO、SBO 均关闭 |

不限制模型名称、expert 数、`top_k`、token 数或固定的 Qwen shape。

### 9.2 Fail fast

显式选择 backend 后，发现不兼容条件时在模型加载或 runner 初始化阶段抛出包含
实际值和期望值的异常。不得自动切换到 Triton 或 CUTLASS。

性能基线必须通过重新启动并显式指定：

```bash
--moe-runner-backend triton
```

或：

```bash
--moe-runner-backend cutlass
```

### 9.3 JIT

不设置 `FLASHINFER_DISABLE_JIT=1`。服务器已有 NVCC 13.0，默认允许 runtime
JIT。warm-up 阶段应先导入并执行 GEMM1/GEMM2，再进入 graph capture。

如果用户禁用了 JIT 且缓存不存在，应保留原始 `MissingJITCacheError` 异常链，
并提示移除该环境变量或安装/预构建匹配的 JIT cache。

后端使用 API feature detection，不仅依赖版本字符串；实验环境和 bootstrap
继续固定已验证的 FlashInfer nightly 版本以保证可复现。

### 9.4 运行时诊断

FlashInfer 异常补充：

```text
operation=gemm1|gemm2
E/M/N/K
block_shape
device_capability
weight/activation dtype
```

启动只记录一次 backend、FlashInfer API、GPU capability、scale contract、kernel
backend 和 CUDA Graph 状态，不逐层或逐请求刷日志。

## 10. 测试设计

### 10.1 注册与能力测试

验证：

- backend enum、CLI 和 runner factory；
- Fp8MoEMethod 只为匹配的 blockwise FP8 创建新 runner；
- 不兼容 GPU、dtype、block shape、并行方式、bias 和 activation 明确失败；
- 不发生静默 fallback。

### 10.2 Packing 与 layout 测试

覆盖：

- `top_k=1/2/8`；
- uniform、skew 和空 expert；
- `T=1/8/128/8192`；
- `m_indptr` 不变量；
- packed token/scale 与 PyTorch reference 一致；
- padding 位置和零填充正确；
- unpermute、top-k 加权和 routed scaling factor 正确。

### 10.3 完整 runner 数值测试

比较 Triton full MoE runner 和新 runner，计入：

```text
routing metadata
+ input quant
+ FP8/scale pack
+ GEMM1
+ SwiGLU + quant
+ GEMM2
+ unpermute/combine
```

覆盖 routed rows：

```text
M=8       小 decode
M=64      batched decode
M=1024    小 prefill
M=65536   chunked-prefill=8192、top_k=8
M=131072  大 prefill 压力测试
```

所有 case 必须输出有限值，并满足 Stage 1 同类指标：

```text
calc_diff < 1e-3
```

### 10.4 CUDA Graph

1. eager warm-up 完成 JIT；
2. capture 新 backend；
3. replay 相同 shape、不同 hidden states；
4. replay 相同 shape、不同 expert 分布；
5. 每次与 eager reference 比较；
6. 连续 replay，确认显存不持续增长。

硬性要求：capture 成功、replay 不触发 JIT、无非法显存访问、无旧输入污染，且
改变输入后输出相应改变。

### 10.5 真实模型服务

使用单卡目标模型和以下关键参数：

```bash
--tp-size 1 \
--ep-size 1 \
--moe-a2a-backend none \
--moe-runner-backend flashinfer_sm120_fp8
```

保持 CUDA Graph 正常开启，覆盖短 prompt、长 decode、batch decode、8192
chunked-prefill、多轮连续请求和并发请求。验证正常生成、无 NaN/Inf/乱码、显存
稳定、日志明确使用新 backend，且没有 fallback。

## 11. 性能验收

### 11.1 对照项

比较：

```text
triton
cutlass
flashinfer_sm120_fp8
```

主 benchmark 测完整 MoE runner。额外 component profile 分别记录 routing、quant、
pack、GEMM1、SwiGLU+quant、GEMM2 和 unpermute/combine，用于定位非 GEMM 开销。
为延续 Stage 1 的可比性，`M=65536` uniform profile 是固定的主判定 case；
synthetic-skew 和真实模型路由回放（可获得时）必须同时记录，但不替代该固定
判定输入。

### 11.2 频率与复现

普通功能测试不锁频。最终性能验收在用户确认 GPU 独占后锁定相同 SM 时钟，沿用
Stage 1 的采样和稳定性检查；测试结束后恢复默认频率。

### 11.3 结果分级

#### GO

- 正确性、CUDA Graph、真实服务全部通过；
- `M=65536` 完整 MoE runner 相对 Triton 至少提升 10%；
- 已测 decode case 不出现超过 5% 的稳定性能回退。

#### FUNCTIONAL_ONLY

功能和 CUDA Graph 正确，但 quant/pack/combine 开销吞掉主要 kernel 收益。保留为
显式实验 backend，不推荐生产启用，并继续优化数据搬运。

#### NO_GO

数值正确性、CUDA Graph 或真实服务稳定性失败，不进入部署。

CUTLASS 是必须记录的对照而不是新 backend 的硬性胜负门槛。如果 CUTLASS 更快，
报告必须如实展示并分析原因。

## 12. 实施与审查顺序

书面设计获批后，先生成独立实施计划，再开始代码修改。实施遵循：

1. 先写能力检查、layout/reference 和 runner 测试；
2. 再注册 backend 和 quant info；
3. 实现 packing/scale layout；
4. 接入两次 FlashInfer GEMM；
5. 完成 eager correctness；
6. 完成 CUDA Graph；
7. 在服务器运行完整 runner 和模型实验；
8. 最终锁频对照 Triton、CUTLASS 和新 backend；
9. 提交前进行代码审查和完整验证。

代码修改开始前必须再次取得用户批准。服务器无法由开发端 SSH 登录，所有服务
器命令由用户执行；虚拟环境中的包管理命令必须使用 `uv pip --python ...`。
