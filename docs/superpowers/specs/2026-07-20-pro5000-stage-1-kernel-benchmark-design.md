# RTX PRO 5000 Stage 1：FlashInfer 与 Triton FP8 MoE 微基准设计

**状态：** 已批准（2026-07-20）

**日期：** 2026-07-20

**目标分支：** `feat/flashinfer-sm120-fp8-moe`

## 1. 背景

Qwen3.5-35B-A3B-FP8 在 NVIDIA RTX PRO 5000 72GB Blackwell（计算能力
12.0）上的 chunked-prefill 性能受到 routed MoE 限制。当前 SGLang 的
blockwise FP8 路径最终使用 Triton `fused_moe_kernel`；待评估的 FlashInfer
入口为：

```python
flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise
```

Stage B 已经在服务器的新 venv 中验证：

- PyTorch `2.11.0+cu130`；
- FlashInfer `0.6.15.dev20260716`；
- CUDA toolkit / NVCC `13.0.48`；
- GPU 计算能力 `(12, 0)`；
- FlashInfer kernel 首次 runtime-JIT 和后续缓存复用均成功；
- GEMM1、GEMM2 小规模正确性 smoke 均通过。

下一步不是直接修改 SGLang runner，而是先回答一个独立问题：在相同 FP8
数据、相同 expert 行数和相同 GEMM 工作量下，FlashInfer kernel 本身是否比
当前 Triton kernel 快到足以承担后续接入成本。

## 2. 目标与非目标

### 2.1 目标

1. 对 GEMM1 和 GEMM2 分别进行低层、kernel-only 的公平比较。
2. 同时覆盖 `cum_m=65536` 和 `cum_m=131072`。
3. 验证均匀路由、固定非均匀路由，以及未来真实路由回放所需的数据契约。
4. 用 FP32 反量化参考检查 FlashInfer 和 Triton 的数值正确性。
5. 在默认 boost 状态下先完成低成本筛选；只在结果接近门槛、波动明显或最终
   验收时要求锁频。
6. 输出可留档、可比较、可定位失败原因的结构化结果。

### 2.2 非目标

Stage 1 明确不做以下工作：

- 不修改 FlashInfer kernel 源码；
- 不修改 SGLang MoE runner、路由器、模型执行路径或后端枚举；
- 不实现生产用 activation quant / scale repack；
- 不测完整 `quant -> GEMM1 -> activation -> quant -> GEMM2` 流水；
- 不启动 Qwen3.5 模型，也不采集真实线上路由；
- 不安装、升级或卸载 Python 包；
- 不把首次 runtime-JIT 时间计入 kernel latency；
- 不在 benchmark 脚本内自动锁定或恢复 GPU 频率；
- 不在本阶段执行完整 NCU/NSYS 分析。只有微基准 gate 通过后，才对决定性
  case 补 profiler 数据。

## 3. 关键术语与真实集成关系

### 3.1 GEMM1 与 GEMM2

| 操作 | 权重 | `(N, K)` | 含义 |
| --- | --- | --- | --- |
| GEMM1 | `w13` | `(1024, 2048)` | hidden state 乘 gate/up 合并权重 |
| GEMM2 | `w2` | `(2048, 512)` | SwiGLU 中间结果乘 down 权重 |

两个操作都会调用待评估的同一个 FlashInfer API，只是矩阵形状不同。

### 3.2 `rows_per_expert` 与 `m_indptr`

`rows_per_expert[e]` 表示 expert `e` 本次收到的 routed token 行数。
`m_indptr` 是它的前缀和：

```text
rows_per_expert = [3, 0, 5, ...]
m_indptr        = [0, 3, 3, 8, ...]
```

微基准没有真实路由器，因此需要构造或加载该数据。正式集成时不依赖静态
JSON；SGLang 会从每次前向的真实 top-k 路由结果生成 expert 分组和 offsets，
FlashInfer adapter 只消费该真实数据，不自行虚构模型路由。

## 4. 被测形状与输入分布

### 4.1 固定形状

```text
E = 256
GEMM1: N = 1024, K = 2048
GEMM2: N = 2048, K = 512
cum_m = 65536, 131072
output dtype = bfloat16
FP8 dtype = float8_e4m3fn
block scale = (128, 128), float32
```

`cum_m=65536` 对应当前重点 chunk size 8192、top-k 8 的 routed 行数；
`cum_m=131072` 用于观察 chunk size 16384 时的扩展性。

### 4.2 Uniform profile

均匀分配总行数：

```text
cum_m=65536  -> 每个 expert 256 行
cum_m=131072 -> 每个 expert 512 行
```

这是性能判定的主 profile。它便于隔离 GEMM 主体效率，但由于行数恰好可被
常见 Triton `BLOCK_SIZE_M=64` 整除，不能代表真实路由的 expert 尾块浪费。

### 4.3 Synthetic-skew profile

固定种子生成可重复的非均匀分布，并满足以下不变量：

- expert 数始终为 256；
- 行数总和严格等于指定 `cum_m`；
- 至少包含一个空 expert；
- 至少包含一个小于 Triton `BLOCK_SIZE_M` 的 expert；
- 至少包含一个明显高于均值的热点 expert；
- 分布和 seed 写入结果 JSON。

该 profile 用于检查空 expert、小 expert、热点 expert、FlashInfer 4 行 scale
对齐，以及 Triton per-expert padding。它是人为压力数据，不冒充 Qwen3.5 的
真实路由，也不单独决定 go/no-go。

### 4.4 真实数据回放接口

脚本预留 `--rows-per-expert-json <path>`。输入至少包含长度为 256 的非负整数
数组；脚本验证长度、总和和数值范围。将来从真实 SGLang 前向导出数据后，可
直接回放而不修改 benchmark 逻辑。Stage 1 首版不负责采集该文件。

## 5. 公平比较设计

### 5.1 采用方案：低层 GEMM 对比

主比较直接调用：

- FlashInfer：`moe_gemm_fp8_nt_groupwise`；
- Triton：SGLang 当前 `fused_moe_kernel` 低层 Triton kernel。

不采用上层 `invoke_fused_moe_kernel` 作为 Stage 1 主对照，因为该 wrapper 会在
每次调用中执行 activation FP8 quant。FlashInfer 生产路径还需要对应的 quant
和 A-scale repack；两者应在 Stage 2 的完整流水微基准中一起衡量，不能与
Stage 1 的 GEMM 主体混在同一个数字里。

### 5.2 共享数据源

每个 case 只生成一份 BF16 原始数据，并只量化一次。两种实现必须共享：

- 相同的 FP8 A 数值；
- 相同的 FP8 B 数值；
- 相同的 FP32 block scale 数值；
- 相同的 `rows_per_expert`；
- 相同的 BF16 输出类型；
- 禁用 router weight 乘法、bias、activation 和 combine。

不得为两种 backend 独立随机生成输入。

### 5.3 A-scale 的两个等价布局

一次 per-token、per-128-K-block 量化后保留两种 scale view：

1. Triton 行主序：`(cum_m, K / 128)`；
2. FlashInfer 布局：`(K / 128, m_padded)`，每个 expert 段按 FlashInfer 契约
   做 4 行边界对齐。

两种布局来自同一组 scale 数值，只改变排布，不重新计算量化参数。

### 5.4 权重与 B-scale

FP8 权重固定为 `(E, N, K)`。B-scale 固定为与 128x128 block 对应的
`(E, N / 128, K / 128)` 语义，并根据两个 API 的实际 stride 契约提供 contiguous
view。任何 transpose 只发生在准备阶段，不进入计时区间。

### 5.5 Triton 路由与 padding 元数据

benchmark 用 `top_k=1` 表示每一行已经完成 expert packing。由
`rows_per_expert` 构造 `topk_ids`，再复用 SGLang 当前对齐逻辑生成：

- `sorted_token_ids`；
- `expert_ids`；
- `num_tokens_post_padded`。

Triton 继续按它选定的 `BLOCK_SIZE_M` 对每个 expert 补齐，因此非均匀 profile
仍能反映 Triton 尾块开销。FlashInfer 使用无 Triton M-block padding 的 packed A
与 CSR `m_indptr`。两边完成的有效 expert GEMM 行数相同。

### 5.6 Triton 配置

默认复用当前 SGLang 对该 shape、FP8 block shape `(128, 128)` 的配置选择逻辑，
并把最终 `BLOCK_SIZE_M/N/K`、warp、stage 等配置写入 JSON。脚本同时允许未来
通过显式配置覆盖，以便重放历史报告中的配置；首版不进行 Triton autotune。

历史延迟 `GEMM1=1.267 ms`、`GEMM2=0.794 ms` 只作为 sanity check 展示，
不参与正式 speedup 计算。

## 6. 正确性

### 6.1 参考值

参考不是原始 BF16 矩阵乘，而是：

1. 按实际 FP8 数值和 FP32 scale 反量化 A；
2. 按实际 FP8 数值和 FP32 scale 反量化各 expert 权重；
3. 对每个 expert 执行 FP32 matmul；
4. 以 backend 的 BF16 输出转换到 FP32 后比较。

这样检查的是两种 kernel 是否正确解释相同的 FP8/scale 契约，不把两者共同的
量化误差误判成 kernel 错误。

### 6.2 必须满足的条件

每个 backend、每个 shape、每个 profile 都必须满足：

- 输出 shape 为 `(cum_m, N)`；
- 输出 dtype 为 `torch.bfloat16`；
- 输出全部为有限值；
- `calc_diff(output, reference) <= 2e-3`；
- 同时记录 FlashInfer 与 Triton 输出之间的 `calc_diff`。

任何正确性失败都终止对应正式计时，并使进程非零退出。性能再好也不能覆盖
正确性失败。

## 7. 计时方法

### 7.1 计时边界

计时区间只包含 kernel launch 和 GPU 执行，不包含：

- 随机输入生成；
- activation/weight 量化；
- scale transpose/repack；
- 路由排序和元数据构造；
- 输出/reference 分配；
- FlashInfer 或 Triton 首次 JIT；
- 正确性 reference 计算；
- `nvidia-smi` 查询。

所有输入和输出 buffer 在计时前完成分配并在迭代间复用。

### 7.2 JIT 与 warmup

每个新 shape/backend 先执行一次 untimed prepare call，完成 runtime-JIT。该次
调用的 wall time 单独记录为 `jit_prepare_seconds`。随后执行：

```text
warmup = 20 calls
iterations = 100 calls per trial
trials = 5
```

每个 trial 使用同一 CUDA stream，以 CUDA Event 包围 100 次调用，并在读取
Event 前同步。最终记录每次 trial 的平均毫秒数以及 min/median/max。

### 7.3 交替顺序

为降低温度与 boost 漂移偏差，trial 顺序交替：

```text
trial 0: Triton -> FlashInfer
trial 1: FlashInfer -> Triton
trial 2: Triton -> FlashInfer
...
```

不在 backend 之间执行 `empty_cache`，也不重新生成输入。

## 8. GPU 频率策略

### 8.1 默认 boost 首测

首轮不要求锁频。运行器在 benchmark 前后保存完整 `nvidia-smi`，并在每个
trial 的计时区间外采样至少以下字段：

- P-state；
- SM clock；
- temperature；
- power draw；
- utilization；
- memory used。

benchmark 不自行调用 `nvidia-smi -lgc`，以免权限失败或影响共享 GPU。

### 8.2 条件锁频

以下情况需要在用户确认 GPU 独占且有权限后，用相同脚本锁定 1732 MHz 复测：

- GEMM1 主 case 的默认 boost speedup 在 15% 到 30% 之间；
- 任一主 backend 的 5 轮相对 spread `(max - min) / median` 大于 5%；
- GPU 时钟、温度、P-state 或 utilization 表明存在明显漂移/竞争；
- 需要形成最终可复现验收报告。

CUDA 13.0 是 toolkit 版本，1732 MHz 是 GPU SM 时钟，两者没有数值对应关系。

## 9. Go / No-Go 规则

### 9.1 决定性 case

正式决策只使用：

```text
operation = GEMM1 / w13
profile = uniform
cum_m = 65536
E = 256, N = 1024, K = 2048
```

定义：

```text
speedup = Triton median_ms / FlashInfer median_ms - 1
```

### 9.2 默认 boost 初筛

在正确性全部通过的前提下：

| 条件 | 决策 |
| --- | --- |
| speedup >= 30%，每轮优势稳定，relative spread <= 5% | `GO` |
| speedup 位于 15% 到 30%，或波动/环境采样异常 | `NEEDS_LOCKED_RERUN` |
| speedup < 15%，且环境与结果稳定 | `NO_GO` |

边界按保守原则处理：恰好 15% 进入 `NEEDS_LOCKED_RERUN`；恰好 30% 只有在
稳定性条件也满足时才为 `GO`。

### 9.3 锁频复测

若执行锁频复测，原计划的最终 gate 不变：GEMM1 决定性 case 必须达到
`speedup >= 20%` 才进入 Stage 2。

### 9.4 非决定性结果

- GEMM1 `cum_m=131072`、synthetic-skew 和默认 boost 的其他 case 全部记录，
  用于识别扩展性与分布风险，但不单独覆盖决定性 case。
- GEMM2 的所有结果全部记录；即使提升不足也不阻塞 Stage 2，因为其短 K
  问题原本计划在后续 tile 调优阶段处理。
- 若任一必测 case 正确性失败，总决策为 `ERROR`，而不是性能 `NO_GO`。

## 10. 文件与职责

计划实现以下文件；最终名称可在实施计划中做不改变职责的微调：

| 文件 | 职责 |
| --- | --- |
| `scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py` | 输入、正确性、两种低层 kernel、计时和 JSON 结果 |
| `scripts/pro5000/run_stage_1_benchmark.sh` | 创建 run 目录、环境检查、GPU 快照、stdout/stderr 留档 |
| `scripts/pro5000/README.md` | 增加 Stage 1 服务器执行与条件锁频说明 |
| `test/registered/unit/test_pro5000_stage_1.py` | CPU 纯逻辑、静态契约和 shell 安全测试 |

优先复用现有 `flashinfer_sm120_fp8_smoke.py` 中已经在服务器验证过的
`compute_padded_offset`、offset 构造、`calc_diff` 和 FlashInfer scale packing
语义。实施时应通过小范围公共 helper 或无副作用导入复用，避免复制后产生两套
不一致的 scale 契约。

Stage 1 不修改 `python/pyproject.toml`，也不执行任何 package install。

## 11. 执行流程

服务器端的一次完整运行按以下顺序进行：

1. 用户从 fork 拉取已批准的精确提交。
2. 激活 `/home/logs/sennian/pro5000-fi-moe/.venv`。
3. wrapper 验证当前 Python、git SHA、工作树和结果目录。
4. 检查 RTX PRO 5000、CC 12.0/12.1、CUDA、NVCC、FlashInfer API 和版本。
5. 采集 environment manifest 与 benchmark 前 GPU 状态。
6. 对每个 shape/profile 构造一次共享 FP8 数据。
7. 完成两种 backend 的 untimed JIT prepare。
8. 运行全部正确性检查。
9. 正确性通过后交替执行 5 轮性能测试。
10. 输出人类可读表格与 `benchmark.json`。
11. 采集 benchmark 后 GPU 状态并给出 `GO`、`NO_GO`、
    `NEEDS_LOCKED_RERUN` 或 `ERROR`。

正常运行不得设置 `FLASHINFER_DISABLE_JIT=1`。FlashInfer JIT cache 继续使用
Stage B 已建立的持久化 cache。

## 12. 产物设计

默认结果目录：

```text
/home/logs/sennian/pro5000-fi-moe/runs/stage-1-<timestamp>-<git-sha>/
├── benchmark.json
├── benchmark.stdout.txt
├── benchmark.stderr.txt
├── environment.json
├── nvidia-smi-before.txt
└── nvidia-smi-after.txt
```

`benchmark.json` 至少包含：

- schema version、时间戳、完整命令行；
- git SHA、branch、dirty 状态；
- Python、PyTorch、CUDA、FlashInfer、Triton、SGLang 版本；
- GPU 名称、计算能力、显存和运行模式；
- 所有 benchmark 参数；
- 每个 profile 的 `rows_per_expert` 与统计摘要；
- Triton 实际 kernel 配置与 padding 行数；
- 两种 backend 的 JIT prepare 时间；
- correctness 指标；
- 每轮 latency、min/median/max、relative spread；
- speedup、历史基线参考；
- GPU trial 采样；
- 最终 decision 和逐条 reason。

大体积 A/B/output tensor 默认不保存，避免结果目录膨胀。JSON 不记录环境变量
全集、token、SSH 配置或其他 secret。

## 13. 退出码与失败处理

- 环境不满足、输入不合法、JIT 失败或正确性失败：非零退出，并在 stderr 和
  JSON（若能够创建）中记录明确阶段、case、异常类型和消息。
- 性能 `NO_GO` 或 `NEEDS_LOCKED_RERUN`：进程正常退出；它们是有效实验结果，
  不能伪装成脚本执行失败。
- wrapper 必须保留 Python 进程原始退出码，并即使失败也尽力采集 after 快照。
- 首次 JIT 很慢属于准备阶段，只要最终成功就不是 latency 回归。
- 不允许脚本静默回退到另一个 FlashInfer API、另一个 dtype 或 BF16 GEMM。

## 14. 测试与验收

### 14.1 本地 CPU 测试

不需要 GPU 的测试至少覆盖：

- uniform 分布长度和总和；
- synthetic-skew 的确定性和不变量；
- 外部 JSON 的错误处理；
- `m_indptr` 与 4 行 FlashInfer scale 对齐 offsets；
- Triton per-expert padding 元数据；
- trial 汇总、relative spread 和边界判定；
- 必需 CLI 默认值；
- wrapper bash syntax、禁止裸 `pip`、禁止自动 `nvidia-smi -lgc`；
- benchmark 不允许设置 `FLASHINFER_DISABLE_JIT=1`。

### 14.2 服务器 GPU 验收

服务器证据必须证明：

- 所有必测 case 均执行，没有静默 skip；
- 两种 backend 均通过正确性；
- 第二次运行不再产生首次 FlashInfer JIT 的长等待；
- stdout 表格与 JSON 数值一致；
- decision reasons 与阈值一致；
- benchmark 前后环境未发生 package 变更。

本地没有 RTX PRO 5000，不能以本地静态测试替代服务器 GPU 证据。

## 15. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| uniform profile 隐藏真实 expert 尾块 | 同时测 synthetic-skew，并预留真实 JSON 回放 |
| 直接调用 Triton 内部 kernel 对版本敏感 | 固定 fork SHA，复用当前 config/metadata helper，JSON 记录完整配置 |
| 默认 boost 产生误判 | backend 交替、多 trial、采样 GPU 状态；中间区间条件锁频 |
| FlashInfer/Triton scale 布局不同导致假比较 | 同一次量化产生两种 layout，正确性使用同一 FP8/scale 反量化参考 |
| 首次 JIT 污染计时 | 独立 untimed prepare，单独记录 wall time，再 warmup |
| 低层 kernel 胜出但接入总收益不足 | Stage 2 单独测 quant/repack 与完整双 GEMM 流水 |
| 低层脚本误用了其他实现 | 记录入口和源码模块；禁止静默 fallback |

## 16. 已批准的设计决策

用户已逐项批准：

1. 没有真实 `m_indptr` 时，首版使用 uniform + 固定 synthetic-skew，并预留
   外部 JSON 回放；正式集成从真实 SGLang 路由在线生成 `m_indptr`。
2. 采用低层 kernel-only Triton 对照；完整 quant/repack 流水延后到 Stage 2。
3. 默认 boost 首测；锁频只在接近门槛、波动明显或最终验收时执行。
4. Stage 1 只新增 benchmark、运行 wrapper、文档和测试，不修改 SGLang runner
   或 Python 依赖。
