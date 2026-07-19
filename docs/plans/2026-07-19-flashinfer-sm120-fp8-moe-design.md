# FlashInfer SM120 FP8 Routed-MoE 接入设计

**状态：** 已批准进入实现规划
**日期：** 2026-07-19
**目标分支：** `feat/flashinfer-sm120-fp8-moe`
**SGLang 基线：** `8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587`
**目标模型：** Qwen3.5-35B-A3B-FP8
**目标 GPU：** NVIDIA RTX PRO 5000 72GB Blackwell，计算能力 12.0

## 1. 目标结果

为 SGLang 增加一个实验性的 routed-MoE 后端，在两次 expert GEMM 中都调用
FlashInfer 已有的 SM120 CuTe kernel：
`flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise`。接入必须保留 SGLang
现有的模型加载、路由、shared expert 和回退路径，不得复制、fork 或重新实现
FlashInfer GEMM kernel。

初始后端通过以下参数显式选择：

```text
--moe-runner-backend flashinfer_sm120_fp8
```

在新路径通过本文档定义的正确性和性能闸门之前，`auto` 继续选择现有后端。

## 2. 背景与基线

目标模型包含 40 个 MoE 层、256 个 routed expert、top-k 8、hidden size 2048、
expert intermediate size 512，以及 1 个 shared expert。checkpoint 使用 FP8 E4M3
权重，配套 float32 `[128, 128]` block scale；激活使用 K-group 为 128 的
per-token 动态量化。

chunked-prefill 工作负载下测得的 Triton 基线如下：

| 操作 | 有效 grouped shape | 延迟 |
| --- | --- | ---: |
| GEMM1 / w13 | M=65536, N=1024, K=2048 | 1.267 ms |
| GEMM2 / w2 | M=65536, N=2048, K=512 | 0.794 ms |
| 40 层两次 Triton expert GEMM 推算合计 | `(1.267 + 0.794) × 40` | 约 82.44 ms |

前两项来自原始计划记录的锁频 NCU `耗时/launch`。最后一项是两次 GEMM 延迟
在 40 层上的算术推算，不是包含路由、量化、激活、combine 和 shared expert 的
完整 routed-MoE 实测时间。

FlashInfer 目标入口使用相同的权重量化粒度，并采用 token-packed zero-padding
模式，从而避免 Triton 路径中的 expert `BLOCK_M` 激活 padding。

## 3. 固定版本与仓库策略

### 3.1 Git 布局

- `origin`：`git@github.com:FiveTreesHigh2/sglang.git`
- `upstream`：`https://github.com/sgl-project/sglang.git`
- `origin/main` 保持为 upstream 镜像。
- 所有开发都在 `feat/flashinfer-sm120-fp8-moe` 上进行，分支起点固定为上述
  SGLang 基线。
- 服务器上不直接编辑代码。每次实验只从 fork 拉取并以 detached HEAD 方式
  checkout 精确 commit SHA。
- 实验 manifest 记录 Git SHA、依赖版本、wheel 哈希、启动命令、相关环境变量
  和 benchmark 输入。

### 3.2 本地与服务器目录

本地开发 clone：

```text
/Users/bsy/Desktop/workspace/Pro5000-Optimize/sglang/
```

服务器持久化根目录：

```text
/home/logs/sennian/pro5000-fi-moe/
  sglang/
  .venv/
  wheelhouse/
  cache/
  runs/
```

仓库和虚拟环境互为同级目录。wheel、cache、模型文件、profile 和运行产物均不
提交到 Git。

### 3.3 旧环境

保留现有 `/home/logs/sennian/py-venv/sglang5.14` 环境，不做原地升级。它继续
作为可以立即启用的运行回退环境。

## 4. 依赖设计

目标 API 首次出现在 FlashInfer PR #3891 合入后的指定 nightly 中。实验固定
使用匹配的 core 和 JIT-cache 构建：

| 包 | 版本或产物 | SHA256 |
| --- | --- | --- |
| `flashinfer-python` | `0.6.15.dev20260716` | `ed0634d9c32f069dafe7583addf74de7a4f366ae07d3093250109bd315b4ba26` |
| `flashinfer-jit-cache` | `0.6.15.dev20260716+cu130` | `86a0944b4cadde0a4227f249e5a0fe466207d7c25e8eb7dee4c3f75fdd5f9bbf` |

feature branch 将 SGLang 稳定版依赖
`flashinfer_python[cu13]==0.6.15` 更新为新 API 所需的精确开发版本。
JIT-cache 的版本前缀必须与 core 版本一致。

经修正后的服务器工具链事实如下：

- Python 3.12
- PyTorch 2.11.0，CUDA 13.0
- NVIDIA driver 580.126.09
- CUDA compiler 13.0，NVCC 13.0.48
- 计算能力 `(12, 0)`

优先使用官方 JIT-cache，以避免首次调用时的编译延迟。生产运行不设置
`FLASHINFER_DISABLE_JIT`。如果 AOT 产物不存在，FlashInfer 可以使用 NVCC 和
Ninja 进行 runtime-JIT 回退。bootstrap 会显式预热目标 kernel，并把编译 cache
放在实验根目录下的持久化位置。

`FLASHINFER_DISABLE_JIT=1` 只作为可选诊断项：若调用成功，说明官方
JIT-cache wheel 包含目标模块；若调用失败，只要取消该变量后能使用 NVCC 编译
并通过同一个 smoke test，就不阻塞环境验收。

## 5. SGLang 架构决策

评估过以下三种接入结构：

1. 在 `Fp8MoEMethod.apply` 中增加直接条件分支。修改量小，但会进一步耦合
   量化、路由和执行逻辑。
2. 注册单个 `none -> FlashInfer` fused function。它与成熟 fused backend 的
   结构接近，但不利于在实验阶段分别隔离和分析流水线各环节。
3. 增加一等的 `MoeRunnerCore`，并注册 standard pre-permute 和 post-permute
   适配器。

选择方案 3。它遵循当前 runner 架构，并使路由、量化、GEMM、激活和 combine
可以独立测试。预计涉及以下文件：

- `python/sglang/srt/layers/moe/utils.py`：backend 枚举和判断方法；
- `python/sglang/srt/layers/moe/moe_runner/runner.py`：实例化新 runner core；
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`：quant info、
  runner input/output、pre-permute、两次 GEMM 调用和 post-permute；
- `python/sglang/srt/layers/quantization/fp8.py`：选择 runner、构造 quant info，
  并缓存 FlashInfer 所需的 weight-scale 布局；
- 聚焦的单元测试、CUDA 正确性测试和 benchmark 文件。

## 6. 初始支持边界

第一版只支持：

- CUDA 计算能力 12.0 或 12.1；
- 已序列化的 FP8 E4M3 routed-expert 权重；
- `[128, 128]` blockwise float32 weight scale；
- 动态激活量化；
- `moe-a2a-backend=none`；
- TP=1、EP=1；
- 目标 Qwen gated-SiLU 语义，且 `gemm1_alpha`、GEMM1 clamp 和
  `swiglu_limit` 均未设置；
- standard combine；目标工作负载的 top-k 为 8；
- 不支持 expert bias、LoRA 和 `no_combine` 模式。

shared expert 继续走 SGLang 现有 dense 路径。本次修改不会把它作为第 257 个
group 追加到 grouped GEMM 中。

显式选择实验 backend 时，如果硬件或模型语义不受支持，应在模型或 runner
初始化阶段报错。运行期异常不得被静默转换成 Triton 调用。现有 `auto` 和显式
Triton 路径保持不变，并作为确定性的回退方案。

## 7. FlashInfer kernel 契约

两次 expert GEMM 都调用同一个官方函数：

```python
from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
```

固定输入契约如下：

- `a`：token-packed FP8，shape 为 `[cum_m, K]`；
- `b`：FP8，shape 为 `[num_experts, N, K]`；
- `a_scale`：float32，shape 为 `[ceil(K/128), m_padded]`；
- `b_scale`：float32，shape 为
  `[num_experts, ceil(K/128), ceil(N/128)]`；
- `m_indptr`：CUDA int32，shape 为 `[num_experts + 1]` 的 CSR 前缀和；
- scale 粒度固定为 `(1, 128, 128)`；
- `scale_major_mode="MN"`，`backend="cute"`；
- 输出为 BF16。

expert `i` 的 A-scale 写入起点为：

```text
floor((m_indptr[i] + 3*i) / 4) * 4
```

并且：

```text
m_padded = floor((cum_m + 3*num_experts) / 4) * 4
```

## 8. 前向数据流

### 8.1 路由与打包

standard dispatcher 提供 BF16 hidden states 以及由 SGLang 计算的 top-k ids 和
weights。pre-permute 适配器执行以下操作：

1. 展平 token/top-k assignment；
2. 统计每个 expert 的 assignment 数量；
3. 生成 CUDA int32 `m_indptr`；
4. 按 expert-major、token-packed 顺序写入 token 行，不做 expert `BLOCK_M`
   padding；
5. 保存 assignment 到 packed row 的逆映射，供 combine 使用。

现有 Triton 和 DeepGEMM preprocess 的输出不能不加修改地传入：前者使用
block padding，后者使用 masked expert tensor；而目标 API 要求紧凑行，仅允许
A-scale 列发生 padding。实现应在契约相同时复用它们的路由语义和辅助操作，
但不能为了减少代码量而复用不兼容的 tensor 布局。

### 8.2 激活量化与 A-scale 布局

正确性优先版本复用 SGLang 现有的 per-token-group FP8 量化，group size 为
128，然后执行最小必要的 scale-layout 适配，将结果转换为 FlashInfer
zero-padding 布局。该适配必须逐项遵循 FlashInfer 官方参考转换。

如果 profile 证明独立 repack launch 明显抵消了 GEMM 收益，后续可以优化
SGLang 量化 helper，使其直接写出目标布局。这仍属于接入适配优化，不是修改
FlashInfer GEMM。

### 8.3 GEMM1

目标模型的 GEMM1 形状为：

```text
A:   [cum_m, 2048]
W13: [256, 1024, 2048]
out: [cum_m, 1024] BF16
```

输出沿用 SGLang 普通 Qwen `silu_and_mul` 路径使用的连续 gate/up 两半语义，
计算 `SiLU(gate) * up`，得到 BF16 `[cum_m, 512]`。实现必须与现有 Triton
结果对照验证该语义，而不能仅通过 `gate_up_interleaved` 标志进行推断；当前
runner 只在特殊 alpha/clamp 激活变体中读取该标志。

### 8.4 GEMM2

激活输出再次使用 group size 128 进行动态量化，并复用同一个 `m_indptr`：

```text
A:  [cum_m, 512]
W2: [256, 2048, 512]
out:[cum_m, 2048] BF16
```

### 8.5 Combine

post-permute 适配器使用保存的逆映射，为每个原始 token 收集 expert 输出，应用
原始 top-k weight 和 routed scaling factor，并将 8 个 routed contribution 求和，
得到 `[num_tokens, 2048]`。

## 9. 权重处理

FP8 权重保持 SGLang 标准的连续 `[E, N, K]` 形式，不为新 kernel 重新量化或
转置。

checkpoint block scale 加载后的布局为 `[E, N_blocks, K_blocks]`。加载完成后，
新 backend 创建非持久化 cache tensor：

```text
[E, N_blocks, K_blocks] -> transpose(1, 2).contiguous()
                         -> [E, K_blocks, N_blocks]
```

保留原始 scale parameter，不进行覆盖，以保证显式 Triton 路径、hot reload
行为和 A/B 对比仍然可用。

## 10. 正确性闸门

验证必须按顺序进行；任何阶段失败都会阻塞下一阶段。

### 10.1 环境与独立 API

- 导入精确的 FlashInfer API；
- 确认 CUDA 可用且计算能力为 `(12, 0)`；
- 记录 NVCC、Torch CUDA、包版本和文件位置；
- 测试均衡、非均衡和空 expert 输入；
- 测试真实 GEMM1 和 GEMM2 shape；
- 验证第二次调用复用已安装或已编译的 cache。

可选 no-JIT 诊断先运行。若失败，记录失败信息，并在启用 runtime JIT 后重新
执行测试。

### 10.2 适配层测试

- `m_indptr` 与 expert histogram 前缀和精确一致；
- packed row 和逆映射能够精确 round-trip；
- A-scale 写入位置与官方参考实现逐元素一致；
- 覆盖空 expert、集中路由、非均衡路由和零 token 输入。

### 10.3 Kernel 与 runner 数值测试

- 独立 FlashInfer GEMM 使用官方 normalized error 指标，与 BF16 reference
  对比时要求 `calc_diff < 1e-3`；
- 集成后的 FlashInfer 与 Triton 路径使用相同的 hidden states、权重、top-k ids
  和 top-k weights；
- 两者分别与同一个 BF16/dequantized reference 比较；
- FlashInfer routed-MoE 误差不得比 Triton 误差高出超过 10%；
- 检查输出有限性，并验证 gate/up 顺序、SiLU、top-k 加权、routed scaling 和
  shared-expert 合成行为。

### 10.4 模型级测试

使用固定 prompt、随机种子和 sampling 参数，分别比较 prefill 与 decode 的
logits、greedy token 和生成文本。FP8 结果不要求逐 bit 相同，但所有实质性差异
都必须记录并排查。

## 11. 性能闸门

完成 warmup 后使用 CUDA event 测量，并分别报告 kernel-only、适配层、完整
routed MoE 和模型级耗时。测试同时覆盖均匀路由和采集到的真实路由分布。

| 闸门 | 要求 |
| --- | --- |
| GEMM1 | 不超过 1.014 ms；相对 1.267 ms 至少提升 20% |
| GEMM2 | 不超过 0.834 ms；相对 0.794 ms 回退不超过 5% |
| 完整 routed MoE | 包含打包、量化、激活和 combine 后至少提升 10% |

如果 kernel-only 性能通过，但适配层抵消了收益，只进行一轮有针对性的适配优化，
然后重新执行完整闸门。如果仍无法获得有意义的完整 MoE 提升，则停止生产接入，
保留实验分支用于分析。

初始 backend 在功能上同时支持 prefill 和 decode：官方 zero-padding 入口特别
面向小 expert-M decode，同时也支持大 M。在设计任何自动选择或按 forward mode
选择的逻辑之前，必须分别测量性能和 CUDA graph 兼容性。

## 12. Stage B：可复现服务器环境

在实现 runner 之前先完成 Stage B：

1. 在 feature branch 上提交已批准的设计、精确依赖 pin、bootstrap、smoke test
   和环境采集脚本。
2. 将 feature branch 推送到 fork，并记录精确 SHA。
3. 在服务器把 fork clone 到持久化根目录，checkout 该 SHA。
4. 使用 `uv` 和 Python 3.12 创建 `.venv`，不修改旧 venv。
5. 支持断点续传地下载 wheel，并在安装前校验 SHA256。
6. 安装 fork 和精确依赖，然后运行 `pip check`。
7. 运行可选 no-JIT 诊断以及必须执行的正常 prewarm/smoke test。
8. 把 manifest 和输出保存到 `runs/`，再返回输出进行审查。

当新 venv 能在 RTX PRO 5000 上成功调用目标 kernel 且数值正确时，Stage B
通过。官方 AOT-cache 加载成功，或者经过记录并预热的 CUDA 13.0 runtime-JIT
编译成功，二者都可接受。

## 13. 同步与回滚

后续每次实验都先在本地 commit 并 push。服务器只执行：

```bash
git fetch origin
git switch --detach <exact-commit-sha>
```

回滚不需要删除任何内容：退出新 venv、返回旧环境，或者 checkout 已记录的历史
commit 即可。大型 wheel 和 cache 会被保留以避免重复下载，但始终位于 Git 之外。

## 14. 延后工作

以下内容不属于第一版接入：

- 修改或调优 FlashInfer CuTe/CUDA kernel 源码；
- TP>1、EP>1 或任意 A2A backend；
- 把 shared expert 融入 grouped GEMM；
- LoRA、expert bias 或 no-combine 支持；
- 静默运行时回退；
- 自动 backend 选择；
- 生产容器化和灰度发布，它们只会在性能闸门通过后进行。

## 15. 权威参考

- [FlashInfer FP8 SM120 zero-padding API](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/flashinfer/grouped_mm/cute_sm120_fp8_groupwise/core.py)
- [FlashInfer FP8 SM120 测试](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/tests/grouped_mm/test_cute_sm120_fp8.py)
- [FlashInfer AOT 注册表](https://github.com/flashinfer-ai/flashinfer/blob/nightly-v0.6.15-20260716/flashinfer/aot.py)
- [SGLang MoE runner](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/moe/moe_runner/runner.py)
- [SGLang DeepGEMM runner 适配模式](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py)
- [SGLang FP8 MoE method](https://github.com/sgl-project/sglang/blob/8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587/python/sglang/srt/layers/quantization/fp8.py)
