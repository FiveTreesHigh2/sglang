# Pro5000 Triton MoE Down/GEMM2 调优设计

日期：2026-07-22  
目标分支：`feat/flashinfer-sm120-fp8-moe`

## 1. 目标

为 NVIDIA RTX PRO 5000 72GB Blackwell（SM120）上的 Qwen3.5-35B-A3B FP8 MoE 生成独立的 Triton down/GEMM2 配置。目标工作负载是持续多并发、等待队列充足的 prefill；服务器使用 `chunked_prefill_size=8192`，因此主要优化单次 forward 总 token 数接近 8192 的场景。

本次只改 benchmark/tuning 工具和测试，不修改 Triton production kernel，也不改变 FlashInfer backend。生成的配置按 SGLang 既有规则落为 `_down.json`。

## 2. 必要的特化边界

Triton 配置天然依赖设备、Triton 版本和 GEMM shape。本次允许配置针对以下契约生成：

- GPU：NVIDIA RTX PRO 5000 72GB Blackwell（SM120）；
- Triton：服务器实际安装版本，当前预期为 3.6.0；
- MoE：`E=256`、`top_k=8`；
- FP8：`fp8_w8a8`、`block_shape=[128, 128]`；
- GEMM2：输入 K=512、输出 N=2048；
- 配置文件沿用 SGLang 命名约定，文件名中的 `N=512` 表示 SwiGLU 后 intermediate size，而不是 GEMM2 输出宽度。

production kernel 不针对 Qwen 类名、固定请求文本或完整 prompt 长度增加分支。

## 3. 当前工具的缺口

现有 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` 已能分别测量 GEMM1/GEMM2，并比较 GEMM2 的 TMA 与非 TMA 路径，但不能直接用于当前目标：

1. `load_topk_ids` 硬编码 61 层、前 3 层 dense 的 DeepSeek 文件命名；
2. `--topk-ids-dir` 强制必填，无法使用通用合成路由；
3. 单个 `--batch-size --tune` 会丢弃返回值，不打印或保存配置；
4. tuning 同时执行 GEMM1 和 GEMM2，down-only 场景浪费时间；
5. 默认 batch-size 列表最大只有 4096，缺少 6144 和 8192；
6. 现有 `BestConfigTrace` 为了同时兼顾两次 GEMM，会要求 GEMM1/GEMM2 共用 `BLOCK_SIZE_M`；down-only 调优不应受此限制。

## 4. 命令行接口

在保持旧命令兼容的前提下，增加：

- `--kernel {up,down,both}`：默认 `both`；本任务使用 `down`；
- `--batch-sizes M [M ...]`：与旧 `--batch-size` 互斥；
- `--route-profiles PROFILE [PROFILE ...]`：支持 `uniform`、`synthetic-skew`；
- `--route-seeds SEED [SEED ...]`：确定性生成多份路由；
- `--output PATH`：显式指定结果 JSON，避免在工作目录静默覆盖文件；
- `--full-search-size`：默认使用 `batch-sizes` 中最大值，本任务为 8192；
- `--shortlist-size`：完整搜索后保留的候选数量，默认 16。

`--topk-ids-dir` 改为可选。显式传入时保留旧的文件路由模式；未传入时使用合成路由。`--batch-size` 继续支持旧的单配置 benchmark 用法。

## 5. 路由生成

合成路由必须满足每个 token 内的 `top_k` expert ID 互不重复，但不同 token 之间允许重复。

- `uniform`：expert 选择尽量均匀覆盖 256 个 expert，并由 seed 改变排列；
- `synthetic-skew`：构造少量热 expert 和长尾冷 expert，但不绑定任何真实请求；
- 所有 profile 均使用 GPU kernel 实际接受的 dtype/shape；
- 相同参数和 seed 必须生成相同路由，便于复现。

真实 serving 路由只用于最终验证，不参与配置选型。

## 6. 搜索策略

### 6.1 主优化点

`num_tokens=8192` 执行完整搜索。实际 GEMM2 routed rows 为 `8192 * 8 = 65536`。

完整搜索覆盖现有 CUDA search space，并继续遵守 FP8 block-shape 对 `BLOCK_SIZE_K` 的约束。每个候选分别测量 TMA 与非 TMA，把 `(config, USE_TMA)` 视为独立候选。

搜索分两阶段：

1. 粗筛：在两个 route profile 上以较少迭代运行全部候选；
2. 稳定复测：取每个 profile 的前若干名并集，在全部 route seed 上增加 warmup/iteration，按 median 选型。

最终按最小化跨 profile 的最大相对 regret 选择；若相同，再比较跨 profile median。这样不会让绝对耗时较低的 profile 掩盖另一个 profile 的明显回退。

### 6.2 非满批与尾批

`2048/4096/6144` 不重新执行完整搜索，只复测：

- 8192 shortlist；
- 当前默认 heuristic 配置；
- shortlist 的有限邻域配置。

每个 M 独立决定配置与 `USE_TMA`。

### 6.3 小 M 保护

同一个 `_down.json` 也会被 decode 和低负载 forward 读取。若文件只有大 M key，最近邻查找会让小 M 错用 prefill 配置。因此结果必须包含 `1/8/32/128/512` 锚点。

这些锚点只评估精简候选集：默认 heuristic、8192 shortlist 和必要的非 TMA 变体；不消耗完整搜索预算。任何小 M 配置都必须经过实测，不能只复制一个未经验证的大 M 配置。

## 7. 输出与安装

调优结果先写到用户显式指定的 `--output`，不得直接覆盖 production 配置。JSON key 是原始 `num_tokens`，不是 `num_tokens * top_k`。

目标配置名为：

```text
E=256,N=512,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
```

结果验证通过后，再复制到：

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<server-version>/
```

也允许通过 `SGLANG_MOE_CONFIG_DIR` 指向独立配置根目录做 A/B 测试。安装前后必须从启动日志确认 `_down.json` 被加载。

## 8. 测试与验收

CPU 单元测试覆盖：

- 新旧 CLI 参数兼容和互斥关系；
- uniform/skew 路由的 shape、范围、token 内无重复和确定性；
- down-only 不执行/记录 GEMM1；
- 单 batch tuning 不再丢失结果；
- `_down.json` 文件名、key 排序和 `USE_TMA` 序列化；
- 结果包含全部大 M 与小 M 锚点；
- robust score 在构造数据上选择预期候选。

服务器验证分三层：

1. preflight：单个小候选确认模型配置、TMA/非 TMA、结果落盘均正常；
2. 正式 tuning：生成完整 `_down.json` 和原始 timing 记录；
3. A/B：同一 Triton backend 下比较默认 down 与 tuned down，首先看 GEMM2 latency，再运行持续多并发 serving。主判断点为 8192-token prefill batch；同时确认小 M 和 decode 没有不可接受回退。

调优只承诺优化 Triton down/GEMM2。最终端到端 prefill 吞吐是否提高，需要 serving A/B 独立确认。

## 9. 非目标

- 不修改 Triton production kernel；
- 不修改 FlashInfer kernel 或 FlashInfer backend；
- 不采集或提交真实请求的 top-k 路由；
- 不针对 4096、6144、14336、30720、63488 等完整 prompt 长度写分支；
- 不在本次实现 shared expert 融合或 FlashInfer scheduler 优化。
