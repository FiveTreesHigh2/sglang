# 优化工作详细报告 · 草稿收纳（2026-08-03）

正式总结报告见 `2026-07-31-optimization-summary-report.md`（提纲级）。本文件收纳正在编写的**详细版报告**已定稿章节，按最终文档顺序排列。

---

## 【已定稿】第一章：背景

### 性能模拟与差距定位

优化工作启动前，首先在 RTX PRO 5000 上对 Qwen3.5-35B-A3B-FP8 进行了实测与性能建模，并向 InferSim 提交了 PR 以补充 PRO 5000 的硬件性能画像（[alibaba/InferSim#12](https://github.com/alibaba/InferSim/pull/12)）。

模拟与实测的对比结论如下（Prefill，Input 4096）：

| | Attention MFU | MoE MFU | Throughput (tok/s) |
|---|---|---|---|
| 理论上限（按 H20 MFU 折算） | 0.80 | 0.60 | 42250 |
| 实测口径模拟上限 | 0.65 | 0.39 | 36649 |
| 实际测试值 | 0.65 | 0.39 | **30033** |

由此得出两个判断：

1. **Decode 受显存带宽约束显著**，kernel 级优化的收益空间有限，更大的收益应来自投机解码等算法级手段，故暂不作为优化主线；
2. **Prefill 实测值（30033 tok/s）与同 MFU 口径的模拟上限（36649 tok/s）存在约 18% 的差距**，说明当前实现存在明显的工程损耗；同时模拟给出的算子 MFU（MoE 仅 0.39）表明 kernel 本身也有进一步提升空间。因此将 prefill 确立为优化主线。

进一步调研发现，SGLang 社区对 SM120（RTX Blackwell）系列的支持尚不完善：针对该架构的优化路径较少，FP8 量化支持也以 MXFP8 为主，对本模型使用的 (128,128) 块量化 FP8 覆盖不足。因此性能优化需要在 SGLang 及其调用的底层 kernel 层面自行修改实现。

### 初步实验：Dense FP8 GEMM 后端修正

首先通过 nsys 对 prefill 稳态执行进行了 kernel 级分析，发现一个直接问题：**所有 dense FP8 GEMM 均回退（fallback）到了 Triton 实现，未能命中 FlashInfer/CUTLASS 算子**。在启动参数中显式指定 `--fp8-gemm-backend flashinfer_cutlass` 后复测，确认 dense GEMM 已正确路由至 CUTLASS 算子，prefill 吞吐由 **30033 提升至 32371 tok/s（+7.8%）**。此配置作为后续所有优化工作的基线。

### 基线 kernel 预算分析与优化方向

在修正 dense GEMM 后端后，对稳态 kernel 时间占比进行了完整分析，得出如下优先级排序：

- **MoE GEMM**：占比最高（~29%）且 MFU 偏低（gemm2 SM 利用率仅 48%），确立为最高优先级；
- **Dense GEMM**：切换 CUTLASS 后 SM 利用率已达 75–84%，接近硬件极限，优化空间有限，不再投入；
- **qkvzba split/cat 及 norm/quant/act 长尾**：均为带宽受限的搬运/逐元素算子，无法通过调优加速，但可通过**消除冗余读写**（去物化、算子融合）回收开销，作为并行推进项。

基于上述分析，确定以 MoE GEMM 为切入点。调研发现 FlashInfer 社区已有面向 SM120 的 FP8 grouped GEMM 算子更新（[flashinfer-ai/flashinfer#3891](https://github.com/flashinfer-ai/flashinfer/pull/3891)），遂决定以集成该算子为基础展开后续优化。

---

## 【已定稿】第二章：MoE Backend 切换：Triton fused_moe → FlashInfer CuTe SM120 Grouped GEMM

### 1. 基线实现：SGLang Triton fused_moe 的工作方式

SGLang 默认的 MoE 执行路径由 `moe_align_block_size` 与 `fused_moe_kernel` 两级构成：

**路由对齐（moe_align_block_size）**。top-k 路由产生 `num_tokens × top_k` 个 (token, expert) 对后，该 kernel 将其按 expert 排序，并把每个 expert 的行段**填充到 BLOCK_M 的整数倍**，产出三个元数据张量：`sorted_token_ids`（含填充哨兵）、`expert_ids`（每个 M-block 归属的 expert）与 `num_tokens_post_padded`。

**融合 GEMM（fused_moe_kernel）**。GEMM 以 2D grid 启动，每个 thread block 通过 `expert_ids` 确定自己服务的 expert，并按 `sorted_token_ids` **在 kernel 内以间接寻址（gather）方式装载激活行**——因此 Triton 路径不需要对激活做物理重排，"permute"的成本隐藏在 GEMM 的加载路径中。GEMM2 通过 `MUL_ROUTED_WEIGHT` 在写出时乘以路由权重，最后由 `moe_sum` 完成 top-k 归约。

该实现的两个结构性开销在 ncu 分析中被量化：

1. **BLOCK_M 填充浪费**：chunk 8192（65536 routed rows）在 256 expert、BM=64 配置下，有效 M-tile 为 1024 个，实际调度 1277 个（**~10% 的 tile 为纯填充计算**）；
2. **GEMM2 配置错配**：本模型 `N=2048, K=512` 形状对应的 `_down.json` tuning 配置缺失，运行时退回复用 up-projection 配置且无 TMA，Tensor pipe 利用率仅 **48%**（GEMM1 为 62%）；主要停顿为 wait（35–38%）与 long_scoreboard（~20%），K 维仅 4 个 128-block 导致主循环过短、无法隐藏访存延迟。

### 2. 引入的实现：FlashInfer CuTe SM120 Zero-Padding Grouped GEMM

FlashInfer 自 [#3891](https://github.com/flashinfer-ai/flashinfer/pull/3891) 起提供面向 SM120 的 FP8 blockwise-scale grouped GEMM（`moe_gemm_fp8_nt_groupwise`），其 zero-padding 模式与本模型的量化格式（weight (128,128) 块 scale + 激活 per-token-group(128) scale）精确匹配。核心设计：

- **Token-packed 激活 + CSR 组描述**：激活按 expert 紧凑排列（无 BLOCK_M 填充），各 expert 段边界由 `m_indptr[E+1]` 前缀和描述。相比 Triton 的对齐填充方案，从机制上消除了 ~10% 的填充 tile 浪费，且对小 per-expert M（decode 场景 m_pe=1）不产生额外的显存与算力开销；
- **Warp-specialized persistent kernel**：grid 固定为 SM 数（110），每 SM 常驻 1 个 384 线程 block，内部按角色分工——TMA 加载 warp（A/B 操作数）、scale 加载 warp、存储 warp 与 8 个 math warp，通过 mbarrier 多级流水线衔接，由 persistent scheduler 依次认领 tile；
- **Tile 尺寸启发式**：按 per-expert M 在四档 kernel 模板（SwapAB 128×8 / M32 / M64 / M128）间选择，其中 SwapAB 变体面向小 M 场景（将 M、N 操作数对调以适配 8 行以下的超窄形状）。

### 3. 直接切换面临的问题

该算子无法直接替换 Triton 路径，根源在于**两者的数据契约完全不同**——Triton 的间接寻址将重排成本隐藏在 GEMM 内部，而 FlashInfer 要求输入在 GEMM 之前已物理就位：

1. **激活需要物理 permute**：必须新增重排步骤把激活按 expert 打包为 token-packed 布局（`moe_permute_prepare`：torch.sort 产出 CSR offsets 与 `src2dst` 映射，随后 scatter 拷贝），GEMM2 之后还需对应的 `unpermute + top-k 加权求和`。这部分在 Triton 路径中是"免费"的，切换后成为显式的 glue kernel 链；
2. **激活 scale 布局转换**：kernel 要求 A-scale 为**列主序 `[k_blocks, m_padded]` 布局，且每个 expert 的 scale 列以 4 行对齐、基地址 16B 对齐**——与 SGLang 通用 per-token-group 量化产出的行主序 `[rows, k_blocks]` 完全不同，需要专用的 Triton 打包 kernel 完成转置 + 按 expert 对齐散射（含填充列清零）；
3. **权重 scale 布局转换**：kernel 消费 `[E, k_blocks, n_blocks]`，checkpoint 加载产出 `[E, n_blocks, k_blocks]`，通过加载后一次性 `transpose(1,2).contiguous()` 解决（一次性成本，不进热路径）；
4. **数值与图兼容约束**：切换必须满足与 Triton 路径的逐位/近逐位对齐（scale 与 fp8 数同源），并保证 CUDA graph 全图捕获兼容（`m_indptr` 常驻 device、无 D2H 同步；graph 补齐行携带 `topk_ids=-1`，所有 glue kernel 必须显式处理该哨兵值）。

切换落地后（配合上述 glue），MoE GEMM 本体的填充浪费与配置错配问题解决；但 nsys 复测显示**新引入的 glue 链（quant → permute → scale-pack → GEMM1 → silu+quant+pack → GEMM2 → unpermute）成为新的开销集中点**：legacy 三段式输入准备耗时 6.9ms/fwd，GEMM1/GEMM2 之间还存在 gate_up 中间张量 134MB/层 的完整写出-读回往返。

### 4. 引出的后续优化

上述 glue 开销确立了下一阶段的优化主线——**在保持 FlashInfer GEMM 数据契约的前提下，压缩 GEMM 前后的数据搬运**：

- **FUSED_A1**：将 quant + permute-scatter + scale-pack 三个 kernel 融合为一（详见下节），6.90 → 6.02 ms/fwd，实现效率达理论带宽下限的 93%；
- **FUSED_A2**：将 GEMM1 与 GEMM2 之间的 silu+mul + quant + scale-pack 融合为单 kernel，消除中间 bf16 张量的一次往返；
- 更彻底的消除路径（GEMM epilogue 内完成激活/量化/归约，将 glue 从"融合"推进到"消失"）在后续章节展开。

---

## 待写章节（规划）

按总结报告（2026-07-31）的结构展开，每章沿用"基线怎么做 → 新方案怎么做 → 问题 → 引出下一步"的叙事模式：

3. glue 融合：FUSED_A1 / FUSED_A2（含带宽核算方法、93% 效率结论、PDL）
4. 去物化三部曲：qkvzba view / z 零拷贝 / qk_rope gate（+4.15% 单项最大）
5. GDN Triton 扫参（SM120 smem 约束、autotune 污染坑、bitwise 验证纪律）
6. 精度回归攻坚（-1 越界 + PDL 触发点两缺陷，排查方法论）
7. decode 回归归因与调度器修复（O(E) 线性扫描 → 上游 #4130 对照）
8. gated GEMM1 / FINALIZE 融合（epilogue 方向的两步落地）
9. 成果汇总与经验（可直接改写自 2026-07-31 报告第七节）

## 写作风格约定

- 结构：动机（含量化证据）→ 机制描述 → 实现 → 实测结果 → 教训/引出下一步
- 所有结论带数字（μs/ms/％），出处为台账（`2026-07-22-prefill-optimization-status.md`、`2026-07-28-decode-regression-ab-sop.md`）
- 中文正文，代码/kernel 名保留英文
