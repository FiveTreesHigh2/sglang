# Qwen3.5-35B-A3B-FP8 @ RTX PRO 5000 单卡推理优化总结报告

日期：2026-07-31 · 范围：2026-07 全部优化工作 · 状态：已全部落地生产

---

## 一、背景与动机（Motivation）

**业务场景**：Qwen3.5-35B-A3B-FP8（40 层，30 层 GDN 线性注意力 + 10 层全注意力交错，256 routed experts top-8 MoE，hidden 2048，moe_intermediate 512，权重 (128,128) 块量化 FP8）部署于单张 RTX PRO 5000（SM120 Blackwell，110 SM，300W，72GB）。生产流量画像：~100 并发 × 4096 输入，prefill 吞吐主导成本。

**核心问题**：SM120 是"消费级架构做数据中心事"的卡——社区优化（DeepGEMM、FlashInfer GDN、大 smem tile 配置等）普遍面向 SM90/SM100（B200 228KB smem、更多 SM），在 SM120（99KB smem 上限、110 SM）上要么不可用、要么直接负优化。基线（sglang + Triton MoE backend）prefill 仅 ~32.1k tok/s，nsys 显示每 forward（8192 token chunk）GPU 时间 241ms，且无单一大瓶颈——优化必须靠**大量中小型算子级改进的累加**。

**目标演进**：
1. 第一阶段：prefill 吞吐 +10%（达成 +12.02%，后续叠加至 +16.1%）
2. 第二阶段：修复优化引入的 decode 回归，最终目标"decode 与 Triton baseline 持平"（bs=128 达成，bs=1 剩 +0.14ms）

---

## 二、方法论

全程 profiling 驱动的闭环，每项优化走完整链条：

```
nsys 稳态采样（60 fwd，kernel 级预算表）
→ ncu SOL/roofline/停顿分析（定性：带宽墙 / 计算墙 / 调度地板 / launch 地板）
→ 带宽核算（理论下限对照，决定"能优化"还是"只能消除"）
→ 实现（改一个变量）→ bitwise 等价验证 → serving A/B（kernel 实锤 + 端到端）
→ 精度门（GSM8K/MMLU）→ env 开关灰度 → 默认开启 → 台账入账
```

关键纪律：锁频（1732MHz）对比、`/proc/<pid>/environ` 三重配置校验、每步只动一个变量、赢家配置必须过 bitwise 等价、非确定性 bug 用 N 轮重复验收捕捉。

---

## 三、阶段一：Prefill 优化（32.1k → 36.6k tok/s，+12.02%）

### 3.1 kernel 预算表（起点，241ms/fwd）

| 模块 | ms/fwd | 占比 | ncu 定性 |
|---|---|---|---|
| Triton fused_moe GEMM×2 | 69.5 | 28.8% | gemm2 SM 仅 48%（config 错配） |
| dense GEMM（cutlass fp8） | 55.4 | 23.0% | SM 75-84%，已到极限 |
| GDN 链（chunk_h 等 6 kernel） | 40.2 | 16.7% | chunk_h Occupancy 仅 15% |
| 全注意力 | 16.1 | 6.7% | 正常 |
| qkvzba split/cat | 10.7 | 4.4% | 纯搬运，DRAM 91%——只能消除 |
| norm/quant/act 长尾 | ~35 | ~15% | 带宽墙 |

### 3.2 各优化项P

**① MoE backend 切换：Triton → FlashInfer CuTe SM120 grouped GEMM（主项）**
- 动机：Triton gemm2 SM 48%（`_down.json` config 缺失）+ moe_align 的 BLOCK_M 填充浪费 ~10%
- 技术：FlashInfer zero-padding 模式 grouped GEMM（token-packed 激活 + m_indptr CSR 描述，warp-specialized persistent kernel）；配套自研 glue：`moe_permute_prepare`（torch.sort + CSR）、scale 打包 Triton kernel（column-major、per-expert 4 对齐）、B4 权重 scale 一次性预转置
- 算子：MoE GEMM1 (65536×1024×2048)、GEMM2 (65536×2048×512)

**② qkvzba view 捷径 + z 零拷贝（+4.15% 端到端，单项最大）**
- 动机：`fused_qkvzba_split_reshape_cat` 10.7ms/fwd 纯搬运，其中 qkv 部分是输入前缀列的恒等复制
- 技术：**去物化（materialization elimination）**——逐坐标核对读写偏移证明恒等后，用 strided tensor view 替代拷贝；下游 `causal_conv1d` 走支持任意 stride 的 Triton 路径；z 分支利用 `_layer_norm_fwd_1pass` 原生 `stride_z_row` 支持改 grouped 模式原地读
- 算子：GDN 前处理 split/reshape/cat、layer_norm

**③ qk_rope gate 去物化（-0.86ms/fwd）**
- 动机：`_fused_qk_rmsnorm_rope_gate` 每层写 67MB gate 纯为解交织，下游 `fused_sigmoid_mul` 本就支持 3D strided 读
- 技术：kernel 加 `WRITE_GATE` 编译期开关，wrapper 返回 `unflatten` strided view 直传下游
- 教训：先前"融合 kernel 净回归"是归因错误——基线的解交织成本藏在 aten elementwise 里；**带宽核算是归因的最终裁判**

**④ GDN chunk_h Triton 扫参（-2.1ms/fwd）**
- 动机：chunk_h SM 21% / Occupancy 15%，block 工作量不足
- 技术：环境变量参数化 BV/warps/stages + 独立进程扫描（禁 autotune——kernel 原地写 state pool 会被污染）。胜出 BV=64/w4/s2（-16.9%）；s=4 全灭（smem 超 SM120 99KB 上限，B200 经验不可移植）；w8 负优化（dot 切碎）
- 算子：GDN chunk_delta_h、recompute_w_u（第二轮扫参 BK/BV=128/s3）

**⑤ FUSED_A1：permute+quant 融合（-0.9ms/fwd，B13）**
- 动机：legacy 三段式（quant → moe_permute → scale pack）6.9ms
- 技术：单 kernel 融合 quant+scatter+scale-pack（PDL 启动、warp 内 reduce_max 求组 scale）；实现效率 93% 贴理论下限后停止投入
- 算子：MoE A1 输入准备（bf16 → packed fp8 + FI scale 布局）

**⑥ FUSED_A2：silu+mul+quant+pack 融合**（先期已有，本期修缺陷）
- 算子：MoE GEMM1 → GEMM2 之间的激活+量化

**⑦ D1-a：norm 直出 fp8（-2.84ms）、GDN1 conv 融合（-6.70ms）及免费档若干**（B1/B3/B8/B9/B10 等：bf16 router GEMM、column-major quant 等）

### 3.3 精度回归攻坚（两个深坑，本项目最重要的正确性教训）

B13 默认开启后 GSM8K/MMLU 掉点，非确定性。开关二分 + 逐位对拍 + PDL 变体判定，最终定位两缺陷：
1. **-1 路由越界**：CUDA graph 补齐行 topk_ids=-1，fused kernel 以 uint32 读取回绕后越界读 17GB + scale 列任意写。修复：`expert >= num_experts` 守卫 + 前缀 scale 列清零
2. **PDL 触发点位置错误**（主要致错源）：`PDLTriggerSecondary` 位于输出存储之前，下游 GEMM（PDL secondary）读到未提交数据。修复：触发点后移至全部 store 之后

修复零性能代价。流程固化：fused 路径必须含 -1 行逐位对拍、N 轮重复验收、精度评测纳入收尾必经。

---

## 四、阶段二：Decode 回归归因与 FlashInfer 调度器修复

### 4.1 问题

prefill 优化收尾后发现 decode bs=1 TPOT 6.18 → 7.45ms（+20%），bs≥4 后回退迅速收敛——典型的小 M 固定开销形态。

### 4.2 归因（三组单变量 A/B + nsys per-step 分解）

排除法逐项：FUSED_A1 无关（A≈C）、dense backend 无关（基线同用 flashinfer_cutlass）。锚定 MoE GEMM：FI grouped GEMM 每 launch ~24μs 地板价（Triton 13.3μs），80 launch/step → +0.72ms，占回退六成。

**根因**（读 kernel 源码定位）：FlashInfer `MGroupedContiguousWithZeroPadding` 调度器用**单线程串行线性扫描** `token_offset[E+1]` 定位 tile（每组 2 次相互依赖的全局读），且 persistent kernel 内 5 个角色各自构造 Scheduler 重复扫描。E=256 时 O(E)×5 的发现成本与实际工作量无关——decode 8 行数据要扫 256 组。prefill 大 M 摊薄故从未暴露；stage-2 eager microbench 被 launch 开销掩盖，也未暴露。

### 4.3 修复

**smem tile-cumsum 协作预计算 + 二分查找**：block 内 warp 协作（shfl prefix-sum）一次性算出 per-expert tile 前缀和存入动态 SharedStorage（static `__shared__` 会破坏 TMA 128B 对齐——踩坑一次），`get_next_block` 改二分，O(E) → O(log E)。tile 枚举顺序逐位不变（与 Triton 参考 `triton_vs_flashinfer=0.0` 硬证明）。

**结果**：kernel 24 → 5.7-9.4μs（反超 Triton 2.2-3.6×），prefill 哨兵 ±2% 内持平，TPOT 7.45 → 6.51ms（收回 84%）。

---

## 五、阶段三：上游 #4130 合入 + gated GEMM1

### 5.1 上游撞车与合入（档 A）

flashinfer PR #4130（07-29 merge）重写了同一调度器（专职调度 warp + shfl prefix-sum/ballot + smem mbarrier 管道发布 tile，比我们的修复更彻底）并新增 `is_gated` fused SwiGLU。决策：撤回自研上游 PR 计划，cherry-pick #4130 到 pinned tag（97 个中间 commit 依赖核查后确认自洽）。

**收益**：prefill MoE GEMM 再降 4.3-12.4%（调度 warp 卸掉消费者 warp 负担，prefill 也受益）；E2E prefill 37405 tok/s、decode 6.45ms——均优于自研补丁版。

### 5.2 gated GEMM1（档 B）

`is_gated`：SiLU(gate)*up 融进 GEMM1 epilogue（fp32 累加器上做激活），要求 w13 布局 up-first。

- sglang 侧：load 时一次性翻转 w13 权重+scale 半区（翻转与分块量化可交换，逐位验证）；A2 kernel 加 `kInputActivated` 模板参数退化为纯 quant+pack；**契约硬断言**防翻转/env 错配静默出错
- 结果：prefill +0.4%、decode 中性（GSM8K 0.832 / MMLU 0.680 精度门签字后默认开启）
- **关键教训**：收益远低于上游宣称的 +34.3%，因为**我们的 plain 路径 silu 早已融合进 A2**——这次融合想吃的肉大半在先前的 A2 融合里已经吃掉。融合收益评估必须以自己的基线为准

---

## 六、阶段四：MoE FINALIZE 融合（unpermute 消除）

### 6.1 动机

目标重设为"decode 与 Triton baseline 持平"（bs=1 差 +0.30ms、bs=128 差 +0.53ms）。差距全部来自 FI 专属 glue 开销，bs=1 最大单项是 `moe_unpermute`（6.6μs×40 层=0.27ms/step）。LM head GEMV（0.83ms/step，带宽利用率仅 47%）虽是更大绝对项，但 backend 无关——修了 baseline 同样受益，不缩相对差距，降级 backlog。

### 6.2 实现

**GEMM2 epilogue 内完成加权 scatter-add**（自研，flashinfer fork `moe-finalize-fusion`）：
- SwapAB pred-stg epilogue 加 finalize 变体：fp32 累加器直接乘 `row_weights[row]`、以 `atomicAdd` 累加进 `out_finalize[dst2token[row], N]`，免 bf16 往返（比原路径少一次舍入）；-1 补齐行跳过（延续守卫惯例）
- 范围刻意收窄：仅 SwapAB tile 路径（decode 全区间命中），非 SwapAB tile 请求 finalize 硬报错不静默回退；prefill 大 M 自动走原路径（unpermute 在彼处已充分摊薄且原子加高冲突）
- sglang 侧：Triton metadata kernel（per-route token 索引 + topk 权重 × routed_scaling 折入，散射为行索引数组）、runner 按 `routes/experts≤8` 镜像 tile 规则分派、fp32→bf16 cast
- 数值策略：Phase 1 用 fp32 归约缓冲（原子加顺序不确定性在 fp32 下 ~1ulp 级）

### 6.3 结果

decode bs=1 6.48→6.32ms；**bs=128 58.49→57.99ms（意外全额收益：1024 routes/256 experts=4 也命中 SwapAB，高并发下 unpermute 摊薄开销全部收回）**；prefill 37582 持平。

---

## 七、最终成果汇总

| 指标 | 项目基线 | 最终 | 变化 |
|---|---|---|---|
| **prefill @4096（生产主工况）** | 32372 tok/s | **37582 tok/s** | **+16.1%** |
| prefill kernel 时间 | 241.4 ms/fwd | ~215 ms/fwd | -11% |
| decode bs=1 TPOT | 6.18 ms | 6.32 ms | +0.14ms（回退期曾 +1.27） |
| decode bs=128 TPOT | 57.96 ms | 57.99 ms | **持平** |
| bs=128 TTFT | 8112 ms | ~7150 ms | **-12%** |
| 精度 GSM8K / MMLU | 基线 | 0.832 / 0.680 | 持平（签字） |

### 涉及算子全景

| 算子 | 优化手段 |
|---|---|
| MoE grouped GEMM ×2 | backend 切换（Triton→FI CuTe SM120）、调度器 O(E)→O(log E)（后被上游 #4130 取代）、gated SwiGLU epilogue、FINALIZE epilogue |
| MoE glue（permute/quant/scale-pack/unpermute） | FUSED_A1 三合一、A2 四合一→瘦身、FINALIZE 消除 unpermute、scale 预转置 |
| GDN 链（chunk_h/wu/conv/norm） | Triton 扫参（smem 约束下的 BV/stage 权衡）、conv 融合、norm 直出 fp8 |
| qkvzba split/cat、qk_rope gate、z norm | 去物化（strided view 替代拷贝） |
| dense GEMM | ncu 判定已到极限，主动不碰（B14 垫零尝试后 revert） |

### 技术清单

- **Kernel 工程**：CUDA/CuTe（epilogue 变体、warp 协作 scan、原子归约、PDL、mbarrier）、Triton（融合 kernel、扫参、metadata scatter）、TVM FFI binding
- **性能分析**：nsys（graph node 展开、稳态窗口、锚点 kernel 步数归一化）、ncu（SOL/roofline/停顿分析）、带宽核算（理论下限对照）
- **正确性体系**：bitwise 等价测试、三指标容差（mean-abs-rel/symmetric/nRMSE）、CUDA graph capture/replay 一致性、-1 补齐行对拍、GSM8K/MMLU 精度门
- **工程化**：env 开关灰度（默认关→验收→翻默认）、契约硬断言（错配报错优于静默出错）、fork 分支管理与上游对齐、单变量 A/B 纪律

### 沉淀的通用经验

1. **launch/调度地板价只在小 batch + CUDA graph 下显形**——eager microbench 会被 launch 开销掩盖出假象，serving graph 口径才是地面真值
2. **融合的收益 = 你自己基线里还没融合的部分**——上游宣称收益不可直接引用
3. **B200 经验不可直接移植 SM120**（smem 99KB、110 SM 改变一切 tile 权衡）
4. **相对目标（与 baseline 持平）与绝对目标（TPOT 下降）会指向不同的优化选择**——LM head GEMV 案例
5. 不锁频的 kernel 比较会因 boost 行为漂 ±10%——kernel 级对比必须锁频

### 遗留事项

- #7：官方 nightly 发布后切 pinned wheel 正规化（顺带 FINALIZE 回馈上游 PR）
- #10（backlog）：LM head GEMV（绝对 TPOT 收益 ~-0.4ms/step，计划已成形）
- bs=1 剩余 +0.14ms：gated GEMM1 小 M 配对税（18.8 vs 10.0μs）为主，待上游 tile 改进或 epilogue 直出 fp8（后者可让 A2 kernel 整个消失，prefill/decode 双收益）
