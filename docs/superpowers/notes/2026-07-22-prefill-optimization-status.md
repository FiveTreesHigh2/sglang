# Prefill 优化工作状态快照（2026-07-22）

## 目标

Qwen3.5-35B-A3B-FP8 @ RTX PRO 5000（SM120）单卡，prefill 吞吐 +10%。

- 基线：Triton MoE backend，chunk 8192，100 并发 × 4096 输入，~32.1k tok/s（input_throughput）
- 每 forward（8192 tok）GPU 时间 241ms，墙钟 ~254ms

## 已定案的事实（nsys/ncu 实测，数据在 `ServerDownload/batch_nsys_ncu/`）

### serving 稳态 kernel 预算（每 forward 241ms，60 forwards 采样）

| 模块 | ms/fwd | 占比 | ncu SOL 判决 |
|---|---|---|---|
| fused_moe GEMM×2 | 69.5 | 28.8% | gemm1 SM 62% / gemm2 SM **48%**（`_down.json` 缺失，config 错配） |
| dense GEMM (cutlass fp8) | 55.4 | 23.0% | SM 75~84%，**已到极限，不要碰** |
| GDN 链（chunk_h/o/conv1d/recompute/kkt/l2norm） | 40.2 | 16.7% | chunk_h SM 21% / DRAM 69% / **Occ 15%**，可调 |
| full attn (BatchPrefill) | 16.1 | 6.7% | 占比比 InferSim 估计小得多 |
| qkvzba_split_reshape_cat | 10.7 | 4.4% | 纯搬运，DRAM ~91%，只能消除不能优化 |
| moe_sum / norm / quant / act 等 | ~35 | ~15% | norm 类已在带宽墙 |

### 关键结论

1. GPU busy ≥95%，**无调度气泡**；之前"墙钟 vs InferSim 差距"是 InferSim 低估了真实 kernel 工作量
2. serving 真实路由下 fused_moe 平均 855μs，仅比 uniform microbench 慢 7%（skew 折扣小）
3. FI MoE backend 已入账 +1.3~1.7% 端到端，**冻结进一步投入**（A1 融合归因实验证明 glue 收益在 prefill CUDA graph 下≈0）
4. chunk 8192→16384 对 FI 边际收益 <1%（m_pe 256 已过 roofline 拐点），**保持 8192 + mem-frac 0.9**

## 行动清单

### ① GDN chunk_h 参数扫描（✅ 已关账：**bv64_w4_s2** 入账 ~+0.9%）

- 脚本：`/home/logs/sennian/gdn_scan.sh`，输出 `/home/logs/sennian/nsys-log/gdn-scan/`（日志平铺，`<combo>.log`）
- 扫描结果（chunk_h avg，60 launches）：16-4-2 421.2；32-4-2 基线 413.6；32-4-3 373.0；**64-4-2 343.7（-16.9%，胜出）**；64-4-3 404.1（smem ~99KB 挤死占用率，反弹）
- **s=4 三组全部 `OutOfResources: shared memory` 崩溃**：bv32_w4_s4 需 115,468B、bv16 系需 107,276B，SM120 上限 101,376B（99KB）。PR #26206 的 s4 收益依赖 B200 228KB smem，**SM120 物理封死，勿再尝试**
- 规律：BV 越大越快（block 工作量不足是真瓶颈），但**大 BV 与深 stage 互斥**（smem 相乘）；BV=128 死刑：V=128 时 grid 第 0 维=1，总 block 数 < SM 数，且累加器寄存器必然 spill；**w8 实测 480.5μs（+40%，归因 dot 切碎+寄存器劣化），warps 旋钮封死在 4**
- 环境变量入口：`SGLANG_GDN_CHUNK_H_BV / _NUM_WARPS / _NUM_STAGES`（`sglang/kernels/ops/attention/fla/chunk_delta_h.py` L24-26），锁定 `BV=64 WARPS=4 STAGES=2`
- 纪律：每组合独立进程；**禁止改成多 config autotune**（kernel 原地写 state pool，autotune 会静默污染）
- 折算收益：-69.9μs × 30 层 ≈ -2.1ms/fwd（241ms 基数）≈ **+0.9% 端到端**
- 数值验证：✅ 已过（`test/manual/test_gdn_chunk_h_bv_equivalence.py`，双子进程 BV=32 vs 64，o/h/final_state 三项 bitwise 全等；脚本已推 `origin/feat/flashinfer-sm120-fp8-moe`，含 venv 导入路径 fallback）
- serving A/B：✅ 端到端不显著但不回退；kernel 实锤：稳态 chunk_h 平均 **353.9μs**（2319 次/15s，中位 357.7）vs 基线 413.6 → **-14.4%**（真实路由折扣后），折算 GPU 时间 -0.92%，与预估吻合
- 坑录：nsys session 随目标进程退出而消失（端口被旧 server 占用时新 server 秒崩 → sessions list 为空）；环境变量用 `/proc/<pid>/environ` 验证确实进了 server 进程
- 落地：生产启动命令需永久带上 `SGLANG_GDN_CHUNK_H_BV=64 SGLANG_GDN_CHUNK_H_NUM_WARPS=4 SGLANG_GDN_CHUNK_H_NUM_STAGES=2`
- 附带：kkt_solve autotune 候选（`fla/chunk_fwd.py` L32-34）从 [1,2,4] 扩到 [4,8]（kkt 各组均 ~90μs，未受 chunk_h 变量影响）

### ② qkvzba view 捷径（✅ 已关账：全栈端到端 +4.15%）

- 现状：`fused_qkvzba_split_reshape_cat_contiguous`（352μs×30层=10.7ms/fwd）的 qkv 部分是输入前 qkv_dim 列的**恒等复制**（已逐坐标核对 kernel 读写偏移）
- 实现（commit `dd3a35d3b`，已推 origin）：
  - `qwen3_5.py` forward：仅 **prefill（is_extend 且非 target_verify）**走 view（`qkvz[:, :qkv_dim]` + z unflatten + b/a 小拷贝）；decode/verify 保留融合 kernel（packed_decode 要 contiguous；verify 路径有 `.view()` 会炸；decode CUDA graph 下 glue 收益≈ 0）
  - 地雷修复 `causal_conv1d_triton.py`：`empty_like` 对非 dense strided view 会退化 row-major，改为显式分配 channel-last dense 输出
  - 注意：is_extend() 包含 TARGET_VERIFY，必须显式排除
- 证据链补充：CUDA 路径 `causal_conv1d_fn` wrapper（`srt/layers/attention/mamba/causal_conv1d.py` L83）对非 contiguous + seq_lens_cpu 自动路由到 Triton 版（接受任意 stride），不会默默 .contiguous()
- 验证脚本：`test/manual/test_gdn_qkvzba_view_equivalence.py` ✅ 9/9 bitwise 全过；已追加第 10 项：grouped norm vs 逐行 norm（待服务器重跑）
- **serving 全量 kernel diff（60 fwd 口径，241.14→238.04ms）揭示三个事实**：
  1. qkvzba -10.72ms ✅、chunk_h -1.13 ✅、conv1d -0.23 ✅
  2. **z 隐藏拷贝 +4.4ms/fwd**：`z.reshape(-1, 128)` 对 strided view 静默降级为 contiguous 拷贝（aten elementwise 5400 次 = z/b/a×30层×60fwd，z 单次 ~140μs）——view 实际只免了 qkv 那 2/3
  3. `_fused_qk_rmsnorm_rope_gate` 净回归 +1.6ms（替代 RMSNorm 0.77+mrope 0.18 却花 2.54），**分支独立问题，待立案**；其余 dense/moe 普涨为窗口切边伪差（次数 +1.7%，归一后持平）
- **z 零拷贝修复（commit `2745f9066`）**：`_layer_norm_fwd_1pass_kernel` 本就支持运行时 `stride_z_row` + group 模式；forward 改为检测 strided z 时走 `[T, H*D]` grouped 路径（group_size=128、weight.repeat(H)、z 原地读），kernel 零改动；contiguous z / CPU / NPU / DP-Attn pad 不匹配时回退原路径
- **z 零拷贝修复后复测（view-ab-fusedqkv，FI MoE 配置）：per-fwd 231.22ms，较基线 -4.2%**：
  - z 拷贝消失：direct_copy 309→54.4ms（avg 11.9→2.1μs），**-4.2ms/fwd** ✅
  - 代价：gated norm strided 读 z，avg 133.6→165.0μs → +0.9ms/fwd（预期内，暂接受）；z 修复净赚 -3.3ms
  - 238.04→231.22 的剩余 ~3.5ms = FI MoE vs Triton（+1.5%），与 FI 入账 +1.3~1.7% 吻合，账本自洽
  - 哨兵全绿：qkvzba 仅 decode 残留；conv1d 238.1μs；chunk_h 350.3μs
- 端到端关账（300条×3次×并发100，取中位数）：新配置 **34013.90** tok/s（34328.87/34013.90/33859.29）vs 基线 **32659.71**（32855.59/32659.71/32580.82）→ **+4.15%**，几乎打满 kernel 层 -4.2% 的理论上限 +4.4%，prefill 纯 GPU-bound 实锤；验证脚本 10 项断言全 PASS
- 待做（已转化为行动 ⑤）：`_fused_qk_rmsnorm_rope_gate` 立案完成，见下
- 预估：+3.3% 端到端；上游未实现（已查 #18590/#20074/#21019/#26206），做完可回馈

### ③ Triton `_down.json` tuning（⏸️ 搁置：用户决定暂不做；gemm2 SM 48% 实锤，预估 +2%）

- 用 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- 前置：dump topk_ids（qwen3_5.py 临时补丁存 `/home/logs/sennian/topk-dump/`，40层×3份）；改脚本硬编码 num_layers 61→40、dense_layers 3→0
- 产物拷到 venv 的 `configs/triton_3_6_0/`，验证 server 日志不再报 `down_moe=False ... sub-optimal`
- 注意：若最终切 FI backend 则此项作废；tuning 后需重跑 Triton(tuned) vs FI 裁决 FI 去留

### ⑤ qk_rope gate 去物化（✅ 已关账）

- **立案修正**：带宽核算证明融合 kernel 并非烂 kernel——每层搬 285MB（读 q_gate 134 + k 8.4，写 q 67 + gate 67 + k 8.4），245μs 已贴带宽墙；之前"+1.6ms 净回归"是归因错误（基线的 gate 解交织藏在 aten elementwise 里 ~1.3ms，真实净差仅 ~0.3ms）
- **真机会：gate 物化多余**（与②同构）：kernel 读 67MB gate 再写 67MB gate_out 纯为解交织；而 `fused_sigmoid_mul` 早已支持 3D strided gate（native 路径 L1018 注释实证）
- 实现（commit `33b4ddd70`）：kernel 加 `WRITE_GATE` 开关 + wrapper `materialize_gate=False` 返回 `q_gate.unflatten(-1,(16,2,256))[:,:,1,:]` strided view；调用点去掉对 gate 的 `.view(seq_len,-1)`（strided view 会抛错）直传 3D view
- 预估：每层 -134MB ≈ -100μs × 10 层 ≈ **-1ms/fwd ≈ +0.45%**
- 验证脚本：`test/manual/test_qk_rope_gate_view_equivalence.py` ✅ 4 项 bitwise 全 PASS（服务器实跑）
- serving 实锤（view-ab-gate）：qk_rope avg **249.9→164.9μs（-34%，-0.86ms/fwd）**；sigmoid_mul 112.8→115.5μs（strided 读代价 +2.7μs，接受）；其余哨兵全平；per-fwd 231.22→**228.41ms**（其中 qk_rope 实锤 -0.86，余为窗口波动不入账）
- 累计：基线 241.4 → 228.4ms/fwd，**GPU 时间 -5.4%**

### ⑥ 待查

- fork 是否已含上游 #20283（GDN state layout [N,HV,K,V]→[N,HV,V,K]）
- FlashInfer GDN kernel（#18361 等）限 SM90/SM100，SM120 不可用

## 冲刺 10% 第二轮（基于 228.4ms/fwd 全量分解，目标 ~217ms，还差 ~11ms）

当前预算大头：cutlass fp8 GEMM 118.4ms（51.8%，含 dense ~56 + FI MoE grouped ~62）；attn 16.1；GDN Triton 家族 ~28.5；deepep permute/reorder 14.4；quant 3.9。

| 行动 | 预估 | 状态 |
|---|---|---|
| A：GDN 第二轮扫参 | -0.3ms（实测，远低于预估） | ✅ 完成；仅采纳 `SGLANG_GDN_WU_BK=128 _BV=128 _NUM_STAGES=3`（recompute_w_u 240→231μs），chunk_o/l2norm/kkt 无有效改进；附：首轮扫描因服务器代码未更新而无效，扫描前必须验证旋钮已生效 |
| B：qkv split 视图化 | 净 -0.65ms（低于显著性阈值） | ✅ 已关闭：保留默认开启，台账计 0，详见「行动 B 详细记录」 |
| C：permute+quant 融合 | -0.9ms（实测） | ✅ 完成：启用既有开关 `SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=1`（配套测试完备）；legacy 三段式 6.90ms → fused quant_scatter 6.02ms；per-fwd 228.41→**226.72ms**；实测远低于 -4ms 预估的原因：两方案均由 134MB packed fp8 写（grouped GEMM 输入契约所要求）主导；带宽核算：融合 kernel 148.3μs vs 理论下限 138μs（实现效率 93%），**此融合形态已达理论上限，不再投入** |
| D：FI GEMM2 tile 调优 | -2ms（下调，62→70% 估算） | 🔨 进行中。瓶颈定性完成（两份 ncu 报告：`ServerDownload/view-ab-gate/moe_glue_and_gemm.ncu-rep` roofline 集 + `ServerDownload/GEMM2/ncu_fi_gemm2_stall.ncu-rep` 停顿集）：GEMM2（N=2048,K=512）tensor pipe 62.1%，相对 GEMM1（81.2%）的差距由 **long_scoreboard 0.83→1.33（+60%）与 barrier 0.12→0.32（+167%）** 主导——K=512 仅 4 个主循环迭代，流水线深度不足以隐藏 TMA 延迟。两 GEMM 同为 warp-specialized persistent（grid 110、block 384、regs 168、smem 77.8KB，占用被 regs 64.5K/65.5K 与 smem 双重限制在 1 block/SM，不可提升）。GEMM2 同时处于计算/带宽联合受限点（流量 ~570MB/次：权重 268MB+输出 C bf16 268MB+A 34MB，DRAM 58%）。调优方向排序：① BK 128→64 加深流水线（针对 long_scoreboard）② 增大 N-tile 撑薄屏障/epilogue（针对 barrier）。待办：核查 FlashInfer tile 配置控制面（可枚举参数 vs 编译期常量） |
| D-ext：gather-aware grouped GEMM + epilogue scatter-back | 理论 -12ms | 📝 已论证：突破 A1/post_reorder 上限的唯一路径——GEMM1 按 gather_idx 逐行 1D TMA 直接从 X_q 取 A 操作数（消除 134MB packed 写往返），GEMM2 epilogue 内完成 scatter-back+加权求和；只读共享无竞争，X_q 33.6MB 可驻留 L2；难点在 CuTe mainloop 流水线重构与 A-scale 间接寻址；建议先做仅改 A 加载路径的原型 |
| E：prefill piecewise CUDA graph | 0（前提失效） | ✅ 已关闭：get_server_info 确认 prefill 默认即运行于 **breakable CUDA graph（BCG，58 个 token 档位至 8192）**，decode 为 full；"prefill 无 CUDA graph" 的早期记录有误，launch 开销无回收空间。剩余可选项 tc_piecewise+inductor（图级融合，预期 <1%，需显式指定 backend 绕过多模态拦截规则）投入产出比不佳，不投入 |

### 外部审计（2026-07-26）采纳结论与后续路线

审计报告：`FP8-kernel-optimize/sglang_audit_2026-07-26.md`（抽验 10 项硬主张 9 项属实；chunk_o/sort/conv1d 估算与实测精确吻合）。关键裁决：

1. **T1（Triton `--enable-fused-moe-sum-all-reduce`，估 -9.5ms）：实测证伪**——用户真机回测性能不升反降（bf16 atomic_add 每输出行 8 次原子加，未过审计自标的 L2 驻留生死线）
2. **T2（补 SM120 Triton config）：小幅提升但仅 ~33000 tps**，仍低于 FI 路线的 34500~34850 → **FI 路线维持**，审计"分叉裁决不公平"的方法论批评成立但重赛结果不变
3. **行动 D 关闭理由修正**：flashinfer 自研 kernel 确有 `static_assert(kTileK == kGranK)`（sf_fp8_tma_load.cuh:50-51），BK=64 需改 SFConfig+promotion 逻辑（M/L 工作量）而非扫参；叠加 store-heavy 定性，D 被删字节路线支配，维持关闭
4. **台账补正**（审计指出的盲区，已用 ab CSV 实测确认）：chunk_o **8.60ms/fwd**、deepep_post_reorder **9.08ms/fwd**、RadixSort **1.19ms/fwd**；memset 不入 kern_sum（需用 `nsys stats -r cuda_gpu_mem_time_sum` 单独核）
5. **免费档第一批已验证（commit `a7e732638`，audit-B-free 采集）**：B1 chunk_o `zeros_like→empty_like`；B10 视图分支删 a/b `.contiguous()`；B9 缓存 `norm.weight.repeat`；B13 FUSED_A1 默认翻 True。实测：per-forward **223.85ms**（项目新低）；FillFunctor 113.8→83 次/fwd（-1.11ms，B1）、D2D memcpy 165→89 次/fwd（-0.77ms，B10）。口径修正：B1 清零实为 FillFunctor kernel（在 kern_sum 内混于 aten 条目）而非 cudaMemset；真正的 kern_sum 盲区是 D2D memcpy
6. **免费档第二批已验证（commits `64f3c998e`+`8454618f8`，audit-B3B4 采集）**：B3 dense 激活量化改 `column_major_scales=True` 直出 MN-major scale（零拷贝视图）；B4 常量 weight_scale 加载期预转置（`weight_scale_inv_fi`）；B14 decode m%4 显式垫零。实测：**direct_copy 335→13 次/fwd（-322，与预期 -324 吻合）**；总量 223.98ms 持平（-0.67ms 结构性收益低于 ±1ms 窗口离散度，以次数证据入账）；另收 BCG 每图约 -160 拷贝节点。**重要教训**：`sglang_per_token_group_quant_fp8_row_padded` 的直调 AOT 分支与包装分支存在 ±1 码位舍入差（0.096%，scale 逐位相同），跨路径复用现成函数时逐位基线必须以目标路径现行实现为准；`column_major_scales=True`（经包装层）已验证与 row-major 逐位一致
7. **免费档第三批已判定（commits `b5b1c9b6c`~`1b9b216f1`，audit-B8M5 采集）**：
   - **B8 ✅ 生效**：chunk_fwd `A=zeros→empty` + recompute_w_u 内因果 mask；memset 18895→1980 次/窗口（**-0.55ms/fwd**，28.5μs 条目消失），w_u kernel 无回归（6.93→6.71ms）；NaN 污染法等价测试全 PASS
   - **M5 ❌ 判定失败已回退**：naive 全局原子 counting sort 净回归 +0.63ms/fwd（新路径 1.90 vs 旧 1.27ms；histogram 单次 23.6μs，64k 次 atomicAdd 在 257 槽热点上串行化）；审计 -1.2ms 定价证伪；`SGLANG_MOE_PERMUTE_COUNTING_SORT` 默认改 0，代码保留。附开发中发现并修复的别名缺陷：`.to(int32).contiguous()` 对已满足条件的张量是 no-op 返回别名，scatter 原子加原地污染了 expert_offsets（C1 断言以 new[e]==ref[e+1] 错位模式暴露）；条件性拷贝需用 `copy=True` 强制
8. **D1-a ✅ 已验证入账（commits `22a70b54a`~`263b77cd1`，audit-D1a 采集，可归因 -2.84ms/fwd）**：GDN gated norm 直出 fp8 喂 out_proj。实现：
   - `layernorm_gated.py` 新增 `rms_norm_gated_fp8_quant`（专用 Triton kernel，不触碰共享 norm kernel）：norm 数学逐行复刻 `_layer_norm_fwd_1pass_kernel`，结果先舍入 bf16 再量化；量化 epilogue 复刻 sgl-kernel v2 kernel，含两项指令级对齐：**除法用 inline PTX `div.full.f32`**（生产 v2 kernel 以 `--use_fast_math` 构建，nvcc 将 `MAX/amax` 降为近似除法；扫描实测 div_rn 差 24937/33.5M、div.full/div.approx 均 0），转换用 inline PTX `cvt.rn.satfinite.e4m3x2.f32`；直出 (k//128, m) contiguous MN-major A scale
   - `fp8_utils.py` 解除 `assert input_scale is None`：cutlass 分支接受预量化 (fp8 codes, MN-major scale)，out_dtype 固定 bf16，m%4 垫零分支同样覆盖；trtllm 分支显式 assert 拒绝
   - `qwen3_5.py` strided-z prefill 分支接入，开关 `SGLANG_GDN_NORM_FP8_OUT`（默认 1）+ 一次性能力探测；decode 与非 strided 分支不受影响
   - **等价性标准裁定（用户选定方案 A，对本项破例）**：量化段对 CUDA v2 kernel 严格逐位（隔离验证 0/33.5M）；norm 段存在不可消除的重结合偏差——PTX 实证两 kernel 的 `tl.sum(xbar²)` 归约结构不同（参考：双链 ×8，14 FMA；融合：单链 ×16，15 FMA），由 epilogue 新增锚点（fp8 单字节存储、tl.max 第二归约）触发的 Triton 布局决策改变所致，源码层无控制手段；实测偏差 11/33.5M 码字（全部 ±1 码位）+ 1/262144 scale（1 bf16 步），与生产 norm kernel 自身的 M 依赖归约序波动（calc_rows_per_block）同类。测试改为定量断言：码字偏差率 ≤2e-6 且全部 ±1 码位；scale 偏差率 ≤2e-5 且相对偏差 ≤1%；wrapper 预量化管路对逐位一致输入严格逐位（含 m%4 垫零分支）
   - 排查过程留档：三方对照（CUDA kernel / 独立 Triton 量化段 / torch fp32 仲裁）定位分歧段；`TRITON_CACHE_DIR` 落盘 PTX 做指令直方图（版本无关，JITFunction 缓存属性 API 不可靠）；中途两次误判（div_rn、硬件 cvt 平局行为）均由扫描数据纠正；另修复 wrapper 尾部 `output.to(input_2d.dtype)` 在预量化路径下错误回转 fp8 的缺陷（`263b77cd1`，由测试 dtype 断言拦截）
   - **serving 验证（audit-D1a，65.1 forwards）**：norm 调用 30.5→0.5 次/fwd、新 kernel 30 次/fwd（130.6μs vs 旧 168.7μs，169MB/130.6μs≈1.29TB/s 达带宽上限）、quant 162.5→132.4 次/fwd（-30 精确吻合）；可归因 norm -1.14 + quant -1.26 + FillFunctor -0.43 = **-2.84ms/fwd**；总量 223.98→**222.02ms（项目新低）**，差额为未触及类目 +0.9ms 同向漂移（已知窗口离散带内）。注意事项：nsys launch 必须含 `--cuda-graph-trace=node`，否则 BCG 图内 kernel（全部 GEMM/norm/quant/MoE）不展开，首次采集因此作废重采
9. **GDN1 ✅ 已验证入账（commits `ed7d3482f`~`24a94d86e`，audit-gdn1 采集，可归因 -6.70ms/fwd）**：conv1d epilogue 融合 qkv 拆分与 qk l2norm。实现：
   - 背景：conv 输出是纯中转张量（写 134MB/层后被 l2norm×2 + extract_v 全量回读，三者实测 3.05+3.52 ms/fwd）；BLOCK_N=256 与 head 128 及 q/k/v 边界（2048/4096）对齐，epilogue 可在寄存器内完成归一化并直写最终 dense q/k/v
   - `causal_conv1d_triton.py`：`_causal_conv1d_fwd_kernel` 增加 constexpr 门控 `SPLIT_QKV`/`QK_L2NORM` epilogue（默认关闭、既有调用方死代码消除；conv 数学逐元素串行，无归约结合序暴露面）；新增 wrapper `causal_conv1d_fn_qkv_split`；conv_state 更新逻辑零改动（kernel 只从输入 x 读状态）
   - `gdn_backend.py`：开关 `SGLANG_GDN_CONV_FUSION={off,v,full}`（默认 full）；v 档仅重定向 v（严格逐位，q/k 走外部 dense l2norm），full 档 l2norm 入 epilogue（有界重结合偏差，D1-a 同标准）；decode/target_verify 不受影响
   - 测试 `test/manual/test_gdn_conv_qkv_fusion_equivalence.py`：v 档全链路严格逐位（含 conv_states 池）；full 档 q/k 定量断言（偏差率 ≤1e-5，实测 3.22e-6，高于 D1-a 的 3.3e-7 系因 epilogue [2,128] 归约形态差异更大；±1 bf16 ulp 硬约束不变）；varlen 覆盖多 chunk、尾块<BLOCK_M、seqlen<state_len、混合 initial_state；全部 PASS
   - conv launch 参数旋钮化（commit `c1be0dc1c`）：`SGLANG_GDN_CONV_BLOCK_M/_BLOCK_N/_NUM_WARPS/_NUM_STAGES`（默认 8/256/4/2 上游原值）；autotune 因 conv_state 非幂等原地更新禁用（chunk_h 同类）；BLOCK_N 兼任对齐契约，改动需重跑等价性测试
   - **serving 验证（audit-gdn1，66.9 forwards）**：l2norm_strided 60→0（-3.05ms）、extract_columns 30→0（-3.52ms）、conv 238→234.9μs（-0.13ms，epilogue 无代价）；可归因 **-6.70ms/fwd**（预估 -6）；总量 222.02→**216.41ms**，差额为未触及类目 +1.1ms 同向漂移
10. 待排期：D-ext 上半段 gather-A（-6.0ms，XL）；conv launch 参数扫描（上限 ≈-0.9ms，conv 现 1.14TB/s vs 1.29 上限）——目标已达成，两项仅在需要进一步余量时启动

✅ **精度回归已修复并验收（commits `e7e975519` + `e655da92e`）**：
- **缺陷 1（-1 补齐路由越界）**：fused A1/A2 kernel 以 `uint32_t` 读 topk_ids，CUDA graph 补齐行的 -1 回绕后 `m_indptr` 越界读 ~17GB、scale 列任意写；离线 16 补齐行复现 illegal access。修复：`expert >= num_experts` 守卫 + expert 0 前缀 scale 列清零；legacy Triton pack-scale 补显式 mask
- **缺陷 2（PDL 触发点位置错误，主要致错源）**：FI SM120 grouped GEMM 以 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动（PDL secondary），其 wait 在 primary 全部 block 触发后即返回；fused kernel 的 `PDLTriggerSecondary` 位于输出存储之前 → GEMM1 读到未提交的 packed/scale。修复：两 kernel 触发点后移至全部存储之后（规范形态参照上游 per_token_group_quant_8bit_v2.cu）。缺陷代码源自 `41ae2a2ec`/`b08848d54`（07-21/22，当时 A1 默认关闭、A2 窗口极小无症状，B13 翻转默认值后暴露）
- **验收**：端到端对拍全净；GSM8K/MMLU 恢复至基线水平（用户确认）；修复后吞吐三轮 36963.98/36592.05/36515.20，无性能代价。排查链全程：开关二分 R0-R5 → -1 对拍（缺陷 1）→ E1/E2/E3 图变量分离 → 端到端离线复现（非确定性 mismatch+NaN）→ S1/S2/S3 PDL 变体判定（缺陷 2）。流程改进已固化：fused 路径独立逐位对拍（含 -1 行）、N 轮重复验收捕捉非确定性、精度评测纳入收尾必经步骤

当前累计（最终入账）：per-forward 241.4 → **216.41ms（-10.35%，audit-gdn1）**；端到端修复后三轮中位数 **36592.05 vs 基线 32664.57 tok/s = +12.02%**（三轮全部高于 +10% 线）；精度 GSM8K/MMLU 与基线持平——**10% 目标以吞吐+精度双口径达成，项目收尾**。生产环境变量清单（已验证）：`SGLANG_GDN_CHUNK_H_BV=64 _NUM_WARPS=4 _NUM_STAGES=2, SGLANG_GDN_WU_BK=128 _BV=128 _NUM_STAGES=3`（FUSED_A1 自 `a7e732638` 起默认开启）；`SGLANG_GDN_QKV_VIEW=1`、`SGLANG_GDN_NORM_FP8_OUT=1`、`SGLANG_GDN_CONV_FUSION=full` 默认保留；`SGLANG_MOE_PERMUTE_COUNTING_SORT` 默认关闭。

旋钮清单（默认值=现状）：
- chunk_o：`SGLANG_GDN_CHUNK_O_BK/_BV/_NUM_WARPS/_NUM_STAGES`（128/64/4/2）
- recompute_w_u：`SGLANG_GDN_WU_BK/_BV/_NUM_WARPS/_NUM_STAGES`（64/64/4/3；生产采纳 128/128/4/3）
- l2norm：`SGLANG_GDN_L2NORM_BT/_NUM_WARPS/_NUM_STAGES`（16/8/3）
- v 提取：`SGLANG_GDN_EXTRACT_V_NUM_WARPS`（8）
- qkv 视图开关：`SGLANG_GDN_QKV_VIEW`（默认 1；行动 B 的 A/B 控制变量）
- kkt：autotune 候选已回退至 w[1,2,4]（w8 扩展无收益且引发选择不稳定，观察到 90→101μs 回归后回退）
- 扫描纪律同①：每组合独立进程；赢家配置必须过 bitwise 等价再进 serving A/B

### 行动 B 详细记录（qkv split 视图化，✅ 已关闭：保留默认开启，台账计 0）

提交链（均已推 origin）：
- `6a3bca47c`：`l2norm_fwd_packed`（strided 读 packed conv 输出）+ `qk_l2norm_applied` 参数（关闭 in-kernel 归一化，避开 `input_guard` 隐式物化）；v 物化初版用 aten 拷贝
- `3d002f759`：v 物化改为专用 `_extract_columns_kernel`（aten 拷贝仅 0.94TB/s → 专用 kernel 117.8μs ≈ 1.14TB/s）
- `044b50013`：v 提取 warps 参数化（规范一致性），删除无作用的 num_stages
- `df4c85961`：kkt 候选回退 + `SGLANG_GDN_QKV_VIEW` 开关（A/B 控制）

验证与测量历程：
1. 等价性：`test/manual/test_gdn_qkv_view_l2norm_equivalence.py` 5 项 bitwise 全 PASS
2. qkv-view 采集：发现 aten v 拷贝效率不足（143μs），修复后重测
3. qkv-view2 采集：可归因净收支 -2.50ms（移除 split 6.40+dense l2norm 2.75，新增 extract_v 3.53+strided l2norm 3.12）；但未触及 kernel 漂移 +4.1ms（cutlass +2.5、attn +0.6、kkt +0.37）
4. **交替 A/B 第一轮（ab-off1/on1/off2/on2）：判定无效**——per-kernel 差分显示四轮（含 off 轮）均存在 `_extract_columns_kernel`/`l2norm_fwd_kernel_strided` 且无 `fused_qkv_split`，即 **`SGLANG_GDN_QKV_VIEW=0` 未生效，四轮全部运行视图路径**；根因待确认（最可能：执行时仓库未拉取含开关的 `df4c85961`）。参数未生效导致整批数据无效的故障模式第二次发生，A/B 脚本已追加运行前 /proc environ 校验（不匹配则 FATAL 退出）
   - 残余价值：四轮同配置重复量化了噪声带——per-forward ±0.25%（226.38~226.95）、吞吐 ±0.5%（34542~34854），后续 A/B 以此为显著性阈值；视图路径四轮均值 226.67ms 与 a1-fused（非视图，226.72）基本相等，支持中性假设但尚非受控对比
5. **交替 A/B 第二轮（仓库更新至含开关版本后重跑，配置校验通过）：最终判定**
   - off 均值 228.43 → on 均值 227.78，净 **-0.65ms（约 +0.3%）**；可归因收支 -2.56ms（与 qkv-view2 批次 -2.50 一致，两批独立复现），被非关联 kernel 同向波动 +1.80ms（cutlass +1.05、kkt +0.39、attn +0.23）部分掩盖
   - 成对比较不一致（off1/on1 差 -1.34，off2/on2 差 +0.04），本批同配置内部离散度 ±1ms 高于上批的 ±0.3ms 噪声带
   - 旁证：kkt 在候选回退后仍存在 89↔102μs 的跨进程选择波动，为独立测量噪声源，与 B 无关
   - **处置：保留 `SGLANG_GDN_QKV_VIEW=1` 默认开启**（等价性验证通过、可归因收支为正、净效应非负），收益台账计 0，不再投入

## 收益汇总（目标 10%，**已实测入账 +4.15%**）

| 措施 | 预估 | 状态 |
|---|---|---|
| FI MoE + GDN 调参 + view/z 零拷贝（全栈） | — | ✅ **端到端 +4.15% 已入账**（34013.9 vs 32659.7，300×3 中位数） |
| qk_rope gate 去物化 | +0.37%（实锤） | ✅ 已关账（验证全 PASS，249.9→164.9μs），per-fwd 累计 228.4ms（-5.4%） |
| `_down.json` tuning | +2%（若裁决回 Triton） | ⏸️ 搁置（用户决定暂不做） |
| dense GEMM 调优 / 调度气泡 | 0 | ❌ 已排除 |

## 环境备忘

- server：`33.243.206.227:30000`，feature 仓库 `/home/logs/sennian/pro5000-fi-moe/sglang`
- **两套 Python 环境，别用错**：`/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3` = feature 仓库 editable（仓库改动立即生效，**验证脚本和 A/B server 都用它**）；`/home/logs/sennian/py-venv/sglang5.14/bin/python` = pip 安装版（仓库改动不生效，仅环境变量类实验可用）
- ncu 需 sudo（宿主机 RmProfilingAdminOnly=1）；nsys launch 建会话用 `--session-new`；`--sample` 只能给 start/profile
- serving 采集流程：见下方《serving 稳态采集 SOP》
- 本机分析：优先服务器端 `nsys stats` 出聚合 CSV；本机只做 CSV 对比，弃用 DumpTimeline 大 JSON 解析

## 附录：serving 稳态采集 SOP（命令可直接复制，换采集时只改 SESSION 名）

以下以 `view-ab2` 为例，每次新采集把 `view-ab2` 全局替换成新名字。

### 第 0 步：清场

```bash
# 确认 30000 端口没被旧 server 占用（占着就先停掉，否则新 server 秒崩、nsys session 随之消失）
ss -ltnp | grep 30000
ps aux | grep sglang.launch_server | grep -v grep
# 确认无残留 nsys 会话
nsys sessions list
```

### 第 1 步：nsys launch 起 server（feature 仓库 .venv + 三个 GDN 环境变量）

```bash
mkdir -p /home/logs/sennian/nsys-log/view-ab2
cd /home/logs/sennian/nsys-log/view-ab2

nsys launch --session-new=view-ab2 \
  --env-var=SGLANG_GDN_CHUNK_H_BV=64,SGLANG_GDN_CHUNK_H_NUM_WARPS=4,SGLANG_GDN_CHUNK_H_NUM_STAGES=2 \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.launch_server \
    --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
    --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
    --host 33.243.206.227 --port 30000 \
    --enable-metrics --tp-size=1 \
    --reasoning-parser=qwen3 --collect-tokens-histogram --tool-call-parser=qwen3_coder \
    --disable-radix-cache --mem-fraction-static 0.9 \
    --fp8-gemm-backend flashinfer_cutlass \
    --enable-layerwise-nvtx-marker \
    > /home/logs/sennian/nsys-log/view-ab2/server.log 2>&1 &
```

注：如果本次要测 FI MoE 配置，在上面追加对应 MoE backend 参数；不加则默认 Triton MoE（与 241.4ms 基线同口径）。

### 第 2 步：验证环境变量真的进了 server 进程

```bash
tr '\0' '\n' < /proc/$(pgrep -f sglang.launch_server | head -1)/environ | grep GDN
# 必须看到三行：BV=64 / NUM_WARPS=4 / NUM_STAGES=2，缺一不可
```

### 第 3 步：等 server 就绪（模型加载约 1-2 分钟，反复执行直到 200）

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://33.243.206.227:30000/health
```

### 第 4 步：后台打 benchmark 进稳态（300 条，保证采集窗口内不断流）

```bash
/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.bench_serving \
  --backend sglang \
  --host 33.243.206.227 --port 30000 \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 1 --random-range-ratio 1.0 \
  --num-prompts 300 --max-concurrency 100 \
  > /home/logs/sennian/nsys-log/view-ab2/bench.log 2>&1 &
```

### 第 5 步：稳态窗口采 15 秒

```bash
sleep 30   # 等 benchmark 进稳态
nsys start --session=view-ab2
sleep 15
nsys stop --session=view-ab2
ls -lt /home/logs/sennian/nsys-log/view-ab2/   # 确认生成 report1.nsys-rep
```

### 第 6 步：导出聚合 CSV 并判读

```bash
nsys stats -r cuda_gpu_kern_sum /home/logs/sennian/nsys-log/view-ab2/report1.nsys-rep --format csv \
  > /home/logs/sennian/nsys-log/view-ab2/kern_sum.csv

# 指标 1：per-forward GPU 总量（总 kernel 时长 ÷ 60 个 forward）
awk -F, 'NR>1 && $2 ~ /^[0-9]+$/ {s+=$2} END {printf "total_kernel_ms=%.1f\nper_forward_ms=%.2f\n", s/1e6, s/1e6/60}' /home/logs/sennian/nsys-log/view-ab2/kern_sum.csv

# 指标 2：z 隐藏拷贝是否消失（bf16 direct_copy 大头应从 5400 次/49μs 降到 3600 次/~2μs）
grep "direct_copy_kernel_cuda" /home/logs/sennian/nsys-log/view-ab2/kern_sum.csv | head -3

# 指标 3：三个哨兵 kernel（qkvzba 仅 decode 残留；conv1d ≤266μs；chunk_h ~353μs）
grep -E "qkvzba|conv1d|chunk_gated_delta_rule_fwd_kernel_h" /home/logs/sennian/nsys-log/view-ab2/kern_sum.csv

# 指标 4：端到端吞吐
grep -iE "throughput|tok/s" /home/logs/sennian/nsys-log/view-ab2/bench.log
```

判读基准（Triton MoE 口径）：基线 241.4ms/fwd；view+bv64（含 z-copy 雷）238.04；z 零拷贝修复后预期 **~234ms**。

### 第 7 步：收尾

```bash
pkill -f sglang.launch_server   # 采完即停，避免影响后续实验
nsys sessions list              # 确认会话已结束
```

### 坐过的坑（每条都是真实事故）

1. 端口被旧 server 占用 → 新 server 秒崩 → `nsys sessions list` 为空（session 随目标进程死亡）
2. 用错 Python：pip 版 sglang5.14 吃不到仓库改动，必须用 feature 仓库 .venv
3. 环境变量前缀式传递不可靠，用 `--env-var` 显式传，并用 `/proc/<pid>/environ` 验证
4. `--session=NAME` 只能 attach 已有会话，建新会话必须 `--session-new=NAME`
5. nsys stats 的 Time(%) 列只有 0.1% 精度，跨 run 对比用第 2 列 Total Time (ns) 的绝对值
