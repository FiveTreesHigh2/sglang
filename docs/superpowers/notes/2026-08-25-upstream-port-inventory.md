# 优化清单与上游移植台账（2026-08-25）

目的：把 `feat/flashinfer-sm120-fp8-moe`（154 commits，52 文件 +13.4k 行）拆成可勾选条目，
供决定 PR 范围；并记录移植到 sglang v0.5.18 / flashinfer 0.6.17 的障碍。

分支：`integrate/v0.5.18`（基于 tag `v0.5.18` = `71de97b26`，2026-08-20）

## 上游核查结论（v0.5.18 实测 grep）

| 检查项 | 结果 |
|---|---|
| flashinfer pin | `flashinfer_python[cu13]==0.6.17` |
| flashinfer 0.6.17（`a0a6b019`，2026-08-10）含 #4130 `is_gated` | **YES**（`92274ba1` 在其历史内）→ A4 可直接用 stock wheel，fork 可弃 |
| flashinfer 0.6.17 含 `epi_pred_stg_finalize` | **NO** → A5 确认仍是我们 fork 独有 |
| `flashinfer_sm120_fp8` / `FlashInferSm120` / `moe_gemm_fp8_nt_groupwise` | **0 命中** —— A 组上游完全没做 |
| `SGLANG_GDN_CONV_FUSION` / `_NORM_FP8_OUT` / `_QKV_VIEW` / `_WU_BK` / `_CHUNK_O_BK` / `_L2NORM_BT` / `_EXTRACT_V_NUM_WARPS` / `_CONV_BLOCK_N` | **0 命中** —— B 组全部独有 |
| `weight_scale_inv_fi` | **0 命中** —— C 组独有 |
| `SGLANG_GDN_CHUNK_H_*` | **上游已有**（随 #30795 进来），我们那一项只是跑参数、未改代码 |

## A 组 — FlashInfer SM120 FP8 MoE 后端（依赖 flashinfer）

| ID | 项目 | 实测收益 | 主要文件 |
|---|---|---|---|
| A1 | 新后端 `flashinfer_sm120_fp8`：注册 + fp8 量化接线 + scale layout 打包 + runner 主体 + 7 条兼容性硬校验 | 载体项；prefill 单项 +1.3~1.7% | `srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`(new 427)、`moe/utils.py`、`moe_runner/runner.py`、`quantization/fp8.py`、`unquant.py`、`arg_groups/overrides.py`、`server_args.py` |
| A2 | Fused A2：SwiGLU + 量化 + pack 单 kernel | 含在 A1 账内 | `kernels/ops/moe/flashinfer_sm120_fp8.py`(new 599)、`*_swiglu_quant.cuh`(new) |
| A3 | Fused A1（`FUSED_A1`，默认开）：permute + 量化 + scatter 单 kernel | **-0.9ms/fwd**（6.90→6.02ms；148.3μs vs 理论下限 138μs，效率 93%，已达上限） | 同上 + `*_quant_scatter.cuh`(new) |
| A4 | gated GEMM1（`GATED`，默认开）：SiLU(gate)×up 进 epilogue，w13 加载期翻 up-first | prefill +0.4%、decode 中性；GSM8K 0.832 | `quantization/fp8.py`、runner、kernels ops |
| A5 | **FINALIZE 融合**（`MOE_FINALIZE`，默认开）：GEMM2 epilogue 加权散射直出 token 输出，unpermute 消失 | decode bs=1 6.48→6.32ms；bs=128 58.49→57.99ms（与 Triton 持平） | 同上 + **flashinfer fork 侧 epilogue 变体** |
| A6 | 正确性三修：`-1` padding route 守卫、PDL trigger 移到 store 后、fast-division 对齐 legacy | 精度 bug（会污染输出） | 两个 `.cuh` |
| A7 | 测试与工具：MoE 单测 1800 行、config 单测 228、stagewise debug 190、scale packing contract、3 个 bench 脚本 ~3.5k 行 | — | `test/registered/...`、`scripts/pro5000/...` |

## B 组 — GDN 线性注意力（纯 sglang，与 flashinfer 无关）

基数：prefill serving 稳态 241.4ms/fwd（60 forwards 采样，4096 输入 / 并发 100）

| ID | 项目 | 实测收益 | 主要文件 |
|---|---|---|---|
| B1 | **GDN1 conv1d epilogue 融合**：conv 输出的 qkv 拆分 + q/k l2norm 在寄存器内完成直写 dense q/k/v，消掉 134MB/层中转张量的全量回读 | **-6.70ms/fwd**（l2norm_strided 60→0 次、extract_columns 30→0 次、conv 无代价） | `kernels/ops/mamba/causal_conv1d_triton.py`(+204)、`attention/linear/gdn_backend.py`(+188) |
| B2 | **D1-a gated norm 直出 fp8**：GDN gated norm epilogue 融合 fp8 量化喂 out_proj | **-2.84ms/fwd**（norm 30.5→0.5 次/fwd、quant 162.5→132.4；新 kernel 1.29TB/s 达带宽上限） | `fla/layernorm_gated.py`(+161)、`quantization/fp8_utils.py`、`models/qwen3_5.py` |
| B3 | qkvzba 恒等复制改 strided view + gated norm grouped 模式读 strided z | -10.72ms（qkvzba）→ z 隐藏拷贝反噬 +4.4ms → 修好净 **-3.3ms** | `causal_conv1d_triton.py`、`models/qwen3_5.py` |
| B4 | 免费档 B1/B9/B10：chunk_o `zeros_like→empty_like`、视图分支删 a/b `.contiguous()`、缓存 `norm.weight.repeat` | **-1.88ms/fwd**（FillFunctor 113.8→83 次、D2D memcpy 165→89 次） | `fla/chunk_o.py`、`models/qwen3_5.py`、`environ.py` |
| B5 | B8：`chunk_fwd` A 缓冲 `zeros→empty` + `recompute_w_u` 内因果 mask | **-0.55ms/fwd**（memset 18895→1980 次/窗口） | `fla/chunk_fwd.py`、`fla/wy_fast.py` |
| B6 | qk_rope gate 去物化（返回 strided view 不复制） | **-0.86ms/fwd**（249.9→164.9μs，-34%） | `kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py`、`qwen3_5.py` |
| B7 | chunk_o / wu / l2norm tile 环境旋钮（chunk_h 上游已有） | -0.3ms（仅 `WU_BK=128 BV=128 STAGES=3` 有效，其余三组无改进） | `fla/chunk_o.py`、`wy_fast.py`、`l2norm.py`、`chunk_fwd.py` |
| B8 | extract_v 专用 Triton kernel + strided l2norm（B1 的前置中间态） | 大部分被 B1 取代 | `jit_kernel/triton/gdn_fused_proj.py`、`fla/l2norm.py`、`gdn_backend.py` |
| B9 | 等价性测试 7 个（逐位 / NaN 污染法） | — | `test/manual/test_gdn_*.py` |

## C 组 — dense FP8（flashinfer_cutlass 路径，纯 sglang）

| ID | 项目 | 实测收益 | 主要文件 |
|---|---|---|---|
| C1 | B3/B4 消除 per-call scale 转置：激活量化直出 MN-major scale（`column_major_scales=True`，零拷贝视图）+ 常量 weight_scale 加载期预转置（`weight_scale_inv_fi`） | **direct_copy 335→13 次/fwd**（-322，与预期 -324 吻合；结构性 -0.67ms；另收 BCG 每图 -160 拷贝节点） | `quantization/fp8_utils.py`、`fp8.py` |
| C2 | bug fix：预量化路径的最终输出 cast 误用 fp8 code dtype | 正确性 | `quantization/fp8_utils.py` |
| C3 | 等价性测试 | — | `test/manual/test_fp8_dense_gemm_scale_layout_equivalence.py` |

## D 组 — 判负 / 工具（不建议进 PR）

| ID | 项目 | 结论 |
|---|---|---|
| D1 | moe_permute counting-sort 快路径 | **判负**：+0.63ms/fwd 回归（64k atomicAdd 在 257 槽热点串行化，histogram 单次 23.6μs），默认关，仅留研究价值 |
| D2 | B14 dense m%4 垫零 | **已自行 revert**（净零；且是 decode bs≤3 回退的真凶之一） |
| D3 | Triton MoE down/up tuning CLI + route 生成器 | 工具链（766 行测试 + 生成器），非运行时优化；当年用户决定搁置 |
| D4 | pro5000 bootstrap / env collector / 3 个 bench 脚本 / 中文设计文档 | 项目专属基建，不适合上游 |

## 累计成绩（口径分开，勿混算）

| 口径 | 数字 |
|---|---|
| prefill serving GPU 时间（B+C 组） | 241.4 → **216.41 ms/fwd（-10.35%）** |
| prefill E2E（B+C 组，三轮中位数） | 36592 vs 32665 tok/s = **+12.02%** |
| prefill E2E @4096（叠加 A 组全链） | 32372 → **37582 tok/s（+16.1%）** |
| decode bs=1 TPOT | 6.18（Triton baseline）→ **6.32ms** |
| decode bs=128 TPOT | 57.96（baseline）→ **57.99ms**（持平） |
| 精度门 | GSM8K 0.832 / MMLU 0.680，用户签字与基线无差异 |

## 移植障碍（必须先处理）

### 1. 上游目录大搬迁（#32045，RFC #29630 Phase 4）

`python/sglang/jit_kernel/` 整树删除，搬到 `python/sglang/kernels/{jit,ops}/`。受影响：

| 我们的路径 | v0.5.18 目标路径 |
|---|---|
| `jit_kernel/csrc/moe/flashinfer_sm120_fp8_quant_scatter.cuh` | `kernels/jit/csrc/moe/` |
| `jit_kernel/csrc/moe/flashinfer_sm120_fp8_swiglu_quant.cuh` | `kernels/jit/csrc/moe/` |
| `jit_kernel/flashinfer_sm120_fp8_moe.py` | `kernels/ops/moe/`（按新 jit binding 惯例） |
| `jit_kernel/moe_permute_prepare.py`（我们改的） | `kernels/ops/moe/moe_permute_prepare.py` |
| `jit_kernel/triton/gdn_fused_proj.py`（我们改的） | `kernels/ops/attention/triton_gdn_fused_proj.py` |
| `kernels/ops/moe/flashinfer_sm120_fp8.py` | 路径不变（已在新布局下） |

### 2. 上游 churn 大的文件（需重新落钩子，不能靠 git merge）

`8f765bc1c..v0.5.18` 的上游改动量：

| 文件 | 上游 churn |
|---|---|
| `srt/server_args.py` | +2483 / -627 |
| `srt/environ.py` | +971 / -540 |
| `srt/arg_groups/overrides.py` | +620 / -66 |
| `srt/layers/quantization/fp8_utils.py` | +381 / -284 |
| `srt/layers/quantization/fp8.py` | +354 / -125 |
| `srt/layers/attention/linear/gdn_backend.py` | +334 / -52 |
| `srt/layers/quantization/unquant.py` | +256 / -31 |
| `srt/models/qwen3_5.py` | +233 / -66 |
| `srt/layers/moe/utils.py` | +183 / -14 |
| `kernels/ops/mamba/causal_conv1d_triton.py` | +12 / -6 |
| `kernels/ops/attention/fla/{chunk_fwd,chunk_o,wy_fast}.py`、`fused_qk_rmsnorm_rope_gate.py`、`kernels/gdn_triton.py` | 未变动 |

### 3. A5 FINALIZE 卡 flashinfer 侧

epilogue 变体是我们 fork 的 CUDA 代码，stock flashinfer 0.6.17 里没有。三条出路：
先给 flashinfer 提 PR / sglang 侧加能力探测且默认关 / 本次 PR 剔出 A5。

### 4. PR base 选择（已实测，2026-08-25）

`upstream/main` = `191244b3f`（2026-08-25）。实测 ancestry：

| 事实 | 数字 |
|---|---|
| v0.5.18 是 main 的祖先？ | **不是** |
| 两者 merge-base（release 切点） | `0111b2903`，2026-08-18 |
| main 自切点起独有 | **392 commits** |
| v0.5.18 独有（release 分支 cherry-pick） | **8 commits** |

所以基于 v0.5.18 开分支再向 main 提 PR，diff 里会混进那 8 个 release 专属 cherry-pick，
且树落后 main 392 个 commit。结论：**PR 分支直接基于 `upstream/main`**，
`integrate/v0.5.18` 只作自用部署分支。

拉取备注：`gh-proxy.com` 被公司域名策略拦截（HTTP 403）；`git fetch upstream --tags`
拉全量会 `early EOF`；可行方式是单 ref 直连 `git fetch upstream main:refs/remotes/upstream/main`。

### 5. 上游 PR 惯例（5 个已合并同类 PR 实测）

| PR | 类型 | 文件 | 增删 |
|---|---|---|---|
| #30541 Add HPC-Ops FP8 MoE runner backend | 新增 MoE 后端 | 11 | +631 / -1 |
| #30272 SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend | 新增 SM120 flashinfer MoE 后端 | 18 | +506 / -237 |
| #34275 fuse cosmos qk norm, rope, kv packing | kernel 融合 | 5 | +570 / -30 |
| #25855 optimize paged_mqa_metadata | kernel 优化 | 4 | +641 / -58 |
| #33208 FlashInfer CUTLASS dense GEMM on SM120 | 后端切换 | 6 | +171 / -495 |

- 规模惯例：**一个 PR 4–18 文件、净 200–650 行** → 我们 52 文件 / +13.4k 必须拆 8–9 个
- 单测惯例：`test/registered/moe/test_*.py` ~234 行、`.../test_*_guard.py` ~63 行（我们 1800 行需瘦身）
- bench 惯例：`benchmark/kernels/bench_*.py`，#25855 只有 57 行（我们 3.5k 行需瘦身并挪位）
- PR 模板四段必填：Motivation / Modifications / Accuracy Tests / Speed Tests and Profiling
- env 旋钮：上游自己在 `kernels/ops/attention/fla/chunk_delta_h.py`、`fla/utils.py` 用裸 `os.getenv` → B7 写法合规
- 合并流程：Ping Merge Oncalls → CODEOWNERS 批准 → 评论 `/tag-and-rerun-ci` → 有写权限者合
- 相关 CODEOWNERS：`srt/layers/quantization` @ch-wan @BBuf @Edwardf0t1 @HaiShaw @b8zhong；
  `kernels/ops/attention/fla` @yizhang2077 @hebiao064 @yuan-luo；
  `srt/layers/attention` @merrymercy @Fridge003 @ispobock @Qiaolin-Yu @hebiao064

---

## PR 拆分方案（2026-08-25，待用户审批）

工作分支：`port/main`（基于 `upstream/main` = `191244b3f`）
上游先例：stacked PR 是被接受的模式（`3836cba9e ... (stack 13/15) (#30075)`）

### 挂点存活验证（upstream/main 实测）

| 挂点 | 状态 |
|---|---|
| `fp8_utils.py` `column_major_scales` / `gemm_fp8_nt_groupwise` | 存活（4 / 7 处） |
| `fla/layernorm_gated.py` `def rms_norm_gated` | 存活 |
| `mamba/causal_conv1d_triton.py` `def causal_conv1d_fn` | 存活 |
| `gdn_backend.py` 调用 `causal_conv1d_fn` | 存活（8 处） |
| `qwen3_5.py` `RMSNormGated` 实例化 | 存活（306 → 402 行，位移与 +95 churn 吻合） |
| `moe/runner.py` `class MoeRunner` / `moe/utils.py` `class MoeRunnerBackend` | 存活 |

### 三个 PR（2026-08-25 敲定：收敛到 3 个，按子系统切）

按子系统重算的生产行数：A(MoE 后端) 1648、B(GDN) 837、C(dense fp8) 212。
上游同类先例支持单 PR 大体量：`[Feature] Add DeepEPv2 MoE A2A backend (#29525)` = 1416 行 / 14 文件。
合并后同主题内的依赖链（旧 PR1→3→2、7→8→9）全部消失，无需 stacked PR。

| PR | 标题（拟） | 内容（对应旧 ID） | 生产+测试预估 |
|---|---|---|---|
| PR-1 | `[Quant] Emit MN-major FP8 activation scales directly on the SM120 CUTLASS path` | C 组：`fp8_utils.py` 直出 MN-major scale + 输出 cast bugfix、`fp8.py` weight_scale 加载期预转置（`weight_scale_inv_fi`）、1 测试（C1+C2+C3） | ~300 |
| PR-2 | `[GDN] Fuse and de-materialize the Qwen3.5 gated-delta-net prefill path` | B 组：conv epilogue 融合(B1,-6.7ms) + gated norm 直出 fp8(B2,-2.84ms) + qkvzba/qk_rope gate 去物化(B3,B6) + 免费档 zero-init/contiguous/repeat(B4,B5)。**不含 B7 tile 旋钮** | ~1500 |
| PR-3 | `[MoE Backend] Add the flashinfer_sm120_fp8 MoE runner backend` | A 组：新后端 + scale 打包 kernel + fused A1 + gated GEMM1 + 正确性三修，一次成型（A1+A2+A3+A4+A6） | ~1700 |

依赖：三者近乎独立；唯一交叉是 `fp8.py`（PR-1 改 weight_scale、PR-3 改 gated 翻转，不同 hunk）→ 按 PR-1 先于 PR-3 发即可。
建议先发 PR-1（最小 ~300 行）热身，摸清 CI 触发与评审节奏，PR-2/PR-3 随后并行。

### B7 tile 旋钮：全部剔除（2026-08-25 用户决定）

旧 PR4 里的 `SGLANG_GDN_CHUNK_O_*` / `_WU_*` / `_L2NORM_*` 旋钮与 `chunk_fwd.py` 的 `num_warps=8`
autotune 候选（commit `5d5ce395b`）**一律不移植**。相关文件在 PR-2 中的最终形态：

- `fla/chunk_o.py`、`fla/l2norm.py`：保持上游原样（PR-2 不碰）
- `fla/wy_fast.py`：只保留 B5 的因果 mask（`tl.where` 上三角置零），不引入 tile 旋钮
- `fla/chunk_fwd.py`：只保留 B5 的 `A = torch.zeros → torch.empty`，不加 autotune 候选

理由：实测仅 `WU_BK=128 BV=128 STAGES=3` 一组有效(-0.3ms)，其余默认值未动、无实测收益，
给上游塞"默认值=原值、无收益"的 env 只会招评审质疑旋钮存在意义。

### 明确不进 PR

A5 FINALIZE（依赖 flashinfer fork）、A7 大 bench（3.5k 行 → 需重写为 `benchmark/kernels/bench_*.py` 小脚本）、
B7 tile 旋钮（见上）、B8 中间态（`gdn_fused_proj.py`，已被 B1 取代）、D1 counting-sort、D2 B14 垫零、
D3 Triton 调参工具（2339 行）、D4 项目脚本与门禁测试（6335 行）、全部中文文档（10968 行）。

---

## Hunk 级归属审计（2026-08-25，移植切分依据）

对跨 PR 边界的共享文件逐 commit 核实，得两处标签修正 + 一处真实文本依赖。

### 混合文件归属表

| 文件 | 归属切分（按 commit / 类边界） |
|---|---|
| `fp8.py` | Fp8**Linear**Method hunk（`64f3c998e`）→ PR-1；Fp8**MoE**Method hunk（`2e7794660`+`201376dfa`）→ PR-3。类边界天然隔开 |
| `fp8_utils.py` | `weight_scale_mn` 参数+消费、`column_major` 直出（`64f3c998e`+`8454618f8`）→ PR-1；`input_scale` 预量化分支+`out_dtype` fix（`22a70b54a`+`263b77cd1`）→ PR-2；m%4 revert（`18af088bd`）→ 弃（main 上 no-op） |
| `environ.py` | `FUSED_A1`/`GATED` → PR-3；`MOE_FINALIZE` → 弃 |
| `qwen3_5.py` | 全 PR-2（B2/B3/B4/B6，无 A 组内容） |
| `chunk_fwd/wy_fast/chunk_o/l2norm.py` | B1/B4/B5 hunk → PR-2；B7 tile 旋钮 hunk（`5d5ce395b`）→ 弃 |
| `causal_conv1d_triton.py` | 全 PR-2（B1 conv 融合含其自身 BLOCK 旋钮 + B3 视图；非 B7） |

### 修正 1：`weight_scale_inv_fi` 属 PR-1（先前误记 PR-2）

commit `64f3c998e`，位于 `Fp8LinearMethod`（dense）。加载期把常量权重 scale 预转置成 CUTLASS MN 布局
存入 `layer.weight_scale_inv_fi`；调用期经 `_block_fp8_extra_kwargs` 作为 `weight_scale_mn` 传入 wrapper，
省掉每次 GEMM 的权重 scale transpose+copy。通用 dense 优化，与 GDN 无关。

### 修正 2：C2 `out_dtype` cast fix 属 PR-2（先前误记 C 组/PR-1）

commit `263b77cd1`，唯一逻辑 `out_dtype = bf16 if input_scale is not None else input_2d.dtype`。
只修预量化路径（`input_scale is not None`，仅 B2 的 out_proj 走）输出误 cast 成 fp8 的 bug → 属 PR-2。

### 真实依赖：PR-1 与 PR-2 改同一函数 `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`

- PR-1 段（`64f3c998e`）：加 `weight_scale_mn` 参数；激活量化改直出 MN（`sglang_per_token_group_quant_fp8_row_padded`）；`if weight_scale_mn is not None: weight_scale = weight_scale_mn`
- PR-2 段（`22a70b54a`+`263b77cd1`）：同函数开头加 `if input_scale is not None:` 预量化分支 + `out_dtype` fix
- 性质：**功能上不依赖**（PR-2 预量化分支自带 MN scale，weight 侧退回 per-call transpose 仍正确），**文本上冲突**（改同段）
- 处理：先合 PR-1，PR-2 基于 PR-1 之后的树来写；PR-3 与 PR-1 在 `fp8.py` 亦仅文本重叠（不同类）→ 顺序 PR-1 → PR-2/PR-3

---

## 撞车检查（2026-08-25，已合并=本地 git 高置信；开放 PR=直接读页面核实）

### 已合并层（upstream/main `191244b3f`，逐符号+逐文件现状核实）

| 项 | main 现状 | 判定 |
|---|---|---|
| PR-3 flashinfer_sm120_fp8 | 4 个符号 0 命中；MoeRunnerBackend 17 值无一是我们的 | 未做 |
| PR-2 B1 conv 融合 | `causal_conv1d_fn` 无 qkv_split/l2norm epilogue | 未做 |
| PR-2 B2 gated norm fp8 | `fla/layernorm_gated.py` 无 fp8/quant/e4m3 | 未做 |
| PR-2 B3/B4 | `b.contiguous()/a.contiguous()` 仍在，`self.norm(x,z)` 仍普通调用 | 未做 |
| PR-2 B5 | `chunk_fwd.py` 仍 `torch.zeros`、`chunk_o.py` 仍 `zeros_like` | 未做 |
| PR-2 B6 | `fused_qk_gemma_rmsnorm_rope_gate` 仍 `torch.empty` 物化 gate_out | 未做 |
| PR-1 dense scale | wrapper 仍 `column_major_scales=(backend=="trtllm")` + per-call `transpose().contiguous()` | 未做 |
| #32994 rmsnorm+quant 融合（已合并，SM90/100/120） | 融进 `layers/layernorm.py` **标准 RMSNorm**，非 GDN gated norm | **不撞 B2**（不同 norm 层） |

### 开放 PR 层（WebFetch 直接读页面核实）

| PR | 状态/作者 | 与我们的关系 |
|---|---|---|
| **#32443** [Qwen3.5] Fuse gated RMSNorm and FP8 quantization | **draft** / JustinTong0323 | **B2 直接重复**：同 target（Qwen3.5 GDN norm→out_proj）、测试文件 `test_rms_norm_gated_fp8_quant.py` 与我们函数名一字不差、同在 fp8_utils.py 冲突 |
| **#30797** [GDN] Fuse the linear-attention prefill prologue | **open** / mattteochen | **B1 功能重复**：独立 `gdn_prefill_fused` kernel 合并 packed-QKV split + gating + Q/K L2norm（我们放 conv epilogue，他们放独立 prologue）；触 `l2norm.py`、`radix_linear_attention.py` |
| #28125 SM120 dispatch for fp8_blockwise_scaled_grouped_mm | open / waynehacking8 | PR-3 **同目标不同机制**：CUTLASS grouped GEMM，非 FlashInfer。竞争"SM120 blockwise fp8 MoE 怎么做" |
| #34827 enable DeepGEMM MoE runner on SM120 | open / qqtang-code | PR-3 **同目标不同机制**：DeepGEMM。触 server_args.py/configurer.py/moe_runner/deep_gemm.py |
| B3/B4/B5/B6（视图去物化 + 免费档） | — | 任何 PR 均未发现 → **仍原创、未被认领** |

### 结论

- **PR-1（dense fp8 scale）：安全**，无重复；仅 `fp8_utils.py` 与 #27896(已合)/#33216/#34731(开放) 有合并冲突风险，无重复劳动。
- **PR-3（FlashInfer SM120 FP8 MoE）：无代码重复**，但有两个竞品后端在飞（CUTLASS #28125、DeepGEMM #34827）抢同一目标。策略风险：维护者可能倾向整合而非再收第三条路。枚举值不撞，但预期评审会问"为何再加 FlashInfer 路径"。
- **PR-2（GDN）：部分已被抢跑。** B2 被 draft #32443 直接重复（连函数名都同），B1 被 #30797 功能覆盖；B3/B4/B5/B6 仍原创。

---

## PR-1 落地记录（分支 `perf/fp8-dense-mn-scale-layout`，基于 `upstream/main` 191244b3f）

### 已实现（2 文件 / +23 / -6）

| 处 | 改动 |
|---|---|
| `fp8_utils.py` wrapper 签名 | 加可选参 `weight_scale_mn`（默认 None，不破坏既有调用方；该 wrapper 全仓仅此一处定义、无其他显式调用点） |
| `fp8_utils.py` cutlass 分支 | 有 `weight_scale_mn` 则直接用，替代 per-call `weight_scale.transpose(-1,-2).contiguous()` |
| `fp8.py` import | 补 `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`（供 `is` 身份判断） |
| `fp8.py` 加载期 | `process_weights_after_loading_block_quant` 末尾，仅当 dispatch 到 flashinfer wrapper 时预转置存 `layer.weight_scale_inv_mn` |
| `fp8.py` apply | block_quant 非 tuple 分支经 `block_fp8_extra_kwargs` 传 `weight_scale_mn`（`getattr` 容错，其他后端收不到未知 kwarg） |

命名决策：`weight_scale_inv_fi` → **`weight_scale_inv_mn`**。`inv` 沿用上游 `register_parameter("weight_scale_inv")`；
上游已有兄弟属性 `weight_scale_inv_swizzled` / `_shuffled` / `_deepgemm`，`_mn` 与之同模式且对应 `scale_major_mode="MN"`。

### 待验证后再定：A-scale 侧 `column_major_scales=True`（2026-08-25 用户决定暂缓）

原计划把 `column_major_scales=(backend == "trtllm")` 改为无条件 `True`，让 cutlass 的
`x_scale.transpose(-1,-2)` 成为零拷贝视图。查证结果：

- **形状/补齐安全**：未传 `scale_tma_aligned`（默认 False）→ 走 `fp8_kernel.py:506-511`，
  `torch.empty((k//block_k, m)).permute(-1,-2)` 返回形状 `(m, k//block_k)`、底层 `(k//block_k, m)` 连续。
  形状与 row-major 一致，assert 照旧成立，转置确为零拷贝，**无 padding**。
- **无条件转置比原条件写法更正确**：原 `if x_scale.shape == (m, k//block_k)` 靠猜形状，
  `m == k//block_k` 方阵时有歧义。
- **⚠ 真实行为变化**：`fp8_kernel.py:620-629` 有快路径
  `if group_size == x.shape[-1] and ... and not column_major_scales: return sglang_per_token_quant_fp8(x)`。
  改 True 后该快路径不再触发（条件为 `block_k == k`，即 **k == 128** 的线性层），换走 group kernel。
  两者数学等价但实现不同，**可能 ±1 码位差**（同族先例：row_padded direct-op vs wrapped 差 0.096%）。
- 缓解证据：trtllm 分支改动前就一直传 True，故"k==128 + column-major + group kernel"
  这条组合上游今天已在跑，非未验证新路径。
- 局限：我们服务器上的逐位实测是在 k≫128 的真实模型（k ∈ {2048,4096,7168}）上做的，**未覆盖 k==128 边界**。

**决定**：本次 PR 该行**保持上游原样**（保守），A-scale 侧优化另行处理。
补一个 **k == block_k == 128** 的逐位用例实测两 kernel 是否一致：一致则再提该项，
不一致则收窄改动，不带未知数进 PR。
