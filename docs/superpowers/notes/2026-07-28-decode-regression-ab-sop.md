# Decode 回退归因 A/B 采集 SOP（2026-07-28）

## ✅ 已定案部分（B14，commit `18af088bd`）

- 前提修正：baseline 与现配置的 dense backend **启动参数**相同（均 flashinfer_cutlass），但**代码版本不同**——B14（`8454618f8` 引入的 m%4 垫零）只存在于现配置，属于代码差异变量，"dense 已排除"的原判断不成立
- 定案证据：TPOT 回归在 bs=2（+1.05ms）与 bs=4（+0.38ms）之间存在 ~0.7ms 台阶，与垫零分支生效边界（m<4）精确对齐；每步 162 次 dense GEMM × 2 个 cat ≈ 324 个额外 kernel ≈ 0.7ms
- 逐位验证：SM120 上 m=1..7 不垫零输出与垫零逐位一致（且 baseline 历史上一直不垫零运行）→ 垫零属不必要防御，已去除（B14 revert，commit `18af088bd`）
- 预期：bs≤3 的 decode TPOT 收回 ~0.7ms；bs≥4 无影响。**待复测确认**
- 残余待归因：全批量段的 +0.3~0.4ms/step，按下方 H1/H2 矩阵继续

## 背景与假设

- 实测：decode bs=1 TPOT 基线 6.33ms → 现配置 7.45ms（**+1.12ms/step**），bs≥4 后差距收敛至 <3%
- stage-2 microbench（eager，07-21 旧代码）显示 FI MoE 在 tokens=1 下反而快 6%，但 serving decode 走 full CUDA graph，launch 开销消失后 kernel 数量地板价才显形；且 microbench 未含 FUSED_A1 与精度修复守卫
- 待验假设（按优先级）：
  - H1：FI MoE 路径（含 glue kernel 数量）在 graph 下劣于 Triton fused_moe，≈ +0.3~0.4ms/step（B14 定案后的修订估值）
  - H2：FUSED_A1 quant_scatter（B13 默认开）在小 M 下劣于 legacy 三段式
  - H3：残差（B3 column-major quant / 精度修复守卫）——B14 已从本项移出并定案

## 实验矩阵（bs=1 decode，一次只动一个变量）

| 组 | MoE backend | FUSED_A1 | 判定 |
|---|---|---|---|
| A | flashinfer_sm120_fp8 | 1（现状） | 现配置基准 |
| B | triton | —（不生效） | A−B = MoE backend 全账（验 H1） |
| C | flashinfer_sm120_fp8 | 0 | A−C = FUSED_A1 份额（验 H2） |

判读：若 B 的 TPOT 回到 ~6.4ms → H1 成立，账基本结清；若 A−B 明显小于 1.1ms → 残差走 H3，需追加 per-kernel diff 定位。

## 采集步骤（每组独立执行，SESSION 名依次 dec-ab-A / dec-ab-B / dec-ab-C）

### 第 0 步：清场（同 prefill SOP）

```bash
ss -ltnp | grep 30000
ps aux | grep sglang.launch_server | grep -v grep
nsys sessions list
```

### 第 1 步：起 server

以 A 组为例（`dec-ab-A`）。**B 组**把 `--moe-runner-backend flashinfer_sm120_fp8` 换成 `--moe-runner-backend triton`；**C 组**在 `--env-var` 里追加 `,SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0`。

```bash
mkdir -p /home/logs/sennian/nsys-log/dec-ab-A
cd /home/logs/sennian/nsys-log/dec-ab-A

nsys launch --session-new=dec-ab-A \
  --env-var=SGLANG_GDN_CHUNK_H_BV=64,SGLANG_GDN_CHUNK_H_NUM_WARPS=4,SGLANG_GDN_CHUNK_H_NUM_STAGES=2,SGLANG_GDN_WU_BK=128,SGLANG_GDN_WU_BV=128,SGLANG_GDN_WU_NUM_STAGES=3 \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.launch_server \
    --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
    --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
    --host 33.243.206.227 --port 30000 \
    --enable-metrics --tp-size=1 \
    --reasoning-parser=qwen3 --collect-tokens-histogram --tool-call-parser=qwen3_coder \
    --disable-radix-cache --mem-fraction-static 0.9 \
    --fp8-gemm-backend flashinfer_cutlass \
    --moe-runner-backend flashinfer_sm120_fp8 \
    --enable-layerwise-nvtx-marker \
    > /home/logs/sennian/nsys-log/dec-ab-A/server.log 2>&1 &

nsys launch --session-new=dec-ab-A      --trace=cuda,nvtx,osrt --cuda-graph-trace=node   python3 -m sglang.launch_server     --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0     --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/     --host 33.243.206.227 --port 30000     --enable-metrics --tp-size=1     --reasoning-parser=qwen3 --collect-tokens-histogram --tool-call-parser=qwen3_coder     --disable-radix-cache --mem-fraction-static 0.9     --fp8-gemm-backend flashinfer_cutlass          --enable-layerwise-nvtx-marker

nsys launch --session-new=dec-ab-A   --env-var=SGLANG_GDN_CHUNK_H_BV=64,SGLANG_GDN_CHUNK_H_NUM_WARPS=4,SGLANG_GDN_CHUNK_H_NUM_STAGES=2,SGLANG_GDN_WU_BK=128,SGLANG_GDN_WU_BV=128,SGLANG_GDN_WU_NUM_STAGES=3,SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0   --trace=cuda,nvtx,osrt --cuda-graph-trace=node   /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.launch_server     --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0     --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/     --host 33.243.206.227 --port 30000     --enable-metrics --tp-size=1     --reasoning-parser=qwen3 --collect-tokens-histogram --tool-call-parser=qwen3_coder     --disable-radix-cache --mem-fraction-static 0.9     --fp8-gemm-backend flashinfer_cutlass     --moe-runner-backend flashinfer_sm120_fp8     --enable-layerwise-nvtx-marker
```

注意：
- `--cuda-graph-trace=node` **必须带**（decode 是 full graph，不展开则全部 kernel 不可见，教训见 D1-a）
- decode 主要走 CUDA graph replay，`--enable-layerwise-nvtx-marker` 仅供 eager 段参考

### 第 2 步：三重配置校验（缺一不可，历史上两次整批作废都栽在这）

```bash
PID=$(pgrep -f sglang.launch_server | head -1)
# 2a. 环境变量进程内校验（C 组必须看到 FUSED_A1=0）
tr '\0' '\n' < /proc/$PID/environ | grep -E "GDN|FUSED_A1"
# 2b. MoE backend 校验（A/C 组应出现 flashinfer_sm120_fp8 A1 prepare mode 日志；B 组不应出现）
grep -iE "moe.*backend|A1 prepare mode" /home/logs/sennian/nsys-log/dec-ab-A/server.log
# 2c. 就绪探测（反复执行直到 200）
curl -s -o /dev/null -w "%{http_code}\n" http://33.243.206.227:30000/health
```

### 第 3 步：decode 稳态负载（bs=1）

```bash
/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 -m sglang.bench_serving \
  --backend sglang \
  --host 33.243.206.227 --port 30000 \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 2048 --random-range-ratio 1.0 \
  --num-prompts 30 --max-concurrency 1 \
  > /home/logs/sennian/nsys-log/dec-ab-A/bench.log 2>&1 &
```

- 单条 = 1 次 prefill（~0.25s）+ 2048 步 decode（~15s），30 条 ≈ 8 分钟，窗口充足
- **采集窗口必须避开 prefill**：等首条请求进入 decode 段后再 start（见第 4 步的对齐方法）

### 第 4 步：稳态窗口采 15 秒

```bash
sleep 45   # 跳过 warmup + 首条 prefill；bs=1 下 95% 墙钟都是 decode，窗口内混入的 prefill ≤1 次
nsys start --session=dec-ab-A
sleep 15
nsys stop --session=dec-ab-A
ls -lt /home/logs/sennian/nsys-log/dec-ab-A/
```

窗口内若恰好跨请求边界会混入 1 次 prefill（~0.25s，占窗口 1.7%）；用第 5 步的步数归一化吸收，不需要精确对齐。

### 第 5 步：导出与判读

```bash
nsys stats -r cuda_gpu_kern_sum report1.nsys-rep --format csv \
  > kern_sum.csv

# 步数锚点：qkvzba 融合 kernel 仅 decode 路径运行、每步 30 次（30 GDN 层）
# steps = instances / 30
grep "qkvzba" kern_sum.csv

# per-step GPU 总量 = 总 kernel 时长 / steps
awk -F, 'NR>1 && $2 ~ /^[0-9]+$/ {s+=$2} END {printf "total_kernel_ms=%.1f\n", s/1e6}' \
  kern_sum.csv

# MoE 相关条目（A/C 组看 FI 家族，B 组看 fused_moe_kernel 家族）
grep -iE "quant_scatter|swiglu_quant|unpermute|moe_permute|cute|fused_moe|moe_sum|moe_align" \
  kern_sum.csv
```

判读口径：
1. **TPOT**：`grep -iE "tpot|itl" bench.log`（端到端裁决）
2. **per-step kernel 总量**：A vs B 差值应 ≈ TPOT 差值（decode 也应是 GPU-bound；若不符说明有 host 开销，另立案）
3. **MoE 类目差分**：A 的 FI 家族总和 vs B 的 Triton 家族总和 = MoE backend 净账
4. 哨兵：dense cutlass GEMM、GDN decode kernel（recurrent/conv update）三组应持平——不平则说明有串扰变量，该组作废重跑

### 第 6 步：收尾

```bash
pkill -f sglang.launch_server
nsys sessions list
```

## 预登记的结论分支

- A−B ≈ +1.0~1.2ms/step 且 MoE 类目差分对得上 → H1 成立；后续可选项：decode 走 Triton 的 hybrid 分支（integration plan 当年预留了 FUNCTIONAL_ONLY 位，但明确禁止静默回退，需加显式 threshold 开关并重新过精度）
- A−C 显著（>0.2ms/step）→ H2 有份额；FUSED_A1 改为仅 prefill 启用是低成本修复
- A−B 明显小于 1.1ms → 残差在 MoE 之外，追加 kern_sum 全量 diff（B14/B3/守卫逐项排查）
- 全组不显著但 TPOT 差仍在 → host/launch 侧问题（graph replay 间隙），改用 nsys osrt + cuda API trace 立案

---

## 2026-07-29 归因结论与调度器修复验收清单

### 归因结果（bs=1，output 4096，干净窗口）

| 类目/步 | A (FI) | B (Triton) | Δ |
|---|---|---|---|
| MoE GEMM | 1.905ms (80×23.8μs) | 1.181ms (80×13.3μs + align) | +0.72ms |
| dense GEMM | 3.452ms | 3.718ms | -0.27ms |
| 总 kernel | 8.35ms | 7.72ms | +0.63ms |

- H1 半成立：FI ZeroPadding 调度器每 launch ~24μs 地板价（`scheduler.cuh` 线性扫 256 expert × 5 个 Scheduler 实例/block），占回退六成；H2（FUSED_A1）证伪（A≈C）
- bs=128 时 A/B 打平（58.65 vs 57.97ms TPOT）——扫描被真实工作摊薄，与全量 sweep 一致
- 残差 ~0.4-0.5ms 未归因（需 D 组：基线 pip 环境同口径）

### 修复：flashinfer `zero-padding-sched-fix` 分支（commit 56108d32）

smem tile-cumsum 协作预计算 + 二分查找，枚举顺序逐位不变。改动：`sm120_common/scheduler.cuh`、`sm120_blockscaling/kernel_impl.cuh`、`tests/grouped_mm/test_cute_sm120_fp8.py`（新增 256-expert 稀疏路由 case）。

### 服务器部署（fork + git URL，pure-python wheel 秒装）

```bash
# 安装（fork 推送后）
/home/logs/sennian/pro5000-fi-moe/.venv/bin/uv pip install --force-reinstall --no-deps \
  "git+https://github.com/FiveTreesHigh2/flashinfer@zero-padding-sched-fix"
# JIT 预热 + 校验
/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 sglang/scripts/pro5000/flashinfer_sm120_fp8_smoke.py
# 回滚 = bootstrap_stage_b.sh 内 pinned 官方 wheel 重装
```

### 验收序列（任一步失败即停）

1. `pytest tests/grouped_mm/test_cute_sm120_fp8.py -v`（含新增 `many_experts_sparse` 两 case）
2. `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`（逐位 glue + full runner 三指标 + CUDA graph 一致性）
3. kernel microbench：`benchmark_flashinfer_sm120_fp8_moe.py --cum-m 8`、`--cum-m 64`（预期 24μs → ≤13μs）；`--cum-m 65536` prefill 哨兵（允差噪声级）
4. E2E decode：本 SOP A 组重跑，MoE GEMM per-step 预期 1.905 → ≤1.2ms，TPOT 回落 ~0.6ms
5. E2E prefill spot-check：4096 输入吞吐 ±1%
6. 精度 smoke（可选）：`python3 -m sglang.test.run_eval --eval-name gsm8k --num-examples 20`

### 2026-07-29 验收结果（全部通过）

- 修复分支最终 commit：`c7bf5278`（56108d32 的 static smem 对齐 bug 修复：表并入动态 SharedStorage，static __shared__ 会破坏 TMA 128B 对齐 → cudaErrorMisalignedAddress）
- 部署方式：因服务器无法直连 GitHub（子模块 clone 失败），实际用文件覆盖 site-packages/flashinfer/data/csrc/（内容与 c7bf5278 一致）；正规化（fork wheel/Release/bootstrap SHA256）待补
- kernel microbench（锁频 1732MHz）：
  - decode：cum_m=8 FI 24μs → **5.7~9.4μs**（vs Triton 20μs，反超 2.2~3.6×）；数值与 Triton **逐位一致**（triton_vs_flashinfer=0.0）
  - prefill 哨兵 cum_m=65536：四 case 相对改前 -1.3% ~ +1.2%，±2% 门限内（不锁频对比会因 boost 行为差异漂 ±10%，勿用）
- E2E decode bs=1（input 4096/output 2048）：TPOT **7.45 → 6.51ms**（基线 6.33，残差 +0.18ms，不再立案）；prefill spot-check 与精度 smoke 正常
- 已知未了：
  - `test_full_runner_uses_single_fused_a2_prepare` 是 B13 旧账（未 patch `_use_fused_a1`，默认翻转后必挂），与本修复无关，待修
  - **上游 PR 4130 已 merge（2026-07-29，含 fp8+mxfp8）**：调度器重写（MoeScheduler：专职调度 warp + shfl prefix-sum/ballot + smem 管道发布，完全替代我们的补丁）+ `is_gated` fused SwiGLU（prefill MPE=1024 +34.3%）。我们自己的上游 PR 计划取消。
  - **合入计划（等含 #4130 的 nightly wheel）**：
    - 档 A（先做）：切官方 nightly。sglang 零代码改动（API 向后兼容，is_gated 默认 False）；改 `pyproject.toml:34` pin + `bootstrap_stage_b.sh:12-14` URL/SHA256；废弃 overlay 与 fork 分支。验收组合拳全跑：锁频 microbench（cum-m 8/64/65536）+ 逐位对照 + CUDA graph 一致性 + E2E decode/prefill + nightly churn 的 attention/GDN spot-check（~1 天）
    - 档 B（另行排期，2-4 天）：接入 is_gated。关键点：w13 权重布局需重排为 up 前 gate 后（load 时一次性）；`fused_swiglu_quant_pack` 瘦身为纯 quant+pack（silu 由 epilogue 接管，半替代非删除）；gate_up buffer 减半；逐位测试按新分工重写；精度门（三指标 + GSM8K/MMLU，epilogue 激活与现路径非逐位一致）
    - 后续上游方向：is_gated + epilogue 直出 fp8（审计项 -3.1ms 的顺路实现），可让 A2 kernel 整个消失

### 2026-07-30 档 B 收尾（is_gated 默认开启）

- 实现：`SGLANG_FLASHINFER_SM120_FP8_GATED`（现默认 True）；w13 load 后翻转为 up-first（fp8.py，B4 挂点）；A2 kernel 加 `kInputActivated` 退化为纯 quant+pack；runner 契约断言堵死翻转/env 错配；stage-2 bench 支持 `a1_fused_gated` 路径与 gated 容差带
- 组件归因（stage-2，t=1/8192）：decode gemm1 +5.3μs vs A2 -5.2μs（graph 下净 +4.4μs/层——上游 gated kernel 小 M 配对税）；prefill 净 -56μs/层（主要来自 A2 免 silu -40μs，store 减半仅 -16μs：compute-bound）
- 关键教训：**plain 路径的 silu 早已融合进 A2（当年 A2 融合），本次融合的增量收益大半已被提前吃掉**；上游 +34.3% 是对未融合基线、MPE=1024 的 GEMM 单体口径
- E2E：prefill 37558（+0.4%）；decode bs=1 6.48ms、bs=128 58.49ms（均噪声级）——microbench 担心的 decode +0.18ms 被生产 graph 的 PDL/重叠遮蔽，未兑现
- 精度：gated vs plain 数值差 ~0.7% mean-rel（silu 位置：fp32 accum vs bf16 舍入后），单测/stage-2 设 gated 容差带（2e-2/2e-3/3e-2），上游同类先例 2e-3
- **精度门已闭环（2026-07-30 用户确认签字）**：on-side GSM8K 131 题 0.832、MMLU 200 题 0.680（stem 0.643 / humanities 0.565 / social 0.735 / other 0.809）；与基线对照由用户确认无差异。默认开启生效
- commit 链：201376dfa → dc23ad0a2 → ad0f50f4f → c72a77980 → eb4d5a896 → 30556a793 → （env 默认翻转）

### decode 后续优化路线（登记）

1. 重建预算表：修复后 build 重采 A 组 nsys，校准 dense/MoE/GDN 各类目 per-step
2. dense GEMM 小 M 调优（~3.4ms/step 大头，粗算距 roofline ~2×；ncu 定位 + FI dense 小 M tile config，潜在 -0.5~1ms）
3. MoE glue 融合双子星（上游对齐）：epilogue 直出 fp8（A2 消失）+ unpermute 进 GEMM2 FINALIZE；做完 gated 在 decode 也翻正
4. 残差清尾：B14 已由 18af088bd revert；B3/守卫 ~0.05-0.1ms 待 D 组归因定夺
5. 算法级：投机解码（MTP/EAGLE）为 bs=1 的数量级杠杆，另立项目

### 2026-07-30 decode 预算表（dec-budget-01，bs=1，gated on，锚点 2195 步）

注：采集时误带 FUSED_A1=0（launch 参数失误，生产无此问题），A1 三行按 fused 修正约 -0.1ms/step；绝对值含 ~10% 窗口膨胀，比例可信。

| 类目 | 发/步 | ms/step | 占比 |
|---|---|---|---|
| dense GEMM | 160 | 3.50 | 45.8% |
| LM head GEMV（cublas bf16，~750GB/s） | 1 | 0.83 | 10.9% |
| MoE GEMM1（gated，18.8μs/发） | 40 | 0.75 | 9.9% |
| MoE GEMM2（10.0μs/发） | 40 | 0.40 | 5.2% |
| 激活 quant | 160* | 0.21* | 2.8% |
| unpermute/combine | 40 | 0.26 | 3.5% |
| GDN recurrent+conv | 60 | 0.25 | 3.2% |
| router+topk sort | 80 | 0.22 | 2.8% |
| attention | 10 | 0.17 | 2.2% |
| norm 类 | 110 | 0.18 | 2.3% |
| A1 fused glue* | 80 | ~0.09* | 1.2% |
| A2 quant_pack（gated） | 40 | 0.05 | 0.7% |

（* = 按 fused A1 修正后的估值）

**修正后的优先级**：① LM head GEMV（单项 0.83ms、带宽利用率仅 ~47%、普适、~0.4ms 上限）②MoE glue 融合（unpermute FINALIZE + gated GEMM1 小 M 税 18.8vs10.0μs，等上游或自研）③ dense GEMM 3.5ms 大头（latency-bound，暂留档）。

### 2026-07-29 档 A 完成（提前启动，未等 nightly）

- 方式：cherry-pick #4130（merge commit 92274ba1）到 pinned tag `b35396c1`，零冲突；依赖核查：97 个中间 commit 仅 #4185 触碰相关路径且只改 cuDNN 测试标记，PR 引用符号在 tag 上全部存在。分支 `adopt-pr4130`（`55b030f3`，含我们移植的 256-expert 稀疏 case ×4）已推 fork。部署仍走文件覆盖（27 文件含 2 个 Python core.py——op 绑定加了 is_gated 参数）
- 验收：flashinfer 单测 47/47 全绿（含 gated）；sglang MoE 测试 1 failed（B13 旧账，与合入前分布一致）15 passed；smoke 数值不变
- microbench（锁频 1732）：prefill 65536 相对我们补丁版再降 **4.3~12.4%**（gemm2 uniform 595→525μs）——专职调度 warp 管道卸掉了消费者 warp 的调度负担，prefill 也受益；decode cum_m=8 8.2~13.7μs（与我们补丁版 5.7~9.4μs 不可比：那批未锁频）
- E2E：prefill @4096 **37405 tok/s**（我们补丁版 36708，baseline 32372，**累计 +15.5%**）；decode bs=1 TPOT **6.45ms**（我们补丁版 6.51，baseline 6.18）
- decode 残差 +0.27ms 结构：FI glue 溢价 ~0.17ms（A/B 组"其余全部"差值，GEMM 前后 sglang kernel 的个数地板）+ 新 build 全局项 ~0.1ms（B14/B3/守卫），随档 B 及后续融合蚕食，不单独立案
- 尾巴：等含 #4130 的官方 nightly 发布后切 pinned wheel 完成正规化（pyproject:34 + bootstrap:12-14）
