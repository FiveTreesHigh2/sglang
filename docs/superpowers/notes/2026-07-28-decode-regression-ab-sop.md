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
nsys stats -r cuda_gpu_kern_sum /home/logs/sennian/nsys-log/dec-ab-A/report1.nsys-rep --format csv \
  > /home/logs/sennian/nsys-log/dec-ab-A/kern_sum.csv

# 步数锚点：qkvzba 融合 kernel 仅 decode 路径运行、每步 30 次（30 GDN 层）
# steps = instances / 30
grep "qkvzba" /home/logs/sennian/nsys-log/dec-ab-A/kern_sum.csv

# per-step GPU 总量 = 总 kernel 时长 / steps
awk -F, 'NR>1 && $2 ~ /^[0-9]+$/ {s+=$2} END {printf "total_kernel_ms=%.1f\n", s/1e6}' \
  /home/logs/sennian/nsys-log/dec-ab-A/kern_sum.csv

# MoE 相关条目（A/C 组看 FI 家族，B 组看 fused_moe_kernel 家族）
grep -iE "quant_scatter|swiglu_quant|unpermute|moe_permute|cute|fused_moe|moe_sum|moe_align" \
  /home/logs/sennian/nsys-log/dec-ab-A/kern_sum.csv
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
