# RTX PRO 5000 Stage B 环境验证

这些命令只创建 `/home/logs/sennian/pro5000-fi-moe` 下的新环境，不修改旧环境
`/home/logs/sennian/py-venv/sglang5.14`。

该环境只安装精确版本的 `flashinfer-python` core，不安装 `flashinfer-jit-cache`。
第一次正常 smoke 使用服务器 CUDA 13.0 NVCC runtime-JIT 编译目标 kernel，第二次
正常 smoke 验证持久化编译缓存能够复用。如果新的 Stage B `.venv` 中已有旧版
bootstrap 遗留的 JIT-cache，脚本只从这个新 venv 卸载它，不触碰旧环境。

Stage B `.venv` 不依赖内置 pip。所有包安装、卸载和依赖检查都使用 `uv pip --python`
指向该 venv；不要在这个 venv 中执行 `pip` 或 `python -m pip`。

## 首次 clone 并固定当前 feature commit

```bash
mkdir -p /home/logs/sennian/pro5000-fi-moe
git clone git@github.com:FiveTreesHigh2/sglang.git \
  /home/logs/sennian/pro5000-fi-moe/sglang
cd /home/logs/sennian/pro5000-fi-moe/sglang
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
git status --short --branch
```

`git status --short --branch` 可以显示 detached HEAD 的 `## HEAD (no branch)`；除此
之外不得有文件状态行。记录 `git rev-parse HEAD` 输出的完整 SHA。

## 已有 clone 更新到最新 feature commit

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
git status --short --branch
```

## 执行 bootstrap

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/bootstrap_stage_b.sh
```

## 从旧版 bootstrap 的混合依赖环境重建一次

如果旧版脚本已经在 Stage B `.venv` 中留下 CUTLASS 4.6.1/cu12 或 protobuf RC，
先保留一份可恢复的备份，再让修正后的脚本从现有 uv cache 重建。不要处理旧环境
`/home/logs/sennian/py-venv/sglang5.14`：

```bash
test -d /home/logs/sennian/pro5000-fi-moe/.venv
test ! -e /home/logs/sennian/pro5000-fi-moe/.venv-before-clean-resolve
mv /home/logs/sennian/pro5000-fi-moe/.venv \
  /home/logs/sennian/pro5000-fi-moe/.venv-before-clean-resolve

cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/bootstrap_stage_b.sh
```

该重命名不会删除文件；确认新环境验收通过前保留备份。

## 安装中断后继续

安装中断后可直接重新运行同一条 bootstrap 命令，不需要删除 `.venv`、wheel 或
cache：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/bootstrap_stage_b.sh
```

脚本会重新校验并跳过已完成的 core wheel；未完成的 `.part` 会继续下载；已有的
Python 3.12 `.venv` 和 uv cache 会被复用。`uv pip install` 会再次收敛到固定依赖，
editable 安装使用按清华、SGLang cu130 排序的双 `--index` 和 `first-index`：普通
包及构建依赖在清华命中后不会查询 SGLang 站点，缺失的 CUDA 包才使用 cu130
源；`sglang-kernel==0.4.4` 从清华 PyPI 镜像显式安装。SGLang 发布流程中，PyPI
上不带 `+cu130` 后缀的 0.4.4 就是去掉本地版本标记后的 CUDA 13.0 构建。预发布
策略使用 `if-necessary-or-explicit`，只接受依赖解析所必需或被精确版本显式指定的
预发布包。

bootstrap 最后一行会打印 `STAGE_B_RUN_DIR`。使用打印出的目录执行：

```bash
find /home/logs/sennian/pro5000-fi-moe/runs -maxdepth 2 -type f \
  \( -name '*.json' -o -name '*.txt' -o -name '*.status' -o -name '*.stderr' \) \
  -print -exec sed -n '1,240p' {} \;
```

把上述输出完整返回。由于没有安装 JIT-cache，`smoke-no-jit.status` 预期为非零，
它只是诊断记录，不阻塞验收。两次 `smoke-normal-*.json` 都必须生成，并且每个
case 都必须满足 `calc_diff < 1e-3`。

## Stage 1：FP8 MoE kernel 微基准

Stage 1 在已经通过 Stage B 的同一个 `.venv` 中，对比 FlashInfer
`moe_gemm_fp8_nt_groupwise` 和 SGLang Triton `fused_moe_kernel`。它不修改模型
代码，也不安装或升级任何 Python 包；依赖检查必须继续使用 `uv pip --python`，
不要运行 `pip` 或 `python -m pip`。

正常测试依赖 CUDA 13.0 NVCC 进行 FlashInfer runtime JIT。开始前应移除以前用于
Stage B 反向诊断的 disable-JIT 环境变量：

```bash
unset FLASHINFER_DISABLE_JIT
```

### 1. 更新到精确 feature commit

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
git status --short --branch
```

记录完整 commit SHA。`git status --short --branch` 除 detached HEAD 的状态行外不能
出现文件状态行。

### 2. 只读检查现有 venv

```bash
uv pip check \
  --python /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
```

该命令只检查依赖，不会修改环境。Stage 1 不需要重新执行 bootstrap，也不需要
安装任何包。

### 3. 先做小规模 preflight

首次运行建议先用较小的 `cum_m` 验证两个 kernel 能启动并通过正确性检查：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
export FLASHINFER_WORKSPACE_BASE=/home/logs/sennian/pro5000-fi-moe/cache/flashinfer-workspace-base
unset FLASHINFER_DISABLE_JIT
/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py \
  --operations gemm1 gemm2 \
  --profiles synthetic-skew \
  --cum-m 4096 \
  --warmup 2 \
  --iterations 5 \
  --trials 1 \
  --clock-mode default \
  --output /tmp/pro5000-stage1-preflight.json
```

首次 FlashInfer JIT 可能需要约 100 秒；脚本会持续打印 `[jit]` 阶段，不要因为这
一段时间没有 kernel 结果就中断。preflight 的每个 case 都应通过正确性检查；由于
它没有包含默认决策 shape，最终 `STAGE_1_DECISION=NOT_EVALUATED` 是预期结果。

### 4. 默认 boost 首测

preflight 通过后执行正式测试：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
unset FLASHINFER_DISABLE_JIT
bash scripts/pro5000/run_stage_1_benchmark.sh
```

wrapper 默认保留 GPU 的动态 boost，不会自动锁频，也不会安装依赖。结束时会打印：

- `STAGE_1_STATUS`：Python benchmark 退出码；
- `STAGE_1_WRAPPER_STATUS`：包含 before/after 环境检查的 wrapper 退出码；
- `STAGE_1_RUN_DIR`：本次所有产物的目录。

两个 status 都应为 `0`。用打印出的目录查看结果：

```bash
RUN_DIR="$(ls -1dt /home/logs/sennian/pro5000-fi-moe/runs/stage-1-* | head -n1)"
sed -n '1,240p' "${RUN_DIR}/benchmark.stdout.txt"
sed -n '1,320p' "${RUN_DIR}/benchmark.stderr.txt"
sed -n '1,360p' "${RUN_DIR}/benchmark.json"
```

### 5. 解读决策

正式决策只读取 `GEMM1 / uniform / cum_m=65536` 主 case：

- `GO`：默认 boost 下中位数加速不低于 30%，每轮配对加速不低于 20%，且波动
  不超过 5%；
- `NEEDS_LOCKED_RERUN`：加速落在 15%–30%、环境采样不稳定，或任一配对轮次
  不满足门槛，需要用条件锁频复测；
- `NO_GO`：环境稳定、正确性通过，但加速低于 15%；这是有效性能结论，不是脚本
  故障；
- `ERROR`：环境、JIT、kernel 启动或正确性检查失败，需要先排查错误。

### 6. 条件锁频

锁频不是首次测试的必要条件。仅当默认结果为 `NEEDS_LOCKED_RERUN`，或需要最终
验收复测时使用。执行前必须确认 GPU 0 为当前任务独占，并确认当前账号有设置和
恢复时钟的权限：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
(
  set -e
  nvidia-smi -i 0 -lgc 1732,1732
  trap 'nvidia-smi -i 0 -rgc' EXIT
  STAGE1_CLOCK_MODE=locked \
    bash scripts/pro5000/run_stage_1_benchmark.sh
)
```

subshell 的 `trap` 会在正常结束或出错时恢复默认 graphics clock。这里的
CUDA toolkit `13.0`、GPU compute capability `12.0` 和运行时 SM clock
`1732 MHz` 是三个不同概念：前两者分别描述编译工具链和 GPU 指令架构，MHz
描述测试时的频率。

### 7. 返回本地分析的产物

请返回本次 `STAGE_1_RUN_DIR` 中的以下文件：

- `benchmark.json`、`benchmark.stdout.txt`、`benchmark.stderr.txt`；
- `environment.json`、`environment-after.json`；
- `nvidia-smi-before.txt`、`nvidia-smi-after.txt`；
- `pip-check-before.txt`、`pip-check-after.txt`；
- `packages-before.txt`、`packages-after.txt`。

这些文件用于同时判断 kernel 正确性、性能、GPU 运行状态和 venv 是否被意外改变。

## Stage 2：完整 FP8 MoE runner 基准

Stage 2 测量的是完整 MoE runner，不是 Stage 1 的单独 GEMM。计时范围包含 input
quant、expert permute、A-scale layout、GEMM1、SwiGLU、第二次 quant/layout、
GEMM2 和 unpermute/combine。FlashInfer 与 Triton 使用完全相同的 FP8 权重、
block scale、hidden states、top-k IDs 和 top-k weights。

默认 workload 使用 Qwen3.5-A3B 的单卡目标 shape：`E=256`、hidden size
`2048`、intermediate size `512`、`top_k=8`。`tokens=8192` 对应
`routed_rows=65536`，是固定的 chunked-prefill 主判定 case。

wrapper 不安装或升级依赖，不执行 `pip`，不锁定 GPU 频率，也不修改旧 venv。
只读依赖检查继续使用 `uv pip --python`。当前 checkout 是 editable 安装，更新
feature commit 后通常不需要重复执行 bootstrap 或安装 SGLang。

### 1. 更新到精确 feature commit

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
git status --short --branch
```

### 2. 运行 CPU 契约测试

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3

"${VENV_PY}" -m pytest \
  test/registered/unit/test_pro5000_stage_2.py \
  -q
```

该测试不启动目标 GPU kernel，检查 CLI、合法 top-k 路由、固定主判定 case、结果
schema、wrapper 不修改依赖且不锁频。

### 3. 先运行缩小的完整 runner preflight

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
export FLASHINFER_WORKSPACE_BASE=/home/logs/sennian/pro5000-fi-moe/cache/flashinfer-workspace-base
unset FLASHINFER_DISABLE_JIT

/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  --tokens 1 8 \
  --top-k 8 \
  --profiles synthetic-skew \
  --warmup 2 \
  --trials 1 \
  --iterations 5 \
  --check-cuda-graph \
  --output-json /tmp/pro5000-stage2-preflight.json
```

脚本首先构造一次共享的 `E=256` blockwise FP8 权重，并每完成 32 个 expert 打印
一条 `[prepare]` 进度。首次目标 kernel JIT 也可能耗时；看到持续的权重量化或 JIT
阶段时不要中断。这个缩小命令不含 `tokens=8192, uniform`，所以最终
`STAGE_2_DECISION=NOT_EVALUATED` 是预期结果；所有列出的 correctness 和 CUDA
Graph 必须为 `PASS`。

### 4. 执行默认 Stage 2 benchmark

preflight 通过后运行：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
unset FLASHINFER_DISABLE_JIT
bash scripts/pro5000/run_stage_2_benchmark.sh
```

默认 wrapper 依次执行 CUTLASS 目标 shape runtime preflight、CUDA Graph 动态路由
replay，以及以下两种 routing profile 的完整 runner 对照：

```text
tokens:   1 8 128 8192 16384
profiles: uniform synthetic-skew
backends: triton flashinfer_sm120_fp8；CUTLASS preflight 可用时再加入 cutlass
```

CUTLASS 实际调用不可用时会记录 `CUTLASS_UNAVAILABLE` 和完整 traceback，但不会
跳过 FlashInfer/Triton correctness 和性能测试，也不会把失败伪造成 0 ms。

结束时输出：

- `STAGE_2_STATUS`：Python benchmark 退出码；
- `STAGE_2_WRAPPER_STATUS`：包含 before/after 环境检查的 wrapper 退出码；
- `STAGE_2_RUN_DIR`：本次完整产物目录。

两个 status 都应为 `0`。查看结果：

```bash
RUN_DIR="$(ls -1dt /home/logs/sennian/pro5000-fi-moe/runs/stage-2-* | head -n1)"
sed -n '1,360p' "${RUN_DIR}/stdout.txt"
sed -n '1,360p' "${RUN_DIR}/stderr.txt"
sed -n '1,1200p' "${RUN_DIR}/benchmark.json"
```

### 5. 解读 Stage 2 决策

- `GO`：所有完整 runner correctness 和 CUDA Graph 通过；固定
  `tokens=8192, uniform` case 相对 Triton 至少加速 10%；所有 `tokens=1/8`
  decode case 的最坏回退不超过 5%。
- `FUNCTIONAL_ONLY`：功能与 CUDA Graph 正确，但主 prefill 收益不足 10%，或
  decode 最坏回退超过 5%。backend 保持显式实验选项，不自动按 token 数切换到
  Triton。
- `NO_GO`：任一 correctness 或 CUDA Graph 检查失败。
- `NOT_EVALUATED`：自定义参数没有包含固定主 case 或 decode case；用于小规模
  preflight，不是正式性能结论。

component profile 记录 production FlashInfer runner 内各 CUDA 操作的时间，排除
Python 与 allocator 的 host 时间；完整 runner 的 backend latency 才是性能判定
依据。

### 6. GPU 频率检查

Stage 2 wrapper 本身不会锁频。运行前后可用以下只读命令检查当前频率和 P-state：

```bash
nvidia-smi \
  --query-gpu=index,pstate,clocks.current.sm,clocks.max.sm,clocks.applications.sm \
  --format=csv,noheader,nounits
```

只有默认 boost 结果接近边界、并且确认 GPU 独占后，才另行进行用户批准的锁频
复测；锁频不属于本 wrapper。

### 7. 打包结果返回本地

```bash
RUN_DIR=/home/logs/sennian/pro5000-fi-moe/runs/stage-2-<timestamp>-<commit>
tar -C "$(dirname "${RUN_DIR}")" -czf \
  /tmp/pro5000-stage2.tar.gz "$(basename "${RUN_DIR}")"
```

返回压缩包即可。它包含 `benchmark.json`、stdout/stderr、before/after 环境、
`uv pip check`、package freeze 和 `nvidia-smi` 记录。
