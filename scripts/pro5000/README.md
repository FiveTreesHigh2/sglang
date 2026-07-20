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
