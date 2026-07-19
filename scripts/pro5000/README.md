# RTX PRO 5000 Stage B 环境验证

这些命令只创建 `/home/logs/sennian/pro5000-fi-moe` 下的新环境，不修改旧环境
`/home/logs/sennian/py-venv/sglang5.14`。

该环境只安装精确版本的 `flashinfer-python` core，不安装 `flashinfer-jit-cache`。
第一次正常 smoke 使用服务器 CUDA 13.0 NVCC runtime-JIT 编译目标 kernel，第二次
正常 smoke 验证持久化编译缓存能够复用。如果新的 Stage B `.venv` 中已有旧版
bootstrap 遗留的 JIT-cache，脚本只从这个新 venv 卸载它，不触碰旧环境。

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
源；`sglang-kernel` 最终仍从 SGLang CUDA 13.0 专用源强制安装。

bootstrap 最后一行会打印 `STAGE_B_RUN_DIR`。使用打印出的目录执行：

```bash
find /home/logs/sennian/pro5000-fi-moe/runs -maxdepth 2 -type f \
  \( -name '*.json' -o -name '*.txt' -o -name '*.status' -o -name '*.stderr' \) \
  -print -exec sed -n '1,240p' {} \;
```

把上述输出完整返回。由于没有安装 JIT-cache，`smoke-no-jit.status` 预期为非零，
它只是诊断记录，不阻塞验收。两次 `smoke-normal-*.json` 都必须生成，并且每个
case 都必须满足 `calc_diff < 1e-3`。
