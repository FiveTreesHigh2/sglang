# RTX PRO 5000 Stage B 环境验证

这些命令只创建 `/home/logs/sennian/pro5000-fi-moe` 下的新环境，不修改旧环境
`/home/logs/sennian/py-venv/sglang5.14`。

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

`git status` 必须为空；记录 `git rev-parse HEAD` 输出的完整 SHA。

## 执行 bootstrap

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
bash scripts/pro5000/bootstrap_stage_b.sh
```

bootstrap 最后一行会打印 `STAGE_B_RUN_DIR`。使用打印出的目录执行：

```bash
find /home/logs/sennian/pro5000-fi-moe/runs -maxdepth 2 -type f \
  \( -name '*.json' -o -name '*.txt' -o -name '*.status' -o -name '*.stderr' \) \
  -print -exec sed -n '1,240p' {} \;
```

把上述输出完整返回。`smoke-no-jit.status` 为非零只表示官方 AOT cache 未命中；
只要两次 `smoke-normal-*.json` 都生成且 `calc_diff < 1e-3`，Stage B 仍可使用
NVCC runtime-JIT 模式继续验收。
