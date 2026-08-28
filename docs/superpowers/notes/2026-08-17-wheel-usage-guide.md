# SM120 FP8 优化版 SGLang + FlashInfer Wheel 使用指南

日期：2026-08-17 · 适用硬件：NVIDIA RTX PRO 5000/6000（SM120/121）· 验证模型：Qwen3.5-35B-A3B-FP8 / Qwen3.6（同架构）

---

## 一、交付物

| 文件 | 说明 |
|---|---|
| `flashinfer_python-0.6.15.dev20260716+aliopt1-py3-none-any.whl`（18MB） | 官方 nightly-20260716 + 上游 #4130（MoeScheduler 重写 + is_gated）+ 自研 FINALIZE 融合。源码分支：`FiveTreesHigh2/flashinfer @ moe-finalize-fusion`（`42d419d5`） |
| `sglang-0.0.0.dev15518+g58001a6f6.d20260817-cp312-cp312-linux_x86_64.whl`（19MB） | feature 分支全部优化：FI MoE runner（gated/FINALIZE/FUSED_A1 默认开）、GDN 扫参入口、qkvzba/gate/z 去物化、精度守卫。源码：`feat/flashinfer-sm120-fp8-moe`（`58001a6f6`） |

**当前存放位置**：`33.243.206.227` 容器内 `/home/logs/sennian/wheels/`

**下载方式**（三选一）：
```bash
# a) 同机其他容器/环境：直接从共享数据盘拷贝
cp /home/logs/sennian/wheels/*.whl <你的目录>/

# b) 其他机器：scp（需有该机 ssh 权限）
scp admin@33.243.206.227:/home/logs/sennian/wheels/*.whl ./

# c) 正式分发（待办）：上传内网 pypi / GitHub Release 后 pip 直装
```

## 二、环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| OS / 架构 | Linux x86_64 | sglang wheel 为 `cp312-linux_x86_64` |
| Python | **3.12** | 与 wheel tag 绑定 |
| GPU | SM120/121（RTX PRO 5000/6000 等） | 优化 kernel 仅在此架构生效 |
| CUDA 工具链 | **CUDA ≥ 12.9（建议 13.0）且 `nvcc` 可用** | FlashInfer kernel 为运行时 JIT 编译，缺 nvcc 会在首个请求处报错 |
| 核心依赖 | torch 2.11.0+cu130、tvm_ffi、triton 3.6 | 随 sglang 依赖解析自动安装（需内网 pypi 源可达） |

CUDA 工具链设置示例（若 nvcc 不在默认 PATH）：
```bash
export CUDA_HOME=/path/to/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# 可选：JIT 编译缓存位置（默认 ~/.cache/flashinfer；多环境共享或磁盘规划时显式指定）
export FLASHINFER_WORKSPACE_BASE=/path/to/flashinfer-workspace
```

## 三、安装

### 场景 A：全新环境
```bash
uv venv myenv --python 3.12
uv pip install --python myenv/bin/python3 \
  flashinfer_python-0.6.15.dev20260716+aliopt1-py3-none-any.whl \
  sglang-0.0.0.dev15518+g58001a6f6.d20260817-cp312-cp312-linux_x86_64.whl
# 依赖（torch 等数 GB）自动从配置的 pypi 源解析安装
```

### 场景 B：已有 sglang 环境原位升级
```bash
uv pip install --force-reinstall --no-deps \
  flashinfer_python-0.6.15.dev20260716+aliopt1-py3-none-any.whl \
  sglang-0.0.0.dev15518+g58001a6f6.d20260817-cp312-cp312-linux_x86_64.whl
```

### 安装校验
```bash
python3 -c "import flashinfer; print(flashinfer.__version__)"
# 期望：0.6.15.dev20260716+aliopt1   ← 带 +aliopt1 才是优化版
python3 -c "from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise as f; import inspect; assert 'finalize_out' in inspect.signature(f).parameters; print('finalize API OK')"
```

## 四、启动与优化开关

### 启动命令模板（关键参数）
```bash
SGLANG_GDN_CHUNK_H_BV=64 SGLANG_GDN_CHUNK_H_NUM_WARPS=4 SGLANG_GDN_CHUNK_H_NUM_STAGES=2 \
SGLANG_GDN_WU_BK=128 SGLANG_GDN_WU_BV=128 SGLANG_GDN_WU_NUM_STAGES=3 \
python3 -m sglang.launch_server \
  --model-path <模型路径> --host <IP> --port <端口> --tp-size 1 \
  --fp8-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_sm120_fp8 \
  <其余业务参数照常>
```

两个 backend 参数与 5 个 GDN 环境变量为**必需**：
- `--fp8-gemm-backend flashinfer_cutlass`：dense FP8 GEMM 走 CUTLASS（缺省会回退 Triton，损失 ~7%）
- `--moe-runner-backend flashinfer_sm120_fp8`：启用本优化的 MoE 路径
- `SGLANG_GDN_*` 五个：GDN kernel 扫参胜者配置（代码默认值不是胜者值，须显式传）

### 优化开关总表（wheel 内默认值）

| 环境变量 | 默认 | 作用 | 关闭方式（回滚） |
|---|---|---|---|
| `SGLANG_FLASHINFER_SM120_FP8_FUSED_A1` | **1** | quant+permute+scale-pack 三合一 | 置 0 |
| `SGLANG_FLASHINFER_SM120_FP8_GATED` | **1** | SiLU 融进 GEMM1 epilogue（权重加载时自动翻转布局） | 置 0 |
| `SGLANG_FLASHINFER_SM120_FP8_MOE_FINALIZE` | **1** | unpermute+combine 融进 GEMM2 epilogue（decode 形状自动启用） | 置 0 |
| `SGLANG_GDN_CHUNK_H_BV/WARPS/STAGES` | 32/4/2（非胜者） | GDN chunk_h tile 配置 | 传 64/4/2 为优化值 |
| `SGLANG_GDN_WU_BK/BV/STAGES` | 64/64/3（非胜者） | GDN recompute_w_u 配置 | 传 128/128/3 为优化值 |

三个 FP8 开关都是**运行时环境变量**——出现问题时置 0 重启即回退，无需换包。

### 生效确认（server 启动日志，三行缺一不可）
```bash
grep -iE "GEMM1 mode|GEMM2 store mode|A1 prepare mode" <server日志>
# flashinfer_sm120_fp8 GEMM1 mode=gated (fused SwiGLU epilogue)
# flashinfer_sm120_fp8 A1 prepare mode=fused
# flashinfer_sm120_fp8 GEMM2 store mode=finalize (fused unpermute+combine)
```

## 五、首次启动：JIT 预热（重要）

FlashInfer kernel 为运行时 JIT：**每个新环境/新 GPU 架构/清缓存后的首次触发要现场编译**（cute_sm120 主模块 ~2 分钟；sampling、各 attention 形状变体按需再编 20-40s）。未预热直接压测会出现首批请求 RT 60s+ 的假性能问题（实测踩过：caption-bench Max RT 69.7s，预热后 13.8s）。

标准预热序列（server 就绪后、放量前执行）：
```bash
# 1. greedy 请求：触发主链路（MoE/attention/GDN）编译
curl -s http://<IP>:<端口>/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<served-model-name>","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'

# 2. 带采样参数的请求：触发 sampling kernel 编译（temperature=0 不会触发！）
curl -s http://<IP>:<端口>/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<served-model-name>","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0.7,"top_p":0.9}'
```

编译产物按源码 hash 缓存（`FLASHINFER_WORKSPACE_BASE` 或 `~/.cache/flashinfer`），同环境后续重启秒级命中，无需重复预热。

## 六、性能预期（RTX PRO 5000 单卡，Qwen3.5-35B-A3B-FP8 实测）

| 指标 | 官方基线（Triton MoE） | 优化版 | 提升 |
|---|---|---|---|
| prefill 吞吐 @4096 输入 | 32372 tok/s | **37582 tok/s** | **+16.1%** |
| decode TPOT bs=1 | 6.18 ms | 6.32 ms | 基本持平（-2%内） |
| decode TPOT bs=128 | 57.96 ms | 57.99 ms | 持平 |
| 高并发 TTFT（128 并发） | 8112 ms | ~7150 ms | -12% |
| 精度（GSM8K / MMLU） | 基线 | 0.832 / 0.680 | 持平（已签字） |

## 七、已知限制

1. **仅 SM120/121**：其他架构会走 sglang 原生路径或直接报不支持
2. sglang wheel 未编 gRPC Rust 扩展（`SGLANG_BUILD_RUST_EXTS=none` 构建）——`grpc_mode` 不可用，HTTP 服务不受影响
3. 验证范围：tp=1、Qwen3.5/3.6 A3B（256 experts / top-8 / (128,128) 块量化 FP8）；其他 MoE 形状理论兼容（tile 规则自适应），建议先跑 smoke 与精度对照
4. sglang 版本号 `0.0.0.dev15518` 为无 tag 分支的 scm fallback，`+g58001a6f6` 携带源码 commit；如需规范版本号可在分支打 tag 后重建

## 八、故障排查速查

| 症状 | 原因 | 处理 |
|---|---|---|
| 启动报 `FlashInfer requires GPUs with sm75 or higher` | JIT 阶段找不到 nvcc/CUDA | 设置 CUDA_HOME/PATH（见第二节） |
| 首批请求 RT 60s+ | 未预热，JIT 编译撞上请求 | 执行第五节预热序列 |
| 日志缺 `mode=gated/finalize` 行 | env 被显式置 0，或 moe-runner-backend 不对 | 检查启动参数与环境变量 |
| 数值/精度异常 | 排查期需要隔离变量 | 三个 FP8 开关逐个置 0 二分，并回传现象 |
| 报 `gated mode and the up-first w13 flip must agree` | GATED env 与权重翻转状态错配（通常是自定义加载路径绕过了 fp8.py 后处理） | 确认权重经标准 `process_weights_after_loading` 加载 |
