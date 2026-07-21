# FlashInfer SM120 FP8 MoE 服务 Prefill 阶段 A 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 FlashInfer 源码的阶段 A 中，建立不可误复用的服务 A/B harness，并把 `flashinfer_sm120_fp8` GEMM1 前的 quant、payload permute 和 A-scale layout 融合为一个通用 CUDA JIT kernel，以真实 `bench_serving` 吞吐判断是否达到 4096 长度至少 10% 的目标。

**Architecture:** 客户端 harness 连接用户手动启动的单个服务器，核验 `/server_info`、commit、依赖和 A1 日志 marker，再用固定长度与三种子生成配对 manifest。生产路径保留 `moe_permute_prepare` 的 device-side 路由准备，用一个新 CUDA kernel 从 BF16 token 直接产生 expert-packed E4M3 payload 与 FlashInfer FP32 A-scale；环境开关允许同一 commit 比较 legacy/fused A1。阶段 A 结束后只有服务 GO 才把 fused A1 改为默认，否则保持 `FUNCTIONAL_ONLY` 并另写阶段 B scheduler 计划。

**Tech Stack:** Python 3.12、PyTorch 2.11、CUDA 13.0、SM120/SM121、SGLang JIT `load_jit`、CUDA C++、TVM-FFI、FlashInfer `0.6.15.dev20260716`、pytest/unittest、`sglang.bench_serving`。

## Global Constraints

- 目标硬件为单卡 NVIDIA RTX PRO 5000 72GB Blackwell；CUDA capability 只允许 `(12, 0)` 或 `(12, 1)`。
- 服务器 Python 固定为 `/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3`；安装和依赖检查只使用 `uv pip --python`，本计划不新增 Python 依赖。
- FlashInfer GEMM API、E4M3 payload、FP32 scale、`(1, 128, 128)` granularity、MN-major A-scale、NT weight、BF16 `out=` 均保持不变。
- dense FP8 GEMM backend 固定为 `flashinfer_cutlass`；只改变 MoE runner backend 或 A1 实验开关。
- 正式服务 A/B 使用同一 GPU UUID 和 `default-unlocked` 频率策略；harness 将两者写入 run key。若之后改为 application clocks 锁频，三种 server 模式必须全部重跑并统一标记为 `application-clocks-locked`，不得混用结果。
- 不修改 FlashInfer 源码，不支持多卡 TP/EP、DeepEP，不实现 Triton 静默 fallback，不特化模型名或 input length。
- 服务端显式使用 `--chunked-prefill-size 8192`；正式 workload 固定为 input lengths `4096 6144 14336 30720 63488`、output length `1`、100 prompts、seeds `17 29 43`。
- 正式 GO：4096 配对吞吐提升中位数 `>= 0.10` 且每个 seed 不回退；其他四个长度的中位数和每个 seed 均 `> 0`；所有 run `completed == 100`。
- 完整 runner 阈值保持 `calc_diff < 0.005`、`symmetric_diff < 1e-4`、`normalized_rmse < 0.01`。
- A1 payload 首先要求与固定环境实际选择的 legacy JIT v2 reference bitwise 一致；不得在测试失败时自动放宽契约。
- `bench_one_batch` 和 Nsight 不属于本计划的常规门槛；只有组件与服务结果矛盾且无法归因时才另行诊断。
- 保留用户未跟踪文件 `test/registered/moe/debug_flashinfer_sm120_fp8_stagewise.py`，不得修改、删除或提交。
- 所有代码改动使用 TDD：先运行目标测试并看到预期 RED，再做最小实现、运行 GREEN、最后独立提交。

---

## 文件结构

### 新文件

- `scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py`：连接当前服务器、核验配置、运行单 backend 的五长度三种子 capture，并比较 Triton/FlashInfer manifests。
- `test/registered/unit/test_pro5000_sm120_fp8_serving.py`：服务 harness 的纯 CPU 契约测试。
- `python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_quant_scatter.cuh`：A1 BF16 quant、route scatter、FlashInfer A-scale pack 和 padding clear CUDA kernel。

### 修改文件

- `python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py`：在现有缓存 module 中注册 A1 low-level wrapper。
- `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`：增加 A1 Python adapter、输入输出 contract 与可复用 `out/out_scale`。
- `python/sglang/srt/environ.py`：注册 `SGLANG_FLASHINFER_SM120_FP8_FUSED_A1`，阶段 A 默认 `False`。
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`：增加一次性模式日志，并在 legacy/fused A1 之间显式选择。
- `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`：A1 bitwise、padding、route、完整 runner、wiring 与 CUDA Graph GPU 测试。
- `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`：环境开关和默认值 CPU 测试。
- `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`：识别 legacy/fused A1 的组件 trace 和稳定 rollup。
- `test/registered/unit/test_pro5000_stage_2.py`：runner component schema 的 CPU 测试。
- `scripts/pro5000/README.md`：服务器命令、manifest、回滚和阶段 A 结论。

---

### Task 1: 建立服务 A/B harness 与不可误复用的结果契约

**Files:**
- Create: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py`
- Create: `test/registered/unit/test_pro5000_sm120_fp8_serving.py`

**Interfaces:**
- Consumes: 活跃服务器的 `GET /server_info`、`python -m sglang.bench_serving` 最后一条 JSONL 结果、当前 Git checkout 和已安装 package metadata。
- Produces: `capture` manifest schema 1、`compare` decision JSON、`build_run_key()`、`verify_server_info()`、`compare_manifests()`。

- [ ] **Step 1: 写 constants、server contract 和 run-key RED 测试**

创建 CPU 测试文件，使用动态 import 加载脚本：

```python
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("pro5000_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def server_info(backend="triton"):
    return {
        "served_model_name": "qwen35-fp8",
        "enable_metrics": True,
        "moe_runner_backend": backend,
        "fp8_gemm_runner_backend": "flashinfer_cutlass",
        "chunked_prefill_size": 8192,
        "tp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "pp_size": 1,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.9,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "model_path": "/models/qwen",
        "dtype": "auto",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "version": "0.5.14",
    }


def test_defaults_and_run_key_include_backend_commit_seed():
    bench = load_bench()
    assert bench.INPUT_LENGTHS == (4096, 6144, 14336, 30720, 63488)
    assert bench.SEEDS == (17, 29, 43)
    runtime = {
        "flashinfer_version": "0.6.15.dev20260716",
        "torch_version": "2.11.0", "torch_cuda_version": "13.0",
        "sglang_kernel_version": "0.4.4",
        "python_executable": "/venv/bin/python3",
        "dense_fp8_backend": "flashinfer_cutlass",
        "chunked_prefill_size": 8192,
    }
    first = bench.build_run_key(
        backend="triton", commit="a" * 40, input_length=4096,
        num_prompts=100, seed=17, server_args_hash="b" * 64,
        a1_mode="not_applicable", flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test", gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    second = bench.build_run_key(
        backend="flashinfer_sm120_fp8", commit="a" * 40,
        input_length=4096, num_prompts=100, seed=17,
        server_args_hash="b" * 64,
        a1_mode="fused", flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test", gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    assert first != second
    legacy = bench.build_run_key(
        backend="flashinfer_sm120_fp8", commit="a" * 40,
        input_length=4096, num_prompts=100, seed=17,
        server_args_hash="b" * 64,
        a1_mode="legacy", flashinfer_artifact_sha256="f" * 64,
        gpu_uuid="GPU-test", gpu_frequency_strategy="default-unlocked",
        runtime_fingerprint=runtime,
    )
    assert legacy != second


def test_server_info_requires_exact_backend_dense_backend_and_chunk():
    bench = load_bench()
    snapshot = bench.verify_server_info(server_info(), "triton")
    assert snapshot["moe_runner_backend"] == "triton"
    for field, value in (
        ("moe_runner_backend", "flashinfer_sm120_fp8"),
        ("fp8_gemm_runner_backend", "triton"),
        ("chunked_prefill_size", 4096),
        ("tp_size", 2),
        ("enable_metrics", False),
        ("mem_fraction_static", 0.8),
    ):
        bad = server_info()
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            bench.verify_server_info(bad, "triton")
```

- [ ] **Step 2: 写 manifest pairing 与正式 GO 判定 RED 测试**

加入精确 case builder；每个 seed 都有结果，防止只比较聚合平均：

```python
def manifest(backend, ratios):
    cases = []
    for input_length in (4096, 6144, 14336, 30720, 63488):
        for seed, ratio in zip((17, 29, 43), ratios[input_length]):
            base = 1000.0
            cases.append({
                "input_length": input_length,
                "seed": seed,
                "completed": 100,
                "total_input_tokens": input_length * 100,
                "input_throughput": base if backend == "triton" else base * ratio,
                "median_ttft_ms": 1.0,
            })
    return {
        "schema_version": 1,
        "metadata": {
            "moe_backend": backend,
            "pairing_hash": "c" * 64,
            "sglang_commit": "d" * 40,
            "flashinfer_artifact_sha256": "f" * 64,
            "flashinfer_version": "0.6.15.dev20260716",
            "torch_version": "2.11.0",
            "torch_cuda_version": "13.0",
            "sglang_kernel_version": "0.4.4",
            "python_executable": "/venv/bin/python3",
            "gpu_uuid": "GPU-test",
            "gpu_frequency_strategy": "default-unlocked",
            "dense_fp8_backend": "flashinfer_cutlass",
            "num_prompts": 100,
        },
        "cases": cases,
    }


def test_compare_requires_4096_ten_percent_and_no_seed_regression():
    bench = load_bench()
    ratios = {
        4096: (1.10, 1.11, 1.12),
        6144: (1.01, 1.02, 1.03),
        14336: (1.01, 1.02, 1.03),
        30720: (1.01, 1.02, 1.03),
        63488: (1.01, 1.02, 1.03),
    }
    result = bench.compare_manifests(
        manifest("triton", ratios),
        manifest("flashinfer_sm120_fp8", ratios),
    )
    assert result["decision"] == "GO"
    assert result["by_input_length"]["4096"]["median_speedup"] >= 0.10

    ratios[4096] = (0.99, 1.11, 1.12)
    assert bench.compare_manifests(
        manifest("triton", ratios),
        manifest("flashinfer_sm120_fp8", ratios),
    )["decision"] == "FUNCTIONAL_ONLY"


def test_compare_rejects_config_drift_or_incomplete_case():
    bench = load_bench()
    ratios = {length: (1.2, 1.2, 1.2) for length in bench.INPUT_LENGTHS}
    triton = manifest("triton", ratios)
    flashinfer = manifest("flashinfer_sm120_fp8", ratios)
    flashinfer["metadata"]["pairing_hash"] = "e" * 64
    with pytest.raises(ValueError, match="pairing_hash"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["pairing_hash"] = "c" * 64
    flashinfer["metadata"]["sglang_commit"] = "e" * 40
    with pytest.raises(ValueError, match="sglang_commit"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["sglang_commit"] = "d" * 40
    flashinfer["metadata"]["gpu_frequency_strategy"] = "application-clocks-locked"
    with pytest.raises(ValueError, match="gpu_frequency_strategy"):
        bench.compare_manifests(triton, flashinfer)
    flashinfer["metadata"]["gpu_frequency_strategy"] = "default-unlocked"
    flashinfer["cases"][0]["completed"] = 99
    with pytest.raises(ValueError, match="completed"):
        bench.compare_manifests(triton, flashinfer)
```

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_sm120_fp8_serving.py -q
```

Expected: FAIL，原因是脚本不存在。

- [ ] **Step 4: 实现纯函数契约**

新脚本顶部必须包含以下固定 schema；`pairing_hash` 排除唯一允许变化的 MoE backend，
而 `server_args_hash` 包含它：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen

import torch


INPUT_LENGTHS = (4096, 6144, 14336, 30720, 63488)
SEEDS = (17, 29, 43)
NUM_PROMPTS = 100
FIXED_SERVER_FIELDS = (
    "model_path", "served_model_name", "dtype", "quantization",
    "kv_cache_dtype", "tp_size", "dp_size", "ep_size", "pp_size",
    "disable_radix_cache", "mem_fraction_static", "attention_backend",
    "prefill_attention_backend", "reasoning_parser", "tool_call_parser",
    "chunked_prefill_size", "fp8_gemm_runner_backend", "moe_runner_backend",
    "enable_metrics",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def query_single_gpu_uuid() -> str:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    uuids = [value.strip() for value in output if value.strip()]
    if len(uuids) != 1:
        raise ValueError(f"stage A requires exactly one visible GPU, got {uuids}")
    return uuids[0]


def verify_server_info(info: dict[str, Any], expected_backend: str) -> dict[str, Any]:
    required = {
        "moe_runner_backend": expected_backend,
        "fp8_gemm_runner_backend": "flashinfer_cutlass",
        "chunked_prefill_size": 8192,
        "tp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "pp_size": 1,
        "disable_radix_cache": True,
        "enable_metrics": True,
        "mem_fraction_static": 0.9,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
    }
    for field, expected in required.items():
        if info.get(field) != expected:
            raise ValueError(
                f"server_info {field} must be {expected!r}, got {info.get(field)!r}"
            )
    if not info.get("model_path"):
        raise ValueError("server_info model_path is required")
    if not info.get("served_model_name"):
        raise ValueError("server_info served_model_name is required")
    return {field: info.get(field) for field in FIXED_SERVER_FIELDS}


def build_run_key(*, backend: str, commit: str, input_length: int,
                  num_prompts: int, seed: int, server_args_hash: str,
                  a1_mode: str, flashinfer_artifact_sha256: str,
                  gpu_uuid: str, gpu_frequency_strategy: str,
                  runtime_fingerprint: dict[str, Any]) -> str:
    return canonical_hash({
        "backend": backend, "commit": commit, "input_length": input_length,
        "num_prompts": num_prompts, "seed": seed,
        "server_args_hash": server_args_hash,
        "a1_mode": a1_mode,
        "flashinfer_artifact_sha256": flashinfer_artifact_sha256,
        "gpu_uuid": gpu_uuid,
        "gpu_frequency_strategy": gpu_frequency_strategy,
        "runtime_fingerprint": runtime_fingerprint,
    })


def compare_manifests(triton: dict[str, Any], flashinfer: dict[str, Any]) -> dict[str, Any]:
    if triton["metadata"]["pairing_hash"] != flashinfer["metadata"]["pairing_hash"]:
        raise ValueError("pairing_hash mismatch")
    for field in (
        "sglang_commit", "flashinfer_artifact_sha256", "flashinfer_version",
        "torch_version", "torch_cuda_version", "sglang_kernel_version",
        "python_executable", "gpu_uuid", "gpu_frequency_strategy",
        "dense_fp8_backend", "num_prompts",
    ):
        if triton["metadata"].get(field) != flashinfer["metadata"].get(field):
            raise ValueError(f"{field} mismatch")
    def indexed(manifest):
        result = {}
        for case in manifest["cases"]:
            if case["completed"] != NUM_PROMPTS:
                raise ValueError(f"completed must be {NUM_PROMPTS}: {case}")
            key = (case["input_length"], case["seed"])
            if key in result:
                raise ValueError(f"duplicate input-length/seed pair: {key}")
            result[key] = case
        return result
    left, right = indexed(triton), indexed(flashinfer)
    expected = {(length, seed) for length in INPUT_LENGTHS for seed in SEEDS}
    if set(left) != expected or set(right) != expected:
        raise ValueError("manifests must contain every input-length/seed pair exactly once")
    by_length = {}
    go = True
    for length in INPUT_LENGTHS:
        samples = []
        for seed in SEEDS:
            a, b = left[(length, seed)], right[(length, seed)]
            if a["total_input_tokens"] != b["total_input_tokens"]:
                raise ValueError("total_input_tokens mismatch")
            samples.append(b["input_throughput"] / a["input_throughput"] - 1.0)
        median = statistics.median(samples)
        if length == 4096:
            passed = median >= 0.10 and min(samples) >= 0.0
        else:
            passed = median > 0.0 and min(samples) > 0.0
        go = go and passed
        by_length[str(length)] = {
            "paired_speedups": samples,
            "median_speedup": median,
            "min_speedup": min(samples),
            "passed": passed,
        }
    return {"decision": "GO" if go else "FUNCTIONAL_ONLY", "by_input_length": by_length}
```

- [ ] **Step 5: 实现 `capture`，固定 metadata、bench command 和原子写入**

`capture` 不启动或关闭服务器。加入以下核心实现；所有 subprocess 必须使用
`sys.executable`，保证进入同一 venv：

```python
def fetch_server_info(host: str, port: int) -> dict[str, Any]:
    with urlopen(f"http://{host}:{port}/server_info", timeout=10) as response:
        return json.load(response)


def parse_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty benchmark output: {path}")
    raw = json.loads(lines[-1])
    return {name: raw[name] for name in (
        "input_throughput", "median_ttft_ms", "completed", "total_input_tokens"
    )}


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    info = fetch_server_info(args.host, args.port)
    snapshot = verify_server_info(info, args.expected_backend)
    commit = git_commit(args.repo)
    gpu_uuid = query_single_gpu_uuid()
    server_args_hash = canonical_hash(snapshot)
    paired_snapshot = dict(snapshot)
    paired_snapshot.pop("moe_runner_backend")
    pairing_hash = canonical_hash(paired_snapshot)
    runtime_fingerprint = {
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "sglang_kernel_version": importlib.metadata.version("sglang-kernel"),
        "python_executable": sys.executable,
        "dense_fp8_backend": snapshot["fp8_gemm_runner_backend"],
        "chunked_prefill_size": 8192,
    }
    metadata = {
        "moe_backend": args.expected_backend,
        "a1_mode": args.a1_mode,
        "dense_fp8_backend": snapshot["fp8_gemm_runner_backend"],
        "sglang_commit": commit,
        "sglang_version": info["version"],
        "flashinfer_artifact_sha256": args.flashinfer_artifact_sha256,
        **runtime_fingerprint,
        "gpu_uuid": gpu_uuid,
        "gpu_frequency_strategy": args.gpu_frequency_strategy,
        "server_args": snapshot,
        "server_args_hash": server_args_hash,
        "pairing_hash": pairing_hash,
        "chunked_prefill_size": 8192,
        "num_prompts": NUM_PROMPTS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    cases = []
    result_dir = args.output.parent / f"{args.expected_backend}-{commit[:12]}"
    result_dir.mkdir(parents=True, exist_ok=True)

    def bench_command(*, input_length: int, num_prompts: int, seed: int,
                      output: Path) -> list[str]:
        return [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang", "--dataset-name", "random",
            "--dataset-path", str(args.dataset_path),
            "--random-input-len", str(input_length),
            "--random-output-len", "1", "--random-range-ratio", "1",
            "--num-prompts", str(num_prompts),
            "--output-file", str(output), "--seed", str(seed),
            "--port", str(args.port), "--host", args.host,
            "--model", snapshot["served_model_name"], "--flush-cache",
        ]

    warmup_output = result_dir / f"warmup-{args.a1_mode}.jsonl"
    warmup_command = bench_command(
        input_length=4096, num_prompts=1, seed=0, output=warmup_output
    )
    if not warmup_output.exists():
        subprocess.run(warmup_command, check=True)
    warmup = parse_last_jsonl(warmup_output)
    if warmup["completed"] != 1:
        raise ValueError(f"warmup completed must be 1: {warmup}")
    metadata["warmup"] = {
        "command": warmup_command,
        "output_file": str(warmup_output),
        **warmup,
    }

    for input_length in INPUT_LENGTHS:
        for seed in SEEDS:
            run_key = build_run_key(
                backend=args.expected_backend, commit=commit,
                input_length=input_length, num_prompts=NUM_PROMPTS, seed=seed,
                server_args_hash=server_args_hash,
                a1_mode=args.a1_mode,
                flashinfer_artifact_sha256=args.flashinfer_artifact_sha256,
                gpu_uuid=gpu_uuid,
                gpu_frequency_strategy=args.gpu_frequency_strategy,
                runtime_fingerprint=runtime_fingerprint,
            )
            output = result_dir / f"{run_key}.jsonl"
            command = bench_command(
                input_length=input_length, num_prompts=NUM_PROMPTS,
                seed=seed, output=output,
            )
            if not output.exists():
                subprocess.run(command, check=True)
            result = parse_last_jsonl(output)
            if result["completed"] != NUM_PROMPTS:
                raise ValueError(f"completed must be {NUM_PROMPTS}: {result}")
            cases.append({
                "run_key": run_key, "input_length": input_length, "seed": seed,
                "command": command, "output_file": str(output), **result,
            })
            write_json_atomic(args.output, {
                "schema_version": 1, "metadata": metadata, "cases": cases,
            })
    if args.expected_backend == "flashinfer_sm120_fp8":
        marker = f"flashinfer_sm120_fp8 A1 prepare mode={args.a1_mode}"
        if marker not in args.server_log.read_text(errors="replace"):
            raise ValueError(f"server log missing A1 marker: {marker}")
    return {"schema_version": 1, "metadata": metadata, "cases": cases}
```

CLI 使用以下精确接口，并验证 SHA256、A1 mode 和 reference backend：

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--host", required=True)
    capture.add_argument("--port", type=int, default=30000)
    capture.add_argument(
        "--expected-backend", choices=("triton", "flashinfer_sm120_fp8"),
        required=True,
    )
    capture.add_argument(
        "--a1-mode", choices=("not_applicable", "legacy", "fused"),
        required=True,
    )
    capture.add_argument("--server-log", type=Path)
    capture.add_argument("--repo", type=Path, required=True)
    capture.add_argument("--dataset-path", type=Path, required=True)
    capture.add_argument("--flashinfer-artifact-sha256", required=True)
    capture.add_argument(
        "--gpu-frequency-strategy",
        choices=("default-unlocked", "application-clocks-locked"),
        required=True,
    )
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--triton", type=Path, required=True)
    compare.add_argument("--flashinfer", type=Path, required=True)
    compare.add_argument(
        "--allow-reference-backend", choices=("flashinfer_sm120_fp8",)
    )
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        digest = args.flashinfer_artifact_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            parser.error("--flashinfer-artifact-sha256 must be 64 hex characters")
        if args.expected_backend == "triton" and args.a1_mode != "not_applicable":
            parser.error("Triton capture requires --a1-mode not_applicable")
        if args.expected_backend == "flashinfer_sm120_fp8":
            if args.a1_mode not in ("legacy", "fused") or args.server_log is None:
                parser.error("FlashInfer capture requires legacy/fused A1 mode and --server-log")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        write_json_atomic(args.output, run_capture(args))
        return 0
    reference = json.loads(args.triton.read_text())
    candidate = json.loads(args.flashinfer.read_text())
    expected_reference = args.allow_reference_backend or "triton"
    if reference["metadata"]["moe_backend"] != expected_reference:
        raise ValueError(
            f"reference backend must be {expected_reference}, got "
            f"{reference['metadata']['moe_backend']}"
        )
    if candidate["metadata"]["moe_backend"] != "flashinfer_sm120_fp8":
        raise ValueError("candidate backend must be flashinfer_sm120_fp8")
    result = compare_manifests(reference, candidate)
    result["reference_backend"] = expected_reference
    result["candidate_backend"] = "flashinfer_sm120_fp8"
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 运行 CPU GREEN、compile 和 diff 检查**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_sm120_fp8_serving.py -q
python3 -m compileall -q scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py
git diff --check
```

Expected: 全部 PASS；后两条无输出。

- [ ] **Step 7: 提交 harness**

```bash
git add scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py \
  test/registered/unit/test_pro5000_sm120_fp8_serving.py
git commit -m "bench: add SM120 FP8 serving A/B harness"
```

---

### Task 2: 用 GPU 测试锁定 A1 adapter 的 bitwise、padding 与通用路由契约

**Files:**
- Modify: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: `sglang_per_token_group_quant_fp8`、`moe_permute` 和 `pack_flashinfer_sm120_fp8_scale` 作为 legacy reference。
- Produces: `fused_quant_scatter_pack_flashinfer_sm120_fp8(hidden_states, topk_ids, src2dst, m_indptr, *, out=None, out_scale=None)` 的 RED/GREEN 行为契约。

- [ ] **Step 1: 写 bitwise reference 与实际 dispatch RED 测试**

在 SM120 test class 中新增：

```python
def test_fused_a1_matches_actual_legacy_v2_reference(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        fused_quant_scatter_pack_flashinfer_sm120_fp8,
        pack_flashinfer_sm120_fp8_scale,
    )
    from sglang.kernels.ops.quantization import fp8_kernel

    torch.manual_seed(101)
    tokens, hidden, experts = 8, 512, 8
    topk_ids = torch.tensor(
        [[3, 3], [0, 7], [1, 1], [6, 2], [7, 7], [4, 0], [5, 6], [2, 3]],
        device="cuda", dtype=torch.int32,
    )
    hidden_states = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16
    )
    with patch.object(
        fp8_kernel,
        "sgl_per_token_group_quant_8bit_jit_v2",
        wraps=fp8_kernel.sgl_per_token_group_quant_8bit_jit_v2,
    ) as v2:
        ref_q, ref_scale = fp8_kernel.sglang_per_token_group_quant_fp8(
            hidden_states, 128
        )
    self.assertEqual(v2.call_count, 1)
    ref_packed, src2dst, m_indptr = moe_permute(ref_q, topk_ids, experts)
    ref_scale_fi = pack_flashinfer_sm120_fp8_scale(
        ref_scale, topk_ids, src2dst, m_indptr, source_is_packed=False
    )
    actual_q, actual_scale = fused_quant_scatter_pack_flashinfer_sm120_fp8(
        hidden_states, topk_ids, src2dst, m_indptr
    )
    torch.testing.assert_close(
        actual_q.view(torch.uint8), ref_packed.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, ref_scale_fi, rtol=0, atol=0)
```

- [ ] **Step 2: 写 route profiles、output reuse 和 `gap==0` RED 测试**

先在 SM120 test class 中增加精确 reference helper，并覆盖 `top_k=1/2/8`、重复
expert、空 expert、均匀和长尾路由：

```python
def _assert_fused_a1_matches_legacy(self, hidden_states, topk_ids, experts):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        fused_quant_scatter_pack_flashinfer_sm120_fp8,
        pack_flashinfer_sm120_fp8_scale,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    ref_q, ref_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
    ref_packed, src2dst, m_indptr = moe_permute(ref_q, topk_ids, experts)
    ref_scale_fi = pack_flashinfer_sm120_fp8_scale(
        ref_scale, topk_ids, src2dst, m_indptr, source_is_packed=False
    )
    actual_q, actual_scale = fused_quant_scatter_pack_flashinfer_sm120_fp8(
        hidden_states, topk_ids, src2dst, m_indptr
    )
    torch.testing.assert_close(
        actual_q.view(torch.uint8), ref_packed.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, ref_scale_fi, rtol=0, atol=0)


def test_fused_a1_route_profiles(self):
    torch.manual_seed(102)
    experts = 8
    profiles = (
        torch.tensor([[0], [1], [2], [3]], dtype=torch.int32),
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32),
        torch.tensor(
            [[0, 0, 0, 1, 1, 2, 3, 7], [7, 7, 6, 5, 4, 3, 2, 1]],
            dtype=torch.int32,
        ),
    )
    for host_topk_ids in profiles:
        topk_ids = host_topk_ids.cuda()
        hidden_states = torch.randn(
            (topk_ids.shape[0], 256), device="cuda", dtype=torch.bfloat16
        )
        with self.subTest(top_k=topk_ids.shape[1]):
            self._assert_fused_a1_matches_legacy(
                hidden_states, topk_ids, experts
            )
```

同一 `out/out_scale` 再先填 NaN/非零字节，然后用两套路由 replay：

```python
def test_fused_a1_reused_outputs_clear_dynamic_padding(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        flashinfer_sm120_m_padded,
        fused_quant_scatter_pack_flashinfer_sm120_fp8,
        pack_flashinfer_sm120_fp8_scale,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    torch.manual_seed(103)
    tokens, hidden, experts, top_k = 4, 256, 8, 2
    routes = (
        torch.tensor([[0, 0], [0, 1], [1, 1], [1, 1]], device="cuda", dtype=torch.int32),
        torch.tensor([[7, 7], [6, 7], [5, 6], [4, 7]], device="cuda", dtype=torch.int32),
    )
    out = torch.empty((tokens * top_k, hidden), device="cuda", dtype=torch.float8_e4m3fn)
    out_scale = torch.full(
        (hidden // 128, flashinfer_sm120_m_padded(tokens * top_k, experts)),
        float("nan"), device="cuda", dtype=torch.float32,
    )
    for topk_ids in routes:
        hidden_states = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
        ref_q, ref_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
        ref_packed, src2dst, m_indptr = moe_permute(ref_q, topk_ids, experts)
        expected_scale = pack_flashinfer_sm120_fp8_scale(
            ref_scale, topk_ids, src2dst, m_indptr, source_is_packed=False
        )
        returned_q, returned_scale = fused_quant_scatter_pack_flashinfer_sm120_fp8(
            hidden_states, topk_ids, src2dst, m_indptr,
            out=out, out_scale=out_scale,
        )
        self.assertEqual(returned_q.data_ptr(), out.data_ptr())
        self.assertEqual(returned_scale.data_ptr(), out_scale.data_ptr())
        torch.testing.assert_close(returned_q.view(torch.uint8), ref_packed.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(returned_scale, expected_scale, rtol=0, atol=0)
        padding = expected_scale == 0
        self.assertTrue(bool(padding.any()))
        self.assertTrue(bool((returned_scale[padding] == 0).all()))
```

最后加入 `T=4,E=1,top_k=1` case，使最后一个 expert 的 `gap==0`：

```python
def test_fused_a1_zero_padding_gap_is_safe(self):
    torch.manual_seed(104)
    hidden_states = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros((4, 1), device="cuda", dtype=torch.int32)
    self._assert_fused_a1_matches_legacy(hidden_states, topk_ids, experts=1)
    torch.cuda.synchronize()
```

该测试必须正常返回，且不能产生 CUDA illegal instruction/divide-by-zero。

- [ ] **Step 3: 写 adapter contract RED 测试**

用合法 CUDA tensors 逐项构造错误输入，并用下列精确测试锁定 dtype、shape、device、
contiguous、output reuse 和 16B alignment 错误：

```python
def test_fused_a1_adapter_contract(self):
    from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
        fused_quant_scatter_pack_flashinfer_sm120_fp8,
    )

    hidden = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    topk = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]], device="cuda", dtype=torch.int32
    )
    _, src2dst, m_indptr = moe_permute(hidden, topk, 8)
    routes = topk.numel()
    scale_shape = (2, ((routes + 3 * 8) // 4) * 4)
    valid_out = torch.empty(
        (routes, 256), device="cuda", dtype=torch.float8_e4m3fn
    )
    valid_scale = torch.empty(scale_shape, device="cuda", dtype=torch.float32)
    misaligned_storage = torch.empty(
        (valid_scale.numel() + 1,), device="cuda", dtype=torch.float32
    )
    misaligned_scale = misaligned_storage[1:].view(scale_shape)

    cases = (
        ("hidden_states", dict(hidden_states=hidden.float())),
        ("hidden_states", dict(hidden_states=hidden[:, :129].contiguous())),
        ("topk_ids", dict(topk_ids=topk.to(torch.int64))),
        ("src2dst", dict(src2dst=src2dst[:-1])),
        ("m_indptr", dict(m_indptr=m_indptr.to(torch.int64))),
        ("CUDA", dict(topk_ids=topk.cpu())),
        ("out", dict(out=valid_out[:, :128])),
        ("out", dict(out=valid_out.view(torch.uint8))),
        ("out_scale", dict(out_scale=valid_scale[:, :-1])),
        ("out_scale", dict(out_scale=valid_scale.to(torch.float64))),
        ("aligned", dict(out_scale=misaligned_scale)),
    )
    defaults = dict(
        hidden_states=hidden, topk_ids=topk, src2dst=src2dst,
        m_indptr=m_indptr, out=valid_out, out_scale=valid_scale,
    )
    for message, override in cases:
        arguments = defaults | override
        with self.subTest(message=message), self.assertRaisesRegex(
            (TypeError, ValueError), message
        ):
            fused_quant_scatter_pack_flashinfer_sm120_fp8(**arguments)
```

另用合法输入固定成功输出 contract：

```python
self.assertEqual(actual_q.shape, (topk_ids.numel(), hidden))
self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
self.assertTrue(actual_q.is_contiguous())
self.assertEqual(
    actual_scale.shape,
    (hidden // 128, ((topk_ids.numel() + 3 * experts) // 4) * 4),
)
self.assertEqual(actual_scale.dtype, torch.float32)
self.assertEqual(actual_scale.data_ptr() % 16, 0)
```

- [ ] **Step 4: 提交并推送 RED 测试**

由于本地没有 SM120，先把只含测试的 RED 提交推到 feature branch：

```bash
git add test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "test: specify fused SM120 FP8 A1 packing"
git push origin feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 5: 服务器同步并确认 RED 原因唯一**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'fused_a1' -q -s
```

Expected: FAIL，唯一主因是无法 import
`fused_quant_scatter_pack_flashinfer_sm120_fp8`；已有 reference 准备不得先失败。

---

### Task 3: 实现 A1 CUDA JIT kernel 与 Python adapter

**Files:**
- Create: `python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_quant_scatter.cuh`
- Modify: `python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py`
- Modify: `python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py`
- Test: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: Task 2 的 exact function signature 和 legacy reference。
- Produces: low-level `flashinfer_sm120_fp8_quant_scatter_pack()` 与 public adapter `fused_quant_scatter_pack_flashinfer_sm120_fp8()`。

- [ ] **Step 1: 新增 CUDA params、token block 和 E 个 padding blocks**

新 header 复用 A2 include 与向量类型，但 grid 的主区是 `num_tokens`，不是
`num_routes`。核心 device 代码固定为：

```cpp
struct FlashInferSm120Fp8QuantScatterParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ topk_ids;
  const int32_t* __restrict__ src2dst;
  const int32_t* __restrict__ m_indptr;
  int64_t hidden_dim;
  int64_t m_padded;
  uint32_t num_tokens;
  uint32_t top_k;
  uint32_t num_experts;
};

template <bool kUsePDL>
__global__ __launch_bounds__(1024, 2)
void flashinfer_sm120_fp8_quant_scatter_kernel(
    const FlashInferSm120Fp8QuantScatterParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  const uint32_t num_groups = params.hidden_dim / kGroupSize;
  PDLWaitPrimary<kUsePDL>();

  if (blockIdx.x < params.num_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t work_id = threadIdx.x / kWorkThreads;
    const uint32_t lane = threadIdx.x % kWorkThreads;
    const bool valid_group = work_id < num_groups;
    const uint32_t vector_id = work_id * kWorkThreads + lane;
    OutputVec quantized;
    float scale = 0.0f;
    if (valid_group) {
      InputVec values;
      values.load(params.input + static_cast<int64_t>(token) * params.hidden_dim, vector_id);
      float local_absmax = 1e-10f;
      float converted[8];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const auto [x, y] = cast<fp32x2_t>(values[i]);
        converted[2 * i] = x;
        converted[2 * i + 1] = y;
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(x), fabsf(y)));
      }
      constexpr uint32_t kMask = (1u << kWorkThreads) - 1u;
      const uint32_t mask = kMask << ((threadIdx.x % device::kWarpThreads) / kWorkThreads * kWorkThreads);
      local_absmax = warp::reduce_max<kWorkThreads>(local_absmax, mask);
      scale = local_absmax * (1.0f / math::FP8_E4M3_MAX);
      const float multiplier = __fdividef(math::FP8_E4M3_MAX, local_absmax);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        quantized[i] = pack_fp8(
            converted[2 * i] * multiplier,
            converted[2 * i + 1] * multiplier);
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    if (valid_group) {
      for (uint32_t choice = 0; choice < params.top_k; ++choice) {
        const uint32_t route = token * params.top_k + choice;
        const uint32_t dst = params.src2dst[route];
        const uint32_t expert = params.topk_ids[route];
        const uint32_t start = params.m_indptr[expert];
        const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
        quantized.store(
            params.output + static_cast<int64_t>(dst) * params.hidden_dim,
            vector_id);
        if (lane == 0) {
          const uint32_t column = aligned + dst - start;
          params.output_scale[static_cast<int64_t>(work_id) * params.m_padded + column] = scale;
        }
      }
    }
    return;
  }

  const uint32_t expert = blockIdx.x - params.num_tokens;
  const uint32_t start = params.m_indptr[expert];
  const uint32_t end = params.m_indptr[expert + 1];
  const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
  const uint32_t valid_end = aligned + end - start;
  const uint32_t next = expert + 1 == params.num_experts
      ? params.m_padded
      : ((end + 3u * (expert + 1)) / 4u) * 4u;
  const uint32_t gap = next - valid_end;
  PDLTriggerSecondary<kUsePDL>();
  if (gap != 0) {
    for (uint32_t i = threadIdx.x; i < num_groups * gap; i += blockDim.x) {
      const uint32_t group = i / gap;
      const uint32_t column = valid_end + i % gap;
      params.output_scale[static_cast<int64_t>(group) * params.m_padded + column] = 0.0f;
    }
  }
}
```

- [ ] **Step 2: 增加 host wrapper 与严格 shape checks**

在同一 header 定义完整 host wrapper：

```cpp
template <bool kUsePDL>
struct FlashInferSm120Fp8QuantScatterKernel {
  static constexpr auto kernel =
      flashinfer_sm120_fp8_quant_scatter_kernel<kUsePDL>;

  static void run(const tvm::ffi::TensorView hidden,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView output_scale,
                  const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView src2dst,
                  const tvm::ffi::TensorView m_indptr) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto T = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"hidden_dim"};
    auto M = SymbolicSize{"num_routes"};
    auto G = SymbolicSize{"num_groups"};
    auto MP = SymbolicSize{"m_padded"};
    auto K = SymbolicSize{"top_k"};
    auto EP = SymbolicSize{"num_experts_plus_one"};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, N}).with_dtype<bf16_t>().with_device(device).verify(hidden);
    TensorMatcher({M, N}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({G, MP}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(src2dst);
    TensorMatcher({EP}).with_dtype<int32_t>().with_device(device).verify(m_indptr);

    RuntimeCheck(T.unwrap() * K.unwrap() == M.unwrap(),
                 "topk_ids must contain one entry per routed row");
    RuntimeCheck(K.unwrap() > 0, "top_k must be positive");
    RuntimeCheck(N.unwrap() > 0 && N.unwrap() % 128 == 0,
                 "hidden_dim must be positive and divisible by 128");
    RuntimeCheck(EP.unwrap() >= 2,
                 "m_indptr must have shape [num_experts + 1]");

    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto num_routes = static_cast<uint32_t>(M.unwrap());
    const auto top_k = static_cast<uint32_t>(K.unwrap());
    const auto hidden_dim = N.unwrap();
    const auto num_groups = static_cast<uint32_t>(hidden_dim / 128);
    const auto num_experts = static_cast<uint32_t>(EP.unwrap() - 1);
    const auto expected_m_padded =
        ((static_cast<int64_t>(num_routes) +
          3 * static_cast<int64_t>(num_experts)) /
         4) *
        4;
    RuntimeCheck(G.unwrap() == num_groups,
                 "invalid number of scale groups");
    RuntimeCheck(MP.unwrap() == expected_m_padded,
                 "invalid FlashInfer m_padded dimension");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output_scale.data_ptr()) % 16 == 0,
        "FlashInfer output_scale must be 16-byte aligned");

    const auto num_threads = ((num_groups * 16u + 31u) / 32u) * 32u;
    RuntimeCheck(num_threads > 0 && num_threads <= 1024,
                 "hidden_dim exceeds single-CTA kernel capacity");
    const auto grid = num_tokens + num_experts;
    const auto params = FlashInferSm120Fp8QuantScatterParams{
        .input = static_cast<const bf16_t*>(hidden.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .src2dst = static_cast<const int32_t*>(src2dst.data_ptr()),
        .m_indptr = static_cast<const int32_t*>(m_indptr.data_ptr()),
        .hidden_dim = hidden_dim,
        .m_padded = expected_m_padded,
        .num_tokens = num_tokens,
        .top_k = top_k,
        .num_experts = num_experts,
    };
    LaunchKernel(grid, num_threads, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
```

不能增加 `out.zero_()` 或 host 读取 `m_indptr`。

- [ ] **Step 3: 把 A1 wrapper 注册到现有缓存 JIT module**

修改 loader：

```python
return load_jit(
    "flashinfer_sm120_fp8_moe",
    *args,
    cuda_files=[
        "moe/flashinfer_sm120_fp8_swiglu_quant.cuh",
        "moe/flashinfer_sm120_fp8_quant_scatter.cuh",
    ],
    cuda_wrappers=[
        ("silu_quant_pack", f"FlashInferSm120Fp8SiluQuantPackKernel<{args}>::run"),
        ("quant_scatter_pack", f"FlashInferSm120Fp8QuantScatterKernel<{args}>::run"),
    ],
)


def flashinfer_sm120_fp8_quant_scatter_pack(
    hidden_states, output, output_scale, topk_ids, src2dst, m_indptr
) -> None:
    module = _jit_flashinfer_sm120_fp8_moe_module(is_arch_support_pdl())
    module.quant_scatter_pack(
        hidden_states, output, output_scale, topk_ids, src2dst, m_indptr
    )
```

不要给整个 module 添加 `--use_fast_math`。需要对齐的除法已由 header 中显式
`__fdividef` 固定，避免改变现有 A2 的精确 `expf` 行为。

- [ ] **Step 4: 实现 public Python adapter**

在现有 op 文件增加：

```python
def fused_quant_scatter_pack_flashinfer_sm120_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise TypeError("hidden_states must be a 2D bfloat16 tensor")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if hidden_states.shape[1] == 0 or hidden_states.shape[1] % 128 != 0:
        raise ValueError("hidden_states last dimension must be positive and divisible by 128")
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2:
        raise TypeError("topk_ids must be a 2D int32 tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.shape[0] != hidden_states.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("topk_ids must have one non-empty routing row per token")
    routes = topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.ndim != 1:
        raise TypeError("src2dst must be a 1D int32 tensor")
    if src2dst.numel() != routes:
        raise ValueError("src2dst must contain one entry per routed slot")
    if not src2dst.is_contiguous():
        raise ValueError("src2dst must be contiguous")
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    if m_indptr.numel() < 2:
        raise ValueError("m_indptr must have shape [num_experts + 1]")
    if not m_indptr.is_contiguous():
        raise ValueError("m_indptr must be contiguous")
    tensors = (hidden_states, topk_ids, src2dst, m_indptr)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("all fused A1 inputs must be CUDA tensors")
    if any(t.device != hidden_states.device for t in tensors[1:]):
        raise ValueError("all fused A1 inputs must be on the same device")
    experts, hidden = m_indptr.numel() - 1, hidden_states.shape[1]
    output_shape = (routes, hidden)
    scale_shape = (hidden // 128, flashinfer_sm120_m_padded(routes, experts))
    if out is None:
        out = torch.empty(output_shape, device=hidden_states.device, dtype=torch.float8_e4m3fn)
    if out.shape != output_shape or out.dtype != torch.float8_e4m3fn or not out.is_contiguous():
        raise ValueError(f"out must be contiguous float8_e4m3fn with shape {output_shape}")
    if out_scale is None:
        out_scale = torch.empty(scale_shape, device=hidden_states.device, dtype=torch.float32)
    if out_scale.shape != scale_shape or out_scale.dtype != torch.float32 or not out_scale.is_contiguous():
        raise ValueError(f"out_scale must be contiguous float32 with shape {scale_shape}")
    if out.device != hidden_states.device or out_scale.device != hidden_states.device:
        raise ValueError("outputs and fused A1 inputs must be on the same device")
    if out_scale.data_ptr() % 16:
        raise ValueError("FlashInfer A-scale output must be 16-byte aligned")
    flashinfer_sm120_fp8_quant_scatter_pack(
        hidden_states, out, out_scale, topk_ids, src2dst, m_indptr
    )
    return out, out_scale
```

- [ ] **Step 5: 本地静态检查、提交并推送实现**

```bash
python3 -m compileall -q \
  python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py
git diff --check
git add python/sglang/jit_kernel/csrc/moe/flashinfer_sm120_fp8_quant_scatter.cuh \
  python/sglang/jit_kernel/flashinfer_sm120_fp8_moe.py \
  python/sglang/kernels/ops/moe/flashinfer_sm120_fp8.py
git commit -m "feat: fuse SM120 FP8 GEMM1 input packing"
git push origin feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 6: 服务器同步并运行 Task 2 GPU tests，确认 GREEN**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'fused_a1' -q -s
```

Expected: 全部 PASS；bitwise mismatch count 为 0；`gap==0` case 无 CUDA 错误。

- [ ] **Step 7: 若服务器发现实现错误，用独立 fix 提交收敛**

失败时只修改实现，不改 bitwise/padding contract；本地提交
`fix: match fused SM120 FP8 A1 contract`，推送、服务器 detached 同步并重跑 Step 6，直到
测试全部 PASS。若必须考虑容差降级，停止本计划并回到已批准设计的 5.4 人工审核门槛。

---

### Task 4: 增加实验开关并把 fused A1 接入 production runner

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py`
- Modify: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`
- Modify: `test/registered/moe/test_flashinfer_sm120_fp8_moe.py`

**Interfaces:**
- Consumes: Task 3 adapter；`moe_permute_prepare(topk_ids, num_experts) -> (m_indptr, src2dst)`。
- Produces: `_use_fused_a1() -> bool`、一次性日志 marker、同 commit 下 legacy/fused A1 production paths。

- [ ] **Step 1: 写默认关闭和缓存行为 CPU RED 测试**

```python
def test_fused_a1_env_defaults_off_and_reads_explicit_value(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner import flashinfer_sm120_fp8 as runner

    monkeypatch.delenv("SGLANG_FLASHINFER_SM120_FP8_FUSED_A1", raising=False)
    runner._use_fused_a1.cache_clear()
    assert envs.SGLANG_FLASHINFER_SM120_FP8_FUSED_A1.get() is False
    assert runner._use_fused_a1() is False
    monkeypatch.setenv("SGLANG_FLASHINFER_SM120_FP8_FUSED_A1", "1")
    runner._use_fused_a1.cache_clear()
    assert runner._use_fused_a1() is True
```

- [ ] **Step 2: 写 runner fused/legacy 精确调用 RED 测试**

在 GPU test 中用同一个 runner case 分别 patch `_use_fused_a1`：

```python
def test_full_runner_uses_fused_a1_only_when_enabled(self):
    from sglang.srt.layers.moe.moe_runner import flashinfer_sm120_fp8 as runner
    topk_ids = torch.arange(16, device="cuda", dtype=torch.int32).remainder(16).view(8, 2)
    dispatch, config, quant_info, _, _ = _make_runner_case(8, 2, topk_ids)

    with patch.object(runner, "_use_fused_a1", return_value=True), \
         patch.object(runner, "fused_quant_scatter_pack_flashinfer_sm120_fp8", wraps=runner.fused_quant_scatter_pack_flashinfer_sm120_fp8) as fused, \
         patch.object(runner, "sglang_per_token_group_quant_fp8", wraps=runner.sglang_per_token_group_quant_fp8) as quant, \
         patch.object(runner, "pack_flashinfer_sm120_fp8_scale", wraps=runner.pack_flashinfer_sm120_fp8_scale) as pack, \
         patch.object(runner, "moe_permute", wraps=runner.moe_permute) as permute, \
         patch.object(runner, "moe_permute_prepare", wraps=runner.moe_permute_prepare) as prepare:
        runner.fused_experts_none_to_flashinfer_sm120_fp8(dispatch, quant_info, config)
    self.assertEqual((fused.call_count, prepare.call_count), (1, 1))
    self.assertEqual((quant.call_count, pack.call_count, permute.call_count), (0, 0, 0))

    with patch.object(runner, "_use_fused_a1", return_value=False), \
         patch.object(runner, "fused_quant_scatter_pack_flashinfer_sm120_fp8", wraps=runner.fused_quant_scatter_pack_flashinfer_sm120_fp8) as fused, \
         patch.object(runner, "moe_permute_prepare", wraps=runner.moe_permute_prepare) as prepare:
        runner.fused_experts_none_to_flashinfer_sm120_fp8(dispatch, quant_info, config)
    self.assertEqual((fused.call_count, prepare.call_count), (0, 0))
```

- [ ] **Step 3: 提交并推送 runner RED 测试**

```bash
git add test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "test: specify fused SM120 FP8 A1 runner path"
git push origin feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 4: 服务器同步并确认 CPU/GPU RED**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py \
  -k fused_a1 -q
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py \
  -k 'full_runner_uses_fused_a1' -q -s
```

Expected: FAIL，分别因为 env symbol、runner symbol/wiring 尚不存在。

- [ ] **Step 5: 注册 env、日志 marker 和两个显式分支**

在 `envs` 增加：

```python
SGLANG_FLASHINFER_SM120_FP8_FUSED_A1 = EnvBool(False)
```

runner 顶部增加 imports 和一次性选择：

```python
import logging

from sglang.jit_kernel.moe_permute_prepare import moe_permute_prepare
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _use_fused_a1() -> bool:
    enabled = envs.SGLANG_FLASHINFER_SM120_FP8_FUSED_A1.get()
    logger.info(
        "flashinfer_sm120_fp8 A1 prepare mode=%s",
        "fused" if enabled else "legacy",
    )
    return enabled
```

把当前 A1 段替换为：

```python
if _use_fused_a1():
    m_indptr, src2dst = moe_permute_prepare(
        topk_ids=topk_ids,
        num_experts=quant_info.w13_weight.shape[0],
    )
    packed_hidden, a1_scale_fi = (
        fused_quant_scatter_pack_flashinfer_sm120_fp8(
            hidden_states,
            topk_ids,
            src2dst,
            m_indptr,
        )
    )
else:
    q_hidden, q_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
    packed_hidden, src2dst, m_indptr = moe_permute(
        q_hidden,
        topk_ids,
        quant_info.w13_weight.shape[0],
    )
    a1_scale_fi = pack_flashinfer_sm120_fp8_scale(
        q_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=False,
    )
```

adapter import 必须位于模块顶部，不能放在 per-layer 热路径函数内。

- [ ] **Step 6: 让完整正确性和 CUDA Graph 测试明确覆盖 fused A1**

在 `test_full_runner_correctness` 和 `test_cuda_graph_replays_new_hidden_and_routing` 的
runner 调用外层 patch：

```python
with patch.object(flashinfer_runner, "_use_fused_a1", return_value=True):
    actual = flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
        dispatch, quant_info, config
    )
```

CUDA Graph warmup、capture、replay 和 eager reference 全部处于同一个 patch scope，防止
capture 和 replay 走不同分支。再保留一个 `return_value=False` 的 legacy smoke，证明旧路径
仍可运行。

- [ ] **Step 7: 本地静态检查、提交并推送 runner 实现**

```bash
python3 -m compileall -q \
  python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py
git diff --check
git add python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/moe_runner/flashinfer_sm120_fp8.py \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py
git commit -m "feat: route SM120 FP8 MoE through fused A1 prepare"
git push origin feat/flashinfer-sm120-fp8-moe
```

- [ ] **Step 8: 服务器同步并运行完整 GREEN**

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py -q -s
```

Expected: 全部 PASS；完整 runner 四组 case 仍满足既有阈值；CUDA Graph replay PASS。

- [ ] **Step 9: 服务器失败时只修实现并重跑完整 GREEN**

不得通过移除 fused-path call-count、关闭 CUDA Graph case 或放宽完整 runner 阈值修复失败。
修复提交使用 `fix: correct fused SM120 FP8 A1 runner wiring`；推送后服务器重新执行
`git fetch`、`git switch --detach origin/feat/flashinfer-sm120-fp8-moe`，并再次运行 config
pytest 与完整 `test_flashinfer_sm120_fp8_moe.py` pytest 两条命令。

---

### Task 5: 扩展 runner component benchmark，准确归因 fused A1

**Files:**
- Modify: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
- Modify: `test/registered/unit/test_pro5000_stage_2.py`

**Interfaces:**
- Consumes: Task 4 两条 A1 production paths。
- Produces: 三种稳定 component path：`a1_legacy_a2_legacy`、`a1_legacy_a2_fused`、`a1_fused_a2_fused`，rollup keys 保持不变。

- [ ] **Step 1: 写三路径 detail/trace RED fixtures**

新增 fully-fused detail：

```python
fully_fused = {
    "moe_permute_prepare": 0.02,
    "fused_quant_scatter_pack_gemm1": 0.04,
    "gemm1": 0.10,
    "fused_swiglu_quant_pack_gemm2": 0.08,
    "gemm2": 0.20,
    "unpermute_combine": 0.07,
}
profile = bench.build_component_profile(fully_fused)
self.assertEqual(profile["path"], "a1_fused_a2_fused")
self.assertAlmostEqual(profile["rollup_ms"]["gemm1_input_prepare"], 0.06)
self.assertAlmostEqual(profile["rollup_ms"]["gemm2_input_prepare"], 0.08)
```

fully-fused trace 固定为：

```python
(
    "moe_permute_prepare",
    "fused_quant_scatter_pack_gemm1",
    "gemm1",
    "fused_swiglu_quant_pack_gemm2",
    "gemm2",
    "unpermute_combine",
)
```

精确 call counts 必须是 `prepare=1,fused_a1=1,quant=0,moe_permute=0,pack=0,gemm=2,
fused_a2=1,unpermute=1`。混合缺 stage、多调一次 fused A1 或 trace 顺序错误均断言
`ValueError`。

- [ ] **Step 2: 运行 CPU 测试确认 RED**

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_2.py -q
```

Expected: FAIL，因为现有 schema 只识别 A2 legacy/fused。

- [ ] **Step 3: 实现三路径稳定 schema**

保留已有 detail keys，新增：

```python
FUSED_A1_DETAIL_KEYS = frozenset((
    "moe_permute_prepare",
    "fused_quant_scatter_pack_gemm1",
))
FUSED_A1_A2_DETAIL_KEYS = frozenset((
    "gemm1", "fused_swiglu_quant_pack_gemm2", "gemm2",
    "unpermute_combine",
)) | FUSED_A1_DETAIL_KEYS
```

`build_component_profile` 对 fully-fused case 使用：

```python
if keys == FUSED_A1_A2_DETAIL_KEYS:
    path = "a1_fused_a2_fused"
    gemm1_input_prepare = (
        detail_ms["moe_permute_prepare"]
        + detail_ms["fused_quant_scatter_pack_gemm1"]
    )
    gemm2_input_prepare = detail_ms["fused_swiglu_quant_pack_gemm2"]
```

旧两种路径分别重命名为 `a1_legacy_a2_legacy` 与 `a1_legacy_a2_fused`；rollup keys 仍为：

```python
(
    "gemm1_input_prepare", "gemm1", "gemm2_input_prepare",
    "gemm2", "unpermute_combine",
)
```

- [ ] **Step 4: 扩展 profiler hooks**

在 `originals` 增加：

```python
"moe_permute_prepare": getattr(flashinfer_runner, "moe_permute_prepare", None),
"fused_a1": getattr(
    flashinfer_runner,
    "fused_quant_scatter_pack_flashinfer_sm120_fp8",
    None,
),
```

分类计数增加 `prepare` 和 `fused_a1`。存在时分别 patch 成：

```python
recorded("moe_permute_prepare", "prepare", originals["moe_permute_prepare"])
recorded("fused_quant_scatter_pack_gemm1", "fused_a1", originals["fused_a1"])
```

trace 决定实际 path，不能根据环境变量文本推断；每个 iteration 清空 trace/counts，且
观察到的 path 在所有 iterations 内必须相同。

- [ ] **Step 5: 运行 CPU GREEN 并提交**

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_2.py -q
python3 -m compileall -q scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py
git diff --check
git add scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  test/registered/unit/test_pro5000_stage_2.py
git commit -m "bench: profile fused SM120 FP8 A1 prepare"
```

Expected: tests PASS；compile/diff check 无输出。

---

### Task 6: 本地回归、服务器 GPU preflight 与完整 runner 归因

**Files:**
- Read: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py`
- Produce on server: `/home/logs/sennian/pro5000-fi-moe/runs/stage-a1-*`

**Interfaces:**
- Consumes: Tasks 1–5 全部提交。
- Produces: fused A1 correctness/CUDA Graph 证据、legacy/fused A1 完整 runner 组件差异；这是服务 A/B 前硬门槛。

- [ ] **Step 1: 本地运行全部 CPU tests 与静态检查**

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_sm120_fp8_serving.py \
  test/registered/unit/test_pro5000_stage_2.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
python3 -m compileall -q \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py
git diff --check
git status --short
```

Expected: tests PASS；只显示用户原有未跟踪 debug 文件。

- [ ] **Step 2: 推送 feature branch，服务器 detached 同步**

本地：

```bash
git push origin feat/flashinfer-sm120-fp8-moe
```

服务器：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
test -z "$(git status --porcelain)"
git fetch origin feat/flashinfer-sm120-fp8-moe
git switch --detach origin/feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
```

- [ ] **Step 3: 服务器运行全部 GPU correctness**

```bash
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
"${VENV_PY}" -m pytest \
  test/registered/moe/test_flashinfer_sm120_fp8_moe.py -q -s
```

Expected: 全部 PASS；A1/A2 bitwise、完整 runner、CUDA Graph 均通过。

- [ ] **Step 4: 用同一 commit 比较 legacy/fused A1 runner**

```bash
RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/stage-a1-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)
mkdir -p "${RUN_ROOT}/legacy" "${RUN_ROOT}/fused"
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3

SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0 "${VENV_PY}" \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  --tokens 8 8192 --top-k 8 --profiles uniform synthetic-skew \
  --warmup 5 --trials 3 --iterations 50 --check-cuda-graph \
  --output-json "${RUN_ROOT}/legacy/benchmark.json" \
  >"${RUN_ROOT}/legacy/stdout.txt" 2>"${RUN_ROOT}/legacy/stderr.txt"

SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=1 "${VENV_PY}" \
  scripts/pro5000/benchmark_flashinfer_sm120_fp8_runner.py \
  --tokens 8 8192 --top-k 8 --profiles uniform synthetic-skew \
  --warmup 5 --trials 3 --iterations 50 --check-cuda-graph \
  --output-json "${RUN_ROOT}/fused/benchmark.json" \
  >"${RUN_ROOT}/fused/stdout.txt" 2>"${RUN_ROOT}/fused/stderr.txt"
```

Expected: 两条命令 exit 0；legacy path 为 `a1_legacy_a2_fused`，fused path 为
`a1_fused_a2_fused`；fused 的 `gemm1_input_prepare` 明显下降，GEMM1/GEMM2 数值接近
legacy，完整 correctness PASS。

- [ ] **Step 5: 归档并回传 runner 证据**

```bash
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RUN_ROOT}.tar.gz" "$(basename "${RUN_ROOT}")"
echo "STAGE_A1_RUN_ROOT=${RUN_ROOT}"
echo "STAGE_A1_ARCHIVE=${RUN_ROOT}.tar.gz"
```

在进入服务 A/B 前检查压缩包同时包含两个 JSON 和 stdout/stderr。

---

### Task 7: 五长度三种子的正式服务 A/B

**Files:**
- Read: `scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py`
- Modify after result: `scripts/pro5000/README.md`
- Produce on server: `/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-${SHORT_COMMIT}/`，其中 `SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"`。

**Interfaces:**
- Consumes: Task 6 通过的精确 commit、用户现有模型目录和三种服务器模式。
- Produces: Triton、FlashInfer legacy A1、FlashInfer fused A1 manifests，以及 Triton-vs-fused 正式 decision。

- [ ] **Step 1: 固定 wheel SHA 和公共启动参数**

以下变量必须在后续使用的每一个新服务器终端中重新执行，shell 变量不会跨终端继承：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export WHEEL=/home/logs/sennian/pro5000-fi-moe/wheelhouse/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl
export FLASHINFER_SHA256="$(sha256sum "${WHEEL}" | awk '{print $1}')"
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
mkdir -p "${RUN_ROOT}"
echo "${FLASHINFER_SHA256}" > "${RUN_ROOT}/flashinfer.sha256"
```

所有服务器都使用原命令中的模型、host、port、metrics、parser、radix cache 和内存参数，
并显式增加 `--chunked-prefill-size 8192` 与相应 MoE backend。

- [ ] **Step 2: 启动 Triton server 并 capture**

在服务器终端 A：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export WHEEL=/home/logs/sennian/pro5000-fi-moe/wheelhouse/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl
export FLASHINFER_SHA256="$(sha256sum "${WHEEL}" | awk '{print $1}')"
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
nvidia-smi -q > "${RUN_ROOT}/gpu-state-triton.txt"
unset SGLANG_FLASHINFER_SM120_FP8_FUSED_A1
"${VENV_PY}" -m sglang.launch_server \
  --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
  --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
  --host 33.243.206.227 --port 30000 --enable-metrics --tp-size 1 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --disable-radix-cache --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --fp8-gemm-backend flashinfer_cutlass \
  --moe-runner-backend triton \
  2>&1 | tee "${RUN_ROOT}/server-triton.log"
```

在终端 B，健康检查通过后：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export WHEEL=/home/logs/sennian/pro5000-fi-moe/wheelhouse/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl
export FLASHINFER_SHA256="$(sha256sum "${WHEEL}" | awk '{print $1}')"
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
cd "${SGLANG_REPO}"
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py capture \
  --host 33.243.206.227 --port 30000 --expected-backend triton \
  --a1-mode not_applicable --repo "$PWD" \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --flashinfer-artifact-sha256 "${FLASHINFER_SHA256}" \
  --gpu-frequency-strategy default-unlocked \
  --output "${RUN_ROOT}/triton.json"
```

完成后在终端 A 用 Ctrl-C 正常停止 server。

- [ ] **Step 3: 启动 FlashInfer legacy A1 server 并 capture**

在终端 A 完整执行：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
nvidia-smi -q > "${RUN_ROOT}/gpu-state-flashinfer-legacy.txt"
export SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0
"${VENV_PY}" -m sglang.launch_server \
  --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
  --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
  --host 33.243.206.227 --port 30000 --enable-metrics --tp-size 1 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --disable-radix-cache --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --fp8-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_sm120_fp8 \
  2>&1 | tee "${RUN_ROOT}/server-flashinfer-legacy.log"
```

capture：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export WHEEL=/home/logs/sennian/pro5000-fi-moe/wheelhouse/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl
export FLASHINFER_SHA256="$(sha256sum "${WHEEL}" | awk '{print $1}')"
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
cd "${SGLANG_REPO}"
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py capture \
  --host 33.243.206.227 --port 30000 \
  --expected-backend flashinfer_sm120_fp8 --a1-mode legacy \
  --server-log "${RUN_ROOT}/server-flashinfer-legacy.log" --repo "$PWD" \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --flashinfer-artifact-sha256 "${FLASHINFER_SHA256}" \
  --gpu-frequency-strategy default-unlocked \
  --output "${RUN_ROOT}/flashinfer-legacy.json"
```

完成后停止 server。该 manifest 用于归因，不参与正式 GO 计算。

- [ ] **Step 4: 启动 FlashInfer fused A1 server 并 capture**

在终端 A 完整执行：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
nvidia-smi -q > "${RUN_ROOT}/gpu-state-flashinfer-fused.txt"
export SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=1
"${VENV_PY}" -m sglang.launch_server \
  --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
  --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
  --host 33.243.206.227 --port 30000 --enable-metrics --tp-size 1 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --disable-radix-cache --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --fp8-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_sm120_fp8 \
  2>&1 | tee "${RUN_ROOT}/server-flashinfer-fused.log"
```

capture：

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export WHEEL=/home/logs/sennian/pro5000-fi-moe/wheelhouse/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl
export FLASHINFER_SHA256="$(sha256sum "${WHEEL}" | awk '{print $1}')"
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
cd "${SGLANG_REPO}"
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py capture \
  --host 33.243.206.227 --port 30000 \
  --expected-backend flashinfer_sm120_fp8 --a1-mode fused \
  --server-log "${RUN_ROOT}/server-flashinfer-fused.log" --repo "$PWD" \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --flashinfer-artifact-sha256 "${FLASHINFER_SHA256}" \
  --gpu-frequency-strategy default-unlocked \
  --output "${RUN_ROOT}/flashinfer-fused.json"
```

- [ ] **Step 5: 生成正式 decision 与 legacy attribution**

```bash
export SGLANG_REPO=/home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git -C "${SGLANG_REPO}" rev-parse --short=12 HEAD)
cd "${SGLANG_REPO}"
"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py compare \
  --triton "${RUN_ROOT}/triton.json" \
  --flashinfer "${RUN_ROOT}/flashinfer-fused.json" \
  --output "${RUN_ROOT}/decision.json"

"${VENV_PY}" scripts/pro5000/benchmark_flashinfer_sm120_fp8_serving.py compare \
  --triton "${RUN_ROOT}/flashinfer-legacy.json" \
  --flashinfer "${RUN_ROOT}/flashinfer-fused.json" \
  --allow-reference-backend flashinfer_sm120_fp8 \
  --output "${RUN_ROOT}/a1-attribution.json"
```

第一条才是正式 GO。第二条仅说明 A1 融合对同一 FlashInfer backend 的净贡献，不得把它
当成 Triton 对照。

- [ ] **Step 6: 归档完整服务证据**

```bash
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RUN_ROOT}.tar.gz" "$(basename "${RUN_ROOT}")"
echo "SERVING_A1_RUN_ROOT=${RUN_ROOT}"
echo "SERVING_A1_ARCHIVE=${RUN_ROOT}.tar.gz"
```

压缩包必须包含 3 个 manifests、2 个 comparisons、3 个 server logs、3 个 `nvidia-smi -q`
快照、wheel SHA、45 个正式 JSONL（每个 backend 15 个）和 3 个不参与统计的
4096/seed-0 warmup JSONL。

---

### Task 8: 根据正式结果收口阶段 A

**Files:**
- Modify: `scripts/pro5000/README.md`
- Conditional modify on GO: `python/sglang/srt/environ.py`
- Conditional test on GO: `test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py`
- Produce on FUNCTIONAL_ONLY: expert distribution `.pt` files和独立阶段 B 实施计划。

**Interfaces:**
- Consumes: Task 7 `decision.json` 和完整 archive。
- Produces: GO 默认开关与部署说明，或 FUNCTIONAL_ONLY 结论与阶段 B 输入证据；两条分支互斥。

- [ ] **Step 1: 若 decision 为 GO，先写默认开启 RED 测试**

仅当 `decision == "GO"` 时，把 CPU test 的默认断言改为：

```python
monkeypatch.delenv("SGLANG_FLASHINFER_SM120_FP8_FUSED_A1", raising=False)
runner._use_fused_a1.cache_clear()
assert envs.SGLANG_FLASHINFER_SM120_FP8_FUSED_A1.get() is True
assert runner._use_fused_a1() is True
```

运行并确认 RED，然后把 `EnvBool(False)` 改为 `EnvBool(True)`；环境值 `0` 仍必须回滚到
legacy。复跑 CPU/GPU/full runner 后提交：

```bash
git add python/sglang/srt/environ.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py \
  scripts/pro5000/README.md
git commit -m "feat: enable fused SM120 FP8 A1 prepare by default"
```

README 记录正式 commit、FlashInfer wheel SHA、五长度逐 seed 数字、GO 公式、启动命令和
`SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=0` 回滚方式。

- [ ] **Step 2: 若 decision 为 FUNCTIONAL_ONLY，保持默认关闭并采集真实路由**

不要修改 EnvBool 默认值。先在终端 A 启动带 recorder 的 fused FlashInfer server：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
export VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
export RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git rev-parse --short=12 HEAD)
export SGLANG_FLASHINFER_SM120_FP8_FUSED_A1=1
"${VENV_PY}" -m sglang.launch_server \
  --served-model-name alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
  --model-path /home/admin/hippo/worker/slave/alimama-public-llm-service-qwen3.5-35a3-fp8-test_alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0_S179908_48_66/suez_worker/runtimedata/cantor/lm_data_Qwen3_5-35B-A3B-FP8/generation_1776070802/partition_0_65535/suez_data/ \
  --host 33.243.206.227 --port 30000 --enable-metrics --tp-size 1 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --disable-radix-cache --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --fp8-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_sm120_fp8 \
  --expert-distribution-recorder-mode per_pass \
  --expert-distribution-recorder-buffer-size -1 \
  2>&1 | tee "${RUN_ROOT}/server-flashinfer-route-recorder.log"
```

健康检查通过后，在终端 B 对每个 input length 单独执行一次不计时的 seed 17、1 prompt
run。`per_pass` 会保存逐层 top-k，因此这里禁止沿用正式吞吐的 100 prompts，避免 63488
长度产生数 GB recorder 数据。下面的 loop 为每个长度创建时间
marker，dump 后只接收该轮新生成的一个 `.pt`，然后立即改成确定文件名：

```bash
cd /home/logs/sennian/pro5000-fi-moe/sglang
VENV_PY=/home/logs/sennian/pro5000-fi-moe/.venv/bin/python3
RUN_ROOT=/home/logs/sennian/pro5000-fi-moe/runs/serving-a1-$(git rev-parse --short=12 HEAD)
ROUTE_DIR="${RUN_ROOT}/expert-routes"
COMMIT="$(git rev-parse --short=12 HEAD)"
mkdir -p "${ROUTE_DIR}"
for INPUT_LENGTH in 4096 6144 14336 30720 63488; do
  MARKER="${ROUTE_DIR}/start-${INPUT_LENGTH}.marker"
  touch "${MARKER}"
  curl -fsS -X POST http://33.243.206.227:30000/start_expert_distribution_record
  "${VENV_PY}" -m sglang.bench_serving \
    --backend sglang --dataset-name random \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len "${INPUT_LENGTH}" --random-output-len 1 --random-range-ratio 1 \
    --num-prompts 1 --seed 17 --host 33.243.206.227 --port 30000 \
    --model alimama-public-llm-service-qwen3.5-35a3-fp8-test-pro5000.default.0 \
    --output-file "${ROUTE_DIR}/bench-${INPUT_LENGTH}.jsonl" \
    --flush-cache
  curl -fsS -X POST http://33.243.206.227:30000/stop_expert_distribution_record
  curl -fsS -X POST http://33.243.206.227:30000/dump_expert_distribution_record
  mapfile -t NEW_DUMPS < <(
    find "$PWD" -maxdepth 1 -type f \
      -name 'expert_distribution_recorder_*.pt' -newer "${MARKER}" -print
  )
  test "${#NEW_DUMPS[@]}" -eq 1
  mv "${NEW_DUMPS[0]}" \
    "${ROUTE_DIR}/expert-distribution-input-${INPUT_LENGTH}-${COMMIT}.pt"
done
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RUN_ROOT}-routes.tar.gz" \
  "$(basename "${RUN_ROOT}")"
echo "STAGE_B_INPUT_ARCHIVE=${RUN_ROOT}-routes.tar.gz"
```

`test "${#NEW_DUMPS[@]}" -eq 1` 失败时停止，不猜测哪一个 dump 属于当前长度。profiling
run 不进入 throughput manifest。

README 记录 `FUNCTIONAL_ONLY`、A1 对 legacy 的净收益、Triton 差距和 route artifact 路径，
然后基于这些 `.pt` 与 runner component 数据另建
`docs/superpowers/plans/2026-07-21-flashinfer-sm120-fp8-scheduler-stage-b.md`。阶段 B 不在
本计划中直接实施。

- [ ] **Step 3: 最终验证与提交结果文档**

无论哪个分支都运行：

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_sm120_fp8_serving.py \
  test/registered/unit/test_pro5000_stage_2.py \
  test/registered/unit/layers/moe/test_flashinfer_sm120_fp8_config.py -q
git diff --check
git status --short
```

若 FUNCTIONAL_ONLY，只提交 README/结果文档，不改 production default。若 GO，必须在默认
开启的精确 commit 上再运行一次 4096 三种子确认；该复核仍需中位数 `>=10%` 且每个 seed
不回退。

---

## 完成定义

阶段 A 只有在以下产物同时存在时完成：

1. 服务 harness CPU tests 与 manifest 防误复用检查通过；
2. A1 payload/scale bitwise、动态 padding、重复 expert 和 `gap==0` GPU tests 通过；
3. 完整 runner 数值与 CUDA Graph replay 通过；
4. legacy/fused A1 component profile 可归因且 archive 完整；
5. 三个 server 模式的五长度三种子 manifests 和原始 JSONL 完整；
6. `decision.json` 明确给出 GO 或 FUNCTIONAL_ONLY；
7. GO 才默认开启 fused A1；FUNCTIONAL_ONLY 则保留默认关闭并准备独立阶段 B 计划；
8. 用户未跟踪 debug 文件保持原状。
