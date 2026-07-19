# RTX PRO 5000 Stage B 环境 Implementation Plan

## 2026-07-19 runtime-JIT 修订

服务器到 GitHub Release 的连接出现 131 秒超时，重试后的下载速度只有约
12–17 KB/s。用户批准采用 core-only runtime-JIT；本修订覆盖本文后续任何旧的
JIT-cache/AOT-cache 描述：

- 依赖固定为 `flashinfer_python==0.6.15.dev20260716`，不使用 `[cu13]` extra；
- 只下载并校验 core wheel，不安装 `flashinfer-jit-cache`；
- 若新的 Stage B venv 中有旧流程残留的 JIT-cache，只从该新 venv 卸载；
- 普通 PyPI 依赖使用清华镜像，`sglang-kernel` 保留 SGLang cu130 专用源；
- `smoke-no-jit.status` 预期非零，仅保留为诊断证据；
- 第一次正常 smoke 使用 CUDA 13.0 NVCC runtime-JIT，第二次验证缓存复用；
- manifest 中 `packages.flashinfer-jit-cache` 预期为 `null`。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改旧 venv 的前提下，为 fork SGLang 建立可复现的 CUDA 13.0 实验环境，并在 RTX PRO 5000 上完成 FlashInfer `moe_gemm_fp8_nt_groupwise` 的独立正确性与缓存预热验证。

**Architecture:** feature branch 固定 FlashInfer nightly 依赖，并提供三个职责单一的工具：GPU smoke test、环境 manifest 采集器和幂等 bootstrap。服务器只从 fork checkout 精确提交并运行 bootstrap；CUDA 验证由用户在服务器执行，本地负责静态测试和脚本测试。

**Tech Stack:** Bash、Python 3.12、uv、PyTorch 2.11.0+CUDA 13.0、FlashInfer `0.6.15.dev20260716`、pytest、Git。

## Global Constraints

- SGLang 基线固定为 `8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587`，开发分支固定为 `feat/flashinfer-sm120-fp8-moe`。
- 服务器根目录固定为 `/home/logs/sennian/pro5000-fi-moe`；repo、`.venv`、`wheelhouse`、`cache` 和 `runs` 必须互相隔离。
- 旧环境 `/home/logs/sennian/py-venv/sglang5.14` 不得写入、升级或删除。
- `flashinfer-python` 固定为 `0.6.15.dev20260716`，core wheel SHA256 固定为 `ed0634d9c32f069dafe7583addf74de7a4f366ae07d3093250109bd315b4ba26`。
- 不安装可选的 `flashinfer-jit-cache`；collector 保留该字段以证明值为 `null`。
- 正常运行不得设置 `FLASHINFER_DISABLE_JIT`；该变量只用于一次可选 no-JIT 诊断。
- runtime-JIT 必须使用服务器已有 NVCC 13.0.48，并把 cache 放在持久化根目录下。
- Stage B 不修改 FlashInfer kernel、不接入 MoE runner、不安装 `flashinfer-cubin`。
- 本地无法代表 RTX PRO 5000 CUDA 环境；GPU smoke 的通过证据必须来自用户返回的服务器输出。
- 每个任务都先写失败测试、确认失败原因、做最小实现、重新验证并单独提交。

---

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `python/pyproject.toml` | 固定 SGLang 对 FlashInfer nightly core 的精确依赖 |
| `scripts/pro5000/flashinfer_sm120_fp8_smoke.py` | 直接调用官方 kernel，验证小型非均匀/空 expert 与真实 GEMM1/GEMM2 shape |
| `scripts/pro5000/collect_stage_b_env.py` | 输出不含 secret 的 JSON 环境 manifest |
| `scripts/pro5000/bootstrap_stage_b.sh` | 创建新 venv、下载并校验 wheels、安装 fork、预热 kernel、保存运行产物 |
| `scripts/pro5000/README.md` | 给不能 SSH 代执行的服务器操作者提供复制即可运行的命令 |
| `test/registered/unit/test_pro5000_stage_b.py` | 检查依赖 pin、纯 helper、collector schema、shell 语法和安全常量 |

---

### Task 1: 固定 FlashInfer nightly 依赖

**Files:**
- Create: `test/registered/unit/test_pro5000_stage_b.py`
- Modify: `python/pyproject.toml:34`

**Interfaces:**
- Consumes: Python 3.11+ 标准库 `tomllib`。
- Produces: SGLang 安装元数据中的精确依赖字符串 `flashinfer_python==0.6.15.dev20260716`。

- [ ] **Step 1: 写依赖 pin 失败测试**

创建 `test/registered/unit/test_pro5000_stage_b.py`：

```python
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "python" / "pyproject.toml"
PRO5000_SCRIPTS = REPO_ROOT / "scripts" / "pro5000"


def test_flashinfer_nightly_dependency_is_pinned() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    assert (
        "flashinfer_python==0.6.15.dev20260716"
        in data["project"]["dependencies"]
    )
```

- [ ] **Step 2: 运行测试，确认因稳定版 pin 而失败**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_flashinfer_nightly_dependency_is_pinned \
  -q
```

Expected: FAIL；依赖列表仍包含 `flashinfer_python[cu13]==0.6.15`。

- [ ] **Step 3: 将依赖修改为精确 nightly 版本**

把 `python/pyproject.toml` 中对应行改为：

```toml
  "flashinfer_python==0.6.15.dev20260716", # Stage B uses the server NVCC for runtime JIT
```

- [ ] **Step 4: 重新运行测试**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_flashinfer_nightly_dependency_is_pinned \
  -q
```

Expected: `1 passed`。

- [ ] **Step 5: 提交依赖 pin**

```bash
git add python/pyproject.toml test/registered/unit/test_pro5000_stage_b.py
git commit -m "build: pin FlashInfer nightly for SM120 FP8 MoE"
```

---

### Task 2: 增加 FlashInfer SM120 独立 smoke test

**Files:**
- Create: `scripts/pro5000/flashinfer_sm120_fp8_smoke.py`
- Modify: `test/registered/unit/test_pro5000_stage_b.py`

**Interfaces:**
- Consumes: `flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise`、`flashinfer.testing.utils.per_token_cast_to_fp8`、`flashinfer.testing.utils.per_block_cast_to_fp8`。
- Produces: stdout 单个 JSON object；字段包含 `gpu`、`compute_capability`、`flashinfer_version`、`results`，进程退出码 0 表示全部 case 的 `calc_diff < 1e-3`。

- [ ] **Step 1: 先增加纯 helper 失败测试**

在 `test/registered/unit/test_pro5000_stage_b.py` 追加：

```python
def _load_script_module(filename: str):
    path = PRO5000_SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_padded_offset_contract() -> None:
    smoke = _load_script_module("flashinfer_sm120_fp8_smoke.py")
    assert smoke.compute_padded_offset(0, 0) == 0
    assert smoke.compute_padded_offset(1, 1) == 4
    assert smoke.compute_padded_offset(9, 3) == 16


def test_smoke_csr_offsets_include_empty_experts() -> None:
    smoke = _load_script_module("flashinfer_sm120_fp8_smoke.py")
    assert smoke.build_offsets([0, 8, 0, 3]) == [0, 0, 8, 8, 11]
```

- [ ] **Step 2: 运行测试，确认脚本不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_smoke_padded_offset_contract \
  test/registered/unit/test_pro5000_stage_b.py::test_smoke_csr_offsets_include_empty_experts \
  -q
```

Expected: FAIL with `FileNotFoundError`。

- [ ] **Step 3: 写完整 GPU smoke 脚本**

创建 `scripts/pro5000/flashinfer_sm120_fp8_smoke.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from typing import Any


DIFF_THRESHOLD = 1e-3
SUPPORTED_CC = {(12, 0), (12, 1)}


def compute_padded_offset(offset: int, problem_idx: int) -> int:
    return (offset + problem_idx * 3) // 4 * 4


def build_offsets(rows_per_expert: list[int]) -> list[int]:
    offsets = [0]
    for rows in rows_per_expert:
        if rows < 0:
            raise ValueError(f"rows_per_expert must be non-negative, got {rows}")
        offsets.append(offsets[-1] + rows)
    return offsets


def calc_diff(x, y) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum().item()
    if denominator == 0:
        return 0.0
    return 1.0 - 2.0 * (x * y).sum().item() / denominator


def quantize_and_pack_a(x, m_indptr):
    import torch
    from flashinfer.testing.utils import per_token_cast_to_fp8

    x_fp8, scale_row_major = per_token_cast_to_fp8(x)
    num_experts = m_indptr.numel() - 1
    k_blocks = scale_row_major.shape[1]
    m_padded = compute_padded_offset(x.shape[0], num_experts)
    packed = torch.zeros(
        (k_blocks, m_padded), dtype=torch.float32, device=x.device
    )
    for expert_id in range(num_experts):
        start = int(m_indptr[expert_id].item())
        end = int(m_indptr[expert_id + 1].item())
        if start == end:
            continue
        packed_start = compute_padded_offset(start, expert_id)
        packed[:, packed_start : packed_start + end - start] = (
            scale_row_major[start:end].t()
        )
    return x_fp8, packed


def make_inputs(rows_per_expert: list[int], n: int, k: int):
    import torch
    from flashinfer.testing.utils import per_block_cast_to_fp8

    torch.manual_seed(0)
    offsets = build_offsets(rows_per_expert)
    total_rows = offsets[-1]
    num_experts = len(rows_per_expert)
    m_indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    a_bf16 = torch.randn((total_rows, k), dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    reference = torch.zeros(
        (total_rows, n), dtype=torch.bfloat16, device="cuda"
    )
    for expert_id, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start < end:
            reference[start:end] = a_bf16[start:end] @ b_bf16[expert_id].t()

    a_fp8, a_scale = quantize_and_pack_a(a_bf16, m_indptr)
    b_fp8_parts = []
    b_scale_parts = []
    for expert_id in range(num_experts):
        b_fp8, b_scale = per_block_cast_to_fp8(b_bf16[expert_id])
        b_fp8_parts.append(b_fp8)
        b_scale_parts.append(b_scale)
    b_fp8 = torch.stack(b_fp8_parts, dim=0)
    b_scale = torch.stack(b_scale_parts, dim=0).transpose(-1, -2).contiguous()
    return a_fp8, b_fp8, a_scale, b_scale, m_indptr, reference


def run_case(name: str, rows_per_expert: list[int], n: int, k: int) -> dict[str, Any]:
    import torch
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    a, b, a_scale, b_scale, m_indptr, reference = make_inputs(
        rows_per_expert, n, k
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    output = moe_gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        out_dtype=torch.bfloat16,
    )
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    kernel_ms = start_event.elapsed_time(end_event)
    difference = calc_diff(output.float(), reference.float())
    if difference >= DIFF_THRESHOLD:
        raise RuntimeError(
            f"{name}: calc_diff={difference:.6e} exceeds {DIFF_THRESHOLD:.1e}"
        )
    return {
        "name": name,
        "num_experts": len(rows_per_expert),
        "cum_m": sum(rows_per_expert),
        "n": n,
        "k": k,
        "calc_diff": difference,
        "kernel_ms": kernel_ms,
        "first_call_wall_seconds": wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-shapes",
        action="store_true",
        help="Also validate the target 256-expert GEMM1 and GEMM2 N/K shapes.",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_CC:
        raise RuntimeError(
            f"Expected compute capability 12.0 or 12.1, got {capability}"
        )

    cases = [("small_irregular_empty", [0, 1, 8, 0, 16], 256, 512)]
    if args.real_shapes:
        one_row_per_expert = [1] * 256
        cases.extend(
            [
                ("qwen_gemm1_shape", one_row_per_expert, 1024, 2048),
                ("qwen_gemm2_shape", one_row_per_expert, 2048, 512),
            ]
        )

    results = [run_case(name, rows, n, k) for name, rows, n, k in cases]
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行纯 helper 测试和语法检查**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_smoke_padded_offset_contract \
  test/registered/unit/test_pro5000_stage_b.py::test_smoke_csr_offsets_include_empty_experts \
  -q
python3 -m py_compile scripts/pro5000/flashinfer_sm120_fp8_smoke.py
```

Expected: `2 passed`，`py_compile` exit 0。此处不声称 GPU case 通过。

- [ ] **Step 5: 提交 smoke test**

```bash
git add scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  test/registered/unit/test_pro5000_stage_b.py
git commit -m "test: add FlashInfer SM120 FP8 environment smoke"
```

---

### Task 3: 增加无 secret 环境 manifest 采集器

**Files:**
- Create: `scripts/pro5000/collect_stage_b_env.py`
- Modify: `test/registered/unit/test_pro5000_stage_b.py`

**Interfaces:**
- Consumes: 当前 Git checkout、标准 Python 包元数据、可选 Torch CUDA、`nvcc` 和 `uname`。
- Produces: stdout 单个 JSON object；只读取白名单 `FLASHINFER_*`/`CUDA_HOME` 环境变量，不读取 token、代理凭据或用户 secret。

- [ ] **Step 1: 写 collector schema 失败测试**

在测试文件追加：

```python
def test_environment_collector_emits_required_schema() -> None:
    script = PRO5000_SCRIPTS / "collect_stage_b_env.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert set(payload) >= {
        "timestamp_utc",
        "platform",
        "git",
        "python",
        "packages",
        "cuda",
        "environment",
    }
    assert set(payload["git"]) >= {"commit", "branch", "dirty"}
    assert "HF_TOKEN" not in payload["environment"]
```

- [ ] **Step 2: 运行测试，确认脚本不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_environment_collector_emits_required_schema \
  -q
```

Expected: FAIL；collector 文件不存在。

- [ ] **Step 3: 实现 collector**

创建 `scripts/pro5000/collect_stage_b_env.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAMES = (
    "torch",
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-jit-cache",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
)
ENV_ALLOWLIST = (
    "CUDA_HOME",
    "FLASHINFER_WORKSPACE_BASE",
    "FLASHINFER_DISABLE_JIT",
    "FLASHINFER_DISABLE_VERSION_CHECK",
    "FLASHINFER_CUBIN_DIR",
)


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as error:
        return {"command": command, "error": repr(error)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cuda_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "nvcc_path": shutil.which("nvcc"),
        "nvcc_version": run_command(["nvcc", "--version"]),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name() if torch.cuda.is_available() else None
                ),
                "compute_capability": (
                    list(torch.cuda.get_device_capability())
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except Exception as error:
        snapshot["torch_probe_error"] = repr(error)
    return snapshot


def main() -> int:
    commit = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "branch", "--show-current"])
    status = run_command(["git", "status", "--porcelain"])
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uname": run_command(["uname", "-a"]),
        },
        "git": {
            "commit": commit.get("stdout"),
            "branch": branch.get("stdout"),
            "dirty": bool(status.get("stdout")),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv": sys.prefix != sys.base_prefix,
        },
        "packages": package_versions(),
        "cuda": cuda_snapshot(),
        "environment": {name: os.environ.get(name) for name in ENV_ALLOWLIST},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 验证 collector**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_environment_collector_emits_required_schema \
  -q
python3 -m py_compile scripts/pro5000/collect_stage_b_env.py
```

Expected: `1 passed`，语法检查 exit 0。无 CUDA 的本地环境允许
`cuda_available=false` 或出现 `torch_probe_error`。

- [ ] **Step 5: 提交 collector**

```bash
git add scripts/pro5000/collect_stage_b_env.py \
  test/registered/unit/test_pro5000_stage_b.py
git commit -m "chore: add Stage B environment manifest collector"
```

---

### Task 4: 增加幂等 Stage B bootstrap

**Files:**
- Create: `scripts/pro5000/bootstrap_stage_b.sh`
- Modify: `test/registered/unit/test_pro5000_stage_b.py`

**Interfaces:**
- Consumes: server repo 位于 `/home/logs/sennian/pro5000-fi-moe/sglang`、`uv`、`curl`、`sha256sum`、Git、NVCC 13.0。
- Produces: `/home/logs/sennian/pro5000-fi-moe/.venv` 和一个新的 `runs/stage-b-*` 结果目录；stdout 最后一行输出 `STAGE_B_RUN_DIR=...`。

- [ ] **Step 1: 写 shell 安全与常量失败测试**

在测试文件追加：

```python
def test_bootstrap_uses_core_only_runtime_jit() -> None:
    script = PRO5000_SCRIPTS / "bootstrap_stage_b.sh"
    completed = subprocess.run(
        ["bash", "-n", str(script)], text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    content = script.read_text()
    assert "/home/logs/sennian/pro5000-fi-moe" in content
    assert "0.6.15.dev20260716" in content
    assert "ed0634d9c32f069dafe7583addf74de7a4f366ae07d3093250109bd315b4ba26" in content
    assert "flashinfer_jit_cache" not in content
    assert "86a0944b4cadde0a4227f249e5a0fe466207d7c25e8eb7dee4c3f75fdd5f9bbf" not in content
    assert "FLASHINFER_DISABLE_JIT=1" in content
    assert "--real-shapes" in content
    assert "nvidia-cutlass-dsl-libs-cu13==4.5.2" in content
    assert "rm -rf" not in content
    assert "/home/logs/sennian/py-venv/sglang5.14" not in content
```

- [ ] **Step 2: 运行测试，确认脚本不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_bootstrap_shell_syntax_and_fixed_artifacts \
  -q
```

Expected: FAIL；bootstrap 文件不存在。

- [ ] **Step 3: 实现 bootstrap**

创建 `scripts/pro5000/bootstrap_stage_b.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

PRO5000_ROOT="${PRO5000_ROOT:-/home/logs/sennian/pro5000-fi-moe}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
VENV_DIR="${PRO5000_ROOT}/.venv"
WHEELHOUSE="${PRO5000_ROOT}/wheelhouse"
CACHE_DIR="${PRO5000_ROOT}/cache"
RUNS_DIR="${PRO5000_ROOT}/runs"
EXPECTED_REPO="${PRO5000_ROOT}/sglang"

CORE_NAME="flashinfer_python-0.6.15.dev20260716-py3-none-any.whl"
CORE_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.15-20260716/flashinfer_python-0.6.15.dev20260716-py3-none-any.whl"
CORE_SHA256="ed0634d9c32f069dafe7583addf74de7a4f366ae07d3093250109bd315b4ba26"

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${command_name}" >&2
    exit 1
  fi
}

verify_sha256() {
  local file_path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${file_path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: SHA256 mismatch for ${file_path}" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    exit 1
  fi
}

download_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local partial="${destination}.part"
  if [[ -f "${destination}" ]]; then
    verify_sha256 "${destination}" "${expected_sha}"
    return
  fi
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${partial}" "${url}"
  verify_sha256 "${partial}" "${expected_sha}"
  mv "${partial}" "${destination}"
}

for command_name in git curl sha256sum uv nvcc uname; do
  require_command "${command_name}"
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: Stage B requires Linux x86_64" >&2
  exit 1
fi
if [[ "${REPO_ROOT}" != "${EXPECTED_REPO}" ]]; then
  echo "ERROR: repo must be located at ${EXPECTED_REPO}; got ${REPO_ROOT}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ERROR: server checkout must be clean before bootstrap" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 1
fi

mkdir -p "${WHEELHOUSE}" "${CACHE_DIR}" "${RUNS_DIR}"
download_verified "${CORE_URL}" "${WHEELHOUSE}/${CORE_NAME}" "${CORE_SHA256}"

if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
  uv venv --python 3.12 --seed "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python3"
if [[ "$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "ERROR: ${VENV_DIR} is not a Python 3.12 venv" >&2
  exit 1
fi

export UV_CACHE_DIR="${CACHE_DIR}/uv"
export FLASHINFER_WORKSPACE_BASE="${CACHE_DIR}/flashinfer-workspace-base"

uv pip install --python "${PYTHON}" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "${WHEELHOUSE}/${CORE_NAME}"
uv pip install --python "${PYTHON}" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --find-links "${WHEELHOUSE}" \
  -e "${REPO_ROOT}/python"
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  --index-url https://docs.sglang.ai/whl/cu130/ \
  sglang-kernel==0.4.4
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  "${WHEELHOUSE}/${CORE_NAME}"
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  nvidia-cutlass-dsl-libs-cu13==4.5.2

if "${PYTHON}" -c 'import importlib.metadata; importlib.metadata.version("flashinfer-jit-cache")' \
  >/dev/null 2>&1; then
  uv pip uninstall --python "${PYTHON}" flashinfer-jit-cache
fi

RUN_ID="stage-b-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

"${PYTHON}" -m pip check >"${RUN_DIR}/pip-check.txt" 2>&1
sha256sum "${WHEELHOUSE}/${CORE_NAME}" >"${RUN_DIR}/wheel-sha256.txt"
"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/collect_stage_b_env.py" \
  >"${RUN_DIR}/environment.json"

set +e
FLASHINFER_DISABLE_JIT=1 "${PYTHON}" \
  "${REPO_ROOT}/scripts/pro5000/flashinfer_sm120_fp8_smoke.py" \
  >"${RUN_DIR}/smoke-no-jit.json" 2>"${RUN_DIR}/smoke-no-jit.stderr"
NO_JIT_STATUS=$?
set -e
printf '%s\n' "${NO_JIT_STATUS}" >"${RUN_DIR}/smoke-no-jit.status"

"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/flashinfer_sm120_fp8_smoke.py" \
  --real-shapes \
  >"${RUN_DIR}/smoke-normal-first.json" \
  2>"${RUN_DIR}/smoke-normal-first.stderr"
"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/flashinfer_sm120_fp8_smoke.py" \
  --real-shapes \
  >"${RUN_DIR}/smoke-normal-second.json" \
  2>"${RUN_DIR}/smoke-normal-second.stderr"

"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/collect_stage_b_env.py" \
  >"${RUN_DIR}/environment-after-smoke.json"

echo "Stage B bootstrap and required smoke tests completed."
echo "NO_JIT_STATUS=${NO_JIT_STATUS}"
echo "STAGE_B_RUN_DIR=${RUN_DIR}"
```

- [ ] **Step 4: 验证 shell、常量和现有 Python 工具**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_b.py -q
bash -n scripts/pro5000/bootstrap_stage_b.sh
python3 -m py_compile \
  scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  scripts/pro5000/collect_stage_b_env.py
```

Expected: 全部 pytest 通过；`bash -n` 和 `py_compile` exit 0。

- [ ] **Step 5: 提交 bootstrap**

```bash
git add scripts/pro5000/bootstrap_stage_b.sh \
  test/registered/unit/test_pro5000_stage_b.py
git commit -m "build: add reproducible PRO 5000 Stage B bootstrap"
```

---

### Task 5: 增加服务器执行与输出回传说明

**Files:**
- Create: `scripts/pro5000/README.md`
- Modify: `test/registered/unit/test_pro5000_stage_b.py`

**Interfaces:**
- Consumes: 已推送的 `origin/feat/flashinfer-sm120-fp8-moe`。
- Produces: 无占位符的首次 clone、detached checkout、bootstrap 和结果打印命令。

- [ ] **Step 1: 写 README 安全约束失败测试**

在测试文件追加：

```python
def test_stage_b_readme_preserves_old_environment_and_uses_detached_checkout() -> None:
    readme = (PRO5000_SCRIPTS / "README.md").read_text()
    assert "/home/logs/sennian/py-venv/sglang5.14" in readme
    assert "git switch --detach origin/feat/flashinfer-sm120-fp8-moe" in readme
    assert "bash scripts/pro5000/bootstrap_stage_b.sh" in readme
    assert "rm -rf" not in readme
```

- [ ] **Step 2: 运行测试，确认 README 不存在**

Run:

```bash
python3 -m pytest \
  test/registered/unit/test_pro5000_stage_b.py::test_stage_b_readme_preserves_old_environment_and_uses_detached_checkout \
  -q
```

Expected: FAIL；README 文件不存在。

- [ ] **Step 3: 写服务器操作说明**

创建 `scripts/pro5000/README.md`：

````markdown
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

把上述输出完整返回。由于不安装 JIT-cache，`smoke-no-jit.status` 预期非零；
只要两次 `smoke-normal-*.json` 都生成且 `calc_diff < 1e-3`，Stage B 使用
NVCC runtime-JIT 模式继续验收。
````

- [ ] **Step 4: 运行 README 测试与全套静态验证**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_b.py -q
git diff --check
```

Expected: 全部测试通过；`git diff --check` 无输出并 exit 0。

- [ ] **Step 5: 提交操作说明**

```bash
git add scripts/pro5000/README.md test/registered/unit/test_pro5000_stage_b.py
git commit -m "docs: add PRO 5000 Stage B server runbook"
```

---

### Task 6: 本地总验证、推送与服务器交接

**Files:**
- Verify only; no new file.

**Interfaces:**
- Consumes: Tasks 1–5 的全部提交。
- Produces: fork feature branch、精确 Git SHA 和用户可执行的服务器命令。

- [ ] **Step 1: 运行完整本地验证**

Run:

```bash
python3 -m pytest test/registered/unit/test_pro5000_stage_b.py -q
bash -n scripts/pro5000/bootstrap_stage_b.sh
python3 -m py_compile \
  scripts/pro5000/flashinfer_sm120_fp8_smoke.py \
  scripts/pro5000/collect_stage_b_env.py
git diff --check
git status --short --branch
```

Expected: pytest 全部通过；三项语法/whitespace 检查 exit 0；工作树干净且当前分支为
`feat/flashinfer-sm120-fp8-moe`。

- [ ] **Step 2: 核对提交范围和依赖版本**

Run:

```bash
git log --oneline 8f765bc1c9542c4ff1c3b62ad16fbfe8882a5587..HEAD
python3 - <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("python/pyproject.toml").read_text())
pins = [x for x in data["project"]["dependencies"] if x.startswith("flashinfer_python")]
assert pins == ["flashinfer_python==0.6.15.dev20260716"], pins
print(pins[0])
PY
```

Expected: 日志只包含已批准的设计/Stage B commits；Python 打印精确 nightly pin。

- [ ] **Step 3: 推送 feature branch**

```bash
git push -u origin feat/flashinfer-sm120-fp8-moe
git rev-parse HEAD
```

Expected: push 成功并打印用于服务器 checkout 的完整 SHA。

- [ ] **Step 4: 用户执行服务器 runbook 并返回产物**

用户按照 `scripts/pro5000/README.md` 执行。收到输出后检查：

```text
environment-after-smoke.json:
  python.venv == true
  packages.torch 以 2.11.0 开头
  packages.flashinfer-python == 0.6.15.dev20260716
  packages.flashinfer-jit-cache == null
  cuda.torch_cuda == 13.0
  cuda.compute_capability == [12, 0]

wheel-sha256.txt:
  core wheel 哈希精确匹配 Global Constraints

smoke-normal-first.json 和 smoke-normal-second.json:
  三个 case 均存在
  每个 calc_diff < 1e-3
```

- [ ] **Step 5: 标记 Stage B 结果**

如果上述 required 项全部通过，记录：

```text
Stage B: PASS
mode: runtime-JIT（smoke-no-jit.status 预期非 0）
git_sha: 服务器 environment-after-smoke.json 中的 git.commit
```

若 required normal smoke 失败，不修改 runner；保留 `runs/stage-b-*` 证据并进入
systematic-debugging，按首次失败栈定位依赖或 NVCC 编译问题。
