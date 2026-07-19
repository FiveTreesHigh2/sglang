from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 compatibility for local test runners.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "python" / "pyproject.toml"
PRO5000_SCRIPTS = REPO_ROOT / "scripts" / "pro5000"


def test_flashinfer_nightly_dependency_uses_runtime_jit():
    data = tomllib.loads(PYPROJECT.read_text())
    dependencies = data["project"]["dependencies"]

    assert "flashinfer_python==0.6.15.dev20260716" in dependencies
    assert not any(dependency.startswith("flashinfer_python[") for dependency in dependencies)


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
    assert 'importlib.metadata.version("flashinfer-jit-cache")' in content
    assert 'uv pip uninstall --python "${PYTHON}" flashinfer-jit-cache' in content
    assert content.count("-i https://pypi.tuna.tsinghua.edu.cn/simple") == 3
    assert "--index-url https://docs.sglang.ai/whl/cu130/" in content
    assert "FLASHINFER_DISABLE_JIT=1" in content
    assert "--real-shapes" in content
    assert "nvidia-cutlass-dsl-libs-cu13==4.5.2" in content
    assert "rm -rf" not in content
    assert "/home/logs/sennian/py-venv/sglang5.14" not in content


def test_stage_b_readme_preserves_old_environment_and_uses_detached_checkout() -> None:
    readme = (PRO5000_SCRIPTS / "README.md").read_text()
    assert "/home/logs/sennian/py-venv/sglang5.14" in readme
    assert "git switch --detach origin/feat/flashinfer-sm120-fp8-moe" in readme
    assert "bash scripts/pro5000/bootstrap_stage_b.sh" in readme
    assert "不安装 `flashinfer-jit-cache`" in readme
    assert "NVCC runtime-JIT" in readme
    assert "安装中断后可直接重新运行" in readme
    assert "rm -rf" not in readme
