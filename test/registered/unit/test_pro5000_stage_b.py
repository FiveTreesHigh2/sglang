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


def test_flashinfer_nightly_dependency_is_pinned():
    data = tomllib.loads(PYPROJECT.read_text())
    dependencies = data["project"]["dependencies"]

    assert "flashinfer_python[cu13]==0.6.15.dev20260716" in dependencies


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
