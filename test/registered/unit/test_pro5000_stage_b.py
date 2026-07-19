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
