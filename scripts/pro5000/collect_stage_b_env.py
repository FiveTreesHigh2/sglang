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
