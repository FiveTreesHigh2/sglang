#!/usr/bin/env python3
"""Capture and compare the Stage A SM120 FP8 serving A/B benchmark."""

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
from typing import Any, Optional, Sequence
from urllib.request import urlopen

import torch


INPUT_LENGTHS = (4096, 6144, 14336, 30720, 63488)
SEEDS = (17, 29, 43)
NUM_PROMPTS = 100
FIXED_SERVER_FIELDS = (
    "model_path",
    "served_model_name",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "tp_size",
    "dp_size",
    "ep_size",
    "pp_size",
    "disable_radix_cache",
    "mem_fraction_static",
    "attention_backend",
    "prefill_attention_backend",
    "reasoning_parser",
    "tool_call_parser",
    "chunked_prefill_size",
    "fp8_gemm_runner_backend",
    "moe_runner_backend",
    "enable_metrics",
)


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible benchmark inputs."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def query_single_gpu_uuid() -> str:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    uuids = [value.strip() for value in output if value.strip()]
    if len(uuids) != 1:
        raise ValueError(f"stage A requires exactly one visible GPU, got {uuids}")
    return uuids[0]


def verify_server_info(info: dict[str, Any], expected_backend: str) -> dict[str, Any]:
    """Reject serving configurations that would make an A/B pairing invalid."""
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


def build_run_key(
    *,
    backend: str,
    commit: str,
    input_length: int,
    num_prompts: int,
    seed: int,
    server_args_hash: str,
    a1_mode: str,
    flashinfer_artifact_sha256: str,
    gpu_uuid: str,
    gpu_frequency_strategy: str,
    runtime_fingerprint: dict[str, Any],
) -> str:
    """Hash every dimension that makes a captured result non-reusable."""
    return canonical_hash(
        {
            "backend": backend,
            "commit": commit,
            "input_length": input_length,
            "num_prompts": num_prompts,
            "seed": seed,
            "server_args_hash": server_args_hash,
            "a1_mode": a1_mode,
            "flashinfer_artifact_sha256": flashinfer_artifact_sha256,
            "gpu_uuid": gpu_uuid,
            "gpu_frequency_strategy": gpu_frequency_strategy,
            "runtime_fingerprint": runtime_fingerprint,
        }
    )


def evaluate_speedup_gate(input_length: int, speedups: Sequence[float]) -> bool:
    """Apply the formal per-length speedup gate to the three paired seeds."""
    median = statistics.median(speedups)
    if input_length == 4096:
        return median >= 0.10 and min(speedups) >= 0.0
    return median > 0.0 and min(speedups) > 0.0


def compare_manifests(
    triton: dict[str, Any], flashinfer: dict[str, Any]
) -> dict[str, Any]:
    """Compare fully paired captures, rejecting any config or case drift."""
    for name, manifest in (("triton", triton), ("flashinfer", flashinfer)):
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"{name} schema_version must be 1, got "
                f"{manifest.get('schema_version')!r}"
            )
    if triton["metadata"]["pairing_hash"] != flashinfer["metadata"][
        "pairing_hash"
    ]:
        raise ValueError("pairing_hash mismatch")
    for field in (
        "sglang_commit",
        "flashinfer_artifact_sha256",
        "flashinfer_version",
        "torch_version",
        "torch_cuda_version",
        "sglang_kernel_version",
        "python_executable",
        "gpu_uuid",
        "gpu_frequency_strategy",
        "dense_fp8_backend",
        "num_prompts",
    ):
        if triton["metadata"].get(field) != flashinfer["metadata"].get(field):
            raise ValueError(f"{field} mismatch")

    def indexed(manifest: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
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
    expected = {
        (input_length, seed) for input_length in INPUT_LENGTHS for seed in SEEDS
    }
    if set(left) != expected or set(right) != expected:
        raise ValueError("manifests must contain every input-length/seed pair exactly once")

    by_length = {}
    go = True
    for input_length in INPUT_LENGTHS:
        samples = []
        for seed in SEEDS:
            baseline, candidate = left[(input_length, seed)], right[(input_length, seed)]
            if baseline["total_input_tokens"] != candidate["total_input_tokens"]:
                raise ValueError("total_input_tokens mismatch")
            samples.append(
                candidate["input_throughput"] / baseline["input_throughput"] - 1.0
            )
        median = statistics.median(samples)
        passed = evaluate_speedup_gate(input_length, samples)
        go = go and passed
        by_length[str(input_length)] = {
            "paired_speedups": samples,
            "median_speedup": median,
            "min_speedup": min(samples),
            "passed": passed,
        }
    return {
        "decision": "GO" if go else "FUNCTIONAL_ONLY",
        "by_input_length": by_length,
    }


def fetch_server_info(host: str, port: int) -> dict[str, Any]:
    with urlopen(f"http://{host}:{port}/server_info", timeout=10) as response:
        return json.load(response)


def parse_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty benchmark output: {path}")
    raw = json.loads(lines[-1])
    return {
        name: raw[name]
        for name in (
            "input_throughput",
            "median_ttft_ms",
            "completed",
            "total_input_tokens",
        )
    }


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    """Capture one warmup and every formal input-length/seed case."""
    if args.expected_backend == "flashinfer_sm120_fp8":
        marker = f"flashinfer_sm120_fp8 A1 prepare mode={args.a1_mode}"
        if args.server_log is None or marker not in args.server_log.read_text(
            errors="replace"
        ):
            raise ValueError(f"server log missing A1 marker: {marker}")

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

    def bench_command(
        *, input_length: int, num_prompts: int, seed: int, output: Path
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--dataset-name",
            "random",
            "--dataset-path",
            str(args.dataset_path),
            "--random-input-len",
            str(input_length),
            "--random-output-len",
            "1",
            "--random-range-ratio",
            "1",
            "--warmup-requests",
            "0",
            "--num-prompts",
            str(num_prompts),
            "--output-file",
            str(output),
            "--seed",
            str(seed),
            "--port",
            str(args.port),
            "--host",
            args.host,
            "--model",
            snapshot["served_model_name"],
            "--flush-cache",
        ]

    warmup_output = result_dir / f"warmup-{args.a1_mode}.jsonl"
    warmup_command = bench_command(
        input_length=4096, num_prompts=1, seed=0, output=warmup_output
    )
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
                backend=args.expected_backend,
                commit=commit,
                input_length=input_length,
                num_prompts=NUM_PROMPTS,
                seed=seed,
                server_args_hash=server_args_hash,
                a1_mode=args.a1_mode,
                flashinfer_artifact_sha256=args.flashinfer_artifact_sha256,
                gpu_uuid=gpu_uuid,
                gpu_frequency_strategy=args.gpu_frequency_strategy,
                runtime_fingerprint=runtime_fingerprint,
            )
            output = result_dir / f"{run_key}.jsonl"
            command = bench_command(
                input_length=input_length,
                num_prompts=NUM_PROMPTS,
                seed=seed,
                output=output,
            )
            if not output.exists():
                subprocess.run(command, check=True)
            result = parse_last_jsonl(output)
            if result["completed"] != NUM_PROMPTS:
                raise ValueError(f"completed must be {NUM_PROMPTS}: {result}")
            cases.append(
                {
                    "run_key": run_key,
                    "input_length": input_length,
                    "seed": seed,
                    "command": command,
                    "output_file": str(output),
                    **result,
                }
            )
            write_json_atomic(
                args.output,
                {"schema_version": 1, "metadata": metadata, "cases": cases},
            )

    return {"schema_version": 1, "metadata": metadata, "cases": cases}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--host", required=True)
    capture.add_argument("--port", type=int, default=30000)
    capture.add_argument(
        "--expected-backend", choices=("triton", "flashinfer_sm120_fp8"), required=True
    )
    capture.add_argument(
        "--a1-mode", choices=("not_applicable", "legacy", "fused"), required=True
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
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            parser.error("--flashinfer-artifact-sha256 must be 64 hex characters")
        args.flashinfer_artifact_sha256 = digest
        if args.expected_backend == "triton" and args.a1_mode != "not_applicable":
            parser.error("Triton capture requires --a1-mode not_applicable")
        if args.expected_backend == "flashinfer_sm120_fp8" and (
            args.a1_mode not in ("legacy", "fused") or args.server_log is None
        ):
            parser.error(
                "FlashInfer capture requires legacy/fused A1 mode and --server-log"
            )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        write_json_atomic(args.output, run_capture(args))
        return 0

    reference = json.loads(args.triton.read_text())
    candidate = json.loads(args.flashinfer.read_text())
    for name, manifest in (("reference", reference), ("candidate", candidate)):
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"{name} schema_version must be 1, got "
                f"{manifest.get('schema_version')!r}"
            )
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
