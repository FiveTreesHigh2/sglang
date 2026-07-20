#!/usr/bin/env bash
set -euo pipefail

PRO5000_ROOT="${PRO5000_ROOT:-/home/logs/sennian/pro5000-fi-moe}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON="${PRO5000_ROOT}/.venv/bin/python3"
RUNS_DIR="${PRO5000_ROOT}/runs"
EXPECTED_REPO="${PRO5000_ROOT}/sglang"
RUN_DIR=""
BENCHMARK_STATUS=1
AFTER_STATUS=0

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${command_name}" >&2
    exit 1
  fi
}

capture_after() {
  if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    return
  fi

  set +e
  "${PYTHON}" "${REPO_ROOT}/scripts/pro5000/collect_stage_b_env.py" \
    >"${RUN_DIR}/environment-after.json"
  if [[ $? -ne 0 ]]; then
    AFTER_STATUS=1
  fi

  uv pip check --python "${PYTHON}" >"${RUN_DIR}/pip-check-after.txt" 2>&1
  if [[ $? -ne 0 ]]; then
    AFTER_STATUS=1
  fi

  uv pip freeze --python "${PYTHON}" >"${RUN_DIR}/packages-after.txt" 2>&1
  if [[ $? -ne 0 ]]; then
    AFTER_STATUS=1
  fi

  nvidia-smi >"${RUN_DIR}/nvidia-smi-after.txt" 2>&1
  if [[ $? -ne 0 ]]; then
    AFTER_STATUS=1
  fi

  if [[ -f "${RUN_DIR}/packages-before.txt" && -f "${RUN_DIR}/packages-after.txt" ]]; then
    if ! cmp -s "${RUN_DIR}/packages-before.txt" "${RUN_DIR}/packages-after.txt"; then
      echo "ERROR: Python package set changed during Stage 1" >&2
      diff -u "${RUN_DIR}/packages-before.txt" "${RUN_DIR}/packages-after.txt" \
        >"${RUN_DIR}/packages.diff" 2>&1 || true
      AFTER_STATUS=1
    fi
  fi
}

finish() {
  local original_status=$?
  local final_status
  trap - EXIT
  capture_after
  final_status="${original_status}"
  if [[ "${final_status}" -eq 0 && "${AFTER_STATUS}" -ne 0 ]]; then
    final_status=1
  fi
  echo "STAGE_1_STATUS=${BENCHMARK_STATUS}"
  echo "STAGE_1_WRAPPER_STATUS=${final_status}"
  if [[ -n "${RUN_DIR}" ]]; then
    echo "STAGE_1_RUN_DIR=${RUN_DIR}"
  fi
  exit "${final_status}"
}

for command_name in git uv nvidia-smi nvcc uname tee cmp diff; do
  require_command "${command_name}"
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: Stage 1 requires Linux x86_64" >&2
  exit 1
fi
if [[ "${REPO_ROOT}" != "${EXPECTED_REPO}" ]]; then
  echo "ERROR: repo must be located at ${EXPECTED_REPO}; got ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Stage B venv Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ "$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "ERROR: Stage 1 requires the existing Python 3.12 Stage B venv" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ERROR: server checkout must be clean before Stage 1" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 1
fi
if [[ -n "${FLASHINFER_DISABLE_JIT:-}" ]]; then
  echo "ERROR: unset FLASHINFER_DISABLE_JIT for Stage 1" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}" "${PRO5000_ROOT}/cache/flashinfer-workspace-base"
export FLASHINFER_WORKSPACE_BASE="${PRO5000_ROOT}/cache/flashinfer-workspace-base"

RUN_ID="stage-1-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
trap finish EXIT

uv pip check --python "${PYTHON}" >"${RUN_DIR}/pip-check-before.txt" 2>&1
uv pip freeze --python "${PYTHON}" >"${RUN_DIR}/packages-before.txt" 2>&1
"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/collect_stage_b_env.py" \
  >"${RUN_DIR}/environment.json"
nvidia-smi >"${RUN_DIR}/nvidia-smi-before.txt" 2>&1

set +e
"${PYTHON}" "${REPO_ROOT}/scripts/pro5000/benchmark_flashinfer_sm120_fp8_moe.py" \
  --clock-mode "${STAGE1_CLOCK_MODE:-default}" \
  --output "${RUN_DIR}/benchmark.json" \
  > >(tee "${RUN_DIR}/benchmark.stdout.txt") \
  2> >(tee "${RUN_DIR}/benchmark.stderr.txt" >&2)
BENCHMARK_STATUS=$?
set -e

exit "${BENCHMARK_STATUS}"
