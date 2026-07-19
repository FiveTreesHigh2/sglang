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
JIT_NAME="flashinfer_jit_cache-0.6.15.dev20260716+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
JIT_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.15-20260716/flashinfer_jit_cache-0.6.15.dev20260716%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl"
JIT_SHA256="86a0944b4cadde0a4227f249e5a0fe466207d7c25e8eb7dee4c3f75fdd5f9bbf"

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
download_verified "${JIT_URL}" "${WHEELHOUSE}/${JIT_NAME}" "${JIT_SHA256}"

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
  "${WHEELHOUSE}/${CORE_NAME}" "${WHEELHOUSE}/${JIT_NAME}"
uv pip install --python "${PYTHON}" \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --find-links "${WHEELHOUSE}" \
  -e "${REPO_ROOT}/python"
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  --index-url https://docs.sglang.ai/whl/cu130/ \
  sglang-kernel==0.4.4
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  "${WHEELHOUSE}/${CORE_NAME}" "${WHEELHOUSE}/${JIT_NAME}"
uv pip install --python "${PYTHON}" --force-reinstall --no-deps \
  nvidia-cutlass-dsl-libs-cu13==4.5.2

RUN_ID="stage-b-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

"${PYTHON}" -m pip check >"${RUN_DIR}/pip-check.txt" 2>&1
sha256sum "${WHEELHOUSE}/${CORE_NAME}" "${WHEELHOUSE}/${JIT_NAME}" \
  >"${RUN_DIR}/wheel-sha256.txt"
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
