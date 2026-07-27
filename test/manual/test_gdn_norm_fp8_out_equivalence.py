"""Equivalence test for the fused GDN gated-norm + fp8 quant epilogue (D1-a).

Acceptance standard (decision recorded in the prefill-optimization status
note): the quant stage replicates the production CUDA v2 kernel bitwise
(verified 0/33.5M in isolation once fed identical inputs), but the norm
stage cannot be replicated bitwise across two compiled Triton kernels —
the tl.sum reduction association is layout-driven (reference kernel: dual
8-element FMA chains; fused kernel: single 16-element chain, PTX-verified)
and the epilogue's extra anchors (fp8 stores, tl.max) legally change it.
The resulting deviation is a reassociation of the same fp32 terms, bounded
in practice at ~3e-7 of codes (all within one e4m3 step) and ~4e-6 of
scales (one bf16 step of the group amax).

Assertions:
  1. fp8 codes: mismatch fraction <= 2e-6 AND every mismatch is exactly
     one e4m3 code step.
  2. A scales: mismatch fraction <= 2e-5 AND relative deviation <= 1%
     (one bf16 step of amax is ~0.78%).
  3. Wrapper plumbing is strictly bitwise: feeding the CUDA-quantized
     (q_ref, s_ref) through the pre-quantized input path reproduces the
     internal-quant GEMM output exactly (torch.equal), including the
     m % 4 != 0 zero-pad branch.
  4. End-to-end: rows whose fused (q, s) are bitwise-identical to the
     reference produce bitwise-identical GEMM outputs; affected rows are
     bounded by the same fraction as (1).

Run on the server:
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
      test/manual/test_gdn_norm_fp8_out_equivalence.py
"""

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import (
    layernorm_fn,
    rms_norm_gated_fp8_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

DEVICE = "cuda"
BLOCK = 128
TOL_CODE_FRACTION = 2e-6  # measured 11 / 33.5M = 3.3e-7
TOL_SCALE_FRACTION = 2e-5  # measured 1 / 262144 = 3.8e-6
TOL_SCALE_REL = 1e-2  # one bf16 step of amax is ~0.78%


def _e4m3_grid():
    codes = torch.arange(256, dtype=torch.uint8, device=DEVICE)
    vals = codes.view(torch.float8_e4m3fn).float()
    return torch.unique(vals[torch.isfinite(vals)])  # sorted ascending


_GRID = None


def assert_code_deviation_bounded(q_new, q_ref, label):
    global _GRID
    if _GRID is None:
        _GRID = _e4m3_grid()
    mism = q_new.view(torch.uint8) != q_ref.view(torch.uint8)
    frac = mism.float().mean().item()
    assert frac <= TOL_CODE_FRACTION, (
        f"{label}: code mismatch fraction {frac:.2e} > {TOL_CODE_FRACTION:.0e}"
    )
    if mism.any():
        va = q_new.float()[mism].contiguous()
        vb = q_ref.float()[mism].contiguous()
        ra = torch.searchsorted(_GRID, va)
        rb = torch.searchsorted(_GRID, vb)
        step = (ra - rb).abs()
        assert (step == 1).all(), (
            f"{label}: found mismatches larger than one e4m3 code step "
            f"(max {int(step.max())})"
        )
    return int(mism.sum()), mism


def assert_scale_deviation_bounded(s_new, s_ref, label):
    mism = s_new != s_ref
    frac = mism.float().mean().item()
    assert frac <= TOL_SCALE_FRACTION, (
        f"{label}: scale mismatch fraction {frac:.2e} > {TOL_SCALE_FRACTION:.0e}"
    )
    if mism.any():
        rel = ((s_new[mism] - s_ref[mism]).abs() / s_ref[mism].abs()).max().item()
        assert rel <= TOL_SCALE_REL, (
            f"{label}: scale relative deviation {rel:.2e} > {TOL_SCALE_REL:.0e}"
        )
    return int(mism.sum()), mism


def run_kernel_case(T, H, D, activation, seed):
    torch.manual_seed(seed)
    N = H * D
    x = torch.randn(T, N, device=DEVICE, dtype=torch.bfloat16) * 2.0
    z = torch.randn(T, N, device=DEVICE, dtype=torch.bfloat16)
    w_head = torch.randn(D, device=DEVICE, dtype=torch.bfloat16) * 0.5 + 1.0
    w = w_head.repeat(H)
    eps = 1e-6

    # Reference: production two-stage pipeline (norm kernel + CUDA v2 quant)
    y_ref = layernorm_fn(
        x,
        w,
        None,
        z=z,
        eps=eps,
        group_size=D,
        norm_before_gate=True,
        is_rms_norm=True,
        activation=activation,
    )
    q_ref, s_ref = sglang_per_token_group_quant_fp8(
        y_ref, BLOCK, column_major_scales=True
    )
    s_ref_mn = s_ref.transpose(-1, -2)  # (k//128, m) zero-copy view

    q_new, s_new = rms_norm_gated_fp8_quant(
        x,
        w,
        z,
        eps=eps,
        group_size=D,
        norm_before_gate=True,
        activation=activation,
    )

    label = f"T={T} act={activation}"
    n_code, _ = assert_code_deviation_bounded(q_new, q_ref, label)
    n_scale, _ = assert_scale_deviation_bounded(s_new, s_ref_mn, label)
    print(
        f"[PASS] kernel  T={T:5d} H={H} D={D} act={activation:<7} "
        f"code_dev={n_code}/{q_new.numel()} scale_dev={n_scale}/{s_new.numel()}"
    )
    return y_ref, q_ref, s_ref_mn, q_new, s_new


def run_gemm_case(T, H, D, activation, seed):
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_gemm_w8a8_block_fp8_linear_with_fallback as fi_gemm,
    )

    y_ref, q_ref, s_ref_mn, q_new, s_new = run_kernel_case(T, H, D, activation, seed)

    N_in = H * D
    n_out = 2048
    torch.manual_seed(seed + 1000)
    weight = (
        (torch.randn(n_out, N_in, device=DEVICE, dtype=torch.float32) * 4.0)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    weight_scale = (
        torch.rand(n_out // BLOCK, N_in // BLOCK, device=DEVICE) * 0.01 + 0.001
    )
    weight_scale_mn = weight_scale.transpose(-1, -2).contiguous()

    out_ref = fi_gemm(
        y_ref,
        weight,
        [BLOCK, BLOCK],
        weight_scale,
        weight_scale_mn=weight_scale_mn,
    )

    # Assertion 3: wrapper pre-quantized plumbing is strictly bitwise when
    # fed the exact codes/scales the internal path would produce.
    out_pre = fi_gemm(
        q_ref,
        weight,
        [BLOCK, BLOCK],
        weight_scale,
        input_scale=s_ref_mn,
        weight_scale_mn=weight_scale_mn,
    )
    assert out_pre.dtype == out_ref.dtype == torch.bfloat16
    assert torch.equal(out_pre, out_ref), (
        f"wrapper pre-quant plumbing not bitwise (T={T})"
    )

    # Assertion 4: rows with bitwise-identical fused inputs must produce
    # bitwise-identical outputs.
    out_fused = fi_gemm(
        q_new,
        weight,
        [BLOCK, BLOCK],
        weight_scale,
        input_scale=s_new,
        weight_scale_mn=weight_scale_mn,
    )
    row_ok = (q_new.view(torch.uint8) == q_ref.view(torch.uint8)).all(dim=1) & (
        s_new == s_ref_mn
    ).all(dim=0)
    assert torch.equal(out_fused[row_ok], out_ref[row_ok]), (
        f"clean rows diverged in GEMM output (T={T})"
    )
    n_dirty = int((~row_ok).sum())
    assert n_dirty <= max(1, int(TOL_CODE_FRACTION * q_new.numel())), (
        f"too many deviating rows: {n_dirty}"
    )
    print(
        f"[PASS] gemm    T={T:5d} H={H} D={D} act={activation:<7} "
        f"plumbing=bitwise dirty_rows={n_dirty}/{T}"
    )


def main():
    assert torch.cuda.is_available()
    run_kernel_case(T=8192, H=32, D=128, activation="sigmoid", seed=0)
    run_kernel_case(T=8192, H=32, D=128, activation="swish", seed=1)
    run_kernel_case(T=4093, H=32, D=128, activation="sigmoid", seed=2)
    run_kernel_case(T=17, H=32, D=128, activation="sigmoid", seed=3)
    run_gemm_case(T=8192, H=32, D=128, activation="sigmoid", seed=4)
    run_gemm_case(T=4093, H=32, D=128, activation="sigmoid", seed=5)  # m%4 pad
    run_gemm_case(T=8192, H=32, D=128, activation="swish", seed=6)
    print("ALL PASS")


if __name__ == "__main__":
    main()
