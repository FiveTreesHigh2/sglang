"""Bitwise equivalence test for the fused GDN gated-norm + fp8 quant epilogue.

Validates rms_norm_gated_fp8_quant (audit D1-a) against the production
two-stage pipeline it replaces:

  stage 1: layernorm_fn(x, w, None, z=z, group_size=head_d, is_rms_norm=True)
           -> bf16 norm output y
  stage 2: sglang_per_token_group_quant_fp8(y, 128, column_major_scales=True)
           inside flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
           -> fp8 codes + MN-major (k//128, m) scale -> CUTLASS GEMM

Checks (all torch.equal, i.e. bitwise):
  1. fp8 codes: fused epilogue vs layernorm_fn + standalone quant.
  2. A scales: fused (ngroups, M) contiguous vs quant col-major transpose view.
  3. End-to-end GEMM output: wrapper fed bf16 y (internal quant) vs wrapper
     fed the fused (q, s) pre-quantized pair, including weight_scale_mn.
  4. m % 4 != 0 shapes exercise the wrapper's zero-pad branch on the
     pre-quantized inputs.

Shapes mirror Qwen3.5 GDN: H=32 v-heads, D=128, hidden 4096 -> out_proj
k=4096, n=2048; activations cover both swish and sigmoid gate types.

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


def run_kernel_case(T, H, D, activation, seed):
    torch.manual_seed(seed)
    N = H * D
    x = torch.randn(T, N, device=DEVICE, dtype=torch.bfloat16) * 2.0
    z = torch.randn(T, N, device=DEVICE, dtype=torch.bfloat16)
    # per-head weight expanded exactly like _norm_weight_grouped
    w_head = torch.randn(D, device=DEVICE, dtype=torch.bfloat16) * 0.5 + 1.0
    w = w_head.repeat(H)
    eps = 1e-6

    # Reference: production two-stage pipeline
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

    # Fused epilogue
    q_new, s_new = rms_norm_gated_fp8_quant(
        x,
        w,
        z,
        eps=eps,
        group_size=D,
        norm_before_gate=True,
        activation=activation,
    )

    assert q_new.shape == q_ref.shape and q_new.dtype == q_ref.dtype
    assert torch.equal(
        q_new.view(torch.uint8), q_ref.view(torch.uint8)
    ), f"fp8 codes differ (T={T}, act={activation})"
    assert s_new.shape == tuple(s_ref_mn.shape) or s_new.shape == s_ref_mn.shape
    assert torch.equal(s_new, s_ref_mn), f"scales differ (T={T}, act={activation})"
    print(f"[PASS] kernel  T={T:5d} H={H} D={D} act={activation}")
    return x, z, w, eps, y_ref, q_new, s_new


def run_gemm_case(T, H, D, activation, seed):
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_gemm_w8a8_block_fp8_linear_with_fallback as fi_gemm,
    )

    x, z, w, eps, y_ref, q_new, s_new = run_kernel_case(T, H, D, activation, seed)

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
    out_new = fi_gemm(
        q_new,
        weight,
        [BLOCK, BLOCK],
        weight_scale,
        input_scale=s_new,
        weight_scale_mn=weight_scale_mn,
    )
    assert out_new.dtype == out_ref.dtype == torch.bfloat16
    assert torch.equal(out_new, out_ref), f"GEMM outputs differ (T={T})"
    print(f"[PASS] gemm    T={T:5d} H={H} D={D} act={activation}")


def main():
    assert torch.cuda.is_available()
    # Kernel-level bitwise checks
    run_kernel_case(T=8192, H=32, D=128, activation="sigmoid", seed=0)
    run_kernel_case(T=8192, H=32, D=128, activation="swish", seed=1)
    run_kernel_case(T=4093, H=32, D=128, activation="sigmoid", seed=2)
    run_kernel_case(T=17, H=32, D=128, activation="sigmoid", seed=3)
    # End-to-end wrapper checks (require flashinfer; run on the server)
    run_gemm_case(T=8192, H=32, D=128, activation="sigmoid", seed=4)
    run_gemm_case(T=4093, H=32, D=128, activation="sigmoid", seed=5)  # m%4 pad
    run_gemm_case(T=8192, H=32, D=128, activation="swish", seed=6)
    print("ALL PASS")


if __name__ == "__main__":
    main()
