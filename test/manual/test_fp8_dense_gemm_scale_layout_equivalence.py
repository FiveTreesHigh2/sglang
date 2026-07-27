"""Bitwise equivalence test for the dense FP8 GEMM scale-layout rework.

Validates the flashinfer_cutlass dense path after switching the activation
quant to column_major_scales=True (A scale written directly into the
CUTLASS MN-major storage, consumed as a zero-copy transpose view) and to
the load-time pre-transposed weight scale (weight_scale_mn).

Note: sglang_per_token_group_quant_fp8_row_padded is deliberately NOT used:
its direct-op call path produces ~0.1% one-code-step rounding differences
vs the wrapped kernel path and fails the bitwise standard.

Checks (all torch.equal, i.e. bitwise):
  1. column_major_scales=True vs False: identical q codes and scale values.
  2. GEMM output: new wrapper (with and without weight_scale_mn) vs the
     legacy call sequence reproduced inline.
  3. Non-multiple-of-4 m (decode shapes m=1..3): output rows match the
     same computation performed on an explicitly zero-padded m=4 input.

Run on the server:
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
      test/manual/test_fp8_dense_gemm_scale_layout_equivalence.py
"""

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

BLOCK = 128
DEVICE = "cuda"


def make_weight(n: int, k: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w_bf16 = torch.randn(n, k, generator=g, device=DEVICE, dtype=torch.float32)
    weight = (w_bf16 / w_bf16.abs().amax()).to(torch.float8_e4m3fn)
    weight_scale = (
        torch.rand(
            n // BLOCK, k // BLOCK, generator=g, device=DEVICE, dtype=torch.float32
        )
        * 0.01
        + 0.001
    )
    return weight, weight_scale


def legacy_gemm(input_2d, weight, weight_scale):
    """Reproduce the pre-rework call sequence exactly."""
    from sglang.srt.layers.quantization.fp8_utils import gemm_fp8_nt_groupwise

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, BLOCK, column_major_scales=False
    )
    x_scale = x_scale.transpose(-1, -2).contiguous()
    ws = weight_scale.transpose(-1, -2).contiguous()
    return gemm_fp8_nt_groupwise(q_input, weight, x_scale, ws, out_dtype=input_2d.dtype)


def main():
    torch.manual_seed(7)
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_gemm_w8a8_block_fp8_linear_with_fallback as new_gemm,
    )

    m, k, n = 8192, 2048, 2048
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
    weight, weight_scale = make_weight(n, k, seed=11)
    weight_scale_mn = weight_scale.transpose(-1, -2).contiguous()

    # --- 1: col-major quant bitwise-equal to row-major quant ---
    q_ref, s_ref = sglang_per_token_group_quant_fp8(x, BLOCK, column_major_scales=False)
    q_new, s_new = sglang_per_token_group_quant_fp8(x, BLOCK, column_major_scales=True)
    assert torch.equal(
        q_new.view(torch.uint8), q_ref.view(torch.uint8)
    ), "quantized activations differ"
    assert torch.equal(s_new.contiguous(), s_ref.contiguous()), "A-scale values differ"
    s_mn = s_new.transpose(-1, -2)
    assert s_mn.shape == (k // BLOCK, m) and s_mn.is_contiguous(), (
        "col-major scale storage is not (k//block, m) contiguous: "
        f"shape={tuple(s_mn.shape)} strides={s_mn.stride()}"
    )
    print("[PASS] 1: col-major quant bitwise-equal to row-major quant (m=8192)")

    # --- 2: full wrapper vs legacy sequence ---
    out_legacy = legacy_gemm(x, weight, weight_scale)
    out_new = new_gemm(x, weight, [BLOCK, BLOCK], weight_scale)
    out_new_mn = new_gemm(
        x, weight, [BLOCK, BLOCK], weight_scale, weight_scale_mn=weight_scale_mn
    )
    assert torch.equal(out_new, out_legacy), "wrapper output != legacy output"
    assert torch.equal(out_new_mn, out_legacy), "weight_scale_mn output differs"
    print("[PASS] 2: new wrapper bitwise-equal to legacy sequence (m=8192)")

    # --- 3: decode shapes m=1..3 vs explicitly zero-padded m=4 ---
    for m_small in (1, 2, 3):
        xs = torch.randn(m_small, k, device=DEVICE, dtype=torch.bfloat16)
        x_pad = torch.zeros(4, k, device=DEVICE, dtype=torch.bfloat16)
        x_pad[:m_small] = xs
        out_small = new_gemm(
            xs, weight, [BLOCK, BLOCK], weight_scale, weight_scale_mn=weight_scale_mn
        )
        out_pad = new_gemm(
            x_pad, weight, [BLOCK, BLOCK], weight_scale, weight_scale_mn=weight_scale_mn
        )
        assert out_small.shape[0] == m_small
        assert torch.equal(out_small, out_pad[:m_small]), f"m={m_small} rows differ"
        print(f"[PASS] 3: m={m_small} matches zero-padded m=4 computation")

    print("ALL PASS")


if __name__ == "__main__":
    main()
