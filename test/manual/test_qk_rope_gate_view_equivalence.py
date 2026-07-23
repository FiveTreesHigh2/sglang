"""Bitwise equivalence check for the qk_rope gate de-materialization.

fused_qk_gemma_rmsnorm_rope_gate(materialize_gate=False) skips the in-kernel
gate copy (67MB read + 67MB write per full-attn layer at T=8192) and returns
the gate as a strided view of the interleaved q_gate instead. This test
asserts, on production Qwen3.5 full-attn shapes (16 q heads / 2 kv heads /
head_dim 256 / rotary_dim 64):

1. q_out / k_out identical between materialize_gate True and False
   (WRITE_GATE must not perturb codegen of the q/k path)
2. gate view == materialized gate_out (bitwise)
3. fused_sigmoid_mul(attn, strided gate view) == (contiguous gate) (bitwise)

Run: python test_qk_rope_gate_view_equivalence.py
"""

import sys

import torch


def _import_paths():
    try:
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )
        from sglang.kernels.ops.layernorm.elementwise import fused_sigmoid_mul
    except ImportError:
        from sglang.srt.layers.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )
        from sglang.srt.layers.elementwise import fused_sigmoid_mul
    return fused_qk_gemma_rmsnorm_rope_gate, fused_sigmoid_mul


def _check(name, ref, new):
    if torch.equal(ref, new):
        print(f"[PASS] {name}: bitwise identical, shape={tuple(ref.shape)}")
        return True
    diff = (ref.float() - new.float()).abs()
    mism = (ref != new).sum().item()
    print(
        f"[FAIL] {name}: {mism}/{ref.numel()} elements differ, "
        f"max_abs_diff={diff.max().item():.3e}"
    )
    return False


def main():
    fused_rope, fused_sigmoid_mul = _import_paths()
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(20260723)

    # Qwen3.5 full-attn shapes (TP1)
    num_q_heads, num_kv_heads = 16, 2
    head_dim, rotary_dim = 256, 64
    eps = 1e-6
    T = 8200
    max_pos = 32768

    # q_gate as a strided split of a wider qkv row (matches production layout)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn(T, q_size * 2 + kv_size * 2, dtype=dtype, device=device)
    q_gate, k, _v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    cos_sin = torch.randn(max_pos, rotary_dim, dtype=dtype, device=device)
    positions = torch.randint(0, max_pos, (T,), dtype=torch.long, device=device)

    def run(materialize):
        return fused_rope(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            has_gate=True,
            materialize_gate=materialize,
        )

    q_ref, k_ref, gate_ref = run(True)
    q_new, k_new, gate_view = run(False)

    ok = True
    ok &= _check("q_out (write_gate on vs off)", q_ref, q_new)
    ok &= _check("k_out (write_gate on vs off)", k_ref, k_new)
    ok &= _check("gate (view vs materialized)", gate_ref, gate_view)
    assert not gate_view.is_contiguous(), "gate view unexpectedly contiguous"

    # downstream: sigmoid-mul gating on strided view vs contiguous gate
    attn = torch.randn(T, q_size, dtype=dtype, device=device)
    out_ref = fused_sigmoid_mul(attn, gate_ref.view(T, num_q_heads, head_dim))
    out_new = fused_sigmoid_mul(attn, gate_view)
    ok &= _check("fused_sigmoid_mul (strided vs contiguous gate)", out_ref, out_new)

    if not ok:
        print("\nqk_rope gate view: NOT equivalent")
        sys.exit(1)
    print("\nqk_rope gate view: all bitwise equivalent, OK")


if __name__ == "__main__":
    main()
