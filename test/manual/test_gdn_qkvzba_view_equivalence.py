"""Bitwise equivalence check for the qkvzba prefill view shortcut.

The contiguous-layout fused kernel (fused_qkvzba_split_reshape_cat_contiguous)
copies the leading qkv_dim columns of projected_states_qkvz verbatim into a
fresh mixed_qkv buffer. The prefill shortcut replaces that copy with a strided
view. This test asserts, on production Qwen3.5 shapes:

1. mixed_qkv view == fused kernel mixed_qkv (bitwise)
2. z / b / a from the view path == fused kernel outputs (bitwise)
3. causal_conv1d_fn(strided view) == causal_conv1d_fn(contiguous copy)
   (bitwise) and the strided-input output buffer stays channel-last dense
4. fused_qkv_split_gdn_prefill(strided view) == (contiguous copy) (bitwise)

Run: python test_gdn_qkvzba_view_equivalence.py
"""

import sys

import torch


def _import_paths():
    try:
        from sglang.jit_kernel.triton.gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
            fused_qkvzba_split_reshape_cat_contiguous,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    except ImportError:
        from sglang.srt.layers.attention.fla.gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
            fused_qkvzba_split_reshape_cat_contiguous,
        )
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
    return (
        fused_qkvzba_split_reshape_cat_contiguous,
        fused_qkv_split_gdn_prefill,
        causal_conv1d_fn,
    )


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
    fused_cat, fused_split, conv_fn = _import_paths()
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(20260722)

    # Qwen3.5-35B-A3B GDN shapes (TP1)
    num_k_heads, num_v_heads = 16, 32
    head_k, head_v = 128, 128
    conv_width = 4
    lengths = [1, 7, 63, 64, 65, 130, 257, 511, 1023, 2048, 4031]
    seq_len = sum(lengths)  # ~8.2k tokens, one prefill-chunk scale

    qkv_dim = 2 * num_k_heads * head_k + num_v_heads * head_v
    qkvz_dim = qkv_dim + num_v_heads * head_v

    qkvz = torch.randn(seq_len, qkvz_dim, dtype=dtype, device=device)
    ba = torch.randn(seq_len, 2 * num_v_heads, dtype=dtype, device=device)

    ok = True

    # ── 1+2. view path vs fused kernel ──
    ref_qkv, ref_z, ref_b, ref_a = fused_cat(
        qkvz, ba, num_k_heads, num_v_heads, head_k, head_v
    )
    view_qkv = qkvz[:, :qkv_dim]
    view_z = qkvz[:, qkv_dim:].unflatten(-1, (num_v_heads, head_v))
    view_b, view_a = ba.split([num_v_heads, num_v_heads], dim=-1)
    ok &= _check("mixed_qkv (view vs fused)", ref_qkv, view_qkv)
    ok &= _check("z (view vs fused)", ref_z, view_z)
    ok &= _check("b (view vs fused)", ref_b, view_b.contiguous())
    ok &= _check("a (view vs fused)", ref_a, view_a.contiguous())

    # ── 3. conv1d on strided view vs contiguous copy ──
    weight = torch.randn(qkv_dim, conv_width, dtype=dtype, device=device)
    bias = torch.randn(qkv_dim, dtype=dtype, device=device)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lengths, device=device), 0).to(torch.int32)
    seq_lens_cpu = torch.tensor(lengths, dtype=torch.int32)
    has_init = torch.zeros(len(lengths), dtype=torch.bool, device=device)
    cache_idx = torch.arange(len(lengths), dtype=torch.int32, device=device)

    def run_conv(x_2d):
        # backend passes (dim, cu_seqlen) channel-last, fresh states each call
        states = torch.zeros(
            len(lengths), qkv_dim, conv_width - 1, dtype=dtype, device=device
        )
        out = conv_fn(
            x_2d.transpose(0, 1),
            weight,
            bias,
            activation="silu",
            conv_states=states,
            has_initial_state=has_init,
            cache_indices=cache_idx,
            query_start_loc=cu,
            seq_lens_cpu=seq_lens_cpu,
        )
        return out.transpose(0, 1)[:seq_len].contiguous(), states

    out_view, states_view = run_conv(view_qkv)
    out_contig, states_contig = run_conv(view_qkv.contiguous())
    ok &= _check("conv1d out (strided vs contiguous)", out_contig, out_view)
    ok &= _check("conv1d states (strided vs contiguous)", states_contig, states_view)

    # ── 4. fused_qkv_split on strided view vs contiguous ──
    num_q_heads = num_k_heads  # q_dim == k_dim for GDN
    sp_view = fused_split(
        view_qkv, num_q_heads, num_k_heads, num_v_heads, head_k, head_k, head_v
    )
    sp_contig = fused_split(
        view_qkv.contiguous(),
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_k,
        head_k,
        head_v,
    )
    for name, r, n in zip(("q", "k", "v"), sp_contig, sp_view):
        ok &= _check(f"fused_qkv_split {name} (strided vs contiguous)", r, n)

    if not ok:
        print("\nqkvzba view shortcut: NOT equivalent")
        sys.exit(1)
    print("\nqkvzba view shortcut: all bitwise equivalent, OK")


if __name__ == "__main__":
    main()
