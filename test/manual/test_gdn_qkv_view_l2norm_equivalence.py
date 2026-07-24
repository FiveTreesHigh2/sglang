"""Bitwise equivalence check for the GDN qkv-split view shortcut (action B).

The prefill path replaces fused_qkv_split_gdn_prefill's q/k copies with
strided views normalized in place by l2norm_fwd_packed, and disables the
in-kernel qk l2norm. This test asserts, on production Qwen3.5 GDN shapes:

1. l2norm_fwd_packed(strided q/k view) == l2norm_fwd(dense q/k copy) (bitwise)
2. v slice .contiguous() == fused split's v output (bitwise)
3. chunk_gated_delta_rule(new path: pre-normalized q/k, in-kernel norm off)
   == (old path: dense q/k/v, in-kernel norm on) for o and the state pool
   (bitwise; separate pool clones because the kernel updates state in place)

Run: python test_gdn_qkv_view_l2norm_equivalence.py
"""

import sys

import torch


def _import_paths():
    try:
        from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.kernels.ops.attention.fla.l2norm import (
            l2norm_fwd,
            l2norm_fwd_packed,
        )
    except ImportError:
        from sglang.srt.layers.attention.fla.gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
        )
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.srt.layers.attention.fla.l2norm import (
            l2norm_fwd,
            l2norm_fwd_packed,
        )
    return fused_qkv_split_gdn_prefill, chunk_gated_delta_rule, l2norm_fwd, l2norm_fwd_packed


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
    fused_split, chunk_rule, l2norm_fwd, l2norm_fwd_packed = _import_paths()
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(20260724)

    # Qwen3.5 GDN shapes (TP1)
    num_q_heads, num_k_heads, num_v_heads = 16, 16, 32
    head_dim = 128
    q_dim = num_q_heads * head_dim
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    qkv_dim = q_dim + k_dim + v_dim
    lengths = [1, 7, 63, 64, 65, 130, 257, 511, 1023, 2048, 4031]
    T = sum(lengths)
    pool_size = 32

    mixed_qkv = torch.randn(T, qkv_dim, dtype=dtype, device=device)

    ok = True

    # ── old path tensors ──
    q_old, k_old, v_old = fused_split(
        mixed_qkv, num_q_heads, num_k_heads, num_v_heads, head_dim, head_dim, head_dim
    )

    # ── 1. packed l2norm vs dense l2norm ──
    q_view = mixed_qkv[:, :q_dim].unflatten(-1, (num_q_heads, head_dim))
    k_view = mixed_qkv[:, q_dim : q_dim + k_dim].unflatten(-1, (num_k_heads, head_dim))
    q_new = l2norm_fwd_packed(q_view)
    k_new = l2norm_fwd_packed(k_view)
    ok &= _check("q l2norm (packed view vs dense)", l2norm_fwd(q_old), q_new)
    ok &= _check("k l2norm (packed view vs dense)", l2norm_fwd(k_old), k_new)

    # ── 2. v extraction ──
    v_new = (
        mixed_qkv[:, q_dim + k_dim :]
        .contiguous()
        .view(1, T, num_v_heads, head_dim)
    )
    ok &= _check("v (slice contiguous vs fused split)", v_old, v_new)

    # ── 3. end-to-end chunk_gated_delta_rule ──
    cu = torch.zeros(len(lengths) + 1, dtype=torch.long, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.long, device=device), 0)
    indices = torch.randperm(pool_size, device=device)[: len(lengths)].to(torch.int32)
    pool_init = (
        torch.randn(pool_size, num_v_heads, head_dim, head_dim,
                    dtype=torch.float32, device=device) * 0.1
    )
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, T, num_v_heads, dtype=torch.float32, device=device)
    )
    beta = torch.sigmoid(torch.randn(1, T, num_v_heads, dtype=dtype, device=device))

    def run(q, k, v, in_kernel_norm):
        pool = pool_init.clone()
        o, _, _ = chunk_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=indices,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=in_kernel_norm,
        )
        return o, pool

    o_ref, pool_ref = run(q_old, k_old, v_old, True)
    o_new, pool_new = run(q_new, k_new, v_new, False)
    ok &= _check("chunk_gated_delta_rule o (new vs old path)", o_ref, o_new)
    ok &= _check("state pool (new vs old path)", pool_ref, pool_new)

    if not ok:
        print("\nqkv view + packed l2norm: NOT equivalent")
        sys.exit(1)
    print("\nqkv view + packed l2norm: all bitwise equivalent, OK")


if __name__ == "__main__":
    main()
