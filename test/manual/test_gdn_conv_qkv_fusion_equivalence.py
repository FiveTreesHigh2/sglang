"""Equivalence test for the GDN1 fused conv qkv-split epilogue.

Validates causal_conv1d_fn_qkv_split against the production pipeline
(causal_conv1d_fn -> packed transit buffer -> l2norm_fwd_packed(q/k) +
extract_v_gdn_prefill(v)):

  1. "v" mode (qk_l2norm=False): strictly bitwise everywhere --
     v equals the extract_v output, raw q/k equal the transit columns,
     external dense l2norm on them equals the packed-view l2norm, and the
     conv_state pools are bitwise identical after both runs.
  2. "full" mode (qk_l2norm=True): v and conv_states stay bitwise; q/k are
     subject to the quantitative reassociation standard established for
     D1-a (deviation fraction <= 2e-6, every mismatch within one bf16 ulp),
     because the in-epilogue sum(x^2) reduction may associate differently
     from l2norm_fwd_kernel_strided.
  3. Varlen coverage: multi-sequence batches with mixed has_initial_state,
     tail chunks shorter than BLOCK_M, and a sequence shorter than the conv
     state length (exercising all conv_state update branches).

Run on the server:
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
      test/manual/test_gdn_conv_qkv_fusion_equivalence.py
"""

import torch

from sglang.jit_kernel.triton.gdn_fused_proj import extract_v_gdn_prefill
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd, l2norm_fwd_packed
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_fn_qkv_split,
)

DEVICE = "cuda"
HQ, HK, HV, D = 16, 16, 32, 128
Q_DIM, K_DIM, V_DIM = HQ * D, HK * D, HV * D
DIM = Q_DIM + K_DIM + V_DIM
WIDTH = 4
TOL_CODE_FRACTION = 2e-6


def bf16_rank(t):
    # Monotonic integer rank of bf16 values (equal rank for +/-0.0):
    # positive codes map above 0x8000, negative codes mirror below it.
    u = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(u < 0x8000, u + 0x8000, 0x10000 - u)


def assert_qk_bounded(a, b, label):
    mism = a.view(torch.int16) != b.view(torch.int16)
    frac = mism.float().mean().item()
    assert frac <= TOL_CODE_FRACTION, (
        f"{label}: deviation fraction {frac:.2e} > {TOL_CODE_FRACTION:.0e}"
    )
    n = int(mism.sum())
    if n:
        ra, rb = bf16_rank(a[mism]), bf16_rank(b[mism])
        step = (ra - rb).abs()
        assert (step <= 1).all(), (
            f"{label}: mismatch beyond one bf16 ulp (max rank step {int(step.max())})"
        )
    return n


def make_case(seq_lens, has_init, seed, with_bias):
    torch.manual_seed(seed)
    T = sum(seq_lens)
    x = torch.randn(T, DIM, device=DEVICE, dtype=torch.bfloat16).transpose(0, 1)
    weight = torch.randn(DIM, WIDTH, device=DEVICE, dtype=torch.bfloat16) * 0.3
    bias = (
        torch.randn(DIM, device=DEVICE, dtype=torch.bfloat16) * 0.1
        if with_bias
        else None
    )
    n_cache = len(seq_lens) + 3
    pool = torch.randn(
        n_cache, DIM, WIDTH - 1, device=DEVICE, dtype=torch.bfloat16
    )
    cache_indices = torch.arange(
        len(seq_lens), device=DEVICE, dtype=torch.int32
    ) + 2
    qsl = torch.zeros(len(seq_lens) + 1, device=DEVICE, dtype=torch.int32)
    qsl[1:] = torch.tensor(seq_lens, device=DEVICE, dtype=torch.int32).cumsum(0)
    his = torch.tensor(has_init, device=DEVICE, dtype=torch.bool)
    return x, weight, bias, pool, cache_indices, qsl, his


def run_case(seq_lens, has_init, seed, with_bias):
    x, weight, bias, pool, cache_indices, qsl, his = make_case(
        seq_lens, has_init, seed, with_bias
    )
    T = sum(seq_lens)

    # ---- reference pipeline ----
    pool_ref = pool.clone()
    mixed = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=pool_ref,
        has_initial_state=his,
        cache_indices=cache_indices,
        query_start_loc=qsl,
        seq_lens_cpu=list(seq_lens),
        activation="silu",
    ).transpose(0, 1)[:T]
    q_ref = l2norm_fwd_packed(mixed[:, :Q_DIM].unflatten(-1, (HQ, D)))
    k_ref = l2norm_fwd_packed(
        mixed[:, Q_DIM : Q_DIM + K_DIM].unflatten(-1, (HK, D))
    )
    v_ref = extract_v_gdn_prefill(mixed, Q_DIM + K_DIM, HV, D)

    common = dict(
        conv_states=None,
        query_start_loc=qsl,
        seq_lens_cpu=list(seq_lens),
        num_q_heads=HQ,
        num_k_heads=HK,
        num_v_heads=HV,
        head_qk_dim=D,
        head_v_dim=D,
        cache_indices=cache_indices,
        has_initial_state=his,
        activation="silu",
    )

    # ---- fused "v" mode: strictly bitwise ----
    pool_v = pool.clone()
    common["conv_states"] = pool_v
    q_raw, k_raw, v_v = causal_conv1d_fn_qkv_split(
        x, weight, bias, qk_l2norm=False, **common
    )
    assert torch.equal(v_v, v_ref), "v differs in v mode"
    assert torch.equal(
        q_raw.view(T, Q_DIM), mixed[:, :Q_DIM]
    ), "raw q differs from transit columns"
    assert torch.equal(
        k_raw.view(T, K_DIM), mixed[:, Q_DIM : Q_DIM + K_DIM]
    ), "raw k differs from transit columns"
    assert torch.equal(l2norm_fwd(q_raw), q_ref), "dense l2norm(q_raw) != q_ref"
    assert torch.equal(l2norm_fwd(k_raw), k_ref), "dense l2norm(k_raw) != k_ref"
    assert torch.equal(pool_v, pool_ref), "conv_states differ in v mode"

    # ---- fused "full" mode: v/states bitwise, q/k bounded ----
    pool_f = pool.clone()
    common["conv_states"] = pool_f
    q_f, k_f, v_f = causal_conv1d_fn_qkv_split(
        x, weight, bias, qk_l2norm=True, **common
    )
    assert torch.equal(v_f, v_ref), "v differs in full mode"
    assert torch.equal(pool_f, pool_ref), "conv_states differ in full mode"
    nq = assert_qk_bounded(q_f, q_ref, "q full mode")
    nk = assert_qk_bounded(k_f, k_ref, "k full mode")

    print(
        f"[PASS] seqs={seq_lens} init={has_init} bias={bias is not None} "
        f"q_dev={nq}/{q_f.numel()} k_dev={nk}/{k_f.numel()}"
    )


def main():
    assert torch.cuda.is_available()
    # single long sequence, no initial state, no bias
    run_case([8192], [False], seed=0, with_bias=False)
    # varlen: multi-chunk + tail < BLOCK_M + seqlen < state_len, mixed init
    run_case(
        [4096, 33, 2048, 7, 2],
        [True, False, True, False, True],
        seed=1,
        with_bias=True,
    )
    # two mid-size sequences, both with initial states
    run_case([1024, 511], [True, True], seed=2, with_bias=False)
    print("ALL PASS")


if __name__ == "__main__":
    main()
