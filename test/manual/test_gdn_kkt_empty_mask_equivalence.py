"""Bitwise equivalence test for the chunk_fwd A-buffer zeros->empty rework (B8).

The kkt+solve kernel stores only the 10 lower-triangular BC-blocks of each
BT x BT tile of A; the 6 upper blocks previously relied on torch.zeros
pre-fill. A is now torch.empty and the sole consumer
(recompute_w_u_fwd_kernel) applies an element-level causal mask
(row >= col) after its full-tile load.

Equivalence argument verified here:
  - strictly-upper BC-blocks: row < col holds element-wise, masked to zero
    (== the old zeros pre-fill);
  - diagonal BC-blocks: their upper-triangular elements are true zeros
    computed and stored by the solve step, so the mask is idempotent there;
  - the where(...).to(dtype) round-trip is value-preserving for bf16 inputs.

Test procedure:
  1. Poison the caching allocator with NaN-filled buffers so torch.empty
     actually observes garbage (a NaN leaking through the mask would
     propagate into w/u and fail torch.equal).
  2. Reference path: allocate A with torch.zeros, launch the kkt kernel and
     recompute_w_u_fwd manually (mask is idempotent on the zeroed tile, so
     this reproduces the pre-rework numerics).
  3. New path: chunk_gated_delta_rule_fwd_intra (torch.empty inside).
  4. torch.equal on w, u and on the causal (row>=col) region of A.

Run on the server:
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
      test/manual/test_gdn_kkt_empty_mask_equivalence.py
"""

import torch

from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
    chunk_gated_delta_rule_fwd_kkt_solve_kernel,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd

DEVICE = "cuda"
BT, BC = 64, 16


def poison_allocator(nbytes: int):
    """Fill (and free) NaN buffers so subsequent torch.empty sees garbage."""
    junk = [
        torch.full((nbytes // 2,), float("nan"), device=DEVICE, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    del junk
    # No empty_cache(): keep the poisoned blocks inside the caching allocator.


def make_inputs(T: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    B, Hg, H, K, V = 1, 16, 32, 128, 128
    k = torch.randn(B, T, Hg, K, generator=g, device=DEVICE, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, generator=g, device=DEVICE, dtype=torch.bfloat16)
    gate = -torch.rand(B, T, H, generator=g, device=DEVICE, dtype=torch.float32).cumsum(
        1
    ) * 0.1
    beta = torch.rand(B, T, H, generator=g, device=DEVICE, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, T], device=DEVICE, dtype=torch.int64)
    return k, v, gate, beta, cu_seqlens


def reference_intra(k, v, gate, beta, cu_seqlens):
    """Pre-rework numerics: zeros-initialized A + same kernels."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k, g=gate, beta=beta, A=A, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, H=H, Hg=Hg, K=K, BT=BT, BC=BC,
    )
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=gate,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    return w, u, A


def main():
    for T, seed in ((8192, 3), (8192 - 24, 5)):  # aligned + ragged tail
        k, v, gate, beta, cu_seqlens = make_inputs(T, seed)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        w_ref, u_ref, A_ref = reference_intra(k, v, gate, beta, cu_seqlens)

        poison_allocator(1 * 8192 * 32 * BT * 2)
        w_new, u_new, A_new = chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=gate, beta=beta,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )

        assert not torch.isnan(w_new.float()).any(), "NaN leaked into w"
        assert not torch.isnan(u_new.float()).any(), "NaN leaked into u"
        assert torch.equal(w_new, w_ref), f"w differs (T={T})"
        assert torch.equal(u_new, u_ref), f"u differs (T={T})"

        # A: compare the causal region only (upper triangle of A_new is
        # unwritten garbage by design; consumers never read it unmasked).
        # rows of A are tile-local: row index = t % BT
        row_in_tile = (torch.arange(T, device=DEVICE) % BT)[:, None]
        col = torch.arange(BT, device=DEVICE)[None, :]
        keep = (row_in_tile >= col).view(1, T, 1, BT)
        assert torch.equal(
            A_new.masked_fill(~keep, 0), A_ref.masked_fill(~keep, 0)
        ), f"A causal region differs (T={T})"
        print(f"[PASS] T={T}: w/u bitwise-equal, A causal region bitwise-equal")

    print("ALL PASS")


if __name__ == "__main__":
    main()
