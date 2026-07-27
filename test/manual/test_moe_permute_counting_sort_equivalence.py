"""Equivalence test for the counting-sort moe_permute_prepare path (M5).

The counting-sort path replaces torch.sort (cub radix sort) for building
expert_offsets / src2dst. Its atomic scatter is NOT stable, so src2dst is
not expected to match the radix-sorted one element-wise. The contract that
downstream consumers (fused quant_scatter pack -> grouped GEMM ->
post_reorder gather) rely on is exactly:

  C1. expert_offsets bitwise-equal to the sorted path (CSR segment starts);
  C2. src2dst is a valid permutation of [0, numel);
  C3. every route lands inside its own expert segment:
      expert_offsets[e] <= src2dst[i] < expert_offsets[e+1] where
      e = topk_ids.flatten()[i].

Rows are computed independently and gathered back through the same src2dst,
so C1-C3 imply the final MoE output is bitwise unchanged.

Run on the server:
  /home/logs/sennian/pro5000-fi-moe/.venv/bin/python3 \
      test/manual/test_moe_permute_counting_sort_equivalence.py
"""

import torch

import sglang.jit_kernel.moe_permute_prepare as mpp

DEVICE = "cuda"


def run_case(T: int, top_k: int, num_experts: int, seed: int, skew: bool):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if skew:
        # Skewed routing: a few hot experts stress the atomic scatter.
        logits = torch.randn(
            T, num_experts, generator=g, device=DEVICE
        ) + torch.linspace(0, 4, num_experts, device=DEVICE)
    else:
        logits = torch.randn(T, num_experts, generator=g, device=DEVICE)
    topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32).contiguous()
    numel = topk_ids.numel()
    flat = topk_ids.view(-1).long()

    # Reference: sorted path (force the legacy branch).
    saved = mpp.MOE_PERMUTE_COUNTING_SORT
    try:
        mpp.MOE_PERMUTE_COUNTING_SORT = False
        off_ref, s2d_ref = mpp.moe_permute_prepare(topk_ids, num_experts)
        mpp.MOE_PERMUTE_COUNTING_SORT = True
        off_new, s2d_new = mpp.moe_permute_prepare(topk_ids, num_experts)
    finally:
        mpp.MOE_PERMUTE_COUNTING_SORT = saved

    # C1: identical CSR offsets.
    assert off_new.dtype == off_ref.dtype, (off_new.dtype, off_ref.dtype)
    assert torch.equal(off_new, off_ref), "expert_offsets differ"

    # C2: valid permutation.
    sorted_dst = s2d_new.long().sort().values
    assert torch.equal(
        sorted_dst, torch.arange(numel, device=DEVICE)
    ), "src2dst is not a permutation"

    # C3: segment membership.
    off64 = off_ref.long()
    lo = off64[flat]
    hi = off64[flat + 1]
    dst = s2d_new.long()
    assert bool(((dst >= lo) & (dst < hi)).all()), "route outside its expert segment"

    tag = "skewed" if skew else "uniform"
    print(f"[PASS] T={T} top_k={top_k} E={num_experts} ({tag})")


def main():
    run_case(T=8192, top_k=8, num_experts=256, seed=3, skew=False)
    run_case(T=8192, top_k=8, num_experts=256, seed=5, skew=True)
    run_case(T=8192 - 24, top_k=8, num_experts=256, seed=7, skew=False)
    run_case(T=3, top_k=8, num_experts=256, seed=9, skew=False)  # decode-like
    print("ALL PASS")


if __name__ == "__main__":
    main()
