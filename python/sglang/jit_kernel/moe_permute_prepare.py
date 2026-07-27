from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Counting sort as an alternative to torch.sort (cub radix sort). Measured
# on RTX PRO 5000 serving (64k routes, 256 experts): the naive global-atomic
# implementation is a net regression (+0.63 ms/fwd vs radix: histogram 23.6us
# per launch from 64k atomicAdds serializing on a 257-slot hot bucket array).
# Beating cub requires smem-privatized histograms + two-level prefix sums;
# not pursued. Default OFF, kept for future study.
MOE_PERMUTE_COUNTING_SORT = (
    os.getenv("SGLANG_MOE_PERMUTE_COUNTING_SORT", "0") == "1"
)


@triton.jit
def _count_kernel(topk_ids, counts, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(topk_ids + offs, mask=mask, other=0)
    # counts is laid out with a leading zero slot: slot e+1 accumulates
    # expert e so that cumsum(counts) directly yields the CSR offsets.
    tl.atomic_add(counts + e + 1, 1, mask=mask)


@triton.jit
def _scatter_rank_kernel(topk_ids, cursors, src2dst, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(topk_ids + offs, mask=mask, other=0)
    pos = tl.atomic_add(cursors + e, 1, mask=mask)
    tl.store(src2dst + offs, pos, mask=mask)


def _moe_permute_prepare_counting(
    topk_ids: torch.Tensor,
    num_experts: int,
    use_int64_offset: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = topk_ids.view(-1)
    numel = flat.numel()
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _count_kernel[grid](flat, counts, numel, BLOCK=BLOCK)
    offset_dtype = torch.int64 if use_int64_offset else torch.int32
    expert_offsets = torch.cumsum(counts, 0, dtype=offset_dtype)
    # copy=True is required: with int32 offsets, .to() would alias
    # expert_offsets and the scatter's atomicAdd would corrupt it in place
    # (offsets[e] += cnt_e, shifting the CSR by one segment).
    cursors = expert_offsets[:num_experts].to(dtype=torch.int32, copy=True)
    src2dst = torch.empty(numel, dtype=torch.int32, device=flat.device)
    _scatter_rank_kernel[grid](flat, cursors, src2dst, numel, BLOCK=BLOCK)
    return expert_offsets, src2dst


@cache_once
def _jit_moe_permute_prepare_module() -> Module:
    return load_jit(
        "moe_permute_prepare",
        cuda_files=["moe/moe_permute_prepare.cu"],
        header_only=False,
    )


@register_custom_op(
    op_name="moe_permute_prepare_out",
    mutates_args=["expert_offsets", "src2dst"],
)
def _moe_permute_prepare_out(
    sorted_topk_ids: torch.Tensor,
    reorder_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    src2dst: torch.Tensor,
    num_experts: int,
    use_int64_offset: bool,
    is_ep: bool,
) -> None:
    module = _jit_moe_permute_prepare_module()
    module.moe_permute_prepare(
        sorted_topk_ids,
        reorder_ids,
        expert_offsets,
        src2dst,
        num_experts,
        use_int64_offset,
        is_ep,
    )


def moe_permute_prepare(
    topk_ids: torch.Tensor,
    num_experts: int,
    use_int64_offset: bool = False,
    is_ep: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")

    # Counting-sort fast path (is_ep needs the negative-id exclusion
    # semantics of the sorted path and stays on it).
    if MOE_PERMUTE_COUNTING_SORT and not is_ep:
        return _moe_permute_prepare_counting(
            topk_ids, num_experts, use_int64_offset
        )

    sorted_topk_ids, reorder_ids = torch.sort(topk_ids.flatten())
    offset_dtype = torch.int64 if use_int64_offset else torch.int32
    expert_offsets = torch.empty(
        (num_experts + 1,), dtype=offset_dtype, device=topk_ids.device
    )
    src2dst = torch.empty(
        (topk_ids.numel(),), dtype=torch.int32, device=topk_ids.device
    )

    _moe_permute_prepare_out(
        sorted_topk_ids,
        reorder_ids,
        expert_offsets,
        src2dst,
        num_experts,
        use_int64_offset,
        is_ep,
    )
    return expert_offsets, src2dst
