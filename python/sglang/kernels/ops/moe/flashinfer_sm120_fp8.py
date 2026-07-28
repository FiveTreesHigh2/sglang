from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.flashinfer_sm120_fp8_moe import (
    flashinfer_sm120_fp8_quant_scatter_pack,
    flashinfer_sm120_fp8_silu_quant_pack,
)


def flashinfer_sm120_m_padded(cum_m: int, num_experts: int) -> int:
    if cum_m < 0 or num_experts <= 0:
        raise ValueError(
            "expected cum_m >= 0 and num_experts > 0, got "
            f"{cum_m=} {num_experts=}"
        )
    return ((cum_m + 3 * num_experts) // 4) * 4


@triton.jit
def _pack_flashinfer_sm120_fp8_scale_kernel(
    source_ptr,
    topk_ids_ptr,
    src2dst_ptr,
    m_indptr_ptr,
    output_ptr,
    source_stride_m,
    source_stride_k,
    output_stride_k,
    output_stride_m,
    num_routes,
    num_k_blocks,
    top_k: tl.constexpr,
    source_is_packed: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    routes = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    k_blocks = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    route_mask = routes < num_routes
    k_mask = k_blocks < num_k_blocks

    experts = tl.load(topk_ids_ptr + routes, mask=route_mask, other=0)
    # CUDA-graph padded rows carry topk_ids == -1: they own no scale column
    # and indexing m_indptr with them is out of bounds. Drop them here
    # explicitly (the previous behavior only worked by accident of layout).
    valid_mask = route_mask & (experts >= 0)
    dst_rows = tl.load(src2dst_ptr + routes, mask=route_mask, other=0)
    expert_starts = tl.load(
        m_indptr_ptr + experts, mask=valid_mask, other=0
    )
    aligned_starts = ((expert_starts + 3 * experts) // 4) * 4
    output_cols = aligned_starts + dst_rows - expert_starts

    if source_is_packed:
        source_rows = dst_rows
    else:
        source_rows = routes // top_k

    mask = valid_mask[:, None] & k_mask[None, :]
    source_offsets = (
        source_rows[:, None] * source_stride_m
        + k_blocks[None, :] * source_stride_k
    )
    values = tl.load(source_ptr + source_offsets, mask=mask, other=0.0)
    output_offsets = (
        k_blocks[None, :] * output_stride_k
        + output_cols[:, None] * output_stride_m
    )
    tl.store(output_ptr + output_offsets, values, mask=mask)


def pack_flashinfer_sm120_fp8_scale(
    source_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    source_is_packed: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if source_scale.dtype != torch.float32 or source_scale.ndim != 2:
        raise TypeError("source_scale must be a 2D float32 tensor")
    if not source_scale.is_contiguous():
        raise ValueError("source_scale must be contiguous")
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2:
        raise TypeError("topk_ids must be a 2D int32 tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.shape[1] == 0:
        raise ValueError("topk_ids must have top_k > 0")
    if (
        src2dst.dtype != torch.int32
        or src2dst.ndim != 1
        or src2dst.numel() != topk_ids.numel()
    ):
        raise TypeError(
            "src2dst must be 1D int32 with one entry per routed slot"
        )
    if not src2dst.is_contiguous():
        raise ValueError("src2dst must be contiguous")
    if (
        m_indptr.dtype != torch.int32
        or m_indptr.ndim != 1
        or m_indptr.numel() < 2
    ):
        raise TypeError(
            "m_indptr must be contiguous int32 with shape [num_experts + 1]"
        )
    if not m_indptr.is_contiguous():
        raise ValueError(
            "m_indptr must be contiguous int32 with shape [num_experts + 1]"
        )
    tensors = (source_scale, topk_ids, src2dst, m_indptr)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all scale-layout inputs must be CUDA tensors")
    if any(tensor.device != source_scale.device for tensor in tensors[1:]):
        raise ValueError("all scale-layout inputs must be on the same device")

    routes = topk_ids.numel()
    num_experts = m_indptr.numel() - 1
    num_k_blocks = source_scale.shape[1]
    expected_rows = routes if source_is_packed else topk_ids.shape[0]
    if source_scale.shape[0] != expected_rows:
        raise ValueError(
            f"source_scale rows must be {expected_rows}, "
            f"got {source_scale.shape[0]}"
        )

    expected_shape = (
        num_k_blocks,
        flashinfer_sm120_m_padded(routes, num_experts),
    )
    if out is None:
        out = torch.empty(
            expected_shape,
            device=source_scale.device,
            dtype=torch.float32,
        )
    if (
        out.shape != expected_shape
        or out.dtype != torch.float32
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous float32 with shape "
            f"{expected_shape}, got dtype={out.dtype} shape={tuple(out.shape)}"
        )
    if out.device != source_scale.device:
        raise ValueError("out and scale-layout inputs must be on the same device")
    if out.data_ptr() % 16 != 0:
        raise ValueError("FlashInfer A-scale output must be 16-byte aligned")

    out.zero_()
    if routes > 0 and num_k_blocks > 0:
        _pack_flashinfer_sm120_fp8_scale_kernel[
            (triton.cdiv(routes, 32), triton.cdiv(num_k_blocks, 16))
        ](
            source_scale,
            topk_ids,
            src2dst,
            m_indptr,
            out,
            source_scale.stride(0),
            source_scale.stride(1),
            out.stride(0),
            out.stride(1),
            routes,
            num_k_blocks,
            top_k=topk_ids.shape[1],
            source_is_packed=source_is_packed,
            BLOCK_ROUTES=32,
            BLOCK_K=16,
        )
    return out


def fused_quant_scatter_pack_flashinfer_sm120_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise TypeError("hidden_states must be a 2D bfloat16 tensor")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if hidden_states.shape[1] == 0 or hidden_states.shape[1] % 128 != 0:
        raise ValueError(
            "hidden_states last dimension must be positive and divisible by 128"
        )
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2:
        raise TypeError("topk_ids must be a 2D int32 tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if (
        topk_ids.shape[0] != hidden_states.shape[0]
        or topk_ids.shape[1] == 0
    ):
        raise ValueError(
            "topk_ids must have one non-empty routing row per token"
        )

    routes = topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.ndim != 1:
        raise TypeError("src2dst must be a 1D int32 tensor")
    if src2dst.numel() != routes:
        raise ValueError("src2dst must contain one entry per routed slot")
    if not src2dst.is_contiguous():
        raise ValueError("src2dst must be contiguous")
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise TypeError("m_indptr must be a 1D int32 tensor")
    if m_indptr.numel() < 2:
        raise ValueError("m_indptr must have shape [num_experts + 1]")
    if not m_indptr.is_contiguous():
        raise ValueError("m_indptr must be contiguous")

    tensors = (hidden_states, topk_ids, src2dst, m_indptr)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all fused A1 inputs must be CUDA tensors")
    if any(tensor.device != hidden_states.device for tensor in tensors[1:]):
        raise ValueError("all fused A1 inputs must be on the same device")

    num_experts = m_indptr.numel() - 1
    hidden_dim = hidden_states.shape[1]
    output_shape = (routes, hidden_dim)
    scale_shape = (
        hidden_dim // 128,
        flashinfer_sm120_m_padded(routes, num_experts),
    )

    if out is None:
        out = torch.empty(
            output_shape,
            device=hidden_states.device,
            dtype=torch.float8_e4m3fn,
        )
    if (
        out.shape != output_shape
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous float8_e4m3fn with shape "
            f"{output_shape}, got dtype={out.dtype} shape={tuple(out.shape)}"
        )

    if out_scale is None:
        out_scale = torch.empty(
            scale_shape,
            device=hidden_states.device,
            dtype=torch.float32,
        )
    if (
        out_scale.shape != scale_shape
        or out_scale.dtype != torch.float32
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            "out_scale must be contiguous float32 with shape "
            f"{scale_shape}, got dtype={out_scale.dtype} "
            f"shape={tuple(out_scale.shape)}"
        )
    if (
        out.device != hidden_states.device
        or out_scale.device != hidden_states.device
    ):
        raise ValueError("outputs and fused A1 inputs must be on the same device")
    if out_scale.data_ptr() % 16 != 0:
        raise ValueError("FlashInfer A-scale output must be 16-byte aligned")

    flashinfer_sm120_fp8_quant_scatter_pack(
        hidden_states,
        out,
        out_scale,
        topk_ids,
        src2dst,
        m_indptr,
    )
    return out, out_scale


def fused_swiglu_quant_pack_flashinfer_sm120_fp8(
    gate_up: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise TypeError("gate_up must be a 2D bfloat16 tensor")
    if not gate_up.is_contiguous():
        raise ValueError("gate_up must be contiguous")
    if gate_up.shape[1] == 0 or gate_up.shape[1] % 256 != 0:
        raise ValueError(
            "gate_up last dimension must be positive and twice a multiple of 128"
        )
    if topk_ids.dtype != torch.int32 or topk_ids.ndim != 2:
        raise TypeError("topk_ids must be a 2D int32 tensor")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.shape[1] == 0:
        raise ValueError("topk_ids must have top_k > 0")

    routes = topk_ids.numel()
    if gate_up.shape[0] != routes:
        raise ValueError(
            f"gate_up rows must equal topk_ids.numel() ({routes}), "
            f"got {gate_up.shape[0]}"
        )
    if (
        src2dst.dtype != torch.int32
        or src2dst.ndim != 1
        or src2dst.numel() != routes
    ):
        raise TypeError(
            "src2dst must be 1D int32 with one entry per routed slot"
        )
    if not src2dst.is_contiguous():
        raise ValueError("src2dst must be contiguous")
    if (
        m_indptr.dtype != torch.int32
        or m_indptr.ndim != 1
        or m_indptr.numel() < 2
    ):
        raise TypeError(
            "m_indptr must be contiguous int32 with shape [num_experts + 1]"
        )
    if not m_indptr.is_contiguous():
        raise ValueError(
            "m_indptr must be contiguous int32 with shape [num_experts + 1]"
        )

    tensors = (gate_up, topk_ids, src2dst, m_indptr)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all fused SwiGLU quant-pack inputs must be CUDA tensors")
    if any(tensor.device != gate_up.device for tensor in tensors[1:]):
        raise ValueError(
            "all fused SwiGLU quant-pack inputs must be on the same device"
        )

    hidden_dim = gate_up.shape[1] // 2
    num_experts = m_indptr.numel() - 1
    output_shape = (routes, hidden_dim)
    scale_shape = (
        hidden_dim // 128,
        flashinfer_sm120_m_padded(routes, num_experts),
    )

    if out is None:
        out = torch.empty(
            output_shape,
            device=gate_up.device,
            dtype=torch.float8_e4m3fn,
        )
    if (
        out.shape != output_shape
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous float8_e4m3fn with shape "
            f"{output_shape}, got dtype={out.dtype} shape={tuple(out.shape)}"
        )
    if out.device != gate_up.device:
        raise ValueError("out and fused quant-pack inputs must be on the same device")

    if out_scale is None:
        out_scale = torch.empty(
            scale_shape,
            device=gate_up.device,
            dtype=torch.float32,
        )
    if (
        out_scale.shape != scale_shape
        or out_scale.dtype != torch.float32
        or not out_scale.is_contiguous()
    ):
        raise ValueError(
            "out_scale must be contiguous float32 with shape "
            f"{scale_shape}, got dtype={out_scale.dtype} "
            f"shape={tuple(out_scale.shape)}"
        )
    if out_scale.device != gate_up.device:
        raise ValueError(
            "out_scale and fused quant-pack inputs must be on the same device"
        )
    if out_scale.data_ptr() % 16 != 0:
        raise ValueError("FlashInfer A-scale output must be 16-byte aligned")

    flashinfer_sm120_fp8_silu_quant_pack(
        gate_up,
        out,
        out_scale,
        topk_ids,
        src2dst,
        m_indptr,
    )
    return out, out_scale
