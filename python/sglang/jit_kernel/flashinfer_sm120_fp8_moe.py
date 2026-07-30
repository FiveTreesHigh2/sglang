from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_flashinfer_sm120_fp8_moe_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "flashinfer_sm120_fp8_moe",
        *args,
        cuda_files=[
            "moe/flashinfer_sm120_fp8_swiglu_quant.cuh",
            "moe/flashinfer_sm120_fp8_quant_scatter.cuh",
        ],
        cuda_wrappers=[
            (
                "silu_quant_pack",
                f"FlashInferSm120Fp8SiluQuantPackKernel<{args}, false>::run",
            ),
            (
                "quant_pack",
                f"FlashInferSm120Fp8SiluQuantPackKernel<{args}, true>::run",
            ),
            (
                "quant_scatter_pack",
                f"FlashInferSm120Fp8QuantScatterKernel<{args}>::run",
            ),
        ],
    )


def flashinfer_sm120_fp8_silu_quant_pack(
    gate_up: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
) -> None:
    module = _jit_flashinfer_sm120_fp8_moe_module(is_arch_support_pdl())
    module.silu_quant_pack(
        gate_up, output, output_scale, topk_ids, src2dst, m_indptr
    )


def flashinfer_sm120_fp8_quant_pack(
    activated: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
) -> None:
    module = _jit_flashinfer_sm120_fp8_moe_module(is_arch_support_pdl())
    module.quant_pack(
        activated, output, output_scale, topk_ids, src2dst, m_indptr
    )


def flashinfer_sm120_fp8_quant_scatter_pack(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    src2dst: torch.Tensor,
    m_indptr: torch.Tensor,
) -> None:
    module = _jit_flashinfer_sm120_fp8_moe_module(is_arch_support_pdl())
    module.quant_scatter_pack(
        hidden_states,
        output,
        output_scale,
        topk_ids,
        src2dst,
        m_indptr,
    )
