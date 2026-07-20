from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.activation import silu_and_mul
from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute, moe_unpermute
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
    pack_flashinfer_sm120_fp8_scale,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class FlashInferSm120Fp8MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale_fi: torch.Tensor
    w2_weight_scale_fi: torch.Tensor
    block_shape: tuple[int, int]


def prepare_flashinfer_sm120_fp8_weight_scales(
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("FlashInfer SM120 FP8 weight scales must be float32")
    if w13_scale.ndim != 3 or w2_scale.ndim != 3:
        raise ValueError("FlashInfer SM120 FP8 weight scales must be rank 3")
    return (
        w13_scale.transpose(1, 2).contiguous(),
        w2_scale.transpose(1, 2).contiguous(),
    )


@functools.lru_cache(maxsize=1)
def _target_grouped_gemm():
    try:
        from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise
    except ImportError as error:
        raise RuntimeError(
            "flashinfer_sm120_fp8 requires "
            "flashinfer.grouped_mm.moe_gemm_fp8_nt_groupwise"
        ) from error
    return moe_gemm_fp8_nt_groupwise


def _validate_contract(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferSm120Fp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    hidden_states = dispatch_output.hidden_states
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability not in {(12, 0), (12, 1)}:
        raise RuntimeError(
            "flashinfer_sm120_fp8 requires capability 12.0 or 12.1, "
            f"got {capability}"
        )
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise TypeError(
            "flashinfer_sm120_fp8 activation input must be 2D bfloat16"
        )
    if not hidden_states.is_contiguous():
        raise ValueError("flashinfer_sm120_fp8 activation input must be contiguous")
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError(
            "flashinfer_sm120_fp8 requires unquantized standard dispatch input"
        )
    if (
        quant_info.w13_weight.dtype != torch.float8_e4m3fn
        or quant_info.w2_weight.dtype != torch.float8_e4m3fn
    ):
        raise TypeError(
            "flashinfer_sm120_fp8 weights must be torch.float8_e4m3fn"
        )
    if (
        not quant_info.w13_weight.is_contiguous()
        or not quant_info.w2_weight.is_contiguous()
    ):
        raise ValueError(
            "flashinfer_sm120_fp8 weights must be contiguous [E, N, K]"
        )
    if tuple(quant_info.block_shape) != (128, 128):
        raise ValueError(
            "flashinfer_sm120_fp8 block_shape must be exactly (128, 128)"
        )

    w13 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("flashinfer_sm120_fp8 weights must be rank 3")
    if w13.shape[0] != w2.shape[0]:
        raise ValueError("w13 and w2 must contain the same number of experts")
    if w13.shape[2] != hidden_states.shape[1] or w2.shape[1] != w13.shape[2]:
        raise ValueError(
            "w13 input and w2 output dimensions must match hidden size"
        )
    if w13.shape[1] != 2 * w2.shape[2]:
        raise ValueError("w13 output must contain gate and up halves for w2")

    payload_tensors = (
        w13,
        w2,
        quant_info.w13_weight_scale_fi,
        quant_info.w2_weight_scale_fi,
    )
    if any(tensor.device != hidden_states.device for tensor in payload_tensors):
        raise ValueError(
            "flashinfer_sm120_fp8 weights and scales must match input device"
        )
    for name, weight, scale in (
        ("w13", w13, quant_info.w13_weight_scale_fi),
        ("w2", w2, quant_info.w2_weight_scale_fi),
    ):
        if weight.shape[1] % 128 or weight.shape[2] % 128:
            raise ValueError(f"{name} N/K dimensions must be divisible by 128")
        expected_scale = (
            weight.shape[0],
            weight.shape[2] // 128,
            weight.shape[1] // 128,
        )
        if scale.dtype != torch.float32 or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"{name} scale must be float32 with shape {expected_scale}, "
                f"got dtype={scale.dtype} shape={tuple(scale.shape)}"
            )
        if not scale.is_contiguous():
            raise ValueError(f"{name} scale must be contiguous")

    topk_output = dispatch_output.topk_output
    if topk_output.topk_ids.ndim != 2:
        raise ValueError("flashinfer_sm120_fp8 topk_ids must be rank 2")
    if topk_output.topk_weights.shape != topk_output.topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have identical shape")
    if topk_output.topk_weights.dtype != torch.float32:
        raise TypeError("flashinfer_sm120_fp8 topk_weights must be float32")
    if topk_output.topk_ids.device != hidden_states.device or (
        topk_output.topk_weights.device != hidden_states.device
    ):
        raise ValueError("routing tensors must match the activation device")

    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError(
            "flashinfer_sm120_fp8 supports gated SiLU/SwiGLU only"
        )
    if runner_config.apply_router_weight_on_input:
        raise ValueError("apply_router_weight_on_input is not supported")
    if runner_config.no_combine:
        raise ValueError("no_combine is not supported")
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
    ):
        raise ValueError("GPT-OSS alpha/limit is not supported")
    if runner_config.swiglu_limit is not None:
        raise ValueError("swiglu_limit is not supported")
    if runner_config.num_local_experts is not None and (
        runner_config.num_local_experts != w13.shape[0]
    ):
        raise ValueError(
            "runner num_local_experts must match the loaded weight tensors"
        )
    if runner_config.hidden_size is not None and (
        runner_config.hidden_size != hidden_states.shape[1]
    ):
        raise ValueError("runner hidden_size must match the activation input")
    if runner_config.intermediate_size_per_partition is not None and (
        runner_config.intermediate_size_per_partition != w2.shape[2]
    ):
        raise ValueError(
            "runner intermediate size must match the w2 input dimension"
        )
    if runner_config.top_k is not None and (
        runner_config.top_k != topk_output.topk_ids.shape[1]
    ):
        raise ValueError("runner top_k must match the routing output")


def _run_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
) -> None:
    result = _target_grouped_gemm()(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        backend="cute",
        out=out,
        out_dtype=torch.bfloat16,
    )
    if not isinstance(result, torch.Tensor) or result.data_ptr() != out.data_ptr():
        raise RuntimeError(
            "FlashInfer SM120 grouped GEMM did not reuse the supplied output"
        )


@register_fused_func("none", "flashinfer_sm120_fp8")
def fused_experts_none_to_flashinfer_sm120_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferSm120Fp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not isinstance(quant_info, FlashInferSm120Fp8MoeQuantInfo):
        raise TypeError(f"unexpected quant_info type: {type(quant_info)}")
    if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
        raise TypeError("flashinfer_sm120_fp8 requires StandardTopKOutput")
    _validate_contract(dispatch_output, quant_info, runner_config)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_output.topk_ids
    topk_weights = dispatch_output.topk_output.topk_weights
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    if topk_ids.numel() == 0:
        return StandardCombineInput(hidden_states=torch.empty_like(hidden_states))

    q_hidden, q_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
    packed_hidden, src2dst, m_indptr = moe_permute(
        q_hidden,
        topk_ids,
        quant_info.w13_weight.shape[0],
    )
    a1_scale_fi = pack_flashinfer_sm120_fp8_scale(
        q_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=False,
    )

    gate_up = torch.empty(
        packed_hidden.shape[0],
        quant_info.w13_weight.shape[1],
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    _run_grouped_gemm(
        packed_hidden,
        quant_info.w13_weight,
        a1_scale_fi,
        quant_info.w13_weight_scale_fi,
        m_indptr,
        gate_up,
    )

    # The v2 quant kernel only fuses SiLU+mul for column-major UE8M0 scales;
    # FlashInfer requires row-major float scales, so keep these as two ops.
    down_input_bf16 = torch.empty(
        packed_hidden.shape[0],
        quant_info.w2_weight.shape[2],
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    silu_and_mul(gate_up, out=down_input_bf16)
    down_input, down_scale = sglang_per_token_group_quant_fp8(
        down_input_bf16,
        128,
    )
    a2_scale_fi = pack_flashinfer_sm120_fp8_scale(
        down_scale,
        topk_ids,
        src2dst,
        m_indptr,
        source_is_packed=True,
    )
    down_output = torch.empty(
        packed_hidden.shape[0],
        quant_info.w2_weight.shape[1],
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    _run_grouped_gemm(
        down_input,
        quant_info.w2_weight,
        a2_scale_fi,
        quant_info.w2_weight_scale_fi,
        m_indptr,
        down_output,
    )

    output = moe_unpermute(
        down_output,
        src2dst,
        topk_ids,
        topk_weights,
        routed_scaling_factor=runner_config.routed_scaling_factor,
    )
    return StandardCombineInput(hidden_states=output)
