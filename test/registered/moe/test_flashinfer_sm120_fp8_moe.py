import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() in {
    (12, 0),
    (12, 1),
}


def _layout_reference(source, topk_ids, src2dst, m_indptr, source_is_packed):
    routes = topk_ids.numel()
    experts = m_indptr.numel() - 1
    top_k = topk_ids.shape[1]
    m_padded = ((routes + 3 * experts) // 4) * 4
    result = torch.zeros(
        source.shape[1], m_padded, dtype=torch.float32, device=source.device
    )
    flat_ids = topk_ids.flatten()
    for route in range(routes):
        expert = int(flat_ids[route].item())
        dst = int(src2dst[route].item())
        expert_start = int(m_indptr[expert].item())
        aligned_start = ((expert_start + 3 * expert) // 4) * 4
        source_row = dst if source_is_packed else route // top_k
        result[:, aligned_start + dst - expert_start] = source[source_row]
    return result


def _make_runner_case(tokens, top_k, topk_ids, seed=11):
    from flashinfer.testing.utils import per_block_cast_to_fp8
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
        FlashInferSm120Fp8MoeQuantInfo,
        prepare_flashinfer_sm120_fp8_weight_scales,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    torch.manual_seed(seed)
    experts, hidden, intermediate = 8, 256, 256
    x = (
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        / 8
    )
    w13_bf16 = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ) / hidden**0.5
    w2_bf16 = torch.randn(
        experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
    ) / intermediate**0.5

    def quantize(weight):
        quantized_parts = []
        scale_parts = []
        for expert in range(weight.shape[0]):
            quantized, scale = per_block_cast_to_fp8(weight[expert])
            quantized_parts.append(quantized)
            scale_parts.append(scale)
        return (
            torch.stack(quantized_parts).contiguous(),
            torch.stack(scale_parts).contiguous(),
        )

    w13, w13_scale = quantize(w13_bf16)
    w2, w2_scale = quantize(w2_bf16)
    w13_scale_fi, w2_scale_fi = prepare_flashinfer_sm120_fp8_weight_scales(
        w13_scale,
        w2_scale,
    )
    topk_weights = torch.rand(
        tokens, top_k, device="cuda", dtype=torch.float32
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_output = StandardTopKOutput(
        topk_weights,
        topk_ids,
        torch.empty(0, device="cuda"),
    )
    dispatch = StandardDispatchOutput(x, None, topk_output)
    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        inplace=False,
        routed_scaling_factor=1.0,
    )
    quant_info = FlashInferSm120Fp8MoeQuantInfo(
        w13,
        w2,
        w13_scale_fi,
        w2_scale_fi,
        (128, 128),
    )
    return dispatch, config, quant_info, w13_scale, w2_scale


def _calc_diff(actual, expected):
    numerator = (actual.float() - expected.float()).abs().mean()
    denominator = expected.float().abs().mean().clamp_min(1e-12)
    return float((numerator / denominator).item())


def _calc_symmetric_diff(actual, expected):
    actual = actual.double()
    expected = expected.double()
    denominator = (
        (actual.square() + expected.square()).sum().clamp_min(1e-24)
    )
    return float((1.0 - 2.0 * (actual * expected).sum() / denominator).item())


def _calc_normalized_rmse(actual, expected):
    error_rms = (actual.float() - expected.float()).square().mean().sqrt()
    expected_rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error_rms / expected_rms).item())


def _grouped_fp32_reference(a, b, a_scale, b_scale, m_indptr):
    boundaries = m_indptr.cpu().tolist()
    num_experts = len(boundaries) - 1
    a_scale_rows = torch.empty(
        (a.shape[0], a_scale.shape[0]),
        device=a.device,
        dtype=torch.float32,
    )
    for expert, (start, end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        aligned_start = ((start + 3 * expert) // 4) * 4
        a_scale_rows[start:end] = a_scale[
            :, aligned_start : aligned_start + end - start
        ].T

    a_dequant = a.float() * a_scale_rows.repeat_interleave(128, dim=1)
    reference = torch.empty(
        (a.shape[0], b.shape[1]),
        device=a.device,
        dtype=torch.float32,
    )
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(num_experts):
            start, end = boundaries[expert : expert + 2]
            if start == end:
                continue
            scale = b_scale[expert].T
            b_dequant = b[expert].float() * scale.repeat_interleave(
                128, dim=0
            ).repeat_interleave(128, dim=1)
            reference[start:end] = a_dequant[start:end] @ b_dequant.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
    return reference


def _run_flashinfer_with_stage_diagnostics(dispatch, config, quant_info):
    from sglang.srt.layers.moe.moe_runner import (
        flashinfer_sm120_fp8 as flashinfer_runner,
    )

    diagnostics = []
    original_grouped_gemm = flashinfer_runner._run_grouped_gemm

    def recording_grouped_gemm(a, b, a_scale, b_scale, m_indptr, out):
        original_grouped_gemm(a, b, a_scale, b_scale, m_indptr, out)
        reference = _grouped_fp32_reference(
            a,
            b,
            a_scale,
            b_scale,
            m_indptr,
        )
        diagnostics.append(
            {
                "diff": _calc_diff(out, reference),
                "actual_abs_mean": float(out.float().abs().mean().item()),
                "reference_abs_mean": float(reference.abs().mean().item()),
            }
        )

    with patch.object(
        flashinfer_runner,
        "_run_grouped_gemm",
        recording_grouped_gemm,
    ):
        output = flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
            dispatch,
            quant_info,
            config,
        ).hidden_states
    return output, diagnostics


def _run_triton_reference(
    dispatch,
    config,
    quant_info,
    w13_scale,
    w2_scale,
):
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe as triton_fused_moe,
    )
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(
        enable_deterministic_inference=False
    ), patch.object(
        triton_fused_moe,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    ):
        return triton_fused_moe.fused_experts(
            dispatch.hidden_states.clone(),
            quant_info.w13_weight,
            quant_info.w2_weight,
            dispatch.topk_output,
            config,
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=[128, 128],
        )


@unittest.skipUnless(_IS_SM120, "SM120/SM121 required")
class TestFlashInferSm120Fp8Packing(unittest.TestCase):
    def test_existing_moe_permute_copies_fp8_bits(self):
        torch.manual_seed(7)
        tokens, hidden, experts, top_k = 8, 256, 8, 2
        q_hidden = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        topk_ids = torch.tensor(
            [
                [0, 7],
                [1, 1],
                [3, 0],
                [7, 2],
                [2, 4],
                [4, 0],
                [6, 6],
                [5, 3],
            ],
            device="cuda",
            dtype=torch.int32,
        )

        packed, src2dst, m_indptr = moe_permute(q_hidden, topk_ids, experts)
        expected = torch.empty_like(packed)
        src2dst_cpu = src2dst.cpu().tolist()
        for route in range(tokens * top_k):
            expected[src2dst_cpu[route]].copy_(q_hidden[route // top_k])

        torch.testing.assert_close(
            packed.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        self.assertEqual(m_indptr.dtype, torch.int32)
        self.assertEqual(m_indptr.shape, (experts + 1,))

    def test_layout_token_and_packed_source_rows(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            pack_flashinfer_sm120_fp8_scale,
        )

        topk_ids = torch.tensor(
            [[3, 0], [3, 3], [1, 0], [7, 3]],
            device="cuda",
            dtype=torch.int32,
        )
        _, src2dst, m_indptr = moe_permute(
            torch.zeros(
                (4, 128), device="cuda", dtype=torch.float8_e4m3fn
            ),
            topk_ids,
            8,
        )
        token_scale = torch.arange(
            8, device="cuda", dtype=torch.float32
        ).view(4, 2)
        packed_scale = torch.arange(
            16, device="cuda", dtype=torch.float32
        ).view(8, 2)

        actual_gemm1 = pack_flashinfer_sm120_fp8_scale(
            token_scale,
            topk_ids,
            src2dst,
            m_indptr,
            source_is_packed=False,
        )
        actual_gemm2 = pack_flashinfer_sm120_fp8_scale(
            packed_scale,
            topk_ids,
            src2dst,
            m_indptr,
            source_is_packed=True,
        )
        torch.testing.assert_close(
            actual_gemm1,
            _layout_reference(
                token_scale, topk_ids, src2dst, m_indptr, False
            ),
        )
        torch.testing.assert_close(
            actual_gemm2,
            _layout_reference(
                packed_scale, topk_ids, src2dst, m_indptr, True
            ),
        )
        self.assertEqual(actual_gemm1.data_ptr() % 16, 0)

    def test_layout_covers_topk_and_empty_expert_profiles(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            pack_flashinfer_sm120_fp8_scale,
        )

        experts, tokens = 8, 8
        for top_k in (1, 2, 8):
            with self.subTest(top_k=top_k):
                topk_ids = (
                    torch.arange(
                        tokens * top_k,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    .remainder(experts - 1)
                    .view(tokens, top_k)
                )
                _, src2dst, m_indptr = moe_permute(
                    torch.zeros(
                        tokens,
                        128,
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    ),
                    topk_ids,
                    experts,
                )
                source = torch.arange(
                    tokens * 2, device="cuda", dtype=torch.float32
                ).view(tokens, 2)
                actual = pack_flashinfer_sm120_fp8_scale(
                    source,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=False,
                )
                expected = _layout_reference(
                    source, topk_ids, src2dst, m_indptr, False
                )
                torch.testing.assert_close(actual, expected)

    def test_reused_out_clears_padding_after_route_change(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            pack_flashinfer_sm120_fp8_scale,
        )

        experts = 8
        route_a = torch.tensor(
            [[0, 0], [0, 1], [1, 1], [1, 1]],
            device="cuda",
            dtype=torch.int32,
        )
        route_b = torch.tensor(
            [[7, 7], [6, 7], [5, 6], [4, 7]],
            device="cuda",
            dtype=torch.int32,
        )
        source = torch.arange(
            8, device="cuda", dtype=torch.float32
        ).view(4, 2)
        out = None
        for topk_ids in (route_a, route_b):
            _, src2dst, m_indptr = moe_permute(
                torch.zeros(
                    (4, 128),
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                ),
                topk_ids,
                experts,
            )
            if out is None:
                out = pack_flashinfer_sm120_fp8_scale(
                    source,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=False,
                )
            else:
                pack_flashinfer_sm120_fp8_scale(
                    source,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=False,
                    out=out,
                )

        expected = _layout_reference(
            source, route_b, src2dst, m_indptr, False
        )
        torch.testing.assert_close(out, expected)

    def test_full_runner_correctness(self):
        strict_failures = []
        cases = [
            (1, 1, torch.tensor([[0]], device="cuda", dtype=torch.int32)),
            (
                8,
                8,
                torch.tensor(
                    [
                        [0, 0, 0, 1, 1, 2, 2, 7],
                        [0, 0, 1, 1, 1, 2, 6, 7],
                        [0, 1, 1, 2, 2, 2, 5, 7],
                        [0, 0, 0, 0, 3, 3, 4, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [7, 7, 7, 6, 6, 5, 5, 4],
                        [0, 0, 0, 0, 0, 0, 0, 7],
                        [1, 2, 3, 4, 5, 6, 7, 7],
                    ],
                    device="cuda",
                    dtype=torch.int32,
                ),
            ),
            (
                128,
                2,
                torch.arange(256, device="cuda", dtype=torch.int32)
                .remainder(8)
                .view(128, 2),
            ),
            (
                1024,
                8,
                torch.arange(8192, device="cuda", dtype=torch.int32)
                .square()
                .remainder(8)
                .view(1024, 8),
            ),
        ]
        for tokens, top_k, topk_ids in cases:
            with self.subTest(tokens=tokens, top_k=top_k):
                dispatch, config, quant_info, w13_scale, w2_scale = (
                    _make_runner_case(tokens, top_k, topk_ids)
                )
                actual, stage_diagnostics = (
                    _run_flashinfer_with_stage_diagnostics(
                        dispatch,
                        config,
                        quant_info,
                    )
                )
                expected = _run_triton_reference(
                    dispatch,
                    config,
                    quant_info,
                    w13_scale,
                    w2_scale,
                )
                full_diff = _calc_diff(actual, expected)
                symmetric_diff = _calc_symmetric_diff(actual, expected)
                normalized_rmse = _calc_normalized_rmse(actual, expected)
                print(
                    "[diagnostic] "
                    f"tokens={tokens} top_k={top_k} "
                    f"gemm1_direct={stage_diagnostics[0]} "
                    f"gemm2_direct={stage_diagnostics[1]} "
                    f"full_vs_triton={full_diff:.6e} "
                    f"symmetric_diff={symmetric_diff:.6e} "
                    f"normalized_rmse={normalized_rmse:.6e} "
                    f"actual_abs_mean={actual.float().abs().mean().item():.6e} "
                    f"triton_abs_mean={expected.float().abs().mean().item():.6e}"
                )
                self.assertTrue(bool(torch.isfinite(actual).all()))
                if full_diff >= 1e-3:
                    strict_failures.append(
                        f"tokens={tokens} top_k={top_k} "
                        f"mean_abs_relative={full_diff:.6e} "
                        f"symmetric_diff={symmetric_diff:.6e} "
                        f"normalized_rmse={normalized_rmse:.6e}"
                    )
        self.assertFalse(strict_failures, "\n".join(strict_failures))
