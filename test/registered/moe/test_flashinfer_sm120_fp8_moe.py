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
_FULL_MEAN_ABS_REL_TOL = 5e-3
_FULL_SYMMETRIC_DIFF_TOL = 1e-4
_FULL_NORMALIZED_RMSE_TOL = 1e-2


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
    experts, hidden, intermediate = 16, 256, 256
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

    def _assert_fused_a1_matches_legacy(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        experts: int,
    ):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            flashinfer_sm120_m_padded,
            fused_quant_scatter_pack_flashinfer_sm120_fp8,
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        ref_q, ref_scale = sglang_per_token_group_quant_fp8(
            hidden_states,
            128,
        )
        ref_packed, src2dst, m_indptr = moe_permute(
            ref_q,
            topk_ids,
            experts,
        )
        ref_scale_fi = pack_flashinfer_sm120_fp8_scale(
            ref_scale,
            topk_ids,
            src2dst,
            m_indptr,
            source_is_packed=False,
        )

        actual_q, actual_scale = (
            fused_quant_scatter_pack_flashinfer_sm120_fp8(
                hidden_states,
                topk_ids,
                src2dst,
                m_indptr,
            )
        )
        torch.testing.assert_close(
            actual_q.view(torch.uint8),
            ref_packed.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_scale,
            ref_scale_fi,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            actual_q.shape,
            (topk_ids.numel(), hidden_states.shape[1]),
        )
        self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
        self.assertTrue(actual_q.is_contiguous())
        self.assertEqual(
            actual_scale.shape,
            (
                hidden_states.shape[1] // 128,
                flashinfer_sm120_m_padded(topk_ids.numel(), experts),
            ),
        )
        self.assertEqual(actual_scale.dtype, torch.float32)
        self.assertTrue(actual_scale.is_contiguous())
        self.assertEqual(actual_scale.data_ptr() % 16, 0)

    def test_fused_a1_matches_actual_legacy_v2_reference(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization import fp8_kernel

        torch.manual_seed(101)
        tokens, hidden, experts = 8, 512, 8
        topk_ids = torch.tensor(
            [
                [3, 3],
                [0, 7],
                [1, 1],
                [6, 2],
                [7, 7],
                [4, 0],
                [5, 6],
                [2, 3],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        hidden_states = torch.randn(
            (tokens, hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )

        with patch.object(
            fp8_kernel,
            "sgl_per_token_group_quant_8bit_jit_v2",
            wraps=fp8_kernel.sgl_per_token_group_quant_8bit_jit_v2,
        ) as v2:
            ref_q, ref_scale = fp8_kernel.sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
            )
        self.assertEqual(v2.call_count, 1)

        ref_packed, src2dst, m_indptr = moe_permute(
            ref_q,
            topk_ids,
            experts,
        )
        ref_scale_fi = pack_flashinfer_sm120_fp8_scale(
            ref_scale,
            topk_ids,
            src2dst,
            m_indptr,
            source_is_packed=False,
        )

        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_quant_scatter_pack_flashinfer_sm120_fp8,
        )

        actual_q, actual_scale = (
            fused_quant_scatter_pack_flashinfer_sm120_fp8(
                hidden_states,
                topk_ids,
                src2dst,
                m_indptr,
            )
        )
        torch.testing.assert_close(
            actual_q.view(torch.uint8),
            ref_packed.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_scale,
            ref_scale_fi,
            rtol=0,
            atol=0,
        )

    def test_fused_a1_route_profiles(self):
        torch.manual_seed(102)
        experts = 8
        profiles = (
            torch.tensor(
                [[0], [1], [2], [3]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [
                    [0, 0, 0, 1, 1, 2, 3, 7],
                    [7, 7, 6, 5, 4, 3, 2, 1],
                ],
                device="cuda",
                dtype=torch.int32,
            ),
        )

        for topk_ids in profiles:
            hidden_states = torch.randn(
                (topk_ids.shape[0], 256),
                device="cuda",
                dtype=torch.bfloat16,
            )
            with self.subTest(top_k=topk_ids.shape[1]):
                self._assert_fused_a1_matches_legacy(
                    hidden_states,
                    topk_ids,
                    experts,
                )

    def test_fused_a1_reused_outputs_clear_dynamic_padding(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            flashinfer_sm120_m_padded,
            fused_quant_scatter_pack_flashinfer_sm120_fp8,
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(103)
        tokens, hidden, experts, top_k = 4, 256, 8, 2
        routes = (
            torch.tensor(
                [[0, 0], [0, 1], [1, 1], [1, 1]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [[7, 7], [6, 7], [5, 6], [4, 7]],
                device="cuda",
                dtype=torch.int32,
            ),
        )
        out = torch.empty(
            (tokens * top_k, hidden),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        out_scale = torch.empty(
            (
                hidden // 128,
                flashinfer_sm120_m_padded(tokens * top_k, experts),
            ),
            device="cuda",
            dtype=torch.float32,
        )
        out_ptr = out.data_ptr()
        scale_ptr = out_scale.data_ptr()

        for topk_ids in routes:
            with self.subTest(topk_ids=topk_ids.cpu().tolist()):
                hidden_states = torch.randn(
                    (tokens, hidden),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                ref_q, ref_scale = sglang_per_token_group_quant_fp8(
                    hidden_states,
                    128,
                )
                ref_packed, src2dst, m_indptr = moe_permute(
                    ref_q,
                    topk_ids,
                    experts,
                )
                expected_scale = pack_flashinfer_sm120_fp8_scale(
                    ref_scale,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=False,
                )

                out.view(torch.uint8).fill_(0x7F)
                out_scale.fill_(float("nan"))
                returned_q, returned_scale = (
                    fused_quant_scatter_pack_flashinfer_sm120_fp8(
                        hidden_states,
                        topk_ids,
                        src2dst,
                        m_indptr,
                        out=out,
                        out_scale=out_scale,
                    )
                )
                self.assertEqual(returned_q.data_ptr(), out_ptr)
                self.assertEqual(returned_scale.data_ptr(), scale_ptr)
                torch.testing.assert_close(
                    returned_q.view(torch.uint8),
                    ref_packed.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    returned_scale,
                    expected_scale,
                    rtol=0,
                    atol=0,
                )
                padding = expected_scale == 0
                self.assertTrue(bool(padding.any()))
                self.assertTrue(bool((returned_scale[padding] == 0).all()))

    def test_fused_a1_zero_padding_gap_is_safe(self):
        torch.manual_seed(104)
        hidden_states = torch.randn(
            (4, 256),
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_ids = torch.zeros(
            (4, 1),
            device="cuda",
            dtype=torch.int32,
        )
        self._assert_fused_a1_matches_legacy(
            hidden_states,
            topk_ids,
            experts=1,
        )
        torch.cuda.synchronize()

    def test_fused_a1_adapter_contract(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_quant_scatter_pack_flashinfer_sm120_fp8,
        )

        hidden = torch.randn(
            (4, 256),
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            device="cuda",
            dtype=torch.int32,
        )
        _, src2dst, m_indptr = moe_permute(hidden, topk, 8)
        routes = topk.numel()
        scale_shape = (2, ((routes + 3 * 8) // 4) * 4)
        valid_out = torch.empty(
            (routes, 256),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        valid_scale = torch.empty(
            scale_shape,
            device="cuda",
            dtype=torch.float32,
        )
        misaligned_storage = torch.empty(
            (valid_scale.numel() + 1,),
            device="cuda",
            dtype=torch.float32,
        )
        misaligned_scale = misaligned_storage[1:].view(scale_shape)

        hidden_noncontiguous = torch.randn(
            (4, 512),
            device="cuda",
            dtype=torch.bfloat16,
        )[:, ::2]
        topk_noncontiguous = torch.tensor(
            [[0, 2, 4, 6], [1, 3, 5, 7]],
            device="cuda",
            dtype=torch.int32,
        ).T
        src2dst_storage = torch.empty(
            (routes * 2,),
            device="cuda",
            dtype=torch.int32,
        )
        src2dst_noncontiguous = src2dst_storage[::2]
        src2dst_noncontiguous.copy_(src2dst)
        m_indptr_storage = torch.empty(
            (m_indptr.numel() * 2,),
            device="cuda",
            dtype=torch.int32,
        )
        m_indptr_noncontiguous = m_indptr_storage[::2]
        m_indptr_noncontiguous.copy_(m_indptr)
        out_noncontiguous = torch.empty(
            (routes, 512),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )[:, ::2]
        scale_noncontiguous = torch.empty(
            (scale_shape[0], scale_shape[1] * 2),
            device="cuda",
            dtype=torch.float32,
        )[:, ::2]

        cases = (
            ("hidden_states", {"hidden_states": hidden.float()}),
            (
                "hidden_states",
                {"hidden_states": hidden[:, :129].contiguous()},
            ),
            ("contiguous", {"hidden_states": hidden_noncontiguous}),
            ("topk_ids", {"topk_ids": topk.to(torch.int64)}),
            ("contiguous", {"topk_ids": topk_noncontiguous}),
            ("src2dst", {"src2dst": src2dst[:-1]}),
            ("contiguous", {"src2dst": src2dst_noncontiguous}),
            ("m_indptr", {"m_indptr": m_indptr.to(torch.int64)}),
            ("contiguous", {"m_indptr": m_indptr_noncontiguous}),
            ("CUDA", {"topk_ids": topk.cpu()}),
            ("out", {"out": valid_out[:, :128]}),
            (
                "out",
                {
                    "out": torch.empty_like(
                        valid_out,
                        dtype=torch.uint8,
                    )
                },
            ),
            ("contiguous", {"out": out_noncontiguous}),
            ("out_scale", {"out_scale": valid_scale[:, :-1]}),
            (
                "out_scale",
                {"out_scale": valid_scale.to(torch.float64)},
            ),
            ("contiguous", {"out_scale": scale_noncontiguous}),
            ("aligned", {"out_scale": misaligned_scale}),
        )
        defaults = {
            "hidden_states": hidden,
            "topk_ids": topk,
            "src2dst": src2dst,
            "m_indptr": m_indptr,
            "out": valid_out,
            "out_scale": valid_scale,
        }
        for message, override in cases:
            arguments = defaults | override
            with self.subTest(message=message), self.assertRaisesRegex(
                (TypeError, ValueError),
                message,
            ):
                fused_quant_scatter_pack_flashinfer_sm120_fp8(**arguments)

    def test_fused_swiglu_quant_pack_matches_legacy_reference(self):
        from sglang.jit_kernel.activation import silu_and_mul
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_swiglu_quant_pack_flashinfer_sm120_fp8,
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(17)
        tokens, top_k, experts, hidden = 8, 2, 8, 512
        topk_ids = torch.tensor(
            [
                [3, 0],
                [1, 1],
                [7, 2],
                [0, 6],
                [5, 3],
                [4, 0],
                [6, 6],
                [2, 7],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        _, src2dst, m_indptr = moe_permute(
            torch.zeros(
                (tokens, 128),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            ),
            topk_ids,
            experts,
        )
        self.assertFalse(
            torch.equal(
                src2dst,
                torch.arange(
                    tokens * top_k,
                    device="cuda",
                    dtype=torch.int32,
                ),
            )
        )

        gate_up = torch.randn(
            (tokens * top_k, hidden * 2),
            device="cuda",
            dtype=torch.bfloat16,
        )
        ref_bf16 = torch.empty(
            (tokens * top_k, hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        silu_and_mul(gate_up, out=ref_bf16)
        ref_q, ref_scale = sglang_per_token_group_quant_fp8(
            ref_bf16,
            128,
        )
        expected_scale = pack_flashinfer_sm120_fp8_scale(
            ref_scale,
            topk_ids,
            src2dst,
            m_indptr,
            source_is_packed=True,
        )

        actual_q, actual_scale = (
            fused_swiglu_quant_pack_flashinfer_sm120_fp8(
                gate_up,
                topk_ids,
                src2dst,
                m_indptr,
            )
        )
        torch.testing.assert_close(
            actual_q.view(torch.uint8),
            ref_q.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_scale,
            expected_scale,
            rtol=1e-6,
            atol=0,
        )

    def test_fused_swiglu_quant_pack_reused_outputs_clear_padding(self):
        from sglang.jit_kernel.activation import silu_and_mul
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_swiglu_quant_pack_flashinfer_sm120_fp8,
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(23)
        experts, tokens, top_k, hidden = 8, 4, 2, 512
        routes = (
            torch.tensor(
                [[0, 0], [0, 1], [1, 1], [1, 1]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [[7, 7], [6, 7], [5, 6], [4, 7]],
                device="cuda",
                dtype=torch.int32,
            ),
        )
        actual_q = torch.empty(
            (tokens * top_k, hidden),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        actual_scale = torch.full(
            (
                hidden // 128,
                ((tokens * top_k + 3 * experts) // 4) * 4,
            ),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
        q_ptr = actual_q.data_ptr()
        scale_ptr = actual_scale.data_ptr()

        for topk_ids in routes:
            with self.subTest(topk_ids=topk_ids.cpu().tolist()):
                _, src2dst, m_indptr = moe_permute(
                    torch.zeros(
                        (tokens, 128),
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    ),
                    topk_ids,
                    experts,
                )
                gate_up = torch.randn(
                    (tokens * top_k, hidden * 2),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                ref_bf16 = torch.empty(
                    (tokens * top_k, hidden),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                silu_and_mul(gate_up, out=ref_bf16)
                ref_q, ref_scale = sglang_per_token_group_quant_fp8(
                    ref_bf16,
                    128,
                )
                expected_scale = pack_flashinfer_sm120_fp8_scale(
                    ref_scale,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=True,
                )

                returned_q, returned_scale = (
                    fused_swiglu_quant_pack_flashinfer_sm120_fp8(
                        gate_up,
                        topk_ids,
                        src2dst,
                        m_indptr,
                        out=actual_q,
                        out_scale=actual_scale,
                    )
                )
                self.assertEqual(returned_q.data_ptr(), q_ptr)
                self.assertEqual(returned_scale.data_ptr(), scale_ptr)
                torch.testing.assert_close(
                    returned_q.view(torch.uint8),
                    ref_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    returned_scale,
                    expected_scale,
                    rtol=1e-6,
                    atol=0,
                )
                padding_mask = expected_scale == 0
                self.assertTrue(bool(padding_mask.any()))
                self.assertTrue(
                    torch.equal(
                        returned_scale[padding_mask],
                        torch.zeros_like(returned_scale[padding_mask]),
                    )
                )

    def test_fused_swiglu_quant_pack_route_profiles(self):
        from sglang.jit_kernel.activation import silu_and_mul
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_swiglu_quant_pack_flashinfer_sm120_fp8,
            pack_flashinfer_sm120_fp8_scale,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(29)
        experts, hidden = 8, 512
        profiles = (
            torch.tensor(
                [[0], [0], [7], [3]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [[3, 3], [0, 7], [3, 0], [7, 7]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [7, 7, 6, 5, 4, 3, 2, 0],
                ],
                device="cuda",
                dtype=torch.int32,
            ),
        )

        for topk_ids in profiles:
            tokens, top_k = topk_ids.shape
            with self.subTest(top_k=top_k):
                _, src2dst, m_indptr = moe_permute(
                    torch.zeros(
                        (tokens, 128),
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    ),
                    topk_ids,
                    experts,
                )
                gate_up = torch.randn(
                    (tokens * top_k, hidden * 2),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                ref_bf16 = torch.empty(
                    (tokens * top_k, hidden),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                silu_and_mul(gate_up, out=ref_bf16)
                ref_q, ref_scale = sglang_per_token_group_quant_fp8(
                    ref_bf16,
                    128,
                )
                expected_scale = pack_flashinfer_sm120_fp8_scale(
                    ref_scale,
                    topk_ids,
                    src2dst,
                    m_indptr,
                    source_is_packed=True,
                )

                actual_q, actual_scale = (
                    fused_swiglu_quant_pack_flashinfer_sm120_fp8(
                        gate_up,
                        topk_ids,
                        src2dst,
                        m_indptr,
                    )
                )
                torch.testing.assert_close(
                    actual_q.view(torch.uint8),
                    ref_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_scale,
                    expected_scale,
                    rtol=1e-6,
                    atol=0,
                )
                self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
                self.assertEqual(actual_q.shape, (tokens * top_k, hidden))
                self.assertTrue(actual_q.is_contiguous())
                self.assertEqual(actual_scale.dtype, torch.float32)
                self.assertEqual(
                    actual_scale.shape,
                    (
                        hidden // 128,
                        ((tokens * top_k + 3 * experts) // 4) * 4,
                    ),
                )
                self.assertTrue(actual_scale.is_contiguous())
                self.assertEqual(actual_scale.data_ptr() % 16, 0)

    def test_full_runner_uses_single_fused_a2_prepare(self):
        from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (
            fused_swiglu_quant_pack_flashinfer_sm120_fp8 as fused_adapter,
        )
        from sglang.srt.layers.moe.moe_runner import (
            flashinfer_sm120_fp8 as flashinfer_runner,
        )

        legacy_silu = getattr(flashinfer_runner, "silu_and_mul", None)
        topk_ids = (
            torch.arange(16, device="cuda", dtype=torch.int32)
            .remainder(16)
            .view(8, 2)
        )
        dispatch, config, quant_info, _, _ = _make_runner_case(
            8,
            2,
            topk_ids,
        )

        with patch.object(
            flashinfer_runner,
            "fused_swiglu_quant_pack_flashinfer_sm120_fp8",
            wraps=fused_adapter,
            create=True,
        ) as fused, patch.object(
            flashinfer_runner,
            "sglang_per_token_group_quant_fp8",
            wraps=flashinfer_runner.sglang_per_token_group_quant_fp8,
        ) as quant, patch.object(
            flashinfer_runner,
            "pack_flashinfer_sm120_fp8_scale",
            wraps=flashinfer_runner.pack_flashinfer_sm120_fp8_scale,
        ) as pack_scale, patch.object(
            flashinfer_runner,
            "silu_and_mul",
            wraps=legacy_silu,
            create=True,
        ) as silu:
            flashinfer_runner.fused_experts_none_to_flashinfer_sm120_fp8(
                dispatch,
                quant_info,
                config,
            )

        self.assertEqual(fused.call_count, 1)
        self.assertEqual(quant.call_count, 1)
        self.assertEqual(pack_scale.call_count, 1)
        self.assertEqual(silu.call_count, 0)

    def test_full_runner_correctness(self):
        correctness_failures = []
        cases = [
            (1, 1, torch.tensor([[0]], device="cuda", dtype=torch.int32)),
            (
                8,
                8,
                torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 8],
                        [0, 1, 2, 3, 4, 5, 7, 9],
                        [0, 1, 2, 3, 4, 6, 8, 10],
                        [0, 1, 2, 3, 5, 7, 9, 11],
                        [0, 1, 2, 4, 6, 8, 10, 12],
                        [0, 1, 3, 5, 7, 9, 11, 13],
                        [0, 2, 4, 6, 8, 10, 12, 14],
                    ],
                    device="cuda",
                    dtype=torch.int32,
                ),
            ),
            (
                128,
                2,
                torch.arange(256, device="cuda", dtype=torch.int32)
                .remainder(16)
                .view(128, 2),
            ),
            (
                1024,
                8,
                torch.cat(
                    (
                        torch.arange(4, device="cuda", dtype=torch.int32)
                        .view(1, 4)
                        .expand(1024, 4),
                        4
                        + (
                            torch.arange(
                                1024, device="cuda", dtype=torch.int32
                            ).view(1024, 1)
                            % 3
                        )
                        * 4
                        + torch.arange(
                            4, device="cuda", dtype=torch.int32
                        ).view(1, 4),
                    ),
                    dim=1,
                ),
            ),
        ]
        for tokens, top_k, topk_ids in cases:
            with self.subTest(tokens=tokens, top_k=top_k):
                sorted_ids = topk_ids.sort(dim=1).values
                self.assertTrue(
                    bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all())
                )
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
                if (
                    full_diff >= _FULL_MEAN_ABS_REL_TOL
                    or symmetric_diff >= _FULL_SYMMETRIC_DIFF_TOL
                    or normalized_rmse >= _FULL_NORMALIZED_RMSE_TOL
                ):
                    correctness_failures.append(
                        f"tokens={tokens} top_k={top_k} "
                        f"mean_abs_relative={full_diff:.6e} "
                        f"symmetric_diff={symmetric_diff:.6e} "
                        f"normalized_rmse={normalized_rmse:.6e}"
                    )
        self.assertFalse(
            correctness_failures,
            "\n".join(correctness_failures),
        )

    def test_cuda_graph_replays_new_hidden_and_routing(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_sm120_fp8 import (
            fused_experts_none_to_flashinfer_sm120_fp8,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        tokens, top_k = 8, 8
        route_a = (
            torch.arange(8, device="cuda", dtype=torch.int32)
            .view(1, 8)
            .expand(tokens, 8)
            .clone()
        )
        route_b = (
            torch.arange(8, 16, device="cuda", dtype=torch.int32)
            .view(1, 8)
            .expand(tokens, 8)
            .clone()
        )
        dispatch, config, quant_info, _, _ = _make_runner_case(
            tokens,
            top_k,
            route_a.clone(),
        )
        static_x = dispatch.hidden_states
        static_ids = dispatch.topk_output.topk_ids
        static_weights = dispatch.topk_output.topk_weights
        static_dispatch = StandardDispatchOutput(
            static_x,
            None,
            StandardTopKOutput(
                static_weights,
                static_ids,
                dispatch.topk_output.router_logits,
            ),
        )

        # Warm every JIT module and the shared FlashInfer .so before capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                fused_experts_none_to_flashinfer_sm120_fp8(
                    static_dispatch,
                    quant_info,
                    config,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused_experts_none_to_flashinfer_sm120_fp8(
                static_dispatch,
                quant_info,
                config,
            ).hidden_states
        captured_output_ptr = graph_output.data_ptr()

        inputs = (
            (torch.randn_like(static_x) / 8, route_a),
            (torch.randn_like(static_x) / 8, route_b),
        )
        for new_x, new_ids in inputs:
            static_x.copy_(new_x)
            static_ids.copy_(new_ids)
            static_weights.fill_(1.0 / top_k)
            graph.replay()
            torch.cuda.synchronize()
            replayed = graph_output.clone()

            eager_dispatch = StandardDispatchOutput(
                new_x,
                None,
                StandardTopKOutput(
                    static_weights.clone(),
                    new_ids,
                    torch.empty(0, device="cuda"),
                ),
            )
            eager = fused_experts_none_to_flashinfer_sm120_fp8(
                eager_dispatch,
                quant_info,
                config,
            ).hidden_states
            torch.cuda.synchronize()

            self.assertEqual(graph_output.data_ptr(), captured_output_ptr)
            self.assertTrue(bool(torch.isfinite(replayed).all()))
            torch.testing.assert_close(replayed, eager, rtol=0, atol=0)

        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()
        self.assertEqual(allocated_after, allocated_before)
