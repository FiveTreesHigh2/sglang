import unittest

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
