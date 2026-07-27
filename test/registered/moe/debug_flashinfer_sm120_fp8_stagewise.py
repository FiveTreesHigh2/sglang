"""Stage-wise diagnosis for test_full_runner_correctness failures.

Replicates the flashinfer_sm120_fp8 runner pipeline step by step and compares
each stage against a PyTorch dequant reference built from the SAME inputs, so
a single run pinpoints which stage diverges:

  gemm1    FlashInfer GEMM1 vs dequant matmul (isolates GEMM1 + a1 scale pack)
  silu_q   fused SwiGLU+quant kernel vs fp32 silu reference
  gemm2    FlashInfer GEMM2 vs dequant matmul (isolates GEMM2 + a2 scale pack)
  combine  moe_unpermute vs manual weighted sum
  triton   Triton runner vs PyTorch full-pipeline reference (checks the
           reference side itself)

Run on the SM120 server:
  <venv_python> test/registered/moe/debug_flashinfer_sm120_fp8_stagewise.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flashinfer_sm120_fp8_moe as t  # noqa: E402

from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise  # noqa: E402

from sglang.kernels.ops.moe.ep_moe_kernels import (  # noqa: E402
    moe_permute,
    moe_unpermute,
)
from sglang.kernels.ops.moe.flashinfer_sm120_fp8 import (  # noqa: E402
    pack_flashinfer_sm120_fp8_scale,
)
from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
    sglang_per_token_group_quant_fp8,
)


def _expand_block_scale(scale: torch.Tensor) -> torch.Tensor:
    # [*, n_blocks, k_blocks] -> [*, n, k] fp32
    return scale.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)


def _grouped_dequant_matmul(a_fp8, a_scale_rows, w_fp8, w_scale, m_indptr):
    """a_fp8 [M, k] + per-row scale [M, k/128]; w [E, n, k] + [E, n/128, k/128]."""
    deq_a = a_fp8.float() * a_scale_rows.repeat_interleave(128, dim=1)
    deq_w = w_fp8.float() * _expand_block_scale(w_scale.float())
    out = torch.zeros(
        a_fp8.shape[0], w_fp8.shape[1], device=a_fp8.device, dtype=torch.float32
    )
    bounds = m_indptr.cpu().tolist()
    for expert in range(w_fp8.shape[0]):
        start, end = bounds[expert], bounds[expert + 1]
        if start < end:
            out[start:end] = deq_a[start:end] @ deq_w[expert].t()
    return out


def diagnose(tokens, top_k, topk_ids):
    dispatch, config, quant_info, w13_scale, w2_scale = t._make_runner_case(
        tokens, top_k, topk_ids
    )
    x = dispatch.hidden_states
    topk_weights = dispatch.topk_output.topk_weights
    experts = quant_info.w13_weight.shape[0]
    intermediate = quant_info.w2_weight.shape[2]
    routes = topk_ids.numel()

    # ---- stage 1: quant + permute + a1 pack + GEMM1 ----
    q_hidden, q_scale = sglang_per_token_group_quant_fp8(x, 128)
    packed_hidden, src2dst, m_indptr = moe_permute(q_hidden, topk_ids, experts)
    a1_scale_fi = pack_flashinfer_sm120_fp8_scale(
        q_scale, topk_ids, src2dst, m_indptr, source_is_packed=False
    )
    gate_up = moe_gemm_fp8_nt_groupwise(
        packed_hidden,
        quant_info.w13_weight,
        a1_scale_fi,
        quant_info.w13_weight_scale_fi,
        m_indptr,
        out_dtype=torch.bfloat16,
    )
    route_idx = torch.arange(routes, device=x.device)
    packed_q_scale = torch.empty(
        routes, q_scale.shape[1], device=x.device, dtype=torch.float32
    )
    packed_q_scale[src2dst.long()] = q_scale[route_idx // top_k]
    ref_gate_up = _grouped_dequant_matmul(
        packed_hidden, packed_q_scale, quant_info.w13_weight, w13_scale, m_indptr
    )
    diff_gemm1 = t._calc_diff(gate_up, ref_gate_up)

    # ---- stage 2: fused SwiGLU + quant ----
    down_input, down_scale = sglang_per_token_group_quant_fp8(
        gate_up, 128, fuse_silu_and_mul=True
    )
    ref_silu = torch.nn.functional.silu(
        gate_up[:, :intermediate].float()
    ) * gate_up[:, intermediate:].float()
    deq_down_input = down_input.float() * down_scale.repeat_interleave(128, dim=1)
    diff_silu_q = t._calc_diff(deq_down_input, ref_silu)

    # ---- stage 3: a2 pack + GEMM2 ----
    a2_scale_fi = pack_flashinfer_sm120_fp8_scale(
        down_scale, topk_ids, src2dst, m_indptr, source_is_packed=True
    )
    down_output = moe_gemm_fp8_nt_groupwise(
        down_input,
        quant_info.w2_weight,
        a2_scale_fi,
        quant_info.w2_weight_scale_fi,
        m_indptr,
        out_dtype=torch.bfloat16,
    )
    ref_down = _grouped_dequant_matmul(
        down_input, down_scale, quant_info.w2_weight, w2_scale, m_indptr
    )
    diff_gemm2 = t._calc_diff(down_output, ref_down)

    # ---- stage 4: combine ----
    combined = moe_unpermute(
        down_output, src2dst, topk_ids, topk_weights, routed_scaling_factor=1.0
    )
    gathered = down_output.float()[src2dst.long()].view(tokens, top_k, -1)
    ref_combined = (gathered * topk_weights.unsqueeze(-1)).sum(dim=1)
    diff_combine = t._calc_diff(combined, ref_combined)

    # ---- reference side: Triton runner vs PyTorch full pipeline ----
    expected = t._run_triton_reference(
        dispatch, config, quant_info, w13_scale, w2_scale
    )
    ref_gathered = ref_down[src2dst.long()].view(tokens, top_k, -1)
    ref_final = (ref_gathered * topk_weights.unsqueeze(-1)).sum(dim=1)
    diff_triton_vs_pyref = t._calc_diff(expected, ref_final)
    diff_runner_vs_triton = t._calc_diff(combined, expected)

    print(
        f"tokens={tokens:<5} top_k={top_k}  "
        f"gemm1={diff_gemm1:.3e}  silu_q={diff_silu_q:.3e}  "
        f"gemm2={diff_gemm2:.3e}  combine={diff_combine:.3e}  "
        f"triton_vs_pyref={diff_triton_vs_pyref:.3e}  "
        f"runner_vs_triton={diff_runner_vs_triton:.3e}"
    )


def main():
    torch.manual_seed(11)
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
        diagnose(tokens, top_k, topk_ids)


if __name__ == "__main__":
    main()
