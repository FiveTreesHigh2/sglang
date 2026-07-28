#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::pack_fp8;

struct FlashInferSm120Fp8SiluQuantPackParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ topk_ids;
  const int32_t* __restrict__ src2dst;
  const int32_t* __restrict__ m_indptr;
  int64_t hidden_dim;
  int64_t m_padded;
  uint32_t num_routes;
  uint32_t num_experts;
};

SGL_DEVICE fp32x2_t flashinfer_sm120_fp8_silu_and_mul(bf16x2_t gate, bf16x2_t up) {
  using namespace device;
  const auto [g0, g1] = cast<fp32x2_t>(gate);
  const auto [u0, u1] = cast<fp32x2_t>(up);
  const auto silu0 = g0 / (1.0f + expf(-g0));
  const auto silu1 = g1 / (1.0f + expf(-g1));

  // Match the legacy production path exactly: the standalone activation
  // kernel stores SwiGLU to BF16 before the generic FP8 quant kernel reads it.
  const auto rounded = cast<bf16x2_t>(fp32x2_t{silu0 * u0, silu1 * u1});
  return cast<fp32x2_t>(rounded);
}

template <bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void flashinfer_sm120_fp8_silu_quant_pack_kernel(
    const FlashInferSm120Fp8SiluQuantPackParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == kGroupSize, "Invalid tiling");

  const uint32_t num_groups = params.hidden_dim / kGroupSize;
  PDLWaitPrimary<kUsePDL>();

  if (blockIdx.x < params.num_routes) {
    const uint32_t route = blockIdx.x;
    const uint32_t expert = params.topk_ids[route];
    if (expert >= params.num_experts) {
      // CUDA-graph padded rows carry topk_ids == -1; indexing m_indptr with
      // the wrapped value is a wild read and the derived scale column a wild
      // store (same defect class as the fused-A1 kernel). Skip the route
      // entirely; trigger PDL so the dependent launch is not held back.
      PDLTriggerSecondary<kUsePDL>();
      return;
    }
    const uint32_t dst = params.src2dst[route];
    const uint32_t expert_start = params.m_indptr[expert];
    const uint32_t aligned_start = ((expert_start + 3u * expert) / 4u) * 4u;
    const uint32_t scale_col = aligned_start + dst - expert_start;
    const auto input = params.input + static_cast<int64_t>(dst) * params.hidden_dim * 2;
    const auto output = params.output + static_cast<int64_t>(dst) * params.hidden_dim;

    const uint32_t work_id = threadIdx.x / kWorkThreads;
    const uint32_t lane_in_work = threadIdx.x % kWorkThreads;
    const bool valid_group = work_id < num_groups;
    const uint32_t vector_id = work_id * kWorkThreads + lane_in_work;
    const uint32_t vectors_per_half = params.hidden_dim / 8u;

    OutputVec out_vec;
    float scale = 0.0f;
    if (valid_group) {
      InputVec gate_vec, up_vec;
      gate_vec.load(input, vector_id);
      up_vec.load(input, vector_id + vectors_per_half);

      float local_max = 1e-10f;
      float results[8];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const auto [x, y] = flashinfer_sm120_fp8_silu_and_mul(gate_vec[i], up_vec[i]);
        results[2 * i] = x;
        results[2 * i + 1] = y;
        local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
      }

      constexpr uint32_t kWorkMask = (1u << kWorkThreads) - 1u;
      const uint32_t work_mask = kWorkMask << ((threadIdx.x % device::kWarpThreads) / kWorkThreads * kWorkThreads);
      local_max = warp::reduce_max<kWorkThreads>(local_max, work_mask);
      constexpr float kMaxInv = 1.0f / math::FP8_E4M3_MAX;
      scale = local_max * kMaxInv;
      // Generic FP8 quant is compiled with --use_fast_math. Keep activation
      // expf precise, but reproduce its fast division at FP8 rounding edges.
      const float quant_multiplier = __fdividef(math::FP8_E4M3_MAX, local_max);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        out_vec[i] = pack_fp8(results[2 * i] * quant_multiplier, results[2 * i + 1] * quant_multiplier);
      }
    }

    PDLTriggerSecondary<kUsePDL>();
    if (valid_group) {
      out_vec.store(output, vector_id);
      if (lane_in_work == 0) {
        params.output_scale[static_cast<int64_t>(work_id) * params.m_padded + scale_col] = scale;
      }
    }
    return;
  }

  const uint32_t expert = blockIdx.x - params.num_routes;
  const uint32_t start = params.m_indptr[expert];
  const uint32_t end = params.m_indptr[expert + 1];
  const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
  const uint32_t valid_end = aligned + end - start;
  const uint32_t next = expert + 1 == params.num_experts ? params.m_padded : ((end + 3u * (expert + 1)) / 4u) * 4u;
  const uint32_t gap = next - valid_end;

  PDLTriggerSecondary<kUsePDL>();
  if (gap != 0) {
    for (uint32_t i = threadIdx.x; i < num_groups * gap; i += blockDim.x) {
      const uint32_t group = i / gap;
      const uint32_t column = valid_end + i % gap;
      params.output_scale[static_cast<int64_t>(group) * params.m_padded + column] = 0.0f;
    }
  }
  // Zero the prefix columns [0, aligned-start-of-expert-0): padded (-1)
  // routes own packed rows below m_indptr[0] but no scale column, and this
  // buffer is allocated with torch.empty.
  if (expert == 0) {
    const uint32_t prefix = (start / 4u) * 4u;
    if (prefix != 0) {
      for (uint32_t i = threadIdx.x; i < num_groups * prefix; i += blockDim.x) {
        const uint32_t group = i / prefix;
        const uint32_t column = i % prefix;
        params.output_scale[static_cast<int64_t>(group) * params.m_padded + column] = 0.0f;
      }
    }
  }
}

}  // namespace

template <bool kUsePDL>
struct FlashInferSm120Fp8SiluQuantPackKernel {
  static constexpr auto kernel = flashinfer_sm120_fp8_silu_quant_pack_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView gate_up,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView src2dst,
      const tvm::ffi::TensorView m_indptr) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_routes"};
    auto D = SymbolicSize{"gate_up_dim"};
    auto N = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    auto MP = SymbolicSize{"m_padded"};
    auto T = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"top_k"};
    auto EP = SymbolicSize{"num_experts_plus_one"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D}).with_dtype<bf16_t>().with_device(device).verify(gate_up);
    TensorMatcher({M, N}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({G, MP}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(src2dst);
    TensorMatcher({EP}).with_dtype<int32_t>().with_device(device).verify(m_indptr);

    RuntimeCheck(D.unwrap() == 2 * N.unwrap(), "gate_up last dim must be 2 * output last dim");
    RuntimeCheck(N.unwrap() > 0 && N.unwrap() % 128 == 0, "hidden_dim must be positive and divisible by 128");
    RuntimeCheck(T.unwrap() * K.unwrap() == M.unwrap(), "topk_ids must contain one entry per routed row");
    RuntimeCheck(K.unwrap() > 0, "top_k must be positive");
    RuntimeCheck(EP.unwrap() >= 2, "m_indptr must have shape [num_experts + 1]");

    const auto num_routes = static_cast<uint32_t>(M.unwrap());
    const auto hidden_dim = N.unwrap();
    const auto num_groups = static_cast<uint32_t>(hidden_dim / 128);
    const auto num_experts = static_cast<uint32_t>(EP.unwrap() - 1);
    const auto expected_m_padded = ((static_cast<int64_t>(num_routes) + 3 * static_cast<int64_t>(num_experts)) / 4) * 4;
    RuntimeCheck(G.unwrap() == num_groups, "invalid number of scale groups");
    RuntimeCheck(MP.unwrap() == expected_m_padded, "invalid FlashInfer m_padded dimension");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output_scale.data_ptr()) % 16 == 0,
        "FlashInfer output_scale must be 16-byte aligned");

    const auto num_threads = ((num_groups * 16u + 31u) / 32u) * 32u;
    RuntimeCheck(num_threads > 0 && num_threads <= 1024, "hidden_dim exceeds single-CTA kernel capacity");
    const auto grid = num_routes + num_experts;

    const auto params = FlashInferSm120Fp8SiluQuantPackParams{
        .input = static_cast<const bf16_t*>(gate_up.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .src2dst = static_cast<const int32_t*>(src2dst.data_ptr()),
        .m_indptr = static_cast<const int32_t*>(m_indptr.data_ptr()),
        .hidden_dim = hidden_dim,
        .m_padded = expected_m_padded,
        .num_routes = num_routes,
        .num_experts = num_experts,
    };

    LaunchKernel(grid, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
