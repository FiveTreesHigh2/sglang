#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::pack_fp8;

struct FlashInferSm120Fp8QuantScatterParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ topk_ids;
  const int32_t* __restrict__ src2dst;
  const int32_t* __restrict__ m_indptr;
  int64_t hidden_dim;
  int64_t m_padded;
  uint32_t num_tokens;
  uint32_t top_k;
  uint32_t num_experts;
};

template <bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void flashinfer_sm120_fp8_quant_scatter_kernel(
    const FlashInferSm120Fp8QuantScatterParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == kGroupSize, "Invalid tiling");

  const uint32_t num_groups = params.hidden_dim / kGroupSize;
  PDLWaitPrimary<kUsePDL>();

  if (blockIdx.x < params.num_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t work_id = threadIdx.x / kWorkThreads;
    const uint32_t lane_in_work = threadIdx.x % kWorkThreads;
    const bool valid_group = work_id < num_groups;
    const uint32_t vector_id = work_id * kWorkThreads + lane_in_work;

    OutputVec quantized;
    float scale = 0.0f;
    if (valid_group) {
      InputVec values;
      values.load(params.input + static_cast<int64_t>(token) * params.hidden_dim, vector_id);

      float local_absmax = 1e-10f;
      float converted[8];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const auto [x, y] = cast<fp32x2_t>(values[i]);
        converted[2 * i] = x;
        converted[2 * i + 1] = y;
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(x), fabsf(y)));
      }

      constexpr uint32_t kWorkMask = (1u << kWorkThreads) - 1u;
      const uint32_t work_mask =
          kWorkMask << ((threadIdx.x % device::kWarpThreads) / kWorkThreads * kWorkThreads);
      local_absmax = warp::reduce_max<kWorkThreads>(local_absmax, work_mask);

      constexpr float kMaxInv = 1.0f / math::FP8_E4M3_MAX;
      scale = local_absmax * kMaxInv;
      // The legacy v2 quant kernel is built with --use_fast_math. Keep this
      // module precise globally, but reproduce that division at FP8 rounding
      // boundaries without changing the existing A2 expf implementation.
      const float quant_multiplier = __fdividef(math::FP8_E4M3_MAX, local_absmax);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        quantized[i] = pack_fp8(
            converted[2 * i] * quant_multiplier, converted[2 * i + 1] * quant_multiplier);
      }
    }

    PDLTriggerSecondary<kUsePDL>();
    if (valid_group) {
      for (uint32_t choice = 0; choice < params.top_k; ++choice) {
        const uint32_t route = token * params.top_k + choice;
        const uint32_t dst = params.src2dst[route];
        const uint32_t expert = params.topk_ids[route];
        const uint32_t start = params.m_indptr[expert];
        const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
        const uint32_t column = aligned + dst - start;

        quantized.store(
            params.output + static_cast<int64_t>(dst) * params.hidden_dim, vector_id);
        if (lane_in_work == 0) {
          params.output_scale[static_cast<int64_t>(work_id) * params.m_padded + column] = scale;
        }
      }
    }
    return;
  }

  const uint32_t expert = blockIdx.x - params.num_tokens;
  const uint32_t start = params.m_indptr[expert];
  const uint32_t end = params.m_indptr[expert + 1];
  const uint32_t aligned = ((start + 3u * expert) / 4u) * 4u;
  const uint32_t valid_end = aligned + end - start;
  const uint32_t next =
      expert + 1 == params.num_experts
      ? params.m_padded
      : ((end + 3u * (expert + 1)) / 4u) * 4u;
  const uint32_t gap = next - valid_end;

  PDLTriggerSecondary<kUsePDL>();
  if (gap != 0) {
    for (uint32_t i = threadIdx.x; i < num_groups * gap; i += blockDim.x) {
      const uint32_t group = i / gap;
      const uint32_t column = valid_end + i % gap;
      params.output_scale[static_cast<int64_t>(group) * params.m_padded + column] = 0.0f;
    }
  }
}

}  // namespace

template <bool kUsePDL>
struct FlashInferSm120Fp8QuantScatterKernel {
  static constexpr auto kernel = flashinfer_sm120_fp8_quant_scatter_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView src2dst,
      const tvm::ffi::TensorView m_indptr) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto T = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"hidden_dim"};
    auto M = SymbolicSize{"num_routes"};
    auto G = SymbolicSize{"num_groups"};
    auto MP = SymbolicSize{"m_padded"};
    auto K = SymbolicSize{"top_k"};
    auto EP = SymbolicSize{"num_experts_plus_one"};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, N}).with_dtype<bf16_t>().with_device(device).verify(hidden);
    TensorMatcher({M, N}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({G, MP}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(src2dst);
    TensorMatcher({EP}).with_dtype<int32_t>().with_device(device).verify(m_indptr);

    RuntimeCheck(T.unwrap() * K.unwrap() == M.unwrap(), "topk_ids must contain one entry per routed row");
    RuntimeCheck(K.unwrap() > 0, "top_k must be positive");
    RuntimeCheck(
        N.unwrap() > 0 && N.unwrap() % 128 == 0, "hidden_dim must be positive and divisible by 128");
    RuntimeCheck(EP.unwrap() >= 2, "m_indptr must have shape [num_experts + 1]");

    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto num_routes = static_cast<uint32_t>(M.unwrap());
    const auto top_k = static_cast<uint32_t>(K.unwrap());
    const auto hidden_dim = N.unwrap();
    const auto num_groups = static_cast<uint32_t>(hidden_dim / 128);
    const auto num_experts = static_cast<uint32_t>(EP.unwrap() - 1);
    const auto expected_m_padded =
        ((static_cast<int64_t>(num_routes) + 3 * static_cast<int64_t>(num_experts)) / 4) * 4;
    RuntimeCheck(G.unwrap() == num_groups, "invalid number of scale groups");
    RuntimeCheck(MP.unwrap() == expected_m_padded, "invalid FlashInfer m_padded dimension");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output_scale.data_ptr()) % 16 == 0,
        "FlashInfer output_scale must be 16-byte aligned");

    const auto num_threads = ((num_groups * 16u + 31u) / 32u) * 32u;
    RuntimeCheck(num_threads > 0 && num_threads <= 1024, "hidden_dim exceeds single-CTA kernel capacity");
    const auto grid = num_tokens + num_experts;
    const auto params = FlashInferSm120Fp8QuantScatterParams{
        .input = static_cast<const bf16_t*>(hidden.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .src2dst = static_cast<const int32_t*>(src2dst.data_ptr()),
        .m_indptr = static_cast<const int32_t*>(m_indptr.data_ptr()),
        .hidden_dim = hidden_dim,
        .m_padded = expected_m_padded,
        .num_tokens = num_tokens,
        .top_k = top_k,
        .num_experts = num_experts,
    };

    LaunchKernel(grid, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
