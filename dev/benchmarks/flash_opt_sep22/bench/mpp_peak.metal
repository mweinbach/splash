#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Peak matrix-unit throughput: repeated M x N x K products on staged tiles.
template <typename TA, typename TB, int M, int N, int K, int SG>
[[kernel]] void mpp_peak(device float *out [[buffer(0)]],
                         constant uint &iters [[buffer(1)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup TA sa[M * K];
  threadgroup TB sb[N * K];
  for (uint i = tid; i < M * K; i += SG * 32) sa[i] = TA(i % 7);
  for (uint i = tid; i < N * K; i += SG * 32) sb[i] = TB(i % 5);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = tensor(sa, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto b = tensor(sb, dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto d = matmul2d_descriptor(M, N, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto acc = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0;
  for (uint it = 0; it < iters; ++it) op.run(a, b, acc);
  float s = 0;
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) s += acc[i];
  if (s == 12345.0f) out[tg.x] = s;
}
#define P(TA, TB, M, N, K, SG, NAME) template [[host_name(NAME)]] [[kernel]] void mpp_peak<TA, TB, M, N, K, SG>(device float *, constant uint &, uint, uint3);
P(bfloat, bfloat, 64, 64, 64, 4, "peak_bf16_64x64x64_sg4")
P(bfloat, bfloat, 32, 64, 64, 4, "peak_bf16_32x64x64_sg4")
P(bfloat, bfloat, 16, 64, 64, 4, "peak_bf16_16x64x64_sg4")
P(half, half, 64, 64, 64, 4, "peak_f16_64x64x64_sg4")
P(bfloat, int8_t, 64, 64, 64, 4, "peak_bf16i8_64x64x64_sg4")
P(bfloat, int8_t, 16, 64, 64, 4, "peak_bf16i8_16x64x64_sg4")
