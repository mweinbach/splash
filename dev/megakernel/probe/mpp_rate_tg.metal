// Matrix-unit rate with B staged in threadgroup memory and A read from a
// cached device tensor, for per-simdgroup vs cooperative tile shapes.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// SG simdgroups per threadgroup; each op spans COOP simdgroups; every op runs
// M x N x K with B slice (SG / COOP distinct slices of the stage).
template <int M, int N, int K, int SG, int COOP, bool ATG>
[[kernel]] void rate_tg(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
                        const device bfloat *asrc [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                        uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int OPS = SG / COOP;
  threadgroup bfloat sb[OPS * N * K];
  threadgroup bfloat sa[ATG ? OPS * M * K : 1];
  for (uint i = tid; i < uint(OPS * N * K); i += SG * 32) sb[i] = bfloat(float(i % 5));
  if (ATG) for (uint i = tid; i < uint(OPS * M * K); i += SG * 32) sa[i] = bfloat(float(i % 3));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint opi = simd / COOP;
  auto b = tensor(sb + opi * N * K, dextents<int, 2>{K, N}, array<int, 2>{1, K}).template slice<K, N>(0, 0);
  auto ad = tensor(const_cast<device bfloat *>(asrc), dextents<int, 2>{1024, 64}, array<int, 2>{1, 1024});
  auto at = tensor(sa + (ATG ? opi * M * K : 0), dextents<int, 2>{K, M}, array<int, 2>{1, K}).template slice<K, M>(0, 0);
  constexpr auto d = matmul2d_descriptor(M, N, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<COOP>> op;
  auto a0 = ad.template slice<K, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0;
  for (uint it = 0; it < iters; ++it) {
    if (ATG) {
      op.run(at, b, acc);
    } else {
      auto as = ad.template slice<K, M>(int((it * K) % 1024), int((opi * M) % 48));
      op.run(as, b, acc);
    }
  }
  float s = 0;
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) s += acc[i];
  if (s == 12345.0f) out[tg.x] = s;
}
#define R(M, N, K, SG, COOP, ATG) \
  template [[host_name("tg_" #M "x" #N "x" #K "_sg" #SG "_coop" #COOP "_a" #ATG)]] [[kernel]] void rate_tg<M, N, K, SG, COOP, ATG>( \
      device float *, constant uint &, const device bfloat *, uint, uint, uint3);
R(16, 64, 32, 8, 1, 0) R(16, 64, 32, 8, 1, 1) R(16, 64, 64, 8, 1, 0) R(16, 128, 32, 8, 1, 0) R(32, 64, 32, 8, 1, 0)
R(64, 128, 32, 8, 8, 0) R(64, 128, 32, 8, 8, 1) R(32, 128, 32, 8, 4, 0) R(32, 128, 32, 8, 4, 1) R(32, 64, 32, 8, 2, 0)
R(16, 64, 32, 4, 1, 0) R(64, 64, 32, 4, 4, 1)
