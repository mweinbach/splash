#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct P { uint K, N, rows, x_stride, y_stride, row0; ulong w_row_stride, p_row_stride; };
// B = pre-dequantized half W[NP, K] in device memory.
template <int M, int NP, int SG>
kernel void tallh(const device bfloat *x [[buffer(0)]], const device half *w [[buffer(1)]],
                  device bfloat *y [[buffer(4)]], constant P &p [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]]) {
  const uint first = tg.x * M;
  if (first >= p.rows) return;
  const int valid = int(min(uint(M), p.rows - first));
  auto a = tensor(const_cast<device bfloat *>(x + ulong(first) * p.x_stride), dextents<int, 2>{int(p.K), valid}, array<int, 2>{1, int(p.x_stride)});
  auto b = tensor(const_cast<device half *>(w), dextents<int, 2>{int(p.K), NP}, array<int, 2>{1, int(p.K)});
  constexpr auto d = matmul2d_descriptor(M, NP, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto acc = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  op.run(a, b, acc);
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    if (int(idx[1]) < valid && uint(idx[0]) < p.N) y[(ulong(first) + idx[1]) * p.y_stride + idx[0]] = bfloat(acc[i]);
  }
}
#define T(M, NP, SG) template [[host_name("tallh_m" #M "_np" #NP "_sg" #SG)]] kernel void tallh<M, NP, SG>(const device bfloat *, const device half *, device bfloat *, constant P &, uint3);
T(16, 64, 1) T(16, 64, 2) T(32, 64, 2) T(32, 64, 4) T(64, 64, 4) T(16, 16, 1) T(32, 16, 1) T(32, 16, 2) T(64, 16, 4) T(16, 32, 1) T(32, 32, 2)
