// Matrix-unit rate for small-M products, one independent matmul per simdgroup
// (execution_simdgroups<1>), B from a cached device tensor (as the GEMV kernels read it).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename TB> struct PerByte { static constant constexpr int V = 1; };
template <> struct PerByte<uint4b_format> { static constant constexpr int V = 2; };
template <> struct PerByte<int4b_format> { static constant constexpr int V = 2; };
template <> struct PerByte<bfloat> { static constant constexpr int V = 0; };
template <> struct PerByte<half> { static constant constexpr int V = 0; };

template <typename TA, typename TB, int M, int N, int K>
[[kernel]] void mpp_rate(device float *out [[buffer(0)]],
                         constant uint &iters [[buffer(1)]],
                         const device uchar *bsrc [[buffer(2)]],
                         const device TA *asrc [[buffer(3)]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int KB = 1024;  // K extent of the source tensors
  auto a = tensor(const_cast<device TA *>(asrc), dextents<int, 2>{KB, M}, array<int, 2>{1, KB});
  constexpr int pb = PerByte<TB>::V;
  const int bstride = pb ? KB : KB;  // elements
  using H = typename tensor<device TB, dextents<int, 2>, tensor_inline>::data_handle_type;
  tensor<device TB, dextents<int, 2>, tensor_inline> b(
      reinterpret_cast<H>(const_cast<device uchar *>(bsrc)), dextents<int, 2>{KB, N}, array<int, 2>{1, bstride});
  constexpr auto d = matmul2d_descriptor(M, N, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<1>> op;
  auto a0 = a.template slice<K, M>(0, 0);
  auto b0 = b.template slice<K, N>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0;
  for (uint it = 0; it < iters; ++it) {
    const int k = int((it * K) % KB);
    auto as = a.template slice<K, M>(k, 0);
    auto bs = b.template slice<K, N>(k, 0);
    op.run(as, bs, acc);
  }
  float s = 0;
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) s += acc[i];
  if (s == 12345.0f) out[tg.x * 4 + simd] = s;
}
#define P(TA, TB, M, N, K, NAME) \
  template [[host_name("rate_" NAME "_" #M "x" #N "x" #K)]] [[kernel]] void mpp_rate<TA, TB, M, N, K>( \
      device float *, constant uint &, const device uchar *, const device TA *, uint, uint3);
#define SHAPES(TA, TB, NAME) \
  P(TA, TB, 8, 32, 64, NAME) P(TA, TB, 16, 32, 64, NAME) P(TA, TB, 16, 64, 64, NAME) P(TA, TB, 32, 32, 64, NAME) \
  P(TA, TB, 16, 32, 128, NAME) P(TA, TB, 32, 64, 64, NAME) P(TA, TB, 64, 64, 64, NAME) P(TA, TB, 8, 64, 64, NAME)
SHAPES(bfloat, bfloat, "bf16xbf16")
SHAPES(half, half, "f16xf16")
SHAPES(bfloat, int8_t, "bf16xi8")
SHAPES(bfloat, uint8_t, "bf16xu8")
SHAPES(bfloat, uint4b_format, "bf16xu4")
SHAPES(half, uint4b_format, "f16xu4")
SHAPES(half, int8_t, "f16xi8")
