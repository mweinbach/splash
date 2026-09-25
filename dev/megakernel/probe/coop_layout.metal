// Dumps the per-lane element coordinates of matmul2d cooperative tensors.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename TA, typename TB, int M, int N, int K>
[[kernel]] void coop_layout(device int *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr auto d = matmul2d_descriptor(M, N, K, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<d, execution_simdgroups<1>> op;
  auto r = op.template get_right_input_cooperative_tensor<TA, TB, float>();
  auto l = op.template get_left_input_cooperative_tensor<TA, TB, float>();
  // header: capacities
  auto ta = tensor(reinterpret_cast<device TA *>(out), dextents<int, 2>{K, M}, array<int, 2>{1, K});
  using HB = typename tensor<device TB, dextents<int, 2>, tensor_inline>::data_handle_type;
  tensor<device TB, dextents<int, 2>, tensor_inline> tb(reinterpret_cast<HB>(out), dextents<int, 2>{K, N}, array<int, 2>{1, K});
  auto sa = ta.template slice<K, M>(0, 0);
  auto sb = tb.template slice<K, N>(0, 0);
  auto dd = op.template get_destination_cooperative_tensor<decltype(sa), decltype(sb), float>();
  if (lane == 0) { out[0] = r.get_capacity(); out[1] = l.get_capacity(); out[2] = dd.get_capacity(); }
  device int *dq = out + 16 + 64 * 256 + lane * 256;
  for (ushort i = 0; i < dd.get_capacity(); ++i) {
    auto idx = dd.get_multidimensional_index(i);
    dq[2 * i] = dd.is_valid_element(i) ? idx[0] : -1;
    dq[2 * i + 1] = dd.is_valid_element(i) ? idx[1] : -1;
  }
  device int *ro = out + 16 + lane * 256;
  for (ushort i = 0; i < r.get_capacity(); ++i) {
    auto idx = r.get_multidimensional_index(i);
    ro[2 * i] = r.is_valid_element(i) ? idx[0] : -1;
    ro[2 * i + 1] = r.is_valid_element(i) ? idx[1] : -1;
  }
  device int *lo = out + 16 + 32 * 256 + lane * 256;
  for (ushort i = 0; i < l.get_capacity(); ++i) {
    auto idx = l.get_multidimensional_index(i);
    lo[2 * i] = l.is_valid_element(i) ? idx[0] : -1;
    lo[2 * i + 1] = l.is_valid_element(i) ? idx[1] : -1;
  }
}
template [[host_name("layout_bf16_u8_16x32x64")]] [[kernel]] void coop_layout<bfloat, uint8_t, 16, 32, 64>(device int *, uint);
template [[host_name("layout_bf16_bf16_16x32x64")]] [[kernel]] void coop_layout<bfloat, bfloat, 16, 32, 64>(device int *, uint);
template [[host_name("layout_bf16_half_16x32x64")]] [[kernel]] void coop_layout<bfloat, half, 16, 32, 64>(device int *, uint);
template [[host_name("layout_half_half_16x32x64")]] [[kernel]] void coop_layout<half, half, 16, 32, 64>(device int *, uint);
template [[host_name("layout_bf16_u8_16x32x128")]] [[kernel]] void coop_layout<bfloat, uint8_t, 16, 32, 128>(device int *, uint);
template [[host_name("layout_bf16_u8_16x64x64")]] [[kernel]] void coop_layout<bfloat, uint8_t, 16, 64, 64>(device int *, uint);
template [[host_name("layout_bf16_bf16_16x64x32")]] [[kernel]] void coop_layout<bfloat, bfloat, 16, 64, 32>(device int *, uint);
