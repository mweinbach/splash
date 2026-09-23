#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
template <int M, int NT, int SGM>
kernel void lay(device int *out [[buffer(0)]], device bfloat *xa [[buffer(1)]], device uchar *wb [[buffer(2)]],
                uint tid [[thread_index_in_threadgroup]]) {
  auto a = tensor(xa, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(wb, dextents<int, 2>{64, NT}, array<int, 2>{1, 64});
  constexpr auto desc = matmul2d_descriptor(M, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SGM>> op;
  auto a0 = a.template slice<64, M>(0, 0);
  auto b0 = b.template slice<64, NT>(0, 0);
  auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  out[tid * 64] = t.get_capacity();
  for (ushort i = 0; i < t.get_capacity(); ++i) {
    auto idx = t.get_multidimensional_index(i);
    out[tid * 64 + 1 + 2 * i] = t.is_valid_element(i) ? idx[0] : -1;
    out[tid * 64 + 2 + 2 * i] = idx[1];
  }
}
template [[host_name("lay_16_32_1")]] kernel void lay<16, 32, 1>(device int *, device bfloat *, device uchar *, uint);
template [[host_name("lay_16_64_1")]] kernel void lay<16, 64, 1>(device int *, device bfloat *, device uchar *, uint);
template [[host_name("lay_16_16_1")]] kernel void lay<16, 16, 1>(device int *, device bfloat *, device uchar *, uint);
template [[host_name("lay_8_32_1")]] kernel void lay<8, 32, 1>(device int *, device bfloat *, device uchar *, uint);
