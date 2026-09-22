// CPU compilation only: device BF16 A and cooperative signed-I8/BF16 B.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename Coefficient, ushort Groups, ushort K>
inline void i8_lut_sdk_probe(device bfloat *a, device const char *source,
                             device float *out, uint tid) {
  constexpr auto descriptor = matmul2d_descriptor(32, 64, K, false, true,
      false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<Groups>> op;
  auto right = op.template get_right_input_cooperative_tensor<bfloat, Coefficient, float>();
  using Left = tensor<device bfloat, dextents<int, 2>>;
  auto dot = op.template get_destination_cooperative_tensor<
      Left, remove_addrspace_t<decltype(right)>, float>();
  for (ushort i = 0; i < right.get_capacity(); ++i) {
    if (!right.is_valid_element(i)) continue;
    const auto index = right.get_multidimensional_index(i);
    right[i] = Coefficient(source[ulong(index[1]) * K + index[0]]);
  }
  for (ushort i = 0; i < dot.get_capacity(); ++i)
    if (dot.is_valid_element(i)) dot[i] = 0.0f;
  auto left = tensor(a, dextents<int, 2>{K, 32}, array<int, 2>{1, K});
  op.run(left, right, dot);
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    out[ulong(index[1]) * 64 + index[0]] = dot[i];
  }
  (void)tid;
}

#define PROBE(T, G, K) \
kernel void i8_lut_sdk_##T##_sg##G##_k##K(device bfloat *a [[buffer(0)]], \
    device const char *b [[buffer(1)]], device float *out [[buffer(2)]], \
    uint tid [[thread_index_in_threadgroup]]) { i8_lut_sdk_probe<T, G, K>(a,b,out,tid); }
PROBE(bfloat, 2, 128)
PROBE(bfloat, 4, 128)
PROBE(bfloat, 2, 256)
PROBE(bfloat, 4, 256)
PROBE(int8_t, 2, 128)
PROBE(int8_t, 4, 128)
PROBE(int8_t, 2, 256)
PROBE(int8_t, 4, 256)
