#include <metal_stdlib>
using namespace metal;
struct PrimitiveParams { uint count, slotWords, x, y, z, reserved; };
kernel void private_indirect_emit(device uint *arguments [[buffer(0)]],
    device const uint *control [[buffer(1)]], constant PrimitiveParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  arguments[p.slotWords] = control[0] ? p.x : 0;
  arguments[p.slotWords + 1] = control[0] ? p.y : 0;
  arguments[p.slotWords + 2] = control[0] ? p.z : 0;
}
kernel void private_indirect_copy(device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]], constant PrimitiveParams &p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < p.count) output[tid] = input[tid];
}
