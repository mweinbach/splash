#include <metal_stdlib>
using namespace metal;

struct Sources1 { array<device const uint *, 1> p [[id(0)]]; };
struct Sources4 { array<device const uint *, 4> p [[id(0)]]; };
struct Sources21 { array<device const uint *, 21> p [[id(0)]]; };

// Exactly one original word per declared native base. No strides or model math.
kernel void idle_probe_read1(constant Sources1 &sources [[buffer(0)]],
                            device uint *output [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid < 1) output[1 + tid] = sources.p[tid][0];
}
kernel void idle_probe_read4(constant Sources4 &sources [[buffer(0)]],
                            device uint *output [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid < 4) output[1 + tid] = sources.p[tid][0];
}
kernel void idle_probe_read21(constant Sources21 &sources [[buffer(0)]],
                             device uint *output [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
  if (tid < 21) output[1 + tid] = sources.p[tid][0];
}
