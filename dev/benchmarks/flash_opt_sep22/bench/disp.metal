#include <metal_stdlib>
using namespace metal;
kernel void tiny(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) { x[tid] = x[tid] * 1.0001f + 1.0f; }
