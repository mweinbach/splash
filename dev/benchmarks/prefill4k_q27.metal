// Private model-shape screen. Original Q4 kernels are compiled from production
// bytes here. No work or memory is assigned to either GPU die.
#include "metal/kernels/prefill/linear_q4.metal"
#include "prefill4k_q27_params.h"
#include "prefill4k_q27_q4_direct.h"
#include "prefill4k_q27_q4_small.h"
#include "prefill4k_q27_q4_pair.h"

#define Q27_PAIR_Q4_ENTRY(Name,M,N,SG) \
[[max_total_threads_per_threadgroup(SG*32)]] kernel void Name( \
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]], \
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    constant Q4PrefillParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]]) { \
  input += ulong(group.x) * M * p.input_size; \
  output += ulong(group.x) * M * p.output_size; \
  sums += ulong(group.x) * M * (p.input_size / 64); \
  prefill4k_q27_pair_tile<M,N,SG>(input,weights,scales,biases,output,sums, \
      p.output_size,p.input_size,group.y * N); \
}
Q27_PAIR_Q4_ENTRY(prefill4k_q27_q4_m32n64_s4_pair,32,64,4)
Q27_PAIR_Q4_ENTRY(prefill4k_q27_q4_m16n64_s2_pair,16,64,2)
#undef Q27_PAIR_Q4_ENTRY

kernel void prefill4k_q27_q4_sums16(device const bfloat *input [[buffer(0)]],
    device float *sums [[buffer(1)]], constant Q4PrefillParams &p [[buffer(2)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint M = 16;
  const uint groups = p.input_size / 64;
  input += ulong(tile) * M * p.input_size;
  sums += ulong(tile) * M * groups;
  for (uint g = 0; g < groups; ++g) for (uint row = simd; row < M; row += 8) {
    const uint origin = row * p.input_size + g * 64 + lane;
    const float value = simd_sum(float(input[origin]) + float(input[origin + 32]));
    if (!lane) sums[g * M + row] = value;
  }
}

#define Q27_SMALL_Q4_ENTRY(Name,M,N,SG) \
[[max_total_threads_per_threadgroup(SG*32)]] kernel void Name( \
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]], \
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    constant Q4PrefillParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
  input += ulong(group.x) * M * p.input_size; \
  output += ulong(group.x) * M * p.output_size; \
  sums += ulong(group.x) * M * (p.input_size / 64); \
  prefill4k_q27_small_q4_tile<M,N,SG,false,false>(input,weights,scales,biases,output,output, \
      p.output_size,p.input_size,sums,group.y * N,lane,simd); \
}
Q27_SMALL_Q4_ENTRY(prefill4k_q27_q4_m32n64_s2_direct,32,64,2)
Q27_SMALL_Q4_ENTRY(prefill4k_q27_q4_m32n64_s4_direct,32,64,4)
Q27_SMALL_Q4_ENTRY(prefill4k_q27_q4_m16n64_s2_direct,16,64,2)
Q27_SMALL_Q4_ENTRY(prefill4k_q27_q4_m16n64_s4_direct,16,64,4)
#undef Q27_SMALL_Q4_ENTRY

kernel void prefill4k_q27_q4_m32n128_s8_direct(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort M = 32, N = 128;
  input += ulong(group.x) * M * p.input_size;
  output += ulong(group.x) * M * p.output_size;
  sums += ulong(group.x) * M * (p.input_size / 64);
  prefill4k_q27_direct_q4_tile<M,N,8,false,false>(input,weights,scales,biases,output,output,
      p.output_size,p.input_size,sums,group.y * N,lane,simd);
}

kernel void prefill4k_q27_q4_sums64(device const bfloat *input [[buffer(0)]],
    device float *sums [[buffer(1)]], constant Q4PrefillParams &p [[buffer(2)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint M = 64;
  const uint groups = p.input_size / 64;
  input += ulong(tile) * M * p.input_size;
  sums += ulong(tile) * M * groups;
  for (uint g = 0; g < groups; ++g) for (uint row = simd; row < M; row += 8) {
    const uint origin = row * p.input_size + g * 64 + lane;
    const float value = simd_sum(float(input[origin]) + float(input[origin + 32]));
    if (!lane) sums[g * M + row] = value;
  }
}

kernel void prefill4k_q27_q4_m64n128_s4(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort M = 64, N = 128;
  input += ulong(group.x) * M * p.input_size;
  output += ulong(group.x) * M * p.output_size;
  sums += ulong(group.x) * M * (p.input_size / 64);
  q4_mpp_prefill_tile<M,N,4,false,false>(input,weights,scales,biases,output,output,
      p.output_size,p.input_size,sums,group.y * N,lane,simd);
}

#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline void prefill4k_q27_error(device atomic_uint *diag, uint flag) {
  atomic_fetch_or_explicit(diag, flag, memory_order_relaxed);
}

template<typename W, ushort M, ushort N, ushort SG>
inline void prefill4k_q27_cache_tile(device bfloat *input,device W *weights,
    device bfloat *output,device atomic_uint *diag,constant Prefill4KQ27Params &p,
    uint3 group,uint3 threads,uint tid) {
  if (!p.rows || p.rows > 8192 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 64 || !p.output_size || p.output_size > 32768 ||
      p.output_size % N || p.tile_rows != M || p.tile_outputs != N ||
      p.traversal > 2 || p.reserved0 || p.reserved1 || group.z ||
      threads.x != SG * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) prefill4k_q27_error(diag,2u); return;
  }
  const uint rowTiles = (p.rows + M - 1) / M, columnTiles = p.output_size / N;
  uint rTile = group.y, nTile = group.x;
  if (p.traversal == 1) { rTile = group.x; nTile = group.y; }
  if (p.traversal == 2) { rTile = group.y * 4 + (group.x & 3); nTile = group.x >> 2; }
  if (rTile >= rowTiles || nTile >= columnTiles) return;
  const uint row = rTile * M, column = nTile * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row) * p.input_size,
      dextents<int,2>{k,M},array<int,2>{1,k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int,2>{k,N},array<int,2>{1,k});
  constexpr auto descriptor = matmul2d_descriptor(M,N,static_cast<int>(dynamic_extent),
      false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  operation.run(a,b,dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (row + index[1] >= p.rows) continue;
    const bfloat value = bfloat(dot[i]);
    if (!isfinite(dot[i]) || !isfinite(float(value))) prefill4k_q27_error(diag,4u);
    output[ulong(row + index[1]) * p.output_size + column + index[0]] = value;
  }
}

#define Q27_CACHE_ENTRY(Name,W,M,N,SG) \
[[max_total_threads_per_threadgroup(SG*32)]] kernel void Name( \
    device bfloat *input [[buffer(0)]],device W *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]],device atomic_uint *diag [[buffer(3)]], \
    constant Prefill4KQ27Params &p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  prefill4k_q27_cache_tile<W,M,N,SG>(input,weights,output,diag,p,group,threads,tid); \
}
Q27_CACHE_ENTRY(prefill4k_q27_bf16_m32n128_s4,bfloat,32,128,4)
Q27_CACHE_ENTRY(prefill4k_q27_bf16_m64n128_s8,bfloat,64,128,8)
Q27_CACHE_ENTRY(prefill4k_q27_bf16_m128n64_s8,bfloat,128,64,8)
Q27_CACHE_ENTRY(prefill4k_q27_f32_m32n128_s4,float,32,128,4)
Q27_CACHE_ENTRY(prefill4k_q27_f32_m64n128_s8,float,64,128,8)
Q27_CACHE_ENTRY(prefill4k_q27_f32_m128n64_s8,float,128,64,8)
#undef Q27_CACHE_ENTRY
