// Private untimed projection tap for direct gathered MPP. The sibling exports
// the timed producer kernels from this same AIR; do not also link their AIR.
// Descriptor equality is not a claim of exact pairing with bucketed MPP.
#if __METAL_VERSION__ >= 410
#include "prefill_moe_sep21/gathered_sg2.metal"
#include "prefill4k_allrows_qmv_probe.h"

#pragma METAL fp math_mode(safe)

inline void prefill_moe_sep21_gathered_sg2_probe_nan(device float *dot, device float *scaled,
    device bfloat *projection, ulong index) {
  const float value = as_type<float>(0x7fc00000u);
  dot[index] = value;
  scaled[index] = value;
  projection[index] = bfloat(value);
}

template <ushort K, ushort Width>
inline void prefill_moe_sep21_gathered_sg2_projection_probe_execute(
    device const bfloat *input, device const int8_t *codes,
    device const float *scales, device const uint *ranks, device const long *ids,
    device float *dot, device float *scaled, device bfloat *projection,
    device uint *diag, uint3 group, uint tid,
    threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite) {
  static_assert((K == 2560 && Width == 640) || (K == 640 && Width == 2560));
  constexpr ushort N = 64;
  const ulong route = ulong(group.y) * 10 + group.z;
  const ulong input_row = K == 2560 ? ulong(group.y) : route;
  const device bfloat *x = input + input_row * K;

  const uint rank = prefill_moe_sep21_gathered_sg2_rank(ids, ranks, route, tid, diag);
  // Use the producer's scan and local sanitization, including hidden-row
  // diagnostics when all IDs are invalid. Excluded down routes never read A.
  // Both the scan predicate and its returned bad flag are threadgroup uniform.
  const bool bad = (K == 2560 || rank != UINT_MAX)
      ? prefill_moe_sep21_gathered_sg2_scan<K>(x, safe_a, nonfinite, tid, diag) : false;
  const uint column = group.x * N;
  if (rank == UINT_MAX) {
    // Standalone tap policy deliberately differs from the timed gate's interim
    // ID-only poison policy: both legal phases poison all taps and retain bits5.
    if (!tid) prefill_moe_sep21_gathered_sg2_error(diag, 5u);
    for (uint n = tid; n < N; n += 64)
      prefill_moe_sep21_gathered_sg2_probe_nan(dot, scaled, projection, route * Width + column + n);
    return;
  }

  // No coefficient pointer is constructed before rank validation. Finite A
  // retains device storage; malformed A uses the producer's local TG fallback.
  auto a = tensor(const_cast<device bfloat *>(x),
      dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto safe = tensor(safe_a, dextents<int, 2>{K, 1}, array<int, 2>{1, K});
  auto b = tensor(const_cast<device int8_t *>(codes) +
      (ulong(rank) * Width + column) * K,
      dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto descriptor = matmul2d_descriptor(16, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<2>> operation;
  auto raw = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  if (bad) operation.run(safe, b, raw); else operation.run(a, b, raw);

#pragma unroll
  for (ushort i = 0; i < raw.get_capacity(); ++i) {
    if (!raw.is_valid_element(i)) continue;
    const auto index = raw.get_multidimensional_index(i);
    if (index[1] != 0) continue;
    const uint n = column + index[0];
    const float scale = scales[ulong(rank) * Width + n];
    const float result = raw[i] * scale;
    const bfloat rounded = bfloat(result);
    if (!(scale > 0.0f) || !prefill_moe_sep21_gathered_sg2_finite(scale) ||
        !prefill_moe_sep21_gathered_sg2_finite(raw[i]) || !prefill_moe_sep21_gathered_sg2_finite(result) ||
        !prefill_moe_sep21_gathered_sg2_finite(float(rounded))) prefill_moe_sep21_gathered_sg2_error(diag, 4u);
    const ulong output = route * Width + n;
    dot[output] = raw[i];
    scaled[output] = result;
    projection[output] = rounded;
  }
}

// Bindings match flash_qmv_probe_c1/c2, with columns=0 and 64 columns/group:
// 0 BF16 input, 1 signed I8 codes, 2 F32 row scales, 3 U32 ranks[512],
// 4 original I64 IDs[R,10], 5 raw F32[R,10,N], 6 scaled F32[R,10,N],
// 7 BF16 projection[R,10,N], 8 sticky diagnostic, 9 FlashQMVProbeParams.
// Phases: K2560/N640/per_route_input=0 or K640/N2560/per_route_input=1.
// Grid {N/64,R,10}, threads {64,1,1}. One runtime-phased entry reserves
// max(K)*sizeof(BF16)=5120 static TG bytes; down uses only the first 1280 bytes.
kernel void prefill_moe_sep21_gathered_sg2_projection_probe(
    device const bfloat *input [[buffer(0)]],
    device const int8_t *codes [[buffer(1)]], device const float *scales [[buffer(2)]],
    device const uint *ranks [[buffer(3)]], device const long *ids [[buffer(4)]],
    device float *dot [[buffer(5)]], device float *scaled [[buffer(6)]],
    device bfloat *projection [[buffer(7)]], device uint *diag [[buffer(8)]],
    constant FlashQMVProbeParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!p.rows || p.rows > 16 || p.selections != 10 || p.experts != 512 ||
      p.reserved || p.columns ||
      !((p.input_size == 2560 && p.output_size == 640 && !p.per_route_input) ||
        (p.input_size == 640 && p.output_size == 2560 && p.per_route_input == 1)) ||
      group.x >= p.output_size / 64 || group.y >= p.rows || group.z >= 10 ||
      threads.x != 64 || threads.y != 1 || threads.z != 1) {
    if (!tid) prefill_moe_sep21_gathered_sg2_error(diag, 2u);
    return;
  }
  alignas(16) threadgroup bfloat safe_a[2560];
  threadgroup atomic_uint nonfinite;
  if (p.input_size == 2560)
    prefill_moe_sep21_gathered_sg2_projection_probe_execute<2560, 640>(input, codes, scales,
        ranks, ids, dot, scaled, projection, diag, group, tid, safe_a, &nonfinite);
  else
    prefill_moe_sep21_gathered_sg2_projection_probe_execute<640, 2560>(input, codes, scales,
        ranks, ids, dot, scaled, projection, diag, group, tid, safe_a, &nonfinite);
}
#endif
