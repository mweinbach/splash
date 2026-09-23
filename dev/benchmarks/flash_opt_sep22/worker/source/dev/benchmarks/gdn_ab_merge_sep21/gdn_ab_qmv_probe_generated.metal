// Generated untimed instrumentation; source journal sealed beside build.
namespace gdn_ab_original_qmv {
template <ushort Bits, ushort GroupSize, bool Fast>
inline void project_math_tap(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    device bfloat *output, device atomic_uint *diagnostics,
    device float *raw_f32,
    constant FlashAffineParams &p, uint out_row, uint expert,
    ulong route, uint lane) {
  constexpr uint PacksPerThread = Fast ? 2 : 1;
  constexpr uint Values = pack_factor_32<Bits>() * PacksPerThread;
  constexpr uint Block = Values * 32;
  static_assert(GroupSize % Values == 0);
  thread float x_thread[Values];
  thread float result[4] = {0};

  if (Fast) {
    // qmv_fast_impl: a complete packed block per lane, then four output rows.
    for (uint k = 0; k < p.input_size; k += Block) {
      const uint channel = k + lane * Values;
      const float sum = mlx_qmv_f32xsum_v1_load_vector<bfloat, float, Values, Bits>(
          x + channel, x_thread);
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, Values, false, result);
    }
  } else {
    // qmv_impl processes all but its final block with the regular helpers.
    // Use subtraction after k<K so tiny K cannot underflow this condition.
    uint k = 0;
    for (; k < p.input_size && p.input_size - k > Block; k += Block) {
      const uint channel = k + lane * Values;
      const float sum = mlx_qmv_f32xsum_v1_load_vector<bfloat, float, Values, Bits>(
          x + channel, x_thread);
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, Values, false, result);
    }
    const uint lane_start = lane * Values;
    const uint available = p.input_size - k;
    const uint remaining = available > lane_start
        ? min(available - lane_start, Values) : 0;
    if (remaining > 0) {
      const uint channel = k + lane_start;
      const float sum = mlx_qmv_f32xsum_v1_load_vector_safe<
          bfloat, float, Values, Bits>(x + channel, x_thread, int(remaining));
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, remaining, true, result);
    }
  }

  for (ushort row = 0; row < 4; ++row) {
    if (out_row + row < p.output_size) {
      const float sum = simd_sum(result[row]);
      if (lane == 0) {
        raw_f32[route * p.output_size + out_row + row] = sum;
        const bfloat value = bfloat(sum);
        if (!finite(sum) || !finite(float(value)))
          atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
        output[route * p.output_size + out_row + row] = value;
      }
    }
  }
}

template <ushort Bits, ushort GroupSize>
inline void project_tap(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, device float *raw_f32,
    constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  static_assert(Bits == 4 || Bits == 5 || Bits == 6 || Bits == 8);
  static_assert(GroupSize == 64 || GroupSize == 128);
  if (!p.rows || !p.selections || !p.experts || !p.output_size ||
      !p.input_size || p.bits != Bits || p.group_size != GroupSize ||
      p.input_size % GroupSize || (p.flags & ~3u) ||
      ((p.flags & 2u) && !(p.flags & 1u)) ||
      (!(p.flags & 1u) && (p.experts != 1 || p.selections != 1)) ||
      p.weight_row_stride_bytes < ulong(p.input_size) * Bits / 8 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / GroupSize) * 2 ||
      p.parameter_row_stride_bytes % 2 ||
      p.parameter_expert_stride_bytes % 2) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint out_row = group.x * 8 + simd_group * 4;
  if (out_row >= p.output_size || group.y >= p.rows ||
      group.z >= p.selections) return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      for (ushort row = 0; row < 4; ++row)
        if (out_row + row < p.output_size)
          output[route * p.output_size + out_row + row] =
              bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  constexpr uint FastBlock = pack_factor_32<Bits>() * 2 * 32;
  if (p.output_size >= 8 && p.input_size % FastBlock == 0) {
    project_math_tap<Bits, GroupSize, true>(
        x, weights, scales, biases, output, diagnostics, raw_f32, p, out_row,
        uint(expert), route, lane);
  } else {
    project_math_tap<Bits, GroupSize, false>(
        x, weights, scales, biases, output, diagnostics, raw_f32, p, out_row,
        uint(expert), route, lane);
  }
}
} // namespace gdn_ab_original_qmv
