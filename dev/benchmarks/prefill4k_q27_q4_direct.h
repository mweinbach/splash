// Private reversible snapshot of the unchanged native Q4 helper.
// Two transformations only: symbol rename and device-sum route at SG8.
// Original helper SHA256: 6b61ba83e034982c527df613399f13714002d8f1a589ba1ed47347e824b33cd8
template <ushort TileM, ushort TileN, ushort Simdgroups, bool AddResidual,
          bool MultiplySiluGate>
inline void prefill4k_q27_direct_q4_tile(device bfloat *input, device uchar *weights,
                                device bfloat *scales, device bfloat *biases,
                                device bfloat *output, device bfloat *auxiliary,
                                uint output_size, uint input_size,
                                device const float *precomputed_sums,
                                uint output_origin, uint simd_lane,
                                uint simd_group,
                                threadgroup float *input_sums = nullptr) {
  constexpr bool StagedSums = false;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  constexpr ushort WeightTileN = 256; // weights are stored in 256-column tiles
  uint tile = output_origin / WeightTileN;
  uint tile_column = output_origin % WeightTileN;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * WeightTileN + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    accumulated[i] = 0.0f;
  }

  auto load_sums = [&](uint start) {
    uint count = min(uint(PrefillSumBatch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM;
         index += Simdgroups * 32) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_sums[index] = precomputed_sums[quant_group * TileM + row];
    }
  };
  if constexpr (StagedSums) {
    load_sums(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, TileM>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * WeightTileN * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);

#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * WeightTileN +
          tile_column + index[0];
      float sum = StagedSums
          ? input_sums[(quant_group % PrefillSumBatch) * TileM + row]
          : precomputed_sums[quant_group * TileM + row];
      accumulated[i] += partial[i] * float(scales[parameter]) +
                        sum * float(biases[parameter]);
    }
    if (StagedSums && quant_group % PrefillSumBatch == PrefillSumBatch - 1 &&
        quant_group + 1 < quant_groups) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_sums(quant_group + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    float value = float(bfloat(accumulated[i]));
    if constexpr (MultiplySiluGate) {
      auto index = accumulated.get_multidimensional_index(i);
      float gate =
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
    }
    if constexpr (AddResidual) {
      auto index = accumulated.get_multidimensional_index(i);
      value +=
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.slice<TileN, TileM>(output_origin, 0));
}
