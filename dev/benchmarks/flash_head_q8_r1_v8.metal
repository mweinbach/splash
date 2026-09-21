#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashAffine.h"

#pragma METAL fp math_mode(safe)

using namespace metal;

// Native source-layout affine primitives. The packed rows are unsigned LE
// bitstreams, not Splash's Q4 StorageN format. Each SIMD group owns one output
// column and reduces K in F32; eight columns share a 256-thread group. This is
// a correctness control for future measured matrix-tile kernels.

inline uint flash_affine_code(const device uchar *row, uint bits,
                              uint channel) {
  const ulong bit = ulong(channel) * bits;
  const ulong byte = bit >> 3;
  const uint shift = uint(bit & 7);
  uint word = uint(row[byte]);
  if (shift + bits > 8) word |= uint(row[byte + 1]) << 8;
  return (word >> shift) & ((1u << bits) - 1);
}

inline bool flash_affine_geometry(uint k, uint bits, uint group_size,
                                  ulong weight_stride,
                                  ulong parameter_stride) {
  return k && (bits == 4 || bits == 5 || bits == 6 || bits == 8) &&
      (group_size == 32 || group_size == 64 || group_size == 128) &&
      k % group_size == 0 && weight_stride >= (ulong(k) * bits + 7) / 8 &&
      parameter_stride >= ulong(k / group_size) * 2 &&
      parameter_stride % 2 == 0;
}

inline float flash_affine_nan() { return as_type<float>(0x7fc00000u); }
inline bool flash_affine_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

template <ushort Bits, ushort GroupSize, ushort Columns, bool ShuffleWords = false, ushort SimdGroups = 8>
inline void flash_head_q8_r1_v8_project(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  static_assert(Bits == 4 || Bits == 8);
  static_assert(GroupSize == 64 || GroupSize == 128);
  const uint nbase = group.x * (SimdGroups * Columns) + simd_group * Columns;
  const bool shape_ok = p.rows && p.selections && p.experts && p.output_size &&
      p.input_size && p.input_size % GroupSize == 0 &&
      p.bits == Bits && p.group_size == GroupSize &&
      !(p.flags & ~3u) && (!(p.flags & 2u) || (p.flags & 1u)) &&
      ((p.flags & 1u) || (p.experts == 1 && p.selections == 1)) &&
      p.weight_row_stride_bytes >= ulong(p.input_size) * Bits / 8 &&
      p.parameter_row_stride_bytes >= ulong(p.input_size / GroupSize) * 2 &&
      p.parameter_row_stride_bytes % 2 == 0 &&
      p.parameter_expert_stride_bytes % 2 == 0 &&
      (!ShuffleWords || (p.weight_row_stride_bytes % 4 == 0 &&
                     p.weight_expert_stride_bytes % 4 == 0));
  if (!shape_ok) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (nbase >= p.output_size || group.y >= p.rows || group.z >= p.selections)
    return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      for (ushort c = 0; c < Columns; ++c)
        if (nbase + c < p.output_size)
          output[route * p.output_size + nbase + c] =
              bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  const device uchar *wbase = weights + ulong(expert) *
      p.weight_expert_stride_bytes + ulong(nbase) * p.weight_row_stride_bytes;
  const ulong coefficient_offset = ulong(expert) *
      p.parameter_expert_stride_bytes + ulong(nbase) *
      p.parameter_row_stride_bytes;
  const device uchar *sbase = scales + coefficient_offset;
  const device uchar *bbase = biases + coefficient_offset;
  float sums[Columns];
  for (ushort c = 0; c < Columns; ++c) sums[c] = 0.0f;
  for (uint g = 0; g < p.input_size / GroupSize; ++g) {
    float sf[Columns];
    float bias[Columns];
    for (ushort c = 0; c < Columns; ++c) {
      if (nbase + c < p.output_size) {
        const device bfloat *s = reinterpret_cast<const device bfloat *>(
            sbase + ulong(c) * p.parameter_row_stride_bytes);
        const device bfloat *b = reinterpret_cast<const device bfloat *>(
            bbase + ulong(c) * p.parameter_row_stride_bytes);
        sf[c] = float(s[g]);
        bias[c] = float(b[g]);
      }
    }
    for (ushort subblock = 0; subblock < GroupSize / 32; ++subblock) {
      const uint k = g * GroupSize + uint(subblock) * 32 + lane;
      const float activation = float(x[k]);
      for (ushort c = 0; c < Columns; ++c) {
        if (nbase + c < p.output_size) {
          const device uchar *w = wbase + ulong(c) * p.weight_row_stride_bytes;
          uint code;
          if (ShuffleWords) {
            // Each aligned quartet loads one original LE uint and broadcasts
            // within its quartet. Byte extraction preserves the source k.
            const device uint *wu = reinterpret_cast<const device uint *>(w);
            uint packed = 0;
            if ((lane & 3u) == 0)
              packed = wu[k >> 2];
            packed = simd_shuffle(packed, lane & ~3u);
            code = (packed >> ((lane & 3u) * 8)) & 255u;
          } else {
            code = uint(w[k]);
          }
          const float coefficient = float(code) * sf[c] + bias[c];
          sums[c] += activation * coefficient;
        }
      }
    }
  }
  for (ushort c = 0; c < Columns; ++c) {
    const float sum = simd_sum(sums[c]);
    if (lane == 0 && nbase + c < p.output_size) {
      const bfloat result = bfloat(sum);
      if (!flash_affine_finite(sum) || !flash_affine_finite(float(result)))
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route * p.output_size + nbase + c] = result;
    }
  }
}


#define FLASH_HEAD_Q8_R1_V8(COLS, SG, SHUFFLE, LABEL) \
kernel void flash_head_q8_r1_v8_c##COLS##_s##SG##_##LABEL( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *expert_ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  flash_head_q8_r1_v8_project<8,64,COLS,SHUFFLE,SG>( \
      input,weights,scales,biases,expert_ids,output,diagnostics,p, \
      group,simd_group,lane); \
}
FLASH_HEAD_Q8_R1_V8(1,4,false,byte)
FLASH_HEAD_Q8_R1_V8(1,4,true,shuffle)
FLASH_HEAD_Q8_R1_V8(1,8,false,byte)
FLASH_HEAD_Q8_R1_V8(1,8,true,shuffle)
FLASH_HEAD_Q8_R1_V8(2,2,false,byte)
FLASH_HEAD_Q8_R1_V8(2,2,true,shuffle)
FLASH_HEAD_Q8_R1_V8(2,4,false,byte)
FLASH_HEAD_Q8_R1_V8(2,4,true,shuffle)
FLASH_HEAD_Q8_R1_V8(2,8,false,byte)
FLASH_HEAD_Q8_R1_V8(2,8,true,shuffle)
FLASH_HEAD_Q8_R1_V8(4,2,false,byte)
FLASH_HEAD_Q8_R1_V8(4,2,true,shuffle)
FLASH_HEAD_Q8_R1_V8(4,4,false,byte)
FLASH_HEAD_Q8_R1_V8(4,4,true,shuffle)
FLASH_HEAD_Q8_R1_V8(4,8,false,byte)
FLASH_HEAD_Q8_R1_V8(4,8,true,shuffle)
FLASH_HEAD_Q8_R1_V8(8,2,false,byte)
FLASH_HEAD_Q8_R1_V8(8,2,true,shuffle)
FLASH_HEAD_Q8_R1_V8(8,4,false,byte)
FLASH_HEAD_Q8_R1_V8(8,4,true,shuffle)
FLASH_HEAD_Q8_R1_V8(8,8,false,byte)
FLASH_HEAD_Q8_R1_V8(8,8,true,shuffle)
#undef FLASH_HEAD_Q8_R1_V8
