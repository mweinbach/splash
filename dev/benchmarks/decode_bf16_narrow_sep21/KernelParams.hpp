#pragma once

#include "metal/abi/FlashDenseSmallRows.h"

// The isolated narrow-tile experiment binds precisely the baseline's eight
// uint32 fields. Timed kernels bind this at buffer 4; untimed tap probes bind
// an additional F32 destination at buffer 4 and these parameters at buffer 5.
using Sep21BF16NarrowParams = FlashDenseSmallRowsParams;
static_assert(sizeof(Sep21BF16NarrowParams) == 32,
              "Narrow BF16 probes reuse the original small-row dense ABI");

#ifndef __METAL_VERSION__
#include <array>
#include <cstddef>
#include <type_traits>

static_assert(std::is_standard_layout_v<Sep21BF16NarrowParams>);
static_assert(offsetof(Sep21BF16NarrowParams, rows) == 0);
static_assert(offsetof(Sep21BF16NarrowParams, padded_rows) == 4);
static_assert(offsetof(Sep21BF16NarrowParams, input_size) == 8);
static_assert(offsetof(Sep21BF16NarrowParams, output_size) == 12);
static_assert(offsetof(Sep21BF16NarrowParams, output_begin) == 16);
static_assert(offsetof(Sep21BF16NarrowParams, output_count) == 20);
static_assert(offsetof(Sep21BF16NarrowParams, tile_rows) == 24);
static_assert(offsetof(Sep21BF16NarrowParams, tile_outputs) == 28);

namespace sep21_bf16_narrow {
inline constexpr uint32_t kInvalidGeometry = 2u;
inline constexpr uint32_t kNonfinite = 4u;

struct Variant final {
  const char *kernel;
  const char *probe;
  uint32_t tileRows;
  uint32_t tileOutputs;
  uint32_t simdgroups;
  constexpr uint32_t threads() const noexcept { return simdgroups * 32; }
};

inline constexpr std::array<Variant, 12> kVariants{{
    {"sep21_bf16_m8_n32_s1", "sep21_bf16_probe_m8_n32_s1", 8, 32, 1},
    {"sep21_bf16_m8_n32_s2", "sep21_bf16_probe_m8_n32_s2", 8, 32, 2},
    {"sep21_bf16_m8_n32_s4", "sep21_bf16_probe_m8_n32_s4", 8, 32, 4},
    {"sep21_bf16_m8_n64_s1", "sep21_bf16_probe_m8_n64_s1", 8, 64, 1},
    {"sep21_bf16_m8_n64_s2", "sep21_bf16_probe_m8_n64_s2", 8, 64, 2},
    {"sep21_bf16_m8_n64_s4", "sep21_bf16_probe_m8_n64_s4", 8, 64, 4},
    {"sep21_bf16_m16_n32_s1", "sep21_bf16_probe_m16_n32_s1", 16, 32, 1},
    {"sep21_bf16_m16_n32_s2", "sep21_bf16_probe_m16_n32_s2", 16, 32, 2},
    {"sep21_bf16_m16_n32_s4", "sep21_bf16_probe_m16_n32_s4", 16, 32, 4},
    {"sep21_bf16_m16_n64_s1", "sep21_bf16_probe_m16_n64_s1", 16, 64, 1},
    {"sep21_bf16_m16_n64_s2", "sep21_bf16_probe_m16_n64_s2", 16, 64, 2},
    {"sep21_bf16_m16_n64_s4", "sep21_bf16_probe_m16_n64_s4", 16, 64, 4},
}};

// The N128 controls retain the original production kernels for timing. Their
// untimed probes expose the raw F32 destination with the identical descriptor.
inline constexpr std::array<Variant, 2> kN128Controls{{
    {"flash_dense_small_rows_m8_n128", "sep21_bf16_probe_m8_n128_s4", 8, 128, 4},
    {"flash_dense_small_rows_m16_n128", "sep21_bf16_probe_m16_n128_s4", 16, 128, 4},
}};

// Mirrors the shader's uniform parameter checks. This deliberately rejects
// M16 padding for a smaller real-row count: M8 owns rows 1/2/4/8 and M16 row16.
constexpr bool geometry(const Sep21BF16NarrowParams &p, const Variant &v) noexcept {
  const bool rows = v.tileRows == 8
      ? (p.rows == 1 || p.rows == 2 || p.rows == 4 || p.rows == 8)
      : v.tileRows == 16 && p.rows == 16;
  return rows && p.tile_rows == v.tileRows && p.tile_outputs == v.tileOutputs &&
      p.padded_rows == v.tileRows && p.input_size && p.input_size <= 32768 &&
      p.input_size % 32 == 0 && p.output_size && p.output_size % 64 == 0 &&
      p.output_count && p.output_count % v.tileOutputs == 0 &&
      p.output_begin <= p.output_size &&
      p.output_count <= p.output_size - p.output_begin &&
      (v.simdgroups == 1 || v.simdgroups == 2 || v.simdgroups == 4);
}
} // namespace sep21_bf16_narrow
#endif
