#pragma once

// Standalone component only. The current main-prefill constructor parameters,
// original M32 jobs, scales, diagnostics and output ownership are unchanged.
#include "metal/abi/FlashInt8ExpertStore.h"

#ifndef __METAL_VERSION__
#include <cstdint>
namespace splash::flash::main_prefill_rhs_tile64_sep22 {
inline constexpr uint32_t kRows = 2048;
inline constexpr uint32_t kSelections = 10;
inline constexpr uint32_t kExperts = 512;
inline constexpr uint32_t kNativeJobRows = 32;
inline constexpr uint32_t kSmallTailDescriptorRows = 16;
inline constexpr uint32_t kTileOutputs = 64;
inline constexpr uint32_t kKStep = 128;
inline constexpr uint32_t kSIMDGroups = 2;
inline constexpr uint32_t kThreads = 64;
inline constexpr uint32_t kGateK = 2560, kGateN = 640;
inline constexpr uint32_t kDownK = 640, kDownN = 2560;
inline constexpr uint32_t kGateKSteps = 20, kDownKSteps = 5;

inline constexpr const char *kBaselineGate = "main_prefill_rhs_tile64_sep22_baseline_gate_up_m16_tail";
inline constexpr const char *kBaselineDown = "main_prefill_rhs_tile64_sep22_baseline_down_scatter_m16_tail";
inline constexpr const char *kCandidateGate = "main_prefill_rhs_tile64_sep22_candidate_gate_up_m16_tail";
inline constexpr const char *kCandidateDown = "main_prefill_rhs_tile64_sep22_candidate_down_scatter_m16_tail";
inline constexpr const char *kBaselineGateProbe = "main_prefill_rhs_tile64_sep22_baseline_gate_up_m16_tail_probe";
inline constexpr const char *kBaselineDownProbe = "main_prefill_rhs_tile64_sep22_baseline_down_scatter_m16_tail_probe";
inline constexpr const char *kCandidateGateProbe = "main_prefill_rhs_tile64_sep22_candidate_gate_up_m16_tail_probe";
inline constexpr const char *kCandidateDownProbe = "main_prefill_rhs_tile64_sep22_candidate_down_scatter_m16_tail_probe";

// GU normal: A0,G1,Gscale2,U3,Uscale4,ranks5,offsets6,jobs7,count8,
// BF16 SwiGLU output9,diagnostics10,original 32-byte parameters11.
// GU probe appends raw F32 G12/U13 and post-scale BF16 G14/U15. Probe output9
// is produced by the same literal arithmetic as its normal counterpart.
inline constexpr uint32_t kGateParamsBinding = 11;
inline constexpr uint32_t kGateRawGBinding = 12, kGateRawUBinding = 13;
inline constexpr uint32_t kGateScaledGBinding = 14, kGateScaledUBinding = 15;
// Down normal: A0,W1,scale2,ranks3,offsets4,jobs5,count6,routeMap7,
// canonical BF16 output8,diagnostics9,original 32-byte parameters10.
// Down probe appends raw unscaled F32 dot11 and post-scale BF16 projection12.
inline constexpr uint32_t kDownParamsBinding = 10;
inline constexpr uint32_t kDownRawBinding = 11, kDownScaledBinding = 12;

// Pure index formulas for the harness's independently checked, bounded pack.
// Caller supplies an admitted rank/N/K and n<N, k<K. No buffer is accessed.
[[nodiscard]] constexpr uint64_t originalCoefficientIndex(uint32_t rank,
    uint32_t n, uint32_t k, uint32_t outputs, uint32_t inputs) noexcept {
  return (uint64_t{rank} * outputs + n) * inputs + k;
}
[[nodiscard]] constexpr uint64_t tile64CoefficientIndex(uint32_t rank,
    uint32_t n, uint32_t k, uint32_t outputs, uint32_t inputs) noexcept {
  return uint64_t{rank} * outputs * inputs + uint64_t{n / 64} * inputs * 64 +
      uint64_t{k} * 64 + n % 64;
}
} // namespace splash::flash::main_prefill_rhs_tile64_sep22
#endif
