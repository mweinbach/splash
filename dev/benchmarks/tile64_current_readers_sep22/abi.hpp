#pragma once
// Standalone native C1/R4 and gathered AR1 readers. Original parameter ABIs.
#include "metal/abi/FlashInt8ExpertStore.h"
#include "flash/FlashGatheredMPP.hpp"
#ifndef __METAL_VERSION__
#include <cstdint>
namespace splash::flash::tile64_current_readers_sep22 {
inline constexpr uint32_t kM = 16, kN = 64, kSIMDGroups = 4, kThreads = 128;
inline constexpr uint32_t kGateK = 2560, kGateN = 640, kDownK = 640, kDownN = 2560;
inline constexpr uint32_t kNativeGateParams = 11, kNativeDownParams = 10;
inline constexpr uint32_t kGatheredGateParams = 9, kGatheredDownParams = 7;
inline constexpr uint32_t kNativeGateRawG = 12, kNativeGateRawU = 13, kNativeGateBF16G = 14, kNativeGateBF16U = 15;
inline constexpr uint32_t kNativeDownRaw = 11, kNativeDownBF16 = 12;
inline constexpr uint32_t kGatheredGateRawG = 10, kGatheredGateRawU = 11, kGatheredGateBF16G = 12, kGatheredGateBF16U = 13;
inline constexpr uint32_t kGatheredDownRaw = 8, kGatheredDownBF16 = 9;
inline constexpr const char *kBaselineNativeGate = "tile64_current_readers_sep22_baseline_native_gate_up_m16_sg4";
inline constexpr const char *kBaselineNativeDown = "tile64_current_readers_sep22_baseline_native_down_m16_sg4";
inline constexpr const char *kCandidateNativeGate = "tile64_current_readers_sep22_candidate_native_gate_up_m16_sg4";
inline constexpr const char *kCandidateNativeDown = "tile64_current_readers_sep22_candidate_native_down_m16_sg4";
inline constexpr const char *kBaselineGatheredGate = "tile64_current_readers_sep22_baseline_gathered_gate_up_m16_sg4";
inline constexpr const char *kBaselineGatheredDown = "tile64_current_readers_sep22_baseline_gathered_down_m16_sg4";
inline constexpr const char *kCandidateGatheredGate = "tile64_current_readers_sep22_candidate_gathered_gate_up_m16_sg4";
inline constexpr const char *kCandidateGatheredDown = "tile64_current_readers_sep22_candidate_gathered_down_m16_sg4";
inline constexpr const char *kBaselineNativeGateProbe = "tile64_current_readers_sep22_baseline_native_gate_up_m16_sg4_probe";
inline constexpr const char *kBaselineNativeDownProbe = "tile64_current_readers_sep22_baseline_native_down_m16_sg4_probe";
inline constexpr const char *kCandidateNativeGateProbe = "tile64_current_readers_sep22_candidate_native_gate_up_m16_sg4_probe";
inline constexpr const char *kCandidateNativeDownProbe = "tile64_current_readers_sep22_candidate_native_down_m16_sg4_probe";
inline constexpr const char *kBaselineGatheredGateProbe = "tile64_current_readers_sep22_baseline_gathered_gate_up_m16_sg4_probe";
inline constexpr const char *kBaselineGatheredDownProbe = "tile64_current_readers_sep22_baseline_gathered_down_m16_sg4_probe";
inline constexpr const char *kCandidateGatheredGateProbe = "tile64_current_readers_sep22_candidate_gathered_gate_up_m16_sg4_probe";
inline constexpr const char *kCandidateGatheredDownProbe = "tile64_current_readers_sep22_candidate_gathered_down_m16_sg4_probe";
// Normal entries access no tap buffers. Probe slots append AFTER the original
// parameter index; copied dispatches must keep graph-owned inline bytes alive.
// Native GU taps are packed [validRoutes,640]; native down taps use canonical
// routeMap output order [routes,2560]. Gathered taps are canonical [R*10,N].
// Every raw tap is a literal unscaled F32 cooperative destination. BF16 taps
// are literal post-scale projection values, before the final GU multiply.
static_assert(sizeof(FlashInt8ExpertStoreParams) == 32);
static_assert(sizeof(FlashGatheredMPPParams) == 16);
} // namespace splash::flash::tile64_current_readers_sep22
#endif
