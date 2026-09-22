#pragma once
// Raw, non-atomic uint64 counters. GPU writes are serialized by the existing
// main-trunk owner; host copies occur only after its command has completed.
#include <array>
#include <cstdint>
#include <cstring>

namespace splash::flash::gdn_wy_hc_sep21::telemetry {
inline constexpr uint64_t kBytes = 256;
inline constexpr uint32_t kHeads = 48;
inline constexpr const char *kEligibleKernel = "private_gdn_wy_telemetry_eligible";
inline constexpr const char *kReplayKernel = "private_gdn_wy_telemetry_replay";
inline constexpr const char *kScope =
    "cumulative completed GPU phase observations since Forward startup; singleton main WY layers; native replay reasons overlap";
inline constexpr const char *kAppliedDefinition =
    "unique heads entering at least one WY value tile on a healthy completed dispatch; equals zero-flag heads after preparation";
struct Params final { uint32_t heads = kHeads, reserved = 0; };
static_assert(sizeof(Params) == 8);
enum Slot : uint32_t {
  PrepareCalls, ReplayCalls, ScheduledHeads, EligibleHeads, AppliedHeads,
  ReplayedHeads, UnflaggedHeads, RangeHeads, CancellationHeads,
  NonfiniteHeads, NormRangeHeads, UnknownReasonHeads, HistogramBegin,
  LastEligibleHeads = 28, LastReplayedHeads, LastReasonMask, Saturated,
};
struct Snapshot final {
  std::array<uint64_t,32> words{};
  [[nodiscard]] uint64_t operator[](Slot slot) const noexcept { return words[slot]; }
  [[nodiscard]] std::array<uint64_t,16> histogram() const noexcept {
    std::array<uint64_t,16> result{};
    for (uint32_t i=0;i<16;++i) result[i]=words[HistogramBegin+i];
    return result;
  }
};
static_assert(sizeof(Snapshot) == kBytes);
inline Snapshot readCompleted(const void *contents) noexcept {
  Snapshot result;
  if (contents) std::memcpy(result.words.data(),contents,kBytes);
  return result;
}
} // namespace splash::flash::gdn_wy_hc_sep21::telemetry
