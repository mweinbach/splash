#pragma once

// Private singleton prefill only. Policy parsing and identities create no GPU
// device or resources and inspect no model payloads. Source hashes are frozen
// by worker_overlay.py, never obtained from mutable runtime counters.
#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"
#include "dev/benchmarks/gdn_wy_hc_worker_sep21/worker_source_hashes.hpp"
#include "dev/benchmarks/gdn_wy_hc_worker_sep21/telemetry.hpp"
#include "engine/Json.hpp"
#include "flash/FlashGDN.hpp"
#include "metal/abi/FlashGDN.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>

namespace splash::flash::gdn_wy_hc_sep21 {

inline constexpr const char *kFlag = "SPLASH_FLASH_GDN_PREFILL_WY_SEP21";
inline constexpr const char *kPolicy =
    "qwen4-gdn-prefill-guarded-wy-v6-f32-v32-t32-sg8-cached-wnorm-singleton-r64to2048-native-v16t16-replay-over-hc-sep21-v1";
inline constexpr const char *kMarker =
    ";private-gdn-prefill-wy-v6-guarded-v32-t32-sg8-cached-wnorm-singleton-r64to2048-native-v16t16-replay-over-hc-sep21-v1";
inline constexpr uint32_t kWorkspaceRows = 2048, kWorkspaceLanes = 1;
inline constexpr uint64_t kCoefficientsBytes = 164757504;
inline constexpr uint64_t kSnapshotBytes = 3145728;
inline constexpr uint64_t kFlagsBytes = 192;
inline constexpr uint64_t kArenaLogicalBytes = kCoefficientsBytes + kSnapshotBytes + kFlagsBytes;
inline constexpr uint64_t kArenaPhysicalBytes = 167919616;
inline constexpr uint64_t kArenaBytes = kArenaPhysicalBytes;
inline constexpr uint64_t kSnapshotOffset = kCoefficientsBytes;
inline constexpr uint64_t kFlagsOffset = kSnapshotOffset + kSnapshotBytes;
inline constexpr uint64_t kTelemetryOffset = kFlagsOffset + kFlagsBytes;
static_assert(kTelemetryOffset % alignof(uint64_t) == 0);
static_assert(kTelemetryOffset + telemetry::kBytes <= kArenaPhysicalBytes);
inline constexpr const char *kScope =
    "main singleton nonverification staged prefill rows64..2048; batch/decode/verify/lazyrollback/replay/pending tapes inherited";

namespace detail {
inline bool parseRequested(const char *raw, const char *stagedRaw) {
  if (!raw || std::strcmp(raw, "0") == 0) return false;
  if (std::strcmp(raw, "1") != 0)
    throw std::invalid_argument(std::string(kFlag) + " must be exactly 0 or 1 (missing means 0)");
  if (!stagedRaw || std::strcmp(stagedRaw, "1") != 0)
    throw std::invalid_argument(std::string(kFlag) + "=1 requires SPLASH_FLASH_GDN_STAGED=1");
  return true;
}
} // namespace detail

inline bool requested() {
  static const bool enabled = detail::parseRequested(std::getenv(kFlag), std::getenv("SPLASH_FLASH_GDN_STAGED"));
  return enabled;
}
constexpr bool eligible(uint64_t rows, uint64_t lanes, bool enabled,
                        bool verification = false) noexcept {
  return enabled && !verification && rows >= 64 && rows <= kWorkspaceRows && lanes == 1;
}
constexpr bool provisioning(uint64_t maximumRows, bool enabled) noexcept {
  return enabled && maximumRows >= 64;
}
constexpr uint64_t plannedBytes(uint64_t maximumRows, bool enabled) noexcept {
  return provisioning(maximumRows, enabled) ? kArenaPhysicalBytes : 0;
}
constexpr const char *selectionMarker(bool enabled) noexcept { return enabled ? kMarker : ""; }

inline std::string numericalIdentity(std::string_view base, bool enabled) {
  if (!enabled) return std::string(base);
  std::string sourceBound(base);
  for (const char *value : {kPolicy, kMarker, kCandidateSourceSHA256,
      kNativeFallbackSourceSHA256, kSnapshotSourceSHA256, kGuardProofSHA256}) {
    sourceBound += '\n'; sourceBound += value;
  }
  return gdn_prefill_fma_sep21::detail::sha256(sourceBound);
}
inline std::string workspaceIdentity() {
  return gdn_prefill_fma_sep21::detail::sha256(numericalIdentity("fixed-shared-arena:R2048:B1:T32:H48:stride53632:coeff0+164757504:snapshot164757504+3145728:flags167903232+192:telemetry167903424+256:physical167919616:align16384", true) + '\n' + kTelemetrySourceSHA256 + '\n' + kTelemetryABISHA256);
}

struct Counters final {
  uint64_t encodedCalls = 0, encodedRows = 0, completedCalls = 0;
  uint32_t lastLayerFlaggedHeads = 0, lastLayerGuardReasons = 0;
  telemetry::Snapshot gpu;
};

inline std::string countersJSON(const Counters &c) {
  using namespace telemetry;
  std::ostringstream out;
  out << R"({"scope":)" << json::quote(telemetry::kScope)
      << R"(,"applied_definition":)" << json::quote(kAppliedDefinition)
      << R"(,"counters_excluded_from_identity":true,"encoded_layer_calls":)" << c.encodedCalls
      << R"(,"encoded_layer_rows":)" << c.encodedRows
      << R"(,"completed_layer_calls":)" << c.completedCalls
      << R"(,"prepare_calls":)" << c.gpu[PrepareCalls]
      << R"(,"replay_calls":)" << c.gpu[ReplayCalls]
      << R"(,"scheduled_heads":)" << c.gpu[ScheduledHeads]
      << R"(,"eligible_heads":)" << c.gpu[EligibleHeads]
      << R"(,"applied_heads":)" << c.gpu[AppliedHeads]
      << R"(,"replayed_heads":)" << c.gpu[ReplayedHeads]
      << R"(,"unflagged_heads":)" << c.gpu[UnflaggedHeads]
      << R"(,"reason_bit_counts":{"range":)" << c.gpu[RangeHeads]
      << R"(,"cancellation":)" << c.gpu[CancellationHeads]
      << R"(,"nonfinite":)" << c.gpu[NonfiniteHeads]
      << R"(,"norm_range":)" << c.gpu[NormRangeHeads]
      << R"(,"unknown":)" << c.gpu[UnknownReasonHeads]
      << R"(},"reason_mask_histogram":[)";
  const auto histogram = c.gpu.histogram();
  for (uint32_t i = 0; i < 16; ++i) { if (i) out << ','; out << histogram[i]; }
  out << R"(],"last_eligible_heads":)" << c.gpu[LastEligibleHeads]
      << R"(,"last_replayed_heads":)" << c.gpu[LastReplayedHeads]
      << R"(,"last_reason_mask":)" << c.gpu[LastReasonMask]
      << R"(,"saturated":)" << (c.gpu[Saturated] ? "true" : "false") << '}';
  return out.str();
}

class Workspace final {
public:
  explicit Workspace(metal::MetalBackend &backend) {
    const auto before = backend.memoryStats().allocatedBytes;
    arena_ = backend.allocateBuffer(kArenaPhysicalBytes, metal::BufferStorage::Shared,
                                   "private-gdn-wy-fixed-singleton-prefill-arena");
    if (!arena_ || arena_.sizeBytes() != kArenaPhysicalBytes || !arena_.contents() ||
        metal::allocationDelta(before, backend.memoryStats().allocatedBytes) != kArenaPhysicalBytes)
      throw std::logic_error("private GDN WY arena allocation/admission extent mismatch");
    coefficients_ = backend.view(arena_, 0, kCoefficientsBytes);
    snapshot_ = backend.view(arena_, kSnapshotOffset, kSnapshotBytes);
    flags_ = backend.view(arena_, kFlagsOffset, kFlagsBytes);
    telemetry_ = backend.view(arena_, kTelemetryOffset, telemetry::kBytes);
    std::memset(telemetry_.contents(), 0, telemetry::kBytes);
  }
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return arena_.sizeBytes(); }
  [[nodiscard]] std::array<metal::MetalBuffer, 4> buffers() const {
    return {coefficients_, snapshot_, flags_, telemetry_};
  }
  [[nodiscard]] Counters counters() const noexcept { return counters_; }
  void complete(uint64_t layerCalls) {
    if (!layerCalls) return;
    // One host copy after existing synchronous command completion. GPU phases
    // accumulate every layer; no flags/tensors are read by the host per layer.
    counters_.completedCalls += layerCalls;
    counters_.gpu = telemetry::readCompleted(telemetry_.contents());
    counters_.lastLayerFlaggedHeads = static_cast<uint32_t>(counters_.gpu[telemetry::LastReplayedHeads]);
    counters_.lastLayerGuardReasons = static_cast<uint32_t>(counters_.gpu[telemetry::LastReasonMask]);
  }
  void addPrefill(metal::CommandGraph &graph, const FlashGDNWeights &w,
                 const FlashGDNBuffers &b, const FlashGDNState &s,
                 uint32_t rows, float normEpsilon) {
    if (!eligible(rows, 1, true))
      throw std::invalid_argument("private GDN WY singleton prefill geometry unsupported");
    metal::CommandGraph validated;
    addGDN(validated, w, b, s, rows, 1, normEpsilon);
    const FlashGDNParams p{rows, 1, 16, 48, 128, 128, 4, normEpsilon,
        s.convolutionLaneStrideBytes ? s.convolutionLaneStrideBytes : flashGDNConvolutionLaneBytes(),
        s.recurrentLaneStrideBytes ? s.recurrentLaneStrideBytes : flashGDNRecurrentLaneBytes()};
    graph.add("flash_gdn_fused_prepare", {b.qkv, b.a, b.b, w.convolution->buffer,
        w.aLog->buffer, w.timeBias->buffer, s.convolution, b.mixed, b.decay, b.beta, b.diagnostics},
        p, {16, rows, 1}, {128, 1, 1});
    // The private backend emits an explicit MTLBarrierScopeBuffers barrier
    // after each of these five exact names, in normal and profiled execution.
    graph.add("private_gdn_wy_snapshot", {s.recurrent, snapshot_, flags_, b.diagnostics},
        p, {48, 64, 1}, {256, 1, 1});
    graph.add("private_gdn_wy_prepare_t32_sg8", {b.mixed, b.decay, b.beta, coefficients_, flags_, b.diagnostics},
        p, {48, (rows + 31) / 32, 1}, {256, 1, 1});
    graph.add("private_gdn_wy_telemetry_eligible", {flags_, telemetry_}, telemetry::Params{},
        {1, 1, 1}, {32, 1, 1});
    graph.add("private_gdn_wy_v32_t32_sg8", {b.mixed, b.decay, b.beta, s.recurrent,
        b.recurrentRows, b.diagnostics, coefficients_, flags_}, p, {48, 4, 1}, {256, 1, 1});
    graph.add("private_gdn_wy_restore", {snapshot_, s.recurrent, flags_, b.diagnostics},
        p, {48, 1, 1}, {256, 1, 1});
    graph.add("private_gdn_wy_native_fallback", {b.mixed, b.decay, b.beta, s.recurrent,
        b.recurrentRows, b.diagnostics, flags_}, p, {48, 8, 1}, {512, 1, 1});
    graph.add("private_gdn_wy_telemetry_replay", {flags_, telemetry_}, telemetry::Params{},
        {1, 1, 1}, {32, 1, 1});
    graph.add("flash_gdn_output", {b.recurrentRows, b.z, w.norm->buffer, b.output, b.diagnostics},
        p, {48, rows, 1}, {32, 1, 1});
    graph.add("flash_gdn_convolution_carry", {b.qkv, s.convolution, b.diagnostics},
        p, {40, 1, 1}, {256, 1, 1});
    ++counters_.encodedCalls; counters_.encodedRows += rows;
  }
private:
  metal::MetalBuffer arena_, coefficients_, snapshot_, flags_, telemetry_;
  Counters counters_;
};

} // namespace splash::flash::gdn_wy_hc_sep21
