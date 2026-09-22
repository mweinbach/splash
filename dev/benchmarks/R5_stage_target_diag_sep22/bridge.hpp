#pragma once

#include "engine/MemoryGovernor.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/ProfilingJson.hpp"
#include "metadata.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

// Private metadata-only diagnostic. It neither submits commands nor reads any
// tensor/weight/request payload. The genuine Worker owns Diagnostic and creates
// the typed cookie immediately around its existing synchronous main-target call.
namespace splash::flash::r5_stage_target_diag_sep22 {
inline constexpr const char *kFlag = "SPLASH_FLASH_DIAG_R5_TARGET_STAGE_SEP22";
inline constexpr const char *kOutput = "SPLASH_FLASH_DIAG_R5_STAGE_OUTPUT_SEP22";
inline constexpr uint64_t kReservationBytes = 64ULL << 20;
inline constexpr size_t kMaximumDispatches = 4096, kMaximumBindings = 32;
inline constexpr size_t kMaximumPipelineBytes = 256, kMaximumReasonBytes = 1024;
inline constexpr size_t kMaximumJsonBytes = 16ULL << 20;

inline std::string environment(const char *name, const char *fallback) {
  const char *value = std::getenv(name); return value ? value : fallback;
}
inline bool strictFlag(std::string_view value) {
  if (value == "0") return false;
  if (value == "1") return true;
  throw std::invalid_argument("R5 stage diagnostic flag must be canonical 0/1");
}
inline uint64_t positiveCanonical(std::string_view text, const char *name) {
  uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || text.front() == '0' || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size() || !value || std::to_string(value) != text)
    throw std::invalid_argument(std::string(name) + " must be a positive canonical integer");
  return value;
}
struct Config final {
  bool enabled = false;
  uint64_t requestId = 1, generation = 1, r5Ordinal = 3, maximumSamples = 1;
  std::string outputPath;
  std::array<std::string, 8> frozen;
  static Config readEnvironment() {
    Config c;
    c.frozen = {environment(kFlag, "0"), environment(kOutput, ""),
        environment("SPLASH_FLASH_DIAG_R5_STAGE_REQUEST_ID_SEP22", "1"),
        environment("SPLASH_FLASH_DIAG_R5_STAGE_GENERATION_SEP22", "1"),
        environment("SPLASH_FLASH_DIAG_R5_STAGE_R5_ORDINAL_SEP22", "3"),
        environment("SPLASH_FLASH_DIAG_R5_STAGE_MAX_SAMPLES_SEP22", "1"),
        environment("SPLASH_FLASH_MTP_DRAFT_DEPTH", ""),
        environment("SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22", "0")};
    c.enabled = strictFlag(c.frozen[0]);
    // Flag0 adds no path/configuration dependency to the original Worker.
    if (!c.enabled) return c;
    c.requestId = positiveCanonical(c.frozen[2], "diagnostic request ID");
    c.generation = positiveCanonical(c.frozen[3], "diagnostic generation");
    c.r5Ordinal = positiveCanonical(c.frozen[4], "diagnostic R5 ordinal");
    c.maximumSamples = positiveCanonical(c.frozen[5], "diagnostic maximum samples");
    if (c.r5Ordinal != 3 || c.maximumSamples != 1 || c.frozen[6] != "4" ||
        c.frozen[7] != "1")
      throw std::invalid_argument("R5 stage diagnostic requires fixed depth4/R5flag1, ordinal3/maxsamples1");
    c.outputPath = c.frozen[1];
    if (c.outputPath.empty() || c.outputPath.size() > 4096 ||
        !std::filesystem::path(c.outputPath).is_absolute())
      throw std::invalid_argument("R5 stage diagnostic requires a fresh absolute output path");
    return c;
  }
  static Config fromEnvironment() {
    const Config now = readEnvironment();
    static const Config frozenConfig = now;
    if (now.enabled != frozenConfig.enabled ||
        (now.enabled && now.frozen != frozenConfig.frozen))
      throw std::logic_error("R5 stage diagnostic configuration changed after admission");
    return frozenConfig;
  }
  void requireFrozen() const {
    const Config now = fromEnvironment();
    if (enabled != now.enabled || (enabled && frozen != now.frozen))
      throw std::logic_error("R5 stage diagnostic instance configuration changed");
  }
};
inline void validateStartup() { (void)Config::fromEnvironment(); }

struct GraphDispatch final {
  std::string pipeline;
  metal::DispatchSize groups, threads;
  std::vector<metal::DispatchProfileBinding> bindings;
  // Recognized, fixed numeric inline-ABI metadata only. The separate private
  // decoder supplies a JSON value; unrecognized parameter layouts stay null.
  std::string parameters = "null";
};
class ScopedVerifyCookie;
inline thread_local ScopedVerifyCookie *currentContext = nullptr;

inline void writeMemory(std::ostream &out, const metal::MetalMemoryStats &m) {
  out << "{\"owner_allocated_bytes\":" << m.allocatedBytes
      << ",\"owner_peak_allocated_bytes\":" << m.peakAllocatedBytes
      << ",\"sampled_device_current_allocated_bytes\":" << m.deviceCurrentAllocatedBytes
      << ",\"sampled_device_peak_allocated_bytes\":" << m.devicePeakAllocatedBytes
      << ",\"sparse_virtual_bytes\":" << m.sparseVirtualBytes
      << ",\"sparse_resident_bytes\":" << m.sparseResidentBytes
      << ",\"peak_sparse_resident_bytes\":" << m.peakSparseResidentBytes
      << ",\"peak_resident_bytes\":" << m.peakResidentBytes
      << ",\"pending_sparse_unmaps\":" << m.pendingSparseUnmaps
      << ",\"device_values_are_sampled_process_counters\":true"
      << ",\"vendor_counter_backing_bytes\":null,\"vendor_counter_backing_known\":false}";
}
inline void writeGovernor(std::ostream &out, const engine::MemoryGovernorSnapshot &g) {
  out << "{\"limit_bytes\":" << g.limitBytes << ",\"observed_resident_bytes\":" << g.observedResidentBytes
      << ",\"reserved_bytes\":" << g.reservedBytes << ",\"headroom_bytes\":" << g.headroomBytes
      << ",\"pressure\":" << unsigned(g.pressure) << ",\"denied_reservations\":" << g.deniedReservations
      << ",\"host_measurement_valid\":" << (g.hostMeasurementValid ? "true" : "false")
      << ",\"host_available_bytes\":" << g.hostAvailableBytes << ",\"host_reserve_bytes\":" << g.hostReserveBytes
      << ",\"host_headroom_bytes\":" << g.hostHeadroomBytes << ",\"system_pressure\":" << unsigned(g.systemPressure)
      << ",\"growth_allowed\":" << (g.growthAllowed ? "true" : "false") << '}';
}

class Diagnostic final {
public:
  explicit Diagnostic(Config config) : config_(std::move(config)) {
    config_.requireFrozen();
    if (!config_.enabled) return;
    fd_ = ::open(config_.outputPath.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd_ < 0) throw std::runtime_error("cannot create fresh R5 stage diagnostic output");
  }
  ~Diagnostic() { if (fd_ >= 0) ::close(fd_); }
  Diagnostic(const Diagnostic &) = delete;
  Diagnostic &operator=(const Diagnostic &) = delete;
  [[nodiscard]] bool enabled() const noexcept { return config_.enabled; }
  [[nodiscard]] bool outputWritten() const noexcept { return outputWritten_; }
  [[nodiscard]] bool outputWriteFailed() const noexcept { return outputWriteFailed_; }
  [[nodiscard]] uint64_t completedR5Calls() const noexcept { return completedR5_; }
  [[nodiscard]] const Config &config() const noexcept { return config_; }
private:
  bool emit(std::string_view bytes) noexcept {
    if (fd_ < 0 || outputWritten_ || bytes.size() > kMaximumJsonBytes) { outputWriteFailed_ = true; return false; }
    size_t written = 0;
    while (written < bytes.size()) {
      const auto count = ::write(fd_, bytes.data() + written, bytes.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) { outputWriteFailed_ = true; return false; }
      written += size_t(count);
    }
    outputWritten_ = true; return true;
  }
  Config config_;
  int fd_ = -1;
  uint64_t completedR5_ = 0, samplesAttempted_ = 0;
  bool outputWritten_ = false, outputWriteFailed_ = false;
  friend class ScopedVerifyCookie;
};

// The cookie's identity is supplied by the real singleton Worker, not inferred
// from dispatch names. Head/batch/AR/prefill do not create this target cookie.
class ScopedVerifyCookie final {
public:
  ScopedVerifyCookie(Diagnostic &owner, uint64_t id, uint64_t generation,
      uint32_t depth, uint64_t physicalRows, engine::MemoryGovernor &governor,
      metal::MetalBackend &backend) noexcept
      : owner_(owner), id_(id), generation_(generation), depth_(depth), rows_(physicalRows),
        governor_(governor), backend_(backend) {
    if (!owner.enabled() || id != owner.config_.requestId || generation != owner.config_.generation ||
        depth != 4 || physicalRows != 5 || owner.samplesAttempted_) return;
    if (currentContext) return; // Never replace another owner's typed scope.
    eligible_ = true; ordinal_ = owner.completedR5_ + 1;
    selected_ = ordinal_ == owner.config_.r5Ordinal;
    currentContext = this;
    if (!selected_) return;
    ++owner.samplesAttempted_;
    try {
      owner.config_.requireFrozen();
      // Setter itself enforces no outstanding ticket. Only this genuine Worker
      // safe point and post-synchronous completion change the stage selector.
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
      restoredOff_ = true;
      const auto stale = backend_.takeCommandDispatchProfiles();
      staleProfiles_ = stale.size();
      if (!stale.empty()) { fail("stale command profiles were present at activation"); return; }
      capability_ = backend_.commandDispatchProfilingCapability();
      beforeMemory_ = backend_.memoryStats(); beforeGovernor_ = governor_.snapshot();
      beforeSnapshotCaptured_ = true;
      if (!capability_.supports(metal::CommandDispatchProfilingMode::StagePerDispatch)) {
        fail("StagePerDispatch unsupported; target remains unprofiled"); return;
      }
      metal::AllocationFailure refusal = metal::AllocationFailure::None;
      reservation_ = governor_.tryReserve(kReservationBytes, &refusal);
      if (!reservation_) { fail(std::string("diagnostic reservation refused: ") + metal::allocationFailureName(refusal)); return; }
      heldGovernor_ = governor_.snapshot();
      heldSnapshotCaptured_ = true;
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::StagePerDispatch);
      active_ = true; restoredOff_ = false;
    } catch (const std::exception &e) { fail(e.what()); restoreOff(); }
      catch (...) { fail("unknown diagnostic activation failure"); restoreOff(); }
  }
  ~ScopedVerifyCookie() {
    if (!finished_ && selected_) { fail("target scope ended without synchronous complete"); completeImpl(false); }
    if (currentContext == this) currentContext = nullptr;
  }
  ScopedVerifyCookie(const ScopedVerifyCookie &) = delete;
  ScopedVerifyCookie &operator=(const ScopedVerifyCookie &) = delete;
  [[nodiscard]] bool selected() const noexcept { return selected_; }
  bool complete() noexcept { return completeImpl(true); }

  void record(const metal::CommandGraph &graph, bool verification, uint64_t rows, uint64_t begin) noexcept {
    if (!eligible_ || finished_) return;
    ++graphCalls_;
    if (graphCalls_ == 1) {
      graphVerification_ = verification; graphRows_ = rows; graphBegin_ = begin;
    }
    if (graphCalls_ != 1 || !verification || rows != 5 || rows != rows_) {
      fail("typed target scope received excluded/multiple Forward graph"); restoreOff(); return;
    }
    if (!selected_) return;
    try {
      const auto dispatches = graph.dispatches();
      graphCount_ = dispatches.size();
      if (dispatches.empty() || dispatches.size() > kMaximumDispatches) {
        fail("Forward graph dispatch count is outside 1..4096"); restoreOff(); return;
      }
      graph_.reserve(dispatches.size());
      for (const auto &d : dispatches) {
        if (d.pipelineName.size() > kMaximumPipelineBytes || d.buffers.size() + d.bytes.size() > kMaximumBindings) {
          fail("Forward graph exceeds bounded pipeline/binding metadata limits"); restoreOff(); graph_.clear(); return;
        }
        GraphDispatch m{d.pipelineName, d.threadgroups, d.threadsPerThreadgroup, {}, "null"};
        m.parameters = decodeParameters(d);
        m.bindings.reserve(d.buffers.size() + d.bytes.size());
        for (const auto &b : d.buffers) m.bindings.push_back({b.index, b.buffer.sizeBytes(), false});
        // Inline payload pointers are deliberately never dereferenced. ABI
        // lengths/index are sufficient fixed inline metadata for this probe.
        for (const auto &b : d.bytes) m.bindings.push_back({b.index, b.sizeBytes, true});
        graph_.push_back(std::move(m));
      }
    } catch (const std::exception &e) { fail(e.what()); restoreOff(); graph_.clear(); }
      catch (...) { fail("graph metadata recording failed"); restoreOff(); graph_.clear(); }
  }
private:
  void fail(std::string_view reason) noexcept {
    invalid_ = true;
    if (!reason_.empty()) return;
    try { reason_.assign(reason.substr(0, kMaximumReasonBytes)); } catch (...) {}
  }
  void restoreOff() noexcept {
    if (!active_) return;
    try {
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
      active_ = false; restoredOff_ = true;
    } catch (...) { fail("could not restore profiling Off at the safe point"); }
  }
  static bool sameSize(const metal::DispatchSize &a, const metal::DispatchSize &b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
  bool validProfile(const metal::CommandDispatchProfile &p) const noexcept {
    if (p.status != metal::CommandDispatchProfileStatus::Complete ||
        p.mode != metal::CommandDispatchProfilingMode::StagePerDispatch ||
        !p.encoderBoundariesAltered || p.samplingBarriers || p.droppedProfilesBefore ||
        p.dispatchMetadataTruncated || p.dispatchCount != graphCount_ ||
        p.dispatches.size() != graph_.size() || graph_.size() != graphCount_) return false;
    const std::array<double, 18> times{p.timing.gpuSeconds, p.timing.wallSeconds,
        p.commandGpuStartSeconds, p.commandGpuEndSeconds, p.commandKernelStartSeconds,
        p.commandKernelEndSeconds, p.hostPreparationSeconds, p.hostEncodingSeconds,
        p.sparseDependencyWaitSeconds, p.hostCommitSeconds, p.hostSubmissionStartSeconds,
        p.hostEncodingStartSeconds, p.hostEncodingEndSeconds, p.hostCommitBeginSeconds,
        p.hostCommitEndSeconds, p.hostScheduledSeconds, p.hostCompletedSeconds, p.hostReadySeconds};
    if (!std::all_of(times.begin(), times.end(), [](double v) { return std::isfinite(v) && v >= 0; }) ||
        p.commandGpuStartSeconds <= 0 || p.commandGpuEndSeconds < p.commandGpuStartSeconds) return false;
    const auto &h = p.timing.host;
    const std::array<double, 14> hostTimes{h.preparationSeconds, h.encodingSeconds,
        h.beforeCommitSeconds, h.dependencyWaitSeconds, h.commitSeconds,
        h.commitToScheduledCallbackSeconds, h.commitToCompletedCallbackSeconds,
        h.completionCallbackBeforeWallEndSeconds, h.submissionReturnSeconds,
        h.ticketBlockingWaitSeconds, h.preCommitMemorySampleSeconds,
        h.postCommitMemorySampleSeconds, h.scheduledMemorySampleSeconds, h.completedMemorySampleSeconds};
    if (!std::all_of(hostTimes.begin(), hostTimes.end(), [](double v) { return std::isfinite(v); })) return false;
    for (const auto *sample : {&p.preCommitDeviceMemorySample, &p.postCommitDeviceMemorySample,
                              &p.scheduledDeviceMemorySample, &p.completedDeviceMemorySample})
      if (!std::isfinite(sample->beganSteadySeconds) || !std::isfinite(sample->endedSteadySeconds) ||
          !std::isfinite(sample->seconds)) return false;
    for (const auto *bridge : {&p.commitClockBridge, &p.completedClockBridge})
      if (!std::isfinite(bridge->beganSteadySeconds) || !std::isfinite(bridge->endedSteadySeconds) ||
          !std::isfinite(bridge->machSeconds) || !std::isfinite(bridge->steadyMinusMachSeconds) ||
          !std::isfinite(bridge->uncertaintySeconds)) return false;
    for (size_t i = 0; i < graph_.size(); ++i) {
      const auto &d = p.dispatches[i]; const auto &g = graph_[i];
      if (d.index != i || d.pipelineName != g.pipeline || !sameSize(d.threadgroups, g.groups) ||
          !sameSize(d.threadsPerThreadgroup, g.threads) || d.bindings.size() != g.bindings.size() ||
          d.bindings.size() > kMaximumBindings || !d.timestampsValid || !d.gpuStartTimestamp ||
          d.gpuEndTimestamp < d.gpuStartTimestamp || !std::isfinite(d.calibratedStartSeconds) ||
          !std::isfinite(d.calibratedEndSeconds) || !std::isfinite(d.gpuSeconds) ||
          d.calibratedStartSeconds < 0 || d.calibratedEndSeconds < d.calibratedStartSeconds || d.gpuSeconds < 0) return false;
      for (size_t j = 0; j < g.bindings.size(); ++j)
        if (d.bindings[j].index != g.bindings[j].index || d.bindings[j].sizeBytes != g.bindings[j].sizeBytes ||
            d.bindings[j].inlineBytes != g.bindings[j].inlineBytes) return false;
    }
    return true;
  }
  bool completeImpl(bool successfulCall) noexcept {
    if (finished_) return !invalid_;
    finished_ = true;
    if (currentContext == this) currentContext = nullptr;
    if (!eligible_) return true;
    if (successfulCall && graphCalls_ == 1 && graphRows_ == 5 && !invalid_) ++owner_.completedR5_;
    if (!selected_) return !invalid_;
    // This runs after the original synchronous target command returns. Off is
    // restored before any commit or trained Head step and before file writing.
    restoreOff();
    try {
      afterCommandMemory_ = backend_.memoryStats(); afterCommandGovernor_ = governor_.snapshot();
      afterCommandSnapshotCaptured_ = true;
      auto profiles = backend_.takeCommandDispatchProfiles();
      profileCount_ = profiles.size();
      if (!successfulCall || graphCalls_ != 1 || graphRows_ != 5) fail("successful genuine R5 target command was not established");
      if (!restoredOff_) fail("profiling Off restoration was not established");
      if (profiles.size() != 1) fail("expected exactly one whole-command stage profile");
      else {
        profile_ = std::move(profiles.front());
        if (!validProfile(*profile_)) fail("stage profile failed complete/mode/flags/count/metadata/timestamp checks");
        if (profile_->reason.size() > kMaximumReasonBytes) {
          fail("profile reason exceeds the diagnostic bound"); profile_->reason.resize(kMaximumReasonBytes);
        }
      }
      reservation_.reset();
      afterReleaseGovernor_ = governor_.snapshot(); afterReleaseMemory_ = backend_.memoryStats();
      afterReleaseSnapshotCaptured_ = true;
      publish();
    } catch (const std::exception &e) {
      fail(e.what()); reservation_.reset();
      try { afterReleaseGovernor_ = governor_.snapshot(); afterReleaseMemory_ = backend_.memoryStats();
        afterReleaseSnapshotCaptured_ = true; publish(); } catch (...) {}
    } catch (...) {
      fail("diagnostic collection/write failed"); reservation_.reset();
      try { publish(); } catch (...) {}
    }
    return !invalid_;
  }
  void publish() {
    std::ostringstream out; profiling::ScopedJsonFormat format(out);
    out << "{\"schema\":\"private-R5-target-stage-diagnostic-sep22-v1\",\"diagnostic_valid\":"
        << (invalid_ ? "false" : "true") << ",\"reason\":" << json::quote(reason_)
        << ",\"instrumentation_is_performance_perturbation\":true,\"throughput_baseline\":false"
        << ",\"single_whole_target_command\":true,\"legacy_dispatch_replay\":false"
        << ",\"request_id\":" << id_ << ",\"generation\":" << generation_ << ",\"draft_depth\":" << depth_
        << ",\"physical_rows\":" << rows_ << ",\"R5_ordinal\":" << ordinal_
        << ",\"prior_completed_unprofiled_R5_calls\":" << (ordinal_ ? ordinal_ - 1 : 0)
        << ",\"maximum_samples\":1,\"samples_attempted\":" << owner_.samplesAttempted_
        << ",\"profiling_restored_off\":" << (restoredOff_ ? "true" : "false")
        << ",\"stale_profiles_at_activation\":" << staleProfiles_ << ",\"profiles_after_target\":" << profileCount_
        << ",\"governor_reservation_bytes\":" << kReservationBytes
        << ",\"reservation_admitted\":" << (heldSnapshotCaptured_ ? "true" : "false")
        << ",\"reservation_released\":" << (!reservation_ ? "true" : "false")
        << ",\"backend_destroyed\":null,\"backend_destruction_proved\":false"
        << ",\"tensor_payload_reads\":0,\"tensor_payload_hashes\":0,\"inline_abi_numeric_metadata_only\":true"
        << ",\"capability\":"; profiling::writeJson(out, capability_);
    out << ",\"memory_snapshots\":{\"before_activation\":";
    if (beforeSnapshotCaptured_) writeMemory(out, beforeMemory_); else out << "null";
    out << ",\"after_target\":";
    if (afterCommandSnapshotCaptured_) writeMemory(out, afterCommandMemory_); else out << "null";
    out << ",\"after_reservation_release\":";
    if (afterReleaseSnapshotCaptured_) writeMemory(out, afterReleaseMemory_); else out << "null"; out << '}';
    out << ",\"governor_snapshots\":{\"before_activation\":";
    if (beforeSnapshotCaptured_) writeGovernor(out, beforeGovernor_); else out << "null";
    out << ",\"reservation_held\":";
    if (heldSnapshotCaptured_) writeGovernor(out, heldGovernor_); else out << "null";
    out << ",\"after_target\":";
    if (afterCommandSnapshotCaptured_) writeGovernor(out, afterCommandGovernor_); else out << "null";
    out << ",\"after_reservation_release\":";
    if (afterReleaseSnapshotCaptured_) writeGovernor(out, afterReleaseGovernor_); else out << "null"; out << '}';
    out << ",\"Forward_graph\":{\"calls\":" << graphCalls_ << ",\"verification\":"
        << (graphVerification_ ? "true" : "false") << ",\"rows\":" << graphRows_
        << ",\"begin\":" << graphBegin_ << ",\"dispatch_count\":" << graphCount_
        << ",\"metadata_count\":" << graph_.size() << ",\"metadata_truncated\":"
        << (graph_.size() != graphCount_ ? "true" : "false") << ",\"dispatches\":[";
    for (size_t i = 0; i < graph_.size(); ++i) {
      if (i) out << ','; const auto &g = graph_[i];
      out << "{\"index\":" << i << ",\"pipeline\":" << json::quote(g.pipeline) << ",\"threadgroups\":";
      profiling::writeJson(out, g.groups); out << ",\"threads_per_threadgroup\":"; profiling::writeJson(out, g.threads);
      out << ",\"parameters\":" << g.parameters << ",\"bindings\":[";
      for (size_t j = 0; j < g.bindings.size(); ++j) {
        if (j) out << ','; const auto &b = g.bindings[j];
        out << "{\"index\":" << b.index << ",\"size_bytes\":" << b.sizeBytes
            << ",\"inline_bytes\":" << (b.inlineBytes ? "true" : "false") << '}';
      }
      out << "]}";
    }
    out << "]},\"raw_command_profile\":";
    if (profile_) profiling::writeJson(out, *profile_); else out << "null";
    out << "}\n";
    auto bytes = out.str();
    if (bytes.size() > kMaximumJsonBytes) {
      fail("bounded diagnostic JSON exceeded its size limit");
      bytes = "{\"schema\":\"private-R5-target-stage-diagnostic-sep22-v1\",\"diagnostic_valid\":false,"
          "\"reason\":\"bounded diagnostic JSON exceeded its size limit\",\"payload_reads\":0,"
          "\"profiling_restored_off\":" + std::string(restoredOff_ ? "true" : "false") +
          ",\"backend_destroyed\":null,\"backend_destruction_proved\":false}\n";
    }
    if (!owner_.emit(bytes)) fail("bounded fresh diagnostic output write failed");
  }
  Diagnostic &owner_;
  uint64_t id_, generation_; uint32_t depth_; uint64_t rows_;
  engine::MemoryGovernor &governor_; metal::MetalBackend &backend_;
  bool eligible_ = false, selected_ = false, active_ = false, restoredOff_ = false;
  bool finished_ = false, invalid_ = false, graphVerification_ = false;
  bool beforeSnapshotCaptured_ = false, heldSnapshotCaptured_ = false;
  bool afterCommandSnapshotCaptured_ = false, afterReleaseSnapshotCaptured_ = false;
  uint64_t ordinal_ = 0, graphCalls_ = 0, graphRows_ = 0, graphBegin_ = 0;
  size_t graphCount_ = 0, staleProfiles_ = 0, profileCount_ = 0;
  std::string reason_;
  std::optional<engine::MemoryGovernor::Reservation> reservation_;
  metal::CommandDispatchProfilingCapability capability_;
  metal::MetalMemoryStats beforeMemory_, afterCommandMemory_, afterReleaseMemory_;
  engine::MemoryGovernorSnapshot beforeGovernor_, heldGovernor_, afterCommandGovernor_, afterReleaseGovernor_;
  std::vector<GraphDispatch> graph_;
  std::optional<metal::CommandDispatchProfile> profile_;
};
inline void recordGraph(const metal::CommandGraph &graph, bool verification,
    uint64_t rows, uint64_t begin) noexcept {
  if (currentContext) currentContext->record(graph, verification, rows, begin);
}
} // namespace splash::flash::r5_stage_target_diag_sep22
