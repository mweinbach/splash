#pragma once

#include "engine/NativeRuntime.hpp"
#include "metal/ProfilingJson.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace splash::engine {
struct NativePrefillTraceOptions {
  std::string file;
  metal::CommandDispatchProfilingMode gpuMode =
      metal::CommandDispatchProfilingMode::Command;
  bool validMode = true;

  [[nodiscard]] static NativePrefillTraceOptions fromEnvironment() {
    NativePrefillTraceOptions result;
    const char *file = std::getenv("SPLASH_PREFILL_TRACE_FILE");
    // An inherited GPU-mode preference alone must never enable profiling.
    if (!file || !*file) return result;
    result.file = file;
    const char *mode = std::getenv("SPLASH_PREFILL_TRACE_GPU_MODE");
    if (!mode || !*mode || std::string_view(mode) == "command") return result;
    if (std::string_view(mode) == "off")
      result.gpuMode = metal::CommandDispatchProfilingMode::Off;
    else if (std::string_view(mode) == "dispatch")
      result.gpuMode = metal::CommandDispatchProfilingMode::DispatchBoundary;
    else if (std::string_view(mode) == "stage")
      result.gpuMode = metal::CommandDispatchProfilingMode::StagePerDispatch;
    else result.validMode = false;
    return result;
  }
};

// Development-only opt-in sink on the native host thread. One JSONL event is
// appended after each completed prefill, joining model/command sequences.
// Failure or a bounded-file limit disables tracing without affecting inference.
// The hook does not create, submit, wait for, or replay GPU commands.
class NativePrefillTrace final {
public:
  static constexpr uint64_t kMaximumFileBytes = 64ULL * 1024 * 1024;
  static constexpr uint64_t kMaximumLineBytes = 8ULL * 1024 * 1024;
  static constexpr uint64_t kStatusReserveBytes = 4096;

  NativePrefillTrace(NativePrefillTraceOptions options,
                    model::RuntimeModel &model, metal::MetalBackend &backend)
      : options_(std::move(options)), model_(model), backend_(backend) {
    if (options_.file.empty()) return;
    try {
      fd_ = ::open(options_.file.c_str(),
                   O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
      if (fd_ < 0) { diagnostic("open_failed"); return; }
      struct stat info {};
      if (::fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode)) {
        // In particular, never write JSON into /dev/stdout or a protocol pipe.
        ::close(fd_);fd_=-1;
        diagnostic("invalid_file"); return;
      }
      if (info.st_size<0 || static_cast<uint64_t>(info.st_size)>
          kMaximumFileBytes-kStatusReserveBytes) {
        status("file_limit_reached", "existing append budget exhausted");
        diagnostic("file_limit_reached");return;
      }
      if (!options_.validMode) {
        status("invalid_mode", "GPU mode must be off, command, dispatch or stage");
        diagnostic("invalid_mode"); return;
      }
      capability_ = backend_.commandDispatchProfilingCapability();
      model_.setModelPhaseProfiling(true);
      modelEnabled_ = true;
      // The backend retains an Unsupported record for an unsupported exact
      // request. Never substitute command/dispatch/stage for another mode.
      backend_.setCommandDispatchProfiling(options_.gpuMode);
      backendEnabled_ = true;
      enabled_ = true;
      (void)model_.takeModelPhaseProfiles();
      (void)backend_.takeCommandDispatchProfiles();
      if (!status(capability_.supports(options_.gpuMode) ? "enabled" : "unsupported",
                  capability_.supports(options_.gpuMode) ? "" : capability_.reason))
        stop();
    } catch (...) {
      diagnostic("initialization_failed");
      status("initialization_failed", "optional trace setup failed");
      stop();
    }
  }
  ~NativePrefillTrace() {
    stop();
    if (fd_ >= 0) ::close(fd_);
  }
  NativePrefillTrace(const NativePrefillTrace &) = delete;
  NativePrefillTrace &operator=(const NativePrefillTrace &) = delete;

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  void completed(const NativePrefillCompletedMetadata &event) noexcept {
    if (!enabled_) return;
    try {
      auto phases = model_.takeModelPhaseProfiles();
      auto commands = backend_.takeCommandDispatchProfiles();
      // The backend profile mode is global. Drain on every batch so intervening
      // decode commands cannot overflow its eight-record queue. Decode records
      // are discarded; only completed-prefill correlation is written.
      if (event.kind != WorkKind::Prefill) return;
      std::unordered_set<uint64_t> sequences;
      for (const auto &phase : phases)
        if (phase.kind == WorkKind::Prefill && phase.commandSequence)
          sequences.insert(phase.commandSequence);
      size_t unrelated = 0;
      for (const auto &command : commands)
        unrelated += !sequences.contains(command.sequence);
      double completionEnd=0.0;
      bool completionFound=false;
      for (const auto &phase : phases) {
        if (phase.kind == WorkKind::Prefill &&
            phase.stage == model::ModelPhaseStage::CompletionHost) {
          completionEnd=std::max(completionEnd,phase.endedSteadySeconds);
          completionFound=true;
        }
      }
      const auto telemetry = model_.telemetry();
      std::ostringstream out;
      profiling::ScopedJsonFormat format(out);
      out << "{\"schema_version\":1,\"event\":\"prefill_completed\",\"process_id\":"
          << ::getpid() << ",\"event_index\":" << ++eventIndex_
          << ",\"gpu_mode_requested\":";
      profiling::writeString(out,metal::commandDispatchProfilingModeName(options_.gpuMode));
      out << ",\"gpu_request_status\":";
      profiling::writeString(out,options_.gpuMode == metal::CommandDispatchProfilingMode::Off
          ? "off" : capability_.supports(options_.gpuMode) ? "enabled" : "unsupported");
      out << ",\"observed_steady_seconds\":";profiling::writeNumber(out,event.observedSteadySeconds);
      out << ",\"completion_observation_scope\":\"after_engine_cache_publication; gap includes host work and possible state-copy waits\""
          << ",\"completion_gap_valid\":";
      const bool gapValid=completionFound && event.observedSteadySeconds>=completionEnd;
      profiling::writeBoolean(out,gapValid);
      out << ",\"completion_gap_seconds\":";
      if(gapValid)profiling::writeNumber(out,event.observedSteadySeconds-completionEnd);
      else out << "null";
      out << ",\"batch\":{\"width\":" << event.width << ",\"input_rows\":" << event.inputRows
          << ",\"model_wall_seconds\":";
      profiling::writeNumber(out,event.modelWallMilliseconds / 1000.0);
      out << ",\"fused_gpu_seconds\":";profiling::writeNumber(out,telemetry.lastPrefillGpuSeconds);
      out << ",\"fused_wall_seconds\":";profiling::writeNumber(out,telemetry.lastPrefillWallSeconds);
      out << "},\"model_profiles_dropped\":" << model_.modelPhaseProfilesDropped()
          << ",\"unrelated_command_profiles_discarded\":" << unrelated
          << ",\"phases\":[";
      bool separator=false;
      for (const auto &phase : phases) {
        if (phase.kind != WorkKind::Prefill) continue;
        if (separator) out << ',';separator=true;profiling::writeJson(out,phase);
      }
      out << "],\"command_profiles\":[";separator=false;
      size_t correlated=0;
      for (const auto &command : commands) {
        if (!sequences.contains(command.sequence)) continue;
        if (separator) out << ',';separator=true;profiling::writeJson(out,command);++correlated;
      }
      out << "],\"prefill_sequences_total\":" << sequences.size()
          << ",\"correlated_command_profiles\":" << correlated
          << ",\"command_profiles_missing\":";
      profiling::writeBoolean(out,options_.gpuMode != metal::CommandDispatchProfilingMode::Off &&
          correlated < sequences.size());
      out << '}';
      if (!append(out.str())) stop();
    } catch (...) {
      diagnostic("collection_failed");
      status("collection_failed", "optional metadata collection failed");
      stop();
    }
  }

private:
  void stop() noexcept {
    enabled_=false;
    if (modelEnabled_) {
      try { model_.setModelPhaseProfiling(false); } catch (...) {}
      modelEnabled_=false;
    }
    if (backendEnabled_) {
      try { backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off); }
      catch (...) {}
      backendEnabled_=false;
    }
  }
  static void diagnostic(std::string_view status) noexcept {
    // Never use stdout: it carries the native binary protocol.
    constexpr std::string_view prefix="prefill_trace_status: ";
    (void)::write(STDERR_FILENO,prefix.data(),prefix.size());
    (void)::write(STDERR_FILENO,status.data(),status.size());
    (void)::write(STDERR_FILENO,"\n",1);
  }
  bool status(std::string_view value,std::string_view reason) noexcept {
    if (fd_<0) return false;
    try {
      std::ostringstream out;
      profiling::ScopedJsonFormat format(out);
      out << "{\"schema_version\":1,\"event\":\"prefill_trace_status\",\"process_id\":"
          << ::getpid() << ",\"status\":";profiling::writeString(out,value);
      out << ",\"reason\":";profiling::writeString(out,reason);
      out << ",\"gpu_mode_requested\":";
      profiling::writeString(out,options_.validMode ? metal::commandDispatchProfilingModeName(options_.gpuMode) : "invalid");
      out << ",\"file_limit_bytes\":" << kMaximumFileBytes
          << ",\"line_limit_bytes\":" << kMaximumLineBytes
          << ",\"host_clock\":\"steady_seconds\",\"counter_calibration_clock\":\"metal_cpu_nanoseconds\",\"capability\":";
      profiling::writeJson(out,capability_);out << '}';
      return append(out.str(),true);
    } catch (...) { return false; }
  }
  bool append(std::string line,bool statusOnly=false) {
    if (fd_<0) return false;
    if (line.size()>kMaximumLineBytes) {
      diagnostic("line_limit_reached");
      if(!statusOnly)status("line_limit_reached", "one metadata event exceeded the bounded line limit");
      return false;
    }
    line.push_back('\n');
    struct stat info {};
    if (::fstat(fd_,&info)!=0 || info.st_size<0) { diagnostic("stat_failed");return false; }
    const uint64_t used=static_cast<uint64_t>(info.st_size);
    const uint64_t reserve=statusOnly?0:kStatusReserveBytes;
    if(used>kMaximumFileBytes || line.size()+reserve>kMaximumFileBytes-used) {
      diagnostic("file_limit_reached");
      if(!statusOnly)status("file_limit_reached", "bounded append budget exhausted; tracing disabled");
      return false;
    }
    size_t offset=0;
    while(offset<line.size()) {
      const auto written=::write(fd_,line.data()+offset,line.size()-offset);
      if(written<0&&errno==EINTR)continue;
      if(written<=0) { diagnostic("write_failed");return false; }
      offset+=static_cast<size_t>(written);
    }
    return true;
  }
  NativePrefillTraceOptions options_;
  model::RuntimeModel &model_;
  metal::MetalBackend &backend_;
  metal::CommandDispatchProfilingCapability capability_;
  int fd_=-1;
  bool enabled_=false,modelEnabled_=false,backendEnabled_=false;
  uint64_t eventIndex_=0;
};
} // namespace splash::engine
