#pragma once
// Optional CPU metadata only. Native clocks are captured by the original
// emission/Done code; this sink submits no GPU work and changes no wire frame.
#include "engine/Json.hpp"
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace splash::flash::batch_clock_sep22 {
class Trace final {
public:
  static std::unique_ptr<Trace> fromEnvironment() {
    const char *flag = std::getenv("SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22");
    const char *path = std::getenv("SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22");
    if (!flag || std::string_view(flag) == "0") {
      if (path) throw std::invalid_argument("native lifecycle trace path requires timestamp marker=1");
      return {};
    }
    if (std::string_view(flag) != "1")
      throw std::invalid_argument("SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22 must be 0 or 1");
    if (!path) throw std::invalid_argument("native lifecycle timestamps require a fresh trace path");
    if (!*path || !std::filesystem::path(path).is_absolute())
      throw std::invalid_argument("native lifecycle trace requires a fresh absolute local JSONL path");
    return std::unique_ptr<Trace>(new Trace(path));
  }
  ~Trace() { if (fd_ >= 0) ::close(fd_); }
  [[nodiscard]] const std::string &clockDomain() const noexcept { return clockDomain_; }
  [[nodiscard]] uint64_t workerPid() const noexcept { return workerPid_; }
  [[nodiscard]] int64_t workerStartTokenNs() const noexcept { return workerStartTokenNs_; }
  void setSourceIdentity(const std::string &identity) { sourceIdentity_ = identity; }
  void record(const char *event, uint64_t instance, uint64_t request,
      uint64_t generation, int64_t nanoseconds, uint32_t emitted,
      uint32_t prompt, const char *finish = nullptr) {
    if (records_.size() == maximumRecords)
      throw std::runtime_error("native lifecycle metadata record budget exhausted");
    records_.push_back({event, instance, request, generation, nanoseconds, emitted, prompt, finish});
  }
  // Flush only once serving has stopped. Record capture never writes a file
  // in the measured first-emission/Done interval and never reallocates.
  void finish() {
    const uint64_t instance = records_.empty() ? 0 : records_.front().instance;
    for (const auto &record : records_) write(record);
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    write({"stop_flush_complete", instance, 0, 0, now, 0, 0, nullptr});
    records_.clear();
  }
private:
  struct Record final {
    const char *event;
    uint64_t instance, request, generation;
    int64_t nanoseconds;
    uint32_t emitted, prompt;
    const char *finish;
  };
  static constexpr size_t maximumRecords = 4096;
  void write(const Record &record) {
    std::string line = "{\"schema\":\"splash-native-lifecycle-trace-sep22-v1\",\"instrumented_metadata\":true";
    line += ",\"clock_kind\":\"native-worker-std-steady-nanoseconds-v1\",\"clock_domain\":" + json::quote(clockDomain_);
    line += ",\"native_worker_pid\":" + std::to_string(workerPid_) + ",\"native_worker_start_token_ns\":" + std::to_string(workerStartTokenNs_);
    line += ",\"source_identity_sha256\":" + json::quote(sourceIdentity_);
    line += ",\"event\":" + json::quote(record.event) + ",\"instance_id\":" + std::to_string(record.instance);
    line += ",\"request_id\":" + std::to_string(record.request) + ",\"generation\":" + std::to_string(record.generation);
    line += ",\"steady_nanoseconds\":" + std::to_string(record.nanoseconds);
    line += ",\"emitted_tokens\":" + std::to_string(record.emitted) + ",\"prompt_tokens\":" + std::to_string(record.prompt);
    line += ",\"finish_reason\":" + (record.finish ? json::quote(record.finish) : "null") + "}\n";
    if (std::string_view(record.event) == "stop_flush_complete") {
      line.resize(line.size() - 2);
      line += ",\"flushed_lifecycle_records\":" + std::to_string(records_.size()) + "}\n";
    }
    size_t written = 0;
    while (written < line.size()) {
      const auto count = ::write(fd_, line.data() + written, line.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::runtime_error("native lifecycle trace write failed");
      written += static_cast<size_t>(count);
    }
  }
  explicit Trace(const char *path) {
    workerPid_ = static_cast<uint64_t>(::getpid());
    workerStartTokenNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    clockDomain_ = "native-worker-std-steady-nanoseconds-v1:" + std::to_string(workerPid_) +
        ":" + std::to_string(workerStartTokenNs_);
    records_.reserve(maximumRecords);
    fd_ = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd_ < 0) throw std::invalid_argument("native lifecycle trace path is not fresh and writable");
  }
  int fd_ = -1;
  std::string sourceIdentity_;
  uint64_t workerPid_ = 0;
  int64_t workerStartTokenNs_ = 0;
  std::string clockDomain_;
  std::vector<Record> records_;
};
}
