#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/ProfilingJson.hpp"

#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
using namespace splash;
using namespace splash::metal;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kGuard = 0xd75a19c3;
constexpr uint32_t kUnwritten = 0xa823745e;
constexpr uint64_t kExpectedBytes = 106320429056ULL;
constexpr const char *kSource = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
uint32_t word(const MetalBuffer &source) {
  require(source && source.contents() && source.sizeBytes() >= 4,
          "invalid Shared source word");
  uint32_t value = 0;
  std::memcpy(&value, source.contents(), sizeof(value));
  return value;
}
uint32_t integer(std::string_view text, uint32_t maximum) {
  require(!text.empty() && text.find_first_not_of("0123456789") == std::string_view::npos,
          "invalid bounded unsigned integer");
  const auto value = std::stoull(std::string(text));
  require(value <= maximum, "unsigned integer exceeds probe bound");
  return static_cast<uint32_t>(value);
}
std::vector<uint32_t> list(std::string_view text, uint32_t maximum) {
  std::vector<uint32_t> values;
  while (true) {
    const auto comma = text.find(',');
    values.push_back(integer(text.substr(0, comma), maximum));
    require(values.size() <= 5, "at most five idle intervals allowed");
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
  }
  return values;
}
double now() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
void number(std::ostream &out, double value) { profiling::writeNumber(out, value); }

struct Geometry {
  std::string name;
  std::string pipeline;
  std::vector<MetalBuffer> sources;
  std::vector<uint32_t> expected;
  std::vector<uint32_t> originalIndices;
  MetalBuffer arguments;
  MetalBuffer output;
  std::unique_ptr<CommandGraph> graph;
  uint64_t nativeSourceBytes = 0;

  Geometry(MetalBackend &backend, std::string label,
           std::vector<MetalBuffer> bases, std::vector<uint32_t> indices = {})
      : name(std::move(label)), sources(std::move(bases)),
        originalIndices(std::move(indices)) {
    require(sources.size() == 1 || sources.size() == 4 || sources.size() == 21,
            "unsupported readonly reflected layout");
    pipeline = "idle_probe_read" + std::to_string(sources.size());
    std::vector<BufferBinding> bindings;
    for (uint32_t index = 0; index < sources.size(); ++index) {
      const auto &source = sources[index];
      require(source.storage() == BufferStorage::Shared && source.sizeBytes() % 16384 == 0,
              "source is not an aligned Shared native base");
      expected.push_back(word(source));
      bindings.push_back({index, source});
      require(nativeSourceBytes <= UINT64_MAX - source.sizeBytes(), "source byte sum overflows");
      nativeSourceBytes += source.sizeBytes();
    }
    // The backend validates exact reflected count, dense readonly pointers,
    // alignment, and same-backend ownership; it retains and declares only these bases.
    arguments = backend.makeReadOnlyArgumentBuffer(pipeline, 0, bindings, name);
    output = backend.allocateBuffer(32 * sizeof(uint32_t), BufferStorage::Shared, name + " output");
    graph = std::make_unique<CommandGraph>();
    graph->add(pipeline, {arguments, output}, {1, 1, 1}, {32, 1, 1});
  }
  Geometry(Geometry &&) = default;
  Geometry &operator=(Geometry &&) = default;
  Geometry(const Geometry &) = delete;
  void clear() {
    auto *values = static_cast<uint32_t *>(output.contents());
    std::fill_n(values, 32, kGuard);
    std::fill_n(values + 1, sources.size(), kUnwritten);
  }
  void validate() const {
    const auto *values = static_cast<const uint32_t *>(output.contents());
    require(values[0] == kGuard, "leading output guard changed");
    for (size_t index = 0; index < sources.size(); ++index)
      require(values[index + 1] == expected[index], "GPU one-word source read differs from CPU");
    for (size_t index = sources.size() + 1; index < 32; ++index)
      require(values[index] == kGuard, "trailing output guard changed");
  }
};

void geometryJson(std::ostream &out, const Geometry &geometry) {
  out << "{\"name\":" << json::quote(geometry.name)
      << ",\"source_base_count\":" << geometry.sources.size()
      << ",\"native_source_bytes\":" << geometry.nativeSourceBytes
      << ",\"gpu_source_words_read\":" << geometry.sources.size()
      << ",\"argument_buffer_bytes\":" << geometry.arguments.sizeBytes()
      << ",\"output_bytes\":" << geometry.output.sizeBytes()
      << ",\"original_base_indices\":[";
  for (size_t index = 0; index < geometry.originalIndices.size(); ++index) {
    if (index) out << ',';
    out << geometry.originalIndices[index];
  }
  out << "],\"source_base_bytes\":[";
  for (size_t index = 0; index < geometry.sources.size(); ++index) {
    if (index) out << ',';
    out << geometry.sources[index].sizeBytes();
  }
  out << "]}";
}

int cpuSelfTest() {
  uint64_t checks = 0;
  static_assert(sizeof(CommandTiming) == 200);
  require(list("1,3,6,9", 9) == std::vector<uint32_t>({1, 3, 6, 9}), "interval parse differs"); ++checks;
  for (const auto bad : {"", "-1", "10", "1,,3", "1,1,1,1,1,1", "1e0"}) {
    bool rejected = false;
    try { (void)list(bad, 9); } catch (...) { rejected = true; }
    require(rejected, "invalid idle selector accepted"); ++checks;
  }
  for (uint32_t count : {1u, 4u, 21u}) {
    std::array<uint32_t, 32> output{};
    std::fill(output.begin(), output.end(), kGuard);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      if (lane < count) output[lane + 1] = lane;
      ++checks;
    }
    require(output[0] == kGuard, "leading CPU guard changed"); ++checks;
    for (uint32_t lane = 0; lane < count; ++lane) { require(output[lane + 1] == lane, "CPU word address differs"); ++checks; }
    for (uint32_t lane = count + 1; lane < 32; ++lane) { require(output[lane] == kGuard, "trailing CPU guard changed"); ++checks; }
  }
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
            << ",\"gpu_commands\":0,\"command_timing_abi_bytes\":200}\n";
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      static_assert(sizeof(CommandTiming) == 200, "fresh current host objects required");
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") return cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "oracle METALLIB PACKAGE REPORT_JSON [--mode tiny-only|model-tiny|original1|original4|original21|all|anonymous4g] [--idle 1,3,6,9] [--repeats 1..3] [--tiny-wake] [--hold-seconds 0..45]\n"
          "Root-only GPU. Only metadata/one-word CPU reads, no Forward/head/caches/arenas. All resources and reflected pipelines prepared before any idle sample.\n"
          "tiny-only skips FlashWeights; model-tiny loads but never binds original bases. anonymous4g skips FlashWeights and wraps a 4GiB owned Shared allocation. --tiny-wake adds wake+geometry samples after each idle interval.\n";
        return 0;
      }
      require(argc >= 4, "usage: oracle METALLIB PACKAGE REPORT_JSON [bounded selectors]");
      const std::filesystem::path reportPath(argv[3]);
      require(!std::filesystem::exists(reportPath) &&
              !std::filesystem::exists(reportPath.string() + ".commands.jsonl"), "choose fresh report files");
      std::string mode = "all";
      auto idles = list("1,3,6,9", 9);
      uint32_t repeats = 1, holdSeconds = 0;
      bool tinyWake = false;
      for (int index = 4; index < argc; ++index) {
        const std::string_view key(argv[index]);
        if (key == "--tiny-wake") { tinyWake = true; continue; }
        require(index + 1 < argc, "selector value missing");
        const std::string_view value(argv[++index]);
        if (key == "--mode") mode = value;
        else if (key == "--idle") idles = list(value, 9);
        else if (key == "--repeats") { repeats = integer(value, 3); require(repeats > 0, "repeats must be1..3"); }
        else if (key == "--hold-seconds") holdSeconds = integer(value, 45);
        else require(false, "unknown bounded selector");
      }
      const std::array<std::string_view, 7> modes{"tiny-only", "model-tiny", "original1", "original4", "original21", "all", "anonymous4g"};
      require(std::find(modes.begin(), modes.end(), mode) != modes.end(), "unsupported probe mode");
      MetalBackend backend(argv[1]);
      require(backend.supportsArgumentBuffersTier2(), "Tier2 readonly arguments unsupported");
      auto tiny = backend.allocateBuffer(16384, BufferStorage::Shared, "idle probe tiny source");
      const uint32_t tinyValue = 0x82761934;
      std::memcpy(tiny.contents(), &tinyValue, sizeof(tinyValue));
      std::vector<Geometry> geometries;
      geometries.reserve(5);
      geometries.emplace_back(backend, "tiny", std::vector<MetalBuffer>{tiny});
      std::unique_ptr<flash::FlashWeights> weights;
      std::vector<MetalBuffer> originals;
      if (mode != "tiny-only" && mode != "anonymous4g") {
        weights = std::make_unique<flash::FlashWeights>(flash::FlashWeights::load(backend, argv[2]));
        originals = weights->immutableWeightBuffers();
        require(originals.size() == 21 && weights->actualAllocatedBytes() == kExpectedBytes,
                "original21 readonly base geometry differs");
        require(weights->sourceIdentity() == kSource && weights->tensorCount() == 3748,
                "qualified original model/source differs");
        uint64_t bytes = 0;
        for (const auto &base : originals) bytes += base.sizeBytes();
        require(bytes == kExpectedBytes, "original mapped bytes differ");
      }
      std::vector<uint32_t> selectedIndices;
      if (!originals.empty()) {
        selectedIndices.resize(originals.size());
        std::iota(selectedIndices.begin(), selectedIndices.end(), 0);
        std::stable_sort(selectedIndices.begin(), selectedIndices.end(), [&](uint32_t left, uint32_t right) {
          return originals[left].sizeBytes() > originals[right].sizeBytes();
        });
      }
      for (uint32_t count : {1u, 4u, 21u}) {
        const std::string name = "original" + std::to_string(count);
        if (mode != "all" && mode != name) continue;
        std::vector<uint32_t> indices(selectedIndices.begin(), selectedIndices.begin() + count);
        if (count == 21) std::sort(indices.begin(), indices.end());
        std::vector<MetalBuffer> bases;
        for (uint32_t index : indices) bases.push_back(originals[index]);
        geometries.emplace_back(backend, name, std::move(bases), std::move(indices));
      }
      if (mode == "anonymous4g") {
        // Allocation is intentionally not memset: touch only the first word.
        // Driver physical/VM policy is measured rather than assuming 4GiB is committed.
        auto anonymous = backend.allocateBuffer(4ULL << 30, BufferStorage::Shared, "idle probe anonymous4g source");
        const uint32_t value = 0x4527139c;
        std::memcpy(anonymous.contents(), &value, sizeof(value));
        geometries.emplace_back(backend, "anonymous4g", std::vector<MetalBuffer>{anonymous});
      }
      backend.setCommandDispatchProfiling(CommandDispatchProfilingMode::Command);
      (void)backend.takeCommandDispatchProfiles();
      std::ofstream report(reportPath), commands(reportPath.string() + ".commands.jsonl");
      require(bool(report) && bool(commands), "cannot open probe reports");
      report << "{\"schema\":\"splash-private-idle-minimal-resource-probe-v1\",\"command_timing_abi_bytes\":200,\"pid\":" << getpid()
        << ",\"mode\":" << json::quote(mode) << ",\"weight_loader_only\":true,\"forward_head_or_cache_created\":false"
        << ",\"weights_loaded\":" << (weights ? "true" : "false")
        << ",\"original_native_base_count\":" << originals.size()
        << ",\"original_native_bytes\":" << (weights ? weights->actualAllocatedBytes() : 0)
        << ",\"source_identity\":" << json::quote(weights ? weights->sourceIdentity() : "")
        << ",\"manifest_fingerprint\":" << json::quote(weights ? weights->manifestFingerprint() : "")
        << ",\"residency_registration\":false,\"kernel_clock_is_os_kernel_not_shader\":true,\"geometries\":[";
      for (size_t index = 0; index < geometries.size(); ++index) { if (index) report << ','; geometryJson(report, geometries[index]); }
      report << "],\"samples\":[";
      report.flush();
      std::cerr << "READY pid=" << getpid() << " mode=" << mode << " geometries=" << geometries.size()
                << " weights_bytes=" << (weights ? weights->actualAllocatedBytes() : 0) << "\n";
      if (holdSeconds) std::this_thread::sleep_for(std::chrono::seconds(holdSeconds));
      uint64_t sampleCount = 0;
      double priorHardwareGpuEndSteady = 0;
      auto execute = [&](Geometry &geometry, std::string_view phase, uint32_t requestedIdle,
                         double actualIdle, uint32_t repeat, bool wakePreceded) {
        // Only the tiny output is written here; all readonly resources/layouts predate sleep.
        geometry.clear();
        const auto before = now();
        const auto timing = backend.submitCommand(geometry.graph->dispatches());
        const auto ended = now();
        geometry.validate();
        const auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == 1, "one command profile required");
        const auto &profile = profiles.front();
        require(profile.mode == CommandDispatchProfilingMode::Command &&
                profile.status == CommandDispatchProfileStatus::Complete &&
                profile.dispatchCount == 1 && !profile.encoderBoundariesAltered && !profile.samplingBarriers,
                "normal command timing profile invalid");
        require(profile.commandGpuStartSeconds > 0 &&
                profile.commandGpuEndSeconds >= profile.commandGpuStartSeconds &&
                profile.commitClockBridge.valid, "hardware GPU clock/bridge missing");
        const double steadyGpuStart = profile.commandGpuStartSeconds + profile.commitClockBridge.steadyMinusMachSeconds;
        if (sampleCount++) report << ',';
        report << "{\"geometry\":" << json::quote(geometry.name)
          << ",\"phase\":" << json::quote(phase)
          << ",\"repeat\":" << repeat << ",\"requested_idle_seconds\":" << requestedIdle
          << ",\"actual_idle_seconds\":"; number(report, actualIdle);
        report << ",\"tiny_wake_preceded\":" << (wakePreceded ? "true" : "false")
          << ",\"host_begin_steady_seconds\":"; number(report, before);
        report << ",\"previous_hardware_gpu_end_to_commit_begin_seconds\":";
        if (priorHardwareGpuEndSteady > 0) number(report, profile.hostCommitBeginSeconds - priorHardwareGpuEndSteady);
        else report << "null";
        report << ",\"host_end_steady_seconds\":"; number(report, ended);
        report << ",\"end_to_end_seconds\":"; number(report, ended - before);
        report << ",\"commit_end_to_hardware_gpu_start_seconds\":"; number(report, steadyGpuStart - profile.hostCommitEndSeconds);
        report << ",\"commit_end_to_os_kernel_start_seconds\":";
        if (profile.commandKernelTimingValid) number(report, profile.commandKernelStartSeconds + profile.commitClockBridge.steadyMinusMachSeconds - profile.hostCommitEndSeconds);
        else report << "null";
        report << ",\"exact_words_and_guards\":true,\"command_sequence\":" << profile.sequence << ",\"command\":";
        profiling::writeJson(report, profile); report << '}'; report.flush();
        commands << "{\"geometry\":" << json::quote(geometry.name) << ",\"phase\":" << json::quote(phase)
                 << ",\"requested_idle_seconds\":" << requestedIdle << ",\"repeat\":" << repeat << ",\"command\":";
        profiling::writeJson(commands, profile); commands << "}\n"; commands.flush();
        std::cerr << "SAMPLE geometry=" << geometry.name << " phase=" << phase << " idle=" << requestedIdle
                  << " wait_ms=" << (steadyGpuStart - profile.hostCommitEndSeconds) * 1000
                  << " gpu_us=" << timing.gpuSeconds * 1e6 << " wall_ms=" << (ended - before) * 1000 << "\n";
        priorHardwareGpuEndSteady = profile.commandGpuEndSeconds + profile.commitClockBridge.steadyMinusMachSeconds;
      };
      // Prepare and execute each geometry twice; no pipeline is first created after idle.
      for (auto &geometry : geometries) {
        execute(geometry, "warmup", 0, 0, 0, false);
        execute(geometry, "immediate", 0, 0, 0, false);
      }
      for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint32_t idle : idles) {
          for (auto &geometry : geometries) {
            const auto sleepBegan = now();
            std::this_thread::sleep_for(std::chrono::seconds(idle));
            const auto elapsed = now() - sleepBegan;
            execute(geometry, "idle_direct", idle, elapsed, repeat, false);
            execute(geometry, "idle_immediate_followup", 0, 0, repeat, false);
            if (tinyWake && geometry.name != "tiny") {
              const auto wakeSleepBegan = now();
              std::this_thread::sleep_for(std::chrono::seconds(idle));
              const auto wakeElapsed = now() - wakeSleepBegan;
              execute(geometries.front(), "idle_tiny_wake", idle, wakeElapsed, repeat, false);
              execute(geometry, "after_tiny_wake", 0, 0, repeat, true);
              execute(geometry, "wake_immediate_followup", 0, 0, repeat, true);
            }
          }
        }
      }
      require(backend.healthy(), "probe backend unhealthy");
      const auto memory = backend.memoryStats();
      report << "],\"sample_count\":" << sampleCount << ",\"execution_complete\":true,\"valid\":true"
        << ",\"final_allocated_bytes\":" << memory.allocatedBytes
        << ",\"peak_allocated_bytes\":" << memory.peakAllocatedBytes
        << ",\"device_current_allocated_bytes\":" << memory.deviceCurrentAllocatedBytes
        << ",\"submission_count\":" << backend.submissionCount() << "}\n";
      report.close(); commands.close();
      std::cout << "{\"pass\":true,\"sample_count\":" << sampleCount
                << ",\"gpu_commands\":" << backend.submissionCount() << "}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "idle-minimal-resource-probe: " << error.what() << '\n';
      return 1;
    }
  }
}
