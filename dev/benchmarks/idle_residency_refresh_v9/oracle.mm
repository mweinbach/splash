#include "Policy.hpp"
#include "../full_original_residency/Policy.hpp"
#include "flash/FlashForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashRequestCommandTrace.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "metal/ProfilingJson.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace splash;
using namespace splash::flash;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
bool enabled(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || std::string_view(raw) == "0") return false;
  require(std::string_view(raw) == "1", "oracle switch must be 0 or 1");
  return true;
}
std::vector<uint32_t> tokens(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  require(data != nil, "cannot read oracle tokens");
  NSError *error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error && [object isKindOfClass:[NSArray class]], "oracle tokens must be a JSON array");
  std::vector<uint32_t> result;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] &&
        CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(), "oracle token must be an integer");
    double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < 248320 &&
        std::floor(value) == value, "oracle token outside vocabulary");
    result.push_back(uint32_t(value));
  }
  require(result.size() == 128, "idle refresh diagnostic requires exactly 128 real prompt tokens");
  return result;
}
std::vector<uint16_t> copyLogits(const FlashForwardResult &result) {
  require(result.logitRows == 1 && result.logitsBF16.contents() &&
      result.logitsBF16.sizeBytes() >= 248320 * 2, "oracle logits missing");
  const auto *p = static_cast<const uint16_t *>(result.logitsBF16.contents());
  return {p, p + 248320};
}
std::string digest(std::span<const uint16_t> values) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> hash{};
  CC_SHA256(values.data(), CC_LONG(values.size_bytes()), hash.data());
  const char *hex = "0123456789abcdef"; std::string result;
  for (unsigned char value : hash) { result += hex[value >> 4]; result += hex[value & 15]; }
  return result;
}
uint32_t greedy(std::span<const uint16_t> logits) {
  uint32_t best = 0; float high = -INFINITY;
  for (uint32_t i = 0; i < logits.size(); ++i) {
    float value = std::bit_cast<float>(uint32_t{logits[i]} << 16);
    require(std::isfinite(value), "nonfinite oracle logits");
    if (value > high) { high = value; best = i; }
  }
  return best;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      static_assert(sizeof(metal::CommandTiming) == 200, "fresh normal timing ABI required");
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "idle-residency-refresh-v9-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON\n"
          "FLASH_IDLE_REFRESH_IDLE_SECONDS=0..15 (default9); FLASH_IDLE_REFRESH_FIRST=off|on\n"
          "Loads normal profile operands; four prefills off/on then idle on/off; two exact continuation steps.\n";
        return 0;
      }
      require(argc == 5, "usage: idle-residency-refresh-v9-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON");
      require(!std::filesystem::exists(argv[4]) &&
          !std::filesystem::exists(std::string(argv[4]) + ".commands.jsonl"), "choose a fresh report path");
      const auto prompt = tokens(argv[3]);
      uint32_t idleSeconds = 9;
      if (const char *raw = std::getenv("FLASH_IDLE_REFRESH_IDLE_SECONDS")) {
        size_t end = 0; const unsigned long value = std::stoul(raw, &end);
        require(end == std::string_view(raw).size() && value <= 15, "idle seconds must be0..15");
        idleSeconds = uint32_t(value);
      }
      const char *order = std::getenv("FLASH_IDLE_REFRESH_FIRST");
      const bool refreshFirst = order && std::string_view(order) == "on";
      require(!order || std::string_view(order) == "on" || std::string_view(order) == "off", "invalid refresh first mode");
      const bool fullUnion = enabled("SPLASH_FLASH_PRIVATE_FULL_ORIGINAL_RESIDENT");
      setenv("SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH", refreshFirst ? "1" : "0", 1);
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      constexpr uint32_t capacity = 8192, maximumRows = 2048;
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(capacity, maximumRows, 16) +
          FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) +
          FlashForward::int8HeadPlannedBytes(weights) + FlashMTPForward::workspacePlannedBytes(capacity, 128) +
          4 * FlashForward::requestStateBytes(capacity) + (16ULL << 20);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights,
          FlashDenseCache::defaultPrefixes(weights, true)) + FlashMTPForward::denseCachePlannedBytes(weights);
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(maximumRows, 10);
      auto reservation = governor.tryReserve(planned);
      require(bool(reservation), "governor denied private idle refresh arenas");
      FlashForward target(backend, weights, capacity, maximumRows, 16);
      FlashMTPForward head(backend, weights, capacity, 128);
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT") || fullUnion) {
        auto operands = target.cachedOperandsOnly(); const auto extra = head.cachedOperandsOnly();
        operands.insert(operands.end(), extra.begin(), extra.end());
        if (fullUnion) {
          const auto original = weights.immutableWeightBuffers();
          uint64_t originalBytes = 0;
          for (const auto &buffer : original) originalBytes += buffer.sizeBytes();
          private_full_residency::validateGeometry(weights.sourceIdentity(),
              weights.manifestFingerprint(), original.size(), originalBytes,
              weights.descriptor().layers, weights.descriptor().experts, weights.descriptor().hiddenSize);
          const auto host = governor.snapshot();
          require(private_full_residency::hostAllowed(host.hostMeasurementValid,
              host.growthAllowed, host.systemPressure == engine::MemoryPressure::Normal &&
              host.pressure == engine::MemoryPressure::Normal, host.hostHeadroomBytes),
              "host reserve denies original-weight union");
          operands.insert(operands.end(), original.begin(), original.end());
        }
        if (!operands.empty()) residency = backend.requestWeightResidency(operands,
            fullUnion ? "private idle refresh full original plus derived union" : "private idle refresh normal saved operands only");
      }
      require(bool(residency), "idle refresh diagnostic requires a residency registration");
      reservation->commit();
      (void)backend.takeCommandDispatchProfiles();
      backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Command);
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl");
      require(bool(commands), "cannot open idle refresh command report");
      std::array<std::vector<uint16_t>, 3> reference{};
      const std::array<bool, 4> modes{refreshFirst, !refreshFirst, true, false};
      std::ostringstream samples;
      for (uint32_t sample = 0; sample < modes.size(); ++sample) {
        const bool refresh = modes[sample];
        if (sample >= 2 && idleSeconds) std::this_thread::sleep_for(std::chrono::seconds(idleSeconds));
        auto state = target.createState();
        // Private backend flag is per submission for this four-case diagnostic.
        setenv("SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH", refresh ? "1" : "0", 1);
        const auto began = Clock::now();
        auto result = target.forward(state, prompt);
        const double elapsed = std::chrono::duration<double>(Clock::now() - began).count();
        const auto execution = metal::private_idle_residency::lastRefresh;
        auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == 1, "private refresh changed graph command count");
        require(result.logicalLength == 128 && !state.poisoned(), "private execution/state mismatch");
        require(!execution.requested || (refresh && sample >= 2 && idleSeconds > 5 && execution.succeeded),
            "refresh ran outside intended idle case");
        require(sample != 2 || idleSeconds <= 5 || execution.requested,
            "idle refresh failed to run");
        uint64_t dispatches = 0; double totalDelay = 0;
        for (uint32_t segment = 0; segment < profiles.size(); ++segment) {
          const auto &profile = profiles[segment];
          require(profile.status == metal::CommandDispatchProfileStatus::Complete, "profile did not complete");
          const auto clocks = compareFlashRequestTraceClocks(profile);
          require(clocks.crossClockValid, "idle refresh clock bridge unavailable");
          dispatches += profile.dispatchCount; totalDelay += clocks.commitEndToGPUStart;

          commands << "{\"sample\":" << sample << ",\"refresh_enabled\":" << (refresh ? "true" : "false")
              << ",\"segment\":" << json::quote("unchanged_whole_graph")
              << ",\"commit_end_to_gpu_start_seconds\":";
          profiling::writeNumber(commands, clocks.commitEndToGPUStart);
          commands << ",\"command\":"; profiling::writeJson(commands, profile); commands << "}\n";
        }
        require(dispatches == profiles[0].dispatchCount, "refresh changed dispatches");
        auto logits = copyLogits(result);
        if (sample == 0) reference[0] = logits;
        else require(logits == reference[0], "refresh prefill BF16 logits differ from control");
        if (sample) samples << ',';
        samples << "{\"sample\":" << sample << ",\"refresh_enabled\":" << (refresh ? "true" : "false")
            << ",\"idle_seconds\":" << (sample >= 2 ? idleSeconds : 0) << ",\"commands\":" << profiles.size()
            << ",\"dispatches\":" << dispatches << ",\"call_seconds\":"; profiling::writeNumber(samples, elapsed);
        samples << ",\"refresh_requested\":" << (execution.requested ? "true" : "false")
            << ",\"refresh_succeeded\":" << (execution.succeeded ? "true" : "false")
            << ",\"refresh_count\":" << execution.count << ",\"refresh_idle_seconds\":";
        profiling::writeNumber(samples, execution.idleSeconds);
        samples << ",\"refresh_api_seconds\":"; profiling::writeNumber(samples, execution.apiSeconds);
        samples << ",\"sum_commit_end_to_gpu_start_seconds\":"; profiling::writeNumber(samples, totalDelay);
        samples << ",\"gpu_seconds\":"; profiling::writeNumber(samples, result.timing.gpuSeconds);
        samples << ",\"command_wall_seconds\":"; profiling::writeNumber(samples, result.timing.wallSeconds);
        samples << ",\"prefill_logits_sha256\":" << json::quote(digest(logits)) << ",\"greedy\":" << greedy(logits) << '}';
        for (uint32_t step = 1; step <= 2; ++step) {
          const uint32_t token = greedy(logits);
          result = target.forward(state, std::span(&token, 1)); logits = copyLogits(result);
          const auto continuation = backend.takeCommandDispatchProfiles();
          require(continuation.size() == 1 && !metal::private_idle_residency::lastRefresh.requested &&
              state.logicalLength() == 128 + step && !state.poisoned(), "continuation route/state changed");
          if (sample == 0) reference[step] = logits;
          else require(logits == reference[step], "refresh continuation BF16 logits differ from control");
        }
      }
      require(bool(commands), "could not write command profiles");
      std::ofstream report(argv[4]);
      report << "{\"schema\":\"splash-private-idle-residency-refresh-v9\",\"valid\":true,\"gpu_work\":true"
          << ",\"diagnostic_not_http_score\":true,\"original_math_graph_state_unchanged\":true"
          << ",\"full_original_union\":" << (fullUnion ? "true" : "false")
          << ",\"source_identity\":" << json::quote(weights.sourceIdentity())
          << ",\"layout_identity\":" << json::quote(weights.manifestFingerprint())
          << ",\"kernel_routes\":" << json::quote(target.kernelRoutes())
          << ",\"resident_union_count\":" << residency.bufferCount()
          << ",\"resident_union_bytes\":" << residency.byteCount()
          << ",\"prefill_and_two_continuations_bit_exact\":true,\"samples\":[" << samples.str() << "]}\n";
      require(bool(report), "could not write idle refresh report");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "idle-residency-refresh-v9-oracle: " << error.what() << '\n';
      return 1;
    }
  }
}
