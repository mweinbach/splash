#include "Diagnostic.hpp"
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
  require(result.size() == 128, "frontier diagnostic requires exactly 128 real prompt tokens");
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
        std::cout << "prefill-frontier-split-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON\n"
          "FLASH_FRONTIER_IDLE_SECONDS=0..15 (default9); FLASH_FRONTIER_FIRST=whole|split\n"
          "Loads normal profile operands; four prefills whole/split then idle split/whole; two exact continuation steps.\n";
        return 0;
      }
      require(argc == 5, "usage: prefill-frontier-split-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON");
      require(!std::filesystem::exists(argv[4]) &&
          !std::filesystem::exists(std::string(argv[4]) + ".commands.jsonl"), "choose a fresh report path");
      const auto prompt = tokens(argv[3]);
      uint32_t idleSeconds = 9;
      if (const char *raw = std::getenv("FLASH_FRONTIER_IDLE_SECONDS")) {
        size_t end = 0; const unsigned long value = std::stoul(raw, &end);
        require(end == std::string_view(raw).size() && value <= 15, "idle seconds must be0..15");
        idleSeconds = uint32_t(value);
      }
      const char *order = std::getenv("FLASH_FRONTIER_FIRST");
      const bool splitFirst = order && std::string_view(order) == "split";
      require(!order || std::string_view(order) == "split" || std::string_view(order) == "whole", "invalid frontier first mode");
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
      require(bool(reservation), "governor denied private frontier arenas");
      FlashForward target(backend, weights, capacity, maximumRows, 16);
      FlashMTPForward head(backend, weights, capacity, 128);
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto operands = target.cachedOperandsOnly(); const auto extra = head.cachedOperandsOnly();
        operands.insert(operands.end(), extra.begin(), extra.end());
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "private frontier normal saved operands only");
      }
      reservation->commit();
      (void)backend.takeCommandDispatchProfiles();
      backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Command);
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl");
      require(bool(commands), "cannot open frontier command report");
      std::array<std::vector<uint16_t>, 3> reference{};
      const std::array<bool, 4> modes{splitFirst, !splitFirst, true, false};
      std::ostringstream samples;
      for (uint32_t sample = 0; sample < modes.size(); ++sample) {
        const bool split = modes[sample];
        if (sample >= 2 && idleSeconds) std::this_thread::sleep_for(std::chrono::seconds(idleSeconds));
        auto state = target.createState();
        setenv("SPLASH_FLASH_PRIVATE_PREFILL_FRONTIER_SPLIT", split ? "1" : "0", 1);
        const auto began = Clock::now();
        auto result = target.forward(state, prompt);
        const double elapsed = std::chrono::duration<double>(Clock::now() - began).count();
        const auto execution = private_prefill_frontier::lastExecution;
        auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == (split ? 2u : 1u), "private graph command count mismatch");
        require(execution.split == split && result.logicalLength == 128 && !state.poisoned(), "private execution/state mismatch");
        uint64_t dispatches = 0; double totalDelay = 0;
        for (uint32_t segment = 0; segment < profiles.size(); ++segment) {
          const auto &profile = profiles[segment];
          require(profile.status == metal::CommandDispatchProfileStatus::Complete, "profile did not complete");
          const auto clocks = compareFlashRequestTraceClocks(profile);
          require(clocks.crossClockValid, "frontier clock bridge unavailable");
          dispatches += profile.dispatchCount; totalDelay += clocks.commitEndToGPUStart;
          if (split && segment == 0) require(profile.dispatchCount == 2, "frontier prefix command must have two dispatches");
          commands << "{\"sample\":" << sample << ",\"split\":" << (split ? "true" : "false")
              << ",\"segment\":" << json::quote(split ? (segment == 0 ? "embedding_expand" : "remaining_layers") : "whole_graph")
              << ",\"commit_end_to_gpu_start_seconds\":";
          profiling::writeNumber(commands, clocks.commitEndToGPUStart);
          commands << ",\"command\":"; profiling::writeJson(commands, profile); commands << "}\n";
        }
        require(dispatches == execution.dispatches, "split omitted or duplicated dispatches");
        auto logits = copyLogits(result);
        if (sample == 0) reference[0] = logits;
        else require(logits == reference[0], "split prefill BF16 logits differ from control");
        if (sample) samples << ',';
        samples << "{\"sample\":" << sample << ",\"split\":" << (split ? "true" : "false")
            << ",\"idle_seconds\":" << (sample >= 2 ? idleSeconds : 0) << ",\"commands\":" << profiles.size()
            << ",\"dispatches\":" << dispatches << ",\"call_seconds\":"; profiling::writeNumber(samples, elapsed);
        samples << ",\"sum_commit_end_to_gpu_start_seconds\":"; profiling::writeNumber(samples, totalDelay);
        samples << ",\"gpu_seconds\":"; profiling::writeNumber(samples, result.timing.gpuSeconds);
        samples << ",\"command_wall_seconds\":"; profiling::writeNumber(samples, result.timing.wallSeconds);
        samples << ",\"prefill_logits_sha256\":" << json::quote(digest(logits)) << ",\"greedy\":" << greedy(logits) << '}';
        for (uint32_t step = 1; step <= 2; ++step) {
          const uint32_t token = greedy(logits);
          result = target.forward(state, std::span(&token, 1)); logits = copyLogits(result);
          const auto continuation = backend.takeCommandDispatchProfiles();
          require(continuation.size() == 1 && !private_prefill_frontier::lastExecution.split &&
              state.logicalLength() == 128 + step && !state.poisoned(), "continuation route/state changed");
          if (sample == 0) reference[step] = logits;
          else require(logits == reference[step], "split continuation BF16 logits differ from control");
        }
      }
      require(bool(commands), "could not write command profiles");
      std::ofstream report(argv[4]);
      report << "{\"schema\":\"splash-private-prefill-frontier-split-v1\",\"valid\":true,\"gpu_work\":true"
          << ",\"diagnostic_not_http_score\":true,\"first_two_dispatches_original_base_bytes\":5204606976"
          << ",\"source_identity\":" << json::quote(weights.sourceIdentity())
          << ",\"layout_identity\":" << json::quote(weights.manifestFingerprint())
          << ",\"kernel_routes\":" << json::quote(target.kernelRoutes())
          << ",\"resident_union_count\":" << residency.bufferCount()
          << ",\"resident_union_bytes\":" << residency.byteCount()
          << ",\"prefill_and_two_continuations_bit_exact\":true,\"samples\":[" << samples.str() << "]}\n";
      require(bool(report), "could not write frontier report");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "prefill-frontier-split-oracle: " << error.what() << '\n';
      return 1;
    }
  }
}
