#include "Policy.hpp"
#include "flash/FlashRequestStateInternal.hpp"
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
#include <cstring>
#include <optional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>


namespace splash::flash {
class FlashDeepPrefixOracle final {
public:
  using Metadata = std::vector<std::pair<const void *, uint64_t>>;
  static std::vector<metal::MetalBuffer> planes(const FlashRequestState &state) {
    if (!state.impl_ || state.impl_->poisoned || state.impl_->pendingVerification)
      throw std::logic_error("private state reset requires healthy state without provisional verification");
    std::vector<metal::MetalBuffer> result;
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const auto &g = state.impl_->gdn[layer]; const auto &q = state.impl_->qsa[layer];
      if (g.convolution) { result.push_back(g.convolution); result.push_back(g.recurrent); }
      if (q.keys) for (const auto &buffer : {q.keys,q.values,q.rawIndexKeys,q.pooledKeys,q.indexPositions})
        result.push_back(buffer);
    }
    result.push_back(state.impl_->pleHistory); result.push_back(state.impl_->pleConvolution);
    return result;
  }
  static Metadata metadata(const FlashRequestState &state) {
    Metadata result;
    for (const auto &buffer : planes(state)) result.emplace_back(buffer.contents(),buffer.sizeBytes());
    if (result.size() != 134) throw std::logic_error("private state reset expects134 exact native planes");
    uint64_t bytes = 0; for (const auto &entry : result) bytes += entry.second;
    if (bytes != 349388800) throw std::logic_error("private state reset expects349388800 exact native bytes");
    return result;
  }
  static void reset(FlashRequestState &state, int64_t eos) {
    const auto before = metadata(state);
    // Allocate the new verification identity before mutation so CPU allocation
    // failure cannot leave partially-reset storage available for inference.
    auto identity = std::make_shared<const uint8_t>(0);
    const auto buffers = planes(state);
    for (const auto &buffer : buffers) {
      if (buffer.storage() != metal::BufferStorage::Shared || !buffer.contents())
        throw std::logic_error("private reset requires existing Shared host-readable storage");
    }
    for (const auto &buffer : buffers) std::memset(buffer.contents(),0,buffer.sizeBytes());
    auto *history = static_cast<int64_t *>(state.impl_->pleHistory.contents());
    history[0] = history[1] = eos;
    state.impl_->length = 0;
    state.impl_->poisoned = false;
    state.impl_->pendingVerification = false;
    state.impl_->identity = std::move(identity);
    if (metadata(state) != before) throw std::logic_error("private reset changednative storage identity");
  }
};
} // namespace splash::flash

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
  require(result.size() == 128, "persistent state diagnostic requires exactly 128 real prompt tokens");
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
        std::cout << "persistent-state-idle-v9-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON\n"
          "FLASH_STATE_IDLE_SECONDS=0..15 (default9); FLASH_STATE_IDLE_FIRST=fresh|reuse\n"
          "Loads normal profile operands; Six prefills: warm fresh/reuse, idle fresh/reuse/reuse/fresh; exact full logits and two continuations.\n";
        return 0;
      }
      require(argc == 5, "usage: persistent-state-idle-v9-oracle METALLIB PACKAGE TOKENS_128_JSON REPORT_JSON");
      require(!std::filesystem::exists(argv[4]) &&
          !std::filesystem::exists(std::string(argv[4]) + ".commands.jsonl"), "choose a fresh report path");
      const auto prompt = tokens(argv[3]);
      uint32_t idleSeconds = 9;
      if (const char *raw = std::getenv("FLASH_STATE_IDLE_SECONDS")) {
        size_t end = 0; const unsigned long value = std::stoul(raw, &end);
        require(end == std::string_view(raw).size() && value <= 15, "idle seconds must be0..15");
        idleSeconds = uint32_t(value);
      }
      const char *order = std::getenv("FLASH_STATE_IDLE_FIRST");
      const bool reuseFirst = order && std::string_view(order) == "reuse";
      require(!order || std::string_view(order) == "reuse" || std::string_view(order) == "fresh", "invalid state idle first mode");
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
      require(bool(reservation), "governor denied private persistent-state arenas");
      FlashForward target(backend, weights, capacity, maximumRows, 16);
      FlashMTPForward head(backend, weights, capacity, 128);
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto operands = target.cachedOperandsOnly(); const auto extra = head.cachedOperandsOnly();
        operands.insert(operands.end(), extra.begin(), extra.end());
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "private persistent-state normal saved operands only");
      }
      reservation->commit();
      (void)backend.takeCommandDispatchProfiles();
      backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Command);
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl");
      require(bool(commands), "cannot open persistent-state command report");
      std::array<std::vector<uint16_t>, 3> reference{};
      // One retained, already-exercised state stays alive across every idle
      // comparison. This controls retention separately from native identity reuse.
      auto persistent = target.createState();
      const auto persistentMetadata = FlashDeepPrefixOracle::metadata(persistent);
      const std::array<bool, 6> modes{false, true, reuseFirst, !reuseFirst, !reuseFirst, reuseFirst};
      std::ostringstream samples;
      for (uint32_t sample = 0; sample < modes.size(); ++sample) {
        const bool reuse = modes[sample];
        if (sample >= 2 && idleSeconds) std::this_thread::sleep_for(std::chrono::seconds(idleSeconds));
        require(backend.healthy() && !backend.needsHealthCheck() && target.ownsState(persistent),
            "private reset requires owned state and healthy quiescent backend");
        const auto began = Clock::now();
        std::optional<FlashRequestState> fresh;
        if (reuse) FlashDeepPrefixOracle::reset(persistent, weights.descriptor().pleHistoryEos);
        else fresh.emplace(target.createState());
        auto &state = reuse ? persistent : *fresh;
        const double statePreparation = std::chrono::duration<double>(Clock::now() - began).count();
        const auto forwardBegan = Clock::now();
        auto result = target.forward(state, prompt);
        const double forwardElapsed = std::chrono::duration<double>(Clock::now() - forwardBegan).count();
        const double elapsed = std::chrono::duration<double>(Clock::now() - began).count();
        require(FlashDeepPrefixOracle::metadata(persistent) == persistentMetadata,
            "persistent native plane identity/bytes changed");
        auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == 1, "persistent state changed target graph command count");
        require(result.logicalLength == 128 && !state.poisoned(), "private execution/state mismatch");
        uint64_t dispatches = 0; double totalDelay = 0;
        for (uint32_t segment = 0; segment < profiles.size(); ++segment) {
          const auto &profile = profiles[segment];
          require(profile.status == metal::CommandDispatchProfileStatus::Complete, "profile did not complete");
          const auto clocks = compareFlashRequestTraceClocks(profile);
          require(clocks.crossClockValid, "persistent-state clock bridge unavailable");
          dispatches += profile.dispatchCount; totalDelay += clocks.commitEndToGPUStart;

          commands << "{\"sample\":" << sample << ",\"reuse\":" << (reuse ? "true" : "false")
              << ",\"segment\":" << json::quote("unchanged_whole_graph")
              << ",\"commit_end_to_gpu_start_seconds\":";
          profiling::writeNumber(commands, clocks.commitEndToGPUStart);
          commands << ",\"command\":"; profiling::writeJson(commands, profile); commands << "}\n";
        }
        require(dispatches == 1709, "persistent state changed real target graph dispatches");
        auto logits = copyLogits(result);
        if (sample == 0) reference[0] = logits;
        else require(logits == reference[0], "reused-state prefill BF16 logits differ from fresh control");
        if (sample) samples << ',';
        samples << "{\"sample\":" << sample << ",\"reuse\":" << (reuse ? "true" : "false")
            << ",\"idle_seconds\":" << (sample >= 2 ? idleSeconds : 0) << ",\"commands\":" << profiles.size()
            << ",\"dispatches\":" << dispatches << ",\"call_seconds\":"; profiling::writeNumber(samples, elapsed);
        samples << ",\"state_prepare_seconds\":"; profiling::writeNumber(samples, statePreparation);
        samples << ",\"forward_seconds\":"; profiling::writeNumber(samples, forwardElapsed);
        samples << ",\"sum_commit_end_to_gpu_start_seconds\":"; profiling::writeNumber(samples, totalDelay);
        samples << ",\"gpu_seconds\":"; profiling::writeNumber(samples, result.timing.gpuSeconds);
        samples << ",\"command_wall_seconds\":"; profiling::writeNumber(samples, result.timing.wallSeconds);
        samples << ",\"prefill_logits_sha256\":" << json::quote(digest(logits)) << ",\"greedy\":" << greedy(logits) << '}';
        for (uint32_t step = 1; step <= 2; ++step) {
          const uint32_t token = greedy(logits);
          result = target.forward(state, std::span(&token, 1)); logits = copyLogits(result);
          const auto continuation = backend.takeCommandDispatchProfiles();
          require(continuation.size() == 1 &&
              state.logicalLength() == 128 + step && !state.poisoned(), "continuation route/state changed");
          if (sample == 0) reference[step] = logits;
          else require(logits == reference[step], "reused-state continuation BF16 logits differ from fresh control");
        }
      }
      require(bool(commands), "could not write command profiles");
      std::ofstream report(argv[4]);
      report << "{\"schema\":\"splash-private-persistent-state-idle-v9\",\"valid\":true,\"gpu_work\":true"
          << ",\"diagnostic_not_http_score\":true,\"original_math_graph_unchanged\":true"
          << ",\"persistent_state_planes\":134,\"persistent_state_bytes\":349388800"
          << ",\"persistent_state_retained_in_fresh_and_reuse_cases\":true"
          << ",\"source_identity\":" << json::quote(weights.sourceIdentity())
          << ",\"layout_identity\":" << json::quote(weights.manifestFingerprint())
          << ",\"kernel_routes\":" << json::quote(target.kernelRoutes())
          << ",\"resident_union_count\":" << residency.bufferCount()
          << ",\"resident_union_bytes\":" << residency.byteCount()
          << ",\"prefill_and_two_continuations_bit_exact\":true,\"samples\":[" << samples.str() << "]}\n";
      require(bool(report), "could not write persistent-state report");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "persistent-state-idle-v9-oracle: " << error.what() << '\n';
      return 1;
    }
  }
}
