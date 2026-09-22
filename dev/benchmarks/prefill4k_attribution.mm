// Full target + real teacher-priming attribution with current profile flags.
// Compilation and --help do not create a Metal backend. Root serializes GPU runs.
#include "flash/FlashForward.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace splash;
using namespace splash::flash;
using Clock = std::chrono::steady_clock;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint32_t integer(const char *name, uint32_t fallback, uint32_t minimum, uint32_t maximum) {
  const char *raw = std::getenv(name); if (!raw) return fallback;
  uint64_t value = 0; require(*raw != 0, "empty attribution integer setting");
  for (const char *p = raw; *p; ++p) {
    require(*p >= '0' && *p <= '9', "attribution integer setting is not decimal");
    value = value * 10 + unsigned(*p - '0'); require(value <= maximum, "attribution integer setting exceeds limit");
  }
  require(value >= minimum, "attribution integer setting is below limit"); return uint32_t(value);
}
bool enabled(const char *name) {
  const char *raw = std::getenv(name); if (!raw) return false;
  require(std::string_view(raw) == "0" || std::string_view(raw) == "1", "attribution switch must be 0 or 1");
  return std::string_view(raw) == "1";
}
std::vector<uint32_t> loadTokens(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read attribution tokens");
  NSError *error = nil; id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSArray class]], "attribution tokens must be a JSON array");
  std::vector<uint32_t> result;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(),
        "attribution token must be an integer, not a boolean");
    const double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < 248320 && std::floor(value) == value, "invalid attribution token");
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), "attribution prompt is empty"); return result;
}
std::string digest(const void *bytes, uint64_t count) {
  CC_SHA256_CTX context{}; CC_SHA256_Init(&context);
  auto *next = static_cast<const uint8_t *>(bytes);
  while (count) { const CC_LONG part = CC_LONG(std::min<uint64_t>(count, UINT32_MAX));
    CC_SHA256_Update(&context, next, part); next += part; count -= part; }
  std::array<uint8_t, 32> result{}; CC_SHA256_Final(result.data(), &context);
  static constexpr char hex[] = "0123456789abcdef"; std::string output;
  for (uint8_t byte : result) { output += hex[byte >> 4]; output += hex[byte & 15]; } return output;
}
std::string hexadecimal(std::span<const uint8_t> bytes) {
  static constexpr char hex[] = "0123456789abcdef"; std::string output;
  for (uint8_t byte : bytes) { output += hex[byte >> 4]; output += hex[byte & 15]; }
  return output;
}
std::string fileDigest(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read attribution provenance"); return digest(data.bytes, data.length);
}
void writeHost(std::ostream &out, const metal::CommandHostTiming &h) {
  out << "{\"timed_commands\":" << h.timedCommands << ",\"commit_samples\":" << h.commitSamples
      << ",\"scheduled_callback_samples\":" << h.scheduledCallbackSamples
      << ",\"completed_callback_samples\":" << h.completedCallbackSamples
      << ",\"pre_commit_memory_samples\":" << h.preCommitMemorySamples
      << ",\"post_commit_memory_samples\":" << h.postCommitMemorySamples
      << ",\"scheduled_memory_samples\":" << h.scheduledMemorySamples
      << ",\"completed_memory_samples\":" << h.completedMemorySamples
      << ",\"ticket_wait_calls\":" << h.ticketWaitCalls;
  const auto field = [&](const char *name, double value) { out << ',' << json::quote(name) << ':'; profiling::writeNumber(out, value); };
  field("preparation_seconds", h.preparationSeconds); field("encoding_seconds", h.encodingSeconds);
  field("encoding_end_to_commit_begin_seconds", h.beforeCommitSeconds);
  field("sparse_dependency_wait_seconds", h.dependencyWaitSeconds); field("commit_seconds", h.commitSeconds);
  field("commit_to_scheduled_callback_seconds", h.commitToScheduledCallbackSeconds);
  field("commit_to_completed_callback_seconds", h.commitToCompletedCallbackSeconds);
  field("completed_callback_to_wall_end_seconds", h.completionCallbackBeforeWallEndSeconds);
  field("submission_return_latency_seconds", h.submissionReturnSeconds);
  field("ticket_blocking_wait_seconds", h.ticketBlockingWaitSeconds);
  field("pre_commit_memory_query_seconds", h.preCommitMemorySampleSeconds);
  field("post_commit_memory_query_seconds", h.postCommitMemorySampleSeconds);
  field("scheduled_memory_query_seconds", h.scheduledMemorySampleSeconds);
  field("completed_memory_query_seconds", h.completedMemorySampleSeconds);
  out << ",\"intervals_can_overlap\":true,\"scheduled_latency_scope\":\"commit begin to callback arrival, not actual GPU scheduling\"}";
}
struct Family { uint64_t dispatches = 0, timed = 0; double seconds = 0; };
struct Classifier {
  bool insideMoE = false, shared = false;
  uint64_t routes = 0;
  static bool dense(std::string_view n) {
    return n.starts_with("flash_affine") || n.starts_with("flash_dense") ||
        n.starts_with("flash_float_dense") || n.starts_with("flash_int8_head");
  }
  std::string family(std::string_view name, bool gatheredExpert = false) {
    if (name == "flash_moe_route") { insideMoE = true; shared = false; ++routes; return "moe"; }
    if (name == "flash_moe_combine") { insideMoE = shared = false; return "moe"; }
    if (insideMoE && !shared && gatheredExpert &&
        (name.starts_with("flash_affine") || name == "flash_moe_silu_multiply")) return "moe";
    if (insideMoE && dense(name)) shared = true;
    if (shared && (dense(name) || name == "flash_moe_silu_multiply")) return "shared_expert";
    if (name.starts_with("flash_shared_expert")) return "shared_expert";
    if (name.starts_with("flash_gdn")) return "gdn";
    if (name.starts_with("flash_qsa")) return "qsa";
    if (name.starts_with("flash_ple")) return "ple";
    if (name.starts_with("flash_hc") || name.starts_with("flash_forward_hc")) return "hc";
    if (name == "flash_affine_embedding") return "embedding";
    if (name == "flash_forward_copy_words") return "copies";
    if (name.starts_with("flash_greedy")) return "greedy";
    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert")) return "moe";
    if (dense(name)) return "dense";
    return "other";
  }
};
void writeFamilies(std::ostream &out, const metal::CommandDispatchProfile &profile, uint32_t expectedLayers) {
  Classifier classifier; std::map<std::string, Family> families;
  double timedSeconds = 0; uint64_t timedDispatches = 0;
  for (const auto &dispatch : profile.dispatches) {
    auto &f = families[classifier.family(dispatch.pipelineName, dispatch.threadgroups.z > 1)]; ++f.dispatches;
    if (dispatch.timestampsValid) { ++f.timed; ++timedDispatches; f.seconds += dispatch.gpuSeconds; timedSeconds += dispatch.gpuSeconds; }
  }
  const bool completeMetadata = !profile.dispatchMetadataTruncated && profile.dispatches.size() == profile.dispatchCount;
  out << "{\"classification_scope\":\"pipeline families; shared dense/activation interval after expert down and before canonical combine, validated by one MoE route per model layer\""
      << ",\"model_layer_routes\":" << classifier.routes << ",\"expected_model_layers\":" << expectedLayers
      << ",\"classification_complete\":" << (completeMetadata && classifier.routes == expectedLayers ? "true" : "false")
      << ",\"timed_dispatches\":" << timedDispatches << ",\"full_command_gpu_seconds\":" << profile.timing.gpuSeconds
      << ",\"sum_timed_dispatch_gpu_seconds\":" << timedSeconds
      << ",\"command_minus_timed_dispatch_seconds\":" << profile.timing.gpuSeconds - timedSeconds << ",\"families\":{";
  bool comma = false;
  for (const auto &[name, f] : families) {
    if (comma) out << ','; comma = true;
    out << json::quote(name) << ":{\"dispatches\":" << f.dispatches << ",\"timed_dispatches\":" << f.timed
        << ",\"gpu_seconds\":" << f.seconds << ",\"fraction_of_full_command_gpu\":";
    if (profile.timing.gpuSeconds > 0 && f.timed) profiling::writeNumber(out, f.seconds / profile.timing.gpuSeconds); else out << "null";
    out << ",\"fraction_of_timed_dispatch_gpu\":";
    if (timedSeconds > 0 && f.timed) profiling::writeNumber(out, f.seconds / timedSeconds); else out << "null";
    out << '}';
  }
  out << "}}";
}
void usage() {
  std::cout << "usage: prefill4k-attribution METALLIB PACKAGE TOKENS_JSON REPORT_JSON\n"
      "PREFILL4K_ATTRIBUTION_MODE=normal|command|stage|dispatch; WARMUP=1; REPEATS=1; ROWS=2048; CAPACITY=8192.\n"
      "Supplied current profile flags are required. TEACHER_PRIME=1 (default) performs exact 128-row singleton head priming.\n"
      "No HTTP scheduler/decode is instantiated. Stage/dispatch are diagnostic only.\n";
}
} // namespace
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
      require(argc == 5, "usage: prefill4k-attribution METALLIB PACKAGE TOKENS_JSON REPORT_JSON");
      const auto tokens = loadTokens(argv[3]);
      const uint32_t rows = integer("PREFILL4K_ATTRIBUTION_ROWS", 2048, 1, 8192);
      const uint32_t capacity = integer("PREFILL4K_ATTRIBUTION_CAPACITY", 8192, 1, 262144);
      const uint32_t warmups = integer("PREFILL4K_ATTRIBUTION_WARMUP", 1, 0, 4);
      const uint32_t repeats = integer("PREFILL4K_ATTRIBUTION_REPEATS", 1, 1, 8);
      const bool teacherCacheOnly = enabled("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");
      const uint32_t primeSetting = integer("PREFILL4K_ATTRIBUTION_TEACHER_PRIME", 1, 0, 1);
      require(tokens.size() <= capacity && rows >= 4, "prompt/arena exceeds capacity");
      const char *rawMode = std::getenv("PREFILL4K_ATTRIBUTION_MODE"); const std::string mode = rawMode ? rawMode : "normal";
      metal::CommandDispatchProfilingMode profileMode = metal::CommandDispatchProfilingMode::Off;
      if (mode == "command") profileMode = metal::CommandDispatchProfilingMode::Command;
      else if (mode == "stage") profileMode = metal::CommandDispatchProfilingMode::StagePerDispatch;
      else if (mode == "dispatch") profileMode = metal::CommandDispatchProfilingMode::DispatchBoundary;
      else require(mode == "normal", "invalid attribution mode");
      for (const auto &path : {std::string(argv[4]), std::string(argv[4]) + ".commands.jsonl", std::string(argv[4]) + ".trace.jsonl"})
        require(!std::filesystem::exists(path), "choose fresh report outputs");
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl"), trace(std::string(argv[4]) + ".trace.jsonl");
      require(commands && trace, "cannot create trace outputs");
      commands << std::setprecision(std::numeric_limits<double>::max_digits10);
      trace << std::setprecision(std::numeric_limits<double>::max_digits10);
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      const auto &descriptor = weights.descriptor();
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(capacity, rows, 4) + FlashForward::requestStateBytes(capacity) +
          FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 16ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE") && rows >= 256) planned += flashMoEBlockedWorkspacePlannedBytes(rows, 10);
      if (primeSetting) {
        planned += FlashMTPForward::workspacePlannedBytes(capacity, 128) + FlashMTPForward::requestStateBytes(capacity);
        if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashMTPForward::denseCachePlannedBytes(weights);
      }
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "governor denied arena before construction");
      FlashForward target(backend, weights, capacity, rows, 4);
      std::unique_ptr<FlashMTPForward> head;
      if (primeSetting) head = std::make_unique<FlashMTPForward>(backend, weights, capacity, 128);
      require(target.workspaceBytes() + FlashForward::requestStateBytes(capacity) +
          (head ? head->workspaceBytes() + FlashMTPForward::requestStateBytes(capacity) : 0) <= planned, "arenas exceed reservation");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto buffers = target.cachedOperandsOnly();
        if (head) { auto headBuffers = head->cachedOperandsOnly(); buffers.insert(buffers.end(), headBuffers.begin(), headBuffers.end()); }
        if (!buffers.empty()) residency = backend.requestWeightResidency(buffers, "prefill4k attribution saved operands only");
      }
      reservation->commit();
      (void)backend.takeCommandDispatchProfiles();
      metal::CommandHostTiming targetHost, primeHost;
      double targetGpu = 0, targetWall = 0, targetCall = 0, primeGpu = 0, primeWall = 0, primeCall = 0;
      uint64_t targetCommands = 0, primeCommands = 0, profiledCommands = 0;
      std::vector<uint32_t> output;
      std::vector<std::string> finalLogitHashes, finalHiddenHashes;
      const auto capture = [&](const char *phase, uint32_t repeat, uint32_t begin, uint32_t count,
                               metal::CommandTiming timing, double callSeconds) {
        commands << "{\"phase\":" << json::quote(phase) << ",\"repeat\":" << repeat << ",\"logical_begin\":" << begin
            << ",\"rows\":" << count << ",\"gpu_seconds\":" << timing.gpuSeconds << ",\"command_wall_seconds\":" << timing.wallSeconds
            << ",\"forward_call_seconds\":" << callSeconds << ",\"host_command_subphases\":"; writeHost(commands, timing.host); commands << "}\n";
        const auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == (profileMode == metal::CommandDispatchProfilingMode::Off ? 0u : 1u), "missing/extra full-command profiles");
        for (const auto &profile : profiles) {
          ++profiledCommands;
          trace << "{\"phase\":" << json::quote(phase) << ",\"repeat\":" << repeat << ",\"logical_begin\":" << begin << ",\"rows\":" << count
              << ",\"family_attribution\":"; writeFamilies(trace, profile, std::string_view(phase) == "target_prefill" ? descriptor.layers : 1);
          trace << ",\"command\":"; profiling::writeJson(trace, profile); trace << "}\n";
        }
        commands.flush(); trace.flush(); require(commands && trace, "could not write attribution records");
      };
      for (uint32_t trial = 0; trial < warmups + repeats; ++trial) {
        const bool measured = trial >= warmups;
        backend.setCommandDispatchProfiling(measured ? profileMode : metal::CommandDispatchProfilingMode::Off);
        auto state = target.createState();
        FlashMTPState headState;
        if (head) headState = head->createState();
        (void)backend.takeCommandDispatchProfiles();
        FlashForwardResult last;
        for (uint32_t begin = 0; begin < tokens.size(); begin += rows) {
          const uint32_t count = std::min<uint32_t>(rows, uint32_t(tokens.size()) - begin);
          const auto started = Clock::now();
          last = target.forward(state, std::span<const uint32_t>(tokens).subspan(begin, count), false, bool(head));
          const double call = std::chrono::duration<double>(Clock::now() - started).count();
          if (measured) {
            ++targetCommands; targetGpu += last.timing.gpuSeconds; targetWall += last.timing.wallSeconds; targetCall += call;
            targetHost.add(last.timing.host); capture("target_prefill", trial - warmups, begin, count, last.timing, call);
          } else (void)backend.takeCommandDispatchProfiles();
          require(last.logitRows == 1, "target prefill unexpectedly computed multiple vocabulary rows");
          if (head) {
            const uint32_t primeRows = std::min<uint32_t>(count, uint32_t(tokens.size()) - begin - 1);
            require(last.hiddenBF16 && last.hiddenBF16.sizeBytes() >= uint64_t{count} * 10240 * 2, "target hidden extent invalid");
            for (uint32_t primeBegin = 0; primeBegin < primeRows; primeBegin += 128) {
              const uint32_t primeCount = std::min<uint32_t>(128, primeRows - primeBegin);
              const auto hidden = backend.view(last.hiddenBF16, uint64_t{primeBegin} * 10240 * 2, uint64_t{primeCount} * 10240 * 2);
              const auto primeStarted = Clock::now();
              FlashMTPResult primed;
              const auto next = std::span<const uint32_t>(tokens).subspan(begin + primeBegin + 1, primeCount);
              if (teacherCacheOnly) primed.timing = head->primeTeacherCache(headState, hidden, next);
              else primed = head->forward(headState, hidden, next, FlashMTPLogits::None);
              const double primeSeconds = std::chrono::duration<double>(Clock::now() - primeStarted).count();
              require(!primed.logitsBF16 && primed.logitRows == 0, "teacher priming unexpectedly computed vocabulary rows");
              if (measured) {
                ++primeCommands; primeGpu += primed.timing.gpuSeconds; primeWall += primed.timing.wallSeconds; primeCall += primeSeconds;
                primeHost.add(primed.timing.host); capture("teacher_prime", trial - warmups, begin + primeBegin, primeCount, primed.timing, primeSeconds);
              } else (void)backend.takeCommandDispatchProfiles();
            }
          }
        }
        require(state.logicalLength() == tokens.size() && (!head || headState.logicalLength() + 1 == tokens.size()), "true prompt/head offsets differ");
        if (!measured) continue;
        require(last.greedyResultsU32 && last.greedyResultsU32.contents(), "current profile GPU greedy result absent");
        output.push_back(greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(last.greedyResultsU32.contents()), descriptor.vocabularySize));
        finalLogitHashes.push_back(digest(last.logitsBF16.contents(), uint64_t{descriptor.vocabularySize} * 2));
        if (head) finalHiddenHashes.push_back(digest(last.hiddenBF16.contents(), last.hiddenBF16.sizeBytes()));
      }
      backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
      std::ofstream report(argv[4]); require(bool(report), "cannot write report");
      report << std::setprecision(std::numeric_limits<double>::max_digits10);
      report << "{\"schema\":\"splash-prefill4k-attribution-v1\",\"execution_complete\":true,\"gpu_executed\":true"
          << ",\"scope\":\"target prefill plus exact 128-row singleton teacher priming; excludes Worker controls and HTTP\""
          << ",\"counter_modes_are_diagnostic\":true,\"profile_mode\":" << json::quote(mode) << ",\"capacity\":" << capacity
          << ",\"maximum_rows\":" << rows << ",\"warmup_trials\":" << warmups << ",\"measured_trials\":" << repeats
          << ",\"prompt_tokens\":" << tokens.size() << ",\"teacher_prime_enabled\":" << (head ? "true" : "false")
          << ",\"token_json_sha256\":" << json::quote(fileDigest(argv[3])) << ",\"teacher_cache_only\":" << (teacherCacheOnly ? "true" : "false")
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity()) << ",\"kernel_routes\":" << json::quote(target.kernelRoutes())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256())) << ",\"tokens_u32le_sha256\":" << json::quote(digest(tokens.data(), tokens.size() * sizeof(uint32_t)))
          << ",\"target_commands\":" << targetCommands << ",\"prime_commands\":" << primeCommands << ",\"profiled_commands\":" << profiledCommands
          << ",\"target_gpu_seconds\":" << targetGpu << ",\"target_command_wall_seconds\":" << targetWall << ",\"target_forward_call_seconds\":" << targetCall
          << ",\"prime_gpu_seconds\":" << primeGpu << ",\"prime_command_wall_seconds\":" << primeWall << ",\"prime_forward_call_seconds\":" << primeCall
          << ",\"target_host_command_subphases\":"; writeHost(report, targetHost);
      report << ",\"prime_host_command_subphases\":"; writeHost(report, primeHost);
      report << ",\"first_tokens\":[";
      for (size_t index = 0; index < output.size(); ++index) { if (index) report << ','; report << output[index]; }
      report << "],\"final_logits_sha256\":[";
      for (size_t index = 0; index < finalLogitHashes.size(); ++index) { if (index) report << ','; report << json::quote(finalLogitHashes[index]); }
      report << "],\"final_hidden_sha256\":[";
      for (size_t index = 0; index < finalHiddenHashes.size(); ++index) { if (index) report << ','; report << json::quote(finalHiddenHashes[index]); }
      report << "]}\n"; require(bool(report), "report write failed");
      std::cout << "prefill4k attribution complete; mode=" << mode << " report=" << argv[4] << '\n';
      return 0;
    } catch (const std::exception &error) { std::cerr << "prefill4k attribution failed: " << error.what() << '\n'; return 1; }
  }
}
