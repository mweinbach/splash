// Root-only real target singleton ordinary AR attribution. Host build/help/CPU
// checks create no Metal backend and touch no model payload.
#include "flash/FlashForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
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
constexpr uint32_t kContext = 2048, kRows = 1, kCapacity = 8192, kVocabulary = 248320;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
bool enabled(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw) return false;
  require(std::string_view(raw) == "0" || std::string_view(raw) == "1", "switch must be exactly0 or1");
  return std::string_view(raw) == "1";
}
uint32_t integer(const char *name, uint32_t fallback, uint32_t low, uint32_t high) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  uint64_t value = 0;
  require(*raw != 0, "empty diagnostic integer");
  for (const char *p = raw; *p; ++p) {
    require(*p >= '0' && *p <= '9', "diagnostic integer must be decimal");
    value = value * 10 + unsigned(*p - '0');
    require(value <= high, "diagnostic integer exceeds limit");
  }
  require(value >= low, "diagnostic integer below limit");
  return uint32_t(value);
}
std::vector<uint32_t> loadTokens(const std::string &path) {
  require(std::filesystem::file_size(path) <= 2ULL << 20, "token fixture exceeds bounded2MiB");
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read token fixture");
  NSError *error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSArray class]], "tokens must be a JSON array");
  std::vector<uint32_t> result;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(),
            "token must be an integer, not a boolean");
    const double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < kVocabulary && std::floor(value) == value, "invalid token ID");
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), "token fixture empty");
  return result;
}
std::string digest(const void *bytes, uint64_t count) {
  CC_SHA256_CTX context{};
  CC_SHA256_Init(&context);
  const auto *next = static_cast<const uint8_t *>(bytes);
  while (count) {
    const CC_LONG part = CC_LONG(std::min<uint64_t>(count, UINT32_MAX));
    CC_SHA256_Update(&context, next, part);
    next += part;
    count -= part;
  }
  std::array<uint8_t, 32> result{};
  CC_SHA256_Final(result.data(), &context);
  constexpr char hex[] = "0123456789abcdef";
  std::string output;
  for (uint8_t byte : result) { output += hex[byte >> 4]; output += hex[byte & 15]; }
  return output;
}
std::string fileDigest(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot read source/artifact provenance");
  CC_SHA256_CTX context{};
  CC_SHA256_Init(&context);
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    if (input.gcount()) CC_SHA256_Update(&context, buffer.data(), CC_LONG(input.gcount()));
  }
  require(input.eof(), "artifact provenance read failed");
  std::array<uint8_t, 32> result{};
  CC_SHA256_Final(result.data(), &context);
  constexpr char hex[] = "0123456789abcdef";
  std::string output;
  for (uint8_t byte : result) { output += hex[byte >> 4]; output += hex[byte & 15]; }
  return output;
}
std::string bufferDigest(const metal::MetalBuffer &buffer, uint64_t bytes) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= bytes, "diagnostic hash view short");
  return digest(buffer.contents(), bytes);
}
uint32_t nextToken(const FlashForwardResult &result, uint32_t row = 0) {
  if (result.greedyResultsU32) {
    require(result.greedyRows > row && result.greedyResultsU32.contents(), "compact greedy record missing");
    return greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(result.greedyResultsU32.contents())[row], kVocabulary);
  }
  require(result.logitRows > row && result.logitsBF16 && result.logitsBF16.contents(), "greedy logits missing");
  const auto *values = static_cast<const uint16_t *>(result.logitsBF16.contents()) + uint64_t{row} * kVocabulary;
  float maximum = -INFINITY;
  uint32_t best = 0;
  for (uint32_t token = 0; token < kVocabulary; ++token) {
    const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
    require(std::isfinite(value), "nonfinite logits");
    if (value > maximum) { best = token; maximum = value; }
  }
  return best;
}
void tokensJSON(std::ostream &out, std::span<const uint32_t> tokens) {
  out << '[';
  for (size_t i = 0; i < tokens.size(); ++i) { if (i) out << ','; out << tokens[i]; }
  out << ']';
}
struct Family { uint64_t calls = 0, timed = 0; double seconds = 0; };
struct Classifier {
  bool insideMoE = false, shared = false;
  uint64_t routes = 0;
  static bool dense(std::string_view name) {
    return name.starts_with("flash_affine") || name.starts_with("flash_dense") || name.starts_with("flash_float_dense");
  }
  std::string family(std::string_view name, const metal::CommandDispatchTimestamp &metadata) {
    if (name.starts_with("flash_greedy")) return "greedy";
    if (name.starts_with("flash_int8_head") || name.starts_with("flash_bf16_q8_head")) return "vocabulary";
    if (name == "flash_affine_embedding") return "embedding";
    if (dense(name)) {
      const uint32_t outputIndex = name.starts_with("flash_affine") ? 5 : 2;
      for (const auto &binding : metadata.bindings)
        if (!binding.inlineBytes && binding.index == outputIndex && binding.sizeBytes == uint64_t{kRows} * kVocabulary * 2)
          return "vocabulary";
    }
    if (name == "flash_moe_route") { insideMoE = true; shared = false; ++routes; return "moe"; }
    if ((name == "flash_moe_combine" || name.starts_with("private_moe_combine"))) { insideMoE = shared = false; return "moe"; }
    if (name.starts_with("flash_shared_expert")) { shared = true; return "shared_expert"; }
    if (insideMoE && dense(name)) shared = true;
    if (shared && (dense(name) || name == "flash_moe_silu_multiply")) return "shared_expert";
    if (name.starts_with("flash_gdn")) return "gdn";
    if (name.starts_with("flash_qsa_out_f32")) return "dense";
    if (name.starts_with("flash_qsa")) return "qsa";
    if (name.starts_with("flash_ple")) return "ple";
    if (name.starts_with("flash_hc") || name.starts_with("flash_forward_hc")) return "hc";
    if (name == "flash_forward_copy_words") return "copies";
    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") ||
        name.starts_with("flash_int8_expert") || name.starts_with("private_moe") || name.starts_with("flash_gathered_mpp")) return "moe";
    if (dense(name)) return "dense";
    return "other";
  }
};
void writeFamilies(std::ostream &out, const metal::CommandDispatchProfile &profile) {
  Classifier classifier;
  std::map<std::string, Family> families, pipelines;
  uint64_t timed = 0;
  double timedSeconds = 0;
  for (const auto &dispatch : profile.dispatches) {
    const auto name = classifier.family(dispatch.pipelineName, dispatch);
    auto &family = families[name], &pipeline = pipelines[dispatch.pipelineName];
    ++family.calls; ++pipeline.calls;
    if (dispatch.timestampsValid) {
      ++family.timed; ++pipeline.timed; ++timed;
      family.seconds += dispatch.gpuSeconds; pipeline.seconds += dispatch.gpuSeconds;
      timedSeconds += dispatch.gpuSeconds;
    }
  }
  const bool complete = !profile.dispatchMetadataTruncated && profile.dispatches.size() == profile.dispatchCount && classifier.routes == 48;
  out << "{\"classification_scope\":\"pipeline names, full vocabulary output extents, shared expert interval between routed down and combine; QSA output projection is dense\""
      << ",\"classification_complete\":" << (complete ? "true" : "false")
      << ",\"model_layer_routes\":" << classifier.routes << ",\"timed_dispatches\":" << timed
      << ",\"full_command_gpu_seconds\":" << profile.timing.gpuSeconds
      << ",\"sum_timed_dispatch_gpu_seconds\":" << timedSeconds << ",\"families\":{";
  const auto fields = [&](const auto &values) {
    bool comma = false;
    for (const auto &[name, family] : values) {
      if (comma) out << ','; comma = true;
      out << json::quote(name) << ":{\"calls\":" << family.calls << ",\"timed_calls\":" << family.timed
          << ",\"gpu_seconds\":" << family.seconds << '}';
    }
  };
  fields(families);
  out << "},\"pipelines\":{"; fields(pipelines); out << "}}";
}
void usage() {
  std::cout << "ar1-attribution --gpu METALLIB PACKAGE PROMPT2048_JSON NEW_REPORT_JSON\n"
      << "--help and --cpu-self-test create no device/backend or model load. CommandTiming ABI200.\n"
      << "DECODE_SEP21_AR_MODE=normal|command|stage|dispatch (default stage), WARMUP=1, REPEATS=1.\n"
      << "Profiles only singleton target.verify on fresh begin2048 state with4 real rows.\n"
      << "One ordinary greedy token is derived once on disposable state. The unprofiled warmup is recorded alongside profiled repeats.\n"
      << "All timing is diagnostic; stage/dispatch modes alter command scheduling. Existing profile+Full512 flags required.\n";
}
void cpuSelfTest() {
  Classifier classifier;
  metal::CommandDispatchTimestamp metadata;
  require(classifier.family("flash_gathered_mpp_gate_up_m16_n64_sg4", metadata) == "moe", "gathered family mismatch");
  require(classifier.family("flash_int8_expert_store_down_scatter_m16_n64", metadata) == "moe", "bucket family mismatch");
  require(classifier.family("flash_qsa_out_f32_n32_m8_n32_s4", metadata) == "dense", "QSA output projection mismatch");
  require(classifier.family("flash_int8_head_m8_n64", metadata) == "vocabulary", "vocabulary family mismatch");
  require(classifier.family("flash_moe_route", metadata) == "moe", "route family mismatch");
  require(classifier.family("flash_float_dense_small_rows_pad", metadata) == "shared_expert", "shared family mismatch");
  require(classifier.family("flash_moe_combine", metadata) == "moe", "combine family mismatch");
  require(classifier.family("flash_gdn_decode_persistent512", metadata) == "gdn", "GDN family mismatch");
  require(classifier.family("flash_hc_norm_bf16_weight", metadata) == "hc", "HC family mismatch");
  std::cout << "{\"cpu_self_test_pass\":true,\"gpu_work\":false,\"command_timing_abi_bytes\":" << sizeof(metal::CommandTiming) << "}\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      static_assert(sizeof(metal::CommandTiming) == 200, "Private host/core objects require normal timing ABI200");
      if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      require((argc == 6) && std::string_view(argv[1]) == "--gpu", "use --gpu and required paths; see --help");
      const std::string library = argv[2], package = argv[3], promptPath = argv[4], reportPath = argv[5];
      const uint32_t repeats = integer("DECODE_SEP21_AR_REPEATS", 1, 1, 8);
      const uint32_t warmup = integer("DECODE_SEP21_AR_WARMUP", 1, 0, 2);
      require(enabled("SPLASH_FLASH_ALLROWS_FULL512_TARGET"), "requires private Full512 all-target policy");
      (void)gathered_mpp::requested(); (void)gathered_mpp::requestedMaximumRows();
      const auto prompt = loadTokens(promptPath);
      require(prompt.size() == kContext, "prompt must contain exactly2048 true tokens");
      std::vector<uint32_t> incoming;
      for (const auto &path : {reportPath, reportPath + ".trace.jsonl"})
        require(!std::filesystem::exists(path), "report exists; choose fresh output");
      const char *rawMode = std::getenv("DECODE_SEP21_AR_MODE");
      const std::string mode = rawMode ? rawMode : "stage";
      auto profileMode = metal::CommandDispatchProfilingMode::Off;
      if (mode == "command") profileMode = metal::CommandDispatchProfilingMode::Command;
      else if (mode == "stage") profileMode = metal::CommandDispatchProfilingMode::StagePerDispatch;
      else if (mode == "dispatch") profileMode = metal::CommandDispatchProfilingMode::DispatchBoundary;
      else require(mode == "normal", "invalid diagnostic mode");
      std::ofstream trace(reportPath + ".trace.jsonl");
      require(bool(trace), "cannot create trace output");
      trace << std::setprecision(std::numeric_limits<double>::max_digits10);
      metal::MetalBackend backend(library);
      const auto weights = FlashWeights::load(backend, package);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory, reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(kCapacity, kContext, kRows) +
          FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) +
          FlashForward::int8HeadPlannedBytes(weights) + FlashForward::requestStateBytes(kCapacity) + (16ULL << 20);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(kContext, 10);
      auto reservation = governor.tryReserve(planned);
      require(bool(reservation), "governor denied diagnostic target/state arenas");
      FlashForward target(backend, weights, kCapacity, kContext, kRows);
      require(target.workspaceBytes() + FlashForward::requestStateBytes(kCapacity) <= planned, "actual diagnostic arena exceeds reservation");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto operands = target.cachedOperandsOnly();
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "sep21 ordinary AR saved target operands only");
      }
      reservation->commit();
      if (incoming.empty()) {
        auto seed = target.createState();
        auto generated = target.forward(seed, prompt);
        incoming.push_back(nextToken(generated));
        for (uint32_t row = 1; row < kRows; ++row) {
          const uint32_t previous = incoming.back();
          generated = target.forward(seed, std::span<const uint32_t>(&previous, 1));
          incoming.push_back(nextToken(generated));
        }
      }
      std::vector<std::string> samples;
      std::string stableOutput, stableHidden, stableContinuation, stablePrefill;
      for (uint32_t trial = 0; trial < warmup + repeats; ++trial) {
        backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
        auto state = target.createState();
        const auto prefilling = target.forward(state, prompt);
        require(state.logicalLength() == kContext && !state.poisoned(), "fresh prefill state invalid");
        const auto prefillHash = bufferDigest(prefilling.logitsBF16, uint64_t{kVocabulary} * 2);
        (void)backend.takeCommandDispatchProfiles();
        const bool measured = trial >= warmup;
        backend.setCommandDispatchProfiling(measured ? profileMode : metal::CommandDispatchProfilingMode::Off);
        const auto start = Clock::now();
        const auto result = target.forward(state, incoming);
        const double call = std::chrono::duration<double>(Clock::now() - start).count();
        const auto profiles = backend.takeCommandDispatchProfiles();
        backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
        require(result.logitRows == kRows && !state.poisoned(), "ordinary AR geometry/state invalid");
        require(profiles.size() == (measured && profileMode != metal::CommandDispatchProfilingMode::Off ? 1u : 0u), "missing/extra command profile");
        const auto outputHash = bufferDigest(result.logitsBF16, uint64_t{kRows} * kVocabulary * 2);
        const auto hiddenHash = result.hiddenBF16 ? bufferDigest(result.hiddenBF16, uint64_t{kRows} * 10240 * 2) : std::string{};
        std::array<uint32_t, kRows> predictions{};
        for (uint32_t row = 0; row < kRows; ++row) predictions[row] = nextToken(result, row);
        require(state.logicalLength() == kContext + kRows && !state.poisoned(), "ordinary AR state did not advance exactly one token");
        const uint32_t next = predictions.back();
        const auto continuation = target.forward(state, std::span<const uint32_t>(&next, 1));
        const auto continuationHash = bufferDigest(continuation.logitsBF16, uint64_t{kVocabulary} * 2);
        require(state.logicalLength() == kContext + kRows + 1 && !state.poisoned(), "continuation state failed");
        if (trial == 0) {
          stableOutput = outputHash; stableHidden = hiddenHash; stableContinuation = continuationHash; stablePrefill = prefillHash;
        } else require(outputHash == stableOutput && hiddenHash == stableHidden && continuationHash == stableContinuation && prefillHash == stablePrefill,
                       "fresh-state repeated output/continuation mismatch");
        std::ostringstream record;
        record << std::setprecision(std::numeric_limits<double>::max_digits10);
        record << "{\"trial\":" << trial << ",\"warmup\":" << (measured ? "false" : "true") << ",\"begin\":2048,\"real_rows\":1,\"gpu_seconds\":" << result.timing.gpuSeconds
            << ",\"command_wall_seconds\":" << result.timing.wallSeconds << ",\"forward_call_seconds\":" << call
            << ",\"logits_sha256\":" << json::quote(outputHash) << ",\"hidden_sha256\":" << json::quote(hiddenHash)
            << ",\"prefill_last_logits_sha256\":" << json::quote(prefillHash) << ",\"continuation_logits_sha256\":" << json::quote(continuationHash)
            << ",\"predicted_tokens\":"; tokensJSON(record, predictions);
        record << ",\"ordinary_ar_state_advanced\":true,\"healthy_continuation_length\":" << state.logicalLength();
        if (!profiles.empty()) {
          record << ",\"family_attribution\":"; writeFamilies(record, profiles[0]);
          trace << "{\"trial\":" << trial << ",\"family_attribution\":";
          writeFamilies(trace, profiles[0]); trace << ",\"command\":"; profiling::writeJson(trace, profiles[0]); trace << "}\n";
        }
        record << '}'; samples.push_back(record.str());
      }
      const auto *store = target.batchInt8ExpertStore();
      require(store, "private Full512 Store absent");
      std::ofstream report(reportPath);
      require(bool(report), "cannot create report");
      report << std::setprecision(std::numeric_limits<double>::max_digits10);
      report << "{\"schema\":\"splash-sep21-singleton-ordinary-ar1-attribution-v1\",\"pass\":true,\"diagnostic_timing_only\":true"
          << ",\"profile_mode\":" << json::quote(mode) << ",\"command_timing_abi_bytes\":" << sizeof(metal::CommandTiming)
          << ",\"metallib_sha256\":" << json::quote(fileDigest(library)) << ",\"executable_sha256\":" << json::quote(fileDigest(argv[0]))
          << ",\"prompt_fixture_sha256\":" << json::quote(fileDigest(promptPath)) << ",\"prompt_tokens_sha256\":" << json::quote(digest(prompt.data(), prompt.size() * 4))
          << ",\"target_derivative_sha256\":" << json::quote(store->numericalIdentitySha256())
          << ",\"target_policy_routes\":" << json::quote(target.kernelRoutes())
          << ",\"gathered_enabled\":" << (store->gatheredMPPEnabled() ? "true" : "false")
          << ",\"gathered_maximum_physical_rows\":" << store->gatheredMPPMaximumRows()
          << ",\"expert_id_capture_adds48_diagnostic_copies\":" << (enabled("SPLASH_FLASH_CAPTURE_EXPERT_IDS") ? "true" : "false")
          << ",\"incoming_policy\":" << json::quote("one ordinary target greedy token on untimed disposable state; no MTP verification or trained proposals")
          << ",\"incoming_tokens\":"; tokensJSON(report, incoming);
      report << ",\"prefill_tokens\":2048,\"forward_begin\":2048,\"real_rows\":1,\"warmups\":" << warmup << ",\"repeats\":" << repeats << ",\"samples\":[";
      for (size_t i = 0; i < samples.size(); ++i) { if (i) report << ','; report << samples[i]; }
      report << "]}\n";
      require(bool(report) && bool(trace), "report/trace write failed");
      std::cout << "{\"pass\":true,\"report\":" << json::quote(reportPath) << ",\"samples\":" << samples.size() << "}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
