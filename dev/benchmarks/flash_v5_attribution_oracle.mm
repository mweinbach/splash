// Root-run diagnostic: compilation and --help perform no Metal inference.
#include "flash/FlashForward.hpp"
#include "flash/FlashBatchPrefill.hpp"
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
#include <iostream>
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
  std::string family(std::string_view name) {
    if (name == "flash_moe_route") { insideMoE = true; shared = false; ++routes; return "moe"; }
    if (name == "flash_moe_combine") { insideMoE = shared = false; return "moe"; }
    if (insideMoE && dense(name)) shared = true;
    if (shared && (dense(name) || name == "flash_moe_silu_multiply")) return "shared_expert";
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
    auto &f = families[classifier.family(dispatch.pipelineName)]; ++f.dispatches;
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
  std::cout << "usage: flash-v5-attribution-oracle METALLIB PACKAGE TOKENS_JSON REPORT_JSON [PROVENANCE_JSON]\n"
      "Root-only GPU run. --help creates no Metal backend.\n"
      "FLASH_V5_ATTRIBUTION_MODE=normal|command|stage|dispatch (default normal).\n"
      "All modes retain normal CommandTiming.host subphases. Stage/dispatch alter encoding and are descriptive only.\n"
      "FLASH_V5_ATTRIBUTION_LANES=1..4 (default 1); optional _LANE_TOKEN_PREFIX reads PREFIX0.json..PREFIX3.json.\n"
      "_PREFILL_ROWS=2048, _CAPACITY=8192, _VERIFY_ROWS=16, _WARMUP=1, _REPEATS=1, _DECODE_STEPS=0 defaults.\n"
      "Uses supplied v5 profile flags and saved operand/expert dirs; requests cached-operands-only residency when enabled.\n"
      "Traces cover the complete 48-layer target trunk, 12 QSA/36 GDN, real batched prefills, and optional AR continuation.\n"
      "No service scheduler/MTP-head engine is instantiated; this is attribution, not an HTTP performance score.\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
      require(argc == 5 || argc == 6, "usage: flash-v5-attribution-oracle METALLIB PACKAGE TOKENS_JSON REPORT_JSON [PROVENANCE_JSON]");
      const auto tokens = loadTokens(argv[3]);
      const uint32_t lanes = integer("FLASH_V5_ATTRIBUTION_LANES", 1, 1, 4);
      const uint32_t rows = integer("FLASH_V5_ATTRIBUTION_PREFILL_ROWS", 2048, 1, 2048);
      const uint32_t capacity = integer("FLASH_V5_ATTRIBUTION_CAPACITY", 8192, 1, 262144);
      const uint32_t verifyRows = integer("FLASH_V5_ATTRIBUTION_VERIFY_ROWS", 16, 0, 16);
      const uint32_t warmups = integer("FLASH_V5_ATTRIBUTION_WARMUP", 1, 0, 4);
      const uint32_t repeats = integer("FLASH_V5_ATTRIBUTION_REPEATS", 1, 1, 8);
      const uint32_t decodeSteps = integer("FLASH_V5_ATTRIBUTION_DECODE_STEPS", 0, 0, 128);
      require(verifyRows <= rows && tokens.size() + decodeSteps <= capacity, "attribution rows/context budget is invalid");
      const char *rawMode = std::getenv("FLASH_V5_ATTRIBUTION_MODE"); const std::string mode = rawMode ? rawMode : "normal";
      metal::CommandDispatchProfilingMode profileMode = metal::CommandDispatchProfilingMode::Off;
      if (mode == "command") profileMode = metal::CommandDispatchProfilingMode::Command;
      else if (mode == "stage") profileMode = metal::CommandDispatchProfilingMode::StagePerDispatch;
      else if (mode == "dispatch") profileMode = metal::CommandDispatchProfilingMode::DispatchBoundary;
      else require(mode == "normal", "invalid attribution mode");
      std::array<std::vector<uint32_t>, 4> laneTokens;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const char *prefix = std::getenv("FLASH_V5_ATTRIBUTION_LANE_TOKEN_PREFIX");
        laneTokens[lane] = prefix ? loadTokens(std::string(prefix) + std::to_string(lane) + ".json") : tokens;
        require(laneTokens[lane].size() == tokens.size(), "attribution requires uniform true lane prompt lengths");
      }
      for (const auto &path : {std::string(argv[4]), std::string(argv[4]) + ".commands.jsonl", std::string(argv[4]) + ".trace.jsonl"})
        require(!std::filesystem::exists(path), "attribution output exists; choose a fresh report name");
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl"), trace(std::string(argv[4]) + ".trace.jsonl");
      require(commands && trace, "cannot create fresh attribution trace outputs");
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      const auto &descriptor = weights.descriptor();
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(capacity, rows, verifyRows) +
          uint64_t{lanes} * FlashForward::requestStateBytes(capacity) + FlashForward::expertCachePlannedBytes(weights) +
          FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 16ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE") && rows >= 256) planned += flashMoEBlockedWorkspacePlannedBytes(rows, 10);
      if (lanes > 1) planned += FlashBatchPrefill::workspacePlannedBytes(capacity, lanes, rows);
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "governor denied attribution arena before construction");
      FlashForward forward(backend, weights, capacity, rows, verifyRows);
      std::unique_ptr<FlashBatchPrefill> batch;
      if (lanes > 1) batch = std::make_unique<FlashBatchPrefill>(backend, weights, forward, capacity, lanes, rows);
      require(forward.workspaceBytes() + (batch ? batch->workspaceBytes() : 0) + uint64_t{lanes} * FlashForward::requestStateBytes(capacity) <= planned,
          "attribution arenas exceed reserved plan");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        const auto buffers = forward.cachedOperandsOnly();
        if (!buffers.empty()) residency = backend.requestWeightResidency(buffers, "v5 attribution saved operands only");
      }
      reservation->commit();
      metal::CommandHostTiming prefillHost, decodeHost;
      double prefillGpu = 0, prefillWall = 0, prefillCall = 0, decodeGpu = 0, decodeWall = 0;
      uint64_t prefillCommands = 0, decodeCommands = 0, profiledCommands = 0;
      const auto capture = [&](const char *phase, uint32_t repeat, uint32_t actualLanes, uint32_t realRows,
                               metal::CommandTiming timing, double callSeconds) {
        commands << "{\"phase\":" << json::quote(phase) << ",\"repeat\":" << repeat << ",\"lanes\":" << actualLanes
            << ",\"real_rows_per_lane\":" << realRows << ",\"gpu_seconds\":" << timing.gpuSeconds
            << ",\"command_wall_seconds\":" << timing.wallSeconds << ",\"forward_call_seconds\":" << callSeconds
            << ",\"host_command_subphases\":"; writeHost(commands, timing.host); commands << "}\n";
        const auto profiles = backend.takeCommandDispatchProfiles();
        require(profiles.size() == (profileMode == metal::CommandDispatchProfilingMode::Off ? 0u : 1u), "missing or extra full-command profiles");
        for (const auto &profile : profiles) {
          ++profiledCommands;
          trace << "{\"phase\":" << json::quote(phase) << ",\"repeat\":" << repeat << ",\"lanes\":" << actualLanes
              << ",\"real_rows_per_lane\":" << realRows
              << ",\"attribution_scope\":\"descriptive counter/profile durations; stage/dispatch alter encoder scheduling; no throughput claim\",\"family_attribution\":";
          writeFamilies(trace, profile, descriptor.layers); trace << ",\"command\":"; profiling::writeJson(trace, profile); trace << "}\n";
        }
        commands.flush(); trace.flush(); require(commands && trace, "could not write attribution records");
      };
      std::vector<std::vector<uint32_t>> output(lanes);
      for (uint32_t trial = 0; trial < warmups + repeats; ++trial) {
        const bool measured = trial >= warmups;
        backend.setCommandDispatchProfiling(measured ? profileMode : metal::CommandDispatchProfilingMode::Off);
        std::array<FlashRequestState, 4> states;
        std::vector<FlashRequestState *> pointers;
        for (uint32_t lane = 0; lane < lanes; ++lane) { states[lane] = forward.createState(); pointers.push_back(&states[lane]); }
        metal::MetalBuffer logits, greedyRecords;
        for (uint32_t begin = 0; begin < tokens.size(); begin += rows) {
          const uint32_t count = std::min<uint32_t>(rows, uint32_t(tokens.size()) - begin);
          std::vector<uint32_t> incoming;
          for (uint32_t lane = 0; lane < lanes; ++lane)
            incoming.insert(incoming.end(), laneTokens[lane].begin() + begin, laneTokens[lane].begin() + begin + count);
          const auto started = Clock::now(); metal::CommandTiming timing;
          if (batch) {
            const auto result = batch->forwardBatch(pointers, incoming, count);
            timing = result.timing; logits = result.logitsBF16; greedyRecords = result.greedyResultsU32;
          } else {
            const auto result = forward.forward(states[0], incoming);
            timing = result.timing; logits = result.logitsBF16; greedyRecords = result.greedyResultsU32;
          }
          const double call = std::chrono::duration<double>(Clock::now() - started).count();
          if (measured) {
            ++prefillCommands; prefillGpu += timing.gpuSeconds; prefillWall += timing.wallSeconds; prefillCall += call;
            prefillHost.add(timing.host); capture("prefill", trial - warmups, lanes, count, timing, call);
          } else (void)backend.takeCommandDispatchProfiles();
        }
        if (!measured) continue;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          require(states[lane].logicalLength() == tokens.size() && forward.ownsState(states[lane]), "prefill consumed an incorrect lane length");
          uint32_t next = 0;
          if (greedyRecords && greedyRecords.contents())
            next = greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(greedyRecords.contents())[lane], descriptor.vocabularySize);
          else {
            const auto *values = static_cast<const uint16_t *>(logits.contents()) + uint64_t{lane} * descriptor.vocabularySize;
            float maximum = -INFINITY;
            for (uint32_t token = 0; token < descriptor.vocabularySize; ++token) {
              const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
              require(std::isfinite(value), "nonfinite attribution logits"); if (value > maximum) { maximum = value; next = token; }
            }
          }
          output[lane].push_back(next);
          for (uint32_t step = 0; step < decodeSteps && next != 248044 && next != 248046; ++step) {
            const auto started = Clock::now(); const auto result = forward.forward(states[lane], std::span<const uint32_t>(&next, 1));
            const double call = std::chrono::duration<double>(Clock::now() - started).count();
            ++decodeCommands; decodeGpu += result.timing.gpuSeconds; decodeWall += result.timing.wallSeconds; decodeHost.add(result.timing.host);
            capture("autoregressive_continuation", trial - warmups, 1, 1, result.timing, call);
            require(result.greedyResultsU32 && result.greedyResultsU32.contents(), "GPU greedy continuation needs SPLASH_FLASH_GPU_GREEDY=1");
            next = greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(result.greedyResultsU32.contents()), descriptor.vocabularySize);
            output[lane].push_back(next);
          }
        }
        require(!batch || batch->canariesIntact(), "batched attribution arena guards changed");
      }
      uint32_t gdn = 0, qsa = 0;
      for (auto kind : descriptor.layerKinds) kind == FlashLayerKind::GatedDeltaNet ? ++gdn : ++qsa;
      const auto persisted = forward.persistedOperandStatus(); const auto *experts = forward.batchInt8ExpertStore();
      std::ofstream report(argv[4]); require(bool(report), "cannot write attribution report");
      report << "{\"schema\":\"splash-v5-full-model-attribution-v1\",\"execution_complete\":true,\"gpu_executed\":true"
          << ",\"execution_scope\":\"complete target trunk, real prefill lanes and optional sequential AR continuation; no HTTP scheduler or trained MTP head\""
          << ",\"timing_scope\":\"diagnostic only; counter stage/dispatch modes perturb encoding and never qualify HTTP performance\""
          << ",\"profile_mode\":" << json::quote(mode) << ",\"capacity\":" << capacity << ",\"lanes\":" << lanes
          << ",\"maximum_rows_per_lane\":" << rows << ",\"verify_arena_rows\":" << verifyRows
          << ",\"warmup_trials\":" << warmups << ",\"measured_trials\":" << repeats << ",\"prompt_tokens_per_lane\":" << tokens.size()
          << ",\"model_layers\":" << descriptor.layers << ",\"gdn_layers\":" << gdn << ",\"qsa_layers\":" << qsa
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint_sha256\":" << json::quote(weights.manifestFingerprint())
          << ",\"kernel_routes\":" << json::quote(forward.kernelRoutes())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256()))
          << ",\"token_json_sha256\":" << json::quote(fileDigest(argv[3]))
          << ",\"tokens_u32le_sha256\":" << json::quote(digest(tokens.data(), tokens.size() * sizeof(uint32_t)))
          << ",\"provenance_json_sha256\":" << (argc == 6 ? json::quote(fileDigest(argv[5])) : "null")
          << ",\"saved_residency_requested\":" << (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT") ? "true" : "false")
          << ",\"saved_resident_buffers\":" << residency.bufferCount() << ",\"saved_resident_bytes\":" << residency.byteCount()
          << ",\"selected_expert_identity_sha256\":" << (experts ? json::quote(experts->identitySha256()) : "null")
          << ",\"selected_expert_mapped_bytes\":" << (experts ? experts->mappedBytes() : 0)
          << ",\"persisted_operand_manifest_sha256\":" << json::quote(persisted.storeManifestSha256)
          << ",\"persisted_bf16_tensors\":" << persisted.bf16Tensors
          << ",\"persisted_f32_tensors\":" << persisted.f32Tensors
          << ",\"persisted_bf16_payload_bytes\":" << persisted.bf16PayloadBytes
          << ",\"persisted_f32_payload_bytes\":" << persisted.f32PayloadBytes
          << ",\"prefill_commands\":" << prefillCommands << ",\"decode_commands\":" << decodeCommands
          << ",\"profiled_commands\":" << profiledCommands << ",\"prefill_gpu_seconds\":" << prefillGpu
          << ",\"prefill_command_wall_seconds\":" << prefillWall << ",\"prefill_forward_call_seconds\":" << prefillCall
          << ",\"prefill_host_command_subphases\":"; writeHost(report, prefillHost);
      report << ",\"decode_gpu_seconds\":" << decodeGpu << ",\"decode_command_wall_seconds\":" << decodeWall
          << ",\"decode_host_command_subphases\":"; writeHost(report, decodeHost);
      report << ",\"first_and_continuation_tokens\":[";
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        if (lane) report << ','; report << '[';
        for (size_t token = 0; token < output[lane].size(); ++token) { if (token) report << ','; report << output[lane][token]; }
        report << ']';
      }
      report << "]}\n"; require(bool(report), "attribution report write failed");
      std::cout << "v5 attribution execution completed; mode=" << mode << " lanes=" << lanes << " report=" << argv[4] << '\n';
      return 0;
    } catch (const std::exception &error) { std::cerr << "v5 attribution failed: " << error.what() << '\n'; return 1; }
  }
}
