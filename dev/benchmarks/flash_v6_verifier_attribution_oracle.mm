// Root-run diagnostic: compilation and --help perform no Metal inference.
#include "flash/FlashForward.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashBatchVerify.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashMTPWindow.hpp"
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
  uint32_t physicalRows = 0;
  bool insideMoE = false, shared = false;
  uint64_t routes = 0;
  static bool dense(std::string_view n) {
    return n.starts_with("flash_affine") || n.starts_with("flash_dense") ||
        n.starts_with("flash_float_dense") || n.starts_with("flash_int8_head");
  }
  std::string family(std::string_view name, const metal::CommandDispatchTimestamp &metadata) {
    if (name.starts_with("flash_int8_head") || name.starts_with("flash_greedy")) return name.starts_with("flash_greedy") ? "greedy" : "vocabulary";
    if (name.starts_with("flash_shared_expert")) return "shared_expert";
    if (dense(name)) {
      const uint32_t outputIndex = name.starts_with("flash_affine") ? 5 : 2;
      for (const auto &binding : metadata.bindings)
        if (!binding.inlineBytes && binding.index == outputIndex && binding.sizeBytes == uint64_t{physicalRows} * 248320 * 2) return "vocabulary";
    }
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
void writeFamilies(std::ostream &out, const metal::CommandDispatchProfile &profile, uint32_t expectedLayers, uint32_t physicalRows) {
  Classifier classifier; classifier.physicalRows = physicalRows; std::map<std::string, Family> families;
  double timedSeconds = 0; uint64_t timedDispatches = 0;
  for (const auto &dispatch : profile.dispatches) {
    auto &f = families[classifier.family(dispatch.pipelineName, dispatch)]; ++f.dispatches;
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
  std::cout << "usage: flash-v6-verifier-attribution-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON\n"
      "Root-only GPU inference. --help creates no Metal backend. ABI must be 200 bytes.\n"
      "FLASH_V6_VERIFY_MODE=normal|command|stage|dispatch (default normal).\n"
      "FLASH_V6_VERIFY_CONTEXT=128|2048; unset profiles both.\n"
      "FLASH_V6_VERIFY_PHYSICAL_ROWS=4|8|16; unset profiles all three.\n"
      "FLASH_V6_VERIFY_EXECUTOR=singleton|joint; unset profiles both.\n"
      "Joint physical R4/R8/R16 is B1/B2/B4 x4 true rows; per-lane R8/R16 is unsupported.\n"
      "FLASH_V6_VERIFY_WARMUP=1 and REPEATS=1; STATE_HASHES=1 defaults.\n"
      "Existing expert-ID capture flag must be1; adds48 diagnostic copy dispatches to each target graph.\n"
      "Each case teacher-primes the trained MTP head with exact target premixer features,\n"
      "generates real anchor+draft tokens, profiles only target verification, and commits its true accepted prefix.\n"
      "Per-family normal host/counter traces include vocabulary and exact per-layer route IDs/histograms.\n"
      "Small windows use gathered expert projections, not blocked MoE bucket jobs.\n"
      "All timings are attribution diagnostics; stage/dispatch modes alter encoder scheduling.\n";
}
std::vector<uint32_t> choices(const char *name, std::initializer_list<uint32_t> defaults) {
  if (!std::getenv(name)) return {defaults};
  const uint32_t value = integer(name, 0, 1, 2048);
  require(std::find(defaults.begin(), defaults.end(), value) != defaults.end(), "unsupported verifier attribution geometry");
  return {value};
}
uint32_t nextToken(const metal::MetalBuffer &records, const metal::MetalBuffer &logits, uint32_t row = 0) {
  if (records && records.contents()) {
    require(records.sizeBytes() >= uint64_t{row + 1} * sizeof(FlashGreedyGPURowResult), "compact greedy row is missing");
    return greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(records.contents())[row], 248320);
  }
  require(logits && logits.contents() && logits.sizeBytes() >= uint64_t{row + 1} * 248320 * 2, "verifier logits have invalid extent");
  const auto *values = static_cast<const uint16_t *>(logits.contents()) + uint64_t{row} * 248320;
  uint32_t best = 0; float maximum = -INFINITY;
  for (uint32_t token = 0; token < 248320; ++token) {
    const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
    require(std::isfinite(value), "nonfinite verifier attribution logits");
    if (value > maximum) { best = token; maximum = value; }
  }
  return best;
}
std::string stateHashes(FlashBatchPrefill &inspector, const FlashRequestState &state) {
  std::ostringstream output; output << '['; bool comma = false;
  for (const auto &plane : inspector.inspectState(state)) {
    if (comma) output << ','; comma = true;
    require(plane.buffer.contents() && plane.activeBytes <= plane.buffer.sizeBytes(), "mutable state hash extent is invalid");
    output << "{\"plane\":" << json::quote(plane.name) << ",\"active_bytes\":" << plane.activeBytes
        << ",\"capacity_bytes\":" << plane.buffer.sizeBytes() << ",\"active_sha256\":"
        << json::quote(digest(plane.buffer.contents(), plane.activeBytes)) << '}';
  }
  output << ']'; return output.str();
}
struct LaneSeed {
  FlashRequestState target;
  FlashMTPState head;
  std::vector<uint32_t> prompt, incoming;
  std::string lastTargetFeatureSha, stateHashJson;
  uint64_t headFoldedLength = 0, headProposedLength = 0;
};
LaneSeed prepareLane(metal::MetalBackend &backend, FlashForward &target, FlashMTPForward &head,
    FlashBatchPrefill &inspector, const std::vector<uint32_t> &prompt, uint32_t rows, bool hashStates) {
  LaneSeed result; result.target = target.createState(); result.head = head.createState(); result.prompt = prompt;
  const auto prefilling = target.forward(result.target, prompt, false, true);
  require(prefilling.hiddenBF16 && prefilling.hiddenBF16.sizeBytes() >= uint64_t{prompt.size()} * 10240 * 2, "actual target features are missing");
  const uint32_t anchor = nextToken(prefilling.greedyResultsU32, prefilling.logitsBF16);
  require(anchor != 248044 && anchor != 248046, "actual anchor is EOS; an exact requested verification window does not exist");
  for (uint32_t begin = 0; begin + 1 < prompt.size(); begin += 128) {
    const uint32_t count = std::min<uint32_t>(128, uint32_t(prompt.size()) - begin - 1);
    const auto features = backend.view(prefilling.hiddenBF16, uint64_t{begin} * 10240 * 2, uint64_t{count} * 10240 * 2);
    (void)head.forward(result.head, features, std::span<const uint32_t>(prompt).subspan(begin + 1, count), FlashMTPLogits::None);
  }
  require(result.target.logicalLength() == prompt.size() && result.head.logicalLength() == prompt.size() - 1,
      "trained head teacher-priming offsets differ from the true prompt");
  const auto last = backend.view(prefilling.hiddenBF16, uint64_t{prompt.size() - 1} * 10240 * 2, 10240 * 2);
  result.lastTargetFeatureSha = digest(last.contents(), last.sizeBytes());
  result.incoming.push_back(anchor);
  auto proposal = head.forward(result.head, last, std::span<const uint32_t>(&anchor, 1), FlashMTPLogits::Last);
  result.headFoldedLength = result.head.logicalLength();
  for (uint32_t draft = 1; draft < rows; ++draft) {
    const uint32_t token = nextToken(proposal.greedyResultsU32, proposal.logitsBF16);
    require(token != 248044 && token != 248046, "actual draft is EOS; refusing to invent an oversized verification window");
    result.incoming.push_back(token);
    if (draft + 1 < rows) {
      const auto feature = backend.view(proposal.hiddenBF16, uint64_t{proposal.hiddenRows - 1} * 10240 * 2, 10240 * 2);
      proposal = head.forward(result.head, feature, std::span<const uint32_t>(&token, 1), FlashMTPLogits::Last);
    }
  }
  result.headProposedLength = result.head.logicalLength();
  if (hashStates) result.stateHashJson = stateHashes(inspector, result.target);
  return result;
}
void writeRoutes(std::ostream &out, const metal::MetalBuffer &captured, uint32_t rows, uint32_t stride, uint32_t layers) {
  require(captured && captured.contents() && rows && rows <= stride && captured.sizeBytes() >= uint64_t{layers} * stride * 10 * 8,
      "diagnostic route capture has invalid shape");
  const auto *ids = static_cast<const int64_t *>(captured.contents());
  out << "{\"actual_gpu_bucket_jobs\":0,\"job_scope\":\"small-window gathered projections; expert histograms describe possible logical grouping, not submitted bucket jobs\""
      << ",\"physical_rows\":" << rows << ",\"selections_per_row\":10,\"layers\":[";
  for (uint32_t layer = 0; layer < layers; ++layer) {
    if (layer) out << ',';
    const auto *begin = ids + uint64_t{layer} * stride * 10;
    std::map<uint32_t, std::vector<uint32_t>> groups;
    for (uint32_t route = 0; route < rows * 10; ++route) {
      require(begin[route] >= 0 && begin[route] < 512, "captured verification expert ID is invalid");
      groups[uint32_t(begin[route])].push_back(route);
    }
    out << "{\"layer\":" << layer << ",\"route_count\":" << rows * 10 << ",\"unique_experts\":" << groups.size()
        << ",\"duplicate_route_assignments_across_rows\":" << rows * 10 - groups.size()
        << ",\"route_ids_i64_sha256\":" << json::quote(digest(begin, uint64_t{rows} * 10 * 8)) << ",\"row_major_expert_ids\":[";
    for (uint32_t route = 0; route < rows * 10; ++route) { if (route) out << ','; out << begin[route]; }
    out << "],\"logical_expert_groups\":["; bool comma = false;
    for (const auto &[expert, members] : groups) {
      if (comma) out << ','; comma = true;
      out << "{\"expert_id\":" << expert << ",\"valid_routes\":" << members.size() << ",\"canonical_route_indices\":[";
      for (size_t member = 0; member < members.size(); ++member) { if (member) out << ','; out << members[member]; }
      out << "]}";
    }
    out << "]}";
  }
  out << "]}";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      static_assert(sizeof(metal::CommandTiming) == 200, "rebuild all objects with the current normal timing ABI");
      if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
      require(argc == 5, "usage: flash-v6-verifier-attribution-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON");
      require(enabled("SPLASH_FLASH_CAPTURE_EXPERT_IDS"), "existing diagnostic expert-ID capture must be enabled");
      const auto contexts = choices("FLASH_V6_VERIFY_CONTEXT", {128, 2048});
      const auto physicalRows = choices("FLASH_V6_VERIFY_PHYSICAL_ROWS", {4, 8, 16});
      const uint32_t warmup = integer("FLASH_V6_VERIFY_WARMUP", 1, 0, 2), repeats = integer("FLASH_V6_VERIFY_REPEATS", 1, 1, 8);
      const uint32_t hashStates = integer("FLASH_V6_VERIFY_STATE_HASHES", 1, 0, 1);
      const char *kind = std::getenv("FLASH_V6_VERIFY_EXECUTOR");
      const std::string executor = kind ? kind : "both";
      require(executor == "both" || executor == "singleton" || executor == "joint", "invalid verifier executor");
      const char *rawMode = std::getenv("FLASH_V6_VERIFY_MODE"); const std::string mode = rawMode ? rawMode : "normal";
      metal::CommandDispatchProfilingMode profilingMode = metal::CommandDispatchProfilingMode::Off;
      if (mode == "command") profilingMode = metal::CommandDispatchProfilingMode::Command;
      else if (mode == "stage") profilingMode = metal::CommandDispatchProfilingMode::StagePerDispatch;
      else if (mode == "dispatch") profilingMode = metal::CommandDispatchProfilingMode::DispatchBoundary;
      else require(mode == "normal", "invalid verifier attribution profile mode");
      for (const auto &path : {std::string(argv[4]), std::string(argv[4]) + ".commands.jsonl", std::string(argv[4]) + ".trace.jsonl", std::string(argv[4]) + ".routes.jsonl"})
        require(!std::filesystem::exists(path), "verifier output exists; choose a fresh report name");
      std::ofstream commands(std::string(argv[4]) + ".commands.jsonl"), trace(std::string(argv[4]) + ".trace.jsonl"), routes(std::string(argv[4]) + ".routes.jsonl");
      require(commands && trace && routes, "cannot open verifier attribution outputs");
      metal::MetalBackend backend(argv[1]); const auto weights = FlashWeights::load(backend, argv[2]);
      const auto &descriptor = weights.descriptor(); constexpr uint32_t capacity = 8192, maximumRows = 2048;
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory, reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(capacity, maximumRows, 16) + FlashForward::expertCachePlannedBytes(weights) +
          FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights) +
          FlashMTPForward::workspacePlannedBytes(capacity, 128) + FlashBatchVerify::workspacePlannedBytes(capacity, 4, 4) +
          FlashBatchPrefill::workspacePlannedBytes(capacity, 1, 1) +
          4 * (FlashForward::requestStateBytes(capacity) + FlashMTPForward::requestStateBytes(capacity)) + (16ULL << 20);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true)) + FlashMTPForward::denseCachePlannedBytes(weights);
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(maximumRows, 10);
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "governor denied verifier diagnostic arenas");
      FlashForward target(backend, weights, capacity, maximumRows, 16);
      FlashMTPForward head(backend, weights, capacity, 128);
      FlashBatchVerify joint(backend, weights, target, capacity, 4, 4, enabled("SPLASH_FLASH_FLOAT_DENSE_CACHE"));
      FlashBatchPrefill inspector(backend, weights, target, capacity, 1, 1);
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto operands = target.cachedOperandsOnly(); const auto extra = head.cachedOperandsOnly();
        operands.insert(operands.end(), extra.begin(), extra.end());
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "v6 verifier saved source/head operands only");
      }
      reservation->commit();
      std::vector<std::string> records; uint64_t cases = 0, normalCommands = 0, counterProfiles = 0;
      for (uint32_t context : contexts) for (uint32_t flat : physicalRows) for (bool batched : {false, true}) {
        if ((batched && executor == "singleton") || (!batched && executor == "joint")) continue;
        const uint32_t lanes = batched ? flat / 4 : 1, realRows = batched ? 4 : flat;
        std::array<std::vector<uint32_t>, 4> prompts;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const std::string name = batched ? "ctx" + std::to_string(context) + "-width4-lane" + std::to_string(lane) + ".json"
              : "ctx" + std::to_string(context) + "-width1-lane0.tokens.json";
          prompts[lane] = loadTokens((std::filesystem::path(argv[3]) / name).string());
          require(prompts[lane].size() == context, "saved verifier prompt has wrong true context length");
        }
        for (uint32_t trial = 0; trial < warmup + repeats; ++trial) {
          backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
          std::vector<LaneSeed> seed; seed.reserve(lanes);
          for (uint32_t lane = 0; lane < lanes; ++lane) seed.push_back(prepareLane(backend, target, head, inspector, prompts[lane], realRows, hashStates && trial >= warmup));
          std::vector<FlashRequestState *> pointers; std::vector<uint32_t> incoming;
          for (auto &lane : seed) { pointers.push_back(&lane.target); incoming.insert(incoming.end(), lane.incoming.begin(), lane.incoming.end()); }
          const bool measured = trial >= warmup;
          backend.setCommandDispatchProfiling(measured ? profilingMode : metal::CommandDispatchProfilingMode::Off);
          const auto started = Clock::now(); metal::CommandTiming timing; metal::MetalBuffer logits, greedy;
          if (batched) {
            const auto result = joint.verifyBatch(pointers, incoming, realRows); timing = result.timing; logits = result.logitsBF16; greedy = result.greedyResultsU32;
            require(result.lanes == lanes && result.rows == realRows, "joint verifier returned different real geometry");
          } else {
            const auto result = target.verify(seed[0].target, incoming); timing = result.timing; logits = result.logitsBF16; greedy = result.greedyResultsU32;
            require(result.logitRows == realRows, "singleton verifier returned different real geometry");
          }
          const double call = std::chrono::duration<double>(Clock::now() - started).count();
          const auto profiles = backend.takeCommandDispatchProfiles();
          backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
          require(profiles.size() == (measured && profilingMode != metal::CommandDispatchProfilingMode::Off ? 1u : 0u), "missing or extra verifier counter profiles");
          std::vector<uint32_t> predictions(flat), retained(lanes);
          for (uint32_t row = 0; row < flat; ++row) predictions[row] = nextToken(greedy, logits, row);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const std::array<uint32_t, 2> stops{248044, 248046};
            const auto accepted = flashMTPAcceptGreedyPrefix(seed[lane].incoming, std::span<const uint32_t>(predictions).subspan(lane * realRows, realRows), 64, stops);
            require(accepted.has_value(), "verifier acceptance geometry is invalid"); retained[lane] = accepted->retainedRows;
          }
          if (measured) {
            ++cases; ++normalCommands;
            const std::string label = (batched ? "joint" : "singleton") + std::string("-ctx") + std::to_string(context) + "-physicalR" + std::to_string(flat) + "-sample" + std::to_string(trial - warmup);
            std::ostringstream record;
            record << "{\"case\":" << json::quote(label) << ",\"executor\":" << json::quote(batched ? "joint" : "singleton")
                << ",\"context_per_lane\":" << context << ",\"lanes\":" << lanes << ",\"true_rows_per_lane\":" << realRows
                << ",\"physical_rows\":" << flat << ",\"gpu_seconds\":" << timing.gpuSeconds << ",\"command_wall_seconds\":" << timing.wallSeconds
                << ",\"forward_call_seconds\":" << call << ",\"diagnostic_copy_dispatches_added\":48,\"actual_gpu_bucket_jobs\":0"
                << ",\"expert_execution\":\"gathered original-affine small-window projections; no blocked buckets\",\"host_command_subphases\":";
            writeHost(record, timing.host); record << ",\"lanes_seed\":[";
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              if (lane) record << ',';
              record << "{\"lane\":" << lane << ",\"target_length_before\":" << context
                  << ",\"head_folded_length\":" << seed[lane].headFoldedLength << ",\"head_proposed_length\":" << seed[lane].headProposedLength
                  << ",\"last_true_target_premixer_bf16_sha256\":" << json::quote(seed[lane].lastTargetFeatureSha)
                  << ",\"prompt_u32le_sha256\":" << json::quote(digest(seed[lane].prompt.data(), seed[lane].prompt.size() * sizeof(uint32_t)))
                  << ",\"prompt_tokens\":[";
              for (size_t token = 0; token < seed[lane].prompt.size(); ++token) { if (token) record << ','; record << seed[lane].prompt[token]; }
              record << "],\"anchor_and_real_trained_drafts\":[";
              for (uint32_t row = 0; row < realRows; ++row) { if (row) record << ','; record << seed[lane].incoming[row]; }
              record << "],\"target_greedy_predictions\":[";
              for (uint32_t row = 0; row < realRows; ++row) { if (row) record << ','; record << predictions[lane * realRows + row]; }
              record << "],\"actual_retained_prefix\":" << retained[lane]
                  << ",\"mutable_active_state_before_verify\":" << (seed[lane].stateHashJson.empty() ? "null" : seed[lane].stateHashJson) << '}';
            }
            record << "]}"; records.push_back(record.str()); commands << record.str() << '\n';
            uint32_t capturedRows = 0, stride = 0;
            const auto ids = batched ? joint.capturedDiagnosticExpertIDs(capturedRows, stride) : target.capturedExpertIDs(capturedRows, stride);
            require(capturedRows == flat, "diagnostic expert capture differs from actual physical rows");
            routes << "{\"case\":" << json::quote(label) << ",\"routes\":"; writeRoutes(routes, ids, capturedRows, stride, descriptor.layers); routes << "}\n";
            for (const auto &profile : profiles) {
              ++counterProfiles; trace << "{\"case\":" << json::quote(label)
                  << ",\"attribution_scope\":\"diagnostic target verification only; stage/dispatch alter encoder scheduling\",\"family_attribution\":";
              writeFamilies(trace, profile, descriptor.layers, flat); trace << ",\"command\":"; profiling::writeJson(trace, profile); trace << "}\n";
            }
            commands.flush(); routes.flush(); trace.flush(); require(commands && routes && trace, "verifier trace write failed");
          }
          const auto commit = batched ? joint.commitBatch(pointers, retained) : target.commitVerify(seed[0].target, retained[0]); (void)commit;
          for (uint32_t lane = 0; lane < lanes; ++lane)
            require(target.ownsState(seed[lane].target) && seed[lane].target.logicalLength() == context + retained[lane], "accepted target prefix commit differs from the true model window");
        }
      }
      const auto persisted = target.persistedOperandStatus(); const auto *experts = target.batchInt8ExpertStore();
      std::ofstream report(argv[4]); require(bool(report), "cannot write verifier attribution report");
      report << "{\"schema\":\"splash-v6-target-verifier-attribution-v1\",\"execution_complete\":true,\"gpu_executed\":true,\"command_timing_abi_bytes\":" << sizeof(metal::CommandTiming)
          << ",\"scope\":\"actual target features and real trained MTP proposals; initial speculative cycle at exact saved prompt contexts; no HTTP performance claim\""
          << ",\"profile_mode\":" << json::quote(mode) << ",\"warmup_per_case\":" << warmup << ",\"measured_repeats_per_case\":" << repeats
          << ",\"state_hash_reads_before_timed_call_can_affect_cache\":" << (hashStates ? "true" : "false")
          << ",\"diagnostic_copy_dispatches_added_per_verifier\":48,\"small_window_gpu_bucket_jobs\":0"
          << ",\"model_layers\":" << descriptor.layers << ",\"gdn_layers\":36,\"qsa_layers\":12"
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity()) << ",\"manifest_fingerprint_sha256\":" << json::quote(weights.manifestFingerprint())
          << ",\"persisted_operand_manifest_sha256\":" << json::quote(persisted.storeManifestSha256)
          << ",\"selected_expert_identity_sha256\":" << (experts ? json::quote(experts->identitySha256()) : "null")
          << ",\"kernel_routes\":" << json::quote(target.kernelRoutes()) << ",\"head_attention_route\":" << json::quote(head.attentionRouteSemantics())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256()))
          << ",\"fixture_provenance_sha256\":" << json::quote(fileDigest((std::filesystem::path(argv[3]) / "fixture-provenance.json").string()))
          << ",\"saved_residency_buffers\":" << residency.bufferCount() << ",\"saved_residency_bytes\":" << residency.byteCount()
          << ",\"cases\":" << cases << ",\"normal_timed_commands\":" << normalCommands << ",\"counter_profiles\":" << counterProfiles << ",\"measurements\":[";
      for (size_t index = 0; index < records.size(); ++index) { if (index) report << ','; report << records[index]; }
      report << "]}\n"; require(bool(report), "verifier attribution report write failed");
      std::cout << "v6 target verifier attribution completed; cases=" << cases << " report=" << argv[4] << '\n'; return 0;
    } catch (const std::exception &error) { std::cerr << "v6 verifier attribution failed: " << error.what() << '\n'; return 1; }
  }
}
