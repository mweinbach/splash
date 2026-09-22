// Compile/CPU mode submits no GPU work. Root alone may run the GPU matrix.
// Include the private Forward implementation to inspect final mixer scratch,
// and filter its ordinary FlashForward.o from the link.
#ifndef PREFILL4K_WIDE_STATE_FORWARD_SOURCE
#error private Forward source macro is required
#endif
#include PREFILL4K_WIDE_STATE_FORWARD_SOURCE
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace splash::flash {
// Forward friendship is granted only by the oracle's private copied header.
// Request-state friendship already exists for native test oracles.
class FlashDeepPrefixOracle final {
public:
  enum class Type { BF16, F32, I64 };
  struct Plane final {
    std::string label;
    Type type;
    uint64_t liveBytes = 0;
    std::vector<uint8_t> bytes;
  };
  struct Snapshot final {
    uint64_t length = 0;
    uint32_t capacity = 0;
    bool poisoned = false, pending = false;
    std::vector<std::pair<std::string, uint64_t>> geometry;
    std::vector<Plane> state, outputs;
  };
  static bool independent(const FlashForward &a, const FlashRequestState &sa,
                          const FlashForward &b, const FlashRequestState &sb) {
    return a.impl_ && b.impl_ && sa.impl_ && sb.impl_ &&
        a.impl_->owner != b.impl_->owner && sa.impl_->owner == a.impl_->owner &&
        sb.impl_->owner == b.impl_->owner && sa.impl_->identity != sb.impl_->identity;
  }
  static Snapshot snapshot(const FlashForward &forward, const FlashRequestState &request,
                           const FlashForwardResult &result, uint32_t rows) {
    if (!forward.impl_ || !request.impl_) throw std::logic_error("missing oracle owner/state");
    std::lock_guard lock(forward.impl_->mutex);
    const auto &s = *request.impl_;
    if (s.owner != forward.impl_->owner || s.poisoned || s.pendingVerification ||
        result.logicalLength != s.length || result.capacity != s.capacity || !rows)
      throw std::logic_error("oracle requires healthy completed plain forward");
    Snapshot out{s.length, s.capacity, s.poisoned, s.pendingVerification, {}, {}, {}};
    const auto add = [&](std::vector<Plane> &to, std::string name, Type type,
                         const metal::MetalBuffer &buffer, uint64_t liveBytes,
                         uint64_t offset = 0, bool physical = true) {
      const uint64_t bytes = physical ? buffer.sizeBytes() : liveBytes;
      if (!buffer || !buffer.contents() || offset > buffer.sizeBytes() ||
          bytes > buffer.sizeBytes() - offset || liveBytes > bytes)
        throw std::logic_error("oracle plane extent is invalid");
      Plane plane{std::move(name), type, liveBytes, std::vector<uint8_t>(bytes)};
      if (bytes) std::memcpy(plane.bytes.data(), static_cast<const uint8_t *>(buffer.contents()) + offset, bytes);
      to.push_back(std::move(plane));
    };
    for (uint32_t layer = 0; layer < forward.impl_->descriptor.layers; ++layer) {
      const std::string prefix = "layer." + std::to_string(layer) + ".";
      if (forward.impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
        const auto &g = s.gdn[layer];
        add(out.state, prefix + "gdn.convolution", Type::BF16, g.convolution, flashGDNConvolutionLaneBytes());
        add(out.state, prefix + "gdn.recurrent", Type::F32, g.recurrent, flashGDNRecurrentLaneBytes());
        out.geometry.emplace_back(prefix + "gdn.convolution_lane_stride", g.convolutionLaneStrideBytes);
        out.geometry.emplace_back(prefix + "gdn.recurrent_lane_stride", g.recurrentLaneStrideBytes);
      } else {
        const auto &q = s.qsa[layer];
        add(out.state, prefix + "qsa.keys", Type::BF16, q.keys, s.length * 512 * 2);
        add(out.state, prefix + "qsa.values", Type::BF16, q.values, s.length * 512 * 2);
        add(out.state, prefix + "qsa.raw_index", Type::BF16, q.rawIndexKeys, s.length * 128 * 2);
        add(out.state, prefix + "qsa.complete_pooled_keys", Type::BF16, q.pooledKeys, (s.length / 4) * 128 * 2);
        add(out.state, prefix + "qsa.positions", Type::I64, q.indexPositions, s.length * 8);
        out.geometry.emplace_back(prefix + "qsa.capacity", q.capacity);
      }
    }
    add(out.state, "ple.token_history", Type::I64, s.pleHistory, 2 * sizeof(int64_t));
    add(out.state, "ple.convolution", Type::BF16, s.pleConvolution, uint64_t{9} * 10240 * 2);
    if (result.logitRows != 1 || result.logitsBF16.sizeBytes() < uint64_t{forward.impl_->descriptor.vocabularySize} * 2 ||
        result.hiddenBF16.sizeBytes() < uint64_t{rows} * 10240 * 2)
      throw std::logic_error("oracle last output contract is invalid");
    add(out.outputs, "last.logits", Type::BF16, result.logitsBF16,
        uint64_t{forward.impl_->descriptor.vocabularySize} * 2, 0, false);
    add(out.outputs, "last.pre_mixer_hyper", Type::BF16, result.hiddenBF16,
        uint64_t{10240} * 2, uint64_t{rows - 1} * 10240 * 2, false);
    const auto mixed = forward.impl_->bf(Scratch::Mixed, rows, 2560);
    add(out.outputs, "last.post_mixer_normalized_hidden", Type::BF16, mixed,
        uint64_t{2560} * 2, uint64_t{rows - 1} * 2560 * 2, false);
    return out; // All borrowed views copied before another trunk call.
  }
};
} // namespace splash::flash

namespace splash::wide_state_oracle {
using namespace flash;
using Access = FlashDeepPrefixOracle;
constexpr uint32_t kCapacity = 16384, kBaselineRows = 2048, kVocabulary = 248320;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
bool enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  require(std::string_view(value) == "0" || std::string_view(value) == "1", "oracle Boolean setting must be 0 or 1");
  return std::string_view(value) == "1";
}
uint32_t candidateRows() {
  const char *value = std::getenv("PREFILL4K_WIDE_STATE_ROWS");
  if (!value || std::string_view(value) == "8192") return 8192;
  require(std::string_view(value) == "4096", "candidate arena must be 4096 or 8192 rows");
  return 4096;
}
std::vector<uint32_t> loadTokens(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  require(data != nil, "cannot read oracle frozen tokens");
  NSError *error = nil; id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error && [object isKindOfClass:[NSArray class]], "oracle tokens must be an array");
  std::vector<uint32_t> out;
  for (id item in static_cast<NSArray *>(object)) {
    require([item isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)item) != CFBooleanGetTypeID(), "token is not an integer");
    const double value = static_cast<NSNumber *>(item).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < kVocabulary && std::floor(value) == value, "invalid frozen token");
    out.push_back(uint32_t(value));
  }
  require(out.size() == 2048 || out.size() == 4096 || out.size() == 8192, "oracle expects exact 2048/4096/8192 token fixture");
  return out;
}
std::string digest(const void *bytes, uint64_t size) {
  CC_SHA256_CTX ctx{}; CC_SHA256_Init(&ctx);
  auto *next = static_cast<const uint8_t *>(bytes);
  while (size) { const CC_LONG count = CC_LONG(std::min<uint64_t>(size, UINT32_MAX)); CC_SHA256_Update(&ctx, next, count); size -= count; next += count; }
  std::array<uint8_t, 32> hash{}; CC_SHA256_Final(hash.data(), &ctx);
  static constexpr char hex[] = "0123456789abcdef"; std::string out;
  for (const uint8_t b : hash) { out += hex[b >> 4]; out += hex[b & 15]; }
  return out;
}
std::string fileDigest(const char *path) {
  std::ifstream file(path, std::ios::binary); require(bool(file), "cannot read oracle build provenance");
  const std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  return digest(bytes.data(), bytes.size());
}
const char *typeName(Access::Type type) {
  if (type == Access::Type::BF16) return "BF16";
  if (type == Access::Type::F32) return "F32";
  return "I64";
}
uint32_t wordBytes(Access::Type type) { return type == Access::Type::BF16 ? 2 : type == Access::Type::F32 ? 4 : 8; }
uint64_t rawWord(const uint8_t *p, uint32_t width) { uint64_t out = 0; std::memcpy(&out, p, width); return out; }
double number(uint64_t word, Access::Type type) {
  return std::bit_cast<float>(type == Access::Type::BF16 ? uint32_t(word) << 16 : uint32_t(word));
}
uint64_t orderedWord(uint64_t word, uint32_t width) {
  const uint64_t sign = uint64_t{1} << (width * 8 - 1), mask = (uint64_t{1} << (width * 8)) - 1;
  return word & sign ? (~word & mask) : (word | sign);
}
struct Difference final {
  uint64_t bytes = 0, words = 0, differingBytes = 0, differingWords = 0;
  uint64_t nonfiniteReference = 0, nonfiniteCandidate = 0, maximumWordDistance = 0;
  double maxAbs = 0, squaredError = 0, squaredReference = 0;
  std::vector<std::array<uint64_t, 3>> first; // offset, reference bits, candidate bits
  bool exact() const { return differingBytes == 0; }
  bool finite() const { return !nonfiniteReference && !nonfiniteCandidate; }
};
Difference compareRange(Access::Type type, const uint8_t *reference, const uint8_t *candidate, uint64_t bytes) {
  const uint32_t width = wordBytes(type); require(bytes % width == 0, "oracle plane is not word aligned");
  Difference out; out.bytes = bytes; out.words = bytes / width;
  for (uint64_t offset = 0; offset < bytes; offset += width) {
    const uint64_t a = rawWord(reference + offset, width), b = rawWord(candidate + offset, width);
    if (a != b) {
      ++out.differingWords;
      for (uint32_t j = 0; j < width; ++j) out.differingBytes += reference[offset + j] != candidate[offset + j];
      if (out.first.size() < 8) out.first.push_back({offset, a, b});
      if (type != Access::Type::I64) {
        const uint64_t oa = orderedWord(a, width), ob = orderedWord(b, width);
        out.maximumWordDistance = std::max(out.maximumWordDistance, oa > ob ? oa - ob : ob - oa);
      }
    }
    if (type != Access::Type::I64) {
      const double av = number(a, type), bv = number(b, type);
      out.nonfiniteReference += !std::isfinite(av); out.nonfiniteCandidate += !std::isfinite(bv);
      if (std::isfinite(av) && std::isfinite(bv)) {
        const double error = av - bv;
        out.maxAbs = std::max(out.maxAbs, std::abs(error));
        out.squaredError += error * error; out.squaredReference += av * av;
      }
    }
  }
  return out;
}
void writeNumber(std::ostream &out, double value) { if (std::isfinite(value)) out << value; else out << "null"; }
void writeDifference(std::ostream &out, Access::Type type, const Difference &diff) {
  out << "{\"exact_bytes_equal\":" << (diff.exact() ? "true" : "false") << ",\"bytes\":" << diff.bytes
      << ",\"words\":" << diff.words << ",\"differing_bytes\":" << diff.differingBytes << ",\"differing_words\":" << diff.differingWords;
  if (type != Access::Type::I64) {
    out << ",\"nonfinite_reference\":" << diff.nonfiniteReference << ",\"nonfinite_candidate\":" << diff.nonfiniteCandidate
        << ",\"max_absolute_error\":"; writeNumber(out, diff.maxAbs);
    out << ",\"rms_absolute_error\":"; writeNumber(out, diff.words ? std::sqrt(diff.squaredError / diff.words) : 0);
    out << ",\"relative_l2_to_reference\":";
    if (diff.squaredReference) writeNumber(out, std::sqrt(diff.squaredError / diff.squaredReference));
    else if (!diff.squaredError) out << '0'; else out << "null";
    out << ",\"maximum_ordered_word_distance\":" << diff.maximumWordDistance;
  }
  out << ",\"first_word_differences\":[";
  for (size_t i = 0; i < diff.first.size(); ++i) {
    if (i) out << ','; const auto &entry = diff.first[i];
    out << "{\"byte_offset\":" << entry[0] << ",\"reference_bits\":" << entry[1] << ",\"candidate_bits\":" << entry[2];
    if (type != Access::Type::I64) { out << ",\"reference_value\":"; writeNumber(out, number(entry[1], type)); out << ",\"candidate_value\":"; writeNumber(out, number(entry[2], type)); }
    out << '}';
  }
  out << "]}";
}
struct Summary final {
  uint64_t checkpoints = 0, statePlanes = 0, outputPlanes = 0, allocatedBytes = 0, liveBytes = 0;
  uint64_t statePhysicalDifferingPlanes = 0, stateLiveDifferingPlanes = 0, outputDifferingPlanes = 0, metadataDifferences = 0, nonfinitePlanes = 0;
  bool exact() const { return !statePhysicalDifferingPlanes && !outputDifferingPlanes && !metadataDifferences && !nonfinitePlanes; }
};
void comparePlanes(std::ostream &out, const std::vector<Access::Plane> &reference, const std::vector<Access::Plane> &candidate,
                   Summary &summary, bool persistent) {
  require(reference.size() == candidate.size(), "oracle plane count differs"); out << '[';
  for (size_t i = 0; i < reference.size(); ++i) {
    const auto &a = reference[i], &b = candidate[i];
    require(a.label == b.label && a.type == b.type && a.liveBytes == b.liveBytes && a.bytes.size() == b.bytes.size(), "oracle plane layouts differ");
    const auto live = compareRange(a.type, a.bytes.data(), b.bytes.data(), a.liveBytes);
    const auto physical = compareRange(a.type, a.bytes.data(), b.bytes.data(), a.bytes.size());
    if (persistent) { ++summary.statePlanes; summary.allocatedBytes += physical.bytes; summary.liveBytes += live.bytes;
      summary.statePhysicalDifferingPlanes += !physical.exact(); summary.stateLiveDifferingPlanes += !live.exact(); }
    else { ++summary.outputPlanes; summary.outputDifferingPlanes += !physical.exact(); }
    summary.nonfinitePlanes += !physical.finite();
    if (i) out << ',';
    out << "{\"plane\":" << json::quote(a.label) << ",\"dtype\":" << json::quote(typeName(a.type))
        << ",\"live_bytes\":" << a.liveBytes << ",\"allocated_bytes\":" << a.bytes.size()
        << ",\"reference_allocated_sha256\":" << json::quote(digest(a.bytes.data(), a.bytes.size()))
        << ",\"candidate_allocated_sha256\":" << json::quote(digest(b.bytes.data(), b.bytes.size())) << ",\"live\":";
    writeDifference(out, a.type, live); out << ",\"allocated\":"; writeDifference(out, a.type, physical); out << '}';
  }
  out << ']';
}
void compareCheckpoint(std::ostream &trace, std::string_view phase, uint32_t index, const Access::Snapshot &a,
                       const Access::Snapshot &b, Summary &summary) {
  ++summary.checkpoints;
  const bool metadataEqual = a.length == b.length && a.capacity == b.capacity && a.poisoned == b.poisoned && a.pending == b.pending && a.geometry == b.geometry;
  summary.metadataDifferences += !metadataEqual;
  trace << "{\"phase\":" << json::quote(phase) << ",\"checkpoint\":" << index << ",\"logical_length\":" << a.length
      << ",\"metadata_equal\":" << (metadataEqual ? "true" : "false") << ",\"reference_metadata\":{\"capacity\":" << a.capacity
      << ",\"length\":" << a.length << ",\"poisoned\":" << (a.poisoned ? "true" : "false") << ",\"pending_verification\":" << (a.pending ? "true" : "false")
      << "},\"candidate_metadata\":{\"capacity\":" << b.capacity << ",\"length\":" << b.length << ",\"poisoned\":" << (b.poisoned ? "true" : "false")
      << ",\"pending_verification\":" << (b.pending ? "true" : "false") << "},\"layer_geometry\":[";
  require(a.geometry.size() == b.geometry.size(), "oracle layer metadata count differs");
  for (size_t i = 0; i < a.geometry.size(); ++i) {
    require(a.geometry[i].first == b.geometry[i].first, "oracle layer metadata layout differs");
    if (i) trace << ',';
    trace << "{\"field\":" << json::quote(a.geometry[i].first) << ",\"reference\":" << a.geometry[i].second << ",\"candidate\":" << b.geometry[i].second << '}';
  }
  trace << "],\"persistent_state\":"; comparePlanes(trace, a.state, b.state, summary, true);
  trace << ",\"last_outputs\":"; comparePlanes(trace, a.outputs, b.outputs, summary, false);
  trace << "}\n"; trace.flush(); require(bool(trace), "oracle checkpoint write failed");
}
uint64_t forwardPlan(const FlashWeights &weights, uint32_t rows) {
  uint64_t plan = FlashForward::workspacePlannedBytes(kCapacity, rows, 0) + FlashForward::requestStateBytes(kCapacity) +
      FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights);
  if (enabled("SPLASH_FLASH_DENSE_CACHE")) plan += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));
  if (enabled("SPLASH_FLASH_QSA_F32")) plan += 16ULL << 20;
  if (enabled("SPLASH_FLASH_BLOCKED_MOE")) plan += flashMoEBlockedWorkspacePlannedBytes(rows, 10);
  return plan;
}
void cpuSelfTest() {
  std::array<uint16_t, 5> a{0, 0x8000, 0x3f80, 0x4000, 0x7f80}, b = a;
  const auto exact = compareRange(Access::Type::BF16, reinterpret_cast<const uint8_t *>(a.data()), reinterpret_cast<const uint8_t *>(b.data()), sizeof(a));
  require(exact.exact() && exact.nonfiniteReference == 1 && exact.nonfiniteCandidate == 1, "CPU exact/nonfinite BF16 scan failed");
  b[0] = 0x8000; b[3] = 0x4001;
  const auto changed = compareRange(Access::Type::BF16, reinterpret_cast<const uint8_t *>(a.data()), reinterpret_cast<const uint8_t *>(b.data()), sizeof(a));
  require(!changed.exact() && changed.differingWords == 2 && changed.first.size() == 2 && changed.first[0][0] == 0 &&
      changed.first[1][0] == 6 && changed.maxAbs == 0.015625, "CPU signed-zero/tail BF16 differences failed");
  const std::array<float, 3> fa{0.0f, 1.0f, -2.0f}, fb{0.0f, 1.00000011920928955078125f, -2.0f};
  const auto f = compareRange(Access::Type::F32, reinterpret_cast<const uint8_t *>(fa.data()), reinterpret_cast<const uint8_t *>(fb.data()), sizeof(fa));
  require(f.differingWords == 1 && f.maximumWordDistance == 1 && f.first[0][0] == 4 && f.finite(), "CPU F32 recurrence word distance failed");
  std::array<int64_t, 3> ia{1, -1, 7}, ib = ia; ib.back() = 8;
  const auto id = compareRange(Access::Type::I64, reinterpret_cast<const uint8_t *>(ia.data()), reinterpret_cast<const uint8_t *>(ib.data()), sizeof(ia));
  require(id.differingWords == 1 && id.first.front()[0] == 16, "CPU exact I64 tail failed");
  std::array<uint8_t, 16> physicalA{}, physicalB{}; physicalB[14] = 1;
  require(compareRange(Access::Type::BF16, physicalA.data(), physicalB.data(), 8).exact() &&
      !compareRange(Access::Type::BF16, physicalA.data(), physicalB.data(), 16).exact(), "CPU live versus padding comparison failed");
  for (const uint32_t rows : {4096u, 8192u}) for (const uint32_t tokens : {2048u, 4096u, 8192u}) {
    uint32_t baselineCalls = 0, wideCalls = 0, baselineConsumed = 0;
    for (uint32_t begin = 0; begin < tokens; begin += rows) {
      const uint32_t end = std::min(tokens, begin + rows); ++wideCalls;
      while (baselineConsumed < end) { baselineConsumed += std::min(2048u, end - baselineConsumed); ++baselineCalls; }
      require(baselineConsumed == end, "CPU aligned checkpoint chunk schedule failed");
    }
    require(baselineConsumed == tokens && baselineCalls == tokens / 2048 && wideCalls == (tokens + rows - 1) / rows &&
        tokens + 7 < kCapacity, "CPU matrix/capacity scheduling failed");
  }
  std::ostringstream jsonTest; writeDifference(jsonTest, Access::Type::BF16, changed);
  require(jsonTest.str().find("\"exact_bytes_equal\":false") != std::string::npos &&
      jsonTest.str().find("\"nonfinite_reference\":1") != std::string::npos, "CPU report does not preserve mismatches/nonfinite");
  std::cout << "{\"schema\":\"splash-private-wide-state-cpu-v1\",\"passed\":true,\"gpu_executed\":false,\"checks\":[\"bf16_nonfinite\",\"signed_zero_exact_difference\",\"bf16_tail_difference\",\"f32_ulp_difference\",\"i64_tail_difference\",\"inactive_padding_is_separate\",\"six_case_chunk_matrix\",\"16384_capacity\",\"mismatch_json\"]}\n";
}
void writeEnvironment(std::ostream &report) {
  static constexpr const char *names[] = {
    "SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_FUSE_HC", "SPLASH_FLASH_FUSE_GDN", "SPLASH_FLASH_QSA_F32",
    "SPLASH_FLASH_DENSE_CACHE", "SPLASH_FLASH_BLOCKED_MOE", "SPLASH_FLASH_EXPERT_QMV", "SPLASH_FLASH_GDN_STAGED",
    "SPLASH_FLASH_QSA_MPP", "SPLASH_FLASH_FLOAT_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE",
    "SPLASH_FLASH_MOE_Q4X8", "SPLASH_FLASH_MOE_M64", "SPLASH_FLASH_PLE_LOOKUP_FUSED", "SPLASH_FLASH_PLE_POST_FUSED",
    "SPLASH_FLASH_GPU_GREEDY", "SPLASH_FLASH_INT8_HEAD", "SPLASH_FLASH_MOE_DIRECT_A", "SPLASH_FLASH_QSA_ROW_TILES",
    "SPLASH_FLASH_QSA_OUT_F32_N32", "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT", "SPLASH_FLASH_SHARED_EXPERT_FUSED",
    "SPLASH_FLASH_DENSE_M64_OUT", "SPLASH_FLASH_HC_UP_F32_MPP", "SPLASH_FLASH_GDN_LAZY_ROLLBACK",
    "SPLASH_FLASH_PLE_SSD_STREAMING", "SPLASH_FLASH_OPERAND_STORE", "SPLASH_FLASH_PREFILL_DENSE_TILES",
    "SPLASH_FLASH_INT8_EXPERT_STORE", "SPLASH_FLASH_HOT_EXPERT_PLAN", "SPLASH_FLASH_DENSE_TRAVERSAL",
    "SPLASH_FLASH_CAPTURE_EXPERT_IDS"
  };
  report << '{';
  for (size_t i = 0; i < std::size(names); ++i) { if (i) report << ','; report << json::quote(names[i]) << ':';
    const char *value = std::getenv(names[i]); if (value) report << json::quote(value); else report << "null"; }
  report << '}';
}
int run(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "usage: state-oracle METALLIB PACKAGE EXACT_TOKENS_JSON REPORT_JSON | --cpu-self-test\n"
        "PREFILL4K_WIDE_STATE_ROWS=4096|8192 (default8192), capacity16384, baseline arena2048.\n"
        "Private full-state/cache/output comparison plus7 identical one-token appends; no MTP/service claim.\n"
        "Exit0 means byte-exact finite full-state/outputs; exit2 records numerical/physical differences.\n"
        "Compilation, --help and --cpu-self-test create no MetalBackend and submit no GPU work.\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
  require(argc == 5, "invalid state oracle arguments; run --help");
  require(!std::filesystem::exists(argv[4]) && !std::filesystem::exists(std::string(argv[4]) + ".checkpoints.jsonl"), "choose a fresh report path");
  const auto tokens = loadTokens(argv[3]); const uint32_t wideRows = candidateRows();
  const std::array<uint32_t, 7> continuation{tokens[0], tokens[1], 248044, 198, tokens[2], 248046, tokens[3]};
  std::ofstream trace(std::string(argv[4]) + ".checkpoints.jsonl"); require(bool(trace), "cannot open oracle checkpoint trace");
  trace << std::setprecision(std::numeric_limits<double>::max_digits10);
  metal::MetalBackend backend(argv[1]); const auto weights = FlashWeights::load(backend, argv[2]);
  const uint64_t physical = NSProcessInfo.processInfo.physicalMemory, reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
  engine::MemoryGovernor governor(backend, physical - reserve, reserve);
  // CPU-owned full physical snapshots are included in admission, not silently
  // retained as borrowed shared scratch or unbudgeted model cache allocations.
  const uint64_t snapshotBudget = 2 * FlashForward::requestStateBytes(kCapacity) + (16ULL << 20);
  const uint64_t planned = forwardPlan(weights, kBaselineRows) + forwardPlan(weights, wideRows) + snapshotBudget;
  auto reservation = governor.tryReserve(planned); require(bool(reservation), "oracle governor denied independent arenas and snapshots");
  FlashForward baseline(backend, weights, kCapacity, kBaselineRows, 0), candidate(backend, weights, kCapacity, wideRows, 0);
  require(baseline.workspaceBytes() + candidate.workspaceBytes() + 2 * FlashForward::requestStateBytes(kCapacity) + snapshotBudget <= planned,
      "oracle actual arenas plus snapshot budget exceed reservation");
  auto baselineState = baseline.createState(), candidateState = candidate.createState();
  require(Access::independent(baseline, baselineState, candidate, candidateState), "oracle must use independent Forward owners and request identities");
  metal::ResidencyLease residency;
  if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
    auto buffers = baseline.cachedOperandsOnly(); const auto extra = candidate.cachedOperandsOnly(); buffers.insert(buffers.end(), extra.begin(), extra.end());
    if (!buffers.empty()) residency = backend.requestWeightResidency(buffers, "wide-state oracle independent saved operands only");
  }
  reservation->commit(); Summary summary;
  uint32_t baselineConsumed = 0, baselineCalls = 0, candidateCalls = 0;
  double baselineGPU = 0, candidateGPU = 0;
  for (uint32_t begin = 0; begin < tokens.size(); begin += wideRows) {
    const uint32_t end = std::min(uint32_t(tokens.size()), begin + wideRows), count = end - begin;
    Access::Snapshot a;
    while (baselineConsumed < end) {
      const uint32_t rows = std::min(kBaselineRows, end - baselineConsumed);
      const auto result = baseline.forward(baselineState, std::span(tokens).subspan(baselineConsumed, rows), false, true);
      ++baselineCalls; baselineGPU += result.timing.gpuSeconds; baselineConsumed += rows;
      // Only the last baseline chunk is a shared logical checkpoint.
      if (baselineConsumed == end) a = Access::snapshot(baseline, baselineState, result, rows);
    }
    const auto result = candidate.forward(candidateState, std::span(tokens).subspan(begin, count), false, true);
    ++candidateCalls; candidateGPU += result.timing.gpuSeconds;
    const auto b = Access::snapshot(candidate, candidateState, result, count);
    compareCheckpoint(trace, "prompt", end, a, b, summary);
    std::cout << "wide-state prompt checkpoint=" << end << " exact_so_far=" << summary.exact() << '\n' << std::flush;
  }
  for (uint32_t step = 0; step < continuation.size(); ++step) {
    const auto input = std::span(&continuation[step], 1);
    const auto ar = baseline.forward(baselineState, input, false, true);
    const auto a = Access::snapshot(baseline, baselineState, ar, 1);
    const auto br = candidate.forward(candidateState, input, false, true);
    const auto b = Access::snapshot(candidate, candidateState, br, 1);
    compareCheckpoint(trace, "identical_one_token_append", step, a, b, summary);
    std::cout << "wide-state append=" << step << " exact_so_far=" << summary.exact() << '\n' << std::flush;
  }
  std::ofstream report(argv[4]); require(bool(report), "cannot open oracle final report");
  report << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "{\"schema\":\"splash-private-wide-state-continuation-v1\",\"gpu_executed\":true,\"completed\":true"
      << ",\"full_allocated_state_and_last_outputs_exact_finite\":" << (summary.exact() ? "true" : "false")
      << ",\"capacity\":" << kCapacity << ",\"prompt_tokens\":" << tokens.size() << ",\"baseline_arena_rows\":" << kBaselineRows
      << ",\"candidate_arena_rows\":" << wideRows << ",\"independent_forward_owners_and_requests\":true,\"borrowed_outputs_copied_before_reuse\":true"
      << ",\"comparison_gate\":\"all allocated persistent bytes, scalar metadata, all last logits, pre-mixer hyper and post-mixer normalized hidden must be byte-exact and finite; numerical metrics are diagnostics only\""
      << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity()) << ",\"metallib_sha256\":" << json::quote(fileDigest(argv[1]))
      << ",\"private_forward_cpp_sha256\":" << json::quote(fileDigest(PREFILL4K_WIDE_STATE_FORWARD_SOURCE))
      << ",\"private_access_header_sha256\":" << json::quote(fileDigest(PREFILL4K_WIDE_STATE_ACCESS_HEADER))
      << ",\"tokens_u32le_sha256\":" << json::quote(digest(tokens.data(), tokens.size() * sizeof(uint32_t)))
      << ",\"baseline_kernel_routes\":" << json::quote(baseline.kernelRoutes()) << ",\"candidate_kernel_routes\":" << json::quote(candidate.kernelRoutes())
      << ",\"planned_reservation_bytes\":" << planned << ",\"cpu_owned_snapshot_budget_bytes\":" << snapshotBudget
      << ",\"baseline_workspace_bytes\":" << baseline.workspaceBytes() << ",\"candidate_workspace_bytes\":" << candidate.workspaceBytes()
      << ",\"baseline_prompt_commands\":" << baselineCalls << ",\"candidate_prompt_commands\":" << candidateCalls
      << ",\"baseline_prompt_gpu_seconds\":" << baselineGPU << ",\"candidate_prompt_gpu_seconds\":" << candidateGPU
      << ",\"timing_scope\":\"diagnostic serialized full-trunk commands interleaved with CPU state copies/hashes; no performance qualification, MTP priming, HTTP scheduler or greedy generation\""
      << ",\"checkpoints\":" << summary.checkpoints << ",\"persistent_plane_comparisons\":" << summary.statePlanes << ",\"output_plane_comparisons\":" << summary.outputPlanes
      << ",\"allocated_state_bytes_compared\":" << summary.allocatedBytes << ",\"live_state_bytes_compared\":" << summary.liveBytes
      << ",\"persistent_allocated_differing_planes\":" << summary.statePhysicalDifferingPlanes << ",\"persistent_live_differing_planes\":" << summary.stateLiveDifferingPlanes
      << ",\"last_output_differing_planes\":" << summary.outputDifferingPlanes << ",\"metadata_differences\":" << summary.metadataDifferences
      << ",\"nonfinite_plane_comparisons\":" << summary.nonfinitePlanes << ",\"identical_append_tokens\":[";
  for (size_t i = 0; i < continuation.size(); ++i) { if (i) report << ','; report << continuation[i]; }
  report << "],\"environment\":"; writeEnvironment(report); report << "}\n"; require(bool(report), "oracle final report write failed");
  std::cout << "wide-state completed; exact=" << summary.exact() << " report=" << argv[4] << '\n';
  return summary.exact() ? 0 : 2;
}
} // namespace splash::wide_state_oracle
int main(int argc, char **argv) {
  @autoreleasepool {
    try { return splash::wide_state_oracle::run(argc, argv); }
    catch (const std::exception &error) { std::cerr << "wide-state oracle error: " << error.what() << '\n'; return 1; }
  }
}
