// Private whole-target correctness oracle. Compilation/--cpu-only read no model
// payload and create no Metal backend. Root alone owns --gpu execution.
#include "flash/FlashForward.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "LazyCopyFusionWholeBuildProvenance.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash {
// Existing request-state test friendship only; no new Forward/public API.
class FlashDeepPrefixOracle final {
public:
  enum class Type : uint32_t { BF16 = 1, F32 = 2, I64 = 3, U32 = 4 };
  struct Plane final {
    std::string label;
    Type type;
    metal::MetalBuffer buffer;
    uint64_t liveBytes;
  };
  static bool pending(const FlashRequestState &state) {
    return state.impl_ && state.impl_->pendingVerification;
  }
  static std::array<uintptr_t, 3> binding(const FlashRequestState &state) {
    return {reinterpret_cast<uintptr_t>(state.impl_.get()),
        state.impl_ ? reinterpret_cast<uintptr_t>(state.impl_->owner.get()) : 0,
        state.impl_ ? reinterpret_cast<uintptr_t>(state.impl_->identity.get()) : 0};
  }
  static std::vector<Plane> planes(const FlashRequestState &request) {
    if (!request.impl_) throw std::invalid_argument("oracle state is uninitialized");
    const auto &s = *request.impl_;
    std::vector<Plane> out;
    const auto add = [&](std::string label, Type type,
                         const metal::MetalBuffer &buffer, uint64_t live) {
      if (!buffer || !buffer.contents() || live > buffer.sizeBytes())
        throw std::runtime_error("persistent oracle plane extent invalid: " + label);
      out.push_back({std::move(label), type, buffer, live});
    };
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const auto prefix = "layer." + std::to_string(layer) + ".";
      if (s.gdn[layer].recurrent) {
        add(prefix + "gdn.convolution", Type::BF16, s.gdn[layer].convolution,
            flashGDNConvolutionLaneBytes());
        add(prefix + "gdn.recurrent", Type::F32, s.gdn[layer].recurrent,
            flashGDNRecurrentLaneBytes());
      } else {
        const auto &q = s.qsa[layer];
        add(prefix + "qsa.keys", Type::BF16, q.keys, s.length * 1024);
        add(prefix + "qsa.values", Type::BF16, q.values, s.length * 1024);
        add(prefix + "qsa.raw_index_keys", Type::BF16, q.rawIndexKeys, s.length * 256);
        add(prefix + "qsa.pooled_keys", Type::BF16, q.pooledKeys, (s.length / 4) * 256);
        add(prefix + "qsa.index_positions", Type::I64, q.indexPositions, s.length * 8);
      }
    }
    add("ple.history", Type::I64, s.pleHistory, 16);
    add("ple.convolution", Type::BF16, s.pleConvolution, uint64_t{9} * 10240 * 2);
    if (out.size() != 134) throw std::runtime_error("persistent plane inventory must be134");
    return out;
  }
  static std::string metadata(const FlashForward &owner,
                              const FlashRequestState &request) {
    if (!request.impl_) throw std::invalid_argument("state metadata is uninitialized");
    const auto &s = *request.impl_;
    std::ostringstream out;
    out << "{\"length\":" << s.length << ",\"capacity\":" << s.capacity
        << ",\"poisoned\":" << (s.poisoned ? "true" : "false")
        << ",\"pending\":" << (s.pendingVerification ? "true" : "false")
        << ",\"owned\":" << (owner.ownsState(request) ? "true" : "false")
        << ",\"geometry\":[";
    for (uint32_t layer = 0; layer < 48; ++layer) {
      if (layer) out << ',';
      if (s.gdn[layer].recurrent)
        out << "[" << layer << ",\"gdn\","
            << s.gdn[layer].convolutionLaneStrideBytes << ','
            << s.gdn[layer].recurrentLaneStrideBytes << ']';
      else out << '[' << layer << ",\"qsa\"," << s.qsa[layer].capacity << ']';
    }
    out << "]}";
    return out.str();
  }
};
} // namespace splash::flash

namespace {
using namespace splash;
using namespace splash::flash;
namespace fs = std::filesystem;
using Access = FlashDeepPrefixOracle;
using Type = Access::Type;
constexpr uint32_t kCapacity = 4096, kPrefillRows = 2048, kVerifyRows = 4;
constexpr uint32_t kVocabulary = 248320, kHyper = 10240;
constexpr uint64_t kScratch = 1ULL << 20, kSpillBound = 32ULL << 30;
constexpr const char *kFlag = "SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21";
constexpr const char *kCanonicalFixtureSHA = "4985e55294b83c72cb9e51e00c40f918460b6c4f560cb5b32d4be3662e540b57";
constexpr const char *kCanonicalTokensSHA = "0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4";
constexpr std::array<uint32_t, 4> kIncoming{71093, 12305, 198, 464};
static_assert(sizeof(metal::CommandTiming) == 200, "private timing ABI200 required");

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
bool flag(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || std::string_view(raw) == "0") return false;
  require(std::string_view(raw) == "1", std::string(name) + " must be0 or1");
  return true;
}
std::string hexDigest(const std::array<uint8_t, 32> &result) {
  std::ostringstream out;
  for (uint8_t byte : result)
    out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
  return out.str();
}
struct Hash final {
  CC_SHA256_CTX state{};
  Hash() { CC_SHA256_Init(&state); }
  void add(const void *raw, uint64_t bytes) {
    auto *next = static_cast<const uint8_t *>(raw);
    while (bytes) {
      const auto count = CC_LONG(std::min<uint64_t>(bytes, UINT32_MAX));
      CC_SHA256_Update(&state, next, count); next += count; bytes -= count;
    }
  }
  std::string finish() {
    std::array<uint8_t, 32> result{}; CC_SHA256_Final(result.data(), &state);
    std::ostringstream out;
    for (uint8_t byte : result)
      out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    return out.str();
  }
};
std::string fileHash(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot hash provenance file: " + path.string());
  std::array<char, 65536> buffer{}; Hash hash;
  while (in) {
    in.read(buffer.data(), buffer.size());
    if (in.gcount()) hash.add(buffer.data(), uint64_t(in.gcount()));
  }
  require(in.eof(), "provenance file read failed: " + path.string());
  return hash.finish();
}
std::string bytesHash(const void *p, uint64_t bytes) {
  Hash hash; hash.add(p, bytes); return hash.finish();
}
std::array<uint8_t, 8> little64(uint64_t value) {
  std::array<uint8_t, 8> out{};
  for (uint32_t i = 0; i < 8; ++i) out[i] = uint8_t(value >> (8 * i));
  return out;
}
uint64_t firstDifferent(const uint8_t *a, const uint8_t *b, uint64_t count) {
  if (!count || std::memcmp(a, b, size_t(count)) == 0) return count;
  uint64_t i = 0;
  while (i < count && a[i] == b[i]) ++i;
  return i;
}
void finite(const uint8_t *raw, uint64_t bytes, Type type) {
  if (type == Type::I64 || type == Type::U32) return;
  const uint32_t width = type == Type::BF16 ? 2 : 4;
  require(bytes % width == 0, "finite scan has partial word");
  for (uint64_t i = 0; i < bytes; i += width) {
    uint32_t bits = 0; std::memcpy(&bits, raw + i, width);
    if (type == Type::BF16) bits <<= 16;
    require(std::isfinite(std::bit_cast<float>(bits)),
        "nonfinite word in checkpoint at byte" + std::to_string(i));
  }
}

struct PlaneView final {
  std::string label;
  Type type;
  const uint8_t *data;
  uint64_t liveBytes, bytes;
};
struct Checkpoint final {
  std::string label, sha256;
  uint64_t bytes = 0, planes = 0, liveBytes = 0;
};
struct Transcript final {
  std::vector<Checkpoint> checkpoints;
  std::array<std::array<uint32_t, 4>, 4> future{}, futurePredictions{};
  uint64_t spillBytes = 0, comparedBytes = 0, checkedPlanes = 0, guardCalls = 0;
  uint64_t controlCheckpoints = 0, candidateCheckpoints = 0;
  std::string phase = "arguments", sourceIdentity, manifestIdentity, derivative;
  std::string controlRoutes, candidateRoutes;
  std::string controlLoadedLibrarySHA, candidateLoadedLibrarySHA;
  uint64_t controlWorkspace = 0, candidateWorkspace = 0;
  uint64_t controlPeak = 0, candidatePeak = 0, controlReservation = 0, candidateReservation = 0;
  bool backendInitialized = false, controlDestroyed = false, candidateDestroyed = false, reportAllowed = false;
  fs::path report, spill;
};

// The same serializer is used for writes and strict streaming comparisons.
// Metadata and plane declarations precede payloads; candidate never imports
// an active trial or reads a second model's cache simultaneously.
class CheckpointStream final {
public:
  CheckpointStream(fs::path path, bool writing, uint64_t expectedBytes)
      : path_(std::move(path)), writing_(writing), expected_(expectedBytes), scratch_(kScratch) {
    if (writing_) {
      require(!fs::exists(path_) && !fs::exists(path_.string() + ".writing"), "fresh checkpoint path required");
      out_.open(path_.string() + ".writing", std::ios::binary);
      require(bool(out_), "checkpoint output cannot be opened");
    } else {
      require(fs::is_regular_file(path_) && fs::file_size(path_) == expected_, "checkpoint exact file extent differs");
      in_.open(path_, std::ios::binary); require(bool(in_), "checkpoint input cannot be opened");
    }
  }
  void part(const void *raw, uint64_t bytes) {
    require(raw || !bytes, "null checkpoint payload");
    require(bytes <= kSpillBound - count_, "checkpoint exceeds hard byte limit");
    const auto *next = static_cast<const uint8_t *>(raw);
    while (bytes) {
      const uint64_t n = std::min<uint64_t>(bytes, scratch_.size());
      if (writing_) {
        out_.write(reinterpret_cast<const char *>(next), std::streamsize(n));
        require(bool(out_), "checkpoint write failed");
      } else {
        in_.read(reinterpret_cast<char *>(scratch_.data()), std::streamsize(n));
        require(in_.gcount() == std::streamsize(n), "checkpoint truncated during comparison");
        const auto offset = firstDifferent(next, scratch_.data(), n);
        require(offset == n, "checkpoint byte mismatch: " + path_.filename().string() +
            " byte=" + std::to_string(count_ + offset) +
            " candidate=" + std::to_string(unsigned(next[offset == n ? 0 : offset])) +
            " control=" + std::to_string(unsigned(scratch_[offset == n ? 0 : offset])));
      }
      hash_.add(next, n); next += n; bytes -= n; count_ += n;
    }
  }
  void number(uint64_t value) { const auto encoded = little64(value); part(encoded.data(), encoded.size()); }
  void string(const std::string &value) { number(value.size()); part(value.data(), value.size()); }
  std::pair<uint64_t, std::string> finish() {
    if (writing_) {
      require(count_ == expected_, "written checkpoint differs from preflight byte declaration");
      out_.close(); require(bool(out_), "checkpoint close failed");
      fs::rename(path_.string() + ".writing", path_);
      require(fs::file_size(path_) == count_, "checkpoint published extent differs");
    } else {
      require(count_ == expected_, "checkpoint consumed extent differs");
      require(in_.peek() == std::char_traits<char>::eof(), "checkpoint has unexpected trailing bytes");
    }
    return {count_, hash_.finish()};
  }
private:
  fs::path path_;
  bool writing_;
  uint64_t expected_, count_ = 0;
  std::ifstream in_; std::ofstream out_;
  std::vector<uint8_t> scratch_;
  Hash hash_;
};

uint64_t serializedBytes(const std::string &metadata, const std::vector<PlaneView> &planes) {
  require(metadata.size() <= 65536 && !planes.empty() && planes.size() <= 134,
      "checkpoint metadata/plane declaration exceeds bounded protocol");
  uint64_t bytes = 16 + 8 + metadata.size() + 8;
  const auto add = [&](uint64_t count) {
    require(count <= kSpillBound - bytes, "declared checkpoint exceeds hard32GiB bound");
    bytes += count;
  };
  for (const auto &plane : planes) {
    require(!plane.label.empty() && plane.label.size() <= 256 && plane.liveBytes <= plane.bytes,
        "bounded plane declaration invalid");
    add(8 + plane.label.size() + 24); add(plane.bytes);
  }
  return bytes;
}

void serialize(CheckpointStream &stream, const std::string &metadata,
               const std::vector<PlaneView> &planes) {
  constexpr std::array<uint8_t, 16> magic{'S','P','L','A','S','H','L','A','Z','Y','C','P',1,0,0,0};
  stream.part(magic.data(), magic.size()); stream.string(metadata); stream.number(planes.size());
  for (const auto &plane : planes) {
    require(plane.liveBytes <= plane.bytes && (plane.data || !plane.bytes), "plane live/physical extent invalid");
    stream.string(plane.label); stream.number(uint32_t(plane.type));
    stream.number(plane.liveBytes); stream.number(plane.bytes);
    finite(plane.data, plane.bytes, plane.type);
    stream.part(plane.data, plane.bytes);
  }
}

class Store final {
public:
  Store(Transcript &transcript, bool candidate) : transcript_(transcript), candidate_(candidate) {}
  void checkpoint(const std::string &label, const std::string &metadata,
                  const std::vector<PlaneView> &planes) {
    transcript_.phase = (candidate_ ? "candidate." : "control.") + label;
    const auto path = transcript_.spill / (label + ".bin");
    const auto found = std::find_if(transcript_.checkpoints.begin(), transcript_.checkpoints.end(),
        [&](const auto &entry) { return entry.label == label; });
    require(candidate_ ? found != transcript_.checkpoints.end() : found == transcript_.checkpoints.end(),
        "checkpoint declaration missing/duplicate");
    const auto declaredBytes = serializedBytes(metadata, planes);
    if (!candidate_) require(declaredBytes <= kSpillBound - transcript_.spillBytes,
        "control spill would exceed preregistered32GiB bound before any write");
    else require(declaredBytes == found->bytes, "candidate declared frame size differs");
    CheckpointStream stream(path, !candidate_, declaredBytes);
    serialize(stream, metadata, planes);
    const auto [bytes, sha256] = stream.finish();
    uint64_t live = 0; for (const auto &plane : planes) live += plane.liveBytes;
    if (candidate_) {
      require(sha256 == found->sha256 && planes.size() == found->planes && live == found->liveBytes,
          "candidate checkpoint provenance/plane/live extent differs");
      transcript_.comparedBytes += bytes; transcript_.checkedPlanes += planes.size();
      ++transcript_.candidateCheckpoints;
    } else {
      require(bytes <= kSpillBound - transcript_.spillBytes, "control spill exceeds preregistered32GiB bound");
      transcript_.spillBytes += bytes; transcript_.checkpoints.push_back({label, sha256, bytes, planes.size(), live});
      ++transcript_.controlCheckpoints;
    }
  }
  void unchanged(const std::string &prior, const FlashForward &owner,
                 const FlashRequestState &state, const std::array<uintptr_t, 3> &binding) {
    require(Access::binding(state) == binding, "host guard changed state owner/identity binding");
    const auto found = std::find_if(transcript_.checkpoints.begin(), transcript_.checkpoints.end(),
        [&](const auto &entry) { return entry.label == prior; });
    require(found != transcript_.checkpoints.end(), "guard checkpoint missing");
    CheckpointStream stream(transcript_.spill / (prior + ".bin"), false, found->bytes);
    const auto planes = stateViews(state); serialize(stream, Access::metadata(owner, state), planes);
    const auto [bytes, sha256] = stream.finish();
    require(sha256 == found->sha256, "host guard changed checkpoint SHA");
    transcript_.comparedBytes += bytes; ++transcript_.guardCalls;
  }
  void state(const std::string &label, const FlashForward &owner, const FlashRequestState &request) {
    const auto planes = stateViews(request);
    uint64_t total = 0; for (const auto &plane : planes) total += plane.bytes;
    require(total == FlashForward::requestStateBytes(request.capacity()), "physical state planes differ from admission formula");
    checkpoint(label, Access::metadata(owner, request), planes);
  }
  void output(const std::string &label, const FlashForwardResult &result, uint32_t consumed) {
    require(result.logitRows > 0 && result.logitRows <= consumed && result.logitRows <= 128,
        "output real logit rows invalid");
    std::vector<PlaneView> planes;
    const auto add = [&](std::string name, Type type, const metal::MetalBuffer &buffer, uint64_t bytes) {
      require(buffer && buffer.contents() && buffer.sizeBytes() == bytes, "borrowed output exact real extent invalid");
      planes.push_back({std::move(name), type, static_cast<const uint8_t *>(buffer.contents()), bytes, bytes});
    };
    add("logits", Type::BF16, result.logitsBF16, uint64_t{result.logitRows} * kVocabulary * 2);
    add("hidden", Type::BF16, result.hiddenBF16, uint64_t{consumed} * kHyper * 2);
    require(result.greedyResultsU32 && result.greedyRows == result.logitRows, "GPU compact greedy records required");
    require(result.greedyResultsU32.contents() && result.greedyResultsU32.sizeBytes() ==
        uint64_t{result.greedyRows} * sizeof(FlashGreedyGPURowResult), "GPU greedy exact frame size invalid");
    const auto *greedy = static_cast<const FlashGreedyGPURowResult *>(result.greedyResultsU32.contents());
    for (uint32_t row = 0; row < result.greedyRows; ++row)
      (void)greedyGPUResultToken(greedy[row], kVocabulary);
    add("compact_greedy", Type::U32, result.greedyResultsU32,
        uint64_t{result.greedyRows} * sizeof(FlashGreedyGPURowResult));
    std::ostringstream metadata;
    metadata << "{\"length\":" << result.logicalLength << ",\"capacity\":" << result.capacity
        << ",\"consumed_rows\":" << consumed << ",\"logit_rows\":" << result.logitRows
        << ",\"greedy_rows\":" << result.greedyRows << '}';
    checkpoint(label, metadata.str(), planes);
  }
  static std::vector<PlaneView> stateViews(const FlashRequestState &state) {
    std::vector<PlaneView> out;
    for (const auto &plane : Access::planes(state))
      out.push_back({plane.label, plane.type, static_cast<const uint8_t *>(plane.buffer.contents()),
          plane.liveBytes, plane.buffer.sizeBytes()});
    return out;
  }
private:
  Transcript &transcript_;
  bool candidate_;
};

uint32_t prediction(const FlashForwardResult &result, uint32_t row = 0) {
  require(result.greedyRows > row && result.greedyResultsU32.contents(), "greedy row absent");
  return greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(result.greedyResultsU32.contents())[row], kVocabulary);
}
uint32_t changedToken(uint32_t greedy, uint32_t step, std::span<const uint32_t> used) {
  uint32_t token = (greedy + 1 + step * 17) % kVocabulary;
  while (std::find(used.begin(), used.end(), token) != used.end() ||
      std::find(kIncoming.begin(), kIncoming.end(), token) != kIncoming.end()) token = (token + 1) % kVocabulary;
  require(token != greedy, "changed future token unexpectedly equals greedy prediction");
  return token;
}
void disjoint(const FlashForward &owner, const FlashRequestState &a,
              const FlashRequestState &b) {
  require(owner.ownsState(a) && owner.ownsState(b), "request owner predicate differs");
  const auto ba = Access::binding(a), bb = Access::binding(b);
  require(ba[0] != bb[0] && ba[1] == bb[1] && ba[2] != bb[2], "request ownership/identity aliases");
  struct Range { uintptr_t begin, end; }; std::vector<Range> ranges;
  for (const auto *state : {&a, &b}) for (const auto &plane : Access::planes(*state)) {
    const auto begin = reinterpret_cast<uintptr_t>(plane.buffer.contents());
    require(plane.buffer.sizeBytes() <= UINTPTR_MAX - begin, "request address extent overflows");
    ranges.push_back({begin, begin + plane.buffer.sizeBytes()});
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto &x, const auto &y) { return x.begin < y.begin; });
  for (size_t i = 1; i < ranges.size(); ++i)
    require(ranges[i].begin >= ranges[i - 1].end, "request persistent allocations overlap");
}
template<class Exception, class Operation> void reject(Operation &&operation, const char *label) {
  bool rejected = false;
  try { operation(); } catch (const Exception &) { rejected = true; }
  require(rejected, std::string("expected host rejection: ") + label);
}

std::vector<uint32_t> tokens(const fs::path &path) {
  require(fs::file_size(path) <= 2ULL << 20, "token fixture exceeds2MiB");
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "token fixture cannot be read");
  NSError *error = nil; id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [value isKindOfClass:[NSArray class]], "token fixture must beJSON array");
  std::vector<uint32_t> out;
  for (id item in static_cast<NSArray *>(value)) {
    require([item isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)item) != CFBooleanGetTypeID(), "token must beinteger");
    const double number = static_cast<NSNumber *>(item).doubleValue;
    require(std::isfinite(number) && number >= 0 && number < kVocabulary && std::floor(number) == number, "token ID out of range");
    out.push_back(uint32_t(number));
  }
  require(out.size() == kPrefillRows, "whole oracle requires exactly2048 tokens");
  return out;
}
NSDictionary *provenance(bool verify) {
  NSString *text = [NSString stringWithUTF8String:kLazyCopyFusionWholeBuildProvenance];
  require(text != nil, "build provenance must beUTF8");
  NSError *error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:&error];
  require(error == nil && [value isKindOfClass:[NSDictionary class]], "build provenance must beJSON dictionary");
  NSDictionary *dictionary = static_cast<NSDictionary *>(value);
  if (verify) {
    for (NSString *key in @[@"sources", @"objects"]) {
      id entries = dictionary[key];
      require([entries isKindOfClass:[NSArray class]] && [entries count] > 0, "build provenance sources/objects must benonempty");
      for (id entry in static_cast<NSArray *>(entries)) {
        require([entry isKindOfClass:[NSDictionary class]], "provenance entry must bedictionary");
        id path = static_cast<NSDictionary *>(entry)[@"path"];
        if (!path) path = static_cast<NSDictionary *>(entry)[@"frozen"];
        id sha = static_cast<NSDictionary *>(entry)[@"sha256"];
        require([path isKindOfClass:[NSString class]] && [sha isKindOfClass:[NSString class]] && [sha length] == 64,
            "provenance path/SHA invalid");
        require(fileHash(static_cast<NSString *>(path).UTF8String) == static_cast<NSString *>(sha).UTF8String,
            "frozen provenance SHA mismatch: " + std::string(static_cast<NSString *>(path).UTF8String));
      }
    }
  }
  return dictionary;
}
void progress(const Transcript &t, bool failed, const std::string &error = {}) {
  if (t.report.empty() || !t.reportAllowed) return;
  const auto destination = t.report.string() + (failed ? ".failure.json" : ".checkpoint.json");
  std::ofstream out(destination + ".writing"); require(bool(out), "progress output cannot beopened");
  out << "{\"schema\":\"splash-lazy-copy-whole-progress-v1\",\"valid\":false,\"qualification_complete\":false"
      << ",\"failed\":" << (failed ? "true" : "false") << ",\"error\":" << json::quote(error)
      << ",\"phase\":" << json::quote(t.phase) << ",\"backend_initialized\":" << (t.backendInitialized ? "true" : "false")
      << ",\"control_backend_destroyed\":" << (t.controlDestroyed ? "true" : "false")
      << ",\"candidate_backend_destroyed\":" << (t.candidateDestroyed ? "true" : "false")
      << ",\"control_checkpoints\":" << t.controlCheckpoints << ",\"candidate_checkpoints\":" << t.candidateCheckpoints
      << ",\"spill_bytes\":" << t.spillBytes << ",\"compared_bytes\":" << t.comparedBytes
      << ",\"checked_planes\":" << t.checkedPlanes << ",\"guard_calls\":" << t.guardCalls << "}\n";
  out.close(); require(bool(out), "progress output close failed"); fs::rename(destination + ".writing", destination);
}
void earlyPolicy() {
  for (const char *name : {"SPLASH_FLASH_ALLROWS_FULL512_TARGET", "SPLASH_FLASH_BLOCKED_MOE",
       "SPLASH_FLASH_FUSE_GDN", "SPLASH_FLASH_GDN_LAZY_ROLLBACK", "SPLASH_FLASH_GPU_GREEDY",
       "SPLASH_FLASH_PLE_SSD_STREAMING", "SPLASH_FLASH_ALLROWS_GATHERED_MPP", "SPLASH_FLASH_MOE_POINTWISE_SEP21"})
    require(flag(name), std::string("required whole-oracle policy: ") + name);
  require(gathered_mpp::requestedMaximumRows() >= 4, "gathered current policy must include physical4 verifier rows");
  require(!flag("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT") && !flag("SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE") &&
      !flag("SPLASH_FLASH_CAPTURE_EXPERT_IDS"), "whole oracle requires original/idle residency and route capture off");
  require(std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE") != nullptr && !std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"),
      "one Full512 expert store required");
}
void pass(Transcript &t, bool candidate, const std::string &library,
          const std::string &package, const std::vector<uint32_t> &prompt,
          const std::string &expectedLibrarySHA) {
  require(!candidate || t.controlDestroyed, "candidate must follow complete control teardown");
  require(setenv(kFlag, candidate ? "1" : "0", 1) == 0, "candidate environment cannot beset");
  t.phase = candidate ? "candidate.backend_construct" : "control.backend_construct"; progress(t, false);
  @autoreleasepool {
    metal::MetalBackend backend(library); t.backendInitialized = true;
    const auto loadedLibrarySHA = hexDigest(backend.metallibSha256());
    require(loadedLibrarySHA == expectedLibrarySHA, "backend immutable loaded library SHA differs from qualified artifact");
    if (candidate) t.candidateLoadedLibrarySHA = loadedLibrarySHA;
    else t.controlLoadedLibrarySHA = loadedLibrarySHA;
    const auto weights = FlashWeights::load(backend, package);
    require(weights.descriptor().vocabularySize == kVocabulary && weights.descriptor().layers == 48,
        "whole-oracle descriptor vocabulary/layer count differs");
    for (const uint32_t token : prompt)
      require(token < weights.descriptor().vocabularySize, "prompt token exceeds actual descriptor vocabulary");
    const auto original = weights.pleSSDStorageStats();
    require(original.allRowsFull512Target && original.gpuMappedBytes == 6370164736ULL,
        "whole oracle original target omission/ledger differs");
    if (candidate) require(weights.sourceIdentity() == t.sourceIdentity && weights.manifestFingerprint() == t.manifestIdentity,
        "candidate source/manifest identity differs");
    else { t.sourceIdentity = weights.sourceIdentity(); t.manifestIdentity = weights.manifestFingerprint(); }
    const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
    const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
    engine::MemoryGovernor governor(backend, physical - reserve, reserve);
    uint64_t planned = FlashForward::workspacePlannedBytes(kCapacity, kPrefillRows, kVerifyRows) +
        FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) +
        FlashForward::int8HeadPlannedBytes(weights) + 2 * FlashForward::requestStateBytes(kCapacity) + (16ULL << 20);
    if (flag("SPLASH_FLASH_DENSE_CACHE"))
      planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));
    if (flag("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(kPrefillRows, 10);
    auto reservation = governor.tryReserve(planned); require(bool(reservation), "governor denied whole target/two-state arenas");
    {
      FlashForward target(backend, weights, kCapacity, kPrefillRows, kVerifyRows);
      require(target.lazyGDNRollbackEnabled() && target.lazyGDNCopyFusionSep21Enabled() == candidate,
          "target constructor did not capture expected lazy/fusion option");
      const auto *store = target.batchInt8ExpertStore();
      require(store && store->mappedBytes() == 121173442560ULL && target.allRowsInt8TargetEnabled(), "actual Full512 target/store differs");
      for (uint32_t layer = 0; layer < 48; ++layer) {
        const auto ids = store->selectedExpertIDs(layer); require(ids.size() == 512, "Full512 inventory incomplete");
        for (uint32_t id = 0; id < 512; ++id) require(ids[id] == id, "Full512 expert IDs noncanonical");
      }
      if (candidate) require(store->numericalIdentitySha256() == t.derivative, "candidate numerical derivative differs");
      else t.derivative = store->numericalIdentitySha256();
      require(target.workspaceBytes() + 2 * FlashForward::requestStateBytes(kCapacity) <= planned, "actual whole arena exceeds reservation");
      if (candidate) { t.candidateWorkspace = target.workspaceBytes(); t.candidateReservation = planned; t.candidateRoutes = target.kernelRoutes(); }
      else { t.controlWorkspace = target.workspaceBytes(); t.controlReservation = planned; t.controlRoutes = target.kernelRoutes(); }
      if (candidate) {
        auto normalized = t.candidateRoutes;
        const auto position = normalized.find(kFlashGDNLazyCopyFusionSep21Route);
        require(position != std::string::npos &&
            t.controlRoutes.find(kFlashGDNLazyCopyFusionSep21Route) == std::string::npos,
            "control/candidate route marker doesnotmatch captured0/1");
        normalized.erase(position, std::strlen(kFlashGDNLazyCopyFusionSep21Route));
        require(normalized == t.controlRoutes, "other kernel policies/operand identities differ between passes");
      }
      metal::ResidencyLease residency;
      if (flag("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        const auto operands = target.cachedOperandsOnly();
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "private sequential lazy-copy whole oracle saved operands only");
      }
      reservation->commit();
      // Prove the option is frozen on every existing instance, including records.
      require(setenv(kFlag, candidate ? "0" : "1", 1) == 0, "post-constructor flag flip failed");
      require(target.lazyGDNCopyFusionSep21Enabled() == candidate, "existing target option changed withenvironment");
      Store files(t, candidate);
      for (uint32_t keep = 1; keep <= 4; ++keep) {
        const auto label = "b1.keep" + std::to_string(keep);
        auto state = target.createState(), peer = target.createState();
        disjoint(target, state, peer);
        FlashRequestState uninitialized;
        require(!target.ownsState(uninitialized) && uninitialized.poisoned() &&
            uninitialized.capacity() == 0 && uninitialized.logicalLength() == 0,
            "default request publication/ownership semantics differ");
        reject<std::invalid_argument>([&] { (void)target.forward(uninitialized, std::span(kIncoming).first(1)); }, "default state");
        const auto originalBinding = Access::binding(state);
        auto moved = std::move(state);
        require(target.ownsState(moved) && !target.ownsState(state) && state.poisoned(), "state move ownership semantics differ");
        reject<std::invalid_argument>([&] { (void)target.forward(state, std::span(kIncoming).first(1)); }, "moved-from state");
        state = std::move(moved); require(Access::binding(state) == originalBinding, "state move changed identity/owner");
        const auto prefill = target.forward(state, prompt, false, true);
        require(state.logicalLength() == 2048 && !state.poisoned() && !Access::pending(state), "fresh canonical prefill publication differs");
        files.output(label + ".prefill.output", prefill, kPrefillRows); files.state(label + ".prefill.state", target, state);
        const auto verified = target.verify(state, kIncoming);
        require(verified.logitRows == 4 && state.logicalLength() == 2052 && Access::pending(state) && !state.poisoned(), "provisional verification publication differs");
        files.output(label + ".verify.output", verified, 4); files.state(label + ".verify.state", target, state);
        const uint32_t next = prediction(verified, keep - 1);
        files.state(label + ".peer.guard", target, peer);
        const auto stateBinding = Access::binding(state), peerBinding = Access::binding(peer);
        reject<std::invalid_argument>([&] { (void)target.commitVerify(state, 0); }, "retained0");
        reject<std::invalid_argument>([&] { (void)target.commitVerify(state, 5); }, "retained5");
        reject<std::invalid_argument>([&] { (void)target.commitVerify(peer, 1); }, "wrong peer commit");
        reject<std::logic_error>([&] { (void)target.forward(peer, std::span(kIncoming).first(1)); }, "healthy peer blocked by pending tape");
        target.abortVerify(peer);
        files.unchanged(label + ".verify.state", target, state, stateBinding);
        files.unchanged(label + ".peer.guard", target, peer, peerBinding);
        const auto committed = target.commitVerify(state, keep);
        if (keep == 4) require(committed.gpuSeconds == 0, "full commit unexpectedly submitted GPU work");
        else require(std::isfinite(committed.gpuSeconds) && committed.gpuSeconds > 0 &&
            std::isfinite(committed.wallSeconds) && committed.wallSeconds > 0, "partial commit timing/ABI invalid");
        require(state.logicalLength() == 2048 + keep && !Access::pending(state) && !state.poisoned(), "retained commit publication differs");
        files.state(label + ".commit.state", target, state);
        uint32_t predicted = next;
        for (uint32_t step = 0; step < 4; ++step) {
          if (!candidate) {
            t.futurePredictions[keep - 1][step] = predicted;
            t.future[keep - 1][step] = changedToken(predicted, step, std::span(t.future[keep - 1]).first(step));
          } else require(predicted == t.futurePredictions[keep - 1][step], "future baseline greedy prediction differs");
          const uint32_t input = t.future[keep - 1][step];
          const auto continuation = target.forward(state, std::span(&input, 1), false, true);
          require(state.logicalLength() == 2048 + keep + step + 1 && !state.poisoned() && !Access::pending(state), "future singleton publication differs");
          const auto futureLabel = label + ".future" + std::to_string(step);
          files.output(futureLabel + ".output", continuation, 1); files.state(futureLabel + ".state", target, state);
          predicted = prediction(continuation);
        }
        require(target.lazyGDNCopyFusionSep21Enabled() == candidate, "existing fusion option notfrozen");
        progress(t, false); std::cout << (candidate ? "candidate" : "control") << " keep=" << keep << " strict checkpoints passed\n" << std::flush;
      }
      // Terminal cancellation has no rollback-to-base claim. A healthy peer
      // continues after abort and after destruction of the cancelled request.
      {
        auto state = target.createState(), peer = target.createState(); disjoint(target, state, peer);
        const auto prefill = target.forward(state, prompt, false, true);
        files.output("b1.abort.prefill.output", prefill, kPrefillRows); files.state("b1.abort.prefill.state", target, state);
        const auto trial = target.verify(state, kIncoming);
        files.output("b1.abort.verify.output", trial, 4); files.state("b1.abort.verify.state", target, state);
        target.abortVerify(state);
        require(state.poisoned() && !Access::pending(state) && state.logicalLength() == 2052, "abort terminal poison/length semantics differ");
        files.state("b1.abort.terminal.state", target, state);
        const auto binding = Access::binding(state);
        reject<std::invalid_argument>([&] { (void)target.forward(state, std::span(kIncoming).first(1)); }, "terminal poisoned state");
        files.unchanged("b1.abort.terminal.state", target, state, binding);
        const auto healthy = target.forward(peer, std::span(kIncoming).first(1), false, true);
        files.output("b1.abort.healthy_peer.output", healthy, 1); files.state("b1.abort.healthy_peer.state", target, peer);
        state = FlashRequestState(); state = target.createState(); disjoint(target, state, peer);
        const auto reset = target.forward(state, prompt, false, true);
        files.output("b1.destroy.prefill.output", reset, kPrefillRows); files.state("b1.destroy.prefill.state", target, state);
        const auto cancelled = target.verify(state, kIncoming);
        files.output("b1.destroy.verify.output", cancelled, 4); files.state("b1.destroy.verify.state", target, state);
        state = FlashRequestState();
        const auto resumed = target.forward(peer, std::span(kIncoming).subspan(1, 1), false, true);
        require(peer.logicalLength() == 2 && !peer.poisoned() && !Access::pending(peer), "healthy peer did notresume after trial destruction");
        files.output("b1.destroy.healthy_peer.output", resumed, 1); files.state("b1.destroy.healthy_peer.state", target, peer);
      }
      require(governor.snapshot().deniedReservations == 0, "whole-oracle governor recorded denied reservation");
      if (candidate) t.candidatePeak = backend.memoryStats().peakAllocatedBytes;
      else t.controlPeak = backend.memoryStats().peakAllocatedBytes;
      // All request/result/operand-vector handles above have left scope before
      // stopping. Residency and Forward leave scope below, then weights/backend.
      backend.stop();
    }
  }
  if (candidate) t.candidateDestroyed = true; else t.controlDestroyed = true;
  require(setenv(kFlag, candidate ? "1" : "0", 1) == 0, "flag restoration failed");
  t.phase = candidate ? "candidate.complete_teardown" : "control.complete_teardown"; progress(t, false);
}

void cpuOnly() {
  require(sizeof(metal::CommandTiming) == 200, "CPU ABI check failed");
  std::array<uint8_t, 16> a{}, b{};
  require(firstDifferent(a.data(), b.data(), a.size()) == a.size(), "CPU equal byte comparator failed");
  b[15] = 1; require(firstDifferent(a.data(), b.data(), a.size()) == 15, "CPU physical tail mismatch missed");
  b = a; b[0] = 1; require(firstDifferent(a.data(), b.data(), a.size()) == 0, "CPU first-byte mismatch missed");
  const auto le = little64(0x123456789abcdef0ULL);
  require(le[0] == 0xf0 && le[7] == 0x12, "CPU little-endian protocol failed");
  require(FlashForward::requestStateBytes(4096) == 232603648ULL &&
      FlashForward::requestStateBytes(8192) == 349388800ULL, "CPU capacity admission metadata failed");
  const uint16_t badBF16 = 0x7f80;
  reject<std::runtime_error>([&] { finite(reinterpret_cast<const uint8_t *>(&badBF16), 2, Type::BF16); }, "matching BF16 infinity cannotpass");
  const float badF32 = std::numeric_limits<float>::quiet_NaN();
  reject<std::runtime_error>([&] { finite(reinterpret_cast<const uint8_t *>(&badF32), 4, Type::F32); }, "matching F32 NaN cannotpass");
  const auto temp = fs::temp_directory_path() / ("splash-lazy-whole-cpu-" + std::to_string(NSProcessInfo.processInfo.processIdentifier));
  require(!fs::exists(temp), "fresh CPU fixture directory required"); fs::create_directory(temp);
  try {
    const std::array<uint16_t, 4> payload{0, 0x8000, 0x3f80, 0x4000};
    const std::vector<PlaneView> planes{{"synthetic.bf16", Type::BF16,
        reinterpret_cast<const uint8_t *>(payload.data()), 4, sizeof(payload)}};
    const std::string metadata = "{\"capacity\":4096,\"length\":2,\"pending\":false}";
    uint64_t bytes = 0; std::string sha;
    {
      CheckpointStream writer(temp / "checkpoint.bin", true, serializedBytes(metadata, planes));
      serialize(writer, metadata, planes); const auto written = writer.finish(); bytes = written.first; sha = written.second;
    }
    {
      CheckpointStream reader(temp / "checkpoint.bin", false, bytes); serialize(reader, metadata, planes);
      const auto matched = reader.finish(); require(matched.first == bytes && matched.second == sha, "CPU streaming protocol equality failed");
    }
    const std::vector<PlaneView> oversized{{"synthetic", Type::U32, a.data(), 0, kSpillBound}};
    reject<std::runtime_error>([&] { (void)serializedBytes(metadata, oversized); }, "hard bound preflight beforepayload access");
    reject<std::runtime_error>([&] {
      CheckpointStream changed(temp / "checkpoint.bin", false, bytes);
      serialize(changed, "{\"capacity\":8192,\"length\":2,\"pending\":false}", planes); (void)changed.finish();
    }, "capacity metadata difference");
    auto physical = payload; physical[3] ^= 1;
    const std::vector<PlaneView> changedPlanes{{"synthetic.bf16", Type::BF16,
        reinterpret_cast<const uint8_t *>(physical.data()), 4, sizeof(physical)}};
    reject<std::runtime_error>([&] {
      CheckpointStream changed(temp / "checkpoint.bin", false, bytes);
      serialize(changed, metadata, changedPlanes); (void)changed.finish();
    }, "inactive physical tail mismatch");
    fs::resize_file(temp / "checkpoint.bin", bytes - 1);
    reject<std::runtime_error>([&] { CheckpointStream truncated(temp / "checkpoint.bin", false, bytes); }, "truncated checkpoint");
    fs::remove_all(temp);
  } catch (...) { fs::remove_all(temp); throw; }
  std::array<uint32_t, 4> used{};
  for (uint32_t step = 0; step < 4; ++step) {
    used[step] = changedToken(198, step, std::span(used).first(step));
    require(used[step] != 198 && used[step] < kVocabulary, "CPU changed future input policy failed");
  }
  std::cout << "{\"schema\":\"splash-lazy-copy-whole-cpu-v1\",\"valid\":true,\"gpu_work\":false,\"model_loaded\":false,\"payload_bytes_read\":0"
      << ",\"command_timing_abi_bytes\":200,\"comparison_scratch_bound_bytes\":" << kScratch
      << ",\"checks\":[\"equal_first_tail_byte_comparator\",\"little_endian_protocol\",\"capacity_4096_8192\",\"bf16_f32_nonfinite_rejection\",\"stream_protocol_hash\",\"capacity_difference\",\"inactive_physical_tail\",\"truncation\",\"changed_future_inputs\"]}\n";
}
void writeEnvironment(std::ostream &out) {
  out << '{'; bool comma = false;
  for (const char *name : {"SPLASH_FLASH_ALLROWS_FULL512_TARGET", "SPLASH_FLASH_INT8_EXPERT_STORE", "SPLASH_FLASH_OPERAND_STORE",
      "SPLASH_FLASH_PLE_SSD_STREAMING", "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT", "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT",
      "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE", "SPLASH_FLASH_FUSE_GDN", "SPLASH_FLASH_GDN_STAGED", "SPLASH_FLASH_GDN_LAZY_ROLLBACK",
      "SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_FLOAT_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE", "SPLASH_FLASH_DENSE_CACHE",
      "SPLASH_FLASH_HC_UP_F32_MPP", "SPLASH_FLASH_FUSE_HC", "SPLASH_FLASH_SHARED_EXPERT_FUSED", "SPLASH_FLASH_GPU_GREEDY",
      "SPLASH_FLASH_INT8_HEAD", "SPLASH_FLASH_MOE_POINTWISE_SEP21", "SPLASH_FLASH_ALLROWS_GATHERED_MPP",
      "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS", "SPLASH_FLASH_QSA_BULK_PREFILL", "SPLASH_FLASH_QSA_BULK_PREFILL_SG8",
      "SPLASH_FLASH_PREFILL_DENSE_TILES", "SPLASH_FLASH_CAPTURE_EXPERT_IDS"}) {
    if (comma) out << ','; comma = true; out << json::quote(name) << ':';
    const char *value = std::getenv(name); if (value) out << json::quote(value); else out << "null";
  }
  out << '}';
}
int run(int argc, char **argv, Transcript &t) {
  if (argc == 2 && (std::string_view(argv[1]) == "--cpu-only" || std::string_view(argv[1]) == "--cpu-self-test")) { cpuOnly(); return 0; }
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "whole-oracle --gpu METALLIB PACKAGE CANONICAL2048_JSON NEW_REPORT_JSON | --cpu-only\n"
      << "Sequential control/candidate Backend+weights+Forward scopes; B1 verify4 keep1/2/3/4, four changed future inputs, abort/destruction/host guards.\n"
      << "Fixed incoming71093,12305,198,464; exact all134 physical persistent planes plus hidden/logits/compact greedy;32GiB spill bound.\n"
      << "Existing lazy policy fixed1; constructor captures custom candidate0/1 and is checked after environment flip. B4 notexercised.\n"
      << "Correctness only; no trained MTP, service performance or semantic quality claim. CPU mode reads no model payload and creates no backend.\n"; return 0;
  }
  require(argc == 6 && std::string_view(argv[1]) == "--gpu", "use --gpu and four required paths; see --help");
  const std::string library = argv[2], package = argv[3], promptPath = argv[4]; t.report = fs::absolute(argv[5]);
  t.spill = t.report.string() + ".spill";
  for (const auto &path : {t.report.string(), t.report.string() + ".writing", t.report.string() + ".checkpoint.json",
       t.report.string() + ".checkpoint.json.writing", t.report.string() + ".failure.json",
       t.report.string() + ".failure.json.writing", t.spill.string()})
    require(!fs::exists(path), "fresh whole-oracle report/spill paths required");
  t.reportAllowed = true;
  earlyPolicy(); NSDictionary *buildProvenance = provenance(true);
  const auto prompt = tokens(promptPath);
  const auto binarySHA = fileHash(argv[0]), librarySHA = fileHash(library), promptSHA = fileHash(promptPath);
  id declaredLibrarySHA = buildProvenance[@"metallib_sha256"];
  require([declaredLibrarySHA isKindOfClass:[NSString class]] && [declaredLibrarySHA length] == 64 &&
      librarySHA == static_cast<NSString *>(declaredLibrarySHA).UTF8String,
      "runtime library SHA differs from immutable build closure before backend construction");
  const auto tokenizerSHA = fileHash(fs::path(package) / "tokenizer.json"), tokenizerConfigSHA = fileHash(fs::path(package) / "tokenizer_config.json");
  const auto tokensSHA = bytesHash(prompt.data(), prompt.size() * sizeof(uint32_t));
  require(promptSHA == kCanonicalFixtureSHA && tokensSHA == kCanonicalTokensSHA,
      "canonical code2048 fixture file/tokens SHA differs beforebackend construction");
  require(fs::space(t.report.parent_path()).available >= kSpillBound, "32GiB bounded spill disk headroom unavailable");
  fs::create_directory(t.spill); progress(t, false);
  pass(t, false, library, package, prompt, librarySHA);
  require(t.controlDestroyed && !t.candidateDestroyed, "control lifetime didnotend before candidate");
  pass(t, true, library, package, prompt, librarySHA);
  require(t.controlDestroyed && t.candidateDestroyed && t.candidateCheckpoints == t.controlCheckpoints,
      "sequential qualification/checkpoint count incomplete");
  require(fileHash(library) == librarySHA && fileHash(argv[0]) == binarySHA, "runtime library/binary mutated during oracle");
  require(fileHash(promptPath) == promptSHA && fileHash(fs::path(package) / "tokenizer.json") == tokenizerSHA &&
      fileHash(fs::path(package) / "tokenizer_config.json") == tokenizerConfigSHA,
      "prompt/tokenizer/config provenance mutated during oracle");
  (void)provenance(true);
  std::ofstream report(t.report.string() + ".writing"); require(bool(report), "whole report cannot beopened");
  report << "{\"schema\":\"splash-lazy-copy-whole-target-exact-v1\",\"valid\":true,\"qualification_complete\":true,\"gpu_executed\":true"
      << ",\"whole_target_48_layers\":true,\"batch_width\":1,\"batch4_exercised\":false,\"capacity\":4096,\"prefill_rows\":2048,\"verify_rows\":4"
      << ",\"kept_rows\":[1,2,3,4],\"fresh_canonical_prefill_each_case\":true,\"future_changed_singleton_rows_per_case\":4"
      << ",\"all134_physical_cache_planes_exact\":true,\"metadata_length_capacity_poison_pending_exact\":true,\"hidden_logits_compact_greedy_exact\":true"
      << ",\"all_floating_checkpoints_finite\":true,\"partial_full_terminal_abort_destroy_and_peer_guards_exercised\":true"
      << ",\"true_foreign_owner_request_exercised\":false,\"request_move_same_owner_disjointness_default_rejections_exercised\":true"
      << ",\"fusion_constructor_capture_after_opposite_env_checked\":true,\"control_backend_destroyed_before_candidate\":true,\"candidate_backend_destroyed\":true"
      << ",\"all_other_kernel_policy_operand_routes_identical\":true"
      << ",\"trained_mtp_instantiated\":false,\"semantic_quality_claim\":false,\"performance_claim\":false,\"command_timing_abi_bytes\":200"
      << ",\"executable_sha256\":" << json::quote(binarySHA) << ",\"metallib_sha256\":" << json::quote(librarySHA)
      << ",\"control_backend_loaded_metallib_sha256\":" << json::quote(t.controlLoadedLibrarySHA)
      << ",\"candidate_backend_loaded_metallib_sha256\":" << json::quote(t.candidateLoadedLibrarySHA)
      << ",\"prompt_fixture_sha256\":" << json::quote(promptSHA) << ",\"tokens_u32le_sha256\":" << json::quote(tokensSHA)
      << ",\"tokenizer_sha256\":" << json::quote(tokenizerSHA) << ",\"tokenizer_config_sha256\":" << json::quote(tokenizerConfigSHA)
      << ",\"source_identity_sha256\":" << json::quote(t.sourceIdentity) << ",\"manifest_fingerprint_sha256\":" << json::quote(t.manifestIdentity)
      << ",\"numerical_derivative_sha256\":" << json::quote(t.derivative)
      << ",\"build_provenance\":" << kLazyCopyFusionWholeBuildProvenance
      << ",\"control_routes\":" << json::quote(t.controlRoutes) << ",\"candidate_routes\":" << json::quote(t.candidateRoutes)
      << ",\"control_workspace_bytes\":" << t.controlWorkspace << ",\"candidate_workspace_bytes\":" << t.candidateWorkspace
      << ",\"control_reservation_bytes\":" << t.controlReservation << ",\"candidate_reservation_bytes\":" << t.candidateReservation
      << ",\"control_backend_peak_allocated_bytes\":" << t.controlPeak << ",\"candidate_backend_peak_allocated_bytes\":" << t.candidatePeak
      << ",\"spill_bound_bytes\":" << kSpillBound << ",\"spill_bytes\":" << t.spillBytes << ",\"comparison_scratch_bound_bytes\":" << kScratch
      << ",\"control_checkpoints\":" << t.controlCheckpoints << ",\"candidate_checkpoints\":" << t.candidateCheckpoints
      << ",\"compared_bytes\":" << t.comparedBytes << ",\"checked_planes\":" << t.checkedPlanes << ",\"guard_calls\":" << t.guardCalls
      << ",\"incoming\":[71093,12305,198,464],\"future_inputs\":[";
  for (uint32_t keep = 0; keep < 4; ++keep) {
    if (keep) report << ','; report << '[';
    for (uint32_t step = 0; step < 4; ++step) { if (step) report << ','; report << t.future[keep][step]; }
    report << ']';
  }
  report << "],\"checkpoints\":[";
  for (size_t i = 0; i < t.checkpoints.size(); ++i) {
    if (i) report << ','; const auto &entry = t.checkpoints[i];
    report << "{\"label\":" << json::quote(entry.label) << ",\"sha256\":" << json::quote(entry.sha256)
        << ",\"bytes\":" << entry.bytes << ",\"planes\":" << entry.planes << ",\"live_bytes\":" << entry.liveBytes << '}';
  }
  report << "],\"environment_except_per_instance_candidate\":"; writeEnvironment(report);
  report << ",\"scope\":\"B1 current Full512 target, identical real verification shape/inputs; every physical persistent byte is hard-gated. Diagnostic synchronous CPU spill interleaves GPU commands; no service timing/quality/MTP acceptance claim. B4, foreign owner and broader contexts unexercised.\"}\n";
  report.close(); require(bool(report), "whole report close failed"); fs::rename(t.report.string() + ".writing", t.report);
  t.phase = "complete"; std::cout << "whole-target strict B1 lazy-copy qualification passed; report=" << t.report << '\n'; return 0;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    Transcript transcript;
    try { return run(argc, argv, transcript); }
    catch (const std::exception &error) {
      try { progress(transcript, true, error.what()); } catch (...) {}
      std::cerr << "whole-target lazy-copy oracle failed: " << error.what() << '\n'; return 2;
    }
  }
}
