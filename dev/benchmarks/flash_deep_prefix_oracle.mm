// Root-run exact state oracle for a 16-row singleton verification tape.
// --cpu-self-test never constructs a Metal backend.
#include "flash/FlashForward.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash {
// This test-only friend reads request-owned state after synchronous GPU waits.
// It exposes no new production API and never changes an immutable weight.
class FlashDeepPrefixOracle final {
public:
  struct Plane final {
    std::string label;
    metal::MetalBuffer buffer;
    uint64_t bytes = 0;
  };
  static std::vector<Plane> planes(const FlashRequestState &state, bool physical) {
    if (!state.impl_) throw std::logic_error("oracle state is uninitialized");
    const auto &s = *state.impl_;
    std::vector<Plane> result;
    const auto add = [&](std::string label, const metal::MetalBuffer &buffer, uint64_t bytes) {
      if (!buffer || bytes > buffer.sizeBytes() || !buffer.contents())
        throw std::logic_error("oracle state plane has invalid storage");
      result.push_back({std::move(label), buffer, physical ? buffer.sizeBytes() : bytes});
    };
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const auto prefix = "layer" + std::to_string(layer) + ".";
      if (s.gdn[layer].recurrent) {
        add(prefix + "gdn_convolution_bf16", s.gdn[layer].convolution,
            s.gdn[layer].convolution.sizeBytes());
        add(prefix + "gdn_recurrent_f32", s.gdn[layer].recurrent,
            s.gdn[layer].recurrent.sizeBytes());
      } else {
        const auto &q = s.qsa[layer];
        add(prefix + "qsa_keys_bf16", q.keys, s.length * 512 * 2);
        add(prefix + "qsa_values_bf16", q.values, s.length * 512 * 2);
        add(prefix + "qsa_raw_index_keys_bf16", q.rawIndexKeys, s.length * 128 * 2);
        add(prefix + "qsa_complete_pooled_keys_bf16", q.pooledKeys, (s.length / 4) * 128 * 2);
        add(prefix + "qsa_positions_i64", q.indexPositions, s.length * 8);
      }
    }
    add("ple_history_i64", s.pleHistory, s.pleHistory.sizeBytes());
    add("ple_convolution_bf16", s.pleConvolution, s.pleConvolution.sizeBytes());
    return result;
  }
  static void clone(FlashRequestState &destination, const FlashRequestState &source) {
    if (!destination.impl_ || !source.impl_ || source.impl_->poisoned ||
        source.impl_->pendingVerification || destination.impl_->pendingVerification ||
        destination.impl_->owner != source.impl_->owner ||
        destination.impl_->capacity != source.impl_->capacity)
      throw std::logic_error("oracle clone requires independent healthy matching states");
    const auto from = planes(source, true), to = planes(destination, true);
    if (from.size() != to.size()) throw std::logic_error("oracle physical state layout differs");
    for (size_t plane = 0; plane < from.size(); ++plane) {
      if (from[plane].bytes != to[plane].bytes || from[plane].label != to[plane].label)
        throw std::logic_error("oracle physical plane differs");
      std::memcpy(to[plane].buffer.contents(), from[plane].buffer.contents(), from[plane].bytes);
    }
    // Preserve the destination's distinct identity and owner.
    destination.impl_->length = source.impl_->length;
    destination.impl_->poisoned = false;
    destination.impl_->pendingVerification = false;
  }
  static bool pending(const FlashRequestState &state) {
    return state.impl_ && state.impl_->pendingVerification;
  }
};
} // namespace splash::flash

namespace {
using namespace splash;
using namespace splash::flash;
constexpr uint32_t kCapacity = 4096, kPrefillRows = 128, kVocabulary = 248320;
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
struct Comparison final {
  uint64_t planes = 0, bytes = 0, mismatchedPlanes = 0;
  std::vector<std::pair<std::string, uint64_t>> first;
  bool pass() const { return !mismatchedPlanes; }
  void add(std::string label, const void *actual, const void *expected, uint64_t count) {
    ++planes; bytes += count;
    if (!count || std::memcmp(actual, expected, count) == 0) return;
    ++mismatchedPlanes;
    if (first.size() >= 8) return;
    const auto *a = static_cast<const std::byte *>(actual);
    const auto *b = static_cast<const std::byte *>(expected);
    uint64_t offset = 0;
    // Only locate the first difference on failing planes. Successful large
    // F32 recurrent planes use one optimized memcmp, without bytewise output.
    while (offset + 8 <= count && std::memcmp(a + offset, b + offset, 8) == 0) offset += 8;
    while (offset < count && a[offset] == b[offset]) ++offset;
    first.emplace_back(std::move(label), offset);
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"planes\":" << planes
        << ",\"bytes_compared\":" << bytes << ",\"mismatched_planes\":" << mismatchedPlanes
        << ",\"first_mismatch_offsets\":[";
    for (size_t index = 0; index < first.size(); ++index) {
      if (index) out << ',';
      out << "{\"plane\":" << json::quote(first[index].first)
          << ",\"byte_offset\":" << first[index].second << '}';
    }
    out << "]}";
  }
};
Comparison compareState(const FlashRequestState &a, const FlashRequestState &b) {
  require(a.logicalLength() == b.logicalLength() && a.capacity() == b.capacity() &&
          a.poisoned() == b.poisoned() && FlashDeepPrefixOracle::pending(a) == FlashDeepPrefixOracle::pending(b),
          "oracle request metadata differs");
  const auto actual = FlashDeepPrefixOracle::planes(a, false);
  const auto expected = FlashDeepPrefixOracle::planes(b, false);
  require(actual.size() == expected.size(), "oracle state plane count differs");
  Comparison result;
  for (size_t index = 0; index < actual.size(); ++index) {
    require(actual[index].label == expected[index].label && actual[index].bytes == expected[index].bytes,
            "oracle state plane geometry differs");
    result.add(actual[index].label, actual[index].buffer.contents(), expected[index].buffer.contents(), actual[index].bytes);
  }
  return result;
}
std::vector<uint16_t> logits(const FlashForwardResult &result) {
  require(result.logitRows == 1 && result.logitsBF16 && result.logitsBF16.contents() &&
          result.logitsBF16.sizeBytes() >= uint64_t{kVocabulary} * 2,
          "oracle continuation logits have invalid storage");
  std::vector<uint16_t> copy(kVocabulary);
  std::memcpy(copy.data(), result.logitsBF16.contents(), copy.size() * 2);
  return copy;
}
std::vector<uint32_t> loadTokens(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  require(data != nil, "oracle prompt JSON cannot be read");
  NSError *error = nil;
  id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSArray class]], "oracle prompt must be an array");
  std::vector<uint32_t> tokens;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] &&
            CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(),
            "oracle token must be an integer");
    const double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < kVocabulary && std::floor(value) == value,
            "oracle token is outside vocabulary");
    tokens.push_back(static_cast<uint32_t>(value));
  }
  require(tokens.size() >= 16, "oracle prompt must contain at least16 tokens");
  return tokens;
}
void pinRawProducer() {
  for (const char *name : {"SPLASH_FLASH_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_CACHE",
       "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE", "SPLASH_FLASH_DENSE_SMALL_ROWS",
       "SPLASH_FLASH_BLOCKED_MOE", "SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP",
       "SPLASH_FLASH_GDN_STAGED", "SPLASH_FLASH_CAPTURE_EXPERT_IDS"})
    require(setenv(name, "0", 1) == 0, "oracle could not pin a raw producer flag");
  require(unsetenv("SPLASH_FLASH_HOT_EXPERT_PLAN") == 0, "oracle could not disable expert cache plan");
  for (const char *name : {"SPLASH_FLASH_FUSE_GDN", "SPLASH_FLASH_FUSE_HC",
       "SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_EXPERT_QMV", "SPLASH_FLASH_MOE_Q4X8"})
    require(setenv(name, "1", 1) == 0, "oracle could not pin a qualified row producer flag");
}
void cpuSelfTest() {
  std::array<uint8_t, 17> a{}, b{};
  Comparison exact;
  exact.add("equal", a.data(), b.data(), a.size());
  require(exact.pass() && exact.bytes == 17, "oracle CPU equal comparison failed");
  b[16] = 1;
  Comparison mismatch;
  mismatch.add("tail", a.data(), b.data(), a.size());
  require(!mismatch.pass() && mismatch.first.size() == 1 && mismatch.first.front().second == 16,
          "oracle CPU tail mismatch comparison failed");
  for (uint32_t rows = 1; rows <= 16; ++rows) {
    uint32_t consumed = 0;
    while (consumed < rows) {
      const auto chunk = flashMTPCommittedFoldChunkRows(rows - consumed);
      require(chunk && *chunk <= 8, "oracle CPU committed fold chunk failed");
      consumed += *chunk;
    }
    require(consumed == rows && flashMTPConvolutionPrefix(rows), "oracle CPU bounded rows failed");
  }
  std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":[\"large_plane_exact_comparison\",\"tail_mismatch\",\"bounded_fold\",\"prefix_geometry\"]}\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        cpuSelfTest(); return 0;
      }
      if (argc != 5)
        throw std::invalid_argument("usage: flash-deep-prefix-oracle METALLIB PACKAGE TOKENS_JSON REPORT_JSON | --cpu-self-test");
      require(!std::filesystem::exists(argv[4]), "choose a fresh oracle report path");
      const auto prompt = loadTokens(argv[3]);
      pinRawProducer();
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      const uint64_t planned = FlashForward::workspacePlannedBytes(kCapacity, kPrefillRows, 16) +
                               3 * FlashForward::requestStateBytes(kCapacity);
      auto reservation = governor.tryReserve(planned);
      require(reservation.has_value(), "oracle physical memory reservation denied before construction");
      FlashForward forward(backend, weights, kCapacity, kPrefillRows, 16);
      require(forward.workspaceBytes() <= FlashForward::workspacePlannedBytes(kCapacity, kPrefillRows, 16),
              "oracle workspace exceeds reserved estimate");
      auto base = forward.createState(), actual = forward.createState(), expected = forward.createState();
      reservation->commit();
      std::array<uint32_t, 512> nonzeroPrompt{};
      for (size_t row = 0; row < nonzeroPrompt.size(); ++row) nonzeroPrompt[row] = prompt[row % prompt.size()];
      for (size_t begin = 0; begin < nonzeroPrompt.size(); begin += kPrefillRows)
        static_cast<void>(forward.forward(base, std::span(nonzeroPrompt).subspan(begin, kPrefillRows)));
      require(base.logicalLength() == 512, "oracle nonzero prefix length differs");
      struct Case final {
        uint32_t begin, retained;
        Comparison restored;
        std::array<Comparison, 2> continuationState, continuationLogits;
        bool pass() const {
          return restored.pass() && continuationState[0].pass() && continuationState[1].pass() &&
                 continuationLogits[0].pass() && continuationLogits[1].pass();
        }
      };
      std::vector<Case> cases;
      for (uint32_t phase = 0; phase < 4; ++phase) {
        std::array<uint32_t, 17> window{};
        for (size_t row = 0; row < window.size(); ++row) window[row] = prompt[(phase * 19 + row) % prompt.size()];
        FlashDeepPrefixOracle::clone(actual, base);
        FlashDeepPrefixOracle::clone(expected, base);
        bool rejected = false;
        try { static_cast<void>(forward.verify(actual, window)); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected && compareState(actual, expected).pass(), "oversized deep window changed request state");
        for (uint32_t retained = 1; retained <= 16; ++retained) {
          FlashDeepPrefixOracle::clone(actual, base);
          FlashDeepPrefixOracle::clone(expected, base);
          const auto verified = forward.verify(actual, std::span(window).first(16));
          require(verified.logitRows == 16 && FlashDeepPrefixOracle::pending(actual), "deep provisional result differs");
          static_cast<void>(forward.commitVerify(actual, retained));
          static_cast<void>(forward.forward(expected, std::span(window).first(retained)));
          Case result{static_cast<uint32_t>(base.logicalLength()), retained, compareState(actual, expected), {}, {}};
          for (uint32_t step = 0; step < 2; ++step) {
            const uint32_t incoming = prompt[(phase * 23 + retained + step) % prompt.size()];
            const auto a = logits(forward.forward(actual, std::span(&incoming, 1)));
            const auto b = logits(forward.forward(expected, std::span(&incoming, 1)));
            result.continuationLogits[step].add("continuation_logits_bf16", a.data(), b.data(), a.size() * 2);
            result.continuationState[step] = compareState(actual, expected);
          }
          std::cerr << "deep_prefix begin=" << result.begin << " retained=" << retained
                    << " restored=" << result.restored.pass() << " future_state="
                    << (result.continuationState[0].pass() && result.continuationState[1].pass())
                    << " future_logits=" << (result.continuationLogits[0].pass() && result.continuationLogits[1].pass()) << '\n';
          cases.push_back(std::move(result));
        }
        const uint32_t token = prompt[(phase + 77) % prompt.size()];
        static_cast<void>(forward.forward(base, std::span(&token, 1)));
      }
      const bool pass = std::all_of(cases.begin(), cases.end(), [](const auto &entry) { return entry.pass(); });
      std::ofstream out(argv[4]);
      out << "{\"pass\":" << (pass ? "true" : "false") << ",\"case_count\":" << cases.size()
          << ",\"source_identity\":" << json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << json::quote(weights.manifestFingerprint())
          << ",\"kernel_routes\":" << json::quote(forward.kernelRoutes())
          << ",\"producer_policy\":\"raw row-independent coefficients; dense caches and QSA MPP/F32 disabled; fused GDN capture enabled\""
          << ",\"state_scope\":\"all physical GDN/PLE state and valid QSA prefixes; provisional QSA rows beyond logical length are ignored\""
          << ",\"nonzero_prefill_rows\":512,\"begin_modulo4\":[0,1,2,3],\"verify_rows\":16,\"continuation_steps\":2,\"cases\":[";
      for (size_t index = 0; index < cases.size(); ++index) {
        if (index) out << ',';
        const auto &entry = cases[index];
        out << "{\"begin\":" << entry.begin << ",\"retained\":" << entry.retained
            << ",\"pass\":" << (entry.pass() ? "true" : "false") << ",\"restored_state\":";
        entry.restored.write(out);
        out << ",\"continuation\":[";
        for (uint32_t step = 0; step < 2; ++step) {
          if (step) out << ',';
          out << "{\"state\":"; entry.continuationState[step].write(out);
          out << ",\"logits\":"; entry.continuationLogits[step].write(out); out << '}';
        }
        out << "]}";
      }
      out << "]}\n";
      require(bool(out), "oracle report write failed");
      return pass ? 0 : 2;
    } catch (const std::exception &error) {
      std::cerr << "flash-deep-prefix-oracle: " << error.what() << '\n'; return 1;
    }
  }
}
