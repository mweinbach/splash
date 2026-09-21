// Root-run isolated primitive qualification. CPU mode creates no Metal backend.
#include "flash/FlashPLESSD.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint8_t kGuard = 0xa7;
constexpr uint64_t kShardRows = 11;
constexpr uint64_t kTableRows = kShardRows * 128;

void require(bool yes, const std::string &message) {
  if (!yes) throw std::runtime_error(message);
}
uint16_t bf16(float value) {
  const uint32_t raw = std::bit_cast<uint32_t>(value);
  if ((raw & 0x7f800000u) == 0x7f800000u)
    return uint16_t((raw >> 16) | ((raw & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}
std::string sha256(const void *bytes, uint64_t count) {
  CC_SHA256_CTX state{};
  CC_SHA256_Init(&state);
  auto *next = static_cast<const uint8_t *>(bytes);
  while (count) {
    const auto n = CC_LONG(std::min<uint64_t>(count, UINT32_MAX));
    CC_SHA256_Update(&state, next, n);
    next += n; count -= n;
  }
  std::array<uint8_t, 32> digest{};
  CC_SHA256_Final(digest.data(), &state);
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (auto byte : digest) { result += digits[byte >> 4]; result += digits[byte & 15]; }
  return result;
}
template<class T> MetalBuffer buffer(MetalBackend &backend, const std::vector<T> &values) {
  auto result = backend.allocateBuffer(values.size() * sizeof(T));
  std::memcpy(result.contents(), values.data(), values.size() * sizeof(T));
  return result;
}
template<class T> FlashTensor tensor(MetalBackend &backend, const std::vector<T> &values,
                                     FlashDType dtype) {
  return {buffer(backend, values), dtype, {values.size()}, values.size() * sizeof(T)};
}
struct Guarded final {
  MetalBuffer base, view;
  uint64_t bytes;
  Guarded(MetalBackend &backend, uint64_t length) : bytes(length) {
    base = backend.allocateBuffer(length + 128);
    std::memset(base.contents(), kGuard, base.sizeBytes());
    view = backend.view(base, 64, length);
  }
  void check() const {
    const auto *p = static_cast<const uint8_t *>(base.contents());
    for (uint64_t i = 0; i < 64; ++i)
      require(p[i] == kGuard && p[64 + bytes + i] == kGuard, "SSD oracle canary changed");
  }
};

struct Payload final {
  std::filesystem::path directory, path;
  uint64_t weightStride, parameterStride;
  std::vector<uint8_t> bytes;
  std::vector<FlashPLESSDStore::Part> parts;
  std::shared_ptr<FlashPLESSDStore> store;
  std::vector<int64_t> multipliers{INT64_MAX, -17, INT64_MIN + 3};
  std::vector<int64_t> sizes = std::vector<int64_t>(16, 67);
  std::vector<int64_t> offsets = std::vector<int64_t>(16);

  explicit Payload(bool padded, bool nonfinite = false, uint64_t cacheBytes = 65536)
      : weightStride(padded ? 96 : 80), parameterStride(padded ? 16 : 10) {
    // Both first/last physical shards and wraparound I64 products are exercised.
    for (uint32_t h = 0; h < 16; ++h)
      offsets[h] = h == 15 ? kTableRows - sizes[h] : (h * 89) % (kTableRows - sizes[h]);
    std::array<char, 128> temporary{};
    const std::string prefix = (std::filesystem::temp_directory_path() /
        "splash-ple-ssd-oracle-XXXXXX").string();
    require(prefix.size() < temporary.size(), "temporary directory path too long");
    std::memcpy(temporary.data(), prefix.c_str(), prefix.size() + 1);
    char *created = ::mkdtemp(temporary.data());
    require(created, "cannot create synthetic payload directory");
    directory = created; path = directory / "original-q4-g32.bin";
    bytes.resize(37, kGuard); // Deliberately nonzero file offsets.
    auto plane = [&](uint64_t stride) {
      const uint64_t offset = bytes.size();
      bytes.resize(offset + kShardRows * stride + 19, kGuard);
      return FlashPLESSDStore::Plane{0, offset, stride};
    };
    for (uint32_t shard = 0; shard < 128; ++shard) {
      auto weights = plane(weightStride), scales = plane(parameterStride), biases = plane(parameterStride);
      parts.push_back({kShardRows, weights, scales, biases});
      for (uint32_t row = 0; row < kShardRows; ++row) {
        const uint64_t id = uint64_t(shard) * kShardRows + row;
        for (uint32_t channel = 0; channel < 160; channel += 2) {
          const uint32_t a = (id + channel * 7) % 16;
          const uint32_t b = (id + (channel + 1) * 7) % 16;
          bytes[weights.offset + row * weightStride + channel / 2] = uint8_t(a | (b << 4));
        }
        for (uint32_t g = 0; g < 5; ++g) {
          const uint16_t s = bf16((shard % 2 ? -1.0f : 1.0f) * float(g + 1) / 32);
          const uint16_t b = nonfinite ? bf16(std::numeric_limits<float>::quiet_NaN()) :
              bf16(float(int((shard + row + g) % 17) - 8) / 16);
          std::memcpy(bytes.data() + scales.offset + row * parameterStride + g * 2, &s, 2);
          std::memcpy(bytes.data() + biases.offset + row * parameterStride + g * 2, &b, 2);
        }
      }
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(bool(file), "cannot create synthetic original payload");
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    file.close(); require(bool(file), "cannot finish synthetic original payload");
    FlashPLESSDStore::Options options;
    options.cacheBytes = cacheBytes; options.maxCoalescedReadBytes = 4096; options.noCache = true;
    store = std::make_shared<FlashPLESSDStore>(
        std::vector<FlashPLESSDStore::Source>{{path, bytes.size()}}, parts, options);
  }
  ~Payload() {
    store.reset();
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
  Payload(const Payload &) = delete;
  Payload &operator=(const Payload &) = delete;
  std::vector<uint8_t> rawRow(uint64_t id) const {
    require(id < kTableRows, "raw synthetic row outside table");
    const auto &part = parts[id / kShardRows];
    const uint64_t row = id % kShardRows;
    std::vector<uint8_t> result(100);
    std::memcpy(result.data(), bytes.data() + part.weights.offset + row * weightStride, 80);
    std::memcpy(result.data() + 80, bytes.data() + part.scales.offset + row * parameterStride, 10);
    std::memcpy(result.data() + 90, bytes.data() + part.biases.offset + row * parameterStride, 10);
    return result;
  }
  FlashPLEWeights resident(MetalBackend &backend) const {
    FlashPLEWeights result;
    result.multipliers = tensor(backend, multipliers, FlashDType::I64);
    result.headVocabularySizes = tensor(backend, sizes, FlashDType::I64);
    result.headOffsets = tensor(backend, offsets, FlashDType::I64);
    result.sharedScale = tensor(backend, std::vector<uint16_t>{bf16(0.00019931793212890625f)}, FlashDType::BF16);
    auto nativePlane = [&](FlashPLESSDStore::Plane plane, uint64_t stride) {
      auto owner = backend.allocateBuffer(kShardRows * stride + 128);
      std::memset(owner.contents(), kGuard, owner.sizeBytes());
      std::memcpy(static_cast<uint8_t *>(owner.contents()) + 64, bytes.data() + plane.offset,
                  kShardRows * stride);
      return backend.view(owner, 64, kShardRows * stride);
    };
    for (const auto &part : parts)
      result.shards.push_back({nativePlane(part.weights, weightStride),
          nativePlane(part.scales, parameterStride), nativePlane(part.biases, parameterStride),
          kShardRows, weightStride, parameterStride});
    return result;
  }
};

std::vector<int64_t> tokensFor(FlashPLEGeometry g, uint64_t seed = 19) {
  std::vector<int64_t> values(uint64_t(g.lanes) * g.rows);
  for (uint32_t lane = 0; lane < g.lanes; ++lane)
    for (uint32_t row = 0; row < g.rows; ++row)
      values[uint64_t(lane) * g.rows + row] = row % 7 == 3 ? g.eosToken :
          int64_t((uint64_t(row) * 1543 + lane * 9137 + seed) % g.vocabularySize);
  return values;
}
std::vector<int64_t> historiesFor(FlashPLEGeometry g, bool cold) {
  std::vector<int64_t> values(uint64_t(g.lanes) * 2);
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    values[lane * 2] = cold || lane == 0 ? g.eosToken : lane * 23 + 7;
    values[lane * 2 + 1] = cold || lane == 0 ? g.eosToken : lane * 13 + 5;
  }
  return values;
}
// Independent formulation: explicit timeline indexing, unsigned 128-bit
// products truncated to I64, and signed 128-bit positive modulo. Production
// uses a running two-token history and native uint64/int64 arithmetic.
std::vector<int64_t> independentIDs(const Payload &payload,
                                  std::span<const int64_t> tokens,
                                  std::span<int64_t> histories,
                                  FlashPLEGeometry g) {
  const auto oldHistory = std::vector<int64_t>(histories.begin(), histories.end());
  std::vector<int64_t> result(uint64_t(g.lanes) * g.rows * 16);
  auto product = [](int64_t token, int64_t multiplier) {
    return uint64_t(static_cast<unsigned __int128>(uint64_t(token)) * uint64_t(multiplier));
  };
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    auto timeline = [&](int64_t at) -> int64_t {
      return at < 0 ? oldHistory[lane * 2 + uint64_t(at + 2)] :
          tokens[uint64_t(lane) * g.rows + uint64_t(at)];
    };
    for (uint32_t row = 0; row < g.rows; ++row) {
      const int64_t current = timeline(row), previous = timeline(int64_t(row) - 1);
      const int64_t older = previous == g.eosToken ? g.eosToken : timeline(int64_t(row) - 2);
      for (uint32_t head = 0; head < 16; ++head) {
        uint64_t bits = product(current, payload.multipliers[0]) ^ product(previous, payload.multipliers[1]);
        if (head >= 8) bits ^= product(older, payload.multipliers[2]);
        const __int128 signedValue = bits & (uint64_t(1) << 63) ?
            static_cast<__int128>(bits) - (static_cast<__int128>(1) << 64) : static_cast<__int128>(bits);
        const __int128 modulus = payload.sizes[head];
        const __int128 remainder = ((signedValue % modulus) + modulus) % modulus;
        result[(uint64_t(lane) * g.rows + row) * 16 + head] = int64_t(remainder) + payload.offsets[head];
      }
    }
    histories[lane * 2] = timeline(int64_t(g.rows) - 2);
    histories[lane * 2 + 1] = timeline(int64_t(g.rows) - 1);
  }
  return result;
}
void compare(const MetalBuffer &a, const MetalBuffer &b, uint64_t bytes, const char *name) {
  require(std::memcmp(a.contents(), b.contents(), bytes) == 0,
          std::string("SSD exact comparison failed: ") + name);
}

uint32_t qualifyPrefixes(MetalBackend &backend, Payload &payload,
                        const FlashPLEWeights &resident, const FlashPLEWeights &disk,
                        FlashPLESSD &staged, FlashPLEGeometry g,
                        const std::vector<int64_t> &tokens,
                        const std::vector<int64_t> &history) {
  if (g.rows > 16) return 0; // Large prefill is not a speculative tape.
  const uint64_t hyper = uint64_t(g.width) * g.streams;
  auto oldHistory = buffer(backend, history), tokenBuffer = buffer(backend, tokens);
  std::vector<uint16_t> before(uint64_t(g.lanes) * 9 * hyper);
  std::vector<uint16_t> normalized(uint64_t(g.lanes) * g.rows * hyper);
  for (uint64_t i = 0; i < before.size(); ++i) before[i] = bf16(float(int(i % 29) - 14) / 8);
  for (uint64_t i = 0; i < normalized.size(); ++i) normalized[i] = bf16(float(int(i % 31) - 15) / 16);
  auto beforeState = buffer(backend, before), normalizedInputs = buffer(backend, normalized);
  Guarded historyA(backend, history.size() * 8), historyB(backend, history.size() * 8);
  Guarded stateA(backend, before.size() * 2), stateB(backend, before.size() * 2);
  std::vector<uint32_t> retained(g.lanes);
  auto kept = buffer(backend, retained), diagA = buffer(backend, std::vector<uint32_t>{kSticky});
  auto diagB = buffer(backend, std::vector<uint32_t>{kSticky});
  std::vector<uint32_t> windows{0, std::min(g.rows, 1u), g.rows / 2, g.rows};
  std::sort(windows.begin(), windows.end()); windows.erase(std::unique(windows.begin(), windows.end()), windows.end());
  uint32_t qualified = 0;
  for (uint32_t retain : windows) {
    for (uint32_t lane = 0; lane < g.lanes; ++lane) retained[lane] = (retain + lane) % (g.rows + 1);
    std::memcpy(kept.contents(), retained.data(), retained.size() * 4);
    *static_cast<uint32_t *>(diagA.contents()) = kSticky;
    *static_cast<uint32_t *>(diagB.contents()) = kSticky;
    CommandGraph a, b;
    addPLERestorePrefix(a, oldHistory, tokenBuffer, beforeState, normalizedInputs, kept,
                        historyA.view, stateA.view, diagA, g);
    addPLERestorePrefix(b, oldHistory, tokenBuffer, beforeState, normalizedInputs, kept,
                        historyB.view, stateB.view, diagB, g);
    (void)backend.submitCommand(a.dispatches()); (void)backend.submitCommand(b.dispatches());
    compare(historyA.view, historyB.view, history.size() * 8, "prefix histories");
    compare(stateA.view, stateB.view, before.size() * 2, "prefix convolution states");
    std::vector<int64_t> expectedHistory(history.size());
    for (uint32_t lane = 0; lane < g.lanes; ++lane) {
      for (uint32_t slot = 0; slot < 2; ++slot) {
        const uint32_t at = retained[lane] + slot;
        expectedHistory[lane * 2 + slot] = at < 2 ? history[lane * 2 + at] :
            tokens[uint64_t(lane) * g.rows + at - 2];
      }
      for (uint32_t row = 0; row < 9; ++row) {
        const uint64_t at = retained[lane] + row;
        const uint16_t *source = at < 9 ? before.data() + (uint64_t(lane) * 9 + at) * hyper :
            normalized.data() + (uint64_t(lane) * g.rows + at - 9) * hyper;
        require(std::memcmp(static_cast<const uint16_t *>(stateA.view.contents()) +
                    (uint64_t(lane) * 9 + row) * hyper, source, hyper * 2) == 0,
                "independent CPU prefix convolution mismatch");
      }
    }
    require(std::memcmp(historyA.view.contents(), expectedHistory.data(), history.size() * 8) == 0,
            "independent CPU prefix history mismatch");
    FlashPLEGeometry next = g; next.rows = 1;
    auto nextTokens = tokensFor(next, 711);
    auto nextTokenBuffer = buffer(backend, nextTokens);
    Guarded idsA(backend, uint64_t(next.lanes) * 16 * 8), idsB(backend, uint64_t(next.lanes) * 16 * 8);
    Guarded outA(backend, uint64_t(next.lanes) * 2560 * 2), outB(backend, uint64_t(next.lanes) * 2560 * 2);
    staged.prepare(nextTokens, expectedHistory, disk, next);
    CommandGraph continueA, continueB;
    addPLENgramIDs(continueA, resident, nextTokenBuffer, historyA.view, idsA.view, diagA, next);
    addPLEGather(continueA, resident, idsA.view, outA.view, diagA, next);
    staged.addHashGather(continueB, disk, nextTokenBuffer, historyB.view, idsB.view, outB.view, diagB, next);
    (void)backend.submitCommand(continueA.dispatches()); (void)backend.submitCommand(continueB.dispatches());
    compare(idsA.view, idsB.view, idsA.bytes, "prefix continuation IDs");
    compare(outA.view, outB.view, outA.bytes, "prefix continuation BF16 gather");
    compare(historyA.view, historyB.view, history.size() * 8, "prefix continuation history");
    require(*static_cast<uint32_t *>(diagA.contents()) == kSticky &&
            *static_cast<uint32_t *>(diagB.contents()) == kSticky,
            "prefix continuation corrupted sticky diagnostics");
    for (const auto *guard : {&historyA, &historyB, &stateA, &stateB, &idsA, &idsB, &outA, &outB}) guard->check();
    ++qualified;
  }
  return qualified;
}

NSDictionary *run(MetalBackend &backend, Payload &payload, uint32_t lanes, uint32_t rows,
                  bool cold, const std::string &label, int invalid = 0) {
  auto resident = payload.resident(backend), disk = resident;
  disk.shards.clear(); disk.diskTableRows = kTableRows;
  FlashPLESSD staged(backend, payload.store, disk, lanes, rows);
  FlashPLEGeometry g; g.lanes = lanes; g.rows = rows;
  auto tokens = tokensFor(g), history = historiesFor(g, cold);
  auto tokenBuffer = buffer(backend, tokens), historyA = buffer(backend, history), historyB = buffer(backend, history);
  const uint64_t selections = uint64_t(lanes) * rows * 16;
  Guarded idsA(backend, selections * 8), idsB(backend, selections * 8);
  Guarded outA(backend, selections * 160 * 2), outB(backend, selections * 160 * 2);
  auto diagA = buffer(backend, std::vector<uint32_t>{kSticky}), diagB = buffer(backend, std::vector<uint32_t>{kSticky});
  const auto beforeStats = payload.store->statistics();
  const auto prepareStart = std::chrono::steady_clock::now();
  staged.prepare(tokens, history, disk, g);
  const double prepareSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepareStart).count();
  auto expectedHistory = history;
  const auto expectedIDs = independentIDs(payload, tokens, expectedHistory, g);
  const auto scratch = staged.scratchBuffers();
  require(scratch.size() == 2 && staged.allocatedBytes() == FlashPLESSD::plannedBytes(lanes, rows),
          "SSD staging allocation differs from admission");
  require(std::memcmp(scratch[0].contents(), expectedIDs.data(), expectedIDs.size() * 8) == 0,
          "prepared SSD IDs differ from independent CPU hash");
  // Check canonical raw rows even when hashes collide or cache evictions occur.
  for (uint64_t i = 0; i < selections; ++i) {
    auto raw = payload.rawRow(expectedIDs[i]);
    require(std::memcmp(static_cast<const uint8_t *>(scratch[1].contents()) + i * 100,
                        raw.data(), 100) == 0, "prepared SSD source row differs");
  }
  if (invalid == 1) {
    // Change only GPU-visible metadata AFTER host preparation. GPU agreement must fail closed.
    auto *multipliers = static_cast<int64_t *>(disk.multipliers.buffer.contents());
    multipliers[0] ^= 0x5f3759df;
  }
  CommandGraph a, b;
  addPLENgramIDs(a, resident, tokenBuffer, historyA, idsA.view, diagA, g);
  addPLEGather(a, resident, idsA.view, outA.view, diagA, g);
  staged.addHashGather(b, disk, tokenBuffer, historyB, idsB.view, outB.view, diagB, g);
  const auto timingA = backend.submitCommand(a.dispatches());
  const auto timingB = backend.submitCommand(b.dispatches());
  {
    CommandGraph stale;
    bool rejected = false;
    try { staged.addHashGather(stale, disk, tokenBuffer, historyB, idsB.view, outB.view, diagB, g); }
    catch (const std::logic_error &) { rejected = true; }
    require(rejected && stale.dispatches().empty(), "SSD preparation was reused by a second graph");
  }
  compare(idsA.view, idsB.view, selections * 8, "IDs");
  compare(historyA, historyB, history.size() * 8, "completed token history");
  uint64_t mismatchedSelections = 0;
  if (invalid == 1) {
    const auto *actual = static_cast<const int64_t *>(idsB.view.contents());
    const auto *outputA = static_cast<const uint16_t *>(outA.view.contents());
    const auto *outputB = static_cast<const uint16_t *>(outB.view.contents());
    for (uint64_t i = 0; i < selections; ++i) {
      if (actual[i] != expectedIDs[i]) {
        ++mismatchedSelections;
        for (uint32_t c = 0; c < 160; ++c)
          require((outputB[i * 160 + c] & 0x7f80) == 0x7f80 &&
                  (outputB[i * 160 + c] & 0x7f) != 0, "stale GPU ID did not produce BF16 NaN");
      } else {
        require(std::memcmp(outputA + i * 160, outputB + i * 160, 320) == 0,
                "matching GPU ID did not retain original gather bytes");
      }
    }
    require(mismatchedSelections && *static_cast<uint32_t *>(diagA.contents()) == kSticky &&
            *static_cast<uint32_t *>(diagB.contents()) == (kSticky | kFlashPLEInvalidIndex),
            "SSD stale-ID agreement did not preserve prior sticky bit and index diagnostic");
    std::memcpy(disk.multipliers.buffer.contents(), payload.multipliers.data(), 24);
  } else {
    compare(outA.view, outB.view, selections * 160 * 2, "BF16 original affine/shared-scale stages");
    compare(diagA, diagB, 4, "sticky diagnostics");
    require(*static_cast<uint32_t *>(diagB.contents()) ==
                (kSticky | (invalid == 2 ? kFlashPLEInvalidNumeric : 0)),
            "SSD gather diagnostics differed from expected sticky mask");
    require(std::memcmp(idsB.view.contents(), expectedIDs.data(), selections * 8) == 0 &&
            std::memcmp(historyB.contents(), expectedHistory.data(), history.size() * 8) == 0,
            "GPU hash/history differed from independent CPU reference");
  }
  require(std::memcmp(tokenBuffer.contents(), tokens.data(), tokens.size() * 8) == 0,
          "SSD gather modified incoming tokens");
  for (const auto *guard : {&idsA, &idsB, &outA, &outB}) guard->check();
  const uint32_t prefixes = !invalid ? qualifyPrefixes(backend, payload, resident, disk, staged, g, tokens, history) : 0;
  // Bad tokens reject host preparation without changing request state or staging payload.
  uint32_t rejectedTokens = 0;
  if (!invalid) {
    const auto stageIDHash = sha256(scratch[0].contents(), scratch[0].sizeBytes());
    const auto stageRowHash = sha256(scratch[1].contents(), scratch[1].sizeBytes());
    const auto requestHistoryHash = sha256(historyB.contents(), history.size() * 8);
    const auto requestIDHash = sha256(idsB.view.contents(), idsB.bytes);
    const auto requestOutHash = sha256(outB.view.contents(), outB.bytes);
    for (int64_t bad : {-1LL, int64_t(g.vocabularySize)}) {
      auto invalidTokens = tokens; invalidTokens[rows / 2] = bad;
      bool rejected = false;
      try { staged.prepare(invalidTokens, history, disk, g); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "SSD host preparation accepted an invalid token");
      require(stageIDHash == sha256(scratch[0].contents(), scratch[0].sizeBytes()) &&
              stageRowHash == sha256(scratch[1].contents(), scratch[1].sizeBytes()) &&
              requestHistoryHash == sha256(historyB.contents(), history.size() * 8) &&
              requestIDHash == sha256(idsB.view.contents(), idsB.bytes) &&
              requestOutHash == sha256(outB.view.contents(), outB.bytes),
              "rejected SSD preparation modified staging or request state");
      CommandGraph rejectedGraph;
      bool gatherRejected = false;
      try { staged.addHashGather(rejectedGraph, disk, tokenBuffer, historyB, idsB.view, outB.view, diagB, g); }
      catch (const std::logic_error &) { gatherRejected = true; }
      require(gatherRejected && rejectedGraph.dispatches().empty(),
              "SSD gather permitted stale preparation after an invalid token");
      ++rejectedTokens;
    }
  }
  const auto afterStats = payload.store->statistics();
  std::cout << label << " B=" << lanes << " R=" << rows << " PASS prefixes=" << prefixes << std::endl;
  return @{@"label": @(label.c_str()), @"lanes": @(lanes), @"rows": @(rows), @"cold_history": @(cold),
      @"invalid_case": @(invalid), @"exact_original_bf16": @(invalid != 1), @"exact_ids_history": @YES,
      @"guards_intact": @YES, @"prior_diagnostic_preserved": @YES, @"prefix_windows": @(prefixes),
      @"second_graph_requires_prepare": @YES,
      @"host_rejected_invalid_tokens": @(rejectedTokens), @"stale_id_nan_selections": @(mismatchedSelections),
      @"source_payload_sha256": @(sha256(payload.bytes.data(), payload.bytes.size()).c_str()),
      @"ids_sha256": @(sha256(idsB.view.contents(), idsB.bytes).c_str()),
      @"output_sha256": @(sha256(outB.view.contents(), outB.bytes).c_str()),
      @"diagnostics": @(*static_cast<uint32_t *>(diagB.contents())),
      @"staging_bytes": @(staged.allocatedBytes()), @"prepare_seconds": @(prepareSeconds),
      @"resident_gpu_seconds": @(timingA.gpuSeconds), @"staged_gpu_seconds": @(timingB.gpuSeconds),
      @"resident_wall_seconds": @(timingA.wallSeconds), @"staged_wall_seconds": @(timingB.wallSeconds),
      @"resident_dispatches": @(a.dispatches().size()), @"staged_dispatches": @(b.dispatches().size()),
      @"read_requests": @(afterStats.readRequests - beforeStats.readRequests),
      @"completed_read_bytes": @(afterStats.completedReadBytes - beforeStats.completedReadBytes),
      @"unique_miss_rows": @(afterStats.uniqueMissRows - beforeStats.uniqueMissRows),
      @"cache_hit_rows": @(afterStats.cacheHitRows - beforeStats.cacheHitRows)};
}

// This path constructs shared Metal buffers but submits NO GPU commands. It
// verifies the execution-level guarantee beyond the store's independent tests.
NSDictionary *sourceFailureNoSubmission(MetalBackend &backend) {
  Payload payload(true);
  auto resident = payload.resident(backend), disk = resident;
  disk.shards.clear(); disk.diskTableRows = kTableRows;
  FlashPLEGeometry g; g.lanes = 2; g.rows = 4;
  FlashPLESSD staged(backend, payload.store, disk, g.lanes, g.rows);
  auto tokens = tokensFor(g), histories = historiesFor(g, false);
  auto tokenBuffer = buffer(backend, tokens), history = buffer(backend, histories);
  Guarded ids(backend, uint64_t(g.lanes) * g.rows * 16 * 8);
  Guarded output(backend, uint64_t(g.lanes) * g.rows * 2560 * 2);
  auto diagnostics = buffer(backend, std::vector<uint32_t>{kSticky});
  staged.prepare(tokens, histories, disk, g);
  const uint64_t beforeSubmissions = backend.submissionCount();
  const auto historyHash = sha256(history.contents(), history.sizeBytes());
  const auto idsHash = sha256(ids.view.contents(), ids.bytes);
  const auto outputHash = sha256(output.view.contents(), output.bytes);
  std::filesystem::resize_file(payload.path, payload.bytes.size() - 1);
  bool rejected = false;
  try { staged.prepare(tokens, histories, disk, g); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected && payload.store->statistics().poisoned,
          "SSD source change was not rejected and poisoned");
  CommandGraph graph;
  bool graphRejected = false;
  try { staged.addHashGather(graph, disk, tokenBuffer, history, ids.view, output.view, diagnostics, g); }
  catch (const std::logic_error &) { graphRejected = true; }
  require(graphRejected && graph.dispatches().empty() && backend.submissionCount() == beforeSubmissions,
          "failed SSD preparation retained a graph or submitted GPU work");
  require(historyHash == sha256(history.contents(), history.sizeBytes()) &&
          idsHash == sha256(ids.view.contents(), ids.bytes) &&
          outputHash == sha256(output.view.contents(), output.bytes) &&
          *static_cast<uint32_t *>(diagnostics.contents()) == kSticky,
          "failed SSD preparation changed request history, IDs, output, or diagnostics");
  ids.check(); output.check();
  return @{@"source_change_rejects_prepare": @YES, @"store_poisoned": @YES,
      @"no_stale_graph": @YES, @"zero_gpu_submissions": @YES, @"request_state_unchanged": @YES};
}

void cpuSelfTest() {
  uint64_t checks = 0;
  for (bool padded : {false, true}) {
    for (uint64_t cacheBytes : {uint64_t(0), uint64_t(4096), uint64_t(65536)}) {
      Payload payload(padded, false, cacheBytes);
      std::vector<int64_t> ids{0, 10, 11, 127 * int64_t(kShardRows), int64_t(kTableRows - 1), 0, 11};
      std::vector<uint8_t> gathered(ids.size() * 100);
      payload.store->lookupRows(ids, gathered);
      for (uint64_t i = 0; i < ids.size(); ++i) {
        auto raw = payload.rawRow(ids[i]);
        require(std::memcmp(raw.data(), gathered.data() + i * 100, 100) == 0,
                "CPU synthetic unchanged source row mismatch"); ++checks;
      }
      for (uint32_t lanes : {1u, 2u, 4u}) {
        for (uint32_t rows : {1u, 4u, 16u, 2048u}) {
          FlashPLEGeometry g; g.lanes = lanes; g.rows = rows;
          auto tokens = tokensFor(g), history = historiesFor(g, false), oldHistory = history;
          auto hashed = computePLENgramIDs(tokens, history, payload.multipliers, payload.sizes,
                                           payload.offsets, g, kTableRows);
          auto independentlyUpdatedHistory = oldHistory;
          const auto independent = independentIDs(payload, tokens, independentlyUpdatedHistory, g);
          require(independent == hashed && independentlyUpdatedHistory == history,
                  "CPU production hash differs from independent 128-bit timeline oracle"); ++checks;
          require(hashed.size() == uint64_t(lanes) * rows * 16, "CPU synthetic ID extent mismatch");
          for (uint64_t i = 0; i < hashed.size(); ++i) {
            const uint32_t head = i % 16;
            require(hashed[i] >= payload.offsets[head] &&
                    hashed[i] < payload.offsets[head] + payload.sizes[head],
                    "CPU wrapped hash outside original head range"); ++checks;
          }
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            require(history[lane * 2] == (rows == 1 ? oldHistory[lane * 2 + 1] :
                    tokens[uint64_t(lane) * rows + rows - 2]) &&
                    history[lane * 2 + 1] == tokens[uint64_t(lane) * rows + rows - 1],
                    "CPU chronological history update mismatch"); ++checks;
          }
          for (int64_t bad : {-1LL, int64_t(g.vocabularySize)}) {
            auto invalid = tokens; invalid[rows / 2] = bad; auto safeHistory = oldHistory;
            bool rejected = false;
            try { (void)computePLENgramIDs(invalid, safeHistory, payload.multipliers,
                    payload.sizes, payload.offsets, g, kTableRows); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected && safeHistory == oldHistory, "CPU invalid hash changed history"); ++checks;
          }
        }
      }
      require(payload.store->statistics().cacheAccountedBytes <= cacheBytes, "CPU cache exceeded admitted bytes");
      ++checks;
    }
  }
  std::cout << "PLE SSD oracle CPU self-test PASS checks=" << checks
            << "; no Metal backend or GPU submissions" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      require(argc == 4 && std::string(argv[3]) == "--run-root-gpu",
          "usage: flash-ple-ssd-oracle <metallib> <fresh-report.json> --run-root-gpu | --cpu-self-test");
      MetalBackend backend(argv[1]);
      NSDictionary *sourceFailure = sourceFailureNoSubmission(backend);
      NSMutableArray *cases = [NSMutableArray array];
      Payload packed(false), padded(true), noCache(true, false, 0), nonfinite(true, true);
      for (uint32_t lanes : {1u, 2u, 4u})
        for (uint32_t rows : {1u, 4u, 16u, 2048u})
          [cases addObject:run(backend, padded, lanes, rows, false, "padded-original-mixed-history")];
      [cases addObject:run(backend, packed, 4, 16, true, "packed-original-cold-history")];
      [cases addObject:run(backend, noCache, 2, 16, false, "padded-original-zero-row-cache")];
      [cases addObject:run(backend, padded, 2, 4, false, "gpu-hash-metadata-changed-after-prepare", 1)];
      [cases addObject:run(backend, nonfinite, 2, 4, false, "nonfinite-original-bias", 2)];
      static constexpr char digits[] = "0123456789abcdef";
      std::string libHash;
      for (auto byte : backend.metallibSha256()) { libHash += digits[byte >> 4]; libHash += digits[byte & 15]; }
      NSDictionary *report = @{@"schema": @"splash-flash-ple-ssd-primitive-qualification-v1", @"valid": @YES,
          @"synthetic_original_q4_g32_only": @YES, @"real_model_loaded": @NO,
          @"strict_original_bf16_and_i64": @YES, @"ssd_raw_rows_unchanged": @YES,
          @"independent_hash_reference": @"explicit-timeline-u128-truncation-s128-positive-modulo",
          @"staging_bounded": @YES, @"head_width": @160, @"source_parts": @128,
          @"disk_only_weights_empty_shards": @YES, @"device": @(backend.capabilities().deviceName.c_str()),
          @"metallib_sha256": @(libHash.c_str()), @"submissions": @(backend.submissionCount()),
          @"source_failure_execution_guard": sourceFailure,
          @"shader_validation": @(std::getenv("MTL_SHADER_VALIDATION") ?: "unset"),
          @"timings_are_single_qualification_samples": @YES, @"cases": cases};
      NSError *error = nil;
      NSData *json = [NSJSONSerialization dataWithJSONObject:report
          options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
      require(json, "cannot encode PLE SSD oracle report");
      require([json writeToFile:@(argv[2]) atomically:YES], "cannot write PLE SSD oracle report");
      std::cout << "PLE SSD primitive all cases PASS: " << cases.count << std::endl;
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "PLE SSD oracle FAIL: " << error.what() << std::endl;
      return 1;
    }
  }
}
