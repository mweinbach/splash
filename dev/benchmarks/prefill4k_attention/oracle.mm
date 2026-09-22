// Root-run GPU comparison. Compilation and --cpu-self-test submit no GPU work.
// Prepared BF16 inputs avoid mixing norm/cache/indexer costs into attention.
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashQSAFast.hpp"
#include "engine/Json.hpp"
#include "metal/abi/FlashQSAFast.h"
#include "metal/abi/FlashQSARowTiles.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::CommandTiming;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
static_assert(sizeof(CommandTiming) == 200, "Recompile the entire private oracle on timing ABI changes");
constexpr uint32_t kSticky = 0x40000000;
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint64_t kGuardWords = 32;
constexpr uint32_t kOutputWidth = 6144;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
float number(uint16_t value) {
  return std::bit_cast<float>(uint32_t(value) << 16);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  if ((word & 0x7f800000u) == 0x7f800000u)
    return uint16_t((word >> 16) | ((word & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
float seeded(uint64_t index, uint64_t seed, float scale) {
  return float(int32_t(randomWord(index + seed) % 2049) - 1024) * (scale / 1024.0f);
}
uint16_t gated(uint16_t attention, uint16_t source) {
  const auto exponential = bf16(std::exp(std::abs(number(source))));
  const auto denominator = bf16(1.0f + number(exponential));
  const auto tail = bf16(1.0f / number(denominator));
  const auto sigmoid = number(source) < 0 ? tail : bf16(1.0f - number(tail));
  return bf16(number(attention) * number(sigmoid));
}

struct SHA256 final {
  CC_SHA256_CTX context{};
  SHA256() { CC_SHA256_Init(&context); }
  void add(const void *data, uint64_t bytes) {
    const auto *next = static_cast<const std::byte *>(data);
    while (bytes) {
      const auto count = CC_LONG(std::min<uint64_t>(bytes, std::numeric_limits<CC_LONG>::max()));
      CC_SHA256_Update(&context, next, count);
      next += count;
      bytes -= count;
    }
  }
  void add(const MetalBuffer &buffer) {
    require(buffer && buffer.contents(), "hash input is not CPU-visible");
    add(buffer.contents(), buffer.sizeBytes());
  }
  std::string finish() {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream out;
    for (auto byte : digest)
      out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    return out.str();
  }
};
std::string inputHash(const FlashQSAState &state, const FlashQSAWorkspace &workspace,
                      const MetalBuffer &projection) {
  SHA256 hash;
  for (const auto &buffer : {state.keys, state.values, state.rawIndexKeys,
                            state.pooledKeys, state.indexPositions, workspace.queries,
                            workspace.indexQueries, workspace.blockScores,
                            workspace.selectedBlocks, projection})
    hash.add(buffer);
  return hash.finish();
}

std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> defaults,
                           uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return std::vector<uint32_t>(defaults);
  std::vector<uint32_t> result;
  std::stringstream stream(raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    require(!item.empty() && item.front() != '-', std::string("invalid ") + name);
    size_t used = 0;
    const auto value = std::stoul(item, &used);
    require(used == item.size() && value <= maximum, std::string("invalid ") + name);
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), std::string("empty ") + name);
  return result;
}
std::vector<std::string> patterns() {
  const char *raw = std::getenv("FLASH_QSA_MPP_PATTERNS");
  std::stringstream stream(raw ? raw : "latest,strided,cutoff_ties");
  std::vector<std::string> result;
  std::string item;
  while (std::getline(stream, item, ',')) {
    require(item == "earliest" || item == "latest" || item == "strided" ||
            item == "cutoff_ties", "unknown FLASH_QSA_MPP_PATTERNS selection");
    result.push_back(item);
  }
  require(!result.empty(), "empty FLASH_QSA_MPP_PATTERNS");
  return result;
}
uint32_t single(const char *name, uint32_t fallback, uint32_t maximum) {
  const auto values = list(name, {fallback}, maximum);
  require(values.size() == 1, std::string(name) + " requires one value");
  return values[0];
}

// The discrete selection is supplied, never recomputed by the candidate.
// cutoff_ties independently applies score-descending, higher-ID cutoff ties,
// then chronological order, as the qualified indexer requires.
std::vector<uint32_t> selected(uint32_t count, const std::string &pattern) {
  const uint32_t complete = count / 4;
  const uint32_t keep = std::min(complete, kFlashQSABlockBudget);
  std::vector<uint32_t> result(keep);
  if (complete <= kFlashQSABlockBudget || pattern == "earliest") {
    std::iota(result.begin(), result.end(), 0u);
  } else if (pattern == "latest") {
    std::iota(result.begin(), result.end(), complete - keep);
  } else if (pattern == "strided") {
    for (uint32_t i = 0; i < keep; ++i)
      result[i] = uint64_t(i) * (complete - 1) / (keep - 1);
  } else {
    std::vector<uint32_t> ranked(complete);
    std::iota(ranked.begin(), ranked.end(), 0u);
    const auto score = [](uint32_t block) { return uint32_t(randomWord(block + 0x7e51)) % 9; };
    std::sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) {
      return score(a) == score(b) ? a > b : score(a) > score(b);
    });
    std::copy_n(ranked.begin(), keep, result.begin());
    std::sort(result.begin(), result.end());
  }
  return result;
}
std::vector<uint32_t> tokens(uint32_t count, std::span<const uint32_t> blocks) {
  std::vector<uint32_t> result;
  for (uint32_t block : blocks)
    for (uint32_t lane = 0; lane < 4; ++lane) result.push_back(block * 4 + lane);
  for (uint32_t token = count / 4 * 4; token < count; ++token) result.push_back(token);
  require(result.size() == std::min(count / 4, 512u) * 4 + count % 4,
          "selected token extent is inconsistent");
  require(std::is_sorted(result.begin(), result.end()) &&
          std::adjacent_find(result.begin(), result.end()) == result.end(),
          "selected tokens are not unique and chronological");
  require(result.empty() || result.back() < count, "selected token violates causality");
  return result;
}

struct Guarded final {
  MetalBuffer base, view;
  uint64_t words;
  Guarded(MetalBackend &backend, uint64_t elements) : words(elements) {
    base = backend.allocateBuffer((words + 2 * kGuardWords) * 2, BufferStorage::Shared,
                                  "QSA MPP oracle guarded output");
    view = backend.view(base, kGuardWords * 2, words * 2);
    clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords, kSentinel);
  }
  void check(bool finite) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kSentinel && data[kGuardWords + words + i] == kSentinel,
              "QSA MPP output guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])), "output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};

struct Error final {
  uint64_t elements = 0, mismatches = 0, nonfinite = 0;
  double squareError = 0, squareReference = 0, squareActual = 0, dot = 0, maxAbs = 0;
  void add(uint16_t actual, uint16_t expected) {
    ++elements;
    mismatches += actual != expected;
    const double a = number(actual), b = number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; return; }
    const double error = a - b;
    maxAbs = std::max(maxAbs, std::abs(error));
    squareError += error * error;
    squareActual += a * a;
    squareReference += b * b;
    dot += a * b;
  }
  double relativeL2() const { return std::sqrt(squareError / std::max(squareReference, 1e-30)); }
  double cosine() const {
    return squareActual == 0 && squareReference == 0 ? 1 :
      dot / std::sqrt(std::max(squareActual * squareReference, 1e-30));
  }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"nonfinite\":" << nonfinite << ",\"relative_l2\":" << relativeL2()
        << ",\"cosine\":" << cosine() << ",\"max_abs\":" << maxAbs << '}';
  }
};
Error compare(std::span<const uint16_t> actual, std::span<const uint16_t> expected) {
  require(actual.size() == expected.size(), "output comparison extents differ");
  Error result;
  for (size_t i = 0; i < actual.size(); ++i) result.add(actual[i], expected[i]);
  return result;
}
void qualify(const Error &error, double tolerance, const char *name) {
  require(!error.nonfinite && error.relativeL2() <= tolerance &&
          error.cosine() >= .9998, std::string(name) + " exceeds numerical tolerance");
}

struct CPUReference final {
  std::vector<uint64_t> offsets;
  std::vector<uint16_t> f32Probabilities, bf16Probabilities;
};
CPUReference reference(const FlashQSAState &state, const FlashQSAWorkspace &workspace,
                        MetalBuffer projection, uint32_t begin, uint32_t rows) {
  const auto *queries = static_cast<const uint16_t *>(workspace.queries.contents());
  const auto *keys = static_cast<const uint16_t *>(state.keys.contents());
  const auto *values = static_cast<const uint16_t *>(state.values.contents());
  const auto *blocks = static_cast<const uint32_t *>(workspace.selectedBlocks.contents());
  const auto *gates = static_cast<const uint16_t *>(projection.contents());
  CPUReference result;
  const std::set<uint32_t> sampleRows{0, rows / 2, rows - 1};
  for (uint32_t row : sampleRows) {
    const uint32_t count = begin + row + 1;
    const auto picked = tokens(count, {blocks + uint64_t(row) * 512, std::min(count / 4, 512u)});
    for (uint32_t head : {0u, 11u, 23u}) {
      std::vector<float> probabilities(picked.size());
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t slot = 0; slot < picked.size(); ++slot) {
        float score = 0;
        for (uint32_t column = 0; column < 256; ++column) {
          const float q = number(queries[(uint64_t(row) * 24 + head) * 256 + column]);
          const float k = number(keys[(uint64_t(picked[slot]) * 2 + head / 12) * 256 + column]);
          score = std::fma(q, k, score);
        }
        probabilities[slot] = score * .0625f;
        maximum = std::max(maximum, probabilities[slot]);
      }
      float denominator = 0;
      for (float &probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (float &probability : probabilities) probability /= denominator;
      for (uint32_t column : {0u, 127u, 255u}) {
        float sumF32 = 0, sumBF16 = 0;
        for (size_t slot = 0; slot < picked.size(); ++slot) {
          const float v = number(values[(uint64_t(picked[slot]) * 2 + head / 12) * 256 + column]);
          sumF32 = std::fma(probabilities[slot], v, sumF32);
          sumBF16 = std::fma(number(bf16(probabilities[slot])), v, sumBF16);
        }
        const auto gate = gates[(uint64_t(row) * 24 + head) * 512 + 256 + column];
        result.offsets.push_back((uint64_t(row) * 24 + head) * 256 + column);
        result.f32Probabilities.push_back(gated(bf16(sumF32), gate));
        result.bf16Probabilities.push_back(gated(bf16(sumBF16), gate));
      }
    }
  }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const CPUReference &expected,
                     bool bfProbabilities) {
  Error result;
  const auto &values = bfProbabilities ? expected.bf16Probabilities : expected.f32Probabilities;
  for (size_t i = 0; i < expected.offsets.size(); ++i)
    result.add(actual[expected.offsets[i]], values[i]);
  return result;
}

double median(std::vector<double> values) {
  require(!values.empty(), "empty timing list");
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Times final {
  std::vector<double> gpu, wall;
  void add(CommandTiming timing) {
    require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 1e-9 && timing.gpuSeconds < 600 &&
            std::isfinite(timing.wallSeconds) && timing.wallSeconds > 1e-9 && timing.wallSeconds < 600, "invalid command timing");
    gpu.push_back(timing.gpuSeconds);
    wall.push_back(timing.wallSeconds);
  }
  void write(std::ostream &out) const {
    out << "{\"samples\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_wall_seconds\":" << median(wall) << '}';
  }
};

void seed(MetalBuffer buffer, uint64_t salt, float scale) {
  require(buffer.sizeBytes() % 2 == 0 && buffer.contents(), "invalid BF16 seed buffer");
  auto *data = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i) data[i] = bf16(seeded(i, salt, scale));
}
void prepare(FlashQSAState &state, FlashQSAWorkspace &workspace, MetalBuffer projection,
             uint32_t begin, uint32_t rows, const std::string &pattern) {
  seed(state.keys, 0x451be, 1.75f);
  seed(state.values, 0xabe18, .75f);
  seed(state.rawIndexKeys, 0x431af, .5f);
  seed(state.pooledKeys, 0x716ad, .25f);
  seed(workspace.queries, 0x44173, 1.75f);
  seed(workspace.indexQueries, 0x981bf, .5f);
  seed(projection, 0x91f73, 2.0f);
  auto *positions = static_cast<int64_t *>(state.indexPositions.contents());
  for (uint32_t token = 0; token < state.capacity; ++token) positions[token] = int64_t(token) * 3 + 7;
  auto *scores = static_cast<float *>(workspace.blockScores.contents());
  for (uint64_t i = 0; i < workspace.blockScores.sizeBytes() / 4; ++i)
    scores[i] = seeded(i, 0x981fa, 3.0f);
  auto *blocks = static_cast<uint32_t *>(workspace.selectedBlocks.contents());
  std::fill_n(blocks, workspace.selectedBlocks.sizeBytes() / 4, 0xdeadbeefu);
  for (uint32_t row = 0; row < rows; ++row) {
    const auto ids = selected(begin + row + 1, pattern);
    (void)tokens(begin + row + 1, ids);
    std::copy(ids.begin(), ids.end(), blocks + uint64_t(row) * 512);
  }
  // Cover BF16 sigmoid saturation, signs, signed zero, and the audited rare
  // precise-exp boundary in every row and both key-sharing query groups.
  const std::array<float, 9> gateValues{0.0f, -0.0f, -6.84375f, -80.0f, 80.0f,
                                      -2.0f, 2.0f, -.125f, .125f};
  auto *gates = static_cast<uint16_t *>(projection.contents());
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < 24; ++head)
      for (size_t i = 0; i < gateValues.size(); ++i)
        gates[(uint64_t(row) * 24 + head) * 512 + 256 + i] = bf16(gateValues[i]);
}

uint32_t validateHost(MetalBackend &backend, MetalBuffer projection,
                       const FlashQSAState &state, const FlashQSAWorkspace &workspace,
                       FlashQSAMPPWorkspace &mpp, MetalBuffer output,
                       MetalBuffer diagnostics, uint32_t begin, uint32_t rows) {
  uint32_t rejected = 0;
  auto reject = [&](MetalBuffer q, const FlashQSAState &s, const FlashQSAWorkspace &w,
                    FlashQSAMPPWorkspace &m, MetalBuffer y, MetalBuffer d,
                    uint32_t start, uint32_t count, uint32_t tile) {
    bool caught = false;
    try {
      CommandGraph graph;
      addQSAAttentionMPP(graph, q, s, w, m, y, d, start, count, tile);
    } catch (const std::invalid_argument &) { caught = true; }
    require(caught, "MPP accepted invalid host arguments");
    ++rejected;
  };
  reject(projection, state, workspace, mpp, output, diagnostics, begin, 0, 64);
  reject(projection, state, workspace, mpp, output, diagnostics, begin, 129, 64);
  reject(projection, state, workspace, mpp, output, diagnostics, state.capacity, 1, 64);
  reject(projection, state, workspace, mpp, output, diagnostics, begin, rows, 32);
  reject({}, state, workspace, mpp, output, diagnostics, begin, rows, 64);
  reject(projection, state, workspace, mpp, {}, diagnostics, begin, rows, 64);
  reject(projection, state, workspace, mpp, output, {}, begin, rows, 64);
  reject(projection, state, workspace, mpp, projection, diagnostics, begin, rows, 64);
  reject(projection, state, workspace, mpp, workspace.queries, diagnostics, begin, rows, 64);
  reject(projection, state, workspace, mpp, output,
         backend.view(output, 0, 4), begin, rows, 64);
  reject(projection, state, workspace, mpp, output,
         backend.view(projection, 2, 4), begin, rows, 64);
  auto shortWorkspace = workspace;
  shortWorkspace.queries = backend.view(workspace.queries, 0, 2);
  reject(projection, state, shortWorkspace, mpp, output, diagnostics, begin, rows, 64);
  auto wrongState = state;
  wrongState.capacity = state.capacity + 1;
  reject(projection, wrongState, workspace, mpp, output, diagnostics, begin, rows, 64);
  auto wrongMPP = mpp;
  wrongMPP.maximumRows = rows - 1;
  reject(projection, state, workspace, wrongMPP, output, diagnostics, begin, rows, 64);
  wrongMPP = mpp;
  wrongMPP.probabilities = mpp.scores;
  reject(projection, state, workspace, wrongMPP, output, diagnostics, begin, rows, 64);
  return rejected;
}

uint32_t validateOnlineHost(MetalBackend &backend, MetalBuffer projection,
                            const FlashQSAState &state,
                            const FlashQSAWorkspace &workspace,
                            const FlashQSAFastWorkspace &fast,
                            MetalBuffer output, MetalBuffer diagnostics,
                            uint32_t begin, uint32_t rows) {
  uint32_t rejected = 0;
  auto reject = [&](FlashQSAFastWorkspace scratch, MetalBuffer q,
                    MetalBuffer y, MetalBuffer d, uint32_t start,
                    uint32_t count, uint32_t partitions) {
    bool caught = false;
    try {
      CommandGraph graph;
      addQSAAttentionOnlineMPP(graph, q, state, workspace, scratch,
                               y, d, start, count, partitions);
    } catch (const std::invalid_argument &) { caught = true; }
    require(caught, "online MPP accepted invalid host arguments");
    ++rejected;
  };
  reject(fast, projection, output, diagnostics, begin, 0, 4);
  reject(fast, projection, output, diagnostics, begin, 129, 4);
  reject(fast, projection, output, diagnostics, state.capacity, 1, 4);
  reject(fast, projection, output, diagnostics, begin, rows, 3);
  reject(fast, projection, output, diagnostics, begin, rows, 64);
  reject(fast, {}, output, diagnostics, begin, rows, 4);
  reject(fast, projection, {}, diagnostics, begin, rows, 4);
  reject(fast, projection, output, {}, begin, rows, 4);
  reject(fast, projection, projection, diagnostics, begin, rows, 4);
  reject(fast, projection, output, backend.view(projection, 2, 4), begin, rows, 4);
  auto wrong = fast;
  wrong.maximumRows = rows - 1;
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  wrong = fast;
  wrong.maximumPartitions = 0;
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  wrong = fast;
  wrong.maximumPartitions = 33;
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  wrong = fast;
  wrong.maximumPartitions = 1;
  reject(wrong, projection, output, diagnostics, begin, rows, 2);
  wrong = fast;
  wrong.partitionStatistics = backend.view(fast.partitionStatistics, 0, 4);
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  wrong = fast;
  wrong.partitionValues = backend.view(fast.partitionValues, 0, 4);
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  wrong = fast;
  wrong.partitionStatistics = fast.partitionValues;
  reject(wrong, projection, output, diagnostics, begin, rows, 4);
  return rejected;
}

void checkPartitionPadding(const FlashQSAFastWorkspace &scratch,
                           uint32_t rows, uint32_t partitions) {
  for (uint32_t row = 0; row < scratch.maximumRows; ++row)
    for (uint32_t head = 0; head < 24; ++head)
      for (uint32_t partition = 0; partition < scratch.maximumPartitions; ++partition) {
        if (row < rows && partition < partitions) continue;
        const uint64_t index = (uint64_t(row) * 24 + head) *
                               scratch.maximumPartitions + partition;
        for (const auto &[buffer, width] : {
               std::pair{scratch.partitionStatistics, 4u},
               std::pair{scratch.partitionValues, 512u}}) {
          const auto *words = static_cast<const uint16_t *>(buffer.contents());
          for (uint32_t word = 0; word < width; ++word)
            require(words[index * width + word] == kSentinel,
                    "online MPP wrote an inactive partition");
        }
      }
}

void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t word = 0; word < 65536; ++word) {
    const auto original = uint16_t(word);
    if (std::isfinite(number(original))) {
      require(bf16(number(original)) == original, "BF16 finite round-trip failed");
      ++checks;
    }
  }
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82, "BF16 ties failed");
  checks += 2;
  for (uint32_t count : {1u, 2u, 3u, 4u, 127u, 2048u, 2049u, 2050u, 2051u, 8193u})
    for (const std::string &pattern : {"earliest", "latest", "strided", "cutoff_ties"}) {
      const auto ids = selected(count, pattern);
      const auto visible = tokens(count, ids);
      require(!visible.empty(), "self-test lost all visible tokens");
      ++checks;
      if (count % 4) require(visible.back() == count - 1, "causal tail not included");
    }
  const auto ties = selected(8192, "cutoff_ties");
  std::set<uint32_t> kept(ties.begin(), ties.end());
  const auto score = [](uint32_t block) { return uint32_t(randomWord(block + 0x7e51)) % 9; };
  for (uint32_t block = 0; block < 2048; ++block)
    if (!kept.contains(block))
      for (uint32_t picked : ties)
        require(score(picked) > score(block) ||
                (score(picked) == score(block) && picked > block), "highest-ID cutoff tie failed");
  ++checks;
  require(number(gated(bf16(1), bf16(0))) == .5f &&
          number(gated(bf16(1), bf16(-100))) == 0 &&
          number(gated(bf16(1), bf16(100))) == 1, "BF16 gate staging failed");
  checks += 3;
  SHA256 hash;
  hash.add("abc", 3);
  require(hash.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA256 self-test failed");
  ++checks;
  Error equal;
  equal.add(bf16(1), bf16(1));
  require(equal.relativeL2() == 0 && equal.cosine() == 1, "comparison self-test failed");
  ++checks;
  for (uint32_t rows : {1u, 8u, 128u})
    for (uint32_t partitions : {1u, 2u, 4u, 8u, 16u, 32u}) {
      require(qsaOnlineMPPWorkspacePlannedBytes(rows, partitions) ==
                uint64_t(rows) * 24 * partitions * 258 * 4,
              "online MPP planned scratch bytes differ");
      ++checks;
    }
  for (const auto &[rows, partitions] : {std::pair{0u, 4u}, std::pair{129u, 4u},
                                        std::pair{1u, 0u}, std::pair{1u, 3u},
                                        std::pair{1u, 33u}}) {
    bool rejected = false;
    try { (void)qsaOnlineMPPWorkspacePlannedBytes(rows, partitions); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "online MPP accepted invalid planned geometry");
    ++checks;
  }
  for (uint32_t rows = 1; rows <= 128; ++rows)
    for (uint32_t begin : {0u, 128u, 511u, 512u, 1023u, 1024u, 4096u}) {
      const uint32_t expected = rows >= 32 ? (begin == 0 ? (rows >= 64 ? 1 : 0) : 4) :
        rows <= 4 && begin >= 1024 ? 32 :
        rows > 4 && rows <= 12 && begin >= 512 ? 16 :
        rows > 12 && rows <= 16 && begin >= 512 ? 8 : 0;
      require(qsaOnlineMPPRoutePartitions(begin, rows) == expected,
              "online MPP context-aware route policy differs");
      ++checks;
    }
  require(qsaOnlineMPPRoutePartitions(4096, 0) == 0 &&
          qsaOnlineMPPRoutePartitions(4096, 129) == 0,
          "online MPP route policy accepted invalid rows");
  checks += 2;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}

void writeReport(const std::filesystem::path &path, const std::string &report) {
  std::ofstream output(path);
  require(bool(output), "cannot open QSA MPP report");
  output << report << '\n';
  output.close();
  require(bool(output), "cannot write QSA MPP report");
}
} // namespace


namespace {
// First build the complete qualified graph so argument/padding validation is
// identical to production. Only the temporal attention geometry is changed.
void addTemporalCandidate(CommandGraph &graph, MetalBuffer projection,
    const FlashQSAState &state, const FlashQSAWorkspace &workspace,
    FlashQSAFastWorkspace &fast, MetalBuffer output, MetalBuffer diagnostics,
    uint32_t begin, uint32_t rows, uint32_t partitions, uint32_t tile,
    bool queryReuse=false, bool valuesN128=false, bool registerPV=false,
    bool registerPVStrict=false, bool relaxedPV=false, bool sg8=false) {
  require(flash_qsa_row_tiles_geometry(begin, rows, partitions),
          "candidate only accepts production dense temporal geometry");
  require(tile == 16 || tile == 32 || tile == 48 || tile == 64, "invalid temporal candidate tile");
  require(uint32_t(queryReuse)+uint32_t(valuesN128)+uint32_t(registerPV)+uint32_t(relaxedPV)+uint32_t(sg8)<=1 &&
          (!(queryReuse || valuesN128 || registerPV || relaxedPV || sg8) || tile==32),
          "Exact staging variants require M32 and cannot be combined");
  CommandGraph qualified;
  addQSAAttentionOnlineMPP(qualified, projection, state, workspace, fast,
                          output, diagnostics, begin, rows, partitions);
  require(qualified.dispatches().size() == 2, "qualified attention graph changed");
  for (size_t index = 0; index < qualified.dispatches().size(); ++index) {
    const auto &dispatch = qualified.dispatches()[index];
    require(dispatch.bytes.size() == 1 &&
            dispatch.bytes[0].sizeBytes == sizeof(FlashQSAFastParams),
            "qualified attention ABI changed");
    FlashQSAFastParams params{};
    std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
    std::vector<MetalBuffer> buffers;
    for (const auto &binding : dispatch.buffers) {
      require(binding.index == buffers.size(), "qualified binding order changed");
      buffers.push_back(binding.buffer);
    }
    if (!index)
      graph.add(queryReuse ? "flash_qsa_mpp_prefill_query_reuse_m32" :
                  valuesN128 ? "flash_qsa_mpp_prefill_values_n128_m32" :
                  registerPV ? (registerPVStrict ? "flash_qsa_mpp_prefill_register_pv_strict_m32" :
                    "flash_qsa_mpp_prefill_register_pv_m32") :
                  relaxedPV ? "flash_qsa_mpp_prefill_relaxed_pv_m32" :
                  sg8 ? "prefill4k_qsa_sg8_rows_m32" :
                  "flash_qsa_mpp_prefill_rows_m" + std::to_string(tile),
                std::move(buffers), params,
                {(uint64_t(rows) * 12 + tile - 1) / tile, 2, partitions}, {sg8 ? 256u : 128u, 1, 1});
    else
      graph.add(dispatch.pipelineName, std::move(buffers), params,
                dispatch.threadgroups, dispatch.threadsPerThreadgroup);
  }
}
}

int main(int argc, char **argv) {
  @autoreleasepool {
    std::vector<std::string> records;
    std::string failureMetrics;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") {
        cpuSelfTest();
        return 0;
      }
      if (argc != 3)
        throw std::invalid_argument("usage: flash-qsa-mpp-oracle METALLIB REPORT_JSON | --cpu-self-test");
      const auto rowCases = list("FLASH_QSA_MPP_ROWS", {32, 128}, 128);
      const auto begins = list("FLASH_QSA_MPP_BEGIN", {512, 1024, 1920}, 262016);
      const auto tiles = list("PREFILL4K_ATTENTION_TILES", {16, 32, 48, 64}, 64);
      const auto probabilityModes = list("FLASH_QSA_MPP_F32_PROBABILITIES", {0, 1}, 1);
      const auto onlinePartitions = list("FLASH_QSA_MPP_ONLINE_PARTITIONS", {4}, 32);
      const bool enableMaterialized = false;
      const bool enableOnline = single("FLASH_QSA_MPP_ONLINE", 1, 1) != 0;
      const bool queryReuse=single("PREFILL4K_ATTENTION_QUERY_REUSE",0,1)!=0;
      const bool valuesN128=single("PREFILL4K_ATTENTION_VALUES_N128",0,1)!=0;
      const bool registerPVStrict=single("PREFILL4K_ATTENTION_REGISTER_PV_STRICT",0,1)!=0;
      const bool registerPV=single("PREFILL4K_ATTENTION_REGISTER_PV",0,1)!=0 || registerPVStrict;
      const bool relaxedPV=single("PREFILL4K_ATTENTION_RELAXED_PV",0,1)!=0;
      const bool sg8=single("PREFILL4K_ATTENTION_SG8",0,1)!=0;
      const auto selections = patterns();
      const uint32_t repeats = single("FLASH_QSA_MPP_REPEATS", 5, 100);
      const uint32_t warmup = single("FLASH_QSA_MPP_WARMUP", 2, 100);
      const double tolerance = .01;
      require(repeats > 0 && warmup > 0, "timing requires positive repeats and warmup");
      for (uint32_t rows : rowCases) require(rows > 0, "rows must be positive");
      for (uint32_t tile : tiles) require(tile == 16 || tile == 32 || tile == 48 || tile == 64, "temporal tile must be16/32/48/64");
      for (uint32_t parts : onlinePartitions)
        require(parts == 1 || parts == 2 || parts == 4 || parts == 8 ||
                parts == 16 || parts == 32,
                "online partitions must be 1,2,4,8,16,32");
      struct Variant final {
        uint32_t tile, partitions;
        bool f32Probabilities, online;
      };
      std::vector<Variant> candidates;
      if (enableMaterialized)
        for (uint32_t tile : tiles)
          for (uint32_t probabilityMode : probabilityModes)
            candidates.push_back({tile, 0, probabilityMode != 0, false});
      if (enableOnline)
        for (uint32_t parts : onlinePartitions)
          for (uint32_t tile : tiles) candidates.push_back({tile, parts, true, true});
      require(!candidates.empty(), "all MPP candidates are disabled");
      const std::string filter = std::getenv("FLASH_QSA_MPP_FILTER") ?
        std::getenv("FLASH_QSA_MPP_FILTER") : "";
      MetalBackend backend(argv[1]);
      uint64_t variants = 0;
      for (uint32_t rows : rowCases) {
        for (uint32_t begin : begins) {
          const bool sparse = (begin + rows) / 4 > 512;
          for (size_t p = 0; p < selections.size(); ++p) {
            if (!sparse && p > 0) continue;
            const std::string pattern = sparse ? selections[p] : "dense";
            const std::string name = "rows" + std::to_string(rows) + "-begin" +
              std::to_string(begin) + "-" + pattern;
            if (!filter.empty() && name.find(filter) == std::string::npos) continue;
            const uint32_t capacity = begin + rows;
            auto state = allocateQSAState(backend, capacity);
            auto workspace = allocateQSAWorkspace(backend, rows, capacity);
            const uint32_t maximumPartitions = std::max(4u,
              *std::max_element(onlinePartitions.begin(), onlinePartitions.end()));
            auto fast = allocateQSAOnlineMPPWorkspace(backend, rows, maximumPartitions);
            Guarded scratchStatistics(backend, fast.partitionStatistics.sizeBytes() / 2);
            Guarded scratchValues(backend, fast.partitionValues.sizeBytes() / 2);
            fast.partitionStatistics = scratchStatistics.view;
            fast.partitionValues = scratchValues.view;
            auto mpp = allocateQSAMPPWorkspace(backend, rows);
            auto projection = backend.allocateBuffer(uint64_t(rows) * 12288 * 2,
              BufferStorage::Shared, "QSA MPP oracle query/gate projection");
            auto diagnostics = backend.allocateBuffer(4, BufferStorage::Shared, "QSA MPP diagnostics");
            auto *status = static_cast<uint32_t *>(diagnostics.contents());
            prepare(state, workspace, projection, begin, rows, selections[p]);
            const std::string beforeHash = inputHash(state, workspace, projection);
            const auto cpu = reference(state, workspace, projection, begin, rows);
            Guarded f32Output(backend, uint64_t(rows) * kOutputWidth);
            Guarded bfOutput(backend, uint64_t(rows) * kOutputWidth);
            Guarded output(backend, uint64_t(rows) * kOutputWidth);
            const uint32_t hostRejections = validateHost(backend, projection, state,
              workspace, mpp, output.view, diagnostics, begin, rows);
            const uint32_t onlineHostRejections = validateOnlineHost(backend, projection,
              state, workspace, fast, output.view, diagnostics, begin, rows);
            CommandGraph f32Graph, bfGraph;
            addTemporalCandidate(f32Graph, projection, state, workspace, fast,
              f32Output.view, diagnostics, begin, rows, 4, 32);
            addQSAAttentionFast(bfGraph, projection, state, workspace, fast,
              bfOutput.view, diagnostics, begin, rows,
              FlashQSAFastMode::CanonicalBF16Probabilities, 4);
            *status = kSticky;
            (void)backend.submitCommand(f32Graph.dispatches());
            const auto bytesSnapshot=[](const MetalBuffer &buffer) {
              const auto *first=static_cast<const uint8_t *>(buffer.contents());
              return std::vector<uint8_t>(first,first+buffer.sizeBytes());
            };
            const auto expectedStatistics=bytesSnapshot(fast.partitionStatistics);
            const auto expectedValues=bytesSnapshot(fast.partitionValues);
            (void)backend.submitCommand(bfGraph.dispatches());
            f32Output.check(true);
            bfOutput.check(true);
            require(*status == kSticky, "control changed sticky diagnostics");
            const auto controlCPU = compareSample(f32Output.values(), cpu, false);
            const auto canonicalCPU = compareSample(bfOutput.values(), cpu, true);
            qualify(controlCPU, tolerance, "F32 control versus independent CPU");
            qualify(canonicalCPU, tolerance, "BF16 control versus independent CPU");
            std::ostringstream record;
            record << std::setprecision(12) << "{\"name\":" << splash::json::quote(name)
              << ",\"rows\":" << rows << ",\"begin\":" << begin << ",\"capacity\":" << capacity
              << ",\"selection\":" << splash::json::quote(pattern)
              << ",\"first_visible_tokens\":" << std::min((begin + 1) / 4, 512u) * 4 + (begin + 1) % 4
              << ",\"last_visible_tokens\":" << std::min(capacity / 4, 512u) * 4 + capacity % 4
              << ",\"input_sha256\":" << splash::json::quote(beforeHash)
              << ",\"host_rejections\":" << hostRejections
              << ",\"online_host_rejections\":" << onlineHostRejections
              << ",\"f32_control_vs_cpu\":";
            controlCPU.write(record);
            record << ",\"canonical_bf16_vs_cpu\":";
            canonicalCPU.write(record);
            record << ",\"variants\":[";
            bool first = true;
            for (const auto &variant : candidates) {
              const uint32_t tile = variant.tile;
              const bool f32Probabilities = variant.f32Probabilities;
              output.clear();
              CommandGraph candidate;
              if (variant.online)
                addTemporalCandidate(candidate, projection, state, workspace, fast,
                  output.view, diagnostics, begin, rows, variant.partitions, tile, queryReuse, valuesN128, registerPV, registerPVStrict, relaxedPV, sg8);
              else
                addQSAAttentionMPP(candidate, projection, state, workspace, mpp,
                  output.view, diagnostics, begin, rows, tile, f32Probabilities);
              *status = kSticky;
              scratchStatistics.clear();
              scratchValues.clear();
              (void)backend.submitCommand(candidate.dispatches());
              scratchStatistics.check(false);
              scratchValues.check(false);
              checkPartitionPadding(fast, rows, variant.online ? variant.partitions : 0);
              const bool statisticsExact=bytesSnapshot(fast.partitionStatistics)==expectedStatistics;
              const bool numeratorsExact=bytesSnapshot(fast.partitionValues)==expectedValues;
              if (registerPV || relaxedPV) require(statisticsExact,"Register PV changed authoritative F32 score/softmax statistics");
              if (queryReuse || valuesN128) {
                require(statisticsExact,
                        "Exact staging changed F32 partition statistics bytes");
                require(numeratorsExact,
                        "Exact staging changed F32 partition values bytes");
              }
              for (uint32_t iteration = 0; iteration < warmup; ++iteration) {
                (void)backend.submitCommand(f32Graph.dispatches());
                (void)backend.submitCommand(candidate.dispatches());
                (void)backend.submitCommand(bfGraph.dispatches());
              }
              output.check(true);
              require(*status == kSticky, "candidate changed sticky diagnostics");
              const std::vector<uint16_t> firstOutput(output.values().begin(), output.values().end());
              Times candidateF32Times, f32Times, candidateBFTimes, bfTimes;
              for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
                if (iteration % 2 == 0) {
                  f32Times.add(backend.submitCommand(f32Graph.dispatches()));
                  candidateF32Times.add(backend.submitCommand(candidate.dispatches()));
                  candidateBFTimes.add(backend.submitCommand(candidate.dispatches()));
                  bfTimes.add(backend.submitCommand(bfGraph.dispatches()));
                } else {
                  candidateF32Times.add(backend.submitCommand(candidate.dispatches()));
                  f32Times.add(backend.submitCommand(f32Graph.dispatches()));
                  bfTimes.add(backend.submitCommand(bfGraph.dispatches()));
                  candidateBFTimes.add(backend.submitCommand(candidate.dispatches()));
                }
              }
              output.check(true);
              f32Output.check(true);
              bfOutput.check(true);
              require(*status == kSticky, "repeated timing diagnostics changed");
              require(compare(output.values(), firstOutput).mismatches == 0,
                      "MPP repeated output is not deterministic");
              const auto vsF32 = compare(output.values(), f32Output.values());
              const auto vsBF = compare(output.values(), bfOutput.values());
              const auto vsCPU = compareSample(output.values(), cpu, !f32Probabilities);
              std::ostringstream diagnostic;
              diagnostic << std::setprecision(12) << "{\"name\":" << splash::json::quote(name)
                  << ",\"tile\":" << tile << ",\"register_pv\":" << (registerPV?"true":"false")
                  << ",\"register_pv_strict\":" << (registerPVStrict?"true":"false")
                  << ",\"f32_statistics_exact\":" << (statisticsExact?"true":"false")
                  << ",\"f32_numerators_exact\":" << (numeratorsExact?"true":"false")
                  << ",\"candidate_timing\":";candidateF32Times.write(diagnostic);
              diagnostic << ",\"control_timing\":";f32Times.write(diagnostic);
              diagnostic << ",\"vs_partitioned_f32\":";vsF32.write(diagnostic);
              diagnostic << ",\"vs_canonical_bf16\":";vsBF.write(diagnostic);
              diagnostic << ",\"vs_cpu\":";vsCPU.write(diagnostic);diagnostic << '}';
              failureMetrics=diagnostic.str();
              qualify(vsF32, tolerance, "MPP versus partitioned F32");
              if (queryReuse || valuesN128) require(vsF32.mismatches==0,
                  "Exact staging must preserve every BF16 output byte");
              qualify(vsBF, tolerance, "MPP versus canonical BF16");
              qualify(vsCPU, tolerance, "MPP versus independent BF16-probability CPU");
              require(inputHash(state, workspace, projection) == beforeHash,
                      "attention modified prepared queries, selection or state inputs");
              if (!first) record << ',';
              first = false;
              record << "{\"score_tile\":" << tile
                << ",\"route\":" << splash::json::quote(variant.online ? "online_mpp" : "materialized_mpp")
                << ",\"partitions\":" << variant.partitions
                << ",\"query_reuse\":" << (queryReuse ? "true" : "false")
                << ",\"sg8\":" << (sg8 ? "true" : "false")
                << ",\"register_pv\":" << (registerPV ? "true" : "false")
                << ",\"register_pv_strict\":" << (registerPVStrict ? "true" : "false")
                << ",\"relaxed_precision_pv\":" << ((relaxedPV || (registerPV && !registerPVStrict)) ? "true" : "false")
                << ",\"f32_partition_statistics_exact\":" << (statisticsExact ? "true" : "false")
                << ",\"f32_partition_numerators_exact\":" << (numeratorsExact ? "true" : "false")
                << ",\"values_n128\":" << (valuesN128 ? "true" : "false")
                << ",\"partition_intermediate_bytes_exact\":" << ((statisticsExact && numeratorsExact) ? "true" : "false")
                << ",\"probabilities\":" << splash::json::quote(f32Probabilities ? "F32" : "BF16")
                << ",\"candidate_dispatches\":" << candidate.dispatches().size()
                << ",\"f32_control_dispatches\":" << f32Graph.dispatches().size()
                << ",\"canonical_bf16_dispatches\":" << bfGraph.dispatches().size()
                << ",\"paired_with_f32_candidate_timing\":";
              candidateF32Times.write(record);
              record << ",\"paired_f32_control_timing\":";
              f32Times.write(record);
              record << ",\"speedup_vs_f32_gpu\":" << median(f32Times.gpu) / median(candidateF32Times.gpu)
                << ",\"paired_with_bf16_candidate_timing\":";
              candidateBFTimes.write(record);
              record << ",\"paired_canonical_bf16_timing\":";
              bfTimes.write(record);
              record << ",\"speedup_vs_canonical_bf16_gpu\":" << median(bfTimes.gpu) / median(candidateBFTimes.gpu)
                << ",\"vs_partitioned_f32\":";
              vsF32.write(record);
              record << ",\"vs_canonical_bf16\":";
              vsBF.write(record);
              record << ",\"vs_matching_probability_cpu_sample\":";
              vsCPU.write(record);
              record << ",\"state_inputs_unchanged\":true,\"output_guards_pass\":true,"
                << "\"partition_scratch_guards_pass\":true,\"inactive_partition_padding_pass\":true} ";
              ++variants;
            }
            record << "]}";
            records.push_back(record.str());
            std::cerr << "QSA MPP " << name << " qualified variants=" << variants << '\n';
          }
        }
      }
      require(variants > 0, "filter selected no QSA MPP variants");
      std::ostringstream report;
      report << std::setprecision(12) << "{\"pass\":true,\"operator\":\"Prefill4k temporal attention M16/M32/M48/M64\","
        << "\"numerical_semantics\":\"BF16 QK operands; F32 MPP scores/global softmax; BF16 or F32 probabilities; mixed P/BF16 V operands/F32 MPP; BF16 attention/staged gate\","
        << "\"f32_control\":\"Production temporal M32/partitions4\","
        << "\"canonical_control\":\"FlashQSAFast CanonicalBF16Probabilities\","
        << "\"cpu_reference\":\"independent serial F32 FMA dot/global softmax; optional BF16 probability cast; serial F32 FMA value dot; BF16 attention/staged gate; sampled rows/heads/columns\","
        << "\"relative_l2_tolerance\":" << tolerance << ",\"warmup_commands_per_route\":" << warmup
        << ",\"paired_order\":\"AB/BA; separate candidate measurements paired with each control\","
        << "\"shader_validation_environment\":" << splash::json::quote(std::getenv("MTL_SHADER_VALIDATION") ?
          std::getenv("MTL_SHADER_VALIDATION") : "unset")
        << ",\"variants\":" << variants << ",\"cases\":[";
      for (size_t i = 0; i < records.size(); ++i) {
        if (i) report << ',';
        report << records[i];
      }
      report << "]}";
      writeReport(argv[2], report.str());
      return 0;
    } catch (const std::exception &error) {
      if (argc == 3) {
        try {
          std::ostringstream report;
          report << "{\"pass\":false,\"error\":" << splash::json::quote(error.what())
              << ",\"failed_variant\":" << (failureMetrics.empty()?"null":failureMetrics) << ",\"completed_cases\":[";
          for (size_t i = 0; i < records.size(); ++i) {
            if (i) report << ',';
            report << records[i];
          }
          report << "]}";
          writeReport(argv[2], report.str());
        } catch (...) {}
      }
      std::cerr << "QSA MPP oracle failure: " << error.what() << '\n';
      return 1;
    }
  }
}
