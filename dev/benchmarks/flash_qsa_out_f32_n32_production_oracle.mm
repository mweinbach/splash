// Root-only GPU oracle. Compilation and --cpu-self-test create no Metal device.
// Cached coefficients must remain bit-exact F32, independent of output reduction.
#include "flash/FlashFloatDenseCache.hpp"
#include "FlashQSAOutF32N32ProductionHostProvenance.hpp"
#include "engine/Json.hpp"
#include "FlashFloatBoundaryAudit.hpp"

#import <Foundation/Foundation.h>

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
#include <optional>
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
using splash::flash::benchmark::bf16;
using splash::flash::benchmark::bf16Double;
using splash::flash::benchmark::bf16ULP;
using splash::flash::benchmark::CellRelation;
using splash::flash::benchmark::cellRelation;
using splash::flash::benchmark::f32DotBound;
using splash::flash::benchmark::gamma;
using splash::flash::benchmark::number;
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint64_t kGuardWords = 32;
constexpr double kRelativeL2Limit = 1e-4;
constexpr std::array<uint32_t, 8> kTileRows{8, 8, 16, 16, 8, 8, 16, 8};
constexpr std::array<uint32_t, 8> kTileOutputs{64, 128, 64, 128, 32, 32, 64, 64};
constexpr std::array<const char *, 8> kTileNames{"m8n64s4", "m8n128s4", "m16n64s4", "m16n128s4", "m8n32s2", "m8n32s4", "m16n64s8", "m8n64s2"};
constexpr const char *kDefaultPrefix = "language_model.lm_head";

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
std::string digestHex(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint8_t byte : digest) out << std::setw(2) << uint32_t(byte);
  return out.str();
}
bool auditSwitch(const char *name) {
  const char *value = std::getenv(name);
  require(!value || std::string(value) == "0" || std::string(value) == "1",
          std::string(name) + " must be absent, 0 or 1");
  return value && std::string(value) == "1";
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
// Independent little-endian byte unpacking, including cross-byte Q5/Q6.
uint32_t code(const std::byte *row, uint32_t bits, uint32_t k) {
  const uint64_t bit = uint64_t(k) * bits;
  const uint32_t shift = uint32_t(bit % 8);
  uint32_t word = std::to_integer<uint8_t>(row[bit / 8]);
  if (shift + bits > 8)
    word |= uint32_t(std::to_integer<uint8_t>(row[bit / 8 + 1])) << 8;
  return (word >> shift) & ((1u << bits) - 1);
}
float coefficient(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *packed = static_cast<const std::byte *>(p.weights->buffer.contents()) +
                       uint64_t(n) * p.weightRowStrideBytes;
  const auto *scales = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.scales->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  const auto *biases = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.biases->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  // This oracle is compiled with -ffp-contract=off. Integer * BF16 is exact
  // in F32 for supported code widths; the addition then rounds once to F32.
  const float product = float(code(packed, p.bits, k)) * number(scales[k / p.groupSize]);
  return product + number(biases[k / p.groupSize]);
}
std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> fallback,
                          uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return std::vector<uint32_t>(fallback);
  std::stringstream stream(raw);
  std::vector<uint32_t> result;
  std::string item;
  while (std::getline(stream, item, ',')) {
    require(!item.empty() && item.front() != '-', std::string("invalid ") + name);
    size_t used = 0;
    const unsigned long value = std::stoul(item, &used);
    require(used == item.size() && value <= maximum, std::string("invalid ") + name);
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), std::string("empty ") + name);
  return result;
}
std::vector<std::string> parsePrefixes(const std::string &raw) {
  require(!raw.empty() && raw.front() != ',' && raw.back() != ',',
          "FLASH_FLOAT_CACHE_PREFIXES must contain nonempty comma-separated names");
  std::vector<std::string> result;
  std::stringstream stream(raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    require(!item.empty(), "empty FLASH_FLOAT_CACHE_PREFIXES entry");result.push_back(item);
  }
  return result;
}
std::vector<std::string> prefixes(const FlashWeights &weights) {
  std::vector<std::string> result;
  const char *filter = std::getenv("FLASH_FLOAT_CACHE_PREFIX");
  if (const char *raw = std::getenv("FLASH_FLOAT_CACHE_PREFIXES")) result = parsePrefixes(raw);
  else if (filter) result = FlashFloatDenseCache::defaultPrefixes(weights, true);
  else result.emplace_back(kDefaultPrefix); // preserve the original bounded head-only CLI
  if (filter) {
    require(*filter != '\0', "FLASH_FLOAT_CACHE_PREFIX must be a nonempty substring");
    std::erase_if(result, [&](const std::string &prefix) {
      return prefix.find(filter) == std::string::npos;
    });
  }
  require(!result.empty(), "float-cache prefix filter selected no eligible matrices");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());return result;
}

struct Error final {
  uint64_t elements = 0, mismatch = 0, nonfinite = 0;
  uint32_t maximumULP = 0;
  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;
  void add(uint16_t actual, uint16_t expected) {
    ++elements;mismatch += actual != expected;
    const float a = number(actual), b = number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite;return; }
    const double delta = double(a) - b;
    maximumAbsolute = std::max(maximumAbsolute, std::abs(delta));
    squaredError += delta * delta;squaredReference += double(b) * b;
    maximumULP = std::max(maximumULP, bf16ULP(actual, expected));
  }
  double relativeL2() const {
    return std::sqrt(squaredError / std::max(1e-30, squaredReference));
  }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatch
        << ",\"nonfinite\":" << nonfinite << ",\"max_bf16_ulp\":" << maximumULP
        << ",\"max_abs\":" << maximumAbsolute << ",\"relative_l2\":" << relativeL2() << '}';
  }
};
Error compare(std::span<const uint16_t> actual, std::span<const uint16_t> expected) {
  require(actual.size() == expected.size(), "comparison extents differ");
  Error result;
  for (size_t i = 0; i < actual.size(); ++i) result.add(actual[i], expected[i]);return result;
}

struct Guarded final {
  MetalBuffer base, view;
  uint64_t words;
  Guarded(MetalBackend &backend, uint64_t elements) : words(elements) {
    base = backend.allocateBuffer((words + 2 * kGuardWords) * 2, BufferStorage::Shared,
                                  "float-cache oracle guarded BF16 buffer");
    view = backend.view(base, kGuardWords * 2, words * 2);clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords, kSentinel);
  }
  void load(std::span<const uint16_t> values) {
    require(values.size() == words, "guarded input extent differs");
    std::memcpy(view.contents(), values.data(), values.size_bytes());
  }
  void check(bool finite = true) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kSentinel && data[kGuardWords + words + i] == kSentinel,
              "float-cache oracle guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])), "output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};

struct Coefficients final {
  uint64_t samples = 0, negativeScales = 0, notBF16Representable = 0;
  void write(std::ostream &out) const {
    out << "{\"samples\":" << samples << ",\"f32_bit_mismatches\":0,"
           "\"sampled_negative_scales\":" << negativeScales
        << ",\"not_bf16_representable\":" << notBF16Representable << '}';
  }
};
Coefficients coefficientSamples(const FlashAffineProjection &p, const FlashTensor &cached) {
  require(p.experts == 1 && cached.dtype == FlashDType::F32 && cached.shape ==
      std::vector<uint64_t>{p.outputSize, p.inputSize} && cached.buffer.contents() &&
      cached.logicalBytes == uint64_t(p.outputSize) * p.inputSize * sizeof(float) &&
      cached.buffer.sizeBytes() >= cached.logicalBytes,
      "cached coefficient dimensions, dtype, extent or visibility invalid");
  const auto *values = static_cast<const float *>(cached.buffer.contents());
  const uint64_t total = uint64_t(p.outputSize) * p.inputSize;
  Coefficients result;
  result.samples = std::min<uint64_t>(total, 16384);
  for (uint64_t i = 0; i < result.samples; ++i) {
    const uint64_t index = i < 2 ? (i ? total - 1 : 0) : randomWord(i + 0xf32c) % total;
    const uint32_t n = uint32_t(index / p.inputSize), k = uint32_t(index % p.inputSize);
    const float expected = coefficient(p, n, k);
    require(std::isfinite(expected) && std::isfinite(values[index]), "nonfinite cached coefficient");
    require(std::bit_cast<uint32_t>(values[index]) == std::bit_cast<uint32_t>(expected),
            "cached F32 coefficient differs from source at n=" + std::to_string(n) +
            " k=" + std::to_string(k));
    result.notBF16Representable += expected != number(bf16(expected));
    const auto *scales = reinterpret_cast<const uint16_t *>(
        static_cast<const std::byte *>(p.scales->buffer.contents()) +
        uint64_t(n) * p.parameterRowStrideBytes);
    result.negativeScales += number(scales[k / p.groupSize]) < 0;
  }
  // Some source matrices may naturally be BF16-representable. Exact original
  // bits remain the contract; independent cancellation traps prove F32 usage.
  return result;
}

struct References final {
  std::vector<uint32_t> columns;
  std::vector<uint16_t> serial, simd32, roundedSerial;
};
float pairwiseSum32(std::array<float, 32> values) {
  // A deterministic independent reduction tree. Its SIMD-lane layout matches
  // the raw control; Metal may choose a different shuffle reduction ordering.
  for (uint32_t stride = 16; stride; stride >>= 1)
    for (uint32_t lane = 0; lane < stride; ++lane) values[lane] += values[lane + stride];
  return values[0];
}
References reference(const FlashAffineProjection &p, std::span<const uint16_t> input,
                     uint32_t rows, uint32_t count) {
  References result;
  count = std::min(count, p.outputSize);
  for (uint32_t c = 0; c < count; ++c)
    result.columns.push_back(count == 1 ? 0 : uint64_t(c) * (p.outputSize - 1) / (count - 1));
  result.serial.resize(uint64_t(rows) * count);
  result.simd32.resize(result.serial.size());result.roundedSerial.resize(result.serial.size());
  std::vector<float> weights(p.inputSize);
  for (uint32_t c = 0; c < count; ++c) {
    for (uint32_t k = 0; k < p.inputSize; ++k)
      weights[k] = coefficient(p, result.columns[c], k);
    for (uint32_t row = 0; row < rows; ++row) {
      float serial = 0, rounded = 0;
      std::array<float, 32> lanes{};
      for (uint32_t k = 0; k < p.inputSize; ++k) {
        const float x = number(input[uint64_t(row) * p.inputSize + k]);
        serial = std::fma(x, weights[k], serial);
        rounded = std::fma(x, number(bf16(weights[k])), rounded);
        lanes[k % 32] = std::fma(x, weights[k], lanes[k % 32]);
      }
      const uint64_t index = uint64_t(row) * count + c;
      result.serial[index] = bf16(serial);result.simd32[index] = bf16(pairwiseSum32(lanes));
      result.roundedSerial[index] = bf16(rounded);
    }
  }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const References &ref,
                    std::span<const uint16_t> expected, uint32_t n, uint32_t rows) {
  require(expected.size() == uint64_t(rows) * ref.columns.size(), "sample reference extent differs");
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (size_t c = 0; c < ref.columns.size(); ++c)
      result.add(actual[uint64_t(row) * n + ref.columns[c]],
                 expected[uint64_t(row) * ref.columns.size() + c]);
  return result;
}

struct DoubleReferences final {
  std::vector<double> sums, absoluteProducts, f32Bounds;
  std::vector<uint16_t> rounded;
  uint32_t columns = 0;
  double gammaK = 0;
  uint64_t sourceCoefficientsVerified = 0, sourceScaleGroups = 0;
  uint64_t negativeScaleGroups = 0, notBF16Representable = 0;
};
DoubleReferences doubleReference(const FlashAffineProjection &p,
                                const FlashTensor &cached,
                                std::span<const uint16_t> input, uint32_t rows) {
  require(input.size() == uint64_t(rows) * p.inputSize, "double reference input extent differs");
  DoubleReferences result;
  result.columns = p.outputSize;
  result.sums.resize(uint64_t(rows) * p.outputSize);
  result.absoluteProducts.resize(result.sums.size());result.f32Bounds.resize(result.sums.size());
  result.rounded.resize(result.sums.size());
  result.gammaK = gamma(p.inputSize, -24);
  const auto *cachedValues = static_cast<const float *>(cached.buffer.contents());
  std::vector<float> weights(p.inputSize);
  for (uint32_t column = 0; column < p.outputSize; ++column) {
    const auto *scales = reinterpret_cast<const uint16_t *>(
        static_cast<const std::byte *>(p.scales->buffer.contents()) +
        uint64_t(column) * p.parameterRowStrideBytes);
    for (uint32_t k = 0; k < p.inputSize; ++k) {
      weights[k] = coefficient(p, column, k);
      const float value = cachedValues[uint64_t(column) * p.inputSize + k];
      require(std::isfinite(weights[k]) && std::bit_cast<uint32_t>(value) ==
              std::bit_cast<uint32_t>(weights[k]),
              "full boundary audit found a changed cached F32 coefficient");
      ++result.sourceCoefficientsVerified;
      result.notBF16Representable += weights[k] != number(bf16(weights[k]));
      if (k % p.groupSize == 0) {
        ++result.sourceScaleGroups;
        result.negativeScaleGroups += number(scales[k / p.groupSize]) < 0;
      }
    }
    for (uint32_t row = 0; row < rows; ++row) {
      double sum = 0, absoluteProducts = 0, flushedInputProducts = 0;
      for (uint32_t k = 0; k < p.inputSize; ++k) {
        const double x = number(input[uint64_t(row) * p.inputSize + k]);
        const double weight = weights[k], term = x * weight;
        sum = std::fma(x, weight, sum);absoluteProducts += std::abs(term);
        if (std::abs(x) < std::numeric_limits<float>::min()) flushedInputProducts += std::abs(term);
      }
      require(std::isfinite(sum) && std::isfinite(absoluteProducts) &&
              absoluteProducts < std::numeric_limits<float>::max(),
              "double reference or finite F32 intermediate bound invalid");
      const uint64_t index = uint64_t(row) * p.outputSize + column;
      result.sums[index] = sum;result.absoluteProducts[index] = absoluteProducts;
      // For any binary reduction of K exact terms, each term sees at most
      // K rounding factors: one product and K-1 additions, or K FMA factors.
      // The standard gamma_K L1 bound therefore covers both FMA and separate
      // multiplication/addition. Include the independent double summation
      // uncertainty and conservative F32 input/intermediate underflow terms.
      result.f32Bounds[index] = f32DotBound(p.inputSize, absoluteProducts, flushedInputProducts);
      result.rounded[index] = bf16Double(sum);
      require(std::isfinite(number(result.rounded[index])), "double reference rounds to nonfinite BF16");
    }
  }
  return result;
}

struct BoundaryStatistics final {
  uint64_t elements = 0, exact = 0, adjacent = 0, violations = 0;
  uint32_t maximumULP = 0;
  double maximumMidpointDistance = 0, maximumF32Bound = 0;
  void add(CellRelation relation, double bound) {
    ++elements;exact += relation.ulp == 0;adjacent += relation.ulp == 1;violations += !relation.pass;
    maximumULP = std::max(maximumULP, relation.ulp);
    maximumMidpointDistance = std::max(maximumMidpointDistance, relation.midpointDistance);
    maximumF32Bound = std::max(maximumF32Bound, bound);
  }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"exact_or_signed_zero\":" << exact
        << ",\"adjacent_bf16\":" << adjacent << ",\"violations\":" << violations
        << ",\"max_bf16_ulp_from_double_gold\":" << maximumULP
        << ",\"max_shared_midpoint_distance\":" << maximumMidpointDistance
        << ",\"max_f32_dot_bound\":" << maximumF32Bound << '}';
  }
};
struct BoundaryEvidence final {
  BoundaryStatistics raw, small;
  Error rawError, smallError;
  uint64_t pairAdjacent = 0, pairViolations = 0, differenceCount = 0;
  std::vector<std::string> details;
  bool pass() const { return !raw.violations && !small.violations && !pairViolations; }
  void write(std::ostream &out) const {
    out << "{\"raw_boundary\":";raw.write(out);
    out << ",\"small_boundary\":";small.write(out);
    out << ",\"raw_vs_double_bf16\":";rawError.write(out);
    out << ",\"small_vs_double_bf16\":";smallError.write(out);
    out << ",\"raw_small_adjacent_pairs\":" << pairAdjacent
        << ",\"raw_small_pair_violations\":" << pairViolations
        << ",\"differing_from_double_gold_or_each_other\":" << differenceCount
        << ",\"differences\":[";
    for (size_t i = 0; i < details.size(); ++i) { if (i) out << ',';out << details[i]; }
    out << "],\"difference_details_truncated\":" << (differenceCount > details.size() ? "true" : "false")
        << ",\"pass\":" << (pass() ? "true" : "false") << '}';
  }
};
BoundaryEvidence boundaryEvidence(std::span<const uint16_t> raw,
                                 std::span<const uint16_t> small,
                                 const DoubleReferences &ref) {
  require(raw.size() == ref.sums.size() && small.size() == ref.sums.size(),
          "full double reference output extents differ");
  BoundaryEvidence result;
  for (uint64_t i = 0; i < raw.size(); ++i) {
    const auto rawRelation = cellRelation(raw[i], ref.rounded[i], ref.sums[i], ref.f32Bounds[i]);
    const auto smallRelation = cellRelation(small[i], ref.rounded[i], ref.sums[i], ref.f32Bounds[i]);
    const uint32_t pairULP = bf16ULP(raw[i], small[i]);
    result.raw.add(rawRelation, ref.f32Bounds[i]);result.small.add(smallRelation, ref.f32Bounds[i]);
    result.rawError.add(raw[i], ref.rounded[i]);result.smallError.add(small[i], ref.rounded[i]);
    result.pairAdjacent += pairULP == 1;result.pairViolations += pairULP > 1;
    if (raw[i] != small[i] || raw[i] != ref.rounded[i] || small[i] != ref.rounded[i]) {
      ++result.differenceCount;
      if (result.details.size() < 128) {
        std::ostringstream detail;
        detail << std::setprecision(17) << "{\"flat_index\":" << i
            << ",\"row\":" << i / ref.columns << ",\"column\":" << i % ref.columns
            << ",\"raw_bf16_bits\":" << raw[i] << ",\"small_bf16_bits\":" << small[i]
            << ",\"double_gold_bf16_bits\":" << ref.rounded[i]
            << ",\"raw_value\":" << number(raw[i]) << ",\"small_value\":" << number(small[i])
            << ",\"double_gold_bf16_value\":" << number(ref.rounded[i])
            << ",\"double_sum\":" << ref.sums[i] << ",\"sum_abs_products\":" << ref.absoluteProducts[i]
            << ",\"f32_dot_absolute_error_bound\":" << ref.f32Bounds[i]
            << ",\"raw_gold_ulp\":" << rawRelation.ulp << ",\"small_gold_ulp\":" << smallRelation.ulp
            << ",\"raw_shared_midpoint_distance\":" << rawRelation.midpointDistance
            << ",\"small_shared_midpoint_distance\":" << smallRelation.midpointDistance
            << ",\"raw_small_ulp\":" << pairULP
            << ",\"boundary_compatible\":"
            << (rawRelation.pass && smallRelation.pass && pairULP <= 1 ? "true" : "false") << '}';
        result.details.push_back(detail.str());
      }
    }
  }
  return result;
}
uint64_t checkPadding(MetalBuffer padded, std::span<const uint16_t> input,
                      uint32_t k, uint32_t rows, uint32_t tileRows) {
  require(padded.contents() && padded.sizeBytes() % 2 == 0, "padded workspace invalid");
  const auto *words = static_cast<const uint16_t *>(padded.contents());
  const uint64_t paddedRows = ((rows + tileRows - 1) / tileRows) * tileRows;
  require(paddedRows * k * 2 <= padded.sizeBytes(), "padded extent exceeds workspace");
  for (size_t i = 0; i < input.size(); ++i)
    require(words[i] == input[i], "input BF16 bits changed during row padding");
  for (uint64_t i = input.size(); i < paddedRows * k; ++i)
    require(words[i] == 0, "padded row was not positive-zero-filled");
  for (uint64_t i = paddedRows * k; i < padded.sizeBytes() / 2; ++i)
    require(words[i] == kSentinel, "unused padded workspace overwritten");
  return padded.sizeBytes() / 2;
}
void clearPadding(MetalBuffer padded) {
  std::fill_n(static_cast<uint16_t *>(padded.contents()), padded.sizeBytes() / 2, kSentinel);
}

double median(std::vector<double> values) {
  require(!values.empty(), "timing sample list empty");std::sort(values.begin(), values.end());
  const size_t n = values.size();return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Times final {
  std::vector<double> gpu, wall;
  void add(CommandTiming timing) {
    require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 0 &&
            std::isfinite(timing.wallSeconds) && timing.wallSeconds > 0, "invalid timing sample");
    gpu.push_back(timing.gpuSeconds);wall.push_back(timing.wallSeconds);
  }
  void write(std::ostream &out) const {
    out << "{\"repeats\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_wall_seconds\":" << median(wall) << ",\"gpu_seconds\":[";
    for (size_t i = 0; i < gpu.size(); ++i) { if (i) out << ',';out << gpu[i]; }
    out << "],\"wall_seconds\":[";
    for (size_t i = 0; i < wall.size(); ++i) { if (i) out << ',';out << wall[i]; }out << "]}";
  }
};
void timed(MetalBackend &backend, const std::array<const CommandGraph *, 3> &graphs,
           uint32_t repeats, std::array<Times, 3> &times) {
  for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    for (uint32_t position = 0; position < 3; ++position) {
      const uint32_t route = (repeat + position) % 3;
      times[route].add(backend.submitCommand(graphs[route]->dispatches()));
    }
}

uint32_t validateHost(MetalBackend &backend, const FlashTensor &weights,
                      MetalBuffer input, MetalBuffer output, MetalBuffer diag,
                      FlashFloatDenseSmallRowsWorkspace &workspace) {
  uint32_t rejected = 0;
  auto reject = [&](MetalBuffer x, const FlashTensor &w, MetalBuffer y, MetalBuffer d,
                    uint32_t rows, uint32_t tile) {
    bool caught = false;
    try { CommandGraph graph;addFloatDenseSmallRows(backend, graph, x, w, y, d, rows, workspace, static_cast<FlashFloatDenseSmallRowsTile>(tile)); }
    catch (const std::exception &) { caught = true; }
    require(caught, "invalid float small-row host parameters accepted");++rejected;
  };
  const uint32_t tile = 1;
  reject(input, weights, output, diag, 0, tile);
  reject(input, weights, output, diag, 17, tile);
  reject({}, weights, output, diag, 1, tile);
  reject(input, weights, {}, diag, 1, tile);
  reject(input, weights, output, {}, 1, tile);
  reject(input, weights, input, diag, 1, tile);
  reject(input, weights, output, diag, 1, 8u);
  reject(weights.buffer, weights, output, diag, 1, tile);
  reject(input, weights, weights.buffer, diag, 1, tile);
  reject(input, weights, output, backend.view(input, 0, sizeof(uint32_t)), 1, tile);
  reject(input, weights, output, backend.view(output, 0, sizeof(uint32_t)), 1, tile);
  reject(workspace.paddedInput(), weights, output, diag, 1, tile);
  reject(input, weights, workspace.paddedInput(), diag, 1, tile);
  FlashTensor invalid = weights;invalid.dtype = FlashDType::BF16;
  reject(input, invalid, output, diag, 1, tile);
  invalid = weights;invalid.shape[0] -= 1;
  reject(input, invalid, output, diag, 1, tile);
  invalid = weights;invalid.logicalBytes -= sizeof(float);
  reject(input, invalid, output, diag, 1, tile);
  return rejected;
}

// Exactly representable cancellation: preserving 1+epsilon produces epsilon;
// rounding the weight to BF16 first produces zero. Exponents -8/-12/-16/-20
// expose BF16 and other implicit narrowing of the declared F32 MPP operand.
void precisionTraps(MetalBackend &backend, std::vector<std::string> &records) {
  auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared,
                                     "float-cache trap diagnostics");
  auto *status = static_cast<uint32_t *>(diag.contents());
  FlashFloatDenseSmallRowsWorkspace workspace(backend, 32);
  for (uint32_t n : {64u, 192u}) {
    constexpr uint32_t k = 32;
    std::vector<float> weights(uint64_t(n) * k, 0);
    for (uint32_t column = 0; column < n; ++column) {
      const float sign = column % 2 ? -1.0f : 1.0f;
      const float epsilon = std::ldexp(1.0f, -std::array<int, 4>{8, 12, 16, 20}[(column / 2) % 4]);
      weights[uint64_t(column) * k] = sign * (1.0f + epsilon);
      weights[uint64_t(column) * k + 1] = -sign;
    }
    FlashTensor matrix;
    matrix.dtype = FlashDType::F32;matrix.shape = {n, k};matrix.logicalBytes = weights.size() * sizeof(float);
    matrix.buffer = backend.allocateBuffer(matrix.logicalBytes, BufferStorage::Shared,
                                           "float-cache exact cancellation matrix");
    std::memcpy(matrix.buffer.contents(), weights.data(), matrix.logicalBytes);
    for (uint32_t rows : {1u, 2u, 4u, 8u, 16u}) {
      std::vector<uint16_t> hostInput(uint64_t(rows) * k, 0), expected(uint64_t(rows) * n);
      for (uint32_t row = 0; row < rows; ++row) {
        const float x = std::array<float, 4>{1.0f, 2.0f, -1.0f, .5f}[row % 4];
        hostInput[uint64_t(row) * k] = bf16(x);hostInput[uint64_t(row) * k + 1] = bf16(x);
        hostInput[uint64_t(row) * k + 2] = 0x8000;
        for (uint32_t column = 0; column < n; ++column) {
          const float epsilon = std::ldexp(1.0f, -std::array<int, 4>{8, 12, 16, 20}[(column / 2) % 4]);
          expected[uint64_t(row) * n + column] = bf16(x * (column % 2 ? -epsilon : epsilon));
        }
      }
      Guarded input(backend, hostInput.size());input.load(hostInput);
      Guarded output(backend, expected.size());
      for (uint32_t tile = 0; tile < 4; ++tile) {
        clearPadding(workspace.paddedInput());output.clear();*status = kSticky;
        CommandGraph graph;
        addFloatDenseSmallRows(backend, graph, input.view, matrix, output.view, diag, rows, workspace, static_cast<FlashFloatDenseSmallRowsTile>(tile));
        (void)backend.submitCommand(graph.dispatches());
        output.check();input.check();
        require(*status == kSticky, "precision trap changed sticky diagnostics");
        require(compare(input.values(), hostInput).mismatch == 0, "precision trap input changed");
        checkPadding(workspace.paddedInput(), hostInput, k, rows, kTileRows[tile]);
        const auto error = compare(output.values(), expected);
        std::ostringstream record;
        record << "{\"n\":" << n << ",\"k\":" << k << ",\"rows\":" << rows
               << ",\"tile\":" << splash::json::quote(kTileNames[tile])
               << ",\"epsilon_exponents\":[-8,-12,-16,-20],\"preserved_first_weight\":1.00390625,\"rounded_weight\":1,"
                  "\"expected_first_output\":0.00390625,\"bf16_rounded_first_output\":0,"
                  "\"error\":";
        error.write(record);record << '}';records.push_back(record.str());
        require(error.mismatch == 0 && !error.nonfinite,
                "mixed MPP lost F32 weight bits in exact cancellation trap");
      }
    }
  }
}

void cpuSelfTest() {
  require(sizeof(splash::metal::CommandTiming) == 200, "current production CommandTiming ABI must be 200 bytes");
  uint64_t checks = 1;
  for (uint32_t word = 0; word < 65536; ++word) {
    const auto original = uint16_t(word);
    if (std::isfinite(number(original))) {
      require(bf16(number(original)) == original, "BF16 finite round-trip failed");++checks;
    }
  }
  for (uint32_t width : {4u, 5u, 6u, 8u}) {
    for (uint32_t length : {32u, 64u, 128u, 640u, 2560u}) {
      std::vector<std::byte> packed((uint64_t(length) * width + 7) / 8);
      for (uint32_t k = 0; k < length; ++k) {
        const uint32_t value = uint32_t(randomWord(k + width)) & ((1u << width) - 1);
        for (uint32_t b = 0; b < width; ++b) {
          const uint64_t bit = uint64_t(k) * width + b;
          packed[bit / 8] |= std::byte(((value >> b) & 1) << (bit % 8));
        }
      }
      for (uint32_t k = 0; k < length; ++k) {
        require(code(packed.data(), width, k) ==
                (uint32_t(randomWord(k + width)) & ((1u << width) - 1)),
                "independent packed code oracle failed");++checks;
      }
    }
  }
  for (uint32_t rows : {1u, 2u, 3u, 4u, 8u, 9u, 16u}) {
    for (uint32_t tileRows : {8u, 16u}) {
      constexpr uint32_t k = 320;
      const uint32_t paddedRows = (rows + tileRows - 1) / tileRows * tileRows;
      std::vector<uint16_t> source(rows * k), padded(16 * k, kSentinel);
      for (size_t i = 0; i < source.size(); ++i) source[i] = uint16_t(randomWord(i));
      std::copy(source.begin(), source.end(), padded.begin());
      std::fill(padded.begin() + source.size(), padded.begin() + paddedRows * k, 0);
      for (size_t i = 0; i < padded.size(); ++i) {
        require(padded[i] == (i < source.size() ? source[i] :
            (i < paddedRows * k ? 0 : kSentinel)), "independent padding extent oracle failed");++checks;
      }
    }
  }
  for (float x : {1.0f, 2.0f, -1.0f, .5f})
    for (float sign : {1.0f, -1.0f})
      for (int exponent : {-8, -12, -16, -20}) {
        const float epsilon = std::ldexp(1.0f, exponent);
        const float preserved = std::fma(x, sign * (1.0f + epsilon), x * -sign);
        const float rounded = std::fma(x, number(bf16(sign * (1.0f + epsilon))), x * -sign);
        require(preserved == x * sign * epsilon && rounded == 0 &&
                number(bf16(preserved)) == preserved, "mixed-operand cancellation trap failed");++checks;
      }
  const float signedAffine = 7.0f * -14.25f + 128.0f;
  require(signedAffine == 28.25f, "signed-affine reconstruction trap failed");++checks;
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82,
          "BF16 round-to-nearest-even ties failed");checks += 2;
  Error exact;exact.add(0x8000, 0);require(exact.maximumULP == 0, "signed-zero ULP mapping failed");++checks;
  std::array<float, 32> lanes{};
  for (uint32_t i = 0; i < lanes.size(); ++i) lanes[i] = float(i);
  require(pairwiseSum32(lanes) == 496, "independent SIMD32 reduction failed");++checks;
  const auto selected = parsePrefixes("a,b,a");
  require(selected == std::vector<std::string>{"a", "b", "a"},
          "explicit prefix parsing failed");++checks;
  for (const char *invalid : {"", ",a", "a,", "a,,b"}) {
    bool caught = false;
    try { (void)parsePrefixes(invalid); } catch (const std::exception &) { caught = true; }
    require(caught, "invalid explicit prefix list accepted");++checks;
  }
  std::cout << "{\"pass\":true,\"command_timing_bytes\":" << sizeof(splash::metal::CommandTiming) << ",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance << ",\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}
void writeReport(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path);require(bool(out), "cannot open float-cache oracle report");
  out << text << '\n';out.close();require(bool(out), "cannot write float-cache oracle report");
}
void writeRecords(std::ostream &out, const std::vector<std::string> &records) {
  out << '[';
  for (size_t i = 0; i < records.size(); ++i) { if (i) out << ',';out << records[i]; }out << ']';
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string metadata;
    std::vector<std::string> records, traps;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest();return 0; }
      if (argc != 4)
        throw std::invalid_argument("usage: flash-float-dense-cache-oracle METALLIB PACKAGE REPORT_JSON | --trap-only METALLIB REPORT_JSON | --cpu-self-test");
      const bool trapOnly = std::string(argv[1]) == "--trap-only";
      const auto rowCounts = list("FLASH_FLOAT_CACHE_ROWS", {4}, 16);
      const auto tiles = list("FLASH_FLOAT_CACHE_TILES", {5}, 5);
      const auto repeatCounts = list("FLASH_FLOAT_CACHE_REPEATS", {6}, 16);
      const auto cpuColumnCounts = list("FLASH_FLOAT_CACHE_CPU_COLUMNS", {19}, 128);
      const bool bodyAudit = auditSwitch("FLASH_DENSE_F32_TILES_V8_BOUNDARY_AUDIT");
      require(!bodyAudit, "private v8 screen requires strict 1e-4 acceptance; boundary fallback is disabled");
      const bool continueAccuracyFailures = auditSwitch("FLASH_FLOAT_CACHE_CONTINUE_ON_ACCURACY_FAILURE");
      require(repeatCounts.size() == 1 && repeatCounts[0] >= 2 && cpuColumnCounts.size() == 1 &&
              cpuColumnCounts[0] > 0, "repeats must be 2..16 and CPU columns one positive value");
      for (uint32_t rows : rowCounts) require(rows > 0, "small-row counts must be positive");
      MetalBackend backend(trapOnly ? argv[2] : argv[1]);
      std::ostringstream info;
      info << "\"schema\":\"flash-qsa-output-f32-n32-production-bridge-oracle-v1\",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance << ",\"command_timing_bytes\":" << sizeof(splash::metal::CommandTiming) << ",\"operand_format\":" << splash::json::quote(kFlashFloatDenseCacheOperandFormat)
          << ",\"metallib_path\":" << splash::json::quote(
              std::filesystem::absolute(trapOnly ? argv[2] : argv[1]).lexically_normal().string())
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(digestHex(backend.metallibSha256()))
          << ",\"small_row_semantics\":" << splash::json::quote(kFlashFloatDenseSmallRowsSemantics)
          << ",\"raw_semantics\":" << splash::json::quote(flashAffineSemantics())
          << ",\"cpu_reference\":\"independent-byte-unpack; exact-f32-affine; serial-f32-fma-and-simd32-lanes\""
          << ",\"device\":" << splash::json::quote(backend.capabilities().deviceName)
          << ",\"gpu_core_count\":" << backend.capabilities().gpuCoreCount
          << ",\"macos\":" << splash::json::quote(backend.capabilities().macosVersion())
          << ",\"shader_validation_environment\":" << splash::json::quote(
              std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset")
          << ",\"timing_order\":\"rotating-raw-candidate-selected; warm-before-sampling\""
          << ",\"relative_l2_acceptance_limit\":" << kRelativeL2Limit
          << ",\"qmv_f32_environment\":" << splash::json::quote(
              std::getenv("SPLASH_FLASH_QMV_F32") ? std::getenv("SPLASH_FLASH_QMV_F32") : "unset")
          << ",\"body_boundary_audit_enabled\":" << (bodyAudit ? "true" : "false")
          << ",\"continue_on_accuracy_failure\":" << (continueAccuracyFailures ? "true" : "false")
          << ",\"head_acceptance_rule\":\"strict-relative-l2-below-1e-4\""
          << ",\"body_boundary_acceptance_rule\":\"strict-relative-l2-below-1e-4-versus-raw-and-selected; boundary-fallback-prohibited\"";
      metadata = info.str();
      precisionTraps(backend, traps);
      if (trapOnly) {
        std::ostringstream report;report << '{' << metadata << ",\"precision_traps\":";
        writeRecords(report, traps);report << ",\"pass\":true,\"original_models_modified\":false} ";
        writeReport(argv[3], report.str());return 0;
      }
      auto weights = FlashWeights::load(backend, argv[2]);
      metadata += ",\"source_identity\":" + splash::json::quote(weights.sourceIdentity()) +
                  ",\"manifest_fingerprint\":" + splash::json::quote(weights.manifestFingerprint());
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared,
                                         "float-cache oracle diagnostics");
      auto *status = static_cast<uint32_t *>(diag.contents());
      uint64_t paddingChecks = 0, accuracyFailures = 0;
      const auto selectedPrefixes = prefixes(weights);
      for (const auto &prefix : selectedPrefixes) {
        const auto &p = weights.projection(prefix);
        require(p.experts == 1 && p.outputSize % 64 == 0 && p.inputSize % 32 == 0 &&
                p.inputSize <= 32768, "selected matrix is incompatible with small-row operator");
        const std::vector<std::string> singlePrefix{prefix};
        const uint64_t planned = FlashFloatDenseCache::plannedBytes(weights, singlePrefix);
        const uint64_t before = backend.memoryStats().allocatedBytes;
        FlashFloatDenseCache cache(backend, weights, singlePrefix); // one selected matrix resident at a time
        require(backend.memoryStats().allocatedBytes - before == cache.allocatedBytes() &&
                cache.allocatedBytes() <= planned, "cache allocation ledger or plan differs");
        require(cache.contains(prefix), "cache omitted selected matrix");
        const auto coefficients = coefficientSamples(p, cache.tensor(prefix));
        FlashFloatDenseSmallRowsWorkspace workspace(backend);
        const MetalBuffer padded = workspace.paddedInput();
        require(padded.contents() && padded.sizeBytes() >= uint64_t(16) * 32768 * 2 &&
                workspace.allocatedBytes() >= padded.sizeBytes(), "workspace reported extent invalid");
        for (uint32_t rows : rowCounts) {
          std::vector<uint16_t> hostInput(uint64_t(rows) * p.inputSize);
          for (size_t i = 0; i < hostInput.size(); ++i) {
            const int32_t value = int32_t(randomWord(i + 0xc0ffee) % 2047) - 1023;
            hostInput[i] = bf16(float(value) / 1024.0f);
          }
          for (size_t i = 0; i < std::min<size_t>(hostInput.size(), 5); ++i)
            hostInput[i] = std::array<uint16_t, 5>{0x8000, 0x0001, 0x007f, 0x0080, 0x3f81}[i];
          Guarded input(backend, hostInput.size());input.load(hostInput);
          Guarded raw(backend, uint64_t(rows) * p.outputSize);
          Guarded small(backend, uint64_t(rows) * p.outputSize);
          const uint32_t hostRejections = validateHost(backend, cache.tensor(prefix), input.view,
                                                      small.view, diag, workspace);
          const auto ref = reference(p, hostInput, rows, cpuColumnCounts[0]);
          const bool auditThisMatrix = bodyAudit && prefix != kDefaultPrefix;
          std::optional<DoubleReferences> doubleRef;
          CommandGraph rawGraph;
          addAffine(rawGraph, input.view, p, raw.view, diag, rows);
          *status = kSticky;(void)backend.submitCommand(rawGraph.dispatches());
          raw.check();input.check();
          require(*status == kSticky && compare(input.values(), hostInput).mismatch == 0,
                  "warm raw control changed diagnostics or input");
          const auto rawSerial = compareSample(raw.values(), ref, ref.serial, p.outputSize, rows);
          const auto rawSIMD = compareSample(raw.values(), ref, ref.simd32, p.outputSize, rows);
          const auto selectedTile = flashFloatDenseSmallRowsPolicy(prefix, rows,
              p.outputSize, p.inputSize, p.bits, p.groupSize);
          Guarded selected(backend, uint64_t(rows) * p.outputSize);
          FlashFloatDenseSmallRowsWorkspace selectedWorkspace(backend);
          clearPadding(selectedWorkspace.paddedInput());
          CommandGraph selectedGraph;
          if (selectedTile)
            addFloatDenseSmallRows(backend, selectedGraph, input.view, cache.tensor(prefix), selected.view, diag,
                rows, selectedWorkspace, *selectedTile);
          else addAffine(selectedGraph, input.view, p, selected.view, diag, rows);
          (void)backend.submitCommand(selectedGraph.dispatches()); selected.check();
          require(*status == kSticky, "selected policy changed diagnostics");
          require(!rawSerial.nonfinite && !rawSIMD.nonfinite,
                  "raw control is nonfinite against independent CPU reference");
          for (uint32_t tile : tiles) {
            clearPadding(padded);small.clear();*status = kSticky;
            CommandGraph smallGraph;
            require(tile == 5 && selectedTile.has_value(), "production bridge screen requires cached source and tile5");
            require(flashQSAOutF32N32Enabled(), "production bridge flag must be enabled");
            const uint64_t beforeDispatches = cache.qsaOutF32N32Dispatches();
            const uint64_t beforeRows = cache.qsaOutF32N32RealRows();
            cache.addSmallRows(smallGraph, prefix, input.view, small.view, diag, rows, workspace, *selectedTile);
            require(smallGraph.dispatches().size() == 2 &&
                smallGraph.dispatches()[0].pipelineName == "flash_float_dense_small_rows_pad" &&
                smallGraph.dispatches()[1].pipelineName == "flash_qsa_out_f32_n32_m8_n32_s4" &&
                smallGraph.dispatches()[1].threadgroups.x == 80 &&
                smallGraph.dispatches()[1].threadgroups.y == (rows + 7) / 8 &&
                cache.qsaOutF32N32Dispatches() == beforeDispatches + 1 &&
                cache.qsaOutF32N32RealRows() == beforeRows + rows,
                "production bridge graph/count differs from qualified geometry");
            (void)backend.submitCommand(smallGraph.dispatches());
            small.check();input.check();require(*status == kSticky, "small-row warm diagnostic changed");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            const std::vector<uint16_t> firstOutput(small.values().begin(), small.values().end());
            std::array<Times, 3> times;
            timed(backend, {&rawGraph, &smallGraph, &selectedGraph}, repeatCounts[0], times);
            raw.check();small.check();input.check();
            require(*status == kSticky && compare(input.values(), hostInput).mismatch == 0,
                    "timed routes changed input or diagnostics");
            require(compare(small.values(), firstOutput).mismatch == 0, "small-row repeat output changed");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            const auto versusRaw = compare(small.values(), raw.values());
            const auto versusSelected = compare(small.values(), selected.values());
            const auto smallSerial = compareSample(small.values(), ref, ref.serial, p.outputSize, rows);
            const auto smallSIMD = compareSample(small.values(), ref, ref.simd32, p.outputSize, rows);
            const auto roundedCounterfactual = compareSample(small.values(), ref, ref.roundedSerial,
                                                             p.outputSize, rows);
            const bool strictAccuracyPass = !versusRaw.nonfinite && !smallSerial.nonfinite &&
                !smallSIMD.nonfinite && !versusSelected.nonfinite && versusSelected.relativeL2() < kRelativeL2Limit && versusRaw.relativeL2() < kRelativeL2Limit &&
                std::min(smallSerial.relativeL2(), smallSIMD.relativeL2()) < kRelativeL2Limit;
            // Full-column source verification and double dots are needed only
            // to justify an exception. Keep screening of already strict-pass
            // matrices bounded, and retain the full evidence on every exception.
            if (auditThisMatrix && !strictAccuracyPass && !doubleRef)
              doubleRef = doubleReference(p, cache.tensor(prefix), hostInput, rows);
            std::optional<BoundaryEvidence> boundaries;
            if (doubleRef) boundaries = boundaryEvidence(raw.values(), small.values(), *doubleRef);
            const bool boundaryFallback = auditThisMatrix && !strictAccuracyPass && boundaries->pass();
            const bool accuracyPass = strictAccuracyPass || boundaryFallback;
            const uint32_t paddedRows = (rows + kTileRows[tile] - 1) / kTileRows[tile] * kTileRows[tile];
            std::ostringstream record;
            record << std::setprecision(12) << "{\"projection\":" << splash::json::quote(prefix)
                << ",\"rows\":" << rows << ",\"padded_rows\":" << paddedRows
                << ",\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"n\":" << p.outputSize << ",\"k\":" << p.inputSize
                << ",\"bits\":" << p.bits << ",\"group_size\":" << p.groupSize
                << ",\"source_weight_row_stride_bytes\":" << p.weightRowStrideBytes
                << ",\"source_parameter_row_stride_bytes\":" << p.parameterRowStrideBytes
                << ",\"n128_column_tail\":" << (p.outputSize % kTileOutputs[tile])
                << ",\"workspace_bytes\":" << workspace.allocatedBytes()
                << ",\"planned_cache_bytes\":" << planned
                << ",\"allocated_cache_bytes\":" << cache.allocatedBytes()
                << ",\"cache_identity_sha256\":" << splash::json::quote(cache.identitySha256())
                << ",\"cache_initialization_timing\":{\"gpu_seconds\":"
                << cache.initializationTiming().gpuSeconds << ",\"wall_seconds\":"
                << cache.initializationTiming().wallSeconds << '}'
                << ",\"cache_coefficients\":";coefficients.write(record);
            record << ",\"sampled_cpu_columns\":" << ref.columns.size()
                << ",\"host_rejections\":" << hostRejections
                << ",\"small_dispatches\":" << smallGraph.dispatches().size()
                << ",\"raw_dispatch_pipelines\":[";
            for (size_t i = 0; i < rawGraph.dispatches().size(); ++i) {
              if (i) record << ',';
              record << splash::json::quote(rawGraph.dispatches()[i].pipelineName);
            }
            record << "],\"raw_timing\":";times[0].write(record);
            record << ",\"small_timing\":";times[1].write(record);
            record << ",\"candidate_tile_n\":" << kTileOutputs[tile] << ",\"candidate_weight_reduction_alternative\":true,\"selected_policy_tile\":" << splash::json::quote(selectedTile ? kTileNames[static_cast<uint32_t>(*selectedTile)] : "raw-f32-qmv");
            record << ",\"selected_policy_timing\":"; times[2].write(record);
            record << ",\"speedup_vs_selected_policy_gpu\":" << median(times[2].gpu) / median(times[1].gpu)
                << ",\"speedup_vs_selected_policy_wall\":" << median(times[2].wall) / median(times[1].wall)
                << ",\"small_vs_selected_policy\":"; versusSelected.write(record);
            record << ",\"speedup_vs_raw_gpu\":" << median(times[0].gpu) / median(times[1].gpu)
                << ",\"speedup_vs_raw_wall\":" << median(times[0].wall) / median(times[1].wall)
                << ",\"small_vs_raw_f32_coefficients\":";versusRaw.write(record);
            record << ",\"small_vs_serial_f32_cpu\":";smallSerial.write(record);
            record << ",\"small_vs_simd32_f32_cpu\":";smallSIMD.write(record);
            record << ",\"raw_vs_serial_f32_cpu\":";rawSerial.write(record);
            record << ",\"raw_vs_simd32_f32_cpu\":";rawSIMD.write(record);
            record << ",\"small_vs_counterfactual_bf16_weight_cpu\":";roundedCounterfactual.write(record);
            record << ",\"strict_relative_l2_pass\":" << (strictAccuracyPass ? "true" : "false")
                << ",\"acceptance_rule\":" << splash::json::quote(boundaryFallback ?
                    "full-independent-double-bf16-boundary-compatibility" : "strict-relative-l2-below-1e-4");
            if (doubleRef) {
              record << ",\"full_source_coefficient_verification\":{\"f32_bit_mismatches\":0"
                  << ",\"coefficients_verified\":" << doubleRef->sourceCoefficientsVerified
                  << ",\"scale_groups\":" << doubleRef->sourceScaleGroups
                  << ",\"negative_scale_groups\":" << doubleRef->negativeScaleGroups
                  << ",\"not_bf16_representable\":" << doubleRef->notBF16Representable
                  << ",\"f32_gamma_k\":" << doubleRef->gammaK << '}'
                  << ",\"full_double_boundary_evidence\":";
              boundaries->write(record);
            }
            record << ",\"accuracy_pass\":" << (accuracyPass ? "true" : "false")
                << ",\"coefficient_bits_exact\":true,\"padding_bits_exact\":true,"
                   "\"output_row_guards_pass\":true,\"unused_workspace_canary_pass\":true}";
            records.push_back(record.str());
            accuracyFailures += !accuracyPass;
            if (!continueAccuracyFailures)
              require(accuracyPass, auditThisMatrix ?
                      "FP32 body projection violates full double BF16 rounding-boundary evidence" :
                      "FP32 projection route exceeds strict 1e-4 relative L2 bound");
            // Diagnose nonfinite copying separately from timed finite input.
            hostInput[0] = 0x7fc5;input.load(hostInput);small.clear();*status = kSticky;
            clearPadding(padded);(void)backend.submitCommand(smallGraph.dispatches());
            small.check(false);input.check(false);
            require(*status == (kSticky | kFlashAffineInvalidNumerics),
                    "small-row nonfinite diagnostic missing or not sticky");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            require(compare(input.values(), hostInput).mismatch == 0, "nonfinite source bits changed");
            hostInput[0] = 0x8000;input.load(hostInput);
            std::cerr << "float-cache projection=" << prefix << " rows=" << rows << " tile=" << kTileNames[tile]
                      << " raw_speedup=" << median(times[0].gpu) / median(times[1].gpu)
                      << " relative_l2=" << versusRaw.relativeL2() << '\n';
          }
        }
      }
      require(!records.empty(), "float-cache oracle produced no variants");
      std::ostringstream report;report << '{' << metadata << ",\"precision_traps\":";
      writeRecords(report, traps);report << ",\"cases\":";writeRecords(report, records);
      report << ",\"variants\":" << records.size() << ",\"padding_word_checks\":" << paddingChecks
             << ",\"selected_matrices\":" << selectedPrefixes.size()
             << ",\"accuracy_failures\":" << accuracyFailures
             << ",\"pass\":" << (accuracyFailures ? "false" : "true")
             << ",\"original_models_modified\":false,\"source_production_routing_default_changed\":false}";
      writeReport(argv[3], report.str());return accuracyFailures ? 1 : 0;
    } catch (const std::exception &error) {
      if (argc == 4) {
        try {
          std::ostringstream report;report << '{';if (!metadata.empty()) report << metadata << ',';
          report << "\"pass\":false,\"error\":" << splash::json::quote(error.what())
                 << ",\"precision_traps\":";writeRecords(report, traps);
          report << ",\"completed_cases\":";writeRecords(report, records);report << '}';
          writeReport(argv[3], report.str());
        } catch (...) {}
      }
      std::cerr << "flash-float-dense-cache-oracle: " << error.what() << '\n';return 1;
    }
  }
}
