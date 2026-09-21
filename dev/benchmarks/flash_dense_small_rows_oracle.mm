// Root-only GPU oracle. --cpu-self-test does not create a Metal device.
// Cached BF16 operands deliberately differ from raw F32 affine coefficients.
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashDenseCache.hpp"
#include "engine/Json.hpp"

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
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint64_t kGuardWords = 32;
constexpr std::array<uint32_t, 4> kTileRows{8, 8, 16, 16};
constexpr std::array<const char *, 4> kTileNames{"m8n64", "m8n128", "m16n64", "m16n128"};
constexpr std::array<const char *, 4> kDefaultPrefixes{
    "language_model.lm_head",
    "language_model.model.layers.1.linear_attn.in_proj_qkv",
    "language_model.model.layers.0.linear_attn.in_proj_qkv",
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
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
uint32_t code(const std::byte *row, uint32_t bits, uint32_t k) {
  const uint64_t bit = uint64_t(k) * bits;
  const uint32_t shift = uint32_t(bit % 8);
  uint32_t word = std::to_integer<uint8_t>(row[bit / 8]);
  if (shift + bits > 8)
    word |= uint32_t(std::to_integer<uint8_t>(row[bit / 8 + 1])) << 8;
  return (word >> shift) & ((1u << bits) - 1);
}
float coefficientF32(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *packed = static_cast<const std::byte *>(p.weights->buffer.contents()) +
                       uint64_t(n) * p.weightRowStrideBytes;
  const auto *scales = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.scales->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  const auto *biases = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.biases->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  const float product = float(code(packed, p.bits, k)) * number(scales[k / p.groupSize]);
  return product + number(biases[k / p.groupSize]);
}
const char *environment(const char *name, const char *alias = nullptr) {
  const char *value = std::getenv(name);
  return value ? value : (alias ? std::getenv(alias) : nullptr);
}
std::vector<uint32_t> list(const char *name, const char *alias,
                          std::initializer_list<uint32_t> fallback, uint32_t maximum) {
  const char *raw = environment(name, alias);
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
std::vector<std::string> prefixes() {
  std::vector<std::string> result;
  if (const char *raw = std::getenv("FLASH_DENSE_SMALL_PREFIXES")) {
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
      require(!item.empty(), "empty FLASH_DENSE_SMALL_PREFIXES entry");result.push_back(item);
    }
  } else {
    for (const char *prefix : kDefaultPrefixes) result.emplace_back(prefix);
    const char *filter = environment("FLASH_DENSE_SMALL_PREFIX", "PREFIX");
    if (!filter) filter = "lm_head";
    std::erase_if(result, [&](const std::string &prefix) {
      return prefix.find(filter) == std::string::npos;
    });
  }
  require(!result.empty(), "small-row prefix filter selected no matrices");
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
    const auto ordered = [](uint16_t x) -> uint32_t {
      return (x & 0x8000) ? uint32_t(0x8000 - (x & 0x7fff)) : uint32_t(0x8000 + x);
    };
    const uint32_t x = ordered(actual), y = ordered(expected);
    maximumULP = std::max(maximumULP, x > y ? x - y : y - x);
  }
  double relativeL2() const { return std::sqrt(squaredError / std::max(1e-30, squaredReference)); }
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
                                  "small-row oracle guarded BF16 buffer");
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
              "small-row oracle guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])), "output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};

struct References final {
  std::vector<uint32_t> columns;
  std::vector<uint16_t> raw, rounded;
};
References reference(const FlashAffineProjection &p, std::span<const uint16_t> input,
                     uint32_t rows, uint32_t count) {
  References result;
  count = std::min(count, p.outputSize);
  for (uint32_t c = 0; c < count; ++c)
    result.columns.push_back(count == 1 ? 0 : uint64_t(c) * (p.outputSize - 1) / (count - 1));
  result.raw.resize(uint64_t(rows) * count);result.rounded.resize(result.raw.size());
  std::vector<float> raw(p.inputSize), rounded(p.inputSize);
  for (uint32_t c = 0; c < count; ++c) {
    for (uint32_t k = 0; k < p.inputSize; ++k) {
      raw[k] = coefficientF32(p, result.columns[c], k);rounded[k] = number(bf16(raw[k]));
    }
    for (uint32_t row = 0; row < rows; ++row) {
      float rawSum = 0, roundedSum = 0;
      for (uint32_t k = 0; k < p.inputSize; ++k) {
        const float x = number(input[uint64_t(row) * p.inputSize + k]);
        rawSum = std::fma(x, raw[k], rawSum);roundedSum = std::fma(x, rounded[k], roundedSum);
      }
      result.raw[uint64_t(row) * count + c] = bf16(rawSum);
      result.rounded[uint64_t(row) * count + c] = bf16(roundedSum);
    }
  }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const References &expected,
                    bool rounded, uint32_t n, uint32_t rows) {
  const auto &values = rounded ? expected.rounded : expected.raw;
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (size_t c = 0; c < expected.columns.size(); ++c)
      result.add(actual[uint64_t(row) * n + expected.columns[c]],
                 values[uint64_t(row) * expected.columns.size() + c]);
  return result;
}
uint64_t coefficientSamples(const FlashAffineProjection &p, const FlashTensor &cached) {
  require(p.experts == 1 && cached.dtype == FlashDType::BF16 && cached.shape ==
      std::vector<uint64_t>{p.outputSize, p.inputSize} && cached.buffer.contents(),
      "cached coefficient dimensions, dtype or visibility invalid");
  const auto *values = static_cast<const uint16_t *>(cached.buffer.contents());
  const uint64_t total = uint64_t(p.outputSize) * p.inputSize;
  const uint64_t count = std::min<uint64_t>(total, 16384);
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t index = i < 2 ? (i ? total - 1 : 0) : randomWord(i + 0xd35e) % total;
    require(values[index] == bf16(coefficientF32(p, uint32_t(index / p.inputSize),
                                                 uint32_t(index % p.inputSize))),
            "cached coefficient differs from independently rounded source");
  }
  return count;
}
uint64_t checkPadding(MetalBuffer padded, std::span<const uint16_t> input,
                      uint32_t k, uint32_t rows, uint32_t tileRows) {
  require(padded.contents() && padded.sizeBytes() % 2 == 0, "padded workspace invalid");
  const auto *words = static_cast<const uint16_t *>(padded.contents());
  const uint64_t paddedRows = ((rows + tileRows - 1) / tileRows) * tileRows;
  require(paddedRows * k * 2 <= padded.sizeBytes(), "padded extent exceeds workspace");
  for (size_t i = 0; i < input.size(); ++i)
    require(words[i] == input[i], "input bits changed during row padding");
  for (uint64_t i = input.size(); i < paddedRows * k; ++i)
    require(words[i] == 0, "padded row was not zero-filled");
  for (uint64_t i = paddedRows * k; i < padded.sizeBytes() / 2; ++i)
    require(words[i] == kSentinel, "unused padded workspace overwritten");
  return padded.sizeBytes() / 2;
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
  // Six permutations balance both direction and position across the routes.
  constexpr std::array<std::array<uint32_t, 3>, 6> orders{{{0, 1, 2}, {2, 1, 0},
      {1, 2, 0}, {0, 2, 1}, {2, 0, 1}, {1, 0, 2}}};
  for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    for (uint32_t route : orders[repeat % orders.size()])
      times[route].add(backend.submitCommand(graphs[route]->dispatches()));
}
uint32_t validateHost(MetalBackend &backend, const FlashTensor &weights,
                      MetalBuffer input, MetalBuffer output, MetalBuffer diag,
                      FlashDenseSmallRowsWorkspace &workspace) {
  uint32_t rejected = 0;
  auto reject = [&](MetalBuffer x, const FlashTensor &w, MetalBuffer y, MetalBuffer d,
                    uint32_t rows, FlashDenseSmallRowsTile tile) {
    bool caught = false;
    try { CommandGraph graph;addDenseBF16SmallRows(backend, graph, x, w, y, d, rows, workspace, tile); }
    catch (const std::exception &) { caught = true; }
    require(caught, "invalid small-row host parameters accepted");++rejected;
  };
  const auto tile = FlashDenseSmallRowsTile::M8N128;
  reject(input, weights, output, diag, 0, tile);
  reject(input, weights, output, diag, 17, tile);
  reject({}, weights, output, diag, 1, tile);
  reject(input, weights, {}, diag, 1, tile);
  reject(input, weights, output, {}, 1, tile);
  reject(input, weights, input, diag, 1, tile);
  reject(input, weights, output, diag, 1, static_cast<FlashDenseSmallRowsTile>(4));
  reject(weights.buffer, weights, output, diag, 1, tile);
  reject(input, weights, weights.buffer, diag, 1, tile);
  reject(input, weights, output, backend.view(input, 0, sizeof(uint32_t)), 1, tile);
  reject(input, weights, output, backend.view(output, 0, sizeof(uint32_t)), 1, tile);
  reject(workspace.paddedInput(), weights, output, diag, 1, tile);
  reject(input, weights, workspace.paddedInput(), diag, 1, tile);
  FlashTensor invalid = weights;invalid.dtype = FlashDType::F32;
  reject(input, invalid, output, diag, 1, tile);
  invalid = weights;invalid.shape[0] -= 1;
  reject(input, invalid, output, diag, 1, tile);
  return rejected;
}

void cpuSelfTest() {
  uint64_t checks = 0;
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
  const float trap = 7.0f * -14.25f + 128.0f;
  require(trap == 28.25f && number(bf16(trap)) == 28.25f,
          "signed-affine coefficient trap failed");++checks;
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82,
          "BF16 round-to-nearest-even ties failed");checks += 2;
  Error exact;exact.add(0x8000, 0);require(exact.maximumULP == 0, "signed-zero ULP mapping failed");++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}
void writeReport(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path);require(bool(out), "cannot open small-row oracle report");
  out << text << '\n';out.close();require(bool(out), "cannot write small-row oracle report");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string metadata;
    std::vector<std::string> records;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest();return 0; }
      if (argc != 4)
        throw std::invalid_argument("usage: flash-dense-small-rows-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const auto selectedPrefixes = prefixes();
      const auto rowCounts = list("FLASH_DENSE_SMALL_ROWS", "ROWS", {4}, 16);
      const auto tiles = list("FLASH_DENSE_SMALL_TILES", "TILES", {1}, 3);
      const auto repeatCounts = list("FLASH_DENSE_SMALL_REPEATS", "REPEATS", {3}, 7);
      const auto cpuColumnCounts = list("FLASH_DENSE_SMALL_CPU_COLUMNS", nullptr, {19}, 128);
      require(repeatCounts.size() == 1 && repeatCounts[0] >= 3 && cpuColumnCounts.size() == 1 &&
              cpuColumnCounts[0] > 0, "repeats must be 3..7 and CPU columns one positive value");
      for (uint32_t rows : rowCounts) require(rows > 0, "small-row counts must be positive");
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared,
                                         "small-row oracle diagnostics");
      auto *status = static_cast<uint32_t *>(diag.contents());
      std::ostringstream info;
      info << "\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"operand_format\":" << splash::json::quote(kFlashDenseCacheOperandFormat)
          << ",\"small_row_semantics\":" << splash::json::quote(kFlashDenseSmallRowsSemantics)
          << ",\"raw_semantics\":" << splash::json::quote(flashAffineSemantics())
          << ",\"cpu_reference\":\"independent-byte-unpack; separate-f32-affine-and-rounded-bf16; serial-f32-fma; sampled-columns\""
          << ",\"device\":" << splash::json::quote(backend.capabilities().deviceName)
          << ",\"gpu_core_count\":" << backend.capabilities().gpuCoreCount
          << ",\"macos\":" << splash::json::quote(backend.capabilities().macosVersion())
          << ",\"shader_validation_environment\":" << splash::json::quote(
              std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset")
          << ",\"timing_order\":\"six-permutation-rotated-raw-vector-small; warm-before-sampling\"";
      metadata = info.str();
      uint64_t variants = 0, paddingChecks = 0, coefficientChecks = 0;
      for (const auto &prefix : selectedPrefixes) {
        const auto &p = weights.projection(prefix);
        require(p.experts == 1 && p.outputSize % 64 == 0 && p.inputSize % 32 == 0 &&
                p.inputSize <= 32768, "selected matrix is incompatible with small-row operator");
        const std::vector<std::string> selected{prefix};
        const uint64_t planned = FlashDenseCache::plannedBytes(weights, selected);
        const uint64_t before = backend.memoryStats().allocatedBytes;
        FlashDenseCache cache(backend, weights, selected); // root serial GPU queue owns this path
        require(backend.memoryStats().allocatedBytes - before == cache.actualAllocatedBytes() &&
                cache.actualAllocatedBytes() <= planned, "cache allocation ledger or plan differs");
        const uint64_t sampledCoefficients = coefficientSamples(p, cache.tensor(prefix));
        coefficientChecks += sampledCoefficients;
        FlashDenseSmallRowsWorkspace workspace(backend);
        const MetalBuffer padded = workspace.paddedInput();
        require(padded.contents() && padded.sizeBytes() >= uint64_t(16) * 32768 * 2 &&
                workspace.allocatedBytes() >= padded.sizeBytes(), "workspace reported extent invalid");
        for (uint32_t rows : rowCounts) {
          std::vector<uint16_t> hostInput(uint64_t(rows) * p.inputSize);
          for (size_t i = 0; i < hostInput.size(); ++i) {
            const int32_t value = int32_t(randomWord(i + 0xc0ffee) % 2047) - 1023;
            hostInput[i] = bf16(float(value) / 1024.0f);
          }
          // Signed zero and finite subnormals exercise the integer bit-copy pad.
          for (size_t i = 0; i < std::min<size_t>(hostInput.size(), 5); ++i)
            hostInput[i] = std::array<uint16_t, 5>{0x8000, 0x0001, 0x007f, 0x0080, 0x3f81}[i];
          Guarded input(backend, hostInput.size());input.load(hostInput);
          Guarded raw(backend, uint64_t(rows) * p.outputSize);
          Guarded vector(backend, uint64_t(rows) * p.outputSize);
          Guarded small(backend, uint64_t(rows) * p.outputSize);
          const uint32_t hostRejections = validateHost(backend, cache.tensor(prefix), input.view,
                                                      small.view, diag, workspace);
          const auto ref = reference(p, hostInput, rows, cpuColumnCounts[0]);
          CommandGraph rawGraph, vectorGraph;
          addAffine(rawGraph, input.view, p, raw.view, diag, rows);
          addDenseBF16(vectorGraph, input.view, cache.tensor(prefix), vector.view, diag, rows);
          *status = kSticky;
          (void)backend.submitCommand(rawGraph.dispatches());
          (void)backend.submitCommand(vectorGraph.dispatches());
          raw.check();vector.check();input.check();
          require(*status == kSticky && compare(input.values(), hostInput).mismatch == 0,
                  "warm controls changed diagnostics or input");
          const auto rawCPU = compareSample(raw.values(), ref, false, p.outputSize, rows);
          const auto vectorCPU = compareSample(vector.values(), ref, true, p.outputSize, rows);
          require(!rawCPU.nonfinite && rawCPU.relativeL2() < .006 &&
                  !vectorCPU.nonfinite && vectorCPU.relativeL2() < .006,
                  "vector controls differ materially from their separate CPU operand references");
          for (uint32_t tile : tiles) {
            std::fill_n(static_cast<uint16_t *>(padded.contents()), padded.sizeBytes() / 2, kSentinel);
            small.clear();*status = kSticky;
            CommandGraph smallGraph;
            addDenseBF16SmallRows(backend, smallGraph, input.view, cache.tensor(prefix), small.view,
                                 diag, rows, workspace, static_cast<FlashDenseSmallRowsTile>(tile));
            (void)backend.submitCommand(smallGraph.dispatches());
            small.check();input.check();require(*status == kSticky, "small-row warm diagnostic changed");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            const std::vector<uint16_t> firstOutput(small.values().begin(), small.values().end());
            std::array<Times, 3> times;
            timed(backend, {&rawGraph, &vectorGraph, &smallGraph}, repeatCounts[0], times);
            raw.check();vector.check();small.check();input.check();
            require(*status == kSticky && compare(input.values(), hostInput).mismatch == 0,
                    "timed routes changed input or diagnostics");
            require(compare(small.values(), firstOutput).mismatch == 0, "small-row repeat output changed");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            const auto versusVector = compare(small.values(), vector.values());
            const auto versusRaw = compare(small.values(), raw.values());
            const auto smallCPU = compareSample(small.values(), ref, true, p.outputSize, rows);
            require(!versusVector.nonfinite && versusVector.relativeL2() < .006 &&
                    !smallCPU.nonfinite && smallCPU.relativeL2() < .006,
                    "small-row route differs materially from BF16 vector or independent CPU reference");
            const uint32_t paddedRows = (rows + kTileRows[tile] - 1) / kTileRows[tile] * kTileRows[tile];
            std::ostringstream record;
            record << std::setprecision(12) << "{\"projection\":" << splash::json::quote(prefix)
                << ",\"rows\":" << rows << ",\"padded_rows\":" << paddedRows
                << ",\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"n\":" << p.outputSize << ",\"k\":" << p.inputSize
                << ",\"bits\":" << p.bits << ",\"group_size\":" << p.groupSize
                << ",\"n128_column_tail\":" << ((tile % 2) ? p.outputSize % 128 : 0)
                << ",\"workspace_bytes\":" << workspace.allocatedBytes()
                << ",\"planned_cache_bytes\":" << planned
                << ",\"allocated_cache_bytes\":" << cache.actualAllocatedBytes()
                << ",\"cache_identity_sha256\":" << splash::json::quote(cache.identitySha256())
                << ",\"cache_initialization_timing\":{\"gpu_seconds\":"
                << cache.initializationTiming().gpuSeconds << ",\"wall_seconds\":"
                << cache.initializationTiming().wallSeconds << '}'
                << ",\"sampled_cache_coefficients\":" << sampledCoefficients
                << ",\"sampled_cpu_columns\":" << ref.columns.size()
                << ",\"host_rejections\":" << hostRejections
                << ",\"small_dispatches\":" << smallGraph.dispatches().size()
                << ",\"raw_timing\":";times[0].write(record);
            record << ",\"cached_vector_timing\":";times[1].write(record);
            record << ",\"small_timing\":";times[2].write(record);
            record << ",\"speedup_vs_raw_gpu\":" << median(times[0].gpu) / median(times[2].gpu)
                << ",\"speedup_vs_cached_vector_gpu\":" << median(times[1].gpu) / median(times[2].gpu)
                << ",\"speedup_vs_raw_wall\":" << median(times[0].wall) / median(times[2].wall)
                << ",\"speedup_vs_cached_vector_wall\":" << median(times[1].wall) / median(times[2].wall)
                << ",\"small_vs_cached_vector\":";versusVector.write(record);
            record << ",\"small_vs_raw_f32_coefficients\":";versusRaw.write(record);
            record << ",\"small_vs_bf16_cpu\":";smallCPU.write(record);
            record << ",\"vector_vs_bf16_cpu\":";vectorCPU.write(record);
            record << ",\"raw_vs_f32_cpu\":";rawCPU.write(record);
            record << ",\"operand_difference_is_explicit\":true,\"padding_bits_exact\":true,"
                      "\"output_row_guards_pass\":true,\"unused_workspace_canary_pass\":true}";
            records.push_back(record.str());++variants;
            // Nonfinite bit copies and sticky diagnostics are separate from timed finite runs.
            hostInput[0] = 0x7fc5;input.load(hostInput);small.clear();*status = kSticky;
            std::fill_n(static_cast<uint16_t *>(padded.contents()), padded.sizeBytes() / 2, kSentinel);
            (void)backend.submitCommand(smallGraph.dispatches());small.check(false);input.check(false);
            require(*status == (kSticky | kFlashAffineInvalidNumerics),
                    "small-row nonfinite diagnostic missing or not sticky");
            paddingChecks += checkPadding(padded, hostInput, p.inputSize, rows, kTileRows[tile]);
            require(compare(input.values(), hostInput).mismatch == 0, "nonfinite source bits changed");
            hostInput[0] = 0x8000;input.load(hostInput);
            std::cerr << "small-row projection=" << prefix << " rows=" << rows << " tile="
                      << kTileNames[tile] << " raw_speedup="
                      << median(times[0].gpu) / median(times[2].gpu) << '\n';
          }
        }
      }
      require(variants > 0, "small-row oracle produced no variants");
      std::ostringstream report;
      report << '{' << metadata << ",\"cases\":[";
      for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
      report << "],\"variants\":" << variants << ",\"sampled_coefficient_bit_checks\":"
          << coefficientChecks << ",\"padding_word_checks\":" << paddingChecks
          << ",\"pass\":true,\"original_models_modified\":false,\"production_routing_changed\":false}";
      writeReport(argv[3], report.str());return 0;
    } catch (const std::exception &error) {
      if (argc == 4) {
        try {
          std::ostringstream report;report << '{';if (!metadata.empty()) report << metadata << ',';
          report << "\"pass\":false,\"error\":" << splash::json::quote(error.what())
                 << ",\"completed_cases\":[";
          for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
          report << "]}";writeReport(argv[3], report.str());
        } catch (...) {}
      }
      std::cerr << "flash-dense-small-rows-oracle: " << error.what() << '\n';return 1;
    }
  }
}
