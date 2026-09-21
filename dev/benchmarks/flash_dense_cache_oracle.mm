// Root-run GPU oracle. Compilation and --cpu-self-test submit no GPU commands.
// The independent unpacker and BF16 rounding below do not call runtime helpers.
#include "flash/FlashDenseCache.hpp"
#include "engine/Json.hpp"

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
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint64_t kGuardWords = 32;
constexpr std::array<uint32_t, 5> kTileRows{8, 16, 16, 32, 32};
constexpr std::array<const char *, 5> kTileNames{
    "m8n64", "m16n64", "m16n128", "m32n64", "m32n128"};
constexpr std::array<const char *, 5> kDefaultCases{
    "language_model.model.layers.0.mlp.shared_expert.gate_proj", // Q8/G128
    "language_model.model.layers.1.linear_attn.in_proj_qkv", // Q4/G64
    "language_model.model.layers.0.linear_attn.in_proj_qkv", // Q6/G64
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down", // Q5/G64, N320
    "language_model.model.layers.0.linear_attn.out_proj", // Q5/G128
};

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
uint32_t code(const std::byte *row, uint32_t bits, uint32_t k) {
  const uint64_t bit = uint64_t(k) * bits;
  const uint32_t shift = uint32_t(bit % 8);
  uint32_t word = std::to_integer<uint8_t>(row[bit / 8]);
  if (shift + bits > 8)
    word |= uint32_t(std::to_integer<uint8_t>(row[bit / 8 + 1])) << 8;
  return (word >> shift) & ((1u << bits) - 1);
}
uint16_t coefficient(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *packed = static_cast<const std::byte *>(p.weights->buffer.contents()) +
                       uint64_t(n) * p.weightRowStrideBytes;
  const auto *scales = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.scales->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  const auto *biases = reinterpret_cast<const uint16_t *>(
      static_cast<const std::byte *>(p.biases->buffer.contents()) +
      uint64_t(n) * p.parameterRowStrideBytes);
  // Integer x BF16 is exactly representable in F32 for all supported widths.
  // The explicit intermediate prevents compiler contraction changing staging.
  const float product = float(code(packed, p.bits, k)) * number(scales[k / p.groupSize]);
  return bf16(product + number(biases[k / p.groupSize]));
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
std::vector<std::string> cases() {
  std::vector<std::string> result;
  if (const char *raw = std::getenv("FLASH_DENSE_CACHE_PREFIXES")) {
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
      require(!item.empty(), "empty FLASH_DENSE_CACHE_PREFIXES item");
      result.push_back(item);
    }
  } else {
    for (const char *prefix : kDefaultCases) result.emplace_back(prefix);
  }
  if (const char *filter = std::getenv("FLASH_DENSE_CACHE_PROJECTION"))
    std::erase_if(result, [&](const std::string &prefix) {
      return prefix.find(filter) == std::string::npos;
    });
  require(!result.empty(), "dense cache filter selected no cases");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

struct SHA256 final {
  CC_SHA256_CTX context{};
  SHA256() { CC_SHA256_Init(&context); }
  void add(const void *data, uint64_t bytes) {
    const auto *next = static_cast<const std::byte *>(data);
    while (bytes) {
      const auto amount = CC_LONG(std::min<uint64_t>(bytes, std::numeric_limits<CC_LONG>::max()));
      CC_SHA256_Update(&context, next, amount);
      next += amount;bytes -= amount;
    }
  }
  std::string finish() {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream out;
    for (auto b : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(b);
    return out.str();
  }
};
struct Coefficients final {
  uint64_t elements = 0, negativeScales = 0, scaleCount = 0;
  std::string expectedSHA, actualSHA;
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":0,\"negative_scales\":"
        << negativeScales << ",\"scale_count\":" << scaleCount
        << ",\"expected_sha256\":" << splash::json::quote(expectedSHA)
        << ",\"actual_sha256\":" << splash::json::quote(actualSHA) << '}';
  }
};
Coefficients checkCoefficients(const FlashAffineProjection &p, const FlashTensor &cache) {
  require(p.weights && p.scales && p.biases && p.experts == 1,
          "coefficient oracle requires a dense affine source");
  require(p.weights->buffer.contents() && p.scales->buffer.contents() && p.biases->buffer.contents(),
          "source coefficient buffers are not CPU-visible");
  require(cache.dtype == FlashDType::BF16 && cache.shape ==
          std::vector<uint64_t>{p.outputSize, p.inputSize} &&
          cache.logicalBytes == uint64_t(p.outputSize) * p.inputSize * 2 &&
          cache.buffer.sizeBytes() >= cache.logicalBytes && cache.buffer.contents(),
          "cached coefficient shape, extent or CPU visibility is invalid");
  const auto *actual = static_cast<const uint16_t *>(cache.buffer.contents());
  std::vector<uint16_t> expectedRow(p.inputSize);
  SHA256 expectedHash, actualHash;
  Coefficients result;
  for (uint32_t n = 0; n < p.outputSize; ++n) {
    const auto *scales = reinterpret_cast<const uint16_t *>(
        static_cast<const std::byte *>(p.scales->buffer.contents()) +
        uint64_t(n) * p.parameterRowStrideBytes);
    for (uint32_t g = 0; g < p.inputSize / p.groupSize; ++g) {
      result.negativeScales += number(scales[g]) < 0;
      ++result.scaleCount;
    }
    for (uint32_t k = 0; k < p.inputSize; ++k) {
      expectedRow[k] = coefficient(p, n, k);
      require(std::isfinite(number(expectedRow[k])), "source reconstruction is nonfinite");
      if (actual[uint64_t(n) * p.inputSize + k] != expectedRow[k])
        throw std::runtime_error("cached coefficient mismatch at n=" + std::to_string(n) +
                                 " k=" + std::to_string(k));
      ++result.elements;
    }
    expectedHash.add(expectedRow.data(), uint64_t(p.inputSize) * 2);
    actualHash.add(actual + uint64_t(n) * p.inputSize, uint64_t(p.inputSize) * 2);
  }
  result.expectedSHA = expectedHash.finish();result.actualSHA = actualHash.finish();
  require(result.expectedSHA == result.actualSHA, "cached coefficient SHA mismatch");
  return result;
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
  for (size_t i = 0; i < actual.size(); ++i) result.add(actual[i], expected[i]);
  return result;
}

struct Guarded final {
  MetalBuffer base, view;
  uint64_t words;
  Guarded(MetalBackend &backend, uint64_t elements) : words(elements) {
    base = backend.allocateBuffer((words + 2 * kGuardWords) * 2, BufferStorage::Shared,
                                  "dense cache oracle guarded output");
    view = backend.view(base, kGuardWords * 2, words * 2);clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords, kSentinel);
  }
  void check(bool finite) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kSentinel && data[kGuardWords + words + i] == kSentinel,
              "dense cache output guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])),
                "dense cache output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};
struct References final {
  std::vector<uint32_t> columns;
  std::vector<uint16_t> values;
};
References reference(const FlashAffineProjection &p, std::span<const uint16_t> input,
                     uint32_t rows, uint32_t count) {
  References result;
  count = std::min(count, p.outputSize);
  for (uint32_t c = 0; c < count; ++c)
    result.columns.push_back(count == 1 ? 0 : uint64_t(c) * (p.outputSize - 1) / (count - 1));
  result.values.resize(uint64_t(rows) * count);
  std::vector<float> decoded(p.inputSize);
  for (uint32_t c = 0; c < count; ++c) {
    for (uint32_t k = 0; k < p.inputSize; ++k) decoded[k] = number(coefficient(p, result.columns[c], k));
    for (uint32_t row = 0; row < rows; ++row) {
      float sum = 0;
      for (uint32_t k = 0; k < p.inputSize; ++k)
        sum = std::fma(number(input[uint64_t(row) * p.inputSize + k]), decoded[k], sum);
      result.values[uint64_t(row) * count + c] = bf16(sum);
    }
  }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const References &expected,
                    uint32_t n, uint32_t rows) {
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (size_t c = 0; c < expected.columns.size(); ++c)
      result.add(actual[uint64_t(row) * n + expected.columns[c]],
                 expected.values[uint64_t(row) * expected.columns.size() + c]);
  return result;
}
double median(std::vector<double> values) {
  require(!values.empty(), "timing sample list empty");
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Times final {
  std::vector<double> gpu, wall;
  void add(CommandTiming timing) {
    require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 0 &&
            std::isfinite(timing.wallSeconds) && timing.wallSeconds > 0,
            "invalid timing sample");
    gpu.push_back(timing.gpuSeconds);wall.push_back(timing.wallSeconds);
  }
  void write(std::ostream &out) const {
    out << "{\"repeats\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_wall_seconds\":" << median(wall) << '}';
  }
};

uint32_t validateHost(MetalBackend &backend, const FlashDenseCache &cache, const std::string &prefix,
                      MetalBuffer input, MetalBuffer output, MetalBuffer diagnostics) {
  uint32_t rejected = 0;
  const auto tile = FlashAffineMPPTile::M16N128;
  auto reject = [&](std::string_view name, MetalBuffer x, MetalBuffer y, MetalBuffer d,
                    uint32_t rows, FlashAffineMPPTile selectedTile) {
    bool caught = false;
    try { CommandGraph graph;cache.addProjection(graph, name, x, y, d, rows, selectedTile); }
    catch (const std::invalid_argument &) { caught = true; }
    catch (const std::out_of_range &) { caught = true; }
    require(caught, "invalid dense cache host parameters accepted");++rejected;
  };
  reject(prefix, input, output, diagnostics, 0, tile);
  reject(prefix, input, output, diagnostics, 2049, tile);
  reject(prefix, {}, output, diagnostics, 1, tile);
  reject(prefix, input, {}, diagnostics, 1, tile);
  reject(prefix, input, output, {}, 1, tile);
  reject(prefix, input, input, diagnostics, 1, tile);
  reject(prefix, input, output, diagnostics, 1, static_cast<FlashAffineMPPTile>(5));
  reject("oracle-missing-prefix", input, output, diagnostics, 1, tile);
  reject(prefix, diagnostics, output, diagnostics, 1, tile);
  reject(prefix, input, diagnostics, diagnostics, 1, tile);
  const auto &weight = cache.tensor(prefix).buffer;
  reject(prefix, weight, output, diagnostics, 1, tile);
  reject(prefix, input, weight, diagnostics, 1, tile);
  reject(prefix, input, output, backend.view(input, 0, sizeof(uint32_t)), 1, tile);
  reject(prefix, input, output, backend.view(output, 0, sizeof(uint32_t)), 1, tile);
  const uint64_t outputRowBytes = cache.tensor(prefix).shape[0] * 2;
  if (input.sizeBytes() >= outputRowBytes + 2)
    reject(prefix, input, backend.view(input, 2, outputRowBytes), diagnostics, 1, tile);
  require(!cache.contains("oracle-missing-prefix"), "contains accepted absent matrix");
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
  require(number(bf16(7.0f * -14.25f + 128.0f)) == 28.25f,
          "signed-affine coefficient trap failed");++checks;
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82,
          "BF16 round-to-nearest-even ties failed");checks += 2;
  SHA256 sha;sha.add("abc", 3);
  require(sha.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "coefficient hash oracle failed");++checks;
  Error exact;exact.add(0x8000, 0);require(exact.maximumULP == 0, "signed-zero ULP mapping failed");++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
            << ",\"gpu_commands\":0}\n";
}

void writeReport(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path);
  require(bool(out), "cannot open dense cache report");
  out << text << '\n';out.close();
  require(bool(out), "cannot write dense cache report");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string reportMetadata;
    std::vector<std::string> records;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") {
        cpuSelfTest();return 0;
      }
      if (argc != 4)
        throw std::invalid_argument("usage: flash-dense-cache-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const auto prefixes = cases();
      const auto rowsList = list("FLASH_DENSE_CACHE_ROWS", {32, 128, 17}, 2048);
      const auto tiles = list("FLASH_DENSE_CACHE_TILES", {1, 2, 4}, 4);
      const auto repeatsList = list("FLASH_DENSE_CACHE_REPEATS", {3}, 20);
      const auto columnsList = list("FLASH_DENSE_CACHE_CPU_COLUMNS", {19}, 128);
      require(repeatsList.size() == 1 && repeatsList[0] &&
              columnsList.size() == 1 && columnsList[0],
              "repeats and CPU columns must each be one positive value");
      for (uint32_t rows : rowsList) require(rows > 0, "dense cache rows must be positive");
      const uint32_t repeats = repeatsList[0], cpuColumns = columnsList[0];
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared,
                                         "dense cache oracle diagnostics");
      auto *status = static_cast<uint32_t *>(diag.contents());
      const auto defaults = FlashDenseCache::defaultPrefixes(weights);
      std::set<std::string> uniqueDefaults(defaults.begin(), defaults.end());
      require(uniqueDefaults.size() == defaults.size(), "default cache list contains duplicates");
      for (const auto &prefix : defaults) {
        require(prefix.find(".experts.") == std::string::npos &&
                prefix.find(".shards.") == std::string::npos &&
                prefix.find("embed_tokens") == std::string::npos &&
                prefix.find("visual.") == std::string::npos,
                "default cache list includes an excluded matrix");
        require(weights.projection(prefix).experts == 1, "default cache list includes experts");
      }
      std::ostringstream metadata;
      metadata << "\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"operand_format\":" << splash::json::quote(kFlashDenseCacheOperandFormat)
          << ",\"execution_semantics\":" << splash::json::quote(kFlashDenseCacheExecutionSemantics)
          << ",\"onfly_control_semantics\":" << splash::json::quote(kFlashAffineMPPReconstructedSemantics)
          << ",\"cpu_reference\":\"independent-byte-unpack; bf16-reconstruction; serial-f32-fma-dot; sampled columns\""
          << ",\"default_matrix_count\":" << defaults.size()
          << ",\"default_planned_bytes\":" << FlashDenseCache::plannedBytes(weights, defaults)
          << ",\"shader_validation_environment\":"
          << splash::json::quote(std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset");
      reportMetadata = metadata.str();
      uint64_t variants = 0, coefficientChecks = 0;
      for (const auto &prefix : prefixes) {
        const auto &p = weights.projection(prefix);
        const std::vector<std::string> selected{prefix};
        const uint64_t planned = FlashDenseCache::plannedBytes(weights, selected);
        const uint64_t before = backend.memoryStats().allocatedBytes;
        FlashDenseCache cache(backend, weights, selected); // only root runs this path
        const uint64_t after = backend.memoryStats().allocatedBytes;
        require(after >= before && after - before == cache.actualAllocatedBytes(),
                "cache allocation ledger delta differs from reported bytes");
        require(cache.actualAllocatedBytes() <= planned &&
                cache.actualAllocatedBytes() >= uint64_t(p.outputSize) * p.inputSize * 2,
                "cache allocation is outside its planned extent");
        require(cache.contains(prefix) && cache.identitySha256().size() == 64 &&
                cache.immutableWeightBuffers().size() == 1, "cache identity or published buffers invalid");
        const auto coeffs = checkCoefficients(p, cache.tensor(prefix));
        coefficientChecks += coeffs.elements;
        for (uint32_t rows : rowsList) {
          std::vector<uint16_t> hostInput(uint64_t(rows) * p.inputSize);
          for (size_t i = 0; i < hostInput.size(); ++i) {
            const int32_t value = int32_t(randomWord(i + 0xc0ffee) % 2047) - 1023;
            hostInput[i] = bf16(float(value) / 1024.0f);
          }
          auto input = backend.allocateBuffer(hostInput.size() * 2, BufferStorage::Shared,
                                               "dense cache oracle BF16 input");
          std::memcpy(input.contents(), hostInput.data(), hostInput.size() * 2);
          Guarded control(backend, uint64_t(rows) * p.outputSize);
          Guarded output(backend, uint64_t(rows) * p.outputSize);
          Guarded vectorOutput(backend, uint64_t(rows) * p.outputSize);
          const uint32_t hostRejections = validateHost(backend, cache, prefix, input, output.view, diag);
          const auto ref = reference(p, hostInput, rows, cpuColumns);
          *status = kSticky;
          CommandGraph vectorGraph;
          addDenseBF16(vectorGraph, input, cache.tensor(prefix), vectorOutput.view, diag, rows);
          (void)backend.submitCommand(vectorGraph.dispatches());
          vectorOutput.check(true);require(*status == kSticky, "BF16 vector diagnostic changed");
          const auto vectorCPU = compareSample(vectorOutput.values(), ref, p.outputSize, rows);
          require(vectorCPU.relativeL2() < .006 && !vectorCPU.nonfinite,
                  "BF16 vector control disagrees with independent CPU dot reference");
          std::ostringstream record;
          record << std::setprecision(12) << "{\"projection\":" << splash::json::quote(prefix)
              << ",\"rows\":" << rows << ",\"n\":" << p.outputSize << ",\"k\":" << p.inputSize
              << ",\"bits\":" << p.bits << ",\"group_size\":" << p.groupSize
              << ",\"planned_bytes\":" << planned << ",\"actual_allocated_bytes\":" << cache.actualAllocatedBytes()
              << ",\"cache_identity_sha256\":" << splash::json::quote(cache.identitySha256())
              << ",\"initialization_timing\":{\"gpu_seconds\":" << cache.initializationTiming().gpuSeconds
              << ",\"wall_seconds\":" << cache.initializationTiming().wallSeconds << '}'
              << ",\"coefficients\":";coeffs.write(record);
          record << ",\"sampled_columns\":" << ref.columns.size()
              << ",\"host_rejections\":" << hostRejections << ",\"vector_vs_bf16_cpu\":";
          vectorCPU.write(record);record << ",\"variants\":[";
          bool firstVariant = true;
          for (uint32_t tile : tiles) {
            const auto selectedTile = static_cast<FlashAffineMPPTile>(tile);
            control.clear();output.clear();*status = kSticky;
            CommandGraph canonical, cached;
            addAffineMPP(canonical, input, p, control.view, diag, rows,
                         FlashAffineMPPMode::ReconstructedBF16, selectedTile);
            cache.addProjection(cached, prefix, input, output.view, diag, rows, selectedTile);
            (void)backend.submitCommand(canonical.dispatches());
            (void)backend.submitCommand(cached.dispatches());
            control.check(true);output.check(true);
            require(*status == kSticky, "finite cached/control diagnostics changed");
            const std::vector<uint16_t> firstOutput(output.values().begin(), output.values().end());
            Times cachedTimes, controlTimes;
            for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
              if (repeat % 2 == 0) {
                controlTimes.add(backend.submitCommand(canonical.dispatches()));
                cachedTimes.add(backend.submitCommand(cached.dispatches()));
              } else {
                cachedTimes.add(backend.submitCommand(cached.dispatches()));
                controlTimes.add(backend.submitCommand(canonical.dispatches()));
              }
            }
            control.check(true);output.check(true);
            require(*status == kSticky, "repeat cached/control diagnostics changed");
            require(compare(output.values(), firstOutput).mismatch == 0, "cached repeated output changed");
            const auto vsControl = compare(output.values(), control.values());
            const auto vsCPU = compareSample(output.values(), ref, p.outputSize, rows);
            const auto controlCPU = compareSample(control.values(), ref, p.outputSize, rows);
            require(vsCPU.relativeL2() < .006 && controlCPU.relativeL2() < .006 &&
                    vsControl.relativeL2() < .006 && !vsCPU.nonfinite && !controlCPU.nonfinite &&
                    !vsControl.nonfinite, "whole-K cache differs materially from BF16 operand references");
            const uint32_t firstTail = rows / kTileRows[tile] * kTileRows[tile];
            const auto tail = compare(output.values().subspan(uint64_t(firstTail) * p.outputSize),
                vectorOutput.values().subspan(uint64_t(firstTail) * p.outputSize));
            require(tail.mismatch == 0, "short row fallback differs from BF16 vector output bits");
            if (!firstVariant) record << ',';firstVariant = false;
            record << "{\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"row_tail_rows\":" << rows - firstTail
                << ",\"n128_column_tail\":" << ((tile == 2 || tile == 4) ? p.outputSize % 128 : 0)
                << ",\"cached_dispatches\":" << cached.dispatches().size()
                << ",\"cached_timing\":";cachedTimes.write(record);
            record << ",\"onfly_paired_timing\":";controlTimes.write(record);
            record << ",\"speedup_gpu\":" << median(controlTimes.gpu) / median(cachedTimes.gpu)
                << ",\"vs_onfly_bf16_gpu\":";vsControl.write(record);
            record << ",\"vs_bf16_cpu\":";vsCPU.write(record);
            record << ",\"onfly_vs_bf16_cpu\":";controlCPU.write(record);
            record << ",\"short_rows_vs_bf16_vector\":";tail.write(record);record << '}';
            ++variants;
            // Infinity must preserve caller diagnostic bits and all guards.
            const uint16_t original = hostInput.front();
            static_cast<uint16_t *>(input.contents())[0] = 0x7f80;
            output.clear();*status = kSticky;
            (void)backend.submitCommand(cached.dispatches());
            output.check(false);
            require(*status == (kSticky | kFlashAffineInvalidNumerics),
                    "cached nonfinite diagnostics missing or not sticky");
            static_cast<uint16_t *>(input.contents())[0] = original;
          }
          record << "]}";records.push_back(record.str());
          std::cerr << "dense cache projection=" << prefix << " rows=" << rows
                    << " variants=" << variants << '\n';
        }
      }
      require(variants > 0, "dense cache oracle produced no variants");
      std::ostringstream report;
      report << '{' << reportMetadata << ",\"cases\":[";
      for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
      report << "],\"variants\":" << variants << ",\"coefficient_bit_checks\":" << coefficientChecks
          << ",\"pass\":true,\"diagnostics_sticky\":true,\"finite_and_nonfinite_guards_pass\":true,"
          << "\"host_rejections_pass\":true,\"original_models_modified\":false}";
      writeReport(argv[3], report.str());return 0;
    } catch (const std::exception &error) {
      if (argc == 4) {
        try {
          std::ostringstream report;report << '{';
          if (!reportMetadata.empty()) report << reportMetadata << ',';
          report << "\"pass\":false,\"error\":" << splash::json::quote(error.what())
                 << ",\"completed_cases\":[";
          for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
          report << "]}";writeReport(argv[3], report.str());
        } catch (...) {}
      }
      std::cerr << "flash-dense-cache-oracle: " << error.what() << '\n';return 1;
    }
  }
}
