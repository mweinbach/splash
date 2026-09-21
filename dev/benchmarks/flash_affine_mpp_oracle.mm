// Root-run GPU oracle. Building and --cpu-self-test submit no GPU commands.
// Alternate MPP reductions are compared, not labelled bit-identical to addAffine.
#include "flash/FlashAffineMPP.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>

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

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
float number(uint16_t value) {
  return std::bit_cast<float>(uint32_t(value) << 16);
}
uint16_t bits(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  if ((word & 0x7f800000u) == 0x7f800000u)
    return static_cast<uint16_t>((word >> 16) | ((word & 0x7fffffu) ? 0x40u : 0u));
  return static_cast<uint16_t>((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> fallback,
                           uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return std::vector<uint32_t>(fallback);
  std::stringstream source(raw);
  std::vector<uint32_t> result;
  std::string item;
  while (std::getline(source, item, ',')) {
    size_t consumed = 0;
    const unsigned long value = std::stoul(item, &consumed);
    require(consumed == item.size() && value <= maximum,
            std::string("invalid ") + name);
    result.push_back(static_cast<uint32_t>(value));
  }
  require(!result.empty(), std::string("empty ") + name);
  return result;
}

struct Error final {
  uint64_t elements = 0, mismatch = 0, nonfinite = 0;
  uint32_t maximumULP = 0;
  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;
  void add(uint16_t actual, uint16_t expected) {
    ++elements;
    mismatch += actual != expected;
    const float a = number(actual), b = number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; return; }
    const double difference = double(a) - b;
    maximumAbsolute = std::max(maximumAbsolute, std::abs(difference));
    squaredError += difference * difference;
    squaredReference += double(b) * b;
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
    base = backend.allocateBuffer((words + 2 * kGuardWords) * 2,
                                  BufferStorage::Shared, "MPP oracle guarded output");
    view = backend.view(base, kGuardWords * 2, words * 2);
    clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords,
                kSentinel);
  }
  void check(bool finite) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kSentinel && data[kGuardWords + words + i] == kSentinel,
              "MPP output guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])),
                "MPP output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), static_cast<size_t>(words)};
  }
};
template <class T>
MetalBuffer buffer(MetalBackend &backend, const std::vector<T> &values, const char *label) {
  auto result = backend.allocateBuffer(values.size() * sizeof(T), BufferStorage::Shared, label);
  std::memcpy(result.contents(), values.data(), values.size() * sizeof(T));
  return result;
}

struct References final {
  std::vector<uint32_t> columns;
  std::vector<uint16_t> f32, bf16;
  uint64_t negativeScales = 0, coefficients = 0;
  double maximumCoefficientRounding = 0;
};
References reference(const FlashAffineProjection &p, std::span<const uint16_t> input,
                     uint32_t rows, uint32_t count) {
  References result;
  const auto *weights = static_cast<const std::byte *>(p.weights->buffer.contents());
  const auto *scales = static_cast<const std::byte *>(p.scales->buffer.contents());
  const auto *biases = static_cast<const std::byte *>(p.biases->buffer.contents());
  require(weights && scales && biases, "source coefficients are not CPU-visible");
  count = std::min(count, p.outputSize);
  for (uint32_t i = 0; i < count; ++i)
    result.columns.push_back(count == 1 ? 0 : uint64_t(i) * (p.outputSize - 1) / (count - 1));
  std::vector<float> w32(p.inputSize), w16(p.inputSize);
  result.f32.resize(uint64_t(rows) * count);
  result.bf16.resize(uint64_t(rows) * count);
  for (uint32_t c = 0; c < count; ++c) {
    const uint32_t n = result.columns[c];
    const auto packed = std::span<const std::byte>(weights + uint64_t(n) * p.weightRowStrideBytes,
        static_cast<size_t>((uint64_t(p.inputSize) * p.bits + 7) / 8));
    const auto *sf = reinterpret_cast<const uint16_t *>(scales + uint64_t(n) * p.parameterRowStrideBytes);
    const auto *bs = reinterpret_cast<const uint16_t *>(biases + uint64_t(n) * p.parameterRowStrideBytes);
    for (uint32_t g = 0; g < p.inputSize / p.groupSize; ++g) {
      result.negativeScales += number(sf[g]) < 0;
      ++result.coefficients;
    }
    for (uint32_t k = 0; k < p.inputSize; ++k) {
      const uint32_t g = k / p.groupSize;
      w32[k] = float(unpackAffineCode(packed, p.bits, k)) * number(sf[g]) + number(bs[g]);
      w16[k] = number(bits(w32[k]));
      result.maximumCoefficientRounding = std::max(result.maximumCoefficientRounding,
                                                   std::abs(double(w32[k]) - w16[k]));
    }
    for (uint32_t row = 0; row < rows; ++row) {
      // A distinct, serial F32 reduction provides an independent dot-product
      // reference. MPP and SIMD32 reductions may legitimately differ by an ULP.
      float f32 = 0, bf16 = 0;
      for (uint32_t k = 0; k < p.inputSize; ++k) {
        const float x = number(input[uint64_t(row) * p.inputSize + k]);
        f32 = std::fma(x, w32[k], f32);
        bf16 = std::fma(x, w16[k], bf16);
      }
      result.f32[uint64_t(row) * count + c] = bits(f32);
      result.bf16[uint64_t(row) * count + c] = bits(bf16);
    }
  }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const References &ref,
                    std::span<const uint16_t> expected, uint32_t n, uint32_t rows) {
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (size_t c = 0; c < ref.columns.size(); ++c)
      result.add(actual[uint64_t(row) * n + ref.columns[c]], expected[uint64_t(row) * ref.columns.size() + c]);
  return result;
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Times final {
  std::vector<double> gpu, wall;
  void add(CommandTiming timing) { gpu.push_back(timing.gpuSeconds);wall.push_back(timing.wallSeconds); }
  void write(std::ostream &out) const {
    out << "{\"repeats\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_wall_seconds\":" << median(wall) << '}';
  }
};

struct Case final { const char *prefix;uint32_t crop = 0; };
constexpr Case kCases[] = {
  {"language_model.model.layers.1.linear_attn.in_proj_qkv"}, // Q4/G64
  {"language_model.model.layers.0.linear_attn.in_proj_qkv"}, // Q6/G64
  {"language_model.model.layers.0.linear_attn.in_proj_a"}, // Q6/G64, N48 tail
  {"language_model.model.layers.0.linear_attn.out_proj"}, // Q5/G128
  {"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down"}, // Q5/G64
  {"language_model.model.layers.3.self_attn.q_proj"}, // Q4/G64 QSA
  {"language_model.model.layers.43.self_attn.v_proj"}, // Q6/G128
  {"language_model.model.layers.0.mlp.shared_expert.gate_proj"}, // Q8/G128
  {"language_model.model.layers.0.mlp.shared_expert_gate"}, // Q8/G64, N1 tail
  {"language_model.model.layers.15.self_attn.v_proj"}, // Q8/G64
  {"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0", 83}, // Q4/G32
};
constexpr const char *kTileNames[] = {"m8n64", "m16n64", "m16n128", "m32n64", "m32n128"};

void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t word = 0; word < 65536; ++word) {
    const auto original = uint16_t(word);
    if (std::isfinite(number(original))) {
      require(bits(number(original)) == original, "BF16 finite round-trip failed");
      ++checks;
    }
  }
  for (uint32_t width : {4u, 5u, 6u, 8u}) {
    std::vector<std::byte> packed((128 * width + 7) / 8);
    for (uint32_t k = 0; k < 128; ++k) {
      const uint32_t value = uint32_t(randomWord(k + width)) & ((1u << width) - 1);
      for (uint32_t b = 0; b < width; ++b) {
        const uint32_t bit = k * width + b;
        packed[bit / 8] |= std::byte(((value >> b) & 1) << (bit % 8));
      }
    }
    for (uint32_t k = 0; k < 128; ++k) {
      require(unpackAffineCode(packed, width, k) ==
          (uint32_t(randomWord(k + width)) & ((1u << width) - 1)), "packed code oracle failed");
      ++checks;
    }
  }
  require(number(bits(7.0f * -14.25f + 128.0f)) == 28.25f,
          "signed-affine coefficient trap failed");
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks + 1 << "}\n";
}

void validateHost(const FlashAffineProjection &p, MetalBuffer input,
                  MetalBuffer output, MetalBuffer diag) {
  uint32_t rejected = 0;
  auto reject = [&](FlashAffineProjection candidate, MetalBuffer x, MetalBuffer y,
                    uint32_t rows, FlashAffineMPPMode mode, FlashAffineMPPTile tile) {
    bool caught = false;
    try { CommandGraph graph;addAffineMPP(graph, x, candidate, y, diag, rows, mode, tile); }
    catch (const std::invalid_argument &) { caught = true; }
    require(caught, "invalid MPP host parameters accepted");++rejected;
  };
  const auto mode = FlashAffineMPPMode::GroupAffineF32;
  const auto tile = FlashAffineMPPTile::M8N64;
  reject(p, input, output, 0, mode, tile);
  reject(p, input, output, 2049, mode, tile);
  reject(p, {}, output, 1, mode, tile);
  reject(p, input, {}, 1, mode, tile);
  reject(p, input, input, 1, mode, tile);
  auto invalid = p;invalid.bits = 3;reject(invalid, input, output, 1, mode, tile);
  invalid = p;invalid.groupSize = 48;reject(invalid, input, output, 1, mode, tile);
  invalid = p;invalid.experts = 2;reject(invalid, input, output, 1, mode, tile);
  reject(p, input, output, 1, static_cast<FlashAffineMPPMode>(2), tile);
  reject(p, input, output, 1, mode, static_cast<FlashAffineMPPTile>(5));
  require(rejected == 10, "host rejection count inconsistent");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest();return 0; }
      if (argc != 4)
        throw std::invalid_argument("usage: flash-affine-mpp-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const auto rowsList = list("FLASH_MPP_ROWS", {8, 13, 32, 65, 128}, 2048);
      const auto tiles = list("FLASH_MPP_TILES", {0, 1, 2, 3, 4}, 4);
      const auto modes = list("FLASH_MPP_MODES", {0, 1}, 1);
      const auto repeatsList = list("FLASH_MPP_REPEATS", {3}, 20);
      const auto countList = list("FLASH_MPP_CPU_COLUMNS", {19}, 128);
      require(repeatsList.size() == 1 && repeatsList[0] && countList.size() == 1 && countList[0],
              "repeats and CPU columns must be single positive values");
      for (auto rows : rowsList) require(rows, "MPP rows must be positive");
      const uint32_t repeats = repeatsList[0], cpuColumns = countList[0];
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "MPP oracle diagnostics");
      auto *status = static_cast<uint32_t *>(diag.contents());
      std::ofstream out(argv[3]);
      require(bool(out), "cannot open MPP report");
      out << std::setprecision(12) << "{\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"canonical_semantics\":" << splash::json::quote(kFlashAffineSemantics)
          << ",\"cpu_reference\":\"serial-f32-fma-dot-bf16-output; sampled columns\",\"cases\":[";
      bool first = true;
      uint64_t variants = 0;
      const char *filter = std::getenv("FLASH_MPP_PROJECTION");
      for (const auto &test : kCases) {
        if (filter && std::string(test.prefix).find(filter) == std::string::npos) continue;
        auto p = weights.projection(test.prefix);
        const uint32_t sourceN = p.outputSize;
        if (test.crop) p.outputSize = std::min(p.outputSize, test.crop);
        for (uint32_t rows : rowsList) {
          std::vector<uint16_t> hostInput(uint64_t(rows) * p.inputSize);
          for (size_t i = 0; i < hostInput.size(); ++i) {
            const int32_t value = int32_t(randomWord(i + 0xc0ffee) % 2047) - 1023;
            hostInput[i] = bits(float(value) / 1024.0f);
          }
          auto input = buffer(backend, hostInput, "MPP oracle BF16 input");
          Guarded control(backend, uint64_t(rows) * p.outputSize);
          Guarded output(backend, uint64_t(rows) * p.outputSize);
          validateHost(p, input, output.view, diag);
          const auto refs = reference(p, hostInput, rows, cpuColumns);
          *status = kSticky;
          CommandGraph canonical;
          addAffine(canonical, input, p, control.view, diag, rows);
          (void)backend.submitCommand(canonical.dispatches()); // compile/residency warmup
          Times controlTimes;
          for (uint32_t i = 0; i < repeats; ++i)
            controlTimes.add(backend.submitCommand(canonical.dispatches()));
          control.check(true);require(*status == kSticky, "canonical diagnostics changed");
          const auto canonicalCPU = compareSample(control.values(), refs, refs.f32, p.outputSize, rows);
          require(canonicalCPU.relativeL2() < .01 && !canonicalCPU.nonfinite,
                  "canonical control disagrees with independent CPU dot reference");
          if (!first) out << ',';first = false;
          out << "{\"projection\":" << splash::json::quote(test.prefix)
              << ",\"rows\":" << rows << ",\"n\":" << p.outputSize << ",\"source_n\":" << sourceN
              << ",\"k\":" << p.inputSize << ",\"bits\":" << p.bits << ",\"group_size\":" << p.groupSize
              << ",\"sampled_columns\":" << refs.columns.size()
              << ",\"sampled_negative_scales\":" << refs.negativeScales
              << ",\"sampled_scale_count\":" << refs.coefficients
              << ",\"max_coefficient_bf16_rounding\":" << refs.maximumCoefficientRounding
              << ",\"canonical_timing\":";controlTimes.write(out);
          out << ",\"canonical_vs_f32_cpu\":";canonicalCPU.write(out);
          out << ",\"variants\":[";
          bool firstVariant = true;
          for (uint32_t mode : modes) for (uint32_t tile : tiles) {
            output.clear();*status = kSticky;
            CommandGraph graph;
            addAffineMPP(graph, input, p, output.view, diag, rows,
                         static_cast<FlashAffineMPPMode>(mode), static_cast<FlashAffineMPPTile>(tile));
            (void)backend.submitCommand(graph.dispatches()); // compile/residency warmup
            output.check(true);require(*status == kSticky, "MPP finite diagnostics changed");
            const std::vector<uint16_t> firstOutput(output.values().begin(), output.values().end());
            Times times, pairedControlTimes;
            for (uint32_t i = 0; i < repeats; ++i) {
              // Alternate command order within each pair. This keeps each
              // candidate's comparison close in time and balances cache and
              // clock changes instead of reusing an earlier global control.
              if (i % 2 == 0) {
                pairedControlTimes.add(backend.submitCommand(canonical.dispatches()));
                times.add(backend.submitCommand(graph.dispatches()));
              } else {
                times.add(backend.submitCommand(graph.dispatches()));
                pairedControlTimes.add(backend.submitCommand(canonical.dispatches()));
              }
            }
            output.check(true);require(*status == kSticky, "MPP repeat diagnostics changed");
            control.check(true);
            require(compare(output.values(), firstOutput).mismatch == 0, "MPP repeated output changed");
            const auto vsControl = compare(output.values(), control.values());
            const auto vsF32 = compareSample(output.values(), refs, refs.f32, p.outputSize, rows);
            const auto vsBF16 = compareSample(output.values(), refs, refs.bf16, p.outputSize, rows);
            const auto &ownReference = mode == 0 ? vsF32 : vsBF16;
            require(ownReference.relativeL2() < .01 && !ownReference.nonfinite,
                    "MPP mode disagrees with its independent CPU dot reference");
            if (!firstVariant) out << ',';firstVariant = false;
            out << "{\"mode\":" << mode << ",\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"semantics\":" << splash::json::quote(mode == 0 ?
                    kFlashAffineMPPGroupSemantics : kFlashAffineMPPReconstructedSemantics)
                << ",\"timing\":";times.write(out);
            out << ",\"paired_control_timing\":";pairedControlTimes.write(out);
            out << ",\"speedup_gpu\":" << median(pairedControlTimes.gpu) / median(times.gpu)
                << ",\"vs_canonical_gpu\":";vsControl.write(out);
            out << ",\"vs_f32_cpu\":";vsF32.write(out);
            out << ",\"vs_bf16_cpu\":";vsBF16.write(out);out << '}';
            ++variants;
          }
          out << "]}";
          out.flush();
          std::cerr << "mpp projection=" << test.prefix << " rows=" << rows << " variants=" << variants << '\n';
          // Nonfinite arithmetic must set sticky bit 4 without clearing caller bits.
          // Only this fixture's first row changes; source weights remain intact.
          hostInput[0] = 0x7f80;std::memcpy(input.contents(), hostInput.data(), hostInput.size() * 2);
          for (uint32_t mode : modes) {
            output.clear();*status = kSticky;
            CommandGraph bad;
            addAffineMPP(bad, input, p, output.view, diag, rows,
                         static_cast<FlashAffineMPPMode>(mode), FlashAffineMPPTile::M8N64);
            (void)backend.submitCommand(bad.dispatches());
            output.check(false);
            require(*status == (kSticky | kFlashAffineInvalidNumerics),
                    "MPP nonfinite diagnostics missing or not sticky");
          }
        }
      }
      require(variants, "MPP projection filter selected no cases");
      out << "],\"variants\":" << variants << ",\"pass\":true,\"diagnostics_sticky\":true,"
          << "\"finite_and_nonfinite_guards_pass\":true,\"host_rejections_pass\":true}\n";
      require(bool(out), "could not write MPP report");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "flash-affine-mpp-oracle: " << error.what() << '\n';return 1;
    }
  }
}
