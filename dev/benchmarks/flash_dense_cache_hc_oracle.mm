// Root-run GPU oracle. Compilation and --cpu-self-test submit no GPU commands.
// HC fusion must be bit-identical to the qualified projection+HC path. Whole-K
// routers are compared numerically because F32 reduction orders can differ.
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashHC.hpp"
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
constexpr std::array<const char *, 4> kDefaultPrefixes{
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.1.attn_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.1.mlp_hyper_connection.input_mix_weight_up",
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
std::vector<std::string> prefixes() {
  std::vector<std::string> result;
  if (const char *raw = std::getenv("FLASH_DENSE_HC_PREFIXES")) {
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
      require(!item.empty(), "empty FLASH_DENSE_HC_PREFIXES item");result.push_back(item);
    }
  } else {
    for (const char *prefix : kDefaultPrefixes) result.emplace_back(prefix);
  }
  if (const char *filter = std::getenv("FLASH_DENSE_HC_PROJECTION"))
    std::erase_if(result, [&](const std::string &prefix) {
      return prefix.find(filter) == std::string::npos;
    });
  require(!result.empty(), "HC prefix filter selected no cases");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());return result;
}
std::string sha256(std::span<const uint16_t> values) {
  require(values.size_bytes() <= std::numeric_limits<CC_LONG>::max(), "SHA256 input too large");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(values.data(), CC_LONG(values.size_bytes()), digest.data());
  std::ostringstream out;
  for (auto value : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
  return out.str();
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
                                  "HC dense oracle guarded buffer");
    view = backend.view(base, kGuardWords * 2, words * 2);clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords, kSentinel);
  }
  void load(std::span<const uint16_t> values) {
    require(values.size() == words, "guarded input size differs");
    std::memcpy(view.contents(), values.data(), values.size_bytes());
  }
  void check(bool finite = true) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kSentinel && data[kGuardWords + words + i] == kSentinel,
              "HC dense oracle guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        require(std::isfinite(number(data[kGuardWords + i])), "output unwritten or nonfinite");
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};
double median(std::vector<double> values) {
  require(!values.empty(), "timing samples empty");
  std::sort(values.begin(), values.end());
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
        << ",\"median_wall_seconds\":" << median(wall) << '}';
  }
};
void paired(MetalBackend &backend, const CommandGraph &control, const CommandGraph &candidate,
            uint32_t repeats, Times &controlTimes, Times &candidateTimes) {
  for (uint32_t i = 0; i < repeats; ++i) {
    if (i % 2 == 0) {
      controlTimes.add(backend.submitCommand(control.dispatches()));
      candidateTimes.add(backend.submitCommand(candidate.dispatches()));
    } else {
      candidateTimes.add(backend.submitCommand(candidate.dispatches()));
      controlTimes.add(backend.submitCommand(control.dispatches()));
    }
  }
}
std::vector<uint16_t> randomInput(uint64_t elements, uint64_t seed, float scale) {
  std::vector<uint16_t> result(elements);
  for (size_t i = 0; i < result.size(); ++i) {
    const int32_t value = int32_t(randomWord(i + seed) % 2047) - 1023;
    result[i] = bf16(float(value) * scale / 1024.0f);
  }
  return result;
}
uint32_t validateHC(MetalBackend &backend, const FlashDenseCache &cache, const std::string &prefix,
                    MetalBuffer down, MetalBuffer normalized, MetalBuffer output, MetalBuffer diag) {
  uint32_t rejected = 0;
  const auto tile = FlashAffineMPPTile::M16N64;
  const auto reject = [&](std::string_view name, MetalBuffer a, MetalBuffer h, MetalBuffer y,
                          MetalBuffer d, uint32_t rows, FlashAffineMPPTile t) {
    bool caught = false;
    try { CommandGraph graph;cache.addHCUpMix(graph, name, a, h, y, d, rows, t); }
    catch (const std::invalid_argument &) { caught = true; }
    require(caught, "invalid HC fused host parameters accepted");++rejected;
  };
  for (uint32_t rows : {0u, 16u, 31u, 33u, 2049u}) reject(prefix, down, normalized, output, diag, rows, tile);
  reject(prefix, down, normalized, output, diag, 32, static_cast<FlashAffineMPPTile>(5));
  reject("oracle-missing-prefix", down, normalized, output, diag, 32, tile);
  reject(prefix, {}, normalized, output, diag, 32, tile);
  reject(prefix, down, {}, output, diag, 32, tile);
  reject(prefix, down, normalized, {}, diag, 32, tile);
  reject(prefix, down, normalized, output, {}, 32, tile);
  reject(prefix, diag, normalized, output, diag, 32, tile);
  reject(prefix, down, diag, output, diag, 32, tile);
  reject(prefix, down, normalized, diag, diag, 32, tile);
  reject(prefix, down, normalized, normalized, diag, 32, tile);
  reject(prefix, down, normalized, output, backend.view(down, 0, 4), 32, tile);
  reject(prefix, down, normalized, output, backend.view(normalized, 0, 4), 32, tile);
  reject(prefix, down, normalized, output, backend.view(output, 0, 4), 32, tile);
  reject(prefix, down, normalized, backend.view(normalized, 2, uint64_t(32) * 2560 * 2), diag, 32, tile);
  const auto weight = cache.tensor(prefix).buffer;
  reject(prefix, weight, normalized, output, diag, 32, tile);
  reject(prefix, down, weight, output, diag, 32, tile);
  reject(prefix, down, normalized, weight, diag, 32, tile);
  reject(prefix, down, normalized, output, backend.view(weight, 0, 4), 32, tile);
  for (uint32_t badRows : {0u, 16u, 31u, 33u, 2049u})
    require(!cache.supportsHCUpMix(prefix, badRows, tile), "supportsHCUpMix accepted invalid rows");
  require(!cache.supportsHCUpMix("oracle-missing-prefix", 32, tile) &&
          !cache.supportsHCUpMix(prefix, 32, static_cast<FlashAffineMPPTile>(5)),
          "supportsHCUpMix accepted absent prefix or invalid tile");
  return rejected;
}

struct Reference final {
  std::vector<uint32_t> columns;
  std::vector<uint16_t> values;
};
Reference routerReference(const FlashTensor &weight, std::span<const uint16_t> input,
                          uint32_t rows, uint32_t count) {
  require(weight.dtype == FlashDType::BF16 && weight.shape.size() == 2 && weight.buffer.contents(),
          "router CPU reference requires CPU-visible BF16 weights");
  const uint32_t n = uint32_t(weight.shape[0]), k = uint32_t(weight.shape[1]);
  const auto *w = static_cast<const uint16_t *>(weight.buffer.contents());
  Reference result;count = std::min(count, n);
  for (uint32_t c = 0; c < count; ++c) result.columns.push_back(count == 1 ? 0 : uint64_t(c) * (n - 1) / (count - 1));
  result.values.resize(uint64_t(rows) * count);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t c = 0; c < count; ++c) {
      float sum = 0;
      for (uint32_t column = 0; column < k; ++column)
        sum = std::fma(number(input[uint64_t(row) * k + column]),
                       number(w[uint64_t(result.columns[c]) * k + column]), sum);
      result.values[uint64_t(row) * count + c] = bf16(sum);
    }
  return result;
}
Error compareSample(std::span<const uint16_t> actual, const Reference &expected, uint32_t n, uint32_t rows) {
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (size_t c = 0; c < expected.columns.size(); ++c)
      result.add(actual[uint64_t(row) * n + expected.columns[c]], expected.values[uint64_t(row) * expected.columns.size() + c]);
  return result;
}
uint32_t validateRouter(MetalBackend &backend, const FlashTensor &weight, MetalBuffer input,
                        MetalBuffer output, MetalBuffer diag) {
  uint32_t rejected = 0;
  const auto tile = FlashAffineMPPTile::M16N64;
  const auto reject = [&](FlashTensor w, MetalBuffer x, MetalBuffer y, MetalBuffer d,
                          uint32_t rows, FlashAffineMPPTile t) {
    bool caught = false;
    try { CommandGraph graph;addDenseBF16WholeK(backend, graph, x, w, y, d, rows, t); }
    catch (const std::invalid_argument &) { caught = true; }
    require(caught, "invalid whole-K router parameters accepted");++rejected;
  };
  reject(weight, input, output, diag, 0, tile);
  reject(weight, input, output, diag, 2049, tile);
  reject(weight, input, output, diag, 1, static_cast<FlashAffineMPPTile>(5));
  reject(weight, {}, output, diag, 1, tile);
  reject(weight, input, {}, diag, 1, tile);
  reject(weight, input, output, {}, 1, tile);
  reject(weight, diag, output, diag, 1, tile);
  reject(weight, input, diag, diag, 1, tile);
  reject(weight, input, input, diag, 1, tile);
  reject(weight, weight.buffer, output, diag, 1, tile);
  reject(weight, input, weight.buffer, diag, 1, tile);
  reject(weight, input, output, backend.view(input, 0, 4), 1, tile);
  reject(weight, input, output, backend.view(output, 0, 4), 1, tile);
  reject(weight, input, backend.view(input, 2, weight.shape[0] * 2), diag, 1, tile);
  auto invalid = weight;invalid.dtype = FlashDType::F32;reject(invalid, input, output, diag, 1, tile);
  invalid = weight;invalid.shape[0] = 511;reject(invalid, input, output, diag, 1, tile);
  invalid = weight;invalid.shape[1] = 2559;reject(invalid, input, output, diag, 1, tile);
  invalid = weight;invalid.logicalBytes = 1;reject(invalid, input, output, diag, 1, tile);
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
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82,
          "BF16 nearest-even rounding failed");checks += 2;
  for (uint16_t value : {uint16_t(1), uint16_t(0x7f), uint16_t(0x8001), uint16_t(0x807f)}) {
    require(std::isfinite(number(value)) && number(value) != 0 &&
            bf16(number(value) * 1.0f) == value, "BF16 subnormal fixture lost before GPU submission");++checks;
  }
  for (uint32_t rowCount : {32u, 128u, 256u, 2048u})
    for (uint32_t tileRows : kTileRows) { require(rowCount % tileRows == 0, "HC row fixture is incomplete");++checks; }
  Error exact;exact.add(0x8000, 0);require(exact.maximumULP == 0, "signed-zero ULP mapping failed");++checks;
  const std::array<uint16_t, 2> vector{0x3f80, 0xbf80};
  require(compare(vector, vector).mismatch == 0 && sha256(vector).size() == 64, "output fingerprint self-check failed");++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}
void writeReport(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path);require(bool(out), "cannot open HC cache report");
  out << text << '\n';out.close();require(bool(out), "cannot write HC cache report");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string reportMetadata;
    std::vector<std::string> records;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest();return 0; }
      if (argc != 4) throw std::invalid_argument("usage: flash-dense-cache-hc-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const auto rowsList = list("FLASH_DENSE_HC_ROWS", {32, 128}, 2048);
      const auto routerRows = list("FLASH_DENSE_ROUTER_ROWS", {32, 128}, 2048);
      const auto tiles = list("FLASH_DENSE_HC_TILES", {1}, 4);
      const auto repeatsList = list("FLASH_DENSE_HC_REPEATS", {3}, 20);
      const auto columnsList = list("FLASH_DENSE_ROUTER_CPU_COLUMNS", {19}, 128);
      require(repeatsList.size() == 1 && repeatsList[0] && columnsList.size() == 1 && columnsList[0],
              "repeats and CPU columns must each be one positive value");
      const uint32_t repeats = repeatsList[0];
      const std::string only = std::getenv("FLASH_DENSE_HC_ONLY") ? std::getenv("FLASH_DENSE_HC_ONLY") : "both";
      require(only == "both" || only == "hc" || only == "router", "FLASH_DENSE_HC_ONLY must be both, hc or router");
      MetalBackend backend(argv[1]);auto weights = FlashWeights::load(backend, argv[2]);
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "HC dense oracle sticky diagnostics");
      auto *status = static_cast<uint32_t *>(diag.contents());
      std::ostringstream metadata;
      metadata << "\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"hc_semantics\":" << splash::json::quote(kFlashDenseCacheHCUpMixSemantics)
          << ",\"router_semantics\":" << splash::json::quote(kFlashDenseCacheExecutionSemantics)
          << ",\"shader_validation_environment\":"
          << splash::json::quote(std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset");
      reportMetadata = metadata.str();
      uint64_t exactHCElements = 0, hcVariants = 0, routerVariants = 0;
      if (only != "router") for (const auto &prefix : prefixes()) {
        const std::vector<std::string> selected{prefix};FlashDenseCache cache(backend, weights, selected);
        const auto &p = weights.projection(prefix);
        require(cache.tensor(prefix).shape == std::vector<uint64_t>{10240, 320}, "HC oracle matrix has unexpected dimensions");
        for (uint32_t rows : rowsList) {
          require(rows >= 32, "HC row count must be at least32");
          auto hostDown = randomInput(uint64_t(rows) * 320, 0xc0ffee, 3.0f);
          auto hostHyper = randomInput(uint64_t(rows) * 10240, 0xbadc0ffe, 4.0f);
          // Zero, saturated gates and signed BF16 subnormal products in each
          // stream supplement the realistic finite activation distribution.
          std::fill_n(hostDown.begin(), 320, uint16_t(0));
          for (uint32_t k = 0; k < 320; ++k) hostDown[320 + k] = bf16((k & 1) ? 48.0f : -48.0f);
          constexpr std::array<uint16_t, 8> specials{1, 0x7f, 0x8001, 0x807f, 0, 0x8000, 0x0080, 0x8080};
          for (uint32_t stream = 0; stream < 4; ++stream)
            for (uint32_t k = 0; k < 64; ++k) hostHyper[uint64_t(stream) * 2560 + k] = specials[k % specials.size()];
          Guarded down(backend, hostDown.size()), normalized(backend, hostHyper.size());
          Guarded raw(backend, uint64_t(rows) * 10240), control(backend, uint64_t(rows) * 2560), output(backend, uint64_t(rows) * 2560);
          down.load(hostDown);normalized.load(hostHyper);
          const uint32_t hostRejections = validateHC(backend, cache, prefix, down.view, normalized.view, output.view, diag);
          for (uint32_t tile : tiles) {
            const auto selectedTile = static_cast<FlashAffineMPPTile>(tile);
            require(cache.supportsHCUpMix(prefix, rows, selectedTile), "requested HC row/tile combination unsupported");
            control.clear();output.clear();raw.clear();*status = kSticky;
            CommandGraph separate, fused;
            cache.addProjection(separate, prefix, down.view, raw.view, diag, rows, selectedTile);
            addHCMix(separate, normalized.view, raw.view, control.view, {rows, 2560, 4, 1e-6f});
            cache.addHCUpMix(fused, prefix, down.view, normalized.view, output.view, diag, rows, selectedTile);
            (void)backend.submitCommand(separate.dispatches());(void)backend.submitCommand(fused.dispatches());
            raw.check();control.check();output.check();down.check();normalized.check();
            require(*status == kSticky, "finite HC diagnostic changed");
            const auto warm = compare(output.values(), control.values());
            require(warm.mismatch == 0, "fused HC output is not bit-exact to same-tile projection+HC control");
            const std::vector<uint16_t> firstOutput(output.values().begin(), output.values().end());
            Times controlTimes, fusedTimes;paired(backend, separate, fused, repeats, controlTimes, fusedTimes);
            control.check();output.check();raw.check();down.check();normalized.check();
            require(*status == kSticky && compare(output.values(), control.values()).mismatch == 0 &&
                    compare(output.values(), firstOutput).mismatch == 0, "fused HC repeat output/diagnostic changed");
            require(compare(down.values(), hostDown).mismatch == 0 && compare(normalized.values(), hostHyper).mismatch == 0,
                    "HC fusion mutated activation inputs");
            uint32_t badPhases = 0;
            for (auto bad : {down.view, normalized.view}) {
              const uint16_t saved = static_cast<uint16_t *>(bad.contents())[0];
              static_cast<uint16_t *>(bad.contents())[0] = 0x7f80;output.clear();*status = kSticky;
              (void)backend.submitCommand(fused.dispatches());output.check(false);down.check(false);normalized.check(false);
              require(*status == (kSticky | kFlashAffineInvalidNumerics), "HC nonfinite diagnostic missing or not sticky");
              static_cast<uint16_t *>(bad.contents())[0] = saved;++badPhases;
            }
            std::ostringstream record;
            record << std::setprecision(12) << "{\"kind\":\"hc-up-mix\",\"projection\":" << splash::json::quote(prefix)
                << ",\"rows\":" << rows << ",\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"source_bits\":" << p.bits << ",\"source_group_size\":" << p.groupSize
                << ",\"cache_identity_sha256\":" << splash::json::quote(cache.identitySha256())
                << ",\"cache_allocated_bytes\":" << cache.actualAllocatedBytes()
                << ",\"control_dispatches\":" << separate.dispatches().size()
                << ",\"fused_dispatches\":" << fused.dispatches().size()
                << ",\"host_rejections\":" << hostRejections << ",\"nonfinite_phases\":" << badPhases
                << ",\"exact_vs_separate\":";warm.write(record);
            record << ",\"finite_output_sha256\":" << splash::json::quote(sha256(firstOutput))
                << ",\"control_paired_timing\":";controlTimes.write(record);
            record << ",\"fused_timing\":";fusedTimes.write(record);
            record << ",\"speedup_gpu\":" << median(controlTimes.gpu) / median(fusedTimes.gpu) << '}';
            records.push_back(record.str());exactHCElements += warm.elements;++hcVariants;
            std::cerr << "cached HC projection=" << prefix << " rows=" << rows << " tile=" << kTileNames[tile] << " exact elements=" << warm.elements << '\n';
          }
        }
      }
      if (only != "hc") {
        const std::string name = std::getenv("FLASH_DENSE_ROUTER_WEIGHT") ? std::getenv("FLASH_DENSE_ROUTER_WEIGHT") :
            "language_model.model.layers.0.mlp.gate.weight";
        const auto &weight = weights.tensor(name);
        require(weight.shape.size() == 2, "router tensor must be a matrix");
        const uint32_t n = uint32_t(weight.shape[0]), k = uint32_t(weight.shape[1]);
        for (uint32_t rows : routerRows) {
          require(rows > 0, "router rows must be positive");
          const auto hostInput = randomInput(uint64_t(rows) * k, 0x5eed1234, 2.0f);
          Guarded input(backend, hostInput.size()), control(backend, uint64_t(rows) * n), output(backend, uint64_t(rows) * n);
          input.load(hostInput);
          const uint32_t hostRejections = validateRouter(backend, weight, input.view, output.view, diag);
          const auto ref = routerReference(weight, hostInput, rows, columnsList[0]);
          CommandGraph vector;
          addDenseBF16(vector, input.view, weight, control.view, diag, rows);*status = kSticky;
          (void)backend.submitCommand(vector.dispatches());control.check();input.check();
          const auto vectorCPU = compareSample(control.values(), ref, n, rows);
          require(*status == kSticky && !vectorCPU.nonfinite && vectorCPU.relativeL2() < .006,
                  "BF16 vector router control disagrees with independent CPU dot reference");
          for (uint32_t tile : tiles) {
            output.clear();*status = kSticky;
            CommandGraph whole;
            addDenseBF16WholeK(backend, whole, input.view, weight, output.view, diag, rows, static_cast<FlashAffineMPPTile>(tile));
            (void)backend.submitCommand(whole.dispatches());output.check();input.check();
            const std::vector<uint16_t> firstOutput(output.values().begin(), output.values().end());
            Times vectorTimes, wholeTimes;paired(backend, vector, whole, repeats, vectorTimes, wholeTimes);
            output.check();control.check();input.check();
            require(*status == kSticky && compare(output.values(), firstOutput).mismatch == 0 &&
                    compare(input.values(), hostInput).mismatch == 0, "whole-K router repeat/diagnostic/input changed");
            const auto vsVector = compare(output.values(), control.values());
            const auto vsCPU = compareSample(output.values(), ref, n, rows);
            require(!vsVector.nonfinite && !vsCPU.nonfinite && vsVector.relativeL2() < .006 && vsCPU.relativeL2() < .006,
                    "whole-K router differs materially from BF16 vector/CPU reference");
            const uint32_t firstTail = rows / kTileRows[tile] * kTileRows[tile];
            const auto tail = compare(output.values().subspan(uint64_t(firstTail) * n), control.values().subspan(uint64_t(firstTail) * n));
            require(tail.mismatch == 0, "router short rows differ from BF16 vector control bits");
            const uint16_t saved = static_cast<uint16_t *>(input.view.contents())[0];
            static_cast<uint16_t *>(input.view.contents())[0] = 0x7f80;output.clear();*status = kSticky;
            (void)backend.submitCommand(whole.dispatches());output.check(false);input.check(false);
            require(*status == (kSticky | kFlashAffineInvalidNumerics), "router nonfinite diagnostic missing or not sticky");
            static_cast<uint16_t *>(input.view.contents())[0] = saved;
            std::ostringstream record;
            record << std::setprecision(12) << "{\"kind\":\"bf16-router\",\"weight\":" << splash::json::quote(name)
                << ",\"rows\":" << rows << ",\"n\":" << n << ",\"k\":" << k
                << ",\"tile\":" << splash::json::quote(kTileNames[tile])
                << ",\"sampled_cpu_columns\":" << ref.columns.size()
                << ",\"host_rejections\":" << hostRejections << ",\"whole_k_dispatches\":" << whole.dispatches().size()
                << ",\"vs_bf16_vector_gpu\":";vsVector.write(record);
            record << ",\"vs_bf16_cpu\":";vsCPU.write(record);
            record << ",\"vector_vs_bf16_cpu\":";vectorCPU.write(record);
            record << ",\"short_rows_vs_vector\":";tail.write(record);
            record << ",\"vector_paired_timing\":";vectorTimes.write(record);
            record << ",\"whole_k_timing\":";wholeTimes.write(record);
            record << ",\"speedup_gpu\":" << median(vectorTimes.gpu) / median(wholeTimes.gpu) << '}';
            records.push_back(record.str());++routerVariants;
            std::cerr << "BF16 router rows=" << rows << " tile=" << kTileNames[tile] << " relative L2=" << vsCPU.relativeL2() << '\n';
          }
        }
      }
      require(hcVariants + routerVariants > 0, "oracle produced no variants");
      std::ostringstream report;report << '{' << reportMetadata << ",\"cases\":[";
      for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
      report << "],\"hc_variants\":" << hcVariants << ",\"router_variants\":" << routerVariants
          << ",\"hc_exact_elements\":" << exactHCElements
          << ",\"pass\":true,\"diagnostics_sticky\":true,\"finite_and_nonfinite_guards_pass\":true,\"host_rejections_pass\":true}";
      writeReport(argv[3], report.str());return 0;
    } catch (const std::exception &error) {
      if (argc == 4) try {
        std::ostringstream report;report << '{';if (!reportMetadata.empty()) report << reportMetadata << ',';
        report << "\"pass\":false,\"error\":" << splash::json::quote(error.what()) << ",\"completed_cases\":[";
        for (size_t i = 0; i < records.size(); ++i) { if (i) report << ',';report << records[i]; }
        report << "]}";writeReport(argv[3], report.str());
      } catch (...) {}
      std::cerr << "flash-dense-cache-hc-oracle: " << error.what() << '\n';return 1;
    }
  }
}
