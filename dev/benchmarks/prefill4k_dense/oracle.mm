// Root coordinates GPU execution. --cpu-self-test does not create a backend.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashDenseCache.h"
#include "metal/abi/FlashDenseSmallRows.h"
#include "flash/FlashDenseTraversal.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "engine/Json.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <filesystem>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint64_t kGuard = 64;
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t bf16(float value) {
  uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
uint64_t hash(const MetalBuffer &buffer) {
  const auto *words = static_cast<const uint16_t *>(buffer.contents());
  uint64_t value = 0xcbf29ce484222325ULL;
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
    value = (value ^ words[i]) * 0x100000001b3ULL;
  return value;
}
DispatchSize traversal(uint32_t rows, uint32_t columns, uint32_t mode) {
  if (mode == 0) return {columns, rows, 1};
  if (mode == 1) return {rows, columns, 1};
  const uint32_t width = 1u << (mode - 1);
  return {uint64_t(columns) * width, (rows + width - 1) / width, 1};
}
std::pair<uint32_t, uint32_t> tileFor(uint32_t x, uint32_t y, uint32_t mode) {
  if (mode == 0) return {y, x};
  if (mode == 1) return {x, y};
  const uint32_t log = mode - 1;
  return {(y << log) + (x & ((1u << log) - 1)), x >> log};
}
void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t rows = 1; rows <= 35; ++rows) {
    for (uint32_t columns : {1u, 3u, 5u, 20u, 81u}) {
      for (uint32_t mode = 0; mode <= 4; ++mode) {
        std::vector<uint32_t> visits(uint64_t(rows) * columns);
        const auto grid = traversal(rows, columns, mode);
        for (uint32_t y = 0; y < grid.y; ++y) {
          for (uint32_t x = 0; x < grid.x; ++x) {
            const auto [r, c] = tileFor(x, y, mode);
            if (r < rows && c < columns) ++visits[uint64_t(r) * columns + c];
          }
        }
        require(std::all_of(visits.begin(), visits.end(), [](uint32_t v) { return v == 1; }),
                "traversal did not cover every tile exactly once");
        ++checks;
      }
    }
  }
  for (uint32_t word = 0; word < 65536; ++word) {
    if (std::isfinite(number(uint16_t(word)))) {
      require(bf16(number(uint16_t(word))) == word, "BF16 round-trip failed"); ++checks;
    }
  }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << "}\n";
}
struct Shape {
  uint32_t k = 0, n = 0;
  std::string projection, weightPath, weightSHA, inputPath, expectedPath;
};
std::vector<uint32_t> parseList(std::string value) {
  std::stringstream stream(value); std::string item; std::vector<uint32_t> result;
  while (std::getline(stream, item, ',')) {
    size_t used = 0; const auto parsed = std::stoul(item, &used);
    require(used == item.size() && parsed > 0 && parsed <= 32768, "invalid integer list");
    result.push_back(uint32_t(parsed));
  }
  require(!result.empty(), "empty integer list"); return result;
}
std::vector<Shape> parseShapes(std::string value) {
  std::stringstream stream(value); std::string item; std::vector<Shape> result;
  while (std::getline(stream, item, ',')) {
    const auto split = item.find('x'); require(split != std::string::npos, "shape must be KxN");
    const auto k = parseList(item.substr(0, split)), n = parseList(item.substr(split + 1));
    require(k.size() == 1 && n.size() == 1 && k[0] % 32 == 0 && n[0] % 64 == 0,
            "shape K and N must be aligned32 and aligned64");
    Shape shape; shape.k = k[0]; shape.n = n[0]; result.push_back(shape);
  }
  require(!result.empty(), "empty shapes"); return result;
}
std::vector<Shape> loadFixtures(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read fixture manifest");
  NSError *error = nil;
  NSDictionary *document = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [document isKindOfClass:[NSDictionary class]], "invalid fixture JSON");
  NSArray *entries = document[@"cases"];
  require([entries isKindOfClass:[NSArray class]] && entries.count > 0, "empty fixture cases");
  std::vector<Shape> result;
  for (NSDictionary *entry in entries) {
    require([entry isKindOfClass:[NSDictionary class]], "invalid fixture case");
    Shape shape;
    shape.k = [entry[@"input_size"] unsignedIntValue];
    shape.n = [entry[@"output_size"] unsignedIntValue];
    require(shape.k && shape.k <= 32768 && shape.k % 32 == 0 && shape.n &&
            shape.n <= 32768 && shape.n % 64 == 0, "invalid fixture shape");
    const auto string = [&](NSString *name, bool required = true) -> std::string {
      id value = entry[name];
      require(!required || [value isKindOfClass:[NSString class]], "missing fixture string");
      return [value isKindOfClass:[NSString class]] ? std::string([value UTF8String]) : std::string{};
    };
    shape.projection = string(@"projection"); shape.weightPath = string(@"weights_file");
    shape.weightSHA = string(@"weights_sha256"); shape.inputPath = string(@"input_file", false);
    shape.expectedPath = string(@"expected_file", false); result.push_back(std::move(shape));
  }
  return result;
}
std::string sha256(const void *pointer, uint64_t bytes) {
  require(bytes <= UINT32_MAX, "SHA256 input exceeds one-shot bound");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  require(CC_SHA256(pointer, CC_LONG(bytes), digest) != nullptr, "SHA256 failed");
  std::ostringstream out;
  for (unsigned char value : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
  return out.str();
}
void readExact(const std::string &path, void *destination, uint64_t bytes) {
  require(std::filesystem::file_size(path) == bytes, "fixture file has wrong extent: " + path);
  std::ifstream input(path, std::ios::binary); require(bool(input), "cannot open fixture file");
  input.read(static_cast<char *>(destination), std::streamsize(bytes));
  require(bool(input), "short fixture file read");
}
struct Error {
  uint64_t elements = 0, mismatches = 0, nonfinite = 0;
  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;
  void add(uint16_t actual, uint16_t expected) {
    ++elements; mismatches += actual != expected;
    const double a = number(actual), b = number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; return; }
    const double delta = a - b;
    maximumAbsolute = std::max(maximumAbsolute, std::abs(delta));
    squaredError += delta * delta; squaredReference += b * b;
  }
  double relativeL2() const { return std::sqrt(squaredError / std::max(1e-30, squaredReference)); }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"nonfinite\":" << nonfinite << ",\"max_abs\":" << maximumAbsolute
        << ",\"relative_l2\":" << relativeL2() << '}';
  }
};
struct Buffers {
  MetalBuffer input, padded, weights, outputBase, output, diagnostics;
  uint64_t elements = 0, inputHash = 0, paddedHash = 0, weightHash = 0;
  Buffers(MetalBackend &backend, uint32_t rows, const Shape &shape) {
    const uint32_t paddedRows = (rows + 127) / 128 * 128;
    input = backend.allocateBuffer(uint64_t(rows) * shape.k * 2, BufferStorage::Shared);
    padded = backend.allocateBuffer(uint64_t(paddedRows) * shape.k * 2, BufferStorage::Shared);
    weights = backend.allocateBuffer(uint64_t(shape.n) * shape.k * 2, BufferStorage::Shared);
    elements = uint64_t(rows) * shape.n;
    outputBase = backend.allocateBuffer((elements + 2 * kGuard) * 2, BufferStorage::Shared);
    output = backend.view(outputBase, kGuard * 2, elements * 2);
    diagnostics = backend.allocateBuffer(64, BufferStorage::Shared);
    auto *x = static_cast<uint16_t *>(input.contents());
    auto *w = static_cast<uint16_t *>(weights.contents());
    for (uint64_t i = 0; i < input.sizeBytes() / 2; ++i)
      x[i] = bf16(float(int32_t(randomWord(i + 7) % 8193) - 4096) / 4096.f);
    for (uint64_t i = 0; i < weights.sizeBytes() / 2; ++i)
      w[i] = bf16(float(int32_t(randomWord(i + 413) % 8193) - 4096) / 65536.f);
    if (!shape.weightPath.empty()) {
      readExact(shape.weightPath, weights.contents(), weights.sizeBytes());
      require(sha256(weights.contents(), weights.sizeBytes()) == shape.weightSHA,
              "actual BF16 coefficient payload SHA256 mismatch");
    }
    if (!shape.inputPath.empty()) readExact(shape.inputPath, input.contents(), input.sizeBytes());
    std::memset(padded.contents(), 0, padded.sizeBytes());
    std::memcpy(padded.contents(), input.contents(), input.sizeBytes());
    inputHash = hash(input); paddedHash = hash(padded); weightHash = hash(weights); reset();
  }
  void reset() {
    std::fill_n(static_cast<uint16_t *>(outputBase.contents()), outputBase.sizeBytes() / 2, kSentinel);
    std::memset(diagnostics.contents(), 0, diagnostics.sizeBytes());
    *static_cast<uint32_t *>(diagnostics.contents()) = kSticky;
  }
  void guards() const {
    const auto *all = static_cast<const uint16_t *>(outputBase.contents());
    for (uint64_t i = 0; i < kGuard; ++i)
      require(all[i] == kSentinel && all[kGuard + elements + i] == kSentinel,
              "output guard changed");
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == kSticky,
            "shader diagnostics changed");
  }
  std::vector<uint16_t> result() const {
    const auto *words = static_cast<const uint16_t *>(output.contents());
    return {words, words + elements};
  }
};
struct Variant {
  std::string name;
  uint32_t m = 0, n = 0, mode = 0, groups = 0;
  bool production = false, vector = false, block512 = false, prefillProduction = false;
  std::vector<double> gpu, wall;
  Error baselineError, geometryError, oracleError;
  double medianGPU = 0, medianWall = 0;
};
std::string modeName(uint32_t mode) {
  static const char *names[] = {"column", "row", "swizzle2", "swizzle4", "swizzle8"};
  return names[mode];
}
Variant variant(uint32_t m, uint32_t n, uint32_t mode, bool production = false,
                uint32_t groups = 0, bool block512 = false) {
  Variant value;
  value.m = m; value.n = n; value.mode = mode; value.production = production;
  value.groups = groups ? groups : m >= 64 ? 8 : 4;
  value.block512 = block512;
  value.name = (production ? "production_" : "candidate_") +
      std::string("m") + std::to_string(m) + "n" + std::to_string(n) +
      "_sg" + std::to_string(value.groups) + (block512 ? "_k512_" : "_kwhole_") + modeName(mode);
  return value;
}
std::vector<Variant> variants(uint32_t rows, Shape shape, bool productionTraversal, bool productionPrefill) {
  const uint32_t outputs = shape.n;
  uint32_t baselineM = rows < 16 ? 8 : rows >= 128 && outputs >= 1024 ? 32 : 16;
  uint32_t baselineN = rows < 16 && outputs >= 1024 ? 128 : baselineM == 32 ? 128 : 64;
  if (rows >= 512 && shape.k == 6144 && shape.n == 2560) {
    baselineM = 64; baselineN = 128;
  }
  std::vector<Variant> result{variant(baselineM, baselineN, 0, true)};
  if (productionPrefill) {
    const auto plan = splash::flash::flashPrefillDenseTilePolicy(rows,shape.n,shape.k);
    if (plan) {
      auto chosen = variant(plan.tileRows,plan.tileOutputs,static_cast<uint32_t>(plan.traversal),true,plan.simdGroups);
      chosen.prefillProduction = true; result.push_back(std::move(chosen));
    }
    return result;
  }
  if (productionTraversal) {
    require(rows >= 16, "production traversal benchmark requires full whole-K row tiles");
    for (uint32_t mode = 1; mode <= 4; ++mode)
      result.push_back(variant(baselineM, baselineN, mode, true));
    return result;
  }
  std::set<std::tuple<uint32_t,uint32_t,uint32_t>> geometries{
      {baselineM,baselineN,baselineM >= 64 ? 8u : 4u},
      {32,128,4},{32,128,8},{64,128,4},{64,128,8},
      {128,64,4},{128,64,8},{128,128,8}};
  for (const auto &[m,n,groups] : geometries)
    for (uint32_t mode = 0; mode <= 4; ++mode) {
      result.push_back(variant(m,n,mode,false,groups));
      if (shape.k % 512 == 0)
        result.push_back(variant(m,n,mode,false,groups,true));
    }
  return result;
}
CommandGraph graphFor(const Variant &v, const Buffers &b, uint32_t rows, Shape shape,
                      uint32_t repetitions) {
  CommandGraph graph;
  for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
    if (v.vector) {
      graph.add("flash_dense_bf16_project", {b.input, b.weights, b.output, b.diagnostics},
          FlashDenseParams{rows, shape.k, shape.n, 0, uint64_t(shape.k) * 2},
          {(shape.n + 7) / 8, rows, 1}, {256, 1, 1});
      continue;
    }
    if (rows < 16) {
      const FlashDenseSmallRowsParams params{rows, (rows + v.m - 1) / v.m * v.m,
          shape.k, shape.n, 0, shape.n, v.m, v.n};
      graph.add("flash_dense_small_rows_pad", {b.input, b.padded, b.diagnostics}, params,
          {(uint64_t(params.padded_rows) * shape.k + 255) / 256, 1, 1});
    }
    const uint32_t full = shape.n / v.n * v.n;
    for (const auto [begin, count, tileN] : {
        std::tuple{0u, full, v.n}, std::tuple{full, shape.n - full, 64u}}) {
      if (!count) continue;
      const std::string suffix = "m" + std::to_string(v.m) + "_n" + std::to_string(tileN);
      if (v.production && rows < 16) {
        const FlashDenseSmallRowsParams params{rows, (rows + v.m - 1) / v.m * v.m,
            shape.k, shape.n, begin, count, v.m, tileN};
        graph.add("flash_dense_small_rows_" + suffix, {b.padded, b.weights, b.output, b.diagnostics},
            params, {count / tileN, params.padded_rows / v.m, 1}, {128, 1, 1});
      } else {
        require(!v.production || rows % v.m == 0, "production tile requires complete rows");
        const FlashDenseCacheParams params{rows, shape.k, shape.n, begin, count, v.m, tileN, v.mode};
        const std::string pipeline = v.prefillProduction
            ? "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(v.groups)
            : (v.production ? "flash_dense_cache_" : "prefill4k_dense_") + suffix +
                (v.production ? v.mode != 0 ? "_traversal" : "" :
                    "_sg" + std::to_string(v.groups) + (v.block512 ? "_k512" : ""));
        graph.add(pipeline,
            {b.padded, b.weights, b.output, b.diagnostics}, params,
            traversal((rows + v.m - 1) / v.m, count / tileN, v.mode),
            {v.groups * 32, 1, 1});
      }
    }
  }
  return graph;
}
Error compare(const std::vector<uint16_t> &actual, const std::vector<uint16_t> &expected) {
  require(actual.size() == expected.size(), "output extent mismatch"); Error result;
  for (uint64_t i = 0; i < actual.size(); ++i) result.add(actual[i], expected[i]);
  return result;
}
Error scalarOracle(const Buffers &b, uint32_t rows, Shape shape,
                   const std::vector<uint16_t> &actual) {
  const auto *x = static_cast<const uint16_t *>(b.input.contents());
  const auto *w = static_cast<const uint16_t *>(b.weights.contents()); Error result;
  const uint32_t sampleRows = std::min(rows, 8u), sampleColumns = std::min(shape.n, 32u);
  for (uint32_t ri = 0; ri < sampleRows; ++ri) {
    const uint32_t r = sampleRows == 1 ? 0 : uint32_t(uint64_t(ri) * (rows - 1) / (sampleRows - 1));
    for (uint32_t ci = 0; ci < sampleColumns; ++ci) {
      const uint32_t c = uint32_t(uint64_t(ci) * (shape.n - 1) / (sampleColumns - 1));
      double dot = 0;
      for (uint32_t k = 0; k < shape.k; ++k)
        dot += double(number(x[uint64_t(r) * shape.k + k])) * number(w[uint64_t(c) * shape.k + k]);
      result.add(actual[uint64_t(r) * shape.n + c], bf16(float(dot)));
    }
  }
  return result;
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end()); const auto middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
uint64_t shaderBoundaryTests(MetalBackend &backend, bool productionTraversal, bool productionPrefill) {
  Shape boundary; boundary.k = 32; boundary.n = 320;
  const uint32_t rows = productionPrefill ? 2048 : 128;
  const uint32_t tileM = productionPrefill ? 128 : 32;
  const uint32_t tileN = productionPrefill ? 64 : 128;
  Buffers b(backend, rows, boundary); uint64_t checks = 0;
  const std::string pipeline = productionPrefill
      ? "flash_dense_cache_prefill_m128_n64_sg4" : productionTraversal
      ? "flash_dense_cache_m32_n128_traversal" : "prefill4k_dense_m32_n128_sg4";
  FlashDenseCacheParams valid{rows,32,320,64,128,tileM,tileN,3};
  CommandGraph graph;
  graph.add(pipeline, {b.padded, b.weights, b.output, b.diagnostics},
      valid, traversal(rows/tileM,128/tileN,3), {128, 1, 1});
  (void)backend.submitCommand(graph.dispatches()); b.guards();
  const auto words = b.result();
  for (uint32_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < 320; ++c)
    require(c >= 64 && c < 192 ? words[uint64_t(r) * 320 + c] != kSentinel
                              : words[uint64_t(r) * 320 + c] == kSentinel,
            "partial output interval changed the wrong elements");
  ++checks;
  for (uint32_t which = 0; which < 7; ++which) {
    auto p = valid;
    switch (which) {
    case 0: p.rows = 0; break;
    case 1: p.rows = 8193; break;
    case 2: p.input_size = 31; break;
    case 3: p.output_begin = 321; break;
    case 4: p.output_count = 32; break;
    case 5: p.tile_rows = 16; break;
    default: p.reserved = 5; break;
    }
    b.reset(); CommandGraph bad;
    bad.add(pipeline, {b.padded, b.weights, b.output, b.diagnostics},
        p, {1, 1, 1}, {128, 1, 1});
    (void)backend.submitCommand(bad.dispatches());
    require(*static_cast<const uint32_t *>(b.diagnostics.contents()) == (kSticky | 2u),
            "invalid params did not set sticky diagnostics");
    const auto *all = static_cast<const uint16_t *>(b.outputBase.contents());
    require(std::all_of(all, all + b.outputBase.sizeBytes() / 2,
                [](uint16_t value) { return value == kSentinel; }),
            "invalid params wrote output"); ++checks;
  }
  return checks;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::string library = "build/prefill4k-dense/locality.metallib", output;
    auto rowsList = parseList("2048");
    auto shapes = parseShapes("2560x10240,2560x12288,10240x320,6144x2560,2560x6144,320x10240");
    uint32_t samples = 7, repetitions = 4; bool productionTraversal = false, productionPrefill = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (arg == "--production-traversal") { productionTraversal = true; continue; }
      if (arg == "--production-prefill") { productionPrefill = true; continue; }
      require(i + 1 < argc, "missing CLI value"); const std::string value = argv[++i];
      if (arg == "--library") library = value;
      else if (arg == "--out") output = value;
      else if (arg == "--rows") rowsList = parseList(value);
      else if (arg == "--shapes") shapes = parseShapes(value);
      else if (arg == "--fixture-manifest") shapes = loadFixtures(value);
      else if (arg == "--samples") samples = parseList(value).at(0);
      else if (arg == "--repeat") repetitions = parseList(value).at(0);
      else throw std::runtime_error("unknown CLI argument: " + arg);
    }
    require(samples <= 99 && repetitions <= 64, "excessive sample or repeat count");
    for (uint32_t rows : rowsList)
      require(rows >= 128 && rows <= 8192 && rows % 128 == 0,
              "prefill screen rows must be128..8192 in multiples of128");
    MetalBackend backend(library);
    const auto boundaryChecks = shaderBoundaryTests(backend, productionTraversal, productionPrefill);
    std::ostringstream report; report << std::setprecision(10)
        << "{\"experiment\":\"prefill4k_dense_v1\",\"arithmetic\":\"whole_k_bf16_mpp\","
        << "\"die_affinity\":false,\"production_traversal\":"
        << (productionTraversal ? "true" : "false") << ",\"boundary_checks\":" << boundaryChecks
        << ",\"production_prefill_tiles\":" << (productionPrefill ? "true" : "false")
        << ",\"samples\":" << samples << ",\"repeat\":" << repetitions << ",\"cases\":[";
    bool firstCase = true;
    for (Shape shape : shapes) for (uint32_t rows : rowsList) {
      Buffers b(backend, rows, shape); auto candidates = variants(rows, shape, productionTraversal, productionPrefill);
      std::vector<uint16_t> reference;
      std::map<std::tuple<uint32_t,uint32_t,uint32_t,bool>, std::vector<uint16_t>> geometryReferences;
      for (auto &candidate : candidates) {
        b.reset(); auto warm = graphFor(candidate, b, rows, shape, 1);
        (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();
        if (reference.empty()) reference = actual;
        if (candidate.production && candidate.mode == 0 && !shape.expectedPath.empty()) {
          std::vector<uint16_t> captured(actual.size());
          readExact(shape.expectedPath, captured.data(), captured.size() * 2);
          require(compare(actual,captured).mismatches == 0,
                  "original production kernel did not match captured model BF16 output");
        }
        candidate.baselineError = compare(actual, reference);
        if (candidate.prefillProduction)
          require(candidate.baselineError.mismatches == 0,
                  "selected production prefill tile changed captured BF16 model output");
        candidate.oracleError = scalarOracle(b, rows, shape, actual);
        require(candidate.oracleError.nonfinite == 0 && candidate.oracleError.relativeL2() <= .004,
                candidate.name + " failed sampled scalar numerical oracle");
        const auto key = std::tuple{candidate.m,candidate.n,candidate.groups,candidate.block512};
        if (!geometryReferences.contains(key)) geometryReferences[key] = actual;
        candidate.geometryError = compare(actual, geometryReferences.at(key));
        require(candidate.geometryError.mismatches == 0,
                candidate.name + " traversal changed same-geometry BF16 output");
      }
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t j = 0; j < candidates.size(); ++j) {
          auto &candidate = candidates[(j + sample * 7) % candidates.size()];
          auto command = graphFor(candidate, b, rows, shape, repetitions);
          const auto timing = backend.submitCommand(command.dispatches());
          b.guards(); candidate.gpu.push_back(timing.gpuSeconds * 1000 / repetitions);
          candidate.wall.push_back(timing.wallSeconds * 1000 / repetitions);
        }
      }
      require(hash(b.input) == b.inputHash && hash(b.padded) == b.paddedHash && hash(b.weights) == b.weightHash,
              "immutable input or weight buffer changed");
      if (!firstCase) report << ','; firstCase = false;
      report << "{\"rows\":" << rows << ",\"input_size\":" << shape.k
          << ",\"output_size\":" << shape.n << ",\"projection\":"
          << splash::json::quote(shape.projection) << ",\"weight_sha256\":"
          << splash::json::quote(shape.weightSHA) << ",\"actual_input\":"
          << (shape.inputPath.empty() ? "false" : "true") << ",\"captured_output_exact\":"
          << (shape.expectedPath.empty() ? "false" : "true") << ",\"selected_policy_mode\":"
          << static_cast<uint32_t>(splash::flash::flashDenseTraversalPolicy(rows, shape.n,
                shape.k, candidates[0].m, candidates[0].n)) << ",\"variants\":[";
      bool firstVariant = true; double fastest = 1e30; std::string winner;
      for (auto &candidate : candidates) {
        candidate.medianGPU = median(candidate.gpu); candidate.medianWall = median(candidate.wall);
        if (candidate.medianGPU < fastest) { fastest = candidate.medianGPU; winner = candidate.name; }
        if (!firstVariant) report << ','; firstVariant = false;
        report << "{\"name\":" << splash::json::quote(candidate.name)
            << ",\"tile_rows\":" << candidate.m << ",\"tile_outputs\":" << candidate.n
            << ",\"simdgroups\":" << candidate.groups << ",\"k512\":"
            << (candidate.block512 ? "true" : "false") << ",\"traversal_mode\":" << candidate.mode
            << ",\"median_gpu_ms\":" << candidate.medianGPU
            << ",\"median_wall_ms\":" << candidate.medianWall
            << ",\"speedup_over_baseline\":" << candidates[0].medianGPU / candidate.medianGPU
            << ",\"baseline_error\":"; candidate.baselineError.write(report);
        report << ",\"same_geometry_error\":"; candidate.geometryError.write(report);
        report << ",\"sampled_scalar_error\":"; candidate.oracleError.write(report);
        report << ",\"gpu_ms\":[";
        for (uint32_t i = 0; i < candidate.gpu.size(); ++i) {
          if (i) report << ','; report << candidate.gpu[i];
        }
        report << "]}";
      }
      report << "]}";
      std::cerr << "rows=" << rows << " K=" << shape.k << " N=" << shape.n
          << " baseline_ms=" << candidates[0].medianGPU << " best=" << winner
          << " speedup=" << candidates[0].medianGPU / fastest << '\n';
    }
    report << "]}\n";
    if (output.empty()) std::cout << report.str();
    else { std::ofstream file(output); require(bool(file), "cannot open report output"); file << report.str(); }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "prefill4k dense oracle failed: " << error.what() << '\n'; return 1;
  }
}
