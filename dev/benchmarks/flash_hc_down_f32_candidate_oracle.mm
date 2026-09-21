// Private Root-only GPU screen. CPU self-test creates no Metal device/backend.
#include "FlashHCDownF32Candidate.hpp"
#include "FlashHCDownF32CandidateABI.h"
#include "FlashFloatBoundaryAudit.hpp"
#include "FlashHCDownHostProvenance.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace splash::flash;
using namespace splash::metal;
namespace c = splash::flash::candidate;
namespace audit = splash::flash::benchmark;
constexpr uint32_t kSticky = 0x40000000u;
void require(bool value, const std::string &reason) { if (!value) throw std::runtime_error(reason); }
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t bf16(float value) { return audit::bf16(value); }
std::string hash(const void *data, uint64_t bytes) {
  CC_SHA256_CTX context{}; require(CC_SHA256_Init(&context), "SHA init failed");
  auto *next = static_cast<const uint8_t *>(data);
  while (bytes) { const auto count = CC_LONG(std::min<uint64_t>(bytes, 1ULL << 30));
    require(CC_SHA256_Update(&context, next, count), "SHA update failed"); next += count; bytes -= count; }
  std::array<uint8_t, 32> digest{}; require(CC_SHA256_Final(digest.data(), &context), "SHA final failed");
  constexpr char hex[] = "0123456789abcdef"; std::string result;
  for (uint8_t v : digest) { result += hex[v >> 4]; result += hex[v & 15]; } return result;
}
std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> fallback, uint32_t limit) {
  const char *raw = std::getenv(name); if (!raw) return fallback;
  std::stringstream stream(raw); std::string item; std::vector<uint32_t> result;
  while (std::getline(stream, item, ',')) { require(!item.empty() && item.front() != '-', std::string("invalid ") + name);
    size_t used = 0; const auto value = std::stoul(item, &used);
    require(used == item.size() && value <= limit, std::string("invalid ") + name); result.push_back(uint32_t(value)); }
  require(!result.empty(), "empty selector"); return result;
}
std::vector<std::string> prefixes() {
  if (const char *raw = std::getenv("FLASH_HC_DOWN_F32_PREFIXES")) {
    std::stringstream stream(raw); std::string item; std::vector<std::string> result;
    while (std::getline(stream, item, ',')) { require(!item.empty(), "empty prefix"); result.push_back(item); } return result;
  }
  return {"language_model.model.layers.1.attn_hyper_connection", "language_model.model.layers.0.attn_hyper_connection",
      "language_model.model.layers.15.attn_hyper_connection", "language_model.model.layers.31.attn_hyper_connection",
      "language_model.model.layers.19.attn_hyper_connection", "language_model.model.hyper_connection_mixer"};
}
struct Guard {
  MetalBuffer base, view; uint64_t bytes;
  Guard(MetalBackend &backend, uint64_t count) : bytes(count) {
    base = backend.allocateBuffer(count + 128, BufferStorage::Shared, "HC down candidate guard");
    std::memset(base.contents(), 0x5a, count + 128); view = backend.view(base, 64, count);
    std::memset(view.contents(), 0, count);
  }
  void check() const {
    const auto *p = static_cast<const uint8_t *>(base.contents());
    for (uint32_t i = 0; i < 64; ++i) require(p[i] == 0x5a && p[64 + bytes + i] == 0x5a, "canary overwritten");
  }
};
struct Error {
  uint64_t elements = 0, mismatches = 0, nonfinite = 0;
  uint32_t maxULP = 0; double sumError = 0, sumReference = 0, maxAbsolute = 0;
  void add(uint16_t a, uint16_t b) {
    ++elements; mismatches += a != b; const double x = number(a), y = number(b);
    if (!std::isfinite(x) || !std::isfinite(y)) { ++nonfinite; return; }
    const double delta = x - y; sumError += delta * delta; sumReference += y * y;
    maxAbsolute = std::max(maxAbsolute, std::abs(delta)); maxULP = std::max(maxULP, audit::bf16ULP(a, b));
  }
  double relative() const { return std::sqrt(sumError / std::max(sumReference, 1e-30)); }
  bool strict() const { return !nonfinite && relative() <= 1e-4; }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches << ",\"nonfinite\":" << nonfinite
        << ",\"maximum_bf16_ulp\":" << maxULP << ",\"maximum_absolute\":" << maxAbsolute << ",\"relative_l2\":" << relative() << '}';
  }
};
Error compare(const MetalBuffer &candidate, const MetalBuffer &control, uint32_t rows, uint32_t width, uint32_t stride) {
  const auto *a = static_cast<const uint16_t *>(candidate.contents()), *b = static_cast<const uint16_t *>(control.contents());
  Error result;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t n = 0; n < width; ++n)
      result.add(a[uint64_t{row} * stride + n], b[uint64_t{row} * stride + n]);
  return result;
}
uint32_t code(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *w = static_cast<const uint8_t *>(p.weights->buffer.contents()) + uint64_t{n} * p.weightRowStrideBytes;
  const uint32_t bit = k * p.bits, shift = bit % 8; uint32_t value = w[bit / 8];
  if (shift + p.bits > 8) value |= uint32_t(w[bit / 8 + 1]) << 8; return value >> shift & ((1u << p.bits) - 1u);
}
float coefficient(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *s = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.scales->buffer.contents()) + uint64_t{n} * p.parameterRowStrideBytes);
  const auto *b = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.biases->buffer.contents()) + uint64_t{n} * p.parameterRowStrideBytes);
  const float product = float(code(p, n, k)) * number(s[k / p.groupSize]); return product + number(b[k / p.groupSize]);
}
void coefficientProof(const FlashAffineProjection &p, const FlashTensor &cached) {
  const auto *actual = static_cast<const float *>(cached.buffer.contents());
  require(cached.dtype == FlashDType::F32 && cached.shape == std::vector<uint64_t>{320, 10240}, "bad saved original-F32 coefficient shape");
  for (uint32_t n = 0; n < 320; ++n) for (uint32_t k = 0; k < 10240; ++k) {
    const float expected = coefficient(p, n, k);
    require(std::isfinite(expected) && std::bit_cast<uint32_t>(actual[uint64_t{n} * 10240 + k]) == std::bit_cast<uint32_t>(expected), "original F32 coefficient bits changed");
  }
}
double median(std::vector<double> values) {
  require(!values.empty(), "empty times"); std::sort(values.begin(), values.end()); const size_t n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
struct Times {
  std::vector<double> gpu, wall;
  void add(CommandTiming t) {
    require(std::isfinite(t.gpuSeconds) && t.gpuSeconds > 1e-9 && std::isfinite(t.wallSeconds) && t.wallSeconds > 1e-9, "invalid timing/possible ABI mismatch");
    gpu.push_back(t.gpuSeconds); wall.push_back(t.wallSeconds);
  }
  void write(std::ostream &out) const {
    out << "{\"median_gpu_ms\":" << median(gpu) * 1000 << ",\"median_wall_ms\":" << median(wall) * 1000 << ",\"gpu_ms\":[";
    for (size_t i = 0; i < gpu.size(); ++i) { if (i) out << ','; out << gpu[i] * 1000; } out << "]}";
  }
};
std::string metadata(const char *path) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal unavailable"); NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
  require(library != nil, "private library unavailable"); std::ostringstream out; out << "{\"pipelines\":["; bool first = true;
  for (const char *name : {"flash_hc_down_f32_literal_witness", "flash_hc_down_f32_literal_reuse_r4", "flash_hc_down_f32_literal_reuse_r8", "flash_hc_down_f32_literal_reuse_r16", "flash_hc_down_f32_mpp_m8_n32", "flash_hc_down_f32_mpp_m8_n64", "flash_hc_down_f32_fold_activate", "flash_hc_down_f32_literal_injection", "flash_hc_down_f32_epilog_from_raw"}) {
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    require(function != nil, std::string("missing ") + name); id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function error:&error];
    const uint32_t requested = std::string_view(name).find("fold_activate") != std::string_view::npos ||
        std::string_view(name).find("epilog_from_raw") != std::string_view::npos ? 256 : 128;
    require(p != nil && p.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength && p.maxTotalThreadsPerThreadgroup >= requested,
        std::string("pipeline limits exceeded ") + name);
    if (!first) out << ','; first = false; out << "{\"name\":" << splash::json::quote(name) << ",\"static_threadgroup_memory_bytes\":" << p.staticThreadgroupMemoryLength << '}';
  }
  out << "],\"device_max_threadgroup_memory_bytes\":" << device.maxThreadgroupMemoryLength << '}'; return out.str();
}
void midpoint(std::ostream &out, const FlashAffineProjection &p, MetalBuffer input,
    MetalBuffer rawControl, MetalBuffer rawCandidate, uint32_t rows) {
  const auto *x = static_cast<const uint16_t *>(input.contents());
  const auto *a = static_cast<const uint16_t *>(rawControl.contents()), *b = static_cast<const uint16_t *>(rawCandidate.contents());
  out << "{\"diagnostic_only\":true,\"overrides_accuracy_pass\":false,\"changed_cells\":["; bool first = true;
  for (uint32_t row = 0; row < rows; ++row) for (uint32_t n = 0; n < 320; ++n) {
    const uint64_t index = uint64_t{row} * 324 + n; if (a[index] == b[index]) continue;
    double sum = 0, l1 = 0, flush = 0;
    for (uint32_t k = 0; k < 10240; ++k) {
      const double product = double(number(x[uint64_t{row} * 10240 + k])) * coefficient(p, n, k);
      sum += product; l1 += std::abs(product); if (std::abs(product) < std::numeric_limits<float>::min()) flush += std::abs(product);
    }
    const double bound = audit::f32DotBound(10240, l1, flush); const uint16_t gold = audit::bf16Double(sum);
    const auto ca = audit::cellRelation(a[index], gold, sum, bound), cb = audit::cellRelation(b[index], gold, sum, bound);
    if (!first) out << ','; first = false;
    out << "{\"row\":" << row << ",\"column\":" << n << ",\"control_bf16\":" << a[index] << ",\"candidate_bf16\":" << b[index]
        << ",\"independent_double\":" << sum << ",\"double_gold_bf16\":" << gold << ",\"f32_dot_bound\":" << bound
        << ",\"control_midpoint_compatible\":" << (ca.pass ? "true" : "false") << ",\"candidate_midpoint_compatible\":" << (cb.pass ? "true" : "false") << '}';
  }
  out << "]}";
}

bool run(MetalBackend &backend, const FlashWeights &weights, const FlashFloatDenseCache &cache,
    const std::string &prefix, uint32_t rows, c::HCDownF32Mode mode, uint32_t pairs, std::ostream &out) {
  const std::string downName = prefix + ".input_mix_weight_down";
  const auto &down = weights.projection(downName);
  const auto &cached = cache.tensor(downName);
  const FlashAffineProjection *inj = weights.contains(prefix + ".block_inject_weight.weight") ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
  Guard input(backend, uint64_t{rows} * 10240 * 2); auto *x = static_cast<uint16_t *>(input.view.contents());
  const char *rawInput = std::getenv("FLASH_HC_DOWN_F32_INPUT");
  if (rawInput) {
    require(std::filesystem::file_size(rawInput) == input.bytes, "raw normalized BF16 fixture extent differs");
    std::ifstream stream(rawInput, std::ios::binary); stream.read(reinterpret_cast<char *>(x), input.bytes); require(bool(stream), "input read failed");
  } else for (uint64_t i = 0; i < input.bytes / 2; ++i) x[i] = bf16(float(int32_t((i * 73 + i / 10240 * 17) % 2047) - 1023) / 1024.0f);
  for (uint64_t i = 0; i < input.bytes / 2; ++i) require(std::isfinite(number(x[i])), "input nonfinite");
  const auto inputSHA = hash(input.view.contents(), input.bytes), cacheSHA = hash(cached.buffer.contents(), cached.logicalBytes);
  std::vector<std::pair<MetalBuffer, std::string>> sources;
  for (const auto &p : {&down, inj}) if (p) for (const auto *t : {p->weights, p->scales, p->biases}) sources.emplace_back(t->buffer, hash(t->buffer.contents(), t->logicalBytes));
  std::array<Guard, 4> activated{Guard(backend, rows * 320 * 2), Guard(backend, rows * 320 * 2), Guard(backend, rows * 320 * 2), Guard(backend, rows * 320 * 2)};
  std::array<Guard, 4> gates{Guard(backend, rows * 4 * 2), Guard(backend, rows * 4 * 2), Guard(backend, rows * 4 * 2), Guard(backend, rows * 4 * 2)};
  std::array<Guard, 4> diag{Guard(backend, 4), Guard(backend, 4), Guard(backend, 4), Guard(backend, 4)};
  for (auto &d : diag) *static_cast<uint32_t *>(d.view.contents()) = kSticky;
  std::array<Guard, 2> rawBF{Guard(backend, rows * 324 * 2), Guard(backend, rows * 324 * 2)};
  std::array<Guard, 2> rawF{Guard(backend, rows * 324 * 4), Guard(backend, rows * 324 * 4)};
  Guard padded(backend, 16 * 10240 * 2), partial(backend, 16 * 8 * 320 * 4);
  const c::HCDownF32Scratch scratch{padded.view, partial.view};
  std::array<CommandGraph, 4> graph;
  addHCFusedDown(graph[0], input.view, down, inj, activated[0].view, gates[0].view, diag[0].view, {rows, 2560, 4, 1e-6f});
  c::addHCDownLiteralWitness(backend, graph[2], input.view, down, inj, activated[2].view, gates[2].view, diag[2].view, rows, {rawBF[0].view, rawF[0].view});
  c::addHCDownF32Candidate(backend, graph[1], input.view, cached, down, inj, activated[1].view, gates[1].view, diag[1].view, rows, mode, scratch);
  c::addHCDownF32Candidate(backend, graph[3], input.view, cached, down, inj, activated[3].view, gates[3].view, diag[3].view, rows, mode, scratch, {rawBF[1].view, rawF[1].view});
  for (uint32_t i : {0u, 2u, 3u, 1u}) (void)backend.submitCommand(graph[i].dispatches());
  require(std::memcmp(activated[0].view.contents(), activated[2].view.contents(), activated[0].bytes) == 0 &&
      (!inj || std::memcmp(gates[0].view.contents(), gates[2].view.contents(), gates[0].bytes) == 0), "literal witness differs from production control");
  const auto rawError = compare(rawBF[1].view, rawBF[0].view, rows, 320, 324);
  const auto activationError = compare(activated[1].view, activated[0].view, rows, 320, 320);
  const auto injectionError = compare(gates[1].view, gates[0].view, rows, inj ? 4 : 0, 4);
  uint64_t f32Mismatches = 0;
  const auto *fa = static_cast<const uint32_t *>(rawF[0].view.contents()), *fb = static_cast<const uint32_t *>(rawF[1].view.contents());
  for (uint32_t row = 0; row < rows; ++row) for (uint32_t n = 0; n < 320 + (inj ? 4 : 0); ++n) f32Mismatches += fa[uint64_t{row} * 324 + n] != fb[uint64_t{row} * 324 + n];
  require(std::memcmp(activated[1].view.contents(), activated[3].view.contents(), activated[1].bytes) == 0 &&
      (!inj || std::memcmp(gates[1].view.contents(), gates[3].view.contents(), gates[1].bytes) == 0), "debug candidate changes numerical output");
  HCDownF32CandidateParams post{}; post.literal.rows = rows; post.literal.has_injection = inj != nullptr;
  CommandGraph postGraph;
  postGraph.add("flash_hc_down_f32_epilog_from_raw", {rawBF[1].view, activated[3].view, gates[3].view, diag[3].view}, post,
      {(uint64_t{rows} * 324 + 255) / 256, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(postGraph.dispatches());
  const bool canonicalPost = std::memcmp(activated[1].view.contents(), activated[3].view.contents(), activated[1].bytes) == 0 &&
      (!inj || std::memcmp(gates[1].view.contents(), gates[3].view.contents(), gates[1].bytes) == 0);
  const bool literal = mode == c::HCDownF32Mode::LiteralRowReuse;
  const bool accuracy = canonicalPost && injectionError.mismatches == 0 && rawError.strict() && activationError.strict() &&
      (!literal || (!f32Mismatches && !rawError.mismatches && !activationError.mismatches));
  const auto healthy = [&] {
    input.check(); padded.check(); partial.check();
    for (uint32_t i = 0; i < 4; ++i) { activated[i].check(); gates[i].check(); diag[i].check(); require(*static_cast<uint32_t *>(diag[i].view.contents()) == kSticky, "sticky diagnostic changed"); }
    for (uint32_t i = 0; i < 2; ++i) { rawBF[i].check(); rawF[i].check(); }
  };
  healthy(); std::array<Times, 2> timing;
  for (uint32_t pair = 0; pair < pairs; ++pair) for (uint32_t order = 0; order < 2; ++order) {
    const uint32_t which = (pair + order) % 2; timing[which].add(backend.submitCommand(graph[which].dispatches())); healthy();
  }
  require(hash(input.view.contents(), input.bytes) == inputSHA && hash(cached.buffer.contents(), cached.logicalBytes) == cacheSHA, "immutable input/cache changed");
  for (const auto &[buffer, before] : sources) require(hash(buffer.contents(), buffer.sizeBytes()) == before, "original source changed");
  out << "{\"prefix\":" << splash::json::quote(prefix) << ",\"source_bits\":" << down.bits << ",\"injection_bits\":" << (inj ? inj->bits : 0)
      << ",\"rows\":" << rows << ",\"mode\":" << splash::json::quote(c::hcDownF32ModeName(mode)) << ",\"accuracy_pass\":" << (accuracy ? "true" : "false")
      << ",\"acceptance_rule\":\"literal-all-bits-exact; MPP-raw-and-activation-relative-L2-at-most1e-4; injection-and-candidate-own-canonical-post-exact; no-midpoint-exception\",\"input_policy\":"
      << splash::json::quote(rawInput ? "captured-raw-normalized-bf16" : "declared-deterministic-synthetic-normalized-bf16")
      << ",\"input_sha256\":" << splash::json::quote(inputSHA) << ",\"cache_sha256\":" << splash::json::quote(cacheSHA)
      << ",\"raw_f32_mismatches\":" << f32Mismatches << ",\"canonical_post_exact\":" << (canonicalPost ? "true" : "false")
      << ",\"canaries_clean\":true,\"immutable_sources_preserved\":true,\"raw_dot_error\":"; rawError.write(out);
  out << ",\"activation_error\":"; activationError.write(out); out << ",\"injection_error\":"; injectionError.write(out);
  out << ",\"control_timing\":"; timing[0].write(out); out << ",\"candidate_timing\":"; timing[1].write(out);
  out << ",\"midpoint_evidence\":"; midpoint(out, down, input.view, rawBF[0].view, rawBF[1].view, rows); out << '}'; return accuracy;
}
void cpuSelfTest() {
  require(sizeof(HCDownF32CandidateParams) == 176 && sizeof(FlashHCFusedParams) == 160, "ABI mismatch");
  require(bf16(-0.0f) == 0x8000 && bf16(1.0f + std::ldexp(1.0f, -8)) == 0x3f80, "BF16 precision boundary mismatch");
  for (uint32_t i = 0; i < 5; ++i) require(*c::hcDownF32ModeName(static_cast<c::HCDownF32Mode>(i)), "invalid mode");
  require(!audit::cellRelation(0x3f81, 0x3f80, 1.0, 1e-8).pass, "false midpoint compatibility");
}
} // namespace
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"host_command_timing_size_bytes\":" << sizeof(CommandTiming)
            << ",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance << "}\n"; return 0;
      }
      require(argc == 4, "usage: flash-hc-down-f32-oracle PRIVATE_METALLIB PACKAGE REPORT_JSON");
      const auto selected = prefixes(); std::vector<std::string> downNames;
      for (const auto &p : selected) downNames.push_back(p + ".input_mix_weight_down");
      const auto rows = list("FLASH_HC_DOWN_F32_ROWS", {4, 8, 16}, 16), modes = list("FLASH_HC_DOWN_F32_MODES", {0, 1, 2, 3, 4}, 4);
      const uint32_t pairs = list("FLASH_HC_DOWN_F32_PAIRS", {6}, 32)[0]; require(pairs > 0, "pairs must be positive");
      const auto pipelines = metadata(argv[1]); MetalBackend backend(argv[1]); const auto weights = FlashWeights::load(backend, argv[2]);
      FlashFloatDenseCache cache(backend, weights, downNames);
      for (const auto &p : downNames) coefficientProof(weights.projection(p), cache.tensor(p));
      const auto digest = backend.metallibSha256(); std::ostringstream libSHA; for (uint8_t v : digest) libSHA << std::hex << std::setfill('0') << std::setw(2) << unsigned(v);
      const std::filesystem::path temporary = std::string(argv[3]) + ".partial"; std::ofstream out(temporary); require(bool(out), "report create failed");
      out << std::setprecision(17) << "{\"schema\":\"flash-hc-down-original-f32-cache-private-v1\",\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_identity\":" << splash::json::quote(weights.manifestFingerprint()) << ",\"cache_identity\":" << splash::json::quote(cache.identitySha256())
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libSHA.str()) << ",\"host_command_timing_size_bytes\":" << sizeof(CommandTiming)
          << ",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance << ",\"pipeline_metadata\":" << pipelines
          << ",\"coefficient_bits_checked\":" << selected.size() * 320ULL * 10240 << ",\"coefficient_dtype\":\"original-F32-not-demoted\",\"strict_relative_l2_limit\":0.0001,\"midpoint_evidence_changes_acceptance\":false,\"timing_scope\":\"complete down+BF16 activation+literal injection; candidate MPP includes padding/partials/fold; alternating matched commands\",\"cases\":[";
      bool first = true, pass = true;
      for (const auto &p : selected) for (uint32_t r : rows) for (uint32_t mode : modes) {
        require(r == 4 || r == 8 || r == 16, "rows must be4/8/16"); if (!first) out << ','; first = false;
        pass = run(backend, weights, cache, p, r, static_cast<c::HCDownF32Mode>(mode), pairs, out) && pass;
      }
      out << "],\"pass\":" << (pass ? "true" : "false") << "}\n"; out.flush(); require(bool(out), "report write failed"); out.close();
      std::filesystem::rename(temporary, argv[3]); std::cout << "{\"pass\":" << (pass ? "true" : "false") << ",\"report\":" << splash::json::quote(argv[3]) << "}\n"; return pass ? 0 : 1;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
