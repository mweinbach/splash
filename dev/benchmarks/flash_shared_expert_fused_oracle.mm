// Root-run private oracle. Compilation and --cpu-self-test create no backend.
#include "FlashSharedExpertFused.hpp"
#include "FlashSharedExpertHostProvenance.hpp"
#include "flash/FlashMoE.hpp"
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
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace splash::flash;
namespace fused = splash::flash::candidate;
using namespace splash::metal;
constexpr uint32_t kSticky = 0x40000000u;
constexpr uint64_t kGuardBytes = 64;
void require(bool value, const std::string &reason) {
  if (!value) throw std::runtime_error(reason);
}
uint16_t bf16(float value) {
  const uint32_t w = std::bit_cast<uint32_t>(value);
  if ((w & 0x7f800000u) == 0x7f800000u)
    return uint16_t((w >> 16) | ((w & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
std::string hash(const void *data, uint64_t bytes) {
  CC_SHA256_CTX context{}; require(CC_SHA256_Init(&context), "SHA init failed");
  const auto *next = static_cast<const uint8_t *>(data);
  while (bytes) {
    const auto size = static_cast<CC_LONG>(std::min<uint64_t>(bytes, 1ULL << 30));
    require(CC_SHA256_Update(&context, next, size), "SHA update failed");
    next += size; bytes -= size;
  }
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  require(CC_SHA256_Final(output.data(), &context), "SHA final failed");
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (uint8_t v : output) { result += digits[v >> 4]; result += digits[v & 15]; }
  return result;
}
std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> fallback,
    uint32_t limit) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  std::stringstream stream(raw); std::string item; std::vector<uint32_t> result;
  while (std::getline(stream, item, ',')) {
    require(!item.empty() && item.front() != '-', std::string("invalid ") + name);
    size_t used = 0; const auto value = std::stoul(item, &used);
    require(used == item.size() && value > 0 && value <= limit, std::string("invalid ") + name);
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), std::string("empty ") + name); return result;
}
FlashAffineMPPTile tile(uint32_t m, uint32_t n) {
  if (m == 16 && n == 64) return FlashAffineMPPTile::M16N64;
  if (m == 16 && n == 128) return FlashAffineMPPTile::M16N128;
  if (m == 32 && n == 64) return FlashAffineMPPTile::M32N64;
  if (m == 32 && n == 128) return FlashAffineMPPTile::M32N128;
  throw std::invalid_argument("M/N must be16|32 and64|128");
}
void cpuSelfTest() {
  require(bf16(0.0f) == 0 && bf16(-0.0f) == 0x8000 &&
      bf16(std::numeric_limits<float>::quiet_NaN()) == 0x7fc0, "BF16 edge conversion differs");
  require(fused::sharedExpertFusedTileRows(tile(16, 64)) == 16 &&
      fused::sharedExpertFusedTileOutputs(tile(16, 64)) == 64 &&
      fused::sharedExpertFusedTileRows(tile(32, 128)) == 32, "descriptor geometry differs");
  require(8191 / 16 * 16 == 8176 && 8191 % 16 == 15 &&
      2049 / 16 * 16 == 2048 && 257 % 16 == 1, "tail windows differ");
  bool rejected = false;
  try { (void)fused::sharedExpertFusedTileRows(FlashAffineMPPTile::M8N64); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "small-row descriptor guard differs");
}
std::string metadata(const char *path, uint32_t m, uint32_t n) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal unavailable");
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
  require(library != nil, "private library unavailable");
  std::ostringstream out;
  out << "{\"device_max_threadgroup_memory_bytes\":" << device.maxThreadgroupMemoryLength << ",\"pipelines\":[";
  bool first = true;
  for (const char *phase : {"", "taps_"}) {
    const std::string name = "flash_shared_expert_fused_" + std::string(phase)
        + "m" + std::to_string(m) + "_n" + std::to_string(n);
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "private function unavailable: " + name);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function error:&error];
    require(p != nil && p.threadExecutionWidth == 32 && p.maxTotalThreadsPerThreadgroup >= 128 &&
        p.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
        "private pipeline exceeds device limits: " + name);
    if (!first) out << ','; first = false;
    out << "{\"name\":" << splash::json::quote(name) << ",\"static_threadgroup_memory_bytes\":"
        << p.staticThreadgroupMemoryLength << ",\"maximum_threads\":" << p.maxTotalThreadsPerThreadgroup << '}';
  }
  out << "]}"; return out.str();
}
struct Guard {
  MetalBuffer allocation; uint64_t bytes;
  bool clean() const {
    const auto *p = static_cast<const uint8_t *>(allocation.contents()) + bytes;
    return std::all_of(p, p + kGuardBytes, [](uint8_t v) { return v == 0x5a; });
  }
};
MetalBuffer guarded(MetalBackend &backend, uint64_t bytes, std::vector<Guard> &guards) {
  auto allocation = backend.allocateBuffer(bytes + kGuardBytes, BufferStorage::Shared,
      "private shared expert guard");
  std::memset(allocation.contents(), 0xa5, bytes);
  std::memset(static_cast<uint8_t *>(allocation.contents()) + bytes, 0x5a, kGuardBytes);
  guards.push_back({allocation, bytes}); return backend.view(allocation, 0, bytes);
}
void equal(const MetalBuffer &a, const MetalBuffer &b, uint64_t bytes, const char *what) {
  require(a.sizeBytes() >= bytes && b.sizeBytes() >= bytes, "comparison extent differs");
  if (std::memcmp(a.contents(), b.contents(), bytes)) {
    const auto *x = static_cast<const uint16_t *>(a.contents());
    const auto *y = static_cast<const uint16_t *>(b.contents());
    for (uint64_t i = 0; i < bytes / 2; ++i)
      if (x[i] != y[i]) throw std::runtime_error(std::string("BF16 comparison differs: ") + what
          + " index=" + std::to_string(i) + " control=" + std::to_string(x[i])
          + " candidate=" + std::to_string(y[i]));
  }
}
void coefficients(const FlashWeights &weights, const FlashDenseCache &cache,
    const std::string &prefix) {
  for (const char *role : {"gate_proj", "up_proj", "down_proj"}) {
    const std::string name = prefix + "." + role;
    const auto &p = weights.projection(name); const auto &saved = cache.tensor(name);
    require(p.bits == 8 && p.groupSize == 128 && p.experts == 1 &&
        saved.dtype == FlashDType::BF16 && saved.shape == std::vector<uint64_t>{p.outputSize, p.inputSize},
        "source shared expert/cache geometry differs");
    const auto *actual = static_cast<const uint16_t *>(saved.buffer.contents());
    for (uint32_t n = 0; n < p.outputSize; ++n) {
      const auto *w = static_cast<const uint8_t *>(p.weights->buffer.contents()) + uint64_t{n} * p.weightRowStrideBytes;
      const auto *s = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.scales->buffer.contents())
          + uint64_t{n} * p.parameterRowStrideBytes);
      const auto *b = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.biases->buffer.contents())
          + uint64_t{n} * p.parameterRowStrideBytes);
      for (uint32_t k = 0; k < p.inputSize; ++k) {
        const float product = float(w[k]) * number(s[k / 128]);
        const uint16_t expected = bf16(product + number(b[k / 128]));
        require(std::isfinite(number(expected)) && actual[uint64_t{n} * p.inputSize + k] == expected,
            "cached original BF16 coefficient differs");
      }
    }
  }
}
void times(std::ostream &out, const std::vector<CommandTiming> &values, bool gpu) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    require(std::isfinite(values[i].gpuSeconds) && values[i].gpuSeconds > 1e-9 &&
        std::isfinite(values[i].wallSeconds) && values[i].wallSeconds > 1e-9,
        "invalid/nonfinite/tiny timing; possible host CommandTiming ABI mismatch");
    if (i) out << ','; out << (gpu ? values[i].gpuSeconds : values[i].wallSeconds) * 1000;
  }
  out << ']';
}
void run(MetalBackend &backend, const FlashDenseCache &cache, const std::string &prefix,
    uint32_t rows, FlashAffineMPPTile selectedTile, uint32_t pairs, bool invalid,
    std::ostream &out) {
  const auto &g = cache.tensor(prefix + ".gate_proj"), &u = cache.tensor(prefix + ".up_proj");
  const auto &d = cache.tensor(prefix + ".down_proj");
  const std::array<std::string, 3> before{
      hash(g.buffer.contents(), g.logicalBytes), hash(u.buffer.contents(), u.logicalBytes),
      hash(d.buffer.contents(), d.logicalBytes)};
  std::vector<Guard> guards;
  const auto input = guarded(backend, uint64_t{rows} * 2560 * 2, guards);
  auto *hidden = static_cast<uint16_t *>(input.contents());
  const char *rawInput = std::getenv("FLASH_SHARED_FUSED_INPUT");
  if (rawInput) {
    require(std::filesystem::file_size(rawInput) == uint64_t{rows} * 2560 * 2, "raw BF16 input extent differs");
    std::ifstream stream(rawInput, std::ios::binary);
    stream.read(reinterpret_cast<char *>(hidden), input.sizeBytes()); require(bool(stream), "raw input read failed");
  } else for (uint64_t i = 0; i < uint64_t{rows} * 2560; ++i)
    hidden[i] = bf16(float(int((i * 73 + i / 2560 * 17) % 257) - 128) / 128.0f);
  for (uint64_t i = 0; i < uint64_t{rows} * 2560; ++i) require(std::isfinite(number(hidden[i])), "fixture input is nonfinite");
  const auto inputSHA = hash(input.contents(), input.sizeBytes());
  std::array<MetalBuffer, 3> activated, down, diagnostic;
  for (uint32_t i = 0; i < 3; ++i) {
    activated[i] = guarded(backend, uint64_t{rows} * 640 * 2, guards);
    down[i] = guarded(backend, uint64_t{rows} * 2560 * 2, guards);
    diagnostic[i] = guarded(backend, 4, guards);
    *static_cast<uint32_t *>(diagnostic[i].contents()) = kSticky;
  }
  const auto controlGate = guarded(backend, uint64_t{rows} * 640 * 2, guards);
  const auto controlUp = guarded(backend, uint64_t{rows} * 640 * 2, guards);
  const auto tapGate = guarded(backend, uint64_t{rows} * 640 * 2, guards);
  const auto tapUp = guarded(backend, uint64_t{rows} * 640 * 2, guards);
  const fused::SharedExpertFusedTail tail{
      guarded(backend, 31ULL * 640 * 2, guards), guarded(backend, 31ULL * 640 * 2, guards)};
  std::array<CommandGraph, 3> graph;
  cache.addProjection(graph[0], prefix + ".gate_proj", input, controlGate, diagnostic[0], rows, FlashAffineMPPTile::M16N64);
  cache.addProjection(graph[0], prefix + ".up_proj", input, controlUp, diagnostic[0], rows, FlashAffineMPPTile::M16N64);
  addSiLUMultiply(graph[0], controlGate, controlUp, activated[0], diagnostic[0], rows, 640);
  fused::addSharedExpertFused(backend, graph[1], g, u, input, activated[1], diagnostic[1], rows, selectedTile, tail);
  fused::addSharedExpertFusedTaps(backend, graph[2], g, u, input, activated[2], diagnostic[2], rows, selectedTile, tapGate, tapUp);
  for (uint32_t i = 0; i < 3; ++i)
    cache.addProjection(graph[i], prefix + ".down_proj", activated[i], down[i], diagnostic[i], rows, FlashAffineMPPTile::M32N128);
  for (uint32_t i = 0; i < 3; ++i) (void)backend.submitCommand(graph[i].dispatches());
  equal(controlGate, tapGate, uint64_t{rows} * 640 * 2, "rounded gate dots");
  equal(controlUp, tapUp, uint64_t{rows} * 640 * 2, "rounded up dots");
  const auto healthy = [&](uint32_t expected) {
    for (uint32_t i = 0; i < 3; ++i) require(*static_cast<uint32_t *>(diagnostic[i].contents()) == expected, "diagnostics differ");
    for (const auto &guard : guards) require(guard.clean(), "output/input canary changed");
    for (uint32_t i = 1; i < 3; ++i) {
      equal(activated[0], activated[i], uint64_t{rows} * 640 * 2, "canonical activated BF16");
      equal(down[0], down[i], uint64_t{rows} * 2560 * 2, "whole shared-expert chain");
    }
  };
  healthy(kSticky);
  uint32_t rejected = 0;
  const auto reject = [&](auto action) {
    CommandGraph negative; bool didReject = false;
    try { action(negative); } catch (const std::invalid_argument &) { didReject = true; }
    require(didReject && negative.dispatches().empty(), "host invalid candidate mutated graph before rejecting"); ++rejected;
  };
  for (uint32_t wrong : {1u, 16u, 255u, 8193u}) reject([&](auto &negative) {
    fused::addSharedExpertFused(backend, negative, g, u, input, activated[1], diagnostic[1], wrong, selectedTile, tail);
  });
  reject([&](auto &negative) {
    fused::addSharedExpertFused(backend, negative, g, u, input, input, diagnostic[1], rows, selectedTile, tail);
  });
  reject([&](auto &negative) {
    auto wrong = g; wrong.dtype = FlashDType::F32;
    fused::addSharedExpertFused(backend, negative, wrong, u, input, activated[1], diagnostic[1], rows, selectedTile, tail);
  });
  reject([&](auto &negative) {
    fused::addSharedExpertFused(backend, negative, g, u, g.buffer, activated[1], diagnostic[1], 256, selectedTile, tail);
  });
  if (rows % fused::sharedExpertFusedTileRows(selectedTile)) reject([&](auto &negative) {
    fused::addSharedExpertFused(backend, negative, g, u, input, activated[1], diagnostic[1], rows, selectedTile, {});
  });
  std::array<std::vector<CommandTiming>, 2> timing;
  for (uint32_t pair = 0; pair < pairs; ++pair)
    for (uint32_t order = 0; order < 2; ++order) {
      const uint32_t which = (pair + order) % 2;
      const auto t = backend.submitCommand(graph[which].dispatches());
      require(std::isfinite(t.gpuSeconds) && t.gpuSeconds > 1e-9 &&
          std::isfinite(t.wallSeconds) && t.wallSeconds > 1e-9,
          "invalid/nonfinite/tiny timing; possible host CommandTiming ABI mismatch");
      timing[which].push_back(t); healthy(kSticky);
    }
  uint32_t invalidCases = 0;
  if (invalid) for (uint16_t value : {uint16_t{0x7fc1}, uint16_t{0x7f80}, uint16_t{0xff80}}) {
    const uint16_t previous = hidden[0]; hidden[0] = value;
    for (uint32_t i = 0; i < 3; ++i) {
      *static_cast<uint32_t *>(diagnostic[i].contents()) = kSticky;
      (void)backend.submitCommand(graph[i].dispatches());
    }
    healthy(kSticky | 4); ++invalidCases; hidden[0] = previous;
  }
  require(hash(input.contents(), input.sizeBytes()) == inputSHA, "input changed");
  require(hash(g.buffer.contents(), g.logicalBytes) == before[0] &&
      hash(u.buffer.contents(), u.logicalBytes) == before[1] &&
      hash(d.buffer.contents(), d.logicalBytes) == before[2], "immutable cached weights changed");
  out << "{\"rows\":" << rows << ",\"candidate_m\":" << fused::sharedExpertFusedTileRows(selectedTile)
      << ",\"candidate_n\":" << fused::sharedExpertFusedTileOutputs(selectedTile)
      << ",\"control_m\":16,\"control_n\":64,\"input_policy\":"
      << splash::json::quote(rawInput ? "captured-raw-bf16" : "declared-deterministic-synthetic-bf16")
      << ",\"input_sha256\":" << splash::json::quote(inputSHA)
      << ",\"dot_planes_bit_exact\":true,\"activated_bit_exact\":true,\"full_chain_bit_exact\":true,\"canaries_clean\":true,\"immutable_inputs_preserved\":true,\"host_rejections\":"
      << rejected << ",\"nonfinite_input_cases\":" << invalidCases
      << ",\"control_dispatches\":" << graph[0].dispatches().size()
      << ",\"candidate_dispatches\":" << graph[1].dispatches().size()
      << ",\"eliminated_temporary_bytes\":" << uint64_t{rows} * 640 * 2 * 2
      << ",\"control_gpu_ms\":"; times(out, timing[0], true);
  out << ",\"candidate_gpu_ms\":"; times(out, timing[1], true);
  out << ",\"control_wall_ms\":"; times(out, timing[0], false);
  out << ",\"candidate_wall_ms\":"; times(out, timing[1], false); out << '}';
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"metal_device_created\":false,\"host_command_timing_size_bytes\":"
            << sizeof(CommandTiming) << ",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance
            << ",\"checks\":[\"BF16_edges\",\"descriptor_geometry\",\"row_tail_windows\",\"small_row_guard\"]}\n"; return 0;
      }
      require(argc == 4, "usage: flash-shared-expert-fused-oracle PRIVATE_METALLIB PACKAGE REPORT_JSON");
      const auto rows = list("FLASH_SHARED_FUSED_ROWS", {256, 257, 2048, 2049, 8191, 8192}, 8192);
      const auto ms = list("FLASH_SHARED_FUSED_M", {16}, 32), ns = list("FLASH_SHARED_FUSED_N", {64}, 128);
      require(ms.size() == 1 && ns.size() == 1, "one descriptor per oracle run");
      const auto selectedTile = tile(ms[0], ns[0]);
      const auto pairs = list("FLASH_SHARED_FUSED_PAIRS", {6}, 32)[0];
      const char *prefixRaw = std::getenv("FLASH_SHARED_FUSED_PREFIX");
      const std::string prefix = prefixRaw ? prefixRaw : "language_model.model.layers.0.mlp.shared_expert";
      require(prefix == "language_model.model.layers.0.mlp.shared_expert" ||
          prefix == "language_model.model.layers.47.mlp.shared_expert", "only inspected layer0/47 supported");
      const char *invalidRaw = std::getenv("FLASH_SHARED_FUSED_INVALID");
      require(!invalidRaw || std::string_view(invalidRaw) == "0" || std::string_view(invalidRaw) == "1", "INVALID must be0/1");
      const bool invalid = !invalidRaw || std::string_view(invalidRaw) == "1";
      const auto pipelineMetadata = metadata(argv[1], ms[0], ns[0]);
      MetalBackend backend(argv[1]); const auto weights = FlashWeights::load(backend, argv[2]);
      const std::array<std::string, 3> prefixes{prefix + ".gate_proj", prefix + ".up_proj", prefix + ".down_proj"};
      FlashDenseCache cache(backend, weights, prefixes); coefficients(weights, cache, prefix);
      const auto digest = backend.metallibSha256();
      std::ostringstream libSHA; for (uint8_t b : digest) libSHA << std::hex << std::setfill('0') << std::setw(2) << unsigned(b);
      std::ofstream out(argv[3]); require(bool(out), "report create failed");
      out << std::setprecision(12) << "{\"schema\":\"flash-shared-expert-fused-prefill-v1\",\"pass\":true,\"source_identity\":"
          << splash::json::quote(weights.sourceIdentity()) << ",\"manifest_identity\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"cache_identity\":" << splash::json::quote(cache.identitySha256())
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libSHA.str())
          << ",\"host_command_timing_size_bytes\":" << sizeof(CommandTiming)
          << ",\"host_object_provenance\":" << kFlashOracleHostObjectProvenance
          << ",\"prefix\":" << splash::json::quote(prefix)
          << ",\"coefficient_bits_checked\":4915200,\"semantics\":" << splash::json::quote(fused::kSharedExpertFusedSemantics)
          << ",\"pipeline_metadata\":" << pipelineMetadata
          << ",\"timing_scope\":\"complete shared gate+up+compiled-BF16-SwiGLU+down, alternating matched commands after untimed rounded-dot taps and warmup; diagnostic input cases outside timing\",\"cases\":[";
      bool first = true;
      for (uint32_t row : rows) {
        require(row >= 256, "rows must be256..8192");
        if (!first) out << ','; first = false;
        run(backend, cache, prefix, row, selectedTile, pairs, invalid, out);
      }
      out << "]}\n"; out.flush(); require(bool(out), "report write failed");
      std::cout << "{\"pass\":true,\"bit_exact\":true,\"report\":" << splash::json::quote(argv[3]) << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
