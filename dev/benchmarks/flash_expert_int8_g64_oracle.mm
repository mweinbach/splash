// Private root-run signed INT8 saved-coefficient expert qualification/performance oracle.
// --cpu-self-test and compilation create no backend and submit no GPU work.
#include "flash/FlashMoEBlocked.hpp"
#include "flash_expert_int8_candidate.h"
#include "engine/MemoryGovernor.hpp"
#include "flash/FlashMoE.hpp"
#include "engine/Json.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "flash_expert_int8_bucket_reference.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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
#include <string_view>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::CommandTiming;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace ref = splash::flash::int8_bucket_reference;
constexpr uint32_t kSelections = 10, kSticky = 0x80000000u;
constexpr uint64_t kGuardBytes = 64;

void require(bool value, const std::string &reason) {
  if (!value) throw std::runtime_error(reason);
}
void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
void cpuSelfTest() {
  const std::array<uint32_t, 2> words{0x76543210u, 0xfedcba98u};
  const std::array<uint16_t, 16> golden{0xbf00, 0xbe80, 0x0000, 0x3e80,
      0x3f00, 0x3f40, 0x3f80, 0x3fa0, 0x3fc0, 0x3fe0, 0x4000, 0x4010,
      0x4020, 0x4030, 0x4040, 0x4050};
  for (uint32_t k = 0; k < 16; ++k) {
    const uint32_t code = (words[k / 8] >> ((k % 8) * 4)) & 15u;
    require(bf16(float(code) * 0.25f - 0.5f) == golden[k],
        "handwritten Q4/BF16 coefficient golden differs");
    const uint16_t signedGolden = k == 2 ? 0 : uint16_t(golden[k] ^ 0x8000u);
    require(bf16(float(code) * -0.25f + 0.5f) == signedGolden,
        "signed scale golden differs");
  }
  require(moEBucketJobCapacity(512, 10, 16) == 831, "512-row job bound differs");
  require(moEBucketJobCapacity(2048, 10, 32) == 1151, "2048-row job bound differs");
}
uint32_t envNumber(const char *name, uint32_t fallback, uint32_t limit) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  require(*raw && *raw != '-', std::string("invalid ") + name);
  size_t consumed = 0;
  const unsigned long parsed = std::stoul(raw, &consumed);
  require(consumed == std::strlen(raw) && parsed > 0 && parsed <= limit,
      std::string("invalid ") + name);
  return uint32_t(parsed);
}

std::string pipelineMetadata(const char *path) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  require(device != nil, "Metal device unavailable");
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
  require(library != nil, "combined candidate library unavailable");
  std::ostringstream out;
  out << "{\"device_max_threadgroup_memory_bytes\":" << device.maxThreadgroupMemoryLength
      << ",\"pipelines\":[";
  bool first = true;
  std::vector<std::string> names;
  for (uint32_t m : {16u, 32u, 64u})
    for (const char *phase : {"gate_up", "down_scatter"})
      names.push_back(std::string("flash_expert_int8_g64_") + phase + "_m" + std::to_string(m) +
          (m == 64 ? "_n64_sg8" : "_n64"));
  for (const auto &name : names) {
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "candidate function missing: " + name);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "candidate pipeline creation failed: " + name);
    const uint32_t requested = name.ends_with("sg8") ? 256 : 128;
    require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= requested &&
        pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
        "candidate pipeline threadgroup limits exceeded: " + name);
    if (!first) out << ',';
    first = false;
    out << "{\"name\":" << splash::json::quote(name) << ",\"execution_width\":"
        << pipeline.threadExecutionWidth << ",\"maximum_threads\":" << pipeline.maxTotalThreadsPerThreadgroup
        << ",\"static_threadgroup_memory_bytes\":" << pipeline.staticThreadgroupMemoryLength << '}';
#if !__has_feature(objc_arc)
    [pipeline release]; [function release];
#endif
  }
  out << "]}";
#if !__has_feature(objc_arc)
  [library release]; [device release];
#endif
  return out.str();
}

template<class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values,
                                     const char *label) {
  auto result = backend.allocateBuffer(values.size() * sizeof(T), BufferStorage::Shared, label);
  std::memcpy(result.contents(), values.data(), values.size() * sizeof(T));
  return result;
}
template<class T> std::vector<T> readFile(const char *path, uint64_t elements) {
  require(std::filesystem::file_size(path) == elements * sizeof(T),
      std::string("raw fixture file has wrong size: ") + path);
  std::vector<T> result(elements);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(T));
  require(bool(in), std::string("raw fixture file read failed: ") + path);
  return result;
}
struct Guard final {
  MetalBuffer allocation;
  uint64_t logicalBytes;
  bool clean() const {
    const auto *bytes = static_cast<const uint8_t *>(allocation.contents());
    return std::all_of(bytes + logicalBytes, bytes + logicalBytes + kGuardBytes,
        [](uint8_t byte) { return byte == 0x5a; });
  }
};
MetalBuffer guarded(MetalBackend &backend, uint64_t bytes, std::vector<Guard> &guards) {
  auto buffer = backend.allocateBuffer(bytes + kGuardBytes, BufferStorage::Shared,
      "Q4x8 oracle guarded output");
  std::memset(buffer.contents(), 0xa5, bytes);
  std::memset(static_cast<uint8_t *>(buffer.contents()) + bytes, 0x5a, kGuardBytes);
  guards.push_back({buffer, bytes});
  return backend.view(buffer, 0, bytes);
}
void sourceAlignment(const FlashAffineProjection &p) {
  require(p.bits == 4 && p.groupSize == 64 && p.experts == 512 &&
      p.weightRowStrideBytes % 4 == 0 && p.weightExpertStrideBytes % 4 == 0 &&
      p.weights && p.weights->buffer.contents() &&
      reinterpret_cast<uintptr_t>(p.weights->buffer.contents()) % 4 == 0,
      "Q4x8 candidate requires aligned original Q4/G64 expert planes");
}

template<class T> void expected(const MetalBuffer &buffer, const std::vector<T> &values,
                               const char *label) {
  require(buffer.sizeBytes() >= values.size() * sizeof(T) &&
      std::memcmp(buffer.contents(), values.data(), values.size() * sizeof(T)) == 0,
      std::string("independent bucket comparison failed: ") + label);
}
void checkBuckets(const FlashMoEBlockedScratch &scratch, const ref::Packed &packed,
                  const ref::Jobs &jobs) {
  require(std::memcmp(scratch.buckets.counts.contents(), packed.counts.data(), 512 * 4) == 0,
      "independent counts differ");
  require(std::memcmp(scratch.buckets.offsets.contents(), packed.offsets.data(), 513 * 4) == 0,
      "independent offsets differ");
  expected(scratch.buckets.routeMap, packed.routeMap, "stable route map");
  expected(scratch.buckets.canonicalToPacked, packed.canonicalToPacked, "inverse route map");
  expected(scratch.buckets.packedInputs, packed.inputs, "BF16 bit-copy inputs");
  require(*static_cast<const uint32_t *>(scratch.buckets.jobCount.contents()) == jobs.count,
      "independent active job count differs");
  require(std::memcmp(scratch.buckets.jobOffsets.contents(), jobs.offsets.data(), 513 * 4) == 0,
      "independent job offsets differ");
  const auto *actualJobs = static_cast<const FlashMoEBucketJob *>(scratch.buckets.tileJobs.contents());
  for (uint32_t index = 0; index < jobs.entries.size(); ++index)
    require(actualJobs[index].expert == jobs.entries[index].expert &&
        actualJobs[index].row_begin == jobs.entries[index].rowBegin,
        "independent active/inactive job records differ at " + std::to_string(index));
}
struct Comparison final {
  uint64_t elements = 0, mismatches = 0;
  double relativeL2 = 0, cosine = 1, maxAbs = 0;
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"relative_l2\":" << relativeL2 << ",\"cosine\":" << cosine << ",\"max_abs\":" << maxAbs << '}';
  }
};
Comparison compareNumerical(const MetalBuffer &a, const MetalBuffer &b, uint64_t elements) {
  const auto *av = static_cast<const uint16_t *>(a.contents());
  const auto *bv = static_cast<const uint16_t *>(b.contents());
  require(a.sizeBytes() >= elements * 2 && b.sizeBytes() >= elements * 2, "comparison exceeds buffer views");
  Comparison result; result.elements = elements;
  double diff = 0, normA = 0, normB = 0, dot = 0;
  for (uint64_t i = 0; i < elements; ++i) {
    const double x = number(av[i]), y = number(bv[i]);
    require(std::isfinite(x) && std::isfinite(y), "nonfinite expert-chain output");
    result.mismatches += av[i] != bv[i];
    const double delta = x - y;
    diff += delta * delta; normA += x * x; normB += y * y; dot += x * y;
    result.maxAbs = std::max(result.maxAbs, std::abs(delta));
  }
  result.relativeL2 = normA ? std::sqrt(diff / normA) : (diff ? std::numeric_limits<double>::infinity() : 0);
  result.cosine = normA && normB ? dot / std::sqrt(normA * normB) : (normA == normB ? 1 : 0);
  return result;
}

std::string digest(const MetalBuffer &buffer) {
  require(buffer.sizeBytes() <= std::numeric_limits<CC_LONG>::max(), "hash extent overflow");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> sha{};
  require(CC_SHA256(buffer.contents(), CC_LONG(buffer.sizeBytes()), sha.data()), "hash failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : sha) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}

struct OfflineExpertInt8 final {
  std::array<MetalBuffer, 3> codes, scales;
  MetalBuffer ranks;
  std::vector<uint32_t> ids;
  std::array<std::string, 6> hashes;
  std::string source;
  explicit OfflineExpertInt8(MetalBackend &backend, const char *directory,
      const std::string &expectedSource, const std::string &expectedPrefix) {
    const auto base = std::filesystem::path(directory);
    std::ifstream file(base / "manifest.json"); require(bool(file), "INT8 manifest missing");
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    NSData *data = [NSData dataWithBytes:text.data() length:text.size()];
    NSError *error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([object isKindOfClass:[NSDictionary class]], "INT8 manifest must be an object");
    NSDictionary *manifest = object;
    require([manifest[@"schema"] isEqual:@"splash-flash-expert-int8-pilot-v1"] &&
        [manifest[@"quantization_format"] isEqual:@"signed-symmetric-int8-g64-f32-scale"] &&
        [manifest[@"scale_precision"] isEqual:@"F32"], "unsupported INT8 payload format");
    NSString *identity = manifest[@"source_identity_sha256"];
    require([identity isKindOfClass:[NSString class]], "INT8 manifest source identity missing");
    source = identity.UTF8String;
    require(source == expectedSource, "INT8 payload source identity differs");
    NSArray *selected = manifest[@"expert_ids"];
    require([selected isKindOfClass:[NSArray class]] && selected.count && selected.count <= 512,
        "INT8 manifest selected expert IDs missing");
    for (id entry in selected) {
      require([entry isKindOfClass:[NSNumber class]] &&
          CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(), "invalid expert ID type");
      const double value = [entry doubleValue];
      require(std::isfinite(value) && value == std::floor(value) && value >= 0 && value < 512,
          "invalid INT8 expert ID");
      ids.push_back(uint32_t(value));
    }
    require(std::is_sorted(ids.begin(), ids.end()) &&
        std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "selected IDs must be sorted and unique");
    std::vector<uint32_t> rank(512, UINT32_MAX);
    for (uint32_t i = 0; i < ids.size(); ++i) rank[ids[i]] = i;
    ranks = upload(backend, rank, "offline INT8 expert rank map");
    for (uint32_t plane = 0; plane < 3; ++plane) {
      const std::string name = plane == 0 ? "gate_proj" : plane == 1 ? "up_proj" : "down_proj";
      const uint64_t n = plane == 2 ? 2560 : 640, k = plane == 2 ? 640 : 2560;
      NSDictionary *matrix = manifest[@"matrices"][[NSString stringWithUTF8String:name.c_str()]];
      require([matrix isKindOfClass:[NSDictionary class]] &&
          [matrix[@"source_prefix"] isEqual:[NSString stringWithUTF8String:(expectedPrefix + "." + name).c_str()]],
          "INT8 payload source layer/plane differs");
      const auto c = readFile<int8_t>((base / "raw-planes" / (name + ".codes.i8.bin")).c_str(), ids.size() * n * k);
      const auto s = readFile<float>((base / "raw-planes" / (name + ".scales.f32.bin")).c_str(), ids.size() * n * (k / 64));
      require(std::none_of(c.begin(), c.end(), [](int8_t v) { return v == INT8_MIN; }),
          "INT8 payload contains excluded -128 code");
      require(std::all_of(s.begin(), s.end(), [](float v) { return std::isfinite(v) && v > 0; }),
          "INT8 payload has nonfinite/nonpositive scale");
      codes[plane] = upload(backend, c, "persisted signed INT8 expert coefficients");
      scales[plane] = upload(backend, s, "persisted F32 row scales");
      hashes[plane * 2] = digest(codes[plane]); hashes[plane * 2 + 1] = digest(scales[plane]);
      require([matrix[@"raw_codes_plane"][@"sha256"] isEqual:
          [NSString stringWithUTF8String:hashes[plane * 2].c_str()]] &&
          [matrix[@"raw_scales_plane"][@"sha256"] isEqual:
          [NSString stringWithUTF8String:hashes[plane * 2 + 1].c_str()]], "INT8 payload SHA256 differs");
    }
  }
  std::span<const uint32_t> selectedExpertIDs() const { return ids; }
  bool canariesIntact() const {
    for (uint32_t i = 0; i < 3; ++i)
      if (digest(codes[i]) != hashes[i * 2] || digest(scales[i]) != hashes[i * 2 + 1]) return false;
    return true;
  }
  void addGateUp(CommandGraph &graph, const FlashMoEBlockedScratch &scratch,
      MetalBuffer diagnostic, uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
    const uint32_t m = uint32_t(tile), capacity = moEBucketJobCapacity(rows, selections, m);
    const FlashExpertInt8Params p{rows, selections, rows * selections, capacity, m, uint32_t(ids.size()), 64, 0};
    const std::string suffix = "_m" + std::to_string(m) + (m == 64 ? "_n64_sg8" : "_n64");
    graph.add("flash_expert_int8_g64_gate_up" + suffix,
        {scratch.buckets.packedInputs, codes[0], scales[0], codes[1], scales[1], ranks,
         scratch.buckets.offsets, scratch.buckets.tileJobs, scratch.buckets.jobCount,
         scratch.packedActivated, diagnostic}, p, {10, capacity, 1}, {m == 64 ? 256u : 128u, 1, 1});
  }
  void addDownScatter(CommandGraph &graph, const FlashMoEBlockedScratch &scratch,
      MetalBuffer diagnostic, uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
    const uint32_t m = uint32_t(tile), capacity = moEBucketJobCapacity(rows, selections, m);
    const FlashExpertInt8Params p{rows, selections, rows * selections, capacity, m, uint32_t(ids.size()), 64, 0};
    const std::string suffix = "_m" + std::to_string(m) + (m == 64 ? "_n64_sg8" : "_n64");
    graph.add("flash_expert_int8_g64_down_scatter" + suffix,
        {scratch.packedActivated, codes[2], scales[2], ranks, scratch.buckets.offsets,
         scratch.buckets.tileJobs, scratch.buckets.jobCount, scratch.buckets.routeMap,
         scratch.scatteredDown, diagnostic}, p, {40, capacity, 1}, {m == 64 ? 256u : 128u, 1, 1});
  }
};

void times(std::ostream &out, const std::vector<CommandTiming> &values, bool gpu) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << (gpu ? values[i].gpuSeconds : values[i].wallSeconds) * 1000;
  }
  out << ']';
}

void runCase(MetalBackend &backend, const FlashWeights &weights, const std::string &prefix,
             uint32_t rows, uint32_t m, const std::string &pattern, uint32_t pairs,
             const OfflineExpertInt8 &cache, std::ostream &out) {
  std::vector<uint16_t> hidden(uint64_t{rows} * 2560);
  std::vector<int64_t> ids(uint64_t{rows} * kSelections);
  const char *rawInput = std::getenv("FLASH_EXPERT_INT8_INPUT");
  const char *rawIDs = std::getenv("FLASH_EXPERT_INT8_ROUTE_IDS");
  require(bool(rawInput) == bool(rawIDs), "raw input and IDs must be supplied together");
  if (rawInput) {
    hidden = readFile<uint16_t>(rawInput, hidden.size());
    ids = readFile<int64_t>(rawIDs, ids.size());
  } else {
    for (uint64_t i = 0; i < hidden.size(); ++i)
      hidden[i] = bf16(float(int((i * 73 + i / 2560 * 17) % 257) - 128) / 512.0f);
    const auto selected = cache.selectedExpertIDs();
    require(selected.size() >= 10, "synthetic patterns require at least10 stored experts");
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t slot = 0; slot < kSelections; ++slot)
        ids[uint64_t{row} * kSelections + slot] = pattern == "concentrated" ? selected[slot] :
            selected[(row * 7 + slot * 3) % selected.size()];
  }
  for (int64_t id : ids)
    require(std::binary_search(cache.ids.begin(), cache.ids.end(), uint32_t(id)),
        "fixture route selects an unstored INT8 expert");
  for (uint16_t value : hidden) require(std::isfinite(number(value)), "nonfinite fixture input");
  const auto packed = ref::pack(hidden, ids, rows, kSelections, kSticky);
  require(packed.diagnostic == kSticky, "fixture has invalid or duplicate route IDs");
  const auto jobs = ref::makeJobs(packed, m);
  const auto &gate = weights.projection(prefix + ".gate_proj");
  const auto &up = weights.projection(prefix + ".up_proj");
  const auto &down = weights.projection(prefix + ".down_proj");
  sourceAlignment(gate); sourceAlignment(up); sourceAlignment(down);
  const auto input = upload(backend, hidden, "Q4x8 hidden");
  const auto expertIDs = upload(backend, ids, "Q4x8 original route IDs");
  const auto route = upload(backend, std::vector<uint16_t>(ids.size(), bf16(0.1f)), "Q4x8 route weights");
  const auto shared = upload(backend, std::vector<uint16_t>(uint64_t{rows} * 2560, 0), "Q4x8 shared zeros");
  const auto sharedGate = upload(backend, std::vector<uint16_t>(rows, 0), "Q4x8 shared gate zeros");
  const auto diagnostic = upload(backend, std::vector<uint32_t>{kSticky}, "Q4x8 sticky diagnostic");
  std::array<FlashMoEBlockedScratch, 2> scratch;
  std::array<CommandGraph, 2> graphs;
  std::array<MetalBuffer, 2> output;
  std::vector<Guard> guards;
  const auto tile = static_cast<FlashMoEBlockedTile>(m);
  for (uint32_t i = 0; i < 2; ++i) {
    scratch[i] = allocateMoEBlockedScratch(backend, rows);
    scratch[i].packedActivated = guarded(backend, uint64_t{rows} * kSelections * 640 * 2, guards);
    scratch[i].scatteredDown = guarded(backend, uint64_t{rows} * kSelections * 2560 * 2, guards);
    output[i] = guarded(backend, uint64_t{rows} * 2560 * 2, guards);
    addMoEBlockedPack(graphs[i], input, expertIDs, scratch[i], diagnostic, rows, tile);
    if (i == 0) {
      addMoEBlockedGateUp(graphs[i], gate, up, scratch[i], diagnostic, rows, tile);
      addMoEBlockedDownScatter(graphs[i], down, scratch[i], diagnostic, rows, tile);
      const auto dispatches = graphs[i].dispatches();
      require(std::count_if(dispatches.begin(), dispatches.end(), [](const ComputeDispatch &d) {
          return d.pipelineName.starts_with("flash_moe_q4x8_gate_up_") ||
                 d.pipelineName.starts_with("flash_moe_q4x8_down_scatter_");
        }) == 2, "control must use the qualified current Q4x8 producer");
    } else {
      cache.addGateUp(graphs[i], scratch[i], diagnostic, rows, tile, kSelections);
      cache.addDownScatter(graphs[i], scratch[i], diagnostic, rows, tile, kSelections);
    }
    addCombine(graphs[i], scratch[i].scatteredDown, expertIDs, route, shared, sharedGate,
        output[i], diagnostic, rows, 2560, 512, kSelections);
  }
  const auto candidate = graphs[1].dispatches();
  (void)backend.submitCommand(graphs[0].dispatches());
  (void)backend.submitCommand(candidate);
  checkBuckets(scratch[0], packed, jobs); checkBuckets(scratch[1], packed, jobs);
  uint32_t wholeKJobs = 0;
  const auto hot = cache.selectedExpertIDs();
  for (uint32_t i = 0; i < jobs.count; ++i) {
    const auto &job = jobs.entries[i];
    if (std::binary_search(hot.begin(), hot.end(), job.expert) && packed.offsets[job.expert + 1] - job.rowBegin >= m)
      ++wholeKJobs;
  }
  const auto activation = compareNumerical(scratch[0].packedActivated, scratch[1].packedActivated,
      uint64_t{rows} * kSelections * 640);
  const auto downComparison = compareNumerical(scratch[0].scatteredDown, scratch[1].scatteredDown,
      uint64_t{rows} * kSelections * 2560);
  const auto combined = compareNumerical(output[0], output[1], uint64_t{rows} * 2560);
  const auto healthy = [&] {
    require(*static_cast<const uint32_t *>(diagnostic.contents()) == kSticky,
        "sticky diagnostics changed or numerical/shape failure occurred");
    for (const auto &guard : guards) require(guard.clean(), "output canary was overwritten");
    require(cache.canariesIntact(), "immutable cache canary was overwritten");
  };
  healthy();
  std::array<std::vector<CommandTiming>, 2> timing;
  for (uint32_t pair = 0; pair < pairs; ++pair) {
    for (uint32_t order = 0; order < 2; ++order) {
      const uint32_t which = (pair + order) % 2;
      timing[which].push_back(backend.submitCommand(which == 0 ? graphs[0].dispatches()
          : std::span<const ComputeDispatch>(candidate)));
      require(*static_cast<const uint32_t *>(diagnostic.contents()) == kSticky, "sticky diagnostics changed during timing");
    }
  }
  (void)compareNumerical(output[0], output[1], uint64_t{rows} * 2560);
  const uint32_t activeExperts = uint32_t(std::count_if(packed.counts.begin(), packed.counts.end(),
      [](uint32_t count) { return count != 0; }));
  out << "{\"rows\":" << rows << ",\"tile_m\":" << m << ",\"pattern\":"
      << splash::json::quote(rawInput ? "raw-fixture" : pattern)
      << ",\"active_experts\":" << activeExperts << ",\"active_jobs\":" << jobs.count
      << ",\"job_capacity\":" << moEBucketJobCapacity(rows, kSelections, m)
      << ",\"row_utilization\":" << double(rows * kSelections) / double(jobs.count * m)
      << ",\"full_tile_jobs\":" << wholeKJobs << ",\"full_tile_routes\":" << wholeKJobs * m
      << ",\"full_tile_route_fraction\":" << double(wholeKJobs * m) / double(rows * kSelections)
      << ",\"whole_k_routes\":" << rows * kSelections
      << ",\"activation\":"; activation.write(out);
  out << ",\"down\":"; downComparison.write(out);
  out << ",\"combine\":"; combined.write(out);
  out << ",\"canaries_clean\":true,\"control_gpu_ms\":";
  times(out, timing[0], true); out << ",\"candidate_gpu_ms\":"; times(out, timing[1], true);
  out << ",\"control_wall_ms\":"; times(out, timing[0], false);
  out << ",\"candidate_wall_ms\":"; times(out, timing[1], false); out << '}';
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":[\"signed_source_BF_coefficients\",\"BF16_RNE\",\"job_bounds\"]}\n";
        return 0;
      }
      require(argc == 5, "usage: flash-expert-int8-oracle COMBINED_METALLIB SOURCE_PACKAGE INT8_DIRECTORY REPORT_JSON");
      const uint32_t pairs = envNumber("FLASH_EXPERT_INT8_PAIRS", 4, 32);
      const uint32_t rows = envNumber("FLASH_EXPERT_INT8_ROWS", 2048, 8192);
      const uint32_t m = envNumber("FLASH_EXPERT_INT8_TILE", 32, 64);
      require(m == 16 || m == 32 || m == 64, "tile must be16/32/64");
      const char *selectedPrefix = std::getenv("FLASH_EXPERT_INT8_PREFIX");
      const std::string prefix = selectedPrefix ? selectedPrefix : "language_model.model.layers.0.mlp.switch_mlp";
      require(setenv("SPLASH_FLASH_MOE_Q4X8", "1", 1) == 0, "cannot force Q4x8 control policy");
      require(setenv("SPLASH_FLASH_MOE_M64", "1", 1) == 0, "cannot enable current M64 qualified control");
      const std::string metadata = pipelineMetadata(argv[1]);
      MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      const OfflineExpertInt8 cache(backend, argv[3], weights.sourceIdentity(), prefix);
      std::ofstream out(argv[4]); require(bool(out), "cannot create report");
      out << std::setprecision(12) << "{\"schema\":\"flash-offline-signed-int8-G64-expert-oracle-v1\",\"pass\":true,\"source_identity\":"
          << splash::json::quote(weights.sourceIdentity()) << ",\"manifest_identity\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"prefix\":" << splash::json::quote(prefix) << ",\"pipeline_metadata\":" << metadata
          << ",\"policy\":\"offline_signed_int8_G64_F32scale_BFactivation_FP32accum_BFstages\",\"numerical_alternative\":true"
          << ",\"validation_scope\":\"finite producer, exact bucket ownership, payload immutability and guard regions; numerical error measured without a model-quality claim\""
          << ",\"timing_scope\":\"complete expert chain, warm alternating matched GPU commands, offline conversion excluded\",\"pairs\":"
          << pairs << ",\"stored_experts\":" << cache.ids.size() << ",\"payload_sha256\":[";
      for (size_t i = 0; i < cache.hashes.size(); ++i) { if (i) out << ','; out << splash::json::quote(cache.hashes[i]); }
      out << "],\"cases\":[";
      bool firstCase = true;
      for (const char *pattern : {"concentrated", "spread"}) {
        if (!firstCase) out << ','; firstCase = false;
        runCase(backend, weights, prefix, rows, m, pattern, pairs, cache, out);
        if (std::getenv("FLASH_EXPERT_INT8_INPUT")) break;
      }
      require(cache.canariesIntact(), "immutable offline coefficient bytes changed");
      out << "]}\n"; out.flush(); require(bool(out), "report write failed");
      std::cout << "{\"pass\":true,\"report\":" << splash::json::quote(argv[4]) << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
