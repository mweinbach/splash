// Private Root-run exact/performance oracle for persistent v5 INT8/Q4 job producers.
// Compilation and --cpu-self-test create no Metal backend and submit no GPU work.
#include "flash/FlashInt8ExpertStore.hpp"
#include "FlashInt8PersistentJobs.hpp"
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "flash/FlashMoE.hpp"
#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "metal/abi/FlashMoEDirectA.h"
#include "metal/abi/FlashInt8ExpertStore.h"
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
#include <memory>
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
uint32_t envNumber(const char *name, uint32_t fallback, uint32_t limit) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  require(*raw && *raw != '-' && *raw != '+', std::string("invalid ") + name);
  size_t consumed = 0;
  const auto parsed = std::stoul(raw, &consumed);
  require(consumed == std::strlen(raw) && parsed > 0 && parsed <= limit,
      std::string("invalid ") + name);
  return uint32_t(parsed);
}
double envPositive(const char *name, double fallback, double limit) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  size_t consumed = 0;
  const double parsed = std::stod(raw, &consumed);
  require(consumed == std::strlen(raw) && std::isfinite(parsed) && parsed > 0 && parsed <= limit,
      std::string("invalid ") + name);
  return parsed;
}
std::vector<uint32_t> parseLayers() {
  const char *raw = std::getenv("FLASH_I8_JOBS_LAYERS");
  if (!raw) return {0, 47};
  std::vector<uint32_t> layers;
  std::stringstream parser(raw); std::string field;
  while (std::getline(parser, field, ',')) {
    require(!field.empty() && std::all_of(field.begin(), field.end(), [](char c) { return c >= '0' && c <= '9'; }),
        "invalid layer list");
    size_t consumed = 0; const auto value = std::stoul(field, &consumed);
    require(consumed == field.size() && value < 48 &&
        std::find(layers.begin(), layers.end(), value) == layers.end(), "invalid or duplicate layer");
    layers.push_back(uint32_t(value));
  }
  require(!layers.empty() && raw[std::strlen(raw) - 1] != ',', "empty or incomplete layer list");
  return layers;
}
std::string prefixFor(uint32_t layer) {
  return "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp";
}
std::vector<uint32_t> coldIDs(std::span<const uint32_t> hot) {
  require(std::is_sorted(hot.begin(), hot.end()) &&
      std::adjacent_find(hot.begin(), hot.end()) == hot.end(), "store selected IDs are not sorted and unique");
  std::vector<uint32_t> cold;
  for (uint32_t expert = 0; expert < 512; ++expert)
    if (!std::binary_search(hot.begin(), hot.end(), expert)) cold.push_back(expert);
  return cold;
}
std::vector<int64_t> patternIDs(uint32_t rows, std::span<const uint32_t> hot,
                               std::string_view pattern) {
  const auto cold = coldIDs(hot);
  require(hot.size() >= 10 && cold.size() >= 10, "synthetic top10 patterns need at least10 stored and10 missing experts");
  require(pattern == "hit-concentrated" || pattern == "hit-spread" || pattern == "miss-only" ||
      pattern == "mixed" || pattern == "spread-all", "unknown route pattern");
  std::vector<int64_t> ids(uint64_t{rows} * kSelections);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t slot = 0; slot < kSelections; ++slot) {
      uint32_t expert = (row * 73 + slot * 53) % 512;
      if (pattern == "hit-concentrated") expert = hot[slot];
      if (pattern == "hit-spread") expert = hot[(row * 7 + slot) % hot.size()];
      if (pattern == "miss-only") expert = cold[(row * 73 + slot) % cold.size()];
      if (pattern == "mixed") expert = slot < 5 ? hot[(row * 7 + slot) % hot.size()]
          : cold[(row * 73 + slot - 5) % cold.size()];
      ids[uint64_t{row} * kSelections + slot] = expert;
    }
  return ids;
}
void persistentCPUTest() {
  for (uint32_t active : {0u,1u,31u,32u,33u,128u,129u,512u,1151u,1791u}) {
    for (uint32_t stride : {1u,8u,16u,32u,64u,128u}) {
      std::vector<uint32_t> visits(active);
      for (uint32_t first = 0; first < stride; ++first)
        for (uint32_t job = first; job < active; job += stride) ++visits[job];
      require(std::all_of(visits.begin(),visits.end(),[](uint32_t n){return n == 1;}),
          "persistent strided job traversal omits or duplicates a GPU job");
    }
  }
  CommandGraph source;
  FlashInt8ExpertStoreParams hit{2048,10,20480,1151,32,64,0,0};
  FlashInt8ExpertStoreGateParams gate{};
  gate.blocked.affine.rows=2048; gate.blocked.affine.selections=10;
  gate.blocked.affine.input_size=2560; gate.blocked.affine.output_size=640;
  gate.blocked.affine.experts=512; gate.blocked.route_capacity=20480;
  gate.blocked.job_capacity=1151; gate.blocked.tile_rows=32;
  gate.stored_experts=64; gate.flags=1;
  FlashInt8ExpertStoreDownParams down{};
  down.blocked.affine.rows=2048; down.blocked.affine.selections=10;
  down.blocked.affine.input_size=640; down.blocked.affine.output_size=2560;
  down.blocked.affine.experts=512; down.blocked.route_capacity=20480;
  down.blocked.job_capacity=1151; down.blocked.tile_rows=32;
  down.stored_experts=64; down.flags=1;
  source.add("flash_int8_expert_store_gate_up_m32_n64",std::vector<MetalBuffer>(11),hit,{10,1151,1},{128,1,1});
  source.add("flash_int8_expert_store_gate_up_miss_direct_m32_n64",std::vector<MetalBuffer>(13),gate,{10,1151,1},{128,1,1});
  source.add("flash_int8_expert_store_down_scatter_m32_n64",std::vector<MetalBuffer>(10),hit,{40,1151,1},{128,1,1});
  source.add("flash_int8_expert_store_down_miss_direct_m32_n64",std::vector<MetalBuffer>(11),down,{40,1151,1},{128,1,1});
  const auto rewritten=splash::flash::candidate::persistentInt8JobDispatches(source.dispatches(),64,16);
  require(rewritten.size()==4 && rewritten[0].threadgroups.y==64 && rewritten[1].threadgroups.y==64 &&
      rewritten[2].threadgroups.y==16 && rewritten[3].threadgroups.y==16 &&
      source.dispatches()[0].threadgroups.y==1151,"private rewrite changed source or grid shape");
  for (size_t i=0;i<4;++i)
    require(rewritten[i].bytes[0].data==source.dispatches()[i].bytes[0].data &&
        rewritten[i].bytes[0].sizeBytes==source.dispatches()[i].bytes[0].sizeBytes &&
        rewritten[i].buffers.size()==source.dispatches()[i].buffers.size(),
        "private rewrite did not retain original ABI/buffer bindings");
  const auto onlyDown=splash::flash::candidate::persistentInt8JobDispatches(source.dispatches(),64,16,false,true);
  require(onlyDown[0].pipelineName==source.dispatches()[0].pipelineName &&
      onlyDown[1].threadgroups.y==1151 && onlyDown[2].threadgroups.y==16,"phase isolation differs");
  for (uint32_t invalid : {0u,129u}) {
    bool rejected=false;
    try {(void)splash::flash::candidate::persistentInt8JobDispatches(source.dispatches(),invalid,16);}
    catch(const std::invalid_argument &){rejected=true;}
    require(rejected,"invalid persistent grid accepted");
  }
  auto malformed=std::vector<ComputeDispatch>(source.dispatches().begin(),source.dispatches().end());
  malformed[0].bytes[0].sizeBytes=31;
  bool rejected=false;
  try {(void)splash::flash::candidate::persistentInt8JobDispatches(malformed);}
  catch(const std::invalid_argument &){rejected=true;}
  require(rejected,"wrong source parameter extent accepted");
  malformed=std::vector<ComputeDispatch>(source.dispatches().begin(),source.dispatches().end());
  malformed[2].threadgroups.x=39; rejected=false;
  try {(void)splash::flash::candidate::persistentInt8JobDispatches(malformed);}
  catch(const std::invalid_argument &){rejected=true;}
  require(rejected,"wrong source output column grid accepted");
}
void cpuSelfTest() {
  persistentCPUTest();
  require(bf16(1.00390625f) == 0x3f80 && bf16(1.01171875f) == 0x3f82,
      "BF16 ties-to-even golden differs");
  require(bf16(-0.0f) == 0x8000, "BF16 signed zero differs");
  require(ref::jobCapacity(129, 10, 16) == moEBucketJobCapacity(129, 10, 16) &&
      ref::jobCapacity(2048, 10, 32) == 1151 &&
      ref::jobCapacity(8192, 10, 64) == moEBucketJobCapacity(8192, 10, 64), "independent job bounds differ");
  for (uint32_t count : {10u, 32u, 128u}) {
    std::vector<uint32_t> hot;
    for (uint32_t rank = 0; rank < count; ++rank) hot.push_back((rank * 173 + 31) % 512);
    std::sort(hot.begin(), hot.end());
    std::array<uint32_t, 512> ranks{}; ranks.fill(UINT32_MAX);
    for (uint32_t rank = 0; rank < hot.size(); ++rank) ranks[hot[rank]] = rank;
    for (const char *pattern : {"hit-concentrated", "hit-spread", "miss-only", "mixed", "spread-all"}) {
      const auto ids = patternIDs(129, hot, pattern);
      std::vector<uint16_t> hidden(129 * 2560, 0x3f80);
      const auto packed = ref::pack(hidden, ids, 129, 10, kSticky);
      require(packed.diagnostic == kSticky && packed.offsets[512] == 1290, "pattern generated invalid top10 ownership");
      uint32_t hits = 0;
      for (uint32_t row = 0; row < 129; ++row)
        for (uint32_t slot = 0; slot < 10; ++slot) {
          const auto expert = uint32_t(ids[row * 10 + slot]);
          const bool hit = ranks[expert] != UINT32_MAX;
          if (hit) require(hot[ranks[expert]] == expert, "arbitrary expert-to-compact-rank round trip differs");
          if (std::string_view(pattern).starts_with("hit-")) require(hit, "hit fixture selected a miss");
          if (std::string_view(pattern) == "miss-only") require(!hit, "miss fixture selected a hit");
          if (std::string_view(pattern) == "mixed") require(hit == (slot < 5), "mixed fixture does not own five hits/five misses");
          hits += hit;
        }
      if (std::string_view(pattern) == "mixed") require(hits == 645, "mixed hit count differs");
      for (uint32_t m : {16u, 32u, 64u}) {
        const auto jobs = ref::makeJobs(packed, m);
        require(jobs.count <= jobs.entries.size(), "independent capacity exceeded");
        for (uint32_t index = 0; index < jobs.count; ++index) {
          const auto &job = jobs.entries[index];
          require(job.expert < 512 && job.rowBegin >= packed.offsets[job.expert] &&
              job.rowBegin < packed.offsets[job.expert + 1], "independent job ownership differs");
        }
      }
    }
  }
  require(kFlashMoEDirectAPaddingRows == 63, "DirectA global tail padding differs");
}
std::string digest(const MetalBuffer &buffer) {
  require(buffer && buffer.contents(), "hash requires an addressable buffer");
  CC_SHA256_CTX context{};
  require(CC_SHA256_Init(&context), "SHA256 initialization failed");
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
  uint64_t remaining = buffer.sizeBytes();
  while (remaining) {
    const auto step = CC_LONG(std::min<uint64_t>(remaining, 1ULL << 30));
    require(CC_SHA256_Update(&context, bytes, step), "SHA256 update failed");
    bytes += step; remaining -= step;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> sha{};
  require(CC_SHA256_Final(sha.data(), &context), "SHA256 finalization failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : sha) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
template<class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values, const char *label) {
  auto buffer = backend.allocateBuffer(values.size() * sizeof(T), BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
template<class T> std::vector<T> readFile(const char *path, uint64_t elements) {
  require(std::filesystem::file_size(path) == elements * sizeof(T), "raw fixture extent differs");
  std::vector<T> result(elements);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(T));
  require(bool(input), "cannot read raw fixture"); return result;
}
struct Guard final {
  MetalBuffer allocation;
  uint64_t bytes;
  bool clean() const {
    const auto *start = static_cast<const uint8_t *>(allocation.contents()) + bytes;
    return std::all_of(start, start + kGuardBytes, [](uint8_t value) { return value == 0x5a; });
  }
};
MetalBuffer guarded(MetalBackend &backend, uint64_t bytes, std::vector<Guard> &guards) {
  auto allocation = backend.allocateBuffer(bytes + kGuardBytes, BufferStorage::Shared, "saved INT8 oracle guarded scratch");
  std::memset(allocation.contents(), 0xa5, bytes);
  std::memset(static_cast<uint8_t *>(allocation.contents()) + bytes, 0x5a, kGuardBytes);
  guards.push_back({allocation, bytes}); return backend.view(allocation, 0, bytes);
}
void guardScratch(MetalBackend &backend, FlashMoEBlockedScratch &s, std::vector<Guard> &guards) {
  for (auto *buffer : {&s.buckets.counts, &s.buckets.offsets, &s.buckets.routeMap,
      &s.buckets.canonicalToPacked, &s.buckets.packedInputs, &s.buckets.jobOffsets,
      &s.buckets.jobCount, &s.buckets.tileJobs, &s.packedActivated, &s.scatteredDown})
    *buffer = guarded(backend, buffer->sizeBytes(), guards);
}
template<class T> void exactVector(const MetalBuffer &buffer, const std::vector<T> &values, const char *label) {
  require(buffer.sizeBytes() >= values.size() * sizeof(T) &&
      std::memcmp(buffer.contents(), values.data(), values.size() * sizeof(T)) == 0,
      std::string("independent bucket mismatch: ") + label);
}
void checkBuckets(const FlashMoEBlockedScratch &s, const ref::Packed &packed, const ref::Jobs &jobs) {
  require(std::memcmp(s.buckets.counts.contents(), packed.counts.data(), 512 * 4) == 0, "independent counts differ");
  require(std::memcmp(s.buckets.offsets.contents(), packed.offsets.data(), 513 * 4) == 0, "independent offsets differ");
  exactVector(s.buckets.routeMap, packed.routeMap, "stable route map");
  exactVector(s.buckets.canonicalToPacked, packed.canonicalToPacked, "canonical inverse map");
  exactVector(s.buckets.packedInputs, packed.inputs, "BF16 input bit copy");
  require(*static_cast<const uint32_t *>(s.buckets.jobCount.contents()) == jobs.count, "active job count differs");
  require(std::memcmp(s.buckets.jobOffsets.contents(), jobs.offsets.data(), 513 * 4) == 0, "job offsets differ");
  const auto *actual = static_cast<const FlashMoEBucketJob *>(s.buckets.tileJobs.contents());
  for (uint32_t i = 0; i < jobs.entries.size(); ++i)
    require(actual[i].expert == jobs.entries[i].expert && actual[i].row_begin == jobs.entries[i].rowBegin,
        "independent active/inactive job ownership differs at " + std::to_string(i));
  if (flashMoEDirectAEnabled()) {
    for (const auto &buffer : {s.buckets.packedInputs, s.packedActivated}) {
      const uint32_t width = buffer.sameView(s.buckets.packedInputs) ? 2560 : 640;
      const auto *tail = static_cast<const uint16_t *>(buffer.contents()) + uint64_t{packed.rows} * packed.selections * width;
      require(std::all_of(tail, tail + uint64_t{kFlashMoEDirectAPaddingRows} * width,
          [](uint16_t value) { return value == 0; }), "DirectA global tail is not sanitized zero");
    }
  }
}
struct Comparison final {
  uint64_t elements = 0, mismatches = 0;
  double relativeL2 = 0, cosine = 1, maxAbs = 0;
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"relative_l2\":" << relativeL2 << ",\"cosine\":" << cosine << ",\"max_abs\":" << maxAbs << '}';
  }
};
Comparison compare(const MetalBuffer &a, const MetalBuffer &b, uint64_t elements,
                   double maxL2, double minCosine) {
  require(a.sizeBytes() >= elements * 2 && b.sizeBytes() >= elements * 2, "output comparison extent differs");
  const auto *av = static_cast<const uint16_t *>(a.contents());
  const auto *bv = static_cast<const uint16_t *>(b.contents());
  Comparison result; result.elements = elements;
  double diff = 0, normA = 0, normB = 0, dot = 0;
  for (uint64_t i = 0; i < elements; ++i) {
    const double x = number(av[i]), y = number(bv[i]);
    require(std::isfinite(x) && std::isfinite(y), "nonfinite expert-chain output");
    result.mismatches += av[i] != bv[i]; const double delta = x - y;
    diff += delta * delta; normA += x * x; normB += y * y; dot += x * y;
    result.maxAbs = std::max(result.maxAbs, std::abs(delta));
  }
  result.relativeL2 = normA ? std::sqrt(diff / normA) : (diff ? std::numeric_limits<double>::infinity() : 0);
  result.cosine = normA && normB ? dot / std::sqrt(normA * normB) : (normA == normB ? 1 : 0);
  require(std::isfinite(result.relativeL2) && result.relativeL2 <= maxL2 && result.cosine >= minCosine,
      "declared INT8 producer sanity guard failed; this is not a model-quality qualification");
  return result;
}
uint64_t exactMissSlices(const std::array<FlashMoEBlockedScratch, 2> &s,
    const ref::Packed &packed, std::span<const int64_t> ids, std::span<const uint32_t> hot) {
  uint64_t misses = 0;
  for (uint32_t expert = 0; expert < 512; ++expert) {
    if (std::binary_search(hot.begin(), hot.end(), expert)) continue;
    for (uint32_t row = packed.offsets[expert]; row < packed.offsets[expert + 1]; ++row) {
      const uint64_t offset = uint64_t{row} * 640 * 2;
      require(std::memcmp(static_cast<const uint8_t *>(s[0].packedActivated.contents()) + offset,
          static_cast<const uint8_t *>(s[1].packedActivated.contents()) + offset, 640 * 2) == 0,
          "mixed/miss Q4 fallback activation is not bit-exact");
    }
  }
  for (uint32_t route = 0; route < ids.size(); ++route) {
    if (std::binary_search(hot.begin(), hot.end(), uint32_t(ids[route]))) continue;
    const uint64_t offset = uint64_t{route} * 2560 * 2;
    require(std::memcmp(static_cast<const uint8_t *>(s[0].scatteredDown.contents()) + offset,
        static_cast<const uint8_t *>(s[1].scatteredDown.contents()) + offset, 2560 * 2) == 0,
        "mixed/miss Q4 fallback down scatter is not bit-exact");
    ++misses;
  }
  return misses;
}
void times(std::ostream &out, const std::vector<CommandTiming> &timing, bool gpu) {
  out << '[';
  for (size_t i = 0; i < timing.size(); ++i) {
    if (i) out << ','; out << (gpu ? timing[i].gpuSeconds : timing[i].wallSeconds) * 1000;
  }
  out << ']';
}
void names(std::ostream &out, std::span<const ComputeDispatch> dispatches) {
  out << '[';
  for (size_t i = 0; i < dispatches.size(); ++i) {
    if (i) out << ','; out << splash::json::quote(dispatches[i].pipelineName);
  }
  out << ']';
}
struct Replay final {
  CommandGraph graph;
  MetalBuffer output, diagnostic;
  std::vector<uint16_t> expected;
  std::vector<Guard> guards;
  std::vector<ComputeDispatch> persistent;
  uint32_t layer = 0;
  std::string pattern;
  void run(MetalBackend &backend) const {
    (void)backend.submitCommand(persistent.empty() ? graph.dispatches() : std::span<const ComputeDispatch>(persistent));
    exactVector(output, expected, "graph mapping lifetime replay");
    require(*static_cast<const uint32_t *>(diagnostic.contents()) == kSticky, "lifetime replay diagnostics differ");
    for (const auto &guard : guards) require(guard.clean(), "lifetime replay canary changed");
  }
};
uint32_t checkRejections(MetalBackend &backend, const FlashWeights &weights,
    const FlashInt8ExpertStore &store, uint32_t layer,
    const FlashMoEBlockedScratch &scratch, MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile) {
  uint32_t checks = 0;
  const auto reject = [&](auto &&operation, const char *label) {
    CommandGraph graph; bool rejected = false;
    try { operation(graph); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && graph.empty(), std::string("graph-construction rejection is not atomic: ") + label);
    ++checks;
  };
  reject([&](CommandGraph &graph) { store.addGateUp(graph, 48, scratch, diagnostics, rows, tile); }, "outside target layer");
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, scratch, diagnostics, 0, tile); }, "zero rows");
  reject([&](CommandGraph &graph) { store.addDownScatter(graph, layer, scratch, diagnostics, rows, tile, 0); }, "zero selections");
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, scratch, diagnostics, rows,
      FlashMoEBlockedTile::M8N64); }, "unsupported INT8 M8 gate tile");
  reject([&](CommandGraph &graph) { store.addDownScatter(graph, layer, scratch, diagnostics, rows,
      FlashMoEBlockedTile::M8N64); }, "unsupported INT8 M8 down tile");
  const auto immutable = store.immutableWeightBuffers();
  const uint32_t other = layer == 0 ? 47 : 0;
  auto bad = scratch;
  bad.packedActivated = backend.view(immutable[other * 2], 16384, scratch.packedActivated.sizeBytes());
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, bad, diagnostics, rows, tile); }, "cross-layer saved payload output");
  const auto savedRanks = backend.view(immutable[other * 2 + 1], 4, 4);
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, scratch, savedRanks, rows, tile); }, "cross-layer rank diagnostics");
  bad = scratch;
  const auto &original = weights.projection(prefixFor(other) + ".gate_proj").weights->buffer;
  bad.scatteredDown = backend.view(original, 0, scratch.scatteredDown.sizeBytes());
  reject([&](CommandGraph &graph) { store.addDownScatter(graph, layer, bad, diagnostics, rows, tile); }, "cross-layer Q4 fallback output");
  bad = scratch;
  bad.buckets.offsets = backend.view(scratch.buckets.packedInputs, 128, 513 * 4);
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, bad, diagnostics, rows, tile); }, "sanitized input overlapping offsets");
  bad = scratch;
  bad.buckets.tileJobs = backend.view(scratch.packedActivated, 128, scratch.buckets.tileJobs.sizeBytes());
  reject([&](CommandGraph &graph) { store.addDownScatter(graph, layer, bad, diagnostics, rows, tile); }, "prepared activation overlapping jobs");
  const auto overlappingDiagnostic = backend.view(scratch.buckets.packedInputs, 4, 4);
  reject([&](CommandGraph &graph) { store.addGateUp(graph, layer, scratch, overlappingDiagnostic, rows, tile); }, "sanitized input overlapping diagnostics");
  return checks;
}
Replay runCase(MetalBackend &backend, const FlashWeights &weights, const FlashInt8ExpertStore &store,
    uint32_t layer, uint32_t rows, uint32_t m, const std::string &pattern, uint32_t pairs,
    double maxL2, double minCosine, std::ostream &out) {
  std::vector<uint16_t> hidden(uint64_t{rows} * 2560);
  const auto hot = store.selectedExpertIDs(layer);
  std::vector<int64_t> ids = patternIDs(rows, hot, pattern);
  const char *rawInput = std::getenv("FLASH_I8_JOBS_INPUT");
  const char *rawIDs = std::getenv("FLASH_I8_JOBS_ROUTE_IDS");
  require(!rawInput || rawIDs, "raw activation input requires route IDs");
  if (rawInput) {
    hidden = readFile<uint16_t>(rawInput, hidden.size());
  } else for (uint64_t i = 0; i < hidden.size(); ++i)
    hidden[i] = bf16(float(int((i * 73 + i / 2560 * 17) % 257) - 128) / 512.0f);
  if (rawIDs) ids=readFile<int64_t>(rawIDs,ids.size());
  for (uint16_t value : hidden) require(std::isfinite(number(value)), "nonfinite fixture hidden value");
  const auto packed = ref::pack(hidden, ids, rows, kSelections, kSticky);
  require(packed.diagnostic == kSticky, "fixture contains invalid or duplicate top10 IDs");
  const auto jobs = ref::makeJobs(packed, m);
  const auto prefix = prefixFor(layer);
  (void)prefix;
  const auto input = upload(backend, hidden, "saved INT8 oracle hidden input");
  const auto expertIDs = upload(backend, ids, "saved INT8 oracle original IDs");
  std::vector<uint16_t> routeValues(ids.size());
  // Unequal route weights make scatter/canonical ownership errors visible.
  for (uint64_t route = 0; route < ids.size(); ++route) routeValues[route] = bf16(float((route % 10) + 1) / 55.0f);
  const auto route = upload(backend, routeValues, "saved INT8 oracle unequal route weights");
  const auto shared = upload(backend, std::vector<uint16_t>(uint64_t{rows} * 2560, 0), "oracle shared expert zeros");
  const auto sharedGate = upload(backend, std::vector<uint16_t>(rows, 0), "oracle shared gate zeros");
  std::array<FlashMoEBlockedScratch, 2> scratch;
  std::array<CommandGraph, 2> graphs;
  std::array<MetalBuffer, 2> output, diagnostic;
  std::vector<Guard> guards;
  const auto tile = static_cast<FlashMoEBlockedTile>(m);
  for (uint32_t which = 0; which < 2; ++which) {
    scratch[which] = allocateMoEBlockedScratch(backend, rows);
    guardScratch(backend, scratch[which], guards);
    output[which] = guarded(backend, uint64_t{rows} * 2560 * 2, guards);
    diagnostic[which] = guarded(backend, 4, guards);
    *static_cast<uint32_t *>(diagnostic[which].contents()) = kSticky;
    addMoEBlockedPack(graphs[which], input, expertIDs, scratch[which], diagnostic[which], rows, tile);
    store.addGateUp(graphs[which], layer, scratch[which], diagnostic[which], rows, tile);
    store.addDownScatter(graphs[which], layer, scratch[which], diagnostic[which], rows, tile);
    addCombine(graphs[which], scratch[which].scatteredDown, expertIDs, route, shared, sharedGate,
        output[which], diagnostic[which], rows, 2560, 512, kSelections);
  }
  const auto healthy = [&] {
    for (const auto &buffer : diagnostic)
      require(*static_cast<const uint32_t *>(buffer.contents()) == kSticky, "sticky diagnostics or numerical flags changed");
    for (const auto &guard : guards) require(guard.clean(), "scratch/output canary overwritten");
  };
  const uint32_t rejectionChecks = checkRejections(backend, weights, store, layer,
      scratch[1], diagnostic[1], rows, tile);
  const uint32_t gateY = envNumber("FLASH_I8_JOBS_GATE_Y",64,128);
  const uint32_t downY = envNumber("FLASH_I8_JOBS_DOWN_Y",16,128);
  const char *phaseRaw = std::getenv("FLASH_I8_JOBS_PHASE");
  const std::string_view phase = phaseRaw ? phaseRaw : "both";
  require(phase == "both" || phase == "gate" || phase == "down", "phase must be both/gate/down");
  const auto persistent = splash::flash::candidate::persistentInt8JobDispatches(
      graphs[1].dispatches(),gateY,downY,phase != "down",phase != "gate");
  (void)backend.submitCommand(graphs[0].dispatches());
  (void)backend.submitCommand(persistent);
  healthy(); checkBuckets(scratch[0], packed, jobs); checkBuckets(scratch[1], packed, jobs);
  const uint64_t routes = uint64_t{rows} * kSelections;
  const uint64_t misses = exactMissSlices(scratch, packed, ids, hot);
  const auto activation = compare(scratch[0].packedActivated, scratch[1].packedActivated, routes * 640, maxL2, minCosine);
  const auto downComparison = compare(scratch[0].scatteredDown, scratch[1].scatteredDown, routes * 2560, maxL2, minCosine);
  const auto combined = compare(output[0], output[1], uint64_t{rows} * 2560, maxL2, minCosine);
  require(!activation.mismatches && !downComparison.mismatches && !combined.mismatches,
      "every persistent INT8/Q4 activation/down/combine byte must match current v5 producer");
  std::array<std::vector<CommandTiming>, 2> timing;
  for (uint32_t pair = 0; pair < pairs; ++pair)
    for (uint32_t order = 0; order < 2; ++order) {
      const uint32_t which = (pair + order) % 2;
      timing[which].push_back(backend.submitCommand(which ? std::span<const ComputeDispatch>(persistent) : graphs[0].dispatches()));
      const auto &sample = timing[which].back();
      require(std::isfinite(sample.gpuSeconds) && std::isfinite(sample.wallSeconds) &&
              sample.gpuSeconds > 1e-9 && sample.wallSeconds > 1e-9,
              "nonfinite/nonpositive command timing; reject stale CommandTiming ABI dependencies");
      // Checks and all checksum scans remain outside the submitted timing scope.
      healthy();
    }
  checkBuckets(scratch[0], packed, jobs); checkBuckets(scratch[1], packed, jobs);
  require(exactMissSlices(scratch, packed, ids, hot) == misses, "timed miss ownership changed");
  require(!compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine).mismatches &&
      !compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine).mismatches &&
      !compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine).mismatches,
      "replayed persistent producer differs in a BF16 byte");
  uint32_t hitJobs = 0, fullHitJobs = 0;
  for (uint32_t i = 0; i < jobs.count; ++i) {
    const auto &job = jobs.entries[i];
    if (std::binary_search(hot.begin(), hot.end(), job.expert)) {
      ++hitJobs; fullHitJobs += packed.offsets[job.expert + 1] - job.rowBegin >= m;
    }
  }
  const auto activeExperts = std::count_if(packed.counts.begin(), packed.counts.end(), [](uint32_t n) { return n != 0; });
  out << "{\"layer\":" << layer << ",\"rows\":" << rows << ",\"tile_m\":" << m
      << ",\"pattern\":" << splash::json::quote(rawInput ? "raw-fixture" : rawIDs ? "captured-route-ids/synthetic-activation" : pattern)
      << ",\"stored_experts\":" << hot.size() << ",\"active_experts\":" << activeExperts
      << ",\"active_jobs\":" << jobs.count << ",\"job_capacity\":" << jobs.entries.size()
      << ",\"hit_jobs\":" << hitJobs << ",\"full_hit_jobs\":" << fullHitJobs
      << ",\"hit_routes\":" << routes - misses << ",\"miss_routes\":" << misses
      << ",\"atomic_graph_rejection_checks\":" << rejectionChecks
      << ",\"miss_activation_and_down_exact\":true,\"all_miss_chain_exact\":" << (misses == routes ? "true" : "null")
      << ",\"independent_bucket_ownership_exact\":true,\"canaries_clean\":true,\"activation\":";
  activation.write(out); out << ",\"down\":"; downComparison.write(out); out << ",\"combine\":"; combined.write(out);
  out << ",\"control_pipelines\":"; names(out, graphs[0].dispatches());
  out << ",\"candidate_pipelines\":"; names(out, persistent);
  out << ",\"persistent_gate_grid_y\":" << gateY << ",\"persistent_down_grid_y\":" << downY;
  out << ",\"control_gpu_ms\":"; times(out, timing[0], true);
  out << ",\"candidate_gpu_ms\":"; times(out, timing[1], true);
  out << ",\"control_wall_ms\":"; times(out, timing[0], false);
  out << ",\"candidate_wall_ms\":"; times(out, timing[1], false); out << '}';
  Replay replay; replay.graph = std::move(graphs[1]); replay.output = output[1]; replay.diagnostic = diagnostic[1]; replay.persistent = persistent;
  const auto *values = static_cast<const uint16_t *>(output[1].contents());
  replay.expected.assign(values, values + uint64_t{rows} * 2560);
  replay.guards = std::move(guards); replay.layer = layer; replay.pattern = rawInput ? "raw-fixture" : rawIDs ? "captured-route-ids/synthetic-activation" : pattern;
  return replay;
}
void checkStoreRanks(const FlashInt8ExpertStore &store, const FlashInt8ExpertStoreMetadata &metadata,
                     std::span<const MetalBuffer> buffers) {
  require(buffers.size() == 96, "production store immutable base/rank buffer list differs");
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto hot = store.selectedExpertIDs(layer);
    const auto &expected = metadata.layers[layer].selectedIDs;
    require(std::equal(hot.begin(), hot.end(), expected.begin(), expected.end()), "actual selected IDs differ from manifest");
    require(buffers[layer * 2].sizeBytes() == metadata.layers[layer].bytes, "mapped layer extent differs");
    const auto &rankBuffer = buffers[layer * 2 + 1];
    require(rankBuffer.sizeBytes() >= 512 * 4, "compact rank map extent differs");
    const auto *ranks = static_cast<const uint32_t *>(rankBuffer.contents());
    for (uint32_t expert = 0; expert < 512; ++expert) {
      const auto it = std::lower_bound(hot.begin(), hot.end(), expert);
      const uint32_t expectedRank = it != hot.end() && *it == expert ? uint32_t(it - hot.begin()) : UINT32_MAX;
      require(ranks[expert] == expectedRank, "arbitrary original expert ID-to-compact-rank mapping differs");
    }
    const auto *padding = static_cast<const uint8_t *>(rankBuffer.contents()) + 512 * 4;
    require(std::all_of(padding, static_cast<const uint8_t *>(rankBuffer.contents()) + rankBuffer.sizeBytes(),
        [](uint8_t value) { return value == 0xff; }), "rank-map immutable padding changed");
  }
}
std::string pipelineMetadata(const char *path, uint32_t m, bool direct) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil, "Metal device unavailable");
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
  require(library != nil, "combined candidate library unavailable");
  const std::string suffix = "_m" + std::to_string(m) + (m == 64 ? "_n64_sg8" : "_n64");
  std::vector<std::string> functions;
  const std::string missKind = direct ? "_direct" : "_staged";
  for (const auto &phase : std::vector<std::string>{"gate_up", "gate_up_miss" + missKind,
      "down_scatter", "down_miss" + missKind})
    functions.push_back(std::string("flash_int8_expert_store_") + phase + suffix);
  require(direct,"persistent job candidate requires the v5 direct-A Q4 miss producer");
  for (const auto &phase : std::vector<std::string>{"gate_up","gate_up_miss_direct","down_scatter","down_miss_direct"})
    functions.push_back("flash_int8_expert_persistent_"+phase+suffix);
  functions.push_back(direct ? "flash_moe_direct_a_prepare_down" : "flash_int8_expert_store_sanitize");
  std::ostringstream out;
  out << "{\"device_max_threadgroup_memory_bytes\":" << device.maxThreadgroupMemoryLength << ",\"pipelines\":[";
  bool first = true;
  for (const auto &name : functions) {
    const uint32_t requested = name.ends_with(suffix) ? (m == 64 ? 256 : 128) : 256;
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "candidate function missing: " + name + " REQUESTED=" + std::to_string(requested) +
        " MAXTHREADS=unavailable STATICTGM=unavailable DEVICE_MAXTGM=" + std::to_string(device.maxThreadgroupMemoryLength) +
        " THREADWIDTH=unavailable");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    const std::string errorText = error ? error.localizedDescription.UTF8String : "no Metal error description";
    require(pipeline != nil, "pipeline creation failed: " + name + " REQUESTED=" + std::to_string(requested) +
        " MAXTHREADS=unavailable STATICTGM=unavailable DEVICE_MAXTGM=" + std::to_string(device.maxThreadgroupMemoryLength) +
        " THREADWIDTH=unavailable ERROR=" + errorText);
    const std::string limits = " REQUESTED=" + std::to_string(requested) +
        " MAXTHREADS=" + std::to_string(pipeline.maxTotalThreadsPerThreadgroup) +
        " STATICTGM=" + std::to_string(pipeline.staticThreadgroupMemoryLength) +
        " DEVICE_MAXTGM=" + std::to_string(device.maxThreadgroupMemoryLength) +
        " THREADWIDTH=" + std::to_string(pipeline.threadExecutionWidth);
    require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= requested &&
        pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
        "pipeline threadgroup limits exceeded: " + name + limits);
    if (!first) out << ','; first = false;
    out << "{\"name\":" << splash::json::quote(name) << ",\"execution_width\":" << pipeline.threadExecutionWidth
        << ",\"requested_threads\":" << requested
        << ",\"maximum_threads\":" << pipeline.maxTotalThreadsPerThreadgroup
        << ",\"static_threadgroup_memory_bytes\":" << pipeline.staticThreadgroupMemoryLength << '}';
  }
  out << "]}"; return out.str();
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"sizeof_command_timing\":"
                  << sizeof(CommandTiming)
                  << ",\"checks\":[\"BF16_RNE\",\"arbitrary_compact_ranks\",\"top10_pattern_ownership\",\"independent_job_bounds\",\"DirectA_padding\",\"persistent_exact_job_coverage\",\"borrowed_source_ABI_and_phase_guards\"]}\n";
        return 0;
      }
      require(argc == 5, "usage: flash-int8-persistent-jobs-oracle COMBINED_METALLIB SOURCE_PACKAGE INT8_STORE REPORT_JSON");
      const uint32_t pairs = envNumber("FLASH_I8_JOBS_PAIRS", 4, 32);
      const uint32_t rows = envNumber("FLASH_I8_JOBS_ROWS", 2048, 8192);
      const uint32_t m = envNumber("FLASH_I8_JOBS_TILE", rows >= 4096 ? 64 : rows >= 1024 ? 32 : 16, 64);
      require(m == 16 || m == 32 || m == 64, "tile must be16/32/64");
      require(m != 64 || rows >= 1024, "production M64 policy requires at least1024 rows");
      const double maxL2 = envPositive("FLASH_I8_JOBS_MAX_RL2", 0.10, 1.0);
      const double minCosine = envPositive("FLASH_I8_JOBS_MIN_COSINE", 0.99, 1.0);
      const auto layers = parseLayers();
      // Freeze the qualified v5 baseline before any production helper reads
      // process-wide feature flags. The candidate changes launch policy only.
      require(setenv("SPLASH_FLASH_MOE_Q4X8", "1", 1) == 0 &&
              setenv("SPLASH_FLASH_MOE_M64", "1", 1) == 0 &&
              setenv("SPLASH_FLASH_MOE_DIRECT_A", "1", 1) == 0,
          "cannot enable qualified Q4x8/M64 control policy");
      const bool direct = flashMoEDirectAEnabled();
      const auto pipelines = pipelineMetadata(argv[1], m, direct);
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      const auto metadata = loadFlashInt8ExpertStoreMetadata(argv[3], weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention());
      const uint64_t planned = FlashInt8ExpertStore::plannedBytes(weights, argv[3]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      require(physical > reserve, "insufficient physical RAM");
      splash::engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      auto reservation = governor.tryReserve(planned);
      require(bool(reservation), "real governor denied production INT8 store before mapping");
      auto store = std::make_unique<FlashInt8ExpertStore>(backend, weights, argv[3]);
      require(store->actualAllocatedBytes() <= planned && store->mappedBytes() == metadata.totalBytes,
          "production mapped/accounted bytes exceed plan");
      reservation->commit();
      auto immutable = store->immutableWeightBuffers();
      checkStoreRanks(*store, metadata, immutable);
      // Initial sidecar plane/file SHA checks are performed by the production
      // constructor. Source hashes are scanned once here and once after all
      // cases; no full payload scan occurs in or between timed pair commands.
      std::vector<MetalBuffer> sourceBuffers;
      std::vector<std::string> sourceHashes;
      for (uint32_t layer : layers)
        for (const char *plane : {"gate_proj", "up_proj", "down_proj"}) {
          const auto &p = weights.projection(prefixFor(layer) + "." + plane);
          for (const auto &buffer : {p.weights->buffer, p.scales->buffer, p.biases->buffer}) {
            sourceBuffers.push_back(buffer); sourceHashes.push_back(digest(buffer));
          }
        }
      std::ofstream out(argv[4]); require(bool(out), "cannot create report");
      out << std::setprecision(12) << "{\"schema\":\"flash-int8-persistent-jobs-oracle-v1\",\"pass\":true"
          << ",\"sizeof_command_timing\":" << sizeof(CommandTiming)
          << ",\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_identity\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"store_identity_sha256\":" << splash::json::quote(store->identitySha256())
          << ",\"plan_sha256\":" << splash::json::quote(store->planSha256())
          << ",\"policy\":" << splash::json::quote(kFlashInt8ExpertStoreSemantics)
          << ",\"control_policy\":" << splash::json::quote(kFlashInt8ExpertStoreSemantics)
          << ",\"direct_a\":" << (direct ? "true" : "false")
          << ",\"numerical_change_from_v5\":false,\"all_output_bytes_exact\":true,\"model_quality_qualified\":false"
          << ",\"producer_sanity_guard\":{\"maximum_relative_l2\":" << maxL2 << ",\"minimum_cosine\":" << minCosine << '}'
          << ",\"validation_scope\":\"every live v5 INT8/Q4 producer output byte, exact arbitrary compact ranks and independent bucket ownership, exact Q4 miss slices, readonly source/sidecar immutability, canaries and retained mapping lifetime; no model-quality claim\""
          << ",\"timing_scope\":\"complete expert chain, warm alternating matched GPU commands; metadata, conversion, mapping, constructor checksum checks, CPU comparisons and final SHA scans excluded\""
          << ",\"pairs\":" << pairs << ",\"planned_bytes\":" << planned << ",\"mapped_bytes\":" << store->mappedBytes()
          << ",\"actual_allocated_bytes\":" << store->actualAllocatedBytes()
          << ",\"constructor_all_payload_checksums_verified\":true,\"pipeline_metadata\":" << pipelines << ",\"selections\":[";
      for (size_t index = 0; index < layers.size(); ++index) {
        if (index) out << ',';
        const auto hot = store->selectedExpertIDs(layers[index]);
        out << "{\"layer\":" << layers[index] << ",\"original_expert_ids\":[";
        for (size_t rank = 0; rank < hot.size(); ++rank) { if (rank) out << ','; out << hot[rank]; }
        out << "]}";
      }
      out << "],\"cases\":[";
      Replay replay; bool first = true;
      const char *selectedPattern = std::getenv("FLASH_I8_JOBS_PATTERN");
      for (uint32_t layer : layers)
        for (const char *pattern : {"hit-concentrated", "hit-spread", "miss-only", "mixed", "spread-all"}) {
          if (selectedPattern && std::string_view(selectedPattern) != pattern) continue;
          if (!first) out << ','; first = false;
          replay = runCase(backend, weights, *store, layer, rows, m, pattern, pairs, maxL2, minCosine, out);
          if (std::getenv("FLASH_I8_JOBS_INPUT") || std::getenv("FLASH_I8_JOBS_ROUTE_IDS")) break;
        }
      require(!first, "selected route pattern is unknown");
      out << "],\"post_run_payload_sha256\":[";
      checkStoreRanks(*store, metadata, immutable);
      for (uint32_t layer = 0; layer < 48; ++layer) {
        const auto actual = digest(immutable[layer * 2]);
        require(actual == metadata.layers[layer].sha256, "readonly persisted INT8 payload changed after all cases");
        if (layer) out << ','; out << splash::json::quote(actual);
      }
      for (size_t index = 0; index < sourceBuffers.size(); ++index)
        require(digest(sourceBuffers[index]) == sourceHashes[index], "original source affine weights changed after all cases");
      out << "],\"payloads_immutable\":true,\"source_affine_weights_immutable\":true,\"compact_rank_maps_exact\":true";
      // The graph retains every bound MetalBuffer, whose base owns the readonly
      // mapping. Replay once after BOTH original producer owners have gone.
      immutable.clear(); sourceBuffers.clear(); store.reset(); weights = FlashWeights();
      replay.run(backend);
      out << ",\"retained_graph_mapping_lifetime\":{\"pass\":true,\"store_destroyed\":true,\"source_weights_destroyed\":true"
          << ",\"layer\":" << replay.layer << ",\"pattern\":" << splash::json::quote(replay.pattern) << "}}\n";
      out.flush(); require(bool(out), "report write failed");
      std::cout << "{\"pass\":true,\"model_quality_qualified\":false,\"report\":" << splash::json::quote(argv[4]) << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
