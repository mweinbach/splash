// Isolated root-run exactness/performance oracle for M64N64 MoE prefill.
// --cpu-self-test and compilation create no backend and submit no GPU work.
#include "flash/FlashMoEBlocked.hpp"
#include "FlashMoETall.hpp"
#include "metal/abi/FlashMoEBlocked.h"
#include "flash/FlashMoE.hpp"
#include "engine/Json.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "FlashMoETallReference.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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
namespace ref = splash::flash::bucket_reference;
constexpr uint32_t kSelections = 10, kSticky = 0x80000000u;
constexpr uint64_t kGuardBytes = 64;

void require(bool value, const std::string &reason) {
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
  require(splash::flash::candidate::moETallJobCapacity(2048, 10) == 831,
      "M64 job capacity differs");
  require(splash::flash::candidate::moETallJobCapacity(8192, 10) == 1791,
      "8192-row M64 job capacity differs");
  for (uint32_t sg : {4u, 8u}) {
    constexpr uint32_t rows = 2048, capacity = 1151;
    FlashMoEBucketParams bucket{rows,10,2560,512,rows*10,32,capacity,0};
    FlashMoEBlockedGateParams gate{};
    gate.affine.rows = rows; gate.affine.selections = 10;
    gate.route_capacity = rows * 10; gate.job_capacity = capacity; gate.tile_rows = 32;
    FlashMoEBlockedDownParams down{};
    down.affine.rows = rows; down.affine.selections = 10;
    down.route_capacity = rows * 10; down.job_capacity = capacity; down.tile_rows = 32;
    std::array<ComputeDispatch,4> source;
    source[0].pipelineName = "flash_moe_bucket_job_prefix";
    source[1].pipelineName = "flash_moe_bucket_jobs";
    source[2].pipelineName = "flash_moe_q4x8_gate_up_m32_n64";
    source[3].pipelineName = "flash_moe_q4x8_down_scatter_m32_n64";
    source[0].threadgroups = {1,1,1}; source[1].threadgroups = {5,1,1};
    source[2].threadgroups = {10,capacity,1}; source[3].threadgroups = {40,capacity,1};
    source[0].threadsPerThreadgroup = {256,1,1}; source[1].threadsPerThreadgroup = {256,1,1};
    source[2].threadsPerThreadgroup = {128,1,1}; source[3].threadsPerThreadgroup = {128,1,1};
    source[0].bytes.push_back({5,&bucket,sizeof(bucket)});
    source[1].bytes.push_back({5,&bucket,sizeof(bucket)});
    source[2].bytes.push_back({12,&gate,sizeof(gate)});
    source[3].bytes.push_back({10,&down,sizeof(down)});
    const splash::flash::candidate::MoETallDispatches candidate(source, sg);
    const auto transformed = candidate.dispatches();
    require(transformed[0].pipelineName == "flash_moe_tall_bucket_job_prefix" &&
        transformed[1].threadgroups.x == 4 && transformed[2].threadgroups.y == 831 &&
        transformed[3].threadsPerThreadgroup.x == sg * 32,
        "M64 phase geometry transform differs");
    FlashMoEBucketParams actual{};
    std::memcpy(&actual, transformed[0].bytes[0].data, sizeof(actual));
    require(actual.tile_rows == 64 && actual.job_capacity == 831 &&
        actual.rows == rows && actual.routes == rows * 10 &&
        transformed[0].bytes[0].data != &bucket,
        "M64 owned ABI rewrite differs");
    bucket.tile_rows = 8;
    std::memcpy(&actual, transformed[0].bytes[0].data, sizeof(actual));
    require(actual.tile_rows == 64, "M64 parameter ownership failed");
    bucket.tile_rows = 32;
    source[2].threadgroups.x = 9;
    bool rejected = false;
    try { const splash::flash::candidate::MoETallDispatches invalid(source,sg); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "M64 matrix geometry guard failed");
  }

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
  for (uint32_t sg : {4u, 8u}) {
    for (const char *phase : {"gate_up", "down_scatter"}) {
      const std::string name = std::string("flash_moe_tall_") + phase + "_m64_n64_sg" + std::to_string(sg);
      id<MTLFunction> function = [library newFunctionWithName:
          [NSString stringWithUTF8String:name.c_str()]];
      require(function != nil, "candidate function missing: " + name);
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      require(pipeline != nil, "candidate pipeline creation failed: " + name);
      require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= sg * 32 &&
          pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
          "candidate pipeline threadgroup limits exceeded: " + name +
          ", static=" + std::to_string(pipeline.staticThreadgroupMemoryLength) +
          ", device maximum=" + std::to_string(device.maxThreadgroupMemoryLength));
      if (!first) out << ',';
      first = false;
      out << "{\"name\":" << splash::json::quote(name) << ",\"execution_width\":"
          << pipeline.threadExecutionWidth << ",\"maximum_threads\":" << pipeline.maxTotalThreadsPerThreadgroup
          << ",\"static_threadgroup_memory_bytes\":" << pipeline.staticThreadgroupMemoryLength << '}';
#if !__has_feature(objc_arc)
      [pipeline release]; [function release];
#endif
    }
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
void equal(const MetalBuffer &a, const MetalBuffer &b, uint64_t bytes, const char *label) {
  require(a.sizeBytes() >= bytes && b.sizeBytes() >= bytes &&
      std::memcmp(a.contents(), b.contents(), bytes) == 0,
      std::string("bit-exact candidate comparison failed: ") + label);
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
ref::Jobs tallJobs(const ref::Packed &packed) {
  ref::Jobs result;
  result.entries.resize(splash::flash::candidate::moETallJobCapacity(packed.rows, packed.selections));
  for (uint32_t expert = 0; expert < 512; ++expert) {
    result.offsets[expert] = result.count;
    require(packed.offsets[expert+1] == packed.offsets[expert] + packed.counts[expert],
        "M64 reference bucket offset/count differs");
    for (uint32_t begin = packed.offsets[expert]; begin < packed.offsets[expert+1]; begin += 64) {
      require(result.count < result.entries.size(), "M64 reference jobs overflow");
      result.entries[result.count++] = ref::Job{expert,begin};
    }
  }
  result.offsets[512] = result.count;
  return result;
}
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
             std::ostream &out) {
  std::vector<uint16_t> hidden(uint64_t{rows} * 2560);
  std::vector<int64_t> ids(uint64_t{rows} * kSelections);
  const char *rawInput = std::getenv("FLASH_MOE_TALL_INPUT");
  const char *rawIDs = std::getenv("FLASH_MOE_TALL_IDS");
  require(bool(rawInput) == bool(rawIDs), "raw input and IDs must be supplied together");
  if (rawInput) {
    hidden = readFile<uint16_t>(rawInput, hidden.size());
    ids = readFile<int64_t>(rawIDs, ids.size());
  } else {
    for (uint64_t i = 0; i < hidden.size(); ++i)
      hidden[i] = bf16(float(int((i * 73 + i / 2560 * 17) % 257) - 128) / 512.0f);
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t slot = 0; slot < kSelections; ++slot)
        ids[uint64_t{row} * kSelections + slot] = pattern == "concentrated" ? slot
            : (row * 73 + slot * 53) % 512;
  }
  for (uint16_t value : hidden) require(std::isfinite(number(value)), "nonfinite fixture input");
  const auto packed = ref::pack(hidden, ids, rows, kSelections, kSticky);
  require(packed.diagnostic == kSticky, "fixture has invalid or duplicate route IDs");
  const auto jobs = ref::makeJobs(packed, m);
  const auto m64Jobs = tallJobs(packed);
  const uint32_t sg = envNumber("FLASH_MOE_TALL_SIMDGROUPS", 4, 8);
  require(sg == 4 || sg == 8, "SIMD groups must be4/8");
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
    addMoEBlockedGateUp(graphs[i], gate, up, scratch[i], diagnostic, rows, tile);
    addMoEBlockedDownScatter(graphs[i], down, scratch[i], diagnostic, rows, tile);
    addCombine(graphs[i], scratch[i].scatteredDown, expertIDs, route, shared, sharedGate,
        output[i], diagnostic, rows, 2560, 512, kSelections);
  }
  // Params point into graphs[1]'s owned storage, which survives every submission.
  const splash::flash::candidate::MoETallDispatches candidateOwner(graphs[1].dispatches(), sg);
  const auto candidate = candidateOwner.dispatches();
  (void)backend.submitCommand(graphs[0].dispatches());
  (void)backend.submitCommand(candidate);
  checkBuckets(scratch[0], packed, jobs); checkBuckets(scratch[1], packed, m64Jobs);
  equal(scratch[0].packedActivated, scratch[1].packedActivated,
      uint64_t{rows} * kSelections * 640 * 2, "activated gate/up");
  equal(scratch[0].scatteredDown, scratch[1].scatteredDown,
      uint64_t{rows} * kSelections * 2560 * 2, "canonical expert down");
  equal(output[0], output[1], uint64_t{rows} * 2560 * 2, "BF16 expert combine");
  const auto healthy = [&] {
    require(*static_cast<const uint32_t *>(diagnostic.contents()) == kSticky,
        "sticky diagnostics changed or numerical/shape failure occurred");
    for (const auto &guard : guards) require(guard.clean(), "output canary was overwritten");
  };
  healthy();
  std::array<std::vector<CommandTiming>, 2> timing;
  for (uint32_t pair = 0; pair < pairs; ++pair) {
    for (uint32_t order = 0; order < 2; ++order) {
      const uint32_t which = (pair + order) % 2;
      timing[which].push_back(backend.submitCommand(which == 0 ? graphs[0].dispatches()
          : std::span<const ComputeDispatch>(candidate)));
      healthy();
    }
  }
  equal(scratch[0].packedActivated, scratch[1].packedActivated,
      uint64_t{rows} * kSelections * 640 * 2, "replayed activation");
  equal(scratch[0].scatteredDown, scratch[1].scatteredDown,
      uint64_t{rows} * kSelections * 2560 * 2, "replayed expert down");
  equal(output[0], output[1], uint64_t{rows} * 2560 * 2, "replayed combine");
  const uint32_t activeExperts = uint32_t(std::count_if(packed.counts.begin(), packed.counts.end(),
      [](uint32_t count) { return count != 0; }));
  out << "{\"rows\":" << rows << ",\"tile_m\":" << m << ",\"pattern\":"
      << splash::json::quote(rawInput ? "raw-fixture" : pattern)
      << ",\"candidate_tile_m\":64,\"candidate_simdgroups\":" << sg
      << ",\"candidate_active_jobs\":" << m64Jobs.count
      << ",\"candidate_job_capacity\":" << splash::flash::candidate::moETallJobCapacity(rows, kSelections)
      << ",\"candidate_row_utilization\":" << double(rows * kSelections) / double(m64Jobs.count * 64)
      << ",\"active_experts\":" << activeExperts << ",\"active_jobs\":" << jobs.count
      << ",\"job_capacity\":" << moEBucketJobCapacity(rows, kSelections, m)
      << ",\"row_utilization\":" << double(rows * kSelections) / double(jobs.count * m)
      << ",\"bit_exact\":true,\"canaries_clean\":true,\"control_gpu_ms\":";
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
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":[\"Q4_nibbles\",\"BF16_signed_coefficients\",\"prefill_job_bounds\",\"tall_job_capacity_owned_abi_and_matrix_shape_guards\"]}\n";
        return 0;
      }
      if (argc == 3 && std::string_view(argv[1]) == "--pipeline-metadata") {
        std::cout << pipelineMetadata(argv[2]) << '\n';
        return 0;
      }
      require(argc == 4, "usage: flash-moe-tall-oracle COMBINED_METALLIB PACKAGE REPORT_JSON");
      const uint32_t pairs = envNumber("FLASH_MOE_TALL_PAIRS", 3, 32);
      const char *selectedPrefix = std::getenv("FLASH_MOE_TALL_PREFIX");
      const std::string prefix = selectedPrefix ? selectedPrefix
          : "language_model.model.layers.0.mlp.switch_mlp";
      // Keep the control graph on the current production Q4x8 N64 producer.
      // The private M64 candidate is selected only by graph rewriting.
      require(setenv("SPLASH_FLASH_MOE_Q4X8", "1", 1) == 0, "cannot force oracle control policy");
      // Validate the compiler's actual static TGM, not source-array estimates,
      // before model loading or any command is committed.
      const std::string metadata = pipelineMetadata(argv[1]);
      MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      std::ofstream out(argv[3]); require(bool(out), "cannot create report");
      out << std::setprecision(12) << "{\"schema\":\"flash-moe-tall-prefill-v1\",\"pass\":true,\"source_identity\":"
          << splash::json::quote(weights.sourceIdentity()) << ",\"manifest_identity\":"
          << splash::json::quote(weights.manifestFingerprint()) << ",\"prefix\":"
          << splash::json::quote(prefix) << ",\"weight_conversion\":false,\"baseline_mpp_math\":"
          << splash::json::quote(kFlashMoEBlockedQ4x8Semantics)
          << ",\"candidate_tile_n\":64,\"candidate_tile_m\":64"
          << ",\"pipeline_metadata\":" << metadata
          << ",\"timing_scope\":\"complete bucket-pack + gate/up/SwiGLU + down/scatter + canonical BF16 combine; alternating matched GPU commands after warmup\",\"pairs\":"
          << pairs << ",\"cases\":[";
      bool first = true;
      const uint32_t selectedRows = std::getenv("FLASH_MOE_TALL_ROWS")
          ? envNumber("FLASH_MOE_TALL_ROWS", 512, 8192) : 0;
      for (uint32_t rows : selectedRows ? std::vector<uint32_t>{selectedRows}
          : std::vector<uint32_t>{512, 2048}) {
        const uint32_t m = envNumber("FLASH_MOE_TALL_TILE", rows < 1024 ? 16 : 32, 32);
        require(m == 8 || m == 16 || m == 32, "tile must be8/16/32");
        const bool raw = std::getenv("FLASH_MOE_TALL_INPUT");
        for (const char *pattern : {"spread", "concentrated"}) {
          if (!first) out << ','; first = false;
          runCase(backend, weights, prefix, rows, m, pattern, pairs, out);
          if (raw) break;
        }
      }
      out << "]}\n"; out.flush(); require(bool(out), "report write failed");
      std::cout << "{\"pass\":true,\"bit_exact\":true,\"report\":" << splash::json::quote(argv[3]) << "}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n'; return 1;
    }
  }
}
