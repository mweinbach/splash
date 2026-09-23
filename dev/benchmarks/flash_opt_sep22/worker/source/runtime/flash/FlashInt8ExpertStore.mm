#include "dev/benchmarks/expert_r4_preflight_bundle_sep22/guard.hpp"
#include <type_traits>
#include "dev/benchmarks/expert_r4_preflight_bundle_sep22/policy.hpp"
#include "dev/benchmarks/expert_r4_compact_verify_worker_sep22/abi.hpp"
#include "dev/benchmarks/expert_r4_compact_verify_worker_sep22/bridge.hpp"
#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"
#include "dev/benchmarks/moe_pointwise_sep21/bridge.hpp"
// Independent private gathered MPP Store overlay v1.
#include "FlashInt8ExpertStore.hpp"
#include "FlashInt8ExpertStoreMetadata.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEDirectA.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
[[noreturn]] void fail(const char *reason) { throw std::invalid_argument(reason); }
std::string hash(const void *pointer, uint64_t count) {
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) fail("INT8 expert SHA256 initialization failed");
  const auto *bytes = static_cast<const uint8_t *>(pointer);
  while (count) {
    const CC_LONG step = static_cast<CC_LONG>(std::min<uint64_t>(count, 1ULL << 30));
    if (!CC_SHA256_Update(&context, bytes, step)) fail("INT8 expert SHA256 update failed");
    bytes += step; count -= step;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256_Final(digest.data(), &context)) fail("INT8 expert SHA256 finalization failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
class Mapping final {
public:
  Mapping(const std::filesystem::path &path, uint64_t expected) : bytes_(expected) {
    const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) fail("cannot open readonly INT8 expert payload");
    struct stat before{};
    if (::fstat(file, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        uint64_t(before.st_size) != expected || expected % kAlignment ||
        expected > std::numeric_limits<size_t>::max()) {
      ::close(file); fail("INT8 expert payload size/alignment changed");
    }
    address_ = ::mmap(nullptr, expected, PROT_READ, MAP_SHARED, file, 0);
    const int saved = errno;
    ::close(file);
    if (address_ == MAP_FAILED) { address_ = nullptr; errno = saved; fail("INT8 expert readonly mapping failed"); }
  }
  ~Mapping() { if (address_) ::munmap(address_, bytes_); }
  void *address() const noexcept { return address_; }
private:
  void *address_ = nullptr;
  uint64_t bytes_;
};
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
      !buffer.contents() || !bytes || buffer.sizeBytes() < bytes)
    fail("INT8 expert store requires sufficient Shared buffer views");
}
void disjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents());
  const auto bb = reinterpret_cast<uintptr_t>(b.contents());
  if (!aa || !bb) fail("INT8 expert store requires addressable buffer views");
  if (aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes())
    fail("INT8 expert writable output overlaps an input or immutable operand");
}
void sourceGeometry(const FlashWeights &weights) {
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp";
    for (uint32_t plane = 0; plane < 3; ++plane) {
      const auto &p = weights.projectionMetadata(prefix + (plane == 0 ? ".gate_proj" : plane == 1 ? ".up_proj" : ".down_proj"));
      const uint32_t n = plane == 2 ? 2560 : 640, k = plane == 2 ? 640 : 2560;
      if (p.experts != 512 || p.outputSize != n || p.inputSize != k || p.bits != 4 || p.groupSize != 64 ||
          p.weightRowStrideBytes != k / 2 || p.weightExpertStrideBytes != uint64_t{n} * k / 2 ||
          p.parameterRowStrideBytes != k / 32 || p.parameterExpertStrideBytes != uint64_t{n} * k / 32 ||
          !p.weights || !p.scales || !p.biases || p.weights->dtype != FlashDType::U32 ||
          p.scales->dtype != FlashDType::BF16 || p.biases->dtype != FlashDType::BF16 ||
          p.weights->logicalBytes != uint64_t{512} * n * k / 2 ||
          p.scales->logicalBytes != uint64_t{512} * n * k / 32 ||
          p.biases->logicalBytes != uint64_t{512} * n * k / 32)
        fail("saved INT8 experts require original contiguous E512 Q4/G64 H2560 I640 source planes");
    }
  }
}
std::string pipeline(const char *phase, FlashMoEBlockedTile tile) {
  const uint32_t m = uint32_t(tile);
  if (m != 16 && m != 32 && m != 64) fail("saved INT8 expert tile must be M16/M32/M64");
  return std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(m) +
      (m == 64 ? "_n64_sg8" : "_n64");
}

// Metadata-only host validation. No source-Q4 buffers/graphs/count readback.
void allRowsScratch(const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
                    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) {
  const uint32_t m = uint32_t(tile);
  if (!rows || rows > 8192 || !selections || selections > 10 ||
      (m != 16 && m != 32 && m != 64) || rows > s.buckets.rowCapacity ||
      selections > s.buckets.selectionCapacity || uint64_t{rows} * selections > s.buckets.routeCapacity ||
      !flashMoEDirectAEnabled()) fail("all-row Full512 unsupported scratch/Direct-A geometry");
  if (m == 64 && (rows < 1024 || flashMoEBlockedTile(4096, false) != FlashMoEBlockedTile::M64N64))
    fail("all-row Full512 M64 requires the enabled wide policy");
  const uint32_t routes = rows * selections, jobs = moEBucketJobCapacity(rows, selections, m);
  if (jobs > s.buckets.jobCapacity) fail("all-row Full512 insufficient declared matrix jobs");
  requireBytes(diagnostics, 4);
  requireBytes(s.buckets.counts, 512 * 4); requireBytes(s.buckets.offsets, 513 * 4);
  requireBytes(s.buckets.routeMap, uint64_t{routes} * 4);
  requireBytes(s.buckets.canonicalToPacked, uint64_t{routes} * 4);
  requireBytes(s.buckets.jobOffsets, 513 * 4); requireBytes(s.buckets.jobCount, 4);
  requireBytes(s.buckets.tileJobs, uint64_t{jobs} * 8);
  requireBytes(s.buckets.packedInputs, uint64_t{routes + 63} * 2560 * 2);
  requireBytes(s.packedActivated, uint64_t{routes + 63} * 640 * 2);
  requireBytes(s.scatteredDown, uint64_t{routes} * 2560 * 2);
  const std::array scratch{s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics};
  for (size_t i = 0; i < scratch.size(); ++i)
    for (size_t j = i + 1; j < scratch.size(); ++j) disjoint(scratch[i], scratch[j]);
}
FlashInt8ExpertStoreParams allRowsParams(uint32_t rows, uint32_t selections,
                                        FlashMoEBlockedTile tile) {
  return {rows, selections, rows * selections,
      moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 512, 0, 0};
}
uint32_t allRowsLaunch(const FlashInt8ExpertStoreParams &p) {
  return p.rows < 256 ? std::min(p.route_capacity, p.job_capacity) : p.job_capacity;
}
} // namespace

// Device-free probes call actual inherited policy, fallback and ABI helpers.
std::string combinedSG2TailPipelineForCPU(bool gate) {
  return fixed_sg2_prefill_sep21::producerName(gate);
}
std::string combinedSG2TailFallbackPipelineForCPU(const char *phase, uint32_t tileRows) {
  return pipeline(phase, static_cast<FlashMoEBlockedTile>(tileRows));
}
bool combinedSG2TailEligibleForCPU(uint32_t rows, uint32_t tileRows, bool verification) {
  return fixed_sg2_prefill_sep21::eligible(rows, static_cast<FlashMoEBlockedTile>(tileRows), verification);
}
FlashInt8ExpertStoreParams combinedSG2TailParamsForCPU(uint32_t rows, uint32_t selections, uint32_t tileRows) {
  return allRowsParams(rows, selections, static_cast<FlashMoEBlockedTile>(tileRows));
}
uint32_t combinedSG2TailLaunchForCPU(const FlashInt8ExpertStoreParams &p) {
  return allRowsLaunch(p);
}

struct FlashInt8ExpertStore::Impl final {
  struct Layer final {
    metal::MetalBuffer base, ranks;
    std::array<metal::MetalBuffer, 3> codes, scales;
  };
  metal::MetalBackend &backend;
  FlashInt8ExpertStoreMetadata metadata;
  std::array<Layer, 48> layers;
  // Address ranges of the immutable layer buffers, captured once after load so
  // the per-dispatch overlap check does not query every buffer's contents().
  std::vector<std::pair<uintptr_t, uint64_t>> immutableRanges;
  uint64_t allocated = 0;
  std::string numericalIdentity;
  mutable std::atomic<uint64_t> fixedSG2GateCalls{0},fixedSG2GateRows{0},fixedSG2DownCalls{0},fixedSG2DownRows{0};
  const bool compactPreflight = compact_r4_preflight_sep22::requested();
  mutable std::atomic<uint64_t> compactPreflightCalls{0},compactPreflightRows{0};
  const bool compactR4Verify = compact_native_r4_verify_sep22::requested();
  mutable std::atomic<uint64_t> compactR4PlanCalls{0},compactR4PlanRows{0},compactR4GateCalls{0},compactR4GateRows{0},compactR4DownCalls{0},compactR4DownRows{0};
  const bool gatheredMPP = gathered_mpp::requested();
  const uint32_t gatheredMPPMaximumRows = gathered_mpp::requestedMaximumRows();
  mutable std::atomic<uint64_t> gatheredMPPGateCalls{0}, gatheredMPPGateRows{0}, gatheredMPPDownCalls{0}, gatheredMPPDownRows{0};
  mutable std::atomic<uint64_t> gateCalls{0}, gateRows{0}, downCalls{0}, downRows{0};
  mutable std::atomic<uint64_t> hitDispatches{0}, missDispatches{0}, fullCalls{0};
  mutable std::atomic<uint64_t> largeGateCalls{0}, largeGateRows{0}, largeDownCalls{0}, largeDownRows{0};
  mutable std::atomic<uint64_t> largeHits{0}, largeMisses{0}, largeFullCalls{0};
  void recordGraph(bool down, uint32_t rows, uint32_t inventory) const noexcept {
    (down ? downCalls : gateCalls).fetch_add(1, std::memory_order_relaxed);
    (down ? downRows : gateRows).fetch_add(rows, std::memory_order_relaxed);
    hitDispatches.fetch_add(1, std::memory_order_relaxed);
    if (inventory <512) missDispatches.fetch_add(1, std::memory_order_relaxed);
    else fullCalls.fetch_add(1, std::memory_order_relaxed);
    if (rows >= 256) {
      (down ? largeDownCalls : largeGateCalls).fetch_add(1, std::memory_order_relaxed);
      (down ? largeDownRows : largeGateRows).fetch_add(rows, std::memory_order_relaxed);
      largeHits.fetch_add(1, std::memory_order_relaxed);
      if (inventory < 512) largeMisses.fetch_add(1, std::memory_order_relaxed);
      else largeFullCalls.fetch_add(1, std::memory_order_relaxed);
    }
  }

  Impl(metal::MetalBackend &b, const FlashWeights &weights, const std::filesystem::path &directory)
      : backend(b), metadata(loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention())) {
    sourceGeometry(weights);
    for (const auto &entry : metadata.layers) {
      if (entry.selectedIDs.size() != 512) fail("all-row target requires Full512 inventory");
      for (uint32_t id = 0; id < 512; ++id)
        if (entry.selectedIDs[id] != id) fail("all-row target requires canonical complete expert IDs");
    }
    std::string derivative = std::string("splash.private-allrows-target-v1\nsource=") +
        weights.manifestFingerprint() + "\nstore=" + metadata.identitySha256 +
        "\npolicy=" + kFlashInt8ExpertStoreSemantics + "\nmtp=original-trained-bank\n";
    if (gatheredMPP)
      derivative += std::string("small_row_policy=") + std::string(gathered_mpp::kPolicy) + "\nsmall_row_cap_policy=" +
          std::string(gathered_mpp::kRowCapPolicy) + "\nsmall_row_max_physical_rows=" +
          std::to_string(gatheredMPPMaximumRows) + "\n";
    if (compactR4Verify)
      derivative += std::string("singleton_r4_verify_execution_policy=")+compact_native_r4_verify_sep22::implementationMarker()+"\n";
    if (compactPreflight)
      derivative += std::string("CPU_only_compact_R4_preflight_policy=")+compact_r4_preflight_sep22::marker()+"\n";
    numericalIdentity = hash(derivative.data(), derivative.size());
    const uint64_t before = backend.memoryStats().allocatedBytes;
    for (uint32_t index = 0; index < 48; ++index) {
      auto &layer = layers[index];
      const auto &entry = metadata.layers[index];
      if (entry.bytes > backend.capabilities().maxBufferLengthBytes) fail("INT8 expert layer exceeds Metal buffer limit");
      auto mapping = std::make_shared<Mapping>(entry.path, entry.bytes);
      const auto *data = static_cast<const uint8_t *>(mapping->address());
      if (hash(data, entry.bytes) != entry.sha256) fail("INT8 expert payload checksum differs");
      uint64_t cursor = 0;
      for (uint32_t plane = 0; plane < 3; ++plane) {
        for (const auto &range : {entry.codes[plane], entry.scales[plane]}) {
          if (!std::all_of(data + cursor, data + range.offset, [](uint8_t v) { return v == 0; }))
            fail("INT8 expert plane alignment padding is nonzero");
          if (hash(data + range.offset, range.length) != range.sha256)
            fail("INT8 expert plane checksum differs");
          cursor = range.offset + range.length;
        }
        const auto &code = entry.codes[plane];
        if (std::find(data + code.offset, data + code.offset + code.length, uint8_t{128}) !=
            data + code.offset + code.length) fail("saved symmetric INT8 contains excluded -128 code");
        const auto &scale = entry.scales[plane];
        const auto *values = reinterpret_cast<const float *>(data + scale.offset);
        if (!std::all_of(values, values + scale.length / sizeof(float),
            [](float v) { return std::isfinite(v) && v > 0.0f; }))
          fail("saved INT8 expert scales must be finite and positive");
      }
      if (!std::all_of(data + cursor, data + entry.bytes, [](uint8_t v) { return v == 0; }))
        fail("INT8 expert file tail padding is nonzero");
      layer.base = backend.wrapSharedMemory(mapping->address(), entry.bytes, mapping, "saved selected signed INT8 expert layer");
      layer.ranks = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "saved INT8 expert ID-to-rank map");
      std::memset(layer.ranks.contents(), 0xff, kAlignment);
      auto *ranks = static_cast<uint32_t *>(layer.ranks.contents());
      for (uint32_t rank = 0; rank < entry.selectedIDs.size(); ++rank) ranks[entry.selectedIDs[rank]] = rank;
      for (uint32_t plane = 0; plane < 3; ++plane) {
        layer.codes[plane] = backend.view(layer.base, entry.codes[plane].offset, entry.codes[plane].length);
        layer.scales[plane] = backend.view(layer.base, entry.scales[plane].offset, entry.scales[plane].length);
      }
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before || after - before > metadata.plannedBytes) fail("saved INT8 store exceeds reserved bytes");
    allocated = after - before;
    for (const auto &layer : layers)
      for (const auto *buffer : {&layer.base, &layer.ranks}) {
        const auto address = reinterpret_cast<uintptr_t>(buffer->contents());
        if (!address) fail("INT8 expert store requires addressable buffer views");
        immutableRanges.emplace_back(address, buffer->sizeBytes());
      }
  }
  const Layer &layer(uint32_t index) const {
    if (index >= 48) fail("saved INT8 expert layer is outside target layers");
    return layers[index];
  }
  void immutableDisjoint(const metal::MetalBuffer &output) const {
    const auto aa = reinterpret_cast<uintptr_t>(output.contents());
    if (!aa || immutableRanges.size() != layers.size() * 2)
      fail("INT8 expert store requires addressable buffer views");
    const uint64_t size = output.sizeBytes();
    for (const auto &[bb, bytes] : immutableRanges)
      if (aa <= bb ? uint64_t(bb - aa) < size : uint64_t(aa - bb) < bytes)
        fail("INT8 expert writable output overlaps an input or immutable operand");
  }
};

FlashInt8ExpertStore::FlashInt8ExpertStore(metal::MetalBackend &b, const FlashWeights &w,
    const std::filesystem::path &directory) : impl_(std::make_unique<Impl>(b, w, directory)) {}
FlashInt8ExpertStore::~FlashInt8ExpertStore() = default;
FlashInt8ExpertStore::FlashInt8ExpertStore(FlashInt8ExpertStore &&) noexcept = default;
FlashInt8ExpertStore &FlashInt8ExpertStore::operator=(FlashInt8ExpertStore &&) noexcept = default;
uint64_t FlashInt8ExpertStore::plannedBytes(const FlashWeights &weights, const std::filesystem::path &directory) {
  sourceGeometry(weights);
  const auto metadata = loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),
      weights.manifestFingerprint(), weights.normConvention());
  for (const auto &entry : metadata.layers)
    if (entry.selectedIDs.size() != 512) fail("all-row target planning requires Full512 inventory");
  return metadata.plannedBytes;
}
const std::string &FlashInt8ExpertStore::identitySha256() const { return impl_->metadata.identitySha256; }
const std::string &FlashInt8ExpertStore::numericalIdentitySha256() const { return impl_->numericalIdentity; }
const std::string &FlashInt8ExpertStore::planSha256() const { return impl_->metadata.planSha256; }
uint64_t FlashInt8ExpertStore::mappedBytes() const noexcept { return impl_ ? impl_->metadata.totalBytes : 0; }
uint64_t FlashInt8ExpertStore::actualAllocatedBytes() const noexcept { return impl_ ? impl_->allocated : 0; }
std::span<const uint32_t> FlashInt8ExpertStore::selectedExpertIDs(uint32_t index) const {
  (void)impl_->layer(index); return impl_->metadata.layers[index].selectedIDs;
}
std::vector<metal::MetalBuffer> FlashInt8ExpertStore::immutableWeightBuffers() const {
  std::vector<metal::MetalBuffer> result;
  for (const auto &layer : impl_->layers) { result.push_back(layer.base); result.push_back(layer.ranks); }
  return result;
}

FlashInt8ExpertStoreGraphCounters FlashInt8ExpertStore::graphCounters() const noexcept {
  if (!impl_) return {};
  return {impl_->gateCalls.load(std::memory_order_relaxed), impl_->gateRows.load(std::memory_order_relaxed),
      impl_->downCalls.load(std::memory_order_relaxed), impl_->downRows.load(std::memory_order_relaxed),
      impl_->hitDispatches.load(std::memory_order_relaxed), impl_->missDispatches.load(std::memory_order_relaxed),
      impl_->fullCalls.load(std::memory_order_relaxed),
      impl_->largeGateCalls.load(std::memory_order_relaxed), impl_->largeGateRows.load(std::memory_order_relaxed),
      impl_->largeDownCalls.load(std::memory_order_relaxed), impl_->largeDownRows.load(std::memory_order_relaxed),
      impl_->largeHits.load(std::memory_order_relaxed), impl_->largeMisses.load(std::memory_order_relaxed),
      impl_->largeFullCalls.load(std::memory_order_relaxed),
      impl_->gatheredMPPGateCalls.load(std::memory_order_relaxed), impl_->gatheredMPPGateRows.load(std::memory_order_relaxed),
      impl_->gatheredMPPDownCalls.load(std::memory_order_relaxed), impl_->gatheredMPPDownRows.load(std::memory_order_relaxed)};
}


void FlashInt8ExpertStore::addGateUp(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("gate_up", tile), {s.buckets.packedInputs, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, p,
      {10, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(false, rows, 512);
}
void FlashInt8ExpertStore::addDownScatter(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const uint32_t routes = rows * selections;
  const FlashMoEBlockedDownParams poison{{rows, selections, 640, 2560, 512, 0, 0, 0,
      320, uint64_t{2560} * 320, 20, uint64_t{2560} * 20},
      routes, moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 0};
  pointwise_sep21::addPoison(graph,
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison);
  graph.add("flash_moe_direct_a_prepare_down",
      {s.packedActivated, s.buckets.offsets, s.packedActivated, diagnostics},
      FlashMoEDirectAPrepareParams{routes, 640, 63, 0}, {routes + 63, 1, 1}, {256, 1, 1});
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("down_scatter", tile), {s.packedActivated, layer.codes[2], layer.scales[2],
      layer.ranks, s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
      s.buckets.routeMap, s.scatteredDown, diagnostics}, p,
      {40, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(true, rows, 512);
}

void FlashInt8ExpertStore::addFixedSG2PrefillGateUp(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections) const {
  if (!fixed_sg2_prefill_sep21::eligible(rows,tile,false) ||selections !=10) fail("fixedSG2 only main R2048/M32 canonical source");
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads=fixed_sg2_prefill_sep21::producerThreads();
  graph.add(fixed_sg2_prefill_sep21::producerName(true), {s.buckets.packedInputs, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, p,
      {10, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(false, rows, 512);
  impl_->fixedSG2GateCalls.fetch_add(1,std::memory_order_relaxed);
  impl_->fixedSG2GateRows.fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addFixedSG2PrefillDownScatter(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections) const {
  if (!fixed_sg2_prefill_sep21::eligible(rows,tile,false) ||selections !=10) fail("fixedSG2 only main R2048/M32 canonical source");
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const uint32_t routes = rows * selections;
  const FlashMoEBlockedDownParams poison{{rows, selections, 640, 2560, 512, 0, 0, 0,
      320, uint64_t{2560} * 320, 20, uint64_t{2560} * 20},
      routes, moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 0};
  pointwise_sep21::addPoison(graph,
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison);
  graph.add("flash_moe_direct_a_prepare_down",
      {s.packedActivated, s.buckets.offsets, s.packedActivated, diagnostics},
      FlashMoEDirectAPrepareParams{routes, 640, 63, 0}, {routes + 63, 1, 1}, {256, 1, 1});
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads=fixed_sg2_prefill_sep21::producerThreads();
  graph.add(fixed_sg2_prefill_sep21::producerName(false), {s.packedActivated, layer.codes[2], layer.scales[2],
      layer.ranks, s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
      s.buckets.routeMap, s.scatteredDown, diagnostics}, p,
      {40, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(true, rows, 512);
  impl_->fixedSG2DownCalls.fetch_add(1,std::memory_order_relaxed);
  impl_->fixedSG2DownRows.fetch_add(rows,std::memory_order_relaxed);
}

fixed_sg2_prefill_sep21::Counters FlashInt8ExpertStore::fixedSG2PrefillCounters() const noexcept {
  return {fixed_sg2_prefill_sep21::requested(),impl_->fixedSG2GateCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2GateRows.load(std::memory_order_relaxed),impl_->fixedSG2DownCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2DownRows.load(std::memory_order_relaxed)};
}

namespace {
void gatheredMPPViews(const gathered_mpp::Geometry &g,
    metal::MetalBuffer input, metal::MetalBuffer ids, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, bool down) {
  requireBytes(input, down ? g.intermediateBytes : g.inputBytes);
  requireBytes(ids, g.idBytes);
  requireBytes(output, down ? g.expertDownBytes : g.intermediateBytes);
  requireBytes(diagnostics, 4);
  const auto view = [](const metal::MetalBuffer &b) {
    return gathered_mpp::ByteView{reinterpret_cast<uintptr_t>(b.contents()), b.sizeBytes()};
  };
  gathered_mpp::validateViews(g, view(input), view(ids), view(output), view(diagnostics), down);
}
} // namespace
bool FlashInt8ExpertStore::gatheredMPPEnabled() const {
  if (!impl_) fail("private gathered I8 MPP Store was moved or disposed");
  if (gathered_mpp::requested() != impl_->gatheredMPP)
    fail("private gathered I8 MPP flag changed after Store construction");
  if (gathered_mpp::requestedMaximumRows() != impl_->gatheredMPPMaximumRows)
    fail("private gathered I8 MPP maximum rows changed after Store construction");
  return impl_->gatheredMPP;
}
uint32_t FlashInt8ExpertStore::gatheredMPPMaximumRows() const {
  (void)gatheredMPPEnabled();
  return impl_->gatheredMPPMaximumRows;
}
void FlashInt8ExpertStore::addGatheredMPPGateUp(metal::CommandGraph &graph, uint32_t index,
    metal::MetalBuffer input, metal::MetalBuffer originalExpertIDs,
    metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer diagnostics,
    uint32_t rows, uint32_t selections) const {
  if (!gatheredMPPEnabled()) fail("private gathered I8 MPP gate/up requires frozen flag1");
  if (rows > gatheredMPPMaximumRows()) fail("private gathered MPP rows exceed frozen route cap");
  const auto g = gathered_mpp::geometry(rows, selections);
  const auto &layer = impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("private gathered I8 MPP requires Full512");
  gatheredMPPViews(g, input, originalExpertIDs, canonicalIntermediate, diagnostics, false);
  for (const auto &buffer : {input, originalExpertIDs, canonicalIntermediate, diagnostics})
    impl_->immutableDisjoint(buffer);
  requireBytes(layer.ranks, 512 * 4);
  for (uint32_t p = 0; p < 2; ++p) {
    requireBytes(layer.codes[p], uint64_t{512} * 640 * 2560);
    requireBytes(layer.scales[p], uint64_t{512} * 640 * 4);
  }
  graph.add("flash_gathered_mpp_gate_up_m16_n64_sg4", {input, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, originalExpertIDs, canonicalIntermediate,
      diagnostics}, FlashGatheredMPPParams{rows, selections, 512, 0},
      {g.gateColumnGroups, rows, selections}, {gathered_mpp::kThreads, 1, 1});
  impl_->recordGraph(false, rows, 512);
  impl_->gatheredMPPGateCalls.fetch_add(1, std::memory_order_relaxed);
  impl_->gatheredMPPGateRows.fetch_add(rows, std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addGatheredMPPDown(metal::CommandGraph &graph, uint32_t index,
    metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer originalExpertIDs,
    metal::MetalBuffer canonicalExpertDown, metal::MetalBuffer diagnostics,
    uint32_t rows, uint32_t selections) const {
  if (!gatheredMPPEnabled()) fail("private gathered I8 MPP down requires frozen flag1");
  if (rows > gatheredMPPMaximumRows()) fail("private gathered MPP rows exceed frozen route cap");
  const auto g = gathered_mpp::geometry(rows, selections);
  const auto &layer = impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("private gathered I8 MPP requires Full512");
  gatheredMPPViews(g, canonicalIntermediate, originalExpertIDs, canonicalExpertDown, diagnostics, true);
  for (const auto &buffer : {canonicalIntermediate, originalExpertIDs, canonicalExpertDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  requireBytes(layer.ranks, 512 * 4);
  requireBytes(layer.codes[2], uint64_t{512} * 2560 * 640);
  requireBytes(layer.scales[2], uint64_t{512} * 2560 * 4);
  graph.add("flash_gathered_mpp_down_m16_n64_sg4", {canonicalIntermediate, layer.codes[2],
      layer.scales[2], layer.ranks, originalExpertIDs, canonicalExpertDown, diagnostics},
      FlashGatheredMPPParams{rows, selections, 512, 0},
      {g.downColumnGroups, rows, selections}, {gathered_mpp::kThreads, 1, 1});
  impl_->recordGraph(true, rows, 512);
  impl_->gatheredMPPDownCalls.fetch_add(1, std::memory_order_relaxed);
  impl_->gatheredMPPDownRows.fetch_add(rows, std::memory_order_relaxed);
}
bool FlashInt8ExpertStore::compactNativeR4VerifyEnabled() const {
  if (!impl_) fail("compact R4 verifier Store disposed");
  if (compact_native_r4_verify_sep22::requested()!=impl_->compactR4Verify)
    fail("compact R4 verifier flag changed after construction");
  if (impl_->compactR4Verify && (!gatheredMPPEnabled()||gatheredMPPMaximumRows()!=4))
    fail("compact R4 verifier requires original gathered cap exactly4");
  return impl_->compactR4Verify;
}
compact_native_r4_verify_sep22::Counters FlashInt8ExpertStore::compactNativeR4VerifyCounters() const {
  return {compactNativeR4VerifyEnabled(),impl_->compactR4PlanCalls.load(std::memory_order_relaxed),impl_->compactR4PlanRows.load(std::memory_order_relaxed),
      impl_->compactR4GateCalls.load(std::memory_order_relaxed),impl_->compactR4GateRows.load(std::memory_order_relaxed),
      impl_->compactR4DownCalls.load(std::memory_order_relaxed),impl_->compactR4DownRows.load(std::memory_order_relaxed)};
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyPack(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier only canonical R4/S10");
  const auto &layer=impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size()!=512) fail("compact verifier requires Full512 inventory");
  allRowsScratch(s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  requireBytes(input,uint64_t{4}*2560*2);requireBytes(ids,40*8);requireBytes(layer.ranks,512*4);
  disjoint(input,ids);
  for (const auto &b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.packedInputs,
      s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,s.packedActivated,s.scatteredDown,diagnostics}) {
    disjoint(input,b);disjoint(ids,b);impl_->immutableDisjoint(b);
  }
  impl_->immutableDisjoint(input);impl_->immutableDisjoint(ids);
  graph.add("expert_r4_compact_native_sep22_plan",{ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
      s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diagnostics},FlashMoEBucketParams{4,10,2560,512,40,16,514,0},{1,1,1},{256,1,1});
  graph.add("flash_moe_direct_a_pack",{input,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diagnostics},
      FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1});
  impl_->compactR4PlanCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4PlanRows.fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyGateUp(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier gate only R4/S10");
  addGateUp(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  impl_->compactR4GateCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4GateRows.fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyDown(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier down only R4/S10");
  addDownScatter(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  impl_->compactR4DownCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4DownRows.fetch_add(rows,std::memory_order_relaxed);
}
compact_r4_preflight_sep22::Counters FlashInt8ExpertStore::compactR4PreflightCounters() const {
  if(!impl_)fail("compact R4 preflight Store disposed");
  if(compact_r4_preflight_sep22::requested()!=impl_->compactPreflight)fail("compact R4 preflight flag changed");
  return {impl_->compactPreflight,impl_->compactPreflightCalls.load(std::memory_order_relaxed),impl_->compactPreflightRows.load(std::memory_order_relaxed)};
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyChain(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if(!compactNativeR4VerifyEnabled()||rows!=4||selections!=10)fail("compact verifier only canonical R4/S10");
  if(!compactR4PreflightCounters().enabled)fail("compact R4 complete preflight requires frozen flag1");
  const auto &sourceLayer=impl_->layer(index);
  if(impl_->metadata.layers[index].selectedIDs.size()!=512)fail("compact verifier requires Full512 inventory");
  // This type has lexical method scope. It cannot escape or be constructed by
  // a caller, and it is never cached. Copies bind exact allocation/view refs.
  struct CapturedViews final {
    const FlashMoEBlockedScratch s;
    const metal::MetalBuffer input,ids,diag;
    const std::remove_cvref_t<decltype(sourceLayer)> source;
    const uint32_t rows,selections;
    const FlashMoEBlockedTile tile;
  };
  // Capture before admission: no caller scratch/source is consulted by append.
  const CapturedViews captured{scratch,input,ids,diagnostics,sourceLayer,rows,selections,FlashMoEBlockedTile::M16N64};
  compact_r4_preflight_sep22::validateComplete(captured.s,captured.input,captured.ids,captured.source.ranks,captured.diag,captured.rows,captured.selections,
      [](const auto &s,auto d,uint32_t r,uint32_t n){allRowsScratch(s,d,r,FlashMoEBlockedTile::M16N64,n);},
      [](const auto &b,uint64_t n){requireBytes(b,n);},
      [](const auto &a,const auto &b){disjoint(a,b);},
      [&](const auto &b){impl_->immutableDisjoint(b);});
  // Noncopyable, nonmovable, single-consume token is constructed ONLY after
  // admission. Its const snapshot reference cannot outlive this method.
  struct ValidatedBundle final {
    const void *const storeEpoch;
    const void *const layerEpoch;
    metal::CommandGraph *const graphOwner;
    const uint32_t layerIndex;
    const CapturedViews &v;
    bool consumed=false;
    ValidatedBundle(const void *owner,const void *selected,metal::CommandGraph *g,uint32_t layer,const CapturedViews &views)
        :storeEpoch(owner),layerEpoch(selected),graphOwner(g),layerIndex(layer),v(views){}
    ValidatedBundle(const ValidatedBundle &)=delete;
    ValidatedBundle &operator=(const ValidatedBundle &)=delete;
    ValidatedBundle(ValidatedBundle &&)=delete;
    ValidatedBundle &operator=(ValidatedBundle &&)=delete;
  };
  static_assert(!std::is_copy_constructible_v<ValidatedBundle> && !std::is_move_constructible_v<ValidatedBundle>);
  ValidatedBundle bundle{impl_.get(),&sourceLayer,&graph,index,captured};
  // Only this private lexical appender can consume the admitted token. Epoch,
  // graph and immutable source-view identities are rechecked at consumption.
  const auto appendValidated=[&](ValidatedBundle &b) {
    if(b.consumed||b.storeEpoch!=impl_.get()||b.layerEpoch!=&sourceLayer||b.graphOwner!=&graph||b.layerIndex!=index||
        b.v.rows!=4||b.v.selections!=10||b.v.tile!=FlashMoEBlockedTile::M16N64||
        !b.v.source.base.sameView(sourceLayer.base)||!b.v.source.ranks.sameView(sourceLayer.ranks))
      fail("compact R4 admitted bundle owner/epoch/source changed");
    for(uint32_t plane=0;plane<3;++plane)
      if(!b.v.source.codes[plane].sameView(sourceLayer.codes[plane])||!b.v.source.scales[plane].sameView(sourceLayer.scales[plane]))
        fail("compact R4 admitted immutable source view changed");
    if(!compactNativeR4VerifyEnabled()||!compactR4PreflightCounters().enabled)
      fail("compact R4 admitted frozen policy changed");
    b.consumed=true;
    const auto &s=b.v.s;const auto &layer=b.v.source;const auto diagnostics=b.v.diag;
    const auto rows=b.v.rows,selections=b.v.selections;const auto tile=b.v.tile;
    graph.add("expert_r4_compact_native_sep22_plan",{b.v.ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
        s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diagnostics},FlashMoEBucketParams{4,10,2560,512,40,16,514,0},{1,1,1},{256,1,1});
    graph.add("flash_moe_direct_a_pack",{b.v.input,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diagnostics},
        FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1});
    impl_->compactR4PlanCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4PlanRows.fetch_add(rows,std::memory_order_relaxed);
    { // Original public gate/up producer suffix, literal and unmodified.
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("gate_up", tile), {s.buckets.packedInputs, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, p,
      {10, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(false, rows, 512);
    }
    impl_->compactR4GateCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4GateRows.fetch_add(rows,std::memory_order_relaxed);
    { // Original public down producer suffix, literal and unmodified.
  const uint32_t routes = rows * selections;
  const FlashMoEBlockedDownParams poison{{rows, selections, 640, 2560, 512, 0, 0, 0,
      320, uint64_t{2560} * 320, 20, uint64_t{2560} * 20},
      routes, moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 0};
  pointwise_sep21::addPoison(graph,
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison);
  graph.add("flash_moe_direct_a_prepare_down",
      {s.packedActivated, s.buckets.offsets, s.packedActivated, diagnostics},
      FlashMoEDirectAPrepareParams{routes, 640, 63, 0}, {routes + 63, 1, 1}, {256, 1, 1});
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("down_scatter", tile), {s.packedActivated, layer.codes[2], layer.scales[2],
      layer.ranks, s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
      s.buckets.routeMap, s.scatteredDown, diagnostics}, p,
      {40, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(true, rows, 512);
    }
    impl_->compactR4DownCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4DownRows.fetch_add(rows,std::memory_order_relaxed);
  };
  appendValidated(bundle);
  impl_->compactPreflightCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactPreflightRows.fetch_add(rows,std::memory_order_relaxed);
}
} // namespace splash::flash
