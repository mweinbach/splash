#include "worker_cache.hpp"

#include "abi.hpp"
#include "quantization.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace splash::flash::dense_w8a8_sep21 {
namespace {
namespace quant = splash::dense_w8a8;

struct StringHash final {
  using is_transparent = void;
  size_t operator()(std::string_view name) const noexcept {
    return std::hash<std::string_view>{}(name);
  }
};

void requireShared(const metal::MetalBuffer &buffer, uint64_t bytes,
                   const std::string &name) {
  if (!buffer || !bytes || buffer.sizeBytes() < bytes ||
      buffer.storage() != metal::BufferStorage::Shared || !buffer.contents())
    throw std::invalid_argument("Private dense W8A8 requires sufficient Shared " + name);
}

void requireSharedBF16(const FlashTensor &source, Geometry expected,
                       const std::string &prefix) {
  const uint64_t bytes = uint64_t{expected.n} * expected.k * 2;
  if (!sourceMetadataMatches(expected, source.dtype, source.shape,
        source.logicalBytes, source.buffer.sizeBytes(), source.buffer.storage()))
    throw std::invalid_argument("Private dense W8A8 source BF16 shape/extent differs: " + prefix);
  requireShared(source.buffer, bytes, "BF16 source " + prefix);
}

std::string sha256(const void *data, uint64_t bytes) {
  if ((!data && bytes) || bytes > std::numeric_limits<CC_LONG>::max())
    throw std::invalid_argument("Private dense W8A8 SHA256 extent is invalid");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(data, static_cast<CC_LONG>(bytes), digest.data()))
    throw std::runtime_error("Private dense W8A8 SHA256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (unsigned char byte : digest) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}

void appendField(std::string &fingerprint, std::string_view key,
                 std::string_view value) {
  fingerprint.append(key);
  fingerprint += ':';
  fingerprint += std::to_string(value.size());
  fingerprint += ':';
  fingerprint.append(value);
  fingerprint += '\n';
}

bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const uintptr_t first = reinterpret_cast<uintptr_t>(a.contents());
  const uintptr_t second = reinterpret_cast<uintptr_t>(b.contents());
  if (!first || !second ||
      first > std::numeric_limits<uintptr_t>::max() - a.sizeBytes() ||
      second > std::numeric_limits<uintptr_t>::max() - b.sizeBytes())
    throw std::invalid_argument("Private dense W8A8 Shared buffer address extent is invalid");
  return first < second + b.sizeBytes() && second < first + a.sizeBytes();
}

void requireIndependent(std::span<const metal::MetalBuffer> buffers) {
  for (size_t i = 0; i < buffers.size(); ++i)
    for (size_t j = i + 1; j < buffers.size(); ++j)
      if (overlaps(buffers[i], buffers[j]))
        throw std::invalid_argument("Private dense W8A8 buffers overlap");
}

void requireDescriptor(const FlashDescriptor &descriptor) {
  if (descriptor.layers != 48 || descriptor.hiddenSize != 2560)
    throw std::invalid_argument("Private dense W8A8 requires the canonical 48-layer descriptor");
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto expected = layer % 4 == 3 ? FlashLayerKind::SparseAttention
                                       : FlashLayerKind::GatedDeltaNet;
    if (descriptor.layerKinds[layer] != expected)
      throw std::invalid_argument("Private dense W8A8 layer kind differs from canonical policy");
  }
}
} // namespace

struct Cache::Impl final {
  std::unordered_map<std::string, Projection, StringHash, std::equal_to<>> projections;
  std::vector<metal::MetalBuffer> coefficientBases;
  std::string identity;
  uint64_t allocatedBytes = 0;

  Impl(metal::MetalBackend &backend, const FlashWeights &weights,
       const FlashDenseCache &denseCache) {
    // Finish metadata-only source validation before allocating or touching
    // coefficient bytes. No raw affine weights, GPU commands or files are used.
    requireDescriptor(weights.descriptor());
    const auto names = selectedPrefixes();
    for (const auto &prefix : names) {
      const auto expected = geometry(prefix);
      if (!expected || !denseCache.contains(prefix))
        throw std::invalid_argument("Private dense W8A8 BF16 source cache is incomplete: " + prefix);
      requireSharedBF16(denseCache.tensor(prefix), expected, prefix);
    }

    std::string fingerprint;
    appendField(fingerprint, "format", kOperandFormat);
    appendField(fingerprint, "execution", kExecutionSemantics);
    appendField(fingerprint, "source", weights.sourceIdentity());
    appendField(fingerprint, "manifest", weights.manifestFingerprint());
    appendField(fingerprint, "norm", weights.normConvention() == NormConvention::OnePlusWeight
                                         ? "one-plus-weight" : "direct-gamma");
    appendField(fingerprint, "bf16-cache", denseCache.identitySha256());
    coefficientBases.reserve(kImmutableBufferCount);
    projections.reserve(kProjectionCount);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    for (const auto &prefix : names) {
      const auto expected = geometry(prefix);
      const auto &source = denseCache.tensor(prefix);
      Projection projection;
      projection.k = expected.k;
      projection.n = expected.n;
      projection.sourceSHA = sha256(source.buffer.contents(), source.logicalBytes);
      const uint64_t codeBytes = uint64_t{expected.n} * expected.k;
      const uint64_t scaleBytes = uint64_t{expected.n} * 4;
      auto codeBase = backend.allocateBuffer(roundedBytes(codeBytes),
          metal::BufferStorage::Shared, "private-dense-w8a8-codes:" + prefix);
      auto scaleBase = backend.allocateBuffer(roundedBytes(scaleBytes),
          metal::BufferStorage::Shared, "private-dense-w8a8-scales:" + prefix);
      projection.codes = backend.view(codeBase, 0, codeBytes);
      projection.scales = backend.view(scaleBase, 0, scaleBytes);
      requireShared(projection.codes, codeBytes, "coefficient codes " + prefix);
      requireShared(projection.scales, scaleBytes, "coefficient scales " + prefix);
      const auto *bf16 = static_cast<const uint16_t *>(source.buffer.contents());
      auto *codes = static_cast<int8_t *>(projection.codes.contents());
      auto *scales = static_cast<float *>(projection.scales.contents());
      quant::QuantError coefficientError;
      for (uint32_t row = 0; row < expected.n; ++row)
        quant::quantizeRow(bf16 + uint64_t{row} * expected.k, expected.k,
                          codes + uint64_t{row} * expected.k, scales[row], coefficientError);
      if (coefficientError.nonfinite() || coefficientError.clippedCount())
        throw std::runtime_error("Private dense W8A8 coefficient fitting diagnostics failed");
      projection.codesSHA = sha256(codes, codeBytes);
      projection.scalesSHA = sha256(scales, scaleBytes);
      appendField(fingerprint, "projection", prefix);
      appendField(fingerprint, "geometry", std::to_string(expected.n) + "," + std::to_string(expected.k));
      appendField(fingerprint, "source-bf16-sha256", projection.sourceSHA);
      appendField(fingerprint, "i8-sha256", projection.codesSHA);
      appendField(fingerprint, "f32-scale-sha256", projection.scalesSHA);
      coefficientBases.push_back(std::move(codeBase));
      coefficientBases.push_back(std::move(scaleBase));
      if (!projections.emplace(prefix, std::move(projection)).second)
        throw std::logic_error("Private dense W8A8 canonical selection contains duplicates");
    }
    if (projections.size() != kProjectionCount || coefficientBases.size() != kImmutableBufferCount)
      throw std::logic_error("Private dense W8A8 coefficient census differs from policy");
    identity = sha256(fingerprint.data(), fingerprint.size());
    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
    if (allocatedBytes > Cache::plannedBytes())
      throw std::runtime_error("Private dense W8A8 cache exceeded conservative planned allocation");
  }
};

Cache::Cache(metal::MetalBackend &backend, const FlashWeights &weights,
             const FlashDenseCache &denseCache)
    : impl_(std::make_unique<Impl>(backend, weights, denseCache)) {}
Cache::~Cache() = default;
Cache::Cache(Cache &&) noexcept = default;
Cache &Cache::operator=(Cache &&) noexcept = default;

bool Cache::contains(std::string_view prefix) const noexcept {
  return impl_ && impl_->projections.contains(prefix);
}

const Projection &Cache::projection(std::string_view prefix) const {
  if (!impl_) throw std::logic_error("Private dense W8A8 cache is not initialized");
  const auto found = impl_->projections.find(prefix);
  if (found == impl_->projections.end())
    throw std::invalid_argument("Private dense W8A8 projection was not selected: " + std::string(prefix));
  return found->second;
}

std::vector<metal::MetalBuffer> Cache::immutableWeightBuffers() const {
  if (!impl_) return {};
  if (impl_->projections.size() != kProjectionCount ||
      impl_->coefficientBases.size() != kImmutableBufferCount)
    throw std::logic_error("Private dense W8A8 immutable coefficient census differs from policy");
  uint64_t backingBytes = 0;
  for (size_t i = 0; i < impl_->coefficientBases.size(); ++i) {
    const auto &base = impl_->coefficientBases[i];
    const uint64_t bytes = base.sizeBytes();
    if (!base || base.storage() != metal::BufferStorage::Shared || !bytes ||
        bytes % kAllocationAlignment ||
        backingBytes > std::numeric_limits<uint64_t>::max() - bytes)
      throw std::logic_error("Private dense W8A8 immutable coefficient backing is invalid");
    for (size_t j = 0; j < i; ++j)
      if (base.sameView(impl_->coefficientBases[j]))
        throw std::logic_error("Private dense W8A8 immutable coefficient backing is duplicated");
    backingBytes += bytes;
  }
  if (backingBytes != plannedBytes())
    throw std::logic_error("Private dense W8A8 immutable coefficient backing bytes differ from policy");
  // Entries are the complete, privately retained allocations, never their
  // exact operand subviews. This audit reads allocation metadata only.
  return impl_->coefficientBases;
}

const std::string &Cache::identitySha256() const {
  if (!impl_) throw std::logic_error("Private dense W8A8 cache is not initialized");
  return impl_->identity;
}

uint64_t Cache::actualAllocatedBytes() const noexcept {
  return impl_ ? impl_->allocatedBytes : 0;
}

Workspace::Workspace(metal::MetalBackend &backend) {
  const auto sizes = logicalBufferBytes();
  const uint64_t before = backend.memoryStats().allocatedBytes;
  constexpr std::array<const char *, 3> labels{
      "private-dense-w8a8-activation-codes", "private-dense-w8a8-activation-scales",
      "private-dense-w8a8-unused-i32-binding"};
  for (size_t i = 0; i < bases_.size(); ++i)
    bases_[i] = backend.allocateBuffer(roundedBytes(sizes[i]), metal::BufferStorage::Shared, labels[i]);
  codes = backend.view(bases_[0], 0, sizes[0]);
  inputScales = backend.view(bases_[1], 0, sizes[1]);
  dummyDot = backend.view(bases_[2], 0, sizes[2]);
  for (size_t i = 0; i < bases_.size(); ++i) requireShared(bases_[i], sizes[i], labels[i]);
  // Establish a harmless binding value on the CPU; no normal GPU entry can
  // access an I32 output, since its Probe template argument is false.
  std::memset(dummyDot.contents(), 0, sizeof(int32_t));
  allocatedBytes_ = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (allocatedBytes_ > plannedBytes())
    throw std::runtime_error("Private dense W8A8 workspace exceeded conservative planned allocation");
}

bool addProjection(metal::CommandGraph &graph, const Cache &cache,
    std::string_view prefix, metal::MetalBuffer input, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, uint32_t rows, bool verify,
    const Workspace &workspace, uint32_t groups) {
  const auto expected = geometry(prefix);
  const auto plan = projectionPlan(prefix, rows, expected.n, expected.k, verify, groups);
  if (!plan || !cache.contains(prefix)) return false;
  const auto &coefficient = cache.projection(prefix);
  if (coefficient.k != expected.k || coefficient.n != expected.n)
    throw std::logic_error("Private dense W8A8 coefficient geometry differs from policy");
  requireShared(input, uint64_t{rows} * expected.k * 2, "BF16 input");
  requireShared(output, uint64_t{rows} * expected.n * 2, "BF16 output");
  requireShared(diagnostics, 4, "diagnostics");
  requireShared(workspace.codes, uint64_t{kRows} * kWorkspaceInputCapacity, "activation code workspace");
  requireShared(workspace.inputScales, uint64_t{kRows} * 4, "activation scale workspace");
  requireShared(workspace.dummyDot, 4, "unused I32 binding");
  requireShared(coefficient.codes, uint64_t{expected.n} * expected.k, "immutable coefficient codes");
  requireShared(coefficient.scales, uint64_t{expected.n} * 4, "immutable coefficient scales");
  const std::array<metal::MetalBuffer, 8> operands{
      input, output, diagnostics, workspace.codes, workspace.inputScales, workspace.dummyDot,
      coefficient.codes, coefficient.scales};
  requireIndependent(operands);

  const DenseW8A8QuantizeParams quantize{rows, expected.k};
  const FlashDenseCacheParams matmul{rows, expected.k, expected.n, 0, expected.n,
      plan.tileRows, plan.tileOutputs, uint32_t(plan.traversal)};
  const auto grid = flashDenseTraversalGrid(rows / plan.tileRows,
      expected.n / plan.tileOutputs, plan.traversal);
  graph.add("dense_w8a8_bf16_to_i8_t256",
      {input, workspace.codes, workspace.inputScales, diagnostics}, quantize,
      {rows, 1, 1}, {256, 1, 1});
  graph.add("dense_w8a8_m128_n64_sg4",
      {workspace.codes, coefficient.codes, output, workspace.dummyDot,
       workspace.inputScales, coefficient.scales, diagnostics}, matmul,
      {grid.x, grid.y, 1}, {128, 1, 1});
  return true;
}

} // namespace splash::flash::dense_w8a8_sep21
