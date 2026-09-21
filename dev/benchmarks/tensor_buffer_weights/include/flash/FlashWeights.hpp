#pragma once

#include "flash/FlashDescriptor.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

enum class FlashDType : uint8_t { U32, BF16, I64, F32 };

struct FlashTensor final {
  metal::MetalBuffer buffer;
  FlashDType dtype = FlashDType::BF16;
  std::vector<uint64_t> shape;
  uint64_t logicalBytes = 0;
};

// Raw, aligned MLX affine storage. Strides are bytes; output/input sizes are
// logical dimensions with no implicit StorageN padding. Rank-2 projections
// have experts=1. Each pointer is stable for the owning FlashWeights lifetime,
// including moves; projection lookup performs no allocation or conversion.
struct FlashAffineProjection final {
  const FlashTensor *weights = nullptr;
  const FlashTensor *scales = nullptr;
  const FlashTensor *biases = nullptr;
  uint32_t experts = 1;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint32_t bits = 0;
  uint32_t groupSize = 0;
  uint64_t weightRowStrideBytes = 0;
  uint64_t weightExpertStrideBytes = 0;
  uint64_t parameterRowStrideBytes = 0;
  uint64_t parameterExpertStrideBytes = 0;
};

struct FlashOriginalTextResidencySelection final {
  std::vector<metal::MetalBuffer> buffers;
  std::vector<std::string> paths;
  uint64_t mappedBytes = 0;
  uint64_t excludedBaseCount = 0;
  uint64_t excludedMappedBytes = 0;
  uint64_t excludedPLEBaseCount = 0;
  uint64_t excludedVisionBaseCount = 0;
  uint64_t excludedUnknownBaseCount = 0;
};

struct FlashPrivateWeightStorageStats final {
  bool perTensorBuffers = false;
  uint64_t nativeBaseCount = 0, sourceMappingCount = 0;
  uint64_t logicalTensorBytes = 0, paddedNativeBytes = 0, paddingBytes = 0;
};

// An immutable, checked view of splash-local-qwen4-affine-v1 derived payloads.
// Original safetensors are not reinterpreted or mutated. No inference, GPU
// commands, residency requests or coefficient transforms occur during load.
class FlashWeights final {
public:
  FlashWeights();
  ~FlashWeights();
  FlashWeights(const FlashWeights &) = delete;
  FlashWeights &operator=(const FlashWeights &) = delete;
  FlashWeights(FlashWeights &&) noexcept;
  FlashWeights &operator=(FlashWeights &&) noexcept;

  [[nodiscard]] static FlashWeights
  load(metal::MetalBackend &backend, const std::filesystem::path &directory,
       bool verifyPayloadHashes = false);
  [[nodiscard]] const FlashDescriptor &descriptor() const;
  [[nodiscard]] const FlashTensor &tensor(std::string_view name) const;
  [[nodiscard]] const FlashAffineProjection &
  projection(std::string_view prefix) const;
  [[nodiscard]] bool contains(std::string_view name) const noexcept;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  // Checked immutable existing native bases only: no new buffers, payload
  // scans, or accounting charge. Whole bases containing PLE/vision/unknown
  // tensors are omitted, even if they also contain active text tensors.
  [[nodiscard]] FlashOriginalTextResidencySelection checkedOriginalTextResidency() const;
  [[nodiscard]] FlashPrivateWeightStorageStats privateStorageStats() const;
  [[nodiscard]] const std::string &manifestFingerprint() const;
  [[nodiscard]] const std::string &sourceIdentity() const;
  [[nodiscard]] const FlashNormAudit &normAudit() const;
  [[nodiscard]] NormConvention normConvention() const;
  // Returns the audited convention for Qwen4 residual RMSNorm tensors and
  // DirectGamma for the distinct gated-GDN and vision normalization weights.
  [[nodiscard]] NormConvention normConvention(std::string_view name) const;
  [[nodiscard]] uint64_t tensorCount() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
