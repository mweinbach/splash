#pragma once

// Private worker-only derived operands. Planning and policy functions below
// inspect fixed geometry only; constructing Cache reads existing BF16 cache
// operands and is reserved for the authorized worker startup path.
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash::dense_w8a8_sep21 {

inline constexpr uint64_t kAllocationAlignment = 16384;
inline constexpr uint32_t kRows = 2048;
inline constexpr uint32_t kWorkspaceInputCapacity = 6144;
inline constexpr uint32_t kProjectionCount = 84;
inline constexpr uint32_t kImmutableBufferCount = 2 * kProjectionCount;
inline constexpr const char *kOperandFormat =
    "private-dense-w8a8-bf16cache-row-symmetric-i8-f32-scale-rne-v1";
inline constexpr const char *kExecutionSemantics =
    "private-dense-w8a8-r2048-qkv-z-qsa-q-m128n64-sg4-integer-whole-k-v1";

[[nodiscard]] constexpr uint64_t roundedBytes(uint64_t bytes) noexcept {
  return (bytes + kAllocationAlignment - 1) & ~(kAllocationAlignment - 1);
}

struct Geometry final {
  uint32_t k = 0, n = 0;
  [[nodiscard]] explicit constexpr operator bool() const noexcept {
    return k != 0;
  }
};

// Canonical, unpadded decimal layer names only. GDN output is deliberately
// excluded following its failed component quality gate. HC/router/shared/
// vocabulary/PLE/MTP and every QSA projection other than Q are excluded.
[[nodiscard]] constexpr Geometry geometry(std::string_view prefix) noexcept {
  constexpr std::string_view leading = "language_model.model.layers.";
  if (!prefix.starts_with(leading)) return {};
  const auto tail = prefix.substr(leading.size());
  const auto split = tail.find('.');
  if (split == std::string_view::npos || !split || split > 2 ||
      (split > 1 && tail.front() == '0')) return {};
  uint32_t layer = 0;
  for (char digit : tail.substr(0, split)) {
    if (digit < '0' || digit > '9') return {};
    layer = layer * 10 + uint32_t(digit - '0');
  }
  if (layer >= 48) return {};
  const auto role = tail.substr(split + 1);
  if (layer % 4 != 3) {
    if (role == "linear_attn.in_proj_qkv") return {2560, 10240};
    if (role == "linear_attn.in_proj_z") return {2560, 6144};
  } else if (role == "self_attn.q_proj") return {2560, 12288};
  return {};
}

[[nodiscard]] constexpr bool requiresCache(uint32_t maximumRows) noexcept {
  // Independent of kernel selection: both members of a paired comparison
  // allocate and fit exactly the same immutable derivative cache.
  return maximumRows >= kRows;
}

// Pure metadata guard, exposed so the exact source contract can be exercised
// on the CPU without constructing Metal buffers or reading coefficient bytes.
[[nodiscard]] constexpr bool sourceMetadataMatches(Geometry expected,
    FlashDType dtype, std::span<const uint64_t> shape, uint64_t logicalBytes,
    uint64_t bufferBytes, metal::BufferStorage storage) noexcept {
  const uint64_t bytes = uint64_t{expected.n} * expected.k * 2;
  return bool(expected) && expected.n && dtype == FlashDType::BF16 &&
      shape.size() == 2 && shape[0] == expected.n && shape[1] == expected.k &&
      logicalBytes == bytes && bufferBytes == bytes &&
      storage == metal::BufferStorage::Shared;
}

[[nodiscard]] constexpr FlashPrefillDenseTilePlan projectionPlan(
    std::string_view prefix, uint32_t rows, uint32_t n, uint32_t k,
    bool verify, uint32_t groups = 4) noexcept {
  const auto expected = geometry(prefix);
  if (verify || rows != kRows || groups != 4 || !expected ||
      expected.k != k || expected.n != n) return {};
  // Retain the frozen measured shape traversal for QKV/Z/QSA-Q. All three
  // component screens selected SG4 independently of baseline selection.
  const auto measured = flashPrefillDenseTilePolicy(prefix, rows, n, k);
  if (!measured || measured.tileRows != 128 || measured.tileOutputs != 64)
    return {};
  return {128, 64, 4, measured.traversal};
}

[[nodiscard]] inline std::vector<std::string> selectedPrefixes() {
  std::vector<std::string> result;
  result.reserve(kProjectionCount);
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const std::string leading = "language_model.model.layers." + std::to_string(layer);
    if (layer % 4 == 3) result.push_back(leading + ".self_attn.q_proj");
    else {
      result.push_back(leading + ".linear_attn.in_proj_qkv");
      result.push_back(leading + ".linear_attn.in_proj_z");
    }
  }
  return result;
}

struct Projection final {
  metal::MetalBuffer codes;       // Exact I8[N,K] view; immutable after fit.
  metal::MetalBuffer scales;      // Exact F32[N] view; immutable after fit.
  uint32_t k = 0, n = 0;
  std::string sourceSHA;          // Exact original cached BF16[N,K] bytes.
  std::string codesSHA;
  std::string scalesSHA;
};

class Cache final {
public:
  // 36 GDN QKV + 36 GDN Z + 12 QSA Q, each with separately rounded backing
  // for I8 codes and F32 scales. This never touches a model or cache object.
  [[nodiscard]] static constexpr uint64_t plannedBytes() noexcept {
    return 36 * (roundedBytes(uint64_t{10240} * 2560) +
                 roundedBytes(uint64_t{10240} * 4)) +
           36 * (roundedBytes(uint64_t{6144} * 2560) +
                 roundedBytes(uint64_t{6144} * 4)) +
           12 * (roundedBytes(uint64_t{12288} * 2560) +
                 roundedBytes(uint64_t{12288} * 4));
  }

  Cache(metal::MetalBackend &backend, const FlashWeights &weights,
        const FlashDenseCache &denseCache);
  ~Cache();
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;
  Cache(Cache &&) noexcept;
  Cache &operator=(Cache &&) noexcept;

  [[nodiscard]] bool contains(std::string_view prefix) const noexcept;
  [[nodiscard]] const Projection &projection(std::string_view prefix) const;
  // Complete coefficient base census for cachedOperandsOnly residency;
  // excludes original BF16 aliases and every activation/workspace buffer.
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Workspace final {
public:
  [[nodiscard]] static constexpr std::array<uint64_t, 3> logicalBufferBytes() noexcept {
    return {uint64_t{kRows} * kWorkspaceInputCapacity, uint64_t{kRows} * 4, 4};
  }
  [[nodiscard]] static constexpr uint64_t plannedBytes() noexcept {
    const auto sizes = logicalBufferBytes();
    return roundedBytes(sizes[0]) + roundedBytes(sizes[1]) + roundedBytes(sizes[2]);
  }
  explicit Workspace(metal::MetalBackend &backend);
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  Workspace(Workspace &&) noexcept = default;
  Workspace &operator=(Workspace &&) noexcept = default;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept { return allocatedBytes_; }

  metal::MetalBuffer codes;       // Exact capacity I8[2048,6144].
  metal::MetalBuffer inputScales; // Exact F32[2048].
  metal::MetalBuffer dummyDot;    // Exact I32[1]; normal kernels never write it.

private:
  std::array<metal::MetalBuffer, 3> bases_;
  uint64_t allocatedBytes_ = 0;
};

// Unsupported rows, verification, roles and geometries return false before
// modifying the graph. Eligible input/output/diagnostics are Shared, unaliased
// BF16[R,K]/BF16[R,N]/U32[1+] views. Conversion and integer MPP are both queued.
[[nodiscard]] bool addProjection(metal::CommandGraph &graph, const Cache &cache,
    std::string_view prefix, metal::MetalBuffer input, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, uint32_t rows, bool verify,
    const Workspace &workspace, uint32_t groups = 4);

} // namespace splash::flash::dense_w8a8_sep21
