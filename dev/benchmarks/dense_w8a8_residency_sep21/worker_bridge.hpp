#pragma once
#include "dev/benchmarks/dense_w8a8_sep21/worker_bridge.hpp"
#include <limits>

namespace splash::flash::dense_w8a8_residency_sep21 {
inline constexpr const char *kFlag = "SPLASH_FLASH_DENSE_W8A8_RESIDENCY_PRUNE_SEP21";
inline constexpr const char *kPolicy =
    "legacy-direct-command-transient-bf16-preserve-all-backing-w8-active-source84-"
    "persistent-omit-w8-off-derived168-unwired-v1";
inline constexpr uint32_t kBF16SourceCount = 84;
inline constexpr uint64_t kBF16SourceBytes = 3774873600ULL;
[[nodiscard]] inline bool parse(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(kFlag) + " must be 0 or 1");
}
[[nodiscard]] inline bool requested() {
  static const bool selected = parse(std::getenv(kFlag));
  return selected;
}
struct Plan final {
  uint32_t omittedBF16Owners = 0, omittedI8Owners = 0, retainedI8Owners = 0;
  uint64_t omittedBF16Bytes = 0, omittedI8Bytes = 0, retainedI8Bytes = 0;
};
[[nodiscard]] constexpr Plan plan(bool hasCache,bool w8,bool prune) noexcept {
  if (!hasCache) return {};
  const bool omitSource = prune && w8, omitDerived = prune && !w8;
  return {omitSource ? kBF16SourceCount : 0,omitDerived ? dense_w8a8_sep21::kImmutableBufferCount : 0,
      omitDerived ? 0 : dense_w8a8_sep21::kImmutableBufferCount,omitSource ? kBF16SourceBytes : 0,
      omitDerived ? dense_w8a8_sep21::Cache::plannedBytes() : 0,omitDerived ? 0 : dense_w8a8_sep21::Cache::plannedBytes()};
}
[[nodiscard]] constexpr uint64_t expectedHybridOwners(bool hasCache,bool w8,bool prune) noexcept {
  const auto p = plan(hasCache,w8,prune);
  return 748 + p.retainedI8Owners - p.omittedBF16Owners;
}
[[nodiscard]] constexpr uint64_t expectedHybridBytes(bool hasCache,bool w8,bool prune) noexcept {
  const auto p = plan(hasCache,w8,prune);
  return 202252746752ULL + p.retainedI8Bytes - p.omittedBF16Bytes;
}
[[nodiscard]] inline bool hasCacheFromRoutes(std::string_view routes) {
  return dense_w8a8_sep21::cacheIdentityFromRoutes(routes) != "none-maxrows-below2048";
}

// Persistent set selection only. The original cache's tensor and saved-owner
// collections remain intact. FlashOperandStore::mapTensor creates one whole
// wrapSharedMemory allocation per selected saved operand, with no subview;
// all selected N*K*2 extents are already16K aligned. Require exact saved view
// membership, geometry and byte census rather than dropping a guessed range.
[[nodiscard]] inline std::vector<metal::MetalBuffer> bf16PersistentOperands(
    const FlashDenseCache &cache,bool hasW8Cache,bool w8,bool prune) {
  const auto saved = cache.persistedWeightBuffers();
  if (!hasW8Cache || !w8 || !prune) return saved;
  std::vector<bool> omitted(saved.size()); uint32_t count = 0; uint64_t bytes = 0;
  for (const auto &prefix : dense_w8a8_sep21::selectedPrefixes()) {
    const auto expected = dense_w8a8_sep21::geometry(prefix);
    const auto &source = cache.tensor(prefix);
    if (!source.buffer || !dense_w8a8_sep21::sourceMetadataMatches(expected,source.dtype,
        source.shape,source.logicalBytes,source.buffer.sizeBytes(),source.buffer.storage()))
      throw std::logic_error("persistent dense W8A8 prune source metadata differs");
    size_t match = saved.size(); uint32_t matches = 0;
    for (size_t i = 0; i < saved.size(); ++i) if (saved[i].sameView(source.buffer)) { match = i; ++matches; }
    if (matches != 1 || omitted[match])
      throw std::logic_error("persistent dense W8A8 prune source is not a unique complete saved owner");
    if (source.buffer.sizeBytes()%dense_w8a8_sep21::kAllocationAlignment ||
        bytes > std::numeric_limits<uint64_t>::max()-source.buffer.sizeBytes())
      throw std::logic_error("persistent dense W8A8 prune source owner byte extent differs");
    omitted[match] = true; ++count; bytes += source.buffer.sizeBytes();
  }
  if (count != kBF16SourceCount || bytes != kBF16SourceBytes)
    throw std::logic_error("persistent dense W8A8 prune source84 census differs");
  std::vector<metal::MetalBuffer> kept; kept.reserve(saved.size()-count);
  for (size_t i = 0; i < saved.size(); ++i) if (!omitted[i]) kept.push_back(saved[i]);
  return kept;
}
[[nodiscard]] constexpr bool includeDerived(bool hasCache,bool w8,bool prune) noexcept {
  return hasCache && !(prune && !w8);
}
} // namespace splash::flash::dense_w8a8_residency_sep21
