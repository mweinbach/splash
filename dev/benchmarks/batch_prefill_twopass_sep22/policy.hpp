#pragma once
#include "dev/benchmarks/prefill_qsa_twopass_sep21/twopass.hpp"
#include <array>
#include <atomic>
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash::batch_prefill_twopass_sep22 {
inline constexpr const char *flag = "SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22";
inline constexpr const char *schema = "batch-real2or4-allfresh2048-existing-packedV-twopass-v1";
inline constexpr const char *policy = "whole real2/4 cohort fresh2048 only; existing singleton QSA packedV floating tree on identical batch projections; original B1/other batch/head arithmetic unchanged; dedicated509607936 arena, legacy234356736 fallback retained";
inline constexpr const char *sourceSHA = "BATCH_SOURCE_SHA_PLACEHOLDER";
inline constexpr const char *shaderSHA = "6bb23ded512839feda31e04a3812802d6aab5eef6d4b8e33474b794a8247a131";
inline constexpr const char *hostSHA = "1f9f56f5ac77bedaa054fb51d0d3ed8b5d6e12d5eb1d5e8688a5b0fd9fde735e";
inline constexpr uint64_t plannedBytes = 509607936;
inline bool parse(const char *raw) {
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument(std::string(flag) + " must be 0 or 1");
}
inline bool requested() {
  static const bool selected = [] {
    if (!parse(std::getenv(flag))) return false;
    for (const char *dependency : {"SPLASH_FLASH_BATCH_PREFILL", "SPLASH_FLASH_BATCH_QSA_BULK_PREFILL",
        "SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21", "SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP",
        "SPLASH_FLASH_QSA_ROW_TILES", "SPLASH_FLASH_QSA_BULK_PREFILL", "SPLASH_FLASH_QSA_BULK_PREFILL_SG8"}) {
      const char *value = std::getenv(dependency);
      if (!value || std::string_view(value) != "1")
        throw std::invalid_argument(std::string(flag) + "=1 requires " + dependency + "=1");
    }
    return true;
  }();
  return selected;
}
constexpr bool eligible(uint32_t lanes, uint32_t rows, bool allFresh, bool selected) noexcept {
  return selected && (lanes == 2 || lanes == 4) && rows == 2048 && allFresh;
}
constexpr uint64_t extraBytes(uint32_t capacity, uint32_t maximumRows, bool selected) noexcept {
  return selected && capacity >= 2048 && maximumRows == 2048 ? plannedBytes : 0;
}
struct Counters {
  std::atomic<uint64_t> arenas{0}, arenaBytes{0}, forwards{0}, layers{0}, lanes{0};
};
inline Counters &counters() { static Counters value; return value; }
inline std::string numericalIdentity(std::string_view parent) {
  if (!requested()) return std::string(parent);
  const std::string material = std::string(parent)+"\n"+schema+"\n"+policy+"\n"+sourceSHA+"\n"+shaderSHA+"\n"+hostSHA;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  if (!CC_SHA256(material.data(),CC_LONG(material.size()),digest)) throw std::runtime_error("batch QSA numerical identity failed");
  constexpr char hex[]="0123456789abcdef"; std::string out; out.reserve(64);
  for (auto byte:digest) {out.push_back(hex[byte>>4]);out.push_back(hex[byte&15]);}
  return out;
}
struct Workspace {
  metal::MetalBuffer arena;
  prefill4k::TwoPassWorkspace qualified;
  uint64_t allocatedBytes = 0;
  Workspace(metal::MetalBackend &backend) {
    const uint64_t before = backend.memoryStats().allocatedBytes;
    arena = backend.allocateBuffer(plannedBytes, metal::BufferStorage::Shared,
        "private batch dedicated packedV two-pass QSA arena");
    const prefill4k::DenseCoalescedWorkspace prepared{2048,
        backend.view(arena, 0, 25165824), backend.view(arena, 25165824, 2097152),
        backend.view(arena, 27262976, 4194304)};
    qualified = {prepared, backend.view(arena, 31457280, 25165824),
        backend.view(arena, 56623104, 402653184), backend.view(arena, 459276288, 50331648)};
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before || after - before != plannedBytes || arena.sizeBytes() != plannedBytes)
      throw std::logic_error("batch two-pass dedicated arena actual charge differs from admitted plan");
    allocatedBytes = after - before;
    counters().arenas.fetch_add(1); counters().arenaBytes.fetch_add(allocatedBytes);
  }
  std::array<metal::MetalBuffer, 6> planes() const {
    return {qualified.prepared.queries, qualified.prepared.indexQueries, qualified.prepared.selectedBlocks,
        qualified.packedQueries, qualified.scoresAndProbabilities, qualified.rawAttention};
  }
};
inline void requireDisjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto x = reinterpret_cast<uintptr_t>(a.contents()), y = reinterpret_cast<uintptr_t>(b.contents());
  if (!x || !y || (x <= y ? y - x < a.sizeBytes() : x - y < b.sizeBytes()))
    throw std::invalid_argument("batch two-pass arena overlaps another batch or request plane");
}
inline void validateWholeCohortArena(const Workspace &w, const std::vector<metal::MetalBuffer> &others) {
  const auto planes = w.planes();
  for (size_t i = 0; i < planes.size(); ++i) {
    for (size_t j = 0; j < i; ++j) requireDisjoint(planes[i], planes[j]);
    // Shared arenas cannot alias an allocation whose storage is Private.
    // Never request a CPU address from an original Private scratch owner.
    for (const auto &other : others)
      if (other && other.storage() == metal::BufferStorage::Shared) requireDisjoint(planes[i], other);
  }
}
} // namespace splash::flash::batch_prefill_twopass_sep22
