#pragma once

// Private bounded oracle helper. Metadata inspection is CPU-only; load() maps
// and validates exactly one Full512 payload only when the GPU oracle calls it.
// The caller must reserve oneLayerPlannedBytes() before calling load().
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "metal/MetalBackend.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash::qmv_one_layer {

inline constexpr uint64_t kAlignment = 16384;
inline constexpr uint64_t kFull512LayerBytes = 2524446720ULL;

namespace detail {
[[noreturn]] inline void fail(std::string_view reason) {
  throw std::invalid_argument("private gathered I8 one-layer oracle: " + std::string(reason));
}

inline void requireDigest(std::string_view value) {
  if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    fail("invalid lowercase SHA256 metadata");
}

inline void validateEntry(const FlashInt8ExpertStoreLayer &entry) {
  if (entry.bytes != kFull512LayerBytes || entry.bytes % kAlignment || entry.selectedIDs.size() != 512)
    fail("requires one canonical aligned Full512 layer");
  for (uint32_t id = 0; id < 512; ++id)
    if (entry.selectedIDs[id] != id) fail("Full512 ID-to-rank inventory is not canonical");
  requireDigest(entry.sha256);
  uint64_t cursor = 0;
  for (uint32_t projection = 0; projection < 3; ++projection) {
    const uint64_t n = projection == 2 ? 2560 : 640;
    const uint64_t k = projection == 2 ? 640 : 2560;
    for (uint32_t plane = 0; plane < 2; ++plane) {
      const auto &range = plane ? entry.scales[projection] : entry.codes[projection];
      cursor = (cursor + kAlignment - 1) & ~(kAlignment - 1);
      const uint64_t length = uint64_t{512} * n * (plane ? sizeof(float) : k);
      if (range.offset != cursor || range.length != length || range.offset > entry.bytes ||
          range.length > entry.bytes - range.offset)
        fail("Full512 plane geometry/packing differs");
      requireDigest(range.sha256);
      cursor += range.length;
    }
  }
  if (((cursor + kAlignment - 1) & ~(kAlignment - 1)) != entry.bytes)
    fail("Full512 layer extent differs");
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
inline std::string hash(const void *pointer, uint64_t count) {
  if (!pointer && count) fail("hash requires an addressable operand");
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) fail("SHA256 initialization failed");
  const auto *bytes = static_cast<const uint8_t *>(pointer);
  while (count) {
    const auto step = static_cast<CC_LONG>(std::min<uint64_t>(count, 1ULL << 30));
    if (!CC_SHA256_Update(&context, bytes, step)) fail("SHA256 update failed");
    bytes += step;
    count -= step;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256_Final(digest.data(), &context)) fail("SHA256 finalization failed");
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (const auto byte : digest) {
    result += alphabet[byte >> 4];
    result += alphabet[byte & 15];
  }
  return result;
}
#pragma clang diagnostic pop

class ReadonlyMapping final {
public:
  ReadonlyMapping(const std::filesystem::path &path, uint64_t expected) : bytes_(expected) {
    if (!expected || expected % kAlignment || expected > std::numeric_limits<size_t>::max())
      fail("payload size/alignment differs");
    file_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file_ < 0) fail("cannot open readonly one-layer payload");
    if (::fstat(file_, &before_) || !S_ISREG(before_.st_mode) || (before_.st_mode & 0222) ||
        before_.st_size <= 0 || uint64_t(before_.st_size) != expected) {
      ::close(file_);
      file_ = -1;
      fail("payload is not a readonly regular file of its exact size");
    }
    address_ = ::mmap(nullptr, expected, PROT_READ, MAP_SHARED, file_, 0);
    if (address_ == MAP_FAILED) {
      const int saved = errno;
      address_ = nullptr;
      ::close(file_);
      file_ = -1;
      errno = saved;
      fail("readonly one-layer mapping failed");
    }
  }
  ~ReadonlyMapping() {
    if (address_) ::munmap(address_, bytes_);
    if (file_ >= 0) ::close(file_);
  }
  ReadonlyMapping(const ReadonlyMapping &) = delete;
  ReadonlyMapping &operator=(const ReadonlyMapping &) = delete;
  void *address() const noexcept { return address_; }
  void requireUnchanged() const {
    struct stat after{};
    if (::fstat(file_, &after) || !S_ISREG(after.st_mode) || (after.st_mode & 0222) ||
        after.st_dev != before_.st_dev || after.st_ino != before_.st_ino ||
        after.st_size != before_.st_size || after.st_mtimespec.tv_sec != before_.st_mtimespec.tv_sec ||
        after.st_mtimespec.tv_nsec != before_.st_mtimespec.tv_nsec ||
        after.st_ctimespec.tv_sec != before_.st_ctimespec.tv_sec ||
        after.st_ctimespec.tv_nsec != before_.st_ctimespec.tv_nsec)
      fail("readonly payload changed during validation");
  }
private:
  int file_ = -1;
  void *address_ = nullptr;
  uint64_t bytes_ = 0;
  struct stat before_{};
};
} // namespace detail

// This plans one base mapping and one rank map, never the 48-layer store.
[[nodiscard]] inline uint64_t oneLayerPlannedBytes(const FlashInt8ExpertStoreLayer &entry) {
  detail::validateEntry(entry);
  return entry.bytes + kAlignment;
}

struct OneLayerPayload final {
  metal::MetalBuffer base;
  metal::MetalBuffer ranks;
  // The same views are bound to the control and every QMV candidate.
  std::array<metal::MetalBuffer, 3> codes;
  std::array<metal::MetalBuffer, 3> scales;
  uint32_t layer = 0;
  uint64_t plannedBytes = 0;
  uint64_t allocatedBytes = 0;

  [[nodiscard]] static OneLayerPayload load(metal::MetalBackend &backend,
      const FlashInt8ExpertStoreMetadata &metadata, uint32_t index) {
    if (index >= metadata.layers.size()) detail::fail("target layer is outside [0,47]");
    const auto &entry = metadata.layers[index];
    OneLayerPayload result;
    result.layer = index;
    result.plannedBytes = oneLayerPlannedBytes(entry);
    if (entry.bytes > backend.capabilities().maxBufferLengthBytes)
      detail::fail("one-layer payload exceeds Metal buffer limit");

    // No FlashWeights or FlashInt8ExpertStore construction occurs here.
    // The caller already validated the source/certified manifest identities.
    auto mapping = std::make_shared<detail::ReadonlyMapping>(entry.path, entry.bytes);
    const auto *data = static_cast<const uint8_t *>(mapping->address());
    if (detail::hash(data, entry.bytes) != entry.sha256)
      detail::fail("one-layer whole-payload checksum differs");
    uint64_t cursor = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      for (const auto &range : {entry.codes[projection], entry.scales[projection]}) {
        if (!std::all_of(data + cursor, data + range.offset, [](uint8_t value) { return value == 0; }))
          detail::fail("one-layer plane alignment padding is nonzero");
        if (detail::hash(data + range.offset, range.length) != range.sha256)
          detail::fail("one-layer plane checksum differs");
        cursor = range.offset + range.length;
      }
      const auto &code = entry.codes[projection];
      if (std::find(data + code.offset, data + code.offset + code.length, uint8_t{128}) !=
          data + code.offset + code.length)
        detail::fail("symmetric I8 contains excluded -128 code");
      const auto &scale = entry.scales[projection];
      const auto *values = reinterpret_cast<const float *>(data + scale.offset);
      if (!std::all_of(values, values + scale.length / sizeof(float),
          [](float value) { return std::isfinite(value) && value > 0.0f; }))
        detail::fail("one-layer row scales must be finite and positive");
    }
    if (!std::all_of(data + cursor, data + entry.bytes, [](uint8_t value) { return value == 0; }))
      detail::fail("one-layer tail padding is nonzero");
    mapping->requireUnchanged();

    // Validation precedes all GPU-visible wrapping. Views keep the same base
    // allocation, whose lifetime token retains the readonly mapping in graphs.
    const uint64_t before = backend.memoryStats().allocatedBytes;
    result.base = backend.wrapSharedMemory(mapping->address(), entry.bytes, mapping,
        "private bounded one-layer Full512 I8 payload");
    result.ranks = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared,
        "private bounded one-layer Full512 ID-to-rank map");
    std::memset(result.ranks.contents(), 0xff, kAlignment);
    auto *rank = static_cast<uint32_t *>(result.ranks.contents());
    for (uint32_t id = 0; id < 512; ++id) rank[entry.selectedIDs[id]] = id;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      result.codes[projection] = backend.view(result.base,
          entry.codes[projection].offset, entry.codes[projection].length);
      result.scales[projection] = backend.view(result.base,
          entry.scales[projection].offset, entry.scales[projection].length);
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before || after - before > result.plannedBytes)
      detail::fail("one-layer backend allocation ledger exceeds the reserved plan");
    result.allocatedBytes = after - before;
    return result;
  }

  using ImmutableHashes = std::array<std::string, 2>;
  // Optional oracle-only scans; these are never called by a CPU self-test.
  [[nodiscard]] ImmutableHashes immutableHashes() const {
    if (!base || !ranks || base.storage() != metal::BufferStorage::Shared ||
        ranks.storage() != metal::BufferStorage::Shared || !base.contents() || !ranks.contents())
      detail::fail("immutable hash witness requires the loaded Shared operands");
    return {detail::hash(base.contents(), base.sizeBytes()),
        detail::hash(ranks.contents(), ranks.sizeBytes())};
  }
  void checkImmutableHashes(const ImmutableHashes &expected) const {
    if (immutableHashes() != expected) detail::fail("immutable one-layer operands changed");
  }
};

} // namespace splash::flash::qmv_one_layer
