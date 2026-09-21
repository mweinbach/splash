#pragma once

// Local derivative artifacts only. No Metal, Foundation, model mutation, or
// payload logging is involved in this checked CPU/file-format helper.
#include "ExperimentalINT8.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model::preconverted_int8 {

inline constexpr uint32_t kSchemaVersion = 1;
inline constexpr uint32_t kConverterVersion = 2;
inline constexpr uint64_t kAlignment = 16 * 1024;
inline constexpr size_t kMetadataBytes = 512;
inline constexpr std::string_view kConverterAlgorithm =
    "splash.affine-q4-to-whole-row-int8-w8a8-profile-v2";

class ArtifactError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct Identity {
  uint32_t n = 0;
  uint32_t k = 0;
  uint32_t policy = 0;
  uint32_t role = 0;
  uint32_t kind = 0;
  std::string sourceSha256;
};

struct ArtifactPaths {
  std::filesystem::path directory;
  std::filesystem::path metadata;
  std::filesystem::path payload;
};

struct Metadata {
  Identity identity;
  uint64_t dataOffset = 0;
  uint64_t scaleOffset = 0;
  uint64_t dataBytes = 0;
  uint64_t scaleBytes = 0;
  uint64_t payloadBytes = 0;
  std::string dataSha256;
  std::string scaleSha256;
  std::string projectionFingerprint;
  std::filesystem::path payloadPath;
};

// metadata.bin: 512 canonical little-endian bytes, never a native struct.
// 0 magic[8], 8 schema u32,12 headerbytes,16 converterversion,20 N,24 K,
// 28 policy,32 role,36 kind; 40 dataOffset u64,48 dataBytes,56 scaleOffset,
// 64 scaleBytes,72 payloadBytes; 80 sourceSHA[32],112 dataSHA,144 scaleSHA,
// 176 projectionFingerprint[32],208 algorithm[128] NUL/zero padded;
// 336 headerSHA[32] (hash all512 with this slot zero),368 reservedzero[144].
// payload.bin: codes at0, FP32-LE scales atalign16KiB(dataBytes), zero padding
// throughalign16KiB(scaleOffset+scaleBytes). Publishing renames BOTH files'
// containing directory, so readers never observe a metadata/payload half pair.

[[nodiscard]] inline std::string digest(const void *bytes, uint64_t count);
// Stream the exact original packed Q4, BF16 scale, BF16 bias bytes in order.
[[nodiscard]] inline std::string sourceDigest(std::span<const uint8_t> packed,
                                            std::span<const uint8_t> scales,
                                            std::span<const uint8_t> biases);
[[nodiscard]] inline std::string cacheKey(const Identity &identity);
[[nodiscard]] inline ArtifactPaths paths(const std::filesystem::path &base,
                                         const Identity &identity);
// Byte-identical to the existing runtime v2 precision-profile fingerprint.
[[nodiscard]] inline std::string projectionFingerprint(
    const Identity &identity, std::string_view dataSha256,
    std::string_view scaleSha256);
// Strict: missing, truncated, corrupt, unknown schema/converter, identity
// mismatch or noncanonical layout all throw; never reconstructs weights.
[[nodiscard]] inline Metadata loadMetadata(const std::filesystem::path &base,
                                           const Identity &expected);
// Check mmap bytes before Metal wrapping: hashes, code range, positive finite
// FP32 row scales, exact byte count and zero canonical padding.
inline void validatePayload(const Metadata &metadata,
                            std::span<const uint8_t> payload);
// Publish both files together with an atomic sibling-directory rename. An
// existing key is strictly validated and reused, never overwritten. Data/SF
// retain the current logical bytes and separately page-aligned footprint.
[[nodiscard]] inline Metadata writeAtomically(
    const std::filesystem::path &base, const Identity &identity,
    std::span<const int8_t> data, std::span<const float> scales,
    std::string_view expectedProjectionFingerprint = {});

namespace detail {

inline constexpr size_t kHeaderDigestOffset = 336;
inline constexpr std::array<uint8_t, 8> kMagic{'S','P','L','I','8','A','0','1'};
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little,
              "INT8 derivative payloads require little-endian FP32 storage");

[[nodiscard]] inline std::string hex(std::span<const uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string value;
  value.reserve(bytes.size() * 2);
  for (uint8_t byte : bytes) {
    value += digits[byte >> 4];
    value += digits[byte & 15];
  }
  return value;
}

inline void checkDigest(std::string_view value) {
  if (value.size() != 64 ||
      value.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    throw ArtifactError("artifact requires a lowercase SHA256 identity");
}

[[nodiscard]] inline std::array<uint8_t, 32> unhex(std::string_view value) {
  checkDigest(value);
  const auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  std::array<uint8_t, 32> result{};
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<uint8_t>((digit(value[i * 2]) << 4) |
                                   digit(value[i * 2 + 1]));
  return result;
}

inline void appendNumber(std::string &out, uint64_t value) {
  std::array<char, 32> bytes{};
  const auto result = std::to_chars(bytes.data(), bytes.data() + bytes.size(), value);
  if (result.ec != std::errc{}) throw ArtifactError("artifact integer encoding failed");
  out.append(bytes.data(), result.ptr);
}

inline void numberLine(std::string &out, std::string_view name, uint64_t value) {
  out.append(name);
  appendNumber(out, value);
  out += '\n';
}

[[nodiscard]] inline uint64_t align(uint64_t value) {
  if (value > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw ArtifactError("artifact byte alignment overflows");
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

[[nodiscard]] inline Metadata layout(const Identity &identity) {
  checkDigest(identity.sourceSha256);
  if (identity.policy > 3 || identity.role > 3 || identity.kind > 1 ||
      identity.n > uint32_t{std::numeric_limits<int32_t>::max()})
    throw ArtifactError("artifact precision profile is invalid");
  try {
    static_cast<void>(experimental_int8::convertedProjectionBytes(identity.n, identity.k));
  } catch (const std::exception &) {
    throw ArtifactError("artifact projection dimensions are invalid");
  }
  Metadata meta;
  meta.identity = identity;
  meta.dataBytes = uint64_t{identity.n} * identity.k;
  meta.scaleBytes = uint64_t{identity.n} * sizeof(float);
  // Match the current converter's per-plane allocation/hash limit exactly.
  if (meta.dataBytes > std::numeric_limits<CC_LONG>::max() ||
      meta.scaleBytes > std::numeric_limits<CC_LONG>::max())
    throw ArtifactError("artifact projection exceeds converter plane limit");
  meta.scaleOffset = align(meta.dataBytes);
  if (meta.scaleBytes > std::numeric_limits<uint64_t>::max() - meta.scaleOffset)
    throw ArtifactError("artifact payload byte count overflows");
  meta.payloadBytes = align(meta.scaleOffset + meta.scaleBytes);
  if (meta.payloadBytes > std::numeric_limits<size_t>::max() ||
      meta.payloadBytes > uint64_t{std::numeric_limits<off_t>::max()})
    throw ArtifactError("artifact payload exceeds host file range");
  return meta;
}

inline bool sameIdentity(const Identity &left, const Identity &right) {
  return left.n == right.n && left.k == right.k && left.policy == right.policy &&
         left.role == right.role && left.kind == right.kind &&
         left.sourceSha256 == right.sourceSha256;
}

inline void checkLayout(const Metadata &meta) {
  const auto expected = layout(meta.identity);
  if (meta.dataOffset != 0 || meta.dataBytes != expected.dataBytes ||
      meta.scaleOffset != expected.scaleOffset || meta.scaleBytes != expected.scaleBytes ||
      meta.payloadBytes != expected.payloadBytes)
    throw ArtifactError("artifact has a noncanonical payload layout");
  checkDigest(meta.dataSha256);
  checkDigest(meta.scaleSha256);
  checkDigest(meta.projectionFingerprint);
  if (meta.projectionFingerprint != projectionFingerprint(
          meta.identity, meta.dataSha256, meta.scaleSha256))
    throw ArtifactError("artifact projection fingerprint mismatch");
}

class File final {
public:
  explicit File(int descriptor = -1) noexcept : descriptor_(descriptor) {}
  ~File() { if (descriptor_ >= 0) ::close(descriptor_); }
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  [[nodiscard]] int get() const noexcept { return descriptor_; }
private:
  int descriptor_;
};

[[noreturn]] inline void fileError(std::string_view action) {
  const int error = errno;
  throw ArtifactError(std::string(action) + ": " +
                      std::error_code(error, std::generic_category()).message());
}

[[nodiscard]] inline uint64_t regularSize(int descriptor) {
  struct stat info{};
  if (::fstat(descriptor, &info)) fileError("artifact file stat failed");
  if (!S_ISREG(info.st_mode) || info.st_size < 0)
    throw ArtifactError("artifact entry is not a regular file");
  return static_cast<uint64_t>(info.st_size);
}

inline void readExactly(int descriptor, std::span<uint8_t> out) {
  while (!out.empty()) {
    const ssize_t count = ::read(descriptor, out.data(), out.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fileError("artifact read failed");
    if (!count) throw ArtifactError("artifact file was truncated during read");
    out = out.subspan(static_cast<size_t>(count));
  }
}

inline void writeExactly(int descriptor, std::span<const uint8_t> bytes) {
  while (!bytes.empty()) {
    const size_t chunk = std::min<size_t>(bytes.size(), 1024 * 1024);
    const ssize_t count = ::write(descriptor, bytes.data(), chunk);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fileError("artifact write failed");
    if (!count) throw ArtifactError("artifact write made no progress");
    bytes = bytes.subspan(static_cast<size_t>(count));
  }
}

inline void zeros(int descriptor, uint64_t count) {
  const std::array<uint8_t, kAlignment> zero{};
  while (count) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(count, zero.size()));
    writeExactly(descriptor, {zero.data(), chunk});
    count -= chunk;
  }
}

inline void put(std::span<uint8_t> bytes, size_t offset, uint64_t value, size_t width) {
  for (size_t i = 0; i < width; ++i)
    bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}

[[nodiscard]] inline uint64_t get(std::span<const uint8_t> bytes,
                                size_t offset, size_t width) {
  uint64_t value = 0;
  for (size_t i = 0; i < width; ++i) value |= uint64_t{bytes[offset + i]} << (i * 8);
  return value;
}

[[nodiscard]] inline std::array<uint8_t, kMetadataBytes> encode(const Metadata &meta) {
  checkLayout(meta);
  std::array<uint8_t, kMetadataBytes> bytes{};
  std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
  const std::array<uint32_t, 8> fields{kSchemaVersion, kMetadataBytes, kConverterVersion,
      meta.identity.n, meta.identity.k, meta.identity.policy, meta.identity.role,
      meta.identity.kind};
  for (size_t i = 0; i < fields.size(); ++i) put(bytes, 8 + i * 4, fields[i], 4);
  const std::array<uint64_t, 5> sizes{meta.dataOffset, meta.dataBytes, meta.scaleOffset,
                                   meta.scaleBytes, meta.payloadBytes};
  for (size_t i = 0; i < sizes.size(); ++i) put(bytes, 40 + i * 8, sizes[i], 8);
  const std::array<std::string_view, 4> hashes{meta.identity.sourceSha256,
      meta.dataSha256, meta.scaleSha256, meta.projectionFingerprint};
  for (size_t i = 0; i < hashes.size(); ++i) {
    const auto raw = unhex(hashes[i]);
    std::copy(raw.begin(), raw.end(), bytes.begin() + 80 + i * 32);
  }
  std::copy(kConverterAlgorithm.begin(), kConverterAlgorithm.end(), bytes.begin() + 208);
  const auto checksum = unhex(digest(bytes.data(), bytes.size()));
  std::copy(checksum.begin(), checksum.end(), bytes.begin() + kHeaderDigestOffset);
  return bytes;
}

[[nodiscard]] inline Metadata decode(std::array<uint8_t, kMetadataBytes> bytes,
                                    const Identity &expected) {
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()) ||
      get(bytes, 8, 4) != kSchemaVersion || get(bytes, 12, 4) != kMetadataBytes ||
      get(bytes, 16, 4) != kConverterVersion)
    throw ArtifactError("artifact magic, schema or converter version is unsupported");
  const auto algorithm = std::span<const uint8_t>(bytes).subspan(208, 128);
  if (!std::equal(kConverterAlgorithm.begin(), kConverterAlgorithm.end(), algorithm.begin()) ||
      !std::all_of(algorithm.begin() + kConverterAlgorithm.size(), algorithm.end(),
                   [](uint8_t byte) { return byte == 0; }) ||
      !std::all_of(bytes.begin() + 368, bytes.end(), [](uint8_t byte) { return byte == 0; }))
    throw ArtifactError("artifact converter algorithm or reserved metadata is invalid");
  const std::string checksum = hex(std::span<const uint8_t>(bytes).subspan(
      kHeaderDigestOffset, 32));
  std::fill_n(bytes.begin() + kHeaderDigestOffset, 32, 0);
  if (checksum != digest(bytes.data(), bytes.size()))
    throw ArtifactError("artifact metadata checksum mismatch");
  Metadata meta;
  meta.identity.n = static_cast<uint32_t>(get(bytes, 20, 4));
  meta.identity.k = static_cast<uint32_t>(get(bytes, 24, 4));
  meta.identity.policy = static_cast<uint32_t>(get(bytes, 28, 4));
  meta.identity.role = static_cast<uint32_t>(get(bytes, 32, 4));
  meta.identity.kind = static_cast<uint32_t>(get(bytes, 36, 4));
  const auto hashAt = [&](size_t offset) {
    return hex(std::span<const uint8_t>(bytes).subspan(offset, 32));
  };
  meta.identity.sourceSha256 = hashAt(80);
  if (!sameIdentity(meta.identity, expected))
    throw ArtifactError("artifact source or precision profile identity mismatch");
  meta.dataOffset = get(bytes, 40, 8);
  meta.dataBytes = get(bytes, 48, 8);
  meta.scaleOffset = get(bytes, 56, 8);
  meta.scaleBytes = get(bytes, 64, 8);
  meta.payloadBytes = get(bytes, 72, 8);
  meta.dataSha256 = hashAt(112);
  meta.scaleSha256 = hashAt(144);
  meta.projectionFingerprint = hashAt(176);
  checkLayout(meta);
  return meta;
}

inline void validateExisting(const Metadata &meta) {
  File file(::open(meta.payloadPath.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0) fileError("artifact payload open failed");
  if (regularSize(file.get()) != meta.payloadBytes)
    throw ArtifactError("artifact payload byte count mismatch");
  void *address = ::mmap(nullptr, static_cast<size_t>(meta.payloadBytes), PROT_READ,
                         MAP_PRIVATE, file.get(), 0);
  if (address == MAP_FAILED) fileError("artifact payload mmap failed");
  try {
    validatePayload(meta, {static_cast<const uint8_t *>(address),
                           static_cast<size_t>(meta.payloadBytes)});
  } catch (...) {
    ::munmap(address, static_cast<size_t>(meta.payloadBytes));
    throw;
  }
  ::munmap(address, static_cast<size_t>(meta.payloadBytes));
}

} // namespace detail

inline std::string digest(const void *bytes, uint64_t count) {
  if (count && !bytes) throw ArtifactError("artifact digest input is missing");
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) throw ArtifactError("artifact SHA256 initialization failed");
  const auto *cursor = static_cast<const uint8_t *>(bytes);
  while (count) {
    const auto chunk = static_cast<CC_LONG>(std::min<uint64_t>(count, 1024 * 1024 * 1024));
    if (!CC_SHA256_Update(&context, cursor, chunk))
      throw ArtifactError("artifact SHA256 update failed");
    cursor += chunk;
    count -= chunk;
  }
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  if (!CC_SHA256_Final(output.data(), &context)) throw ArtifactError("artifact SHA256 failed");
  return detail::hex(output);
}

inline std::string sourceDigest(std::span<const uint8_t> packed,
                               std::span<const uint8_t> scales,
                               std::span<const uint8_t> biases) {
  if (packed.empty() || scales.empty() || biases.empty() || scales.size() != biases.size() ||
      scales.size() > std::numeric_limits<size_t>::max() / 16 ||
      packed.size() != scales.size() * 16)
    throw ArtifactError("artifact source planes are incomplete affine Q4 storage");
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) throw ArtifactError("artifact SHA256 initialization failed");
  for (auto bytes : {packed, scales, biases}) {
    while (!bytes.empty()) {
      const auto count = static_cast<CC_LONG>(std::min<size_t>(bytes.size(), 1024 * 1024 * 1024));
      if (!CC_SHA256_Update(&context, bytes.data(), count))
        throw ArtifactError("artifact source SHA256 update failed");
      bytes = bytes.subspan(count);
    }
  }
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  if (!CC_SHA256_Final(output.data(), &context)) throw ArtifactError("artifact source SHA256 failed");
  return detail::hex(output);
}

inline std::string cacheKey(const Identity &identity) {
  static_cast<void>(detail::layout(identity));
  std::string key = "splash.preconverted-whole-row-int8-artifact-key-v1\nalgorithm=";
  key.append(kConverterAlgorithm);
  key += '\n';
  detail::numberLine(key, "converter=", kConverterVersion);
  detail::numberLine(key, "n=", identity.n);
  detail::numberLine(key, "k=", identity.k);
  detail::numberLine(key, "policy=", identity.policy);
  detail::numberLine(key, "role=", identity.role);
  detail::numberLine(key, "kind=", identity.kind);
  key += "source=" + identity.sourceSha256 + '\n';
  return digest(key.data(), key.size());
}

inline ArtifactPaths paths(const std::filesystem::path &base, const Identity &identity) {
  if (base.empty()) throw ArtifactError("artifact derivative directory is empty");
  const auto directory = base / cacheKey(identity);
  return {directory, directory / "metadata.bin", directory / "payload.bin"};
}

inline std::string projectionFingerprint(const Identity &identity,
                                         std::string_view dataSha256,
                                         std::string_view scaleSha256) {
  static_cast<void>(detail::layout(identity));
  detail::checkDigest(dataSha256);
  detail::checkDigest(scaleSha256);
  constexpr std::array<std::string_view, 4> policyNames{"all", "target", "hybrid", "q4"};
  std::string canonical(kConverterAlgorithm);
  canonical += "\npolicy=";
  canonical.append(policyNames[identity.policy]);
  canonical += '\n';
  detail::numberLine(canonical, "role=", identity.role);
  detail::numberLine(canonical, "kind=", identity.kind);
  detail::numberLine(canonical, "n=", identity.n);
  detail::numberLine(canonical, "k=", identity.k);
  canonical += "weight_scale=maxabs-whole-output-row-div127\n"
               "activation_scale=maxabs-whole-input-row-div127\n"
               "rounding=nearest-even\nclamp=-127,127\nzero_scale=1\n"
               "layout=output-major-n-k\nphases=prefill-decode-replay\n"
               "diagnostic_abi=sticky-nonfinite-shape-16-v1\n";
  canonical += "data=";
  canonical.append(dataSha256);
  canonical += "\nscales=";
  canonical.append(scaleSha256);
  canonical += '\n';
  return digest(canonical.data(), canonical.size());
}

inline Metadata loadMetadata(const std::filesystem::path &base, const Identity &expected) {
  const auto location = paths(base, expected);
  detail::File directory(::open(location.directory.c_str(), O_RDONLY | O_DIRECTORY |
                                                             O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0) detail::fileError("artifact key directory open failed");
  detail::File file(::openat(directory.get(), "metadata.bin", O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0) detail::fileError("artifact metadata open failed");
  if (detail::regularSize(file.get()) != kMetadataBytes)
    throw ArtifactError("artifact metadata byte count mismatch");
  std::array<uint8_t, kMetadataBytes> bytes{};
  detail::readExactly(file.get(), bytes);
  auto meta = detail::decode(bytes, expected);
  detail::File payload(::openat(directory.get(), "payload.bin", O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (payload.get() < 0) detail::fileError("artifact payload open failed");
  if (detail::regularSize(payload.get()) != meta.payloadBytes)
    throw ArtifactError("artifact payload byte count mismatch");
  meta.payloadPath = location.payload;
  return meta;
}

inline void validatePayload(const Metadata &metadata, std::span<const uint8_t> payload) {
  detail::checkLayout(metadata);
  if (payload.size() != metadata.payloadBytes)
    throw ArtifactError("artifact mapped payload byte count mismatch");
  const auto data = payload.subspan(static_cast<size_t>(metadata.dataOffset),
                                   static_cast<size_t>(metadata.dataBytes));
  const auto scales = payload.subspan(static_cast<size_t>(metadata.scaleOffset),
                                     static_cast<size_t>(metadata.scaleBytes));
  const auto validateDomain = [&] {
    if (std::memchr(data.data(), 128, data.size()))
      throw ArtifactError("artifact contains an out-of-domain INT8 weight code");
    for (size_t offset = 0; offset < scales.size(); offset += sizeof(float)) {
      float scale;
      std::memcpy(&scale, scales.data() + offset, sizeof(scale));
      if (!std::isfinite(scale) || scale <= 0.0f)
        throw ArtifactError("artifact row scale is not finite and positive");
    }
    for (const auto padding : {payload.subspan(static_cast<size_t>(metadata.dataBytes),
                                              static_cast<size_t>(metadata.scaleOffset - metadata.dataBytes)),
                              payload.subspan(static_cast<size_t>(metadata.scaleOffset + metadata.scaleBytes))})
      if (!std::all_of(padding.begin(), padding.end(), [](uint8_t byte) { return byte == 0; }))
        throw ArtifactError("artifact padding is not canonical zero storage");
  };
  std::string dataHash;
  std::string scaleHash;
  if (data.size() >= 8 * 1024 * 1024) {
    // One plane only: the caller's scan can fault pages in ahead of the full
    // sequential SHA pass. Neither hashing nor any semantic check is omitted.
    std::exception_ptr hashFailure;
    std::jthread hashWorker([&] {
      try {
        dataHash = digest(data.data(), data.size());
      } catch (...) {
        hashFailure = std::current_exception();
      }
    });
    // If a caller-side check throws, jthread joins before spans/mapping can
    // leave scope. Results and worker exceptions are read only after join.
    validateDomain();
    scaleHash = digest(scales.data(), scales.size());
    hashWorker.join();
    if (hashFailure) std::rethrow_exception(hashFailure);
  } else {
    dataHash = digest(data.data(), data.size());
    scaleHash = digest(scales.data(), scales.size());
    validateDomain();
  }
  if (dataHash != metadata.dataSha256 || scaleHash != metadata.scaleSha256)
    throw ArtifactError("artifact converted payload checksum mismatch");
}

inline Metadata writeAtomically(const std::filesystem::path &base, const Identity &identity,
                                 std::span<const int8_t> data, std::span<const float> scales,
                                 std::string_view expectedProjectionFingerprint) {
  auto meta = detail::layout(identity);
  if (data.size() != meta.dataBytes || scales.size() != identity.n)
    throw ArtifactError("artifact writer projection byte count mismatch");
  for (int8_t code : data)
    if (code == std::numeric_limits<int8_t>::min())
      throw ArtifactError("artifact writer code is outside the converter domain");
  for (float scale : scales)
    if (!std::isfinite(scale) || scale <= 0.0f)
      throw ArtifactError("artifact writer scale is not finite and positive");
  meta.dataSha256 = digest(data.data(), data.size());
  meta.scaleSha256 = digest(scales.data(), meta.scaleBytes);
  meta.projectionFingerprint = projectionFingerprint(identity, meta.dataSha256, meta.scaleSha256);
  if (!expectedProjectionFingerprint.empty() &&
      expectedProjectionFingerprint != meta.projectionFingerprint)
    throw ArtifactError("artifact writer differs from the runtime projection fingerprint");
  const auto location = paths(base, identity);
  const auto reuse = [&] {
    auto existing = loadMetadata(base, identity);
    if (existing.projectionFingerprint != meta.projectionFingerprint ||
        existing.dataSha256 != meta.dataSha256 || existing.scaleSha256 != meta.scaleSha256)
      throw ArtifactError("existing artifact key contains different converted payload");
    detail::validateExisting(existing);
    return existing;
  };
  std::error_code error;
  if (std::filesystem::exists(location.directory, error)) return reuse();
  if (error) throw ArtifactError("artifact directory lookup failed: " + error.message());
  std::filesystem::create_directories(base, error);
  if (error) throw ArtifactError("artifact derivative directory creation failed: " + error.message());
  std::string temporary = (base / (".int8-artifact-" + cacheKey(identity) + "-XXXXXX")).string();
  if (!::mkdtemp(temporary.data())) detail::fileError("artifact temporary directory creation failed");
  const std::filesystem::path temp(temporary);
  try {
    detail::File directory(::open(temp.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (directory.get() < 0) detail::fileError("artifact temporary directory open failed");
    detail::File payload(::openat(directory.get(), "payload.bin",
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (payload.get() < 0) detail::fileError("artifact temporary payload creation failed");
    detail::writeExactly(payload.get(), {reinterpret_cast<const uint8_t *>(data.data()), data.size()});
    detail::zeros(payload.get(), meta.scaleOffset - meta.dataBytes);
    detail::writeExactly(payload.get(), {reinterpret_cast<const uint8_t *>(scales.data()),
                                         static_cast<size_t>(meta.scaleBytes)});
    detail::zeros(payload.get(), meta.payloadBytes - meta.scaleOffset - meta.scaleBytes);
    if (::fsync(payload.get())) detail::fileError("artifact payload sync failed");
    detail::File metadata(::openat(directory.get(), "metadata.bin",
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (metadata.get() < 0) detail::fileError("artifact temporary metadata creation failed");
    const auto encoded = detail::encode(meta);
    detail::writeExactly(metadata.get(), encoded);
    if (::fsync(metadata.get()) || ::fsync(directory.get()))
      detail::fileError("artifact metadata directory sync failed");
    if (::renameatx_np(AT_FDCWD, temp.c_str(), AT_FDCWD,
                       location.directory.c_str(), RENAME_EXCL)) {
      const int failure = errno;
      if (failure != EEXIST && failure != ENOTEMPTY) detail::fileError("artifact atomic publication failed");
      std::filesystem::remove_all(temp, error);
      return reuse();
    }
    detail::File parent(::open(base.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (parent.get() < 0 || ::fsync(parent.get())) detail::fileError("artifact parent directory sync failed");
    meta.payloadPath = location.payload;
    return meta;
  } catch (...) {
    std::filesystem::remove_all(temp, error);
    throw;
  }
}

} // namespace splash::model::preconverted_int8
