#pragma once
// Metadata::load reads bounded JSON and file metadata only. BoundedPayload::load
// is Root's --gpu-only entry point: reserve Metadata::plannedBytes() first. It
// maps exactly one original and one packed layer, never a full-model store.
// All byte certification, hashes and immutable checks run outside timed work.
#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#import <Foundation/Foundation.h>

#include <array>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <span>
#include <vector>

namespace splash::bench::i8_lut {
namespace one = splash::flash::qmv_one_layer;
using metal::MetalBackend;
using metal::MetalBuffer;
inline constexpr uint64_t kAlignment = one::kAlignment;
inline constexpr uint64_t kPackedLayerBytes = 1895301120ULL;
inline constexpr uint64_t kCoefficientBytes = 2516582400ULL;
inline constexpr uint64_t kScaleBytes = 7864320ULL;
inline constexpr uint64_t kMetadataLimit = 2ULL << 20;
inline constexpr const char *kSourceIdentitySHA256 = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr const char *kSourceManifestSHA256 = "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402";
inline constexpr const char *kSourceStoreManifestSHA256 = "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1";
inline constexpr const char *kCoefficientCertificateSHA256 = "48ac7919aa82d245647e1bd262b193e8cf70b9dcee5729ec0d6f20787f1148d9";
inline constexpr const char *kConversionReportSHA256 = "0aea438a13cf047b3348d22d0eb5076822258f5fa8fb2ca161d280d654112615";
inline constexpr std::array<const char *, 3> kRoles{"gate_proj", "up_proj", "down_proj"};

inline void require(bool value, std::string_view reason) {
  if (!value) throw std::invalid_argument("private bounded saved-I8 LUT oracle: " + std::string(reason));
}
inline uint64_t checkedAdd(uint64_t a, uint64_t b) {
  require(b <= UINT64_MAX - a, "extent addition overflow"); return a + b;
}
inline uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  require(!a || b <= UINT64_MAX / a, "geometry multiplication overflow"); return a * b;
}
inline uint64_t rounded(uint64_t bytes) {
  return checkedAdd(bytes, kAlignment - 1) & ~(kAlignment - 1);
}
struct Geometry final {uint64_t experts = 512, outputs = 640, input = 2560;};
struct Layout final {uint64_t rows = 0, groups = 0, codes = 0, ids = 0, lut = 0, scales = 0;};
inline Layout layout(Geometry g) {
  require(g.experts && g.outputs && g.input && g.input % 64 == 0, "empty/non-G64 geometry");
  Layout result; result.rows = checkedMultiply(g.experts, g.outputs); result.groups = g.input / 64;
  result.codes = checkedMultiply(result.rows, g.input);
  result.ids = result.codes / 2;
  result.lut = checkedMultiply(checkedMultiply(result.rows, result.groups), 16);
  result.scales = checkedMultiply(result.rows, sizeof(float));
  require(result.codes <= SIZE_MAX && result.ids <= SIZE_MAX && result.lut <= SIZE_MAX &&
      result.scales <= SIZE_MAX, "geometry exceeds addressable host spans");
  return result;
}
inline Geometry geometry(uint32_t projection) {
  require(projection < 3, "projection outside gate/up/down");
  return projection == 2 ? Geometry{512, 2560, 640} : Geometry{512, 640, 2560};
}

// The verifier enumerates original coefficient positions independently of
// the packer's group/chunk write loop. IDs preserve original U32 nibble order.
inline uint64_t verifyProjection(Geometry g, std::span<const int8_t> saved,
    std::span<const uint8_t> ids, std::span<const int8_t> lut,
    std::span<const float> originalScales, std::span<const float> packedScales) {
  const auto shape = layout(g);
  require(saved.size() == shape.codes && ids.size() == shape.ids && lut.size() == shape.lut &&
      originalScales.size() == shape.rows && packedScales.size() == shape.rows,
      "coefficient/ID/LUT/scale span differs from geometry");
  require(std::find(lut.begin(), lut.end(), int8_t{-128}) == lut.end(), "LUT contains excluded -128 code");
  for (uint64_t row = 0; row < shape.rows; ++row) {
    require(std::isfinite(originalScales[row]) && originalScales[row] > 0.0f,
        "original late F32 scale is nonfinite/nonpositive");
    require(std::bit_cast<uint32_t>(originalScales[row]) == std::bit_cast<uint32_t>(packedScales[row]),
        "duplicated late F32 scale bytes differ");
    for (uint64_t channel = 0; channel < g.input; ++channel) {
      const auto code = saved[row * g.input + channel];
      require(code != -128, "saved I8 contains excluded -128 code");
      const uint8_t pair = ids[row * (g.input / 2) + channel / 2];
      const uint8_t symbol = (channel & 1) ? pair >> 4 : pair & 15;
      const uint64_t address = (row * shape.groups + channel / 64) * 16 + symbol;
      require(lut[address] == code, "packed ID/LUT does not reconstruct an original saved I8 coefficient");
    }
  }
  return shape.codes;
}
inline void requireCanonicalRanks(std::span<const uint32_t> ranks) {
  require(ranks.size() == kAlignment / sizeof(uint32_t), "rank-map allocation extent differs");
  for (uint32_t id = 0; id < 512; ++id) require(ranks[id] == id, "Full512 ID-to-rank map is not canonical");
  for (uint64_t id = 512; id < ranks.size(); ++id) require(ranks[id] == UINT32_MAX, "rank-map unused tail differs");
}

struct Plane final {uint64_t offset = 0, length = 0; std::string sha256;};
struct Projection final {Geometry dimensions; Plane ids, lut, scales;};
struct Q4Binding final {
  std::string projection, sourcePrefix, shard, sourceShard, sourceShardSHA256, rangeSHA256;
  std::filesystem::path path;
  uint64_t offset = 0, length = 0, sourceOffset = 0, sourceBytes = 0;
  std::array<uint64_t, 5> sourceSnapshot{};
};
namespace detail {
inline NSDictionary *object(id value) {
  require([value isKindOfClass:[NSDictionary class]], "JSON object missing/invalid"); return value;
}
inline NSArray *array(id value) {
  require([value isKindOfClass:[NSArray class]], "JSON array missing/invalid"); return value;
}
inline std::string string(id value) {
  require([value isKindOfClass:[NSString class]], "JSON string missing/invalid");
  const char *utf8 = [value UTF8String]; require(utf8 != nullptr, "JSON string UTF8 invalid");
  const auto bytes = [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
  require(std::strlen(utf8) == bytes, "JSON string contains embedded NUL"); return {utf8, size_t(bytes)};
}
inline uint64_t integer(id value) {
  require([value isKindOfClass:[NSNumber class]] &&
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
      !CFNumberIsFloatType((__bridge CFNumberRef)value) && [value longLongValue] >= 0,
      "JSON integer missing/negative/floating/boolean");
  return [value unsignedLongLongValue];
}
inline bool boolean(id value) {
  require([value isKindOfClass:[NSNumber class]] &&
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID(), "JSON boolean missing/invalid");
  return [value boolValue];
}
inline NSString *key(const char *value) {return [NSString stringWithUTF8String:value];}
inline void shape(id value, std::initializer_list<uint64_t> expected) {
  NSArray *actual = array(value); require(actual.count == expected.size(), "plane/dimension shape rank differs");
  NSUInteger index = 0;
  for (const auto dimension : expected) require(integer(actual[index++]) == dimension, "plane/dimension shape differs");
}
inline std::vector<uint8_t> readMetadata(const std::filesystem::path &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  require(fd >= 0, "cannot open bounded metadata JSON");
  struct stat state{};
  if (::fstat(fd, &state) || !S_ISREG(state.st_mode) || state.st_size <= 0 ||
      uint64_t(state.st_size) > kMetadataLimit) {
    ::close(fd); require(false, "metadata JSON is not a bounded regular file");
  }
  std::vector<uint8_t> bytes(size_t(state.st_size)); size_t cursor = 0;
  while (cursor < bytes.size()) {
    const auto count = ::read(fd, bytes.data() + cursor, bytes.size() - cursor);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {::close(fd); require(false, "metadata JSON read was incomplete");}
    cursor += size_t(count);
  }
  struct stat after{};
  const bool unchanged = !::fstat(fd, &after) && after.st_size == state.st_size &&
      after.st_dev == state.st_dev && after.st_ino == state.st_ino &&
      after.st_mtimespec.tv_sec == state.st_mtimespec.tv_sec &&
      after.st_mtimespec.tv_nsec == state.st_mtimespec.tv_nsec &&
      after.st_ctimespec.tv_sec == state.st_ctimespec.tv_sec &&
      after.st_ctimespec.tv_nsec == state.st_ctimespec.tv_nsec;
  ::close(fd); require(unchanged, "metadata JSON changed during bounded read"); return bytes;
}
inline NSDictionary *parse(const std::vector<uint8_t> &bytes) {
  NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()]; NSError *error = nil;
  id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error, "metadata JSON parsing failed"); return object(parsed);
}
inline std::string jsonValue(id value) {
  NSError *error = nil; NSData *data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:&error];
  require(data && !error && data.length <= kMetadataLimit, "witness JSON serialization failed");
  return {static_cast<const char *>(data.bytes), size_t(data.length)};
}
inline std::array<uint64_t, 5> snapshot(const std::filesystem::path &path) {
  struct stat state{};
  require(!::lstat(path.c_str(), &state) && S_ISREG(state.st_mode) && state.st_size > 0 &&
      state.st_mtimespec.tv_sec >= 0 && state.st_ctimespec.tv_sec >= 0, "Q4 source snapshot file metadata invalid");
  return {uint64_t(state.st_dev), uint64_t(state.st_ino), uint64_t(state.st_size),
      checkedAdd(checkedMultiply(uint64_t(state.st_mtimespec.tv_sec), 1000000000), uint64_t(state.st_mtimespec.tv_nsec)),
      checkedAdd(checkedMultiply(uint64_t(state.st_ctimespec.tv_sec), 1000000000), uint64_t(state.st_ctimespec.tv_nsec))};
}
inline std::array<uint64_t, 5> snapshot(id value) {
  NSArray *record = array(value); require(record.count == 5, "source snapshot certificate extent differs");
  std::array<uint64_t, 5> result{};
  for (uint32_t field = 0; field < 5; ++field) result[field] = integer(record[field]); return result;
}
inline void readonlyFile(const std::filesystem::path &path, uint64_t bytes) {
  struct stat state{};
  require(!::lstat(path.c_str(), &state) && S_ISREG(state.st_mode) && !(state.st_mode & 0222) &&
      state.st_size > 0 && uint64_t(state.st_size) == bytes, "payload is not an exact-sized readonly regular file");
}
inline Plane plane(id value, const char *dtype, std::initializer_list<uint64_t> dimensions,
    uint64_t length, uint64_t &cursor, uint64_t ownerBytes) {
  NSDictionary *record = object(value); require(string(record[@"dtype"]) == dtype, "plane dtype differs");
  shape(record[@"shape"], dimensions); Plane result;
  result.offset = integer(record[@"offset"]); result.length = integer(record[@"length"]);
  result.sha256 = string(record[@"sha256"]); one::detail::requireDigest(result.sha256);
  cursor = rounded(cursor);
  require(result.offset == cursor && result.length == length && result.offset <= ownerBytes &&
      result.length <= ownerBytes - result.offset, "plane alignment/order/extent differs");
  cursor = checkedAdd(result.offset, result.length); return result;
}
inline void requirePlaneHash(const uint8_t *bytes, const Plane &plane) {
  require(one::detail::hash(bytes + plane.offset, plane.length) == plane.sha256, "packed plane hash differs");
}
inline void requireZeroPadding(const uint8_t *bytes, uint64_t begin, uint64_t end) {
  require(begin <= end && std::all_of(bytes + begin, bytes + end, [](uint8_t value) {return value == 0;}),
      "payload alignment/tail padding is nonzero");
}
} // namespace detail

struct Metadata final {
  std::filesystem::path directory, path;
  std::string manifestSHA256, certificateSHA256, sha256;
  std::string sourceIdentitySHA256, sourceManifestSHA256, sourceStoreManifestSHA256;
  std::filesystem::path sourceDirectory;
  std::array<Q4Binding, 3> q4Bindings;
  std::string originalQ4IDInputsJSON, sourceMetadataCertificatesJSON;
  std::string sourceCoefficientCertificateSHA256, sourceConversionReportSHA256, sourceShardHashPolicy;
  uint32_t layerIndex = 0;
  uint64_t bytes = 0;
  std::array<Projection, 3> projections;
  flash::FlashInt8ExpertStoreLayer original;
  void validate() const {
    require(layerIndex < 48 && bytes == kPackedLayerBytes && original.path.is_absolute() &&
        original.path == original.path.lexically_normal() && original.path != path,
        "bounded metadata layer/path/extent differs");
    one::detail::validateEntry(original); one::detail::requireDigest(sha256);
    require(sourceIdentitySHA256 == kSourceIdentitySHA256 && sourceManifestSHA256 == kSourceManifestSHA256 &&
        sourceStoreManifestSHA256 == kSourceStoreManifestSHA256, "qualified source/store identities differ");
    uint64_t cursor = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const auto g = geometry(projection); const auto sizes = layout(g); const auto &p = projections[projection];
      require(p.dimensions.experts == g.experts && p.dimensions.outputs == g.outputs && p.dimensions.input == g.input,
          "packed projection geometry differs");
      const auto plane = [&](const Plane &range, uint64_t length) {
        cursor = rounded(cursor);
        require(range.offset == cursor && range.length == length && range.offset <= bytes &&
            range.length <= bytes - range.offset, "packed plane geometry/order/extent differs");
        one::detail::requireDigest(range.sha256); cursor = checkedAdd(range.offset, range.length);
      };
      plane(p.ids, sizes.ids); plane(p.lut, sizes.lut); plane(p.scales, sizes.scales);
      require(p.scales.sha256 == original.scales[projection].sha256, "copied scale hashes differ");
      const auto &q4 = q4Bindings[projection];
      require(q4.projection == kRoles[projection] && q4.rangeSHA256 == p.ids.sha256 && q4.path.is_absolute() &&
          q4.sourcePrefix == "language_model.model.layers." + std::to_string(layerIndex) + ".mlp.switch_mlp." + q4.projection &&
          q4.length == sizes.ids && q4.offset % kAlignment == 0 && q4.offset <= q4.sourceSnapshot[2] &&
          q4.length <= q4.sourceSnapshot[2] - q4.offset, "original Q4 input ID witness binding/extent differs");
      one::detail::requireDigest(q4.sourceShardSHA256);
    }
    require(rounded(cursor) == bytes, "packed final extent differs");
  }
  // The original checkpoint remains writable on disk; these metadata-only
  // checks preserve its prior full-shard certificate without reading a shard.
  void checkSourceWitness() const {
    for (const auto &binding : q4Bindings)
      require(detail::snapshot(binding.path) == binding.sourceSnapshot, "original Q4 certified source snapshot changed");
    const auto manifest = detail::readMetadata(sourceDirectory / "manifest.json");
    const auto certificate = detail::readMetadata(original.path.parent_path() / "coefficient-certificate.json");
    const auto report = detail::readMetadata(original.path.parent_path() / "conversion-report.json");
    require(one::detail::hash(manifest.data(), manifest.size()) == kSourceManifestSHA256 &&
        one::detail::hash(certificate.data(), certificate.size()) == kCoefficientCertificateSHA256 &&
        one::detail::hash(report.data(), report.size()) == kConversionReportSHA256,
        "original Q4 source/certificate metadata witness changed");
  }
  [[nodiscard]] uint64_t plannedBytes() const {
    validate();
    return checkedAdd(one::oneLayerPlannedBytes(original), bytes);
  }
  [[nodiscard]] static Metadata load(const std::filesystem::path &package) {
    Metadata result; result.directory = std::filesystem::canonical(package);
    require(std::filesystem::is_directory(result.directory), "packed root is not a directory");
    const auto manifestBytes = detail::readMetadata(result.directory / "manifest.json");
    const auto certificateBytes = detail::readMetadata(result.directory / "certificate.json");
    result.manifestSHA256 = one::detail::hash(manifestBytes.data(), manifestBytes.size());
    result.certificateSHA256 = one::detail::hash(certificateBytes.data(), certificateBytes.size());
    NSDictionary *manifest = detail::parse(manifestBytes), *certificate = detail::parse(certificateBytes);
    require(detail::string(manifest[@"schema"]) == "splash-prefill-i8-lut-one-layer-v1" &&
        detail::integer(manifest[@"alignment"]) == kAlignment && detail::integer(manifest[@"group_size"]) == 64,
        "packed manifest schema/alignment/group differs");
    const uint64_t index = detail::integer(manifest[@"layer_index"]); require(index < 48, "layer outside [0,47]");
    result.layerIndex = uint32_t(index);
    NSArray *selected = detail::array(manifest[@"selected_experts"]);
    require(selected.count == 512 && detail::string(manifest[@"rank_order"]) == "expert_id", "Full512 rank inventory differs");
    for (uint32_t id = 0; id < 512; ++id) {
      require(detail::integer(selected[id]) == id, "Full512 selected ID inventory is not canonical");
      result.original.selectedIDs.push_back(id);
    }
    char basename[32]{}; std::snprintf(basename, sizeof(basename), "layer-%02u.bin", result.layerIndex);
    require(detail::string(manifest[@"path"]) == basename, "packed payload path is not the canonical layer basename");
    result.path = result.directory / basename; result.bytes = detail::integer(manifest[@"bytes"]);
    require(result.bytes == kPackedLayerBytes, "packed Full512 layer extent differs");
    result.sha256 = detail::string(manifest[@"sha256"]); one::detail::requireDigest(result.sha256);
    result.sourceIdentitySHA256 = detail::string(manifest[@"source_identity_sha256"]);
    result.sourceManifestSHA256 = detail::string(manifest[@"source_manifest_sha256"]);
    result.sourceStoreManifestSHA256 = detail::string(manifest[@"source_store_manifest_sha256"]);
    for (const auto &digest : {result.sourceIdentitySHA256, result.sourceManifestSHA256, result.sourceStoreManifestSHA256})
      one::detail::requireDigest(digest);
    require(result.sourceIdentitySHA256 == kSourceIdentitySHA256 && result.sourceManifestSHA256 == kSourceManifestSHA256 &&
        result.sourceStoreManifestSHA256 == kSourceStoreManifestSHA256, "qualified source/checkpoint/Full512 store identity differs");
    NSDictionary *source = detail::object(manifest[@"source_i8_layer"]);
    require(detail::integer(source[@"layer_index"]) == index, "original and packed layer indices differ");
    result.original.path = detail::string(source[@"path"]);
    require(result.original.path.is_absolute() && result.original.path == result.original.path.lexically_normal() &&
        result.original.path != result.path, "original I8 path must be distinct, absolute and normalized");
    result.original.bytes = detail::integer(source[@"logical_size"]);
    result.original.sha256 = detail::string(source[@"sha256"]); one::detail::requireDigest(result.original.sha256);
    require(result.original.bytes == one::kFull512LayerBytes, "original Full512 layer extent differs");
    // Authenticate the chosen original layer against the pinned store JSON,
    // without invoking the production validator or touching payload bytes.
    const auto storeBytes = detail::readMetadata(result.original.path.parent_path() / "manifest.json");
    require(one::detail::hash(storeBytes.data(), storeBytes.size()) == kSourceStoreManifestSHA256,
        "original Full512 store manifest metadata seal differs");
    NSDictionary *store = detail::parse(storeBytes);
    require(detail::string(store[@"schema"]) == "splash-flash-int8-expert-store-v1" &&
        detail::string(store[@"source_identity_sha256"]) == kSourceIdentitySHA256 &&
        detail::string(store[@"source_manifest_sha256"]) == kSourceManifestSHA256 &&
        detail::integer(store[@"alignment"]) == kAlignment && detail::integer(store[@"target_layers"]) == 48,
        "original Full512 store source/schema metadata differs");
    NSArray *storeLayers = detail::array(store[@"layers"]), *storeInventory = detail::array(store[@"selected_experts"]);
    require(storeLayers.count == 48 && storeInventory.count == 48, "original Full512 layer/inventory count differs");
    NSArray *storeSelected = detail::array(storeInventory[index]); require(storeSelected.count == 512, "original Full512 expert count differs");
    for (uint32_t id = 0; id < 512; ++id) require(detail::integer(storeSelected[id]) == id, "original Full512 rank inventory differs");
    NSDictionary *storeLayer = detail::object(storeLayers[index]);
    require(detail::integer(storeLayer[@"layer_index"]) == index && detail::string(storeLayer[@"path"]) == basename &&
        result.original.path.filename() == basename && detail::integer(storeLayer[@"bytes"]) == result.original.bytes &&
        detail::string(storeLayer[@"sha256"]) == result.original.sha256,
        "packed source layer is not the chosen pinned Full512 manifest record");
    NSDictionary *certifiedProjections = detail::object(storeLayer[@"projections"]);
    NSDictionary *packedProjections = detail::object(manifest[@"projections"]);
    NSDictionary *originalProjections = detail::object(source[@"projections"]);
    uint64_t packedCursor = 0, originalCursor = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const auto g = geometry(projection); const auto sizes = layout(g);
      auto &target = result.projections[projection]; target.dimensions = g;
      NSDictionary *packed = detail::object(packedProjections[detail::key(kRoles[projection])]);
      NSDictionary *saved = detail::object(originalProjections[detail::key(kRoles[projection])]);
      NSDictionary *certified = detail::object(certifiedProjections[detail::key(kRoles[projection])]);
      require([saved isEqualToDictionary:certified], "packed original projection metadata differs from pinned Full512 store");
      const std::string prefix = "language_model.model.layers." + std::to_string(index) + ".mlp.switch_mlp." + kRoles[projection];
      require(detail::string(packed[@"source_prefix"]) == prefix && detail::string(saved[@"source_prefix"]) == prefix,
          "original projection source prefix differs");
      detail::shape(packed[@"dimensions"], {g.experts, g.outputs, g.input});
      detail::shape(saved[@"dimensions"], {g.experts, g.outputs, g.input});
      target.ids = detail::plane(packed[@"ids"], "U8", {g.experts, g.outputs, g.input / 2}, sizes.ids, packedCursor, result.bytes);
      target.lut = detail::plane(packed[@"lut"], "I8", {g.experts, g.outputs, g.input / 64, 16}, sizes.lut, packedCursor, result.bytes);
      target.scales = detail::plane(packed[@"scales"], "F32", {g.experts, g.outputs}, sizes.scales, packedCursor, result.bytes);
      const auto codes = detail::plane(saved[@"codes"], "I8", {g.experts, g.outputs, g.input}, sizes.codes, originalCursor, result.original.bytes);
      const auto scales = detail::plane(saved[@"scales"], "F32", {g.experts, g.outputs}, sizes.scales, originalCursor, result.original.bytes);
      result.original.codes[projection] = {codes.offset, codes.length, codes.sha256};
      result.original.scales[projection] = {scales.offset, scales.length, scales.sha256};
      require(target.scales.sha256 == scales.sha256, "copied late-scale metadata hashes differ");
    }
    require(rounded(packedCursor) == result.bytes && rounded(originalCursor) == result.original.bytes,
        "packed/original final layer extent differs");
    one::detail::validateEntry(result.original);
    // Original Q4 bytes stay unmapped here. Their exact range hashes are the
    // packed ID plane hashes; full-shard identity comes from the pinned prior
    // conversion report and exact dev/ino/size/mtime/ctime snapshot witnesses.
    NSArray *q4Inputs = detail::array(manifest[@"original_q4_id_inputs"]);
    require(q4Inputs.count == 3, "original Q4 input witness inventory differs");
    result.originalQ4IDInputsJSON = detail::jsonValue(q4Inputs);
    std::array<bool, 3> bound{};
    for (NSDictionary *entry in q4Inputs) {
      NSDictionary *record = detail::object(entry); const auto role = detail::string(record[@"projection"]);
      uint32_t projection = 0; while (projection < 3 && role != kRoles[projection]) ++projection;
      require(projection < 3 && !bound[projection], "original Q4 input projection witness duplicates/unknown role"); bound[projection] = true;
      const auto g = geometry(projection); const auto sizes = layout(g); auto &binding = result.q4Bindings[projection];
      binding.projection = role; binding.sourcePrefix = "language_model.model.layers." + std::to_string(index) + ".mlp.switch_mlp." + role;
      require(detail::string(record[@"dtype"]) == "U32", "original Q4 ID witness dtype differs");
      detail::shape(record[@"shape"], {g.experts, g.outputs, g.input / 8});
      binding.path = detail::string(record[@"path"]); binding.shard = detail::string(record[@"shard"]);
      binding.offset = detail::integer(record[@"offset"]); binding.length = detail::integer(record[@"length"]);
      binding.sourceShard = detail::string(record[@"source_shard"]); binding.sourceOffset = detail::integer(record[@"source_offset"]);
      binding.sourceShardSHA256 = detail::string(record[@"source_shard_sha256"]);
      binding.rangeSHA256 = detail::string(record[@"range_sha256"]); binding.sourceSnapshot = detail::snapshot(record[@"source_snapshot"]);
      one::detail::requireDigest(binding.sourceShardSHA256); one::detail::requireDigest(binding.rangeSHA256);
      require(binding.rangeSHA256 == result.projections[projection].ids.sha256 && binding.length == sizes.ids &&
          binding.offset % kAlignment == 0 && binding.offset <= binding.sourceSnapshot[2] &&
          binding.length <= binding.sourceSnapshot[2] - binding.offset && binding.path.is_absolute() &&
          binding.path == binding.path.lexically_normal(), "original Q4 input range/ID hash/path differs");
      const std::filesystem::path relative(binding.shard);
      require(!relative.empty() && !relative.is_absolute() && relative == relative.lexically_normal() &&
          std::filesystem::canonical(binding.path) == binding.path, "Q4 shard path is not relative/canonical");
      auto sourceRoot = binding.path;
      for (const auto &component : relative) {require(component != ".." && component != ".", "Q4 shard path escapes source root"); sourceRoot = sourceRoot.parent_path();}
      require(sourceRoot / relative == binding.path, "Q4 shard absolute/relative path binding differs");
      if (result.sourceDirectory.empty()) result.sourceDirectory = sourceRoot;
      require(result.sourceDirectory == sourceRoot && detail::snapshot(binding.path) == binding.sourceSnapshot,
          "Q4 input source root/certified snapshot differs");
    }
    const auto originalManifestBytes = detail::readMetadata(result.sourceDirectory / "manifest.json");
    require(one::detail::hash(originalManifestBytes.data(), originalManifestBytes.size()) == kSourceManifestSHA256,
        "original Q4 source manifest metadata seal differs");
    NSDictionary *originalManifest = detail::parse(originalManifestBytes);
    NSDictionary *sourceCertificates = detail::object(manifest[@"source_metadata_certificates"]);
    result.sourceMetadataCertificatesJSON = detail::jsonValue(sourceCertificates);
    result.sourceCoefficientCertificateSHA256 = detail::string(sourceCertificates[@"coefficient_certificate_sha256"]);
    result.sourceConversionReportSHA256 = detail::string(sourceCertificates[@"conversion_report_sha256"]);
    require(result.sourceCoefficientCertificateSHA256 == kCoefficientCertificateSHA256 &&
        result.sourceConversionReportSHA256 == kConversionReportSHA256, "source metadata certificate identities differ");
    const auto coefficientCertificateBytes = detail::readMetadata(result.original.path.parent_path() / "coefficient-certificate.json");
    const auto conversionReportBytes = detail::readMetadata(result.original.path.parent_path() / "conversion-report.json");
    require(one::detail::hash(coefficientCertificateBytes.data(), coefficientCertificateBytes.size()) == kCoefficientCertificateSHA256 &&
        one::detail::hash(conversionReportBytes.data(), conversionReportBytes.size()) == kConversionReportSHA256,
        "published coefficient/snapshot certificate metadata seal differs");
    NSDictionary *conversionReport = detail::parse(conversionReportBytes);
    require(detail::string(conversionReport[@"source"]) == result.sourceDirectory.string() &&
        detail::boolean(conversionReport[@"source_manifest_and_plan_sha_reverified_at_end"]) &&
        detail::boolean(conversionReport[@"source_plan_and_shard_snapshots_unchanged"]) &&
        detail::integer(conversionReport[@"source_shards_verified_against_original_manifest_sha"]) == 14,
        "source full-shard/snapshot certificate scope differs");
    NSDictionary *priorSnapshots = detail::object(conversionReport[@"source_readonly_snapshots"]);
    NSDictionary *nativeTensors = detail::object(originalManifest[@"tensors"]);
    NSArray *nativeShards = detail::array(originalManifest[@"shards"]);
    for (uint32_t projection = 0; projection < 3; ++projection) {
      auto &binding = result.q4Bindings[projection];
      require(detail::snapshot(priorSnapshots[detail::key(binding.path.c_str())]) == binding.sourceSnapshot,
          "Q4 source snapshot differs from pinned full-shard verification certificate");
      NSDictionary *native = detail::object(nativeTensors[detail::key((binding.sourcePrefix + ".weight").c_str())]);
      NSDictionary *witness = nil;
      for (NSDictionary *candidate in q4Inputs) if (detail::string(candidate[@"projection"]) == binding.projection) witness = candidate;
      NSMutableDictionary *descriptor = [witness mutableCopy];
      for (NSString *extra in @[@"projection", @"path", @"source_shard_sha256", @"source_snapshot", @"range_sha256"]) [descriptor removeObjectForKey:extra];
      require([native isEqualToDictionary:descriptor], "Q4 input witness differs from original sealed tensor descriptor");
      bool foundShard = false;
      for (NSDictionary *candidate in nativeShards) if (detail::string(candidate[@"path"]) == binding.shard) {
        require(!foundShard && detail::string(candidate[@"sha256"]) == binding.sourceShardSHA256 &&
            detail::integer(candidate[@"bytes"]) == binding.sourceSnapshot[2] &&
            detail::string(candidate[@"source_path"]) == binding.sourceShard, "Q4 native/original source shard identity differs");
        binding.sourceBytes = detail::integer(candidate[@"source_bytes"]); foundShard = true;
      }
      require(foundShard && binding.sourceOffset >= 8 && binding.sourceOffset <= binding.sourceBytes &&
          binding.length <= binding.sourceBytes - binding.sourceOffset, "original checkpoint Q4 range exceeds source shard");
    }
    require(detail::string(certificate[@"schema"]) == "splash-prefill-i8-lut-exact-byte-certificate-v1" &&
        detail::boolean(certificate[@"pass"]) && !detail::boolean(certificate[@"gpu_work"]) &&
        detail::string(certificate[@"manifest_sha256"]) == result.manifestSHA256 &&
        detail::integer(certificate[@"layer_index"]) == index &&
        detail::string(certificate[@"source_store_manifest_sha256"]) == result.sourceStoreManifestSHA256 &&
        detail::boolean(certificate[@"all_saved_i8_bytes_reconstructed_exactly"]) &&
        detail::boolean(certificate[@"all_saved_scale_bytes_copied_exactly"]) &&
        detail::boolean(certificate[@"stored_output_readback_verified"]) &&
        detail::boolean(certificate[@"source_and_store_snapshots_unchanged"]) &&
        detail::integer(certificate[@"coefficient_code_bytes"]) == kCoefficientBytes &&
        detail::integer(certificate[@"scale_bytes"]) == kScaleBytes, "packed exact-byte certificate differs");
    require(detail::integer(certificate[@"original_q4_id_bytes"]) == kCoefficientBytes / 2 &&
        !detail::boolean(certificate[@"source_shards_full_rehashed"]) && !detail::boolean(certificate[@"full_store_payload_rehashed"]),
        "bounded pack certificate source-read scope differs");
    result.sourceShardHashPolicy = detail::string(certificate[@"source_shard_hash_policy"]);
    require(result.sourceShardHashPolicy == "prior conversion-report full-shard verification plus exact certified dev/ino/size/mtime/ctime snapshots with O_RDONLY mappings, rechecked before and after",
        "original Q4 full-shard/snapshot proof policy differs");
    NSDictionary *checks = detail::object(certificate[@"projection_checks"]);
    for (uint32_t projection = 0; projection < 3; ++projection) {
      NSDictionary *check = detail::object(checks[detail::key(kRoles[projection])]); const auto sizes = layout(geometry(projection));
      require(detail::integer(check[@"coefficient_code_bytes"]) == sizes.codes &&
          detail::integer(check[@"scale_bytes"]) == sizes.scales && detail::boolean(check[@"exact_code_bytes"]) &&
          detail::boolean(check[@"exact_scale_bytes"]), "packed per-projection certificate differs");
    }
    detail::readonlyFile(result.path, result.bytes); detail::readonlyFile(result.original.path, result.original.bytes);
    require(result.plannedBytes() == 4419764224ULL, "bounded two-mapping/rank ledger differs"); return result;
  }
};

struct BoundedPayload final {
  one::OneLayerPayload original;
  MetalBuffer base;
  std::array<MetalBuffer, 3> ids, luts, scales;
  uint32_t layerIndex = 0;
  uint64_t plannedBytes = 0, allocatedBytes = 0, exactCodeBytesCertified = 0;
  [[nodiscard]] static BoundedPayload load(MetalBackend &backend, const Metadata &metadata) {
    BoundedPayload result; result.metadata_ = metadata; result.layerIndex = metadata.layerIndex;
    result.plannedBytes = metadata.plannedBytes();
    metadata.checkSourceWitness();
    require(metadata.bytes <= backend.capabilities().maxBufferLengthBytes &&
        metadata.original.bytes <= backend.capabilities().maxBufferLengthBytes, "one-layer mapping exceeds Metal buffer limit");
    result.originalMapping_ = std::make_shared<one::detail::ReadonlyMapping>(metadata.original.path, metadata.original.bytes);
    result.packedMapping_ = std::make_shared<one::detail::ReadonlyMapping>(metadata.path, metadata.bytes);
    result.requireMappedHashes(); result.exactCodeBytesCertified = result.verifyMapped();
    result.originalMapping_->requireUnchanged(); result.packedMapping_->requireUnchanged();
    metadata.checkSourceWitness();
    const uint64_t before = backend.memoryStats().allocatedBytes;
    auto &source = result.original; source.layer = metadata.layerIndex; source.plannedBytes = one::oneLayerPlannedBytes(metadata.original);
    source.base = backend.wrapSharedMemory(result.originalMapping_->address(), metadata.original.bytes,
        result.originalMapping_, "Root readonly saved-I8 Full512 one-layer control");
    source.ranks = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "Root Full512 canonical expert-ID rank map");
    std::memset(source.ranks.contents(), 0xff, kAlignment);
    auto *ranks = static_cast<uint32_t *>(source.ranks.contents()); for (uint32_t id = 0; id < 512; ++id) ranks[id] = id;
    result.rankSHA256_ = one::detail::hash(source.ranks.contents(), kAlignment);
    source.allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
    require(source.allocatedBytes <= source.plannedBytes, "original/rank allocation exceeded reserved ledger");
    result.base = backend.wrapSharedMemory(result.packedMapping_->address(), metadata.bytes,
        result.packedMapping_, "Root readonly original-Q4-ID signed-I8-LUT one-layer payload");
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const auto &packed = metadata.projections[projection]; const auto &entry = metadata.original;
      source.codes[projection] = backend.view(source.base, entry.codes[projection].offset, entry.codes[projection].length);
      source.scales[projection] = backend.view(source.base, entry.scales[projection].offset, entry.scales[projection].length);
      result.ids[projection] = backend.view(result.base, packed.ids.offset, packed.ids.length);
      result.luts[projection] = backend.view(result.base, packed.lut.offset, packed.lut.length);
      result.scales[projection] = backend.view(result.base, packed.scales.offset, packed.scales.length);
    }
    result.allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
    require(result.allocatedBytes <= result.plannedBytes, "two mappings/rank allocation exceeded reserved ledger");
    requireCanonicalRanks({ranks, size_t(kAlignment / sizeof(uint32_t))}); return result;
  }
  [[nodiscard]] uint64_t verify() const {
    require(originalMapping_ && packedMapping_, "exact reconstruction requires a loaded bounded payload");
    originalMapping_->requireUnchanged(); packedMapping_->requireUnchanged(); requireBoundViews();
    const auto count = verifyMapped(); require(count == kCoefficientBytes, "exact code-byte certificate incomplete");
    requireCanonicalRanks({static_cast<const uint32_t *>(original.ranks.contents()), size_t(kAlignment / sizeof(uint32_t))});
    originalMapping_->requireUnchanged(); packedMapping_->requireUnchanged(); return count;
  }
  void checkImmutable() const {
    require(originalMapping_ && packedMapping_, "immutable check requires a loaded bounded payload");
    requireBoundViews(); originalMapping_->requireUnchanged(); packedMapping_->requireUnchanged(); requireMappedHashes();
    metadata_.checkSourceWitness();
    require(one::detail::hash(original.ranks.contents(), kAlignment) == rankSHA256_, "canonical rank bytes changed");
    const auto manifest = detail::readMetadata(metadata_.directory / "manifest.json");
    const auto certificate = detail::readMetadata(metadata_.directory / "certificate.json");
    const auto store = detail::readMetadata(metadata_.original.path.parent_path() / "manifest.json");
    require(one::detail::hash(manifest.data(), manifest.size()) == metadata_.manifestSHA256 &&
        one::detail::hash(certificate.data(), certificate.size()) == metadata_.certificateSHA256 &&
        one::detail::hash(store.data(), store.size()) == kSourceStoreManifestSHA256,
        "packed manifest/certificate metadata changed during Root timing");
    require(verify() == exactCodeBytesCertified, "post-timing exact reconstruction certificate differs");
    originalMapping_->requireUnchanged(); packedMapping_->requireUnchanged();
    metadata_.checkSourceWitness();
  }
private:
  Metadata metadata_;
  std::shared_ptr<one::detail::ReadonlyMapping> originalMapping_, packedMapping_;
  std::string rankSHA256_;
  [[nodiscard]] uint64_t verifyMapped() const {
    const auto *saved = static_cast<const uint8_t *>(originalMapping_->address());
    const auto *packed = static_cast<const uint8_t *>(packedMapping_->address()); uint64_t count = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const auto g = geometry(projection); const auto sizes = layout(g); const auto &target = metadata_.projections[projection];
      count += verifyProjection(g,
          {reinterpret_cast<const int8_t *>(saved + metadata_.original.codes[projection].offset), size_t(sizes.codes)},
          {packed + target.ids.offset, size_t(sizes.ids)},
          {reinterpret_cast<const int8_t *>(packed + target.lut.offset), size_t(sizes.lut)},
          {reinterpret_cast<const float *>(saved + metadata_.original.scales[projection].offset), size_t(sizes.rows)},
          {reinterpret_cast<const float *>(packed + target.scales.offset), size_t(sizes.rows)});
    }
    return count;
  }
  void requireMappedHashes() const {
    const auto *saved = static_cast<const uint8_t *>(originalMapping_->address());
    const auto *packed = static_cast<const uint8_t *>(packedMapping_->address());
    require(one::detail::hash(saved, metadata_.original.bytes) == metadata_.original.sha256, "original whole-layer checksum differs");
    require(one::detail::hash(packed, metadata_.bytes) == metadata_.sha256, "packed whole-layer checksum differs");
    uint64_t savedCursor = 0, packedCursor = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      for (const auto &range : {metadata_.original.codes[projection], metadata_.original.scales[projection]}) {
        detail::requireZeroPadding(saved, savedCursor, range.offset);
        require(one::detail::hash(saved + range.offset, range.length) == range.sha256, "original plane checksum differs");
        savedCursor = range.offset + range.length;
      }
      const auto &target = metadata_.projections[projection];
      for (const auto &range : {target.ids, target.lut, target.scales}) {
        detail::requireZeroPadding(packed, packedCursor, range.offset); detail::requirePlaneHash(packed, range);
        packedCursor = range.offset + range.length;
      }
    }
    detail::requireZeroPadding(saved, savedCursor, metadata_.original.bytes);
    detail::requireZeroPadding(packed, packedCursor, metadata_.bytes);
  }
  void requireBoundViews() const {
    require(layerIndex == metadata_.layerIndex && original.layer == layerIndex && plannedBytes == metadata_.plannedBytes() &&
        original.plannedBytes == one::oneLayerPlannedBytes(metadata_.original) && allocatedBytes <= plannedBytes &&
        base && original.base && original.ranks && base.storage() == metal::BufferStorage::Shared &&
        original.base.storage() == metal::BufferStorage::Shared && original.ranks.storage() == metal::BufferStorage::Shared &&
        base.contents() == packedMapping_->address() && original.base.contents() == originalMapping_->address() &&
        original.ranks.contents() && base.sizeBytes() == metadata_.bytes &&
        original.base.sizeBytes() == metadata_.original.bytes && original.ranks.sizeBytes() == kAlignment,
        "loaded bounded mapping/rank identity/ledger differs");
    const auto *saved = static_cast<const uint8_t *>(originalMapping_->address());
    const auto *packed = static_cast<const uint8_t *>(packedMapping_->address());
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const auto check = [](const MetalBuffer &view, const uint8_t *start, uint64_t offset, uint64_t length) {
        require(view && view.storage() == metal::BufferStorage::Shared && view.contents() == start + offset &&
            view.sizeBytes() == length, "immutable source/packed plane view identity differs");
      };
      const auto &source = metadata_.original; const auto &target = metadata_.projections[projection];
      check(original.codes[projection], saved, source.codes[projection].offset, source.codes[projection].length);
      check(original.scales[projection], saved, source.scales[projection].offset, source.scales[projection].length);
      check(ids[projection], packed, target.ids.offset, target.ids.length); check(luts[projection], packed, target.lut.offset, target.lut.length);
      check(scales[projection], packed, target.scales.offset, target.scales.length);
    }
  }
};

struct Variant final {const char *suffix; uint32_t sg, k; bool stagedI8;};
inline constexpr std::array<Variant, 8> variants{{
    {"m32_n64_k128_sg2_bf16", 2, 128, false}, {"m32_n64_k128_sg2_i8", 2, 128, true},
    {"m32_n64_k128_sg4_bf16", 4, 128, false}, {"m32_n64_k128_sg4_i8", 4, 128, true},
    {"m32_n64_k256_sg2_bf16", 2, 256, false}, {"m32_n64_k256_sg2_i8", 2, 256, true},
    {"m32_n64_k256_sg4_bf16", 4, 256, false}, {"m32_n64_k256_sg4_i8", 4, 256, true}
}};
inline std::string pipelineName(const Variant &variant, bool gate) {
  return std::string("prefill_i8_lut_") + (gate ? "gate_up_" : "down_scatter_") + variant.suffix;
}
class Commands final {
public:
  std::vector<metal::ComputeDispatch> commands;
  Commands(std::span<const metal::ComputeDispatch> source, uint32_t rows, const Variant &variant,
      const BoundedPayload &payload) {
    require((variant.sg == 2 || variant.sg == 4) && (variant.k == 128 || variant.k == 256), "candidate tile geometry differs");
    bool gateSeen = false, downSeen = false;
    for (const auto &original : source) {
      auto dispatch = original;
      const bool gate = dispatch.pipelineName.starts_with("flash_int8_expert_store_gate_up_m32_") ||
          dispatch.pipelineName.starts_with("prefill_i8_lut_gate_up_m32_n64_");
      const bool down = dispatch.pipelineName.starts_with("flash_int8_expert_store_down_scatter_m32_") ||
          dispatch.pipelineName.starts_with("prefill_i8_lut_down_scatter_m32_n64_");
      if (!gate && !down) {commands.push_back(dispatch); continue;}
      require(dispatch.bytes.size() == 1 && dispatch.bytes[0].sizeBytes == sizeof(FlashInt8ExpertStoreParams) &&
          dispatch.bytes[0].data, "native producer parameter ABI differs");
      FlashInt8ExpertStoreParams p{}; std::memcpy(&p, dispatch.bytes[0].data, sizeof(p));
      require(rows >= 1 && rows <= 2048 && p.rows == rows && p.selections == 10 && p.route_capacity == rows * 10 &&
          p.tile_rows == 32 && p.job_capacity == (rows * 10 + 31) / 32 + 511 && p.stored_experts == 512 &&
          !p.scale_group_size && !p.reserved && dispatch.threadgroups.x == (gate ? 10u : 40u) &&
          dispatch.threadgroups.y == p.job_capacity && dispatch.threadgroups.z == 1,
          "native M32 bucket geometry differs");
      const auto &layer = payload.original;
      require((gate && !gateSeen && !downSeen && dispatch.buffers.size() == 11 && dispatch.bytes[0].index == 11 &&
          dispatch.buffers[1].buffer.sameView(layer.codes[0]) && dispatch.buffers[2].buffer.sameView(layer.scales[0]) &&
          dispatch.buffers[3].buffer.sameView(layer.codes[1]) && dispatch.buffers[4].buffer.sameView(layer.scales[1]) &&
          dispatch.buffers[5].buffer.sameView(layer.ranks)) ||
          (down && gateSeen && !downSeen && dispatch.buffers.size() == 10 && dispatch.bytes[0].index == 10 &&
          dispatch.buffers[1].buffer.sameView(layer.codes[2]) && dispatch.buffers[2].buffer.sameView(layer.scales[2]) &&
          dispatch.buffers[3].buffer.sameView(layer.ranks)), "immutable producer operands/dispatch order differs");
      for (uint32_t index = 0; index < dispatch.buffers.size(); ++index)
        require(dispatch.buffers[index].index == index, "native producer buffer-index inventory differs");
      if (gate) {gateSeen = true; dispatch.buffers[1].buffer = payload.ids[0]; dispatch.buffers[3].buffer = payload.ids[1];
        dispatch.buffers.push_back({12, payload.luts[0]}); dispatch.buffers.push_back({13, payload.luts[1]});
      } else {downSeen = true; dispatch.buffers[1].buffer = payload.ids[2]; dispatch.buffers.push_back({11, payload.luts[2]});}
      dispatch.pipelineName = pipelineName(variant, gate); dispatch.threadsPerThreadgroup = {uint64_t(variant.sg) * 32, 1, 1};
      commands.push_back(std::move(dispatch));
    }
    require(gateSeen && downSeen && commands.size() == source.size(), "candidate full-chain dispatch inventory differs");
  }
};

inline void cpuSelfTest() {
  const auto mustFail = [](const auto &operation) {
    bool rejected = false; try {operation();} catch (const std::invalid_argument &) {rejected = true;}
    require(rejected, "CPU malformed fixture was accepted");
  };
  require(rounded(0) == 0 && kPackedLayerBytes + one::kFull512LayerBytes + kAlignment == 4419764224ULL,
      "canonical packed/original/rank ledger differs");
  uint64_t packedBytes = 0, codeBytes = 0, scaleBytes = 0;
  for (uint32_t projection = 0; projection < 3; ++projection) {
    const auto shape = layout(geometry(projection)); packedBytes += shape.ids + shape.lut + shape.scales;
    codeBytes += shape.codes; scaleBytes += shape.scales;
  }
  require(packedBytes == kPackedLayerBytes && codeBytes == kCoefficientBytes && scaleBytes == kScaleBytes, "canonical plane sizes differ");
  const Geometry small{2, 2, 128}; const auto sizes = layout(small);
  std::vector<int8_t> saved(size_t(sizes.codes)), lut(size_t(sizes.lut));
  std::vector<uint8_t> ids(size_t(sizes.ids));
  std::vector<float> scales{std::bit_cast<float>(0x3f800001u), std::bit_cast<float>(0x00000001u),
      std::bit_cast<float>(0x00800001u), std::bit_cast<float>(0x7f7fffffu)}, copied = scales;
  for (uint64_t row = 0; row < sizes.rows; ++row) for (uint64_t group = 0; group < sizes.groups; ++group) {
    const uint64_t table = (row * sizes.groups + group) * 16;
    for (uint32_t symbol = 0; symbol < 16; ++symbol)
      lut[table + symbol] = symbol == 0 ? -127 : (symbol == 15 ? 127 : int8_t(int(symbol) * 7 - 60 + int(row + group)));
    for (uint64_t pair = 0; pair < 32; ++pair) {
      const uint8_t low = uint8_t((pair + row + group) % 16), high = uint8_t((15 - pair % 16 + row + group) % 16);
      ids[row * 64 + group * 32 + pair] = low | (high << 4);
      saved[row * 128 + group * 64 + pair * 2] = lut[table + low];
      saved[row * 128 + group * 64 + pair * 2 + 1] = lut[table + high];
    }
  }
  const auto verify = [&] {return verifyProjection(small, saved, ids, lut, scales, copied);};
  require(verify() == 512, "small complete coefficient reconstruction failed");
  ids.back() ^= 0x10; mustFail([&] {(void)verify();}); ids.back() ^= 0x10;
  const auto prior = lut.back(); lut.back() = -128; mustFail([&] {(void)verify();}); lut.back() = prior;
  const auto last = saved.back(); saved.back() = -128; mustFail([&] {(void)verify();}); saved.back() = last;
  copied[0] = std::bit_cast<float>(std::bit_cast<uint32_t>(scales[0]) + 1); mustFail([&] {(void)verify();}); copied[0] = scales[0];
  for (const auto bits : {0u, 0x80000000u, 0xbf800000u, 0x7fc00001u, 0x7f800000u}) {
    const auto value = scales[0]; scales[0] = copied[0] = std::bit_cast<float>(bits);
    mustFail([&] {(void)verify();}); scales[0] = copied[0] = value;
  }
  mustFail([&] {(void)verifyProjection(small, {saved.data(), saved.size() - 1}, ids, lut, scales, copied);});
  mustFail([&] {(void)verifyProjection(small, saved, {ids.data(), ids.size() - 1}, lut, scales, copied);});
  mustFail([&] {(void)verifyProjection(small, saved, ids, {lut.data(), lut.size() - 1}, scales, copied);});
  mustFail([&] {(void)verifyProjection(small, saved, ids, lut, {scales.data(), scales.size() - 1}, copied);});
  mustFail([&] {(void)verifyProjection(small, saved, ids, lut, scales, {copied.data(), copied.size() - 1});});
  std::vector<uint32_t> ranks(size_t(kAlignment / 4), UINT32_MAX);
  for (uint32_t id = 0; id < 512; ++id) ranks[id] = id;
  requireCanonicalRanks(ranks); ranks[511] = 510; mustFail([&] {requireCanonicalRanks(ranks);}); ranks[511] = 511;
  ranks.back() = 0; mustFail([&] {requireCanonicalRanks(ranks);}); ranks.back() = UINT32_MAX;
  mustFail([&] {requireCanonicalRanks({ranks.data(), ranks.size() - 1});});
  for (const auto g : {Geometry{0, 2, 128}, Geometry{2, 0, 128}, Geometry{2, 2, 0},
      Geometry{2, 2, 63}, Geometry{2, 2, 65}, Geometry{UINT64_MAX, 2, 128}, Geometry{2, UINT64_MAX, 128}})
    mustFail([&] {(void)layout(g);});
  mustFail([] {(void)rounded(UINT64_MAX);}); mustFail([] {(void)checkedAdd(UINT64_MAX, 1);});
  for (const auto &variant : variants) {
    const auto expected = "m32_n64_k" + std::to_string(variant.k) + "_sg" + std::to_string(variant.sg) +
        (variant.stagedI8 ? "_i8" : "_bf16");
    require(expected == variant.suffix && pipelineName(variant, true) != pipelineName(variant, false), "eight-candidate names differ");
  }
  NSDictionary *record = @{@"dtype": @"U8", @"shape": @[@2, @2, @64], @"offset": @0, @"length": @256,
      @"sha256": @"0000000000000000000000000000000000000000000000000000000000000000"};
  uint64_t cursor = 0; (void)detail::plane(record, "U8", {2, 2, 64}, 256, cursor, 256);
  mustFail([&] {uint64_t c = 1; (void)detail::plane(record, "U8", {2, 2, 64}, 256, c, 256);});
  mustFail([&] {uint64_t c = 0; (void)detail::plane(record, "I8", {2, 2, 64}, 256, c, 256);});
  mustFail([&] {uint64_t c = 0; (void)detail::plane(record, "U8", {2, 2, 63}, 256, c, 256);});
  mustFail([&] {uint64_t c = 0; (void)detail::plane(record, "U8", {2, 2, 64}, 255, c, 256);});
  mustFail([&] {uint64_t c = 0; (void)detail::plane(record, "U8", {2, 2, 64}, 256, c, 255);});
  mustFail([] {(void)detail::integer(@YES);}); mustFail([] {(void)detail::integer(@(-1));});
  mustFail([] {(void)detail::integer(@1.5);});
  const unichar nulText[] = {'l', 'a', 'y', 'e', 'r', 0, 'x'};
  NSString *embeddedNUL = [NSString stringWithCharacters:nulText length:7];
  mustFail([&] {(void)detail::string(embeddedNUL);});
}
} // namespace splash::bench::i8_lut
