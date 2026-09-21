#include "flash/FlashWeights.hpp"
#include "flash/FlashOriginalResidency.hpp"
#include "engine/Json.hpp"
#include "PrivateStorage.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
constexpr std::string_view kSchema = "splash-local-qwen4-affine-v1";
using PrivateClock = std::chrono::steady_clock;
thread_local metal::AllocationAdmission privateAdmission;
thread_local PrivateOwnedOriginalStorage privateStorage;

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error("Flash weights: " + std::string(message));
}

NSString *ns(std::string_view value) {
  NSString *result = [[NSString alloc] initWithBytes:value.data()
                                            length:value.size()
                                          encoding:NSUTF8StringEncoding];
  if (!result) fail("invalid UTF-8 metadata");
  return result;
}

std::string stringValue(id value) {
  if (![value isKindOfClass:[NSString class]]) fail("expected a string field");
  NSString *text = value;
  const char *bytes = text.UTF8String;
  if (!bytes) fail("invalid UTF-8 string field");
  std::string result(bytes, [text lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
  if (result.find('\0') != std::string::npos) fail("embedded NUL in metadata");
  return result;
}

NSDictionary *object(id value) {
  if (![value isKindOfClass:[NSDictionary class]]) fail("expected an object field");
  return value;
}

NSArray *array(id value) {
  if (![value isKindOfClass:[NSArray class]]) fail("expected an array field");
  return value;
}

uint64_t integer(id value) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    fail("expected an integer field");
  NSNumber *number = value;
  const char type = number.objCType[0];
  if (std::string_view("cCsSiIlLqQ").find(type) == std::string_view::npos ||
      (std::string_view("csilq").find(type) != std::string_view::npos && number.longLongValue < 0))
    fail("integer field is fractional or negative");
  return number.unsignedLongLongValue;
}

uint32_t narrow(uint64_t value) {
  if (!value || value > std::numeric_limits<uint32_t>::max()) fail("dimension exceeds uint32 range");
  return static_cast<uint32_t>(value);
}

uint64_t multiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left) fail("byte count overflows");
  return left * right;
}

uint64_t aligned(uint64_t value) {
  if (value > std::numeric_limits<uint64_t>::max() - (kAlignment - 1)) fail("alignment overflows");
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

void checkDigest(std::string_view digest) {
  if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    fail("invalid lowercase SHA256 field");
}

std::string hash(std::span<const uint8_t> bytes) {
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) fail("SHA256 initialization failed");
  while (!bytes.empty()) {
    const auto count = static_cast<CC_LONG>(std::min<size_t>(bytes.size(), 1ULL << 30));
    if (!CC_SHA256_Update(&context, bytes.data(), count)) fail("SHA256 update failed");
    bytes = bytes.subspan(count);
  }
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  if (!CC_SHA256_Final(output.data(), &context)) fail("SHA256 finalization failed");
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : output) { result += digits[byte >> 4]; result += digits[byte & 15]; }
  return result;
}

NSData *readSmall(const std::filesystem::path &path) {
  struct stat status{};
  if (::stat(path.c_str(), &status) || !S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<uint64_t>(status.st_size) > (32ULL << 20))
    fail("JSON/metadata file exceeds the bounded startup read");
  NSError *error = nil;
  NSData *data = [NSData dataWithContentsOfFile:ns(path.string()) options:0 error:&error];
  if (!data || data.length > (32ULL << 20)) fail("unable to read bounded JSON/metadata file");
  return data;
}

NSDictionary *parse(NSData *data) {
  NSError *error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  if (!value || error) fail("invalid JSON metadata");
  return object(value);
}

std::span<const uint8_t> bytes(NSData *data) {
  return {static_cast<const uint8_t *>(data.bytes), data.length};
}

bool within(const std::filesystem::path &path, const std::filesystem::path &root) {
  auto next = path.begin();
  for (const auto &part : root) { if (next == path.end() || *next++ != part) return false; }
  return true;
}

std::filesystem::path relativeFile(const std::filesystem::path &root, std::string_view name) {
  const std::filesystem::path relative(name);
  if (relative.empty() || relative.is_absolute()) fail("payload path is not relative");
  for (const auto &part : relative) if (part == "." || part == "..") fail("noncanonical relative path");
  const auto result = std::filesystem::canonical(root / relative);
  if (!within(result, root) || !std::filesystem::is_regular_file(result)) fail("file escapes the derived directory");
  return result;
}

class Mapping final {
public:
  static std::shared_ptr<Mapping> open(const std::filesystem::path &path, uint64_t expectedBytes) {
    const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) fail("unable to open readonly derived payload");
    struct stat status{};
    if (::fstat(file, &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
        static_cast<uint64_t>(status.st_size) != expectedBytes || expectedBytes % kAlignment ||
        expectedBytes > std::numeric_limits<size_t>::max()) {
      ::close(file); fail("derived payload byte count/alignment mismatch");
    }
    void *address = ::mmap(nullptr, expectedBytes, PROT_READ, MAP_SHARED, file, 0);
    ::close(file);
    if (address == MAP_FAILED) fail("readonly derived payload mmap failed");
    Mapping *owner = nullptr;
    try {
      owner = new Mapping(address, expectedBytes);
    } catch (...) {
      ::munmap(address, expectedBytes);
      throw;
    }
    // shared_ptr's pointer constructor deletes owner if its control-block
    // allocation fails; Mapping then releases mmap exactly once.
    return std::shared_ptr<Mapping>(owner);
  }
  ~Mapping() { if (address_) ::munmap(address_, bytes_); }
  void *address() const noexcept { return address_; }
  uint64_t size() const noexcept { return bytes_; }
private:
  Mapping(void *address, uint64_t count) : address_(address), bytes_(count) {}
  void *address_;
  uint64_t bytes_;
};

FlashDType dtype(std::string_view name) {
  if (name == "U32") return FlashDType::U32;
  if (name == "BF16") return FlashDType::BF16;
  if (name == "I64") return FlashDType::I64;
  if (name == "F32") return FlashDType::F32;
  fail("unsupported tensor dtype");
}

uint64_t typeBytes(FlashDType type) {
  switch (type) { case FlashDType::BF16: return 2; case FlashDType::I64: return 8;
  case FlashDType::U32: case FlashDType::F32: return 4; }
  fail("unsupported tensor dtype");
}

float floating(const FlashTensor &tensor, size_t index) {
  if (index >= tensor.logicalBytes / typeBytes(tensor.dtype)) fail("float tensor index out of bounds");
  const auto *base = static_cast<const uint8_t *>(tensor.buffer.contents());
  if (!base) fail("weight tensor is not shared");
  if (tensor.dtype == FlashDType::BF16) {
    uint16_t value; std::memcpy(&value, base + index * 2, 2);
    return std::bit_cast<float>(uint32_t{value} << 16);
  }
  if (tensor.dtype == FlashDType::F32) { float value; std::memcpy(&value, base + index * 4, 4); return value; }
  fail("expected a floating tensor");
}

template <size_t Size>
std::array<int64_t, Size> intArray(const FlashTensor &tensor) {
  if (tensor.dtype != FlashDType::I64 || tensor.shape != std::vector<uint64_t>{Size} ||
      tensor.logicalBytes != Size * sizeof(int64_t) || !tensor.buffer.contents())
    fail("stored PLE I64 array has incorrect dtype/shape");
  std::array<int64_t, Size> result{};
  std::memcpy(result.data(), tensor.buffer.contents(), sizeof(result));
  return result;
}

bool qwenNorm(std::string_view name) {
  return name.ends_with(".hc_norm.weight") || name.ends_with(".self_attn.q_norm.weight") ||
      name.ends_with(".self_attn.k_norm.weight") || name.ends_with(".indexer.q_layernorm.weight") ||
      name.ends_with(".indexer.k_layernorm.weight") || name.ends_with(".ple.norm_key.weight") ||
      name.ends_with(".ple.norm_query.weight") || name.ends_with(".ple.norm_conv.weight") ||
      name == "mtp.pre_fc_norm_embedding.weight" || name == "mtp.pre_fc_norm_hidden.weight";
}

// Python's source-record identity uses ensure_ascii=True. Encode UTF16 units
// for non-ASCII filenames so the native verifier has exactly the same bytes.
std::string asciiQuote(std::string_view value) {
  NSString *text = ns(value);
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (NSUInteger index = 0; index < text.length; ++index) {
    const unichar character = [text characterAtIndex:index];
    if (character < 128) {
      const std::string escaped = json::quote(std::string(1, static_cast<char>(character)));
      result.append(escaped, 1, escaped.size() - 2);
    } else {
      result += "\\u";
      for (int shift = 12; shift >= 0; shift -= 4) result += hex[(character >> shift) & 15];
    }
  }
  result += '"'; return result;
}

} // namespace

bool privateOwnedOriginalRequested() {
  const char *mode = std::getenv("SPLASH_FLASH_PRIVATE_OWNED_ORIGINAL");
  if (mode && std::string_view(mode) != "0" && std::string_view(mode) != "1")
    fail("SPLASH_FLASH_PRIVATE_OWNED_ORIGINAL must be 0 or 1");
  return mode && std::string_view(mode) == "1";
}

void privateOwnedOriginalSetAdmission(metal::AllocationAdmission admission) {
  privateAdmission = std::move(admission);
}

PrivateOwnedOriginalStorage privateOwnedOriginalStorage() noexcept {
  return privateStorage;
}

struct FlashWeights::Impl final {
  FlashDescriptor descriptor;
  FlashNormAudit norm;
  std::string fingerprint;
  std::string sourceIdentity;
  uint64_t allocatedBytes = 0;
  std::map<std::string, FlashTensor, std::less<>> tensors;
  std::map<std::string, FlashAffineProjection, std::less<>> projections;
  std::vector<metal::MetalBuffer> bases;
  FlashOriginalTextResidencySelection originalTextResidency;
};

FlashWeights::FlashWeights() = default;
FlashWeights::~FlashWeights() = default;
FlashWeights::FlashWeights(FlashWeights &&) noexcept = default;
FlashWeights &FlashWeights::operator=(FlashWeights &&) noexcept = default;

FlashWeights FlashWeights::load(metal::MetalBackend &backend,
                               const std::filesystem::path &directory,
                               bool verifyPayloadHashes) {
  @autoreleasepool {
    const auto loadBegan = PrivateClock::now();
    const bool owned = privateOwnedOriginalRequested();
    if (owned && !privateAdmission)
      fail("private owned original storage requires startup allocation admission");
    privateStorage = {};
    privateStorage.enabled = owned;
    privateStorage.verifiedPayloadHashes = verifyPayloadHashes;
    const auto root = std::filesystem::canonical(directory);
    if (!std::filesystem::is_directory(root)) fail("derived root is not a directory");
    NSData *manifestBytes = readSmall(relativeFile(root, "manifest.json"));
    const std::string manifestDigest = hash(bytes(manifestBytes));
    NSData *checksumBytes = readSmall(relativeFile(root, "manifest.sha256"));
    const std::string checksum(static_cast<const char *>(checksumBytes.bytes), checksumBytes.length);
    if (checksum.size() != 65 || checksum.back() != '\n' ||
        checksum.substr(0, 64) != manifestDigest) fail("manifest byte checksum mismatch");
    NSDictionary *manifest = parse(manifestBytes);
    if (stringValue(manifest[@"schema"]) != kSchema || integer(manifest[@"alignment"]) != kAlignment)
      fail("unsupported aligned affine manifest schema");
    const std::string sourceIdentity = stringValue(manifest[@"source_identity_sha256"]);
    checkDigest(sourceIdentity);

    struct SourceRecord { uint64_t bytes; std::string sha; };
    std::map<std::string, SourceRecord, std::less<>> sourceRecords;
    const auto sourceRecord = [&](const std::string &path, uint64_t count, const std::string &sha) {
      checkDigest(sha);
      if (!count || !sourceRecords.emplace(path, SourceRecord{count, sha}).second)
        fail("duplicate or empty source identity record");
    };
    std::map<std::string, std::string, std::less<>> smallHashes;
    for (id value in array(manifest[@"small_files"])) {
      NSDictionary *record = object(value);
      const std::string name = stringValue(record[@"path"]);
      const uint64_t count = integer(record[@"bytes"]);
      const std::string sha = stringValue(record[@"sha256"]);
      NSData *data = readSmall(relativeFile(root, name));
      if (data.length != count || hash(bytes(data)) != sha) fail("copied small-file checksum mismatch");
      sourceRecord(name, count, sha); smallHashes.emplace(name, sha);
    }
    const std::string configName = stringValue(manifest[@"config"]);
    const std::string indexName = stringValue(manifest[@"index"]);
    if (!smallHashes.contains(configName) || !smallHashes.contains(indexName))
      fail("config/index missing from checked small-file identities");
    NSDictionary *config = parse(readSmall(relativeFile(root, configName)));
    NSDictionary *quantization = object(manifest[@"quantization"]);
    NSDictionary *configQuantization = object(config[@"quantization"] ?: config[@"quantization_config"]);
    if (![quantization isEqualToDictionary:configQuantization]) fail("manifest/config quantization mismatch");
    NSDictionary *index = parse(readSmall(relativeFile(root, indexName)));
    NSDictionary *weightMap = object(index[@"weight_map"]);

    FlashWeights result;
    result.impl_ = std::make_unique<Impl>();
    Impl &impl = *result.impl_;
    impl.descriptor = FlashDescriptor::fromConfig(relativeFile(root, configName));
    impl.sourceIdentity = sourceIdentity;
    const uint64_t before = backend.memoryStats().allocatedBytes;
    struct Shard {
      metal::MetalBuffer base;
      std::shared_ptr<Mapping> mapping;
      std::string source;
      uint64_t sourceBytes;
      uint64_t backingBytes;
      std::vector<std::pair<uint64_t, uint64_t>> ranges;
      uint32_t residencyCategories = 0;
    };
    std::map<std::string, Shard, std::less<>> shards;
    for (id value in array(manifest[@"shards"])) {
      NSDictionary *record = object(value);
      const std::string name = stringValue(record[@"path"]);
      const std::string source = stringValue(record[@"source_path"]);
      const uint64_t count = integer(record[@"bytes"]);
      const uint64_t sourceCount = integer(record[@"source_bytes"]);
      sourceRecord(source, sourceCount, stringValue(record[@"source_sha256"]));
      const std::string expectedHash = stringValue(record[@"sha256"]); checkDigest(expectedHash);
      if (!count || count > backend.capabilities().maxBufferLengthBytes) fail("derived shard exceeds Metal buffer limit");
      std::shared_ptr<Mapping> mapping;
      metal::MetalBuffer base;
      const auto verifyMapping = [&] {
        if (!verifyPayloadHashes) return;
        const auto began = PrivateClock::now();
        const std::string digest = hash({static_cast<const uint8_t *>(mapping->address()),
                                        static_cast<size_t>(count)});
        privateStorage.payloadHashSeconds +=
            std::chrono::duration<double>(PrivateClock::now() - began).count();
        if (digest != expectedHash) fail("derived shard checksum mismatch");
      };
      const auto releaseOwnedMapping = [&] {
        if (!mapping) return;
        mapping.reset();
        --privateStorage.activeMappings;
        ++privateStorage.releasedMappingCount;
      };
      if (owned) {
        // The source mapping is never wrapped by Metal. Hold admission for
        // destination plus worst-case source pages, and destroy the source
        // before committing the reservation or moving to the next shard.
        const uint64_t admissionBytes = multiply(count, 2);
        privateStorage.maximumAdmissionBytes =
            std::max(privateStorage.maximumAdmissionBytes, admissionBytes);
        const auto outcome = privateAdmission(admissionBytes, [&] {
          try {
          mapping = Mapping::open(relativeFile(root, name), count);
          ++privateStorage.openedMappingCount;
          ++privateStorage.activeMappings;
          privateStorage.maximumActiveMappings =
              std::max(privateStorage.maximumActiveMappings, privateStorage.activeMappings);
          privateStorage.temporaryMappingPeakBytes =
              std::max(privateStorage.temporaryMappingPeakBytes, count);
          base = backend.allocateBuffer(count, metal::BufferStorage::Shared,
                                        "private-owned-original:" + name);
          if (!base.contents()) fail("owned original allocation is not CPU Shared");
          const auto copyBegan = PrivateClock::now();
          std::memcpy(base.contents(), mapping->address(), static_cast<size_t>(count));
          privateStorage.copySeconds +=
              std::chrono::duration<double>(PrivateClock::now() - copyBegan).count();
          privateStorage.copiedBytes += count;
          const auto compareBegan = PrivateClock::now();
          const bool equal = std::memcmp(base.contents(), mapping->address(), static_cast<size_t>(count)) == 0;
          privateStorage.comparisonSeconds +=
              std::chrono::duration<double>(PrivateClock::now() - compareBegan).count();
          if (!equal) fail("owned original byte copy differs from readonly source");
          privateStorage.verifiedCopyBytes += count;
          verifyMapping();
          releaseOwnedMapping();
          } catch (...) {
            // Clear both transient and destination ownership while admission
            // is still held. This also handles driver denial mid-startup.
            releaseOwnedMapping();
            base = {};
            throw;
          }
        });
        if (!outcome)
          throw metal::MetalAllocationError("private owned original shard admission denied", outcome.failure);
        ++privateStorage.admittedShardCopies;
      } else {
        mapping = Mapping::open(relativeFile(root, name), count);
        ++privateStorage.openedMappingCount;
        ++privateStorage.activeMappings;
        privateStorage.maximumActiveMappings =
            std::max(privateStorage.maximumActiveMappings, privateStorage.activeMappings);
        verifyMapping();
        base = backend.wrapSharedMemory(mapping->address(), count, mapping, name);
      }
      if (!shards.emplace(name, Shard{base, mapping, source, sourceCount, count, {}}).second)
        fail("duplicate derived shard path");
      impl.bases.push_back(std::move(base));
    }
    std::string canonical = "{\"schema\":" + asciiQuote(kSchema) + ",\"source_files\":[";
    bool first = true;
    for (const auto &[path, record] : sourceRecords) {
      if (!first) canonical += ','; first = false;
      canonical += "{\"bytes\":" + std::to_string(record.bytes) + ",\"path\":" + asciiQuote(path) +
                   ",\"sha256\":" + asciiQuote(record.sha) + '}';
    }
    canonical += "]}";
    if (hash({reinterpret_cast<const uint8_t *>(canonical.data()), canonical.size()}) != sourceIdentity)
      fail("canonical source-record identity mismatch");

    NSDictionary *tensorMap = object(manifest[@"tensors"]);
    if (tensorMap.count != weightMap.count || tensorMap.count != 3748 || shards.size() != 21)
      fail("checkpoint tensor/shard inventory is incomplete");
    for (NSString *key in tensorMap) {
      const std::string name = stringValue(key);
      NSDictionary *record = object(tensorMap[key]);
      const std::string shardName = stringValue(record[@"shard"]);
      auto found = shards.find(shardName);
      if (found == shards.end()) fail("tensor references an unknown derived shard");
      Shard &shard = found->second;
      shard.residencyCategories |= flashOriginalResidencyCategory(name);
      const std::string source = stringValue(record[@"source_shard"]);
      const uint64_t sourceOffset = integer(record[@"source_offset"]);
      if (source != shard.source || stringValue(weightMap[key]) != source) fail("source index/tensor shard disagreement");
      const uint64_t offset = integer(record[@"offset"]);
      const uint64_t length = integer(record[@"length"]);
      FlashTensor tensor;
      tensor.dtype = dtype(stringValue(record[@"dtype"]));
      uint64_t count = 1;
      NSArray *shape = array(record[@"shape"]);
      if (shape.count > 8) fail("tensor rank exceeds native route limit");
      for (id dimension in shape) { const uint64_t size = integer(dimension);
        if (!size) fail("tensor contains a zero dimension");
        tensor.shape.push_back(size); count = multiply(count, size); }
      tensor.logicalBytes = multiply(count, typeBytes(tensor.dtype));
      if (length != tensor.logicalBytes || !length || offset % kAlignment ||
          offset > shard.backingBytes || length > shard.backingBytes - offset ||
          sourceOffset < 8 || sourceOffset > shard.sourceBytes || length > shard.sourceBytes - sourceOffset)
        fail("tensor shape/byte range/alignment mismatch");
      tensor.buffer = backend.view(shard.base, offset, length);
      shard.ranges.emplace_back(offset, offset + length);
      if (!impl.tensors.emplace(name, std::move(tensor)).second) fail("duplicate tensor name");
    }
    std::array<uint64_t, 4> dtypeCounts{};
    for (const auto &[name, tensor] : impl.tensors) {
      static_cast<void>(name);
      ++dtypeCounts[static_cast<size_t>(tensor.dtype)];
    }
    if (dtypeCounts != std::array<uint64_t, 4>{1020, 2725, 3, 0})
      fail("checkpoint dtype inventory differs from preserved U32/BF16/I64 storage");
    for (auto &[name, shard] : shards) {
      static_cast<void>(name);
      std::sort(shard.ranges.begin(), shard.ranges.end());
      uint64_t cursor = 0;
      const auto *data = static_cast<const uint8_t *>(shard.base.contents());
      if (!data) fail("immutable shard backing is not Shared");
      for (const auto &[begin, end] : shard.ranges) {
        if (begin != aligned(cursor)) fail("derived tensor packing is not canonical/nonoverlapping");
        if (!std::all_of(data + cursor, data + begin, [](uint8_t value) { return value == 0; }))
          fail("nonzero derived tensor padding");
        cursor = end;
      }
      if (shard.ranges.empty() || aligned(cursor) != shard.backingBytes ||
          !std::all_of(data + cursor, data + shard.backingBytes, [](uint8_t value) { return value == 0; }))
        fail("derived shard tail padding/layout mismatch");
    }
    for (const auto &[name, weight] : impl.tensors) {
      if (weight.dtype != FlashDType::U32) continue;
      if (!name.ends_with(".weight") || (weight.shape.size() != 2 && weight.shape.size() != 3))
        fail("packed affine tensor has unsupported name/rank");
      const std::string prefix = name.substr(0, name.size() - 7);
      const auto sf = impl.tensors.find(prefix + ".scales");
      const auto bias = impl.tensors.find(prefix + ".biases");
      if (sf == impl.tensors.end() || bias == impl.tensors.end() ||
          sf->second.dtype != FlashDType::BF16 || bias->second.dtype != FlashDType::BF16 ||
          sf->second.shape != bias->second.shape || sf->second.shape.size() != weight.shape.size())
        fail("affine projection parameters are missing or inconsistent");
      NSDictionary *override = quantization[ns(prefix)];
      if (override) override = object(override);
      const uint32_t bits = narrow(integer(override[@"bits"] ?: quantization[@"bits"]));
      const uint32_t group = narrow(integer(override[@"group_size"] ?: quantization[@"group_size"]));
      if (stringValue(override[@"mode"] ?: quantization[@"mode"]) != "affine" ||
          (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
          (group != 32 && group != 64 && group != 128)) fail("unsupported affine bits/group/mode");
      const size_t rank = weight.shape.size();
      for (size_t axis = 0; axis + 1 < rank; ++axis)
        if (weight.shape[axis] != sf->second.shape[axis]) fail("affine parameter row/expert shape mismatch");
      const uint64_t k = multiply(sf->second.shape.back(), group);
      if (multiply(k, bits) % 32 || multiply(k, bits) / 32 != weight.shape.back())
        fail("affine packed K dimension disagrees with parameters");
      FlashAffineProjection projection;
      projection.weights = &weight; projection.scales = &sf->second; projection.biases = &bias->second;
      projection.experts = rank == 3 ? narrow(weight.shape[0]) : 1;
      projection.outputSize = narrow(weight.shape[rank - 2]); projection.inputSize = narrow(k);
      projection.bits = bits; projection.groupSize = group;
      projection.weightRowStrideBytes = multiply(weight.shape.back(), 4);
      projection.parameterRowStrideBytes = multiply(sf->second.shape.back(), 2);
      projection.weightExpertStrideBytes = multiply(projection.outputSize, projection.weightRowStrideBytes);
      projection.parameterExpertStrideBytes = multiply(projection.outputSize, projection.parameterRowStrideBytes);
      impl.projections.emplace(prefix, projection);
    }
    if (impl.projections.size() != 1020) fail("packed affine inventory is incomplete");

    const auto requireTensor = [&](const std::string &name, FlashDType expectedType,
                                   std::vector<uint64_t> expectedShape) {
      const auto &value = result.tensor(name);
      if (value.dtype != expectedType || value.shape != expectedShape)
        fail("semantic tensor dtype/geometry mismatch: " + name);
    };
    const auto requireProjection = [&](const std::string &prefix, uint32_t e,
                                       uint32_t n, uint32_t k,
                                       uint32_t expectedBits = 0, uint32_t expectedGroup = 0) {
      const auto &value = result.projection(prefix);
      if (value.experts != e || value.outputSize != n || value.inputSize != k ||
          (expectedBits && value.bits != expectedBits) ||
          (expectedGroup && value.groupSize != expectedGroup))
        fail("semantic projection geometry/format mismatch: " + prefix);
    };
    const auto requireHC = [&](const std::string &prefix, bool injection) {
      requireTensor(prefix + ".hc_norm.weight", FlashDType::BF16, {10240});
      requireProjection(prefix + ".input_mix_weight_down", 1, 320, 10240);
      requireProjection(prefix + ".input_mix_weight_up", 1, 10240, 320);
      if (injection) requireProjection(prefix + ".block_inject_weight", 1, 4, 10240);
    };
    const auto requireLayer = [&](const std::string &prefix, bool linear) {
      requireHC(prefix + ".attn_hyper_connection", true);
      requireHC(prefix + ".mlp_hyper_connection", true);
      if (linear) {
        const auto attention = prefix + ".linear_attn";
        requireProjection(attention + ".in_proj_qkv", 1, 10240, 2560);
        requireProjection(attention + ".in_proj_z", 1, 6144, 2560);
        requireProjection(attention + ".in_proj_a", 1, 48, 2560);
        requireProjection(attention + ".in_proj_b", 1, 48, 2560);
        requireProjection(attention + ".out_proj", 1, 2560, 6144, 5, 128);
        requireTensor(attention + ".conv1d.weight", FlashDType::BF16, {10240, 4, 1});
        requireTensor(attention + ".A_log", FlashDType::BF16, {48});
        requireTensor(attention + ".dt_bias", FlashDType::BF16, {48});
        requireTensor(attention + ".norm.weight", FlashDType::BF16, {128});
      } else {
        const auto attention = prefix + ".self_attn";
        requireProjection(attention + ".q_proj", 1, 12288, 2560);
        requireProjection(attention + ".k_proj", 1, 512, 2560);
        requireProjection(attention + ".v_proj", 1, 512, 2560);
        requireProjection(attention + ".o_proj", 1, 2560, 6144);
        requireProjection(attention + ".indexer.index_qk_proj", 1, 640, 2560);
        requireTensor(attention + ".q_norm.weight", FlashDType::BF16, {256});
        requireTensor(attention + ".k_norm.weight", FlashDType::BF16, {256});
        requireTensor(attention + ".indexer.q_layernorm.weight", FlashDType::BF16, {128});
        requireTensor(attention + ".indexer.k_layernorm.weight", FlashDType::BF16, {128});
      }
      const auto mlp = prefix + ".mlp";
      requireTensor(mlp + ".gate.weight", FlashDType::BF16, {512, 2560});
      requireProjection(mlp + ".switch_mlp.gate_proj", 512, 640, 2560, 4, 64);
      requireProjection(mlp + ".switch_mlp.up_proj", 512, 640, 2560, 4, 64);
      requireProjection(mlp + ".switch_mlp.down_proj", 512, 2560, 640, 4, 64);
      requireProjection(mlp + ".shared_expert.gate_proj", 1, 640, 2560, 8, 128);
      requireProjection(mlp + ".shared_expert.up_proj", 1, 640, 2560, 8, 128);
      requireProjection(mlp + ".shared_expert.down_proj", 1, 2560, 640, 8, 128);
      requireProjection(mlp + ".shared_expert_gate", 1, 1, 2560, 8, 64);
    };
    for (uint32_t layer = 0; layer < impl.descriptor.layers; ++layer)
      requireLayer("language_model.model.layers." + std::to_string(layer),
                   impl.descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet);
    requireHC("language_model.model.hyper_connection_mixer", false);
    requireProjection("language_model.model.embed_tokens", 1, 248320, 2560, 8, 64);
    requireProjection("language_model.lm_head", 1, 248320, 2560, 8, 64);
    requireLayer("mtp.layers.0", false);
    requireHC("mtp.hyper_connection_mixer", false);
    requireProjection("mtp.fc_embedding", 1, 2560, 2560, 4, 64);
    requireProjection("mtp.fc_hidden", 1, 2560, 2560, 4, 64);
    requireTensor("mtp.pre_fc_norm_embedding.weight", FlashDType::BF16, {2560});
    requireTensor("mtp.pre_fc_norm_hidden.weight", FlashDType::BF16, {10240});
    const std::string pleLayer = "language_model.model.layers.1.ple";
    requireProjection(pleLayer + ".key_proj", 1, 10240, 2560);
    requireProjection(pleLayer + ".value_proj", 1, 2560, 2560);
    for (const auto suffix : {".norm_key.weight", ".norm_query.weight", ".norm_conv.weight"})
      requireTensor(pleLayer + suffix, FlashDType::BF16, {10240});
    requireTensor(pleLayer + ".conv1d.weight", FlashDType::BF16, {10240, 4, 1});

    const std::string ple = "language_model.model.layers.1.ple.ple_embedding.";
    impl.descriptor.pleLayerMultipliers = intArray<3>(result.tensor(ple + "layer_multipliers"));
    impl.descriptor.pleHeadVocabularySizes = intArray<16>(result.tensor(ple + "ngram_heads_vocab_sizes"));
    impl.descriptor.pleHeadOffsets = intArray<16>(result.tensor(ple + "ngram_heads_offsets"));
    const auto &sharedScale = result.tensor(ple + "ngram_embedding.weight_scale");
    if (sharedScale.logicalBytes != 2 || sharedScale.dtype != FlashDType::BF16)
      fail("PLE shared weight scale is not one stored BF16 scalar");
    impl.descriptor.pleSharedWeightScale = floating(sharedScale, 0);
    if (!std::isfinite(impl.descriptor.pleSharedWeightScale)) fail("nonfinite PLE shared weight scale");
    uint64_t rows = 0;
    for (size_t index = 0; index < 16; ++index) {
      const int64_t vocabulary = impl.descriptor.pleHeadVocabularySizes[index];
      if (vocabulary < impl.descriptor.pleVocabularyBase ||
          impl.descriptor.pleHeadOffsets[index] < 0 ||
          static_cast<uint64_t>(impl.descriptor.pleHeadOffsets[index]) != rows)
        fail("stored PLE vocabulary/offset arrays are inconsistent");
      if (static_cast<uint64_t>(vocabulary) > std::numeric_limits<uint64_t>::max() - rows)
        fail("stored PLE vocabulary sum overflows");
      rows += static_cast<uint64_t>(vocabulary);
    }
    if (rows > std::numeric_limits<uint64_t>::max() - (impl.descriptor.pleVocabularyAlignment - 1))
      fail("stored PLE vocabulary padding overflows");
    const uint64_t paddedRows = (rows + impl.descriptor.pleVocabularyAlignment - 1) /
        impl.descriptor.pleVocabularyAlignment * impl.descriptor.pleVocabularyAlignment;
    uint64_t tableRows = 0;
    for (uint32_t part = 0; part < impl.descriptor.pleParts; ++part) {
      const auto &projection = result.projection(ple + "ngram_embedding.shards." + std::to_string(part));
      if (projection.experts != 1 || projection.bits != 4 || projection.groupSize != 32 ||
          projection.inputSize != impl.descriptor.pleHeadDimension() ||
          projection.outputSize != paddedRows / impl.descriptor.pleParts)
        fail("PLE shard affine layout mismatch");
      tableRows += projection.outputSize;
    }
    if (tableRows != paddedRows) fail("PLE shard vocabulary does not match stored head parameters");
    for (int64_t multiplier : impl.descriptor.pleLayerMultipliers)
      if (multiplier <= 0 || !(multiplier & 1) ||
          multiplier > std::numeric_limits<int64_t>::max() / impl.descriptor.vocabularySize)
        fail("PLE hash multiplier exceeds checked signed token arithmetic");
    impl.descriptor.pleTableRows = tableRows;
    impl.descriptor.pleParametersLoaded = true;
    impl.descriptor.validate();

    std::vector<float> means;
    uint32_t ones = 0;
    for (uint32_t layer = 0; layer < impl.descriptor.layers; ++layer) {
      const auto &anchor = result.tensor("language_model.model.layers." + std::to_string(layer) +
                                         ".attn_hyper_connection.hc_norm.weight");
      if (anchor.shape != std::vector<uint64_t>{impl.descriptor.hyperHiddenSize()} ||
          (anchor.dtype != FlashDType::BF16 && anchor.dtype != FlashDType::F32)) fail("HC norm anchor shape/dtype mismatch");
      float sum = 0.0F;
      for (uint32_t index = 0; index < impl.descriptor.hyperHiddenSize(); ++index) {
        const float value = floating(anchor, index);
        if (!std::isfinite(value)) fail("nonfinite HC norm anchor");
        sum += value;
      }
      const float mean = sum / impl.descriptor.hyperHiddenSize();
      means.push_back(mean); if (mean > 0.5F) ++ones;
    }
    std::sort(means.begin(), means.end());
    impl.norm.anchors = static_cast<uint32_t>(means.size());
    impl.norm.medianMean = (means[means.size() / 2 - 1] + means[means.size() / 2]) * 0.5F;
    impl.norm.onesCenteredFraction = static_cast<double>(ones) / means.size();
    if (impl.norm.onesCenteredFraction >= 0.9 && impl.norm.medianMean >= 0.75 && impl.norm.medianMean <= 1.5)
      impl.norm.convention = NormConvention::DirectGamma;
    else if (impl.norm.onesCenteredFraction <= 0.1 && impl.norm.medianMean >= -0.5 && impl.norm.medianMean <= 0.25)
      impl.norm.convention = NormConvention::OnePlusWeight;
    else fail("checkpoint norm convention is ambiguous; refusing unqualified native execution");
    // Source records identify the original checkpoint; the verified derived
    // manifest also binds names to actual payload offsets/layouts. Include both
    // so a changed tensor binding cannot reuse another numerical model's state.
    const std::string effectiveIdentity = "splash.native-flash-weights-v1\nsource=" +
        sourceIdentity + "\nmanifest=" + manifestDigest + "\nnorm=" +
        (impl.norm.convention == NormConvention::OnePlusWeight ? "one-plus-weight\n" : "direct-gamma\n");
    impl.fingerprint = hash({reinterpret_cast<const uint8_t *>(effectiveIdentity.data()),
                             effectiveIdentity.size()});
    // Residency is an allocation property. Classify every tensor first, then
    // omit an entire base if any view is PLE, vision or an unknown family.
    // Reusing these existing buffers requests no second model allocation.
    for (id value in array(manifest[@"shards"])) {
      const std::string path = stringValue(object(value)[@"path"]);
      const auto &shard = shards.at(path);
      auto &selection = impl.originalTextResidency;
      if (flashOriginalResidencyBaseEligible(shard.residencyCategories)) {
        if (selection.mappedBytes > UINT64_MAX - shard.base.sizeBytes()) fail("original residency byte sum overflows");
        selection.mappedBytes += shard.base.sizeBytes();
        selection.buffers.push_back(shard.base); selection.paths.push_back(path);
      } else {
        ++selection.excludedBaseCount;
        if (selection.excludedMappedBytes > UINT64_MAX - shard.base.sizeBytes()) fail("excluded residency byte sum overflows");
        selection.excludedMappedBytes += shard.base.sizeBytes();
        if (shard.residencyCategories & static_cast<uint32_t>(OriginalResidencyCategory::PLE)) ++selection.excludedPLEBaseCount;
        if (shard.residencyCategories & static_cast<uint32_t>(OriginalResidencyCategory::Vision)) ++selection.excludedVisionBaseCount;
        if (shard.residencyCategories & static_cast<uint32_t>(OriginalResidencyCategory::Unknown)) ++selection.excludedUnknownBaseCount;
      }
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) fail("allocation ledger regressed during load");
    impl.allocatedBytes = after - before;
    privateStorage.nativeBaseCount = impl.bases.size();
    privateStorage.loadSeconds =
        std::chrono::duration<double>(PrivateClock::now() - loadBegan).count();
    if (owned && (privateStorage.activeMappings || privateStorage.maximumActiveMappings != 1 ||
                  privateStorage.openedMappingCount != 21 || privateStorage.releasedMappingCount != 21 ||
                  privateStorage.admittedShardCopies != 21 ||
                  privateStorage.copiedBytes != 106320429056ULL ||
                  privateStorage.verifiedCopyBytes != privateStorage.copiedBytes))
      fail("private owned original copy/lifetime inventory is incomplete");
    return result;
  }
}

const FlashDescriptor &FlashWeights::descriptor() const {
  if (!impl_) fail("weights are not loaded"); return impl_->descriptor;
}
const FlashTensor &FlashWeights::tensor(std::string_view name) const {
  if (!impl_) fail("weights are not loaded");
  const auto found = impl_->tensors.find(name);
  if (found == impl_->tensors.end()) fail("missing tensor: " + std::string(name)); return found->second;
}
const FlashAffineProjection &FlashWeights::projection(std::string_view prefix) const {
  if (!impl_) fail("weights are not loaded");
  const auto found = impl_->projections.find(prefix);
  if (found == impl_->projections.end()) fail("missing affine projection: " + std::string(prefix)); return found->second;
}
bool FlashWeights::contains(std::string_view name) const noexcept {
  return impl_ && impl_->tensors.contains(name);
}
std::vector<metal::MetalBuffer> FlashWeights::immutableWeightBuffers() const {
  if (!impl_) fail("weights are not loaded"); return impl_->bases;
}
FlashOriginalTextResidencySelection FlashWeights::checkedOriginalTextResidency() const {
  if (!impl_) fail("weights are not loaded");
  constexpr std::string_view source = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
  constexpr std::string_view layout = "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";
  const auto &selection = impl_->originalTextResidency;
  if (impl_->sourceIdentity != source || impl_->fingerprint != layout ||
      impl_->descriptor.layers != 48 || impl_->descriptor.experts != 512 || impl_->descriptor.hiddenSize != 2560 ||
      selection.buffers.size() != 13 || selection.paths.size() != 13 || selection.mappedBytes != 66060288000ULL ||
      selection.excludedBaseCount != 8 || selection.excludedPLEBaseCount != 8 || selection.excludedVisionBaseCount != 1 ||
      selection.excludedUnknownBaseCount != 0)
    fail("original text residency requires the qualified 13 pure-text bases and source/layout geometry");
  for (size_t index = 0; index < selection.buffers.size(); ++index) {
    const auto &buffer = selection.buffers[index];
    if (!buffer || buffer.storage() != metal::BufferStorage::Shared || !buffer.contents() || !buffer.sizeBytes())
      fail("original text residency contains an invalid readonly Shared base");
  }
  return selection;
}
const std::string &FlashWeights::manifestFingerprint() const {
  if (!impl_) fail("weights are not loaded"); return impl_->fingerprint;
}
const std::string &FlashWeights::sourceIdentity() const {
  if (!impl_) fail("weights are not loaded"); return impl_->sourceIdentity;
}
const FlashNormAudit &FlashWeights::normAudit() const {
  if (!impl_) fail("weights are not loaded"); return impl_->norm;
}
NormConvention FlashWeights::normConvention() const { return normAudit().convention; }
NormConvention FlashWeights::normConvention(std::string_view name) const {
  static_cast<void>(tensor(name)); return qwenNorm(name) ? normConvention() : NormConvention::DirectGamma;
}
uint64_t FlashWeights::tensorCount() const noexcept { return impl_ ? impl_->tensors.size() : 0; }
uint64_t FlashWeights::actualAllocatedBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }

} // namespace splash::flash
