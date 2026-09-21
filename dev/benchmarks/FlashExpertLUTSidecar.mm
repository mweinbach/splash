#include "FlashExpertLUTSidecar.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <fcntl.h>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash::candidate {
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(std::string("Expert LUT sidecar: ") + message);
}
NSString *ns(const std::filesystem::path &path) {
  NSString *value = [NSString stringWithUTF8String:path.c_str()];
  require(value != nil, "path is not UTF8"); return value;
}
std::string string(id value) {
  require([value isKindOfClass:[NSString class]], "expected string metadata");
  const char *bytes = [value UTF8String];
  require(bytes != nullptr, "invalid UTF8 metadata"); return bytes;
}
uint64_t integer(id value) {
  require([value isKindOfClass:[NSNumber class]] &&
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
      "expected integer metadata; boolean is invalid");
  NSNumber *number = value;
  const char type = number.objCType[0];
  require(std::string_view("cCsSiIlLqQ").find(type) != std::string_view::npos,
      "fractional metadata is invalid");
  require(std::string_view("csilq").find(type) == std::string_view::npos ||
      number.longLongValue >= 0, "negative metadata is invalid");
  return number.unsignedLongLongValue;
}
std::string hash(std::span<const uint8_t> bytes) {
  CC_SHA256_CTX context{}; require(CC_SHA256_Init(&context), "SHA initialization failed");
  while (!bytes.empty()) {
    const auto count = static_cast<CC_LONG>(std::min<size_t>(bytes.size(), 1ULL << 30));
    require(CC_SHA256_Update(&context, bytes.data(), count), "SHA update failed");
    bytes = bytes.subspan(count);
  }
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
  require(CC_SHA256_Final(digest.data(), &context), "SHA final failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t b : digest) { result += hex[b >> 4]; result += hex[b & 15]; }
  return result;
}
NSData *smallFile(const std::filesystem::path &path) {
  require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) <= 32 * 1024 * 1024,
      "invalid bounded metadata file");
  NSData *data = [NSData dataWithContentsOfFile:ns(path)];
  require(data != nil, "metadata read failed"); return data;
}
NSDictionary *object(NSData *data) {
  NSError *error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require([value isKindOfClass:[NSDictionary class]] && !error, "invalid JSON object");
  return value;
}
std::span<const uint8_t> bytes(NSData *data) {
  return {static_cast<const uint8_t *>(data.bytes), data.length};
}
} // namespace

struct ExpertLUTSidecar::Plane {
  void *address = nullptr;
  uint64_t count = 0;
  ~Plane() { if (address) ::munmap(address, count); }
};

ExpertLUTSidecar ExpertLUTSidecar::load(const std::filesystem::path &directory,
    const std::filesystem::path &sourcePackage, const std::string &expectedPrefix) {
  const auto root = std::filesystem::canonical(directory);
  const auto sourceRoot = std::filesystem::canonical(sourcePackage);
  NSData *metadata = smallFile(root / "manifest.json");
  NSDictionary *manifest = object(metadata);
  ExpertLUTSidecar result;
  result.manifestSHA_ = hash(bytes(metadata));
  std::ifstream checksumFile(root / "manifest.sha256");
  std::string checksum; checksumFile >> checksum;
  require(checksum == result.manifestSHA_, "manifest byte hash differs");
  require(string(manifest[@"schema"]) == "splash-flash-exact-expert-bf16-lut-v1" &&
      string(manifest[@"layout"]) == kExactExpertLUTLayout &&
      string(manifest[@"coefficient_policy"]) == kExactExpertLUTPolicy,
      "schema, layout or coefficient policy differs");
  require(string(manifest[@"prefix"]) == expectedPrefix &&
      std::filesystem::canonical(string(manifest[@"source_package"])) == sourceRoot,
      "requested prefix/package differs");
  require(manifest[@"source_codes_reused"] == (__bridge id)kCFBooleanTrue &&
      manifest[@"original_checkpoint_modified"] == (__bridge id)kCFBooleanFalse,
      "source code/immutability contract differs");
  require(integer(manifest[@"alignment"]) == 16384 &&
      integer(manifest[@"saved_lut_bytes"]) == result.savedBytes() &&
      integer(manifest[@"all_48_layers_lut_bytes"]) == 48 * result.savedBytes(),
      "saved allocation geometry differs");
  NSData *sourceManifest = smallFile(sourceRoot / "manifest.json");
  NSDictionary *source = object(sourceManifest);
  result.sourceIdentity_ = string(manifest[@"source_identity_sha256"]);
  require(string(source[@"source_identity_sha256"]) == result.sourceIdentity_ &&
      string(manifest[@"source_manifest_sha256"]) == hash(bytes(sourceManifest)),
      "immutable source identity differs");
  id payloadValue = manifest[@"payloads"];
  require([payloadValue isKindOfClass:[NSArray class]] && [payloadValue count] == 3,
      "three payloads are required");
  NSArray *payloads = payloadValue;
  const std::array<std::string, 3> names{"gate_proj", "up_proj", "down_proj"};
  for (uint32_t index = 0; index < 3; ++index) {
    id raw = payloads[index];
    require([raw isKindOfClass:[NSDictionary class]], "invalid payload metadata");
    NSDictionary *payload = raw;
    require(string(payload[@"projection"]) == names[index] &&
        string(payload[@"prefix"]) == expectedPrefix + "." + names[index] &&
        string(payload[@"path"]) == names[index] + ".bf16" &&
        string(payload[@"dtype"]) == "BF16" && integer(payload[@"bytes"]) == 419430400 &&
        integer(payload[@"groups_checked"]) == 13107200 &&
        integer(payload[@"coefficients_checked"]) == 209715200,
        "projection/dtype/byte/verification extent differs");
    id shapeValue = payload[@"shape"];
    require([shapeValue isKindOfClass:[NSArray class]] && [shapeValue count] == 5,
        "invalid tiled shape rank");
    NSArray *shape = shapeValue;
    const std::array<uint64_t, 5> dimensions = index == 2
        ? std::array<uint64_t, 5>{512, 40, 10, 64, 16}
        : std::array<uint64_t, 5>{512, 10, 40, 64, 16};
    for (uint32_t d = 0; d < 5; ++d)
      require(integer(shape[d]) == dimensions[d], "invalid tiled shape extent");
    const auto path = root / (names[index] + ".bf16");
    const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    require(file >= 0, "payload readonly open failed");
    struct stat status{};
    if (::fstat(file, &status) || !S_ISREG(status.st_mode) || status.st_size != 419430400) {
      ::close(file); require(false, "payload extent/type differs");
    }
    void *address = ::mmap(nullptr, 419430400, PROT_READ, MAP_SHARED, file, 0);
    ::close(file); require(address != MAP_FAILED, "payload readonly mmap failed");
    auto plane = std::make_shared<Plane>(); plane->address = address; plane->count = 419430400;
    require(hash({static_cast<const uint8_t *>(address), plane->count}) == string(payload[@"sha256"]),
        "payload hash differs");
    const auto *coefficients = static_cast<const uint16_t *>(address);
    for (uint64_t i = 0; i < plane->count / 2; ++i)
      require((coefficients[i] & 0x7f80u) != 0x7f80u, "payload contains NaN or infinity");
    result.planes_[index] = std::move(plane);
  }
  return result;
}

std::array<metal::MetalBuffer, 3>
ExpertLUTSidecar::buffers(metal::MetalBackend &backend) const {
  std::array<metal::MetalBuffer, 3> result;
  for (uint32_t i = 0; i < 3; ++i) {
    require(planes_[i] && planes_[i]->address, "sidecar is not loaded");
    result[i] = backend.wrapSharedMemory(planes_[i]->address, planes_[i]->count,
        planes_[i], "readonly exact expert BF16 LUT");
  }
  return result;
}

} // namespace splash::flash::candidate
