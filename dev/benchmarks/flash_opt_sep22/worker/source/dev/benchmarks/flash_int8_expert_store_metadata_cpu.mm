#include "flash/FlashInt8ExpertStoreMetadata.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

// This fixture never opens a backend or loads model operands. Sparse read-only
// payloads exercise only the metadata and filesystem contract; payload hashes
// deliberately remain placeholders for the consuming constructor to verify.
namespace {
using namespace splash::flash;
namespace fs = std::filesystem;
constexpr uint64_t kAlignment = 16384;
const std::string kSource(64, 'a');
const std::string kSourceManifest(64, 'b');
const std::string kDigest(64, 'c');

NSString *native(const std::string &value) {
  return [[NSString alloc] initWithBytes:value.data() length:value.size()
                              encoding:NSUTF8StringEncoding];
}
std::string digest(NSData *data) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes{};
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), bytes.data());
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : bytes) {
    result += alphabet[byte >> 4];
    result += alphabet[byte & 15];
  }
  return result;
}
std::string fingerprint(NormConvention convention) {
  const std::string text = "splash.native-flash-weights-v1\nsource=" + kSource +
      "\nmanifest=" + kSourceManifest + "\nnorm=" +
      (convention == NormConvention::OnePlusWeight ? "one-plus-weight\n" : "direct-gamma\n");
  return digest([NSData dataWithBytes:text.data() length:text.size()]);
}
uint64_t aligned(uint64_t value) { return (value + kAlignment - 1) & ~(kAlignment - 1); }

NSMutableDictionary *plane(NSString *dtype, NSArray *shape, uint64_t offset,
                           uint64_t length) {
  return [@{@"dtype": dtype, @"shape": [shape mutableCopy], @"offset": @(offset),
            @"length": @(length), @"sha256": native(kDigest)} mutableCopy];
}
NSMutableDictionary *projection(uint32_t layer, const char *name, uint64_t n,
                                uint64_t k, uint64_t hot, uint64_t &cursor) {
  cursor = aligned(cursor);
  NSMutableDictionary *codes = plane(@"I8", @[@(hot), @(n), @(k)], cursor, hot * n * k);
  cursor = aligned(cursor + hot * n * k);
  NSMutableDictionary *scales = plane(@"F32", @[@(hot), @(n)], cursor, hot * n * 4);
  cursor = aligned(cursor + hot * n * 4);
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) +
      ".mlp.switch_mlp." + name;
  return [@{@"source_prefix": native(prefix), @"dimensions": [@[@(hot), @(n), @(k)] mutableCopy],
            @"codes": codes, @"scales": scales} mutableCopy];
}

struct Fixture final {
  fs::path parent;
  fs::path directory;
  NSMutableDictionary *base = nil;
  uint64_t layerBytes = 0;
  std::array<uint32_t, 48> counts{};
  std::array<uint64_t, 48> fileBytes{};

  Fixture() {
    const std::string value = (fs::temp_directory_path() / "splash-int8-metadata-XXXXXX").string();
    std::vector<char> pattern(value.begin(), value.end());
    pattern.push_back('\0');
    char *created = ::mkdtemp(pattern.data());
    if (!created) throw std::runtime_error("mkdtemp failed");
    parent = fs::canonical(created);
    directory = parent / "store";
    fs::create_directory(directory);
    configure(32);
  }
  // Resize the same sparse files. No payload bytes are read, materialized or
  // scanned, even when the logical geometry covers 128 experts per layer.
  void configure(uint32_t hot) {
    std::array<uint32_t, 48> uniform{};
    uniform.fill(hot);
    configure(uniform);
  }
  void configure(const std::array<uint32_t, 48> &inventory) {
    counts = inventory;
    NSMutableArray *layers = [NSMutableArray array], *selected = [NSMutableArray array];
    uint64_t total = 0;
    for (uint32_t layer = 0; layer < 48; ++layer) {
      uint64_t cursor = 0;
      NSMutableDictionary *projections = [NSMutableDictionary dictionary];
      projections[@"gate_proj"] = projection(layer, "gate_proj", 640, 2560, counts[layer], cursor);
      projections[@"up_proj"] = projection(layer, "up_proj", 640, 2560, counts[layer], cursor);
      projections[@"down_proj"] = projection(layer, "down_proj", 2560, 640, counts[layer], cursor);
      layerBytes = cursor;
      fileBytes[layer] = cursor;
      char filename[32]{};
      std::snprintf(filename, sizeof(filename), "layer-%02u.bin", layer);
      const fs::path payload = directory / filename;
      if (fs::exists(payload) && ::chmod(payload.c_str(), 0644))
        throw std::runtime_error("payload resize chmod failed");
      FILE *file = std::fopen(payload.c_str(), "wb");
      if (!file || ::ftruncate(::fileno(file), static_cast<off_t>(layerBytes)))
        throw std::runtime_error("sparse payload creation failed");
      std::fclose(file);
      if (::chmod(payload.c_str(), 0444)) throw std::runtime_error("payload chmod failed");
      [layers addObject:[@{@"layer_index": @(layer), @"path": native(filename),
                          @"bytes": @(layerBytes), @"sha256": native(kDigest),
                          @"projections": projections} mutableCopy]];
      NSMutableArray *ids = [NSMutableArray array];
      for (uint32_t rank = 0; rank < counts[layer]; ++rank) [ids addObject:@(counts[layer] ==512 ? rank : layer + rank)];
      [selected addObject:ids];
      total += layerBytes;
    }
    base = [@{@"schema": @"splash-flash-int8-expert-store-v1",
              @"source_identity_sha256": native(kSource),
              @"source_manifest_sha256": native(kSourceManifest),
              @"plan_sha256": native(kDigest), @"alignment": @(kAlignment),
              @"target_layers": @48, @"selected_experts": selected,
              @"coefficient_policy": @"source_q4_g64_f32_separate_multiply_add_then_bf16_rne_v1",
              @"quantization_format": @"signed-symmetric-int8-rowwise-f32-scale",
              @"integer_rounding": @"F32 absmax/127; F32 division; nearest-even integer; clamp [-127,127]; zero row scale=1",
              @"layers": layers, @"total_bytes": @(total),
              @"planned_allocation_bytes": @(total + 48 * kAlignment)} mutableCopy];
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(parent, ignored);
  }
  NSData *bytes(NSDictionary *root) {
    NSError *error = nil;
    NSData *result = [NSJSONSerialization dataWithJSONObject:root
                           options:NSJSONWritingSortedKeys error:&error];
    if (!result || error) throw std::runtime_error("fixture JSON serialization failed");
    return result;
  }
  NSMutableDictionary *copy() {
    NSError *error = nil;
    id result = [NSJSONSerialization JSONObjectWithData:bytes(base)
                            options:NSJSONReadingMutableContainers error:&error];
    if (!result || error) throw std::runtime_error("fixture JSON copy failed");
    return result;
  }
  void write(NSData *data) {
    const fs::path path = directory / "manifest.json";
    std::error_code ignored;
    fs::remove(path, ignored);
    if (![data writeToFile:native(path.string()) atomically:NO])
      throw std::runtime_error("fixture manifest write failed");
  }
  void write(NSDictionary *root) { write(bytes(root)); }
  FlashInt8ExpertStoreMetadata load(const fs::path &path, NormConvention norm,
                                    std::string_view source = kSource,
                                    std::string_view expected = {}) {
    const std::string actualExpected = expected.empty() ? fingerprint(norm) : std::string(expected);
    return loadFlashInt8ExpertStoreMetadata(path, source, actualExpected, norm);
  }
};

NSMutableDictionary *layer(NSMutableDictionary *root, NSUInteger index = 0) {
  return root[@"layers"][index];
}
NSMutableDictionary *proj(NSMutableDictionary *root, NSUInteger projection = 0) {
  const std::array<NSString *, 3> names{@"gate_proj", @"up_proj", @"down_proj"};
  return layer(root)[@"projections"][names[projection]];
}
NSMutableDictionary *codes(NSMutableDictionary *root) { return proj(root)[@"codes"]; }
NSMutableDictionary *scales(NSMutableDictionary *root) { return proj(root)[@"scales"]; }

struct Runner final {
  Fixture fixture;
  uint32_t tests = 0;
  uint32_t failures = 0;
  void check(bool condition, const std::string &name) {
    ++tests;
    if (!condition) { ++failures; std::cerr << "FAIL " << name << '\n'; }
  }
  void rejects(const std::string &name, const std::function<void()> &body,
               std::string_view expectedMessage = {}) {
    ++tests;
    try { body(); ++failures; std::cerr << "FAIL accepted " << name << '\n'; }
    catch (const std::exception &error) {
      if (!expectedMessage.empty() && std::string_view(error.what()).find(expectedMessage) == std::string_view::npos) {
        ++failures;
        std::cerr << "FAIL wrong rejection for " << name << ": " << error.what() << '\n';
      }
    }
  }
  void mutation(const std::string &name,
                const std::function<void(NSMutableDictionary *)> &change) {
    NSMutableDictionary *root = fixture.copy();
    change(root);
    fixture.write(root);
    rejects(name, [&] { fixture.load(fixture.directory, NormConvention::OnePlusWeight); });
  }
  void rawMutation(const std::string &name,
                   const std::function<std::string(std::string)> &change) {
    NSData *data = fixture.bytes(fixture.base);
    std::string raw(static_cast<const char *>(data.bytes), data.length);
    raw = change(raw);
    fixture.write([NSData dataWithBytes:raw.data() length:raw.size()]);
    rejects(name, [&] { fixture.load(fixture.directory, NormConvention::OnePlusWeight); });
  }
};

std::string replaceFirst(std::string raw, const std::string &from,
                         const std::string &to) {
  const size_t found = raw.find(from);
  if (found == std::string::npos) throw std::runtime_error("raw fixture token not found: " + from);
  raw.replace(found, from.size(), to);
  return raw;
}

void run(Runner &r) {
  auto &f = r.fixture;
  const std::string rawSha = digest(f.bytes(f.base));
  for (uint32_t inventory : {32, 64, 128, 256, 512}) {
    f.configure(inventory);
    f.write(f.base);
    const std::string inventorySha = digest(f.bytes(f.base));
    const std::string cohort = " inventory=" + std::to_string(inventory);
    for (NormConvention norm : {NormConvention::OnePlusWeight, NormConvention::DirectGamma}) {
      const auto result = f.load(f.directory, norm);
      r.check(result.sourceIdentity == kSource && result.sourceManifestSha256 == kSourceManifest &&
              result.planSha256 == kDigest && result.identitySha256 == inventorySha,
              "valid source and raw manifest identities" + cohort);
      r.check(result.directory == fs::canonical(f.directory) &&
              result.totalBytes == f.layerBytes * 48 &&
              result.plannedBytes == f.layerBytes * 48 + 48 * kAlignment,
              "valid canonical path and allocation counts" + cohort);
      bool allLayers = true;
      for (uint32_t i = 0; i < 48; ++i) {
        const auto &item = result.layers[i];
        std::vector<uint32_t> expectedIDs;
        for (uint32_t rank = 0; rank < inventory; ++rank) expectedIDs.push_back(inventory ==512 ? rank : i + rank);
        allLayers &= item.bytes == f.fileBytes[i] && item.sha256 == kDigest &&
                     item.selectedIDs == expectedIDs && fs::exists(item.path);
        for (uint32_t p = 0; p < 3; ++p) {
          allLayers &= item.codes[p].length == uint64_t(inventory) * 640 * 2560 &&
                       item.scales[p].length == uint64_t(inventory) * (p == 2 ? 2560 : 640) * 4 &&
                       item.codes[p].sha256 == kDigest && item.scales[p].sha256 == kDigest &&
                       item.codes[p].offset % kAlignment == 0 && item.scales[p].offset % kAlignment == 0;
        }
      }
      r.check(allLayers, "all 48 ordered layers and six plane descriptors retained" + cohort);
    }
  }
  // Unsupported counts and mixed supported counts have fully corresponding
  // code/scale shapes, offsets, file lengths and allocation ledgers. Their
  // failure therefore exercises the inventory guard independently of geometry.
  for (uint32_t inventory : {1, 31, 33, 65, 127}) {
    f.configure(inventory);
    f.write(f.base);
    r.rejects("unsupported uniform inventory=" + std::to_string(inventory),
              [&] { f.load(f.directory, NormConvention::OnePlusWeight); },
              "selected expert inventory must contain exactly 32, 64, 128, 256 or512 IDs per layer");
  }
  for (uint32_t inventory : {32, 64}) {
    std::array<uint32_t, 48> mixed{};
    mixed.fill(inventory);
    mixed.back() = inventory == 32 ? 64 : 32;
    f.configure(mixed);
    f.write(f.base);
    r.rejects("mixed 32/64 fully matching inventory with first layer=" + std::to_string(inventory),
              [&] { f.load(f.directory, NormConvention::OnePlusWeight); },
              "metadata count, geometry or extent mismatch");
  }
  f.configure(32);
  f.write(f.base);
  r.check(digest(f.bytes(f.base)) == rawSha, "default 32-expert fixture identity restored after inventory cases");
  r.rejects("wrong expected source", [&] { f.load(f.directory, NormConvention::OnePlusWeight, std::string(64, 'd')); });
  r.rejects("wrong expected effective fingerprint", [&] { f.load(f.directory, NormConvention::OnePlusWeight, kSource, std::string(64, 'd')); });
  r.rejects("raw source manifest digest is not effective fingerprint", [&] { f.load(f.directory, NormConvention::OnePlusWeight, kSource, kSourceManifest); });
  r.rejects("other norm fingerprint", [&] { f.load(f.directory, NormConvention::DirectGamma, kSource, fingerprint(NormConvention::OnePlusWeight)); });

  r.mutation("schema version", [](auto root) { root[@"schema"] = @"splash-flash-int8-expert-store-v2"; });
  r.mutation("source mismatch", [](auto root) { root[@"source_identity_sha256"] = native(std::string(64, 'd')); });
  r.mutation("source manifest mismatch", [](auto root) { root[@"source_manifest_sha256"] = native(std::string(64, 'd')); });
  r.mutation("coefficient policy", [](auto root) { root[@"coefficient_policy"] = @"BF16 multiplication"; });
  r.mutation("quantization format", [](auto root) { root[@"quantization_format"] = @"signed-symmetric-int8-g64-f32-scale"; });
  r.mutation("rounding policy", [](auto root) { root[@"integer_rounding"] = @"truncate"; });
  r.mutation("alignment", [](auto root) { root[@"alignment"] = @8192; });
  r.mutation("target layers", [](auto root) { root[@"target_layers"] = @47; });
  r.mutation("total byte count", [](auto root) { root[@"total_bytes"] = @0; });
  r.mutation("planned byte count", [](auto root) { root[@"planned_allocation_bytes"] = root[@"total_bytes"]; });
  for (NSString *key : @[@"schema", @"source_identity_sha256", @"source_manifest_sha256",
                         @"plan_sha256", @"alignment", @"target_layers", @"selected_experts",
                         @"coefficient_policy", @"quantization_format", @"integer_rounding",
                         @"layers", @"total_bytes", @"planned_allocation_bytes"]) {
    const std::string name = "missing root key " + std::string(key.UTF8String);
    r.mutation(name, [key](auto root) { [root removeObjectForKey:key]; });
  }
  r.mutation("unknown root key", [](auto root) { root[@"unexpected"] = @1; });
  r.mutation("unknown layer key", [](auto root) { layer(root)[@"unexpected"] = @1; });
  r.mutation("unknown projection key", [](auto root) { proj(root)[@"unexpected"] = @1; });
  r.mutation("unknown plane key", [](auto root) { codes(root)[@"unexpected"] = @1; });
  r.mutation("missing layer key", [](auto root) { [layer(root) removeObjectForKey:@"path"]; });
  r.mutation("missing projection key", [](auto root) { [proj(root) removeObjectForKey:@"source_prefix"]; });
  r.mutation("missing plane key", [](auto root) { [codes(root) removeObjectForKey:@"offset"]; });
  r.mutation("root null", [](auto root) { root[@"layers"] = [NSNull null]; });
  r.mutation("layer non-object", [](auto root) { root[@"layers"][0] = @1; });
  r.mutation("projection non-object", [](auto root) { layer(root)[@"projections"][@"gate_proj"] = @1; });
  r.mutation("plane non-object", [](auto root) { proj(root)[@"codes"] = @1; });
  r.mutation("string integer", [](auto root) { root[@"target_layers"] = @"48"; });
  r.mutation("boolean integer", [](auto root) { root[@"target_layers"] = @YES; });
  r.mutation("negative integer", [](auto root) { codes(root)[@"offset"] = @-1; });
  r.rawMutation("whole floating count", [](auto raw) { return replaceFirst(raw, "\"target_layers\":48", "\"target_layers\":48.0"); });
  r.rawMutation("exponent count", [](auto raw) { return replaceFirst(raw, "\"target_layers\":48", "\"target_layers\":4.8e1"); });
  r.rawMutation("fractional ID", [](auto raw) { return replaceFirst(raw, "\"selected_experts\":[[0,1,", "\"selected_experts\":[[0.5,1,"); });
  r.rawMutation("overflow integer", [](auto raw) { return replaceFirst(raw, "\"target_layers\":48", "\"target_layers\":18446744073709551616"); });
  r.rawMutation("duplicate root key", [](auto raw) { raw.insert(1, "\"target_layers\":48,"); return raw; });
  r.rawMutation("escaped duplicate root key", [](auto raw) { raw.insert(1, "\"target_\\u006cayers\":48,"); return raw; });
  r.rawMutation("duplicate layer key", [](auto raw) { return replaceFirst(raw, "\"layer_index\":0", "\"layer_index\":0,\"layer_index\":0"); });
  r.rawMutation("duplicate plane key", [](auto raw) { return replaceFirst(raw, "\"dtype\":\"I8\"", "\"dtype\":\"I8\",\"dtype\":\"I8\""); });
  r.rawMutation("truncated JSON", [](auto raw) { raw.resize(raw.size() - 1); return raw; });

  r.mutation("layers count 47", [](auto root) { [root[@"layers"] removeLastObject]; });
  r.mutation("layers count 49", [](auto root) { [root[@"layers"] addObject:layer(root)]; });
  r.mutation("layer order", [](auto root) { layer(root)[@"layer_index"] = @1; });
  r.mutation("projection count two", [](auto root) { [layer(root)[@"projections"] removeObjectForKey:@"down_proj"]; });
  r.mutation("projection binding order", [](auto root) {
    NSMutableDictionary *projections = layer(root)[@"projections"];
    id gate = projections[@"gate_proj"];
    projections[@"gate_proj"] = projections[@"up_proj"];
    projections[@"up_proj"] = gate;
  });
  r.mutation("wrong source prefix", [](auto root) { proj(root)[@"source_prefix"] = @"language_model.model.layers.1.mlp.switch_mlp.gate_proj"; });
  r.mutation("selected layer count 47", [](auto root) { [root[@"selected_experts"] removeLastObject]; });
  r.mutation("selected empty", [](auto root) { root[@"selected_experts"][0] = [NSMutableArray array]; });
  r.mutation("selected duplicate", [](auto root) { root[@"selected_experts"][0][1] = @0; });
  r.mutation("selected unsorted", [](auto root) { [root[@"selected_experts"][0] exchangeObjectAtIndex:0 withObjectAtIndex:1]; });
  r.mutation("selected ID 512", [](auto root) { root[@"selected_experts"][0][0] = @512; });
  r.mutation("selected negative ID", [](auto root) { root[@"selected_experts"][0][0] = @-1; });
  r.mutation("selected bool ID", [](auto root) { root[@"selected_experts"][0][0] = @YES; });
  r.mutation("selected more than 128", [](auto root) {
    NSMutableArray *ids = [NSMutableArray array];
    for (uint32_t i = 0; i < 129; ++i) [ids addObject:@(i)];
    root[@"selected_experts"][0] = ids;
  });
  r.mutation("dimension compact count", [](auto root) { proj(root)[@"dimensions"][0] = @2; });
  r.mutation("dimension output count", [](auto root) { proj(root)[@"dimensions"][1] = @641; });
  r.mutation("dimension input count", [](auto root) { proj(root)[@"dimensions"][2] = @2559; });
  r.mutation("dimension missing", [](auto root) { [proj(root)[@"dimensions"] removeLastObject]; });
  r.mutation("codes dtype", [](auto root) { codes(root)[@"dtype"] = @"U8"; });
  r.mutation("scales dtype", [](auto root) { scales(root)[@"dtype"] = @"BF16"; });
  r.mutation("codes shape", [](auto root) { codes(root)[@"shape"][2] = @2559; });
  r.mutation("scales shape", [](auto root) { scales(root)[@"shape"] = @[@32, @640, @1]; });
  r.mutation("codes compact inventory shape mismatch", [](auto root) { codes(root)[@"shape"][0] = @31; });
  r.mutation("scales compact inventory shape mismatch", [](auto root) { scales(root)[@"shape"][0] = @64; });
  r.mutation("codes offset unaligned", [](auto root) { codes(root)[@"offset"] = @1; });
  r.mutation("codes offset noncanonical", [](auto root) { codes(root)[@"offset"] = @(kAlignment); });
  r.mutation("scales offset overlap", [](auto root) { scales(root)[@"offset"] = @0; });
  r.mutation("codes length", [](auto root) { codes(root)[@"length"] = @1; });
  r.mutation("scales length", [](auto root) { scales(root)[@"length"] = @1; });
  r.mutation("layer declared bytes", [](auto root) { layer(root)[@"bytes"] = @1; });
  r.mutation("uppercase hash", [](auto root) { root[@"plan_sha256"] = native(std::string(64, 'A')); });
  r.mutation("short hash", [](auto root) { layer(root)[@"sha256"] = @"abc"; });
  r.mutation("invalid hash character", [](auto root) { codes(root)[@"sha256"] = native(std::string(64, 'g')); });
  r.mutation("nonstrings hash", [](auto root) { scales(root)[@"sha256"] = @1; });
  r.mutation("embedded NUL", [](auto root) { layer(root)[@"path"] = native(std::string("layer-00.bin\0escape", 18)); });
  r.mutation("empty path", [](auto root) { layer(root)[@"path"] = @""; });
  r.mutation("absolute path", [&f](auto root) { layer(root)[@"path"] = native((f.directory / "layer-00.bin").string()); });
  r.mutation("parent path", [](auto root) { layer(root)[@"path"] = @"../store/layer-00.bin"; });
  r.mutation("current-directory path", [](auto root) { layer(root)[@"path"] = @"./layer-00.bin"; });
  r.mutation("nested path", [](auto root) { layer(root)[@"path"] = @"subdir/layer-00.bin"; });
  r.mutation("other layer payload", [](auto root) { layer(root)[@"path"] = @"layer-01.bin"; });

  f.write(f.base);
  const fs::path payload = f.directory / "layer-00.bin", saved = f.directory / "saved.bin";
  fs::rename(payload, saved);
  r.rejects("missing payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::create_symlink(saved.filename(), payload);
  r.rejects("payload symlink", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::remove(payload);
  fs::create_directory(payload);
  r.rejects("payload nonregular directory", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::remove(payload);
  if (::mkfifo(payload.c_str(), 0444)) throw std::runtime_error("mkfifo fixture failed");
  r.rejects("payload nonregular FIFO", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::remove(payload);
  fs::rename(saved, payload);
  if (::chmod(payload.c_str(), 0644)) throw std::runtime_error("writable payload chmod failed");
  r.rejects("owner-writable payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  if (::chmod(payload.c_str(), 0464)) throw std::runtime_error("group-writable payload chmod failed");
  r.rejects("group-writable payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  if (::chmod(payload.c_str(), 0446)) throw std::runtime_error("other-writable payload chmod failed");
  r.rejects("other-writable payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  if (::chmod(payload.c_str(), 0644)) throw std::runtime_error("resize payload chmod failed");
  fs::resize_file(payload, f.layerBytes - 1);
  if (::chmod(payload.c_str(), 0444)) throw std::runtime_error("readonly payload chmod failed");
  r.rejects("truncated payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  if (::chmod(payload.c_str(), 0644)) throw std::runtime_error("resize payload chmod failed");
  fs::resize_file(payload, f.layerBytes + 1);
  if (::chmod(payload.c_str(), 0444)) throw std::runtime_error("readonly payload chmod failed");
  r.rejects("oversized payload", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  if (::chmod(payload.c_str(), 0644)) throw std::runtime_error("resize payload chmod failed");
  fs::resize_file(payload, f.layerBytes);
  if (::chmod(payload.c_str(), 0444)) throw std::runtime_error("readonly payload chmod failed");

  const fs::path alias = f.parent / "alias";
  fs::create_directory_symlink(f.directory, alias);
  r.rejects("store directory symlink", [&] { f.load(alias, NormConvention::OnePlusWeight); });
  r.rejects("store path is regular file", [&] { f.load(payload, NormConvention::OnePlusWeight); });
  r.rejects("missing store directory", [&] { f.load(f.parent / "absent", NormConvention::OnePlusWeight); });
  const fs::path manifest = f.directory / "manifest.json", manifestSaved = f.directory / "manifest-saved.json";
  fs::rename(manifest, manifestSaved);
  r.rejects("missing manifest", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::create_symlink(manifestSaved.filename(), manifest);
  r.rejects("manifest symlink", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::remove(manifest);
  fs::create_directory(manifest);
  r.rejects("manifest nonregular", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  fs::remove(manifest);
  fs::rename(manifestSaved, manifest);
  f.write([NSData data]);
  r.rejects("empty manifest", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  NSMutableData *large = [NSMutableData dataWithLength:(8ULL << 20) + 1];
  f.write(large);
  r.rejects("manifest over 8MiB", [&] { f.load(f.directory, NormConvention::OnePlusWeight); });
  f.write(f.base);
  const auto restored = f.load(f.directory, NormConvention::OnePlusWeight);
  r.check(restored.identitySha256 == rawSha, "fixture remains valid after filesystem negatives");
}
} // namespace

int main() {
  @autoreleasepool {
    try {
      Runner runner;
      run(runner);
      std::cout << "Flash INT8 expert-store metadata CPU: " << runner.tests << " checks, "
                << runner.failures << " failures; no GPU or model operands loaded\n";
      return runner.failures ? 1 : 0;
    } catch (const std::exception &error) {
      std::cerr << "Flash INT8 metadata fixture error: " << error.what() << '\n';
      return 2;
    }
  }
}
