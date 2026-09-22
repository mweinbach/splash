// Isolated scheduling screen on the installed layer's original Q4/G64 bytes.
// --cpu-self-test and compilation do not create a backend or submit GPU work.
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashAffine.h"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
constexpr uint16_t kSentinel = 0x7fc1;
constexpr uint32_t kSticky = 0x80000000u;
constexpr uint64_t kGuard = 64;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
uint64_t bytesHash(const void *pointer, uint64_t count, uint64_t hash = 0xcbf29ce484222325ULL) {
  const auto *bytes = static_cast<const uint8_t *>(pointer);
  for (uint64_t i = 0; i < count; ++i) hash = (hash ^ bytes[i]) * 0x100000001b3ULL;
  return hash;
}
std::vector<uint32_t> parseList(const std::string &value) {
  std::stringstream stream(value); std::string item; std::vector<uint32_t> result;
  while (std::getline(stream, item, ',')) {
    size_t used = 0; const auto parsed = std::stoul(item, &used);
    require(used == item.size() && parsed > 0 && parsed <= 4096, "invalid integer list");
    result.push_back(uint32_t(parsed));
  }
  require(!result.empty(), "empty integer list"); return result;
}
std::vector<std::string> stringList(const std::string &value) {
  std::stringstream stream(value); std::string item; std::vector<std::string> result;
  while (std::getline(stream, item, ',')) result.push_back(item);
  require(!result.empty(), "empty string list"); return result;
}
const char *modeName(uint32_t mode) {
  constexpr std::array names{"output", "row", "selection", "route", "row_swizzle2",
      "row_swizzle4", "row_swizzle8", "selection_swizzle2", "selection_swizzle4"};
  return names.at(mode);
}
DispatchSize grid(uint32_t columns, uint32_t rows, uint32_t selections, uint32_t mode) {
  if (mode == 9) return {columns, uint64_t(rows) * selections, 1};
  if (mode == 10) return {uint64_t(rows) * selections, columns, 1};
  if (mode == 1) return {rows, columns, selections};
  if (mode == 2) return {selections, columns, rows};
  if (mode == 3) return {uint64_t(rows) * selections, columns, 1};
  if (mode >= 4 && mode <= 6) {
    const uint32_t width = 1u << (mode - 3);
    return {uint64_t(columns) * width, (rows + width - 1) / width, selections};
  }
  if (mode >= 7) {
    const uint32_t width = 1u << (mode - 6);
    return {uint64_t(columns) * width, rows, (selections + width - 1) / width};
  }
  return {columns, rows, selections};
}
std::array<uint32_t, 3> mapped(uint32_t x, uint32_t y, uint32_t z,
                              uint32_t selections, uint32_t mode) {
  if (mode == 1) return {y, x, z};
  if (mode == 2) return {y, z, x};
  if (mode == 3) return {y, x / selections, x % selections};
  if (mode >= 4 && mode <= 6) {
    const uint32_t width = 1u << (mode - 3); return {x / width, y * width + x % width, z};
  }
  if (mode >= 7) {
    const uint32_t width = 1u << (mode - 6); return {x / width, y, z * width + x % width};
  }
  return {x, y, z};
}
void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t rows = 1; rows <= 17; ++rows) for (uint32_t selections : {1u, 3u, 10u})
    for (uint32_t columns : {1u, 7u, 80u, 320u}) for (uint32_t mode = 0; mode < 9; ++mode) {
      const auto size = grid(columns, rows, selections, mode);
      std::vector<uint32_t> visits(uint64_t(columns) * rows * selections);
      for (uint32_t z = 0; z < size.z; ++z) for (uint32_t y = 0; y < size.y; ++y)
        for (uint32_t x = 0; x < size.x; ++x) {
          const auto m = mapped(x, y, z, selections, mode);
          if (m[0] < columns && m[1] < rows && m[2] < selections)
            ++visits[(uint64_t(m[1]) * selections + m[2]) * columns + m[0]];
        }
      require(std::all_of(visits.begin(), visits.end(), [](uint32_t v) { return v == 1; }),
              "traversal duplicates or omits a tile"); ++checks;
    }
  for (uint32_t word = 0; word < 65536; ++word) if (std::isfinite(number(uint16_t(word)))) {
    require(bf16(number(uint16_t(word))) == word, "finite BF16 round trip failed"); ++checks;
  }
  for (uint32_t count : {1u, 10u, 40u, 160u}) for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    std::vector<int64_t> ids(count); std::vector<uint32_t> ranked(count), expected(count);
    for (uint32_t i = 0; i < count; ++i) {
      ids[i] = pattern == 0 ? 17 : pattern == 1 ? int64_t(i % 10) :
          pattern == 2 ? int64_t(randomWord(i + 31) % 512) : int64_t(count - i);
      expected[i] = i;
    }
    std::stable_sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) { return ids[a] < ids[b]; });
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t rank = 0;
      for (uint32_t j = 0; j < count; ++j) rank += ids[j] < ids[i] || (ids[j] == ids[i] && j < i);
      ranked[rank] = i;
    }
    require(ranked == expected, "route rank differs from independent stable sort"); ++checks;
    const uint32_t selections = count == 1 ? 1 : 10, rows = count / selections;
    for (uint32_t columns : {80u, 320u}) for (uint32_t mode : {9u, 10u}) {
      const auto size = grid(columns, rows, selections, mode);
      std::vector<uint32_t> visits(uint64_t(count) * columns);
      for (uint32_t z = 0; z < size.z; ++z) for (uint32_t y = 0; y < size.y; ++y)
        for (uint32_t x = 0; x < size.x; ++x) {
          const uint32_t job = mode == 10 ? x : y, column = mode == 10 ? y : x;
          if (job >= count || column >= columns || z) continue;
          ++visits[uint64_t(expected[job]) * columns + column];
        }
      require(std::all_of(visits.begin(), visits.end(), [](uint32_t value) { return value == 1; }),
              "sorted grid omits or aliases canonical route/column outputs"); ++checks;
    }
  }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << "}\n";
}
struct Mapping {
  void *address = MAP_FAILED; uint64_t bytes = 0;
  ~Mapping() { if (address != MAP_FAILED) munmap(address, bytes); }
};
MetalBuffer tensor(MetalBackend &backend, NSDictionary *manifest,
                   const std::filesystem::path &package, const std::string &name,
                   const std::string &dtype, std::array<uint64_t, 3> shape) {
  NSDictionary *record = manifest[@"tensors"][[NSString stringWithUTF8String:name.c_str()]];
  require([record isKindOfClass:[NSDictionary class]] &&
      [record[@"dtype"] isEqual:[NSString stringWithUTF8String:dtype.c_str()]],
      "tensor record missing or dtype differs: " + name);
  NSArray *dimensions = record[@"shape"];
  require(dimensions.count == 3, "tensor rank differs: " + name);
  for (uint32_t i = 0; i < 3; ++i)
    require([dimensions[i] unsignedLongLongValue] == shape[i], "tensor geometry differs: " + name);
  const uint64_t offset = [record[@"offset"] unsignedLongLongValue];
  const uint64_t length = [record[@"length"] unsignedLongLongValue];
  require(length == shape[0] * shape[1] * shape[2] * (dtype == "U32" ? 4 : 2),
          "tensor byte extent differs: " + name);
  NSString *shard = record[@"shard"];
  const std::filesystem::path relative(shard.UTF8String);
  require(!relative.is_absolute() && relative.string().find("..") == std::string::npos,
          "tensor shard path escapes package");
  const auto path = package / relative;
  const uint64_t fileBytes = std::filesystem::file_size(path);
  const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
  const uint64_t begin = offset / page * page;
  const uint64_t end = (offset + length + page - 1) / page * page;
  require(offset <= fileBytes && length <= fileBytes - offset && end <= fileBytes,
          "mapped tensor exceeds shard");
  const int file = open(path.c_str(), O_RDONLY);
  require(file >= 0, "cannot open tensor shard");
  auto owner = std::make_shared<Mapping>(); owner->bytes = end - begin;
  owner->address = mmap(nullptr, owner->bytes, PROT_READ, MAP_SHARED, file, off_t(begin));
  close(file); require(owner->address != MAP_FAILED, "tensor mapping failed");
  const auto base = backend.wrapSharedMemory(owner->address, owner->bytes, owner, name);
  return backend.view(base, offset - begin, length);
}
struct Plane {
  std::string name; uint32_t k = 0, n = 0; MetalBuffer weights, scales, biases;
  Plane(MetalBackend &backend, NSDictionary *manifest, const std::filesystem::path &package,
        const std::string &prefix, const std::string &phase) : name(phase) {
    k = phase == "down" ? 640 : 2560; n = phase == "down" ? 2560 : 640;
    const std::string root = prefix + "." + phase + "_proj";
    weights = tensor(backend, manifest, package, root + ".weight", "U32", {512, n, k / 8});
    scales = tensor(backend, manifest, package, root + ".scales", "BF16", {512, n, k / 64});
    biases = tensor(backend, manifest, package, root + ".biases", "BF16", {512, n, k / 64});
  }
};
struct Buffers {
  MetalBuffer input, ids, jobs, outputBase, output, diagnostics; uint32_t rows, selections;
  uint64_t elements = 0;
  Buffers(MetalBackend &backend, const Plane &p, uint32_t r, uint32_t s,
          const std::string &pattern, bool adversarial) : rows(r), selections(s) {
    const uint64_t inputRows = uint64_t(rows) * (p.name == "down" ? selections : 1);
    input = backend.allocateBuffer(inputRows * p.k * 2, BufferStorage::Shared);
    ids = backend.allocateBuffer(uint64_t(rows) * selections * 8, BufferStorage::Shared);
    jobs = backend.allocateBuffer(uint64_t(rows) * selections * 4, BufferStorage::Shared);
    elements = uint64_t(rows) * selections * p.n;
    outputBase = backend.allocateBuffer((elements + 2 * kGuard) * 2, BufferStorage::Shared);
    output = backend.view(outputBase, kGuard * 2, elements * 2);
    diagnostics = backend.allocateBuffer(64, BufferStorage::Shared);
    auto *x = static_cast<uint16_t *>(input.contents());
    for (uint64_t i = 0; i < inputRows * p.k; ++i) {
      float value = float(int(randomWord(i + 7) % 8193) - 4096) / 4096.f;
      if (adversarial) value = float(int(randomWord(i / 2 + 19) % 2049) - 1024) /
          float((i / 32) % 4 == 0 ? 16 : 1024) * (i % 2 ? -1.f : 1.f);
      x[i] = bf16(value);
    }
    auto *routes = static_cast<int64_t *>(ids.contents());
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t slot = 0; slot < selections; ++slot) {
      const uint32_t position = pattern == "shifted" ? (slot + row * 3) % selections : slot;
      const uint32_t shift = pattern == "disjoint" ||
          (pattern == "shifted" && position >= selections * 3 / 4) ? 13 * row : 0;
      routes[uint64_t(row) * selections + slot] = (17 + 29 * position + shift) % 512;
    }
    reset();
  }
  void reset() {
    std::fill_n(static_cast<uint16_t *>(outputBase.contents()), elements + 2 * kGuard, kSentinel);
    std::memset(diagnostics.contents(), 0, diagnostics.sizeBytes());
    std::memset(jobs.contents(), 0xff, jobs.sizeBytes());
    *static_cast<uint32_t *>(diagnostics.contents()) = kSticky;
  }
  void check() const {
    const auto *all = static_cast<const uint16_t *>(outputBase.contents());
    for (uint64_t i = 0; i < kGuard; ++i)
      require(all[i] == kSentinel && all[elements + kGuard + i] == kSentinel, "output guard changed");
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == kSticky, "shader diagnostics changed");
    const auto *values = static_cast<const uint16_t *>(output.contents());
    for (uint64_t i = 0; i < elements; ++i) require(std::isfinite(number(values[i])), "nonfinite output");
  }
  std::vector<uint16_t> result() const {
    const auto *all = static_cast<const uint16_t *>(output.contents()); return {all, all + elements};
  }
  void checkJobs() const {
    const uint32_t count = rows * selections;
    const auto *routes = static_cast<const int64_t *>(ids.contents());
    const auto *actual = static_cast<const uint32_t *>(jobs.contents());
    std::vector<uint32_t> expected(count);
    for (uint32_t i = 0; i < count; ++i) expected[i] = i;
    std::stable_sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) { return routes[a] < routes[b]; });
    require(std::equal(expected.begin(), expected.end(), actual), "GPU jobs differ from independent stable sort");
  }
};
uint64_t selectedHash(const Plane &p, const Buffers &b) {
  const auto *ids = static_cast<const int64_t *>(b.ids.contents());
  const std::set<int64_t> unique(ids, ids + uint64_t(b.rows) * b.selections);
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (int64_t id : unique) for (const auto &source : {p.weights, p.scales, p.biases}) {
    const uint64_t bytes = source.sizeBytes() / 512;
    hash = bytesHash(static_cast<const uint8_t *>(source.contents()) + uint64_t(id) * bytes, bytes, hash);
  }
  return hash;
}
struct Variant { std::string name, pipeline; uint32_t mode = 0, threads = 0; bool production = false;
  std::vector<double> gpu, wall; double medianGPU = 0, medianWall = 0; };
std::vector<Variant> variants(const Plane &p, const std::string &family, bool sortedOnly) {
  const bool contig = family == "contig";
  const uint32_t threads = contig ? (p.k == 2560 ? 128 : 64) : 256;
  const std::string candidate = contig ? (p.k == 2560 ? "ultra_contig16_m" : "ultra_contig8_m") : "ultra_affine_m";
  const std::string production = contig ? (p.k == 2560 ? "flash_expert_qmv_contig_k16_sg4_c2" :
      "flash_expert_qmv_contig_k8_sg2_c4") : "flash_affine_q4_g64_u32_c1";
  std::vector<Variant> result;
  result.push_back({"production_output", production, 0, threads, true, {}, {}, 0, 0});
  for (uint32_t mode = 0; mode < (sortedOnly ? 1u : 9u); ++mode)
    result.push_back({std::string("candidate_") + modeName(mode), candidate + std::to_string(mode),
        mode, threads, false, {}, {}, 0, 0});
  const std::string sorted = contig ? (p.k == 2560 ? "ultra_contig16_sorted" : "ultra_contig8_sorted") : "ultra_affine_sorted";
  for (uint32_t fast = 0; fast < 2; ++fast)
    result.push_back({fast ? "candidate_expert_sorted_route" : "candidate_expert_sorted_output", sorted + std::to_string(fast),
        9 + fast, threads, false, {}, {}, 0, 0});
  return result;
}
CommandGraph graphFor(const Plane &p, const Buffers &b, const Variant &v, uint32_t repeat) {
  CommandGraph graph;
  const FlashAffineParams params{b.rows, b.selections, p.k, p.n, 512, 4, 64,
      p.name == "down" ? 3u : 1u, p.k / 2, uint64_t(p.n) * p.k / 2,
      p.k / 32, uint64_t(p.n) * p.k / 32};
  for (uint32_t i = 0; i < repeat; ++i) {
    auto bindings = std::vector<MetalBuffer>{b.input, p.weights, p.scales, p.biases, b.ids, b.output, b.diagnostics};
    if (v.mode >= 9) {
      graph.add("ultra_quantized_route_jobs", {b.ids, b.jobs, b.diagnostics}, params, {1, 1, 1}, {256, 1, 1});
      bindings.push_back(b.jobs);
    }
    graph.add(v.pipeline, std::move(bindings), params,
        grid(p.n / 8, b.rows, b.selections, v.mode), {v.threads, 1, 1});
  }
  return graph;
}
std::vector<uint32_t> greedy(const std::vector<uint16_t> &values, uint32_t width) {
  std::vector<uint32_t> indices;
  for (uint64_t begin = 0; begin < values.size(); begin += width) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < width; ++i) if (number(values[begin + i]) > number(values[begin + best])) best = i;
    indices.push_back(best);
  }
  return indices;
}
double sampledRelativeL2(const Plane &p, const Buffers &b, const std::vector<uint16_t> &values) {
  const auto *x = static_cast<const uint16_t *>(b.input.contents());
  const auto *weights = static_cast<const uint8_t *>(p.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(p.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(p.biases.contents());
  const auto *ids = static_cast<const int64_t *>(b.ids.contents());
  double error = 0, norm = 0;
  const uint32_t routes = b.rows * b.selections;
  for (uint32_t ri = 0; ri < std::min(routes, 8u); ++ri) {
    const uint32_t route = ri * (routes - 1) / std::max(1u, std::min(routes, 8u) - 1);
    for (uint32_t ci = 0; ci < 16; ++ci) {
      const uint32_t column = ci * (p.n - 1) / 15;
      const uint64_t source = uint64_t(ids[route]) * p.n + column;
      const uint64_t inputRow = p.name == "down" ? route : route / b.selections;
      double dot = 0;
      for (uint32_t k = 0; k < p.k; ++k) {
        const uint8_t packed = weights[source * (p.k / 2) + k / 2];
        const uint32_t code = (packed >> ((k % 2) * 4)) & 15;
        const uint64_t coefficient = source * (p.k / 64) + k / 64;
        dot += double(number(x[inputRow * p.k + k])) *
            (double(code) * number(scales[coefficient]) + number(biases[coefficient]));
      }
      const double delta = number(values[uint64_t(route) * p.n + column]) - dot;
      error += delta * delta; norm += dot * dot;
    }
  }
  return std::sqrt(error / std::max(1e-30, norm));
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end()); const auto middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
} // namespace

int main(int argc, char **argv) {
  try { @autoreleasepool {
    std::string library = (std::filesystem::path(argv[0]).parent_path() / "locality.metallib").string(), output;
    std::filesystem::path package = "install/local-models/Flash-Next-oQ4e-mtp-v1";
    std::string prefix = "language_model.model.layers.0.mlp.switch_mlp";
    auto rowsList = parseList("1,4,16"), selectionsList = parseList("1,10");
    auto planes = stringList("gate,up,down"), families = stringList("affine,contig"), patterns = stringList("shared,disjoint");
    uint32_t samples = 7, repetitions = 8; bool sortedOnly = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      require(i + 1 < argc, "missing CLI value"); const std::string value = argv[++i];
      if (arg == "--library") library = value;
      else if (arg == "--out") output = value;
      else if (arg == "--package") package = value;
      else if (arg == "--prefix") prefix = value;
      else if (arg == "--rows") rowsList = parseList(value);
      else if (arg == "--selections") selectionsList = parseList(value);
      else if (arg == "--planes") planes = stringList(value);
      else if (arg == "--families") families = stringList(value);
      else if (arg == "--patterns") patterns = stringList(value);
      else if (arg == "--samples") samples = parseList(value).at(0);
      else if (arg == "--repeat") repetitions = parseList(value).at(0);
      else if (arg == "--sorted-only") {
        require(value == "0" || value == "1", "--sorted-only must be0 or1"); sortedOnly = value == "1";
      }
      else throw std::runtime_error("unknown argument: " + arg);
    }
    require(samples <= 99 && repetitions <= 256, "excessive samples or repeat count");
    for (uint32_t rows : rowsList) require(rows <= 16, "rows must be 1..16");
    for (uint32_t selections : selectionsList) require(selections == 1 || selections == 10, "selections must be 1 or10");
    for (const auto &name : planes) require(name == "gate" || name == "up" || name == "down", "invalid plane");
    for (const auto &name : families) require(name == "affine" || name == "contig", "invalid family");
    for (const auto &name : patterns) require(name == "shared" || name == "disjoint" || name == "shifted", "invalid route pattern");
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(package / "manifest.json").c_str()]];
    require(data != nil, "cannot read model manifest"); NSError *error = nil;
    NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([manifest isKindOfClass:[NSDictionary class]] &&
        [manifest[@"schema"] isEqual:@"splash-local-qwen4-affine-v1"], "invalid model manifest");
    MetalBackend backend(library);
    std::ostringstream report; report << std::setprecision(10)
        << "{\"experiment\":\"ultra_quantized_locality_v2\",\"die_affinity\":false,"
        << "\"sorted_jobs\":\"GPU stable ranks included before every sorted projection; canonical output scatter\","
        << "\"validation\":\"full BF16 bytes and per-route argmax against production for random and cancellation inputs; sampled FP64 dot numerical check\","
        << "\"source_identity\":" << splash::json::quote([manifest[@"source_identity_sha256"] UTF8String])
        << ",\"prefix\":" << splash::json::quote(prefix) << ",\"samples\":" << samples
        << ",\"repeat\":" << repetitions << ",\"cases\":[";
    bool firstCase = true;
    for (const auto &phase : planes) {
      Plane plane(backend, manifest, package, prefix, phase);
      for (uint32_t rows : rowsList) for (uint32_t selections : selectionsList)
        for (const auto &pattern : patterns) for (const auto &family : families) {
          if (family == "contig" && selections != 10) continue;
          Buffers b(backend, plane, rows, selections, pattern, false);
          auto candidates = variants(plane, family, sortedOnly);
          const uint64_t inputHash = bytesHash(b.input.contents(), b.input.sizeBytes());
          const uint64_t idsHash = bytesHash(b.ids.contents(), b.ids.sizeBytes());
          const uint64_t weightHash = selectedHash(plane, b);
          uint64_t validatedElements = 0; double numericalError = 0;
          for (bool adversarial : {false, true}) {
            Buffers validation(backend, plane, rows, selections, pattern, adversarial);
            std::vector<uint16_t> reference; std::vector<uint32_t> expectedGreedy;
            for (const auto &v : candidates) {
              validation.reset(); auto command = graphFor(plane, validation, v, 1);
              (void)backend.submitCommand(command.dispatches()); validation.check();
              if (v.mode >= 9) validation.checkJobs();
              const auto actual = validation.result();
              if (reference.empty()) {
                reference = actual; expectedGreedy = greedy(reference, plane.n);
                numericalError = std::max(numericalError, sampledRelativeL2(plane, validation, reference));
              }
              require(actual == reference, v.name + " changed BF16 bytes");
              require(greedy(actual, plane.n) == expectedGreedy, v.name + " changed per-route greedy selection");
              validatedElements += actual.size();
            }
          }
          require(numericalError < .006, "production projection failed sampled FP64 numerical oracle");
          for (uint32_t sample = 0; sample < samples; ++sample) for (uint32_t index = 0; index < candidates.size(); ++index) {
            auto &v = candidates[(index + sample * 3) % candidates.size()];
            auto command = graphFor(plane, b, v, repetitions);
            const auto timing = backend.submitCommand(command.dispatches()); b.check();
            if (v.mode >= 9) b.checkJobs();
            v.gpu.push_back(timing.gpuSeconds * 1000 / repetitions);
            v.wall.push_back(timing.wallSeconds * 1000 / repetitions);
          }
          require(inputHash == bytesHash(b.input.contents(), b.input.sizeBytes()) &&
              idsHash == bytesHash(b.ids.contents(), b.ids.sizeBytes()) && weightHash == selectedHash(plane, b),
              "immutable selected source or producer bytes changed");
          if (!firstCase) report << ','; firstCase = false;
          report << "{\"plane\":" << splash::json::quote(phase) << ",\"rows\":" << rows
              << ",\"selections\":" << selections << ",\"pattern\":" << splash::json::quote(pattern)
              << ",\"family\":" << splash::json::quote(family) << ",\"input_size\":" << plane.k
              << ",\"output_size\":" << plane.n << ",\"validated_elements\":" << validatedElements
              << ",\"bf16_mismatches\":0,\"greedy_mismatches\":0,\"sampled_relative_l2\":" << numericalError
              << ",\"selected_source_hash\":" << splash::json::quote(std::to_string(weightHash)) << ",\"variants\":[";
          for (auto &v : candidates) { v.medianGPU = median(v.gpu); v.medianWall = median(v.wall); }
          bool firstVariant = true; const Variant *winner = &candidates[0];
          for (const auto &v : candidates) {
            if (v.medianGPU < winner->medianGPU) winner = &v;
            if (!firstVariant) report << ','; firstVariant = false;
            report << "{\"name\":" << splash::json::quote(v.name) << ",\"median_gpu_ms\":" << v.medianGPU
                << ",\"median_wall_ms\":" << v.medianWall << ",\"speedup\":" << candidates[0].medianGPU / v.medianGPU
                << ",\"gpu_ms\":[";
            for (uint32_t i = 0; i < v.gpu.size(); ++i) { if (i) report << ','; report << v.gpu[i]; }
            report << "]}";
          }
          report << "]}";
          std::cerr << phase << " rows=" << rows << " selections=" << selections << " " << pattern << " " << family
              << " baseline_ms=" << candidates[0].medianGPU << " best=" << winner->name
              << " speedup=" << candidates[0].medianGPU / winner->medianGPU << '\n';
        }
    }
    report << "]}\n";
    if (output.empty()) std::cout << report.str();
    else { std::ofstream file(output); require(bool(file), "cannot open report output"); file << report.str(); }
    return 0;
  } } catch (const std::exception &error) {
    std::cerr << "ultra quantized locality oracle failed: " << error.what() << '\n'; return 1;
  }
}
