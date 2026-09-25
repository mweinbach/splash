#include "flash/OptQmv.hpp"

#include <algorithm>
#include <dispatch/dispatch.h>
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>

namespace splash::flash::opt {
namespace {

// Must match OptQmvParams in opt_qmv.metal.
struct QmvParams {
  uint32_t K, N, rows, xStride, yStride, row0;
  uint64_t wRowStride, pRowStride;
};

// Balanced chunks of at most kRowCapacities' largest entry (5 rows).
template <class Fn>
void forEachChunk(uint32_t rows, Fn &&fn) {
  const uint32_t chunks = (rows + 4) / 5;
  const uint32_t per = (rows + chunks - 1) / chunks;
  for (uint32_t row0 = 0; row0 < rows; row0 += per) fn(row0, std::min(per, rows - row0));
}
static_assert(sizeof(QmvParams) == 40);

struct Variant {
  uint32_t rn, sn, sk, vpt;
};

// Kernel row capacities that are compiled; the smallest one >= rows is used.
// Eight-row variants spill registers (4-8x slower), so the route stops at 5.
constexpr uint32_t kRowCapacities[] = {1, 2, 3, 4, 5};

// From the standalone sweep (build/opt-sep22/bench/sweep1.txt) over this
// model's projection shapes on M5 Ultra.
Variant choose(uint32_t K, uint32_t N, uint32_t capacity) {
  if (K <= 1024) return {4, 1, 4, 8};
  if (capacity >= 3) return N <= 1024 ? Variant{2, 1, 16, 8} : Variant{2, 1, 8, 8};
  if (N <= 1024 || K >= 8192) return {2, 1, 16, 8};
  if (K >= 4096) return {4, 1, 4, 16};
  return {4, 1, 8, 16};
}

} // namespace

uint32_t draftVocabulary() noexcept {
  static const uint32_t rows = [] {
    const char *value = std::getenv("SPLASH_OPT_DRAFT_VOCAB");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 98304u;
  }();
  return rows;
}

bool qmvEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_QMV");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

namespace {
std::mutex &repackMutex() { static std::mutex mutex; return mutex; }
std::unordered_map<const void *, metal::MetalBuffer> &repackRegistry() {
  static std::unordered_map<const void *, metal::MetalBuffer> registry;
  return registry;
}
const metal::MetalBuffer *repackedCodes(const void *codes) {
  std::lock_guard lock(repackMutex());
  const auto found = repackRegistry().find(codes);
  return found == repackRegistry().end() ? nullptr : &found->second;
}
} // namespace

uint32_t prefillQmvRows() noexcept {
  static const uint32_t rows = [] {
    const char *value = std::getenv("SPLASH_OPT_PREFILL_QMV_ROWS");
    const unsigned long parsed = value && *value ? std::strtoul(value, nullptr, 10) : 128ul;
    return static_cast<uint32_t>(std::min<unsigned long>(parsed, kQmvMaximumPrefillRows));
  }();
  return rows;
}

bool prefillQmvEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_PREFILL_QMV");
    return qmvEnabled() && (!value || std::strcmp(value, "0") != 0);
  }();
  return enabled;
}

bool repackEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_REPACK");
    return qmvEnabled() && (!value || std::strcmp(value, "0") != 0);
  }();
  return enabled;
}

bool repackEligible(const FlashAffineProjection &p) noexcept {
  return (p.bits == 5 || p.bits == 6) && p.experts == 1 && p.weights && p.scales && p.biases &&
      (p.groupSize == 64 || p.groupSize == 128) && p.outputSize > 1024 &&
      p.outputSize % 32 == 0 && p.inputSize % p.groupSize == 0 &&
      p.inputSize / p.groupSize <= 160 && p.weights->buffer.contents() &&
      p.weightRowStrideBytes >= (uint64_t(p.inputSize) * p.bits + 7) / 8;
}

metal::MetalBuffer repackCodes(metal::MetalBackend &backend, const FlashAffineProjection &p) {
  const uint32_t N = p.outputSize, K = p.inputSize, bits = p.bits;
  auto out = backend.allocateBuffer((uint64_t(N) * K + 16383) & ~uint64_t(16383),
      metal::BufferStorage::Shared, "opt-repacked-u8-codes");
  const auto *source = static_cast<const uint8_t *>(p.weights->buffer.contents());
  auto *destination = static_cast<uint8_t *>(out.contents());
  if (!source || !destination) throw std::logic_error("repacked codes require Shared buffers");
  const uint64_t rowBytes = p.weightRowStrideBytes;
  const uint32_t mask = (1u << bits) - 1u;
  // MLX packs each row as a little-endian bitstream of `bits`-wide codes.
  dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
    const uint8_t *row = source + n * rowBytes;
    uint8_t *codes = destination + n * K;
    for (uint32_t k = 0; k < K; ++k) {
      const uint64_t bit = uint64_t(k) * bits;
      const uint64_t byte = bit >> 3;
      uint32_t value = row[byte];
      if (byte + 1 < rowBytes) value |= uint32_t(row[byte + 1]) << 8;
      codes[k] = uint8_t((value >> (bit & 7)) & mask);
    }
  });
  return out;
}

void registerRepackedCodes(const void *codes, const metal::MetalBuffer &repacked) {
  std::lock_guard lock(repackMutex());
  repackRegistry()[codes] = repacked;
}

bool mkDenseEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_MK_QMV");
    return qmvEnabled() && value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

namespace {
std::unordered_map<const void *, MkTiledProjection> &mkTiledRegistry() {
  static std::unordered_map<const void *, MkTiledProjection> registry;
  return registry;
}
const MkTiledProjection *mkTiled(const void *codes) {
  std::lock_guard lock(repackMutex());
  const auto found = mkTiledRegistry().find(codes);
  return found == mkTiledRegistry().end() ? nullptr : &found->second;
}
constexpr uint32_t kMkTileN = 32, kMkTileK = 64;
uint32_t mkPaddedN(uint32_t n) { return (n + kMkTileN - 1) / kMkTileN * kMkTileN; }
} // namespace

bool mkTileEligible(const FlashAffineProjection &p) noexcept {
  return p.experts == 1 && p.weights && p.scales && p.biases && p.weights->buffer.contents() &&
      p.scales->buffer.contents() && p.biases->buffer.contents() &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.groupSize == 64 || p.groupSize == 128) && p.outputSize >= 1024 &&
      p.inputSize % kMkTileK == 0 && p.inputSize % p.groupSize == 0 && p.inputSize <= 10240 &&
      p.weightRowStrideBytes >= uint64_t(p.inputSize) * p.bits / 8 &&
      p.parameterRowStrideBytes >= uint64_t(p.inputSize / p.groupSize) * 2;
}

uint64_t mkTileBytes(const FlashAffineProjection &p) noexcept {
  const uint64_t n = mkPaddedN(p.outputSize);
  const uint64_t codes = n * p.inputSize * p.bits / 8;
  const uint64_t parameters = 2 * n * (p.inputSize / p.groupSize) * 2;
  return ((codes + 16383) & ~uint64_t(16383)) + ((parameters + 16383) & ~uint64_t(16383));
}

namespace {
uint64_t mkRoundUp(uint64_t bytes) { return (bytes + 16383) & ~uint64_t(16383); }
uint64_t mkCodeBytes(const FlashAffineProjection &p) {
  return uint64_t(mkPaddedN(p.outputSize)) * p.inputSize * p.bits / 8;
}
uint64_t mkParameterCount(const FlashAffineProjection &p) {
  return uint64_t(2) * mkPaddedN(p.outputSize) * (p.inputSize / p.groupSize);
}
// Lane-major tiles of column blocks [nt0, nt1) and their [K/G][NP] scales then
// biases. With a column map, tile column c holds original output columns[c].
void mkWriteTileBlocks(const uint8_t *source, const uint16_t *scales, const uint16_t *biases,
                       uint64_t rowBytes, uint64_t parameterRow, uint32_t N, uint32_t NP, uint32_t K,
                       uint32_t bits, uint32_t groupSize, const uint32_t *map, uint8_t *destination,
                       uint16_t *parameters, uint32_t nt0, uint32_t nt1) {
  const uint32_t tiles = K / kMkTileK, tileBytes = 256 * bits, groups = K / groupSize;
  const uint32_t mask = (1u << bits) - 1u;
  const auto code = [&](uint32_t column, uint32_t k) -> uint32_t {
    const uint32_t n = map ? map[column] : column;
    if (n >= N) return 0;
    const uint8_t *row = source + uint64_t(n) * rowBytes;
    const uint64_t bit = uint64_t(k) * bits, byte = bit >> 3;
    uint32_t value = row[byte];
    if (byte + 1 < rowBytes) value |= uint32_t(row[byte + 1]) << 8;
    return (value >> (bit & 7)) & mask;
  };
  // Element e of lane L is (k, n) of the uint8 right-operand cooperative tensor
  // of a 16x32x64 matmul (see kernels/mk_tiles.h).
  for (uint32_t nt = nt0; nt < nt1; ++nt) {
    uint8_t *block = destination + uint64_t(nt) * tiles * tileBytes;
    std::memset(block, 0, uint64_t(tiles) * tileBytes);
    for (uint32_t kt = 0; kt < tiles; ++kt) {
      uint8_t *t = block + uint64_t(kt) * tileBytes;
      for (uint32_t L = 0; L < 32; ++L) {
        const uint32_t kL = 4 * (L & 1) + 8 * ((L >> 3) & 1), nL = ((L >> 1) & 3) + 4 * ((L >> 4) & 1);
        for (uint32_t e = 0; e < 64; ++e) {
          const uint32_t k = kt * kMkTileK + kL + 16 * (e >> 4) + (e & 3);
          const uint32_t c = code(nt * kMkTileN + nL + 8 * ((e >> 2) & 3), k);
          if (bits == 8) { t[L * 64 + e] = uint8_t(c); continue; }
          t[L * 32 + 4 * (e / 8) + (e & 3)] |= uint8_t((c & 15u) << (4 * ((e / 4) & 1)));
          if (bits == 5) t[1024 + L * 8 + e / 8] |= uint8_t(((c >> 4) & 1u) << (e % 8));
          if (bits == 6) t[1024 + L * 16 + e / 4] |= uint8_t(((c >> 4) & 3u) << (2 * (e % 4)));
        }
      }
    }
    for (uint32_t column = nt * kMkTileN; column < (nt + 1) * kMkTileN; ++column) {
      const uint32_t n = map ? map[column] : column;
      for (uint32_t g = 0; g < groups; ++g) {
        parameters[uint64_t(g) * NP + column] = n < N ? scales[uint64_t(n) * parameterRow + g] : 0;
        parameters[uint64_t(groups + g) * NP + column] = n < N ? biases[uint64_t(n) * parameterRow + g] : 0;
      }
    }
  }
}

// Whole projection (expert 0 of a rank-2 projection).
void mkWriteTiles(const FlashAffineProjection &p, uint8_t *destination, uint16_t *parameters,
                  const std::vector<uint32_t> *columns = nullptr) {
  const uint32_t NP = columns ? uint32_t(columns->size()) : mkPaddedN(p.outputSize);
  const auto *source = static_cast<const uint8_t *>(p.weights->buffer.contents());
  const auto *scales = static_cast<const uint16_t *>(p.scales->buffer.contents());
  const auto *biases = static_cast<const uint16_t *>(p.biases->buffer.contents());
  if (!source || !scales || !biases) throw std::logic_error("mk dense tiles require Shared buffers");
  const uint32_t *map = columns ? columns->data() : nullptr;
  dispatch_apply(NP / kMkTileN, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t nt) {
    mkWriteTileBlocks(source, scales, biases, p.weightRowStrideBytes, p.parameterRowStrideBytes / 2,
                      p.outputSize, NP, p.inputSize, p.bits, p.groupSize, map, destination, parameters,
                      uint32_t(nt), uint32_t(nt) + 1);
  });
}

bool mkFormat(const FlashAffineProjection &p) {
  return p.experts == 1 && p.weights && p.scales && p.biases && p.weights->buffer.contents() &&
      p.scales->buffer.contents() && p.biases->buffer.contents() &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.groupSize == 64 || p.groupSize == 128) && p.outputSize > 0 &&
      p.inputSize % kMkTileK == 0 && p.inputSize % p.groupSize == 0 && p.inputSize <= 10240 &&
      p.weightRowStrideBytes >= uint64_t(p.inputSize) * p.bits / 8 &&
      p.parameterRowStrideBytes >= uint64_t(p.inputSize / p.groupSize) * 2;
}
// Must match MkSegment / MkMultiParams in kernels/mk_dense.metal.
struct MkSegmentHost {
  uint32_t blockBegin, n, paddedN, bits, group, outIndex, pad0, pad1;
  uint64_t tileOffset, parameterOffset, biasOffset, pad2;
};
struct MkMultiHost {
  uint32_t K, rows, segments, xStride;
  uint32_t yStride[4];
  MkSegmentHost segment[4];
};
static_assert(sizeof(MkSegmentHost) == 64 && sizeof(MkMultiHost) == 288);
} // namespace

MkTiledProjection mkTileProjection(metal::MetalBackend &backend, const FlashAffineProjection &p) {
  MkTiledProjection out;
  out.paddedN = mkPaddedN(p.outputSize);
  out.tiles = backend.allocateBuffer(mkRoundUp(mkCodeBytes(p)), metal::BufferStorage::Shared, "mk-dense-tiles");
  out.parameters = backend.allocateBuffer(mkRoundUp(mkParameterCount(p) * 2), metal::BufferStorage::Shared,
                                          "mk-dense-parameters");
  mkWriteTiles(p, static_cast<uint8_t *>(out.tiles.contents()), static_cast<uint16_t *>(out.parameters.contents()));
  return out;
}

bool mkGroupEligible(const std::vector<const FlashAffineProjection *> &projections) noexcept {
  if (projections.empty() || projections.size() > 4) return false;
  for (const auto *p : projections)
    if (!p || !mkFormat(*p) || p->inputSize != projections.front()->inputSize) return false;
  return true;
}

uint64_t mkGroupBytes(const std::vector<const FlashAffineProjection *> &projections) noexcept {
  uint64_t codes = 0, parameters = 0;
  for (const auto *p : projections) { codes += mkCodeBytes(*p); parameters += mkParameterCount(*p) * 2; }
  return mkRoundUp(codes) + mkRoundUp(parameters);
}

MkTiledGroup mkTileGroup(metal::MetalBackend &backend, const std::vector<const FlashAffineProjection *> &projections) {
  if (!mkGroupEligible(projections)) throw std::invalid_argument("mk tiled group format unsupported");
  MkTiledGroup group;
  uint64_t codes = 0, parameters = 0;
  for (const auto *p : projections) { codes += mkCodeBytes(*p); parameters += mkParameterCount(*p); }
  group.tiles = backend.allocateBuffer(mkRoundUp(codes), metal::BufferStorage::Shared, "mk-dense-group-tiles");
  group.parameters = backend.allocateBuffer(mkRoundUp(parameters * 2), metal::BufferStorage::Shared,
                                            "mk-dense-group-parameters");
  MkMultiHost table{};
  table.K = projections.front()->inputSize;
  table.segments = uint32_t(projections.size());
  uint64_t tileOffset = 0, parameterOffset = 0;
  for (uint32_t i = 0; i < projections.size(); ++i) {
    const auto &p = *projections[i];
    const uint32_t NP = mkPaddedN(p.outputSize);
    mkWriteTiles(p, static_cast<uint8_t *>(group.tiles.contents()) + tileOffset,
                 static_cast<uint16_t *>(group.parameters.contents()) + parameterOffset);
    table.segment[i] = {group.blocks, p.outputSize, NP, p.bits, p.groupSize, i, 0, 0, tileOffset,
                        parameterOffset, uint64_t(p.inputSize / p.groupSize) * NP, 0};
    group.n[i] = p.outputSize;
    group.blocks += NP / kMkTileN;
    tileOffset += mkCodeBytes(p);
    parameterOffset += mkParameterCount(p);
  }
  group.K = table.K;
  group.count = table.segments;
  group.table.resize(sizeof(table));
  std::memcpy(group.table.data(), &table, sizeof(table));
  return group;
}

bool addMkGroup(metal::CommandGraph &graph, const metal::MetalBuffer &input, const MkTiledGroup &group,
                const std::vector<metal::MetalBuffer> &outputs, uint32_t rows) {
  if (rows < 2 || rows > 8 || outputs.size() != group.count) return false;
  MkMultiHost table;
  std::memcpy(&table, group.table.data(), sizeof(table));
  table.rows = rows;
  table.xStride = group.K;
  for (uint32_t i = 0; i < 4; ++i) table.yStride[i] = i < group.count ? group.n[i] : 0;
  std::vector<metal::MetalBuffer> buffers{input, group.tiles, group.parameters};
  for (uint32_t i = 0; i < 4; ++i) buffers.push_back(outputs[i < group.count ? i : 0]);
  graph.add("mk_mpt_multi_sk4", std::move(buffers), table, {group.blocks, 1, 1}, {128, 1, 1});
  return true;
}

namespace {
// Must match MkHCSegment / MkHCDownParams / MkHCUpParams in kernels/mk_hc.metal.
struct MkHCSegmentHost {
  uint32_t bits, group, paddedN, pad0;
  uint64_t tileOffset, parameterOffset, biasOffset, pad1;
};
struct MkHCDownHost { uint32_t rows, splits, blocks, pad0; MkHCSegmentHost down, inject; };
struct MkHCUpHost { uint32_t rows, splits, hasInjection, pad0; MkHCSegmentHost up; };
static_assert(sizeof(MkHCSegmentHost) == 48 && sizeof(MkHCDownHost) == 112 && sizeof(MkHCUpHost) == 64);
constexpr uint32_t kMkHCSplits = 4;
bool mkHCFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  return mkFormat(p) && p.inputSize == K && p.outputSize == N && p.groupSize == 64;
}
} // namespace

bool mkHCEligible(const FlashAffineProjection &down, const FlashAffineProjection *inject,
                  const FlashAffineProjection &up) noexcept {
  return mkHCFormat(down, 10240, 320) && mkHCFormat(up, 320, 10240) &&
      (!inject || (mkFormat(*inject) && inject->inputSize == 10240 && inject->outputSize == 4));
}

uint64_t mkHCBytes(const FlashAffineProjection &down, const FlashAffineProjection *inject,
                   const FlashAffineProjection &up) noexcept {
  std::vector<const FlashAffineProjection *> group{&down};
  if (inject) group.push_back(inject);
  return mkGroupBytes(group) + mkRoundUp(mkCodeBytes(up)) + mkRoundUp(mkParameterCount(up) * 2);
}

MkHC mkTileHC(metal::MetalBackend &backend, const FlashAffineProjection &down,
              const FlashAffineProjection *inject, const FlashAffineProjection &up) {
  if (!mkHCEligible(down, inject, up)) throw std::invalid_argument("mk HC format unsupported");
  MkHC hc;
  std::vector<const FlashAffineProjection *> group{&down};
  if (inject) group.push_back(inject);
  hc.down = mkTileGroup(backend, group);
  hc.injection = inject != nullptr;
  std::vector<uint32_t> columns(10240);
  for (uint32_t c = 0; c < 10240; ++c) columns[c] = (c % 4) * 2560 + 8 * (c / 32) + (c % 32) / 4;
  hc.upTiles = backend.allocateBuffer(mkRoundUp(mkCodeBytes(up)), metal::BufferStorage::Shared, "mk-hc-up-tiles");
  hc.upParameters = backend.allocateBuffer(mkRoundUp(mkParameterCount(up) * 2), metal::BufferStorage::Shared,
                                           "mk-hc-up-parameters");
  mkWriteTiles(up, static_cast<uint8_t *>(hc.upTiles.contents()), static_cast<uint16_t *>(hc.upParameters.contents()),
               &columns);
  hc.upBits = up.bits;
  return hc;
}

metal::MetalBuffer mkHCPartials(metal::MetalBackend &backend) {
  return backend.allocateBuffer(mkRoundUp(uint64_t(kMkHCSplits) * 8 * 352 * 4), metal::BufferStorage::Shared,
                                "mk-hc-partials");
}

bool addMkHC(metal::CommandGraph &graph, const metal::MetalBuffer &normalized, const MkHC &hc,
             const metal::MetalBuffer &partials, const metal::MetalBuffer &mixed,
             const metal::MetalBuffer &gates, uint32_t rows) {
  if (rows < 2 || rows > 8) return false;
  MkMultiHost table;
  std::memcpy(&table, hc.down.table.data(), sizeof(table));
  const auto segment = [](const MkSegmentHost &s) {
    return MkHCSegmentHost{s.bits, s.group, s.paddedN, 0, s.tileOffset, s.parameterOffset, s.biasOffset, 0};
  };
  MkHCDownHost down{rows, kMkHCSplits, hc.down.blocks, 0, segment(table.segment[0]),
                    hc.injection ? segment(table.segment[1]) : segment(table.segment[0])};
  graph.add("mk_hc_down_sk8", {normalized, hc.down.tiles, hc.down.parameters, partials}, down,
            {hc.down.blocks, kMkHCSplits, 1}, {256, 1, 1});
  const MkHCUpHost up{rows, kMkHCSplits, hc.injection ? 1u : 0u, 0,
                      {hc.upBits, 64, 10240, 0, 0, 0, uint64_t(5) * 10240, 0}};
  graph.add("mk_hc_up", {normalized, partials, hc.upTiles, hc.upParameters, mixed, hc.injection ? gates : mixed}, up,
            {80, 1, 1}, {128, 1, 1});
  return true;
}

bool mkExpertTilesEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_MK_MOE_TILED");
    return mkDenseEnabled() && value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

namespace {
bool mkExpertFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  return p.experts == 512 && p.weights && p.scales && p.biases && p.weights->buffer.contents() &&
      p.scales->buffer.contents() && p.biases->buffer.contents() && p.bits == 4 && p.groupSize == 64 &&
      p.inputSize == K && p.outputSize == N && p.weightRowStrideBytes == uint64_t(K) / 2 &&
      p.weightExpertStrideBytes == uint64_t(N) * K / 2 && p.parameterRowStrideBytes == uint64_t(K / 64) * 2 &&
      p.parameterExpertStrideBytes == uint64_t(N) * (K / 64) * 2;
}
} // namespace

bool mkExpertTilesEligible(const FlashAffineProjection &gate, const FlashAffineProjection &up,
                           const FlashAffineProjection &down) noexcept {
  return mkExpertFormat(gate, 2560, 640) && mkExpertFormat(up, 2560, 640) && mkExpertFormat(down, 640, 2560);
}

uint64_t mkExpertTileBytes(const FlashAffineProjection &gate, const FlashAffineProjection &up,
                           const FlashAffineProjection &down) noexcept {
  uint64_t total = 0;
  for (const auto *p : {&gate, &up, &down})
    total += mkRoundUp(p->weightExpertStrideBytes * p->experts) + mkRoundUp(p->parameterExpertStrideBytes * 2 * p->experts);
  return total;
}

MkExpertTiles mkTileExperts(metal::MetalBackend &backend, const FlashAffineProjection &gate,
                            const FlashAffineProjection &up, const FlashAffineProjection &down) {
  if (!mkExpertTilesEligible(gate, up, down)) throw std::invalid_argument("mk expert tile format unsupported");
  MkExpertTiles out;
  const FlashAffineProjection *sources[3] = {&gate, &up, &down};
  metal::MetalBuffer *codes[3] = {&out.gate, &out.up, &out.down};
  metal::MetalBuffer *parameters[3] = {&out.gateParameters, &out.upParameters, &out.downParameters};
  for (int i = 0; i < 3; ++i) {
    const auto &p = *sources[i];
    *codes[i] = backend.allocateBuffer(mkRoundUp(p.weightExpertStrideBytes * p.experts),
                                       metal::BufferStorage::Shared, "mk-expert-tiles");
    *parameters[i] = backend.allocateBuffer(mkRoundUp(p.parameterExpertStrideBytes * 2 * p.experts),
                                            metal::BufferStorage::Shared, "mk-expert-parameters");
    const auto *source = static_cast<const uint8_t *>(p.weights->buffer.contents());
    const auto *scales = static_cast<const uint16_t *>(p.scales->buffer.contents());
    const auto *biases = static_cast<const uint16_t *>(p.biases->buffer.contents());
    auto *destination = static_cast<uint8_t *>(codes[i]->contents());
    auto *parameterOut = static_cast<uint16_t *>(parameters[i]->contents());
    const uint32_t blocks = p.outputSize / kMkTileN;
    const uint64_t codeStride = p.weightExpertStrideBytes, parameterStride = p.parameterExpertStrideBytes / 2;
    dispatch_apply(size_t(p.experts) * blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t job) {
      const uint64_t expert = job / blocks;
      const uint32_t nt = uint32_t(job % blocks);
      mkWriteTileBlocks(source + expert * codeStride, scales + expert * parameterStride,
                        biases + expert * parameterStride, p.weightRowStrideBytes, p.parameterRowStrideBytes / 2,
                        p.outputSize, p.outputSize, p.inputSize, 4, 64, nullptr,
                        destination + expert * codeStride, parameterOut + expert * 2 * parameterStride, nt, nt + 1);
    });
  }
  return out;
}

namespace {
std::unordered_map<const void *, MkExpertTiles> &mkExpertRegistry() {
  static std::unordered_map<const void *, MkExpertTiles> registry;
  return registry;
}
} // namespace

void registerMkExpertTiles(const FlashAffineProjection &gate, const FlashAffineProjection &down,
                           const MkExpertTiles &tiles) {
  std::lock_guard lock(repackMutex());
  mkExpertRegistry()[gate.weights->buffer.contents()] = tiles;
  mkExpertRegistry()[down.weights->buffer.contents()] = tiles;
}

const MkExpertTiles *mkExpertTilesFor(const FlashAffineProjection &p) noexcept {
  if (!p.weights) return nullptr;
  std::lock_guard lock(repackMutex());
  const auto found = mkExpertRegistry().find(p.weights->buffer.contents());
  return found == mkExpertRegistry().end() ? nullptr : &found->second;
}

const MkTiledProjection *mkTiledProjection(const FlashAffineProjection &p) noexcept {
  return p.weights ? mkTiled(p.weights->buffer.contents()) : nullptr;
}

void registerMkTiled(const void *codes, const MkTiledProjection &tiled) {
  std::lock_guard lock(repackMutex());
  mkTiledRegistry()[codes] = tiled;
}

bool addQmv(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const FlashAffineProjection &p, const metal::MetalBuffer &output,
            uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || p.experts != 1 ||
      !p.weights || !p.scales || !p.biases)
    return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.groupSize != 64 && p.groupSize != 128) return false;
  if (p.bits == 4 && p.groupSize != 64) return false;
  const uint32_t K = p.inputSize, N = p.outputSize;
  if (!K || !N || K % 16 || K % p.groupSize) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 4) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  if (input.sameView(output)) return false;
  if (mkDenseEnabled() && rows >= 2 && rows <= 16) {
    if (const auto *tiled = mkTiled(base)) {
      struct { uint32_t K, N, rows, xStride, yStride, row0, paddedN, pad0; uint64_t biasOffset; } params{
          K, N, rows, K, N, 0, tiled->paddedN, 0, uint64_t(K / p.groupSize) * tiled->paddedN};
      static_assert(sizeof(params) == 40);
      // Sixteen-row windows keep threadgroup memory within 16 KB (at most eight simdgroups).
      const bool wide = rows > 8;
      const uint32_t sk = K >= 4096 ? (wide ? 8 : 16) : 4;
      graph.add((wide ? "mk_mpt16_b" : "mk_mpt_b") + std::to_string(p.bits) + "_g" + std::to_string(p.groupSize) +
                    "_sk" + std::to_string(sk) + (wide ? "" : "_x1"),
                {input, tiled->tiles, tiled->parameters, output}, params,
                {tiled->paddedN / 32, 1, 1}, {32 * sk, 1, 1});
      return true;
    }
  }
  // Windows wider than five rows (batched verification, short prefill): the
  // matrix units read each 4/8-bit code (or the 8-bit repack of a 5/6-bit
  // projection) once per sixteen-row chunk; a <= 5-row tail uses the SIMD
  // kernel. Row ranges written by the kernels are disjoint.
  uint32_t simdRow0 = 0;
  const metal::MetalBuffer *repacked = (p.bits == 5 || p.bits == 6) && repackEnabled()
      ? repackedCodes(base) : nullptr;
  const bool matrixFormat = (p.bits == 4 && p.groupSize == 64) || p.bits == 8 || repacked;
  // Narrow outputs (N <= 1024) measured faster in-model as SIMD chunks for
  // verification windows; prefill-sized windows (> 20 rows) use the matrix
  // units for them too, with sixteen-column tiles and a 16-way K split.
  const bool narrow = N <= 1024;
  // SPLASH_MK_DENSE=1: wide projections of 2..5-row windows also use the
  // matrix units, whose cost does not grow with the row count.
  static const uint32_t minimumMatrixRows = [] {
    const char *value = std::getenv("SPLASH_MK_DENSE");
    return value && std::strcmp(value, "1") == 0 ? 2u : kQmvMaximumRows + 1;
  }();
  const bool fewRows = rows <= kQmvMaximumRows;
  if (rows >= minimumMatrixRows && (!fewRows || (!narrow && !repacked)) && mppqEnabled() &&
      matrixFormat && N % 32 == 0 &&
      K / p.groupSize <= 160 && (!narrow || rows > kQmvMaximumSplitRows)) {
    const uint32_t nt = narrow ? 16 : 32, sk = narrow ? 16 : 8;
    const uint32_t bits = repacked ? 8 : p.bits;
    const std::string pipeline = "opt_mppq_b" + std::to_string(bits) + "_g" +
        std::to_string(p.groupSize) + "_nt" + std::to_string(nt) + "_sk" + std::to_string(sk);
    const metal::MetalBuffer &codes = repacked ? *repacked : p.weights->buffer;
    const uint64_t codeRowStride = repacked ? uint64_t(K) : p.weightRowStrideBytes;
    while (rows - simdRow0 > kQmvMaximumRows ||
           (simdRow0 == 0 && rows >= minimumMatrixRows && rows <= kQmvMaximumRows)) {
      const uint32_t matrixRows = std::min<uint32_t>(rows - simdRow0, 16);
      const QmvParams params{K, N, matrixRows, K, N, simdRow0, codeRowStride, p.parameterRowStrideBytes};
      graph.add(pipeline, {input, codes, p.scales->buffer, p.biases->buffer, output},
                params, {N / nt, 1, 1}, {32 * sk, 1, 1});
      simdRow0 += matrixRows;
    }
  }
  if (simdRow0 == rows) return true;
  forEachChunk(rows - simdRow0, [&](uint32_t chunk0, uint32_t count) {
    const uint32_t row0 = simdRow0 + chunk0;
    uint32_t capacity = 0;
    for (uint32_t candidate : kRowCapacities)
      if (candidate >= count) { capacity = candidate; break; }
    const Variant v = choose(K, N, capacity);
    const std::string pipeline = "opt_qmv_b" + std::to_string(p.bits) + "_g" +
        std::to_string(p.groupSize) + "_r" + std::to_string(capacity) + "_n" +
        std::to_string(v.rn) + "_sn" + std::to_string(v.sn) + "_sk" +
        std::to_string(v.sk) + "_v" + std::to_string(v.vpt);
    const QmvParams params{K, N, count, K, N, row0, p.weightRowStrideBytes, p.parameterRowStrideBytes};
    const uint32_t perGroup = v.rn * v.sn;
    graph.add(pipeline, {input, p.weights->buffer, p.scales->buffer, p.biases->buffer, output},
              params, {(N + perGroup - 1) / perGroup, 1, 1}, {32 * v.sn * v.sk, 1, 1});
  });
  return true;
}

bool addSharedSwiGLU(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                     const FlashAffineProjection &gate, const FlashAffineProjection &up,
                     const metal::MetalBuffer &output, const metal::MetalBuffer &diagnostics,
                     uint32_t rows) {
  if (!qmvEnabled() || !fusedSwiGLUEnabled() || !rows || rows > kQmvMaximumSplitRows) return false;
  for (const auto *p : {&gate, &up})
    if (p->experts != 1 || !p->weights || !p->scales || !p->biases || p->bits != 8 ||
        (p->groupSize != 64 && p->groupSize != 128) || p->weightRowStrideBytes % 4 ||
        p->parameterRowStrideBytes % 2 || !p->weights->buffer.contents() ||
        reinterpret_cast<uintptr_t>(p->weights->buffer.contents()) % 4)
      return false;
  if (gate.inputSize != up.inputSize || gate.outputSize != up.outputSize ||
      gate.groupSize != up.groupSize || gate.weightRowStrideBytes != up.weightRowStrideBytes ||
      gate.parameterRowStrideBytes != up.parameterRowStrideBytes)
    return false;
  const uint32_t K = gate.inputSize, N = gate.outputSize;
  if (!K || !N || K % 16 || K % gate.groupSize || !diagnostics) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    uint32_t capacity = 0;
    for (uint32_t candidate : kRowCapacities)
      if (candidate >= count) { capacity = candidate; break; }
    // Same variant as addQmv would choose for each projection, so every gate
    // and up value is summed exactly as by the separate projections.
    const Variant v = choose(K, N, capacity);
    const std::string pipeline = "opt_qmv_swiglu_b8_g" + std::to_string(gate.groupSize) +
        "_r" + std::to_string(capacity) + "_n" + std::to_string(v.rn) + "_sn" +
        std::to_string(v.sn) + "_sk" + std::to_string(v.sk) + "_v" + std::to_string(v.vpt);
    const QmvParams params{K, N, count, K, N, row0, gate.weightRowStrideBytes, gate.parameterRowStrideBytes};
    const uint32_t perGroup = v.rn * v.sn;
    graph.add(pipeline, {input, gate.weights->buffer, gate.scales->buffer, gate.biases->buffer,
              up.weights->buffer, up.scales->buffer, up.biases->buffer, output, diagnostics},
              params, {(N + perGroup - 1) / perGroup, 1, 1}, {32 * v.sn * v.sk, 1, 1});
  });
  return true;
}

bool addQmvTall(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                const FlashAffineProjection &p, const metal::MetalBuffer &output,
                uint32_t rows) {
  if (!qmvEnabled() || rows <= kQmvMaximumSplitRows || p.experts != 1 || p.outputSize > 64 ||
      !p.weights || !p.scales || !p.biases)
    return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.groupSize != 64 && p.groupSize != 128) return false;
  if (p.bits == 4 && p.groupSize != 64) return false;
  const uint32_t K = p.inputSize, N = p.outputSize;
  if (!K || !N || K % 16 || K % p.groupSize) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 4) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  // Four-row blocks along grid y. Activation rows come from memory, so the
  // variant maximizes parallel loads: long K splits 16 ways for <= 4 outputs;
  // wide outputs use four 4-output groups with a 2-way K split.
  uint32_t sn = 2, sk = 1;
  if (N <= 4) { sn = 1; sk = 16; }
  else if (N > 8) { sn = 4; sk = 2; }
  const std::string pipeline = "opt_qmv_b" + std::to_string(p.bits) + "_g" +
      std::to_string(p.groupSize) + "_r4_n4_sn" + std::to_string(sn) + "_sk" +
      std::to_string(sk) + "_v8";
  const QmvParams params{K, N, rows, K, N, 0, p.weightRowStrideBytes, p.parameterRowStrideBytes};
  const uint32_t perGroup = 4 * sn;
  graph.add(pipeline, {input, p.weights->buffer, p.scales->buffer, p.biases->buffer, output},
            params, {(N + perGroup - 1) / perGroup, (rows + 3) / 4, 1}, {32 * sn * sk, 1, 1});
  return true;
}

namespace {

struct HCParams {
  uint32_t rows, hasInjection, injectionBits, injectionGroup;
  uint64_t downWRow, downPRow, injWRow, injPRow, upWRow, upPRow;
  uint32_t row0, reserved;
};
static_assert(sizeof(HCParams) == 72);

bool hcFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  if (p.experts != 1 || !p.weights || !p.scales || !p.biases) return false;
  if (p.inputSize != K || p.outputSize != N || p.groupSize != 64) return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  return base && reinterpret_cast<uintptr_t>(base) % 4 == 0;
}

uint32_t rowCapacity(uint32_t rows) {
  for (uint32_t candidate : kRowCapacities)
    if (candidate >= rows) return candidate;
  return 0;
}

} // namespace

bool addHCDown(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
               const FlashAffineProjection &down, const FlashAffineProjection *injection,
               const metal::MetalBuffer &activated, const metal::MetalBuffer &gates,
               uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || !hcFormat(down, 10240, 320)) return false;
  if (injection) {
    if (injection->experts != 1 || injection->inputSize != 10240 || injection->outputSize != 4 ||
        !injection->weights || !injection->scales || !injection->biases ||
        (injection->bits != 4 && injection->bits != 5 && injection->bits != 6 && injection->bits != 8) ||
        (injection->groupSize != 64 && injection->groupSize != 128) ||
        injection->weightRowStrideBytes % 4 ||
        injection->parameterRowStrideBytes % 2 || !gates)
      return false;
  }
  const FlashAffineProjection &i = injection ? *injection : down;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const HCParams params{count, injection ? 1u : 0u, injection ? injection->bits : 0u,
        injection ? injection->groupSize : 0u, down.weightRowStrideBytes, down.parameterRowStrideBytes,
        injection ? injection->weightRowStrideBytes : 0u, injection ? injection->parameterRowStrideBytes : 0u,
        0, 0, row0, 0};
    const std::string pipeline = "opt_hc_down_b" + std::to_string(down.bits) + "_g64_r" +
        std::to_string(rowCapacity(count));
    graph.add(pipeline, {normalized, down.weights->buffer, down.scales->buffer, down.biases->buffer,
        i.weights->buffer, i.scales->buffer, i.biases->buffer, activated,
        injection ? gates : activated}, params, {injection ? 161u : 160u, 1, 1}, {512, 1, 1});
  });
  return true;
}

bool addHCUpMix(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
                const metal::MetalBuffer &activated, const FlashAffineProjection &up,
                const metal::MetalBuffer &mixed, uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || !hcFormat(up, 320, 10240)) return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const HCParams params{count, 0, 0, 0, 0, 0, 0, 0, up.weightRowStrideBytes,
        up.parameterRowStrideBytes, row0, 0};
    const std::string pipeline = "opt_hc_up_mix_b" + std::to_string(up.bits) + "_g64_r" +
        std::to_string(rowCapacity(count));
    graph.add(pipeline, {normalized, activated, up.weights->buffer, up.scales->buffer,
        up.biases->buffer, mixed}, params, {640, 1, 1}, {128, 1, 1});
  });
  return true;
}

bool addDenseBF16Rows(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                      const FlashTensor &weights, const metal::MetalBuffer &output,
                      uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumSplitRows || weights.dtype != FlashDType::BF16 ||
      weights.shape.size() != 2)
    return false;
  const uint64_t n = weights.shape[0], k = weights.shape[1];
  if (!n || !k || n > UINT32_MAX || k % 8 || k > UINT32_MAX) return false;
  const auto *base = static_cast<const unsigned char *>(weights.buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 8) return false;
  const uint32_t N = uint32_t(n), K = uint32_t(k);
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const QmvParams params{K, N, count, K, N, row0, 0, 0};
    graph.add("opt_dense_bf16_r" + std::to_string(rowCapacity(count)),
              {input, weights.buffer, output}, params, {(N + 3) / 4, 1, 1}, {128, 1, 1});
  });
  return true;
}

namespace {
int moeMode() noexcept {
  static const int mode = [] {
    const char *value = std::getenv("SPLASH_OPT_MOE");
    if (!qmvEnabled() || !value) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "2") == 0) return 2;
    return 0;
  }();
  return mode;
}
} // namespace

bool moeEnabled() noexcept { return moeMode() != 0; }
bool fusedSwiGLUEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_FUSED_SWIGLU");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool mppqEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_MPPQ");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool statePoolEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_STATE_POOL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool gdnKernelEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_GDN_KERNEL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool qsaKernelEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_QSA_KERNEL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool draftChainEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_DRAFT_CHAIN");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool prepareVerifyEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_PREPARE_VERIFY");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool moeReplacesInt8() noexcept { return moeMode() == 1; }

namespace {
struct MoEParams {
  uint32_t K, N, rows, selections, experts, reserved;
  uint64_t wRow, wExpert, pRow, pExpert;
};
static_assert(sizeof(MoEParams) == 56);

bool expertFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  if (!p.weights || !p.scales || !p.biases || p.experts < 2) return false;
  if (p.inputSize != K || p.outputSize != N || p.bits != 4 || p.groupSize != 64) return false;
  if (p.weightRowStrideBytes % 4 || p.weightExpertStrideBytes % 4 ||
      p.parameterRowStrideBytes % 2 || p.parameterExpertStrideBytes % 2)
    return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  return base && reinterpret_cast<uintptr_t>(base) % 4 == 0;
}
} // namespace

bool addMoEExperts(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                   const FlashAffineProjection &gate, const FlashAffineProjection &up,
                   const FlashAffineProjection &down, const metal::MetalBuffer &expertIDs,
                   const metal::MetalBuffer &intermediate, const metal::MetalBuffer &expertDown,
                   uint32_t rows, uint32_t selections) {
  if (!moeEnabled() || !rows || rows > kQmvMaximumRows || !selections || selections > 16) return false;
  if (!expertFormat(gate, 2560, 640) || !expertFormat(up, 2560, 640) || !expertFormat(down, 640, 2560))
    return false;
  if (gate.experts != up.experts || gate.experts != down.experts ||
      gate.weightRowStrideBytes != up.weightRowStrideBytes ||
      gate.weightExpertStrideBytes != up.weightExpertStrideBytes ||
      gate.parameterRowStrideBytes != up.parameterRowStrideBytes ||
      gate.parameterExpertStrideBytes != up.parameterExpertStrideBytes)
    return false;
  const uint32_t routes = rows * selections;
  if (expertIDs.sizeBytes() < uint64_t(routes) * 8 || intermediate.sizeBytes() < uint64_t(routes) * 640 * 2 ||
      expertDown.sizeBytes() < uint64_t(routes) * 2560 * 2)
    return false;
  const MoEParams gu{2560, 640, rows, selections, gate.experts, 0, gate.weightRowStrideBytes,
      gate.weightExpertStrideBytes, gate.parameterRowStrideBytes, gate.parameterExpertStrideBytes};
  graph.add("opt_moe_gate_up_b4_g64", {input, gate.weights->buffer, gate.scales->buffer,
      gate.biases->buffer, up.weights->buffer, up.scales->buffer, up.biases->buffer, expertIDs,
      intermediate}, gu, {640 / 8, routes, 1}, {64, 1, 1});
  const MoEParams dn{640, 2560, rows, selections, down.experts, 0, down.weightRowStrideBytes,
      down.weightExpertStrideBytes, down.parameterRowStrideBytes, down.parameterExpertStrideBytes};
  graph.add("opt_moe_down_b4_g64", {intermediate, down.weights->buffer, down.scales->buffer,
      down.biases->buffer, down.weights->buffer, down.scales->buffer, down.biases->buffer, expertIDs,
      expertDown}, dn, {2560 / 8, routes, 1}, {64, 1, 1});
  return true;
}

bool addRoute(metal::CommandGraph &graph, const metal::MetalBuffer &logits,
              const metal::MetalBuffer &expertIDs, const metal::MetalBuffer &routeWeights,
              uint32_t rows, uint32_t experts, uint32_t selections, bool normalizeTopK) {
  if (!qmvEnabled() || !rows || experts != 512 || !selections || selections > 10) return false;
  if (logits.sizeBytes() < uint64_t(rows) * experts * 2 ||
      expertIDs.sizeBytes() < uint64_t(rows) * selections * 8 ||
      routeWeights.sizeBytes() < uint64_t(rows) * selections * 2)
    return false;
  const struct { uint32_t rows, experts, selections, normalize; } params{
      rows, experts, selections, normalizeTopK ? 1u : 0u};
  graph.add("opt_moe_route", {logits, expertIDs, routeWeights}, params, {rows, 1, 1}, {32, 1, 1});
  return true;
}

} // namespace splash::flash::opt

namespace splash::flash::opt {
namespace {
bool timersEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_TIMERS");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
struct TimerSlot { const char *label = nullptr; uint64_t count = 0, nanoseconds = 0; };
std::mutex &timerMutex() { static std::mutex mutex; return mutex; }
std::vector<TimerSlot> &timerSlots() { static std::vector<TimerSlot> slots; return slots; }
uint64_t nowNanoseconds() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

ScopedTimer::ScopedTimer(const char *label) noexcept : label_(label) {
  if (timersEnabled()) start_ = nowNanoseconds();
}
ScopedTimer::~ScopedTimer() {
  if (!start_) return;
  const uint64_t elapsed = nowNanoseconds() - start_;
  std::lock_guard lock(timerMutex());
  auto &slots = timerSlots();
  auto slot = std::find_if(slots.begin(), slots.end(),
      [&](const TimerSlot &candidate) { return std::strcmp(candidate.label, label_) == 0; });
  if (slot == slots.end()) { slots.push_back({label_, 0, 0}); slot = slots.end() - 1; }
  slot->count += 1; slot->nanoseconds += elapsed;
  static const uint64_t every = [] {
    const char *value = std::getenv("SPLASH_OPT_TIMERS_EVERY");
    return value && *value ? std::max<uint64_t>(1, std::strtoull(value, nullptr, 10)) : uint64_t{256};
  }();
  if (slot->count % every == 0)
    std::fprintf(stderr, "opt-timer %s count=%llu mean_us=%.1f\n", slot->label,
        static_cast<unsigned long long>(slot->count),
        double(slot->nanoseconds) / double(slot->count) / 1000.0);
}
} // namespace splash::flash::opt
