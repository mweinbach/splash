#pragma once
// Root-only temporary pairing of existing saved-I8 Gate/Up code rows.
// Call pack() only after reserving pairPlannedBytes() in addition to the
// already-loaded oneLayerPlannedBytes(). No loader, mmap, full-store constructor
// or GPU submission is called here. Initialization, full byte certification,
// SHA witnesses and unchanged() scans belong OUTSIDE the timed command train.
// Exact code rearrangement is not a GPU arithmetic/runtime qualification:
// BF16 raw/scaled/full-chain equality and numerical W8 qualification remain
// separate oracle obligations. Original code/scale/rank bytes are untouched.
#include "prefill4k_allrows_qmv_one_layer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::bench::paired_gate {
namespace one = splash::flash::qmv_one_layer;
using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
inline constexpr uint64_t kAlignment = 16384, kGuardBytes = 128;
inline constexpr uint8_t kCanary = 0x5a;
inline constexpr uint64_t kStoredExperts = 512, kOutputsPerSide = 640, kInputSize = 2560;
inline constexpr uint64_t kOutputsPerBlock = 64;
inline constexpr uint64_t kSourceCodeBytes = 838860800ULL;
inline constexpr uint64_t kPairedCodeBytes = 1677721600ULL;

inline void require(bool value, const char *message) {
  if (!value) throw std::invalid_argument(std::string("private paired Gate: ") + message);
}
inline uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  require(!a || b <= std::numeric_limits<uint64_t>::max() / a, "packing geometry multiplication overflow");
  return a * b;
}
inline uint64_t checkedAdd(uint64_t a, uint64_t b) {
  require(b <= std::numeric_limits<uint64_t>::max() - a, "packing geometry addition overflow");
  return a + b;
}
inline uint64_t roundedBytes(uint64_t bytes) {
  require(bytes > 0, "empty allocation extent");
  return checkedAdd(bytes, kAlignment - 1) & ~(kAlignment - 1);
}
struct Geometry {
  uint64_t experts = kStoredExperts, outputsPerSide = kOutputsPerSide, inputSize = kInputSize;
};
struct Layout {
  uint64_t sourceRows = 0, pairedRows = 0, sourceBytes = 0, pairedBytes = 0, blocks = 0;
};
inline Layout layout(Geometry geometry) {
  require(geometry.experts > 0 && geometry.outputsPerSide > 0 && geometry.inputSize > 0,
      "empty packing geometry");
  require(geometry.outputsPerSide % kOutputsPerBlock == 0, "outputs must consist of complete64-column blocks");
  Layout result;
  result.sourceRows = checkedMultiply(geometry.experts, geometry.outputsPerSide);
  result.pairedRows = checkedMultiply(result.sourceRows, 2);
  result.sourceBytes = checkedMultiply(result.sourceRows, geometry.inputSize);
  result.pairedBytes = checkedMultiply(result.pairedRows, geometry.inputSize);
  result.blocks = geometry.outputsPerSide / kOutputsPerBlock;
  require(result.pairedBytes <= std::numeric_limits<size_t>::max(), "packing extent exceeds host size_t");
  return result;
}
[[nodiscard]] inline uint64_t pairPlannedBytes() {
  return roundedBytes(checkedAdd(kPairedCodeBytes, 2 * kGuardBytes));
}
[[nodiscard]] inline uint64_t pairPlannedBytes(Geometry geometry) {
  return roundedBytes(checkedAdd(layout(geometry).pairedBytes, 2 * kGuardBytes));
}
inline uint64_t pairedRow(Geometry geometry, uint64_t rank, uint64_t side, uint64_t column) {
  const auto shape = layout(geometry);
  require(rank < geometry.experts && side < 2 && column < geometry.outputsPerSide, "pair index outside geometry");
  const uint64_t row = rank * (2 * geometry.outputsPerSide) + (column / 64) * 128 + side * 64 + column % 64;
  require(row < shape.pairedRows, "paired row outside extent");
  return row;
}
struct OriginalRow { uint64_t rank = 0, side = 0, column = 0; };
inline OriginalRow inverseRow(Geometry geometry, uint64_t row) {
  const auto shape = layout(geometry);
  require(row < shape.pairedRows, "inverse row outside geometry");
  const uint64_t local = row % (2 * geometry.outputsPerSide);
  return {row / (2 * geometry.outputsPerSide), (local % 128) / 64,
          (local / 128) * 64 + local % 64};
}
inline bool rangesOverlap(const void *a, uint64_t aBytes, const void *b, uint64_t bBytes) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  require(a && b && aBytes && bBytes, "packing requires addressable nonempty spans");
  require(aBytes <= std::numeric_limits<uintptr_t>::max() - x && bBytes <= std::numeric_limits<uintptr_t>::max() - y,
      "packing pointer extent overflow");
  return x < y + bBytes && y < x + aBytes;
}
inline void validateSpans(Geometry geometry, std::span<const uint8_t> gate,
    std::span<const uint8_t> up, std::span<const uint8_t> paired) {
  const auto shape = layout(geometry);
  require(gate.size() == shape.sourceBytes && up.size() == shape.sourceBytes && paired.size() == shape.pairedBytes,
      "code-plane spans differ from geometry");
  require(!rangesOverlap(gate.data(), gate.size(), paired.data(), paired.size()) &&
      !rangesOverlap(up.data(), up.size(), paired.data(), paired.size()), "destination aliases original source codes");
}
// Verification enumerates paired rows and uses the inverse mapping, rather
// than replaying the original-side iteration used for copying.
inline uint64_t verifyRows(Geometry geometry, std::span<const uint8_t> gate,
    std::span<const uint8_t> up, std::span<const uint8_t> paired) {
  validateSpans(geometry, gate, up, paired);
  const auto shape = layout(geometry);
  for (uint64_t row = 0; row < shape.pairedRows; ++row) {
    const auto original = inverseRow(geometry, row);
    const auto source = original.side ? up : gate;
    const uint64_t sourceRow = original.rank * geometry.outputsPerSide + original.column;
    require(!std::memcmp(paired.data() + row * geometry.inputSize,
        source.data() + sourceRow * geometry.inputSize, size_t(geometry.inputSize)), "inverse paired-row byte certificate differs");
  }
  return shape.pairedBytes;
}
inline uint64_t packRows(Geometry geometry, std::span<const uint8_t> gate,
    std::span<const uint8_t> up, std::span<uint8_t> paired) {
  validateSpans(geometry, gate, up, paired);
  for (uint64_t rank = 0; rank < geometry.experts; ++rank)
    for (uint64_t side = 0; side < 2; ++side)
      for (uint64_t column = 0; column < geometry.outputsPerSide; ++column) {
        const auto source = side ? up : gate;
        const uint64_t sourceRow = rank * geometry.outputsPerSide + column;
        const uint64_t destinationRow = pairedRow(geometry, rank, side, column);
        std::memcpy(paired.data() + destinationRow * geometry.inputSize,
            source.data() + sourceRow * geometry.inputSize, size_t(geometry.inputSize));
      }
  return verifyRows(geometry, gate, up, paired);
}

struct PairPayload final {
  MetalBuffer base, paired;
  uint64_t plannedBytes = 0, allocatedBytes = 0, codeBytesCertified = 0;
  std::string pairedSHA;
  std::array<std::string, 2> originalCodeSHA;
  one::OneLayerPayload::ImmutableHashes sourceImmutableWitness;
  uint32_t sourceLayer = 0;

  [[nodiscard]] static PairPayload pack(MetalBackend &backend, const one::OneLayerPayload &source) {
    validateSource(source);
    const uint64_t plan = pairPlannedBytes();
    require(plan <= backend.capabilities().maxBufferLengthBytes, "paired base exceeds Metal buffer limit");
    PairPayload result; result.plannedBytes = plan; result.sourceLayer = source.layer;
    result.sourceBase_ = source.base; result.sourceRanks_ = source.ranks;
    result.sourceCodes_ = {source.codes[0], source.codes[1]};
    result.sourceImmutableWitness = source.immutableHashes();
    for (uint32_t side = 0; side < 2; ++side)
      result.originalCodeSHA[side] = one::detail::hash(source.codes[side].contents(), kSourceCodeBytes);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    result.base = backend.allocateBuffer(plan, BufferStorage::Shared, "Root temporary guarded paired Gate/Up I8 plane");
    result.paired = backend.view(result.base, kGuardBytes, kPairedCodeBytes);
    auto *bytes = static_cast<uint8_t *>(result.base.contents());
    require(bytes && result.paired.contents(), "paired Shared buffer is not addressable");
    // Initialize canaries only; every logical byte is subsequently copied
    // exactly once. The page slack is protected by the tail canary as well.
    std::memset(bytes, kCanary, size_t(kGuardBytes));
    std::memset(bytes + kGuardBytes + kPairedCodeBytes, kCanary,
        size_t(plan - kGuardBytes - kPairedCodeBytes));
    result.codeBytesCertified = packRows({},
        {static_cast<const uint8_t *>(source.codes[0].contents()), size_t(kSourceCodeBytes)},
        {static_cast<const uint8_t *>(source.codes[1].contents()), size_t(kSourceCodeBytes)},
        {static_cast<uint8_t *>(result.paired.contents()), size_t(kPairedCodeBytes)});
    result.pairedSHA = one::detail::hash(result.paired.contents(), kPairedCodeBytes);
    source.checkImmutableHashes(result.sourceImmutableWitness);
    require(result.guardsClean(), "packing changed a paired-plane canary");
    result.allocatedBytes = splash::metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
    require(result.allocatedBytes <= plan, "paired allocation exceeds reserved plan");
    return result;
  }
  [[nodiscard]] bool guardsClean() const {
    if (!base || !paired || base.storage() != BufferStorage::Shared || paired.storage() != BufferStorage::Shared ||
        base.sizeBytes() != pairPlannedBytes() || paired.sizeBytes() != kPairedCodeBytes || !base.contents() || !paired.contents()) return false;
    const auto *bytes = static_cast<const uint8_t *>(base.contents());
    if (paired.contents() != bytes + kGuardBytes) return false;
    return std::all_of(bytes, bytes + kGuardBytes, [](uint8_t value) { return value == kCanary; }) &&
        std::all_of(bytes + kGuardBytes + kPairedCodeBytes, bytes + base.sizeBytes(),
            [](uint8_t value) { return value == kCanary; });
  }
  void unchanged(const one::OneLayerPayload &source) const {
    validateSource(source);
    require(source.layer == sourceLayer && source.base.sameView(sourceBase_) && source.ranks.sameView(sourceRanks_) &&
        source.codes[0].sameView(sourceCodes_[0]) && source.codes[1].sameView(sourceCodes_[1]), "original source views changed identity");
    require(guardsClean() && codeBytesCertified == kPairedCodeBytes && plannedBytes == pairPlannedBytes() &&
        allocatedBytes <= plannedBytes, "paired metadata/canaries changed");
    require(one::detail::hash(paired.contents(), kPairedCodeBytes) == pairedSHA, "paired code bytes changed during Root timing");
    source.checkImmutableHashes(sourceImmutableWitness);
  }
private:
  MetalBuffer sourceBase_, sourceRanks_;
  std::array<MetalBuffer, 2> sourceCodes_;
  static void validateSource(const one::OneLayerPayload &source) {
    require(source.layer < 48 && source.base && source.ranks && source.base.storage() == BufferStorage::Shared &&
        source.ranks.storage() == BufferStorage::Shared && source.base.contents() && source.ranks.contents() &&
        source.base.sizeBytes() == one::kFull512LayerBytes && source.ranks.sizeBytes() == one::kAlignment &&
        source.plannedBytes == one::kFull512LayerBytes + one::kAlignment, "source is not the canonical loaded one-layer payload");
    const auto *data = static_cast<const uint8_t *>(source.base.contents()); uint64_t cursor = 0;
    for (uint32_t projection = 0; projection < 3; ++projection) {
      const uint64_t n = projection == 2 ? 2560 : 640, k = projection == 2 ? 640 : 2560;
      cursor = cursor ? roundedBytes(cursor) : 0;
      const auto &code = source.codes[projection]; const uint64_t codeBytes = 512 * n * k;
      require(code && code.storage() == BufferStorage::Shared && code.sizeBytes() == codeBytes &&
          code.contents() == data + cursor, "source code-plane shape/offset differs"); cursor += codeBytes;
      cursor = roundedBytes(cursor);
      const auto &scale = source.scales[projection]; const uint64_t scaleBytes = 512 * n * sizeof(float);
      require(scale && scale.storage() == BufferStorage::Shared && scale.sizeBytes() == scaleBytes &&
          scale.contents() == data + cursor, "source scale-plane shape/offset differs"); cursor += scaleBytes;
    }
    require(roundedBytes(cursor) == source.base.sizeBytes(), "source base plane inventory differs");
  }
};

inline void cpuSelfTest() {
  const auto mustFail = [](const auto &operation) {
    bool rejected = false; try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "malformed geometry was accepted");
  };
  require(kStoredExperts * 2 * kOutputsPerSide * kInputSize == kPairedCodeBytes &&
      pairPlannedBytes() == 1677737984ULL, "canonical pair ledger differs");
  const Geometry small{2, 128, 8}; const auto shape = layout(small);
  require(shape.pairedBytes == 4096, "small synthetic certificate size differs");
  std::vector<uint8_t> gate(shape.sourceBytes), up(shape.sourceBytes), paired(shape.pairedBytes), visits(shape.pairedBytes);
  uint32_t random = 0x41c64e6du;
  for (auto *plane : {&gate, &up}) for (auto &value : *plane) { random = random * 1664525u + 1013904223u; value = uint8_t(random >> 24); }
  const auto gateBefore = gate, upBefore = up;
  require(packRows(small, gate, up, paired) == shape.pairedBytes, "small paired byte certificate is incomplete");
  for (uint64_t rank = 0; rank < small.experts; ++rank)
    for (uint64_t side = 0; side < 2; ++side)
      for (uint64_t column = 0; column < small.outputsPerSide; ++column) {
        const uint64_t row = pairedRow(small, rank, side, column); const auto inverse = inverseRow(small, row);
        require(inverse.rank == rank && inverse.side == side && inverse.column == column, "small inverse table differs");
        for (uint64_t k = 0; k < small.inputSize; ++k) require(++visits[row * small.inputSize + k] == 1, "paired cell duplicate");
      }
  require(std::all_of(visits.begin(), visits.end(), [](uint8_t value) { return value == 1; }) && gate == gateBefore && up == upBefore,
      "small bijection/source immutability failed");
  paired[77] ^= 1; mustFail([&] { (void)verifyRows(small, gate, up, paired); }); paired[77] ^= 1;
  mustFail([] { (void)layout({0, 128, 8}); }); mustFail([] { (void)layout({2, 127, 8}); });
  mustFail([] { (void)layout({2, 128, 0}); });
  mustFail([] { (void)layout({UINT64_MAX, 64, 1}); });
  mustFail([] { (void)layout({1, 64, UINT64_MAX}); });
  mustFail([] { (void)roundedBytes(UINT64_MAX); });
  mustFail([&] { (void)pairedRow(small, 2, 0, 0); }); mustFail([&] { (void)pairedRow(small, 0, 2, 0); });
  mustFail([&] { (void)inverseRow(small, shape.pairedRows); });
  mustFail([&] { (void)packRows(small, {gate.data(), gate.size() - 1}, up, paired); });
  mustFail([&] { (void)packRows(small, {paired.data(), size_t(shape.sourceBytes)}, up, paired); });
}
} // namespace splash::bench::paired_gate
