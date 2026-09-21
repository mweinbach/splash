#include "FlashSharedExpertFused.hpp"
#include "flash/FlashMoE.hpp"
#include "metal/abi/FlashDenseCache.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace splash::flash::candidate {
namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::invalid_argument(std::string("Shared expert fusion: ") + reason);
}
void bytes(const metal::MetalBuffer &buffer, uint64_t expected) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= expected,
      "Shared buffer extent is insufficient");
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto x = reinterpret_cast<uintptr_t>(a.contents());
  const auto y = reinterpret_cast<uintptr_t>(b.contents());
  require(x && y && x <= std::numeric_limits<uintptr_t>::max() - a.sizeBytes() &&
      y <= std::numeric_limits<uintptr_t>::max() - b.sizeBytes(), "invalid Shared address extent");
  return x < y + b.sizeBytes() && y < x + a.sizeBytes();
}
void tensor(const FlashTensor &value) {
  require(value.dtype == FlashDType::BF16 &&
      value.shape == std::vector<uint64_t>{640, 2560} &&
      value.logicalBytes == 640ULL * 2560 * 2, "requires original cached shared-expert BF16 operands");
  bytes(value.buffer, value.logicalBytes);
}
void validate(const FlashTensor &gate, const FlashTensor &up,
    const metal::MetalBuffer &input, const metal::MetalBuffer &activated,
    const metal::MetalBuffer &diagnostics, uint32_t rows) {
  require(rows >= 256 && rows <= 8192, "rows must be256..8192");
  tensor(gate); tensor(up);
  bytes(input, uint64_t{rows} * 2560 * 2);
  bytes(activated, uint64_t{rows} * 640 * 2); bytes(diagnostics, 4);
  require(!overlaps(gate.buffer, up.buffer), "gate/up operand planes overlap");
  for (const auto &weight : {gate.buffer, up.buffer})
    for (const auto &write : {input, activated, diagnostics})
      require(!overlaps(weight, write), "input/writer overlaps immutable operand plane");
  require(!overlaps(activated, input) && !overlaps(diagnostics, input) &&
      !overlaps(diagnostics, activated), "writer/input overlap");
}
void complete(metal::CommandGraph &graph, const FlashTensor &gate,
    const FlashTensor &up, metal::MetalBuffer input, metal::MetalBuffer activated,
    metal::MetalBuffer diagnostic, uint32_t rows, FlashAffineMPPTile tile,
    const metal::MetalBuffer &gateTap = {}, const metal::MetalBuffer &upTap = {}) {
  const uint32_t m = sharedExpertFusedTileRows(tile), n = sharedExpertFusedTileOutputs(tile);
  const FlashDenseCacheParams p{rows, 2560, 640, 0, 640, m, n, 0};
  const std::string name = "flash_shared_expert_fused_" + std::string(gateTap ? "taps_" : "")
      + "m" + std::to_string(m) + "_n" + std::to_string(n);
  if (gateTap) graph.add(name, {input, gate.buffer, up.buffer, activated, diagnostic,
      gateTap, upTap}, p, {640 / n, rows / m, 1}, {128, 1, 1});
  else graph.add(name, {input, gate.buffer, up.buffer, activated, diagnostic}, p,
      {640 / n, rows / m, 1}, {128, 1, 1});
}
} // namespace

uint32_t sharedExpertFusedTileRows(FlashAffineMPPTile tile) {
  switch (tile) {
  case FlashAffineMPPTile::M16N64:
  case FlashAffineMPPTile::M16N128: return 16;
  case FlashAffineMPPTile::M32N64:
  case FlashAffineMPPTile::M32N128: return 32;
  default: throw std::invalid_argument("Shared expert fusion: unsupported descriptor");
  }
}
uint32_t sharedExpertFusedTileOutputs(FlashAffineMPPTile tile) {
  switch (tile) {
  case FlashAffineMPPTile::M16N64:
  case FlashAffineMPPTile::M32N64: return 64;
  case FlashAffineMPPTile::M16N128:
  case FlashAffineMPPTile::M32N128: return 128;
  default: throw std::invalid_argument("Shared expert fusion: unsupported descriptor");
  }
}

void addSharedExpertFused(metal::MetalBackend &backend, metal::CommandGraph &graph,
    const FlashTensor &gate, const FlashTensor &up, metal::MetalBuffer input,
    metal::MetalBuffer activated, metal::MetalBuffer diagnostics, uint32_t rows,
    FlashAffineMPPTile tile, const SharedExpertFusedTail &tail) {
  validate(gate, up, input, activated, diagnostics, rows);
  const uint32_t m = sharedExpertFusedTileRows(tile);
  const uint32_t fullRows = rows / m * m, remaining = rows - fullRows;
  if (remaining) {
    bytes(tail.gate, uint64_t{remaining} * 640 * 2);
    bytes(tail.up, uint64_t{remaining} * 640 * 2);
    require(!overlaps(tail.gate, tail.up), "tail gate/up overlap");
    for (const auto &temporary : {tail.gate, tail.up})
      for (const auto &value : {input, activated, diagnostics, gate.buffer, up.buffer})
        require(!overlaps(temporary, value), "tail scratch overlap");
  }
  complete(graph, gate, up, input, activated, diagnostics, fullRows, tile);
  if (remaining) {
    const auto tailInput = backend.view(input, uint64_t{fullRows} * 2560 * 2,
        uint64_t{remaining} * 2560 * 2);
    const auto tailOutput = backend.view(activated, uint64_t{fullRows} * 640 * 2,
        uint64_t{remaining} * 640 * 2);
    addDenseBF16WholeK(backend, graph, tailInput, gate, tail.gate, diagnostics,
        remaining, tile);
    addDenseBF16WholeK(backend, graph, tailInput, up, tail.up, diagnostics,
        remaining, tile);
    addSiLUMultiply(graph, tail.gate, tail.up, tailOutput, diagnostics, remaining, 640);
  }
}

void addSharedExpertFusedTaps(metal::MetalBackend &backend, metal::CommandGraph &graph,
    const FlashTensor &gate, const FlashTensor &up, metal::MetalBuffer input,
    metal::MetalBuffer activated, metal::MetalBuffer diagnostics, uint32_t rows,
    FlashAffineMPPTile tile, metal::MetalBuffer gateTap, metal::MetalBuffer upTap) {
  validate(gate, up, input, activated, diagnostics, rows);
  const uint32_t m = sharedExpertFusedTileRows(tile);
  bytes(gateTap, uint64_t{rows} * 640 * 2); bytes(upTap, uint64_t{rows} * 640 * 2);
  require(!overlaps(gateTap, upTap), "tap planes overlap");
  for (const auto &tap : {gateTap, upTap})
    for (const auto &value : {input, activated, diagnostics, gate.buffer, up.buffer})
      require(!overlaps(tap, value), "tap overlap");
  const uint32_t fullRows = rows / m * m, remaining = rows - fullRows;
  complete(graph, gate, up, input, activated, diagnostics, fullRows, tile, gateTap, upTap);
  if (remaining) {
    const auto tailInput = backend.view(input, uint64_t{fullRows} * 2560 * 2,
        uint64_t{remaining} * 2560 * 2);
    const auto tailGate = backend.view(gateTap, uint64_t{fullRows} * 640 * 2,
        uint64_t{remaining} * 640 * 2);
    const auto tailUp = backend.view(upTap, uint64_t{fullRows} * 640 * 2,
        uint64_t{remaining} * 640 * 2);
    const auto tailOutput = backend.view(activated, uint64_t{fullRows} * 640 * 2,
        uint64_t{remaining} * 640 * 2);
    addDenseBF16WholeK(backend, graph, tailInput, gate, tailGate, diagnostics,
        remaining, tile);
    addDenseBF16WholeK(backend, graph, tailInput, up, tailUp, diagnostics,
        remaining, tile);
    addSiLUMultiply(graph, tailGate, tailUp, tailOutput, diagnostics, remaining, 640);
  }
}

} // namespace splash::flash::candidate
