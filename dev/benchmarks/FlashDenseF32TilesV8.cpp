#include "FlashDenseF32TilesV8.hpp"
#include "metal/abi/FlashFloatDenseCache.h"
#include <array>
#include <limits>
#include <string>
namespace splash::flash::candidate {
namespace {
uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash float small-row dense byte extent overflows");
  return a * b;
}
void requireBuffer(const metal::MetalBuffer &b, uint64_t bytes, const char *name) {
  if (!b || !bytes || b.sizeBytes() < bytes || !b.contents())
    throw std::invalid_argument(std::string("Flash float small-row dense invalid Shared ") + name);
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const uintptr_t first = reinterpret_cast<uintptr_t>(a.contents());
  const uintptr_t second = reinterpret_cast<uintptr_t>(b.contents());
  if (!first || !second || first > UINTPTR_MAX - a.sizeBytes() ||
      second > UINTPTR_MAX - b.sizeBytes())
    throw std::invalid_argument("Flash float small-row dense invalid Shared address extent");
  return first < second + b.sizeBytes() && second < first + a.sizeBytes();
}
}
void addDenseF32TileV8(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weight,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashFloatDenseSmallRowsWorkspace &workspace,
                           uint32_t tile) {
  if (!workspace.belongsTo(backend) || !rows || rows > 16 ||
      weight.dtype != FlashDType::F32 || weight.shape.size() != 2 ||
      !weight.shape[0] || weight.shape[0] > UINT32_MAX || weight.shape[0] % 64 ||
      !weight.shape[1] || weight.shape[1] > workspace.maximumInputSize() || weight.shape[1] % 32)
    throw std::invalid_argument("Flash float small-row dense invalid matrix/rows/workspace");
  if (tile < 4) {
    addFloatDenseSmallRows(backend, graph, input, weight, output, diagnostics,
        rows, workspace, static_cast<FlashFloatDenseSmallRowsTile>(tile));
    return;
  }
  if (tile > 7) throw std::invalid_argument("invalid private F32 tile");
  constexpr std::array<uint32_t, 4> ms{8, 8, 16, 8};
  constexpr std::array<uint32_t, 4> ns{32, 32, 64, 64};
  constexpr std::array<uint32_t, 4> simds{2, 4, 8, 2};
  constexpr std::array<const char *, 4> pipelines{
      "flash_dense_f32_tiles_v8_m8_n32_s2", "flash_dense_f32_tiles_v8_m8_n32_s4",
      "flash_dense_f32_tiles_v8_m16_n64_s8", "flash_dense_f32_tiles_v8_m8_n64_s2"};
  const uint32_t m = ms[tile - 4], n = ns[tile - 4];
  const uint32_t k = static_cast<uint32_t>(weight.shape[1]);
  const uint32_t outputs = static_cast<uint32_t>(weight.shape[0]);
  const uint32_t paddedRows = (rows + m - 1) / m * m;
  const uint64_t weightBytes = product(product(outputs, k), 4);
  if (weight.logicalBytes < weightBytes)
    throw std::invalid_argument("Flash float small-row dense weight logical extent is short");
  requireBuffer(weight.buffer, weightBytes, "weights");
  requireBuffer(input, product(product(rows, k), 2), "input");
  requireBuffer(output, product(product(rows, outputs), 2), "output");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  requireBuffer(workspace.paddedInput(), product(product(paddedRows, k), 2), "padding");
  for (const auto &b : {input, output, diagnostics, workspace.paddedInput()})
    if (overlaps(b, weight.buffer))
      throw std::invalid_argument("Flash float small-row dense aliases immutable weights");
  if (overlaps(input, output) || overlaps(input, diagnostics) || overlaps(output, diagnostics) ||
      overlaps(workspace.paddedInput(), input) || overlaps(workspace.paddedInput(), output) ||
      overlaps(workspace.paddedInput(), diagnostics))
    throw std::invalid_argument("Flash float small-row dense buffer overlap");
  FlashFloatDenseSmallRowsParams params{rows, paddedRows, k, outputs, 0, outputs, m, 64};
  // The shared production pad validates its original N64/N128 geometry.
  // Padding does not depend on N; restore private N only for the matrix tile.
  graph.add("flash_float_dense_small_rows_pad", {input, workspace.paddedInput(), diagnostics}, params,
      {(product(paddedRows, k) - 1) / 256 + 1, 1, 1});
  params.tile_outputs = n;
  graph.add(pipelines[tile - 4], {workspace.paddedInput(), weight.buffer, output, diagnostics},
      params, {outputs / n, paddedRows / m, 1}, {simds[tile - 4] * 32, 1, 1});
}

}
