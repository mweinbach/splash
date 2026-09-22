// CPU-only host policy/grid coverage; no Metal backend or GPU submission.
#include "flash/FlashDenseTraversal.hpp"
#include "metal/abi/FlashDenseCache.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static_assert(sizeof(FlashDenseCacheParams) == 32);
static_assert(offsetof(FlashDenseCacheParams, reserved) == 28);
using namespace splash::flash;
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
} // namespace

int main() {
  try {
    uint64_t checks = 0;
    require(flashDenseTraversalPolicy(2048,2560,6144,64,128) == FlashDenseTraversal::RowFast,
            "attention output traversal not selected"); ++checks;
    require(flashDenseTraversalPolicy(512,6144,2560,32,128) == FlashDenseTraversal::Swizzle8,
            "wide projection traversal not selected"); ++checks;
    require(flashDenseTraversalPolicy(512,2560,2560,32,128) == FlashDenseTraversal::Swizzle4,
            "square projection traversal not selected"); ++checks;
    require(flashDenseTraversalPolicy(2048,320,2560,16,64) == FlashDenseTraversal::ColumnFast,
            "non-model synthetic shape selected"); ++checks;
    for (const auto &shape : std::array<std::array<uint32_t,5>,3>{{
        {2048,2560,6144,64,128}, {512,6144,2560,32,128},
        {512,2560,2560,32,128}}}) {
      for (uint32_t field = 0; field < shape.size(); ++field) {
        for (int32_t delta : {-1,1}) {
          auto changed = shape; changed[field] = uint32_t(int64_t(changed[field]) + delta);
          require(flashDenseTraversalPolicy(changed[0],changed[1],changed[2],changed[3],changed[4]) ==
                  FlashDenseTraversal::ColumnFast, "policy extrapolated beyond a measured shape");
          ++checks;
        }
      }
    }
    for (uint32_t rowTiles : {1u,2u,3u,7u,15u,16u,31u,32u,64u,127u,128u}) {
      for (uint32_t columnTiles : {1u,2u,3u,5u,20u,48u,81u,128u}) {
        for (uint32_t mode = 0; mode <= 4; ++mode) {
          const auto grid = flashDenseTraversalGrid(rowTiles,columnTiles,
              static_cast<FlashDenseTraversal>(mode));
          std::vector<uint32_t> visits(uint64_t(rowTiles)*columnTiles);
          for (uint32_t y = 0; y < grid.y; ++y) for (uint32_t x = 0; x < grid.x; ++x) {
            uint32_t row = y, column = x;
            if (mode == 1) { row = x; column = y; }
            else if (mode >= 2) {
              const uint32_t log = mode - 1;
              row = (y << log) + (x & ((1u << log)-1)); column = x >> log;
            }
            if (row < rowTiles && column < columnTiles)
              ++visits[uint64_t(row)*columnTiles+column];
          }
          for (uint32_t value : visits) require(value == 1, "grid lost or duplicated a logical tile");
          ++checks;
        }
      }
    }
    const auto invalid = flashDenseTraversalGrid(32,20,static_cast<FlashDenseTraversal>(5));
    require(invalid.x == 0 && invalid.y == 0, "invalid traversal mode produced a dispatch"); ++checks;
    std::cout << "{\"dense_traversal_cpu_policy\":\"passed\",\"checks\":" << checks << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
