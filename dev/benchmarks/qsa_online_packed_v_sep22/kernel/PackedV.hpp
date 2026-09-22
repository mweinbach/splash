#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
struct QSAOnlinePackedVParams { uint rows, kvHeads, dimensions, reserved; };
#else
#include <cstdint>
struct QSAOnlinePackedVParams {
  std::uint32_t rows, kvHeads, dimensions, reserved;
};
namespace splash::flash::qsa_online_packed_v_sep22 {
constexpr bool eligibleFor(bool mainPrefill, bool verification, bool fresh,
                           std::uint32_t begin, std::uint32_t rows,
                           std::uint32_t capacity, bool originalOnlineSG8) {
  return mainPrefill && !verification && fresh && !begin && rows == 2048 &&
         capacity >= 2048 && capacity <= 262144 && originalOnlineSG8;
}
constexpr std::uint64_t logicalPackedBytes = 2048ULL * 2 * 256 * 2;
}
#endif
static_assert(sizeof(QSAOnlinePackedVParams) == 16, "Private packed-V inline ABI");
