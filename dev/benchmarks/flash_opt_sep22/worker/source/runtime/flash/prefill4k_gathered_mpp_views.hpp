#pragma once

// One shared host/shader ABI. This is copied only into a private source tree.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif

struct FlashGatheredMPPViewParams {
  uint32_t rows, selections, experts, reserved;
};
static_assert(sizeof(FlashGatheredMPPViewParams) == 16);

#ifndef __METAL_VERSION__
#include <array>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>

namespace splash::flash::gathered_mpp_view {
inline constexpr std::string_view kPolicy =
    "private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c1-rows1to16-v1";
inline constexpr uint32_t kSimds = 4, kColumns = 4, kThreads = 128;
inline constexpr uint32_t kInvalidOriginalExpertID = 1, kInvalidGeometry = 2, kInvalidNumerics = 4;
inline constexpr const char *kFlag = "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV";
inline bool requested() {
  const char *raw = std::getenv(kFlag);
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV must be exactly0 or1");
}

// Descriptor checks use metadata only. Runtime IDs belong to a preceding GPU
// route producer, so graph construction must never read them on the CPU.
inline constexpr bool validGeometry(uint32_t rows, uint32_t selections,
                                    uint32_t experts = 512) {
  return rows >= 1 && rows <= 16 && selections == 10 && experts == 512;
}
struct Geometry {
  uint64_t inputBytes, idBytes, intermediateBytes, expertDownBytes;
  uint32_t gateColumnGroups, downColumnGroups, rows, selections;
};
inline Geometry geometry(uint32_t rows, uint32_t selections = 10) {
  if (!validGeometry(rows, selections))
    throw std::invalid_argument("private gathered I8 QMV requires rows1..16 E512 H2560 I640 K10");
  const uint64_t routes = uint64_t{rows} * selections;
  return {uint64_t{rows} * 2560 * 2, routes * 8, routes * 640 * 2,
      routes * 2560 * 2, 640 / kColumns, 2560 / kColumns, rows, selections};
}
struct ByteView {
  uintptr_t address;
  uint64_t bytes;
};
inline bool validView(ByteView view, uint64_t required, uint64_t alignment) {
  return view.address && required && alignment && view.address % alignment == 0 &&
      view.bytes >= required && view.bytes <= std::numeric_limits<uintptr_t>::max() - view.address;
}
inline bool overlaps(ByteView a, ByteView b) {
  // Both ends must have been admitted by validView; subtraction avoids overflow.
  return a.address <= b.address ? uint64_t(b.address - a.address) < a.bytes
                               : uint64_t(a.address - b.address) < b.bytes;
}
inline void validateViews(const Geometry &g, ByteView input, ByteView ids,
                          ByteView output, ByteView diagnostics, bool down) {
  if (!validView(input, down ? g.intermediateBytes : g.inputBytes, 2) ||
      !validView(ids, g.idBytes, 8) ||
      !validView(output, down ? g.expertDownBytes : g.intermediateBytes, 2) ||
      !validView(diagnostics, 4, 4))
    throw std::invalid_argument("private gathered I8 QMV invalid view size, alignment, or address");
  const std::array views{input, ids, output, diagnostics};
  for (size_t i = 0; i < views.size(); ++i)
    for (size_t j = i + 1; j < views.size(); ++j)
      if (overlaps(views[i], views[j]))
        throw std::invalid_argument("private gathered I8 QMV operands must be disjoint");
}
inline uint32_t checkedRank(int64_t id, std::span<const uint32_t> ranks,
                            uint32_t &stickyDiagnostics) {
  if (id < 0 || id >= 512 || ranks.size() != 512 || ranks[size_t(id)] >= 512) {
    stickyDiagnostics |= kInvalidOriginalExpertID;
    return UINT32_MAX;
  }
  return ranks[size_t(id)];
}
// A fixture/reference guard only. Deliberately not called by graph construction.
inline bool fixtureIDsValid(std::span<const int64_t> ids, uint32_t rows,
                            uint32_t selections = 10) {
  if (!validGeometry(rows, selections) || ids.size() != uint64_t{rows} * selections) return false;
  for (int64_t id : ids) if (id < 0 || id >= 512) return false;
  return true;
}
inline uint32_t fixtureIDDiagnostics(std::span<const int64_t> ids, uint32_t rows,
                                     uint32_t sticky = 0, uint32_t selections = 10) {
  if (!validGeometry(rows, selections) || ids.size() != uint64_t{rows} * selections)
    return sticky | kInvalidGeometry;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t slot = 0; slot < selections; ++slot) {
      const int64_t id = ids[uint64_t{row} * selections + slot];
      if (id < 0 || id >= 512) sticky |= kInvalidOriginalExpertID;
      else for (uint32_t prior = 0; prior < slot; ++prior)
        if (ids[uint64_t{row} * selections + prior] == id) sticky |= kInvalidOriginalExpertID;
    }
  return sticky;
}
} // namespace splash::flash::gathered_mpp_view
#endif
