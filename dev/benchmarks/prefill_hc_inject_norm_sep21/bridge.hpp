#pragma once
#include "flash/FlashHC.hpp"
#include "metal/abi/FlashHCFused.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

// A private, separately named route. No public HC-down/up geometry is widened.
namespace splash::flash::prefill_hc_inject_norm_sep21 {
inline FlashHCFusedParams checkedParams(FlashHCGeometry g, const FlashTensor &weight,
                                      NormConvention convention) {
  if (g.rows < 512 || g.rows > 2048 || g.width != 2560 || g.streams != 4 ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0)
    throw std::invalid_argument("private prefill HC inject/norm requires R512..2048/H2560/S4");
  FlashHCFusedParams p{};
  p.rows = g.rows; p.width = g.width; p.streams = g.streams; p.lowrank = 320;
  p.simdgroups = 4; p.norm_epsilon = g.epsilon;
  p.norm_is_float = weight.dtype == FlashDType::F32;
  switch (convention) {
    case NormConvention::OnePlusWeight: p.norm_convention = 0; break;
    case NormConvention::DirectGamma: p.norm_convention = 1; break;
    default: throw std::invalid_argument("invalid private prefill HC norm convention");
  }
  const uint64_t count = uint64_t(g.width) * g.streams;
  if ((weight.dtype != FlashDType::BF16 && !p.norm_is_float) ||
      weight.shape != std::vector<uint64_t>{count} ||
      weight.logicalBytes != count * (p.norm_is_float ? 4 : 2))
    throw std::invalid_argument("invalid private prefill HC norm weight");
  return p;
}
inline void requireBytes(const metal::MetalBuffer &b, uint64_t bytes) {
  if (!b || b.sizeBytes() < bytes)
    throw std::invalid_argument("private prefill HC buffer is smaller than logical shape");
}
inline void requireDisjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto *ap = a.contents(); const auto *bp = b.contents();
  if (!ap || !bp)
    throw std::invalid_argument("private prefill HC requires shared addressable views");
  const auto aa = reinterpret_cast<uintptr_t>(ap), bb = reinterpret_cast<uintptr_t>(bp);
  const bool overlaps = aa <= bb ? uint64_t(bb - aa) < a.sizeBytes()
                                : uint64_t(aa - bb) < b.sizeBytes();
  if (overlaps) throw std::invalid_argument("unsupported private prefill HC buffer alias");
}
inline void addPrivatePrefillHCInjectNormSep21(metal::CommandGraph &graph,
    metal::MetalBuffer hyperInput, metal::MetalBuffer branch, metal::MetalBuffer gates,
    const FlashTensor &weight, metal::MetalBuffer updated, metal::MetalBuffer normalized,
    metal::MetalBuffer diagnostics, FlashHCGeometry g, NormConvention convention) {
  const auto p = checkedParams(g, weight, convention);
  const uint64_t hyperBytes = uint64_t(g.rows) * g.width * g.streams * 2;
  requireBytes(weight.buffer, weight.logicalBytes); requireBytes(hyperInput, hyperBytes);
  requireBytes(updated, hyperBytes); requireBytes(normalized, hyperBytes);
  requireBytes(branch, uint64_t(g.rows) * g.width * 2);
  requireBytes(gates, uint64_t(g.rows) * g.streams * 2); requireBytes(diagnostics, 4);
  for (const auto &b : {hyperInput, branch, gates, weight.buffer, updated, normalized})
    requireDisjoint(diagnostics, b);
  for (const auto &b : {branch, gates, weight.buffer}) requireDisjoint(updated, b);
  if (!updated.sameView(hyperInput)) requireDisjoint(updated, hyperInput);
  for (const auto &b : {hyperInput, updated, branch, gates, weight.buffer})
    requireDisjoint(normalized, b);
  graph.add("private_prefill_hc_inject_norm_sep21",
      {hyperInput, branch, gates, weight.buffer, updated, normalized, diagnostics},
      p, {g.rows, g.streams, 1}, {640, 1, 1});
}
} // namespace splash::flash::prefill_hc_inject_norm_sep21
