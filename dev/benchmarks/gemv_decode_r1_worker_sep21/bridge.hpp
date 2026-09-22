#pragma once
// Device-free policy for the isolated R1 numerical-alternative worker. The
// generated identity binds the sealed producer, certificate and shipping
// extraction. This bridge creates no model buffers or execution scratch.
#include "source_identity.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::gemv_decode_r1_sep21 {
inline constexpr const char *kFlag="SPLASH_FLASH_GEMV_DECODE_R1_SEP21";
inline constexpr std::string_view kSemantics=
    ";private-R1-nonverification-decode-only-numerical-alternative"
    ";direct-BF16xI8-vector4-L32O4-fourpartials-descendingXOR"
    ";kernel=gemv_decode_sep21_v4_l32_o4"
    ";cert=RNorRTZ-u23-FTZ4lambda-lateF32scale-BF16SwiGLU-v1b"
    ";sourceSha256=";

[[nodiscard]] constexpr bool sourceIdentityValid(std::string_view value) noexcept {
  if(value.size()!=64) return false;
  for(char c:value)
    if(!((c>='0'&&c<='9')||(c>='a'&&c<='f'))) return false;
  return true;
}
static_assert(sourceIdentityValid(kSourceIdentitySha256),
    "R1 worker requires the generated lowercase SHA256 source identity");

enum class SwitchState:uint8_t { Missing,Disabled,Enabled };
[[nodiscard]] inline SwitchState parseState(const char *raw) {
  if(!raw) return SwitchState::Missing;
  if(std::string_view(raw)=="0") return SwitchState::Disabled;
  if(std::string_view(raw)=="1") return SwitchState::Enabled;
  throw std::invalid_argument("SPLASH_FLASH_GEMV_DECODE_R1_SEP21 must be exactly 0 or 1 when present");
}
[[nodiscard]] inline bool parseSwitch(const char *raw) {
  return parseState(raw)==SwitchState::Enabled;
}
[[nodiscard]] inline bool requested() {
  // A throwing first initialization does not initialize the static, allowing
  // an invalid startup setting to be corrected and retried. Later reads check
  // the exact optional setting, including missing versus explicit zero.
  static const SwitchState frozen=parseState(std::getenv(kFlag));
  const SwitchState current=parseState(std::getenv(kFlag));
  if(current!=frozen)
    throw std::logic_error("SPLASH_FLASH_GEMV_DECODE_R1_SEP21 changed after policy was frozen");
  return frozen==SwitchState::Enabled;
}
[[nodiscard]] constexpr bool eligibleFor(uint32_t rows,bool verification,bool enabled) noexcept {
  return enabled&&rows==1&&!verification;
}
[[nodiscard]] inline bool eligible(uint32_t rows,bool verification) {
  return eligibleFor(rows,verification,requested());
}
[[nodiscard]] inline std::string markerFor(bool enabled) {
  if(!enabled) return {};
  std::string marker(kSemantics);
  marker+=kSourceIdentitySha256;
  return marker;
}
[[nodiscard]] inline std::string implementationMarker() {
  return markerFor(requested());
}
// These describe graph construction, rather than GPU completion. Store owns
// any atomics and snapshots them here without allocating GPU resources.
struct Counters final {
  bool enabled=false;
  uint64_t gateCalls=0,gateRows=0,downCalls=0,downRows=0;
};
} // namespace splash::flash::gemv_decode_r1_sep21
