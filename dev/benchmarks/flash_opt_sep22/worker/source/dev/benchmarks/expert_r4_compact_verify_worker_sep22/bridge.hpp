#pragma once
// Device-free policy for the private physical-R4 singleton-main verification
// route. Forward supplies caller scope; this bridge creates no GPU resources.
#include "source_identity.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::compact_native_r4_verify_sep22 {
inline constexpr const char *kFlag="SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22";
inline constexpr std::string_view kSemantics=
    ";private-physicalR4-singleton-main-verification-only"
    ";INTEGERONLY-compact-native-M16-exact-original-float-kernels-six-stages"
    ";planner=expert_r4_compact_native_sep22_plan-parallel-SIMDcount-TG2432B"
    ";native-pack-GU-excludedPoison-prepareDown-down-unchanged"
    ";component-exact-native-stages-no-model-quality-claim"
    ";sourceSha256=";
[[nodiscard]] constexpr bool sourceIdentityValid(std::string_view s) noexcept {
  if(s.size()!=64)return false;
  for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
}
static_assert(sourceIdentityValid(kSourceIdentitySha256),"Compact verify needs its generated lowercase SHA256 source identity");
enum class SwitchState:uint8_t {Missing,Disabled,Enabled};
[[nodiscard]] inline SwitchState parseState(const char *raw) {
  if(!raw)return SwitchState::Missing;
  if(std::string_view(raw)=="0")return SwitchState::Disabled;
  if(std::string_view(raw)=="1")return SwitchState::Enabled;
  throw std::invalid_argument("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22 must be exactly 0 or 1 when present");
}
[[nodiscard]] inline bool parseSwitch(const char *raw){return parseState(raw)==SwitchState::Enabled;}
[[nodiscard]] inline bool requested() {
  // Invalid first initialization throws without freezing; correcting the value
  // can retry. Every later call rechecks the exact optional environment state.
  static const SwitchState frozen=parseState(std::getenv(kFlag));
  if(parseState(std::getenv(kFlag))!=frozen)
    throw std::logic_error("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22 changed after policy was frozen");
  return frozen==SwitchState::Enabled;
}
[[nodiscard]] constexpr bool eligibleFor(uint32_t rows,bool verification,bool enabled) noexcept {
  return enabled&&rows==4&&verification;
}
[[nodiscard]] inline bool eligible(uint32_t rows,bool verification){return eligibleFor(rows,verification,requested());}
[[nodiscard]] constexpr bool dependenciesValid(bool full512,bool gather,uint32_t cap,
    bool blocked,bool directA,bool q4x8) noexcept {
  return full512&&gather&&cap==4&&blocked&&directA&&q4x8;
}
inline void validateDependenciesFor(bool enabled,bool full512,bool gather,uint32_t cap,
    bool blocked,bool directA,bool q4x8) {
  if(enabled&&!dependenciesValid(full512,gather,cap,blocked,directA,q4x8))
    throw std::invalid_argument("Compact native R4 verify requires Full512, gathered MPP, cap exactly4, original blocked, direct A and Q4x8");
}
inline void validateDependencies(bool full512,bool gather,uint32_t cap,bool blocked,bool directA,bool q4x8) {
  validateDependenciesFor(requested(),full512,gather,cap,blocked,directA,q4x8);
}
[[nodiscard]] inline std::string markerFor(bool enabled) {
  if(!enabled)return {};
  std::string marker(kSemantics);marker+=kSourceIdentitySha256;return marker;
}
[[nodiscard]] inline std::string implementationMarker(){return markerFor(requested());}
// Store owns any atomics; these snapshots count graph construction only.
struct Counters final {
  bool enabled=false;
  uint64_t planCalls=0,planRows=0,gateCalls=0,gateRows=0,downCalls=0,downRows=0;
};
} // namespace splash::flash::compact_native_r4_verify_sep22
