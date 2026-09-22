#pragma once
// Device-free policy for the private physical-R8/R16 batch-target verification
// route. BatchVerify supplies caller scope; this bridge creates no GPU resources.
#include "source_identity.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::compact_native_batch_verify_sep22 {
inline constexpr const char *kFlag="SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22";
inline constexpr std::string_view kSemantics=
    ";private-physicalR8R16-batch-target-verification-rows4-only"
    ";INTEGERONLY-compact-native-M16-exact-original-float-kernels-six-stages"
    ";planner=expert_batch_r8r16_compact_native_sep22_plan-parallel-SIMDcount"
    ";native-pack-GU-excludedPoison-prepareDown-down-unchanged"
    ";actual-active-lanes2or4-partial3or1-and-rows1to3-original"
    ";component-exact-native-stages-no-model-quality-claim"
    ";sourceSha256=";
[[nodiscard]] constexpr bool sourceIdentityValid(std::string_view s) noexcept {
  if(s.size()!=64)return false;
  for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
}
static_assert(sourceIdentityValid(kSourceIdentitySha256),"Compact batch verify needs its generated lowercase SHA256 source identity");
enum class SwitchState:uint8_t {Missing,Disabled,Enabled};
[[nodiscard]] inline SwitchState parseState(const char *raw) {
  if(!raw)return SwitchState::Missing;
  if(std::string_view(raw)=="0")return SwitchState::Disabled;
  if(std::string_view(raw)=="1")return SwitchState::Enabled;
  throw std::invalid_argument("SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22 must be exactly 0 or 1 when present");
}
[[nodiscard]] inline bool parseSwitch(const char *raw){return parseState(raw)==SwitchState::Enabled;}
[[nodiscard]] inline bool requested() {
  // Invalid first initialization throws without freezing; correcting the value
  // can retry. Every later call rechecks the exact optional environment state.
  static const SwitchState frozen=parseState(std::getenv(kFlag));
  if(parseState(std::getenv(kFlag))!=frozen)
    throw std::logic_error("SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22 changed after policy was frozen");
  return frozen==SwitchState::Enabled;
}
[[nodiscard]] constexpr bool eligibleFor(uint32_t lanes,uint32_t rows,bool verification,bool enabled) noexcept {
  return enabled&&verification&&rows==4&&(lanes==2||lanes==4);
}
[[nodiscard]] inline bool eligible(uint32_t lanes,uint32_t rows,bool verification){return eligibleFor(lanes,rows,verification,requested());}
[[nodiscard]] constexpr bool dependenciesValid(bool full512,bool gather,uint32_t cap,
    bool blocked,bool directA,bool q4x8,bool pointwise) noexcept {
  return full512&&gather&&cap==4&&blocked&&directA&&q4x8&&pointwise;
}
inline void validateDependenciesFor(bool enabled,bool full512,bool gather,uint32_t cap,
    bool blocked,bool directA,bool q4x8,bool pointwise) {
  if(enabled&&!dependenciesValid(full512,gather,cap,blocked,directA,q4x8,pointwise))
    throw std::invalid_argument("Compact native batch verify requires Full512, gathered MPP cap exactly4, original blocked/directA/Q4x8 and pointwiseM16");
}
inline void validateDependencies(bool full512,bool gather,uint32_t cap,bool blocked,bool directA,bool q4x8,bool pointwise) {
  validateDependenciesFor(requested(),full512,gather,cap,blocked,directA,q4x8,pointwise);
}
[[nodiscard]] inline std::string markerFor(bool enabled) {
  if(!enabled)return {};
  std::string marker(kSemantics);marker+=kSourceIdentitySha256;return marker;
}
[[nodiscard]] inline std::string implementationMarker(){return markerFor(requested());}
// Store owns any atomics; these snapshots count graph construction only.
struct WidthCounters final {
 uint64_t planCalls=0,planRows=0,gateCalls=0,gateRows=0,downCalls=0,downRows=0;
};
struct Counters final {bool enabled=false;WidthCounters r8{},r16{};};
} // namespace splash::flash::compact_native_batch_verify_sep22
