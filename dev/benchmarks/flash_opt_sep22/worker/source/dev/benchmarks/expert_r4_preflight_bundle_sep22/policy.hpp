#pragma once
#include "source_identity.hpp"
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::compact_r4_preflight_sep22 {
inline constexpr const char *kFlag="SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22";
enum class State:uint8_t {Missing,Disabled,Enabled};
inline State parse(const char *p) {
  if(!p)return State::Missing;
  if(std::string_view(p)=="0")return State::Disabled;
  if(std::string_view(p)=="1")return State::Enabled;
  throw std::invalid_argument("SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22 must be exactly0 or1");
}
inline bool requested() {
  static const State frozen=parse(std::getenv(kFlag));
  if(parse(std::getenv(kFlag))!=frozen)
    throw std::logic_error("Compact R4 preflight flag changed after freezing");
  return frozen==State::Enabled;
}
inline void validateDependencies(bool originalCompact) {
  if(requested()&&!originalCompact)
    throw std::invalid_argument("Compact R4 preflight requires original compact R4 verify enabled");
}
inline std::string marker() {
  return requested()?std::string(";private-CPU-only-R4-unique13view-preflight-localStoreLayerGraphBundle-original6stage-unchanged-public-guards-sourceSha256=")+kPreflightSourceIdentitySha256:std::string{};
}
struct Counters {bool enabled=false;uint64_t calls=0,rows=0;};
} // namespace splash::flash::compact_r4_preflight_sep22
