#pragma once
#include "source_identity.hpp"
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
namespace splash::flash::guard_hc_fast_composite_sep22 {
inline constexpr const char *flag="SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22";
enum class State:uint8_t {Missing,Disabled,Enabled};
inline State parse(const char *p){if(!p)return State::Missing;if(std::string_view(p)=="0")return State::Disabled;if(std::string_view(p)=="1")return State::Enabled;throw std::invalid_argument("SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22 must be exactly0 or1");}
inline bool requested(){static const State frozen=parse(std::getenv(flag));if(parse(std::getenv(flag))!=frozen)throw std::logic_error("Guard/HC fast composite flag changed after freezing");return frozen==State::Enabled;}
inline void validate(bool compact,bool guard,bool hc){if(requested()&&(!compact||!guard||!hc))throw std::invalid_argument("Guard/HC fast composite requires compact1/guardBundle1/HCpad1");if(!requested()&&hc)throw std::invalid_argument("HC fast in this composite worker requires its registered composite flag1");}
inline std::string marker(){return requested()?std::string(";private-guardBundle-C1-HCfastV8-down7cc3-R4-only-sourceSha256=")+kCompositeSourceIdentitySha256:std::string{};}
}
