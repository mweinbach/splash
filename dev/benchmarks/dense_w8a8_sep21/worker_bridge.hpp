#pragma once
#include "worker_cache.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace splash::flash::dense_w8a8_sep21 {
inline constexpr const char *kFlag = "SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21";
inline constexpr std::string_view kNumericalPolicy =
    "private-main-prefill-r2048-qkv-z-qsaq-cached-original-bf16-rowmax-i8-f32scales-"
    "gpu-activation-rne-i8-i32-wholek-late-f32-scales-bf16-84roles-v1";
inline constexpr std::string_view kCacheMarker = ";dense-w8a8-cache:";
[[nodiscard]] inline bool parse(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(kFlag) + " must be 0 or 1");
}
[[nodiscard]] inline bool requested() {
  static const bool value = parse(std::getenv(kFlag));
  return value;
}
[[nodiscard]] inline const char *selectionMarker(bool selected) noexcept {
  return selected ? ";dense-w8a8-selected-r2048-main-nonverify-v1"
                  : ";dense-w8a8-selected0-base-graphs-v1";
}
[[nodiscard]] inline std::string cacheIdentityFromRoutes(std::string_view routes) {
  const auto begin = routes.find(kCacheMarker);
  if (begin == std::string_view::npos)
    throw std::logic_error("private dense W8A8 worker omitted its fixed cache identity");
  const auto start = begin + kCacheMarker.size(), end = routes.find(';',start);
  const auto identity = routes.substr(start,end == std::string_view::npos ? end : end-start);
  if (identity == "none-maxrows-below2048") return std::string(identity);
  if (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    throw std::logic_error("private dense W8A8 cache identity is not SHA256");
  return std::string(identity);
}
// The worker's fixed numerical/cache policy is independent of the graph
// selector. Actual verified BF16 coefficient hashes enter through Cache's
// identity; selector state is reported separately in kernel_routes/status.
[[nodiscard]] inline std::string numericalIdentity(std::string_view base,std::string_view routes) {
  const std::string identity = std::string(base) + "\n" + std::string(kNumericalPolicy) + "\n" + cacheIdentityFromRoutes(routes);
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  if (!CC_SHA256(identity.data(),CC_LONG(identity.size()),digest)) throw std::runtime_error("private dense W8A8 identity SHA256 failed");
  std::ostringstream out;
  for (unsigned char value : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
  return out.str();
}
[[nodiscard]] constexpr bool mainProjectionEligible(std::string_view prefix,uint32_t rows,
    uint32_t n,uint32_t k,bool verify,bool singletonMain,bool selected) noexcept {
  return selected && singletonMain && bool(projectionPlan(prefix,rows,n,k,verify));
}
} // namespace splash::flash::dense_w8a8_sep21
