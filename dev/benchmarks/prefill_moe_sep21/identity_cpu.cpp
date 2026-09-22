#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"
#include "flash/FlashGatheredMPP.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <iostream>
#include <stdexcept>
std::string digest(const std::string &text) {
  unsigned char value[CC_SHA256_DIGEST_LENGTH];
  if (!CC_SHA256(text.data(),static_cast<CC_LONG>(text.size()),value)) throw std::runtime_error("identity SHA failed");
  const char *hex="0123456789abcdef";std::string result;
  for (auto c :value) { result +=hex[c >>4];result +=hex[c &15]; }
  return result;
}
int main() {
  using namespace splash::flash;
  std::string base="splash.private-allrows-target-v1\nsource=edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
      "\nstore=ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1\npolicy=" +
      std::string(kFlashInt8ExpertStoreSemantics) +"\nmtp=original-trained-bank\n";
  const bool gathered=gathered_mpp::requested();
  const auto cap=gathered_mpp::requestedMaximumRows();
  if (gathered) base +="small_row_policy=" +std::string(gathered_mpp::kPolicy) +"\nsmall_row_cap_policy=" +
      std::string(gathered_mpp::kRowCapPolicy) +"\nsmall_row_max_physical_rows=" +std::to_string(cap) +"\n";
  const auto policy=prefill_moe_sep21::policyIdentity();
  const auto selected=policy.empty() ? base :base +"prefill_moe_policy=" +policy +"\n";
  const auto baseHash=digest(base),selectedHash=digest(selected);
  if (!gathered &&baseHash !="2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d")
    throw std::runtime_error("base numerical derivative identity drift");
  if (policy.empty() &&selectedHash !=baseHash) throw std::runtime_error("control derivative changed");
  std::cout <<"{\"gpu_work\":false,\"variant\":" <<prefill_moe_sep21::requestedVariantIndex()
      <<",\"gathered_mpp\":" <<(gathered ? "true" :"false") <<",\"gathered_physical_row_cap\":" <<cap
      <<",\"base_numerical_derivative_sha256\":\"" <<baseHash
      <<"\",\"selected_numerical_derivative_sha256\":\"" <<selectedHash
      <<"\",\"selected_policy\":\"" <<policy <<"\"}\n";
}
