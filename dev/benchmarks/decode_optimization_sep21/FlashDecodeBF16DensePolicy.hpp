#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::bf16_decode {
inline constexpr std::string_view kSemantics =
    "private-main-target-cached-original-rounded-bf16-coefficients-bf16-input-f32-whole-k-mpp-output-bf16-m8n64or128-m16n64or128-rows1_2_4_8_16-known-attention-shared-ple-roles-hcup1_2-original-bf16-precise-mix-stages-v2";
enum class Scope { AllCached, ExistingF32Route };
enum class Roles { All, Attention, Shared, PLE, HCUpSmall };
struct Policy {
  bool enabled = false;
  Scope scope = Scope::AllCached;
  Roles roles = Roles::All;
  bool f32Selective = false;
  bool operator==(const Policy &) const = default;
};
inline bool switchValue(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be exactly0 or1");
}
inline Policy requested() {
  Policy policy;
  policy.enabled = switchValue("SPLASH_FLASH_DECODE_BF16_DENSE");
  policy.f32Selective = switchValue("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");
  if (const char *raw = std::getenv("SPLASH_FLASH_DECODE_BF16_DENSE_SCOPE")) {
    if (std::string_view(raw) == "all_cached") policy.scope = Scope::AllCached;
    else if (std::string_view(raw) == "existing_f32_route") policy.scope = Scope::ExistingF32Route;
    else throw std::invalid_argument("DECODE_BF16_DENSE_SCOPE must be all_cached or existing_f32_route");
  }
  if (const char *raw = std::getenv("SPLASH_FLASH_DECODE_BF16_DENSE_ROLES")) {
    const std::string_view value(raw);
    if (value == "all") policy.roles = Roles::All;
    else if (value == "attention") policy.roles = Roles::Attention;
    else if (value == "shared") policy.roles = Roles::Shared;
    else if (value == "ple") policy.roles = Roles::PLE;
    else if (value == "hc_up_small") policy.roles = Roles::HCUpSmall;
    else throw std::invalid_argument("DECODE_BF16_DENSE_ROLES must be all, attention, shared, ple or hc_up_small");
  }
  return policy;
}
inline const char *scopeName(Scope scope) { return scope == Scope::AllCached ? "all_cached" : "existing_f32_route"; }
inline const char *rolesName(Roles roles) {
  switch (roles) {
  case Roles::All: return "all";
  case Roles::Attention: return "attention";
  case Roles::Shared: return "shared";
  case Roles::PLE: return "ple";
  case Roles::HCUpSmall: return "hc_up_small";
  }
  throw std::logic_error("invalid BF16 decode role set");
}
inline std::string identity(const Policy &policy) {
  return std::string(kSemantics) + ";scope=" + scopeName(policy.scope) + ";roles=" + rolesName(policy.roles) +
      (policy.scope == Scope::ExistingF32Route ? std::string(";existing_f32_selective=") + (policy.f32Selective ? "1" : "0") : "");
}
inline void validateDependencies(const Policy &policy) {
  if (policy.enabled && (!switchValue("SPLASH_FLASH_DENSE_CACHE") ||
      !switchValue("SPLASH_FLASH_FLOAT_DENSE_CACHE") || switchValue("SPLASH_FLASH_DENSE_SMALL_ROWS")))
    throw std::invalid_argument("private BF16 decode selector requires DENSE_CACHE=1, FLOAT_DENSE_CACHE=1 and DENSE_SMALL_ROWS=0");
}
inline void validateFrozen(const Policy &policy) {
  if (requested() != policy) throw std::logic_error("private BF16 decode selector changed after construction");
}
inline uint64_t workspaceBytes(const Policy &policy) { return policy.enabled ? uint64_t{16} * 32768 * 2 : 0; }
inline bool rowEligible(uint32_t rows) { return rows == 1 || rows == 2 || rows == 4 || rows == 8 || rows == 16; }
inline bool geometry(const Policy &policy, std::string_view prefix, uint32_t rows, uint32_t outputs, uint32_t inputs) {
  if (!policy.enabled || !rowEligible(rows)) return false;
  const bool smallHC = policy.scope == Scope::AllCached && (policy.roles == Roles::All || policy.roles == Roles::HCUpSmall) &&
      (rows == 1 || rows == 2) && outputs == 10240 && inputs == 320;
  if (smallHC && prefix == "language_model.model.hyper_connection_mixer.input_mix_weight_up") return true;
  constexpr std::string_view layerPrefix = "language_model.model.layers.";
  if (!prefix.starts_with(layerPrefix)) return false;
  const auto tail = prefix.substr(layerPrefix.size());
  const auto split = tail.find('.');
  if (split == std::string_view::npos || !split || split > 2) return false;
  uint32_t layer = 0;
  for (char digit : tail.substr(0, split)) {
    if (digit < '0' || digit > '9') return false;
    layer = layer * 10 + uint32_t(digit - '0');
  }
  if (layer >= 48) return false;
  const auto role = tail.substr(split + 1);
  if (smallHC && (role == "attn_hyper_connection.input_mix_weight_up" || role == "mlp_hyper_connection.input_mix_weight_up")) return true;
  const auto permitted = [&](Roles roleSet) { return policy.roles == Roles::All || policy.roles == roleSet; };
  if (permitted(Roles::Attention)) {
    if ((role == "linear_attn.in_proj_qkv" && outputs == 10240 && inputs == 2560) ||
        (role == "linear_attn.in_proj_z" && outputs == 6144 && inputs == 2560) ||
        (role == "linear_attn.out_proj" && outputs == 2560 && inputs == 6144) ||
        (role == "self_attn.q_proj" && outputs == 12288 && inputs == 2560) ||
        ((role == "self_attn.k_proj" || role == "self_attn.v_proj") && outputs == 512 && inputs == 2560) ||
        (role == "self_attn.o_proj" && outputs == 2560 && inputs == 6144) ||
        (role == "self_attn.indexer.index_qk_proj" && outputs == 640 && inputs == 2560)) return true;
  }
  if (permitted(Roles::Shared) &&
      (((role == "mlp.shared_expert.gate_proj" || role == "mlp.shared_expert.up_proj") && outputs == 640 && inputs == 2560) ||
       (role == "mlp.shared_expert.down_proj" && outputs == 2560 && inputs == 640))) return true;
  if (permitted(Roles::PLE) &&
      ((role == "ple.key_proj" && outputs == 10240 && inputs == 2560) ||
       (role == "ple.value_proj" && outputs == 2560 && inputs == 2560))) return true;
  return false;
}
} // namespace splash::flash::bf16_decode
