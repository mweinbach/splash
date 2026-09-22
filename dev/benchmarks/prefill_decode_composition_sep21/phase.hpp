#pragma once
#include "dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include <atomic>
#include <array>
#include <CommonCrypto/CommonDigest.h>
namespace splash::flash::phase_q4_sep21 {
inline constexpr const char *flag="SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21";
inline constexpr const char *schema="prefill-full512-i8-allrows-originalq4-decode-verify-v1";
inline constexpr const char *policy="all singleton Prefill rows Full512-I8; explicit Decode and every singleton Verify original packed Q4; original trained MTP";
inline bool parse(const char *value){
  if(!value || std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21 must be 0 or 1");
}
inline bool requested(){static const bool value=parse(std::getenv(flag));return value;}
enum class Phase:uint8_t {Prefill,Decode,Verify};
struct Stage {
  std::atomic<uint64_t> i8Calls{0},i8Rows{0},q4Calls{0},q4Rows{0};
};
inline std::array<Stage,3> counters;
inline void record(Phase phase,bool i8,uint32_t rows){
  if(!requested())return;
  auto &value=counters[static_cast<uint8_t>(phase)];
  (i8?value.i8Calls:value.q4Calls).fetch_add(1,std::memory_order_relaxed);
  (i8?value.i8Rows:value.q4Rows).fetch_add(rows,std::memory_order_relaxed);
}
inline std::string identity(std::string_view base){
  if(!requested())return std::string(base);
  const std::string text=std::string(base)+"\nphase="+schema+"\n"+policy+
      "\nf32_constructed296_bytes12097945600\nf32_parent508_virtual_selector_membership_null_policy_raw_prefill_v2\n";
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(text.data(),static_cast<CC_LONG>(text.size()),digest.data());
  static constexpr char hex[]="0123456789abcdef";std::string out;
  for(auto byte:digest){out.push_back(hex[byte>>4]);out.push_back(hex[byte&15]);}return out;
}
inline std::vector<metal::MetalBuffer> persistentF32(const FlashWeights &weights,
    const FlashFloatDenseCache &cache){
  const auto saved=cache.persistedWeightBuffers();
  if(!requested())return saved;
  if(cache.persistedTensorCount()!=296 || cache.persistedPayloadBytes()!=12097945600ULL || saved.size()!=296)
    throw std::logic_error("phase F32 union296 saved backing census differs");
  std::vector<metal::MetalBuffer> kept;uint64_t bytes=0;
  for(const auto &name:FlashDenseCache::defaultPrefixes(weights,false)){
    if(name.find(".ple.")!=std::string::npos && !name.starts_with("language_model.model.layers.1.ple."))continue;
    const auto &p=weights.projection(name);
    if(!flashFloatDenseSmallRowsPolicy(name,4,p.outputSize,p.inputSize,p.bits,p.groupSize))continue;
    if(!cache.contains(name))throw std::logic_error("phase persistent F32 selection lacks backing");
    const auto &tensor=cache.tensor(name);
    const uint64_t expected=uint64_t{p.outputSize}*p.inputSize*4;
    if(tensor.dtype!=FlashDType::F32 || tensor.logicalBytes!=expected || tensor.buffer.sizeBytes()!=expected ||
        tensor.shape!=std::vector<uint64_t>{p.outputSize,p.inputSize})
      throw std::logic_error("phase persistent F32 view/shape differs");
    uint32_t matches=0;for(const auto &owner:saved)if(owner.sameView(tensor.buffer))++matches;
    if(matches!=1)throw std::logic_error("phase persistent F32 owner is not unique full view");
    for(const auto &owner:kept)if(owner.sameView(tensor.buffer))throw std::logic_error("phase duplicate F32 owner");
    kept.push_back(tensor.buffer);bytes+=tensor.buffer.sizeBytes();
  }
  if(kept.size()!=118 || bytes!=3247964160ULL)throw std::logic_error("phase persistent118 F32 census differs");
  return kept;
}
} // namespace splash::flash::phase_q4_sep21
