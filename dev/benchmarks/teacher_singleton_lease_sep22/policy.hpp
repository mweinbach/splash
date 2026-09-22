#pragma once
#include "dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include <limits>
namespace splash::flash::teacher_singleton_lease_sep22 {
inline constexpr const char *flag="SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22";
inline constexpr const char *profile="pure-I8-teacher-parent508-startup-persistent807-r4F32118-omitBF1684-v1";
inline constexpr const char *policy="startup lease selection only; preserve actual508F32/509BF16 backing and math; fixed R4 F32118 persistent; BF16 W8 source84 direct-transient; W8derived168 already included; all backing/governor reservations retained; B1 qualification only; batch math/capabilities unchanged";
inline constexpr const char *sourceSHA="SOURCE_SHA_PLACEHOLDER";
inline bool parse(const char *value){
  if(!value || std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument(std::string(flag)+" must be 0 or 1");
}
inline bool requested(){static const bool value=parse(std::getenv(flag));return value;}
inline std::vector<metal::MetalBuffer> persistentF32(const FlashWeights &weights,const FlashFloatDenseCache &cache){
  const auto saved=cache.persistedWeightBuffers();
  if(!requested())return saved;
  if(cache.persistedTensorCount()!=508 || cache.persistedPayloadBytes()!=14391705600ULL || saved.size()!=508 || cache.prefixes().size()!=508)
    throw std::logic_error("teacher lease-only full508 actual F32 backing census differs");
  std::vector<metal::MetalBuffer> kept;uint64_t bytes=0;
  for(const auto &name:FlashDenseCache::defaultPrefixes(weights,false)){
    const auto &p=weights.projection(name);
    if(!flashFloatDenseSmallRowsPolicy(name,4,p.outputSize,p.inputSize,p.bits,p.groupSize))continue;
    if(!cache.contains(name))throw std::logic_error("teacher lease-only persistent F32 selection lacks backing");
    const auto &tensor=cache.tensor(name);const uint64_t expected=uint64_t{p.outputSize}*p.inputSize*4;
    if(tensor.dtype!=FlashDType::F32 || tensor.logicalBytes!=expected || tensor.buffer.sizeBytes()!=expected ||
        tensor.shape!=std::vector<uint64_t>{p.outputSize,p.inputSize} || tensor.buffer.storage()!=metal::BufferStorage::Shared)
      throw std::logic_error("teacher lease-only persistent F32 view/shape differs");
    uint32_t matches=0;for(const auto &owner:saved)if(owner.sameView(tensor.buffer))++matches;
    if(matches!=1)throw std::logic_error("teacher lease-only persistent F32 owner is not unique full view");
    for(const auto &owner:kept)if(owner.sameView(tensor.buffer))throw std::logic_error("teacher lease-only duplicate F32 owner");
    if(bytes>std::numeric_limits<uint64_t>::max()-tensor.buffer.sizeBytes())throw std::overflow_error("teacher lease-only F32 census overflow");
    kept.push_back(tensor.buffer);bytes+=tensor.buffer.sizeBytes();
  }
  if(kept.size()!=118 || bytes!=3247964160ULL)throw std::logic_error("teacher lease-only persistent118 F32 census differs");
  return kept;
}
inline std::vector<metal::MetalBuffer> persistentBF16(const FlashDenseCache &cache){
  const auto saved=cache.persistedWeightBuffers();if(!requested())return saved;
  if(cache.persistedTensorCount()!=509 || cache.persistedPayloadBytes()!=8467251200ULL || saved.size()!=509)
    throw std::logic_error("teacher lease-only full509 BF16 backing census differs");
  return dense_w8a8_residency_sep21::bf16PersistentOperands(cache,true,true,true);
}
} // namespace splash::flash::teacher_singleton_lease_sep22
