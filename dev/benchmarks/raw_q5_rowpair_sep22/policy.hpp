#pragma once
#include "metal/abi/FlashAffine.h"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace raw_q5_rowpair_sep22 {
inline constexpr uint32_t kRows=4,kK=6144,kN=2560;
inline constexpr uint64_t kWeights=9830400,kParameters=245760,kInput=49152,
    kOutput=20480,kRaw=40960,kAlignment=16384;
struct Span { uintptr_t address=0;uint64_t bytes=0; };
inline bool valid(const FlashAffineParams &p) noexcept {
  return p.rows==4 && p.selections==1 && p.experts==1 && p.input_size==6144 &&
      p.output_size==2560 && p.bits==5 && p.group_size==128 && !p.flags &&
      p.weight_row_stride_bytes>=3840 && p.parameter_row_stride_bytes>=96 &&
      !(p.parameter_row_stride_bytes%2) && !(p.parameter_expert_stride_bytes%2);
}
inline uint64_t extent(uint64_t stride,uint64_t width) {
  if (stride>(UINT64_MAX-width)/(kN-1)) throw std::invalid_argument("rowpair stride extent overflow");
  return uint64_t(kN-1)*stride+width;
}
inline bool overlap(Span a,Span b) noexcept {
  if(!a.address||!b.address||!a.bytes||!b.bytes||a.bytes>UINTPTR_MAX-a.address||b.bytes>UINTPTR_MAX-b.address)return true;
  return a.address<b.address+b.bytes&&b.address<a.address+a.bytes;
}
inline void validate(const FlashAffineParams &p,const std::array<Span,7>&s,
    uint64_t groupsX,uint64_t groupsY,uint64_t groupsZ,uint64_t threads,
    bool paired,Span raw={}) {
  if(!valid(p)||groupsX!=320||groupsY!=(paired?2u:4u)||groupsZ!=1||threads!=64)
    throw std::invalid_argument("rowpair unqualified params or complete dispatch grid");
  const std::array<uint64_t,7> sizes{kInput,extent(p.weight_row_stride_bytes,3840),
      extent(p.parameter_row_stride_bytes,96),extent(p.parameter_row_stride_bytes,96),
      1,kOutput,4};
  for(size_t i=0;i<s.size();++i) {
    if(!s[i].address||s[i].bytes<sizes[i]||s[i].bytes>UINTPTR_MAX-s[i].address)
      throw std::invalid_argument("rowpair short or invalid Shared span");
  }
  if(s[0].address%2||s[2].address%2||s[3].address%2||s[5].address%2||s[6].address%4)
    throw std::invalid_argument("rowpair typed Shared span alignment");
  for(size_t w:{5u,6u})for(size_t r:{0u,1u,2u,3u,4u})if(overlap(s[w],s[r]))
    throw std::invalid_argument("rowpair writable span aliases immutable input");
  if(overlap(s[5],s[6]))throw std::invalid_argument("rowpair result/diagnostic aliases");
  if(raw.address||raw.bytes) {
    if(!raw.address||raw.bytes<kRaw||raw.address%4||raw.bytes>UINTPTR_MAX-raw.address)
      throw std::invalid_argument("rowpair invalid F32 tap span");
    for(auto q:s)if(overlap(raw,q))throw std::invalid_argument("rowpair tap aliases another span");
  }
}
inline FlashAffineParams parameters(uint64_t weightRow=3840,uint64_t parameterRow=96) {
  return {4,1,6144,2560,1,5,128,0,weightRow,0,parameterRow,0};
}
} // namespace raw_q5_rowpair_sep22
