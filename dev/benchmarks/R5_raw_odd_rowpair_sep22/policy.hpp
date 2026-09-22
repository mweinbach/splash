#pragma once
#include "metal/abi/FlashAffine.h"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace R5_raw_odd_rowpair_sep22 {
inline constexpr uint32_t rows = 5;
inline constexpr uint64_t governorLimit = 256ULL << 20;
inline constexpr uint64_t selectedSourceLimit = 64ULL << 20;
struct Shape {
  uint32_t K=0,N=0,bits=0,group=0;
  std::string prefix,sourceDType="U32";
  uint64_t weightRowStride=0,parameterRowStride=0;
};
inline constexpr std::array<std::array<uint32_t,4>,7> observed{{
  {{6144,2560,5,128}},{{2560,10240,4,64}},{{2560,6144,5,128}},
  {{2560,6144,6,64}},{{2560,12288,4,64}},{{2560,10240,5,64}},
  {{6144,2560,4,64}}
}};
inline uint64_t add(uint64_t a,uint64_t b) {
  if(a>UINT64_MAX-b)throw std::invalid_argument("R5 extent addition overflow");
  return a+b;
}
inline uint64_t multiply(uint64_t a,uint64_t b) {
  if(b&&a>UINT64_MAX/b)throw std::invalid_argument("R5 extent multiplication overflow");
  return a*b;
}
inline uint64_t rounded(uint64_t n) {
  return add(n,16383)&~uint64_t{16383};
}
inline bool observedShape(const Shape&s) noexcept {
  for(const auto&a:observed)if(a==std::array<uint32_t,4>{s.K,s.N,s.bits,s.group})return true;
  return false;
}
inline Shape shape(uint32_t i) {
  if(i>=observed.size())throw std::invalid_argument("R5 shape index must be0..6");
  const auto&a=observed[i];Shape s;
  s.K=a[0];s.N=a[1];s.bits=a[2];s.group=a[3];
  s.weightRowStride=uint64_t(s.K)*s.bits/8;s.parameterRowStride=uint64_t(s.K/s.group)*2;
  return s;
}
inline uint64_t inputBytes(const Shape&s){return multiply(multiply(rows,s.K),2);}
inline uint64_t outputBytes(const Shape&s){return multiply(multiply(rows,s.N),2);}
inline uint64_t rawBytes(const Shape&s){return multiply(multiply(rows,s.N),4);}
inline uint64_t codeRowBytes(const Shape&s){return multiply(s.K,s.bits)/8;}
inline uint64_t parameterRowBytes(const Shape&s){return multiply(s.K/s.group,2);}
inline uint64_t extent(uint32_t N,uint64_t stride,uint64_t width) {
  if(!N||stride<width)throw std::invalid_argument("R5 invalid tensor stride");
  return add(multiply(N-1,stride),width);
}
inline std::array<uint64_t,3> coefficientBytes(const Shape&s) {
  return {extent(s.N,s.weightRowStride,codeRowBytes(s)),
    extent(s.N,s.parameterRowStride,parameterRowBytes(s)),
    extent(s.N,s.parameterRowStride,parameterRowBytes(s))};
}
inline FlashAffineParams parameters(const Shape&s,uint64_t w=0,uint64_t p=0) {
  return {5,1,s.K,s.N,1,s.bits,s.group,0,
    w?w:s.weightRowStride,0,p?p:s.parameterRowStride,0};
}
inline bool valid(const Shape&s,const FlashAffineParams&p) noexcept {
  return observedShape(s)&&p.rows==5&&p.selections==1&&p.experts==1&&
    p.input_size==s.K&&p.output_size==s.N&&p.bits==s.bits&&p.group_size==s.group&&
    !p.flags&&p.weight_row_stride_bytes>=uint64_t(s.K)*s.bits/8&&
    p.parameter_row_stride_bytes>=uint64_t(s.K/s.group)*2&&
    !(p.parameter_row_stride_bytes%2)&&!(p.parameter_expert_stride_bytes%2);
}
struct Span {uintptr_t address=0;uint64_t bytes=0;};
inline bool overlap(Span a,Span b) noexcept {
  if(!a.address||!b.address||!a.bytes||!b.bytes||
      a.bytes>UINTPTR_MAX-a.address||b.bytes>UINTPTR_MAX-b.address)return true;
  return a.address<b.address+b.bytes&&b.address<a.address+a.bytes;
}
inline void validate(const Shape&s,const FlashAffineParams&p,const std::array<Span,7>&v,
    bool paired,uint64_t gx,uint64_t gy,uint64_t gz,uint64_t tx,uint64_t ty=1,uint64_t tz=1,
    Span raw={}) {
  if(!valid(s,p)||gx!=s.N/8||gy!=(paired?3u:5u)||gz!=1||tx!=64||ty!=1||tz!=1)
    throw std::invalid_argument("R5 params or complete dispatch geometry rejected");
  const std::array<uint64_t,7>minimum{inputBytes(s),
    extent(s.N,p.weight_row_stride_bytes,codeRowBytes(s)),
    extent(s.N,p.parameter_row_stride_bytes,parameterRowBytes(s)),
    extent(s.N,p.parameter_row_stride_bytes,parameterRowBytes(s)),1,outputBytes(s),4};
  for(size_t i=0;i<v.size();++i)
    if(!v[i].address||v[i].bytes<minimum[i]||v[i].bytes>UINTPTR_MAX-v[i].address)
      throw std::invalid_argument("R5 short or overflowing Shared span");
  if(v[0].address%2||v[2].address%2||v[3].address%2||v[5].address%2||v[6].address%4)
    throw std::invalid_argument("R5 typed span alignment");
  for(size_t w:{5u,6u})for(size_t r:{0u,1u,2u,3u,4u})
    if(overlap(v[w],v[r]))throw std::invalid_argument("R5 writable immutable-source alias");
  if(overlap(v[5],v[6]))throw std::invalid_argument("R5 output diagnostic alias");
  if(raw.address||raw.bytes) {
    if(!raw.address||raw.bytes<rawBytes(s)||raw.address%4||raw.bytes>UINTPTR_MAX-raw.address)
      throw std::invalid_argument("R5 raw tap span");
    for(const auto&a:v)if(overlap(raw,a))throw std::invalid_argument("R5 raw tap alias");
  }
}
// Sum actual planned rounded base sizes. This is reserved before any source
// payload read or tensor allocation; only bounded metadata precedes admission.
inline uint64_t admissionBytes(const Shape&s,bool capture) {
  uint64_t total=0;
  const auto charge=[&](uint64_t n){total=add(total,rounded(add(n,144)));};
  const auto native=coefficientBytes(s);
  uint64_t source=0;for(uint64_t n:native){source=add(source,n);charge(n);}
  if(source>selectedSourceLimit)throw std::invalid_argument("R5 selected native spans exceed64MiB");
  charge(multiply(s.N,add(s.weightRowStride,4)));
  charge(multiply(s.N,add(s.parameterRowStride,4)));
  charge(multiply(s.N,add(s.parameterRowStride,4)));
  charge(inputBytes(s));if(capture)charge(inputBytes(s));
  for(uint32_t i=0;i<2;++i){charge(outputBytes(s));charge(outputBytes(s));charge(rawBytes(s));charge(4);}
  charge(add(add(inputBytes(s),outputBytes(s)),32768));
  if(total>governorLimit)throw std::invalid_argument("R5 component admission exceeds256MiB");
  return total;
}
inline std::string format(const Shape&s) {
  return "q"+std::to_string(s.bits)+"_g"+std::to_string(s.group);
}
inline std::string kernel(const Shape&s,bool paired,bool tap) {
  if(!paired&&!tap)return "flash_affine_mlx_qmv_f32xsum_v1_"+format(s);
  return "r5_raw_odd_rowpair_sep22_"+std::string(paired?(tap?"candidate_probe_":"timed_"):"control_probe_")+format(s);
}
} // namespace R5_raw_odd_rowpair_sep22
