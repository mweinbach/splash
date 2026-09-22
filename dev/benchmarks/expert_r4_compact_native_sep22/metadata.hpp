#pragma once
// Device-free exact reference for the original native M16 metadata contract.
#include "abi.hpp"
#include <array>
#include <span>
#include <stdexcept>
namespace compact_native_r4_metadata {
struct Expected {
  std::array<uint32_t,512>counts{};
  std::array<uint32_t,513>offsets{},jobOffsets{};
  std::array<uint32_t,40>routeMap{},inverse{};
  std::array<FlashMoEBucketJob,514>jobs{};
  uint32_t jobCount=0,sticky=0;
};
inline Expected expected(std::span<const int64_t>ids) {
  if(ids.size()!=40)throw std::invalid_argument("compact native reference requires R4/S10 IDs");
  Expected r;r.routeMap.fill(UINT32_MAX);r.inverse.fill(UINT32_MAX);
  for(auto&j:r.jobs)j={UINT32_MAX,0};
  for(uint32_t route=0;route<40;++route) {
    const auto id=ids[route];if(id<0||id>=512){r.sticky|=1;continue;}
    for(uint32_t p=route/10*10;p<route;++p)if(ids[p]==id)r.sticky|=1;
    ++r.counts[size_t(id)];
  }
  uint32_t total=0;
  for(uint32_t e=0;e<512;++e) {
    r.offsets[e]=total;r.jobOffsets[e]=r.jobCount;
    for(uint32_t route=0;route<40;++route)if(ids[route]==int64_t(e)){r.routeMap[total]=route;r.inverse[route]=total++;}
    for(uint32_t begin=r.offsets[e];begin<total;begin+=16)r.jobs[r.jobCount++]={e,begin};
  }
  r.offsets[512]=total;r.jobOffsets[512]=r.jobCount;return r;
}
inline bool cpuSelfTest() {
  std::array<int64_t,40>ids;ids.fill(7);const auto r=expected(ids);
  if(r.counts[7]!=40||r.offsets[7]!=0||r.offsets[8]!=40||r.jobCount!=3||r.sticky!=1)return false;
  for(uint32_t i=0;i<40;++i)if(r.routeMap[i]!=i||r.inverse[i]!=i)return false;
  if(r.jobs[0].expert!=7||r.jobs[0].row_begin!=0||r.jobs[1].row_begin!=16||r.jobs[2].row_begin!=32)return false;
  for(uint32_t j=3;j<514;++j)if(r.jobs[j].expert!=UINT32_MAX||r.jobs[j].row_begin)return false;
  ids.fill(-1);const auto bad=expected(ids);if(bad.jobCount||bad.offsets[512]||bad.sticky!=1)return false;
  for(auto v:bad.inverse)if(v!=UINT32_MAX)return false;
  for(uint32_t i=0;i<40;++i)ids[i]=i;const auto spread=expected(ids);
  return spread.jobCount==40&&spread.offsets[512]==40&&!spread.sticky;
}
} // namespace compact_native_r4_metadata
