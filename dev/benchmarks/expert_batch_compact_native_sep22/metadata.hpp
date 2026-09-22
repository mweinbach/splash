#pragma once
// Device-free exact reference for the original native M16 metadata contract.
#include "abi.hpp"
#include <array>
#include <span>
#include <stdexcept>
namespace compact_native_batch_metadata {
struct Expected {
  std::array<uint32_t,512>counts{};
  std::array<uint32_t,513>offsets{},jobOffsets{};
  std::array<uint32_t,kCompactNativeBatchRoutes>routeMap{},inverse{};
  std::array<FlashMoEBucketJob,kCompactNativeBatchJobCapacity>jobs{};
  uint32_t jobCount=0,sticky=0;
};
inline Expected expected(std::span<const int64_t>ids) {
  if(ids.size()!=kCompactNativeBatchRoutes)throw std::invalid_argument("compact native reference requires fixed R8/R16 S10 IDs");
  Expected r;r.routeMap.fill(UINT32_MAX);r.inverse.fill(UINT32_MAX);
  for(auto&j:r.jobs)j={UINT32_MAX,0};
  for(uint32_t route=0;route<kCompactNativeBatchRoutes;++route) {
    const auto id=ids[route];if(id<0||id>=512){r.sticky|=1;continue;}
    for(uint32_t p=route/10*10;p<route;++p)if(ids[p]==id)r.sticky|=1;
    ++r.counts[size_t(id)];
  }
  uint32_t total=0;
  for(uint32_t e=0;e<512;++e) {
    r.offsets[e]=total;r.jobOffsets[e]=r.jobCount;
    for(uint32_t route=0;route<kCompactNativeBatchRoutes;++route)if(ids[route]==int64_t(e)){r.routeMap[total]=route;r.inverse[route]=total++;}
    for(uint32_t begin=r.offsets[e];begin<total;begin+=16)r.jobs[r.jobCount++]={e,begin};
  }
  r.offsets[512]=total;r.jobOffsets[512]=r.jobCount;return r;
}
inline bool cpuSelfTest() {
  std::array<int64_t,kCompactNativeBatchRoutes>ids;ids.fill(7);const auto r=expected(ids);
  if(r.counts[7]!=kCompactNativeBatchRoutes||r.offsets[7]!=0||r.offsets[8]!=kCompactNativeBatchRoutes||r.jobCount!=(kCompactNativeBatchRoutes+15)/16||r.sticky!=1)return false;
  for(uint32_t i=0;i<kCompactNativeBatchRoutes;++i)if(r.routeMap[i]!=i||r.inverse[i]!=i)return false;
  for(uint32_t j=0;j<r.jobCount;++j)if(r.jobs[j].expert!=7||r.jobs[j].row_begin!=j*16)return false;
  for(uint32_t j=r.jobCount;j<kCompactNativeBatchJobCapacity;++j)if(r.jobs[j].expert!=UINT32_MAX||r.jobs[j].row_begin)return false;
  ids.fill(-1);const auto bad=expected(ids);if(bad.jobCount||bad.offsets[512]||bad.sticky!=1)return false;
  for(auto v:bad.inverse)if(v!=UINT32_MAX)return false;
  for(uint32_t i=0;i<kCompactNativeBatchRoutes;++i)ids[i]=i;const auto spread=expected(ids);
  return spread.jobCount==kCompactNativeBatchRoutes&&spread.offsets[512]==kCompactNativeBatchRoutes&&!spread.sticky;
}
} // namespace compact_native_batch_metadata
