#pragma once
// Native policy: invalid IDs are excluded; duplicate legal selections remain
// counted/mapped and set diagnostic1. This helper independently stable-sorts
// canonical routes, rather than reproducing GPU histogram/scan scheduling.
#include "dev/benchmarks/flash_expert_int8_bucket_reference.hpp"
#include <array>
#include <numeric>

namespace splash::bench::i8_lut::guard_v2 {
inline constexpr uint32_t kInvalid = UINT32_MAX;
struct Expected final {
  std::array<uint32_t,512> counts{};
  std::array<uint32_t,513> offsets{};
  std::vector<uint32_t> routeMap,inverse,invalidRoutes,duplicateRoutes;
  uint32_t diagnostic = 0, jobs = 0;
  uint32_t total() const { return offsets[512]; }
};
inline Expected expected(std::span<const int64_t> ids,uint32_t rows,
                         uint32_t selections,uint32_t sticky,uint32_t tile=32) {
  if (!rows || rows>8192 || !selections || selections>16 ||
      ids.size()!=uint64_t{rows}*selections || !tile)
    throw std::invalid_argument("guard-v2 fixture geometry differs");
  Expected result; result.diagnostic=sticky;
  result.routeMap.assign(ids.size(),kInvalid);result.inverse.assign(ids.size(),kInvalid);
  std::vector<uint32_t> valid;
  for (uint32_t route=0;route<ids.size();++route) {
    const auto id=ids[route];
    if (id<0 || id>=512) {
      result.invalidRoutes.push_back(route);result.diagnostic|=1u;continue;
    }
    ++result.counts[uint32_t(id)];valid.push_back(route);
    const uint32_t rowBegin=route-route%selections;
    for (uint32_t previous=rowBegin;previous<route;++previous)
      if (ids[previous]==id) {
        result.duplicateRoutes.push_back(route);result.diagnostic|=1u;break;
      }
  }
  for (uint32_t id=0;id<512;++id) {
    result.offsets[id+1]=result.offsets[id]+result.counts[id];
    result.jobs+=(result.counts[id]+tile-1)/tile;
  }
  std::stable_sort(valid.begin(),valid.end(),[&](uint32_t a,uint32_t b){return ids[a]<ids[b];});
  for (uint32_t packed=0;packed<valid.size();++packed) {
    result.routeMap[packed]=valid[packed];result.inverse[valid[packed]]=packed;
  }
  return result;
}
inline void cpuSelfTest() {
  constexpr uint32_t sticky=0x80000000u;
  const auto check=[](bool okay){if (!okay) throw std::logic_error("guard-v2 independent CPU golden differs");};
  const std::array<int64_t,3> duplicate{7,2,7};
  const auto d=expected(duplicate,1,3,sticky);
  check(d.total()==3 && d.counts[2]==1 && d.counts[7]==2 && d.jobs==2 &&
        d.diagnostic==(sticky|1u) && d.invalidRoutes.empty() && d.duplicateRoutes==std::vector<uint32_t>{2});
  check(d.routeMap==std::vector<uint32_t>({1,0,2}) && d.inverse==std::vector<uint32_t>({1,0,2}));
  const std::array<int64_t,3> invalid{-1,7,512};
  const auto i=expected(invalid,1,3,sticky);
  check(i.total()==1 && i.counts[7]==1 && i.jobs==1 && i.diagnostic==(sticky|1u) &&
        i.invalidRoutes==std::vector<uint32_t>({0,2}) && i.duplicateRoutes.empty());
  check(i.routeMap==std::vector<uint32_t>({1,kInvalid,kInvalid}) &&
        i.inverse==std::vector<uint32_t>({kInvalid,0,kInvalid}));
  const std::array<int64_t,6> crossRow{2,7,11,2,7,11};
  const auto c=expected(crossRow,2,3,sticky);
  check(c.total()==6 && c.diagnostic==sticky && c.duplicateRoutes.empty());
  // Contrast against the preexisting independently maintained pack reference.
  for (const auto &ids:{std::vector<int64_t>(duplicate.begin(),duplicate.end()),
                       std::vector<int64_t>(invalid.begin(),invalid.end()),
                       std::vector<int64_t>(crossRow.begin(),crossRow.end())}) {
    const uint32_t rows=uint32_t(ids.size()/3);
    const auto a=expected(ids,rows,3,sticky);
    const auto b=flash::int8_bucket_reference::pack(std::vector<uint16_t>(uint64_t{rows}*2560),ids,rows,3,sticky);
    check(a.counts==b.counts && a.offsets==b.offsets && a.routeMap==b.routeMap &&
          a.inverse==b.canonicalToPacked && a.diagnostic==b.diagnostic);
  }
}
} // namespace splash::bench::i8_lut::guard_v2
