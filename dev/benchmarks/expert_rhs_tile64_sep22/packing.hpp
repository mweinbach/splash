#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
namespace rhs_tile64 {
inline void requireInlineStrides(uint32_t first,uint32_t second) {
 if(first!=1 || !second)throw std::invalid_argument("MSL inline tensor requires first stride1");
}
inline uint64_t bankBytes(uint32_t width,uint32_t k) {
 if(!width || width%64 || !k || uint64_t(width)>std::numeric_limits<uint64_t>::max()/k)
  throw std::invalid_argument("RHS tile64 packing geometry");
 return uint64_t(width)*k;
}
inline uint64_t oldIndex(uint32_t width,uint32_t k,uint32_t n,uint32_t kk) {
 (void)bankBytes(width,k);if(n>=width || kk>=k)throw std::invalid_argument("RHS tile64 packing index");
 return uint64_t(n)*k+kk;
}
inline uint64_t newIndex(uint32_t width,uint32_t k,uint32_t n,uint32_t kk) {
 (void)bankBytes(width,k);if(n>=width || kk>=k)throw std::invalid_argument("RHS tile64 packing index");
 return uint64_t(n/64)*k*64+uint64_t(kk)*64+n%64;
}
inline std::vector<int8_t> pack(const std::vector<int8_t>&old,uint32_t width,uint32_t k) {
 if(old.size()!=bankBytes(width,k))throw std::invalid_argument("RHS tile64 exact code bank extent");
 std::vector<int8_t> result(old.size());
 for(uint32_t n=0;n<width;++n)for(uint32_t kk=0;kk<k;++kk)result[newIndex(width,k,n,kk)]=old[oldIndex(width,k,n,kk)];
 return result;
}
inline uint64_t cpuBijection() {
 uint64_t checked=0;
 for(const auto shape:{std::pair<uint32_t,uint32_t>{640,2560},{2560,640}}) {
  const auto [width,k]=shape;const uint64_t bytes=bankBytes(width,k);std::vector<uint8_t> visits(bytes);
  for(uint32_t n=0;n<width;++n)for(uint32_t kk=0;kk<k;++kk) {
   const auto old=oldIndex(width,k,n,kk),now=newIndex(width,k,n,kk);
   if(old>=bytes || now>=bytes || visits[now]++)throw std::runtime_error("RHS tile64 bijection failure");
   const uint64_t tile=now/(uint64_t(k)*64),within=now%(uint64_t(k)*64);
   if(tile*64+within%64!=n || within/64!=kk)throw std::runtime_error("RHS tile64 inverse failure");
   // Old transposed B tensor coordinates(k,nLocal) map to nLocal*K+k.
   // Corrected untransposed B coordinates(nLocal,k) map to k*64+nLocal.
   requireInlineStrides(1,k);requireInlineStrides(1,64);
   const uint64_t oldLogical=uint64_t(n/64)*k*64+uint64_t(n%64)*k+kk;
   const uint64_t newLogical=uint64_t(n/64)*k*64+uint64_t(kk)*64+n%64;
   if(oldLogical!=old || newLogical!=now)throw std::runtime_error("RHS original/corrected view logical coefficient mismatch");
   if(within>uint64_t(k)*64-1)throw std::runtime_error("RHS tile64 tile bounds failure");
   ++checked;
  }
  for(auto count:visits)if(count!=1)throw std::runtime_error("RHS tile64 omitted destination");
 }
 return checked;
}
}
