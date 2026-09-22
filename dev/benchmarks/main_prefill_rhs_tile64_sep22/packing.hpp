#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace splash::flash::main_prefill_rhs_tile64_sep22 {
inline constexpr uint32_t rows=2048,selections=10,routes=rows*selections,experts=512;
inline constexpr uint64_t alignment=16384,guard=64,hostAllowance=512ULL<<20;
constexpr uint64_t rounded(uint64_t bytes){return (bytes+alignment-1)&~(alignment-1);}
constexpr uint64_t guarded(uint64_t bytes){return rounded(bytes+2*guard);}
constexpr uint64_t originalIndex(uint32_t n,uint32_t k,uint32_t rank,uint32_t column,uint32_t inner){return (uint64_t(rank)*n+column)*k+inner;}
constexpr uint64_t tiledIndex(uint32_t n,uint32_t k,uint32_t rank,uint32_t column,uint32_t inner){return uint64_t(rank)*n*k+uint64_t(column/64)*k*64+uint64_t(inner)*64+column%64;}
inline constexpr uint32_t jobCapacity=(routes+7)/8+511,usedJobCapacity=(routes+31)/32+511;
// Original mapping+rank; one temporary transformed guarded layer; fixtures;
// two guarded scratch/output arms and both literal producer probe planes.
constexpr uint64_t fixtureBytes(){return guarded(uint64_t(rows)*2560*2)+guarded(uint64_t(routes)*8)+guarded(uint64_t(routes)*2)+guarded(uint64_t(rows)*2560*2)+guarded(uint64_t(rows)*2);}
constexpr uint64_t scratchBytes(){return guarded(512*4)+guarded(513*4)+2*guarded(uint64_t(routes)*4)+guarded(uint64_t(routes+63)*2560*2)+guarded(513*4)+guarded(4)+guarded(uint64_t(jobCapacity)*8)+guarded(uint64_t(routes+63)*640*2)+guarded(uint64_t(routes)*2560*2)+guarded(uint64_t(rows)*2560*2)+guarded(4);}
constexpr uint64_t probeBytes(){return 2*guarded(uint64_t(routes)*640*4)+2*guarded(uint64_t(routes)*640*2)+guarded(uint64_t(routes)*2560*4)+guarded(uint64_t(routes)*2560*2);}
constexpr uint64_t nativePlannedBytes(){return 2524446720ULL+alignment+guarded(2524446720ULL)+fixtureBytes()+2*(scratchBytes()+probeBytes());}
inline uint64_t cpuBijection(){uint64_t count=0;for(auto shape:std::array<std::array<uint32_t,2>,3>{{{640,2560},{640,2560},{2560,640}}}){const auto n=shape[0],k=shape[1];for(uint32_t c: {0u,1u,63u,64u,n-1})for(uint32_t i:{0u,1u,127u,128u,k-1})for(uint32_t r:{0u,511u}){auto at=tiledIndex(n,k,r,c,i);if(at>=uint64_t(experts)*n*k||at/(uint64_t(n)*k)!=r)throw std::logic_error("tile64 bounded index");++count;}}return count;}
}
