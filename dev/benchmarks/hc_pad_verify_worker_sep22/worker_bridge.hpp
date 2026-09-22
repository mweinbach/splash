#pragma once
#include "dev/benchmarks/hc_pad_producer_sep22/abi.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "metal/CommandGraph.hpp"
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::hc_pad_verify_sep22 {
inline constexpr const char *flag="SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22";
inline constexpr const char *policy="exact-main-VerifyR4-HCdown-SG4-positive-zero-inactive4to7-unchanged-HCup-F32-M8N32SG4-parent-v1";
inline constexpr const char *marker=";private-hc-pad-verify-r4-down-producer-unchanged-up-v1";
inline constexpr const char *certificate="97roles-rawDownF32-BF16-Silu-gates-UpF32-BF16-mix-padded8-guards-exact-sep22-hc-pad-producer-r4-97chain-v2";
inline bool parse(const char *value){if(!value||std::string_view(value)=="0")return false;if(std::string_view(value)=="1")return true;throw std::invalid_argument("SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22 must be0 or1");}
inline bool requested(){static const bool value=parse(std::getenv(flag));return value;}
inline void requireFrozen(){if(parse(std::getenv(flag))!=requested())throw std::logic_error("HC padding flag changed after startup");}
inline void validateDependencies(bool hcUpF32,bool fuseHC,bool floatDense){if(requested()&&(!hcUpF32||!fuseHC||!floatDense))throw std::invalid_argument("HC padding VerifyR4 requires HC_UP_F32_MPP=1/FUSE_HC=1/FLOAT_DENSE_CACHE=1");}
inline constexpr bool eligible(bool verification,uint32_t rows,uint32_t maximumRows){return verification&&rows==4&&maximumRows>=8;}
inline std::atomic<uint64_t> graphCalls{0},graphRows{0},savedPaddingDispatches{0};
inline void record(uint32_t rows){graphCalls.fetch_add(1,std::memory_order_relaxed);graphRows.fetch_add(rows,std::memory_order_relaxed);savedPaddingDispatches.fetch_add(1,std::memory_order_relaxed);}
inline void disjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b){if(!a||!b||!a.contents()||!b.contents())throw std::invalid_argument("HC padding needs Shared views");
  const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());if(x<=y?uint64_t(y-x)<a.sizeBytes():uint64_t(x-y)<b.sizeBytes())throw std::invalid_argument("HC padding expanded scratch aliases source/output");}
template<class Params>inline Params params(const metal::ComputeDispatch &dispatch,uint32_t index){Params p{};if(dispatch.bytes.size()!=1||dispatch.bytes[0].index!=index||dispatch.bytes[0].sizeBytes!=sizeof(p))throw std::logic_error("HC original parameter binding changed");std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));return p;}
inline std::vector<metal::MetalBuffer> bindings(const metal::ComputeDispatch &dispatch){std::vector<metal::MetalBuffer> out;for(const auto &b:dispatch.buffers)out.push_back(b.buffer);return out;}
// Both legacy host validations and the expanded consumer aliases complete
// before changing the caller graph. All borrowed coefficient views are retained.
inline void addChain(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const metal::MetalBuffer &normalized,const FlashAffineProjection &down,
    const FlashAffineProjection *injection,const metal::MetalBuffer &expanded,
    const metal::MetalBuffer &gates,const FlashTensor &up,const metal::MetalBuffer &mixed,
    const metal::MetalBuffer &diag,FlashFloatDenseSmallRowsWorkspace &workspace,
    FlashHCGeometry geometry,bool verification,uint32_t maximumRows){
  requireFrozen();if(!requested()||!eligible(verification,geometry.rows,maximumRows)||expanded.sizeBytes()!=5120)throw std::invalid_argument("HC padding requires admitted main VerifyR4 eight-row scratch");
  metal::CommandGraph originalDown,originalUp;
  addHCFusedDown(originalDown,normalized,down,injection,expanded,gates,diag,geometry);
  addHCFusedUpMixF32Cache(backend,originalUp,normalized,backend.view(expanded,0,2560),up,mixed,diag,geometry,workspace);
  if(originalDown.dispatches().size()!=1||originalUp.dispatches().size()!=2)throw std::logic_error("HC original chain changed");
  const auto &d=originalDown.dispatches()[0],&u=originalUp.dispatches()[1];auto p=params<FlashHCFusedParams>(d,10);const auto q=params<FlashFloatDenseSmallRowsParams>(u,6);
  if(p.rows!=4||p.simdgroups!=4||p.arithmetic_mode||p.write_raw_up)throw std::logic_error("HC padding source helper requires literal explicit SG4");
  const FlashFloatDenseSmallRowsParams expected{4,8,320,2560,0,2560,8,32};if(std::memcmp(&q,&expected,sizeof(q)))throw std::logic_error("HC padded up descriptor changed");
  for(const auto &b:{normalized,up.buffer,mixed,diag})disjoint(expanded,b);
  if(injection)disjoint(expanded,gates);
  auto db=bindings(d),ub=bindings(u);if(db.size()!=10||ub.size()!=6)throw std::logic_error("HC original chain buffer slots changed");
  ub[0]=expanded;hc_pad_sep22::FlashHCDownPadParams privateParams{};privateParams.literal=p;privateParams.padded_rows=8;
  graph.add("flash_hc_down_pad_r4_q"+std::to_string(down.bits)+"_g"+std::to_string(down.groupSize)+"_s4",std::move(db),privateParams,d.threadgroups,d.threadsPerThreadgroup);
  graph.add(u.pipelineName,std::move(ub),q,u.threadgroups,u.threadsPerThreadgroup);record(geometry.rows);
}
}
