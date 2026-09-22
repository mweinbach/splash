#pragma once
#include "abi.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "metal/abi/FlashFloatDenseCache.h"
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::flash::hc_pad_sep22 {
enum class Scope:uint32_t {Prefill,Ordinary,Verify,Head,Batch};
inline constexpr bool eligible(Scope scope,uint32_t rows,uint64_t activatedBytes) {
  return scope==Scope::Verify && rows==4 && activatedBytes>=8ULL*320*2;
}
struct DownDebug {metal::MetalBuffer rawF32,rawBF16;};
inline void disjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  if(!a||!b||!a.contents()||!b.contents())throw std::invalid_argument("HC pad needs Shared full views");
  const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  if(x<=y?uint64_t(y-x)<a.sizeBytes():uint64_t(x-y)<b.sizeBytes())throw std::invalid_argument("HC pad source/output aliases");
}
inline std::vector<metal::MetalBuffer> buffers(const metal::ComputeDispatch &dispatch) {
  std::vector<metal::MetalBuffer> out;for(const auto &binding:dispatch.buffers)out.push_back(binding.buffer);return out;
}
template<class Params> inline Params params(const metal::ComputeDispatch &dispatch,uint32_t index) {
  Params p{};
  if(dispatch.bytes.size()!=1||dispatch.bytes[0].index!=index||dispatch.bytes[0].sizeBytes!=sizeof(p))throw std::logic_error("HC pad original parameter binding changed");
  std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));return p;
}
inline void addDown(metal::CommandGraph &graph,metal::MetalBuffer normalized,
    const FlashAffineProjection &down,const FlashAffineProjection *injection,
    metal::MetalBuffer activation,metal::MetalBuffer gates,metal::MetalBuffer diag,
    Scope scope,bool candidate,const DownDebug &debug={}) {
  if(!eligible(scope,4,activation.sizeBytes())||activation.sizeBytes()!=8ULL*320*2)
    throw std::invalid_argument("HC producer padding is Verify R4 with exact eight-row scratch only");
  metal::CommandGraph validated;
  addHCFusedDown(validated,normalized,down,injection,activation,gates,diag,{4,2560,4,1e-6f});
  if(validated.dispatches().size()!=1)throw std::logic_error("HC pad down control shape changed");
  const auto &original=validated.dispatches()[0];auto p=params<FlashHCFusedParams>(original,10);
  if(p.rows!=4||p.simdgroups!=4||p.arithmetic_mode!=0)throw std::logic_error("HC pad requires unchanged explicit coefficient SG4 helper");
  FlashHCDownPadParams privateParams{};privateParams.literal=p;privateParams.padded_rows=8;
  auto bound=buffers(original);const bool probe=bool(debug.rawF32)||bool(debug.rawBF16);
  if(probe){const uint64_t count=4ULL*(320+(injection?4:0));
    if(!debug.rawF32||!debug.rawBF16||debug.rawF32.sizeBytes()!=count*4||debug.rawBF16.sizeBytes()!=count*2)throw std::invalid_argument("HC producer probe extent differs");
    for(const auto &b:bound){disjoint(debug.rawF32,b);disjoint(debug.rawBF16,b);}disjoint(debug.rawF32,debug.rawBF16);
    bound.push_back(debug.rawF32);bound.push_back(debug.rawBF16);privateParams.literal.write_raw_up=1;}
  if(!candidate&&!probe){addHCFusedDown(graph,normalized,down,injection,activation,gates,diag,{4,2560,4,1e-6f});return;}
  const std::string suffix="q"+std::to_string(down.bits)+"_g"+std::to_string(down.groupSize)+"_s4";
  const auto name=probe?(candidate?"flash_hc_down_pad_probe_r4_":"flash_hc_down_control_probe_r4_"):("flash_hc_down_pad_r4_");
  graph.add(name+suffix,std::move(bound),privateParams,{(320+(injection?4u:0u)+3)/4,4,1},{128,1,1});
}
struct UpDebug {metal::MetalBuffer rawBF16,rawF32;};
inline void addUp(metal::MetalBackend &backend,metal::CommandGraph &graph,
    metal::MetalBuffer normalized,metal::MetalBuffer activation,const FlashTensor &weight,
    metal::MetalBuffer mixed,metal::MetalBuffer diag,FlashFloatDenseSmallRowsWorkspace &workspace,
    metal::MetalBuffer controlPadding,Scope scope,bool candidate,const UpDebug &debug={}) {
  if(!eligible(scope,4,activation.sizeBytes())||activation.sizeBytes()!=5120||controlPadding.sizeBytes()!=5120)
    throw std::invalid_argument("HC up requires Verify R4 expanded activation/padding extent");
  const auto active=backend.view(activation,0,2560);metal::CommandGraph validated;
  addHCFusedUpMixF32Cache(backend,validated,normalized,active,weight,mixed,diag,{4,2560,4,1e-6f},workspace,debug.rawBF16);
  if(validated.dispatches().size()!=2)throw std::logic_error("HC original pad/up graph changed");
  const auto &pad=validated.dispatches()[0],&up=validated.dispatches()[1];
  const auto paddingParams=params<FlashFloatDenseSmallRowsParams>(pad,3);const auto p=params<FlashFloatDenseSmallRowsParams>(up,6);
  const FlashFloatDenseSmallRowsParams expected{4,8,320,2560,0,2560,8,32};
  if(std::memcmp(&p,&expected,sizeof(p)))throw std::logic_error("HC up descriptor/geometry changed");
  const auto source=candidate?activation:controlPadding;
  for(const auto &b:{normalized,weight.buffer,mixed,diag})disjoint(source,b);
  if(debug.rawBF16)disjoint(source,debug.rawBF16);
  if(debug.rawF32){
    if(!debug.rawBF16||debug.rawF32.sizeBytes()!=4ULL*10240*4)throw std::invalid_argument("HC up F32 probe extent differs");
    for(const auto &b:{source,normalized,weight.buffer,mixed,diag,debug.rawBF16})disjoint(debug.rawF32,b);}
  if(!candidate){disjoint(controlPadding,activation);graph.add(pad.pipelineName,{active,controlPadding,diag},paddingParams,{10,1,1});}
  auto bound=buffers(up);if(bound.size()!=6)throw std::logic_error("HC original up bindings changed");bound[0]=source;
  if(debug.rawF32){bound.push_back(debug.rawF32);graph.add("flash_hc_pad_up_probe_f32_m8_n32_s4",std::move(bound),p,{80,1,1},{128,1,1});}
  else graph.add(up.pipelineName,std::move(bound),p,{80,1,1},{128,1,1});
}
}
