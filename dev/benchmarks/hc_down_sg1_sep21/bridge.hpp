#pragma once
#include "flash/FlashHCFused.hpp"
#include "metal/abi/FlashHCFused.h"
#include <cstring>
#include <stdexcept>

namespace splash::flash::hcdown_sg1 {
enum class Mode:uint32_t{Only=0};
inline const char *modeName(Mode mode){if(mode!=Mode::Only)throw std::invalid_argument("invalidSG1mode");return "exact-fullK-lane32-oneCTA-oneSIMD-peroutput";}
struct Debug{metal::MetalBuffer rawBF16,rawF32;};
inline void disjoint(metal::MetalBuffer a,metal::MetalBuffer b){
  if(!a||!b||!a.contents()||!b.contents())throw std::invalid_argument("HCsg1tap needsSharedviews");
  const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  if(x<=y?uint64_t(y-x)<a.sizeBytes():uint64_t(x-y)<b.sizeBytes())throw std::invalid_argument("HCsg1tapaliasesexistingoperand");
}
inline void append(metal::CommandGraph &graph,metal::MetalBuffer normalized,const FlashAffineProjection &down,const FlashAffineProjection *injection,
    metal::MetalBuffer activated,metal::MetalBuffer gates,metal::MetalBuffer diag,uint32_t rows,uint32_t sg,const Debug &debug={}) {
  if((rows!=1&&rows!=4)||(sg!=1&&sg!=4))throw std::invalid_argument("HCsg1screen needsR1/R4 SG1/4");
  metal::CommandGraph validated;addHCFusedDown(validated,normalized,down,injection,activated,gates,diag,{rows,2560,4,1e-6f});
  if(validated.dispatches().size()!=1)throw std::logic_error("HCsg1validationgraphshape changed");
  const auto &original=validated.dispatches()[0];FlashHCFusedParams p{};
  if(original.bytes.size()!=1||original.bytes[0].index!=10||original.bytes[0].sizeBytes!=sizeof(p))throw std::logic_error("HCsg1baselineparameterABI changed");
  std::memcpy(&p,original.bytes[0].data,sizeof(p));p.simdgroups=sg;
  std::vector<metal::MetalBuffer> buffers;for(const auto &binding:original.buffers)buffers.push_back(binding.buffer);
  const bool probe=bool(debug.rawBF16)||bool(debug.rawF32);
  if(probe){
    const uint64_t count=uint64_t{rows}*(320+(injection?4:0));
    if(!debug.rawBF16||!debug.rawF32||debug.rawBF16.sizeBytes()<count*2||debug.rawF32.sizeBytes()<count*4)throw std::invalid_argument("HCsg1tapextent short");
    for(const auto &buffer:buffers){disjoint(debug.rawBF16,buffer);disjoint(debug.rawF32,buffer);}disjoint(debug.rawBF16,debug.rawF32);
    buffers.push_back(debug.rawF32);buffers.push_back(debug.rawBF16);p.write_raw_up=1;
  }
  const std::string suffix="q"+std::to_string(down.bits)+"_g"+std::to_string(down.groupSize);
  if(!probe&&sg==4){addHCFusedDown(graph,normalized,down,injection,activated,gates,diag,{rows,2560,4,1e-6f});return;}
  graph.add(probe?"flash_hc_down_probe_"+suffix+"_s"+std::to_string(sg):"flash_hc_down_sg1_"+suffix,std::move(buffers),p,
      {sg==1?uint64_t(320+(injection?4:0)):uint64_t((320+(injection?4:0)+3)/4),rows,1},{sg*32,1,1});
}
inline void addCandidate(metal::MetalBackend &,metal::CommandGraph &graph,metal::MetalBuffer x,const FlashAffineProjection &down,const FlashAffineProjection *injection,
    metal::MetalBuffer activation,metal::MetalBuffer gates,metal::MetalBuffer diag,uint32_t rows,Mode mode,const Debug &debug={}) {
  (void)modeName(mode);append(graph,x,down,injection,activation,gates,diag,rows,1,debug);
}
inline void addWitness(metal::MetalBackend &,metal::CommandGraph &graph,metal::MetalBuffer x,const FlashAffineProjection &down,const FlashAffineProjection *injection,
    metal::MetalBuffer activation,metal::MetalBuffer gates,metal::MetalBuffer diag,uint32_t rows,const Debug &debug) {
  append(graph,x,down,injection,activation,gates,diag,rows,4,debug);
}
} // namespace splash::flash::hcdown_sg1
