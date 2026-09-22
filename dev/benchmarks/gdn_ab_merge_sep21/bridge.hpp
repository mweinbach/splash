#pragma once
#include "abi.hpp"
#include "flash/FlashAffine.hpp"
#include "metal/CommandGraph.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::flash::gdn_ab_merge {
inline constexpr const char *kFlag="SPLASH_FLASH_GDN_AB_MERGE_SEP21";
inline constexpr const char *kSemantics="exact-main-GDN-AB-paired-scheduling-R1R4-K2560N48-Q5G128orQ6G64-original-lane-xsum-qdot-simdsum-BF16-boundaries-v1";
inline bool switchValue(const char *name){const char *value=std::getenv(name);if(!value||std::string_view(value)=="0")return false;if(std::string_view(value)=="1")return true;throw std::invalid_argument(std::string(name)+" must be0 or1");}
inline bool requested(){
  const bool now=switchValue(kFlag);
  static const bool frozen=now;
  if(now!=frozen)throw std::logic_error("GDNABpair flag changed after frozenconstruction");
  if(now&&!switchValue("SPLASH_FLASH_QMV_F32"))throw std::invalid_argument("GDNABpair requiresactive QMV_F32=1");
  return now;
}
inline std::atomic<uint64_t> graphCalls{0},graphRows{0};
struct Counters{uint64_t calls,rows;};
inline Counters counters(){return{graphCalls.load(std::memory_order_relaxed),graphRows.load(std::memory_order_relaxed)};}
inline void disjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  if(!a||!b||a.storage()!=metal::BufferStorage::Shared||b.storage()!=metal::BufferStorage::Shared||!a.contents()||!b.contents())
    throw std::invalid_argument("GDNABpair requiresShared addressable views");
  const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  if(x<=y?uint64_t(y-x)<a.sizeBytes():uint64_t(x-y)<b.sizeBytes())throw std::invalid_argument("GDNABpair writable/readonly views overlap");
}
inline bool geometry(const FlashAffineProjection &p,uint32_t rows) {
  return (rows==1||rows==4)&&p.experts==1&&p.outputSize==48&&p.inputSize==2560&&
      ((p.bits==5&&p.groupSize==128)||(p.bits==6&&p.groupSize==64))&&p.weights&&p.scales&&p.biases&&
      p.weights->dtype==FlashDType::U32&&p.scales->dtype==FlashDType::BF16&&p.biases->dtype==FlashDType::BF16;
}
struct Taps{metal::MetalBuffer aRaw,bRaw;};
inline void add(metal::CommandGraph &graph,metal::MetalBuffer input,const FlashAffineProjection &a,const FlashAffineProjection &b,
    metal::MetalBuffer aOut,metal::MetalBuffer bOut,metal::MetalBuffer diagnostics,uint32_t rows,const Taps &taps={},int probePlane=-1) {
  if(probePlane<-1||probePlane>1)throw std::invalid_argument("GDNABpair invalidprobeplane");
  if(!geometry(a,rows)||!geometry(b,rows)||!flashAffineFastEnabled())throw std::invalid_argument("GDNABpair unqualified source/row/route");
  metal::CommandGraph av,bv;
  addAffine(av,input,a,aOut,diagnostics,rows);addAffine(bv,input,b,bOut,diagnostics,rows);
  if(av.dispatches().size()!=1||bv.dispatches().size()!=1)throw std::logic_error("GDNABpair originalgraph changed");
  const auto &ad=av.dispatches()[0],&bd=bv.dispatches()[0];
  if(ad.pipelineName!="flash_affine_mlx_qmv_f32xsum_v1_q"+std::to_string(a.bits)+"_g"+std::to_string(a.groupSize)||
     bd.pipelineName!="flash_affine_mlx_qmv_f32xsum_v1_q"+std::to_string(b.bits)+"_g"+std::to_string(b.groupSize))
    throw std::invalid_argument("GDNABpair mustpreserve originalactiveQMV route");
  const std::array<metal::MetalBuffer,7> readonly{input,a.weights->buffer,a.scales->buffer,a.biases->buffer,b.weights->buffer,b.scales->buffer,b.biases->buffer};
  std::vector<metal::MetalBuffer> writable{aOut,bOut,diagnostics};
  const bool probe=bool(taps.aRaw)||bool(taps.bRaw);
  if(probe){
    if(!taps.aRaw||!taps.bRaw||taps.aRaw.sizeBytes()<uint64_t{rows}*48*4||taps.bRaw.sizeBytes()<uint64_t{rows}*48*4)
      throw std::invalid_argument("GDNABpair tapextent short");
    writable.push_back(taps.aRaw);writable.push_back(taps.bRaw);
  }
  if(reinterpret_cast<uintptr_t>(input.contents())%2||reinterpret_cast<uintptr_t>(aOut.contents())%2||reinterpret_cast<uintptr_t>(bOut.contents())%2||
     reinterpret_cast<uintptr_t>(diagnostics.contents())%4||(probe&&(reinterpret_cast<uintptr_t>(taps.aRaw.contents())%4||reinterpret_cast<uintptr_t>(taps.bRaw.contents())%4)))
    throw std::invalid_argument("GDNABpair typedviewalignment invalid");
  for(const auto &w:writable)for(const auto &r:readonly)disjoint(w,r);
  for(size_t x=0;x<writable.size();++x)for(size_t y=x+1;y<writable.size();++y)disjoint(writable[x],writable[y]);
  GDNABMergeParams params{};
  if(ad.bytes.size()!=1||bd.bytes.size()!=1||ad.bytes[0].index!=7||bd.bytes[0].index!=7||ad.bytes[0].sizeBytes!=64||bd.bytes[0].sizeBytes!=64)
    throw std::logic_error("GDNABpair parameterABI changed");
  std::memcpy(&params.a,ad.bytes[0].data,64);std::memcpy(&params.b,bd.bytes[0].data,64);
  std::vector<metal::MetalBuffer> buffers{input,a.weights->buffer,a.scales->buffer,a.biases->buffer,b.weights->buffer,b.scales->buffer,b.biases->buffer,aOut,bOut,diagnostics};
  std::string pipeline="flash_gdn_ab_merge_qmv_f32_v1";
  if(probe){buffers.push_back(taps.aRaw);buffers.push_back(taps.bRaw);
    pipeline=probePlane==0?"flash_gdn_ab_single_probe_a_qmv_f32_v1":probePlane==1?"flash_gdn_ab_single_probe_b_qmv_f32_v1":"flash_gdn_ab_merge_probe_qmv_f32_v1";}
  else if(probePlane!=-1)throw std::invalid_argument("GDNABpair planechoice onlyforprobe");
  graph.add(pipeline,std::move(buffers),params,{6,rows,probe&&probePlane>=0?1u:2u},{64,1,1});
  if(!probe){graphCalls.fetch_add(1,std::memory_order_relaxed);graphRows.fetch_add(rows,std::memory_order_relaxed);}
}
} // namespace splash::flash::gdn_ab_merge
