#pragma once
#include "source_identity.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "dev/benchmarks/raw_q4_verify_worker_sep22/policy.hpp"
#include "dev/benchmarks/raw_q5_rowpair_sep22/policy.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace splash::flash::raw_q5_verify_sep22 {
inline constexpr const char*kFlag="SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22";
inline constexpr const char*kAIR="99d57fd78957cfcc68ecc30d426c754b0dd702e42a32acef0008155234482a69";
inline constexpr const char*kScope="singleton VerifyR4 only; authenticated main GDN output Q5/G128/K6144/N2560; selected F32 cache and all other roles excluded";
inline constexpr const char*kMarker=";private-rawQ5-rowpair-mainGDNout-VerifyR4-sourceSha256=";
inline bool parse(const char*v){if(!v||std::string_view(v)=="0")return false;if(std::string_view(v)=="1")return true;throw std::invalid_argument(std::string(kFlag)+" must be0 or1");}
inline bool requested(){const bool now=parse(std::getenv(kFlag));static const bool frozen=now;if(now!=frozen)throw std::logic_error("rawQ5 rowpair selector changed after frozen construction");return frozen;}
inline void validateDependencies(){if(!requested())return;if(!raw_q4_verify_sep22::requested())throw std::invalid_argument("rawQ5 rowpair requires SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22=1");raw_q4_verify_sep22::validateDependencies();for(const char*n:{"SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22","SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22","SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22","SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22","SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"}){const char*v=std::getenv(n);if(!v||std::string_view(v)!="1")throw std::invalid_argument(std::string("rawQ5 rowpair requires ")+n+"=1");}const char*phase=std::getenv("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21");if(phase&&std::string_view(phase)!="0")throw std::invalid_argument("rawQ5 rowpair excludes phase-Q4 target");}
inline bool role(std::string_view prefix)noexcept{constexpr std::string_view lead="language_model.model.layers.";if(!prefix.starts_with(lead))return false;const auto t=prefix.substr(lead.size());const auto dot=t.find('.');if(dot==std::string_view::npos||dot==0||dot>2||(dot==2&&t[0]=='0'))return false;uint32_t layer=0;for(char x:t.substr(0,dot)){if(x<'0'||x>'9')return false;layer=layer*10+uint32_t(x-'0');}return layer<48&&layer%4!=3&&t.substr(dot+1)=="linear_attn.out_proj";}
inline bool geometry(uint32_t rows,bool verify,bool singleton,const FlashAffineProjection&p)noexcept{return rows==4&&verify&&singleton&&p.experts==1&&p.inputSize==6144&&p.outputSize==2560&&p.bits==5&&p.groupSize==128;}
inline bool selected(std::string_view prefix,uint32_t rows,bool verify,bool singleton,const FlashAffineProjection&p)noexcept{return role(prefix)&&geometry(rows,verify,singleton,p);}
inline std::atomic<uint64_t>graphCalls{0},graphRows{0};
inline bool tensorShape(const FlashTensor&t,uint64_t n,uint64_t k)noexcept{return t.shape.size()==2&&t.shape[0]==n&&t.shape[1]==k;}
inline bool planeMetadata(const FlashAffineProjection&p)noexcept {
  return geometry(4,true,true,p)&&p.weights&&p.scales&&p.biases&&
      p.weights->dtype==FlashDType::U32&&p.scales->dtype==FlashDType::BF16&&p.biases->dtype==FlashDType::BF16&&
      tensorShape(*p.weights,2560,960)&&tensorShape(*p.scales,2560,48)&&tensorShape(*p.biases,2560,48)&&
      p.weights->logicalBytes==9830400&&p.scales->logicalBytes==245760&&p.biases->logicalBytes==245760&&
      p.weightRowStrideBytes==3840&&p.weightExpertStrideBytes==9830400&&p.parameterRowStrideBytes==96&&p.parameterExpertStrideBytes==245760;
}
inline void validateInventory(const FlashWeights&w){
  if(!requested())return;
  if(w.sourceIdentity()!="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e")throw std::invalid_argument("rawQ5 original model source differs");
  uint32_t count=0;
  for(uint32_t i=0;i<48;++i){
    if(i%4==3)continue;
    const auto prefix="language_model.model.layers."+std::to_string(i)+".linear_attn.out_proj";
    if(!w.contains(prefix+".weight"))throw std::invalid_argument("rawQ5 main GDN output role absent");
    const auto&p=w.projection(prefix);
    if(!selected(prefix,4,true,true,p)||!planeMetadata(p)||flashFloatDenseSmallRowsPolicy(prefix,4,p.outputSize,p.inputSize,p.bits,p.groupSize)||
       !p.weights->buffer||!p.scales->buffer||!p.biases->buffer)
      throw std::invalid_argument("rawQ5 role inventory/cache selector/original source planes differ");
    ++count;
  }
  if(count!=36)throw std::invalid_argument("rawQ5 main GDN output inventory must contain36roles");
}
inline void validateCachePresence(const FlashWeights&w,const FlashFloatDenseCache&cache){
  if(!requested())return;uint32_t count=0;
  for(uint32_t i=0;i<48;++i){
    if(i%4==3)continue;
    const auto prefix="language_model.model.layers."+std::to_string(i)+".linear_attn.out_proj";
    if(!w.contains(prefix+".weight")||!planeMetadata(w.projection(prefix))||!cache.contains(prefix))
      throw std::invalid_argument("rawQ5 original F32 selector membership absent");
    const auto&t=cache.tensor(prefix);
    if(t.dtype!=FlashDType::F32||!tensorShape(t,2560,6144)||t.logicalBytes!=62914560)
      throw std::invalid_argument("rawQ5 original F32 backing identity/extent differs");
    ++count;
  }
  if(count!=36)throw std::invalid_argument("rawQ5 original cache member census differs");
}
inline std::string marker(){return requested()?std::string(kMarker)+kSourceIdentitySha256:std::string{};}
inline void add(metal::CommandGraph&graph,std::string_view prefix,metal::MetalBuffer input,const FlashAffineProjection&p,
    metal::MetalBuffer output,metal::MetalBuffer diag,uint32_t rows,bool verify,bool singleton){
  validateDependencies();if(!requested()||!selected(prefix,rows,verify,singleton,p)||flashFloatDenseSmallRowsPolicy(prefix,rows,p.outputSize,p.inputSize,p.bits,p.groupSize))throw std::invalid_argument("rawQ5 unqualified role/context/cache choice");
  metal::CommandGraph original;addAffine(original,input,p,output,diag,rows);
  if(original.dispatches().size()!=1)throw std::logic_error("rawQ5 original descriptor count differs");const auto&d=original.dispatches()[0];
  if(d.pipelineName!="flash_affine_mlx_qmv_f32xsum_v1_q5_g128"||d.threadgroups.x!=320||d.threadgroups.y!=4||d.threadgroups.z!=1||d.threadsPerThreadgroup.x!=64||d.threadsPerThreadgroup.y!=1||d.threadsPerThreadgroup.z!=1||d.buffers.size()!=7||d.bytes.size()!=1||d.bytes[0].index!=7||d.bytes[0].sizeBytes!=64||!d.bytes[0].data)throw std::logic_error("rawQ5 original native raw route differs");
  FlashAffineParams params{};std::memcpy(&params,d.bytes[0].data,64);std::array<::raw_q5_rowpair_sep22::Span,7>spans{};std::vector<metal::MetalBuffer>b;for(uint32_t i=0;i<7;++i){if(d.buffers[i].index!=i||!d.buffers[i].buffer||d.buffers[i].buffer.storage()!=metal::BufferStorage::Shared||!d.buffers[i].buffer.contents())throw std::invalid_argument("rawQ5 original Shared binding invalid");b.push_back(d.buffers[i].buffer);spans[i]={reinterpret_cast<uintptr_t>(b.back().contents()),b.back().sizeBytes()};}
  ::raw_q5_rowpair_sep22::validate(params,spans,320,2,1,64,true);
  graph.add("raw_q5_rowpair_sep22_timed",std::move(b),params,{320,2,1},{64,1,1});graphCalls.fetch_add(1,std::memory_order_relaxed);graphRows.fetch_add(4,std::memory_order_relaxed);
}
} // namespace splash::flash::raw_q5_verify_sep22
