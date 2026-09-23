#pragma once
#include "source_identity.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "dev/benchmarks/raw_q4_rowpair_sep22/policy.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace splash::flash::raw_q4_verify_sep22 {
inline constexpr const char*kFlag="SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22";
inline constexpr const char*kAIR="4a671caf5641a451f5d60d5d2295b9064e4d78bf23effa74c5b86eaeeaec1003";
inline constexpr const char*kScope="singleton VerifyR4 only; authenticated main GDN QKV Q4/G64/K2560/N10240; selected F32 cache and PLE excluded";
inline constexpr const char*kMarker=";private-rawQ4-rowpair-mainGDNQKV-VerifyR4-sourceSha256=";
inline bool parse(const char*v){if(!v||std::string_view(v)=="0")return false;if(std::string_view(v)=="1")return true;throw std::invalid_argument(std::string(kFlag)+" must be0 or1");}
inline bool requested(){const bool now=parse(std::getenv(kFlag));static const bool frozen=now;if(now!=frozen)throw std::logic_error("rawQ4 rowpair selector changed after frozen construction");return frozen;}
inline void validateDependencies(){if(!requested())return;for(const char*n:{"SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22","SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22","SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22","SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22","SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"}){const char*v=std::getenv(n);if(!v||std::string_view(v)!="1")throw std::invalid_argument(std::string("rawQ4 rowpair requires ")+n+"=1");}const char*phase=std::getenv("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21");if(phase&&std::string_view(phase)!="0")throw std::invalid_argument("rawQ4 rowpair excludes phase-Q4 target");}
inline bool role(std::string_view prefix)noexcept{constexpr std::string_view lead="language_model.model.layers.";if(!prefix.starts_with(lead))return false;const auto t=prefix.substr(lead.size());const auto dot=t.find('.');if(dot==std::string_view::npos||dot==0||dot>2||(dot==2&&t[0]=='0'))return false;uint32_t layer=0;for(char x:t.substr(0,dot)){if(x<'0'||x>'9')return false;layer=layer*10+uint32_t(x-'0');}return layer<48&&layer%4!=3&&t.substr(dot+1)=="linear_attn.in_proj_qkv";}
inline bool geometry(uint32_t rows,bool verify,bool singleton,const FlashAffineProjection&p)noexcept{return rows==4&&verify&&singleton&&p.experts==1&&p.inputSize==2560&&p.outputSize==10240&&p.bits==4&&p.groupSize==64;}
inline bool selected(std::string_view prefix,uint32_t rows,bool verify,bool singleton,const FlashAffineProjection&p)noexcept{return role(prefix)&&geometry(rows,verify,singleton,p);}
inline std::atomic<uint64_t>graphCalls{0},graphRows{0};
inline void validateInventory(const FlashWeights&w){if(!requested())return;if(w.sourceIdentity()!="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e")throw std::invalid_argument("rawQ4 original model source differs");uint32_t count=0;for(uint32_t i=0;i<48;++i){const auto prefix="language_model.model.layers."+std::to_string(i)+".linear_attn.in_proj_qkv";if(i%4==3||!w.contains(prefix+".weight"))continue;const auto&p=w.projection(prefix);if(p.bits!=4)continue;if(!selected(prefix,4,true,true,p)||flashFloatDenseSmallRowsPolicy(prefix,4,p.outputSize,p.inputSize,p.bits,p.groupSize)||!p.weights||!p.scales||!p.biases||p.weights->dtype!=FlashDType::U32||p.scales->dtype!=FlashDType::BF16||p.biases->dtype!=FlashDType::BF16||p.weights->logicalBytes!=13107200||p.scales->logicalBytes!=819200||p.biases->logicalBytes!=819200||p.weightRowStrideBytes!=1280||p.parameterRowStrideBytes!=80)throw std::invalid_argument("rawQ4 role inventory/cache selector/original source planes differ");++count;}if(count!=26)throw std::invalid_argument("rawQ4 main GDN QKV inventory must contain26roles");}
inline void validateCachePresence(const FlashWeights&w,const FlashFloatDenseCache&cache){if(!requested())return;uint32_t count=0;for(uint32_t i=0;i<48;++i){const auto p="language_model.model.layers."+std::to_string(i)+".linear_attn.in_proj_qkv";if(i%4==3||!w.contains(p+".weight")||w.projection(p).bits!=4)continue;if(!cache.contains(p))throw std::invalid_argument("rawQ4 original F32 selector membership absent");const auto&t=cache.tensor(p);if(t.dtype!=FlashDType::F32||t.logicalBytes!=104857600)throw std::invalid_argument("rawQ4 original F32 backing identity/extent differs");++count;}if(count!=26)throw std::invalid_argument("rawQ4 original cache member census differs");}
inline std::string marker(){return requested()?std::string(kMarker)+kSourceIdentitySha256:std::string{};}
inline void add(metal::CommandGraph&graph,std::string_view prefix,metal::MetalBuffer input,const FlashAffineProjection&p,
    metal::MetalBuffer output,metal::MetalBuffer diag,uint32_t rows,bool verify,bool singleton){
  validateDependencies();if(!requested()||!selected(prefix,rows,verify,singleton,p)||flashFloatDenseSmallRowsPolicy(prefix,rows,p.outputSize,p.inputSize,p.bits,p.groupSize))throw std::invalid_argument("rawQ4 unqualified role/context/cache choice");
  metal::CommandGraph original;addAffine(original,input,p,output,diag,rows);
  if(original.dispatches().size()!=1)throw std::logic_error("rawQ4 original descriptor count differs");const auto&d=original.dispatches()[0];
  if(d.pipelineName!="flash_affine_mlx_qmv_f32xsum_v1_q4_g64"||d.threadgroups.x!=1280||d.threadgroups.y!=4||d.threadgroups.z!=1||d.threadsPerThreadgroup.x!=64||d.threadsPerThreadgroup.y!=1||d.threadsPerThreadgroup.z!=1||d.buffers.size()!=7||d.bytes.size()!=1||d.bytes[0].index!=7||d.bytes[0].sizeBytes!=64||!d.bytes[0].data)throw std::logic_error("rawQ4 original native raw route differs");
  FlashAffineParams params{};std::memcpy(&params,d.bytes[0].data,64);std::array<::raw_q4_rowpair_sep22::Span,7>spans{};std::vector<metal::MetalBuffer>b;for(uint32_t i=0;i<7;++i){if(d.buffers[i].index!=i||!d.buffers[i].buffer||d.buffers[i].buffer.storage()!=metal::BufferStorage::Shared||!d.buffers[i].buffer.contents())throw std::invalid_argument("rawQ4 original Shared binding invalid");b.push_back(d.buffers[i].buffer);spans[i]={reinterpret_cast<uintptr_t>(b.back().contents()),b.back().sizeBytes()};}
  ::raw_q4_rowpair_sep22::validate(params,spans,1280,2,1,64,true);
  graph.add("raw_q4_rowpair_sep22_timed",std::move(b),params,{1280,2,1},{64,1,1});graphCalls.fetch_add(1,std::memory_order_relaxed);graphRows.fetch_add(4,std::memory_order_relaxed);
}
} // namespace splash::flash::raw_q4_verify_sep22
