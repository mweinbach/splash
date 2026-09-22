#pragma once
#include "flash/FlashDenseSmallRows.hpp"
#include "dev/benchmarks/decode_bf16_narrow_sep21/KernelParams.hpp"
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::bf16_target_r4 {
inline constexpr const char *kFlag="SPLASH_FLASH_DECODE_BF16_R4_TARGETED";
inline constexpr std::string_view kSemantics=
  "private-main-physicalR4-only-sourceU32-affine-BF16scaleBias-cached-roundedBF16-wholeK-M8-qkvN10240K2560Q6G64-N64SG4-zN6144K2560Q6G64-N32SG2-plevalueN2560K2560Q4G64-N32SG2-qsaqN12288K2560Q4G64-N32SG2-v1";
inline bool readSwitch(const char *name) {
  const char *raw=std::getenv(name);
  if(!raw||std::string_view(raw)=="0")return false;
  if(std::string_view(raw)=="1")return true;
  throw std::invalid_argument(std::string(name)+" must be exactly0 or1");
}
inline bool requested() {
  if(readSwitch("SPLASH_FLASH_DECODE_BF16_DENSE"))
    throw std::invalid_argument("targeted R4 BF16 forbids legacy DECODE_BF16_DENSE policy");
  return readSwitch(kFlag);
}
inline void validateDependencies(bool enabled) {
  if(enabled&&(!readSwitch("SPLASH_FLASH_DENSE_CACHE")||!readSwitch("SPLASH_FLASH_FLOAT_DENSE_CACHE")||readSwitch("SPLASH_FLASH_DENSE_SMALL_ROWS")))
    throw std::invalid_argument("targeted R4 BF16 requires DENSE_CACHE=1 FLOAT_DENSE_CACHE=1 DENSE_SMALL_ROWS=0");
}
inline void validateFrozen(bool enabled) {
  if(requested()!=enabled)throw std::logic_error("targeted R4 BF16 changed after construction");
}
inline uint64_t workspaceBytes(bool enabled) { return enabled?uint64_t{16}*32768*2:0; }
inline std::optional<sep21_bf16_narrow::Variant> select(std::string_view prefix,uint32_t rows,uint32_t outputs,uint32_t inputs,
    uint32_t bits,uint32_t group,uint32_t experts,FlashDType sourceCodes,FlashDType sourceScales,FlashDType sourceBiases) {
  if(rows!=4||experts!=1||group!=64||sourceCodes!=FlashDType::U32||sourceScales!=FlashDType::BF16||sourceBiases!=FlashDType::BF16)return std::nullopt;
  constexpr std::string_view first="language_model.model.layers.";
  if(!prefix.starts_with(first))return std::nullopt;
  const auto tail=prefix.substr(first.size());const auto dot=tail.find('.');
  if(dot==std::string_view::npos||!dot||dot>2||(dot==2&&tail[0]=='0'))return std::nullopt;
  uint32_t layer=0;for(char digit:tail.substr(0,dot)){if(digit<'0'||digit>'9')return std::nullopt;layer=layer*10+uint32_t(digit-'0');}
  if(layer>=48)return std::nullopt;const auto role=tail.substr(dot+1);
  if(role=="linear_attn.in_proj_qkv"&&inputs==2560&&outputs==10240&&bits==6)return sep21_bf16_narrow::kVariants[5];
  if(role=="linear_attn.in_proj_z"&&inputs==2560&&outputs==6144&&bits==6)return sep21_bf16_narrow::kVariants[1];
  if(role=="ple.value_proj"&&inputs==2560&&outputs==2560&&bits==4)return sep21_bf16_narrow::kVariants[1];
  if(role=="self_attn.q_proj"&&inputs==2560&&outputs==12288&&bits==4)return sep21_bf16_narrow::kVariants[1];
  return std::nullopt;
}
inline void addProjection(metal::MetalBackend &backend,metal::CommandGraph &graph,metal::MetalBuffer input,const FlashTensor &weight,
    metal::MetalBuffer output,metal::MetalBuffer diagnostics,uint32_t rows,FlashDenseSmallRowsWorkspace &workspace,
    const sep21_bf16_narrow::Variant &variant) {
  // The existing host API validates shared views, alignment, extents, immutable
  // operand disjointness and arena ownership before these two exact dispatches.
  metal::CommandGraph validated;
  addDenseBF16SmallRows(backend,validated,input,weight,output,diagnostics,rows,workspace,FlashDenseSmallRowsTile::M8N64);
  if(rows!=4||weight.shape.size()!=2||variant.tileRows!=8||
      (variant.kernel!=sep21_bf16_narrow::kVariants[1].kernel&&variant.kernel!=sep21_bf16_narrow::kVariants[5].kernel))
    throw std::invalid_argument("targeted R4 BF16 unsupported selectedproducer");
  const uint32_t outputs=uint32_t(weight.shape[0]),inputs=uint32_t(weight.shape[1]);
  const FlashDenseSmallRowsParams pad{rows,8,inputs,outputs,0,outputs,8,64};
  graph.add("flash_dense_small_rows_pad",{input,workspace.paddedInput(),diagnostics},pad,
      {(uint64_t{8}*inputs+255)/256,1,1},{256,1,1});
  auto params=pad;params.tile_outputs=variant.tileOutputs;
  if(!sep21_bf16_narrow::geometry(params,variant))throw std::invalid_argument("targeted R4 BF16 launchgeometry invalid");
  graph.add(variant.kernel,{workspace.paddedInput(),weight.buffer,output,diagnostics},params,
      {outputs/variant.tileOutputs,1,1},{variant.threads(),1,1});
}
} // namespace splash::flash::bf16_target_r4
