#include "bridge.hpp"
#include <iostream>

int main() {
  using namespace splash::flash;using namespace bf16_target_r4;
  const auto check=[](bool value){if(!value)throw std::runtime_error("targeted R4 BF16 CPUcheckfailed");};
  unsetenv(kFlag);unsetenv("SPLASH_FLASH_DECODE_BF16_DENSE");check(!requested()&&workspaceBytes(false)==0);
  setenv(kFlag,"1",1);setenv("SPLASH_FLASH_DENSE_CACHE","1",1);setenv("SPLASH_FLASH_FLOAT_DENSE_CACHE","1",1);setenv("SPLASH_FLASH_DENSE_SMALL_ROWS","0",1);
  validateDependencies(requested());validateFrozen(true);check(workspaceBytes(true)==1048576);
  uint32_t legal=0,rejected=0;
  for(uint32_t layer=0;layer<48;++layer){const auto first="language_model.model.layers."+std::to_string(layer)+".";
    for(const auto tuple:std::array<std::array<uint32_t,4>,4>{{{10240,2560,6,5},{6144,2560,6,1},{2560,2560,4,1},{12288,2560,4,1}}}){
      const uint32_t roleIndex=tuple[0]==10240?0:tuple[0]==6144?1:tuple[0]==2560?2:3;
      const std::array<const char *,4> roles{"linear_attn.in_proj_qkv","linear_attn.in_proj_z","ple.value_proj","self_attn.q_proj"};
      const auto prefix=first+roles[roleIndex];const auto variant=select(prefix,4,tuple[0],tuple[1],tuple[2],64,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16);
      check(bool(variant)&&variant->tileRows==8&&variant->tileOutputs==sep21_bf16_narrow::kVariants[tuple[3]].tileOutputs&&variant->simdgroups==sep21_bf16_narrow::kVariants[tuple[3]].simdgroups);++legal;
      for(uint32_t rows:{0u,1u,2u,3u,5u,8u,16u,2048u}){check(!select(prefix,rows,tuple[0],tuple[1],tuple[2],64,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;}
      for(uint32_t bits:{4u,5u,6u,8u})if(bits!=tuple[2]){check(!select(prefix,4,tuple[0],tuple[1],bits,64,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;}
      for(uint32_t group:{32u,128u}){check(!select(prefix,4,tuple[0],tuple[1],tuple[2],group,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;}
      check(!select(prefix,4,tuple[0],tuple[1],tuple[2],64,512,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;
      check(!select(prefix,4,tuple[0],tuple[1],tuple[2],64,1,FlashDType::BF16,FlashDType::BF16,FlashDType::BF16));++rejected;
    }
    for(const auto role:{"linear_attn.out_proj","self_attn.o_proj","self_attn.k_proj","self_attn.v_proj","self_attn.indexer.index_qk_proj","attn_hyper_connection.input_mix_weight_up"}){
      check(!select(first+role,4,2560,2560,4,64,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;}
  }
  for(const auto malformed:{"language_model.model.layers.48.self_attn.q_proj","language_model.model.layers.03.self_attn.q_proj","mtp.layers.0.self_attn.q_proj","language_model.lm_head"}){
    check(!select(malformed,4,12288,2560,4,64,1,FlashDType::U32,FlashDType::BF16,FlashDType::BF16));++rejected;}
  setenv(kFlag,"0",1);bool frozen=false;try{validateFrozen(true);}catch(const std::logic_error &){frozen=true;}check(frozen);
  for(const auto invalid:{"","2","true","01"," 1","1 "}){setenv(kFlag,invalid,1);bool failure=false;try{(void)requested();}catch(const std::invalid_argument &){failure=true;}check(failure);++rejected;}
  setenv(kFlag,"0",1);setenv("SPLASH_FLASH_DECODE_BF16_DENSE","1",1);bool legacy=false;try{(void)requested();}catch(const std::invalid_argument &){legacy=true;}check(legacy);
  std::cout<<"{\"pass\":true,\"legal_cases\":"<<legal<<",\"rejected_cases\":"<<rejected<<",\"gpu_work\":false,\"model_payload_bytes_read\":0}\n";
}
