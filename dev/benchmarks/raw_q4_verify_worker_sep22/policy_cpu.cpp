#include "policy.hpp"
#include <iostream>
using namespace splash::flash;
int main(int argc,char**argv){try{uint64_t n=0;const auto require=[&](bool x){if(!x)throw std::runtime_error("rawQ4 CPU policy failure");++n;};FlashAffineProjection p{};p.experts=1;p.inputSize=2560;p.outputSize=10240;p.bits=4;p.groupSize=64;
  for(uint32_t i=0;i<64;++i){const auto prefix="language_model.model.layers."+std::to_string(i)+".linear_attn.in_proj_qkv";require(raw_q4_verify_sep22::role(prefix)==(i<48&&i%4!=3));for(uint32_t r=0;r<17;++r)for(bool v:{false,true})for(bool s:{false,true})require(raw_q4_verify_sep22::selected(prefix,r,v,s,p)==(i<48&&i%4!=3&&r==4&&v&&s));}
  for(const char*x:{"language_model.model.layers.01.linear_attn.in_proj_qkv","language_model.model.layers.1.ple.key_proj","language_model.model.layers.1.linear_attn.in_proj_z","mtp.fc_hidden","language_model.lm_head","language_model.model.layers.x.linear_attn.in_proj_qkv"})require(!raw_q4_verify_sep22::role(x));
  for(uint32_t field=0;field<6;++field){auto bad=p;switch(field){case 0:bad.experts=2;break;case 1:bad.bits=5;break;case 2:bad.groupSize=128;break;case 3:bad.inputSize=6144;break;case 4:bad.outputSize=12288;break;case 5:bad.bits=6;break;}require(!raw_q4_verify_sep22::selected("language_model.model.layers.1.linear_attn.in_proj_qkv",4,true,true,bad));}
  require(!raw_q4_verify_sep22::parse(nullptr));require(!raw_q4_verify_sep22::parse("0"));require(raw_q4_verify_sep22::parse("1"));for(const char*x:{"","2","true","01"," 1","-1"}){bool bad=false;try{raw_q4_verify_sep22::parse(x);}catch(const std::invalid_argument&){bad=true;}require(bad);}
  if(argc==2&&std::string_view(argv[1])=="--dependency"){raw_q4_verify_sep22::validateDependencies();std::cout<<"{\"pass\":true,\"GPU_work\":false}\n";return 0;}
  if(argc==2&&std::string_view(argv[1])=="--lifetime"){const bool first=raw_q4_verify_sep22::requested();setenv(raw_q4_verify_sep22::kFlag,first?"0":"1",1);bool bad=false;try{(void)raw_q4_verify_sep22::requested();}catch(const std::logic_error&){bad=true;}require(bad);std::cout<<"{\"pass\":true,\"GPU_work\":false}\n";return 0;}
  std::cout<<"{\"pass\":true,\"checks\":"<<n<<",\"GPU_work\":false,\"payload_reads\":0}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
