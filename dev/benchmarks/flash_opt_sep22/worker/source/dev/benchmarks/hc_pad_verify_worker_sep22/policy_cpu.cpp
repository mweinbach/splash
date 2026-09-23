#include "worker_bridge.hpp"
#include <iostream>
int main(){using namespace splash::flash::hc_pad_verify_sep22;uint64_t checks=0;
  for(const char *v:std::array<const char*,3>{nullptr,"0","1"}){const bool expected=v&&*v=='1';if(parse(v)!=expected)return 1;++checks;}
  for(const char *v:{""," 1","01","true","2","-1","1\n"}){bool bad=false;try{(void)parse(v);}catch(const std::invalid_argument &){bad=true;}if(!bad)return 2;++checks;}
  for(uint32_t max=0;max<=2048;++max)for(uint32_t rows=0;rows<=32;++rows)for(bool verify:{false,true}){if(eligible(verify,rows,max)!=(verify&&rows==4&&max>=8))return 3;++checks;}
  if(sizeof(splash::flash::hc_pad_sep22::FlashHCDownPadParams)!=176||sizeof(splash::metal::CommandTiming)!=200)return 4;
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"checks\":"<<checks<<",\"scope\":\"mainVerifyR4 only/admitted8rows\"}\n";
}
