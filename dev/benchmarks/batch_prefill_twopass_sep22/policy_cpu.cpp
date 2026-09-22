#include "policy.hpp"
#include <iostream>
using namespace splash::flash::batch_prefill_twopass_sep22;
int main() {
  uint64_t checks=0;
  for (uint32_t lanes=0;lanes<=8;++lanes) for(uint32_t rows=0;rows<=8192;++rows)
    for(bool fresh:{false,true}) for(bool enabled:{false,true}) {
      const bool expected=enabled&&fresh&&(lanes==2||lanes==4)&&rows==2048;
      if(eligible(lanes,rows,fresh,enabled)!=expected) return 1;
      ++checks;
    }
  const uint64_t offsets[]{0,25165824,27262976,31457280,56623104,459276288};
  const uint64_t sizes[]{25165824,2097152,4194304,25165824,402653184,50331648};
  for(size_t i=0;i<6;++i) {
    if(offsets[i]%16384||sizes[i]%16384||offsets[i]+sizes[i]>(i==5?plannedBytes:offsets[i+1]))return 2;
    ++checks;
  }
  if(offsets[5]+sizes[5]!=509607936||parse(nullptr)||parse("0")||!parse("1"))return 3;
  checks+=4;
  bool rejected=false;try{(void)parse("2");}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 4;++checks;
  std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"GPU_work\":false,\"device_or_payload_access\":false}\n";
}
