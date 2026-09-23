#include "policy.hpp"
#include <iostream>
using namespace splash::flash;
namespace r5=compact_native_r5_verify_sep22;
int main(int argc,char**argv) {
  try {
    uint64_t checks=0;auto require=[&](bool value){if(!value)throw std::runtime_error("R5 CPU contract failed");++checks;};
    if(argc==2&&std::string_view(argv[1])=="--dependency"){r5::validateDependencies();std::cout<<"{\"pass\":true,\"GPU_work\":false}\n";return 0;}
    if(argc==2&&std::string_view(argv[1])=="--depth"){const char*value=std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH");r5::validateDepth(r5::env("SPLASH_FLASH_MTP"),value&&std::string_view(value)=="4"?4:3,value!=nullptr);std::cout<<"{\"pass\":true,\"GPU_work\":false}\n";return 0;}
    if(argc==2&&std::string_view(argv[1])=="--lifetime"){const bool initial=r5::requested();::setenv(r5::kFlag,initial?"0":"1",1);bool rejected=false;try{(void)r5::requested();}catch(const std::logic_error&){rejected=true;}require(rejected);std::cout<<"{\"pass\":true,\"GPU_work\":false}\n";return 0;}
    for(uint32_t rows=0;rows<=8192;++rows)for(bool verification:{false,true})for(bool singleton:{false,true})for(bool enabled:{false,true})
      require(r5::eligibleFor(rows,verification,singleton,enabled)==(rows==5&&verification&&singleton&&enabled));
    require(!r5::parse(nullptr));require(!r5::parse("0"));require(r5::parse("1"));
    for(const char*bad:{"","2","true","01"," 1","-1"}){bool rejected=false;try{(void)r5::parse(bad);}catch(const std::invalid_argument&){rejected=true;}require(rejected);}
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"GPU_work\":false,\"model_operand_capture_payload_reads\":0}\n";
    return 0;
  }catch(const std::exception&error){std::cerr<<error.what()<<'\n';return 1;}
}
