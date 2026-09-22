#include "policy.hpp"
#include <iostream>
using namespace splash::flash::immutable_interval_worker_sep22;
int main(int argc,char**argv){
  try{
    if(argc==2&&std::string_view(argv[1])=="--startup"){startup();std::cout<<"{\"pass\":true,\"GPU_executed\":false}\n";return 0;}
    if(argc==2&&std::string_view(argv[1])=="--lifetime"){
      const bool first=requested();setenv(flag,first?"0":"1",1);bool rejected=false;
      try{(void)requested();}catch(const std::logic_error&){rejected=true;}
      if(!rejected)throw std::runtime_error("index flag lifetime mutation accepted");
      std::cout<<"{\"pass\":true,\"GPU_executed\":false}\n";return 0;
    }
    uint64_t checks=0;const auto require=[&](bool yes){if(!yes)throw std::runtime_error("index policy CPU refusal");++checks;};
    require(!parse(nullptr));require(!parse("0"));require(parse("1"));
    for(const char*bad:{"","2","01","true"," 1","-1"}){bool rejected=false;try{(void)parse(bad);}catch(const std::invalid_argument&){rejected=true;}require(rejected);}
    require(finalizedTables.load()==0&&finalizedSpans.load()==0&&indexableTables.load()==0&&indexedAccepts.load()==0&&originalCallbacks.load()==0);
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"GPU_executed\":false,\"model_payload_reads\":0}\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
