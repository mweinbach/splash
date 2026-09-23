#include "worker_bridge.hpp"
#include "flash/FlashForward.hpp"
#include <cstring>
#include <vector>
#include <iostream>
namespace qsa=splash::flash::prefill_qsa_twopass_sep21;
int main(int argc,char **argv) {
  try {
    uint64_t checks=0;auto require=[&](bool value){if (!value) throw std::runtime_error("QSA worker CPU policy failed");++checks;};
    const std::string mode=argc>1?argv[1]:"";
    if (mode=="--freeze0" || mode=="--freeze1") {const bool selected=mode=="--freeze1";setenv(qsa::kFlag,selected?"1":"0",1);
      for (const char *dep:{"SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8"}) setenv(dep,"1",1);
      require(qsa::requested()==selected);setenv(qsa::kFlag,selected?"0":"1",1);require(qsa::requested()==selected);
      std::vector<std::pair<uint32_t,uint64_t>> plans;
      for (uint32_t rows:{128u,512u,2048u,8192u}) {const auto plan=splash::flash::FlashForward::workspacePlannedBytes(16384,rows,4);
        require(plan>=qsa::plannedExtraBytes(rows,selected));plans.emplace_back(rows,plan);}
      std::cout<<"{\"pass\":true,\"freeze\":"<<selected<<",\"checks\":"<<checks<<",\"gpu_commands\":0,\"workspace_plans\":{";
      for (uint32_t i=0;i<plans.size();++i) {if (i) std::cout<<',';std::cout<<'"'<<plans[i].first<<"\":"<<plans[i].second;}
      std::cout<<"}}\n";return 0;}
    require(!qsa::parse(nullptr)&&!qsa::parse("0")&&qsa::parse("1"));
    for (const char *bad:{"","2","true","01"," 1","1 ","-1"}) {bool rejected=false;try {(void)qsa::parse(bad);}catch (const std::invalid_argument &) {rejected=true;}require(rejected);}
    for (uint32_t begin:{0u,1u,128u,2048u}) for (uint32_t rows:{0u,1u,4u,16u,128u,512u,2047u,2048u,2049u,4096u,8192u})
      for (bool verify:{false,true}) for (bool singleton:{false,true}) for (bool selected:{false,true})
        require(qsa::mainEligible(begin,rows,verify,singleton,selected)==(selected&&singleton&&!verify&&!begin&&rows==2048));
    for (uint32_t rows:{0u,128u,2047u,2048u,8192u}) for (bool selected:{false,true}) require(qsa::plannedExtraBytes(rows,selected)==(selected&&rows>=2048?478150656:0));
    require(std::string(qsa::selectionMarker(false)).empty());const std::string base(64,'a');require(qsa::numericalIdentity(base,false)==base);
    const auto derived=qsa::numericalIdentity(base,true);require(derived.size()==64&&derived!=base);
    qsa::recordForward();qsa::recordLayer();require(qsa::numericalIdentity(base,true)==derived);
    require(qsa::encodedCounters().forwards.load()==1&&qsa::encodedCounters().attentionLayers.load()==1&&qsa::encodedCounters().constructedArenas.load()==0);
    require(25165824+402653184+50331648==478150656);require(25165824%16384==0&&427819008%16384==0&&478150656%16384==0);
    std::vector<uint8_t> snapshot(25165824),candidate(snapshot);candidate.back()=1;require(std::memcmp(snapshot.data(),candidate.data(),snapshot.size())!=0);
    require(sizeof(decltype(snapshot)::value_type)==1&&snapshot.size()==25165824);
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<",\"gpu_commands\":0,\"extra_arena_bytes\":478150656,\"prepared_Q_full_byte_compare_witness\":true}\n";return 0;
  } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
