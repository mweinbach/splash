#include "worker_bridge.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_bridge.hpp"
#include <iostream>
using namespace splash::flash;
namespace hc = prefill_hc_inject_norm_sep21;
int main(int argc,char **argv) {
  try {
    const std::string mode=argc>1?argv[1]:"";
    if(mode=="--freeze0"||mode=="--freeze1") {
      const bool wanted=mode=="--freeze1";setenv(hc::kFlag,wanted?"1":"0",1);
      if(hc::requested()!=wanted)throw std::runtime_error("initial HC selector freeze");
      setenv(hc::kFlag,wanted?"0":"1",1);if(hc::requested()!=wanted)throw std::runtime_error("HC selector changed after freeze");
      std::cout<<"HC prefill selector freeze passed\n";return 0;
    }
    uint64_t checks=0;auto require=[&](bool v){if(!v)throw std::runtime_error("HC prefill CPU policy failed");++checks;};
    require(!hc::parse(nullptr)&&!hc::parse("0")&&hc::parse("1"));
    for(const char *bad:{"","2","true","01"," 1","1 ","-1"}) {
      bool rejected=false;try{(void)hc::parse(bad);}catch(const std::invalid_argument &){rejected=true;}require(rejected);
    }
    for(uint32_t rows:{0u,1u,4u,16u,32u,128u,511u,512u,513u,1024u,2048u,2049u,4096u,8192u})
      for(bool verify:{false,true})for(bool singleton:{false,true})for(bool selected:{false,true}) {
        const bool expected=selected&&singleton&&!verify&&rows>=512&&rows<=2048;
        require(hc::mainEligible(rows,verify,singleton,selected)==expected);
        for(bool ple:{false,true})for(bool terminal:{false,true})
          require(hc::nextNormEligible(expected,ple,terminal)==(expected&&!ple&&!terminal));
      }
    for(uint32_t rows:{512u,1024u,2048u})for(bool f32:{false,true})for(auto c:{NormConvention::OnePlusWeight,NormConvention::DirectGamma}) {
      const FlashTensor w{{},f32?FlashDType::F32:FlashDType::BF16,{10240},uint64_t(10240)*(f32?4:2)};
      const auto p=hc::checkedParams({rows,2560,4,1e-6f},w,c);require(p.rows==rows&&p.norm_is_float==f32);
    }
    const std::string cache=std::string(dense_w8a8_sep21::kCacheMarker)+std::string(64,'a');
    for(bool denseSelected:{false,true}) {
      const std::string routes=std::string(dense_w8a8_sep21::selectionMarker(denseSelected))+cache;
      const auto base=dense_w8a8_sep21::numericalIdentity(std::string(64,'b'),routes);
      for(bool hcSelected:{false,true})require(base==dense_w8a8_sep21::numericalIdentity(std::string(64,'b'),routes+hc::selectionMarker(hcSelected)));
    }
    require(hc::encodedCounters().forwards.load()==0&&hc::encodedCounters().attentionToMlp.load()==0&&
        hc::encodedCounters().mlpToNext.load()==0&&hc::encodedCounters().pleExcluded.load()==0&&hc::encodedCounters().terminalExcluded.load()==0);
    hc::recordForward();hc::recordAttentionToMlp();hc::recordMlpToNext();hc::recordPLEExcluded();hc::recordTerminalExcluded();
    require(hc::encodedCounters().forwards.load()==1&&hc::encodedCounters().attentionToMlp.load()==1&&
        hc::encodedCounters().mlpToNext.load()==1&&hc::encodedCounters().pleExcluded.load()==1&&hc::encodedCounters().terminalExcluded.load()==1);
    std::cout<<"{\"prefill_hc_worker_cpu_policy\":\"passed\",\"checks\":"<<checks
        <<",\"added_workspace_bytes\":0,\"gpu_work\":false,\"payload_bytes_read\":0}\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
