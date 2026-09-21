// Private real-feature four-proposal proof. CPU self-test creates no device.
// Every host object must use the private indirect-dispatch header overlay.
#include "FlashMTPGPUChainFour.hpp"
#include "FlashMTPGPUChainFourPolicy.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashForward.hpp"
#include "metal/abi/FlashForward.h"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace chain=splash::flash::mtp_gpu_chain_four_candidate;
namespace policy=splash::flash::mtp_gpu_chain_four_policy;
namespace metal=splash::metal;
constexpr uint32_t kHyper=10240,kVocabulary=248320,kCapacity=4096;
void require(bool value,std::string_view message) {
  if(!value) throw std::runtime_error(std::string(message));
}
float number(uint16_t value) {return std::bit_cast<float>(uint32_t{value}<<16);}
std::vector<uint16_t> words(const metal::MetalBuffer &buffer,uint64_t count) {
  require(buffer&&buffer.contents()&&buffer.storage()==metal::BufferStorage::Shared&&
          buffer.sizeBytes()>=count*2,"invalid borrowed BF16 storage");
  const auto *begin=static_cast<const uint16_t *>(buffer.contents());
  std::vector<uint16_t> result(begin,begin+count);
  for(auto word:result) require(std::isfinite(number(word)),"nonfinite head output");
  return result;
}
FlashGreedyGPURowResult record(const metal::MetalBuffer &buffer) {
  require(buffer&&buffer.contents()&&buffer.sizeBytes()>=sizeof(FlashGreedyGPURowResult),
          "invalid exact greedy record storage");
  FlashGreedyGPURowResult result;
  std::memcpy(&result,buffer.contents(),sizeof(result));
  return result;
}
bool sameRecord(const FlashGreedyGPURowResult &a,const FlashGreedyGPURowResult &b) {
  return std::memcmp(&a,&b,sizeof(a))==0;
}
void exact(std::span<const uint16_t> actual,std::span<const uint16_t> expected,
           std::string_view message) {
  require(actual.size()==expected.size()&&std::equal(actual.begin(),actual.end(),expected.begin()),message);
}
bool sentinel(const metal::MetalBuffer &buffer,uint64_t begin=0) {
  require(buffer.contents()&&begin<=buffer.sizeBytes(),"invalid sentinel extent");
  const auto *bytes=static_cast<const uint8_t *>(buffer.contents());
  return std::all_of(bytes+begin,bytes+buffer.sizeBytes(),[](uint8_t byte){return byte==0xa7;});
}
using RawBuffers=std::vector<std::vector<uint8_t>>;
RawBuffers rawBuffers(metal::MetalBackend &backend,
                      const std::vector<metal::MetalBuffer> &buffers) {
  RawBuffers result;
  for(const auto &source:buffers) {
    require(source&&source.sizeBytes(),"empty qualification buffer");
    auto readable=source;
    if(!source.contents()) {
      require(source.sizeBytes()%4==0,"private inspection requires complete U32 words");
      readable=backend.allocateBuffer(source.sizeBytes(),metal::BufferStorage::Shared,
          "private-oracle-inspection-copy");
      const FlashForwardCopyParams params{source.sizeBytes()/4};
      metal::CommandGraph inspection;
      inspection.add("flash_forward_copy_words",{source,readable},params,
          {(params.words+255)/256,1,1},{256,1,1});
      (void)backend.submitCommand(inspection.dispatches());
    }
    require(readable.contents(),"qualification buffer is not readable after inspection");
    result.emplace_back(source.sizeBytes());
    std::memcpy(result.back().data(),readable.contents(),source.sizeBytes());
  }
  return result;
}
uint64_t exactRaw(const RawBuffers &actual,const RawBuffers &expected,
                  std::string_view message) {
  require(actual.size()==expected.size(),message);
  uint64_t bytes=0;
  for(size_t index=0;index<actual.size();++index) {
    require(actual[index].size()==expected[index].size()&&
            std::memcmp(actual[index].data(),expected[index].data(),actual[index].size())==0,message);
    bytes+=actual[index].size();
  }
  return bytes;
}
void timing(const metal::CommandTiming &value,bool body) {
  require(std::isfinite(value.gpuSeconds)&&value.gpuSeconds>=0&&
          std::isfinite(value.wallSeconds)&&value.wallSeconds>1e-9,
          "invalid command timing; every host object must match the private header ABI");
  if(body) require(value.gpuSeconds>1e-9,"active head body returned zero GPU duration");
}
struct Snapshot {
  std::vector<uint16_t> hidden,logits;
  FlashGreedyGPURowResult greedy{};
  metal::CommandTiming timing;
};
Snapshot snapshot(const chain::FlashMTPResult &result) {
  require(result.hiddenRows&&result.logitRows==1&&result.greedyRows==1,"scalar seed/result metadata differs");
  Snapshot out{words(result.hiddenBF16,uint64_t{result.hiddenRows}*kHyper),
               words(result.logitsBF16,kVocabulary),record(result.greedyResultsU32),result.timing};
  (void)splash::flash::greedyGPUResultToken(out.greedy,kVocabulary);
  timing(out.timing,true);
  return out;
}
std::vector<uint32_t> prompt(const char *path,uint32_t expectedCount) {
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  require(data&&data.length<=(1ULL<<20),"cannot read bounded prompt token JSON");
  NSError *error=nil;
  id decoded=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error&&[decoded isKindOfClass:[NSArray class]],"prompt JSON must be an array");
  NSArray *array=decoded;require(array.count==expectedCount,"prompt JSON has wrong exact token count");
  std::vector<uint32_t> out;
  for(id item in array) {
    require([item isKindOfClass:[NSNumber class]]&&
            CFGetTypeID((__bridge CFTypeRef)item)!=CFBooleanGetTypeID(),"prompt token must be an integer");
    NSNumber *number=item;
    require(std::string_view("cCsSiIlLqQ").find(number.objCType[0])!=std::string_view::npos&&
            number.longLongValue>=0&&number.unsignedLongLongValue<kVocabulary,"invalid integer prompt token");
    out.push_back(uint32_t(number.unsignedLongLongValue));
  }
  return out;
}
struct Features {std::vector<uint16_t> hidden;uint32_t anchor=0;double gpu=0,wall=0;};
Features features(splash::flash::FlashForward &target,std::span<const uint32_t> tokens) {
  auto state=target.createState();Features out;out.hidden.reserve(uint64_t{tokens.size()}*kHyper);
  for(uint32_t begin=0;begin<tokens.size();) {
    const uint32_t count=uint32_t(std::min<size_t>(128,tokens.size()-begin));
    const auto result=target.forward(state,tokens.subspan(begin,count),false,true);
    const auto captured=words(result.hiddenBF16,uint64_t{count}*kHyper);
    out.hidden.insert(out.hidden.end(),captured.begin(),captured.end()); // Before target arena reuse.
    require(result.greedyRows==1&&result.logitRows==1,"target exact greedy anchor unavailable");
    out.anchor=splash::flash::greedyGPUResultToken(record(result.greedyResultsU32),kVocabulary);
    out.gpu+=result.timing.gpuSeconds;out.wall+=result.timing.wallSeconds;begin+=count;
  }
  require(state.logicalLength()==tokens.size()&&!state.poisoned(),"target feature capture offset/health differs");
  require(!policy::greedy::stopToken(out.anchor),"target anchor is EOS; prompt has no live MTP draft cycle");
  return out;
}
struct Stepwise {
  policy::Control control;
  std::array<Snapshot,3> outputs;
  double gpu=0,wall=0,host=0;
};
Stepwise stepwise(chain::FlashMTPForward &head,chain::FlashMTPState &state,
                  const chain::FlashMTPResult &seed,const FlashGreedyGPURowResult &first) {
  Stepwise out;auto previous=seed.hiddenBF16;const uint32_t initial=uint32_t(state.logicalLength());
  for(uint32_t body=0;body<3;++body) {
    const policy::Params params{248320,4,5,initial,kCapacity,97,body,0};
    (void)policy::beginStep(out.control,params,first);
    if(out.control.bodyEnabled) {
      const std::array<uint32_t,1> next{out.control.proposals[body]};
      const auto start=std::chrono::steady_clock::now();
      const auto result=head.forward(state,previous,next);
      out.host+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      out.outputs[body]=snapshot(result);
      out.gpu+=result.timing.gpuSeconds;out.wall+=result.timing.wallSeconds;
      previous=result.hiddenBF16;
      (void)policy::finishStep(out.control,params,out.outputs[body].greedy);
    } else (void)policy::finishStep(out.control,params,FlashGreedyGPURowResult{UINT32_MAX,0,1,1},UINT32_MAX);
  }
  require(!out.control.errors&&state.logicalLength()==initial+out.control.consumedPairs,
          "stepwise scalar proposal count/length differs");
  return out;
}
void controlExact(const FlashMTPGPUChainFourControl &actual,const policy::Control &expected) {
  require(actual.proposal_count==expected.proposalCount&&actual.consumed_pairs==expected.consumedPairs&&
          actual.body_enabled==expected.bodyEnabled&&actual.finished_eos==expected.finishedEOS&&
          actual.errors==expected.errors&&!actual.reserved0&&!actual.reserved1&&!actual.reserved2,
          "four-chain control differs from scalar CPU reference");
  for(uint32_t i=0;i<4;++i)require(actual.proposals[i]==expected.proposals[i],"four-chain proposal prefix/suffix differs");
}
void dimensions(const chain::FourProposalResult &result,
                const std::vector<metal::MetalBuffer> &buffers,uint32_t executed) {
  require(buffers.size()==17,"four-chain mutable buffer inventory differs");
  for(uint32_t body=0;body<3;++body) {
    const uint32_t offset=2+body*5,count=result.indirectDispatches[body];
    require(count&&count<=256,"four-chain body dispatch count invalid");
    const auto *original=static_cast<const uint32_t *>(buffers[offset].contents());
    const auto *indirect=static_cast<const uint32_t *>(buffers[offset+1].contents());
    for(uint32_t index=0;index<count*3;++index)
      require(indirect[index]==(body<executed?original[index]:0u),"four-chain active/zero indirect dimension differs");
    require(sentinel(buffers[offset+1],uint64_t{count}*12),"four-chain guard wrote unused dimensions");
    if(body>=executed)
      require(sentinel(buffers[offset+2])&&sentinel(buffers[offset+3])&&sentinel(buffers[offset+4]),
              "four-chain skipped I64/argmax/hidden suffix changed");
  }
}
void compareNatural(metal::MetalBackend &backend,chain::FlashMTPForward &baseline,
    const chain::FlashMTPState &reference,chain::FlashMTPForward &chained,
    const chain::FlashMTPState &actual,const Stepwise &expected,const chain::FourProposalResult &result,
    const std::vector<metal::MetalBuffer> &buffers) {
  controlExact(result.control,expected.control);
  require(result.originalLength+expected.control.consumedPairs==result.logicalLength&&
          actual.logicalLength()==reference.logicalLength()&&!actual.poisoned(),"four-chain logical publication differs");
  dimensions(result,buffers,expected.control.consumedPairs);
  for(uint32_t body=0;body<expected.control.consumedPairs;++body) {
    require(sameRecord(record(result.greedyRecords[body]),expected.outputs[body].greedy),
            "four-chain exact per-step greedy record differs");
    exact(words(result.hiddenSnapshots[body],kHyper),expected.outputs[body].hidden,
          "four-chain full per-step BF16 hidden differs");
    require(sentinel(result.hiddenSnapshots[body],uint64_t{kHyper}*2),"four-chain hidden allocation padding changed");
  }
  (void)exactRaw(rawBuffers(backend,chained.stateBuffers(actual)),
                rawBuffers(backend,baseline.stateBuffers(reference)),"four-chain full QSA state differs");
}
double median(std::vector<double> values) {
  require(!values.empty(),"empty paired timing samples");std::sort(values.begin(),values.end());
  return (values[values.size()/2-1]+values[values.size()/2])*.5;
}
struct Samples {
  std::vector<double> gpu,wall,host;
  void add(double a,double b,double c){gpu.push_back(a);wall.push_back(b);host.push_back(c);}
  void write(std::ostream &out)const {
    out<<"{\"samples\":"<<gpu.size()<<",\"median_gpu_seconds\":"<<median(gpu)
       <<",\"median_command_wall_seconds\":"<<median(wall)<<",\"median_api_host_seconds\":"<<median(host)
       <<",\"gpu_seconds\":[";
    for(size_t i=0;i<gpu.size();++i){if(i)out<<',';out<<gpu[i];}out<<"]}";
  }
};
struct GuardCase {const char *name;uint32_t depth,budget,diagnostics;FlashGreedyGPURowResult seed;};
void cpuSelfTest() {
  uint64_t checks=0;
  for(uint32_t budget=1;budget<=17;++budget)for(uint32_t depth=0;depth<=4;++depth) {
    policy::Control control;
    for(uint32_t body=0;body<3;++body) {
      const policy::Params params{248320,depth,budget,3,kCapacity,97,body,0};
      (void)policy::beginStep(control,params,{17,0xbf80,0,0});
      (void)policy::finishStep(control,params,{18+body,0xbf80,0,0});
      const auto pending=policy::publish(control,3,false);
      require(!pending.ready&&pending.logicalLength==3,"CPU early publication");++checks;
    }
    const uint32_t proposals=std::min(depth,budget-1);
    require(control.proposalCount==proposals&&control.consumedPairs==(proposals?proposals-1:0),
            "CPU four-proposal quota/count contract");++checks;
  }
  std::cout<<"{\"pass\":true,\"cpu_checks\":"<<checks<<",\"gpu_commands\":0,\"model_loaded\":false}\n";
}
void write(const char *path,const std::string &text) {
  std::ofstream file(path);require(bool(file),"cannot open four-oracle report");
  file<<text<<'\n';require(bool(file),"cannot write four-oracle report");
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    std::vector<std::string> contexts;
    try {
      if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){cpuSelfTest();return 0;}
      require(argc==6,"usage: flash-mtp-gpu-chain-four-oracle METALLIB PACKAGE PROMPT128_JSON PROMPT2048_JSON REPORT");
      const char *flag=std::getenv("SPLASH_FLASH_GPU_GREEDY");
      require(flag&&std::string_view(flag)=="1","Root must set SPLASH_FLASH_GPU_GREEDY=1; oracle never changes flags");
      const std::array<std::vector<uint32_t>,2> prompts{prompt(argv[3],128),prompt(argv[4],2048)};
      metal::MetalBackend backend(argv[1]);
      const auto weights=splash::flash::FlashWeights::load(backend,argv[2]);
      splash::flash::FlashForward target(backend,weights,kCapacity,128);
      chain::FlashMTPForward baseline(backend,weights,kCapacity,128),chained(backend,weights,kCapacity,128);
      require(std::string_view(baseline.attentionRouteSemantics())==chained.attentionRouteSemantics()&&
              std::string_view(baseline.projectionRouteSemantics())==chained.projectionRouteSemantics(),
              "four-oracle twin math profiles differ");
      chain::FourProposalWorkspace workspace(chained);const auto buffers=workspace.mutableBuffers();
      const auto ownedHidden=backend.allocateBuffer(uint64_t{kHyper}*2,metal::BufferStorage::Shared,"private-four-owned-seed-hidden");
      const auto ownedGreedy=backend.allocateBuffer(16,metal::BufferStorage::Shared,"private-four-owned-seed-record");
      const auto prior=backend.allocateBuffer(4,metal::BufferStorage::Shared,"private-four-prior-diagnostics");
      uint64_t executed=0;bool fullWidth=false;
      for(const auto &tokens:prompts) {
        const auto trueFeatures=features(target,tokens);
        const uint32_t n=uint32_t(tokens.size());
        const auto targetHidden=backend.allocateBuffer(uint64_t{n}*kHyper*2,metal::BufferStorage::Shared,
            "private-four-real-target-hidden");
        std::memcpy(targetHidden.contents(),trueFeatures.hidden.data(),uint64_t{n}*kHyper*2);
        auto reference=baseline.createState();auto actual=chained.createState();
        for(uint32_t begin=0;begin<n-1;) {
          const uint32_t count=std::min(uint32_t{128},n-1-begin);
          const auto hidden=backend.view(targetHidden,uint64_t{begin}*kHyper*2,uint64_t{count}*kHyper*2);
          const auto pairs=std::span<const uint32_t>(tokens).subspan(begin+1,count);
          const auto a=baseline.forward(reference,hidden,pairs,chain::FlashMTPLogits::None);
          const auto copied=words(a.hiddenBF16,uint64_t{count}*kHyper);
          const auto b=chained.forward(actual,hidden,pairs,chain::FlashMTPLogits::None);
          exact(words(b.hiddenBF16,uint64_t{count}*kHyper),copied,"real-target-pair head priming differs");
          begin+=count;
        }
        const auto last=backend.view(targetHidden,uint64_t{n-1}*kHyper*2,uint64_t{kHyper}*2);
        const std::array<uint32_t,1> anchor{trueFeatures.anchor};
        const auto originalSeed=baseline.forward(reference,last,anchor);
        const auto seedSnapshot=snapshot(originalSeed);
        auto seed=chained.forward(actual,last,anchor);const auto twinSeed=snapshot(seed);
        exact(twinSeed.hidden,seedSnapshot.hidden,"real-feature seed hidden differs");
        exact(twinSeed.logits,seedSnapshot.logits,"real-feature seed vocabulary differs");
        require(sameRecord(twinSeed.greedy,seedSnapshot.greedy)&&reference.logicalLength()==n&&
                actual.logicalLength()==n,"real-feature seed record/offset differs");
        std::memcpy(ownedHidden.contents(),twinSeed.hidden.data(),uint64_t{kHyper}*2);
        std::memcpy(ownedGreedy.contents(),&twinSeed.greedy,16);std::memset(prior.contents(),0,4);
        seed.hiddenBF16=ownedHidden;seed.hiddenRows=1;seed.greedyResultsU32=ownedGreedy;
        seed.logitRows=seed.greedyRows=1;seed.logicalLength=n;
        const auto rewind=[&]{baseline.truncate(reference,n);chained.truncate(actual,n);};
        const auto runChain=[&] {
          const auto start=std::chrono::steady_clock::now();
          const auto result=workspace.run(actual,seed,4,5,prior);
          const double host=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
          timing(result.timing,bool(result.control.consumed_pairs));return std::pair{result,host};
        };
        rewind();
        const auto warm=stepwise(baseline,reference,seed,twinSeed.greedy);
        const auto warmChain=runChain();
        compareNatural(backend,baseline,reference,chained,actual,warm,warmChain.first,buffers);
        executed+=warm.control.consumedPairs;fullWidth|=warm.control.proposalCount==4;
        Samples baselineTimes,chainTimes;
        for(uint32_t repetition=0;repetition<6;++repetition) {
          rewind();Stepwise expected;chain::FourProposalResult result;double host=0;
          if(repetition%2){auto paired=runChain();result=paired.first;host=paired.second;
            expected=stepwise(baseline,reference,seed,twinSeed.greedy);}
          else {expected=stepwise(baseline,reference,seed,twinSeed.greedy);
            auto paired=runChain();result=paired.first;host=paired.second;}
          compareNatural(backend,baseline,reference,chained,actual,expected,result,buffers);
          require(expected.control.proposalCount==warm.control.proposalCount&&
                  expected.control.consumedPairs==warm.control.consumedPairs,"same prefix produced a different natural EOS/count");
          for(uint32_t body=0;body<expected.control.consumedPairs;++body) {
            exact(expected.outputs[body].hidden,warm.outputs[body].hidden,"stepwise repeated hidden differs");
            require(sameRecord(expected.outputs[body].greedy,warm.outputs[body].greedy),"stepwise repeated greedy differs");
          }
          baselineTimes.add(expected.gpu,expected.wall,expected.host);
          chainTimes.add(result.timing.gpuSeconds,result.timing.wallSeconds,host);
        }
        // Roll back all speculative pairs and append a different real pair,
        // with the same captured target-derived seed carry for both routes.
        rewind();const std::array<uint32_t,1> changed{tokens.back()};
        const auto resumedA=snapshot(baseline.forward(reference,ownedHidden,changed));
        const auto resumedB=snapshot(chained.forward(actual,ownedHidden,changed));
        exact(resumedB.hidden,resumedA.hidden,"four-chain rollback continuation hidden differs");
        exact(resumedB.logits,resumedA.logits,"four-chain rollback continuation full vocabulary differs");
        require(sameRecord(resumedB.greedy,resumedA.greedy),"four-chain rollback continuation greedy differs");
        std::vector<std::string> guards;
        const auto natural=twinSeed.greedy;
        const std::array<GuardCase,10> cases{{
          {"depth0",0,1,4,{17,0xbf80,1,0}},{"quota1",4,1,0,{248320,0xbf80,0,0}},
          {"depth1",1,5,0,natural},{"quota2",4,2,0,natural},
          {"firstEOS248044",4,5,0,{248044,0xbf80,0,0}},{"firstEOS248046",4,5,0,{248046,0xbf80,0,0}},
          {"badToken",4,5,0,{248320,0xbf80,0,0}},{"badRank",4,5,0,{17,0x7fff,0,0}},
          {"nonfiniteSeed",4,5,0,{17,0xbf80,1,0}},{"priorDiagnostics",4,5,4,natural}
        }};
        for(const auto &test:cases) {
          rewind();std::memcpy(ownedGreedy.contents(),&test.seed,16);std::memcpy(prior.contents(),&test.diagnostics,4);
          const auto beforeState=rawBuffers(backend,chained.stateBuffers(actual));
          const auto beforeScratch=rawBuffers(backend,chained.scratchBuffers());
          const auto result=workspace.run(actual,seed,test.depth,test.budget,prior);
          policy::Control expected;
          for(uint32_t body=0;body<3;++body) {
            const policy::Params p{248320,test.depth,test.budget,n,kCapacity,result.indirectDispatches[body],body,0};
            (void)policy::beginStep(expected,p,test.seed,test.diagnostics);
            require(!expected.bodyEnabled,"guard fixture unexpectedly requires a numerical body");
            (void)policy::finishStep(expected,p,{UINT32_MAX,0,1,1},UINT32_MAX);
          }
          controlExact(result.control,expected);dimensions(result,buffers,0);
          require(actual.logicalLength()==n&&!actual.poisoned(),"skipped four-chain guard mutated logical state/health");
          (void)exactRaw(rawBuffers(backend,chained.stateBuffers(actual)),beforeState,"skipped four-chain guard wrote QSA state");
          (void)exactRaw(rawBuffers(backend,chained.scratchBuffers()),beforeScratch,"skipped four-chain guard wrote head scratch");
          std::ostringstream entry;entry<<"{\"name\":"<<splash::json::quote(test.name)
            <<",\"proposal_count\":"<<result.control.proposal_count<<",\"errors\":"<<result.control.errors
            <<",\"all_body_dimensions_zero\":true,\"all_suffixes_untouched\":true,\"qsa_scratch_unchanged\":true,\"pass\":true}";
          guards.push_back(entry.str());
        }
        std::memcpy(ownedGreedy.contents(),&natural,16);std::memset(prior.contents(),0,4);
        std::ostringstream entry;entry<<std::setprecision(12)<<"{\"context_tokens\":"<<n
          <<",\"target_anchor\":"<<trueFeatures.anchor<<",\"real_target_hidden_rows\":"<<n
          <<",\"feature_generation_gpu_seconds\":"<<trueFeatures.gpu<<",\"feature_generation_wall_seconds\":"<<trueFeatures.wall
          <<",\"natural_proposal_count\":"<<warm.control.proposalCount<<",\"natural_consumed_pairs\":"<<warm.control.consumedPairs
          <<",\"observed_natural_eos_position\":"<<(warm.control.finishedEOS?std::to_string(warm.control.proposalCount):"null")
          <<",\"baseline\":";baselineTimes.write(entry);entry<<",\"one_command_chain\":";chainTimes.write(entry);
        entry<<",\"gpu_speedup\":";
        if(median(chainTimes.gpu)>1e-9&&warm.control.consumedPairs)entry<<median(baselineTimes.gpu)/median(chainTimes.gpu);
        else entry<<"null";
        entry<<",\"command_wall_speedup\":"<<median(baselineTimes.wall)/median(chainTimes.wall)
             <<",\"api_host_speedup\":"<<median(baselineTimes.host)/median(chainTimes.host)
             <<",\"all_active_hidden_greedy_qsa_exact\":true,\"rollback_continuation_full_logits_exact\":true"
             <<",\"forced_eos_positions_2_3_4_gpu_tested\":false,\"later_eos_policy_cpu_covered\":true,\"guards\":[";
        for(size_t i=0;i<guards.size();++i){if(i)entry<<',';entry<<guards[i];}entry<<"],\"pass\":true}";
        contexts.push_back(entry.str());std::cerr<<"real-feature four-chain context"<<n<<" exact PASS\n";
      }
      require(executed&&fullWidth,"natural fixtures did not exercise the full four-proposal chain");
      std::ostringstream report;report<<"{\"pass\":true,\"private_prototype\":true,\"http_qualified\":false"
        <<",\"default_policy_qualified\":false,\"input_features\":\"unchanged-target-prefill-captured-bf16\""
        <<",\"target_kernel_routes\":"<<splash::json::quote(target.kernelRoutes())
        <<",\"head_projection_route\":"<<splash::json::quote(chained.projectionRouteSemantics())
        <<",\"source_identity\":"<<splash::json::quote(weights.sourceIdentity())
        <<",\"paired_samples_per_context\":6,\"order\":\"AB-BA-alternating\""
        <<",\"timing_scope\":\"head-only; feature generation/model construction/snapshots/inspection copies excluded; baseline sums forward API and command spans, chain spans workspace.run\""
        <<",\"full_four_proposal_width_exercised\":true,\"production_modified\":false,\"contexts\":[";
      for(size_t i=0;i<contexts.size();++i){if(i)report<<',';report<<contexts[i];}report<<"]}";
      write(argv[5],report.str());return 0;
    }catch(const std::exception &error){
      std::cerr<<"private real-feature four-chain oracle: "<<error.what()<<'\n';
      if(argc==6){std::ostringstream report;report<<"{\"pass\":false,\"private_prototype\":true,\"http_qualified\":false,\"error\":"
        <<splash::json::quote(error.what())<<",\"completed_contexts\":[";
        for(size_t i=0;i<contexts.size();++i){if(i)report<<',';report<<contexts[i];}report<<"]}";
        try{write(argv[5],report.str());}catch(...){}}
      return 1;
    }
  }
}
