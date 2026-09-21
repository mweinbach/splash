// Private two-proposal proof. --cpu-self-test loads no model/device/fixture.
// Every host object must use the private indirect-dispatch header overlay.
#include "FlashMTPGPUChainTwo.hpp"
#include "FlashMTPGPUChainPolicy.hpp"
#include "flash/FlashGreedyGPU.hpp"
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
namespace chain=splash::flash::mtp_gpu_chain_candidate;
namespace policy=splash::flash::mtp_gpu_chain_policy;
namespace metal=splash::metal;
constexpr uint32_t kHyper=10240,kVocabulary=248320,kCapacity=4096;
void require(bool value,std::string_view message) {
  if(!value) throw std::runtime_error(std::string(message));
}
uint32_t primeSetting() {
  const char *raw=std::getenv("FLASH_MTP_GPU_CHAIN_TWO_PRIME_ROWS");
  if(!raw) return 128;
  const std::string_view text(raw);
  require(!text.empty()&&text.find_first_not_of("0123456789")==std::string_view::npos&&
          !(text.size()>1&&text.front()=='0'),"prime rows require canonical decimal1..2048");
  uint64_t value=0;
  for(char digit:text) {
    require(value<2048,"prime rows exceed2048");
    value=value*10+uint32_t(digit-'0');
  }
  require(value>=1&&value<=2048,"prime rows must be1..2048");
  return uint32_t(value);
}
uint16_t bf16(float value) {
  const uint32_t bits=std::bit_cast<uint32_t>(value);
  return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
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
std::vector<uint16_t> fixture() {
  const char *path=std::getenv("FLASH_MTP_GPU_CHAIN_TWO_HIDDEN_BF16");
  if(!path) return {};
  require(*path,"captured BF16 fixture path cannot be empty");
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  require(bool(file),"cannot open captured BF16 fixture");
  const auto bytes=file.tellg();
  require(bytes>0&&uint64_t(bytes)%(uint64_t{kHyper}*2)==0&&uint64_t(bytes)<=(64ULL<<20),
          "fixture requires complete BF16[rows,10240], bounded64MiB");
  std::vector<uint16_t> out(uint64_t(bytes)/2);
  file.seekg(0);file.read(reinterpret_cast<char *>(out.data()),std::streamsize(bytes));
  require(bool(file),"captured BF16 fixture is short");
  for(auto word:out) require(std::isfinite(number(word)),"captured BF16 fixture is nonfinite");
  return out;
}
std::vector<uint32_t> fill(const metal::MetalBuffer &buffer,uint32_t begin,uint32_t rows,
                          std::span<const uint16_t> captured) {
  require(buffer.contents()&&buffer.sizeBytes()>=uint64_t{rows}*kHyper*2,"input fixture extent");
  auto *data=static_cast<uint16_t *>(buffer.contents());
  std::vector<uint32_t> tokens;
  for(uint32_t row=0;row<rows;++row) {
    const uint32_t position=begin+row;
    tokens.push_back((position*31+17)%240000);
    for(uint32_t column=0;column<kHyper;++column)
      data[uint64_t{row}*kHyper+column]=captured.empty()?
          bf16(std::sin(float(column%997)*.041f+float(position)*.173f)*(.2f+float(column/2560)*.12f)):
          captured[uint64_t{position%(captured.size()/kHyper)}*kHyper+column];
  }
  return tokens;
}
enum class Injection { Natural,EOS044,EOS046,BadToken,BadRank,Nonfinite };
struct Case {const char *name;uint32_t modulo,depth,remaining,prior;Injection injection;};
const std::array<Case,14> kCases{{
  {"natural-mod0",0,2,3,0,Injection::Natural},{"natural-mod1",1,2,3,0,Injection::Natural},
  {"natural-mod2",2,2,3,0,Injection::Natural},{"natural-mod3",3,2,3,0,Injection::Natural},
  {"depth0-ignores-seed-and-diag",0,0,1,4,Injection::Nonfinite},
  {"budget1-ignores-seed",1,2,1,0,Injection::BadToken},
  {"requested-depth1",2,1,3,0,Injection::Natural},
  {"budget2-clamps-second",3,2,2,0,Injection::Natural},
  {"first-eos248044",0,2,3,0,Injection::EOS044},
  {"first-eos248046",1,2,3,0,Injection::EOS046},
  {"invalid-first-token",2,2,3,0,Injection::BadToken},
  {"invalid-first-rank",3,2,3,0,Injection::BadRank},
  {"nonfinite-first-record",0,2,3,0,Injection::Nonfinite},
  {"prior-diagnostics",1,2,3,4,Injection::Natural},
}};
void compareControl(const FlashMTPGPUChainControl &actual,const policy::Control &expected) {
  require(actual.proposal_count==expected.proposalCount&&actual.consumed_pairs==expected.consumedPairs&&
          actual.body_enabled==expected.bodyEnabled&&actual.finished_eos==expected.finishedEOS&&
          actual.errors==expected.errors&&!actual.reserved0&&!actual.reserved1&&!actual.reserved2&&
          actual.proposals[0]==expected.proposals[0]&&actual.proposals[1]==expected.proposals[1],
          "GPU guard control disagrees with pure CPU reference");
}
void cpuSelfTest() {
  uint64_t checks=0;
  for(uint32_t budget=1;budget<=17;++budget) for(uint32_t depth=0;depth<=2;++depth) {
    policy::Params params{248320,depth,budget,3,4096,61,{0,0}};
    const auto prepared=policy::prepare(params,FlashGreedyGPURowResult{17,0xbf80,0,0});
    require(bool(prepared.control.bodyEnabled)==(std::min(depth,budget-1)==2),"CPU depth/bonus contract");++checks;
    for(uint32_t eos:{248044u,248046u}) {
      const auto stopped=policy::prepare(params,FlashGreedyGPURowResult{eos,0xbf80,0,0});
      require(!stopped.control.bodyEnabled&&policy::indirectGroups(stopped,{17,3,1})==policy::Groups{0,0,0},
              "CPU EOS/no-work indirect contract");++checks;
    }
  }
  static_assert(sizeof(FlashMTPGPUChainGuardParams)==32&&sizeof(FlashMTPGPUChainControl)==40);
  std::cout<<"{\"pass\":true,\"cpu_checks\":"<<checks<<",\"gpu_commands\":0,\"model_loaded\":false}\n";
}
void write(const char *path,const std::string &text) {
  std::ofstream file(path);require(bool(file),"cannot open private oracle report");
  file<<text<<'\n';require(bool(file),"cannot write private oracle report");
}
}

int main(int argc,char **argv) {
  @autoreleasepool {
    std::vector<std::string> records;
    try {
      if(argc==2&&std::string_view(argv[1])=="--cpu-self-test") {cpuSelfTest();return 0;}
      require(argc==4,"usage: flash-mtp-gpu-chain-two-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const char *greedyFlag=std::getenv("SPLASH_FLASH_GPU_GREEDY");
      require(greedyFlag&&std::string_view(greedyFlag)=="1",
              "Root must explicitly set SPLASH_FLASH_GPU_GREEDY=1; oracle never changes flags");
      const uint32_t primeRows=primeSetting();
      const auto captured=fixture();
      metal::MetalBackend backend(argv[1]);
      const auto weights=splash::flash::FlashWeights::load(backend,argv[2]);
      chain::FlashMTPForward baseline(backend,weights,kCapacity,128);
      chain::FlashMTPForward chained(backend,weights,kCapacity,128);
      require(std::string_view(baseline.attentionRouteSemantics())==chained.attentionRouteSemantics()&&
              std::string_view(baseline.projectionRouteSemantics())==chained.projectionRouteSemantics(),
              "twin heads use different math profiles");
      chain::TwoProposalWorkspace workspace(chained);
      const auto buffers=workspace.mutableBuffers();require(buffers.size()==7,"workspace buffer inventory differs");
      const auto input=backend.allocateBuffer(uint64_t{128}*kHyper*2,metal::BufferStorage::Shared,"private-chain-fixed-input");
      const auto seedHidden=backend.allocateBuffer(uint64_t{kHyper}*2,metal::BufferStorage::Shared,"private-chain-owned-seed-hidden");
      const auto seedRecord=backend.allocateBuffer(sizeof(FlashGreedyGPURowResult),metal::BufferStorage::Shared,"private-chain-owned-seed-record");
      const auto prior=backend.allocateBuffer(4,metal::BufferStorage::Shared,"private-chain-prior-diagnostics");
      uint64_t exactWords=0,naturalBodies=0,skippedCases=0,indirectTriplets=0;
      uint64_t exactQsaBytes=0,untouchedScratchBytes=0;
      for(const auto &test:kCases) {
        auto reference=baseline.createState();auto actual=chained.createState();
        const uint32_t prefix=primeRows+(test.modulo+4-(primeRows+1)%4)%4;
        for(uint32_t begin=0;begin<prefix;) {
          const uint32_t rows=std::min(uint32_t{128},prefix-begin);
          const auto tokens=fill(input,begin,rows,captured);
          const auto view=backend.view(input,0,uint64_t{rows}*kHyper*2);
          const auto a=baseline.forward(reference,view,tokens,chain::FlashMTPLogits::None);
          const auto hiddenA=words(a.hiddenBF16,uint64_t{rows}*kHyper);
          const auto b=chained.forward(actual,view,tokens,chain::FlashMTPLogits::None);
          exact(words(b.hiddenBF16,uint64_t{rows}*kHyper),hiddenA,"twin priming hidden differs");
          begin+=rows;
        }
        const auto token=fill(input,prefix,1,captured);
        const auto one=backend.view(input,0,uint64_t{kHyper}*2);
        const auto a=baseline.forward(reference,one,token);
        const auto first=snapshot(a); // Snapshot before reusing this owner.
        auto seed=chained.forward(actual,one,token);
        const auto secondSeed=snapshot(seed);
        exact(secondSeed.hidden,first.hidden,"seed BF16 hidden differs");
        exact(secondSeed.logits,first.logits,"seed BF16 vocabulary differs");
        require(sameRecord(secondSeed.greedy,first.greedy),"seed exact greedy record differs");
        const uint32_t begin=uint32_t(actual.logicalLength());
        require(reference.logicalLength()==begin&&begin%4==test.modulo,"seed begin/modulo differs");
        std::memcpy(seedHidden.contents(),secondSeed.hidden.data(),uint64_t{kHyper}*2);
        auto chosen=secondSeed.greedy;
        if(test.injection==Injection::EOS044)chosen={248044,0xbf80,0,0};
        if(test.injection==Injection::EOS046)chosen={248046,0xbf80,0,0};
        if(test.injection==Injection::BadToken)chosen={248320,0xbf80,0,0};
        if(test.injection==Injection::BadRank)chosen={17,0x7fff,0,0};
        if(test.injection==Injection::Nonfinite)chosen={17,0xbf80,kFlashGreedyGPUErrorNonfinite,0};
        std::memcpy(seedRecord.contents(),&chosen,sizeof(chosen));
        std::memcpy(prior.contents(),&test.prior,4);
        seed.hiddenBF16=seedHidden;seed.hiddenRows=1;seed.greedyResultsU32=seedRecord;
        seed.greedyRows=1;seed.logitRows=1;seed.logicalLength=begin;
        const policy::Params params{248320,test.depth,test.remaining,begin,kCapacity,1,{0,0}};
        const auto prepared=policy::prepare(params,chosen,test.prior);
        Snapshot expected;
        if(prepared.control.bodyEnabled) {
          const std::array<uint32_t,1> next{chosen.token};
          expected=snapshot(baseline.forward(reference,seedHidden,next));
        }
        RawBuffers qsaBefore,scratchBefore;
        if(!prepared.control.bodyEnabled) {
          qsaBefore=rawBuffers(backend,chained.stateBuffers(actual));
          scratchBefore=rawBuffers(backend,chained.scratchBuffers());
        }
        const auto result=workspace.run(actual,seed,test.depth,test.remaining,prior);
        timing(result.timing,bool(prepared.control.bodyEnabled));
        const auto completion=policy::complete(prepared,expected.greedy,0,true);
        compareControl(result.control,completion.control);
        require(result.originalLength==begin&&result.logicalLength==completion.logicalLength&&
                actual.logicalLength()==completion.logicalLength&&!actual.poisoned(),
                "chain published a wrong logical length or poisoned a skipped seed");
        require(result.indirectDispatches>0&&result.indirectDispatches<=256,"body dispatch count outside guard capacity");
        const auto *staticGroups=static_cast<const uint32_t *>(buffers[0].contents());
        const auto *indirectGroups=static_cast<const uint32_t *>(buffers[1].contents());
        for(uint32_t index=0;index<result.indirectDispatches;++index) {
          for(uint32_t axis=0;axis<3;++axis)
            require(indirectGroups[index*3+axis]==(prepared.control.bodyEnabled?staticGroups[index*3+axis]:0u),
                    "indirect group triplet differs from guarded static dimensions");
          ++indirectTriplets;
        }
        require(sentinel(buffers[1],uint64_t{result.indirectDispatches}*12),"guard wrote beyond used indirect dimensions");
        if(prepared.control.bodyEnabled) {
          ++naturalBodies;
          require(test.injection==Injection::Natural,"injected invalid/EOS seed entered a head pair");
          require(sameRecord(record(result.secondGreedy),expected.greedy),"second exact greedy record differs");
          require(result.control.proposals[0]==first.greedy.token&&
                  result.control.proposals[1]==expected.greedy.token,"natural proposal IDs differ");
          exact(words(result.hiddenSnapshot,kHyper),expected.hidden,"indirect R1 head hidden differs from stepwise baseline");
          require(sentinel(result.hiddenSnapshot,uint64_t{kHyper}*2),"owned snapshot wrote its allocation padding");
          exactWords+=kHyper;
          int64_t gpuToken=0;std::memcpy(&gpuToken,buffers[2].contents(),8);
          require(gpuToken==int64_t(chosen.token),"GPU token bridge disagrees with I64 embedding ABI");
          exactQsaBytes+=exactRaw(rawBuffers(backend,chained.stateBuffers(actual)),
              rawBuffers(backend,baseline.stateBuffers(reference)),
              "active indirect body QSA cache differs from stepwise baseline");
        } else {
          ++skippedCases;
          require(sentinel(buffers[2])&&sentinel(buffers[3])&&sentinel(buffers[6]),
                  "skipped body modified I64, second-greedy or hidden suffix");
          require(!result.control.consumed_pairs&&result.logicalLength==begin,
                  "skipped body consumed a head pair");
          exactQsaBytes+=exactRaw(rawBuffers(backend,chained.stateBuffers(actual)),qsaBefore,
              "skipped indirect body wrote QSA state bytes");
          untouchedScratchBytes+=exactRaw(rawBuffers(backend,chained.scratchBuffers()),scratchBefore,
              "skipped indirect body wrote head/QSA/greedy scratch bytes");
        }
        // Every healthy case restores its original fold and overwrites a real
        // changed-token pair, independently exercising future rollback behavior.
        baseline.truncate(reference,begin);chained.truncate(actual,begin);
        const std::array<uint32_t,1> changed{37};
        const auto continuedA=snapshot(baseline.forward(reference,seedHidden,changed));
        const auto continuedB=snapshot(chained.forward(actual,seedHidden,changed));
        exact(continuedB.hidden,continuedA.hidden,"rollback/continued hidden differs");
        exact(continuedB.logits,continuedA.logits,"rollback/continued full vocabulary differs");
        require(sameRecord(continuedB.greedy,continuedA.greedy)&&reference.logicalLength()==begin+1&&
                actual.logicalLength()==begin+1,"rollback/continued greedy or state differs");
        exactWords+=uint64_t{kHyper}+kVocabulary;
        std::ostringstream entry;entry<<std::setprecision(12)<<"{\"name\":"<<splash::json::quote(test.name)
            <<",\"begin\":"<<begin<<",\"begin_mod4\":"<<test.modulo<<",\"requested_depth\":"<<test.depth
            <<",\"remaining\":"<<test.remaining<<",\"proposal_count\":"<<result.control.proposal_count
            <<",\"consumed_pairs\":"<<result.control.consumed_pairs<<",\"errors\":"<<result.control.errors
            <<",\"body_enabled\":"<<(result.control.body_enabled?"true":"false")
            <<",\"returned_hidden_valid\":"<<(result.control.body_enabled&&!result.control.errors?"true":"false")
            <<",\"indirect_dispatches\":"<<result.indirectDispatches<<",\"chain_gpu_seconds\":"<<result.timing.gpuSeconds
            <<",\"chain_wall_seconds\":"<<result.timing.wallSeconds
            <<",\"baseline_step_gpu_seconds\":"<<expected.timing.gpuSeconds
            <<",\"baseline_step_wall_seconds\":"<<expected.timing.wallSeconds
            <<",\"seed_hidden_and_logits_exact\":true,\"continuation_hidden_logits_greedy_exact\":true"
            <<",\"qsa_contents_exact\":true,\"skipped_scratch_bytes_unchanged\":"
            <<(prepared.control.bodyEnabled?"null":"true")<<",\"pass\":true}";
        records.push_back(entry.str());std::cerr<<"private two-proposal "<<test.name<<" exact/skip PASS\n";
      }
      require(naturalBodies>0,"natural seeds selected EOS; no executed-body parity was established");
      std::ostringstream report;report<<"{\"pass\":true,\"private_prototype\":true,\"end_to_end_qualified\":false"
          <<",\"timing_is_observation_only\":true,\"model_source_identity\":"<<splash::json::quote(weights.sourceIdentity())
          <<",\"head_projection_route\":"<<splash::json::quote(chained.projectionRouteSemantics())
          <<",\"head_attention_route\":"<<splash::json::quote(chained.attentionRouteSemantics())
          <<",\"fixture\":"<<splash::json::quote(captured.empty()?"deterministic-synthetic-bf16":"captured-bf16-real-rows-cycled")
          <<",\"prime_rows\":"<<primeRows<<",\"natural_bodies\":"<<naturalBodies<<",\"skipped_cases\":"<<skippedCases
          <<",\"indirect_triplets_checked\":"<<indirectTriplets<<",\"exact_bf16_words_compared\":"<<exactWords
          <<",\"exact_qsa_state_bytes_compared\":"<<exactQsaBytes
          <<",\"untouched_skipped_scratch_bytes_compared\":"<<untouchedScratchBytes
          <<",\"inspection_gpu_copies_excluded_from_chain_timing\":true"
          <<",\"state_contents_directly_snapshot_capability\":true,\"production_modified\":false,\"cases\":[";
      for(size_t index=0;index<records.size();++index){if(index)report<<',';report<<records[index];}
      report<<"]}";write(argv[3],report.str());return 0;
    } catch(const std::exception &error) {
      std::cerr<<"private two-proposal oracle: "<<error.what()<<'\n';
      if(argc==4) {
        std::ostringstream report;report<<"{\"pass\":false,\"private_prototype\":true,\"end_to_end_qualified\":false,\"error\":"
            <<splash::json::quote(error.what())<<",\"completed_cases\":[";
        for(size_t index=0;index<records.size();++index){if(index)report<<',';report<<records[index];}
        report<<"]}";try{write(argv[3],report.str());}catch(...){}
      }
      return 1;
    }
  }
}
