// --cpu-self-test and compilation never construct a Metal device.
// --gpu is Root-only, after model unload and under the exclusive GPU slot.
#include "bandwidth.hpp"
#include "metal/CommandGraph.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
constexpr uint64_t GiB=uint64_t{1}<<30,Guard=64;
constexpr std::array<uint32_t,2> Seeds{0x29158a31u,0xbd64e907u};
static_assert(sizeof(CommandTiming)==200,"fresh ABI200 core backend required");
void require(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
uint32_t word(uint64_t index,uint32_t seed){return uint32_t(index)*kBWMultiplier+seed;}

// Sum floor((a*i+b)/m), exact nonnegative integer arithmetic. Inputs normalize
// to modulo2^32 before use; 128-bit intermediates avoid pre-cancellation overflow.
__uint128_t floorSum(uint64_t n,uint64_t m,uint64_t a,uint64_t b){
  __uint128_t result=0;
  while(true){
    if(a>=m){result+=__uint128_t(n)*(n-1)/2*(a/m);a%=m;}
    if(b>=m){result+=__uint128_t(n)*(b/m);b%=m;}
    const __uint128_t y=__uint128_t(a)*n+b;
    if(y<m)return result;
    n=uint64_t(y/m);b=uint64_t(y%m);std::swap(m,a);
  }
}
uint64_t componentSum(uint64_t begin,uint64_t n,uint32_t component,uint32_t seed){
  constexpr uint64_t modulus=uint64_t{1}<<32;
  const uint64_t a=uint32_t(4u*kBWMultiplier),b=word(begin*4+component,seed);
  const auto result=__uint128_t(n)*b+__uint128_t(a)*n*(n-1)/2-
      __uint128_t(modulus)*floorSum(n,modulus,a,b);
  require(result<=std::numeric_limits<uint64_t>::max(),"expected checksum exceeds64 bits");
  return uint64_t(result);
}
BWRecord expected(uint64_t begin,uint64_t n,uint32_t seed){
  require(n>0&&n<=kBWChunkVectors,"bad expected CTA range");BWRecord r{};
  for(uint32_t c=0;c<4;++c)r.sums[c]=componentSum(begin,n,c,seed);
  r.vectors=n;r.first=word(begin*4,seed);r.last=word((begin+n)*4-1,seed);return r;
}
void cpuSelfTest(){
  uint64_t checks=0;
  for(uint64_t size:{1ull,17ull,255ull,256ull,257ull,65535ull,65536ull,65537ull}){
    std::vector<uint8_t> visits(size);
    for(uint64_t group=0;group<(size+kBWChunkVectors-1)/kBWChunkVectors;++group){
      const auto begin=group*kBWChunkVectors,end=std::min(begin+kBWChunkVectors,size);
      for(uint64_t tid=0;tid<kBWThreads;++tid)for(uint64_t i=begin+tid;i<end;i+=kBWThreads)++visits[i];
    }
    for(auto count:visits){require(count==1,"CTA coverage notexactlyonce");++checks;}
  }
  for(uint32_t seed:{0u,1u,Seeds[0],Seeds[1],0xffffffffu})
    for(uint64_t begin:{0ull,65536ull,536805376ull,1073676288ull})
      for(uint64_t n:{1ull,17ull,256ull,65536ull}){
        const auto reference=expected(begin,n,seed);std::array<uint64_t,4> brute{};
        for(uint64_t i=begin;i<begin+n;++i)for(uint32_t c=0;c<4;++c)brute[c]+=word(i*4+c,seed);
        for(uint32_t c=0;c<4;++c){require(reference.sums[c]==brute[c],"closed checksum mismatch");++checks;}
      }
  require(expected(0,65536,Seeds[0]).first!=expected(65536,65536,Seeds[0]).first,"CTA address fingerprint collapsed");
  require(3*8*GiB+2*8*GiB/16/kBWChunkVectors*sizeof(BWRecord)+2*Guard<32*GiB,"allocation exceeds32GiB");
  std::cout<<"{\"valid\":true,\"cpu_checks\":"<<checks<<",\"gpu_commands\":0,\"model_files_opened\":0,\"command_timing_abi_bytes\":"<<sizeof(CommandTiming)<<"}\n";
}
double median(std::vector<double> v){require(!v.empty(),"missing samples");std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
void doubles(std::ostream &out,const std::vector<double>&values){out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';}
struct Records {
  MetalBuffer allocation,buffer;uint64_t capacity;
  Records(MetalBackend &backend,uint64_t groups):capacity(groups){
    allocation=backend.allocateBuffer(groups*sizeof(BWRecord)+2*Guard,BufferStorage::Shared,"bandwidth checksum CTA records withguards");
    std::memset(allocation.contents(),0xcd,allocation.sizeBytes());
    buffer=backend.view(allocation,Guard,groups*sizeof(BWRecord));
  }
  void guards()const{
    const auto*p=static_cast<const uint8_t*>(allocation.contents());
    for(uint64_t i=0;i<Guard;++i)require(p[i]==0xcd&&p[Guard+capacity*sizeof(BWRecord)+i]==0xcd,"record canary changed");
  }
  uint64_t check(const std::vector<BWRecord>&a,const std::vector<BWRecord>&b,
      uint64_t vectors,uint32_t stamp,bool copyCounts=false)const{
    guards();const auto*p=static_cast<const BWRecord*>(buffer.contents());uint64_t observed=0;
    const auto groups=(vectors+kBWChunkVectors-1)/kBWChunkVectors;
    for(uint64_t group=0;group<groups;++group){
      const BWRecord&ref=group<a.size()?a[group]:b.at(group-a.size());const auto &r=p[group];
      require(r.stamp==stamp&&r.vectors==ref.vectors&&r.badWords==0,"CTA stamp/count/mismatch check failed");
      if(!copyCounts){
        for(uint32_t c=0;c<4;++c)require(r.sums[c]==ref.sums[c],"CTA exact64 checksum failed");
        require(r.first==ref.first&&r.last==ref.last,"CTA address fingerprint failed");
      }
      observed+=r.vectors;
    }
    require(observed==vectors,"global uint64vectorcount differs");return observed;
  }
};
BWParams params(uint64_t vectorsPerBuffer,uint64_t vectors,uint32_t seedA,uint32_t seedB,uint32_t stamp){return {vectorsPerBuffer,vectors,kBWChunkVectors,seedA,seedB,stamp,0};}
void command(CommandGraph&graph,const std::string&name,std::vector<MetalBuffer>buffers,const BWParams&p){
  graph.add(name,std::move(buffers),p,{(p.vectors+kBWChunkVectors-1)/kBWChunkVectors,1,1},{kBWThreads,1,1});
}
} //namespace

int main(int argc,char**argv){@autoreleasepool{try{
  if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpuSelfTest();return 0;}
  require(argc>=4&&std::string(argv[1])=="--gpu","usage: bandwidth --cpu-self-test | --gpu METALLIB FRESH_REPORT [--buffer-gib4|8] [--warmups3] [--samples7]");
  const std::filesystem::path output=argv[3];require(!std::filesystem::exists(output),"fresh report path required");
  uint32_t gib=8,warmups=3,samples=7;
  for(int i=4;i<argc;i+=2){require(i+1<argc,"missing argument value");size_t used=0;const std::string raw=argv[i+1];const auto value=std::stoul(raw,&used);require(used==raw.size(),"invalid integer");const std::string flag=argv[i];
    if(flag=="--buffer-gib"){require(value==4||value==8,"bufferGiB mustbe4or8");gib=uint32_t(value);}
    else if(flag=="--warmups"){require(value>=3&&value<=20,"warmups mustbe3..20");warmups=uint32_t(value);}
    else if(flag=="--samples"){require(value>=7&&value<=31,"samples mustbe7..31");samples=uint32_t(value);}
    else require(false,"unknown option");
  }
  const uint64_t bufferBytes=gib*GiB,bufferVectors=bufferBytes/16,groups=bufferVectors/kBWChunkVectors;
  const uint64_t plannedBytes=3*bufferBytes+2*groups*sizeof(BWRecord)+2*Guard;
  require(plannedBytes<32*GiB,"allocation mustremainbelow32GiB");
  std::ofstream report(output);require(bool(report),"cannotcreate report");
  report<<std::setprecision(15)<<"{\"schema\":\"splash-sustained-memory-payload-abi200-v1\",\"execution_started\":true,\"gpu_executed\":true,\"model_files_opened\":0,\"command_timing_abi_bytes\":200,\"metallib\":"<<splash::json::quote(argv[2])<<",\"buffer_bytes\":"<<bufferBytes<<",\"planned_total_allocation_bytes\":"<<plannedBytes<<",\"memory_cap_bytes\":"<<32*GiB<<",\"warmups_per_case\":"<<warmups<<",\"measured_samples_per_case\":"<<samples<<",\"vector_bytes\":16,\"threads_per_cta\":256,\"vectors_per_cta\":65536,\"measurement\":\"hardwareGPUcommandtime and synchronous commandwall; initialization,pipelinewarmups,CPUchecksumchecks and wholebuffer GPUwordvalidation excluded\",\"physical_DRAM_counter_collected\":false,\"metric\":\"effective explicitoperandpayloadbytes per GPUduration; physicaltransactions,caching,instructions and interdie traffic unresolved\",\"cases\":[";report.flush();
  MetalBackend backend(argv[2]);
  require(backend.capabilities().hasUnifiedMemory,"Shared sustainedbenchmark requiresunifiedmemory");
  require(backend.capabilities().maxBufferLengthBytes>=bufferBytes,"device maximum bufferlength istoo small");
  std::array<MetalBuffer,3> data;
  for(uint32_t i=0;i<3;++i)data[i]=backend.allocateBuffer(bufferBytes,BufferStorage::Shared,"large standalone bandwidth residentbuffer"+std::to_string(i));
  Records results(backend,2*groups);
  require(backend.memoryStats().allocatedBytes<32*GiB,"actualGPUallocation exceeds32GiB");
  std::array<std::vector<BWRecord>,2> references;
  for(uint32_t s=0;s<2;++s){
    references[s].reserve(groups);for(uint64_t group=0;group<groups;++group)references[s].push_back(expected(group*kBWChunkVectors,kBWChunkVectors,Seeds[s]));
    CommandGraph init;command(init,"bw_initialize",{data[s]},params(bufferVectors,bufferVectors,Seeds[s],0,0));(void)backend.submitCommand(init.dispatches());
  }
  uint32_t stamp=0;uint64_t untimedValidationCommands=0,timedCommands=0;
  auto validate=[&](uint32_t buffer,uint32_t seedIndex){
    CommandGraph validation;const auto p=params(bufferVectors,bufferVectors,Seeds[seedIndex],0,++stamp);
    command(validation,"bw_validate",{data[buffer],results.buffer},p);(void)backend.submitCommand(validation.dispatches());++untimedValidationCommands;
    (void)results.check(references[seedIndex],{},bufferVectors,p.stamp);
  };
  validate(0,0);validate(1,1);
  bool firstCase=true;
  for(uint32_t mode=0;mode<3;++mode){
    const uint64_t vectors=mode==1?bufferVectors*2:bufferVectors,caseGroups=vectors/kBWChunkVectors;
    std::vector<double> gpu,wall;std::vector<uint32_t> sourceIndices;
    for(uint32_t sample=0;sample<warmups+samples;++sample){
      const uint32_t source=sample%2,other=1-source;const auto p=params(bufferVectors,vectors,Seeds[source],Seeds[other],++stamp);
      CommandGraph graph;if(mode<2)command(graph,"bw_read",{data[source],data[other],results.buffer},p);
      else command(graph,"bw_copy",{data[source],data[2],results.buffer},p);
      const auto timing=backend.submitCommand(graph.dispatches());++timedCommands;
      require(timing.gpuSeconds>0&&timing.wallSeconds>0,"missing GPU/walltimestamps");
      (void)results.check(references[source],mode==1?references[other]:std::vector<BWRecord>{},vectors,p.stamp,mode==2);
      if(sample>=warmups){gpu.push_back(timing.gpuSeconds);wall.push_back(timing.wallSeconds);sourceIndices.push_back(source);}
      if(mode==2)validate(2,source);
    }
    const uint64_t readBytes=bufferBytes*(mode==1?2:1),writeBytes=mode==2?bufferBytes:0;
    const uint64_t extraReadBytes=mode==2?0:caseGroups*8,auxWriteBytes=caseGroups*sizeof(BWRecord);
    const double gpuMedian=median(gpu),wallMedian=median(wall);
    if(!firstCase)report<<',';firstCase=false;
    report<<"{\"mode\":"<<splash::json::quote(mode==0?"read_one_rotating_source":mode==1?"read_two_rotating_source_order":"copy_rotating_source_to_destination")<<",\"explicit_data_bytes_read\":"<<readBytes<<",\"explicit_data_bytes_written\":"<<writeBytes<<",\"fingerprint_extra_read_bytes\":"<<extraReadBytes<<",\"checksum_record_write_bytes\":"<<auxWriteBytes<<",\"vectors_counted_per_sample\":"<<vectors<<",\"cta_records_checked_per_sample\":"<<caseGroups<<",\"all_cta_checksums_counts_fingerprints_pass\":true,\"copy_every_destination_word_checked_outside_timing\":"<<(mode==2?"true":"false")<<",\"median_gpu_seconds\":"<<gpuMedian<<",\"median_command_wall_seconds\":"<<wallMedian<<",\"median_data_payload_GBps\":"<<double(readBytes+writeBytes)/gpuMedian/1e9<<",\"median_all_explicit_payload_GBps\":"<<double(readBytes+writeBytes+extraReadBytes+auxWriteBytes)/gpuMedian/1e9<<",\"median_command_wall_data_GBps\":"<<double(readBytes+writeBytes)/wallMedian/1e9<<",\"gpu_seconds\":";doubles(report,gpu);report<<",\"wall_seconds\":";doubles(report,wall);report<<",\"source_buffer_indices\":[";
    for(size_t i=0;i<sourceIndices.size();++i){if(i)report<<',';report<<sourceIndices[i];}report<<"]}";report.flush();
  }
  validate(0,0);validate(1,1);results.guards();
  report<<"],\"total_read_copy_commands_including_warmups\":"<<timedCommands<<",\"untimed_full_word_validation_commands\":"<<untimedValidationCommands<<",\"actual_allocated_gpu_bytes\":"<<backend.memoryStats().allocatedBytes<<",\"source_patterns_unchanged_full_word_validation\":true,\"record_guards_pass\":true,\"execution_complete\":true,\"completed\":true,\"valid\":true}\n";
  // Completion markers are written only after every validation succeeds.
  std::cout<<"Completed standalone memorypayload benchmark: "<<output<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"bandwidth oracle: "<<e.what()<<'\n';return 1;}}}
