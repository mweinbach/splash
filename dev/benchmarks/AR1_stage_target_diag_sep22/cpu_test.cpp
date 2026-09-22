#include "bridge.hpp"
#include <iostream>
#include <limits>
using namespace splash::flash::ar1_stage_target_diag_sep22;
void need(bool v){if(!v)throw std::runtime_error("AR1 pure CPU metadata gate failed");}
splash::metal::CommandDispatchProfile profile(){
  using namespace splash::metal;CommandDispatchProfile p;
  p.mode=CommandDispatchProfilingMode::StagePerDispatch;p.status=CommandDispatchProfileStatus::Complete;
  p.encoderBoundariesAltered=true;p.commandKernelTimingValid=true;p.dispatchCount=1;
  p.timing.gpuSeconds=.001;p.timing.wallSeconds=.002;p.commandGpuStartSeconds=1;p.commandGpuEndSeconds=2;
  CommandDispatchTimestamp d;d.pipelineName="synthetic_CPU_metadata_only";d.threadgroups={1,1,1};d.threadsPerThreadgroup={32,1,1};
  d.timestampsValid=true;d.gpuStartTimestamp=1;d.gpuEndTimestamp=2;d.calibratedStartSeconds=1;d.calibratedEndSeconds=2;d.gpuSeconds=.001;
  d.bindings={{0,64,false},{1,40,true}};p.dispatches.push_back(d);return p;
}
int main(){
  uint64_t tests=0;
  need(!flag("0")&&flag("1"));++tests;
  for(auto value:{"","00","true","2"}){bool rejected=false;try{(void)flag(value);}catch(...){rejected=true;}need(rejected);++tests;}
  need(ordinaryCookie(1,1,true,2048,2048));++tests;
  need(!ordinaryCookie(2,1,true,2048,2048)&&!ordinaryCookie(1,2,true,2048,2048)&&
      !ordinaryCookie(1,1,false,2048,2048)&&!ordinaryCookie(1,1,true,2047,2048));++tests;
  const auto original=profile();need(validProfile(original));++tests;
  {auto p=original;p.droppedProfilesBefore=1;need(!validProfile(p));++tests;}
  {auto p=original;p.dispatchMetadataTruncated=true;need(!validProfile(p));++tests;}
  {auto p=original;p.dispatchCount=4097;need(!validProfile(p));++tests;}
  {auto p=original;p.dispatches[0].timestampsValid=false;need(!validProfile(p));++tests;}
  {auto p=original;p.dispatches[0].gpuSeconds=std::numeric_limits<double>::quiet_NaN();need(!validProfile(p));++tests;}
  {auto p=original;p.dispatches[0].bindings[1].index=0;need(!validProfile(p));++tests;}
  {auto p=original;p.dispatches[0].bindings[1].index=32;need(!validProfile(p));++tests;}
  {auto p=original;p.hostCommitSeconds=std::numeric_limits<double>::infinity();need(!validProfile(p));++tests;}
  {auto p=original;p.timing.host.commitSeconds=std::numeric_limits<double>::infinity();need(!validProfile(p));++tests;}
  {auto p=original;p.mode=splash::metal::CommandDispatchProfilingMode::Command;need(!validProfile(p));++tests;}
  {auto p=original;p.commandKernelTimingValid=false;need(!validProfile(p));++tests;}
  {auto p=original;p.encoderBoundariesAltered=false;need(!validProfile(p));++tests;}
  std::cout<<"{\"valid\":true,\"tests\":"<<tests<<",\"GPU_work\":false,\"model_tensor_or_token_file_reads\":0}\n";
}
