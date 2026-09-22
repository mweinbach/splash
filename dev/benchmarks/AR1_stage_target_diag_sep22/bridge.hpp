#pragma once
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/ProfilingJson.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <fcntl.h>
#include <unistd.h>

#ifndef SPLASH_AR1_DIAGNOSTIC_SOURCE_ID
#define SPLASH_AR1_DIAGNOSTIC_SOURCE_ID "0000000000000000000000000000000000000000000000000000000000000000"
#endif
namespace splash::flash::ar1_stage_target_diag_sep22 {
inline constexpr char kDiagnosticSourceId[] = SPLASH_AR1_DIAGNOSTIC_SOURCE_ID;
inline constexpr char kOriginalSourceId[] = "162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a";
inline constexpr char kFlag[] = "SPLASH_FLASH_DIAG_AR1_TARGET_STAGE_SEP22";
inline constexpr char kOutput[] = "SPLASH_FLASH_DIAG_AR1_STAGE_OUTPUT_SEP22";
inline constexpr uint64_t kReservationBytes = 64ULL<<20;
inline constexpr size_t kMaximumDispatches = 4096,kMaximumBindings = 32,kMaximumJsonBytes = 16ULL<<20;
static_assert(sizeof(metal::CommandTiming)==200);
static_assert(sizeof(kDiagnosticSourceId)==65);

inline std::string env(const char*name,const char*fallback){const char*v=std::getenv(name);return v?v:fallback;}
inline bool flag(std::string_view value){
  if(value=="0")return false;if(value=="1")return true;
  throw std::invalid_argument("AR1 diagnostic flag must be canonical0/1");
}
struct Config {
  bool enabled=false;std::string output;
  std::array<std::string,9> frozen;
  static Config read(){
    Config c;c.frozen={env(kFlag,"0"),env(kOutput,""),env("SPLASH_FLASH_MTP",""),
      env("SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21",""),
      env("SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21",""),env("SPLASH_FLASH_GDN_PREFILL_FMA_SEP21",""),
      env("SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21",""),env("SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21",""),
      env("SPLASH_FLASH_PREFILL_ROWS","")};
    c.enabled=flag(c.frozen[0]);if(!c.enabled)return c;
    if(c.frozen[2]!="0"||c.frozen[3]!="0"||c.frozen[4]!="1"||c.frozen[5]!="1"||
        c.frozen[6]!="1"||c.frozen[7]!="1"||c.frozen[8]!="2048")
      throw std::invalid_argument("AR1 diagnostic requires MTP0/teacherbulk0/allfourprefill1/rows2048");
    c.output=c.frozen[1];
    if(c.output.empty()||c.output.size()>4096||!std::filesystem::path(c.output).is_absolute())
      throw std::invalid_argument("AR1 diagnostic requires fresh absolute output path");
    return c;
  }
  static Config fromEnvironment(){
    const Config now=read();static const Config frozen=now;
    if(now.enabled!=frozen.enabled||(now.enabled&&now.frozen!=frozen.frozen))
      throw std::logic_error("AR1 diagnostic environment changed after admission");
    return frozen;
  }
  void check()const{const auto now=fromEnvironment();if(now.enabled!=enabled||(enabled&&now.frozen!=frozen))
    throw std::logic_error("AR1 diagnostic instance configuration changed");}
};
inline void validateStartup(){(void)Config::fromEnvironment();}
inline bool ordinaryCookie(uint64_t id,uint64_t generation,bool noMTP,uint64_t offset,uint64_t count){
  return id==1&&generation==1&&noMTP&&offset==2048&&count==2048;
}
inline void writeMemory(std::ostream&out,const metal::MetalMemoryStats&m){
  out<<"{\"owner_allocated_bytes\":"<<m.allocatedBytes<<",\"owner_peak_allocated_bytes\":"<<m.peakAllocatedBytes
    <<",\"sampled_device_current_allocated_bytes\":"<<m.deviceCurrentAllocatedBytes
    <<",\"sampled_device_peak_allocated_bytes\":"<<m.devicePeakAllocatedBytes
    <<",\"pending_sparse_unmaps\":"<<m.pendingSparseUnmaps
    <<",\"device_values_are_sampled_process_counters\":true,\"vendor_counter_backing_bytes\":null,\"vendor_counter_backing_known\":false}";
}
inline void writeGovernor(std::ostream&out,const engine::MemoryGovernorSnapshot&g){
  out<<"{\"limit_bytes\":"<<g.limitBytes<<",\"observed_resident_bytes\":"<<g.observedResidentBytes
    <<",\"reserved_bytes\":"<<g.reservedBytes<<",\"headroom_bytes\":"<<g.headroomBytes
    <<",\"denied_reservations\":"<<g.deniedReservations<<",\"host_available_bytes\":"<<g.hostAvailableBytes
    <<",\"host_reserve_bytes\":"<<g.hostReserveBytes<<",\"growth_allowed\":"<<(g.growthAllowed?"true":"false")<<'}';
}
inline bool validProfile(const metal::CommandDispatchProfile&p){
  if(p.status!=metal::CommandDispatchProfileStatus::Complete||p.mode!=metal::CommandDispatchProfilingMode::StagePerDispatch||
      !p.encoderBoundariesAltered||p.samplingBarriers||p.droppedProfilesBefore||p.dispatchMetadataTruncated||
      !p.commandKernelTimingValid||p.reason.size()>1024||p.dispatchCount==0||p.dispatchCount>kMaximumDispatches||p.dispatches.size()!=p.dispatchCount)return false;
  const std::array times{p.timing.gpuSeconds,p.timing.wallSeconds,p.commandGpuStartSeconds,p.commandGpuEndSeconds,
    p.commandKernelStartSeconds,p.commandKernelEndSeconds,p.hostPreparationSeconds,p.hostEncodingSeconds,
    p.sparseDependencyWaitSeconds,p.hostCommitSeconds,p.hostSubmissionStartSeconds,p.hostEncodingStartSeconds,
    p.hostEncodingEndSeconds,p.hostCommitBeginSeconds,p.hostCommitEndSeconds,p.hostScheduledSeconds,p.hostCompletedSeconds,p.hostReadySeconds};
  if(!std::all_of(times.begin(),times.end(),[](double t){return std::isfinite(t)&&t>=0;})||
      p.timing.gpuSeconds<=0||p.timing.wallSeconds<=0||p.commandGpuStartSeconds<=0||p.commandGpuEndSeconds<p.commandGpuStartSeconds)return false;
  const auto&h=p.timing.host;
  const std::array hostTimes{h.preparationSeconds,h.encodingSeconds,h.beforeCommitSeconds,h.dependencyWaitSeconds,
    h.commitSeconds,h.commitToScheduledCallbackSeconds,h.commitToCompletedCallbackSeconds,h.completionCallbackBeforeWallEndSeconds,
    h.submissionReturnSeconds,h.ticketBlockingWaitSeconds,h.preCommitMemorySampleSeconds,h.postCommitMemorySampleSeconds,
    h.scheduledMemorySampleSeconds,h.completedMemorySampleSeconds};
  if(!std::all_of(hostTimes.begin(),hostTimes.end(),[](double t){return std::isfinite(t);}))return false;
  for(const auto*sample:{&p.preCommitDeviceMemorySample,&p.postCommitDeviceMemorySample,&p.scheduledDeviceMemorySample,&p.completedDeviceMemorySample})
    if(!std::isfinite(sample->beganSteadySeconds)||!std::isfinite(sample->endedSteadySeconds)||!std::isfinite(sample->seconds))return false;
  for(const auto*bridge:{&p.commitClockBridge,&p.completedClockBridge})
    if(!std::isfinite(bridge->beganSteadySeconds)||!std::isfinite(bridge->endedSteadySeconds)||!std::isfinite(bridge->machSeconds)||
        !std::isfinite(bridge->steadyMinusMachSeconds)||!std::isfinite(bridge->uncertaintySeconds))return false;
  for(size_t i=0;i<p.dispatches.size();++i){const auto&d=p.dispatches[i];
    if(d.index!=i||d.pipelineName.empty()||d.pipelineName.size()>256||d.bindings.size()>kMaximumBindings||
        !d.timestampsValid||!d.gpuStartTimestamp||d.gpuEndTimestamp<d.gpuStartTimestamp||
        !std::isfinite(d.gpuSeconds)||d.gpuSeconds<0||!std::isfinite(d.calibratedStartSeconds)||
        !std::isfinite(d.calibratedEndSeconds)||d.calibratedStartSeconds<0||d.calibratedEndSeconds<d.calibratedStartSeconds||
        !d.threadgroups.x||!d.threadgroups.y||!d.threadgroups.z||!d.threadsPerThreadgroup.x||
        !d.threadsPerThreadgroup.y||!d.threadsPerThreadgroup.z)return false;
    std::array<bool,kMaximumBindings> seen{};
    for(const auto&b:d.bindings){if(b.index>=kMaximumBindings||seen[b.index])return false;seen[b.index]=true;}
  }
  return true;
}
class ScopedAR1;
class Diagnostic {
public:
  explicit Diagnostic(Config c):config_(std::move(c)){
    config_.check();if(!config_.enabled)return;
    fd_=::open(config_.output.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd_<0)throw std::runtime_error("cannot create fresh AR1 diagnostic output");
  }
  ~Diagnostic(){if(fd_>=0)::close(fd_);}
  Diagnostic(const Diagnostic&)=delete;Diagnostic&operator=(const Diagnostic&)=delete;
private:
  bool emit(std::string_view bytes)noexcept{
    if(fd_<0||written_||bytes.size()>kMaximumJsonBytes)return false;
    size_t position=0;while(position<bytes.size()){
      const auto count=::write(fd_,bytes.data()+position,bytes.size()-position);
      if(count<0&&errno==EINTR)continue;if(count<=0)return false;position+=size_t(count);
    }written_=true;return true;
  }
  Config config_;int fd_=-1;uint64_t completedAR1_=0;bool attempted_=false,written_=false;
  friend class ScopedAR1;
};
class ScopedAR1 {
public:
  ScopedAR1(Diagnostic&owner,uint64_t id,uint64_t generation,bool noMTP,uint64_t offset,uint64_t promptCount,
      uint64_t begin,engine::MemoryGovernor&governor,metal::MetalBackend&backend)noexcept:
      owner_(owner),id_(id),generation_(generation),begin_(begin),governor_(governor),backend_(backend){
    if(!owner_.config_.enabled||owner_.attempted_||!ordinaryCookie(id,generation,noMTP,offset,promptCount))return;
    eligible_=true;ordinal_=owner_.completedAR1_+1;selected_=ordinal_==3;
    try{
      owner_.config_.check();
      if(begin_!=2048+ordinal_-1){fail("ordinaryAR1 true offset differs from ordinal");return;}
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);restoredOff_=true;
      const auto stale=backend_.takeCommandDispatchProfiles();staleCount_=stale.size();
      if(!stale.empty()){fail("stale profiles before actual AR1");return;}
      if(!selected_)return;owner_.attempted_=true;
      beforeMemory_=backend_.memoryStats();beforeGovernor_=governor_.snapshot();beforeCaptured_=true;
      capability_=backend_.commandDispatchProfilingCapability();
      if(!capability_.supports(metal::CommandDispatchProfilingMode::StagePerDispatch)){
        fail("StagePerDispatch unsupported; genuine AR1 remains unprofiled");return;}
      metal::AllocationFailure refusal=metal::AllocationFailure::None;
      reservation_=governor_.tryReserve(kReservationBytes,&refusal);
      if(!reservation_){fail("64MiB diagnostic admission refused");return;}
      heldGovernor_=governor_.snapshot();admitted_=true;
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::StagePerDispatch);active_=true;restoredOff_=false;
    }catch(const std::exception&e){fail(e.what());off();}catch(...){fail("AR1 diagnostic activation failed");off();}
  }
  ~ScopedAR1(){if(!finished_&&selected_){fail("genuine AR1 scope abandoned");finish(false,0,0);}else off();}
  ScopedAR1(const ScopedAR1&)=delete;ScopedAR1&operator=(const ScopedAR1&)=delete;
  void complete(uint64_t returnedLength,uint32_t logitRows)noexcept{finish(true,returnedLength,logitRows);}
private:
  void fail(std::string_view reason)noexcept{invalid_=true;if(!reason_.empty())return;try{reason_.assign(reason.substr(0,1024));}catch(...){}}
  void off()noexcept{
    if(!active_)return;try{backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);active_=false;restoredOff_=true;}
    catch(...){fail("profiling Off restoration failed");}
  }
  void finish(bool successful,uint64_t length,uint32_t logits)noexcept{
    if(finished_)return;finished_=true;if(!eligible_)return;
    returnedLength_=length;logitRows_=logits;
    if(!successful||length!=begin_+1||logits!=1)fail("ordinary AR1 returned unexpected true length or logits");
    if(!invalid_&&restoredOff_&&!selected_)++owner_.completedAR1_;
    if(!selected_)return;
    off();
    try{
      afterMemory_=backend_.memoryStats();afterGovernor_=governor_.snapshot();afterCaptured_=true;
      auto profiles=backend_.takeCommandDispatchProfiles();profileCount_=profiles.size();
      if(!restoredOff_)fail("profiling did not return Off before next command");
      if(profiles.size()!=1)fail("exactly one real whole-AR1 profile required");
      else{profile_=std::move(profiles.front());if(!validProfile(*profile_))fail("AR1 profile complete/count/timestamp/metadata gate failed");}
      reservation_.reset();releasedMemory_=backend_.memoryStats();releasedGovernor_=governor_.snapshot();releaseCaptured_=true;
      if(releasedGovernor_.reservedBytes!=beforeGovernor_.reservedBytes)fail("diagnostic reservation not fully released");
      publish();
    }catch(const std::exception&e){fail(e.what());reservation_.reset();try{publish();}catch(...){}}
    catch(...){fail("AR1 diagnostic collection failed");reservation_.reset();try{publish();}catch(...){}}
  }
  void publish(){
    std::ostringstream out;profiling::ScopedJsonFormat format(out);
    out<<"{\"schema\":\"private-standard-AR1-stage-diagnostic-sep22-v1\",\"diagnostic_valid\":"<<(invalid_?"false":"true")
      <<",\"reason\":"<<json::quote(reason_)<<",\"baseline_source_identity_sha256\":"<<json::quote(kOriginalSourceId)
      <<",\"diagnostic_source_id\":"<<json::quote(kDiagnosticSourceId)
      <<",\"request_id\":"<<id_<<",\"generation\":"<<generation_
      <<",\"ordinary_AR1\":true,\"MTP_state_present\":false,\"physical_rows\":1,\"logit_rows\":"<<logitRows_
      <<",\"begin\":"<<begin_<<",\"returned_length\":"<<returnedLength_<<",\"AR1_ordinal\":"<<ordinal_
      <<",\"prior_completed_unprofiled_AR1_calls\":"<<owner_.completedAR1_
      <<",\"maximum_samples\":1,\"samples_attempted\":"<<(owner_.attempted_?1:0)
      <<",\"single_whole_target_command\":true,\"legacy_dispatch_replay\":false,\"profiling_restored_off\":"<<(restoredOff_?"true":"false")
      <<",\"stale_profiles_at_activation\":"<<staleCount_<<",\"profiles_after_target\":"<<profileCount_
      <<",\"governor_reservation_bytes\":"<<kReservationBytes<<",\"reservation_admitted\":"<<(admitted_?"true":"false")
      <<",\"reservation_released\":"<<(!reservation_?"true":"false")
      <<",\"instrumentation_is_performance_perturbation\":true,\"throughput_baseline\":false,\"SourceWorld_qualified\":false"
      <<",\"Forward_graph_snapshot_hook\":false,\"numeric_inline_ABI_values\":null"
      <<",\"metadata_scope\":\"Core authentic pipeline/grid/threads/bindingindex/extent/inlinekind; missing numericABI remainsunknown\""
      <<",\"tensor_payload_reads\":0,\"tensor_payload_hashes\":0,\"token_payload_reads\":0"
      <<",\"backend_destroyed\":null,\"vendor_counter_backing_bytes\":null,\"vendor_counter_backing_known\":false,\"capability\":";
    profiling::writeJson(out,capability_);
    out<<",\"memory_snapshots\":{\"before_activation\":";if(beforeCaptured_)writeMemory(out,beforeMemory_);else out<<"null";
    out<<",\"after_target\":";if(afterCaptured_)writeMemory(out,afterMemory_);else out<<"null";
    out<<",\"after_reservation_release\":";if(releaseCaptured_)writeMemory(out,releasedMemory_);else out<<"null";out<<'}';
    out<<",\"governor_snapshots\":{\"before_activation\":";if(beforeCaptured_)writeGovernor(out,beforeGovernor_);else out<<"null";
    out<<",\"reservation_held\":";if(admitted_)writeGovernor(out,heldGovernor_);else out<<"null";
    out<<",\"after_target\":";if(afterCaptured_)writeGovernor(out,afterGovernor_);else out<<"null";
    out<<",\"after_reservation_release\":";if(releaseCaptured_)writeGovernor(out,releasedGovernor_);else out<<"null";out<<'}';
    out<<",\"raw_command_profile\":";if(profile_)profiling::writeJson(out,*profile_);else out<<"null";out<<"}\n";
    const auto text=out.str();if(text.size()>kMaximumJsonBytes||!owner_.emit(text))fail("bounded AR1 diagnostic write failed");
  }
  Diagnostic&owner_;uint64_t id_,generation_,begin_,ordinal_=0,returnedLength_=0;uint32_t logitRows_=0;
  engine::MemoryGovernor&governor_;metal::MetalBackend&backend_;
  bool eligible_=false,selected_=false,active_=false,restoredOff_=false,finished_=false,invalid_=false;
  bool beforeCaptured_=false,afterCaptured_=false,releaseCaptured_=false,admitted_=false;
  size_t staleCount_=0,profileCount_=0;std::string reason_;
  metal::MetalMemoryStats beforeMemory_,afterMemory_,releasedMemory_;
  engine::MemoryGovernorSnapshot beforeGovernor_,heldGovernor_,afterGovernor_,releasedGovernor_;
  metal::CommandDispatchProfilingCapability capability_;
  std::optional<engine::MemoryGovernor::Reservation> reservation_;
  std::optional<metal::CommandDispatchProfile> profile_;
};
} // namespace splash::flash::ar1_stage_target_diag_sep22
