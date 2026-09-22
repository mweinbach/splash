// Observation only. --help and --cpu-self-test never initialize Metal.
#define main splash_teacher_profile_attribution_unused_main
#include "dev/benchmarks/prefill4k_attribution.mm"
#undef main
#include "dev/benchmarks/mtp_teacher_bulk_sep21/bulk.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_bridge.hpp"
#include "dev/benchmarks/mtp_teacher_bulk_sep21/policy.hpp"
#include <cstring>

namespace {
constexpr uint32_t profileRows = 1920, profileHyper = 10240;
constexpr uint64_t qWeightBytes = uint64_t{12288} * 2560 * 2;
using Profile = metal::CommandDispatchProfile;
using Dispatch = metal::CommandDispatchTimestamp;
using Mode = metal::CommandDispatchProfilingMode;
struct InjectionContract {
  uint64_t weights = 0, scales = 0, biases = 0;
};
uint64_t binding(const Dispatch &d, uint32_t index) {
  uint64_t result = UINT64_MAX;
  for (const auto &b : d.bindings) if (b.index == index) {
    require(result == UINT64_MAX && !b.inlineBytes, "duplicate/inline role buffer binding");
    result = b.sizeBytes;
  }
  require(result != UINT64_MAX, "missing role buffer binding");
  return result;
}
bool isQProjection(const Dispatch &d) {
  if (!d.pipelineName.starts_with("flash_dense")) return false;
  for (const auto &b : d.bindings)
    if (b.index == 1 && !b.inlineBytes && b.sizeBytes == qWeightBytes) return true;
  return false;
}
bool isInjection(const Dispatch &d, const InjectionContract &c) {
  if (!d.pipelineName.starts_with("flash_affine_q5_g64")) return false;
  for (const auto &b : d.bindings)
    if (b.index == 1 && !b.inlineBytes && b.sizeBytes == c.weights) return true;
  return false;
}
void validateBulkQ(const Dispatch &d) {
  require(d.pipelineName == "flash_dense_cache_m32_n128" &&
      d.threadgroups.x == 96 && d.threadgroups.y == 60 && d.threadgroups.z == 1 &&
      d.threadsPerThreadgroup.x == 128 && d.threadsPerThreadgroup.y == 1 && d.threadsPerThreadgroup.z == 1,
      "current bulk q projection pipeline/grid/thread contract changed");
  require(binding(d,0) == uint64_t{profileRows} * 2560 * 2 && binding(d,1) == qWeightBytes &&
      binding(d,2) == uint64_t{profileRows} * 12288 * 2 && binding(d,3) == 4,
      "current bulk q projection binding contract changed");
}
void validateBulkInjection(const Dispatch &d, const InjectionContract &c) {
  require(d.pipelineName == "flash_affine_q5_g64_c1" &&
      d.threadgroups.x == 1 && d.threadgroups.y == profileRows && d.threadgroups.z == 1 &&
      d.threadsPerThreadgroup.x == 256 && d.threadsPerThreadgroup.y == 1 && d.threadsPerThreadgroup.z == 1,
      "current bulk raw injection pipeline/grid/thread contract changed");
  require(binding(d,0) == uint64_t{profileRows} * profileHyper * 2 && binding(d,1) == c.weights &&
      binding(d,2) == c.scales && binding(d,3) == c.biases &&
      binding(d,4) == binding(d,0) && binding(d,5) == uint64_t{profileRows} * 4 * 2 && binding(d,6) == 4,
      "current bulk raw injection binding contract changed");
}
void validatePrepare(const Dispatch &d, uint32_t rows) {
  require(d.threadgroups.x == rows && d.threadgroups.y == 30 && d.threadgroups.z == 1 &&
      d.threadsPerThreadgroup.x == 256 && d.threadsPerThreadgroup.y == 1 && d.threadsPerThreadgroup.z == 1,
      "current teacher prepare grid/thread contract changed");
  require(binding(d,0) == uint64_t{rows} * 12288 * 2 && binding(d,1) == uint64_t{rows} * 512 * 2 &&
      binding(d,2) == uint64_t{rows} * 512 * 2 && binding(d,3) == uint64_t{rows} * 640 * 2 && binding(d,14) == 4,
      "current teacher prepare binding contract changed");
}
struct Roles {
  uint32_t q = 0, injection = 0, mix = 0, prepare = 0, pool = 0, other = 0;
  double qSeconds = 0, injectionSeconds = 0, mixSeconds = 0, prepareSeconds = 0, poolSeconds = 0, otherSeconds = 0;
};
Roles roles(const Profile &p, const InjectionContract &c, bool bulk) {
  require(p.status == metal::CommandDispatchProfileStatus::Complete && p.droppedProfilesBefore == 0 &&
      !p.dispatchMetadataTruncated && p.dispatches.size() == p.dispatchCount,
      "teacher dispatch profile is incomplete/dropped/unsupported");
  Roles r;
  uint64_t qIndex = UINT64_MAX, injectionIndex = UINT64_MAX, mixIndex = UINT64_MAX, firstPrepare = UINT64_MAX;
  bool waitingForPool = false;
  for (const auto &d : p.dispatches) {
    require(d.timestampsValid && std::isfinite(d.gpuSeconds) && d.gpuSeconds >= 0,
        "teacher dispatch timestamp invalid");
    if (isQProjection(d)) {
      if (bulk) validateBulkQ(d);
      ++r.q; r.qSeconds += d.gpuSeconds; qIndex = std::min(qIndex,d.index);
    } else if (isInjection(d,c)) {
      if (bulk) validateBulkInjection(d,c);
      ++r.injection; r.injectionSeconds += d.gpuSeconds; injectionIndex = std::min(injectionIndex,d.index);
    } else if (d.pipelineName == "flash_hc_mix_with_injection") {
      ++r.mix; r.mixSeconds += d.gpuSeconds; mixIndex = d.index;
    } else if (d.pipelineName == "flash_qsa_fast_prepare") {
      validatePrepare(d,bulk ? 128 : 127);
      require(!waitingForPool,"teacher chronological prepare/pool order changed");
      waitingForPool = true; ++r.prepare; r.prepareSeconds += d.gpuSeconds;
      firstPrepare = std::min(firstPrepare,d.index);
    } else if (d.pipelineName.starts_with("flash_qsa_pool_rope_")) {
      require(waitingForPool,"teacher pool lacks preceding original prepare");
      waitingForPool = false; ++r.pool; r.poolSeconds += d.gpuSeconds;
    } else {
      require(!waitingForPool,"unexpected dispatch between teacher prepare and pool");
      ++r.other; r.otherSeconds += d.gpuSeconds;
    }
  }
  require(r.q == (bulk ? 1u : 2u) && r.injection == 1 && r.mix == 1 &&
      r.prepare == (bulk ? 15u : 1u) && r.pool == r.prepare && !waitingForPool &&
      injectionIndex < mixIndex && mixIndex < qIndex && qIndex < firstPrepare,
      "teacher role counts/order changed");
  return r;
}
void writeRoles(std::ostream &out, const Roles &r, const Profile &p) {
  const double sum = r.qSeconds+r.injectionSeconds+r.mixSeconds+r.prepareSeconds+r.poolSeconds+r.otherSeconds;
  out << "{\"q_projection_dispatches\":" << r.q << ",\"q_projection_gpu_seconds\":" << r.qSeconds
      << ",\"raw_injection_dispatches\":" << r.injection << ",\"raw_injection_gpu_seconds\":" << r.injectionSeconds
      << ",\"mix_dispatches\":" << r.mix << ",\"mix_gpu_seconds\":" << r.mixSeconds
      << ",\"prepare_dispatches\":" << r.prepare << ",\"whole_prepare_gpu_seconds\":" << r.prepareSeconds
      << ",\"query_normalization_cost_isolated\":false,\"query_normalization_gpu_seconds_upper_bound\":" << r.prepareSeconds
      << ",\"pool_dispatches\":" << r.pool << ",\"pool_gpu_seconds\":" << r.poolSeconds
      << ",\"other_dispatches\":" << r.other << ",\"other_gpu_seconds\":" << r.otherSeconds
      << ",\"sum_dispatch_gpu_seconds\":" << sum << ",\"full_command_gpu_seconds\":" << p.timing.gpuSeconds
      << ",\"command_minus_dispatch_seconds\":" << p.timing.gpuSeconds-sum
      << ",\"sum_is_not_uninstrumented_elision_saving\":true}";
}
void validTiming(const metal::CommandTiming &t) {
  require(std::isfinite(t.gpuSeconds) && t.gpuSeconds > 0 && std::isfinite(t.wallSeconds) && t.wallSeconds > 0,
      "teacher profile invalid command timing");
}
struct Sequence {
  bool instrumented = false;
  uint32_t pair = 0, position = 0;
  double gpu = 0, call = 0, commandWall = 0;
  std::array<metal::CommandTiming,2> timings;
  std::vector<Profile> profiles;
};
void cpuSelfTest() {
  Dispatch d; d.pipelineName="flash_dense_cache_m32_n128";
  d.threadgroups={96,60,1};d.threadsPerThreadgroup={128,1,1};
  d.bindings={{0,uint64_t{profileRows}*2560*2,false},{1,qWeightBytes,false},
      {2,uint64_t{profileRows}*12288*2,false},{3,4,false}};
  require(isQProjection(d),"CPU q role not selected");validateBulkQ(d);
  auto bad=d;bad.bindings[2].sizeBytes-=2;bool rejected=false;
  try{validateBulkQ(bad);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"CPU malformed q role not rejected");
  bad=d;bad.bindings.push_back(d.bindings[0]);rejected=false;
  try{validateBulkQ(bad);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"CPU duplicate role binding not rejected");
  std::cout << "teacher-profile role metadata CPU checks passed\n";
}
}

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc==2 && std::string_view(argv[1])=="--help") {
        std::cout << "usage: teacher-bulk-profile METALLIB PACKAGE EXACT_2048_TOKENS_JSON FRESH_REPORT_JSON\n"
            "Root-only GPU observation, original bulk1920+127 math. No elision or Worker.\n"
            "TEACHER_BULK_PROFILE_MODE=auto|dispatch|stage (default auto), PAIRS=10 (even2..20).\n";
        return 0;
      }
      if(argc==2 && std::string_view(argv[1])=="--cpu-self-test") {cpuSelfTest();return 0;}
      require(argc==5,"invalid teacher bulk profile arguments");
      const bool bulkFlag=teacher_bulk_sep21::parse(std::getenv(teacher_bulk_sep21::flag));
      teacher_bulk_sep21::validate(bulkFlag,{enabled("SPLASH_FLASH_MTP"),enabled("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY"),
          enabled("SPLASH_FLASH_DENSE_CACHE"),enabled("SPLASH_FLASH_MTP_QSA_F32"),enabled("SPLASH_FLASH_MTP_QSA_MPP")});
      require(bulkFlag,"teacher profile requires qualified bulk flag");
      const uint32_t pairs=integer("TEACHER_BULK_PROFILE_PAIRS",10,2,20);
      require(pairs%2==0,"teacher profile pairs must balance even AB/BA positions");
      const char *rawMode=std::getenv("TEACHER_BULK_PROFILE_MODE");const std::string requested=rawMode?rawMode:"auto";
      require(requested=="auto" || requested=="dispatch" || requested=="stage","invalid teacher profile mode");
      require(!std::filesystem::exists(argv[4]) && !std::filesystem::exists(std::string(argv[4])+".trace.jsonl"),
          "teacher profile outputs must be fresh");
      const auto tokens=loadTokens(argv[3]);require(tokens.size()==2048,"teacher profile requires true2048 token fixture");
      metal::MetalBackend backend(argv[1]);
      const auto capability=backend.commandDispatchProfilingCapability();
      Mode profileMode=requested=="stage"?Mode::StagePerDispatch:Mode::DispatchBoundary;
      if(requested=="auto" && !capability.supports(profileMode) && capability.supports(Mode::StagePerDispatch))profileMode=Mode::StagePerDispatch;
      const bool supported=capability.supports(profileMode);
      const auto weights=FlashWeights::load(backend,argv[2]);
      const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve=std::max<uint64_t>(16ULL<<30,physical/10);
      engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      uint64_t planned=FlashForward::workspacePlannedBytes(8192,2048,4)+FlashForward::requestStateBytes(8192)+
          FlashForward::expertCachePlannedBytes(weights)+FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights)+
          2*FlashMTPForward::workspacePlannedBytes(8192,128)+4*FlashMTPForward::requestStateBytes(8192)+(256ULL<<20)+FlashMTPTeacherBulkForward::plannedBytes;
      if(dense_w8a8_sep21::requiresCache(2048))planned+=dense_w8a8_sep21::Cache::plannedBytes();
      if(enabled("SPLASH_FLASH_DENSE_CACHE"))planned+=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true))+2*FlashMTPForward::denseCachePlannedBytes(weights);
      if(enabled("SPLASH_FLASH_QSA_F32"))planned+=16ULL<<20;
      if(enabled("SPLASH_FLASH_BLOCKED_MOE"))planned+=flashMoEBlockedWorkspacePlannedBytes(2048,10);
      const auto initialAllocation=backend.memoryStats().allocatedBytes;
      auto reservation=governor.tryReserve(planned);require(bool(reservation),"teacher profile governor denied construction");
      FlashForward target(backend,weights,8192,2048,4);
      FlashMTPForward head(backend,weights,8192,128),candidateHead(backend,weights,8192,128);
      FlashMTPTeacherBulkForward bulk(candidateHead);
      require(bulk.workspaceBytes()<=FlashMTPTeacherBulkForward::plannedBytes,"teacher profile bulk arena exceeds plan");
      auto targetState=target.createState();
      auto targetFeatures=backend.allocateBuffer(uint64_t{2048}*profileHyper*2,metal::BufferStorage::Shared,"teacher-oracle-owned-real-target-features");
      auto input=backend.allocateBuffer(uint64_t{2048}*profileHyper*2,metal::BufferStorage::Shared,"teacher-oracle-owned-real-pair-features");
      auto chained=backend.allocateBuffer(uint64_t{4}*profileHyper*2,metal::BufferStorage::Shared,"teacher-oracle-owned-chained-features");
      (void)input;(void)chained;
      auto normalState=candidateHead.createState(),profiledState=candidateHead.createState();
      auto originalState0=head.createState(),originalState1=head.createState();
      (void)originalState0;(void)originalState1;
      metal::ResidencyLease residency;
      if(enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto buffers=target.cachedOperandsOnly();const auto hb=head.cachedOperandsOnly();buffers.insert(buffers.end(),hb.begin(),hb.end());
        const auto cb=candidateHead.cachedOperandsOnly();buffers.insert(buffers.end(),cb.begin(),cb.end());
        if(!buffers.empty())residency=backend.requestWeightResidency(buffers,"teacher oracle saved operands only");
      }
      reservation->commit();
      const auto main=target.forward(targetState,tokens,false,true);validTiming(main.timing);
      require(main.hiddenBF16 && main.hiddenBF16.contents() && main.hiddenBF16.sizeBytes()>=targetFeatures.sizeBytes(),"teacher profile actual target features absent");
      std::memcpy(targetFeatures.contents(),main.hiddenBF16.contents(),targetFeatures.sizeBytes());
      const auto &ip=weights.projection("mtp.layers.0.attn_hyper_connection.block_inject_weight");
      require(ip.bits==5 && ip.groupSize==64 && ip.inputSize==profileHyper && ip.outputSize==4 && ip.experts==1,
          "teacher raw injection original projection metadata changed");
      const InjectionContract injection{ip.weights->buffer.sizeBytes(),ip.scales->buffer.sizeBytes(),ip.biases->buffer.sizeBytes()};
      (void)backend.takeCommandDispatchProfiles();
      const auto sequence=[&](bool profiled) {
        auto &state=profiled?profiledState:normalState;candidateHead.truncate(state,0);
        backend.setCommandDispatchProfiling(profiled?profileMode:Mode::Off);
        Sequence result;result.instrumented=profiled;const auto began=Clock::now();
        for(uint32_t command=0;command<2;++command) {
          const uint32_t begin=command?profileRows:0,count=command?127:profileRows;
          const auto features=backend.view(targetFeatures,uint64_t{begin}*profileHyper*2,uint64_t{count}*profileHyper*2);
          const auto next=std::span<const uint32_t>(tokens).subspan(begin+1,count);
          result.timings[command]=command?candidateHead.primeTeacherCache(state,features,next):bulk.primeTeacherCache(state,features,next);
          validTiming(result.timings[command]);result.gpu+=result.timings[command].gpuSeconds;result.commandWall+=result.timings[command].wallSeconds;
        }
        result.call=std::chrono::duration<double>(Clock::now()-began).count();
        require(state.logicalLength()==2047 && !state.poisoned(),"teacher profile sequence publication differs");
        result.profiles=backend.takeCommandDispatchProfiles();
        require(result.profiles.size()==(profiled?2u:0u),"teacher profile command retention mismatch");
        return result;
      };
      std::array<double,2> warmGPU{};std::array<uint32_t,2> warmSequences{};
      const uint32_t paths=supported?2:1;
      for(uint32_t path=0;path<paths;++path)while(warmGPU[path]<.150 || warmSequences[path]<10) {
        auto s=sequence(path!=0);warmGPU[path]+=s.gpu;++warmSequences[path];
        require(warmSequences[path]<=1000,"teacher profile warmup did not progress");
      }
      std::vector<Sequence> samples;samples.reserve(pairs*paths);
      for(uint32_t pair=0;pair<pairs;++pair)for(uint32_t position=0;position<paths;++position) {
        const bool profiled=paths==2 && (pair+position)%2;
        auto s=sequence(profiled);s.pair=pair;s.position=position;samples.push_back(std::move(s));
      }
      backend.setCommandDispatchProfiling(Mode::Off);
      const auto memory=backend.memoryStats();
      require(memory.peakAllocatedBytes>=initialAllocation && memory.peakAllocatedBytes-initialAllocation<=planned,
          "teacher setup/counter-buffer peak exceeds governor reservation");
      // Validation and I/O happen after every warm/measured sequence is over.
      bool roleCostsValid=supported;
      std::vector<std::array<std::string,2>> roleIssues(samples.size());
      for(size_t sample=0;sample<samples.size();++sample)if(samples[sample].instrumented)
        for(uint32_t command=0;command<2;++command)try {
          require(samples[sample].profiles[command].mode==profileMode,"teacher selected profile mode changed");
          (void)roles(samples[sample].profiles[command],injection,command==0);
        }catch(const std::exception &error) {
          roleCostsValid=false;roleIssues[sample][command]=error.what();
        }
      std::ofstream trace(std::string(argv[4])+".trace.jsonl");require(bool(trace),"cannot write teacher profile trace");
      trace<<std::setprecision(std::numeric_limits<double>::max_digits10);
      std::array<double,2> gpu{},call{},wall{};std::array<uint32_t,2> measured{};
      for(size_t sample=0;sample<samples.size();++sample) {
        const auto &s=samples[sample];
        const uint32_t path=s.instrumented?1:0;gpu[path]+=s.gpu;call[path]+=s.call;wall[path]+=s.commandWall;++measured[path];
        for(uint32_t command=0;command<2;++command) {
          trace<<"{\"pair\":"<<s.pair<<",\"position\":"<<s.position<<",\"instrumented\":"<<(s.instrumented?"true":"false")
              <<",\"logical_begin\":"<<(command?1920:0)<<",\"rows\":"<<(command?127:1920)
              <<",\"sequence_gpu_seconds\":"<<s.gpu<<",\"sequence_api_seconds\":"<<s.call
              <<",\"command_gpu_seconds\":"<<s.timings[command].gpuSeconds<<",\"command_wall_seconds\":"<<s.timings[command].wallSeconds
              <<",\"host_command_subphases\":";writeHost(trace,s.timings[command].host);
          if(s.instrumented) {
            trace<<",\"roles\":";
            if(roleIssues[sample][command].empty())writeRoles(trace,roles(s.profiles[command],injection,command==0),s.profiles[command]);
            else trace<<"null,\"role_costs_unavailable_reason\":"<<json::quote(roleIssues[sample][command]);
            trace<<",\"command\":";profiling::writeJson(trace,s.profiles[command]);
          }
          trace<<"}\n";
        }
      }
      require(bool(trace),"teacher profile trace write failed");trace.close();
      std::ofstream report(argv[4]);require(bool(report),"cannot write teacher profile report");
      report<<std::setprecision(std::numeric_limits<double>::max_digits10)
          <<"{\"schema\":\"splash-current-teacher-bulk-role-profile-v1\",\"execution_complete\":true,\"gpu_executed\":true"
          <<",\"observation_only\":true,\"elision_enabled\":false,\"original_graph_math_preserved\":true,\"qualification_complete\":false"
          <<",\"scope\":\"actual QSA premixer features; original CacheOnlyBulk1920 then original127 tail; excludes Worker/HTTP\""
          <<",\"profile_supported\":"<<(supported?"true":"false")<<",\"requested_profile_mode\":"<<json::quote(requested)
          <<",\"selected_profile_mode\":"<<json::quote(metal::commandDispatchProfilingModeName(profileMode))
          <<",\"capability_reason\":"<<json::quote(capability.reason)
          <<",\"timestamp_counter_set\":"<<(capability.timestampCounterSet?"true":"false")
          <<",\"dispatch_boundary_supported\":"<<(capability.dispatchBoundary?"true":"false")
          <<",\"stage_boundary_supported\":"<<(capability.stageBoundary?"true":"false")
          <<",\"counter_modes_are_diagnostic\":true,\"ordinary_timings_are_uninstrumented\":true"
          <<",\"source_identity\":"<<json::quote(weights.sourceIdentity())<<",\"metallib_sha256\":"<<json::quote(hexadecimal(backend.metallibSha256()))
          <<",\"source_snapshot_parent\":\"build/mtp-teacher-bulk-sep21-v4\",\"parent_source_manifest\":\"build/mtp-teacher-bulk-sep21-v4/source-manifest.json\""
          <<",\"new_source_hashes_computed\":false,\"payload_hashes_computed\":false,\"owned_capture_bytes\":"<<targetFeatures.sizeBytes()
          <<",\"model_workspace_allocations_unchanged\":true,\"bulk_workspace_planned_bytes\":"<<FlashMTPTeacherBulkForward::plannedBytes
          <<",\"bulk_workspace_actual_bytes\":"<<bulk.workspaceBytes()<<",\"reserved_growth_bytes\":"<<planned
          <<",\"initial_allocated_bytes\":"<<initialAllocation<<",\"final_allocated_bytes\":"<<memory.allocatedBytes
          <<",\"peak_allocated_bytes\":"<<memory.peakAllocatedBytes<<",\"peak_growth_bytes\":"<<memory.peakAllocatedBytes-initialAllocation
          <<",\"counter_allocations_included_in_reserved_peak_check\":true,\"normal_warm_sequences\":"<<warmSequences[0]
          <<",\"normal_warm_gpu_seconds\":"<<warmGPU[0]<<",\"profile_warm_sequences\":"<<warmSequences[1]
          <<",\"profile_warm_gpu_seconds\":"<<warmGPU[1]<<",\"balanced_ab_ba_pairs\":"<<(supported?pairs:0)
          <<",\"commands_per_sequence\":2,\"pairs_per_sequence\":2047,\"logical_cache_prefixes_per_sequence\":16"
          <<",\"all_role_timestamps_and_contracts_valid\":"<<(roleCostsValid?"true":"false")
          <<",\"role_costs_unavailable\":"<<(roleCostsValid?"false":"true")
          <<",\"query_normalization_cost_isolated\":false,\"whole_prepare_is_query_cost_upper_bound_only\":true"
          <<",\"sequence_api_timing_excludes_truncate_mode_selection_profile_drain_and_report_validation\":true"
          <<",\"no_cpu_tensor_reads_or_file_writes_between_measured_sequences\":true,\"original_token_writes_and_diag_reads_included\":true"
          <<",\"normal_measured_sequences\":"<<measured[0]<<",\"normal_whole_gpu_ms\":"<<gpu[0]*1000/measured[0]
          <<",\"normal_whole_api_ms\":"<<call[0]*1000/measured[0]<<",\"normal_whole_command_wall_ms\":"<<wall[0]*1000/measured[0];
      if(supported)report<<",\"profile_measured_sequences\":"<<measured[1]<<",\"profile_whole_gpu_ms\":"<<gpu[1]*1000/measured[1]
          <<",\"profile_whole_api_ms\":"<<call[1]*1000/measured[1]<<",\"profile_whole_command_wall_ms\":"<<wall[1]*1000/measured[1]
          <<",\"profile_over_normal_gpu_ratio\":"<<(gpu[1]/measured[1])/(gpu[0]/measured[0])
          <<",\"profile_over_normal_api_ratio\":"<<(call[1]/measured[1])/(call[0]/measured[0]);
      report<<"}\n";require(bool(report),"teacher profile report write failed");
      std::cout<<"teacher bulk observation complete; report="<<argv[4]<<'\n';return 0;
    }catch(const std::exception &error){std::cerr<<"teacher bulk profile failed: "<<error.what()<<'\n';return 1;}
  }
}
