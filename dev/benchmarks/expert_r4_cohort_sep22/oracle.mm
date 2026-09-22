// Root-run one-layer FIRST N16/R4 component screen. CPU entry creates no
// device, reads no model/capture payload and performs no payload hashing.
#define main preserved_qmv_oracle_entry_not_invoked
#include "../prefill4k_allrows_qmv_oracle.mm"
#undef main
#include "flash/FlashGatheredMPP.hpp"
#include "abi.hpp"
#include "quality.hpp"
#include "malformed.hpp"

namespace {
constexpr uint32_t rows=4,routes=40,variants=3,sampledColumns=64;
constexpr uint64_t actElements=uint64_t{routes}*640,downElements=uint64_t{routes}*2560;
constexpr std::array<const char *,variants> names{"nativeBucketMPP","currentGatheredSG4","expertR4CohortN16"};
constexpr FlashExpertR4CohortParams cohortParams{4,10,512,16,10,1,0,0};

struct CohortPlan {
  MetalBuffer counts,offsets,routeMap,inverse,jobs,jobCount,stages;
};
CohortPlan allocatePlan(MetalBackend &backend,std::vector<Guard> &guards) {
  return {guarded(backend,512*4,guards),guarded(backend,513*4,guards),
      guarded(backend,routes*4,guards),guarded(backend,routes*4,guards),
      guarded(backend,routes*sizeof(FlashMoEBucketJob),guards),guarded(backend,4,guards),
      guarded(backend,513*4,guards)};
}
void planGraph(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer hidden,
    MetalBuffer ids,const CohortPlan &plan,MetalBuffer diag) {
  graph.add("expert_r4_cohort_sep22_plan",{hidden,ids,layer.ranks,plan.counts,plan.offsets,
      plan.routeMap,plan.inverse,plan.jobs,plan.jobCount,plan.stages,diag},
      cohortParams,{1,1,1},{256,1,1});
}
void cohortGraph(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer hidden,
    MetalBuffer ids,const CohortPlan &plan,MetalBuffer act,MetalBuffer down,MetalBuffer diag,bool audit=false) {
  planGraph(graph,layer,hidden,ids,plan,diag);
  graph.add(audit?"expert_r4_cohort_sep22_gate_up_n16_audit":"expert_r4_cohort_sep22_gate_up_n16",
      {hidden,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],layer.ranks,ids,
       plan.offsets,plan.routeMap,plan.inverse,plan.jobs,plan.jobCount,plan.stages,act,diag},
      cohortParams,{40,10,1},{128,1,1});
  graph.add(audit?"expert_r4_cohort_sep22_down_n16_audit":"expert_r4_cohort_sep22_down_n16",
      {act,layer.codes[2],layer.scales[2],layer.ranks,ids,plan.offsets,plan.routeMap,plan.inverse,
       plan.jobs,plan.jobCount,plan.stages,down,diag},cohortParams,{160,10,1},{128,1,1});
}
void gatheredGraph(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer hidden,
    MetalBuffer ids,MetalBuffer act,MetalBuffer down,MetalBuffer diag) {
  const FlashGatheredMPPParams p{4,10,512,0};
  graph.add("flash_gathered_mpp_gate_up_m16_n64_sg4",{hidden,layer.codes[0],layer.scales[0],
      layer.codes[1],layer.scales[1],layer.ranks,ids,act,diag},p,{10,4,10},{128,1,1});
  graph.add("flash_gathered_mpp_down_m16_n64_sg4",{act,layer.codes[2],layer.scales[2],
      layer.ranks,ids,down,diag},p,{40,4,10},{128,1,1});
}
uint64_t differing(MetalBuffer a,MetalBuffer b,uint64_t count,uint32_t bytes) {
  const auto *aa=static_cast<const uint8_t *>(a.contents()),*bb=static_cast<const uint8_t *>(b.contents());
  uint64_t mismatches=0;
  for(uint64_t i=0;i<count;++i) mismatches+=std::memcmp(aa+i*bytes,bb+i*bytes,bytes)!=0;
  return mismatches;
}
std::span<const uint16_t> bf16View(MetalBuffer b) {
  return {static_cast<const uint16_t *>(b.contents()),size_t(b.sizeBytes()/2)};
}
void poisonBF16(MetalBuffer b) {std::fill_n(static_cast<uint16_t *>(b.contents()),b.sizeBytes()/2,0x7fc0u);}
void poisonCapture(const Outputs &c) {
  for(auto b:{c.dot,c.scaled}) std::fill_n(static_cast<float *>(b.contents()),b.sizeBytes()/4,
      std::bit_cast<float>(0x7fc00000u));
  poisonBF16(c.bf16);
}
std::vector<uint16_t> hiddenFixture() {
  std::vector<uint16_t> x(rows*2560);
  for(uint32_t row=0;row<rows;++row) {
    double squares=0;
    for(uint32_t k=0;k<2560;++k) {const double v=double(int32_t((k*173+row*31)%211)-105);squares+=v*v;}
    const double normalization=std::sqrt(squares/2560);
    for(uint32_t k=0;k<2560;++k)
      x[row*2560+k]=reference::bf16FromF64(double(int32_t((k*173+row*31)%211)-105)/normalization);
  }
  return x;
}
std::vector<int64_t> caseIDs(std::string_view name) {
  require(name=="u10"||name=="u20"||name=="u30"||name=="u40"||name=="oldmixed"||name=="permuted",
      "unsupported frozen R4 case");
  std::vector<int64_t> ids(routes);
  for(uint32_t row=0;row<rows;++row) {
    const uint32_t shift=name=="u10"?0:name=="u20"?(row/2)*73:
        name=="u30"?(row<2?0:(row-1)*73):name=="u40"?row*73:(row%2?row*73:0);
    for(uint32_t slot=0;slot<10;++slot) ids[row*10+slot]=(slot*53+shift)%512;
  }
  return ids;
}
uint32_t uniqueIDs(const std::vector<int64_t> &ids) {
  std::array<bool,512> seen{};
  for(int64_t id:ids) {require(id>=0&&id<512,"case IDs malformed");seen[size_t(id)]=true;}
  return uint32_t(std::count(seen.begin(),seen.end(),true));
}
std::vector<uint32_t> selectedPairs(uint32_t plane) {
  const uint32_t n=plane==2?2560:640,stride=n/sampledColumns;
  std::vector<uint32_t> pairs;
  for(uint32_t route=0;route<routes;++route) for(uint32_t i=0;i<sampledColumns;++i) {
    pairs.push_back(route);pairs.push_back(i*stride+(route*17+plane*7)%stride);
  }
  return pairs;
}
struct StageReport {
  std::array<uint32_t,6> words{};
  bool reservedZero=false,audit=false;
  bool pass() const {
    return words[0]==cohortParams.epoch&&words[1]==kExpertR4MetaReady&&
        words[2]==(audit?kExpertR4GateCTAs:0u)&&words[3]==(audit?kExpertR4DownCTAs:0u)&&
        words[4]==kExpertR4GateReady&&words[5]==kExpertR4DownReady&&reservedZero;
  }
  void json(std::ostream &out) const {
    out<<"{\"audit\":"<<(audit?"true":"false")<<",\"words\":[";
    for(size_t i=0;i<words.size();++i) {if(i)out<<',';out<<words[i];}
    out<<"],\"reserved_zero\":"<<(reservedZero?"true":"false")<<",\"pass\":"<<(pass()?"true":"false")<<'}';
  }
};
StageReport stages(const CohortPlan &plan,bool audit) {
  StageReport report; report.audit=audit;
  const auto *values=static_cast<const uint32_t *>(plan.stages.contents());
  std::copy_n(values,report.words.size(),report.words.begin());
  report.reservedZero=std::all_of(values+6,values+513,[](uint32_t value){return !value;});
  return report;
}
// Independent complete tiny-metadata checker for the known valid fixture.
bool metadataMatches(const CohortPlan &plan,const std::vector<int64_t> &ids) {
  const auto *counts=static_cast<const uint32_t *>(plan.counts.contents());
  const auto *offsets=static_cast<const uint32_t *>(plan.offsets.contents());
  const auto *map=static_cast<const uint32_t *>(plan.routeMap.contents());
  const auto *inverse=static_cast<const uint32_t *>(plan.inverse.contents());
  const auto *jobs=static_cast<const FlashMoEBucketJob *>(plan.jobs.contents());
  uint32_t cursor=0,job=0;
  for(uint32_t expert=0;expert<512;++expert) {
    uint32_t count=0;
    if(offsets[expert]!=cursor) return false;
    const uint32_t begin=cursor;
    for(uint32_t route=0;route<routes;++route) if(ids[route]==int64_t(expert)) {
      if(map[cursor]!=route||inverse[route]!=cursor) return false;
      ++cursor;++count;
    }
    if(counts[expert]!=count) return false;
    for(uint32_t first=begin;first<cursor;first+=4) {
      if(job>=routes||jobs[job].expert!=expert||jobs[job].row_begin!=first) return false;
      ++job;
    }
  }
  if(cursor!=routes||offsets[512]!=routes||*static_cast<const uint32_t *>(plan.jobCount.contents())!=job) return false;
  for(uint32_t i=job;i<routes;++i) if(jobs[i].expert!=UINT32_MAX||jobs[i].row_begin) return false;
  return true;
}
void denseProbe(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer input,
    MetalBuffer ids,const CohortPlan &plan,const Outputs &capture,uint32_t plane,uint32_t variant,
    const FlashMoEBlockedScratch &scratch) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
  if(variant==0)
    graph.add("flash_qmv_probe_mpp_m16",{plane==2?scratch.packedActivated:scratch.buckets.packedInputs,
        layer.codes[plane],layer.scales[plane],layer.ranks,scratch.buckets.offsets,scratch.buckets.tileJobs,
        scratch.buckets.jobCount,scratch.buckets.routeMap,capture.dot,capture.scaled,capture.bf16,capture.diag},
        FlashQMVProbeParams{4,10,k,n,uint32_t(plane==2),0,512,0},{n/64,40,1},{128,1,1});
  else if(variant==1)
    graph.add("flash_gathered_mpp_projection_probe",{input,layer.codes[plane],layer.scales[plane],layer.ranks,
        ids,capture.dot,capture.scaled,capture.bf16,capture.diag},
        FlashQMVProbeParams{4,10,k,n,uint32_t(plane==2),0,512,0},{n/64,4,10},{128,1,1});
  else
    graph.add("expert_r4_cohort_sep22_projection_probe_n16",{input,layer.codes[plane],layer.scales[plane],
        layer.ranks,ids,plan.offsets,plan.routeMap,plan.inverse,plan.jobs,plan.jobCount,plan.stages,
        capture.dot,capture.scaled,capture.bf16,capture.diag},
        FlashExpertR4CohortProbeParams{cohortParams,k,n,uint32_t(plane==2),0},{n/16,10,1},{128,1,1});
}
struct SampleReport {
  Summary strict;
  uint64_t certifiedFailures=0,diagnosticFailures=0;
  double maxScaledBoundRatio=0;
  bool pass() const {return !certifiedFailures;}
  void json(std::ostream &out) const {
    out<<"{\"sampled_only\":true,\"strict_certificate\":";strict.json(out);
    out<<",\"certified_bound_sign_finite_failures\":"<<certifiedFailures
       <<",\"diagnostic_coverage_failures\":"<<diagnosticFailures
       <<",\"max_scaled_bound_ratio\":"<<finiteJSON(maxScaledBoundRatio)
       <<",\"certified_bound_sign_finite_pass\":"<<(pass()?"true":"false")<<'}';
  }
};
SampleReport sampledReference(const one::OneLayerPayload &layer,uint32_t plane,
    const std::vector<int64_t> &ids,MetalBuffer input,const Outputs &capture,
    const std::vector<uint32_t> &pairs,uint32_t variant,bool compact=false) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
  const auto *x=static_cast<const uint16_t *>(input.contents());
  const auto *codes=static_cast<const int8_t *>(layer.codes[plane].contents());
  const auto *scale=static_cast<const float *>(layer.scales[plane].contents());
  const auto *rank=static_cast<const uint32_t *>(layer.ranks.contents());
  const auto *dot=static_cast<const float *>(capture.dot.contents()),*scaled=static_cast<const float *>(capture.scaled.contents());
  const auto *bf16=static_cast<const uint16_t *>(capture.bf16.contents());
  SampleReport result;
  for(size_t sample=0;sample<pairs.size()/2;++sample) {
    const uint32_t route=pairs[sample*2],column=pairs[sample*2+1];
    const uint64_t coefficient=uint64_t{rank[ids[route]]}*n+column,index=compact?sample:uint64_t{route}*n+column;
    const std::span<const uint16_t> a{x+uint64_t{plane==2?route:route/10}*k,k};
    const std::span<const int8_t> w{codes+coefficient*k,k};
    const auto ref=variant==2?expert_r4_quality::vectorReference(a,w,scale[coefficient],32):
        variant==3?expert_r4_quality::scalarReference(a,w,scale[coefficient]):
        reference::reference(a,w,scale[coefficient],32,true);
    const auto report=reference::assess(ref,dot[index],scaled[index],bf16[index],diagnostic(capture.diag));
    auto &s=result.strict;++s.dots;s.boundFailures+=!report.dotWithinBound||!report.scaledWithinBound;
    s.strictSensitive+=report.strictSensitive;s.strictFailures+=report.strictSensitive&&!report.strictBF16Pass;
    s.signFailures+=!report.signMatches;s.exceptional+=ref.exceptional;s.bf16ReferenceMismatches+=!report.bf16Exact;
    s.maxBF16ULPs=std::max(s.maxBF16ULPs,report.bf16ULPs);
    if(ref.dotAbsoluteBound)s.maxBoundRatio=std::max(s.maxBoundRatio,report.dotAbsoluteError/ref.dotAbsoluteBound);
    if(ref.scaledAbsoluteBound)result.maxScaledBoundRatio=std::max(result.maxScaledBoundRatio,report.scaledAbsoluteError/ref.scaledAbsoluteBound);
    const bool certified=!ref.exceptional&&report.finiteCapture&&report.dotWithinBound&&report.scaledWithinBound&&
        report.signMatches&&report.negativeZeroMatches&&report.diagnosticsCoverExpected;
    result.certifiedFailures+=!certified;result.diagnosticFailures+=!report.diagnosticsCoverExpected;
    if((!certified||(report.strictSensitive&&!report.strictBF16Pass))&&s.failures.size()<16) {
      std::ostringstream failure;failure<<std::setprecision(17)<<"{\"route\":"<<route<<",\"column\":"<<column
        <<",\"reference_dot\":"<<finiteJSON(ref.dot)<<",\"observed_dot\":"<<finiteJSON(dot[index])
        <<",\"dot_abs_bound\":"<<finiteJSON(ref.dotAbsoluteBound)<<",\"reference_scaled\":"<<finiteJSON(ref.scaled)
        <<",\"observed_scaled\":"<<finiteJSON(scaled[index])<<",\"scaled_abs_bound\":"<<finiteJSON(ref.scaledAbsoluteBound)
        <<",\"reference_bf16\":"<<ref.bf16<<",\"observed_bf16\":"<<bf16[index]
        <<",\"strict_sensitive\":"<<(report.strictSensitive?"true":"false")
        <<",\"strict_sensitive_bf16_pass\":"<<(report.strictBF16Pass?"true":"false")
        <<",\"certified_bound_sign_finite_pass\":"<<(certified?"true":"false")<<'}';
      s.failures.push_back(failure.str());
    }
  }
  return result;
}
void cpuSelfTest() {
  cpuTest();require(expert_r4_quality::cpuSelfTest(),"balanced char4 certificate CPU self-test failed");
  require(sizeof(FlashExpertR4CohortParams)==32&&sizeof(FlashExpertR4CohortProbeParams)==48&&
      sizeof(FlashMoEBucketJob)==8,"cohort/probe/job ABI differs");
  require(512*4+513*4+40*4+40*4+40*sizeof(FlashMoEBucketJob)+4+513*4==kExpertR4MetadataBytes,
      "metadata6796-byte bound differs");
  for(std::string_view name:{"u10","u20","u30","u40","oldmixed","permuted"}) {
    const auto ids=caseIDs(name);require(gathered_i8_qmv::fixtureIDDiagnostics(ids,4)==0,"case has invalid or duplicate per-row IDs");
    const uint32_t expected=name=="u10"?10:name=="u20"?20:name=="u40"?40:30;
    require(uniqueIDs(ids)==expected,"frozen case distinct-expert count differs");
  }
  const auto old=caseIDs("oldmixed");require(old[19]==38&&old[20]==0&&old[30]==219,"oldmixed fidelity fixture differs");
  require(hiddenFixture().size()==10240,"normalized R4 fixture extent differs");
  for(uint32_t plane=0;plane<3;++plane) {const auto pairs=selectedPairs(plane);require(pairs.size()==routes*64*2,"sample extent differs");
    for(size_t i=0;i<pairs.size()/2;++i)require(pairs[i*2]<routes&&pairs[i*2+1]<(plane==2?2560u:640u),"sample coordinates invalid");}
  std::cout<<"{\"expert_r4_cohort_cpu_checks\":\"passed\",\"rows\":4,\"cases\":6,\"metadata_bytes\":6796,"
      "\"gpu_work\":false,\"payload_reads\":false,\"hashing_performed\":false,\"quality_thresholds_unchanged\":true}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test") {cpuSelfTest();return 0;}
    require(argc>=7&&std::string_view(argv[1])=="--gpu",
      "usage: cohort-oracle --gpu metallib Full512Store layer 4 NEWreport [--case oldmixed|u10|u20|u30|u40|permuted] [--rank-shift0..511] [--pairs1..60]");
    const uint32_t layerIndex=numeric(argv[4],0,47);require(numeric(argv[5],4,4)==4,"R4 only");
    const std::filesystem::path reportPath(argv[6]);require(!std::filesystem::exists(reportPath),"NEW report path required");
    std::string caseName="oldmixed";uint32_t rankShift=0,pairs=18;bool explicitRank=false;
    for(int i=7;i<argc;i+=2) {require(i+1<argc,"missing CLI option value");const std::string option(argv[i]);
      if(option=="--case")caseName=argv[i+1];else if(option=="--rank-shift"){rankShift=numeric(argv[i+1],0,511);explicitRank=true;}
      else if(option=="--pairs")pairs=numeric(argv[i+1],1,60);else throw std::invalid_argument("unknown cohort oracle option");}
    if(caseName=="permuted"&&!explicitRank)rankShift=73;
    const uint32_t requestedPairs=pairs;pairs=(pairs+2)/3*3;
    const auto hidden=hiddenFixture();const auto ids=caseIDs(caseName);
    const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],
      "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
      "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
    require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","Full512 certificate differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1);::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);
    @autoreleasepool {
      MetalBackend backend(argv[2]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=4ULL<<30;
      require(physical>reserve,"host reserve unavailable");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(128ULL<<20);
      auto reservation=governor.tryReserve(planned);require(bool(reservation),"one-layer cohort reservation denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      if(rankShift){auto *rank=static_cast<uint32_t *>(layer.ranks.contents());for(uint32_t id=0;id<512;++id)rank[id]=(id+rankShift)%512;}
      const auto immutable=layer.immutableHashes();std::vector<Guard> guards;
      auto x=upload(backend,hidden,guards),idBuffer=upload(backend,ids,guards);
      auto scratch=allocateMoEBlockedScratch(backend,4,10);guardScratch(backend,scratch,guards);
      const auto plan=allocatePlan(backend,guards);std::array<MetalBuffer,variants> act,down,diag;
      const splash::flash::expert_r4_cohort_sep22_safety::Plan safetyPlan{
          plan.counts,plan.offsets,plan.routeMap,plan.inverse,plan.jobs,plan.jobCount,plan.stages};
      std::array<CommandGraph,variants> chain;
      for(uint32_t v=0;v<variants;++v) {act[v]=guarded(backend,actElements*2,guards);diag[v]=guarded(backend,4,guards);
        down[v]=v?guarded(backend,downElements*2,guards):scratch.scatteredDown;poisonBF16(act[v]);poisonBF16(down[v]);clear(diag[v]);
        if(v==0)mppChain(chain[v],layer,x,idBuffer,scratch,diag[v],4);
        else if(v==1)gatheredGraph(chain[v],layer,x,idBuffer,act[v],down[v],diag[v]);
        else {splash::flash::expert_r4_cohort_sep22_safety::validate(layer,safetyPlan,x,idBuffer,act[v],down[v],diag[v]);
          cohortGraph(chain[v],layer,x,idBuffer,plan,act[v],down[v],diag[v]);}
        (void)backend.submitCommand(chain[v].dispatches());}
      const StageReport shippingStage=stages(plan,false);const bool metadataPass=metadataMatches(plan,ids);
      std::array<uint32_t,variants> initialDiagnostics;for(uint32_t v=0;v<variants;++v)initialDiagnostics[v]=diagnostic(diag[v]);
      CommandGraph unpack;unpack.add("flash_qmv_probe_unpack_activation",{scratch.packedActivated,scratch.buckets.canonicalToPacked,act[0],diag[0]},
          FlashQMVProbeParams{4,10,2560,640,0,0,512,0},{3,40,1},{256,1,1});(void)backend.submitCommand(unpack.dispatches());
      auto auditAct=guarded(backend,actElements*2,guards),auditDown=guarded(backend,downElements*2,guards),auditDiag=guarded(backend,4,guards);
      splash::flash::expert_r4_cohort_sep22_safety::validate(layer,safetyPlan,x,idBuffer,auditAct,auditDown,auditDiag);
      poisonBF16(auditAct);poisonBF16(auditDown);clear(auditDiag);CommandGraph auditChain;
      cohortGraph(auditChain,layer,x,idBuffer,plan,auditAct,auditDown,auditDiag,true);(void)backend.submitCommand(auditChain.dispatches());
      const StageReport auditStage=stages(plan,true);
      const uint64_t auditActMismatch=differing(act[2],auditAct,actElements,2),auditDownMismatch=differing(down[2],auditDown,downElements,2);
      const bool auditPairPass=auditStage.pass()&&!auditActMismatch&&!auditDownMismatch&&diagnostic(auditDiag)==0x80000000u;
      std::array<std::array<Outputs,variants>,3> caps;
      std::array<std::array<SampleReport,variants>,3> f64;
      std::array<SampleReport,3> scalarF64;
      std::array<std::array<expert_r4_quality::MetricReport,variants>,3> projectionQuality;
      std::array<std::array<std::array<uint64_t,3>,variants>,3> differences{},scalarDifferences{};
      std::array<bool,variants> primitive;primitive.fill(true);std::array<bool,variants> strict;strict.fill(true);
      for(uint32_t plane=0;plane<3;++plane) {const uint32_t n=plane==2?2560:640,k=plane==2?640:2560;
        const auto input=plane==2?act[1]:x;const auto pairsForPlane=selectedPairs(plane);
        // Native bucket projection uses its own sanitized packed activation.
        // For a common down operand, rebuild only its packed A from SG4 output.
        if(plane==2) {
          const auto *inverse=static_cast<const uint32_t *>(scratch.buckets.canonicalToPacked.contents());
          const auto *canonical=static_cast<const uint16_t *>(act[1].contents());
          auto *packed=static_cast<uint16_t *>(scratch.packedActivated.contents());
          for(uint32_t route=0;route<routes;++route)std::memcpy(packed+uint64_t{inverse[route]}*640,canonical+uint64_t{route}*640,640*2);
        }
        for(uint32_t v=0;v<variants;++v) {auto &capture=caps[plane][v];capture=outputs(backend,uint64_t{routes}*n,guards);
          poisonCapture(capture);clear(capture.diag);CommandGraph probe;denseProbe(probe,layer,input,idBuffer,plan,capture,plane,v,scratch);
          (void)backend.submitCommand(probe.dispatches());f64[plane][v]=sampledReference(layer,plane,ids,input,capture,pairsForPlane,v);
          primitive[v]&=f64[plane][v].pass()&&diagnostic(capture.diag)==0x80000000u;strict[v]&=f64[plane][v].strict.pass();}
        for(uint32_t v=0;v<variants;++v) {projectionQuality[plane][v]=expert_r4_quality::metrics(bf16View(caps[plane][1].bf16),bf16View(caps[plane][v].bf16),n);
          const uint64_t count=uint64_t{routes}*n;differences[plane][v]={differing(caps[plane][1].dot,caps[plane][v].dot,count,4),
            differing(caps[plane][1].scaled,caps[plane][v].scaled,count,4),differing(caps[plane][1].bf16,caps[plane][v].bf16,count,2)};}
        const uint32_t sampleCount=uint32_t(pairsForPlane.size()/2);auto scalar=outputs(backend,sampleCount,guards);poisonCapture(scalar);clear(scalar.diag);
        auto selected=upload(backend,pairsForPlane,guards),p=upload(backend,std::vector<FlashQMVProbeParams>{{4,10,k,n,uint32_t(plane==2),4,512,0}},guards);
        auto count=upload(backend,std::vector<uint32_t>{sampleCount},guards);CommandGraph probe;
        probe.add("expert_r4_cohort_sep22_scalar_projection_samples",{input,layer.codes[plane],layer.scales[plane],layer.ranks,idBuffer,
            scalar.dot,scalar.scaled,scalar.bf16,scalar.diag,p,selected,count},{(sampleCount+255)/256,1,1},{256,1,1});
        (void)backend.submitCommand(probe.dispatches());scalarF64[plane]=sampledReference(layer,plane,ids,input,scalar,pairsForPlane,3,true);
        for(uint32_t v=0;v<variants;++v)for(uint32_t i=0;i<sampleCount;++i) {const uint64_t at=uint64_t{pairsForPlane[i*2]}*n+pairsForPlane[i*2+1];
          scalarDifferences[plane][v][0]+=static_cast<const uint32_t *>(scalar.dot.contents())[i]!=static_cast<const uint32_t *>(caps[plane][v].dot.contents())[at];
          scalarDifferences[plane][v][1]+=static_cast<const uint32_t *>(scalar.scaled.contents())[i]!=static_cast<const uint32_t *>(caps[plane][v].scaled.contents())[at];
          scalarDifferences[plane][v][2]+=static_cast<const uint16_t *>(scalar.bf16.contents())[i]!=static_cast<const uint16_t *>(caps[plane][v].bf16.contents())[at];}
      }
      std::array<expert_r4_quality::MetricReport,variants> activationQuality,downQuality;
      std::array<SampleReport,variants> ownDownF64;
      std::array<uint64_t,variants> swigluMismatch{},ownDownMismatch{};
      std::array<bool,variants> qualified;qualified.fill(true);
      for(uint32_t v=0;v<variants;++v) {activationQuality[v]=expert_r4_quality::metrics(bf16View(act[1]),bf16View(act[v]),640);
        downQuality[v]=expert_r4_quality::metrics(bf16View(down[1]),bf16View(down[v]),2560);
        auto expected=guarded(backend,actElements*2,guards);poisonBF16(expected);clear(diag[v]);CommandGraph swiglu;
        addSiLUMultiply(swiglu,caps[0][v].bf16,caps[1][v].bf16,expected,diag[v],4,640,10);(void)backend.submitCommand(swiglu.dispatches());
        swigluMismatch[v]=differing(act[v],expected,actElements,2);qualified[v]&=!swigluMismatch[v]&&diagnostic(diag[v])==0x80000000u;
        auto own=outputs(backend,downElements,guards);poisonCapture(own);clear(own.diag);
        if(v==0) {const auto *inverse=static_cast<const uint32_t *>(scratch.buckets.canonicalToPacked.contents());
          for(uint32_t route=0;route<routes;++route)std::memcpy(static_cast<uint16_t *>(scratch.packedActivated.contents())+uint64_t{inverse[route]}*640,
              static_cast<const uint16_t *>(act[v].contents())+uint64_t{route}*640,640*2);}
        CommandGraph ownProbe;denseProbe(ownProbe,layer,act[v],idBuffer,plan,own,2,v,scratch);(void)backend.submitCommand(ownProbe.dispatches());
        ownDownF64[v]=sampledReference(layer,2,ids,act[v],own,selectedPairs(2),v);
        strict[v]&=ownDownF64[v].strict.pass();
        ownDownMismatch[v]=differing(down[v],own.bf16,downElements,2);
        qualified[v]&=!ownDownMismatch[v]&&ownDownF64[v].pass()&&diagnostic(own.diag)==0x80000000u;
      }
      std::array<std::vector<uint8_t>,variants> savedAct,savedDown;
      for(uint32_t v=0;v<variants;++v) {savedAct[v].resize(act[v].sizeBytes());savedDown[v].resize(down[v].sizeBytes());
        std::memcpy(savedAct[v].data(),act[v].contents(),savedAct[v].size());std::memcpy(savedDown[v].data(),down[v].contents(),savedDown[v].size());}
      for(auto b:{scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.canonicalToPacked,
          scratch.buckets.packedInputs,scratch.buckets.jobOffsets,scratch.buckets.jobCount,scratch.buckets.tileJobs,
          scratch.packedActivated,scratch.scatteredDown,plan.counts,plan.offsets,plan.routeMap,plan.inverse,plan.jobs,plan.jobCount,plan.stages})
        std::memset(b.contents(),0xa5,b.sizeBytes());
      std::array<bool,variants> replay;
      for(uint32_t v=0;v<variants;++v) {poisonBF16(act[v]);poisonBF16(down[v]);clear(diag[v]);(void)backend.submitCommand(chain[v].dispatches());
        if(v==0)(void)backend.submitCommand(unpack.dispatches());
        replay[v]=!std::memcmp(savedAct[v].data(),act[v].contents(),savedAct[v].size())&&!std::memcmp(savedDown[v].data(),down[v].contents(),savedDown[v].size())&&diagnostic(diag[v])==0x80000000u;}
      const StageReport replayStage=stages(plan,false);
      const auto safety=splash::flash::expert_r4_cohort_sep22_safety::run(backend,layer,hidden,ids,
          [&](uint64_t bytes){return guarded(backend,bytes,guards);});
      const uint64_t after=backend.memoryStats().allocatedBytes;require(after>=before&&after-before<=planned,"cohort oracle exceeded one layer plus128MiB admission");reservation->commit();
      bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      bool inputsImmutable=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      // Payload verification, references, copied comparisons, audit and safety
      // checks finish before warmup and are never part of a timed shipping loop.
      layer.checkImmutableHashes(immutable);
      bool allPrequalified=true;
      for(uint32_t v=0;v<variants;++v) {qualified[v]&=primitive[v]&&primitive[1]&&canaries&&inputsImmutable&&replay[v]&&
          initialDiagnostics[v]==0x80000000u&&activationQuality[v].pass()&&downQuality[v].pass();
        for(uint32_t plane=0;plane<3;++plane)qualified[v]&=projectionQuality[plane][v].pass();
        if(v==2)qualified[v]&=shippingStage.pass()&&metadataPass&&auditPairPass&&replayStage.pass()&&safety.pass();
        allPrequalified&=qualified[v];}
      std::array<double,variants> warmGPU{};std::array<uint32_t,variants> warmRuns{};
      std::array<std::vector<CommandTiming>,variants> times;
      const bool timingAttempted=allPrequalified;
      if(timingAttempted) {
        for(auto d:diag)clear(d); // Once outside both loops; sticky bits survive all replays.
        uint32_t cycle=0;
        while(std::any_of(warmGPU.begin(),warmGPU.end(),[](double s){return s<0.150;})) {
          require(cycle<20000,"GPU timestamps did not accumulate150ms within bounded warm replays");
          for(uint32_t step=0;step<variants;++step) {const uint32_t v=(cycle+step)%variants;
            if(warmGPU[v]<0.150) {const auto t=backend.submitCommand(chain[v].dispatches());warmGPU[v]+=t.gpuSeconds;++warmRuns[v];}}
          ++cycle;
        }
        for(uint32_t sample=0;sample<pairs;++sample)for(uint32_t step=0;step<variants;++step) {
          const uint32_t v=(sample+step)%variants;times[v].push_back(backend.submitCommand(chain[v].dispatches()));}
      }
      // Only now inspect buffers again. Native canonical activation needs an
      // untimed unpack; shipping commands include their original work only.
      if(timingAttempted)(void)backend.submitCommand(unpack.dispatches());
      const StageReport finalStage=stages(plan,false);
      canaries&=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      inputsImmutable&=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      bool timingPass=timingAttempted;
      for(uint32_t v=0;v<variants;++v) {replay[v]&=!std::memcmp(savedAct[v].data(),act[v].contents(),savedAct[v].size())&&
          !std::memcmp(savedDown[v].data(),down[v].contents(),savedDown[v].size());
        qualified[v]&=canaries&&inputsImmutable&&replay[v]&&diagnostic(diag[v])==0x80000000u;
        if(v==2)qualified[v]&=finalStage.pass();timingPass&=qualified[v];}
      std::ofstream out(reportPath);require(bool(out),"cannot create NEW component report");out<<std::setprecision(17);
      out<<"{\"schema\":\"splash-private-expert-r4-cohort-n16-component-v1\",\"layer\":"<<layerIndex<<",\"rows\":4"
         <<",\"case\":"<<splash::json::quote(caseName)<<",\"distinct_experts\":"<<uniqueIDs(ids)<<",\"rank_shift\":"<<rankShift
         <<",\"hidden_policy\":\"frozen synthetic normalized BF16, row-distinct; no model capture input\",\"one_shared_readonly_layer\":true"
         <<",\"mapped_layer_bytes\":"<<metadata.layers[layerIndex].bytes<<",\"metadata_bytes\":6796,\"scratch_allowance_bytes\":"<<(128ULL<<20)
         <<",\"allocated_delta_bytes\":"<<after-before<<",\"canaries\":"<<(canaries?"true":"false")<<",\"inputs_immutable\":"<<(inputsImmutable?"true":"false")
         <<",\"weights_immutable_pre_timing\":true,\"model_quality_qualified\":false,\"verification_state_qualified\":false"
         <<",\"all_pre_timing_semantic_checks_pass\":"<<(allPrequalified?"true":"false")<<",\"timing_attempted\":"<<(timingAttempted?"true":"false")
         <<",\"timing_pass\":"<<(timingPass?"true":"false")<<",\"requested_samples\":"<<requestedPairs<<",\"matched_samples\":"<<pairs
         <<",\"metadata_matches_fixture\":"<<(metadataPass?"true":"false")<<",\"shipping_stages\":";shippingStage.json(out);
      out<<",\"audit_stages\":";auditStage.json(out);out<<",\"audit_shipping_activated_mismatches\":"<<auditActMismatch
         <<",\"audit_shipping_down_mismatches\":"<<auditDownMismatch<<",\"audit_shipping_exact_pass\":"<<(auditPairPass?"true":"false")
         <<",\"replay_stages\":";replayStage.json(out);out<<",\"final_stages\":";finalStage.json(out);
      out<<",\"safety\":"<<safety.json()
         <<",\"sampled_f64_policy\":\"64 fixed columns per everyroute, chosen before payload inspection; independent compensated F64 BF16xI8. New balancedchar4→singlelaneAccumulator→descendingXOR32 has D=ceil(K/128)+2+5 (27/12), N=K+31. Generic u32=2^-23, 4lambda peraddition, conservative F64positive uncertainty/upward envelope, rawoperand/scaledresult FTZ allowances, F64lateproduct uncertainty and conservative nonoverflow risk exclusions. Native bucket/currentSG4 opaque frozen references remain unchanged. Scalar sequentialK contrast independent of new association. No exhaustive per-dot claim.\""
         <<",\"stage_policy\":\"Shipping enforces exact full grids in its kernel admission and uses inherited serial dispatch ordering, single-writer readyflags and no global counter atomics. Untimed audit uses the same arithmetic and verifies400/1600 completedCTAs; exact audit/shipping BF16act/down pairing is required before timing. Partial shipping grids are rejected. Literal compiled SiLUMultiply and eachvariant ownactivationDown taps must bit-match; ownactivationDown raw/scaled/BF16 sampled F64 certificates are also required.\""
         <<",\"quality_policy\":\"Current gatheredSG4 baseline; identical normalized H/IDs/rankmap and common SG4 activation for dense down projections; actual own activation for shipping fullchain down. Aggregate and perroute relativeL2≤1e-4/cosine≥.999999 for projected gate/up/down and activated/fullchainDown; thresholds unchanged. Strict sensitive-dot failures remain independently failed reports.\""
         <<",\"timing_policy\":\"All semantic/control/audit/safety/reference/hash assertions before shippingwarm. Eachcontrol/newvariant accumulates≥150msGPUwarm. Three-way rotated samples rounded to multiples3 (default18) balance everyvariant's position; only realshippingchains timed. No CPUbuffer/model/diag touches inside warm or samples, no resets between replays.\",\"variants\":[";
      for(uint32_t v=0;v<variants;++v) {if(v)out<<',';out<<"{\"name\":"<<splash::json::quote(names[v])
        <<",\"numerical_alternative_primitive_qualified\":"<<(qualified[v]?"true":"false")<<",\"sampled_strict_certificate_pass\":"<<(strict[v]?"true":"false")
        <<",\"sampled_certified_bounds_pass\":"<<(primitive[v]?"true":"false")<<",\"initial_sticky\":"<<initialDiagnostics[v]
        <<",\"poisoned_and_post_timing_replay_exact\":"<<(replay[v]?"true":"false")<<",\"compiled_swiglu_mismatches\":"<<swigluMismatch[v]
        <<",\"own_activation_down_tap_mismatches\":"<<ownDownMismatch[v]<<",\"warm_gpu_ms\":"<<warmGPU[v]*1000<<",\"warm_replays\":"<<warmRuns[v]
        <<",\"own_activation_down_sampled_f64\":";ownDownF64[v].json(out);out
        <<",\"chain_activated_quality\":";activationQuality[v].json(out);out<<",\"chain_down_quality\":";downQuality[v].json(out);
        out<<",\"timings\":[";for(size_t i=0;i<times[v].size();++i){if(i)out<<',';out<<"{\"gpu_ms\":"<<times[v][i].gpuSeconds*1000<<",\"wall_ms\":"<<times[v][i].wallSeconds*1000<<'}';}
        out<<"],\"projections\":[";
        for(uint32_t plane=0;plane<3;++plane) {if(plane)out<<',';const auto &d=differences[plane][v],&s=scalarDifferences[plane][v];
          out<<"{\"plane\":"<<plane<<",\"bf16_quality\":";projectionQuality[plane][v].json(out);
          out<<",\"current_sg4_mismatches\":{\"raw\":"<<d[0]<<",\"scaled\":"<<d[1]<<",\"bf16\":"<<d[2]<<'}'
             <<",\"scalar_sample_mismatches\":{\"raw\":"<<s[0]<<",\"scaled\":"<<s[1]<<",\"bf16\":"<<s[2]<<'}'
             <<",\"sampled_f64\":";f64[plane][v].json(out);out<<'}';}out<<"]}";}
      out<<"],\"scalar_sample_certificates\":[";for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';scalarF64[plane].json(out);}out<<"]}\n";
      backend.stop();std::cout<<"{\"all_pre_timing_semantic_checks_pass\":"<<(allPrequalified?"true":"false")
          <<",\"timing_pass\":"<<(timingPass?"true":"false")<<",\"report\":"<<splash::json::quote(reportPath.string())<<"}\n";
      return timingPass?0:2;
    }
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
