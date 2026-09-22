// Root-run compact INTEGER setup feeding unchanged native M16 producers.
// CPU entry creates no device and reads/hashes no model/capture payload.
#define main preserved_qmv_oracle_entry_not_invoked
#include "../prefill4k_allrows_qmv_oracle.mm"
#undef main
#include "flash/FlashGatheredMPP.hpp"
#include "abi.hpp"
#include "metadata.hpp"
#include "quality.hpp" // copied unchanged; only existing BF16 norm metrics used
#include "safety.hpp"

namespace {
constexpr uint32_t rows=4,routes=40,variants=3,jobSlots=514,operandRows=103;
constexpr uint64_t actElements=uint64_t{routes}*640,downElements=uint64_t{routes}*2560;
constexpr std::array<const char *,variants> names{"oldNativeM16","compactNativeM16","currentGatheredSG4"};
constexpr FlashMoEBucketParams compactParams{4,10,2560,512,40,16,514,0};
constexpr FlashMoEBucketParams packParams{4,10,2560,512,40,0,0,0};

std::vector<uint16_t> normalizedHidden() {
  std::vector<uint16_t> values(rows*2560);
  for(uint32_t row=0;row<rows;++row) {
    double squares=0;for(uint32_t k=0;k<2560;++k){const double x=double(int32_t((k*173+row*31)%211)-105);squares+=x*x;}
    const double normalization=std::sqrt(squares/2560);
    for(uint32_t k=0;k<2560;++k)values[row*2560+k]=reference::bf16FromF64(double(int32_t((k*173+row*31)%211)-105)/normalization);
  }
  return values;
}
std::vector<int64_t> fixtureIDs(std::string_view name) {
  require(name=="u10"||name=="u20"||name=="u30"||name=="u40"||name=="oldmixed"||name=="permuted","unsupported compact native case");
  std::vector<int64_t> ids(routes);
  for(uint32_t row=0;row<rows;++row) {
    const uint32_t shift=name=="u10"?0:name=="u20"?(row/2)*73:name=="u30"?(row<2?0:(row-1)*73):
        name=="u40"?row*73:(row%2?row*73:0);
    for(uint32_t slot=0;slot<10;++slot){const uint32_t selection=name=="permuted"?(slot*3+row)%10:slot;
      ids[row*10+slot]=(selection*53+shift)%512;}
  }
  return ids;
}
template<class T> std::vector<T> copyBuffer(MetalBuffer b,uint64_t elements=0) {
  if(!elements)elements=b.sizeBytes()/sizeof(T);
  require(elements*sizeof(T)<=b.sizeBytes(),"snapshot byte extent exceeds view");
  std::vector<T> result(elements);std::memcpy(result.data(),b.contents(),elements*sizeof(T));return result;
}
template<class T> bool same(const std::vector<T> &a,const std::vector<T> &b) {
  return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}
std::span<const uint16_t> bf16View(const std::vector<uint16_t> &a){return {a.data(),a.size()};}
void poison(MetalBuffer b){std::memset(b.contents(),0xa5,b.sizeBytes());}
void resetNative(const FlashMoEBlockedScratch &s,MetalBuffer diag) {
  for(auto b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
      s.buckets.packedInputs,s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,
      s.packedActivated,s.scatteredDown})poison(b);
  clear(diag);
}
void setup(CommandGraph &g,MetalBuffer h,MetalBuffer ids,const FlashMoEBlockedScratch &s,MetalBuffer diag,bool compact) {
  if(!compact){addMoEBlockedPack(g,h,ids,s,diag,4,FlashMoEBlockedTile::M16N64,10);return;}
  g.add("expert_r4_compact_native_sep22_plan",{ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,
      s.buckets.canonicalToPacked,s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diag},compactParams,{1,1,1},{256,1,1});
  g.add("flash_moe_direct_a_pack",{h,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diag},
      packParams,{103,1,1},{256,1,1});
}
void gate(CommandGraph &g,const one::OneLayerPayload &layer,const FlashMoEBlockedScratch &s,MetalBuffer diag) {
  const FlashInt8ExpertStoreParams p{4,10,40,514,16,512,0,0};
  g.add("flash_int8_expert_store_gate_up_m16_n64",{s.buckets.packedInputs,layer.codes[0],layer.scales[0],
      layer.codes[1],layer.scales[1],layer.ranks,s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.packedActivated,diag},
      p,{10,40,1},{128,1,1});
}
void downTail(CommandGraph &g,const one::OneLayerPayload &layer,const FlashMoEBlockedScratch &s,MetalBuffer diag,bool includeDown) {
  const FlashMoEBlockedDownParams p{{4,10,640,2560,512,0,0,0,320,uint64_t{2560}*320,20,uint64_t{2560}*20},40,514,16,0};
  g.add("flash_moe_blocked_poison_excluded_routes",{s.buckets.canonicalToPacked,s.scatteredDown,diag},p,{10,40,1},{256,1,1});
  g.add("flash_moe_direct_a_prepare_down",{s.packedActivated,s.buckets.offsets,s.packedActivated,diag},
      FlashMoEDirectAPrepareParams{40,640,63,0},{103,1,1},{256,1,1});
  if(includeDown)g.add("flash_int8_expert_store_down_scatter_m16_n64",{s.packedActivated,layer.codes[2],layer.scales[2],
      layer.ranks,s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.buckets.routeMap,s.scatteredDown,diag},
      FlashInt8ExpertStoreParams{4,10,40,514,16,512,0,0},{40,40,1},{128,1,1});
}
void nativeGraph(CommandGraph &g,const one::OneLayerPayload &layer,MetalBuffer h,MetalBuffer ids,
    const FlashMoEBlockedScratch &s,MetalBuffer diag,bool compact,uint32_t phase) {
  setup(g,h,ids,s,diag,compact);gate(g,layer,s,diag);
  if(phase)downTail(g,layer,s,diag,phase==2);
}
void gatheredGraph(CommandGraph &g,const one::OneLayerPayload &layer,MetalBuffer h,MetalBuffer ids,MetalBuffer act,MetalBuffer down,MetalBuffer diag) {
  const FlashGatheredMPPParams p{4,10,512,0};
  g.add("flash_gathered_mpp_gate_up_m16_n64_sg4",{h,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],layer.ranks,ids,act,diag},p,{10,4,10},{128,1,1});
  g.add("flash_gathered_mpp_down_m16_n64_sg4",{act,layer.codes[2],layer.scales[2],layer.ranks,ids,down,diag},p,{40,4,10},{128,1,1});
}
compact_native_r4_metadata::Expected metadataSnapshot(const FlashMoEBlockedScratch &s) {
  compact_native_r4_metadata::Expected m;
  std::memcpy(m.counts.data(),s.buckets.counts.contents(),sizeof(m.counts));
  std::memcpy(m.offsets.data(),s.buckets.offsets.contents(),sizeof(m.offsets));
  std::memcpy(m.routeMap.data(),s.buckets.routeMap.contents(),sizeof(m.routeMap));
  std::memcpy(m.inverse.data(),s.buckets.canonicalToPacked.contents(),sizeof(m.inverse));
  std::memcpy(m.jobOffsets.data(),s.buckets.jobOffsets.contents(),sizeof(m.jobOffsets));
  std::memcpy(m.jobs.data(),s.buckets.tileJobs.contents(),sizeof(m.jobs));
  m.jobCount=*static_cast<const uint32_t *>(s.buckets.jobCount.contents());return m;
}
bool sameMetadata(const compact_native_r4_metadata::Expected &a,const compact_native_r4_metadata::Expected &b) {
  return a.counts==b.counts&&a.offsets==b.offsets&&a.routeMap==b.routeMap&&a.inverse==b.inverse&&
      a.jobOffsets==b.jobOffsets&&a.jobCount==b.jobCount&&!std::memcmp(a.jobs.data(),b.jobs.data(),sizeof(a.jobs));
}
std::vector<uint16_t> canonicalize(const std::vector<uint16_t> &packed,const compact_native_r4_metadata::Expected &m) {
  std::vector<uint16_t> out(actElements,0x7fc0);
  for(uint32_t route=0;route<routes;++route)if(m.inverse[route]!=UINT32_MAX) {
    require(m.inverse[route]<routes,"canonical observer sees malformed inverse");
    std::copy_n(packed.data()+uint64_t{m.inverse[route]}*640,640,out.data()+uint64_t{route}*640);
  }
  return out;
}
struct Projection {
  std::vector<float> dot,scaled;std::vector<uint16_t> bf16;uint32_t diagnostic=0;
};
Projection capture(const Outputs &c){return {copyBuffer<float>(c.dot),copyBuffer<float>(c.scaled),copyBuffer<uint16_t>(c.bf16),diagnostic(c.diag)};}
void resetCapture(const Outputs &c,uint32_t sticky) {
  const float nan=std::bit_cast<float>(0x7fc00000u);
  std::fill_n(static_cast<float *>(c.dot.contents()),c.dot.sizeBytes()/4,nan);
  std::fill_n(static_cast<float *>(c.scaled.contents()),c.scaled.sizeBytes()/4,nan);
  std::fill_n(static_cast<uint16_t *>(c.bf16.contents()),c.bf16.sizeBytes()/2,0x7fc0u);
  *static_cast<uint32_t *>(c.diag.contents())=sticky;
}
void nativeProbe(CommandGraph &g,const one::OneLayerPayload &layer,const FlashMoEBlockedScratch &s,const Outputs &c,uint32_t plane) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
  g.add("flash_qmv_probe_mpp_m16",{plane==2?s.packedActivated:s.buckets.packedInputs,layer.codes[plane],layer.scales[plane],layer.ranks,
      s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.buckets.routeMap,c.dot,c.scaled,c.bf16,c.diag},
      FlashQMVProbeParams{4,10,k,n,uint32_t(plane==2),0,512,0},{n/64,40,1},{128,1,1});
}
struct SampleReport {
  Summary strict;uint64_t certifiedFailures=0;
  bool pass()const{return !certifiedFailures;}
  void json(std::ostream &o)const{o<<"{\"sampled_only\":true,\"strict_certificate\":";strict.json(o);o
      <<",\"certified_bound_sign_finite_failures\":"<<certifiedFailures<<",\"certified_bound_sign_finite_pass\":"<<(pass()?"true":"false")<<'}';}
};
SampleReport f64(const one::OneLayerPayload &layer,uint32_t plane,const std::vector<int64_t> &ids,
    const std::vector<uint16_t> &input,const Projection &c) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640,stride=n/64;
  const auto *codes=static_cast<const int8_t *>(layer.codes[plane].contents());
  const auto *scales=static_cast<const float *>(layer.scales[plane].contents());
  const auto *ranks=static_cast<const uint32_t *>(layer.ranks.contents());SampleReport result;
  for(uint32_t route=0;route<40;++route)for(uint32_t sample=0;sample<64;++sample) {
    const uint32_t column=sample*stride+(route*17+plane*7)%stride;const uint64_t row=uint64_t{ranks[ids[route]]}*n+column,index=uint64_t{route}*n+column;
    const auto ref=reference::reference({input.data()+uint64_t{plane==2?route:route/10}*k,k},{codes+row*k,k},scales[row],32,true);
    const auto report=reference::assess(ref,c.dot[index],c.scaled[index],c.bf16[index],c.diagnostic);auto &s=result.strict;
    ++s.dots;s.boundFailures+=!report.dotWithinBound||!report.scaledWithinBound;s.strictSensitive+=report.strictSensitive;
    s.strictFailures+=report.strictSensitive&&!report.strictBF16Pass;s.signFailures+=!report.signMatches;s.exceptional+=ref.exceptional;
    s.bf16ReferenceMismatches+=!report.bf16Exact;s.maxBF16ULPs=std::max(s.maxBF16ULPs,report.bf16ULPs);
    if(ref.dotAbsoluteBound)s.maxBoundRatio=std::max(s.maxBoundRatio,report.dotAbsoluteError/ref.dotAbsoluteBound);
    const bool pass=!ref.exceptional&&report.finiteCapture&&report.dotWithinBound&&report.scaledWithinBound&&
        report.signMatches&&report.negativeZeroMatches&&report.diagnosticsCoverExpected;result.certifiedFailures+=!pass;
    if((!pass||(report.strictSensitive&&!report.strictBF16Pass))&&s.failures.size()<16){std::ostringstream e;e<<std::setprecision(17)
      <<"{\"route\":"<<route<<",\"column\":"<<column<<",\"reference_dot\":"<<finiteJSON(ref.dot)<<",\"observed_dot\":"<<finiteJSON(c.dot[index])
      <<",\"dot_abs_bound\":"<<finiteJSON(ref.dotAbsoluteBound)<<",\"reference_scaled\":"<<finiteJSON(ref.scaled)<<",\"observed_scaled\":"<<finiteJSON(c.scaled[index])
      <<",\"scaled_abs_bound\":"<<finiteJSON(ref.scaledAbsoluteBound)<<",\"strict_sensitive\":"<<(report.strictSensitive?"true":"false")
      <<",\"strict_bf16_pass\":"<<(report.strictBF16Pass?"true":"false")<<",\"certified_pass\":"<<(pass?"true":"false")<<'}';s.failures.push_back(e.str());}
  }
  return result;
}
struct NativeSnapshot {
  compact_native_r4_metadata::Expected metadata;
  std::vector<uint16_t> packA,rawGU,preparedGU,canonicalRaw,canonicalPrepared,down;
  std::vector<uint8_t> jobBackingTail;
  std::array<Projection,3> projections;
  std::array<SampleReport,3> reports;
  uint32_t gateDiagnostic=0,preparedDiagnostic=0,fullDiagnostic=0;
  uint32_t compiledSiluDiagnostic=0;
  uint64_t compiledSiluMismatches=0,downTapMismatches=0;
  bool metadataExpected=false,tailUntouched=false,packPaddingZero=false,preparedPaddingZero=false;
};
void selfTest() {
  cpuTest();require(compact_native_r4_metadata::cpuSelfTest(),"native M16 CPU metadata reference failed");
  require(sizeof(FlashMoEBucketParams)==32&&sizeof(FlashMoEBucketJob)==8,"native metadata ABI differs");
  require(operandRows==routes+63&&jobSlots==(routes+15)/16+511,"103-row padding or514-job capacity differs");
  for(std::string_view name:{"u10","u20","u30","u40","oldmixed","permuted"}) {
    const auto ids=fixtureIDs(name);require(gathered_i8_qmv::fixtureIDDiagnostics(ids,4)==0,"fixture invalid or duplicate");
    const auto m=compact_native_r4_metadata::expected(ids);require(m.offsets[512]==40&&!m.sticky,"valid metadata differs");
    require(m.jobCount==(name=="u10"?10u:name=="u20"?20u:name=="u40"?40u:30u),"fixture job count differs");}
  const auto ids=fixtureIDs("oldmixed");require(ids[34]==431&&ids[19]==38&&ids[20]==0,"failed route34 fidelity fixture changed");
  std::cout<<"{\"compact_native_cpu_checks\":\"passed\",\"cases\":6,\"tile_rows\":16,\"job_slots\":514,\"operand_rows\":103,"
      "\"gpu_work\":false,\"payload_reads\":false,\"hashing_performed\":false}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){selfTest();return 0;}
    require(argc>=7&&std::string_view(argv[1])=="--gpu","usage: compact-native-oracle --gpu metallib Full512Store layer 4 NEWreport [--case oldmixed|u10|u20|u30|u40|permuted] [--rank-shift0..511] [--pairs1..60]");
    const uint32_t layerIndex=numeric(argv[4],0,47);require(numeric(argv[5],4,4)==4,"R4 only");
    const std::filesystem::path path(argv[6]);require(!std::filesystem::exists(path),"NEW report path required");
    std::string caseName="oldmixed";uint32_t rankShift=0,pairs=18;bool explicitRank=false;
    for(int i=7;i<argc;i+=2){require(i+1<argc,"missing CLI value");const std::string option(argv[i]);
      if(option=="--case")caseName=argv[i+1];else if(option=="--rank-shift"){rankShift=numeric(argv[i+1],0,511);explicitRank=true;}
      else if(option=="--pairs")pairs=numeric(argv[i+1],1,60);else throw std::invalid_argument("unknown compact-native option");}
    if(caseName=="permuted"&&!explicitRank)rankShift=73;const uint32_t requestedPairs=pairs;pairs=(pairs+2)/3*3;
    const auto hidden=normalizedHidden();const auto ids=fixtureIDs(caseName);const auto expected=compact_native_r4_metadata::expected(ids);
    const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],"ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
      "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
    require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","Full512 certificate differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1);::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);
    @autoreleasepool {
      MetalBackend backend(argv[2]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=4ULL<<30;
      require(physical>reserve,"host reserve unavailable");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(128ULL<<20);
      auto reservation=governor.tryReserve(planned);require(bool(reservation),"one-layer reservation denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      if(rankShift){auto *r=static_cast<uint32_t *>(layer.ranks.contents());for(uint32_t id=0;id<512;++id)r[id]=(id+rankShift)%512;}
      const auto immutable=layer.immutableHashes();std::vector<Guard> guards;
      auto h=upload(backend,hidden,guards),id=upload(backend,ids,guards);auto s=allocateMoEBlockedScratch(backend,4,10);guardScratch(backend,s,guards);
      auto diag=guarded(backend,4,guards),gatherDiag=guarded(backend,4,guards),observerDiag=guarded(backend,4,guards);
      splash::flash::compact_native_r4_safety::validate(layer,s,h,id,diag);
      auto gatherAct=guarded(backend,actElements*2,guards),gatherDown=guarded(backend,downElements*2,guards);
      auto canonical=guarded(backend,actElements*2,guards),compiled=guarded(backend,actElements*2,guards);
      std::array<Outputs,3> commonCapture;for(uint32_t plane=0;plane<3;++plane)commonCapture[plane]=outputs(backend,uint64_t{routes}*(plane==2?2560:640),guards);
      std::array<CommandGraph,2> prefix,prepared,shipping;
      for(uint32_t v=0;v<2;++v){nativeGraph(prefix[v],layer,h,id,s,diag,v==1,0);nativeGraph(prepared[v],layer,h,id,s,diag,v==1,1);
        nativeGraph(shipping[v],layer,h,id,s,diag,v==1,2);require(shipping[v].dispatches().size()==(v?6:10),"shipping dispatch count differs");}
      CommandGraph gather;gatheredGraph(gather,layer,h,id,gatherAct,gatherDown,gatherDiag);require(gather.dispatches().size()==2,"gather control dispatch count differs");
      std::array<NativeSnapshot,2> snapshots;
      for(uint32_t v=0;v<2;++v) {
        auto &out=snapshots[v];resetNative(s,diag);(void)backend.submitCommand(prefix[v].dispatches());
        // Capture the actual gate-only diagnostic before any observer can add
        // numeric poison flags. CPU canonicalization changes no shipping buffer.
        out.gateDiagnostic=diagnostic(diag);out.metadata=metadataSnapshot(s);out.metadataExpected=sameMetadata(out.metadata,expected);
        out.packA=copyBuffer<uint16_t>(s.buckets.packedInputs,uint64_t{operandRows}*2560);
        out.rawGU=copyBuffer<uint16_t>(s.packedActivated,uint64_t{operandRows}*640);out.canonicalRaw=canonicalize(out.rawGU,out.metadata);
        const auto *tail=static_cast<const uint8_t *>(s.buckets.tileJobs.contents())+jobSlots*sizeof(FlashMoEBucketJob);
        out.jobBackingTail.assign(tail,static_cast<const uint8_t *>(s.buckets.tileJobs.contents())+s.buckets.tileJobs.sizeBytes());
        out.tailUntouched=std::all_of(out.jobBackingTail.begin(),out.jobBackingTail.end(),[](uint8_t byte){return byte==0xa5;});
        out.packPaddingZero=std::all_of(out.packA.begin()+uint64_t{out.metadata.offsets[512]}*2560,out.packA.end(),[](uint16_t x){return !x;});
        for(uint32_t plane=0;plane<2;++plane){resetCapture(commonCapture[plane],out.gateDiagnostic);CommandGraph probe;nativeProbe(probe,layer,s,commonCapture[plane],plane);
          (void)backend.submitCommand(probe.dispatches());out.projections[plane]=capture(commonCapture[plane]);out.reports[plane]=f64(layer,plane,ids,hidden,out.projections[plane]);}
        std::memcpy(canonical.contents(),out.canonicalRaw.data(),canonical.sizeBytes());clear(observerDiag);poison(compiled);CommandGraph silu;
        addSiLUMultiply(silu,commonCapture[0].bf16,commonCapture[1].bf16,compiled,observerDiag,4,640,10);(void)backend.submitCommand(silu.dispatches());
        out.compiledSiluDiagnostic=diagnostic(observerDiag);
        const auto expectedRaw=copyBuffer<uint16_t>(compiled);for(size_t i=0;i<expectedRaw.size();++i)out.compiledSiluMismatches+=expectedRaw[i]!=out.canonicalRaw[i];
        // A separate authoritative native prefix captures ALL prepared103 rows
        // after the unchanged in-place down sanitizer, before down reads them.
        resetNative(s,diag);(void)backend.submitCommand(prepared[v].dispatches());out.preparedDiagnostic=diagnostic(diag);
        out.preparedGU=copyBuffer<uint16_t>(s.packedActivated,uint64_t{operandRows}*640);out.canonicalPrepared=canonicalize(out.preparedGU,metadataSnapshot(s));
        out.preparedPaddingZero=std::all_of(out.preparedGU.begin()+uint64_t{out.metadata.offsets[512]}*640,out.preparedGU.end(),[](uint16_t x){return !x;});
        resetCapture(commonCapture[2],out.preparedDiagnostic);CommandGraph downProbe;nativeProbe(downProbe,layer,s,commonCapture[2],2);(void)backend.submitCommand(downProbe.dispatches());
        out.projections[2]=capture(commonCapture[2]);out.reports[2]=f64(layer,2,ids,out.canonicalPrepared,out.projections[2]);
        resetNative(s,diag);(void)backend.submitCommand(shipping[v].dispatches());out.fullDiagnostic=diagnostic(diag);out.down=copyBuffer<uint16_t>(s.scatteredDown);
        for(size_t i=0;i<out.down.size();++i)out.downTapMismatches+=out.down[i]!=out.projections[2].bf16[i];
      }
      clear(gatherDiag);poison(gatherAct);poison(gatherDown);(void)backend.submitCommand(gather.dispatches());
      const uint32_t gatheredDiagnostic=diagnostic(gatherDiag);const auto gatheredAct=copyBuffer<uint16_t>(gatherAct),gatheredDown=copyBuffer<uint16_t>(gatherDown);
      std::array<Projection,3> gatheredProjections;std::array<SampleReport,3> gatheredF64;
      for(uint32_t plane=0;plane<3;++plane){const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
        resetCapture(commonCapture[plane],0x80000000u);CommandGraph probe;
        probe.add("flash_gathered_mpp_projection_probe",{plane==2?gatherAct:h,layer.codes[plane],layer.scales[plane],layer.ranks,id,
            commonCapture[plane].dot,commonCapture[plane].scaled,commonCapture[plane].bf16,commonCapture[plane].diag},
            FlashQMVProbeParams{4,10,k,n,uint32_t(plane==2),0,512,0},{n/64,4,10},{128,1,1});
        (void)backend.submitCommand(probe.dispatches());gatheredProjections[plane]=capture(commonCapture[plane]);
        gatheredF64[plane]=f64(layer,plane,ids,plane==2?gatheredAct:hidden,gatheredProjections[plane]);}
      clear(observerDiag);poison(compiled);CommandGraph gatherSilu;addSiLUMultiply(gatherSilu,commonCapture[0].bf16,commonCapture[1].bf16,compiled,observerDiag,4,640,10);
      (void)backend.submitCommand(gatherSilu.dispatches());const auto gatherCompiled=copyBuffer<uint16_t>(compiled);
      const bool gatherLiteral=same(gatherCompiled,gatheredAct)&&same(gatheredProjections[2].bf16,gatheredDown)&&diagnostic(observerDiag)==0x80000000u;
      bool exactNative=sameMetadata(snapshots[0].metadata,snapshots[1].metadata)&&same(snapshots[0].packA,snapshots[1].packA)&&
          same(snapshots[0].rawGU,snapshots[1].rawGU)&&same(snapshots[0].preparedGU,snapshots[1].preparedGU)&&same(snapshots[0].down,snapshots[1].down)&&
          same(snapshots[0].jobBackingTail,snapshots[1].jobBackingTail)&&snapshots[0].gateDiagnostic==snapshots[1].gateDiagnostic&&
          snapshots[0].preparedDiagnostic==snapshots[1].preparedDiagnostic&&snapshots[0].fullDiagnostic==snapshots[1].fullDiagnostic;
      std::array<std::array<uint64_t,3>,3> probeMismatches{};
      for(uint32_t plane=0;plane<3;++plane){const auto &a=snapshots[0].projections[plane],&b=snapshots[1].projections[plane];
        for(size_t i=0;i<a.dot.size();++i){probeMismatches[plane][0]+=std::bit_cast<uint32_t>(a.dot[i])!=std::bit_cast<uint32_t>(b.dot[i]);
          probeMismatches[plane][1]+=std::bit_cast<uint32_t>(a.scaled[i])!=std::bit_cast<uint32_t>(b.scaled[i]);probeMismatches[plane][2]+=a.bf16[i]!=b.bf16[i];}
        exactNative&=!probeMismatches[plane][0]&&!probeMismatches[plane][1]&&!probeMismatches[plane][2]&&a.diagnostic==b.diagnostic;}
      std::array<expert_r4_quality::MetricReport,3> gatherQuality;
      for(uint32_t plane=0;plane<3;++plane)gatherQuality[plane]=expert_r4_quality::metrics(bf16View(gatheredProjections[plane].bf16),bf16View(snapshots[0].projections[plane].bf16),plane==2?2560:640);
      const auto activatedQuality=expert_r4_quality::metrics(bf16View(gatheredAct),bf16View(snapshots[0].canonicalRaw),640);
      const auto downQuality=expert_r4_quality::metrics(bf16View(gatheredDown),bf16View(snapshots[0].down),2560);
      // A poisoned replay of each whole shipping chain regenerates the same
      // shared bases; compare before another chain overwrites that storage.
      std::array<bool,2> replay;
      for(uint32_t v=0;v<2;++v){resetNative(s,diag);(void)backend.submitCommand(shipping[v].dispatches());
        replay[v]=same(copyBuffer<uint16_t>(s.scatteredDown),snapshots[v].down)&&same(copyBuffer<uint16_t>(s.packedActivated,uint64_t{operandRows}*640),snapshots[v].preparedGU)&&
            sameMetadata(metadataSnapshot(s),snapshots[v].metadata)&&diagnostic(diag)==snapshots[v].fullDiagnostic;}
      const auto safety=splash::flash::compact_native_r4_safety::run(backend,layer,hidden,ids,
          [&](uint64_t bytes){return guarded(backend,bytes,guards);});
      // Execute the pre-existing bad-rank difference explicitly. Integer plans
      // include original valid IDs regardless of rank. Native jobs skip rank512
      // with bit1, leaving that job's raw GU/down poison untouched; gather writes
      // canonical NaN and retains bit5. No gathered equality is claimed here.
      auto corruptLayer=layer;auto corruptRanks=copyBuffer<uint32_t>(layer.ranks);
      corruptRanks[size_t(ids[0])]=512;corruptLayer.ranks=upload(backend,corruptRanks,guards);
      std::array<uint32_t,2> badRankGateDiag{},badRankFullDiag{};
      std::array<std::vector<uint16_t>,2> badRankRaw,badRankDown;
      bool badRankNativeMetadata=true,badRankNativeSkip=true;
      for(uint32_t v=0;v<2;++v) {
        splash::flash::compact_native_r4_safety::validate(corruptLayer,s,h,id,diag);
        resetNative(s,diag);CommandGraph gu;nativeGraph(gu,corruptLayer,h,id,s,diag,v==1,0);
        (void)backend.submitCommand(gu.dispatches());badRankGateDiag[v]=diagnostic(diag);
        const auto m=metadataSnapshot(s);badRankNativeMetadata&=sameMetadata(m,expected);
        badRankRaw[v]=canonicalize(copyBuffer<uint16_t>(s.packedActivated,uint64_t{operandRows}*640),m);
        resetNative(s,diag);CommandGraph full;nativeGraph(full,corruptLayer,h,id,s,diag,v==1,2);
        (void)backend.submitCommand(full.dispatches());badRankFullDiag[v]=diagnostic(diag);badRankDown[v]=copyBuffer<uint16_t>(s.scatteredDown);
        for(uint32_t route=0;route<40;++route)if(ids[route]==ids[0]) {
          badRankNativeSkip&=std::all_of(badRankRaw[v].begin()+uint64_t{route}*640,badRankRaw[v].begin()+uint64_t{route+1}*640,[](uint16_t x){return x==0xa5a5;})&&
              std::all_of(badRankDown[v].begin()+uint64_t{route}*2560,badRankDown[v].begin()+uint64_t{route+1}*2560,[](uint16_t x){return x==0xa5a5;});
        }
      }
      clear(gatherDiag);poison(gatherAct);poison(gatherDown);CommandGraph corruptGather;
      gatheredGraph(corruptGather,corruptLayer,h,id,gatherAct,gatherDown,gatherDiag);(void)backend.submitCommand(corruptGather.dispatches());
      const uint32_t badRankGatherDiag=diagnostic(gatherDiag);bool badRankGatherPoison=true;
      const auto badGatherAct=copyBuffer<uint16_t>(gatherAct),badGatherDown=copyBuffer<uint16_t>(gatherDown);
      for(uint32_t route=0;route<40;++route)if(ids[route]==ids[0])badRankGatherPoison&=
          std::all_of(badGatherAct.begin()+uint64_t{route}*640,badGatherAct.begin()+uint64_t{route+1}*640,[](uint16_t x){return x==0x7fc0;})&&
          std::all_of(badGatherDown.begin()+uint64_t{route}*2560,badGatherDown.begin()+uint64_t{route+1}*2560,[](uint16_t x){return x==0x7fc0;});
      const bool badRankDifferencePass=badRankNativeMetadata&&badRankNativeSkip&&badRankGatherPoison&&
          same(badRankRaw[0],badRankRaw[1])&&same(badRankDown[0],badRankDown[1])&&
          badRankGateDiag[0]==0x80000001u&&badRankGateDiag[1]==0x80000001u&&
          badRankFullDiag[0]==0x80000001u&&badRankFullDiag[1]==0x80000001u&&badRankGatherDiag==0x80000005u;
      // Restore clean gather publication before any warm/timed shipping work.
      clear(gatherDiag);(void)backend.submitCommand(gather.dispatches());
      const uint64_t after=backend.memoryStats().allocatedBytes;require(after>=before&&after-before<=planned,"compact oracle exceeded one layer plus128MiB reservation");reservation->commit();
      bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      bool inputsImmutable=!std::memcmp(hidden.data(),h.contents(),h.sizeBytes())&&!std::memcmp(ids.data(),id.contents(),id.sizeBytes());
      layer.checkImmutableHashes(immutable);
      bool prequalified=exactNative&&gatherLiteral&&gatheredDiagnostic==0x80000000u&&activatedQuality.pass()&&downQuality.pass()&&
          replay[0]&&replay[1]&&canaries&&inputsImmutable&&safety.pass()&&badRankDifferencePass;
      for(uint32_t v=0;v<2;++v){const auto &n=snapshots[v];prequalified&=n.metadataExpected&&n.tailUntouched&&n.packPaddingZero&&n.preparedPaddingZero&&
          !n.compiledSiluMismatches&&!n.downTapMismatches&&n.gateDiagnostic==0x80000000u&&n.preparedDiagnostic==0x80000000u&&n.fullDiagnostic==0x80000000u;
        prequalified&=n.compiledSiluDiagnostic==0x80000000u;
        for(uint32_t plane=0;plane<3;++plane)prequalified&=n.reports[plane].pass()&&n.projections[plane].diagnostic==0x80000000u;}
      for(uint32_t plane=0;plane<3;++plane)prequalified&=gatherQuality[plane].pass()&&gatheredF64[plane].pass()&&gatheredProjections[plane].diagnostic==0x80000000u;
      std::array<double,3> warmGPU{};std::array<uint32_t,3> warmRuns{};std::array<std::vector<CommandTiming>,3> times;
      auto dispatches=[&](uint32_t v){return v==2?gather.dispatches():shipping[v].dispatches();};
      if(prequalified){clear(diag);clear(gatherDiag);uint32_t cycle=0;
        while(std::any_of(warmGPU.begin(),warmGPU.end(),[](double x){return x<.150;})){
          require(cycle<20000,"150msGPUwarm failed within bounded replays");for(uint32_t step=0;step<3;++step){const uint32_t v=(cycle+step)%3;
            if(warmGPU[v]<.150){const auto t=backend.submitCommand(dispatches(v));warmGPU[v]+=t.gpuSeconds;++warmRuns[v];}}++cycle;}
        for(uint32_t sample=0;sample<pairs;++sample)for(uint32_t step=0;step<3;++step){const uint32_t v=(sample+step)%3;times[v].push_back(backend.submitCommand(dispatches(v)));}}
      // No CPU buffer/diagnostic touches occur in either loop above. Inspect
      // the final shared result only after the final sample, then replay each
      // control separately for exact output publication against its snapshots.
      bool timingPass=prequalified&&diagnostic(diag)==0x80000000u&&diagnostic(gatherDiag)==0x80000000u;
      if(prequalified)for(uint32_t v=0;v<2;++v){(void)backend.submitCommand(shipping[v].dispatches());replay[v]&=same(copyBuffer<uint16_t>(s.scatteredDown),snapshots[v].down)&&
          same(copyBuffer<uint16_t>(s.packedActivated,uint64_t{operandRows}*640),snapshots[v].preparedGU)&&diagnostic(diag)==0x80000000u;timingPass&=replay[v];}
      canaries&=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});inputsImmutable&=!std::memcmp(hidden.data(),h.contents(),h.sizeBytes())&&!std::memcmp(ids.data(),id.contents(),id.sizeBytes());
      timingPass&=canaries&&inputsImmutable&&same(copyBuffer<uint16_t>(gatherAct),gatheredAct)&&same(copyBuffer<uint16_t>(gatherDown),gatheredDown);
      std::ofstream out(path);require(bool(out),"cannot create NEW compact report");out<<std::setprecision(17);
      out<<"{\"schema\":\"splash-private-compact-integer-native-m16-r4-v1\",\"layer\":"<<layerIndex<<",\"rows\":4,\"case\":"<<splash::json::quote(caseName)
        <<",\"rank_shift\":"<<rankShift<<",\"source_generated_normalized_h\":true,\"route34_id\":"<<ids[34]
        <<",\"one_shared_readonly_layer\":true,\"mapped_layer_bytes\":"<<metadata.layers[layerIndex].bytes<<",\"allocated_delta_bytes\":"<<after-before
        <<",\"same_native_scratch_and_diag_bases\":true,\"same_native_probe_output_bases\":true,\"job_slots_compared\":514,\"job_backing_slots\":"<<s.buckets.tileJobs.sizeBytes()/8
        <<",\"operand_rows\":103,\"exact_old_native_all_stage_pass\":"<<(exactNative?"true":"false")<<",\"all_pre_timing_semantic_checks_pass\":"<<(prequalified?"true":"false")
        <<",\"timing_attempted\":"<<(prequalified?"true":"false")<<",\"timing_pass\":"<<(timingPass?"true":"false")<<",\"canaries\":"<<(canaries?"true":"false")
        <<",\"inputs_immutable\":"<<(inputsImmutable?"true":"false")<<",\"weights_immutable_pre_timing\":true,\"model_quality_qualified\":false,\"verification_state_qualified\":false"
        <<",\"requested_samples\":"<<requestedPairs<<",\"matched_samples\":"<<pairs<<",\"safety\":"<<safety.json()
        <<",\"executed_preexisting_bad_rank_difference\":{\"pass\":"<<(badRankDifferencePass?"true":"false")
        <<",\"native_metadata_rank_independent\":"<<(badRankNativeMetadata?"true":"false")
        <<",\"native_raw_and_down_job_skipped_untouched\":"<<(badRankNativeSkip?"true":"false")
        <<",\"native_gate_sticky\":["<<badRankGateDiag[0]<<','<<badRankGateDiag[1]<<"],\"native_full_sticky\":["<<badRankFullDiag[0]<<','<<badRankFullDiag[1]
        <<"],\"gather_nan_poison\":"<<(badRankGatherPoison?"true":"false")<<",\"gather_full_sticky\":"<<badRankGatherDiag<<'}'
        <<",\"raw_gate_observer_policy\":\"Separate untimed setup+pack+GUP prefix, actual gate diagnostic captured first, RootCPUcanonicalization copies raw live GU without diagnostic mutation. RawGU103rows startsidenticalA5 onboth native variants; dead/paddingGU is not claimed initialized until original prepareDown. Originalprepared103rows and packedH103rows fullycompared.\""
        <<",\"math_policy\":\"No softwaredot, new association, producer/preparation/store fusion or other tile. All pack/M16GU/excludedpoison/prepareDown/M16down kernels and bindings literal native controls. Frozen original opaqueMPP F64certificates sampled64 fixed columns perroute remain fullyreported; strictfailures independent of finitebound/normqualification. Metadata no rankfilter; pre-existing nativebad-rank skip1 versus gatheredNaN5 remains explicit safety difference.\""
        <<",\"timing_policy\":\"All exact native metadata/padding/raw/scaled/BF16/compiledSiLU/rawGU/preparedGU/down/diagnostic/replay+gatherglobal/perroute1e-4/.999999+safety assertions beforewarm. >=150msGPUwarm EACH3variant; shipping-only10/6/2dispatches, rotated18default/multiple3balanced samples. NoCPUbuffer/model/diagnostic touches insidewarm/samples; nativevariants shareexactbuffers and diagnostic.\""
        <<",\"gather_activated_quality\":";activatedQuality.json(out);out<<",\"gather_full_down_quality\":";downQuality.json(out);
      out<<",\"gather_projection_qualities\":[";for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';gatherQuality[plane].json(out);}out<<"],\"native_stage_reports\":[";
      for(uint32_t v=0;v<2;++v){if(v)out<<',';const auto &n=snapshots[v];out<<"{\"name\":"<<splash::json::quote(names[v])<<",\"metadata_expected\":"<<(n.metadataExpected?"true":"false")
        <<",\"unused_backing_tail_untouched\":"<<(n.tailUntouched?"true":"false")<<",\"packed_input_padding_zero\":"<<(n.packPaddingZero?"true":"false")
        <<",\"prepared_activation_padding_zero\":"<<(n.preparedPaddingZero?"true":"false")<<",\"gate_sticky_before_observer\":"<<n.gateDiagnostic<<",\"prepared_sticky\":"<<n.preparedDiagnostic
        <<",\"full_sticky\":"<<n.fullDiagnostic<<",\"compiled_silu_mismatches\":"<<n.compiledSiluMismatches<<",\"own_prepared_down_tap_mismatches\":"<<n.downTapMismatches
        <<",\"compiled_silu_sticky\":"<<n.compiledSiluDiagnostic
        <<",\"poisoned_post_timing_replay_exact\":"<<(replay[v]?"true":"false")<<",\"f64\":[";for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';n.reports[plane].json(out);}out<<"]}";}
      out<<"],\"projection_exact_mismatches\":[";for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';out<<"{\"plane\":"<<plane<<",\"raw\":"<<probeMismatches[plane][0]
        <<",\"scaled\":"<<probeMismatches[plane][1]<<",\"bf16\":"<<probeMismatches[plane][2]<<'}';}out<<"],\"gather_f64\":[";
      for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';gatheredF64[plane].json(out);}out<<"],\"timings\":[";
      for(uint32_t v=0;v<3;++v){if(v)out<<',';out<<"{\"name\":"<<splash::json::quote(names[v])<<",\"dispatches\":"<<(v==0?10:v==1?6:2)
        <<",\"warm_gpu_ms\":"<<warmGPU[v]*1000<<",\"warm_replays\":"<<warmRuns[v]<<",\"samples\":[";for(size_t i=0;i<times[v].size();++i){if(i)out<<',';out
          <<"{\"gpu_ms\":"<<times[v][i].gpuSeconds*1000<<",\"wall_ms\":"<<times[v][i].wallSeconds*1000<<'}';}out<<"]}";}out<<"]}\n";
      backend.stop();std::cout<<"{\"exact_old_native_all_stage_pass\":"<<(exactNative?"true":"false")<<",\"all_pre_timing_semantic_checks_pass\":"<<(prequalified?"true":"false")
        <<",\"timing_pass\":"<<(timingPass?"true":"false")<<",\"report\":"<<splash::json::quote(path.string())<<"}\n";return timingPass?0:2;
    }
  }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
