// Private one-layer numerical-alternative oracle. CPU entry creates no device,
// hashes no payload, and reads no model data. Root owns every --gpu invocation.
#define main preserved_qmv_oracle_entry_not_invoked
#include "../prefill4k_allrows_qmv_oracle.mm"
#undef main
#include "flash/FlashGatheredMPP.hpp"
#include "quality.hpp"
#include "malformed.hpp"

namespace {
constexpr uint32_t variantCount = 4;
constexpr uint32_t sampledColumns = 64;
constexpr std::array<const char *,variantCount> variantNames{
    "oldBucketMPP", "directGatheredSG4", "vectorGEMVL32O4", "vectorGEMVL16O8"};
constexpr std::array<const char *,variantCount> probeNames{
    "", "flash_gathered_mpp_projection_probe",
    "gemv_decode_sep21_v4_l32_o4_projection_probe",
    "gemv_decode_sep21_v4_l16_o8_projection_probe"};
constexpr std::array<const char *,variantCount> gateNames{
    "", "flash_gathered_mpp_gate_up_m16_n64_sg4",
    "gemv_decode_sep21_v4_l32_o4_gate_up", "gemv_decode_sep21_v4_l16_o8_gate_up"};
constexpr std::array<const char *,variantCount> downNames{
    "", "flash_gathered_mpp_down_m16_n64_sg4",
    "gemv_decode_sep21_v4_l32_o4_down", "gemv_decode_sep21_v4_l16_o8_down"};
constexpr std::array<uint32_t,variantCount> outputRows{64,64,4,8};

uint64_t differing(MetalBuffer a, MetalBuffer b, uint64_t count, uint32_t bytes) {
  const auto *aa=static_cast<const uint8_t *>(a.contents());
  const auto *bb=static_cast<const uint8_t *>(b.contents());
  uint64_t result=0;
  for(uint64_t i=0;i<count;++i) result+=std::memcmp(aa+i*bytes,bb+i*bytes,bytes)!=0;
  return result;
}
std::span<const uint16_t> bf16View(MetalBuffer b) {
  return {static_cast<const uint16_t *>(b.contents()),size_t(b.sizeBytes()/2)};
}
std::vector<uint16_t> normalizedHidden(uint32_t rows) {
  std::vector<uint16_t> result(uint64_t{rows}*2560);
  for(uint32_t row=0;row<rows;++row) {
    double squares=0;
    for(uint32_t k=0;k<2560;++k) {
      const double value=double(int32_t((k*173+row*31)%211)-105); squares+=value*value;
    }
    const double normalization=std::sqrt(squares/2560);
    for(uint32_t k=0;k<2560;++k)
      result[row*2560+k]=reference::bf16FromF64(double(int32_t((k*173+row*31)%211)-105)/normalization);
  }
  return result;
}
std::vector<int64_t> routeFixture(uint32_t rows,std::string_view pattern) {
  require(pattern=="mixed"||pattern=="repeated"||pattern=="spread"||pattern=="permuted",
      "unsupported frozen route fixture");
  std::vector<int64_t> result(uint64_t{rows}*10);
  for(uint32_t row=0;row<rows;++row) for(uint32_t slot=0;slot<10;++slot) {
    const uint32_t shift=pattern=="repeated"?0:pattern=="mixed"?(row%2?row*73:0):row*73;
    const uint32_t selection=pattern=="permuted"?(slot*3+row)%10:slot;
    result[row*10+slot]=(selection*53+shift)%512;
    if(pattern=="repeated"&&slot==9) result[row*10+slot]=511;
  }
  return result;
}
void directGate(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer hidden,
    MetalBuffer ids,MetalBuffer act,MetalBuffer diag,uint32_t rows,uint32_t variant) {
  graph.add(gateNames[variant],{hidden,layer.codes[0],layer.scales[0],layer.codes[1],
      layer.scales[1],layer.ranks,ids,act,diag},FlashGatheredI8QMVParams{rows,10,512,0},
      {640/outputRows[variant],rows,10},{128,1,1});
}
void directDown(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer act,
    MetalBuffer ids,MetalBuffer down,MetalBuffer diag,uint32_t rows,uint32_t variant) {
  graph.add(downNames[variant],{act,layer.codes[2],layer.scales[2],layer.ranks,ids,down,diag},
      FlashGatheredI8QMVParams{rows,10,512,0},{2560/outputRows[variant],rows,10},{128,1,1});
}
void directChain(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer hidden,
    MetalBuffer ids,MetalBuffer act,MetalBuffer down,MetalBuffer diag,uint32_t rows,uint32_t variant) {
  directGate(graph,layer,hidden,ids,act,diag,rows,variant);
  directDown(graph,layer,act,ids,down,diag,rows,variant);
}
void projection(CommandGraph &graph,const one::OneLayerPayload &layer,MetalBuffer input,
    MetalBuffer ids,const Outputs &capture,uint32_t rows,uint32_t plane,uint32_t variant) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
  graph.add(probeNames[variant],{input,layer.codes[plane],layer.scales[plane],layer.ranks,
      ids,capture.dot,capture.scaled,capture.bf16,capture.diag},
      FlashQMVProbeParams{rows,10,k,n,uint32_t(plane==2),variant>=2?4u:0u,512,0},
      {n/outputRows[variant],rows,10},{128,1,1});
}
// Fixed before any payload inspection. All routes and exactly64 columns/route
// are sampled; full dense BF16 comparison covers every projected output.
std::vector<uint32_t> samplePairs(uint32_t rows,uint32_t plane) {
  const uint32_t n=plane==2?2560:640,stride=n/sampledColumns;
  std::vector<uint32_t> pairs;
  pairs.reserve(uint64_t{rows}*10*sampledColumns*2);
  for(uint32_t route=0;route<rows*10;++route) for(uint32_t i=0;i<sampledColumns;++i) {
    pairs.push_back(route); pairs.push_back(i*stride+(route*17+plane*7)%stride);
  }
  return pairs;
}
struct SampleSummary {
  Summary frozen;
  uint64_t certifiedFailures=0, diagnosticsFailures=0;
  double maxScaledBoundRatio=0;
  bool certifiedPass() const { return !certifiedFailures; }
  void json(std::ostream &out) const {
    out<<"{\"sampled_only\":true,\"frozen_strict\":"; frozen.json(out);
    out<<",\"certified_bound_sign_finite_failures\":"<<certifiedFailures
        <<",\"diagnostics_coverage_failures\":"<<diagnosticsFailures
        <<",\"max_scaled_bound_ratio\":"<<finiteJSON(maxScaledBoundRatio)
        <<",\"certified_bound_sign_finite_pass\":"<<(certifiedPass()?"true":"false")<<'}';
  }
};
// reducer0: opaque MPP; 1/2: unchanged C1/C2 certificate; 3/4: new L32/L16;
// reducer5: independent sequential-K scalar GPU sample certificate.
SampleSummary sampledF64(const one::OneLayerPayload &layer,uint32_t plane,
    const std::vector<int64_t> &ids,MetalBuffer input,const Outputs &capture,
    const std::vector<uint32_t> &pairs,uint32_t reducer,bool compact=false) {
  const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
  const auto *x=static_cast<const uint16_t *>(input.contents());
  const auto *codes=static_cast<const int8_t *>(layer.codes[plane].contents());
  const auto *scales=static_cast<const float *>(layer.scales[plane].contents());
  const auto *ranks=static_cast<const uint32_t *>(layer.ranks.contents());
  const auto *dots=static_cast<const float *>(capture.dot.contents());
  const auto *scaled=static_cast<const float *>(capture.scaled.contents());
  const auto *bf=static_cast<const uint16_t *>(capture.bf16.contents());
  SampleSummary result;
  for(size_t sample=0;sample<pairs.size()/2;++sample) {
    const uint32_t route=pairs[sample*2],column=pairs[sample*2+1];
    const uint64_t row=uint64_t{ranks[ids[route]]}*n+column,index=compact?sample:uint64_t{route}*n+column;
    const std::span<const uint16_t> xv{x+uint64_t{plane==2?route:route/10}*k,k};
    const std::span<const int8_t> cv{codes+row*k,k};
    const auto ref=reducer==3||reducer==4?gemv_quality::vectorReference(xv,cv,scales[row],reducer==3?32:16):
        reducer==5?gemv_quality::scalarReference(xv,cv,scales[row]):
        reference::reference(xv,cv,scales[row],32,reducer==0);
    const auto report=reference::assess(ref,dots[index],scaled[index],bf[index],diagnostic(capture.diag));
    auto &s=result.frozen; ++s.dots;
    s.boundFailures+=!report.dotWithinBound||!report.scaledWithinBound;
    s.strictSensitive+=report.strictSensitive;
    s.strictFailures+=report.strictSensitive&&!report.strictBF16Pass;
    s.signFailures+=!report.signMatches; s.exceptional+=ref.exceptional;
    s.bf16ReferenceMismatches+=!report.bf16Exact; s.maxBF16ULPs=std::max(s.maxBF16ULPs,report.bf16ULPs);
    if(ref.dotAbsoluteBound) s.maxBoundRatio=std::max(s.maxBoundRatio,report.dotAbsoluteError/ref.dotAbsoluteBound);
    if(ref.scaledAbsoluteBound) result.maxScaledBoundRatio=std::max(result.maxScaledBoundRatio,report.scaledAbsoluteError/ref.scaledAbsoluteBound);
    result.diagnosticsFailures+=!report.diagnosticsCoverExpected;
    const bool certified=!ref.exceptional&&report.finiteCapture&&report.dotWithinBound&&
        report.scaledWithinBound&&report.signMatches&&report.negativeZeroMatches&&report.diagnosticsCoverExpected;
    result.certifiedFailures+=!certified;
    if((!certified||(report.strictSensitive&&!report.strictBF16Pass))&&s.failures.size()<16) {
      std::ostringstream f; f<<std::setprecision(17)<<"{\"route\":"<<route<<",\"column\":"<<column
        <<",\"reference_dot\":"<<finiteJSON(ref.dot)<<",\"observed_dot\":"<<finiteJSON(dots[index])
        <<",\"dot_abs_bound\":"<<finiteJSON(ref.dotAbsoluteBound)<<",\"reference_scaled\":"<<finiteJSON(ref.scaled)
        <<",\"observed_scaled\":"<<finiteJSON(scaled[index])<<",\"scaled_abs_bound\":"<<finiteJSON(ref.scaledAbsoluteBound)
        <<",\"reference_bf16\":"<<ref.bf16<<",\"observed_bf16\":"<<bf[index]
        <<",\"strict_sensitive\":"<<(report.strictSensitive?"true":"false")
        <<",\"strict_sensitive_pass\":"<<(report.strictBF16Pass?"true":"false")
        <<",\"certified_bound_sign_finite_pass\":"<<(certified?"true":"false")<<'}';
      s.failures.push_back(f.str());
    }
  }
  return result;
}
void gemvCPU() {
  cpuTest();
  require(sizeof(FlashGatheredI8QMVParams)==16&&sizeof(FlashGatheredMPPParams)==16&&sizeof(FlashQMVProbeParams)==32,
      "GEMV producer/probe ABI differs");
  require(gemv_quality::cpuSelfTest(),"independent GEMV F64 certificate or frozen quality gate CPU test failed");
  for(uint32_t rows:{1u,2u,4u,8u,16u}) for(std::string_view pattern:{"mixed","repeated","spread","permuted"}) {
    require(gathered_i8_qmv::fixtureIDDiagnostics(routeFixture(rows,pattern),rows)==0,"route fixture is malformed");
    for(uint32_t plane=0;plane<3;++plane) {
      const auto pairs=samplePairs(rows,plane); const uint32_t n=plane==2?2560:640;
      require(pairs.size()==uint64_t{rows}*10*sampledColumns*2,"sample extent differs");
      for(size_t i=0;i<pairs.size()/2;++i) require(pairs[i*2]<rows*10&&pairs[i*2+1]<n,"sample coordinates are invalid");
    }
  }
  std::cout<<"{\"gemv_cpu_checks\":\"passed\",\"gpu_work\":false,\"hashing_performed\":false,"
      "\"payload_reads\":false,\"quality_thresholds_frozen\":true,\"columns_sampled_per_route\":64}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test") { gemvCPU(); return 0; }
    require(argc>=7&&std::string_view(argv[1])=="--gpu",
      "usage: gemv-oracle --gpu metallib Full512Store layer rows NEWreport [--pattern mixed|repeated|spread|permuted] [--hidden bf16] [--ids i64] [--pairs1..31] [--rank-shift0..511]");
    const uint32_t layerIndex=numeric(argv[4],0,47),rows=numeric(argv[5],1,16);
    const std::filesystem::path reportPath(argv[6]); require(!std::filesystem::exists(reportPath),"NEW report path required");
    std::string pattern="mixed",hiddenPath,idsPath; uint32_t pairs=16,rankShift=0;
    for(int i=7;i<argc;i+=2) {
      require(i+1<argc,"missing option value"); const std::string option(argv[i]);
      if(option=="--pattern") pattern=argv[i+1]; else if(option=="--hidden") hiddenPath=argv[i+1];
      else if(option=="--ids") idsPath=argv[i+1]; else if(option=="--pairs") pairs=numeric(argv[i+1],1,31);
      else if(option=="--rank-shift") rankShift=numeric(argv[i+1],0,511); else throw std::invalid_argument("unknown GEMV oracle option");
    }
    const uint32_t requestedPairs=pairs;
    pairs=(pairs+variantCount-1)/variantCount*variantCount;
    auto hidden=normalizedHidden(rows); auto ids=routeFixture(rows,pattern);
    if(!hiddenPath.empty()) hidden=fixture<uint16_t>(hiddenPath,hidden.size());
    if(!idsPath.empty()) ids=fixture<int64_t>(idsPath,ids.size());
    require(std::all_of(hidden.begin(),hidden.end(),[](uint16_t x){return std::isfinite(reference::bf16Number(x));}),
        "finite normalized or caller BF16 hidden fixture required");
    require(gathered_i8_qmv::fixtureIDDiagnostics(ids,rows)==0,"valid unique original I64 IDs required");
    const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],
      "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
      "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
    require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","Full512 certificate differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1); ::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);
    @autoreleasepool {
      MetalBackend backend(argv[2]); const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=4ULL<<30;
      require(physical>reserve,"host reserve unavailable"); splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(128ULL<<20);
      auto reservation=governor.tryReserve(planned); require(bool(reservation),"one-layer GEMV reservation refused");
      const uint64_t before=backend.memoryStats().allocatedBytes; auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      if(rankShift) { auto *ranks=static_cast<uint32_t *>(layer.ranks.contents()); for(uint32_t id=0;id<512;++id) ranks[id]=(id+rankShift)%512; }
      const auto immutable=layer.immutableHashes(); std::vector<Guard> guards;
      auto x=upload(backend,hidden,guards),idBuffer=upload(backend,ids,guards);
      auto scratch=allocateMoEBlockedScratch(backend,rows,10); guardScratch(backend,scratch,guards);
      const uint64_t actElements=uint64_t{rows}*10*640,downElements=uint64_t{rows}*10*2560;
      std::array<MetalBuffer,variantCount> act,down,diag;
      std::array<CommandGraph,variantCount> chain;
      for(uint32_t v=0;v<variantCount;++v) {
        act[v]=guarded(backend,actElements*2,guards); diag[v]=guarded(backend,4,guards);
        down[v]=v?guarded(backend,downElements*2,guards):scratch.scatteredDown;
        if(v) directChain(chain[v],layer,x,idBuffer,act[v],down[v],diag[v],rows,v);
        else mppChain(chain[v],layer,x,idBuffer,scratch,diag[v],rows);
        clear(diag[v]);
      }
      // No data/diagnostic CPU touches occur inside either timing loop. GPU
      // timestamps alone decide when each variant has accumulated150ms warmup.
      std::array<double,variantCount> warmGPU{}; std::array<uint32_t,variantCount> warmRuns{};
      uint32_t warmCycle=0;
      while(std::any_of(warmGPU.begin(),warmGPU.end(),[](double s){return s<0.150;})) {
        require(warmCycle<20000,"GPU warm timestamps failed to reach150ms within bounded replays");
        for(uint32_t step=0;step<variantCount;++step) {
          const uint32_t v=(warmCycle+step)%variantCount;
          if(warmGPU[v]<0.150) { const auto timing=backend.submitCommand(chain[v].dispatches()); warmGPU[v]+=timing.gpuSeconds; ++warmRuns[v]; }
        }
        ++warmCycle;
      }
      std::array<std::vector<CommandTiming>,variantCount> times;
      for(uint32_t pair=0;pair<pairs;++pair) for(uint32_t step=0;step<variantCount;++step) {
        const uint32_t v=(pair+step)%variantCount; times[v].push_back(backend.submitCommand(chain[v].dispatches()));
      }
      // All GPU assertions, diagnostics, reference scans and data comparisons
      // start only after the final matched timing sample has completed.
      std::array<uint32_t,variantCount> shippingDiagnostics;
      for(uint32_t v=0;v<variantCount;++v) shippingDiagnostics[v]=diagnostic(diag[v]);
      CommandGraph unpack;
      unpack.add("flash_qmv_probe_unpack_activation",{scratch.packedActivated,scratch.buckets.canonicalToPacked,act[0],diag[0]},
          FlashQMVProbeParams{rows,10,2560,640,0,0,512,0},{3,rows*10,1},{256,1,1});
      (void)backend.submitCommand(unpack.dispatches());
      std::array<std::vector<uint8_t>,variantCount> savedAct,savedDown;
      for(uint32_t v=0;v<variantCount;++v) {
        savedAct[v].resize(act[v].sizeBytes()); savedDown[v].resize(down[v].sizeBytes());
        std::memcpy(savedAct[v].data(),act[v].contents(),savedAct[v].size());
        std::memcpy(savedDown[v].data(),down[v].contents(),savedDown[v].size());
      }
      std::array<std::array<Outputs,variantCount>,3> caps;
      std::array<std::array<SampleSummary,variantCount>,3> f64;
      std::array<std::array<gemv_quality::MetricReport,variantCount>,3> projectionQuality;
      std::array<std::array<std::array<uint64_t,3>,variantCount>,3> projectionDifferences{};
      std::array<std::array<SampleSummary,2>,3> c1c2F64;
      std::array<SampleSummary,3> scalarF64;
      std::array<std::array<std::array<uint64_t,3>,variantCount>,3> scalarDifferences{};
      std::array<bool,variantCount> primitive; primitive.fill(true);
      std::array<bool,variantCount> frozenStrict; frozenStrict.fill(true);
      std::array<uint64_t,variantCount> swigluMismatch{},chainDownTapMismatch{};
      for(uint32_t plane=0;plane<3;++plane) {
        const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
        const auto input=plane==2?act[0]:x; const uint64_t elements=uint64_t{rows}*10*n;
        const auto selected=samplePairs(rows,plane);
        for(uint32_t v=0;v<variantCount;++v) {
          auto &capture=caps[plane][v]; capture=outputs(backend,elements,guards); clear(capture.diag); CommandGraph probe;
          if(!v) probe.add("flash_qmv_probe_mpp_m16",{plane==2?scratch.packedActivated:scratch.buckets.packedInputs,
              layer.codes[plane],layer.scales[plane],layer.ranks,scratch.buckets.offsets,scratch.buckets.tileJobs,
              scratch.buckets.jobCount,scratch.buckets.routeMap,capture.dot,capture.scaled,capture.bf16,capture.diag},
              FlashQMVProbeParams{rows,10,k,n,uint32_t(plane==2),0,512,0},{n/64,rows*10,1},{128,1,1});
          else projection(probe,layer,input,idBuffer,capture,rows,plane,v);
          (void)backend.submitCommand(probe.dispatches());
          f64[plane][v]=sampledF64(layer,plane,ids,input,capture,selected,v>=2?v+1:0);
          primitive[v]&=f64[plane][v].certifiedPass()&&diagnostic(capture.diag)==0x80000000u;
          frozenStrict[v]&=f64[plane][v].frozen.pass();
          projectionDifferences[plane][v]={differing(caps[plane][0].dot,capture.dot,elements,4),
              differing(caps[plane][0].scaled,capture.scaled,elements,4),differing(caps[plane][0].bf16,capture.bf16,elements,2)};
        }
        for(uint32_t v=0;v<variantCount;++v)
          projectionQuality[plane][v]=gemv_quality::metrics(bf16View(caps[plane][1].bf16),bf16View(caps[plane][v].bf16),n);
        for(uint32_t c=1;c<=2;++c) {
          auto capture=outputs(backend,elements,guards); clear(capture.diag); CommandGraph probe;
          probe.add("flash_qmv_probe_c"+std::to_string(c),{input,layer.codes[plane],layer.scales[plane],layer.ranks,
              idBuffer,capture.dot,capture.scaled,capture.bf16,capture.diag},
              FlashQMVProbeParams{rows,10,k,n,uint32_t(plane==2),c,512,0},{n/(4*c),rows,10},{128,1,1});
          (void)backend.submitCommand(probe.dispatches());
          c1c2F64[plane][c-1]=sampledF64(layer,plane,ids,input,capture,selected,c);
        }
        const uint32_t sampleCount=uint32_t(selected.size()/2);
        auto scalar=outputs(backend,sampleCount,guards); clear(scalar.diag);
        auto selectedBuffer=upload(backend,selected,guards);
        auto params=upload(backend,std::vector<FlashQMVProbeParams>{{rows,10,k,n,uint32_t(plane==2),4,512,0}},guards);
        auto countBuffer=upload(backend,std::vector<uint32_t>{sampleCount},guards); CommandGraph scalarProbe;
        scalarProbe.add("gemv_decode_sep21_scalar_projection_samples",{input,layer.codes[plane],layer.scales[plane],
            layer.ranks,idBuffer,scalar.dot,scalar.scaled,scalar.bf16,scalar.diag,params,selectedBuffer,countBuffer},
            {(sampleCount+255)/256,1,1},{256,1,1});
        (void)backend.submitCommand(scalarProbe.dispatches());
        scalarF64[plane]=sampledF64(layer,plane,ids,input,scalar,selected,5,true);
        for(uint32_t v=0;v<variantCount;++v) for(uint32_t i=0;i<sampleCount;++i) {
          const uint64_t index=uint64_t{selected[i*2]}*n+selected[i*2+1];
          scalarDifferences[plane][v][0]+=static_cast<const uint32_t *>(scalar.dot.contents())[i]!=static_cast<const uint32_t *>(caps[plane][v].dot.contents())[index];
          scalarDifferences[plane][v][1]+=static_cast<const uint32_t *>(scalar.scaled.contents())[i]!=static_cast<const uint32_t *>(caps[plane][v].scaled.contents())[index];
          scalarDifferences[plane][v][2]+=static_cast<const uint16_t *>(scalar.bf16.contents())[i]!=static_cast<const uint16_t *>(caps[plane][v].bf16.contents())[index];
        }
      }
      std::array<gemv_quality::MetricReport,variantCount> activationQuality,chainDownQuality;
      std::array<bool,variantCount> replay; replay.fill(true);
      std::array<bool,variantCount> qualified; qualified.fill(true);
      for(uint32_t v=0;v<variantCount;++v) {
        activationQuality[v]=gemv_quality::metrics(bf16View(act[1]),bf16View(act[v]),640);
        chainDownQuality[v]=gemv_quality::metrics(bf16View(down[1]),bf16View(down[v]),2560);
        auto expected=guarded(backend,actElements*2,guards); clear(diag[v]); CommandGraph swiglu;
        addSiLUMultiply(swiglu,caps[0][v].bf16,caps[1][v].bf16,expected,diag[v],rows,640,10);
        (void)backend.submitCommand(swiglu.dispatches()); swigluMismatch[v]=differing(act[v],expected,actElements,2);
        qualified[v]&=diagnostic(diag[v])==0x80000000u&&!swigluMismatch[v];
        if(v) {
          auto capture=outputs(backend,downElements,guards); clear(capture.diag); CommandGraph ownDown;
          projection(ownDown,layer,act[v],idBuffer,capture,rows,2,v); (void)backend.submitCommand(ownDown.dispatches());
          chainDownTapMismatch[v]=differing(down[v],capture.bf16,downElements,2);
          qualified[v]&=diagnostic(capture.diag)==0x80000000u&&!chainDownTapMismatch[v];
        } else { chainDownTapMismatch[v]=differing(down[v],caps[2][v].bf16,downElements,2); qualified[v]&=!chainDownTapMismatch[v]; }
      }
      // Poison old bucket buffers as well as every output. Direct kernels must
      // regenerate canonical outputs independently; baseline must rebuild its
      // original bucket metadata and packed operands before its exact replay.
      for(auto b:{scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.canonicalToPacked,
          scratch.buckets.packedInputs,scratch.buckets.jobOffsets,scratch.buckets.jobCount,scratch.buckets.tileJobs,
          scratch.packedActivated,scratch.scatteredDown}) std::memset(b.contents(),0xa5,b.sizeBytes());
      for(uint32_t v=1;v<variantCount;++v) {
        std::memset(act[v].contents(),0xa5,act[v].sizeBytes()); std::memset(down[v].contents(),0xa5,down[v].sizeBytes());
        clear(diag[v]); (void)backend.submitCommand(chain[v].dispatches());
        replay[v]=!std::memcmp(savedAct[v].data(),act[v].contents(),savedAct[v].size())&&
            !std::memcmp(savedDown[v].data(),down[v].contents(),savedDown[v].size())&&diagnostic(diag[v])==0x80000000u;
      }
      std::memset(act[0].contents(),0xa5,act[0].sizeBytes()); clear(diag[0]);
      (void)backend.submitCommand(chain[0].dispatches()); (void)backend.submitCommand(unpack.dispatches());
      replay[0]=!std::memcmp(savedAct[0].data(),act[0].contents(),savedAct[0].size())&&
          !std::memcmp(savedDown[0].data(),down[0].contents(),savedDown[0].size())&&diagnostic(diag[0])==0x80000000u;
      const auto malformed=splash::flash::gemv_decode_sep21_safety::run(backend,layer,rows,hidden,ids,
          [&](uint64_t bytes){return guarded(backend,bytes,guards);});
      std::array<bool,variantCount> malformedPass; malformedPass.fill(true);
      for(uint32_t v=2;v<variantCount;++v) malformedPass[v]=malformed.variants[v-2].pass();
      const uint64_t after=backend.memoryStats().allocatedBytes;
      require(after>=before&&after-before<=planned,"GEMV oracle exceeded one-layer plus128MiB memory reservation"); reservation->commit();
      const bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &guard){return guard.valid();});
      const bool inputsImmutable=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      layer.checkImmutableHashes(immutable);
      bool allQualified=true;
      for(uint32_t v=0;v<variantCount;++v) {
        qualified[v]&=primitive[0]&&primitive[v]&&canaries&&inputsImmutable&&replay[v]&&malformedPass[v]&&
            shippingDiagnostics[v]==0x80000000u&&activationQuality[v].pass()&&chainDownQuality[v].pass();
        for(uint32_t plane=0;plane<3;++plane) qualified[v]&=projectionQuality[plane][v].pass();
        allQualified&=qualified[v];
      }
      std::ofstream out(reportPath); require(bool(out),"cannot create NEW GEMV report"); out<<std::setprecision(17);
      out<<"{\"schema\":\"splash-private-vector-gemv-one-layer-v1\",\"layer\":"<<layerIndex<<",\"rows\":"<<rows
        <<",\"route_pattern\":"<<splash::json::quote(pattern)<<",\"rank_shift_fixture\":"<<rankShift
        <<",\"hidden_policy\":"<<splash::json::quote(hiddenPath.empty()?"synthetic normalized BF16; actual decode provenance unqualified":"caller BF16 fixture; provenance independently required")
        <<",\"one_shared_readonly_layer\":true,\"mapped_layer_bytes\":"<<metadata.layers[layerIndex].bytes
        <<",\"scratch_allowance_bytes\":"<<(128ULL<<20)<<",\"allocated_delta_bytes\":"<<after-before
        <<",\"canaries\":"<<(canaries?"true":"false")<<",\"inputs_immutable\":"<<(inputsImmutable?"true":"false")
        <<",\"weights_immutable\":true,\"model_quality_qualified\":false,\"real_normalized_decode_activation_qualified\":false"
        <<",\"all_numerical_alternatives_qualified\":"<<(allQualified?"true":"false")
        <<",\"sampled_f64_policy\":\"64 fixed columns per route before payload inspection; independent compensated F64 BF16xI8; product exact for regular normal products; vector accumulation depth ceil(K/(4*lanes))+3+log2(lanes); scalar depth K; F64 uncertainty gamma(4K,u64)*sumAbs; sampled certificate is not exhaustive\""
        <<",\"strict_policy\":\"Frozen C1/C2 strict-sensitive BF16 certificates remain unchanged and fully visible. All alternative strict-sensitive failures remain failures in their own reports. Pre-registered dense BF16 relativeL2 and cosine gates plus certified F64 raw/scaled bounds qualify only the numerical alternative primitive; no model-quality claim.\""
        <<",\"bf16_quality_baseline\":\"original directGatheredSG4; every dense projection uses identical input, down uses oldBucketMPP canonical activation; shipping chain quality uses each variant's own activation and down\""
        <<",\"timing_policy\":\"Each shipping variant accumulates at least150ms GPU warm in rotating order. Sixteen/default matched samples; sample counts rounded up to a multiple of4 before payload inspection, balancing every variant equally in every position. No CPU operand/model/diagnostic touches or resets inside loops. Every assertion is evaluated after all matched samples. Old MPP includes bucket/downPrepare/scatter; direct chains include their two actual producers.\""
        <<",\"requested_matched_samples\":"<<requestedPairs<<",\"matched_samples\":"<<pairs<<",\"variants\":[";
      for(uint32_t v=0;v<variantCount;++v) {
        if(v) out<<',';
        out<<"{\"name\":"<<splash::json::quote(variantNames[v])<<",\"numerical_alternative_qualified\":"<<(qualified[v]?"true":"false")
          <<",\"sampled_f64_frozen_strict_pass\":"<<(frozenStrict[v]?"true":"false")
          <<",\"sampled_certified_finite_sign_bound_pass\":"<<(primitive[v]?"true":"false")
          <<",\"shipping_sticky\":"<<shippingDiagnostics[v]<<",\"poisoned_replay_exact\":"<<(replay[v]?"true":"false")
          <<",\"malformed_shader_checks_applicable\":"<<(v>=2?"true":"false")
          <<",\"malformed_shader_checks_pass\":"<<(v>=2?(malformedPass[v]?"true":"false"):"null")
          <<",\"compiled_swiglu_mismatches\":"<<swigluMismatch[v]<<",\"own_activation_down_tap_mismatches\":"<<chainDownTapMismatch[v]
          <<",\"warm_gpu_ms\":"<<warmGPU[v]*1000<<",\"warm_replays\":"<<warmRuns[v]
          <<",\"chain_activated_quality\":"; activationQuality[v].json(out);
        out<<",\"chain_down_quality\":"; chainDownQuality[v].json(out);
        out<<",\"timings\":[";
        for(size_t i=0;i<times[v].size();++i) { if(i) out<<','; out<<"{\"wall_ms\":"<<times[v][i].wallSeconds*1000<<",\"gpu_ms\":"<<times[v][i].gpuSeconds*1000<<'}'; }
        out<<"],\"projections\":[";
        for(uint32_t plane=0;plane<3;++plane) {
          if(plane) out<<','; const auto &d=projectionDifferences[plane][v],&scalar=scalarDifferences[plane][v];
          out<<"{\"plane\":"<<plane<<",\"bf16_quality\":"; projectionQuality[plane][v].json(out);
          out<<",\"old_mpp_mismatches\":{\"raw\":"<<d[0]<<",\"scaled\":"<<d[1]<<",\"bf16\":"<<d[2]<<'}'
            <<",\"scalar_sample_mismatches\":{\"raw\":"<<scalar[0]<<",\"scaled\":"<<scalar[1]<<",\"bf16\":"<<scalar[2]<<'}'
            <<",\"sampled_f64\":"; f64[plane][v].json(out); out<<'}';
        }
        out<<"]}";
      }
      out<<"],\"unchanged_C1_C2_scalar_sample_contrasts\":[";
      for(uint32_t plane=0;plane<3;++plane) { if(plane) out<<','; out<<"{\"plane\":"<<plane<<",\"C1\":"; c1c2F64[plane][0].json(out);
        out<<",\"C2\":"; c1c2F64[plane][1].json(out); out<<",\"scalar_sequential_K\":"; scalarF64[plane].json(out); out<<'}'; }
      out<<"],\"malformed_shader_only_reports\":"<<malformed.json()<<"}\n"; backend.stop();
      std::cout<<"{\"all_numerical_alternatives_qualified\":"<<(allQualified?"true":"false")
          <<",\"report\":"<<splash::json::quote(reportPath.string())<<"}\n";
      return allQualified?0:2;
    }
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
