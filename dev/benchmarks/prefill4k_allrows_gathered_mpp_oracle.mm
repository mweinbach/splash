// Independent Root-run gathered-MPP oracle. Existing QMV/C2 sources are read-only.
// Reuse the frozen one-layer mapping, guards, old-MPP producer and F64 reporting.
#define main preserved_qmv_oracle_entry_not_invoked
#include "prefill4k_allrows_qmv_oracle.mm"
#undef main
#include "flash/FlashGatheredMPP.hpp"

namespace {
void gatheredMPPChain(CommandGraph &graph, const one::OneLayerPayload &layer,
    MetalBuffer hidden, MetalBuffer ids, MetalBuffer activated, MetalBuffer down,
    MetalBuffer diag, uint32_t rows) {
  const FlashGatheredMPPParams p{rows,10,512,0};
  graph.add("flash_gathered_mpp_gate_up_m16_n64_sg4", {hidden,layer.codes[0],layer.scales[0],
      layer.codes[1],layer.scales[1],layer.ranks,ids,activated,diag}, p, {10,rows,10},{128,1,1});
  graph.add("flash_gathered_mpp_down_m16_n64_sg4", {activated,layer.codes[2],layer.scales[2],
      layer.ranks,ids,down,diag}, p, {40,rows,10},{128,1,1});
}
uint64_t different(MetalBuffer a, MetalBuffer b, uint64_t elements, uint32_t bytes) {
  const auto *aa=static_cast<const uint8_t *>(a.contents()),*bb=static_cast<const uint8_t *>(b.contents());
  uint64_t count=0; for(uint64_t i=0;i<elements;++i) count+=std::memcmp(aa+i*bytes,bb+i*bytes,bytes)!=0; return count;
}
std::vector<uint16_t> normalizedFixture(uint32_t rows) {
  std::vector<uint16_t> result(uint64_t{rows}*2560);
  for(uint32_t row=0;row<rows;++row) {
    double squares=0;
    for(uint32_t k=0;k<2560;++k) {const double x=double(int32_t((k*173+row*31)%211)-105);squares+=x*x;}
    const double normalization=std::sqrt(squares/2560);
    for(uint32_t k=0;k<2560;++k)result[row*2560+k]=reference::bf16FromF64(double(int32_t((k*173+row*31)%211)-105)/normalization);
  }return result;
}
std::vector<int64_t> fixtureRoutes(uint32_t rows,std::string_view pattern) {
  require(pattern=="mixed"||pattern=="repeated"||pattern=="spread"||pattern=="permuted","unsupported frozen route pattern");
  std::vector<int64_t> ids(uint64_t{rows}*10);
  for(uint32_t row=0;row<rows;++row) {
   for(uint32_t slot=0;slot<10;++slot) {
    const uint32_t shift=pattern=="repeated"?0:pattern=="mixed"?(row%2?row*73:0):row*73;
    const uint32_t selection=pattern=="permuted"?(slot*3+row)%10:slot;
    ids[row*10+slot]=(selection*53+shift)%512;
    if(pattern=="repeated"&&slot==9)ids[row*10+slot]=511;
   }
  }
  return ids;
}
void gatheredCPU() {
  require(sizeof(FlashGatheredMPPParams)==16&&sizeof(FlashQMVProbeParams)==32,"gathered/probe ABI differs");
  for(uint32_t rows:{1u,2u,4u,8u,16u}) {
    auto g=gathered_mpp::geometry(rows);require(g.gateColumnGroups==10&&g.downColumnGroups==40,"fixed gathered M16 grids differ");
    for(std::string_view p:{"mixed","repeated","spread","permuted"})require(gathered_i8_qmv::fixtureIDDiagnostics(fixtureRoutes(rows,p),rows)==0,"frozen pattern has invalid duplicate ownership");
  }
  const auto old=fixtureRoutes(2,"mixed");require(old[19]==38,"initial mixed frame changed baseline failing fixture");
  std::cout<<"{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"hashing_performed\":false,\"exact_old_mpp_parity\":\"pending\"}\n";
}
}
int main(int argc,char **argv) {
  try {
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){gatheredCPU();return 0;}
    require(argc>=7&&std::string_view(argv[1])=="--gpu","usage: gathered-oracle --gpu metallib fullStore layer rows NEWreport [--pattern mixed|repeated|spread|permuted] [--hidden bf16] [--ids i64] [--pairs1..9] [--rank-shift0..511]");
    const uint32_t layerIndex=numeric(argv[4],0,47),rows=numeric(argv[5],1,16);
    const std::filesystem::path output(argv[6]);require(!std::filesystem::exists(output),"oracle requires NEW report path to preserve prior failures");
    std::string pattern="mixed",hiddenPath,idsPath;uint32_t pairs=3,rankShift=0;
    for(int i=7;i<argc;i+=2) {
      require(i+1<argc,"missing CLI value");const std::string option(argv[i]);
      if(option=="--pattern")pattern=argv[i+1];else if(option=="--hidden")hiddenPath=argv[i+1];else if(option=="--ids")idsPath=argv[i+1];
      else if(option=="--pairs")pairs=numeric(argv[i+1],1,9);else if(option=="--rank-shift")rankShift=numeric(argv[i+1],0,511);else throw std::invalid_argument("unknown gathered oracle option");
    }
    auto hidden=normalizedFixture(rows);auto ids=fixtureRoutes(rows,pattern);
    if(!hiddenPath.empty())hidden=fixture<uint16_t>(hiddenPath,hidden.size());if(!idsPath.empty())ids=fixture<int64_t>(idsPath,ids.size());
    require(gathered_i8_qmv::fixtureIDDiagnostics(ids,rows)==0,"finite oracle needs valid unique original I64 IDs");
    const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],"ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
    require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","certified Full512 metadata differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1);::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);
    @autoreleasepool {
      MetalBackend backend(argv[2]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=4ULL<<30;
      require(physical>reserve,"host reserve cannot be protected");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(128ULL<<20);
      auto reservation=governor.tryReserve(planned);require(bool(reservation),"one-layer gathered reservation denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      if(rankShift){auto *r=static_cast<uint32_t *>(layer.ranks.contents());for(uint32_t id=0;id<512;++id)r[id]=(id+rankShift)%512;}
      const auto immutable=layer.immutableHashes();std::vector<Guard> guards;
      auto x=upload(backend,hidden,guards),idBuffer=upload(backend,ids,guards);auto scratch=allocateMoEBlockedScratch(backend,rows,10);guardScratch(backend,scratch,guards);
      std::array<MetalBuffer,2>diag{guarded(backend,4,guards),guarded(backend,4,guards)};
      auto baselineAct=guarded(backend,uint64_t{rows}*10*640*2,guards),candidateAct=guarded(backend,baselineAct.sizeBytes(),guards);
      auto candidateDown=guarded(backend,uint64_t{rows}*10*2560*2,guards);std::array<CommandGraph,2>chain;
      mppChain(chain[0],layer,x,idBuffer,scratch,diag[0],rows);gatheredMPPChain(chain[1],layer,x,idBuffer,candidateAct,candidateDown,diag[1],rows);
      for(uint32_t v=0;v<2;++v){clear(diag[v]);(void)backend.submitCommand(chain[v].dispatches());}
      CommandGraph unpack;unpack.add("flash_qmv_probe_unpack_activation",{scratch.packedActivated,scratch.buckets.canonicalToPacked,baselineAct,diag[0]},FlashQMVProbeParams{rows,10,2560,640,0,0,512,0},{3,rows*10,1},{256,1,1});(void)backend.submitCommand(unpack.dispatches());
      bool exact=true,finiteBoundSign=true,baselineF64Strict=true,candidateF64Strict=true;
      std::array<std::array<Outputs,2>,3>caps;std::array<std::array<Summary,2>,3>sums;
      std::array<std::array<uint64_t,3>,3>mismatches{};
      for(uint32_t plane=0;plane<3;++plane) {
        const uint32_t n=plane==2?2560:640,k=plane==2?640:2560;const auto input=plane==2?baselineAct:x;
        for(uint32_t v=0;v<2;++v) {
          auto &c=caps[plane][v];c=outputs(backend,uint64_t{rows}*10*n,guards);clear(c.diag);CommandGraph probe;
          const FlashQMVProbeParams p{rows,10,k,n,uint32_t(plane==2),0,512,0};
          if(!v)probe.add("flash_qmv_probe_mpp_m16",{plane==2?scratch.packedActivated:scratch.buckets.packedInputs,layer.codes[plane],layer.scales[plane],layer.ranks,scratch.buckets.offsets,scratch.buckets.tileJobs,scratch.buckets.jobCount,scratch.buckets.routeMap,c.dot,c.scaled,c.bf16,c.diag},p,{n/64,rows*10,1},{128,1,1});
          else probe.add("flash_gathered_mpp_projection_probe",{input,layer.codes[plane],layer.scales[plane],layer.ranks,idBuffer,c.dot,c.scaled,c.bf16,c.diag},p,{n/64,rows,10},{128,1,1});
          (void)backend.submitCommand(probe.dispatches());sums[plane][v]=evaluate(layer,plane,rows,ids,input,c,0);
          finiteBoundSign&=sums[plane][v].finiteBoundSignPass()&&diagnostic(c.diag)==0x80000000u;
          if(!v)baselineF64Strict&=sums[plane][v].pass();else candidateF64Strict&=sums[plane][v].pass();
        }
        const uint64_t elements=uint64_t{rows}*10*n;
        mismatches[plane]={different(caps[plane][0].dot,caps[plane][1].dot,elements,4),different(caps[plane][0].scaled,caps[plane][1].scaled,elements,4),different(caps[plane][0].bf16,caps[plane][1].bf16,elements,2)};
        exact&=!mismatches[plane][0]&&!mismatches[plane][1]&&!mismatches[plane][2];
      }
      const uint64_t actMismatch=different(baselineAct,candidateAct,uint64_t{rows}*10*640,2),downMismatch=different(scratch.scatteredDown,candidateDown,uint64_t{rows}*10*2560,2);
      exact&=!actMismatch&&!downMismatch&&diagnostic(diag[0])==0x80000000u&&diagnostic(diag[1])==0x80000000u;
      std::array<uint64_t,2>swigluMismatch{};std::array<MetalBuffer,2>expectedAct;
      for(uint32_t v=0;v<2;++v){expectedAct[v]=guarded(backend,baselineAct.sizeBytes(),guards);clear(diag[v]);CommandGraph swiglu;addSiLUMultiply(swiglu,caps[0][v].bf16,caps[1][v].bf16,expectedAct[v],diag[v],rows,640,10);(void)backend.submitCommand(swiglu.dispatches());swigluMismatch[v]=different(v?candidateAct:baselineAct,expectedAct[v],uint64_t{rows}*10*640,2);exact&=!swigluMismatch[v]&&diagnostic(diag[v])==0x80000000u;}
      std::vector<uint8_t>savedAct(candidateAct.sizeBytes()),savedDown(candidateDown.sizeBytes()),savedBaselineDown(scratch.scatteredDown.sizeBytes());
      std::memcpy(savedAct.data(),candidateAct.contents(),savedAct.size());std::memcpy(savedDown.data(),candidateDown.contents(),savedDown.size());std::memcpy(savedBaselineDown.data(),scratch.scatteredDown.contents(),savedBaselineDown.size());
      for(auto b:{scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.canonicalToPacked,scratch.buckets.packedInputs,scratch.buckets.jobOffsets,scratch.buckets.jobCount,scratch.buckets.tileJobs,scratch.packedActivated,scratch.scatteredDown})std::memset(b.contents(),0xa5,b.sizeBytes());
      std::memset(candidateAct.contents(),0xa5,candidateAct.sizeBytes());std::memset(candidateDown.contents(),0xa5,candidateDown.sizeBytes());clear(diag[1]);(void)backend.submitCommand(chain[1].dispatches());
      const bool staleIndependent=!std::memcmp(savedAct.data(),candidateAct.contents(),savedAct.size())&&!std::memcmp(savedDown.data(),candidateDown.contents(),savedDown.size())&&diagnostic(diag[1])==0x80000000u;
      exact&=staleIndependent;
      const uint64_t after=backend.memoryStats().allocatedBytes;require(after>=before&&after-before<=planned,"gathered oracle exceeded reservation");reservation->commit();
      bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      bool inputsImmutable=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      bool timingPass=exact&&finiteBoundSign&&canaries&&inputsImmutable;std::array<std::vector<CommandTiming>,2>times;
      const bool timingAttempted=timingPass;
      if(timingPass) {for(uint32_t v=0;v<2;++v){clear(diag[v]);(void)backend.submitCommand(chain[v].dispatches());timingPass&=diagnostic(diag[v])==0x80000000u;}
        for(uint32_t pair=0;pair<pairs;++pair)for(uint32_t step=0;step<2;++step){const uint32_t v=(pair+step)%2;clear(diag[v]);times[v].push_back(backend.submitCommand(chain[v].dispatches()));timingPass&=diagnostic(diag[v])==0x80000000u;}}
      canaries&=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});inputsImmutable&=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      const bool candidateReplayExact=!std::memcmp(savedAct.data(),candidateAct.contents(),savedAct.size())&&!std::memcmp(savedDown.data(),candidateDown.contents(),savedDown.size());
      const bool baselineReplayExact=!timingAttempted||!std::memcmp(savedBaselineDown.data(),scratch.scatteredDown.contents(),savedBaselineDown.size());
      exact&=candidateReplayExact&&baselineReplayExact;timingPass&=canaries&&inputsImmutable&&candidateReplayExact&&baselineReplayExact;layer.checkImmutableHashes(immutable);
      std::ofstream out(output);require(bool(out),"cannot create NEW gathered report");out<<std::setprecision(17);
      out<<"{\"schema\":\"splash-private-gathered-mpp-one-layer-v1\",\"exact_old_mpp_parity_pass\":"<<(exact?"true":"false")<<",\"timing_pass\":"<<(timingPass?"true":"false")
        <<",\"baseline_f64_strict_pass\":"<<(baselineF64Strict?"true":"false")<<",\"candidate_f64_strict_pass\":"<<(candidateF64Strict?"true":"false")<<",\"finite_bound_sign_pass\":"<<(finiteBoundSign?"true":"false")
        <<",\"model_quality_qualified\":false,\"real_normalized_decode_activation_qualified\":false,\"layer\":"<<layerIndex<<",\"rows\":"<<rows<<",\"route_pattern\":"<<splash::json::quote(pattern)<<",\"rank_shift_fixture\":"<<rankShift
        <<",\"hidden_policy\":"<<splash::json::quote(hiddenPath.empty()?"synthetic normalized BF16; not actual decode activation":"caller fixture; provenance independently required")<<",\"one_shared_readonly_layer\":true,\"mapped_layer_bytes\":"<<metadata.layers[layerIndex].bytes
        <<",\"input_base_mod128\":"<<(reinterpret_cast<uintptr_t>(x.contents())%128)<<",\"activated_base_mod128\":"<<(reinterpret_cast<uintptr_t>(candidateAct.contents())%128)
        <<",\"activated_mismatches\":"<<actMismatch<<",\"chain_down_mismatches\":"<<downMismatch<<",\"swiglu_mismatches\":["<<swigluMismatch[0]<<','<<swigluMismatch[1]<<"]"
        <<",\"stale_bucket_independent\":"<<(staleIndependent?"true":"false")<<",\"canaries\":"<<(canaries?"true":"false")<<",\"inputs_immutable\":"<<(inputsImmutable?"true":"false")<<",\"weights_immutable\":true,\"projection_reports\":[";
      for(uint32_t plane=0;plane<3;++plane){if(plane)out<<',';out<<"{\"plane\":"<<plane<<",\"f32_dot_mismatches\":"<<mismatches[plane][0]<<",\"f32_scaled_mismatches\":"<<mismatches[plane][1]<<",\"bf16_mismatches\":"<<mismatches[plane][2]<<",\"baseline_f64\":";sums[plane][0].json(out);out<<",\"gathered_f64\":";sums[plane][1].json(out);out<<'}';}
      out<<"],\"timing_policy\":\"EXACT old-MPP raw/scaled/BF16/stage parity plus frozen conservative opaqueMPP F64 absolute/finite/sign bounds; baseline strict F64 failures remain failures and do not claim exact F64 qualification\",\"timing_scope\":\"warm alternating shipping chains; old-MPP includes bucketPrelude/downPrepare/scatter; gathered two direct dispatches; untimed probes/CPUcomparison/hashes excluded\",\"timings\":[";
      for(uint32_t v=0;v<2;++v){if(v)out<<',';out<<"{\"variant\":"<<splash::json::quote(v?"gatheredMPP":"oldMPP")<<",\"samples\":[";for(size_t i=0;i<times[v].size();++i){if(i)out<<',';out<<"{\"wall_ms\":"<<times[v][i].wallSeconds*1000<<",\"gpu_ms\":"<<times[v][i].gpuSeconds*1000<<'}';}out<<"]}";}out<<"]}\n";backend.stop();
      std::cout<<"{\"exact_old_mpp_parity_pass\":"<<(exact?"true":"false")<<",\"timing_pass\":"<<(timingPass?"true":"false")<<",\"baseline_f64_strict_pass\":"<<(baselineF64Strict?"true":"false")<<",\"report\":"<<splash::json::quote(output.string())<<"}\n";return timingPass?0:2;
    }
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
