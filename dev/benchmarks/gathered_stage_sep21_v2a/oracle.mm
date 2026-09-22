// Root-only GPU qualification. CPU entry creates no device and reads no payload.
#define main preserved_qmv_oracle_entry_not_invoked
#include "prefill4k_allrows_qmv_oracle.mm"
#undef main
#include "flash/FlashGatheredMPP.hpp"

namespace {
constexpr uint32_t variants = 5;
constexpr std::array<const char *, variants> names{
    "oldMPP", "stagedGatheredSG4", "localScanGatheredSG4",
    "localScanGatheredSG1NumericalAlternative", "localScanGatheredSG2NumericalAlternative"};
constexpr std::array<const char *, variants> taps{
    "", "flash_gathered_stage_projection_probe", "flash_gathered_mpp_projection_probe",
    "prefill_moe_sep21_gathered_sg1_projection_probe", "prefill_moe_sep21_gathered_sg2_projection_probe"};
constexpr std::array<uint32_t, variants> threadCounts{128,128,128,32,64};

uint64_t different(MetalBuffer a, MetalBuffer b, uint64_t elements, uint32_t bytes) {
  const auto *aa = static_cast<const uint8_t *>(a.contents());
  const auto *bb = static_cast<const uint8_t *>(b.contents());
  uint64_t count = 0;
  for (uint64_t i = 0; i < elements; ++i) count += std::memcmp(aa+i*bytes,bb+i*bytes,bytes) != 0;
  return count;
}
std::vector<uint16_t> normalizedFixture(uint32_t rows) {
  std::vector<uint16_t> result(uint64_t{rows}*2560);
  for (uint32_t row = 0; row < rows; ++row) {
    double squares = 0;
    for (uint32_t k = 0; k < 2560; ++k) {
      const double x = double(int32_t((k*173+row*31)%211)-105); squares += x*x;
    }
    const double normalization = std::sqrt(squares/2560);
    for (uint32_t k = 0; k < 2560; ++k)
      result[row*2560+k] = reference::bf16FromF64(double(int32_t((k*173+row*31)%211)-105)/normalization);
  }
  return result;
}
std::vector<int64_t> fixtureRoutes(uint32_t rows, std::string_view pattern) {
  require(pattern=="mixed" || pattern=="repeated" || pattern=="spread" || pattern=="permuted",
      "unsupported frozen route pattern");
  std::vector<int64_t> ids(uint64_t{rows}*10);
  for (uint32_t row = 0; row < rows; ++row) for (uint32_t slot = 0; slot < 10; ++slot) {
    const uint32_t shift = pattern=="repeated" ? 0 : pattern=="mixed" ? (row%2 ? row*73 : 0) : row*73;
    const uint32_t selection = pattern=="permuted" ? (slot*3+row)%10 : slot;
    ids[row*10+slot] = (selection*53+shift)%512;
    if (pattern=="repeated" && slot==9) ids[row*10+slot] = 511;
  }
  return ids;
}
void sanitize(CommandGraph &graph, const one::OneLayerPayload &layer, MetalBuffer input,
    MetalBuffer ids, MetalBuffer safe, MetalBuffer diag, uint32_t rows, bool down) {
  graph.add(down ? "flash_gathered_stage_sanitize_down" : "flash_gathered_stage_sanitize_hidden",
      {input,layer.ranks,ids,safe,diag}, FlashGatheredMPPParams{rows,10,512,0},
      {down?3u:10u,rows,down?10u:1u},{256,1,1});
}
void gatheredChain(CommandGraph &graph, const one::OneLayerPayload &layer,
    MetalBuffer hidden, MetalBuffer ids, MetalBuffer activated, MetalBuffer down,
    MetalBuffer diag, MetalBuffer safeHidden, MetalBuffer safeDown, uint32_t rows, uint32_t variant) {
  require(variant && variant < variants, "gathered variant outside bounds");
  const FlashGatheredMPPParams p{rows,10,512,0};
  std::string gateName, downName;
  MetalBuffer gateInput = hidden, downInput = activated;
  if (variant==1) {
    sanitize(graph,layer,hidden,ids,safeHidden,diag,rows,false);
    gateInput=safeHidden; downInput=safeDown;
    gateName="flash_gathered_stage_gate_up_m16_n64_sg4";
    downName="flash_gathered_stage_down_m16_n64_sg4";
  } else if (variant==2) {
    gateName="flash_gathered_mpp_gate_up_m16_n64_sg4";
    downName="flash_gathered_mpp_down_m16_n64_sg4";
  } else {
    const std::string sg=std::to_string(variant-2), prefix="prefill_moe_sep21_gathered_sg"+sg;
    gateName=prefix+"_gate_up_m16_n64_sg"+sg;
    downName=prefix+"_down_m16_n64_sg"+sg;
  }
  graph.add(gateName,{gateInput,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],
      layer.ranks,ids,activated,diag},p,{10,rows,10},{threadCounts[variant],1,1});
  if (variant==1) sanitize(graph,layer,activated,ids,safeDown,diag,rows,true);
  graph.add(downName,{downInput,layer.codes[2],layer.scales[2],layer.ranks,ids,down,diag},
      p,{40,rows,10},{threadCounts[variant],1,1});
}
void cpuSelfTest() {
  require(sizeof(FlashGatheredMPPParams)==16 && sizeof(FlashQMVProbeParams)==32,"ABI differs");
  for (uint32_t rows : {1u,2u,4u,8u,16u})
    for (std::string_view pattern : {"mixed","repeated","spread","permuted"})
      require(gathered_i8_qmv::fixtureIDDiagnostics(fixtureRoutes(rows,pattern),rows)==0,
          "frozen fixture has duplicate or invalid ID");
  uint32_t exceptional=0;
  for (uint32_t raw=0;raw<65536;++raw) {
    const uint16_t bits=uint16_t(raw); const bool bad=(bits&0x7f80u)==0x7f80u;
    const uint16_t safe=bad?0:bits; exceptional+=bad;
    require(bad?safe==0:safe==bits,"bitwise sanitizer policy differs");
  }
  require(exceptional==256,"BF16 exceptional pattern count differs");
  std::cout << "{\"cpu_checks\":\"passed\",\"variants\":5,\"bf16_bit_policy_patterns\":65536,"
      "\"gpu_work\":false,\"hashing_performed\":false,\"exact_old_mpp_parity\":\"pending\"}\n";
}
}

int main(int argc, char **argv) {
  try {
    if (argc==2 && std::string_view(argv[1])=="--cpu-self-test") { cpuSelfTest(); return 0; }
    require(argc>=7 && std::string_view(argv[1])=="--gpu",
        "usage: gathered-stage-v2 --gpu metallib fullStore layer rows NEWreport [--pattern mixed|repeated|spread|permuted] [--hidden bf16] [--ids i64] [--pairs1..9] [--warm-cycles1..16] [--rank-shift0..511]");
    const uint32_t layerIndex=numeric(argv[4],0,47), rows=numeric(argv[5],1,16);
    const std::filesystem::path output(argv[6]);
    require(!std::filesystem::exists(output),"NEW report path required");
    std::string pattern="mixed",hiddenPath,idsPath; uint32_t pairs=9,warmCycles=5,rankShift=0;
    for (int i=7;i<argc;i+=2) {
      require(i+1<argc,"missing CLI value"); const std::string option(argv[i]);
      if (option=="--pattern") pattern=argv[i+1];
      else if (option=="--hidden") hiddenPath=argv[i+1];
      else if (option=="--ids") idsPath=argv[i+1];
      else if (option=="--pairs") pairs=numeric(argv[i+1],1,9);
      else if (option=="--warm-cycles") warmCycles=numeric(argv[i+1],1,16);
      else if (option=="--rank-shift") rankShift=numeric(argv[i+1],0,511);
      else throw std::invalid_argument("unknown gathered-stage-v2 option");
    }
    auto hidden=normalizedFixture(rows); auto ids=fixtureRoutes(rows,pattern);
    if (!hiddenPath.empty()) hidden=fixture<uint16_t>(hiddenPath,hidden.size());
    if (!idsPath.empty()) ids=fixture<int64_t>(idsPath,ids.size());
    require(gathered_i8_qmv::fixtureIDDiagnostics(ids,rows)==0,"finite oracle requires valid unique I64 IDs");
    const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],
        "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
        "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
    require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1",
        "certified Full512 metadata differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1); ::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);
    @autoreleasepool {
      MetalBackend backend(argv[2]);
      const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=4ULL<<30;
      require(physical>reserve,"host reserve cannot be protected");
      splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(128ULL<<20);
      auto reservation=governor.tryReserve(planned); require(bool(reservation),"admission denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;
      auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      if (rankShift) {
        auto *r=static_cast<uint32_t *>(layer.ranks.contents());
        for (uint32_t id=0;id<512;++id) r[id]=(id+rankShift)%512;
      }
      const auto immutable=layer.immutableHashes(); std::vector<Guard> guards;
      auto x=upload(backend,hidden,guards),idBuffer=upload(backend,ids,guards);
      auto scratch=allocateMoEBlockedScratch(backend,rows,10); guardScratch(backend,scratch,guards);
      std::array<MetalBuffer,variants>diag,act,down;
      const uint64_t actElements=uint64_t{rows}*10*640,downElements=uint64_t{rows}*10*2560;
      for (uint32_t v=0;v<variants;++v) {
        diag[v]=guarded(backend,4,guards); act[v]=guarded(backend,actElements*2,guards);
        if (v) down[v]=guarded(backend,downElements*2,guards);
      }
      down[0]=scratch.scatteredDown;
      auto safeHidden=guarded(backend,x.sizeBytes(),guards),safeDown=guarded(backend,actElements*2,guards);
      const uint64_t stageBytes=safeHidden.sizeBytes()+safeDown.sizeBytes();
      std::array<CommandGraph,variants> chain;
      mppChain(chain[0],layer,x,idBuffer,scratch,diag[0],rows);
      for (uint32_t v=1;v<variants;++v)
        gatheredChain(chain[v],layer,x,idBuffer,act[v],down[v],diag[v],safeHidden,safeDown,rows,v);
      std::array<bool,variants>chainDiagnostics;
      for (uint32_t v=0;v<variants;++v) { clear(diag[v]); (void)backend.submitCommand(chain[v].dispatches());
        chainDiagnostics[v]=diagnostic(diag[v])==0x80000000u; }
      CommandGraph unpack;
      unpack.add("flash_qmv_probe_unpack_activation",{scratch.packedActivated,scratch.buckets.canonicalToPacked,act[0],diag[0]},
          FlashQMVProbeParams{rows,10,2560,640,0,0,512,0},{3,rows*10,1},{256,1,1});
      (void)backend.submitCommand(unpack.dispatches());
      std::array<bool,variants>rawExact,bf16Exact,stageExact,finiteBoundSign,f64Strict,timingEligible;
      rawExact.fill(true); bf16Exact.fill(true); stageExact.fill(true);
      finiteBoundSign.fill(true); f64Strict.fill(true); timingEligible.fill(false);
      std::array<std::array<Outputs,variants>,3>caps;
      std::array<std::array<Summary,variants>,3>sums;
      std::array<std::array<std::array<uint64_t,3>,variants>,3> mismatches{};
      for (uint32_t plane=0;plane<3;++plane) {
        const uint32_t n=plane==2?2560:640,k=plane==2?640:2560;
        const auto input=plane==2?act[0]:x;
        for (uint32_t v=0;v<variants;++v) {
          auto &c=caps[plane][v]; c=outputs(backend,uint64_t{rows}*10*n,guards);
          clear(c.diag); CommandGraph probe;
          const FlashQMVProbeParams p{rows,10,k,n,uint32_t(plane==2),0,512,0};
          if (!v) probe.add("flash_qmv_probe_mpp_m16",{
              plane==2?scratch.packedActivated:scratch.buckets.packedInputs,layer.codes[plane],layer.scales[plane],
              layer.ranks,scratch.buckets.offsets,scratch.buckets.tileJobs,scratch.buckets.jobCount,
              scratch.buckets.routeMap,c.dot,c.scaled,c.bf16,c.diag},p,{n/64,rows*10,1},{128,1,1});
          else {
            auto operand=input;
            if (v==1) {
              operand=plane==2?safeDown:safeHidden;
              sanitize(probe,layer,input,idBuffer,operand,c.diag,rows,plane==2);
            }
            probe.add(taps[v],{operand,layer.codes[plane],layer.scales[plane],layer.ranks,idBuffer,
                c.dot,c.scaled,c.bf16,c.diag},p,{n/64,rows,10},{threadCounts[v],1,1});
          }
          (void)backend.submitCommand(probe.dispatches());
          sums[plane][v]=evaluate(layer,plane,rows,ids,input,c,0);
          finiteBoundSign[v]&=sums[plane][v].finiteBoundSignPass()&&diagnostic(c.diag)==0x80000000u;
          f64Strict[v]&=sums[plane][v].pass();
          const uint64_t elements=uint64_t{rows}*10*n;
          mismatches[plane][v]={different(caps[plane][0].dot,c.dot,elements,4),
              different(caps[plane][0].scaled,c.scaled,elements,4),different(caps[plane][0].bf16,c.bf16,elements,2)};
          rawExact[v]&=!mismatches[plane][v][0]&&!mismatches[plane][v][1];
          bf16Exact[v]&=!mismatches[plane][v][2];
        }
      }
      std::array<std::array<uint64_t,3>,variants> stageMismatches{};
      for (uint32_t v=0;v<variants;++v) {
        stageMismatches[v][0]=different(act[0],act[v],actElements,2);
        stageMismatches[v][1]=different(down[0],down[v],downElements,2);
        auto expected=guarded(backend,actElements*2,guards); clear(diag[v]); CommandGraph swiglu;
        addSiLUMultiply(swiglu,caps[0][v].bf16,caps[1][v].bf16,expected,diag[v],rows,640,10);
        (void)backend.submitCommand(swiglu.dispatches());
        stageMismatches[v][2]=different(act[v],expected,actElements,2);
        stageExact[v]=!stageMismatches[v][0]&&!stageMismatches[v][1]&&!stageMismatches[v][2]&&
            chainDiagnostics[v]&&diagnostic(diag[v])==0x80000000u;
      }
      std::array<std::vector<uint8_t>,variants>savedAct,savedDown;
      for (uint32_t v=0;v<variants;++v) {
        savedAct[v].resize(act[v].sizeBytes()); savedDown[v].resize(down[v].sizeBytes());
        std::memcpy(savedAct[v].data(),act[v].contents(),savedAct[v].size());
        std::memcpy(savedDown[v].data(),down[v].contents(),savedDown[v].size());
      }
      for (auto b : {scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,
          scratch.buckets.canonicalToPacked,scratch.buckets.packedInputs,scratch.buckets.jobOffsets,
          scratch.buckets.jobCount,scratch.buckets.tileJobs,scratch.packedActivated,scratch.scatteredDown,
          safeHidden,safeDown}) std::memset(b.contents(),0xa5,b.sizeBytes());
      std::array<bool,variants>replayExact; replayExact.fill(true);
      for (uint32_t v=1;v<variants;++v) {
        std::memset(act[v].contents(),0xa5,act[v].sizeBytes());
        std::memset(down[v].contents(),0xa5,down[v].sizeBytes()); clear(diag[v]);
        (void)backend.submitCommand(chain[v].dispatches());
        replayExact[v]=!std::memcmp(savedAct[v].data(),act[v].contents(),savedAct[v].size())&&
            !std::memcmp(savedDown[v].data(),down[v].contents(),savedDown[v].size())&&diagnostic(diag[v])==0x80000000u;
      }
      // Explicit baseline poisoned replay regenerates all original buckets and
      // both original outputs. It is a replay guarantee, not bucket independence.
      std::memset(act[0].contents(),0xa5,act[0].sizeBytes()); clear(diag[0]);
      (void)backend.submitCommand(chain[0].dispatches());
      (void)backend.submitCommand(unpack.dispatches());
      replayExact[0]=!std::memcmp(savedAct[0].data(),act[0].contents(),savedAct[0].size())&&
          !std::memcmp(savedDown[0].data(),down[0].contents(),savedDown[0].size())&&diagnostic(diag[0])==0x80000000u;
      bool stageFiniteBits=!std::memcmp(hidden.data(),safeHidden.contents(),safeHidden.sizeBytes())&&
          !std::memcmp(act[1].contents(),safeDown.contents(),safeDown.sizeBytes());
      const uint64_t after=backend.memoryStats().allocatedBytes;
      require(after>=before && after-before<=planned,"oracle exceeded reservation"); reservation->commit();
      bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      bool inputsImmutable=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&
          !std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      std::array<std::vector<CommandTiming>,variants>times;
      for (uint32_t v=0;v<variants;++v) {
        timingEligible[v]=finiteBoundSign[0]&&finiteBoundSign[v]&&bf16Exact[v]&&stageExact[v]&&
            replayExact[v]&&canaries&&inputsImmutable&&(v!=1||stageFiniteBits);
      }
      // Numerical taps and CPU reference scans may leave GPU clocks idle. Warm
      // every eligible shipping chain in rotating order immediately before the
      // matched samples, with all copy/sanitize work included in each chain.
      for (uint32_t cycle=0;cycle<warmCycles;++cycle) for (uint32_t step=0;step<variants;++step) {
        const uint32_t v=(cycle+step)%variants;
        if (timingEligible[v]) { clear(diag[v]); (void)backend.submitCommand(chain[v].dispatches());
          timingEligible[v]&=diagnostic(diag[v])==0x80000000u; }
      }
      for (uint32_t pair=0;pair<pairs;++pair) for (uint32_t step=0;step<variants;++step) {
        const uint32_t v=(pair+step)%variants;
        if (timingEligible[v]) { clear(diag[v]); times[v].push_back(backend.submitCommand(chain[v].dispatches()));
          timingEligible[v]&=diagnostic(diag[v])==0x80000000u; }
      }
      canaries&=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.valid();});
      inputsImmutable&=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&
          !std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());
      for (uint32_t v=0;v<variants;++v) {
        if (!times[v].empty()) replayExact[v]&=!std::memcmp(savedDown[v].data(),down[v].contents(),savedDown[v].size());
        if (v) replayExact[v]&=!std::memcmp(savedAct[v].data(),act[v].contents(),savedAct[v].size());
        timingEligible[v]&=canaries&&inputsImmutable&&replayExact[v];
      }
      stageFiniteBits&=!std::memcmp(hidden.data(),safeHidden.contents(),safeHidden.sizeBytes())&&
          !std::memcmp(act[1].contents(),safeDown.contents(),safeDown.sizeBytes());
      timingEligible[1]&=stageFiniteBits; layer.checkImmutableHashes(immutable);
      std::ofstream out(output); require(bool(out),"cannot create NEW report"); out<<std::setprecision(17);
      out<<"{\"schema\":\"splash-private-gathered-stage-one-layer-v2a\",\"layer\":"<<layerIndex<<",\"rows\":"<<rows
          <<",\"route_pattern\":"<<splash::json::quote(pattern)<<",\"rank_shift_fixture\":"<<rankShift
          <<",\"hidden_policy\":"<<splash::json::quote(hiddenPath.empty()?"synthetic normalized BF16; not actual decode activation":"caller fixture; provenance independently required")
          <<",\"one_shared_readonly_layer\":true,\"mapped_layer_bytes\":"<<metadata.layers[layerIndex].bytes
          <<",\"admitted_canonical_stage_bytes\":"<<stageBytes<<",\"warm_cycles\":"<<warmCycles<<",\"matched_pairs\":"<<pairs
          <<",\"finite_stage_bitcopy_pass\":"<<(stageFiniteBits?"true":"false")
          <<",\"canaries\":"<<(canaries?"true":"false")<<",\"inputs_immutable\":"<<(inputsImmutable?"true":"false")
          <<",\"weights_immutable\":true,\"model_quality_qualified\":false,\"real_normalized_decode_activation_qualified\":false"
          <<",\"timing_policy\":\"Per-variant exact BF16 projections and activated/down/compiledSwiGLU stage parity, conservative opaqueMPP F64 finite/sign/absolute bounds and diagnostics/guards/replay; raw/scaled F32 differences remain numerical alternatives; strict F64 failures remain failures; no model qualification\""
          <<",\"timing_scope\":\"warm rotating matched shipping chains; oldMPP includes bucketPrelude/downPrepare/scatter; stagedSG4 includes both sanitize dispatches; localScanSG4/SG1/SG2 include original perCTA scan/fallback; untimed probes/CPU comparisons/immutable hashes excluded\",\"variants\":[";
      bool allTiming=true;
      for (uint32_t v=0;v<variants;++v) {
        allTiming&=timingEligible[v]; if (v) out<<',';
        const bool exact=rawExact[v]&&bf16Exact[v]&&stageExact[v]&&replayExact[v]&&canaries&&inputsImmutable&&(v!=1||stageFiniteBits);
        out<<"{\"variant\":"<<splash::json::quote(names[v])<<",\"exact_old_mpp_raw_scaled_pass\":"<<(rawExact[v]?"true":"false")
            <<",\"exact_old_mpp_bf16_projection_pass\":"<<(bf16Exact[v]?"true":"false")<<",\"exact_old_mpp_stage_pass\":"<<(stageExact[v]?"true":"false")
            <<",\"exact_old_mpp_parity_pass\":"<<(exact?"true":"false")<<",\"finite_bound_sign_pass\":"<<(finiteBoundSign[v]?"true":"false")
            <<",\"f64_strict_pass\":"<<(f64Strict[v]?"true":"false")<<",\"stale_buffer_replay_pass\":"<<(replayExact[v]?"true":"false")
            <<",\"timing_pass\":"<<(timingEligible[v]?"true":"false")<<",\"activated_mismatches\":"<<stageMismatches[v][0]
            <<",\"chain_down_mismatches\":"<<stageMismatches[v][1]<<",\"swiglu_mismatches\":"<<stageMismatches[v][2]<<",\"projection_reports\":[";
        for (uint32_t plane=0;plane<3;++plane) {
          if (plane) out<<',';
          out<<"{\"plane\":"<<plane<<",\"f32_dot_mismatches\":"<<mismatches[plane][v][0]
              <<",\"f32_scaled_mismatches\":"<<mismatches[plane][v][1]<<",\"bf16_mismatches\":"<<mismatches[plane][v][2]<<",\"f64\":";
          sums[plane][v].json(out); out<<'}';
        }
        out<<"],\"samples\":[";
        for (size_t i=0;i<times[v].size();++i) {
          if (i) out<<',';
          out<<"{\"wall_ms\":"<<times[v][i].wallSeconds*1000<<",\"gpu_ms\":"<<times[v][i].gpuSeconds*1000<<'}';
        }
        out<<"]}";
      }
      out<<"]}\n"; backend.stop();
      std::cout<<"{\"all_variant_timing_pass\":"<<(allTiming?"true":"false")<<",\"staged_exact_old_mpp_parity_pass\":"
          <<(rawExact[1]&&bf16Exact[1]&&stageExact[1]&&replayExact[1]&&canaries&&inputsImmutable&&stageFiniteBits?"true":"false")
          <<",\"report\":"<<splash::json::quote(output.string())<<"}\n";
      return timingEligible[1]?0:2;
    }
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
