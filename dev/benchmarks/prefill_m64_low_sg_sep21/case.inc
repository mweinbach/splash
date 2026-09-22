bool runCase(MetalBackend &backend,const NativeStoreView &store,
    uint32_t rows,const std::string &pattern,uint32_t pairs,
    double maxL2,double minCosine,bool strict,std::ostream &out) {
  require(strict && pairs>=4 && !(pairs%2),"M64 low-SIMD strict parity/even timing pairs required");
  const uint32_t layer=store.layerIndex;
  std::vector<uint32_t> hot(512);for (uint32_t e=0;e<512;++e) hot[e]=e;
  std::vector<uint16_t> hidden(uint64_t{rows}*2560);auto ids=patternIDs(rows,hot,pattern);
  const char *rawInput=std::getenv("FLASH_INT8_STORE_INPUT"),*rawIDs=std::getenv("FLASH_INT8_STORE_ROUTE_IDS");
  require(bool(rawInput)==bool(rawIDs),"raw hidden and IDs must be supplied together");
  if (rawInput) {hidden=readFile<uint16_t>(rawInput,hidden.size());ids=readFile<int64_t>(rawIDs,ids.size());}
  else hidden=syntheticHidden(rows,inputPolicy());
  const auto rms=rowRMS(hidden,rows);const auto packed=ref::pack(hidden,ids,rows,10,kSticky);
  require(packed.diagnostic==kSticky,"fixture invalid/duplicate top10 ownership");
  const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,64)};
  const auto input=upload(backend,hidden,"M64 low-SIMD shared BF16 input");
  const auto expertIDs=upload(backend,ids,"M64 low-SIMD shared original I64 IDs");
  std::vector<uint16_t> weights(ids.size());
  for (uint64_t i=0;i<ids.size();++i) weights[i]=bf16(float((i%10)+1)/55.0f);
  const auto route=upload(backend,weights,"M64 low-SIMD unequal route weights");
  const auto shared=upload(backend,std::vector<uint16_t>(uint64_t{rows}*2560,0),"M64 shared expert zeros");
  const auto sharedGate=upload(backend,std::vector<uint16_t>(rows,0),"M64 shared gate zeros");
  std::array<FlashMoEBlockedScratch,2> scratch;std::array<CommandGraph,2> native;
  std::array<MetalBuffer,2> output,diagnostic;std::vector<Guard> guards;
  for (uint32_t which=0;which<2;++which) {
    const auto tile=which ? FlashMoEBlockedTile::M64N64 : FlashMoEBlockedTile::M32N64;
    scratch[which]=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,scratch[which],guards);
    output[which]=guarded(backend,uint64_t{rows}*2560*2,guards);
    diagnostic[which]=guarded(backend,4,guards);*static_cast<uint32_t *>(diagnostic[which].contents())=kSticky;
    addMoEBlockedPack(native[which],input,expertIDs,scratch[which],diagnostic[which],rows,tile);
    store.addGateUp(native[which],layer,scratch[which],diagnostic[which],rows,tile);
    store.addDownScatter(native[which],layer,scratch[which],diagnostic[which],rows,tile);
    addCombine(native[which],scratch[which].scatteredDown,expertIDs,route,shared,sharedGate,
        output[which],diagnostic[which],rows,2560,512,10);
  }
  require(native[0].dispatches().size()==native[1].dispatches().size(),"Native M64 adds unexpected dispatches");
  const uint64_t routes=uint64_t{rows}*10;
  ProbeBuffers controlProbe(backend,routes,guards),candidateProbe(backend,routes,guards);
  const auto control=m64LowCommands(native[0].dispatches(),rows,0);
  const auto controlProbeCommands=m64LowCommands(native[0].dispatches(),rows,0,&controlProbe);
  struct Result final {
    uint32_t sg=0;std::vector<ComputeDispatch> commands,probeCommands;
    std::array<Comparison,3> initial,final,probeChain;ProbeComparisons probes;
    std::vector<CommandTiming> controlTimes,candidateTimes;std::vector<std::string> failures;
    double controlWarmGPU=0,candidateWarmGPU=0;uint32_t warmPairs=0;
    bool initialChecked=false,finalChecked=false,probeChecked=false,eligible=false;
    std::string outputHash;
  };
  std::array<Result,2> results;
  for (uint32_t i=0;i<2;++i) {
    auto &r=results[i];r.sg=i ? 2 : 4;
    r.commands=m64LowCommands(native[1].dispatches(),rows,r.sg);
    r.probeCommands=m64LowCommands(native[1].dispatches(),rows,r.sg,&candidateProbe);
  }
  const auto healthy=[&] {
    for (const auto &d:diagnostic) require(*static_cast<const uint32_t *>(d.contents())==kSticky,"M64 sticky diagnostic changed");
    for (const auto &g:guards) require(g.clean(),"M64 guarded scratch/probe/output overwritten");
  };
  const auto buckets=[&](uint32_t which) {checkBuckets(scratch[which],packed,jobs[which]);};
  const auto poison=[&] {
    std::memset(scratch[1].packedActivated.contents(),0xff,scratch[1].packedActivated.sizeBytes());
    std::memset(scratch[1].scatteredDown.contents(),0xff,scratch[1].scatteredDown.sizeBytes());
    std::memset(output[1].contents(),0xff,output[1].sizeBytes());
    *static_cast<uint32_t *>(diagnostic[1].contents())=kSticky;
  };
  const auto chain=[&] {
    return std::array<Comparison,3>{compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine),
        compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine),
        compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine)};
  };
  const auto fail=[&](Result &r,const char *reason) {r.failures.emplace_back(reason);r.eligible=false;};
  (void)backend.submitCommand(control);healthy();buckets(0);
  const std::array<std::string,3> controlHashes{digest(scratch[0].packedActivated),digest(scratch[0].scatteredDown),digest(output[0])};
  controlProbe.poison();(void)backend.submitCommand(controlProbeCommands);healthy();buckets(0);
  const auto controlFinite=compareProbes(controlProbe,controlProbe,routes);
  const bool controlProbeQualified=exactProbes(controlFinite) &&
      digest(scratch[0].packedActivated)==controlHashes[0] && digest(scratch[0].scatteredDown)==controlHashes[1] && digest(output[0])==controlHashes[2];
  for (auto &r:results) {
    if (!controlProbeQualified) {fail(r,"qualified control probe changed complete BF16 bytes or nonfinite raw/scaled probes");continue;}
    poison();candidateProbe.poison();(void)backend.submitCommand(r.probeCommands);
    healthy();buckets(1);r.probeChain=chain();r.probes=compareProbes(controlProbe,candidateProbe,routes);r.probeChecked=true;
    if (!exactChain(r.probeChain) || !exactProbes(r.probes)) {fail(r,"native M64 raw F32/scaled BF16 or complete BF16 chain differs");continue;}
    poison();(void)backend.submitCommand(r.commands);healthy();buckets(1);
    r.initial=chain();r.initialChecked=true;r.eligible=exactChain(r.initial);
    if (!r.eligible) fail(r,"native M64 non-probe complete BF16 chain differs before timing");
  }
  // Mandatory paired GPU warm-up after ALL CPU comparisons/hash reads.
  // Submissions only until completed; no .contents()/guards/checksum scans.
  for (auto &r:results) if (r.eligible) {
    for (;r.controlWarmGPU<.100 || r.candidateWarmGPU<.100;++r.warmPairs) {
      require(r.warmPairs<10000,"M64 GPU warm duration unavailable");CommandTiming ct,vt;
      if (r.warmPairs&1) {vt=backend.submitCommand(r.commands);ct=backend.submitCommand(control);}
      else {ct=backend.submitCommand(control);vt=backend.submitCommand(r.commands);}
      require(ct.gpuSeconds>0 && std::isfinite(ct.gpuSeconds) && vt.gpuSeconds>0 && std::isfinite(vt.gpuSeconds),"M64 GPU warm timestamps unavailable");
      r.controlWarmGPU+=ct.gpuSeconds;r.candidateWarmGPU+=vt.gpuSeconds;
    }
  }
  // Timed region: no CPU buffer access; alternating pair order and rotated
  // candidate position evenly cover both positions for each variant.
  for (uint32_t pair=0;pair<pairs;++pair) for (uint32_t order=0;order<results.size();++order) {
    const uint32_t position=(pair+order)%results.size();auto &r=results[position];if (!r.eligible) continue;
    if ((pair+position)&1) {r.candidateTimes.push_back(backend.submitCommand(r.commands));r.controlTimes.push_back(backend.submitCommand(control));}
    else {r.controlTimes.push_back(backend.submitCommand(control));r.candidateTimes.push_back(backend.submitCommand(r.commands));}
  }
  healthy();buckets(0);buckets(1);
  require(digest(output[0])==controlHashes[2],"qualified baseline changed during timing");
  bool pass=controlProbeQualified;
  for (auto &r:results) {
    if (r.eligible) {poison();(void)backend.submitCommand(r.commands);healthy();buckets(1);r.final=chain();r.finalChecked=true;r.outputHash=digest(output[1]);if (!exactChain(r.final)) fail(r,"native M64 full BF16 chain changed after timing");}
    pass=pass && r.eligible && r.failures.empty();
  }
  require(!std::memcmp(hidden.data(),input.contents(),hidden.size()*2) && !std::memcmp(ids.data(),expertIDs.contents(),ids.size()*8),"M64 shared fixtures mutated");
  out<<"{\"layer\":"<<layer<<",\"rows\":"<<rows<<",\"pattern\":"<<splash::json::quote(rawInput ? "raw-fixture" : pattern)
      <<",\"input_row_rms_min\":"<<rms[0]<<",\"input_row_rms_max\":"<<rms[1]
      <<",\"native_job_tiles\":[32,64],\"active_jobs\":["<<jobs[0].count<<','<<jobs[1].count<<']'
      <<",\"job_capacities\":["<<jobs[0].entries.size()<<','<<jobs[1].entries.size()<<']'
      <<",\"control_policy\":\"qualified SG2 fixedK128 M32 jobs with M16 adaptive tail\""
      <<",\"candidate_policy\":\"native M64 jobs; fixedK128 static full64-row operands within existing63-row pad; valid_rows masked stores\""
      <<",\"dispatch_counts\":["<<control.size()<<','<<native[1].dispatches().size()<<']'
      <<",\"additional_hit_list_dispatches\":0,\"additional_prefix_dispatches\":0,\"all_native_job_generation_cost_timed\":true"
      <<",\"strict_raw_f32_scaled_bf16_and_full_chain_required\":true,\"probe_dispatches_excluded_from_timing\":true"
      <<",\"minimum_warm_gpu_ms_per_variant_and_matched_control\":100,\"timing_has_cpu_buffer_reads\":false,\"balanced_positions_and_pair_order\":true"
      <<",\"synthetic_input_policy\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" : "per-row RMS normalized before BF16 rounding")
      <<",\"qualified_control_probe_complete_chain_exact\":"<<(controlProbeQualified ? "true" : "false")
      <<",\"control_probe_finite\":";writeProbes(out,controlFinite);
  out<<",\"control_output_sha256\":"<<splash::json::quote(controlHashes[2])<<",\"control_pipelines\":";names(out,control);
  out<<",\"variants\":[";
  for (uint32_t i=0;i<results.size();++i) {
    if (i) out<<',';const auto &r=results[i];
    out<<"{\"variant\":"<<i+1<<",\"simdgroups\":"<<r.sg<<",\"threads\":"<<r.sg*32
        <<",\"job_tile_rows\":64,\"fixed_k\":128,\"static_all64rows\":true"
        <<",\"probe_qualification_completed\":"<<(r.probeChecked ? "true" : "false")
        <<",\"initial_chain_checked\":"<<(r.initialChecked ? "true" : "false")<<",\"final_chain_checked\":"<<(r.finalChecked ? "true" : "false")
        <<",\"raw_f32_and_scaled_bf16_bit_exact\":"<<(r.probeChecked && exactProbes(r.probes) ? "true" : "false")
        <<",\"strict_full_bf16_exact\":"<<(r.initialChecked && r.finalChecked && exactChain(r.initial) && exactChain(r.final) ? "true" : "false")
        <<",\"screen_pass\":"<<(r.eligible && r.failures.empty() ? "true" : "false")<<",\"model_quality_qualified\":false,\"probes\":";writeProbes(out,r.probes);
    out<<",\"probe_activation\":";r.probeChain[0].write(out);out<<",\"probe_down\":";r.probeChain[1].write(out);out<<",\"probe_combine\":";r.probeChain[2].write(out);
    out<<",\"initial_activation\":";r.initial[0].write(out);out<<",\"initial_down\":";r.initial[1].write(out);out<<",\"initial_combine\":";r.initial[2].write(out);
    out<<",\"final_activation\":";r.final[0].write(out);out<<",\"final_down\":";r.final[1].write(out);out<<",\"final_combine\":";r.final[2].write(out);
    out<<",\"control_warm_gpu_ms\":"<<r.controlWarmGPU*1000<<",\"candidate_warm_gpu_ms\":"<<r.candidateWarmGPU*1000<<",\"warm_pairs\":"<<r.warmPairs
        <<",\"timing_attempted\":"<<(!r.candidateTimes.empty() ? "true" : "false")<<",\"control_gpu_ms\":";times(out,r.controlTimes,true);
    out<<",\"candidate_gpu_ms\":";times(out,r.candidateTimes,true);out<<",\"control_wall_ms\":";times(out,r.controlTimes,false);out<<",\"candidate_wall_ms\":";times(out,r.candidateTimes,false);
    out<<",\"candidate_output_sha256\":"<<splash::json::quote(r.outputHash)<<",\"candidate_pipelines\":";names(out,r.commands);out<<",\"failures\":[";
    for (uint32_t j=0;j<r.failures.size();++j) {if (j) out<<',';out<<splash::json::quote(r.failures[j]);}out<<"]}";
  }
  out<<"],\"screen_pass\":"<<(pass ? "true" : "false")<<'}';return pass;
}
