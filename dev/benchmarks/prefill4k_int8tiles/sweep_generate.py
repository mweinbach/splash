from pathlib import Path
import argparse

BODY =r'''
  struct Variant { uint32_t tile,sg; bool relaxed; };
  const std::array<Variant,7> variants{{{32,4,true},{64,8,false},{64,8,true},{128,8,false},{128,8,true},{128,16,false},{128,16,true}}};
  std::vector<prefill4k_int8tiles::HitCommands> plans;
  std::vector<std::vector<CommandTiming>> controlTimes(7),candidateTimes(7);
  std::array<bool,7> exact; exact.fill(true);
  const uint32_t rejectionChecks =checkRejections(backend,weights,store,layer,scratch[1],diagnostic[1],rows,tile);
  plans.reserve(7);
  for (const auto &v :variants) {
    const uint32_t cap =(rows *10 +v.tile -1) /v.tile +511;
    const auto rounded =[](uint64_t n) { return (n +16383) &~uint64_t(16383); };
    auto admission =governor.tryReserve(rounded(513 *4 +64) +rounded(4 +64) +rounded(uint64_t(cap) *8 +64));
    require(bool(admission),"sweep compact hit lists denied before allocation");
    plans.emplace_back(backend,graphs[1].dispatches(),rows,v.tile,v.sg,v.relaxed); admission->commit();
  }
  const auto healthy =[&] {
    for (const auto &d :diagnostic) require(*static_cast<const uint32_t *>(d.contents()) ==kSticky,"sweep sticky diagnostics changed");
    for (const auto &g :guards) require(g.clean(),"sweep output canary changed");
    for (const auto &p :plans) require(p.canaries(),"sweep compact list canary changed");
  };
  const auto checkJobs =[&](uint32_t index) {
    const auto &plan =plans[index]; const uint32_t tile =variants[index].tile;
    std::vector<uint32_t> prefix(513,0);
    std::vector<FlashMoEBucketJob> expected(plan.capacity,{UINT32_MAX,0}); uint32_t count =0;
    for (uint32_t expert =0; expert <512; ++expert) {
      if (std::binary_search(hot.begin(),hot.end(),expert))
        for (uint32_t row =packed.offsets[expert]; row <packed.offsets[expert +1]; row +=tile) expected[count++] ={expert,row};
      prefix[expert +1] =count;
    }
    exactVector(plan.offsets,prefix,"sweep CPU hit offsets");
    require(*static_cast<const uint32_t *>(plan.count.contents()) ==count,"sweep CPU hit count differs");
    const auto *actual =static_cast<const FlashMoEBucketJob *>(plan.jobs.contents());
    for (uint32_t i =0; i <plan.capacity; ++i)
      require(actual[i].expert ==expected[i].expert && actual[i].row_begin ==expected[i].row_begin,"sweep CPU job ownership differs");
  };
  const uint64_t routes =uint64_t(rows) *10;
  const auto equalBuffers =[&] {
    return std::memcmp(scratch[0].packedActivated.contents(),scratch[1].packedActivated.contents(),routes *640 *2) ==0 &&
        std::memcmp(scratch[0].scatteredDown.contents(),scratch[1].scatteredDown.contents(),routes *2560 *2) ==0 &&
        std::memcmp(output[0].contents(),output[1].contents(),uint64_t(rows) *2560 *2) ==0;
  };
  const auto poisonCandidate =[&] {
    std::memset(scratch[1].packedActivated.contents(),0xa5,scratch[1].packedActivated.sizeBytes());
    std::memset(scratch[1].scatteredDown.contents(),0xa5,scratch[1].scatteredDown.sizeBytes());
    std::memset(output[1].contents(),0xa5,output[1].sizeBytes());
  };
  (void)backend.submitCommand(graphs[0].dispatches());
  const auto controlHash =digest(output[0]);
  for (uint32_t index =0; index <7; ++index) {
    poisonCandidate(); (void)backend.submitCommand(plans[index].commands);
    healthy(); checkJobs(index); checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
    (void)exactMissSlices(scratch,packed,ids,hot);
    exact[index] =equalBuffers();
  }
  for (uint32_t pair =0; pair <pairs; ++pair)
    for (uint32_t order =0; order <7; ++order) {
      const uint32_t index =(pair +order) %7;
      poisonCandidate();
      if ((pair +index) &1) {
        candidateTimes[index].push_back(backend.submitCommand(plans[index].commands));
        controlTimes[index].push_back(backend.submitCommand(graphs[0].dispatches()));
      } else {
        controlTimes[index].push_back(backend.submitCommand(graphs[0].dispatches()));
        candidateTimes[index].push_back(backend.submitCommand(plans[index].commands));
      }
      healthy(); checkJobs(index); exact[index] =exact[index] &&equalBuffers();
      (void)exactMissSlices(scratch,packed,ids,hot);
      require(digest(output[0]) ==controlHash,"same persisted control output changed");
    }
  out << "{\"layer\":" <<layer <<",\"rows\":" <<rows <<",\"source_tile\":" <<m
      <<",\"pattern\":" <<splash::json::quote(pattern) <<",\"actual_code_cancellation_fixture\":" <<(edge ? "true" :"false")
      <<",\"shared_source_control_graph\":true,\"shared_fixture_buffers\":true,\"rotation\":\"variant index=(pair+order)%7; paired control/candidate order alternates\""
      <<",\"atomic_graph_rejection_checks\":" <<rejectionChecks
      <<",\"control_output_sha256\":" <<splash::json::quote(controlHash) <<",\"variants\":[";
  for (uint32_t index =0; index <7; ++index) {
    if (index) out <<','; const auto &v =variants[index];
    poisonCandidate(); (void)backend.submitCommand(plans[index].commands); healthy();checkJobs(index);
    exact[index] =exact[index] &&equalBuffers();
    const auto activation =compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine);
    const auto down =compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes *2560,maxL2,minCosine);
    const auto combined =compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);
    out <<"{\"tile\":" <<v.tile <<",\"sg\":" <<v.sg <<",\"relaxed\":" <<(v.relaxed ? "true" :"false")
        <<",\"strict_bf16_exact\":" <<(exact[index] ? "true" :"false") <<",\"activation\":";
    activation.write(out);out <<",\"down\":";down.write(out);out <<",\"combine\":";combined.write(out);
    out <<",\"control_gpu_ms\":";times(out,controlTimes[index],true);out <<",\"candidate_gpu_ms\":";times(out,candidateTimes[index],true);
    out <<",\"control_wall_ms\":";times(out,controlTimes[index],false);out <<",\"candidate_wall_ms\":";times(out,candidateTimes[index],false);
    out <<",\"candidate_output_sha256\":" <<splash::json::quote(digest(output[1])) <<",\"candidate_pipelines\":";names(out,plans[index].commands);
    out <<",\"compact_jobs_checked\":true,\"canaries_clean\":true}";
  }
  out <<"]}";
  Replay replay; replay.graph =std::move(graphs[1]); replay.hitPlan =std::move(plans.back()); replay.commands =replay.hitPlan.commands;
  replay.output =output[1];replay.diagnostic =diagnostic[1];
  const auto *values =static_cast<const uint16_t *>(output[1].contents()); replay.expected.assign(values,values +uint64_t(rows) *2560);
  replay.guards =std::move(guards); replay.layer =layer;replay.pattern =pattern; return replay;
}
'''

def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    original =Path('build/prefill4k-int8tiles/oracle.mm').read_text()
    begin =original.index('  const uint32_t candidateTile = envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end =original.index('\nvoid checkStoreRanks(',begin)
    source =original[:begin] +BODY +original[end:]
    source =source.replace('prefill4k-current-int8-hit-only-tile-oracle-v1','prefill4k-current-int8-hit-only-all-variant-sweep-v1')
    point ='  functions.push_back("prefill4k_int8tiles_hit_prefix");'
    insert ='''  for (uint32_t tile :{32u,64u,128u}) {
    const uint32_t sg =tile ==32 ? 4 :8;
    for (const char *prefix :{"prefill4k_int8tiles_","prefill4k_int8relaxed_"})
      for (const char *phase :{"gate_up","down_scatter"}) {
        functions.push_back(std::string(prefix) +phase +"_m" +std::to_string(tile) +"_n64_sg" +std::to_string(sg));
        if (tile ==128) functions.push_back(std::string(prefix) +phase +"_m128_n64_sg16");
      }
  }
'''
    if source.count(point) !=1:raise RuntimeError('sweep metadata contract drift')
    source =source.replace(point,insert +point)
    before ='const uint32_t requested =(name.starts_with("prefill4k_int8tiles_") || name.starts_with("prefill4k_int8relaxed_")) && name.find("_sg") !=std::string::npos ? candidateSG *32 : name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;'
    after ='const uint32_t requested =name.ends_with("_sg16") ? 512 :name.ends_with("_sg8") ? 256 :name.ends_with("_sg4") ? 128 :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;'
    if source.count(before) !=1:raise RuntimeError('sweep pipeline threads contract drift')
    source =source.replace(before,after)
    source =source.replace('      const char *selectedPattern = std::getenv("FLASH_INT8_STORE_PATTERN");','      const char *selectedPattern = std::getenv("FLASH_INT8_STORE_PATTERN");')
    call ='          replay = runCase(backend,weights,*store,governor,metadata,layer,rows, m, pattern, pairs, maxL2, minCosine, out);'
    # Source whitespace is checked rather than guessed at generation time.
    if call not in source:
        call=next(line for line in source.splitlines() if 'replay = runCase(' in line)
    extra ='''          if (!selectedPattern && std::string_view(pattern) =="mixed") {
            require(setenv("PREFILL4K_INT8_HIT_EDGE","1",1) ==0,"cannot enable actual-code cancellation fixture");
            out <<',';
            replay =runCase(backend,weights,*store,governor,metadata,layer,rows,m,"hit-concentrated",pairs,maxL2,minCosine,out);
            unsetenv("PREFILL4K_INT8_HIT_EDGE");
          }
'''
    source=source.replace(call,call +'\n'+extra)
    # candidateSG becomes unused after validating every variant explicitly.
    source=source.replace('  const uint32_t candidateSG =envNumber("PREFILL4K_INT8_HIT_SG",8,16);','')
    source=source.replace(' +std::to_string(candidateSG));',' +std::to_string(candidateTile ==32 ? 4 :8));')
    (destination /'oracle.mm').write_text(source)

if __name__ =='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);generate(p.parse_args().destination)
