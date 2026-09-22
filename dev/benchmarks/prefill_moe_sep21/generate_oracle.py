from pathlib import Path
import argparse


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f'private expert oracle source drift: {before!r}')
    return text.replace(before, after)


BODY = r'''
  (void)governor;
  require(m ==32,"private expert matmul oracle requires unchanged native M32 jobs");
  using prefill_moe_sep21::variants;
  const char *selection=std::getenv("PREFILL_MOE_SEP21_VARIANT");
  const uint32_t selected=selection ? envNumber("PREFILL_MOE_SEP21_VARIANT",1,variants.size()) -1 :UINT32_MAX;
  std::vector<uint32_t> active;
  std::vector<prefill_moe_sep21::HitCommands> plans;
  for (uint32_t index=0;index <variants.size();++index)
    if (selected ==UINT32_MAX ||selected ==index) {
      active.push_back(index);plans.emplace_back(graphs[1].dispatches(),rows,variants[index]);
    }
  require(!active.empty(),"private expert matmul variant selection empty");
  const uint64_t routes=uint64_t(rows) *10;
  const uint32_t rejectionChecks=checkRejections(backend,weights,store,layer,scratch[1],diagnostic[1],rows,tile);
  const auto healthy=[&] {
    for (const auto &d :diagnostic) require(*static_cast<const uint32_t *>(d.contents()) ==kSticky,"private matmul sticky diagnostics changed");
    for (const auto &g :guards) require(g.clean(),"private matmul output canary changed");
  };
  const auto equalBuffers=[&] {
    return std::memcmp(scratch[0].packedActivated.contents(),scratch[1].packedActivated.contents(),routes *640 *2) ==0 &&
        std::memcmp(scratch[0].scatteredDown.contents(),scratch[1].scatteredDown.contents(),routes *2560 *2) ==0 &&
        std::memcmp(output[0].contents(),output[1].contents(),uint64_t(rows) *2560 *2) ==0;
  };
  const auto poison=[&] {
    std::memset(scratch[1].packedActivated.contents(),0xa5,scratch[1].packedActivated.sizeBytes());
    std::memset(scratch[1].scatteredDown.contents(),0xa5,scratch[1].scatteredDown.sizeBytes());
    std::memset(output[1].contents(),0xa5,output[1].sizeBytes());
  };
  std::vector<std::vector<CommandTiming>> controlTimes(active.size()),candidateTimes(active.size());
  std::vector<bool> exact(active.size(),true);
  const bool strict=std::getenv("PREFILL_MOE_SEP21_STRICT") !=nullptr;
  // Warm every candidate, validate independent jobs/ownership, finite output,
  // unchanged Q4 misses and reportable full-output error before any timing pair.
  (void)backend.submitCommand(graphs[0].dispatches());healthy();
  const auto controlHash=digest(output[0]);
  for (uint32_t position=0;position <active.size();++position) {
    poison();(void)backend.submitCommand(plans[position].commands);healthy();
    checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
    (void)exactMissSlices(scratch,packed,ids,hot);
    exact[position]=equalBuffers();
    (void)compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine);
    (void)compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes *2560,maxL2,minCosine);
    (void)compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);
    require(!strict ||exact[position],"private matmul strict full BF16 chain differs before timing");
  }
  for (uint32_t pair=0;pair <pairs;++pair)
    for (uint32_t order=0;order <active.size();++order) {
      const uint32_t position=(pair +order) %active.size();poison();
      if ((pair +position) &1) {
        candidateTimes[position].push_back(backend.submitCommand(plans[position].commands));
        controlTimes[position].push_back(backend.submitCommand(graphs[0].dispatches()));
      } else {
        controlTimes[position].push_back(backend.submitCommand(graphs[0].dispatches()));
        candidateTimes[position].push_back(backend.submitCommand(plans[position].commands));
      }
      healthy();checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
      exact[position]=exact[position] &&equalBuffers();
      require(!strict ||exact[position],"private matmul timed strict full BF16 chain differs");
      (void)exactMissSlices(scratch,packed,ids,hot);
      require(digest(output[0]) ==controlHash,"private matmul persisted control changed");
    }
  out <<"{\"layer\":" <<layer <<",\"rows\":" <<rows <<",\"native_job_tile\":32,\"pattern\":" <<splash::json::quote(rawInput ? "raw-fixture" :pattern)
      <<",\"normalized_synthetic_input\":" <<(!rawInput &&!edge ? "true" :"false") <<",\"actual_code_cancellation_fixture\":" <<(edge ? "true" :"false")
      <<",\"original_jobs_and_parameters_unchanged\":true,\"additional_prefix_dispatches\":0,\"original_dispatch_count\":" <<graphs[1].dispatches().size()
      <<",\"atomic_graph_rejection_checks\":" <<rejectionChecks
      <<",\"shared_fixture_and_source_graph\":true,\"rotation\":\"(pair+order)%active_variants; paired control/candidate order alternates\",\"control_output_sha256\":" <<splash::json::quote(controlHash)
      <<",\"variants\":[";
  for (uint32_t position=0;position <active.size();++position) {
    if (position) out <<',';const uint32_t index=active[position];const auto &v=variants[index];
    poison();(void)backend.submitCommand(plans[position].commands);healthy();
    exact[position]=exact[position] &&equalBuffers();
    require(!strict ||exact[position],"private matmul final strict full BF16 chain differs");
    const auto activation=compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine);
    const auto down=compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes *2560,maxL2,minCosine);
    const auto combined=compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);
    out <<"{\"variant\":" <<index +1 <<",\"kind\":" <<splash::json::quote(v.kind) <<",\"tile_m\":32,\"tile_n\":64,\"k\":" <<v.k <<",\"sg\":" <<v.sg
        <<",\"register_i8_to_bf16_exact_conversion\":" <<(v.registerOperand ? "true" :"false") <<",\"static_full_row_extents\":" <<(v.staticExtent ? "true" :"false")
        <<",\"register_part_rows\":" <<(v.rowParts ? 16 :32) <<",\"different_reduction_descriptor\":" <<(v.differentReduction ? "true" :"false") <<",\"strict_full_bf16_exact\":" <<(exact[position] ? "true" :"false")
        <<",\"numerical_alternative\":" <<(exact[position] ? "false" :"true") <<",\"model_quality_qualified\":false,\"activation\":";
    activation.write(out);out <<",\"down\":";down.write(out);out <<",\"combine\":";combined.write(out);
    out <<",\"control_gpu_ms\":";times(out,controlTimes[position],true);out <<",\"candidate_gpu_ms\":";times(out,candidateTimes[position],true);
    out <<",\"control_wall_ms\":";times(out,controlTimes[position],false);out <<",\"candidate_wall_ms\":";times(out,candidateTimes[position],false);
    out <<",\"candidate_output_sha256\":" <<splash::json::quote(digest(output[1])) <<",\"candidate_pipelines\":";names(out,plans[position].commands);
    if (std::getenv("PREFILL_MOE_SEP21_PHASES")) {
      out <<",\"candidate_phase_replay_scope\":\"separate per-dispatch GPU commands excluded from complete-chain timing\",\"candidate_phase_replay\":[";
      bool firstPhase=true;
      for (const auto &d :plans[position].commands) {
        const auto timing=backend.submitCommand(std::span<const ComputeDispatch>(&d,1));
        if (!firstPhase) out <<',';firstPhase=false;
        out <<"{\"pipeline\":" <<splash::json::quote(d.pipelineName) <<",\"gpu_ms\":" <<timing.gpuSeconds *1000 <<'}';
      }
      healthy();out <<']';
    }
    out <<",\"original_jobs_checked\":true,\"canaries_clean\":true}";
  }
  out <<"]}";
  Replay replay;replay.graph=std::move(graphs[1]);replay.hitPlan=std::move(plans.back());replay.commands=replay.hitPlan.commands;
  replay.output=output[1];replay.diagnostic=diagnostic[1];
  const auto *words=static_cast<const uint16_t *>(output[1].contents());replay.expected.assign(words,words +uint64_t(rows) *2560);
  replay.guards=std::move(guards);replay.layer=layer;replay.pattern=rawInput ? "raw-fixture" :pattern;return replay;
}
'''


def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    source=Path('build/prefill4k-int8tiles/oracle.mm').read_text()
    source=replace(source,'#include "dev/benchmarks/prefill4k_int8tiles/bridge.hpp"','#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"\n#include "dev/benchmarks/prefill4k_int8columns/precision.hpp"')
    source=source.replace('prefill4k_int8tiles::HitCommands','prefill_moe_sep21::HitCommands')
    begin=source.index('  const uint32_t candidateTile = envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=source.index('\nvoid checkStoreRanks(',begin)
    source=source[:begin] +BODY +source[end:]
    begin=source.index('  const uint32_t candidateTile =envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=source.index('  functions.push_back(direct ?',begin)
    source=source[:begin] +'''  for (const auto &v :prefill_moe_sep21::variants) {
    functions.push_back(prefill_moe_sep21::pipelineName(v,true));
    functions.push_back(prefill_moe_sep21::pipelineName(v,false));
  }
''' +source[end:]
    source=replace(source,'const uint32_t requested =(name.starts_with("prefill4k_int8tiles_") || name.starts_with("prefill4k_int8relaxed_")) && name.find("_sg") !=std::string::npos ? candidateSG *32 : name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;',
      'const uint32_t requested =name.starts_with("prefill_moe_sep21_") ? (name.find("_sg1") !=std::string::npos ? 32 :64) :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;')
    source=replace(source,'      cpuSelfTest();','      cpuSelfTest();prefill_moe_sep21::cpuSelfTest();prefill4k_int8columns::selfTest();')
    source=replace(source,'hot.size() >= 10 && cold.size() >= 10','hot.size() >=10 &&((pattern !="mixed" &&pattern !="miss-only") ||cold.size() >=10)')
    # Match the normalized BF16 fixture amplitude used by the newer column oracle.
    source=replace(source,'/ 512.0f);','/ 74.0f);')
    source=source.replace('prefill4k-current-int8-hit-only-tile-oracle-v1','prefill-moe-sep21-low-simd-register-oracle-v1')
    source=source.replace('"numerical_alternative\\\":false','"numerical_alternatives_reported_per_variant\\\":true')
    source=replace(source,'envPositive("FLASH_INT8_STORE_MAX_RL2", 0.10, 1.0)','envPositive("FLASH_INT8_STORE_MAX_RL2", 0.001, 1.0)')
    source=replace(source,'envPositive("FLASH_INT8_STORE_MIN_COSINE", 0.99, 1.0)','envPositive("FLASH_INT8_STORE_MIN_COSINE", 0.999999, 1.0)')
    (destination /'oracle.mm').write_text(source)


if __name__ =='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
