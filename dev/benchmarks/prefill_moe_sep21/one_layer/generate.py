#!/usr/bin/env python3
"""Generate the bounded Full512 layer eleven-variant screen without loading it."""
from pathlib import Path
import argparse
import importlib.util


BODY = r'''
  using prefill_moe_sep21::variants;
  require(graphs[0].dispatches().size()==graphs[1].dispatches().size(),
      "matched M32 source graph dispatch counts differ");
  const char *selection=std::getenv("PREFILL_MOE_SEP21_VARIANT");
  const uint32_t selected=selection ?
      envNumber("PREFILL_MOE_SEP21_VARIANT",1,variants.size())-1 : UINT32_MAX;
  std::vector<uint32_t> active;
  std::vector<prefill_moe_sep21::HitCommands> plans;
  for (uint32_t index=0;index<variants.size();++index)
    if (selected==UINT32_MAX || selected==index) {
      active.push_back(index);
      plans.emplace_back(graphs[1].dispatches(),rows,variants[index]);
    }
  require(!active.empty(),"bounded one-layer variant selection empty");
  const uint64_t routes=uint64_t{rows}*10;
  struct Result final {
    std::array<Comparison,3> initial,final;
    std::vector<CommandTiming> controlTimes,candidateTimes;
    std::vector<std::string> failures;
    std::string outputHash;
    uint32_t diagnostic=kSticky;
    bool initialChecked=false,finalChecked=false,initialExact=false,finalExact=false;
    bool errorGuard=false,buckets=false,canaries=false,eligible=false;
  };
  std::vector<Result> results(active.size());
  const auto fail=[&](Result &r,const std::string &reason) {
    if (std::find(r.failures.begin(),r.failures.end(),reason)==r.failures.end())
      r.failures.push_back(reason);
  };
  const auto canariesClean=[&] {
    return std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});
  };
  const auto policy=[&](const Comparison &c) {
    return !c.nonfinite && std::isfinite(c.relativeL2) &&
        c.relativeL2<=maxL2 && std::isfinite(c.cosine) && c.cosine>=minCosine;
  };
  const auto poison=[&] {
    std::memset(scratch[1].packedActivated.contents(),0xa5,scratch[1].packedActivated.sizeBytes());
    std::memset(scratch[1].scatteredDown.contents(),0xa5,scratch[1].scatteredDown.sizeBytes());
    std::memset(output[1].contents(),0xa5,output[1].sizeBytes());
    *static_cast<uint32_t *>(diagnostic[1].contents())=kSticky;
  };
  const auto inspect=[&](uint32_t position,bool initial) {
    auto &r=results[position];
    r.diagnostic=*static_cast<const uint32_t *>(diagnostic[1].contents());
    if (r.diagnostic!=kSticky) fail(r,"candidate sticky diagnostic/numerical flags changed");
    r.canaries=canariesClean();
    if (!r.canaries) fail(r,"guarded scratch/output canary overwritten");
    r.buckets=false;
    try {checkBuckets(scratch[1],packed,jobs[1]);r.buckets=true;}
    catch (const std::exception &e) {fail(r,e.what());}
    auto &comparisons=initial ? r.initial : r.final;
    comparisons={
        compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine),
        compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine),
        compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine)};
    const bool exact=std::all_of(comparisons.begin(),comparisons.end(),
        [](const Comparison &c){return !c.mismatches && !c.nonfinite;});
    const bool guard=std::all_of(comparisons.begin(),comparisons.end(),policy);
    if (!guard) fail(r,"complete BF16 chain exceeds .001 relative-L2/.999999 cosine or finite guard");
    if (strict && !exact) fail(r,"strict complete BF16 chain differs from native M32");
    if (initial) {
      r.initialChecked=true;r.initialExact=exact;r.errorGuard=guard;
      r.eligible=r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);
    } else {
      r.finalChecked=true;r.finalExact=exact;r.errorGuard=r.errorGuard && guard;
      r.eligible=r.eligible && r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);
    }
  };
  // Original native control is computed once before qualification. These scans
  // and constructor checks are excluded from every submitted command timing.
  (void)backend.submitCommand(graphs[0].dispatches());
  require(*static_cast<const uint32_t *>(diagnostic[0].contents())==kSticky &&
      canariesClean(),"native M32 control diagnostics/canaries failed");
  checkBuckets(scratch[0],packed,jobs[0]);
  const auto baselineFinite=compare(output[0],output[0],uint64_t{rows}*2560,maxL2,minCosine);
  require(!baselineFinite.nonfinite,"native M32 control has nonfinite output");
  const auto controlHash=digest(output[0]);
  bool sharedMemoryHealthy=true;
  for (uint32_t position=0;position<active.size();++position) {
    auto &r=results[position];
    if (!sharedMemoryHealthy) {fail(r,"screen stopped after earlier guard corruption");continue;}
    poison();
    try {
      (void)backend.submitCommand(plans[position].commands);
      inspect(position,true);
    } catch (const std::exception &e) {fail(r,std::string("candidate warm qualification: ")+e.what());}
    sharedMemoryHealthy=canariesClean();
  }
  // The rotating sweep gives each qualified variant a matched original-control
  // sample per pair. Candidate/control order alternates; no large scans or
  // poisoning occurs between timed commands, and all outputs are checked again.
  for (uint32_t pair=0;pair<pairs && sharedMemoryHealthy;++pair)
    for (uint32_t order=0;order<active.size();++order) {
      const uint32_t position=(pair+order)%active.size();auto &r=results[position];
      if (!r.eligible) continue;
      *static_cast<uint32_t *>(diagnostic[1].contents())=kSticky;
      try {
        if ((pair+position)&1) {
          r.candidateTimes.push_back(backend.submitCommand(plans[position].commands));
          r.controlTimes.push_back(backend.submitCommand(graphs[0].dispatches()));
        } else {
          r.controlTimes.push_back(backend.submitCommand(graphs[0].dispatches()));
          r.candidateTimes.push_back(backend.submitCommand(plans[position].commands));
        }
        r.diagnostic=*static_cast<const uint32_t *>(diagnostic[1].contents());
        if (r.diagnostic!=kSticky) {
          fail(r,"timed candidate diagnostics changed");r.eligible=false;
        }
        require(*static_cast<const uint32_t *>(diagnostic[0].contents())==kSticky,
            "timed native M32 control diagnostic changed");
      } catch (const std::exception &e) {
        fail(r,std::string("candidate matched timing: ")+e.what());r.eligible=false;
      }
      sharedMemoryHealthy=canariesClean();
      if (!sharedMemoryHealthy) {
        fail(r,"timed guard corruption stopped sweep");r.eligible=false;break;
      }
    }
  require(digest(output[0])==controlHash,"timed native M32 control output changed");
  checkBuckets(scratch[0],packed,jobs[0]);
  bool sweepPass=sharedMemoryHealthy;
  // Replay every warmed variant to associate final full-output measurements
  // with that variant, because all candidates deliberately share one scratch.
  for (uint32_t position=0;position<active.size();++position) {
    auto &r=results[position];
    if (sharedMemoryHealthy && r.initialChecked) {
      poison();
      try {
        (void)backend.submitCommand(plans[position].commands);
        inspect(position,false);r.outputHash=digest(output[1]);
      } catch (const std::exception &e) {fail(r,std::string("candidate final replay: ")+e.what());r.eligible=false;}
      sharedMemoryHealthy=canariesClean();
    }
    if (r.controlTimes.size()!=r.candidateTimes.size())
      fail(r,"candidate/control timing sample count differs");
    sweepPass=sweepPass && r.initialChecked && r.finalChecked && r.errorGuard &&
        r.buckets && r.canaries && r.diagnostic==kSticky && r.failures.empty();
  }
  require(!std::memcmp(hidden.data(),input.contents(),hidden.size()*2) &&
      !std::memcmp(ids.data(),expertIDs.contents(),ids.size()*8),"shared one-layer fixture mutated");
  out<<"{\"layer\":"<<layer<<",\"rows\":"<<rows<<",\"native_job_tile\":32"
      <<",\"pattern\":"<<splash::json::quote(rawInput ? "raw-fixture" : pattern)
      <<",\"stored_experts\":512,\"same_original_m32_jobs_and_parameters\":true"
      <<",\"additional_prefix_dispatches\":0,\"original_dispatch_count\":"<<graphs[0].dispatches().size()
      <<",\"production_store_negative_tests\":\"omitted for explicitly bounded native dispatch adapter\""
      <<",\"synthetic_input_policy\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" :
          std::getenv("PREFILL_MOE_SEP21_NORMALIZED") ? "private normalized /74 BF16 fixture" : "frozen inherited /512 BF16 fixture")
      <<",\"strict_full_bf16_requested\":"<<(strict ? "true" : "false")
      <<",\"producer_sanity_guard\":{\"maximum_relative_l2\":"<<maxL2
      <<",\"minimum_cosine\":"<<minCosine<<'}'
      <<",\"rotation\":\"(pair+order)%active_variants; paired candidate/control order alternates\""
      <<",\"control_output_sha256\":"<<splash::json::quote(controlHash)
      <<",\"control_pipelines\":";names(out,graphs[0].dispatches());
  out<<",\"variants\":[";
  for (uint32_t position=0;position<active.size();++position) {
    if (position) out<<',';const uint32_t index=active[position];const auto &v=variants[index];const auto &r=results[position];
    const bool exact=r.initialChecked && r.finalChecked && r.initialExact && r.finalExact;
    const bool accepted=r.initialChecked && r.finalChecked && r.errorGuard &&
        r.buckets && r.canaries && r.diagnostic==kSticky && r.failures.empty();
    out<<"{\"variant\":"<<index+1<<",\"kind\":"<<splash::json::quote(v.kind)
        <<",\"tile_m\":32,\"tile_n\":64,\"k\":"<<v.k<<",\"sg\":"<<v.sg
        <<",\"register_i8_to_bf16_exact_conversion\":"<<(v.registerOperand ? "true" : "false")
        <<",\"static_full_row_extents\":"<<(v.staticExtent ? "true" : "false")
        <<",\"register_part_rows\":"<<(v.rowParts ? 16 : 32)
        <<",\"different_reduction_descriptor\":"<<(v.differentReduction ? "true" : "false")
        <<",\"qualification_completed_before_timing\":"<<(r.initialChecked ? "true" : "false")
        <<",\"full_bf16_error_guard_pass\":"<<(r.errorGuard ? "true" : "false")
        <<",\"strict_full_bf16_exact\":"<<(exact ? "true" : "false")
        <<",\"numerical_alternative\":"<<(exact ? "false" : "true")
        <<",\"screen_pass\":"<<(accepted ? "true" : "false")
        <<",\"model_quality_qualified\":false,\"diagnostic\":"<<r.diagnostic
        <<",\"independent_m32_bucket_jobs_exact\":"<<(r.buckets ? "true" : "false")
        <<",\"canaries_clean\":"<<(r.canaries ? "true" : "false")
        <<",\"initial_activation\":";r.initial[0].write(out);
    out<<",\"initial_down\":";r.initial[1].write(out);out<<",\"initial_combine\":";r.initial[2].write(out);
    out<<",\"final_activation\":";r.final[0].write(out);
    out<<",\"final_down\":";r.final[1].write(out);out<<",\"final_combine\":";r.final[2].write(out);
    out<<",\"timing_attempted\":"<<(!r.candidateTimes.empty() ? "true" : "false")
        <<",\"control_gpu_ms\":";times(out,r.controlTimes,true);
    out<<",\"candidate_gpu_ms\":";times(out,r.candidateTimes,true);
    out<<",\"control_wall_ms\":";times(out,r.controlTimes,false);
    out<<",\"candidate_wall_ms\":";times(out,r.candidateTimes,false);
    out<<",\"candidate_output_sha256\":"<<splash::json::quote(r.outputHash)
        <<",\"candidate_pipelines\":";names(out,plans[position].commands);
    out<<",\"failures\":[";
    for (uint32_t failure=0;failure<r.failures.size();++failure) {
      if (failure) out<<',';out<<splash::json::quote(r.failures[failure]);
    }
    out<<"]}";
  }
  out<<"],\"screen_pass\":"<<(sweepPass ? "true" : "false")<<'}';
  return sweepPass;
}
'''


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f'bounded one-layer source drift: {before!r}')
    return source.replace(before, after)


def generate(destination):
    spec = importlib.util.spec_from_file_location('native_m64_generator',
        Path('dev/benchmarks/prefill_moe_sep21/native_m64/generate.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(destination)
    source = (destination / 'oracle.mm').read_text()
    source = replace(source, '#include "metal/abi/FlashInt8ExpertStore.h"',
        '#include "metal/abi/FlashInt8ExpertStore.h"\n'
        '#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"')
    source = replace(source, 'cpuSelfTest();\n  std::vector<uint32_t> hot(512);',
        'cpuSelfTest();prefill_moe_sep21::cpuSelfTest();\n  std::vector<uint32_t> hot(512);')
    source = replace(source,
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,64)};',
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,32)};')
    source = replace(source,
        'const auto tile=which ? FlashMoEBlockedTile::M64N64 : FlashMoEBlockedTile::M32N64;',
        'constexpr auto tile=FlashMoEBlockedTile::M32N64;')
    source = replace(source, '/512.0f);',
        '/(std::getenv("PREFILL_MOE_SEP21_NORMALIZED") ? 74.0f : 512.0f));')
    begin = source.index('  require(graphs[0].dispatches().size()==graphs[1].dispatches().size(),')
    end = source.index('} // namespace\n\nint main(', begin)
    source = source[:begin] + BODY + source[end:]
    # Report nonfinite outputs and failed error guards rather than throwing away
    # the evidence. Unavailable floating-point summaries remain valid JSON null.
    source = replace(source, 'uint64_t elements = 0, mismatches = 0;',
        'uint64_t elements = 0, mismatches = 0, nonfinite = 0;')
    source = replace(source, 'struct Comparison final {',
        'std::string finiteJSON(double value) {\n'
        '  if (!std::isfinite(value)) return "null";\n'
        '  std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();\n'
        '}\nstruct Comparison final {')
    source = replace(source,
        '<< ",\\\"relative_l2\\\":" << relativeL2 << ",\\\"cosine\\\":" << cosine << ",\\\"max_abs\\\":" << maxAbs',
        '<< ",\\\"nonfinite_pairs\\\":" << nonfinite\n'
        '        << ",\\\"relative_l2\\\":" << finiteJSON(relativeL2) << ",\\\"cosine\\\":" << finiteJSON(cosine) << ",\\\"max_abs\\\":" << finiteJSON(maxAbs)')
    source = replace(source,
        'require(std::isfinite(x) && std::isfinite(y), "nonfinite expert-chain output");',
        'if (!std::isfinite(x) || !std::isfinite(y)) {\n'
        '      ++result.nonfinite;result.mismatches += av[i] != bv[i];continue;\n'
        '    }')
    source = replace(source,
        '  require(std::isfinite(result.relativeL2) && result.relativeL2 <= maxL2 && result.cosine >= minCosine,\n'
        '      "declared INT8 producer sanity guard failed; this is not a model-quality qualification");',
        '  (void)maxL2;(void)minCosine; // Per-variant policy records rejection after reporting all elements.')
    source = replace(source, 'std::getenv("PREFILL_MOE_NATIVE_M64_STRICT")',
        'std::getenv("PREFILL_MOE_SEP21_STRICT")')
    source = source.replace('prefill-moe-sep21-native-m32-m64-one-layer-v1',
        'prefill-moe-sep21-eleven-variant-one-layer-v1')
    source = replace(source, '      bool first=true;', '      bool first=true;bool sweepPass=true;')
    source = replace(source,
        'const bool exact=runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry);',
        'const bool accepted=runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry);\n'
        '          sweepPass=sweepPass && accepted;')
    source = replace(source,
        '          if (strict && !exact)\n'
        '            throw std::runtime_error("strict native M32/M64 complete BF16 chain differs");', '')
    source = replace(source, 'out<<"],\\\"pass\\\":true,\\\"one_layer_payload_and_ranks_immutable\\\":true"',
        'out<<"],\\\"pass\\\":"<<(sweepPass ? "true" : "false")<<",\\\"one_layer_payload_and_ranks_immutable\\\":true"')
    source = replace(source,
        'std::cout<<"{\\\"pass\\\":true,\\\"gpu_executed\\\":true,\\\"report\\\":"<<splash::json::quote(argv[6])<<"}\\n";\n      return 0;',
        'std::cout<<"{\\\"pass\\\":"<<(sweepPass ? "true" : "false")<<",\\\"gpu_executed\\\":true,\\\"report\\\":"<<splash::json::quote(argv[6])<<"}\\n";\n'
        '      return sweepPass ? 0 : 2;')
    source = source.replace('warm alternating M32/M64', 'warm rotating private variants paired with native M32')
    source = source.replace('native_m32_m64_gpu_parity', 'eleven_variant_gpu_parity')
    (destination / 'oracle.mm').write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination)
