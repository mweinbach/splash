#!/usr/bin/env python3
"""Generate the bounded unscaled BF16-code oracle using source files only."""
from pathlib import Path
import argparse
import importlib.util


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f'exact BF16 code oracle source drift: {before!r}')
    return text.replace(before, after)


def generate(destination):
    spec=importlib.util.spec_from_file_location('one_layer_generator',
        Path('dev/benchmarks/prefill_moe_sep21/one_layer/generate.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.generate(destination)
    source=(destination/'oracle.mm').read_text()
    source=replace(source,'#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"',
        '#include "dev/benchmarks/prefill_moe_sep21/bf16_codes/cache.hpp"\n#include <chrono>')
    source=source.replace('prefill_moe_sep21::','prefill_moe_bf16_codes::')
    source=replace(source,'bool runCase(MetalBackend &backend, const NativeStoreView &store,',
        'bool runCase(MetalBackend &backend, const NativeStoreView &store,\n'
        '    const prefill_moe_bf16_codes::ExactCodeCache &cache,')
    source=replace(source,
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,32)};',
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,64)};')
    source=replace(source,'  using prefill_moe_bf16_codes::variants;',r'''
  // Source graphs for both native job sizes share the same two scratch sets.
  // Each candidate uses the exact native-I8 control with its own M32/M64 jobs.
  std::array<CommandGraph,2> wideGraphs;
  for (uint32_t which=0;which<2;++which) {
    constexpr auto tile=FlashMoEBlockedTile::M64N64;
    addMoEBlockedPack(wideGraphs[which],input,expertIDs,scratch[which],diagnostic[which],rows,tile);
    store.addGateUp(wideGraphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    store.addDownScatter(wideGraphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    addCombine(wideGraphs[which],scratch[which].scatteredDown,expertIDs,route,shared,sharedGate,
        output[which],diagnostic[which],rows,2560,512,10);
  }
  using prefill_moe_bf16_codes::variants;''')
    source=source.replace('prefill_moe_bf16_codes::HitCommands','prefill_moe_bf16_codes::CacheCommands')
    source=replace(source,
        'plans.emplace_back(graphs[1].dispatches(),rows,variants[index]);',
        'plans.emplace_back(variants[index].m==64 ? wideGraphs[1].dispatches() : graphs[1].dispatches(),\n'
        '          rows,variants[index],store.layer,cache);')
    source=replace(source,'  const uint64_t routes=uint64_t{rows}*10;',r'''
  const auto controlCommands=[&](uint32_t position) {
    return variants[active[position]].m==64 ? wideGraphs[0].dispatches() : graphs[0].dispatches();
  };
  const uint64_t routes=uint64_t{rows}*10;''')
    source=replace(source,
        'return std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});',
        'return cache.guardsClean() && std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});')
    source=replace(source,'try {checkBuckets(scratch[1],packed,jobs[1]);r.buckets=true;}',
        'try {\n'
        '      const auto &expected=jobs[variants[active[position]].m==64 ? 1 : 0];\n'
        '      checkBuckets(scratch[0],packed,expected);checkBuckets(scratch[1],packed,expected);r.buckets=true;\n'
        '    }')
    begin=source.index('  // Original native control is computed once before qualification.')
    end=source.index('  bool sharedMemoryHealthy=true;',begin)
    source=source[:begin]+r'''
  std::array<std::string,2> controlHashes;
  for (uint32_t wide=0;wide<2;++wide) {
    (void)backend.submitCommand(wide ? wideGraphs[0].dispatches() : graphs[0].dispatches());
    require(*static_cast<const uint32_t *>(diagnostic[0].contents())==kSticky &&
        canariesClean(),"native I8 control diagnostics/canaries failed");
    checkBuckets(scratch[0],packed,jobs[wide]);
    const auto finite=compare(output[0],output[0],uint64_t{rows}*2560,maxL2,minCosine);
    require(!finite.nonfinite,"native I8 control has nonfinite output");
    controlHashes[wide]=digest(output[0]);
  }
'''+source[end:]
    source=replace(source,
        '(void)backend.submitCommand(plans[position].commands);\n      inspect(position,true);',
        '(void)backend.submitCommand(controlCommands(position));\n'
        '      (void)backend.submitCommand(plans[position].commands);\n      inspect(position,true);')
    source=source.replace('r.controlTimes.push_back(backend.submitCommand(graphs[0].dispatches()));',
        'r.controlTimes.push_back(backend.submitCommand(controlCommands(position)));')
    source=replace(source,
        '  require(digest(output[0])==controlHash,"timed native M32 control output changed");\n'
        '  checkBuckets(scratch[0],packed,jobs[0]);',r'''
  for (uint32_t wide=0;wide<2;++wide) {
    (void)backend.submitCommand(wide ? wideGraphs[0].dispatches() : graphs[0].dispatches());
    require(digest(output[0])==controlHashes[wide],"timed native I8 control output changed");
    checkBuckets(scratch[0],packed,jobs[wide]);
  }''')
    source=replace(source,
        '(void)backend.submitCommand(plans[position].commands);\n        inspect(position,false);',
        '(void)backend.submitCommand(controlCommands(position));\n'
        '        (void)backend.submitCommand(plans[position].commands);\n        inspect(position,false);')
    source=replace(source,'<<",\\\"native_job_tile\\\":32"',
        '<<",\\\"native_job_tiles\\\":[32,64]"')
    source=source.replace('same_original_m32_jobs_and_parameters','same_original_native_jobs_and_parameters')
    source=replace(source,
        '<<",\\\"control_output_sha256\\\":"<<splash::json::quote(controlHash)',
        '<<",\\\"native_m32_output_sha256\\\":"<<splash::json::quote(controlHashes[0])\n'
        '      <<",\\\"native_m64_output_sha256\\\":"<<splash::json::quote(controlHashes[1])')
    source=replace(source,'<<",\\\"tile_m\\\":32,\\\"tile_n\\\":64,\\\"k\\\":"<<v.k',
        '<<",\\\"tile_m\\\":"<<v.m<<",\\\"tile_n\\\":64,\\\"k\\\":"<<v.k')
    source=replace(source,
        '<<",\\\"register_i8_to_bf16_exact_conversion\\\":"<<(v.registerOperand ? "true" : "false")',
        '<<",\\\"persistent_unscaled_bf16_integer_codes\\\":true"')
    source=replace(source,
        '<<",\\\"register_part_rows\\\":"<<(v.rowParts ? 16 : 32)',
        '<<",\\\"cache_includes_f32_scale\\\":false"')
    source=replace(source,
        '<<",\\\"candidate_pipelines\\\":";names(out,plans[position].commands);',
        '<<",\\\"candidate_pipelines\\\":";names(out,plans[position].commands);\n'
        '    out<<",\\\"matched_native_i8_control_pipelines\\\":";names(out,controlCommands(position));')
    source=source.replace('independent_m32_bucket_jobs_exact','independent_native_bucket_jobs_exact')
    source=replace(source,'splash::json::quote(v.kind)',
        'splash::json::quote(v.k ? "fixed" : "whole")')
    source=source.replace('native M32','matched native I8')
    source=source.replace('PREFILL_MOE_SEP21_VARIANT','PREFILL_MOE_BF16_CODES_VARIANT')
    source=source.replace('PREFILL_MOE_SEP21_STRICT','PREFILL_MOE_BF16_CODES_STRICT')
    source=source.replace('PREFILL_MOE_SEP21_NORMALIZED','PREFILL_MOE_BF16_CODES_NORMALIZED')
    source=replace(source,
        '      std::ofstream out(argv[6]);require(bool(out),"cannot create native report");',r'''
      const uint64_t cachePlanned=prefill_moe_bf16_codes::ExactCodeCache::plannedBytes();
      auto cacheAdmission=governor.tryReserve(cachePlanned);
      require(bool(cacheAdmission),"independent exact BF16 code cache admission denied");
      const auto conversionStart=std::chrono::steady_clock::now();
      auto cache=prefill_moe_bf16_codes::ExactCodeCache::convert(backend,layer);
      const double conversionCPUms=std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-conversionStart).count();
      cacheAdmission->commit();
      const auto verificationStart=std::chrono::steady_clock::now();
      const uint64_t beforeMismatches=cache.verifyEveryWord(layer);
      const double beforeVerificationCPUms=std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-verificationStart).count();
      require(!beforeMismatches && cache.guardsClean(),"exact unscaled BF16 cache certification failed before GPU timing");
      std::ofstream out(argv[6]);require(bool(out),"cannot create native report");''')
    source=replace(source,
        '<<",\\\"planned_bytes\\\":"<<planned<<",\\\"host_reserve_bytes\\\":"<<reserve',
        '<<",\\\"planned_bytes\\\":"<<planned+cachePlanned<<",\\\"host_reserve_bytes\\\":"<<reserve\n'
        '          <<",\\\"cache_logical_bytes\\\":"<<cache.logicalBytes()<<",\\\"cache_admitted_bytes\\\":"<<cachePlanned\n'
        '          <<",\\\"cache_words_verified_before_timing\\\":"<<cache.words()\n'
        '          <<",\\\"cache_word_mismatches_before_timing\\\":"<<beforeMismatches\n'
        '          <<",\\\"conversion_cpu_ms_excluded_from_timing\\\":"<<conversionCPUms\n'
        '          <<",\\\"before_verification_cpu_ms_excluded_from_timing\\\":"<<beforeVerificationCPUms\n'
        '          <<",\\\"cache_includes_scale\\\":false,\\\"f32_row_scale_views_original\\\":true"')
    source=replace(source,
        'runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry)',
        'runCase(backend,store,cache,rows,pattern,pairs,maxL2,minCosine,strict,entry)')
    source=replace(source,'      layer.checkImmutableHashes(immutable);',r'''
      const auto finalVerificationStart=std::chrono::steady_clock::now();
      const uint64_t afterMismatches=cache.verifyEveryWord(layer);
      const double afterVerificationCPUms=std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-finalVerificationStart).count();
      const bool cacheCertified=!afterMismatches && cache.guardsClean();
      sweepPass=sweepPass && cacheCertified;
      layer.checkImmutableHashes(immutable);''')
    source=replace(source,'after-before<=planned,"one-layer native ledger exceeded reservation"',
        'after-before<=planned+cachePlanned,"one-layer native/cache ledger exceeded independent reservations"')
    source=replace(source,
        '<<",\\\"final_allocated_bytes\\\":"<<after-before<<"}\\n";',
        '<<",\\\"cache_words_verified_after_timing\\\":"<<cache.words()\n'
        '          <<",\\\"cache_word_mismatches_after_timing\\\":"<<afterMismatches\n'
        '          <<",\\\"cache_guards_clean\\\":"<<(cache.guardsClean() ? "true" : "false")\n'
        '          <<",\\\"cache_certified_before_and_after\\\":"<<(cacheCertified ? "true" : "false")\n'
        '          <<",\\\"after_verification_cpu_ms_excluded_from_timing\\\":"<<afterVerificationCPUms\n'
        '          <<",\\\"final_allocated_bytes\\\":"<<after-before<<"}\\n";')
    source=source.replace('prefill-moe-sep21-eleven-variant-one-layer-v1',
        'prefill-moe-sep21-exact-unscaled-bf16-code-one-layer-v1')
    source=source.replace('eleven_variant_gpu_parity','exact_unscaled_bf16_code_gpu_parity')
    source=source.replace('private variants paired with matched native I8',
        'persistent unscaled BF16-code variants paired with matched native I8 M32/M64')
    (destination/'oracle.mm').write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
