#!/usr/bin/env python3
"""Generate the independent wide-W8A8 screen without touching frozen outputs."""
from pathlib import Path
import argparse
import importlib.util


def replace(source,before,after,count=1):
    if source.count(before)!=count:raise RuntimeError(f'wide W8A8 oracle source drift: {before!r}')
    return source.replace(before,after)


def generate(destination):
    spec=importlib.util.spec_from_file_location('w8a8_generator',Path('dev/benchmarks/prefill_moe_sep21/w8a8/generate.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.generate(destination)
    source=(destination/'oracle.mm').read_text()
    source=replace(source,'#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"',
        '#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"\n'
        '#include "dev/benchmarks/prefill_moe_sep21/w8a8_wide/cache.hpp"')
    source=replace(source,'prefill_moe_w8a8::cpuSelfTest();',
        'prefill_moe_w8a8_wide::cpuSelfTest();')
    source=replace(source,
        'const std::string suffix = m == 64 ? "_m64_n64_sg8" : "_m32_n64";',
        'const std::string suffix = m == 64 ? "_m64_n64_sg8" : m == 16 ? "_m16_n64" : "_m32_n64";',2)
    source=replace(source,
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,32)};',
        'const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,16)};')
    source=replace(source,'  using prefill_moe_w8a8::variants;',r'''
  std::array<CommandGraph,2> smallGraphs;
  for (uint32_t which=0;which<2;++which) {
    constexpr auto tile=FlashMoEBlockedTile::M16N64;
    addMoEBlockedPack(smallGraphs[which],input,expertIDs,scratch[which],diagnostic[which],rows,tile);
    store.addGateUp(smallGraphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    store.addDownScatter(smallGraphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    addCombine(smallGraphs[which],scratch[which].scatteredDown,expertIDs,route,shared,sharedGate,
        output[which],diagnostic[which],rows,2560,512,10);
  }
  using prefill_moe_w8a8_wide::variants;''')
    source=source.replace('prefill_moe_w8a8::QuantizedCommands','prefill_moe_w8a8_wide::QuantizedCommands')
    source=replace(source,
        'plans.emplace_back(graphs[1].dispatches(),rows,variants[index],store.layer,workspace);',
        'plans.emplace_back(variants[index].m==16 ? smallGraphs[1].dispatches() : graphs[1].dispatches(),\n'
        '          rows,variants[index],store.layer,workspace);')
    source=replace(source,'  const uint64_t routes=uint64_t{rows}*10;',r'''
  const auto controlCommands=[&](uint32_t position) {
    return variants[active[position]].m==16 ? smallGraphs[0].dispatches() : graphs[0].dispatches();
  };
  prefill_moe_w8a8_wide::QuantizedCommands bestW8(graphs[1].dispatches(),rows,variants[0],store.layer,workspace);
  const uint64_t routes=uint64_t{rows}*10;''')
    source=replace(source,'std::vector<CommandTiming> controlTimes,candidateTimes;',
        'std::vector<CommandTiming> controlTimes,candidateTimes,bestW8Times;')
    source=replace(source,'bool initialRawPass=false,finalRawPass=false;',
        'bool initialRawPass=false,finalRawPass=false,initialGeometryExact=false,finalGeometryExact=false;')
    source=replace(source,'try {checkBuckets(scratch[1],packed,jobs[1]);r.buckets=true;}',
        'try {\n'
        '      const auto &expected=jobs[variants[active[position]].m==16 ? 1 : 0];\n'
        '      checkBuckets(scratch[0],packed,expected);checkBuckets(scratch[1],packed,expected);r.buckets=true;\n'
        '    }')
    source=replace(source,'r.eligible=r.initialRawPass && r.diagnostic==kSticky',
        'r.eligible=r.initialGeometryExact && r.initialRawPass && r.diagnostic==kSticky')
    source=replace(source,'r.eligible=r.eligible && r.finalRawPass && r.diagnostic==kSticky',
        'r.eligible=r.eligible && r.finalGeometryExact && r.finalRawPass && r.diagnostic==kSticky')
    begin=source.index('  // Original native control is computed once before qualification.')
    end=source.index('  bool sharedMemoryHealthy=true;',begin)
    source=source[:begin]+r'''
  std::array<std::string,2> controlHashes;
  for (uint32_t small=0;small<2;++small) {
    (void)backend.submitCommand(small ? smallGraphs[0].dispatches() : graphs[0].dispatches());
    require(*static_cast<const uint32_t *>(diagnostic[0].contents())==kSticky && canariesClean(),
        "native I8 M16/M32 control diagnostics/canaries failed");
    checkBuckets(scratch[0],packed,jobs[small]);
    require(!compare(output[0],output[0],uint64_t{rows}*2560,maxL2,minCosine).nonfinite,
        "native I8 control has nonfinite output");
    controlHashes[small]=digest(output[0]);
  }
  // Qualify the previously measured fastest W8 geometry again with this exact
  // fixture. Its actual two quantizers are included in all reference timings.
  (void)backend.submitCommand(graphs[0].dispatches());poison();
  (void)backend.submitCommand(bestW8.auditCommands);
  const auto bestCertificate=certify(store,scratch,packed,workspace);
  require(bestCertificate.pass(),"best M32N64SG2 W8 raw certificate failed");
  checkBuckets(scratch[1],packed,jobs[0]);
  for (const auto &comparison:std::array<Comparison,3>{
      compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine),
      compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine),
      compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine)})
    require(policy(comparison),"best M32N64SG2 W8 producer guard failed");
  const std::array<std::string,3> bestW8Hashes{
      digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
  (void)backend.submitCommand(bestW8.commands);
  require(bestW8Hashes==std::array<std::string,3>{
      digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])},
      "best W8 audit/timed BF16 producers differ");
'''+source[end:]
    source=replace(source,
        '(void)backend.submitCommand(plans[position].auditCommands);\n      const auto certificate=',
        '(void)backend.submitCommand(controlCommands(position));\n'
        '      (void)backend.submitCommand(plans[position].auditCommands);\n      const auto certificate=')
    source=replace(source,'      inspect(position,true);',r'''
      r.initialGeometryExact=bestW8Hashes==std::array<std::string,3>{
          digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
      if (!r.initialGeometryExact) fail(r,"initial wide geometry differs from exact best M32N64SG2 W8 BF16 outputs");
      inspect(position,true);''')
    begin=source.index('        if ((pair+position)&1) {')
    end=source.index('        r.diagnostic=',begin)
    source=source[:begin]+r'''
        if (!variants[active[position]].wide) {
          if ((pair+position)&1) {
            const auto reference=backend.submitCommand(plans[position].commands);
            r.candidateTimes.push_back(reference);r.bestW8Times.push_back(reference);
            r.controlTimes.push_back(backend.submitCommand(controlCommands(position)));
          } else {
            r.controlTimes.push_back(backend.submitCommand(controlCommands(position)));
            const auto reference=backend.submitCommand(plans[position].commands);
            r.candidateTimes.push_back(reference);r.bestW8Times.push_back(reference);
          }
        } else {
          for (uint32_t step=0;step<3;++step) {
            const uint32_t category=(pair+position+step)%3;
            if (!category) r.controlTimes.push_back(backend.submitCommand(controlCommands(position)));
            else if (category==1) r.bestW8Times.push_back(backend.submitCommand(bestW8.commands));
            else r.candidateTimes.push_back(backend.submitCommand(plans[position].commands));
          }
        }
'''+source[end:]
    source=replace(source,
        '  require(digest(output[0])==controlHash,"timed native M32 control output changed");\n'
        '  checkBuckets(scratch[0],packed,jobs[0]);',r'''
  for (uint32_t small=0;small<2;++small) {
    (void)backend.submitCommand(small ? smallGraphs[0].dispatches() : graphs[0].dispatches());
    require(digest(output[0])==controlHashes[small],"timed native I8 control output changed");
    checkBuckets(scratch[0],packed,jobs[small]);
  }''')
    source=replace(source,
        '(void)backend.submitCommand(plans[position].auditCommands);\n        const auto certificate=',
        '(void)backend.submitCommand(controlCommands(position));\n'
        '        (void)backend.submitCommand(plans[position].auditCommands);\n        const auto certificate=')
    source=replace(source,'        inspect(position,false);',r'''
        r.finalGeometryExact=bestW8Hashes==std::array<std::string,3>{
            digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
        if (!r.finalGeometryExact) fail(r,"final wide geometry differs from exact best M32N64SG2 W8 BF16 outputs");
        inspect(position,false);''')
    source=source.replace('r.initialRawPass && r.finalRawPass && r.errorGuard &&',
        'r.initialRawPass && r.finalRawPass && r.initialGeometryExact && r.finalGeometryExact && r.errorGuard &&')
    source=replace(source,'<<",\\\"native_job_tile\\\":32"','<<",\\\"native_job_tiles\\\":[32,16]"')
    source=source.replace('same_original_m32_jobs_and_parameters','same_original_selected_m16_m32_jobs_and_parameters')
    source=replace(source,
        '<<",\\\"control_output_sha256\\\":"<<splash::json::quote(controlHash)',
        '<<",\\\"native_m32_output_sha256\\\":"<<splash::json::quote(controlHashes[0])\n'
        '      <<",\\\"native_m16_output_sha256\\\":"<<splash::json::quote(controlHashes[1])\n'
        '      <<",\\\"best_w8_m32_n64_sg2_output_sha256\\\":"<<splash::json::quote(bestW8Hashes[2])')
    source=replace(source,'<<",\\\"tile_m\\\":32,\\\"tile_n\\\":64,\\\"k\\\":"',
        '<<",\\\"tile_m\\\":"<<v.m<<",\\\"tile_n\\\":"<<v.n<<",\\\"k\\\":"')
    source=replace(source,'<<",\\\"initial_raw_f64_certificate\\\":"',
        '<<",\\\"geometry_exact_to_best_w8_m32_n64_sg2\\\":"<<((r.initialGeometryExact && r.finalGeometryExact)?"true":"false")\n'
        '        <<",\\\"initial_raw_f64_certificate\\\":"')
    source=replace(source,'out<<",\\\"candidate_wall_ms\\\":";times(out,r.candidateTimes,false);',
        'out<<",\\\"candidate_wall_ms\\\":";times(out,r.candidateTimes,false);\n'
        '    out<<",\\\"best_w8_m32_n64_sg2_gpu_ms\\\":";times(out,r.bestW8Times,true);\n'
        '    out<<",\\\"best_w8_m32_n64_sg2_wall_ms\\\":";times(out,r.bestW8Times,false);')
    source=replace(source,'<<",\\\"candidate_pipelines\\\":";names(out,plans[position].commands);',
        '<<",\\\"candidate_pipelines\\\":";names(out,plans[position].commands);\n'
        '    out<<",\\\"matched_native_i8_control_pipelines\\\":";names(out,controlCommands(position));\n'
        '    out<<",\\\"best_w8_m32_n64_sg2_pipelines\\\":";names(out,bestW8.commands);')
    source=source.replace('independent_m32_bucket_jobs_exact','independent_selected_m16_m32_bucket_jobs_exact')
    source=source.replace('PREFILL_MOE_W8A8_VARIANT','PREFILL_MOE_W8A8_WIDE_VARIANT')
    source=source.replace('PREFILL_MOE_W8A8_STRICT','PREFILL_MOE_W8A8_WIDE_STRICT')
    source=source.replace('prefill-moe-sep21-w8a8-one-layer-v1','prefill-moe-sep21-w8a8-wide-one-layer-v1')
    source=source.replace('w8a8_gpu_certificate_and_parity','w8a8_wide_gpu_certificate_and_geometry_parity')
    source=source.replace('warm rotating W8A8 variants paired with native M32; both actual quantizer dispatches included',
        'warm rotating W8A8 geometry variants with matched original I8 M16/M32 and best W8 M32N64SG2 references; both actual quantizer dispatches included')
    (destination/'oracle.mm').write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
