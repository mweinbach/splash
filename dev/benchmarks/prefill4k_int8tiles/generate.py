from pathlib import Path
import argparse

def replacement(text,before,after,count=1):
    if text.count(before) !=count: raise RuntimeError(f"Hit oracle source drift: {before!r}")
    return text.replace(before,after)

def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    relaxed =Path('dev/benchmarks/prefill4k_int8tiles/candidate.metal').read_text()
    relaxed =replacement(relaxed,'false, true, false, matmul2d_descriptor::mode::multiply','false, true, true, matmul2d_descriptor::mode::multiply',2)
    relaxed =relaxed.replace('prefill4k_int8tiles_','prefill4k_int8relaxed_')
    (destination /'relaxed.metal').write_text(relaxed)
    text = Path('dev/benchmarks/flash_int8_expert_store_oracle.mm').read_text()
    text = replacement(text,'#include "flash_expert_int8_bucket_reference.hpp"','#include "dev/benchmarks/flash_expert_int8_bucket_reference.hpp"\n#include "dev/benchmarks/prefill4k_int8tiles/bridge.hpp"')
    for role in ('gate','up','down'):
        text = replacement(text,f'  const auto &{role} = weights.projection(prefix + ".{role}_proj");\n','')
    text = replacement(text,'  const auto prefix = prefixFor(layer);\n','')
    begin = text.index('    if (!which) {\n      addMoEBlockedGateUp')
    end = text.index('    addCombine(graphs[which]',begin)
    text = text[:begin] + '    store.addGateUp(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);\n    store.addDownScatter(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);\n' + text[end:]
    point = '  const auto healthy = [&] {'
    addition = r'''  const uint32_t candidateTile = envNumber("PREFILL4K_INT8_HIT_TILE",64,128);
  const uint32_t candidateSG = envNumber("PREFILL4K_INT8_HIT_SG",8,16);
  const bool relaxed =std::getenv("PREFILL4K_INT8_HIT_RELAXED") !=nullptr;
  const uint32_t candidateCapacity =(rows *10 +candidateTile -1) /candidateTile +511;
  constexpr uint64_t alignment =16384;
  const uint64_t planBytes =((513 *4 +64 +alignment -1) &~(alignment -1)) +
      ((4 +64 +alignment -1) &~(alignment -1)) +((uint64_t(candidateCapacity) *8 +64 +alignment -1) &~(alignment -1));
  auto admission =governor.tryReserve(planBytes);
  require(bool(admission),"hit-only job list MemoryGovernor admission failed");
  prefill4k_int8tiles::HitCommands hitPlan(backend,graphs[1].dispatches(),rows,candidateTile,candidateSG,relaxed);
  admission->commit();
  std::array<std::vector<ComputeDispatch>,2> commands;
  commands[0].assign(graphs[0].dispatches().begin(),graphs[0].dispatches().end());
  commands[1] =hitPlan.commands;
  const auto checkHitJobs =[&] {
    std::vector<uint32_t> prefix(513,0);
    std::vector<FlashMoEBucketJob> expected(candidateCapacity,{UINT32_MAX,0});
    uint32_t count =0;
    for (uint32_t expert =0; expert <512; ++expert) {
      if (std::binary_search(hot.begin(),hot.end(),expert))
        for (uint32_t row =packed.offsets[expert]; row <packed.offsets[expert +1]; row +=candidateTile)
          expected[count++] ={expert,row};
      prefix[expert +1] =count;
    }
    exactVector(hitPlan.offsets,prefix,"independent hit-only job offsets");
    require(*static_cast<const uint32_t *>(hitPlan.count.contents()) ==count,"independent hit-only count differs");
    const auto *actual =static_cast<const FlashMoEBucketJob *>(hitPlan.jobs.contents());
    for (uint32_t i =0; i <candidateCapacity; ++i)
      require(actual[i].expert ==expected[i].expert && actual[i].row_begin ==expected[i].row_begin,"independent hit-only ownership differs");
    require(hitPlan.canaries(),"hit-only list canary differs");
  };
'''
    text = replacement(text,point,addition +point)
    text = replacement(text,'Replay runCase(MetalBackend &backend, const FlashWeights &weights, const FlashInt8ExpertStore &store,','Replay runCase(MetalBackend &backend,const FlashWeights &weights,const FlashInt8ExpertStore &store,splash::engine::MemoryGovernor &governor,')
    text = replacement(text,'runCase(backend, weights, *store, layer, rows','runCase(backend,weights,*store,governor,layer,rows')
    text = replacement(text,'splash::engine::MemoryGovernor &governor,\n    uint32_t layer','splash::engine::MemoryGovernor &governor,const FlashInt8ExpertStoreMetadata &metadata,\n    uint32_t layer')
    text = replacement(text,'runCase(backend,weights,*store,governor,layer,rows','runCase(backend,weights,*store,governor,metadata,layer,rows')
    text = replacement(text,'  for (uint16_t value : hidden) require(std::isfinite(number(value)), "nonfinite fixture hidden value");',r'''  const bool edge =std::getenv("PREFILL4K_INT8_HIT_EDGE") !=nullptr;
  if (edge) {
    const auto immutable =store.immutableWeightBuffers();
    const auto *codes =reinterpret_cast<const int8_t *>(static_cast<const uint8_t *>(immutable[layer *2].contents()) +metadata.layers[layer].codes[0].offset);
    uint32_t k1 =0,k2 =64;
    while (k1 <64 && codes[k1] ==0) ++k1;
    while (k2 <2559 && codes[k2] ==0) ++k2;
    require(k1 <64 && k2 <2559,"edge fixture needs two nonzero actual persisted codes");
    std::fill(hidden.begin(),hidden.end(),uint16_t(0));
    for (uint32_t row =0; row <rows; ++row) {
      const float sign =row &1 ? -1.0f :1.0f;
      hidden[uint64_t(row) *2560 +k1] =bf16(sign *float(codes[k2]));
      hidden[uint64_t(row) *2560 +k2] =bf16(-sign *float(codes[k1]));
      hidden[uint64_t(row) *2560 +2559] =bf16(sign *0x1p-12f);
    }
    // Selected original expert hot[0], gate output0 has exactly cancelling
    // integer products plus a tiny BF16 residual. Other channels stay finite.
  }
  for (uint16_t value :hidden) require(std::isfinite(number(value)),"nonfinite fixture hidden value");''')
    text = text.replace('backend.submitCommand(graphs[0].dispatches())','backend.submitCommand(commands[0])').replace('backend.submitCommand(graphs[1].dispatches())','backend.submitCommand(commands[1])').replace('backend.submitCommand(graphs[which].dispatches())','backend.submitCommand(commands[which])')
    text = replacement(text,'  healthy(); checkBuckets(scratch[0], packed, jobs); checkBuckets(scratch[1], packed, jobs);','  healthy(); checkHitJobs(); checkBuckets(scratch[0],packed,jobs); checkBuckets(scratch[1],packed,jobs);')
    text = replacement(text,'  if (misses == routes) require(!activation.mismatches && !downComparison.mismatches && !combined.mismatches,','  require(!activation.mismatches && !downComparison.mismatches && !combined.mismatches,')
    text = replacement(text,'  (void)compare(output[0], output[1], uint64_t{rows} * 2560, maxL2, minCosine);','  checkHitJobs();\n  require(compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine).mismatches ==0 &&\n      compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes *2560,maxL2,minCosine).mismatches ==0 &&\n      compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine).mismatches ==0,"timed same-INT8 source equality failed");')
    text = replacement(text,'names(out, graphs[0].dispatches())','names(out,commands[0])').replace('names(out, graphs[1].dispatches())','names(out,commands[1])')
    text = replacement(text,'  CommandGraph graph;\n  MetalBuffer output','  CommandGraph graph;\n  prefill4k_int8tiles::HitCommands hitPlan;\n  std::vector<ComputeDispatch> commands;\n  MetalBuffer output')
    text = replacement(text,'backend.submitCommand(graph.dispatches())','backend.submitCommand(commands)')
    text = replacement(text,'Replay replay; replay.graph = std::move(graphs[1]);','Replay replay; replay.graph =std::move(graphs[1]); replay.hitPlan =std::move(hitPlan); replay.commands =std::move(commands[1]);')
    text = replacement(text,'<< ",\\\"pattern\\\":" << splash::json::quote(rawInput','<< ",\\\"candidate_hit_tile\\\":" << candidateTile << ",\\\"candidate_hit_sg\\\":" << candidateSG << ",\\\"candidate_relaxed_precision\\\":" << (relaxed ? "true" :"false") << ",\\\"cancellation_fixture\\\":" << (edge ? "true" :"false")\n      << ",\\\"pattern\\\":" << splash::json::quote(rawInput')
    text = replacement(text,'  functions.push_back(direct ? "flash_moe_direct_a_prepare_down"',r'''  const uint32_t candidateTile =envNumber("PREFILL4K_INT8_HIT_TILE",64,128);
  const uint32_t candidateSG =envNumber("PREFILL4K_INT8_HIT_SG",8,16);
  const bool relaxed =std::getenv("PREFILL4K_INT8_HIT_RELAXED") !=nullptr;
  for (const char *phase : {"gate_up","down_scatter"})
    functions.push_back(std::string(relaxed ? "prefill4k_int8relaxed_" :"prefill4k_int8tiles_") +phase +"_m" +std::to_string(candidateTile) +"_n64_sg" +std::to_string(candidateSG));
  functions.push_back("prefill4k_int8tiles_hit_prefix"); functions.push_back("prefill4k_int8tiles_hit_jobs");
  functions.push_back(direct ? "flash_moe_direct_a_prepare_down"''')
    text = replacement(text,'const uint32_t requested = name.ends_with(suffix) ? (m == 64 ? 256 : 128) : 256;','const uint32_t requested =(name.starts_with("prefill4k_int8tiles_") || name.starts_with("prefill4k_int8relaxed_")) && name.find("_sg") !=std::string::npos ? candidateSG *32 : name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;')
    text = text.replace('flash-production-selected-int8-expert-store-oracle-v1','prefill4k-current-int8-hit-only-tile-oracle-v1').replace('\\\"numerical_alternative\\\":true','\\\"numerical_alternative\\\":false')
    (destination /'oracle.mm').write_text(text)

if __name__ =='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);generate(p.parse_args().destination)
