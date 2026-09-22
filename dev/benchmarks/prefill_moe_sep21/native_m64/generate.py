#!/usr/bin/env python3
"""Generate a bounded one-layer native M32/M64 oracle; never load payloads."""
from pathlib import Path
import argparse


BODY = r'''
namespace one = splash::flash::qmv_one_layer;

// Reproduce the frozen Full512 store's native dispatches over one certified
// layer. Production constructor/source validators are intentionally excluded:
// the one-layer loader verifies identical payload geometry, hashes and ranks.
struct NativeStoreView final {
  const one::OneLayerPayload &layer;
  uint32_t layerIndex;
  void addGateUp(CommandGraph &graph, uint32_t index,
      const FlashMoEBlockedScratch &s, MetalBuffer diag, uint32_t rows,
      FlashMoEBlockedTile tile) const {
    require(index == layerIndex, "one-layer native store index differs");
    const uint32_t m = uint32_t(tile), jobs = moEBucketJobCapacity(rows,10,m);
    const FlashInt8ExpertStoreParams p{rows,10,rows*10,jobs,m,512,0,0};
    const std::string suffix = m == 64 ? "_m64_n64_sg8" : "_m32_n64";
    graph.add("flash_int8_expert_store_gate_up" + suffix,
        {s.buckets.packedInputs,layer.codes[0],layer.scales[0],layer.codes[1],
         layer.scales[1],layer.ranks,s.buckets.offsets,s.buckets.tileJobs,
         s.buckets.jobCount,s.packedActivated,diag},p,{10,jobs,1},
        {m == 64 ? 256u : 128u,1,1});
  }
  void addDownScatter(CommandGraph &graph, uint32_t index,
      const FlashMoEBlockedScratch &s, MetalBuffer diag, uint32_t rows,
      FlashMoEBlockedTile tile) const {
    require(index == layerIndex, "one-layer native store index differs");
    const uint32_t m = uint32_t(tile), routes = rows*10;
    const uint32_t jobs = moEBucketJobCapacity(rows,10,m);
    const FlashMoEBlockedDownParams poison{{rows,10,640,2560,512,0,0,0,
        320,uint64_t{2560}*320,20,uint64_t{2560}*20},routes,jobs,m,0};
    graph.add("flash_moe_blocked_poison_excluded_routes",
        {s.buckets.canonicalToPacked,s.scatteredDown,diag},poison,
        {10,routes,1},{256,1,1});
    graph.add("flash_moe_direct_a_prepare_down",
        {s.packedActivated,s.buckets.offsets,s.packedActivated,diag},
        FlashMoEDirectAPrepareParams{routes,640,63,0},{routes+63,1,1},{256,1,1});
    const FlashInt8ExpertStoreParams p{rows,10,routes,jobs,m,512,0,0};
    const std::string suffix = m == 64 ? "_m64_n64_sg8" : "_m32_n64";
    graph.add("flash_int8_expert_store_down_scatter" + suffix,
        {s.packedActivated,layer.codes[2],layer.scales[2],layer.ranks,
         s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,
         s.buckets.routeMap,s.scatteredDown,diag},p,{40,jobs,1},
        {m == 64 ? 256u : 128u,1,1});
  }
};

void nativeCPU() {
  cpuSelfTest();
  std::vector<uint32_t> hot(512);
  for (uint32_t expert=0;expert<512;++expert) hot[expert]=expert;
  for (std::string_view pattern : {"hit-concentrated","hit-spread","spread-all"}) {
    const auto ids=patternIDs(129,hot,pattern);
    const auto packed=ref::pack(std::vector<uint16_t>(129*2560,0x3f80),ids,129,10,kSticky);
    require(packed.diagnostic==kSticky,"Full512 CPU pattern ownership differs");
    for (uint32_t m : {32u,64u}) {
      const auto jobs=ref::makeJobs(packed,m);
      require(jobs.count<=jobs.entries.size(),"Full512 native job capacity differs");
    }
  }
  require(sizeof(FlashInt8ExpertStoreParams)==32,"native I8 parameter ABI differs");
}

uint32_t numeric(const char *raw, uint32_t minimum, uint32_t maximum) {
  const std::string value(raw); size_t used=0;
  require(!value.empty() && std::all_of(value.begin(),value.end(),
      [](char c){return c>='0' && c<='9';}),"invalid numeric CLI value");
  const auto parsed=std::stoul(value,&used);
  require(used==value.size() && parsed>=minimum && parsed<=maximum,
      "numeric CLI value outside bounded native oracle");
  return uint32_t(parsed);
}

bool runCase(MetalBackend &backend, const NativeStoreView &store,
    uint32_t rows, const std::string &pattern, uint32_t pairs,
    double maxL2, double minCosine, bool strict, std::ostream &out) {
  const uint32_t layer=store.layerIndex;
  std::vector<uint32_t> hot(512);
  for (uint32_t expert=0;expert<512;++expert) hot[expert]=expert;
  std::vector<uint16_t> hidden(uint64_t{rows}*2560);
  auto ids=patternIDs(rows,hot,pattern);
  const char *rawInput=std::getenv("FLASH_INT8_STORE_INPUT");
  const char *rawIDs=std::getenv("FLASH_INT8_STORE_ROUTE_IDS");
  require(bool(rawInput)==bool(rawIDs),"raw hidden and IDs must be supplied together");
  if (rawInput) {
    hidden=readFile<uint16_t>(rawInput,hidden.size());
    ids=readFile<int64_t>(rawIDs,ids.size());
  } else for (uint64_t i=0;i<hidden.size();++i)
    hidden[i]=bf16(float(int((i*73+i/2560*17)%257)-128)/512.0f);
  for (uint16_t value:hidden) require(std::isfinite(number(value)),"nonfinite hidden fixture");
  const auto packed=ref::pack(hidden,ids,rows,10,kSticky);
  require(packed.diagnostic==kSticky,"fixture invalid/duplicate top10 ownership");
  const std::array<ref::Jobs,2> jobs{ref::makeJobs(packed,32),ref::makeJobs(packed,64)};
  const auto input=upload(backend,hidden,"native M32/M64 shared hidden fixture");
  const auto expertIDs=upload(backend,ids,"native M32/M64 shared original IDs");
  std::vector<uint16_t> routeValues(ids.size());
  for (uint64_t route=0;route<ids.size();++route)
    routeValues[route]=bf16(float((route%10)+1)/55.0f);
  const auto route=upload(backend,routeValues,"native M32/M64 shared unequal route weights");
  const auto shared=upload(backend,std::vector<uint16_t>(uint64_t{rows}*2560,0),"native shared expert zeros");
  const auto sharedGate=upload(backend,std::vector<uint16_t>(rows,0),"native shared gate zeros");
  std::array<FlashMoEBlockedScratch,2> scratch;
  std::array<CommandGraph,2> graphs;
  std::array<MetalBuffer,2> output,diagnostic;
  std::vector<Guard> guards;
  for (uint32_t which=0;which<2;++which) {
    const auto tile=which ? FlashMoEBlockedTile::M64N64 : FlashMoEBlockedTile::M32N64;
    scratch[which]=allocateMoEBlockedScratch(backend,rows);
    guardScratch(backend,scratch[which],guards);
    output[which]=guarded(backend,uint64_t{rows}*2560*2,guards);
    diagnostic[which]=guarded(backend,4,guards);
    *static_cast<uint32_t *>(diagnostic[which].contents())=kSticky;
    addMoEBlockedPack(graphs[which],input,expertIDs,scratch[which],diagnostic[which],rows,tile);
    store.addGateUp(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    store.addDownScatter(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    addCombine(graphs[which],scratch[which].scatteredDown,expertIDs,route,shared,sharedGate,
        output[which],diagnostic[which],rows,2560,512,10);
  }
  require(graphs[0].dispatches().size()==graphs[1].dispatches().size(),
      "matched native chain dispatch count differs");
  const auto healthy=[&] {
    for (const auto &d:diagnostic)
      require(*static_cast<const uint32_t *>(d.contents())==kSticky,"native sticky diagnostics differ");
    for (const auto &g:guards) require(g.clean(),"native guarded scratch overwritten");
  };
  for (uint32_t which=0;which<2;++which) {
    (void)backend.submitCommand(graphs[which].dispatches());
    healthy();checkBuckets(scratch[which],packed,jobs[which]);
  }
  const uint64_t routes=uint64_t{rows}*10;
  auto activation=compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine);
  auto down=compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine);
  auto combined=compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine);
  bool exact=!activation.mismatches && !down.mismatches && !combined.mismatches;
  std::array<std::vector<CommandTiming>,2> timing;
  const bool timingAttempted=!strict || exact;
  if (timingAttempted)
    for (uint32_t pair=0;pair<pairs;++pair)
      for (uint32_t order=0;order<2;++order) {
        const uint32_t which=(pair+order)%2;
        timing[which].push_back(backend.submitCommand(graphs[which].dispatches()));
        healthy();
      }
  for (uint32_t which=0;which<2;++which) checkBuckets(scratch[which],packed,jobs[which]);
  activation=compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine);
  down=compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine);
  combined=compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine);
  exact=exact && !activation.mismatches && !down.mismatches && !combined.mismatches;
  require(!std::memcmp(hidden.data(),input.contents(),hidden.size()*2) &&
      !std::memcmp(ids.data(),expertIDs.contents(),ids.size()*8),"native shared fixture mutated");
  out<<"{\"layer\":"<<layer<<",\"rows\":"<<rows
      <<",\"pattern\":"<<splash::json::quote(rawInput ? "raw-fixture" : pattern)
      <<",\"stored_experts\":512,\"native_job_tiles\":[32,64],\"producer_simdgroups\":[4,8]"
      <<",\"active_jobs\":["<<jobs[0].count<<','<<jobs[1].count<<']'
      <<",\"job_capacities\":["<<jobs[0].entries.size()<<','<<jobs[1].entries.size()<<']'
      <<",\"dispatch_counts\":["<<graphs[0].dispatches().size()<<','<<graphs[1].dispatches().size()<<']'
      <<",\"additional_hit_list_dispatches\":0,\"shared_fixture_and_layer\":true"
      <<",\"synthetic_input_policy\":\"frozen inherited /512 BF16 fixture; not actual model activation\""
      <<",\"strict_full_bf16_requested\":"<<(strict ? "true" : "false")
      <<",\"strict_full_bf16_exact\":"<<(exact ? "true" : "false")
      <<",\"numerical_alternative\":"<<(exact ? "false" : "true")
      <<",\"timing_attempted\":"<<(timingAttempted ? "true" : "false")
      <<",\"independent_m32_and_m64_buckets_exact\":true,\"canaries_clean\":true"
      <<",\"activation\":";activation.write(out);
  out<<",\"down\":";down.write(out);out<<",\"combine\":";combined.write(out);
  out<<",\"control_pipelines\":";names(out,graphs[0].dispatches());
  out<<",\"candidate_pipelines\":";names(out,graphs[1].dispatches());
  out<<",\"control_gpu_ms\":";times(out,timing[0],true);
  out<<",\"candidate_gpu_ms\":";times(out,timing[1],true);
  out<<",\"control_wall_ms\":";times(out,timing[0],false);
  out<<",\"candidate_wall_ms\":";times(out,timing[1],false);
  out<<",\"control_output_sha256\":"<<splash::json::quote(digest(output[0]))
      <<",\"candidate_output_sha256\":"<<splash::json::quote(digest(output[1]))<<'}';
  return exact;
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc==2 && std::string_view(argv[1])=="--cpu-self-test") {
        nativeCPU();
        std::cout<<"{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"payload_reads\":false,\"native_m32_m64_gpu_parity\":\"pending\"}\n";
        return 0;
      }
      require(argc==7 && std::string_view(argv[1])=="--gpu",
          "usage: native-m64-oracle --gpu METALLIB FULL512_STORE LAYER ROWS NEW_REPORT");
      const uint32_t layerIndex=numeric(argv[4],0,47),rows=numeric(argv[5],1024,2048);
      const uint32_t pairs=envNumber("FLASH_INT8_STORE_PAIRS",4,32);
      const double maxL2=envPositive("FLASH_INT8_STORE_MAX_RL2",0.001,1.0);
      const double minCosine=envPositive("FLASH_INT8_STORE_MIN_COSINE",0.999999,1.0);
      const bool strict=std::getenv("PREFILL_MOE_NATIVE_M64_STRICT")!=nullptr;
      const char *selected=std::getenv("FLASH_INT8_STORE_PATTERN");
      if (selected) require(std::string_view(selected)=="hit-concentrated" ||
          std::string_view(selected)=="hit-spread" || std::string_view(selected)=="spread-all",
          "Full512 native oracle supports hit-concentrated/hit-spread/spread-all");
      require(!std::filesystem::exists(argv[6]),"choose fresh native report path");
      const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[3],
          "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
          NormConvention::OnePlusWeight);
      require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1",
          "certified Full512 manifest differs");
      require(setenv("SPLASH_FLASH_MOE_Q4X8","1",1)==0 &&
          setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1)==0 &&
          setenv("SPLASH_FLASH_MOE_M64","1",1)==0,"cannot enable native control policy");
      MetalBackend backend(argv[2]);
      const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve=std::max<uint64_t>(16ULL<<30,physical/10);
      require(physical>reserve,"cannot protect host reserve");
      splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      // Includes transient guarded replacements, both complete scratch sets,
      // fixture buffers and allocator rounding at the bounded R<=2048 extent.
      const uint64_t planned=one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(1ULL<<30);
      auto admission=governor.tryReserve(planned);
      require(bool(admission),"one-layer native oracle reservation denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;
      auto layer=one::OneLayerPayload::load(backend,metadata,layerIndex);
      const one::OneLayerPayload::ImmutableHashes immutable{
          metadata.layers[layerIndex].sha256,digest(layer.ranks)};
      std::ofstream out(argv[6]);require(bool(out),"cannot create native report");
      out<<std::setprecision(12)<<"{\"schema\":\"prefill-moe-sep21-native-m32-m64-one-layer-v1\""
          <<",\"model_quality_qualified\":false,\"one_shared_readonly_layer\":true"
          <<",\"mapped_layer_bytes\":"<<layer.base.sizeBytes()
          <<",\"planned_bytes\":"<<planned<<",\"host_reserve_bytes\":"<<reserve
          <<",\"full_model_loaded\":false,\"all_48_layer_store_loaded\":false"
          <<",\"production_store_constructor_used\":false,\"native_full512_dispatch_adapter\":true"
          <<",\"timing_scope\":\"complete original native Full512 expert chains; warm alternating M32/M64; one shared layer and fixtures; checks, constructor scans and final hashes excluded\""
          <<",\"pairs\":"<<pairs<<",\"cases\":[";
      bool first=true;
      try {
        const NativeStoreView store{layer,layerIndex};
        for (const char *pattern : {"hit-concentrated","hit-spread","spread-all"}) {
          if (selected && std::string_view(selected)!=pattern) continue;
          std::ostringstream entry;
          const bool exact=runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry);
          if (!first) out<<',';first=false;
          out<<entry.str();
          if (strict && !exact)
            throw std::runtime_error("strict native M32/M64 complete BF16 chain differs");
          if (std::getenv("FLASH_INT8_STORE_INPUT")) break;
        }
      } catch (const std::exception &e) {
        out<<"],\"pass\":false,\"failure\":"<<splash::json::quote(e.what())<<"}\n";
        out.flush();throw;
      }
      require(!first,"no native patterns selected");
      layer.checkImmutableHashes(immutable);
      const uint64_t after=backend.memoryStats().allocatedBytes;
      require(after>=before && after-before<=planned,"one-layer native ledger exceeded reservation");
      admission->commit();
      out<<"],\"pass\":true,\"one_layer_payload_and_ranks_immutable\":true"
          <<",\"final_allocated_bytes\":"<<after-before<<"}\n";
      out.flush();require(bool(out),"native report write failed");
      backend.stop();
      std::cout<<"{\"pass\":true,\"gpu_executed\":true,\"report\":"<<splash::json::quote(argv[6])<<"}\n";
      return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
  }
}
'''


def generate(destination: Path):
    inherited = Path('build/prefill4k-int8tiles/oracle.mm').read_text()
    source = inherited[:inherited.index('uint64_t exactMissSlices(')]
    source = source.replace('#include "dev/benchmarks/prefill4k_int8tiles/bridge.hpp"',
        '#include "metal/abi/FlashInt8ExpertStore.h"\n'
        '#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"')
    start = source.index('std::vector<uint32_t> parseLayers()')
    end = source.index('std::vector<uint32_t> coldIDs(', start)
    source = source[:start] + source[end:]
    old = 'hot.size() >= 10 && cold.size() >= 10'
    if source.count(old) != 1:
        raise RuntimeError('frozen pattern validation drift')
    source = source.replace(old,
        'hot.size() >= 10 && ((pattern != "mixed" && pattern != "miss-only") || cold.size() >= 10)')
    source += inherited[inherited.index('void times('):inherited.index('struct Replay final')]
    source += BODY
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'oracle.mm').write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination)
