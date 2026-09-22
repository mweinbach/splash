#!/usr/bin/env python3
"""Generate a private bounded M32-job adaptive-tail oracle; never read payloads."""
from pathlib import Path
import argparse
import importlib.util


SUPPORT = r'''
struct TailVariant final { const char *name; uint32_t tailRows; };
constexpr std::array<TailVariant,2> tailVariants{{{"m16-tail",16},{"m8-tail",8}}};
std::string tailPipeline(uint32_t tailRows,bool gate,bool probe=false) {
  require(tailRows==0 || tailRows==16 || tailRows==8,"private tail selector differs");
  return std::string("adaptive_expert_tail_sep21_")+(gate ? "gate_up_" : "down_scatter_")+
      (tailRows ? "m"+std::to_string(tailRows)+"_tail" : "m32_control")+
      (probe ? "_probe" : "");
}
bool validTailParams(const FlashInt8ExpertStoreParams &p,uint32_t rows) {
  return rows>=1024 && rows<=2048 && p.rows==rows && p.selections==10 &&
      p.route_capacity==rows*10 && p.job_capacity==(rows*10+31)/32+511 &&
      p.tile_rows==32 && p.stored_experts==512 && !p.scale_group_size && !p.reserved;
}
// An independent CPU model of original M32 job ownership. Tail choice cannot
// change the original job's canonical begin/end or permit malformed ownership.
bool validTailJob(const FlashMoEBucketJob &job,std::span<const uint32_t> offsets,
    std::span<const uint32_t> ranks,uint32_t routes,uint32_t &remaining) {
  if (offsets.size()!=513 || ranks.size()!=512 || job.expert>=512) return false;
  const uint32_t start=offsets[job.expert],end=offsets[job.expert+1];
  if (start>end || end>routes || job.row_begin<start || job.row_begin>=end ||
      ranks[job.expert]>=512) return false;
  remaining=std::min(32u,end-job.row_begin);return true;
}
void tailCPU() {
  nativeCPU();
  const FlashInt8ExpertStoreParams good{2048,10,20480,1151,32,512,0,0};
  require(validTailParams(good,2048),"adaptive original-M32 parameter golden differs");
  for (uint32_t index=0;index<8;++index) {
    auto bad=good;auto *words=reinterpret_cast<uint32_t *>(&bad);
    words[index]=index==6 || index==7 ? 1 : 0;
    require(!validTailParams(bad,2048),"adaptive malformed parameter accepted");
  }
  auto bad=good;bad.tile_rows=16;
  require(!validTailParams(bad,2048),"adaptive native-M16 jobs accepted");
  bad=good;bad.stored_experts=511;
  require(!validTailParams(bad,2048),"adaptive partial expert inventory accepted");
  std::array<uint32_t,513> offsets{};std::array<uint32_t,512> ranks{};
  for (uint32_t expert=0;expert<512;++expert) {offsets[expert+1]=(expert+1)*40;ranks[expert]=expert;}
  uint32_t remaining=0;
  require(validTailJob({17,17*40},offsets,ranks,20480,remaining) && remaining==32 &&
      validTailJob({17,17*40+32},offsets,ranks,20480,remaining) && remaining==8,
      "adaptive original-M32 full/tail ownership golden differs");
  require(validTailJob({17,17*40+1},offsets,ranks,20480,remaining) && remaining==32,
      "adaptive copied validator changed valid within-bucket job semantics");
  for (const FlashMoEBucketJob job : {FlashMoEBucketJob{512,0}, {17,17*40-1}, {17,18*40}})
    require(!validTailJob(job,offsets,ranks,20480,remaining),"adaptive malformed job accepted");
  auto brokenOffsets=offsets;brokenOffsets[18]=brokenOffsets[17]-1;
  require(!validTailJob({17,17*40},brokenOffsets,ranks,20480,remaining),"descending offset accepted");
  brokenOffsets=offsets;brokenOffsets[18]=20481;
  require(!validTailJob({17,17*40},brokenOffsets,ranks,20480,remaining),"out-of-range offset accepted");
  auto brokenRanks=ranks;brokenRanks[17]=UINT32_MAX;
  require(!validTailJob({17,17*40},offsets,brokenRanks,20480,remaining),"invalid compact rank accepted");
  for (const auto &v:tailVariants)
    require(tailPipeline(v.tailRows,true)!=tailPipeline(v.tailRows,false) &&
        tailPipeline(v.tailRows,true,true)==tailPipeline(v.tailRows,true)+"_probe",
        "adaptive private pipeline inventory differs");
}

struct ProbeBuffers final {
  MetalBuffer rawGate,rawUp,scaledGate,scaledUp,rawDown,scaledDown;
  ProbeBuffers(MetalBackend &backend,uint64_t routes,std::vector<Guard> &guards) {
    rawGate=guarded(backend,routes*640*4,guards);rawUp=guarded(backend,routes*640*4,guards);
    scaledGate=guarded(backend,routes*640*2,guards);scaledUp=guarded(backend,routes*640*2,guards);
    rawDown=guarded(backend,routes*2560*4,guards);scaledDown=guarded(backend,routes*2560*2,guards);
  }
  void poison() const {
    for (const auto &b:{rawGate,rawUp,scaledGate,scaledUp,rawDown,scaledDown})
      std::memset(b.contents(),0xff,b.sizeBytes());
  }
};
std::vector<ComputeDispatch> tailCommands(std::span<const ComputeDispatch> source,
    uint32_t rows,uint32_t tailRows,const ProbeBuffers *probe=nullptr) {
  std::vector<ComputeDispatch> result;bool gateSeen=false,downSeen=false;
  MetalBuffer offsets,jobs,count;
  for (const auto &original:source) {
    auto d=original;
    const bool gate=d.pipelineName=="flash_int8_expert_store_gate_up_m32_n64";
    const bool down=d.pipelineName=="flash_int8_expert_store_down_scatter_m32_n64";
    if (!gate && !down) {result.push_back(d);continue;}
    require(d.bytes.size()==1 && d.bytes[0].data &&
        d.bytes[0].sizeBytes==sizeof(FlashInt8ExpertStoreParams),"adaptive producer parameter ABI differs");
    FlashInt8ExpertStoreParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));
    require(validTailParams(p,rows) && d.threadgroups.y==p.job_capacity &&
        d.threadgroups.z==1 && d.threadsPerThreadgroup.x==128 &&
        d.threadsPerThreadgroup.y==1 && d.threadsPerThreadgroup.z==1,
        "adaptive producer requires untouched M32 parameters/grid/128 threads");
    for (uint32_t index=0;index<d.buffers.size();++index)
      require(d.buffers[index].index==index,"adaptive original buffer indices differ");
    if (gate) {
      require(!gateSeen && !downSeen && d.buffers.size()==11 &&
          d.bytes[0].index==11 && d.threadgroups.x==10,"adaptive gate ABI/order differs");
      gateSeen=true;offsets=d.buffers[6].buffer;jobs=d.buffers[7].buffer;count=d.buffers[8].buffer;
      if (probe) {
        d.buffers.push_back({12,probe->rawGate});d.buffers.push_back({13,probe->rawUp});
        d.buffers.push_back({14,probe->scaledGate});d.buffers.push_back({15,probe->scaledUp});
      }
    } else {
      require(gateSeen && !downSeen && d.buffers.size()==10 && d.bytes[0].index==10 &&
          d.threadgroups.x==40 && d.buffers[4].buffer.sameView(offsets) &&
          d.buffers[5].buffer.sameView(jobs) && d.buffers[6].buffer.sameView(count),
          "adaptive down ABI/order/original jobs differ");
      downSeen=true;
      if (probe) {d.buffers.push_back({11,probe->rawDown});d.buffers.push_back({12,probe->scaledDown});}
    }
    d.pipelineName=tailPipeline(tailRows,gate,probe!=nullptr);result.push_back(d);
  }
  require(gateSeen && downSeen && result.size()==source.size(),"adaptive producer inventory differs");
  return result;
}
void matchingNativeGraphs(std::span<const ComputeDispatch> a,std::span<const ComputeDispatch> b) {
  require(a.size()==b.size(),"adaptive native graph dispatch counts differ");
  const auto sameSize=[](const auto &x,const auto &y) {return x.x==y.x && x.y==y.y && x.z==y.z;};
  for (uint32_t i=0;i<a.size();++i) {
    require(a[i].pipelineName==b[i].pipelineName && sameSize(a[i].threadgroups,b[i].threadgroups) &&
        sameSize(a[i].threadsPerThreadgroup,b[i].threadsPerThreadgroup) &&
        a[i].buffers.size()==b[i].buffers.size() && a[i].bytes.size()==b[i].bytes.size(),
        "adaptive native graph names/geometry/ABI differ");
    for (uint32_t j=0;j<a[i].buffers.size();++j)
      require(a[i].buffers[j].index==b[i].buffers[j].index &&
          a[i].buffers[j].buffer.sizeBytes()==b[i].buffers[j].buffer.sizeBytes(),
          "adaptive native graph buffer roles/extents differ");
    for (uint32_t j=0;j<a[i].bytes.size();++j)
      require(a[i].bytes[j].index==b[i].bytes[j].index && a[i].bytes[j].sizeBytes==b[i].bytes[j].sizeBytes &&
          !std::memcmp(a[i].bytes[j].data,b[i].bytes[j].data,a[i].bytes[j].sizeBytes),
          "adaptive native graph parameter bytes differ");
  }
}
struct BitComparison final {
  uint64_t elements=0,mismatches=0,nonfinite=0;
  bool exact() const {return !mismatches && !nonfinite;}
  void write(std::ostream &out) const {
    out<<"{\"elements\":"<<elements<<",\"bit_mismatches\":"<<mismatches
        <<",\"nonfinite\":"<<nonfinite<<",\"bit_exact_and_finite\":"<<(exact() ? "true" : "false")<<'}';
  }
};
BitComparison bitCompare(const MetalBuffer &a,const MetalBuffer &b,uint64_t elements,bool f32) {
  require(a.sizeBytes()>=elements*(f32 ? 4 : 2) && b.sizeBytes()>=elements*(f32 ? 4 : 2),
      "adaptive probe comparison extent differs");
  BitComparison result;result.elements=elements;
  if (f32) {
    const auto *av=static_cast<const uint32_t *>(a.contents()),*bv=static_cast<const uint32_t *>(b.contents());
    for (uint64_t i=0;i<elements;++i) {
      result.mismatches+=av[i]!=bv[i];
      result.nonfinite+=!std::isfinite(std::bit_cast<float>(av[i])) || !std::isfinite(std::bit_cast<float>(bv[i]));
    }
  } else {
    const auto *av=static_cast<const uint16_t *>(a.contents()),*bv=static_cast<const uint16_t *>(b.contents());
    for (uint64_t i=0;i<elements;++i) {
      result.mismatches+=av[i]!=bv[i];
      result.nonfinite+=!std::isfinite(number(av[i])) || !std::isfinite(number(bv[i]));
    }
  }
  return result;
}
using ProbeComparisons=std::array<BitComparison,6>;
ProbeComparisons compareProbes(const ProbeBuffers &a,const ProbeBuffers &b,uint64_t routes) {
  return {bitCompare(a.rawGate,b.rawGate,routes*640,true),bitCompare(a.rawUp,b.rawUp,routes*640,true),
      bitCompare(a.scaledGate,b.scaledGate,routes*640,false),bitCompare(a.scaledUp,b.scaledUp,routes*640,false),
      bitCompare(a.rawDown,b.rawDown,routes*2560,true),bitCompare(a.scaledDown,b.scaledDown,routes*2560,false)};
}
void writeProbes(std::ostream &out,const ProbeComparisons &values) {
  constexpr std::array<const char *,6> keys{"raw_f32_gate","raw_f32_up","scaled_bf16_gate",
      "scaled_bf16_up","raw_f32_down_canonical","scaled_bf16_down_canonical"};
  out<<'{';for (uint32_t i=0;i<values.size();++i) {if (i) out<<',';out<<splash::json::quote(keys[i])<<':';values[i].write(out);}out<<'}';
}
bool exactChain(const std::array<Comparison,3> &values) {
  return std::all_of(values.begin(),values.end(),[](const Comparison &c){return !c.mismatches && !c.nonfinite;});
}
bool exactProbes(const ProbeComparisons &values) {
  return std::all_of(values.begin(),values.end(),[](const BitComparison &c){return c.exact();});
}
'''


CASE = r'''
bool runCase(MetalBackend &backend,const NativeStoreView &store,
    uint32_t rows,const std::string &pattern,uint32_t pairs,
    double maxL2,double minCosine,bool strict,std::ostream &out) {
  require(strict && pairs>=2 && !(pairs%2),"adaptive strict parity/even timing pairs required");
  const uint32_t layer=store.layerIndex;
  std::vector<uint32_t> hot(512);for (uint32_t expert=0;expert<512;++expert) hot[expert]=expert;
  std::vector<uint16_t> hidden(uint64_t{rows}*2560);auto ids=patternIDs(rows,hot,pattern);
  const char *rawInput=std::getenv("FLASH_INT8_STORE_INPUT"),*rawIDs=std::getenv("FLASH_INT8_STORE_ROUTE_IDS");
  require(bool(rawInput)==bool(rawIDs),"raw hidden and IDs must be supplied together");
  if (rawInput) {hidden=readFile<uint16_t>(rawInput,hidden.size());ids=readFile<int64_t>(rawIDs,ids.size());}
  else hidden=syntheticHidden(rows,inputPolicy());
  const auto rms=rowRMS(hidden,rows);const auto packed=ref::pack(hidden,ids,rows,10,kSticky);
  require(packed.diagnostic==kSticky,"fixture invalid/duplicate top10 ownership");
  const auto jobs=ref::makeJobs(packed,32);
  const auto input=upload(backend,hidden,"adaptive shared BF16 input");
  const auto expertIDs=upload(backend,ids,"adaptive shared original I64 route IDs");
  std::vector<uint16_t> weights(ids.size());
  for (uint64_t route=0;route<ids.size();++route) weights[route]=bf16(float((route%10)+1)/55.0f);
  const auto route=upload(backend,weights,"adaptive shared unequal route weights");
  const auto shared=upload(backend,std::vector<uint16_t>(uint64_t{rows}*2560,0),"adaptive shared zeros");
  const auto sharedGate=upload(backend,std::vector<uint16_t>(rows,0),"adaptive shared gate zeros");
  std::array<FlashMoEBlockedScratch,2> scratch;std::array<CommandGraph,2> graphs;
  std::array<MetalBuffer,2> output,diagnostic;std::vector<Guard> guards;
  for (uint32_t which=0;which<2;++which) {
    constexpr auto tile=FlashMoEBlockedTile::M32N64;
    scratch[which]=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,scratch[which],guards);
    output[which]=guarded(backend,uint64_t{rows}*2560*2,guards);diagnostic[which]=guarded(backend,4,guards);
    *static_cast<uint32_t *>(diagnostic[which].contents())=kSticky;
    addMoEBlockedPack(graphs[which],input,expertIDs,scratch[which],diagnostic[which],rows,tile);
    store.addGateUp(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    store.addDownScatter(graphs[which],layer,scratch[which],diagnostic[which],rows,tile);
    addCombine(graphs[which],scratch[which].scatteredDown,expertIDs,route,shared,sharedGate,
        output[which],diagnostic[which],rows,2560,512,10);
  }
  matchingNativeGraphs(graphs[0].dispatches(),graphs[1].dispatches());
  const uint64_t routes=uint64_t{rows}*10;
  ProbeBuffers controlProbe(backend,routes,guards),candidateProbe(backend,routes,guards);
  const auto copiedM32Probe=tailCommands(graphs[1].dispatches(),rows,0,&controlProbe);
  const char *selection=std::getenv("ADAPTIVE_EXPERT_TAIL_SEP21_VARIANT");
  const uint32_t selected=selection ? envNumber("ADAPTIVE_EXPERT_TAIL_SEP21_VARIANT",1,2)-1 : UINT32_MAX;
  struct Result final {
    uint32_t index=0;std::vector<ComputeDispatch> commands,probeCommands;
    std::array<Comparison,3> initial,final,probeChain;ProbeComparisons probes;
    std::vector<CommandTiming> controlTimes,candidateTimes;std::vector<std::string> failures;
    double controlWarmGPU=0,candidateWarmGPU=0;uint32_t warmPairs=0;
    bool initialChecked=false,finalChecked=false,probeChecked=false,eligible=false;
    std::string outputHash;
  };
  std::vector<Result> results;
  for (uint32_t index=0;index<tailVariants.size();++index) if (selected==UINT32_MAX || selected==index) {
    Result r;r.index=index;r.commands=tailCommands(graphs[1].dispatches(),rows,tailVariants[index].tailRows);
    r.probeCommands=tailCommands(graphs[1].dispatches(),rows,tailVariants[index].tailRows,&candidateProbe);
    results.push_back(std::move(r));
  }
  require(!results.empty() && pairs%results.size()==0,"adaptive balanced position sweep differs");
  const auto healthy=[&] {
    for (const auto &d:diagnostic)
      require(*static_cast<const uint32_t *>(d.contents())==kSticky,"adaptive sticky diagnostic changed");
    for (const auto &g:guards) require(g.clean(),"adaptive guarded scratch/probe/output overwritten");
  };
  const auto poison=[&] {
    std::memset(scratch[1].packedActivated.contents(),0xff,scratch[1].packedActivated.sizeBytes());
    std::memset(scratch[1].scatteredDown.contents(),0xff,scratch[1].scatteredDown.sizeBytes());
    std::memset(output[1].contents(),0xff,output[1].sizeBytes());
    *static_cast<uint32_t *>(diagnostic[1].contents())=kSticky;
  };
  const auto chain=[&] {
    return std::array<Comparison,3>{
        compare(scratch[0].packedActivated,scratch[1].packedActivated,routes*640,maxL2,minCosine),
        compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes*2560,maxL2,minCosine),
        compare(output[0],output[1],uint64_t{rows}*2560,maxL2,minCosine)};
  };
  const auto fail=[&](Result &r,const char *reason) {r.failures.emplace_back(reason);r.eligible=false;};
  // Native baseline and copied M32 probe qualification are entirely untimed.
  (void)backend.submitCommand(graphs[0].dispatches());healthy();checkBuckets(scratch[0],packed,jobs);
  const auto controlHash=digest(output[0]);
  require(exactChain({compare(scratch[0].packedActivated,scratch[0].packedActivated,routes*640,maxL2,minCosine),
      compare(scratch[0].scatteredDown,scratch[0].scatteredDown,routes*2560,maxL2,minCosine),
      compare(output[0],output[0],uint64_t{rows}*2560,maxL2,minCosine)}),"native M32 nonfinite full chain");
  poison();controlProbe.poison();(void)backend.submitCommand(copiedM32Probe);
  healthy();checkBuckets(scratch[1],packed,jobs);
  const auto copiedM32Chain=chain();const auto copiedM32ProbeFinite=compareProbes(controlProbe,controlProbe,routes);
  const bool copiedM32Qualified=exactChain(copiedM32Chain) && exactProbes(copiedM32ProbeFinite);
  for (auto &r:results) {
    if (!copiedM32Qualified) {fail(r,"copied M32 probe full chain differs or raw/scaled probes nonfinite");continue;}
    poison();candidateProbe.poison();(void)backend.submitCommand(r.probeCommands);
    healthy();checkBuckets(scratch[1],packed,jobs);r.probeChain=chain();
    r.probes=compareProbes(controlProbe,candidateProbe,routes);r.probeChecked=true;
    if (!exactChain(r.probeChain) || !exactProbes(r.probes)) {
      fail(r,"adaptive probe raw F32/scaled BF16 or complete BF16 chain differs");continue;
    }
    poison();(void)backend.submitCommand(r.commands);healthy();checkBuckets(scratch[1],packed,jobs);
    r.initial=chain();r.initialChecked=true;r.eligible=exactChain(r.initial);
    if (!r.eligible) fail(r,"adaptive non-probe complete BF16 chain differs before timing");
  }
  // Every accepted candidate receives at least 100ms of measured GPU warm work
  // and its own matched production control. No CPU buffer inspection occurs in
  // this loop; all diagnostics, canaries and bucket/output scans follow it.
  for (auto &r:results) if (r.eligible) {
    for (;r.controlWarmGPU<.100 || r.candidateWarmGPU<.100;++r.warmPairs) {
      require(r.warmPairs<10000,"adaptive GPU warm duration unavailable");
      CommandTiming ct,vt;
      if (r.warmPairs&1) {vt=backend.submitCommand(r.commands);ct=backend.submitCommand(graphs[0].dispatches());}
      else {ct=backend.submitCommand(graphs[0].dispatches());vt=backend.submitCommand(r.commands);}
      require(ct.gpuSeconds>0 && std::isfinite(ct.gpuSeconds) && vt.gpuSeconds>0 && std::isfinite(vt.gpuSeconds),
          "adaptive GPU warm timestamps unavailable");
      r.controlWarmGPU+=ct.gpuSeconds;r.candidateWarmGPU+=vt.gpuSeconds;
    }
    healthy();checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
    if (!exactChain(chain())) fail(r,"adaptive full BF16 chain changed during warm-up");
  }
  healthy();require(digest(output[0])==controlHash,"native baseline changed before paired timings");
  // Timed region: submissions only, balanced rotated positions and alternating
  // matched pair order. No .contents(), diagnostics, guards or output scans.
  for (uint32_t pair=0;pair<pairs;++pair)
    for (uint32_t order=0;order<results.size();++order) {
      const uint32_t position=(pair+order)%results.size();auto &r=results[position];
      if (!r.eligible) continue;
      if ((pair+position)&1) {
        r.candidateTimes.push_back(backend.submitCommand(r.commands));
        r.controlTimes.push_back(backend.submitCommand(graphs[0].dispatches()));
      } else {
        r.controlTimes.push_back(backend.submitCommand(graphs[0].dispatches()));
        r.candidateTimes.push_back(backend.submitCommand(r.commands));
      }
    }
  // Final scans and per-variant replays are excluded from timings. Replays are
  // needed because all adaptive candidates deliberately share one scratch.
  healthy();checkBuckets(scratch[0],packed,jobs);
  require(digest(output[0])==controlHash,"timed native M32 output changed");
  bool sweepPass=copiedM32Qualified;
  for (auto &r:results) {
    if (r.initialChecked) {
      poison();(void)backend.submitCommand(r.commands);healthy();checkBuckets(scratch[1],packed,jobs);
      r.final=chain();r.finalChecked=true;r.outputHash=digest(output[1]);
      if (!exactChain(r.final)) fail(r,"adaptive complete BF16 chain differs after timed replay");
    }
    if (r.eligible && (r.controlTimes.size()!=pairs || r.candidateTimes.size()!=pairs))
      fail(r,"adaptive paired sample count incomplete");
    for (const auto &values:{r.controlTimes,r.candidateTimes})
      for (const auto &t:values)
        if (!std::isfinite(t.gpuSeconds) || t.gpuSeconds<=0 ||
            !std::isfinite(t.wallSeconds) || t.wallSeconds<=0)
          fail(r,"adaptive nonfinite/nonpositive command timing");
    sweepPass=sweepPass && r.eligible && r.initialChecked && r.finalChecked && r.probeChecked && r.failures.empty();
  }
  require(!std::memcmp(hidden.data(),input.contents(),hidden.size()*2) &&
      !std::memcmp(ids.data(),expertIDs.contents(),ids.size()*8),"adaptive shared fixtures mutated");
  out<<"{\"layer\":"<<layer<<",\"rows\":"<<rows<<",\"native_job_tile\":32"
      <<",\"pattern\":"<<splash::json::quote(rawInput ? "raw-fixture" : pattern)
      <<",\"input_row_rms_min\":"<<rms[0]<<",\"input_row_rms_max\":"<<rms[1]
      <<",\"same_original_m32_jobs_params_grids_threads\":true,\"producer_threads\":128"
      <<",\"active_jobs\":"<<jobs.count<<",\"job_capacity\":"<<jobs.entries.size()
      <<",\"scratch_job_capacity\":"<<scratch[0].buckets.jobCapacity
      <<",\"original_dispatch_count\":"<<graphs[0].dispatches().size()
      <<",\"additional_hit_list_dispatches\":0,\"additional_prefix_dispatches\":0"
      <<",\"strict_full_bf16_required\":true,\"probe_dispatches_excluded_from_timing\":true"
      <<",\"minimum_warm_gpu_ms_per_variant_and_matched_control\":100"
      <<",\"timing_has_cpu_buffer_reads\":false,\"balanced_positions_and_pair_order\":true"
      <<",\"synthetic_input_policy\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" :
          inputPolicy()==InputPolicy::RowRMS ? "per-row RMS normalized before BF16 rounding; synthetic qualification" :
          inputPolicy()==InputPolicy::Divisor74 ? "approximate /74 synthetic fixture" : "inherited /512 synthetic fixture")
      <<",\"copied_m32_probe_qualified_against_native_control\":"<<(copiedM32Qualified ? "true" : "false")
      <<",\"copied_m32_probe_activation\":";copiedM32Chain[0].write(out);
  out<<",\"copied_m32_probe_down\":";copiedM32Chain[1].write(out);
  out<<",\"copied_m32_probe_combine\":";copiedM32Chain[2].write(out);
  out<<",\"copied_m32_probe_finite\":";writeProbes(out,copiedM32ProbeFinite);
  out<<",\"control_output_sha256\":"<<splash::json::quote(controlHash)<<",\"control_pipelines\":";
  names(out,graphs[0].dispatches());out<<",\"variants\":[";
  for (uint32_t position=0;position<results.size();++position) {
    if (position) out<<',';const auto &r=results[position];const auto &v=tailVariants[r.index];
    out<<"{\"variant\":"<<r.index+1<<",\"kind\":"<<splash::json::quote(v.name)
        <<",\"adaptive_tail_rows\":"<<v.tailRows<<",\"job_tile_rows\":32"
        <<",\"probe_qualification_completed\":"<<(r.probeChecked ? "true" : "false")
        <<",\"initial_chain_checked\":"<<(r.initialChecked ? "true" : "false")
        <<",\"final_chain_checked\":"<<(r.finalChecked ? "true" : "false")
        <<",\"strict_full_bf16_exact\":"<<(r.initialChecked && r.finalChecked && exactChain(r.initial) && exactChain(r.final) ? "true" : "false")
        <<",\"numerical_alternative\":"<<(r.probeChecked && !exactProbes(r.probes) ? "true" : "false")
        <<",\"raw_f32_and_scaled_bf16_bit_exact\":"<<(r.probeChecked && exactProbes(r.probes) ? "true" : "false")
        <<",\"screen_pass\":"<<(r.eligible && r.failures.empty() ? "true" : "false")
        <<",\"model_quality_qualified\":false,\"probes\":";writeProbes(out,r.probes);
    out<<",\"probe_activation\":";r.probeChain[0].write(out);out<<",\"probe_down\":";r.probeChain[1].write(out);
    out<<",\"probe_combine\":";r.probeChain[2].write(out);
    out<<",\"initial_activation\":";r.initial[0].write(out);out<<",\"initial_down\":";r.initial[1].write(out);
    out<<",\"initial_combine\":";r.initial[2].write(out);
    out<<",\"final_activation\":";r.final[0].write(out);out<<",\"final_down\":";r.final[1].write(out);
    out<<",\"final_combine\":";r.final[2].write(out);
    out<<",\"control_warm_gpu_ms\":"<<r.controlWarmGPU*1000
        <<",\"candidate_warm_gpu_ms\":"<<r.candidateWarmGPU*1000<<",\"warm_pairs\":"<<r.warmPairs
        <<",\"timing_attempted\":"<<(!r.candidateTimes.empty() ? "true" : "false")
        <<",\"control_gpu_ms\":";times(out,r.controlTimes,true);
    out<<",\"candidate_gpu_ms\":";times(out,r.candidateTimes,true);
    out<<",\"control_wall_ms\":";times(out,r.controlTimes,false);
    out<<",\"candidate_wall_ms\":";times(out,r.candidateTimes,false);
    out<<",\"candidate_output_sha256\":"<<splash::json::quote(r.outputHash)<<",\"candidate_pipelines\":";
    names(out,r.commands);out<<",\"probe_pipelines\":";names(out,r.probeCommands);
    out<<",\"failures\":[";for (uint32_t i=0;i<r.failures.size();++i) {if (i) out<<',';out<<splash::json::quote(r.failures[i]);}out<<"]}";
  }
  out<<"],\"screen_pass\":"<<(sweepPass ? "true" : "false")<<'}';return sweepPass;
}
'''


def replace(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise RuntimeError(f'adaptive source drift: {before!r}: {actual} != {count}')
    return source.replace(before, after)


def generate(destination: Path):
    path = Path('dev/benchmarks/prefill_moe_sep21/native_m16/generate.py')
    spec = importlib.util.spec_from_file_location('adaptive_native_m16', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(destination)
    source = (destination / 'oracle.mm').read_text()
    begin = source.index('bool runCase(')
    end = source.index('} // namespace\n\nint main(', begin)
    source = source[:begin] + SUPPORT + CASE + source[end:]
    source = replace(source, '        nativeCPU();', '        tailCPU();')
    source = source.replace('prefill-moe-sep21-native-m32-m16-one-layer-v1',
                            'adaptive-expert-tail-sep21-one-layer-v1')
    source = source.replace('native-m16-oracle', 'adaptive-expert-tail-oracle')
    source = replace(source, 'one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(1ULL<<30)',
                     'one::oneLayerPlannedBytes(metadata.layers[layerIndex])+(3ULL<<30)')
    source = replace(source, '      bool first=true;', '      bool first=true;bool sweepPass=true;')
    source = replace(source,
        'const bool exact=runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry);',
        'const bool accepted=runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry);\n'
        '          sweepPass=sweepPass && accepted;')
    source = replace(source,
        '          if (strict && !exact)\n'
        '            throw std::runtime_error("strict native M32/M16 complete BF16 chain differs");', '')
    source = replace(source,
        'out<<"],\\\"pass\\\":true,\\\"one_layer_payload_and_ranks_immutable\\\":true"',
        'out<<"],\\\"pass\\\":"<<(sweepPass ? "true" : "false")<<",\\\"one_layer_payload_and_ranks_immutable\\\":true"')
    source = replace(source,
        'std::cout<<"{\\\"pass\\\":true,\\\"gpu_executed\\\":true,\\\"report\\\":"<<splash::json::quote(argv[6])<<"}\\n";\n'
        '      return 0;',
        'std::cout<<"{\\\"pass\\\":"<<(sweepPass ? "true" : "false")<<",\\\"gpu_executed\\\":true,\\\"report\\\":"<<splash::json::quote(argv[6])<<"}\\n";\n'
        '      return sweepPass ? 0 : 2;')
    source = source.replace('native_m32_m16_gpu_parity', 'adaptive_tail_gpu_parity')
    cpu_line = next(line for line in source.splitlines() if 'std::cout<<' in line and '"cpu_checks' in line)
    source = replace(source, cpu_line,
        '        std::cout<<"{\\\"cpu_checks\\\":\\\"passed\\\",\\\"gpu_work\\\":false,\\\"payload_reads\\\":false,'
        '\\\"adaptive_tail_gpu_parity\\\":\\\"pending\\\",\\\"native_job_tile\\\":32,'
        '\\\"original_m32_job_capacity\\\":1151,\\\"uniform_active_m32_jobs\\\":1024,'
        '\\\"uniform_effective_matrix_rows_m32_m16_tail_m8_tail\\\":[32768,24576,20480]}\\n";')
    source = source.replace('complete original native Full512 expert chains; warm alternating M32/M16; one shared layer and fixtures; checks, constructor scans and final hashes excluded',
        'complete native M32 control and private adaptive original-M32-job chains; 100ms GPU warm per variant and matched control; balanced alternating pairs; no CPU buffer reads in timed loop; probes/checks/final replays excluded')
    source = source.replace('both complete scratch sets,', 'both complete scratch sets and both raw/scaled probe sets,')
    (destination / 'oracle.mm').write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination)
