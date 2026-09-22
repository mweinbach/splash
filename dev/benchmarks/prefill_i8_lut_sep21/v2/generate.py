#!/usr/bin/env python3
"""Generate a bounded strict I8-LUT oracle from source only; never read payloads."""
from pathlib import Path
import argparse
import importlib.util


BODY = r'''
namespace lut = splash::bench::i8_lut;

std::ofstream gCheckpoints;
std::string gPattern="none";
uint32_t gVariant=0;
template <typename Writer> void checkpoint(const char *event,const Writer &writer) {
  if (!gCheckpoints.is_open()) return;
  std::ostringstream out;out<<std::setprecision(12)<<"{\"event\":"<<splash::json::quote(event)
      <<",\"pattern\":"<<splash::json::quote(gPattern)<<",\"variant\":"<<gVariant;
  writer(out);out<<"}\n";gCheckpoints<<out.str();gCheckpoints.flush();
  require(bool(gCheckpoints),"v2 checkpoint write failed");
}
template <typename Values> void integers(std::ostream &out,const Values &values) {
  out<<'[';bool first=true;for (auto value:values) {if (!first) out<<',';first=false;out<<value;}out<<']';
}
void openCheckpoints(const std::filesystem::path &report) {
  const auto path=report.string()+".checkpoints.jsonl";
  require(!std::filesystem::exists(path),"choose fresh v2 checkpoint path");
  gCheckpoints.open(path);require(bool(gCheckpoints),"cannot create v2 checkpoints");
}


uint32_t numeric(const char *raw,uint32_t minimum,uint32_t maximum) {
  const std::string value(raw);size_t consumed=0;
  require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,
      "invalid bounded integer argument");
  const auto parsed=std::stoul(value,&consumed);
  require(consumed==value.size() && parsed>=minimum && parsed<=maximum,
      "bounded integer argument outside range");
  return uint32_t(parsed);
}

std::vector<uint16_t> normalizedHidden(uint32_t rows) {
  std::vector<uint16_t> result(uint64_t{rows}*2560);
  for (uint32_t row=0;row<rows;++row) {
    double square=0;
    for (uint32_t col=0;col<2560;++col) {
      const uint64_t i=uint64_t{row}*2560+col;
      const double value=int((i*73+row*17)%257)-128;square+=value*value;
    }
    const float divisor=float(std::sqrt(square/2560));
    require(std::isfinite(divisor) && divisor>0,"synthetic row RMS differs");
    for (uint32_t col=0;col<2560;++col) {
      const uint64_t i=uint64_t{row}*2560+col;
      result[i]=bf16(float(int((i*73+row*17)%257)-128)/divisor);
    }
  }
  return result;
}

struct Exact final {
  uint64_t elements=0,mismatches=0,nonfinite=0;
  void write(std::ostream &out) const {
    out<<"{\"elements\":"<<elements<<",\"bit_mismatches\":"<<mismatches
        <<",\"nonfinite\":"<<nonfinite<<",\"pass\":"
        <<(!mismatches && !nonfinite ? "true" : "false")<<'}';
  }
  bool pass() const {return !mismatches && !nonfinite;}
};
Exact exactBF16(const MetalBuffer &a,const MetalBuffer &b,uint64_t count) {
  require(a.contents() && b.contents() && a.sizeBytes()>=count*2 && b.sizeBytes()>=count*2,
      "BF16 audit extent differs");
  const auto *x=static_cast<const uint16_t *>(a.contents());
  const auto *y=static_cast<const uint16_t *>(b.contents());Exact result;result.elements=count;
  for (uint64_t i=0;i<count;++i) {
    result.mismatches+=x[i]!=y[i];result.nonfinite+=!std::isfinite(number(x[i])) || !std::isfinite(number(y[i]));
  }
  return result;
}
Exact exactF32(const MetalBuffer &a,const MetalBuffer &b,uint64_t count) {
  require(a.contents() && b.contents() && a.sizeBytes()>=count*4 && b.sizeBytes()>=count*4,
      "F32 audit extent differs");
  const auto *x=static_cast<const float *>(a.contents());
  const auto *y=static_cast<const float *>(b.contents());Exact result;result.elements=count;
  for (uint64_t i=0;i<count;++i) {
    result.mismatches+=std::bit_cast<uint32_t>(x[i])!=std::bit_cast<uint32_t>(y[i]);
    result.nonfinite+=!std::isfinite(x[i]) || !std::isfinite(y[i]);
  }
  return result;
}

struct Audit final {
  std::array<MetalBuffer,3> raw,scaled,boundary;
  static Audit allocate(MetalBackend &backend,uint32_t rows,std::vector<Guard> &guards) {
    Audit result;
    for (uint32_t plane=0;plane<3;++plane) {
      const uint64_t count=uint64_t{rows}*10*(plane==2 ? 2560 : 640);
      result.raw[plane]=guarded(backend,count*4,guards);
      result.scaled[plane]=guarded(backend,count*4,guards);
      result.boundary[plane]=guarded(backend,count*2,guards);
    }
    return result;
  }
};

std::string controlName(uint32_t index,bool gate,bool audit=false) {
  require(index<2,"control index outside bounded screen");
  return std::string("prefill_i8_lut_")+(gate ? "gate_up_" : "down_scatter_")+
      "m32_n64_"+(index ? "fixed_k128_sg2" : "whole_sg4")+(audit ? "_audit" : "");
}

void nativeChain(CommandGraph &graph,MetalBackend &backend,const lut::BoundedPayload &payload,
    FlashMoEBlockedScratch &s,MetalBuffer input,MetalBuffer expertIDs,MetalBuffer route,
    MetalBuffer shared,MetalBuffer sharedGate,MetalBuffer output,MetalBuffer diag,uint32_t rows) {
  (void)backend;
  constexpr auto tile=FlashMoEBlockedTile::M32N64;
  const uint32_t routes=rows*10,jobs=moEBucketJobCapacity(rows,10,32);
  const auto &layer=payload.original;
  addMoEBlockedPack(graph,input,expertIDs,s,diag,rows,tile);
  const FlashInt8ExpertStoreParams params{rows,10,routes,jobs,32,512,0,0};
  graph.add("flash_int8_expert_store_gate_up_m32_n64",
      {s.buckets.packedInputs,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],
       layer.ranks,s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.packedActivated,diag},
      params,{10,jobs,1},{128,1,1});
  const FlashMoEBlockedDownParams poison{{rows,10,640,2560,512,0,0,0,
      320,uint64_t{2560}*320,20,uint64_t{2560}*20},routes,jobs,32,0};
  graph.add("flash_moe_blocked_poison_excluded_routes",
      {s.buckets.canonicalToPacked,s.scatteredDown,diag},poison,{10,routes,1},{256,1,1});
  graph.add("flash_moe_direct_a_prepare_down",{s.packedActivated,s.buckets.offsets,s.packedActivated,diag},
      FlashMoEDirectAPrepareParams{routes,640,63,0},{routes+63,1,1},{256,1,1});
  graph.add("flash_int8_expert_store_down_scatter_m32_n64",
      {s.packedActivated,layer.codes[2],layer.scales[2],layer.ranks,s.buckets.offsets,
       s.buckets.tileJobs,s.buckets.jobCount,s.buckets.routeMap,s.scatteredDown,diag},
      params,{40,jobs,1},{128,1,1});
  addCombine(graph,s.scatteredDown,expertIDs,route,shared,sharedGate,output,diag,rows,2560,512,10);
}

std::vector<ComputeDispatch> controlCommands(std::span<const ComputeDispatch> source,uint32_t index) {
  std::vector<ComputeDispatch> result(source.begin(),source.end());
  uint32_t producers=0;
  for (auto &d:result) {
    const bool gate=d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m");
    const bool down=d.pipelineName.starts_with("flash_int8_expert_store_down_scatter_m");
    if (!gate && !down) continue;
    d.pipelineName=controlName(index,gate);d.threadsPerThreadgroup={index ? 64u : 128u,1,1};++producers;
  }
  require(producers==2,"native source producers missing");return result;
}

std::vector<ComputeDispatch> uncompressedCommands(std::span<const ComputeDispatch> source,
    const lut::BoundedPayload &payload) {
  std::vector<ComputeDispatch> result(source.begin(),source.end());uint32_t producers=0;
  for (auto &d:result) {
    const bool gate=d.pipelineName.starts_with("prefill_i8_lut_gate_up_");
    const bool down=d.pipelineName.starts_with("prefill_i8_lut_down_scatter_");
    if (!gate && !down) continue;
    d.pipelineName+="_uncompressed";d.buffers[1].buffer=payload.original.codes[gate ? 0 : 2];
    if (gate) d.buffers[3].buffer=payload.original.codes[1];++producers;
  }
  require(producers==2,"matched uncompressed source producers missing");return result;
}

std::vector<ComputeDispatch> auditCommands(std::span<const ComputeDispatch> source,
    const lut::BoundedPayload &payload,const Audit &a) {
  std::vector<ComputeDispatch> result;
  for (const auto &original:source) {
    auto d=original;
    const bool gate=d.pipelineName.find("gate_up_m32_n64")!=std::string::npos;
    const bool down=d.pipelineName.find("down_scatter_m32_n64")!=std::string::npos;
    if (!gate && !down) continue;
    d.pipelineName+="_audit";
    if (gate) {
      require(d.buffers.size()==11 || d.buffers.size()==13,"gate audit binding count differs");
      d.buffers[9].buffer=a.raw[0];
      if (d.buffers.size()==11) {
        d.buffers.push_back({12,payload.luts[0]});d.buffers.push_back({13,payload.luts[1]});
      }
      d.buffers.push_back({14,a.raw[1]});d.buffers.push_back({15,a.scaled[0]});
      d.buffers.push_back({16,a.scaled[1]});d.buffers.push_back({17,a.boundary[0]});
      d.buffers.push_back({18,a.boundary[1]});
    } else {
      require(d.buffers.size()==10 || d.buffers.size()==11,"down audit binding count differs");
      d.buffers[8].buffer=a.raw[2];
      if (d.buffers.size()==10) d.buffers.push_back({11,payload.luts[2]});
      d.buffers.push_back({12,a.scaled[2]});d.buffers.push_back({13,a.boundary[2]});
    }
    result.push_back(std::move(d));
  }
  require(result.size()==2,"audit producers missing");return result;
}

struct Qualification final {
  std::array<Exact,3> raw,scaled,boundary,chain;
  bool pass() const {
    for (const auto *values:{&raw,&scaled,&boundary,&chain})
      for (const auto &value:*values) if (!value.pass()) return false;
    return true;
  }
  bool chainPass() const {return std::all_of(chain.begin(),chain.end(),[](const Exact &v){return v.pass();});}
  void write(std::ostream &out) const {
    bool first=true;out<<'{';
    for (const auto &[name,values]:std::array<std::pair<const char *,const std::array<Exact,3> *>,4>{{
        {"raw_f32",&raw},{"scaled_f32",&scaled},{"scaled_bf16",&boundary},{"full_chain_bf16",&chain}}}) {
      if (!first) out<<',';first=false;out<<splash::json::quote(name)<<":[";
      for (uint32_t plane=0;plane<3;++plane) {if (plane) out<<',';(*values)[plane].write(out);}out<<']';
    }
    out<<",\"pass\":"<<(pass() ? "true" : "false")<<'}';
  }
};

Qualification qualify(const Audit &a,const Audit &b,const FlashMoEBlockedScratch &sa,
    const FlashMoEBlockedScratch &sb,MetalBuffer oa,MetalBuffer ob,uint32_t rows) {
  Qualification result;
  for (uint32_t plane=0;plane<3;++plane) {
    const uint64_t count=uint64_t{rows}*10*(plane==2 ? 2560 : 640);
    result.raw[plane]=exactF32(a.raw[plane],b.raw[plane],count);
    result.scaled[plane]=exactF32(a.scaled[plane],b.scaled[plane],count);
    result.boundary[plane]=exactBF16(a.boundary[plane],b.boundary[plane],count);
  }
  result.chain[0]=exactBF16(sa.packedActivated,sb.packedActivated,uint64_t{rows}*10*640);
  result.chain[1]=exactBF16(sa.scatteredDown,sb.scatteredDown,uint64_t{rows}*10*2560);
  result.chain[2]=exactBF16(oa,ob,uint64_t{rows}*2560);return result;
}

std::vector<ComputeDispatch> producers(std::span<const ComputeDispatch> source,bool gate) {
  std::vector<ComputeDispatch> result;
  for (const auto &d:source)
    if (d.pipelineName.find(gate ? "gate_up_m32_n64" : "down_scatter_m32_n64")!=std::string::npos)
      result.push_back(d);
  require(result.size()==1,"isolated producer missing");return result;
}

struct TimingScope final {
  const char *name;
  std::array<std::vector<CommandTiming>,2> control,candidate;
  std::array<double,3> warm{};uint32_t warmTriplets=0;
  void write(std::ostream &out) const {
    out<<"{\"scope\":"<<splash::json::quote(name)<<",\"warm_gpu_ms\":["
        <<warm[0]<<','<<warm[1]<<','<<warm[2]<<"],\"warm_triplets\":"<<warmTriplets
        <<",\"control_names\":["<<splash::json::quote(std::string_view(name).find("matched")!=std::string_view::npos ?
          "same_descriptor_uncompressed" : "whole_sg4")<<",\"fixed_k128_sg2\"],\"against_controls\":[";
    for (uint32_t c=0;c<2;++c) {
      if (c) out<<',';out<<"{\"control_gpu_ms\":";times(out,control[c],true);
      out<<",\"candidate_gpu_ms\":";times(out,candidate[c],true);
      out<<",\"control_wall_ms\":";times(out,control[c],false);
      out<<",\"candidate_wall_ms\":";times(out,candidate[c],false);out<<'}';
    }
    out<<"]}";
  }
};

TimingScope timeScope(MetalBackend &backend,const char *name,
    const std::array<std::vector<ComputeDispatch>,2> &controls,
    std::span<const ComputeDispatch> candidate,uint32_t pairs,bool eligible) {
  TimingScope result;result.name=name;if (!eligible) return result;
  while (std::any_of(result.warm.begin(),result.warm.end(),[](double ms){return ms<150.0;})) {
    require(result.warmTriplets<4096,"bounded GPU warm failed to reach150ms");
    for (uint32_t order=0;order<3;++order) {
      const uint32_t arm=(result.warmTriplets+order)%3;
      const auto t=backend.submitCommand(arm<2 ? std::span<const ComputeDispatch>(controls[arm]) : candidate);
      require(std::isfinite(t.gpuSeconds) && t.gpuSeconds>0,"GPU warm timing unavailable");
      result.warm[arm]+=t.gpuSeconds*1000;
    }
    ++result.warmTriplets;
  }
  for (uint32_t pair=0;pair<pairs;++pair)
    for (uint32_t c=0;c<2;++c)
      for (uint32_t order=0;order<2;++order) {
        if ((pair+c+order)%2) result.candidate[c].push_back(backend.submitCommand(candidate));
        else result.control[c].push_back(backend.submitCommand(controls[c]));
      }
  return result;
}

void poisonAudit(const Audit &a) {
  for (const auto *planes:{&a.raw,&a.scaled,&a.boundary})
    for (const auto &buffer:*planes) std::memset(buffer.contents(),0xa5,buffer.sizeBytes());
}
void requirePoison(const Audit &a,bool downOnly) {
  for (const auto *planes:{&a.raw,&a.scaled,&a.boundary})
    for (uint32_t plane=downOnly ? 2 : 0;plane<3;++plane) {
      const auto &buffer=(*planes)[plane];const auto *bytes=static_cast<const uint8_t *>(buffer.contents());
      require(std::all_of(bytes,bytes+buffer.sizeBytes(),[](uint8_t value){return value==0xa5;}),
          "malformed producer overwrote audit poison");
    }
}

uint32_t negativeProducers(MetalBackend &backend,std::span<const ComputeDispatch> source,
    const Audit &audit,const FlashMoEBlockedScratch &scratch,uint32_t rows,std::vector<Guard> &guards) {
  uint32_t checked=0;
  const uint32_t capacity=moEBucketJobCapacity(rows,10,32);
  for (uint32_t kind=0;kind<7;++kind) {
    auto commands=std::vector<ComputeDispatch>(source.begin(),source.end());
    const bool downOnly=kind==6;
    if (downOnly) commands=producers(source,false);
    const auto diag=guarded(backend,4,guards);*static_cast<uint32_t *>(diag.contents())=kSticky;
    MetalBuffer operand;
    if (kind<2) operand=upload(backend,std::vector<uint32_t>(512,kind ? 512u : UINT32_MAX),"negative all-expert ranks");
    else if (kind==2 || kind==3) {
      std::vector<FlashMoEBucketJob> jobs(capacity);
      std::memcpy(jobs.data(),scratch.buckets.tileJobs.contents(),jobs.size()*sizeof(jobs[0]));
      for (auto &job:jobs) {if (kind==2) job.expert=512;else job.row_begin=rows*10;}
      operand=upload(backend,jobs,"negative expert/job row bounds");
    } else if (kind==4) operand=upload(backend,std::vector<uint32_t>{capacity+1},"negative active job count");
    else if (kind==5) {
      std::vector<uint32_t> offsets(513);std::memcpy(offsets.data(),scratch.buckets.offsets.contents(),513*4);
      offsets[512]=rows*10+1;operand=upload(backend,offsets,"negative offsets terminal bound");
    } else operand=upload(backend,std::vector<uint32_t>(uint64_t{rows}*10,rows*10),"negative canonical route bounds");
    for (auto &d:commands) {
      const bool gate=d.pipelineName.find("gate_up_m32_n64")!=std::string::npos;
      const uint32_t binding=kind<2 ? (gate ? 5 : 3) :
          kind==2 || kind==3 ? (gate ? 7 : 5) :kind==4 ? (gate ? 8 : 6) :kind==5 ? (gate ? 6 : 4) : 7;
      d.buffers[binding].buffer=operand;d.buffers[gate ? 10 : 9].buffer=diag;
    }
    poisonAudit(audit);(void)backend.submitCommand(commands);
    require(*static_cast<const uint32_t *>(diag.contents())==(kSticky|(kind<3 || kind==6 ? 1u : 2u)),
        "malformed producer sticky diagnostic differs");requirePoison(audit,downOnly);
    for (const auto &guard:guards) require(guard.clean(),"malformed producer guard changed");++checked;
  }
  return checked;
}


uint32_t negativeIDs(MetalBackend &backend,MetalBuffer input,const std::vector<int64_t> &ids,
    FlashMoEBlockedScratch &scratch,uint32_t rows,std::vector<Guard> &guards) {
  uint32_t checked=0;
  for (uint32_t kind=0;kind<3;++kind) {
    auto malformed=ids;malformed[0]=kind==0 ? -1 :kind==1 ? 512 : malformed[1];
    const auto expected=lut::guard_v2::expected(malformed,rows,10,kSticky);
    const auto routes=upload(backend,malformed,"v2 exact native malformed route IDs");
    const auto diag=guarded(backend,4,guards);*static_cast<uint32_t *>(diag.contents())=kSticky;
    CommandGraph graph;addMoEBlockedPack(graph,input,routes,scratch,diag,rows,FlashMoEBlockedTile::M32N64);
    (void)backend.submitCommand(graph.dispatches());
    const auto actualDiag=*static_cast<const uint32_t *>(diag.contents());
    const auto actualJobs=*static_cast<const uint32_t *>(scratch.buckets.jobCount.contents());
    std::array<uint32_t,512> counts{};std::array<uint32_t,513> offsets{};
    std::memcpy(counts.data(),scratch.buckets.counts.contents(),512*4);
    std::memcpy(offsets.data(),scratch.buckets.offsets.contents(),513*4);
    std::vector<uint32_t> map(ids.size()),inverse(ids.size());
    std::memcpy(map.data(),scratch.buckets.routeMap.contents(),map.size()*4);
    std::memcpy(inverse.data(),scratch.buckets.canonicalToPacked.contents(),inverse.size()*4);
    const bool guardsClean=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});
    const bool pass=actualDiag==expected.diagnostic && counts==expected.counts && offsets==expected.offsets &&
        actualJobs==expected.jobs && actualJobs<=moEBucketJobCapacity(rows,10,32) &&
        map==expected.routeMap && inverse==expected.inverse && guardsClean;
    checkpoint("malformed_id_native_bucket",[&](std::ostream &out) {
      out<<",\"kind\":"<<kind<<",\"fixture\":"<<splash::json::quote(kind==0 ? "ID -1" :kind==1 ? "ID512" :"duplicate ID")
          <<",\"native_producer\":\"addMoEBlockedPack/v1 metallib\",\"timing_included\":false"
          <<",\"expected_ids\":";integers(out,malformed);
      out<<",\"actual_diagnostic\":"<<actualDiag<<",\"expected_diagnostic\":"<<expected.diagnostic
          <<",\"actual_hist_total\":"<<std::accumulate(counts.begin(),counts.end(),uint32_t{0})
          <<",\"expected_hist_total\":"<<expected.total()<<",\"actual_excluded_count\":"<<ids.size()-offsets[512]
          <<",\"expected_excluded_count\":"<<expected.invalidRoutes.size()
          <<",\"expected_duplicate_count\":"<<expected.duplicateRoutes.size()
          <<",\"actual_job_count\":"<<actualJobs<<",\"expected_job_count\":"<<expected.jobs
          <<",\"actual_counts\":";integers(out,counts);out<<",\"expected_counts\":";integers(out,expected.counts);
      out<<",\"actual_offsets\":";integers(out,offsets);out<<",\"expected_offsets\":";integers(out,expected.offsets);
      out<<",\"route_map_exact\":"<<(map==expected.routeMap ? "true" :"false")
          <<",\"canonical_inverse_exact\":"<<(inverse==expected.inverse ? "true" :"false")
          <<",\"guards_clean\":"<<(guardsClean ? "true" :"false")<<",\"pass\":"<<(pass ? "true" :"false");
    });
    require(actualDiag==expected.diagnostic,"v2 malformed ID sticky diagnostic differs");
    require(actualJobs==expected.jobs && actualJobs<=moEBucketJobCapacity(rows,10,32),"v2 malformed ID job count differs");
    require(counts==expected.counts,"v2 exact native malformed histogram differs");
    require(offsets==expected.offsets,"v2 exact native malformed offsets differ");
    require(map==expected.routeMap && inverse==expected.inverse,"v2 exact native malformed maps differ");
    require(guardsClean,"v2 malformed ID canary changed");++checked;
  }
  return checked;
}

bool runCase(MetalBackend &backend,const lut::BoundedPayload &payload,uint32_t rows,
    const std::string &pattern,uint32_t pairs,uint32_t selection,std::ostream &out) {
  gPattern=pattern;gVariant=0;
  std::vector<uint32_t> hot(512);for (uint32_t i=0;i<512;++i) hot[i]=i;
  auto hidden=normalizedHidden(rows);auto ids=patternIDs(rows,hot,pattern);
  const char *rawInput=std::getenv("PREFILL_I8_LUT_INPUT");
  const char *rawIDs=std::getenv("PREFILL_I8_LUT_IDS");
  require(bool(rawInput)==bool(rawIDs),"raw input/IDs must be paired");
  if (rawInput) {hidden=readFile<uint16_t>(rawInput,hidden.size());ids=readFile<int64_t>(rawIDs,ids.size());}
  for (uint16_t value:hidden) require(std::isfinite(number(value)),"nonfinite input fixture");
  const auto packed=ref::pack(hidden,ids,rows,10,kSticky);
  require(packed.diagnostic==kSticky,"invalid/duplicate route IDs");
  const auto jobs=ref::makeJobs(packed,32);
  const auto input=upload(backend,hidden,"I8 LUT shared row RMS fixture");
  const auto expertIDs=upload(backend,ids,"I8 LUT shared top10 IDs");
  std::vector<uint16_t> weights(ids.size());
  for (uint64_t i=0;i<weights.size();++i) weights[i]=bf16(float(i%10+1)/55.0f);
  const auto route=upload(backend,weights,"I8 LUT unequal route weights");
  const auto shared=upload(backend,std::vector<uint16_t>(uint64_t{rows}*2560,0),"shared expert zero");
  const auto sharedGate=upload(backend,std::vector<uint16_t>(rows,0),"shared gate zero");
  std::array<FlashMoEBlockedScratch,3> scratch;std::array<CommandGraph,3> graphs;
  std::array<MetalBuffer,3> output,diag;std::array<Audit,3> audits;std::vector<Guard> guards;
  for (uint32_t arm=0;arm<3;++arm) {
    scratch[arm]=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,scratch[arm],guards);
    output[arm]=guarded(backend,uint64_t{rows}*2560*2,guards);diag[arm]=guarded(backend,4,guards);
    *static_cast<uint32_t *>(diag[arm].contents())=kSticky;
    audits[arm]=Audit::allocate(backend,rows,guards);
    nativeChain(graphs[arm],backend,payload,scratch[arm],input,expertIDs,route,shared,sharedGate,output[arm],diag[arm],rows);
  }
  std::array<std::vector<ComputeDispatch>,2> controls{
      controlCommands(graphs[0].dispatches(),0),controlCommands(graphs[1].dispatches(),1)};
  const auto healthy=[&] {
    for (const auto &d:diag) require(*static_cast<const uint32_t *>(d.contents())==kSticky,"sticky diagnostics changed");
    for (const auto &g:guards) require(g.clean(),"scratch/output guard changed");
    for (const auto &s:scratch) checkBuckets(s,packed,jobs);
  };
  for (uint32_t arm=0;arm<2;++arm) {
    (void)backend.submitCommand(controls[arm]);
    // Audit down consumes the control's saved BF16 activated boundary. The
    // candidate audit is separately bound to the whole-SG4 control below.
    const auto commonInput=controlCommands(graphs[0].dispatches(),arm);
    (void)backend.submitCommand(auditCommands(commonInput,payload,audits[arm]));
  }
  const auto controlParity=qualify(audits[0],audits[1],scratch[0],scratch[1],output[0],output[1],rows);
  out<<"{\"layer\":"<<payload.layerIndex<<",\"rows\":"<<rows<<",\"pattern\":"
      <<splash::json::quote(rawInput ? "raw-fixture" : pattern)
      <<",\"true_row_rms_fixture\":"<<(rawInput ? "false" : "true")
      <<",\"controls\":[\"whole_sg4\",\"fixed_k128_sg2\"],\"active_jobs\":"<<jobs.count
      <<",\"no_cpu_touches_between_timing_submissions\":true,\"control_vs_control\":";
  controlParity.write(out);out<<",\"variants\":[";
  bool first=true,allPass=true;
  for (uint32_t index=0;index<lut::variants.size();++index) {
    if (selection && selection!=index+1) continue;
    gVariant=index+1;
    const auto &v=lut::variants[index];
    // Arm1 is reused by the matched uncompressed control. Restore the actual
    // current-best chain and its common-input dot audit for every candidate.
    for (uint32_t c=0;c<2;++c) {
      (void)backend.submitCommand(controls[c]);
      const auto commonInput=controlCommands(graphs[0].dispatches(),c);
      (void)backend.submitCommand(auditCommands(commonInput,payload,audits[c]));
    }
    lut::Commands plan(graphs[2].dispatches(),rows,v,payload);
    lut::Commands sameInputPlan(graphs[0].dispatches(),rows,v,payload);
    const auto candidateAudit=auditCommands(sameInputPlan.commands,payload,audits[2]);
    (void)backend.submitCommand(plan.commands);
    (void)backend.submitCommand(candidateAudit);
    healthy();
    std::array<Qualification,2> initial;
    for (uint32_t control=0;control<2;++control)
      initial[control]=qualify(audits[control],audits[2],scratch[control],scratch[2],output[control],output[2],rows);
    // Preregistered before GPU: whole-SG4 raw agreement is an honest contrast.
    // Compression itself must agree with the same descriptor and K-loop over
    // saved uncompressed I8. Every candidate must retain the current best's
    // complete BF16 chain; SG2/K128 must also retain its raw/scaled arithmetic.
    lut::Commands matchedSource(graphs[1].dispatches(),rows,v,payload);
    auto matched=uncompressedCommands(matchedSource.commands,payload);
    auto matchedCommon=uncompressedCommands(sameInputPlan.commands,payload);
    auto matchedAudit=auditCommands(matchedCommon,payload,audits[1]);
    (void)backend.submitCommand(matched);(void)backend.submitCommand(matchedAudit);healthy();
    const auto initialMatched=qualify(audits[1],audits[2],scratch[1],scratch[2],output[1],output[2],rows);
    const bool sameBestLoop=v.sg==2 && v.k==128;
    const bool eligible=initialMatched.pass() && initial[1].chainPass() && (!sameBestLoop || initial[1].pass());
    checkpoint("candidate_initial_fidelity",[&](std::ostream &out) {
      out<<",\"suffix\":"<<splash::json::quote(v.suffix)<<",\"eligible_before_timing\":"<<(eligible ? "true" :"false")
          <<",\"timing_attempted\":false,\"against_whole_sg4\":";initial[0].write(out);
      out<<",\"against_current_best_sg2k128\":";initial[1].write(out);
      out<<",\"against_matched_uncompressed\":";initialMatched.write(out);
    });
    const uint32_t producerNegativeChecks=negativeProducers(backend,candidateAudit,audits[2],scratch[0],rows,guards);
    const uint32_t idNegativeChecks=negativeIDs(backend,input,ids,scratch[2],rows,guards);
    (void)backend.submitCommand(controls[0]);(void)backend.submitCommand(controls[1]);
    (void)backend.submitCommand(plan.commands);healthy();
    checkpoint("malformed_checks_before_timing_pass",[&](std::ostream &out) {
      out<<",\"producer_checks\":"<<producerNegativeChecks<<",\"id_checks\":"<<idNegativeChecks
          <<",\"clean_states_restored\":true,\"timing_attempted\":false";
    });

    // GPU-only residency and balanced pairs in all three scopes. No CPU
    // diagnostics, output reads, scans, hashes or writes occur between them.
    std::array<std::vector<ComputeDispatch>,2> gateControls{producers(controls[0],true),producers(controls[1],true)};
    std::array<std::vector<ComputeDispatch>,2> downControls{
        producers(controlCommands(graphs[0].dispatches(),0),false),
        producers(controlCommands(graphs[0].dispatches(),1),false)};
    downControls[1][0].buffers[8].buffer=scratch[1].scatteredDown;
    downControls[1][0].buffers[9].buffer=diag[1];
    auto gateCandidate=producers(plan.commands,true);
    auto downCandidate=producers(sameInputPlan.commands,false);
    downCandidate[0].buffers[8].buffer=scratch[2].scatteredDown;downCandidate[0].buffers[9].buffer=diag[2];
    std::array<std::vector<ComputeDispatch>,2> matchedControls{matched,controls[1]};
    std::array<std::vector<ComputeDispatch>,2> matchedGateControls{producers(matched,true),gateControls[1]};
    std::array<std::vector<ComputeDispatch>,2> matchedDownControls{producers(matchedCommon,false),downControls[1]};
    matchedDownControls[0][0].buffers[8].buffer=scratch[1].scatteredDown;
    matchedDownControls[0][0].buffers[9].buffer=diag[1];
    std::array<TimingScope,6> measured{
        timeScope(backend,"full_chain_whole_and_current_best",controls,plan.commands,pairs,eligible),
        timeScope(backend,"gate_up_only_whole_and_current_best",gateControls,gateCandidate,pairs,eligible),
        timeScope(backend,"down_only_whole_and_current_best_common_original_bf16",downControls,downCandidate,pairs,eligible),
        timeScope(backend,"full_chain_matched_and_current_best",matchedControls,plan.commands,pairs,eligible),
        timeScope(backend,"gate_up_only_matched_and_current_best",matchedGateControls,gateCandidate,pairs,eligible),
        timeScope(backend,"down_only_matched_and_current_best_common_original_bf16",matchedDownControls,downCandidate,pairs,eligible)};
    // Recompute isolated audits and complete chains after the timing scope.
    for (uint32_t c=0;c<2;++c) {
      (void)backend.submitCommand(controls[c]);
      const auto commonInput=controlCommands(graphs[0].dispatches(),c);
      (void)backend.submitCommand(auditCommands(commonInput,payload,audits[c]));
    }
    (void)backend.submitCommand(plan.commands);(void)backend.submitCommand(candidateAudit);healthy();
    std::array<Qualification,2> final;
    for (uint32_t c=0;c<2;++c)
      final[c]=qualify(audits[c],audits[2],scratch[c],scratch[2],output[c],output[2],rows);
    (void)backend.submitCommand(matched);(void)backend.submitCommand(matchedAudit);healthy();
    const auto finalMatched=qualify(audits[1],audits[2],scratch[1],scratch[2],output[1],output[2],rows);
    const bool pass=eligible && finalMatched.pass() && final[1].chainPass() && (!sameBestLoop || final[1].pass());
    allPass=allPass && pass;

    checkpoint("candidate_final_fidelity_and_timings",[&](std::ostream &out) {
      out<<",\"pass\":"<<(pass ? "true" :"false")<<",\"timing_attempted\":"<<(eligible ? "true" :"false")
          <<",\"against_whole_sg4\":";final[0].write(out);
      out<<",\"against_current_best_sg2k128\":";final[1].write(out);
      out<<",\"against_matched_uncompressed\":";finalMatched.write(out);
      out<<",\"timing_scopes\":[";for (uint32_t i=0;i<measured.size();++i) {if (i) out<<',';measured[i].write(out);}out<<']';
    });
    (void)backend.submitCommand(plan.commands);healthy();
    if (!first) out<<',';first=false;
    out<<"{\"variant\":"<<index+1<<",\"suffix\":"<<splash::json::quote(v.suffix)
        <<",\"coefficient_representation\":\"original Q4 nibble IDs + exact signed I8 LUT/G64\""
        <<",\"operand_storage\":\"cooperative threadgroup staged B; SDK rejects cooperative input register B for SG2/SG4\""
        <<",\"preregistered_equality_policy\":\"raw/scaled/boundary/full-chain exact to same-descriptor uncompressed; full-chain exact currentbest; SG2K128 also raw/scaled/boundary exact currentbest; wholeSG4 contrast only\""
        <<",\"strict_raw_scaled_and_full_chain_prerequisites\":true,\"eligible_before_timing\":"<<(eligible ? "true" : "false")
        <<",\"timing_attempted\":"<<(eligible ? "true" : "false")
        <<",\"malformed_producer_gpu_checks\":"<<producerNegativeChecks<<",\"malformed_id_gpu_checks\":"<<idNegativeChecks
        <<",\"malformed_outputs_poison_unchanged\":true,\"malformed_guards_clean\":true"
        <<",\"initial_against_controls\":[";initial[0].write(out);out<<',';initial[1].write(out);
    out<<"],\"final_against_controls\":[";final[0].write(out);out<<',';final[1].write(out);
    out<<"],\"initial_against_same_descriptor_uncompressed\":";initialMatched.write(out);
    out<<",\"final_against_same_descriptor_uncompressed\":";finalMatched.write(out);
    out<<",\"timing_scopes\":[";
    for (uint32_t scope=0;scope<measured.size();++scope) {
      if (scope) out<<',';measured[scope].write(out);
    }
    out<<"],\"candidate_pipelines\":";names(out,plan.commands);
    out<<",\"candidate_output_sha256\":"<<splash::json::quote(digest(output[2]))
        <<",\"pass\":"<<(pass ? "true" : "false")<<'}';
  }
  require(!first,"no LUT variants selected");
  require(!std::memcmp(input.contents(),hidden.data(),hidden.size()*2) &&
      !std::memcmp(expertIDs.contents(),ids.data(),ids.size()*8),"shared fixture mutated");
  out<<"],\"guards_clean\":true,\"canonical_buckets_exact\":true,\"pass\":"
      <<(allPass ? "true" : "false")<<'}';return allPass;
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc==2 && std::string_view(argv[1])=="--cpu-self-test") {
        cpuSelfTest();lut::cpuSelfTest();lut::guard_v2::cpuSelfTest();
        const auto hidden=normalizedHidden(3);
        for (uint32_t row=0;row<3;++row) {
          double square=0;for (uint32_t col=0;col<2560;++col) {
            const double x=number(hidden[uint64_t{row}*2560+col]);square+=x*x;
          }
          require(std::sqrt(square/2560)>.999 && std::sqrt(square/2560)<1.001,"row RMS CPU golden differs");
        }
        std::cout<<"{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"payload_reads\":false,\"model_quality_qualified\":false}\n";return 0;
      }

      if (argc==5 && std::string_view(argv[1])=="--guards") {
        const uint32_t rows=numeric(argv[3],2048,2048);const std::filesystem::path report(argv[4]);
        require(!std::filesystem::exists(report),"choose fresh v2 guard report");openCheckpoints(report);
        require(setenv("SPLASH_FLASH_MOE_Q4X8","1",1)==0 && setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1)==0,
            "cannot enable original native guard topology");
        MetalBackend backend(argv[2]);std::vector<Guard> guards;
        const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
        const uint64_t reserve=std::max<uint64_t>(16ULL<<30,physical/10);
        splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
        constexpr uint64_t planned=1ULL<<30;
        auto admission=governor.tryReserve(planned);require(bool(admission),"v2 guard admission denied");
        const uint64_t before=backend.memoryStats().allocatedBytes;
        const auto hidden=normalizedHidden(rows);const auto input=upload(backend,hidden,"guard-only row RMS input");
        auto scratch=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,scratch,guards);
        std::vector<uint32_t> hot(512);std::iota(hot.begin(),hot.end(),0u);uint32_t checks=0;
        for (const char *pattern:{"spread-all","hit-concentrated"}) {
          gPattern=pattern;gVariant=0;const auto ids=patternIDs(rows,hot,pattern);
          checks+=negativeIDs(backend,input,ids,scratch,rows,guards);
        }
        const uint64_t actual=backend.memoryStats().allocatedBytes-before;
        require(actual<=planned,"v2 guard allocation exceeded1GiB reservation");admission->commit();
        std::ofstream out(report);require(bool(out),"cannot create v2 guard report");
        out<<"{\"schema\":\"prefill-i8-lut-native-guards-v2\",\"pass\":true,\"gpu_executed\":true"
            <<",\"guard_checks\":"<<checks<<",\"planned_bytes\":"<<planned<<",\"actual_bytes\":"<<actual
            <<",\"coefficient_payload_loaded\":false"
            <<",\"candidate_fidelity_measured\":false,\"timing_attempted\":false}\n";out.flush();
        require(bool(out),"v2 guard report write failed");backend.stop();return 0;
      }
      require(argc==6 && std::string_view(argv[1])=="--gpu",
          "usage: oracle --gpu METALLIB PACKDIR ROWS NEW_REPORT");
      const uint32_t rows=numeric(argv[4],2048,2048);
      const uint32_t pairs=envNumber("PREFILL_I8_LUT_PAIRS",4,32);
      require(pairs>=2 && pairs%2==0,"balanced timing requires even pairs2..32");
      const uint32_t selection=std::getenv("PREFILL_I8_LUT_VARIANT") ? envNumber("PREFILL_I8_LUT_VARIANT",1,8) : 0;
      const char *selected=std::getenv("PREFILL_I8_LUT_PATTERN");
      if (selected) require(std::string_view(selected)=="spread-all" || std::string_view(selected)=="hit-concentrated",
          "LUT bounded oracle supports spread-all/hit-concentrated");
      require(!std::filesystem::exists(argv[5]),"choose fresh LUT report path");
      openCheckpoints(argv[5]);
      const auto metadata=lut::Metadata::load(argv[3]);
      require(setenv("SPLASH_FLASH_MOE_Q4X8","1",1)==0 && setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1)==0,
          "cannot enable matched original native policy");
      MetalBackend backend(argv[2]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve=std::max<uint64_t>(16ULL<<30,physical/10);require(physical>reserve,"host reserve unavailable");
      splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      const uint64_t planned=metadata.plannedBytes()+(5ULL<<30);
      auto admission=governor.tryReserve(planned);require(bool(admission),"bounded LUT admission denied");
      const uint64_t before=backend.memoryStats().allocatedBytes;
      auto payload=lut::BoundedPayload::load(backend,metadata);
      const uint64_t certifiedBefore=payload.verify();
      std::ofstream out(argv[5]);require(bool(out),"cannot create LUT report");
      out<<std::setprecision(12)<<"{\"schema\":\"prefill-i8-lut-sep21-strict-bounded-v2\",\"model_quality_qualified\":false"
          <<",\"full_model_loaded\":false,\"production_metadata_loader_used\":false,\"planned_bytes\":"<<planned
          <<",\"host_reserve_bytes\":"<<reserve<<",\"strict_source_codes_certified_before\":"<<certifiedBefore
          <<",\"dot_audit_plane_order\":[\"gate\",\"up\",\"down\"],\"chain_audit_plane_order\":[\"activated_bf16\",\"scattered_down_bf16\",\"combined_bf16\"]"
          <<",\"original_q4_id_inputs\":"<<metadata.originalQ4IDInputsJSON
          <<",\"source_metadata_certificates\":"<<metadata.sourceMetadataCertificatesJSON
          <<",\"source_shard_hash_policy\":"<<splash::json::quote(metadata.sourceShardHashPolicy)
          <<",\"packed_manifest_sha256\":"<<splash::json::quote(metadata.manifestSHA256)
          <<",\"packed_certificate_sha256\":"<<splash::json::quote(metadata.certificateSHA256)
          <<",\"pairs_per_control\":"<<pairs<<",\"warm_gpu_minimum_ms_per_arm\":150,\"cases\":[";
      bool pass=true,first=true;
      try {
        for (const char *pattern:{"spread-all","hit-concentrated"}) {
          if (selected && std::string_view(selected)!=pattern) continue;
          std::ostringstream entry;const bool accepted=runCase(backend,payload,rows,pattern,pairs,selection,entry);
          pass=pass && accepted;if (!first) out<<',';first=false;out<<entry.str();out.flush();
          if (std::getenv("PREFILL_I8_LUT_INPUT")) break;
        }
        require(!first,"no cases selected");
        const uint64_t certifiedAfter=payload.verify();payload.checkImmutable();
        const uint64_t after=backend.memoryStats().allocatedBytes;
        require(after>=before && after-before<=planned,"bounded LUT ledger exceeded reservation");admission->commit();
        out<<"],\"strict_source_codes_certified_after\":"<<certifiedAfter
            <<",\"source_and_pack_immutable\":true,\"final_allocated_bytes\":"<<after-before
            <<",\"pass\":"<<(pass ? "true" : "false")<<"}\n";
      } catch (const std::exception &e) {
        out<<"],\"pass\":false,\"failure\":"<<splash::json::quote(e.what())<<"}\n";out.flush();throw;
      }
      out.flush();require(bool(out),"LUT report write failed");backend.stop();
      std::cout<<"{\"pass\":"<<(pass ? "true" : "false")<<",\"gpu_executed\":true,\"report\":"
          <<splash::json::quote(argv[5])<<"}\n";return pass ? 0 : 2;
    } catch (const std::exception &e) {
      checkpoint("failure",[&](std::ostream &out){out<<",\"failure\":"<<splash::json::quote(e.what());});
      std::cerr<<e.what()<<'\n';return 1;
    }
  }
}
'''


def generate(destination: Path):
    module_path=Path('dev/benchmarks/prefill_moe_sep21/native_m64/generate.py')
    spec=importlib.util.spec_from_file_location('native_m64_source',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.generate(destination)
    source=(destination/'oracle.mm').read_text()
    source=source[:source.index('namespace one = splash::flash::qmv_one_layer;')]
    start=source.index('double envPositive(')
    end=source.index('std::vector<uint32_t> coldIDs(',start)
    source=source[:start]+source[end:]
    start=source.index('struct Comparison final {')
    end=source.index('void times(',start)
    source=source[:start]+source[end:]
    source=source.replace('#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"',
        '#include "dev/benchmarks/prefill_i8_lut_sep21/v2/cache.hpp"\n#include "dev/benchmarks/prefill_i8_lut_sep21/v2/guard_reference.hpp"')
    source+=BODY
    (destination/'oracle.mm').write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
