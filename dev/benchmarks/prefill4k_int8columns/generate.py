from pathlib import Path
import argparse

def replace(text,before,after,count=1):
    if text.count(before)!=count:raise RuntimeError(f'wide-column source drift: {before!r}')
    return text.replace(before,after)

BODY=r'''
  require(m ==32,"wide-column oracle requires original M32 jobs");
  constexpr std::array<uint32_t,2> widths{128,256};
  std::array<MetalBuffer,3> linear;
  const auto rounded=[](uint64_t x) { return (x +16383) &~uint64_t(16383); };
  const uint64_t routes=uint64_t(rows) *10;
  const uint64_t auditBytes=rounded(routes *640 *4 +64) *2 +rounded(routes *2560 *4 +64);
  auto auditAdmission=governor.tryReserve(auditBytes);
  require(bool(auditAdmission),"wide-column audit buffers denied before allocation");
  linear[0]=guarded(backend,routes *640 *4,guards);
  linear[1]=guarded(backend,routes *640 *4,guards);
  linear[2]=guarded(backend,routes *2560 *4,guards);auditAdmission->commit();
  std::vector<prefill4k_int8columns::HitCommands> plans,audits;
  plans.reserve(2);audits.reserve(2);
  for (uint32_t n :widths) {
    plans.emplace_back(graphs[1].dispatches(),rows,n,n /16,linear);
    audits.emplace_back(graphs[1].dispatches(),rows,n,n /16,linear,true);
  }
  const uint32_t rejectionChecks=checkRejections(backend,weights,store,layer,scratch[1],diagnostic[1],rows,tile);
  const auto healthy=[&] {
    for (const auto &d :diagnostic) require(*static_cast<const uint32_t *>(d.contents()) ==kSticky,"wide-column sticky diagnostics changed");
    for (const auto &g :guards) require(g.clean(),"wide-column output/audit canary changed");
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
  std::array<std::vector<CommandTiming>,2> controlTimes,candidateTimes;
  std::array<uint64_t,2> certified{};
  std::array<double,2> maximumAbsoluteError{},maximumEnvelopeRatio{},maximumNearZeroAbsoluteError{};
  const auto *payload=static_cast<const uint8_t *>(store.immutableWeightBuffers()[layer *2].contents());
  (void)backend.submitCommand(graphs[0].dispatches());healthy();
  const auto controlHash=digest(output[0]);
  for (uint32_t variant=0;variant <2;++variant) {
    poison();(void)backend.submitCommand(plans[variant].commands);healthy();
    checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
    require(equalBuffers(),"wide-column warm full BF16 projection chain differs");
    (void)exactMissSlices(scratch,packed,ids,hot);
    const std::array<std::string,3> fastHashes{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
    for (const auto &buffer :linear) std::fill_n(static_cast<float *>(buffer.contents()),buffer.sizeBytes() /4,std::numeric_limits<float>::quiet_NaN());
    (void)backend.submitCommand(audits[variant].commands);healthy();
    require(std::array<std::string,3>{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])} ==fastHashes,
        "wide-column audit full outputs differ from timed fast path");
    for (uint32_t plane=0;plane <3;++plane) {
      const uint64_t beforeSamples=certified[variant];
      const uint32_t k=plane ==2 ? 640 :2560,n=plane ==2 ? 2560 :640;
      const auto *inputWords=static_cast<const uint16_t *>(plane ==2 ? scratch[1].packedActivated.contents() :scratch[1].buckets.packedInputs.contents());
      const auto *codes=reinterpret_cast<const int8_t *>(payload +metadata.layers[layer].codes[plane].offset);
      const auto *scales=reinterpret_cast<const float *>(payload +metadata.layers[layer].scales[plane].offset);
      const auto *actual=static_cast<const float *>(linear[plane].contents());
      std::vector<uint32_t> columns{0,1,31,63,64,127,128,255,256,511,639,n -1};
      std::sort(columns.begin(),columns.end());columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
      uint32_t expertsChecked=0;
      for (uint32_t expert=0;expert <512 &&expertsChecked <16;++expert) {
        if (!packed.counts[expert] ||!std::binary_search(hot.begin(),hot.end(),expert)) continue;
        ++expertsChecked;
        const uint32_t rank=uint32_t(std::lower_bound(hot.begin(),hot.end(),expert) -hot.begin());
        for (uint32_t row :{packed.offsets[expert],packed.offsets[expert +1] -1})
          for (uint32_t col :columns) {
            const auto certificate=prefill4k_int8columns::certificate(codes +(uint64_t(rank) *n +col) *k,k,
                scales[uint64_t(rank) *n +col],inputWords +uint64_t(row) *k);
            const double got=actual[uint64_t(row) *n +col];
            require(std::isfinite(got),"wide-column raw scaled F32 projection audit absent or nonfinite");
            const double error=std::abs(got -certificate.dotScaled);
            const double envelope=prefill4k_int8columns::comparisonEnvelope(got,certificate);
            require(error <=envelope,"wide-column sampled FP64 absolute envelope failed");
            maximumAbsoluteError[variant]=std::max(maximumAbsoluteError[variant],error);
            if (envelope >0) maximumEnvelopeRatio[variant]=std::max(maximumEnvelopeRatio[variant],error /envelope);
            if (std::abs(certificate.dotScaled) <=1e-5) maximumNearZeroAbsoluteError[variant]=std::max(maximumNearZeroAbsoluteError[variant],error);
            ++certified[variant];
          }
      }
      uint32_t activeHits=0;for (uint32_t expert :hot) activeHits +=uint32_t(packed.counts[expert] !=0);
      const uint64_t expected=uint64_t(std::min(activeHits,16u)) *2 *columns.size();
      require(expertsChecked ==std::min(activeHits,16u),"wide-column FP64 expert cardinality differs");
      require(certified[variant] -beforeSamples ==expected,"wide-column FP64 sample cardinality differs");
    }
    require(equalBuffers(),"wide-column certified full BF16 projection chain differs");
  }
  // All full BF16 tensors, native jobs, finite guards and raw-F32 FP64 windows
  // are qualified above before the first submitted matched timing pair.
  for (uint32_t pair=0;pair <pairs;++pair)
    for (uint32_t order=0;order <2;++order) {
      const uint32_t variant=(pair +order) %2;poison();
      if ((pair +variant) &1) {
        candidateTimes[variant].push_back(backend.submitCommand(plans[variant].commands));
        controlTimes[variant].push_back(backend.submitCommand(graphs[0].dispatches()));
      } else {
        controlTimes[variant].push_back(backend.submitCommand(graphs[0].dispatches()));
        candidateTimes[variant].push_back(backend.submitCommand(plans[variant].commands));
      }
      healthy();checkBuckets(scratch[0],packed,jobs);checkBuckets(scratch[1],packed,jobs);
      require(equalBuffers(),"wide-column timed full BF16 chain differs");
      (void)exactMissSlices(scratch,packed,ids,hot);
      require(digest(output[0]) ==controlHash,"wide-column persisted control changed");
    }
  out <<"{\"layer\":" <<layer <<",\"rows\":" <<rows <<",\"native_job_tile\":32,\"pattern\":" <<splash::json::quote(rawInput ? "raw-fixture" :pattern)
      <<",\"normalized_synthetic_input\":" <<(!rawInput &&!edge ? "true" :"false") <<",\"actual_code_cancellation_fixture\":" <<(edge ? "true" :"false")
      <<",\"original_jobs_and_parameters_unchanged\":true,\"additional_prefix_dispatches\":0,\"original_dispatch_count\":" <<graphs[1].dispatches().size()
      <<",\"audit_admitted_bytes\":" <<auditBytes <<",\"atomic_graph_rejection_checks\":" <<rejectionChecks
      <<",\"shared_fixture_and_source_graph\":true,\"rotation\":\"(pair+order)%2; paired control/candidate order alternates\",\"control_output_sha256\":" <<splash::json::quote(controlHash)
      <<",\"variants\":[";
  for (uint32_t variant=0;variant <2;++variant) {
    if (variant) out <<',';poison();(void)backend.submitCommand(plans[variant].commands);healthy();
    require(equalBuffers(),"wide-column final full BF16 chain differs");
    const auto activation=compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine);
    const auto down=compare(scratch[0].scatteredDown,scratch[1].scatteredDown,routes *2560,maxL2,minCosine);
    const auto combined=compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);
    out <<"{\"tile_m\":32,\"tile_n\":" <<widths[variant] <<",\"sg\":" <<widths[variant] /16 <<",\"strict_full_bf16_exact\":true"
        <<",\"gate_ctas_per_job\":" <<(640 +widths[variant] -1) /widths[variant] <<",\"down_ctas_per_job\":" <<2560 /widths[variant]
        <<",\"fp64_projection_samples\":" <<certified[variant] <<",\"maximum_fp64_absolute_error\":" <<maximumAbsoluteError[variant]
        <<",\"maximum_fp64_error_over_absolute_envelope\":" <<maximumEnvelopeRatio[variant] <<",\"maximum_near_zero_absolute_error\":" <<maximumNearZeroAbsoluteError[variant]
        <<",\"activation\":";activation.write(out);out <<",\"down\":";down.write(out);out <<",\"combine\":";combined.write(out);
    out <<",\"control_gpu_ms\":";times(out,controlTimes[variant],true);out <<",\"candidate_gpu_ms\":";times(out,candidateTimes[variant],true);
    out <<",\"control_wall_ms\":";times(out,controlTimes[variant],false);out <<",\"candidate_wall_ms\":";times(out,candidateTimes[variant],false);
    out <<",\"candidate_output_sha256\":" <<splash::json::quote(digest(output[1])) <<",\"candidate_pipelines\":";names(out,plans[variant].commands);
    out <<",\"original_jobs_checked\":true,\"canaries_clean\":true}";
  }
  out <<"]}";
  Replay replay;replay.graph=std::move(graphs[1]);replay.hitPlan=std::move(plans.back());replay.commands=replay.hitPlan.commands;
  replay.output=output[1];replay.diagnostic=diagnostic[1];
  const auto *words=static_cast<const uint16_t *>(output[1].contents());replay.expected.assign(words,words +uint64_t(rows) *2560);
  replay.guards=std::move(guards);replay.layer=layer;replay.pattern=rawInput ? "raw-fixture" :pattern;return replay;
}
'''

def shader_source():
    s=Path('dev/benchmarks/prefill4k_int8tiles/candidate.metal').read_text()
    s=s.replace('prefill4k_int8tiles_','prefill4k_int8columns_')
    s=replace(s,'template <ushort M, ushort SG>','template <ushort M, ushort N, ushort SG>',3)
    s=s.replace('prefill4k_int8columns_job<M, SG>','prefill4k_int8columns_job<M, N, SG>')
    s=replace(s,'  constexpr ushort N = 64;\n','',2)
    s=replace(s,'if (group.x >= 10)','if (group.x >= (640 + N - 1) / N)')
    s=replace(s,'if (group.x >= 40)','if (group.x >= (2560 + N - 1) / N)')
    first=s.index('  const uint column = group.x * N;')
    s=s[:first]+s[first:].replace('  const uint column = group.x * N;','  const uint column = group.x * N;\n  const uint valid_cols = min(uint(N), 640u - column);',1)
    second=s.index('  const uint column = group.x * N;',first +100)
    s=s[:second]+s[second:].replace('  const uint column = group.x * N;','  const uint column = group.x * N;\n  const uint valid_cols = min(uint(N), 2560u - column);',1)
    s=replace(s,'dextents<int, 2>{2560, N}','dextents<int, 2>{2560, int(valid_cols)}',2)
    s=replace(s,'dextents<int, 2>{640, N}','dextents<int, 2>{640, int(valid_cols)}')
    s=replace(s,'if (uint(index[1]) >= valid_rows) continue;','if (uint(index[1]) >= valid_rows || uint(index[0]) >= valid_cols) continue;',2)
    s=replace(s,'device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,\n    uint3 group, uint3 threads, uint tid)',
        'device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,\n    device float *linearGate, device float *linearUp, uint3 group, uint3 threads, uint tid)')
    s=replace(s,'constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid)',
        'constant FlashInt8ExpertStoreParams &p, device float *linear, uint3 group, uint3 threads, uint tid)')
    s=replace(s,'  if (group.x >= (640 + N - 1) / N)','  (void)linearGate; (void)linearUp;\n  if (group.x >= (640 + N - 1) / N)')
    s=replace(s,'  if (group.x >= (2560 + N - 1) / N)','  (void)linear;\n  if (group.x >= (2560 + N - 1) / N)')
    s=replace(s,'    output[ulong(begin + index[1]) * 640 + n] = value;',
        '#if PREFILL4K_INT8COLUMNS_AUDIT\n    linearGate[ulong(begin + index[1]) * 640 + n] = gf;\n    linearUp[ulong(begin + index[1]) * 640 + n] = uf;\n#endif\n    output[ulong(begin + index[1]) * 640 + n] = value;')
    s=replace(s,'    output[ulong(route) * 2560 + n] = value;',
        '#if PREFILL4K_INT8COLUMNS_AUDIT\n    linear[ulong(begin + index[1]) * 2560 + n] = result;\n#endif\n    output[ulong(route) * 2560 + n] = value;')
    s=replace(s,'#define PREFILL4K_INT8TILES_GATE(NAME, M, SG)','#define PREFILL4K_INT8TILES_GATE(NAME, M, N, SG)')
    s=replace(s,'#define PREFILL4K_INT8TILES_DOWN(NAME, M, SG)','#define PREFILL4K_INT8TILES_DOWN(NAME, M, N, SG)')
    s=replace(s,'constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \\',
        'constant FlashInt8ExpertStoreParams &p [[buffer(11)]], device float *lg [[buffer(12)]], device float *lu [[buffer(13)]], \\')
    s=replace(s,'constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group',
        'constant FlashInt8ExpertStoreParams &p [[buffer(10)]], device float *linear [[buffer(11)]], uint3 group')
    s=replace(s,'prefill4k_int8columns_gate<M, SG>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid);',
        'prefill4k_int8columns_gate<M, N, SG>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, lg, lu, group, threads, tid);')
    s=replace(s,'prefill4k_int8columns_down<M, SG>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid);',
        'prefill4k_int8columns_down<M, N, SG>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, linear, group, threads, tid);')
    begin=s.index('// M128 requires');end=s.index('#undef PREFILL4K_INT8TILES_GATE',begin)
    s=s[:begin]+'''// Width varies; original M32 native job list is unchanged.
PREFILL4K_INT8TILES_GATE(prefill4k_int8columns_gate_up_m32_n128_sg8, 32, 128, 8)
PREFILL4K_INT8TILES_GATE(prefill4k_int8columns_gate_up_m32_n256_sg16, 32, 256, 16)
PREFILL4K_INT8TILES_DOWN(prefill4k_int8columns_down_scatter_m32_n128_sg8, 32, 128, 8)
PREFILL4K_INT8TILES_DOWN(prefill4k_int8columns_down_scatter_m32_n256_sg16, 32, 256, 16)
'''+s[end:]
    return '#ifndef PREFILL4K_INT8COLUMNS_AUDIT\n#define PREFILL4K_INT8COLUMNS_AUDIT 0\n#endif\n'+s

def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    shader=shader_source();(destination/'candidate.metal').write_text(shader)
    (destination/'audit.metal').write_text('#define PREFILL4K_INT8COLUMNS_AUDIT 1\n'+shader.replace('prefill4k_int8columns_','prefill4k_int8columnsaudit_'))
    s=Path('build/prefill4k-int8tiles/oracle.mm').read_text()
    s=replace(s,'#include "dev/benchmarks/prefill4k_int8tiles/bridge.hpp"','#include "dev/benchmarks/prefill4k_int8columns/bridge.hpp"\n#include "dev/benchmarks/prefill4k_int8columns/precision.hpp"')
    s=s.replace('prefill4k_int8tiles::HitCommands','prefill4k_int8columns::HitCommands')
    point='  const bool edge =std::getenv("PREFILL4K_INT8_HIT_EDGE") !=nullptr;'
    normalization=r'''  if (!rawInput) for (uint32_t row=0;row <rows;++row) {
    double sum=0;for (uint32_t k=0;k <2560;++k) { const double x=number(hidden[uint64_t(row) *2560 +k]);sum +=x *x; }
    const double reciprocal=1.0 /std::sqrt(sum /2560 +1e-6);
    for (uint32_t k=0;k <2560;++k) hidden[uint64_t(row) *2560 +k]=bf16(float(number(hidden[uint64_t(row) *2560 +k]) *reciprocal));
  }
'''
    s=replace(s,point,normalization+point)
    begin=s.index('  const uint32_t candidateTile = envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=s.index('\nvoid checkStoreRanks(',begin)
    s=s[:begin]+BODY+s[end:]
    begin=s.index('  const uint32_t candidateTile =envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=s.index('  functions.push_back(direct ?',begin)
    s=s[:begin]+r'''  for (const char *prefix :{"prefill4k_int8columns_","prefill4k_int8columnsaudit_"})
    for (uint32_t n :{128u,256u})
      for (const char *phase :{"gate_up","down_scatter"})
        functions.push_back(std::string(prefix) +phase +"_m32_n" +std::to_string(n) +"_sg" +std::to_string(n /16));
'''+s[end:]
    before='const uint32_t requested =(name.starts_with("prefill4k_int8tiles_") || name.starts_with("prefill4k_int8relaxed_")) && name.find("_sg") !=std::string::npos ? candidateSG *32 : name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;'
    s=replace(s,before,'const uint32_t requested =name.ends_with("_sg16") ? 512 :name.ends_with("_sg8") ? 256 :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;')
    s=replace(s,'      require(m == 16 || m == 32 || m == 64, "tile must be16/32/64");','      require(m ==32,"wide-column oracle requires unchanged native M32 jobs");')
    s=s.replace('prefill4k-current-int8-hit-only-tile-oracle-v1','prefill4k-current-int8-hit-only-column-sweep-v1')
    call='          replay = runCase(backend,weights,*store,governor,metadata,layer,rows, m, pattern, pairs, maxL2, minCosine, out);'
    extra=r'''          if (!selectedPattern &&std::string_view(pattern) =="mixed") {
            require(setenv("PREFILL4K_INT8_HIT_EDGE","1",1) ==0,"cannot enable actual-code cancellation fixture");
            out <<',';replay=runCase(backend,weights,*store,governor,metadata,layer,rows,m,"hit-concentrated",pairs,maxL2,minCosine,out);
            unsetenv("PREFILL4K_INT8_HIT_EDGE");
          }
'''
    s=replace(s,call,call+'\n'+extra)
    s=replace(s,'      cpuSelfTest();','      cpuSelfTest();prefill4k_int8columns::selfTest();')
    (destination/'oracle.mm').write_text(s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);generate(p.parse_args().destination)
