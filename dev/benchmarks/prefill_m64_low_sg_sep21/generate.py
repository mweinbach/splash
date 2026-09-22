#!/usr/bin/env python3
"""CPU-only prepare native M64 low-SIMD expert kernels and a bounded oracle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / 'build/adaptive-expert-tail-sg2k128-sep21'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPERS = r'''
template <ushort SG, bool Probe>
inline void m64_low_sg_gate(device bfloat *a, device int8_t *g,
    device const float *gs, device int8_t *u, device const float *us,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device bfloat *out, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid, device float *raw_g,
    device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!m64_low_sg_job<64, SG>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  m64_low_sg_gate_math<SG, Probe>(a,g,gs,u,us,out,diag,group,
      rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u);
}
template <ushort SG, bool Probe>
inline void m64_low_sg_down(device bfloat *a, device int8_t *w,
    device const float *s, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device const uint *map, device bfloat *out, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads,
    uint tid, device float *raw, device bfloat *scaled) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!m64_low_sg_job<64, SG>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  m64_low_sg_down_math<SG, Probe>(a,w,s,map,out,diag,p,group,
      rank,begin,valid_rows,raw,scaled);
}
'''


ADAPTER = r'''
std::vector<ComputeDispatch> m64LowCommands(std::span<const ComputeDispatch> source,
    uint32_t rows,uint32_t sg,const ProbeBuffers *probe=nullptr) {
  const bool control=sg==0;
  require(control || sg==2 || sg==4,"private native M64 SIMD count differs");
  std::vector<ComputeDispatch> result;bool gateSeen=false,downSeen=false;
  MetalBuffer offsets,jobs,count;
  for (const auto &original:source) {
    auto d=original;
    const uint32_t m=control ? 32 : 64;
    const std::string suffix=control ? "_m32_n64" : "_m64_n64_sg8";
    const bool gate=d.pipelineName=="flash_int8_expert_store_gate_up"+suffix;
    const bool down=d.pipelineName=="flash_int8_expert_store_down_scatter"+suffix;
    if (!gate && !down) {result.push_back(d);continue;}
    require(d.bytes.size()==1 && d.bytes[0].data &&
        d.bytes[0].sizeBytes==sizeof(FlashInt8ExpertStoreParams),"M64 low-SIMD parameter ABI differs");
    FlashInt8ExpertStoreParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));
    require(rows>=1024 && rows<=2048 && p.rows==rows && p.selections==10 &&
        p.route_capacity==rows*10 && p.tile_rows==m && p.stored_experts==512 &&
        p.job_capacity==(rows*10+m-1)/m+511 && !p.scale_group_size && !p.reserved &&
        d.threadgroups.y==p.job_capacity && d.threadgroups.z==1 &&
        d.threadsPerThreadgroup.x==(control ? 128u : 256u) &&
        d.threadsPerThreadgroup.y==1 && d.threadsPerThreadgroup.z==1,
        "M64 low-SIMD requires original native M32/M64 jobs and grids");
    for (uint32_t i=0;i<d.buffers.size();++i)
      require(d.buffers[i].index==i,"M64 low-SIMD source binding index differs");
    if (gate) {
      require(!gateSeen && !downSeen && d.buffers.size()==11 &&
          d.bytes[0].index==11 && d.threadgroups.x==10,"M64 low-SIMD gate ABI/order differs");
      require(d.buffers[0].buffer.sizeBytes()>=uint64_t(rows*10+63)*2560*2,
          "M64 full static reads require the existing 63-row input pad");
      gateSeen=true;offsets=d.buffers[6].buffer;jobs=d.buffers[7].buffer;count=d.buffers[8].buffer;
      if (probe) {
        d.buffers.push_back({12,probe->rawGate});d.buffers.push_back({13,probe->rawUp});
        d.buffers.push_back({14,probe->scaledGate});d.buffers.push_back({15,probe->scaledUp});
      }
    } else {
      require(gateSeen && !downSeen && d.buffers.size()==10 && d.bytes[0].index==10 &&
          d.threadgroups.x==40 && d.buffers[4].buffer.sameView(offsets) &&
          d.buffers[5].buffer.sameView(jobs) && d.buffers[6].buffer.sameView(count),
          "M64 low-SIMD down ABI/order/native jobs differ");
      require(d.buffers[0].buffer.sizeBytes()>=uint64_t(rows*10+63)*640*2,
          "M64 full static reads require the existing 63-row activated pad");
      downSeen=true;
      if (probe) {d.buffers.push_back({11,probe->rawDown});d.buffers.push_back({12,probe->scaledDown});}
    }
    const auto kind=gate ? "gate_up_" : "down_scatter_";
    d.pipelineName=control ? std::string("adaptive_expert_tail_sg2k128_sep21_")+kind+"m16_tail" :
        std::string("prefill_m64_low_sg_sep21_")+kind+"m64_k128_sg"+std::to_string(sg);
    if (probe) d.pipelineName+="_probe";
    d.threadsPerThreadgroup={control ? 64u : sg*32u,1,1};result.push_back(d);
  }
  require(gateSeen && downSeen && result.size()==source.size(),"M64 low-SIMD source producer inventory differs");
  return result;
}
void m64LowCPU() {
  nativeCPU();
  std::vector<uint32_t> hot(512);for (uint32_t e=0;e<512;++e) hot[e]=e;
  const auto ids=patternIDs(2048,hot,"spread-all");
  const auto packed=ref::pack(std::vector<uint16_t>(2048*2560,0x3f80),ids,2048,10,kSticky);
  const auto j32=ref::makeJobs(packed,32),j64=ref::makeJobs(packed,64);
  require(j32.count==1024 && j64.count==512 && j32.entries.size()==1151 && j64.entries.size()==831,
      "Native M32/M64 uniform forty-row job goldens differ");
  for (uint32_t n:{1u,7u,8u,16u,31u,32u,33u,40u,63u,64u,65u,2048u}) {
    uint32_t owned=0;
    for (uint32_t begin=0;begin<n;begin+=64) {
      const uint32_t valid=std::min(64u,n-begin);
      require(valid && valid<=64 && begin+63<n+63,"Native M64 padded read extent invalid");
      owned+=valid;
    }
    require(owned==n,"Native M64 masked store ownership differs");
  }
}
'''


def generate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    control_manifest = json.loads((CONTROL / 'shader-manifest.json').read_text())
    if sha(CONTROL / 'adaptive.metal') != control_manifest['shader_sha256']:
        raise RuntimeError('Qualified SG2/K128 tail control source drift')
    control_shader = (CONTROL / 'adaptive.metal').read_text()
    start = control_shader.index('template <ushort M, bool Probe>\n')
    wrappers = control_shader.index('template <ushort TailM, bool Probe>\n', start)
    shared = control_shader[:start].replace('prefill_moe_sep21_memory_job','m64_low_sg_job')
    shared = shared.replace('prefill_moe_sep21_memory_sigmoid','m64_low_sg_sigmoid')
    math = control_shader[start:wrappers]
    math = math.replace('template <ushort M, bool Probe>', 'template <ushort SG, bool Probe>')
    math = math.replace('constexpr ushort SG = 2, K = 128;', 'constexpr ushort M = 64, K = 128;')
    math = math.replace('adaptive_expert_tail_sg2k128_gate_math','m64_low_sg_gate_math')
    math = math.replace('adaptive_expert_tail_sg2k128_down_math','m64_low_sg_down_math')
    math = math.replace('prefill_moe_sep21_memory_sigmoid','m64_low_sg_sigmoid')
    if math.count('if (valid_rows == M)') != 4:
        raise RuntimeError('Expected four frozen static row extent branches')
    # Each matrix output row is independent. Read the complete static M64 tile
    # within the existing +63 global pad; retain ALL original valid-store masks.
    math = math.replace('if (valid_rows == M)', 'if (true) /* existing +63 global pad; stores remain valid_rows masked */')
    entry = load(ROOT / 'dev/benchmarks/adaptive_expert_tail_sep21/generate_shader.py', 'm64_entries').entry
    shader = shared + math + WRAPPERS
    pipelines = []
    for sg in (4,2):
        for gate in (True,False):
            for probe in (False,True):
                name = 'prefill_m64_low_sg_sep21_' + ('gate_up_' if gate else 'down_scatter_') + f'm64_k128_sg{sg}' + ('_probe' if probe else '')
                shader += entry(name,sg,gate,probe).replace('adaptive_expert_tail_gate<','m64_low_sg_gate<').replace('adaptive_expert_tail_down<','m64_low_sg_down<')
                pipelines.append(dict(name=name,simdgroups=sg,threads=sg*32,probe=probe))
    shader += '#endif\n'
    (destination / 'candidate.metal').write_text(shader)

    original = (CONTROL / 'oracle.mm').read_text()
    support_start = original.index('struct TailVariant final')
    probe_start = original.index('struct ProbeBuffers final')
    adapter_start = original.index('std::vector<ComputeDispatch> fixedCommands(')
    compare_start = original.index('struct BitComparison final')
    case_start = original.index('bool runCase(')
    main_start = original.index('} // namespace\n\nint main(',case_start)
    prefix = original[:support_start]
    prefix = prefix.replace('const std::string suffix = "_m" + std::to_string(m) + "_n64";',
                            'const std::string suffix = m==64 ? "_m64_n64_sg8" : "_m32_n64";')
    prefix = prefix.replace('{128u,1,1});','{m==64 ? 256u : 128u,1,1});')
    source = prefix + original[probe_start:adapter_start] + ADAPTER + original[compare_start:case_start]
    source += Path(__file__).with_name('case.inc').read_text()
    main = original[main_start:].replace('tailCPU();','m64LowCPU();')
    main = main.replace('adaptive-expert-tail-sg2k128-sep21-one-layer-v1','prefill-m64-low-sg-sep21-one-layer-v1')
    main = main.replace('adaptive-expert-tail-oracle','prefill-m64-low-sg-oracle')
    main = main.replace('native_job_tile\\\":32','control_native_job_tile\\\":32,\\\"candidate_native_job_tile\\\":64')
    main = main.replace('complete existing variant7 SG2/static-K128 M32 control and private SG2/static-K128 adaptive original-M32-job chains',
                        'complete qualified SG2/K128/M16-tail M32 control and private native M64 low-SIMD chains')
    cpu_line = next(line for line in main.splitlines() if 'std::cout<<' in line and '\"cpu_checks' in line)
    main = main.replace(cpu_line,
        '        std::cout<<"{\\\"cpu_checks\\\":\\\"passed\\\",\\\"gpu_work\\\":false,\\\"payload_reads\\\":false,'
        '\\\"native_m64_low_sg_gpu_parity\\\":\\\"pending\\\",\\\"native_job_tiles\\\":[32,64],'
        '\\\"uniform_active_jobs\\\":[1024,512],\\\"uniform_job_capacities\\\":[1151,831],'
        '\\\"control_sg\\\":2,\\\"candidate_sg\\\":[4,2],\\\"fixed_k\\\":128,'
        '\\\"existing_global_pad_rows\\\":63,\\\"masked_valid_rows_stores\\\":true}\\n";')
    source += main
    (destination / 'oracle.mm').write_text(source)
    inputs = [CONTROL / 'adaptive.metal', CONTROL / 'oracle.mm', CONTROL / 'shader-manifest.json',
              Path(__file__),Path(__file__).with_name('case.inc'),
              ROOT / 'dev/benchmarks/adaptive_expert_tail_sep21/generate_shader.py']
    manifest = dict(schema='private-native-m64-low-simd-source-v1',gpu_executed=False,payload_bytes_read=0,
                    inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
                    shader_sha256=sha(destination/'candidate.metal'),oracle_sha256=sha(destination/'oracle.mm'),
                    pipelines=pipelines,control='qualified M32 SG2 fixedK128 + M16 adaptive tail',
                    native_job_tiles=[32,64],candidate_fixed_k=128,candidate_static_all64rows=True,
                    global_pad_rows=63,masked_valid_rows_stores=True,additional_job_dispatches=0,
                    mandatory_raw_f32_scaled_bf16_full_chain_exact=True,
                    minimum_gpu_warm_ms=100,balanced_pairs=True,model_quality_qualified=False)
    (destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(prepared=str(destination),gpu_executed=False,payload_bytes_read=0,pipelines=len(pipelines))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path)
    generate(p.parse_args().destination.resolve())
