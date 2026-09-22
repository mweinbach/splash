#!/usr/bin/env python3
"""CPU-only exact old two-dot gate controls with raw/scaled F32 audits.

The arithmetic is copied from the tested SG2 whole/fixedK128 source. A distinct
Audit template specialization writes observations without weak-symbol aliases.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'dev/benchmarks/prefill_moe_sep21/memory.metal'
OUTPUT=ROOT/'dev/benchmarks/prefill_paired_gate_sep21/control.metal'


def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Control source anchor drift:'+before[:100])
    return text.replace(before,after,1)


def main():
    raw=SOURCE.read_bytes();text=raw.decode();end=text.index('template <ushort M, ushort SG, ushort K = 0, bool Static = false>\ninline void prefill_moe_sep21_memory_down')
    text=text[:end].replace('prefill_moe_sep21_memory_','prefill_paired_control_')
    text=once(text,'template <ushort M, ushort SG, ushort K = 0, bool Static = false>\ninline void prefill_paired_control_gate',
        'template <ushort M, ushort SG, ushort K = 0, bool Static = false, bool Audit = false>\ninline void prefill_paired_control_gate')
    text=once(text,'    uint3 group, uint3 threads, uint tid) {\n  if (group.x >= 10)',
        '    uint3 group, uint3 threads, uint tid, device float *rawG, device float *rawU,\n'
        '    device float *scaledG, device float *scaledU) {\n  if (group.x >= 10)')
    text=once(text,'    const float gf = gd[i] * gs, uf = ud[i] * us;',
        '    const float gf = gd[i] * gs, uf = ud[i] * us;\n'
        '    if constexpr(Audit){const ulong at=ulong(begin+index[1])*640+n;\n'
        '      rawG[at]=gd[i];rawU[at]=ud[i];scaledG[at]=gf;scaledU[at]=uf;}')
    text+=r'''
#define PAIRED_CONTROL_COMMON \
    device bfloat *a [[buffer(0)]],device int8_t *g [[buffer(1)]],device const float *gs [[buffer(2)]], \
    device int8_t *u [[buffer(3)]],device const float *us [[buffer(4)]],device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]],device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]],device bfloat *out [[buffer(9)]],device uint *diag [[buffer(10)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(11)]]
#define PAIRED_CONTROL_POSITION uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]
#define PAIRED_CONTROL_ENTRY(LABEL,K,STATIC) \
kernel void prefill_paired_control_gate_m32_n64_##LABEL(PAIRED_CONTROL_COMMON,PAIRED_CONTROL_POSITION){ \
  prefill_paired_control_gate<32,2,K,STATIC,false>(a,g,gs,u,us,ranks,offsets,jobs,count,out,diag,p,group,threads,tid,nullptr,nullptr,nullptr,nullptr); \
} \
kernel void prefill_paired_control_gate_m32_n64_##LABEL##_audit(PAIRED_CONTROL_COMMON, \
    device float *rawG [[buffer(12)]],device float *rawU [[buffer(13)]], \
    device float *scaledG [[buffer(14)]],device float *scaledU [[buffer(15)]],PAIRED_CONTROL_POSITION){ \
  prefill_paired_control_gate<32,2,K,STATIC,true>(a,g,gs,u,us,ranks,offsets,jobs,count,out,diag,p,group,threads,tid,rawG,rawU,scaledG,scaledU); \
}
PAIRED_CONTROL_ENTRY(whole_sg2,0,false)
PAIRED_CONTROL_ENTRY(k128_sg2,128,true)
#undef PAIRED_CONTROL_COMMON
#undef PAIRED_CONTROL_POSITION
#undef PAIRED_CONTROL_ENTRY
#endif
'''
    OUTPUT.write_text(text)
    out=ROOT/'build/prefill-paired-gate-sep21';out.mkdir(parents=True,exist_ok=True)
    (out/'control-source-audit.json').write_text(json.dumps({'gpu_executed':False,'payload_bytes_read':0,
        'original_source_sha256':hashlib.sha256(raw).hexdigest(),'generated_control_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'old_two_N64_MPP_runs_and_scale_SwiGLU_arithmetic_unchanged':True,'whole_SG2_and_best_fixedK128_SG2_controls':True,
        'audit_helper_specializations_unique':True},indent=2)+'\n')
    print(OUTPUT)


if __name__=='__main__':main()
