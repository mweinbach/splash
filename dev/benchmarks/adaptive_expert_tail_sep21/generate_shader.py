#!/usr/bin/env python3
"""Freeze native M32 validation/math with smaller descriptors only for tail jobs.

Source preparation only: this tool never loads model payloads or creates a GPU.
The original job validator remains byte-for-byte identical. Its M32 result is
the sole source of rank, begin and valid_rows for each producer invocation.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / 'runtime/metal/kernels/shared/flash_int8_expert_store.metal'


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def replace(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise ValueError(f'Adaptive expert native source drift: {before!r}: {actual} != {count}')
    return source.replace(before, after)


HEADER = '''// Private CPU/source-prepared M32 native-job adaptive-tail experiment.
// Original BF16 A, persisted signed I8 B, F32 accumulation, late F32 row scale,
// native BF16 SwiGLU and canonical down scatter are preserved.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEBuckets.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
'''


GATE_WRAPPER = r'''
template <ushort TailM, bool Probe>
inline void adaptive_expert_tail_gate(device bfloat *a, device int8_t *g,
    device const float *gs, device int8_t *u, device const float *us,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device bfloat *out, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid, device float *raw_g,
    device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  // Validate ORIGINAL M32 params/jobs once, independent of selected math M.
  if (!int8_expert_store_job<32, 4>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  if constexpr (TailM == 8) {
    if (valid_rows <= 8) {
      adaptive_expert_tail_gate_math<8, Probe>(a,g,gs,u,us,out,diag,group,
          rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u); return;
    }
  }
  if constexpr (TailM <= 16) {
    if (valid_rows <= 16) {
      adaptive_expert_tail_gate_math<16, Probe>(a,g,gs,u,us,out,diag,group,
          rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u); return;
    }
  }
  adaptive_expert_tail_gate_math<32, Probe>(a,g,gs,u,us,out,diag,group,
      rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u);
}
'''

DOWN_WRAPPER = r'''
template <ushort TailM, bool Probe>
inline void adaptive_expert_tail_down(device bfloat *a, device int8_t *w,
    device const float *s, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device const uint *map, device bfloat *out, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads,
    uint tid, device float *raw, device bfloat *scaled) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<32, 4>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  if constexpr (TailM == 8) {
    if (valid_rows <= 8) {
      adaptive_expert_tail_down_math<8, Probe>(a,w,s,map,out,diag,p,group,
          rank,begin,valid_rows,raw,scaled); return;
    }
  }
  if constexpr (TailM <= 16) {
    if (valid_rows <= 16) {
      adaptive_expert_tail_down_math<16, Probe>(a,w,s,map,out,diag,p,group,
          rank,begin,valid_rows,raw,scaled); return;
    }
  }
  adaptive_expert_tail_down_math<32, Probe>(a,w,s,map,out,diag,p,group,
      rank,begin,valid_rows,raw,scaled);
}
'''


def entry(name, tail, gate, probe):
    if gate:
        extra = '''    device float *raw_g [[buffer(12)]], device float *raw_u [[buffer(13)]],
    device bfloat *scaled_g [[buffer(14)]], device bfloat *scaled_u [[buffer(15)]],
''' if probe else ''
        outputs = 'raw_g,raw_u,scaled_g,scaled_u' if probe else 'nullptr,nullptr,nullptr,nullptr'
        return f'''kernel void {name}(
    device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]],
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]],
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]],
{extra}    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {{
  adaptive_expert_tail_gate<{tail}, {'true' if probe else 'false'}>(
      a,g,gs,u,us,ranks,offsets,jobs,count,out,diag,p,group,threads,tid,{outputs});
}}
'''
    extra = '''    device float *raw [[buffer(11)]], device bfloat *scaled [[buffer(12)]],
''' if probe else ''
    outputs = 'raw,scaled' if probe else 'nullptr,nullptr'
    return f'''kernel void {name}(
    device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]],
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]],
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]],
{extra}    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {{
  adaptive_expert_tail_down<{tail}, {'true' if probe else 'false'}>(
      a,w,s,ranks,offsets,jobs,count,map,out,diag,p,group,threads,tid,{outputs});
}}
'''


def generate(destination):
    native = NATIVE.read_text()
    start = native.index('#pragma METAL fp math_mode(fast)\ninline bfloat int8_expert_store_sigmoid(')
    gate_start = native.index('template <ushort M, ushort SG>\ninline void int8_expert_store_gate(', start)
    down_start = native.index('template <ushort M, ushort SG>\ninline void int8_expert_store_down(', gate_start)
    end = native.index('#define INT8_GATE(', down_start)
    shared = native[start:gate_start]
    job_start = shared.index('template <ushort M, ushort SG>\ninline bool int8_expert_store_job(')
    validator = shared[job_start:]
    gate_original, down_original = native[gate_start:down_start], native[down_start:end]
    gate_body = gate_original[gate_original.index('  constexpr ushort N = 64;'):]
    down_body = down_original[down_original.index('  constexpr ushort N = 64;'):]
    gate_body = replace(gate_body, 'execution_simdgroups<SG>', 'execution_simdgroups<4>')
    down_body = replace(down_body, 'execution_simdgroups<SG>', 'execution_simdgroups<4>')
    gate_body = replace(gate_body, '    const bfloat gv = bfloat(gf), uv = bfloat(uf);',
        '''    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    if constexpr (Probe) {
      const ulong at = ulong(begin + index[1]) * 640 + n;
      raw_g[at] = gd[i]; raw_u[at] = ud[i];
      scaled_g[at] = gv; scaled_u[at] = uv;
    }''')
    down_body = replace(down_body, '    const bfloat value = bfloat(result);',
        '''    const bfloat value = bfloat(result);
    if constexpr (Probe) {
      const ulong at = ulong(route) * 2560 + n;
      raw[at] = dot[i]; scaled[at] = value;
    }''')
    gate_math = '''template <ushort M, bool Probe>
inline void adaptive_expert_tail_gate_math(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device bfloat *output, device uint *diag, uint3 group,
    uint rank, uint begin, uint valid_rows, device float *raw_g,
    device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
''' + gate_body
    down_math = '''template <ushort M, bool Probe>
inline void adaptive_expert_tail_down_math(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *route_map,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint rank, uint begin, uint valid_rows,
    device float *raw, device bfloat *scaled) {
''' + down_body
    source = HEADER + shared + gate_math + down_math + GATE_WRAPPER + DOWN_WRAPPER
    inventory = []
    for tail, suffix in ((32, 'm32_control'), (16, 'm16_tail'), (8, 'm8_tail')):
        for gate in (True, False):
            for probe in (False, True):
                name = 'adaptive_expert_tail_sep21_' + ('gate_up_' if gate else 'down_scatter_') + suffix + ('_probe' if probe else '')
                source += entry(name, tail, gate, probe)
                inventory.append({'name': name, 'minimum_tail_descriptor_m': tail, 'probe': probe,
                                  'native_job_tile': 32, 'threads': 128, 'simdgroups': 4})
    source += '#endif\n'
    if validator not in source or source.count('int8_expert_store_job<32, 4>(p, ranks, offsets, jobs, count, diag,') != 2:
        raise ValueError('Original validator or exactly-once validation source check failed')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'adaptive.metal'
    path.write_text(source)
    manifest = {
        'schema': 'private-native-m32-adaptive-expert-tail-source-v1',
        'gpu_executed': False, 'model_payload_bytes_read': 0,
        'native_source': str(NATIVE.relative_to(ROOT)), 'native_source_sha256': sha(native),
        'generator_source_sha256': sha(Path(__file__).read_bytes()),
        'original_validator_sha256': sha(validator), 'validator_byte_identical': True,
        'gate_original_helper_sha256': sha(gate_original), 'down_original_helper_sha256': sha(down_original),
        'shader_sha256': sha(source), 'pipelines': inventory,
        'params_and_jobs_validated_as_m32_once_per_producer': True,
        'no_new_jobs_or_launches_or_resources_for_regular_kernels': True,
        'dynamic_a_extent': 'original min(32, expert_end - original_job_begin)',
        'numerical_policy': 'BF16 A x native signed I8 B -> F32 dot -> original F32 late row scale -> BF16; native SwiGLU/scatter',
        'descriptor_exactness': 'pending GPU raw F32, scaled BF16 and complete chain comparisons',
        'tail_m16': 'valid_rows <= 16 uses M16; otherwise original M32',
        'tail_m8': 'valid_rows <= 8 uses M8, <= 16 uses M16; otherwise original M32',
        'uniform_r2048_jobs': 1024,
        'uniform_r2048_padded_rows': {'native_m32': 32768, 'tail_m16': 24576, 'tail_m8': 20480},
    }
    (destination / 'shader-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': str(path), 'pipelines': len(inventory), 'gpu_executed': False,
                      'model_payload_bytes_read': 0, 'original_validator_byte_identical': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination.resolve())
