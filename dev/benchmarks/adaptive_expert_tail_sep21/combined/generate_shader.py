#!/usr/bin/env python3
"""Freeze existing SG2/K128 variant7 math with M16 descriptors for M32 tail jobs.

Both full and tail paths use exactly the same K128 loop, F32 accumulation and
static/full-row operand policy. No SG4, whole-K or FMA prefill code is composed.
This source-preparation tool never executes GPU work or reads model payloads.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[4]
FIXED = ROOT / 'build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1'
SOURCE_RELATIVE = Path('dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/candidate.metal')


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def replace(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise ValueError(f'Combined SG2K128 source drift: {before!r}: {actual} != {count}')
    return text.replace(before, after)


GATE = r'''
template <ushort TailM, bool Probe>
inline void adaptive_expert_tail_sg2k128_gate(device bfloat *a, device int8_t *g,
    device const float *gs, device int8_t *u, device const float *us,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device bfloat *out, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid, device float *raw_g,
    device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  // ORIGINAL M32 jobs/parameters, with the existing variant7 SG2 launch.
  if (!prefill_moe_sep21_memory_job<32, 2>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  if constexpr (TailM == 16) {
    if (valid_rows <= 16) {
      adaptive_expert_tail_sg2k128_gate_math<16, Probe>(a,g,gs,u,us,out,diag,group,
          rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u); return;
    }
  }
  adaptive_expert_tail_sg2k128_gate_math<32, Probe>(a,g,gs,u,us,out,diag,group,
      rank,begin,valid_rows,raw_g,raw_u,scaled_g,scaled_u);
}
'''

DOWN = r'''
template <ushort TailM, bool Probe>
inline void adaptive_expert_tail_sg2k128_down(device bfloat *a, device int8_t *w,
    device const float *s, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *count,
    device const uint *map, device bfloat *out, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads,
    uint tid, device float *raw, device bfloat *scaled) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!prefill_moe_sep21_memory_job<32, 2>(p, ranks, offsets, jobs, count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  if constexpr (TailM == 16) {
    if (valid_rows <= 16) {
      adaptive_expert_tail_sg2k128_down_math<16, Probe>(a,w,s,map,out,diag,p,group,
          rank,begin,valid_rows,raw,scaled); return;
    }
  }
  adaptive_expert_tail_sg2k128_down_math<32, Probe>(a,w,s,map,out,diag,p,group,
      rank,begin,valid_rows,raw,scaled);
}
'''


def generate(destination, fixed):
    source_path = fixed / 'source' / SOURCE_RELATIVE
    original = source_path.read_text()
    parent = json.loads((fixed / 'overlay-manifest.json').read_text())
    record = next(r for r in parent['files'] if r['path'] == SOURCE_RELATIVE.as_posix())
    if sha(original) != record['overlay_sha256']:
        raise ValueError('Frozen SG2/K128 source differs from whole-worker snapshot')
    signature = 'template <ushort M, ushort SG, ushort K = 0, bool Static = false>\n'
    job_start = original.index(signature + 'inline bool prefill_moe_sep21_memory_job(')
    gate_start = original.index(signature + 'inline void prefill_moe_sep21_memory_gate(', job_start)
    down_start = original.index(signature + 'inline void prefill_moe_sep21_memory_down(', gate_start)
    end = original.index('#define PREFILL4K_INT8TILES_GATE(', down_start)
    shared = original[:gate_start]
    validator = original[job_start:gate_start]
    gate_original, down_original = original[gate_start:down_start], original[down_start:end]
    gate_body = gate_original[gate_original.index('  constexpr ushort N = 64;'):]
    down_body = down_original[down_original.index('  constexpr ushort N = 64;'):]
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
inline void adaptive_expert_tail_sg2k128_gate_math(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device bfloat *output, device uint *diag, uint3 group,
    uint rank, uint begin, uint valid_rows, device float *raw_g,
    device float *raw_u, device bfloat *scaled_g, device bfloat *scaled_u) {
  constexpr ushort SG = 2, K = 128;
  constexpr bool Static = true;
''' + gate_body
    down_math = '''template <ushort M, bool Probe>
inline void adaptive_expert_tail_sg2k128_down_math(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *route_map,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint rank, uint begin, uint valid_rows,
    device float *raw, device bfloat *scaled) {
  constexpr ushort SG = 2, K = 128;
  constexpr bool Static = true;
''' + down_body
    source = '// Private combined SG2/K128 M16-tail experiment over original M32 jobs.\n' + shared
    source += gate_math + down_math + GATE + DOWN
    spec = importlib.util.spec_from_file_location('adaptive_tail_entries', Path(__file__).parents[1] / 'generate_shader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = []
    for tail, suffix in ((32, 'm32_control'), (16, 'm16_tail')):
        for gate in (True, False):
            for probe in (False, True):
                name = 'adaptive_expert_tail_sg2k128_sep21_' + ('gate_up_' if gate else 'down_scatter_') + suffix + ('_probe' if probe else '')
                source += module.entry(name, tail, gate, probe).replace('adaptive_expert_tail_gate<', 'adaptive_expert_tail_sg2k128_gate<').replace('adaptive_expert_tail_down<', 'adaptive_expert_tail_sg2k128_down<')
                inventory.append({'name': name, 'native_job_tile': 32, 'threads': 64, 'simdgroups': 2,
                                  'fixed_k': 128, 'static_full_row_extent': True, 'probe': probe,
                                  'minimum_tail_descriptor_m': tail})
    source += '#endif\n'
    if validator not in source or source.count('prefill_moe_sep21_memory_job<32, 2>(p, ranks, offsets, jobs, count, diag,') != 2:
        raise ValueError('Original M32/SG2 validation must occur exactly once per producer')
    if source.count('for (uint k = 0; k < 2560; k += K)') != 1 or source.count('for (uint k = 0; k < 640; k += K)') != 1:
        raise ValueError('Existing SG2/K128 per-K iteration changed')
    if 'execution_simdgroups<4>' in source or 'adaptive_expert_tail_gate_math' in source:
        raise ValueError('SG4 tail path may not be composed into SG2/K128')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'adaptive.metal').write_text(source)
    manifest = {
        'schema': 'private-combined-sg2-k128-m32-job-m16-tail-source-v1',
        'source_path': str(source_path), 'source_sha256': sha(original),
        'fixed_worker_manifest_sha256': sha((fixed / 'overlay-manifest.json').read_bytes()),
        'generator_source_sha256': sha(Path(__file__).read_bytes()),
        'entry_generator_path': str(Path(__file__).parents[1] / 'generate_shader.py'),
        'entry_generator_sha256': sha((Path(__file__).parents[1] / 'generate_shader.py').read_bytes()),
        'shader_sha256': sha(source), 'original_validator_sha256': sha(validator),
        'validator_byte_identical': True, 'gate_original_helper_sha256': sha(gate_original),
        'down_original_helper_sha256': sha(down_original), 'pipelines': inventory,
        'scope_policy': 'SG2 fixed K128 and Static=true for BOTH M32 and adaptive M16; per-K loop/order unchanged',
        'native_job_tile': 32, 'fixed_k': 128, 'producer_threads': 64, 'producer_simdgroups': 2,
        'gate_k_steps': 20, 'down_k_steps': 5, 'accumulation': 'original ordered F32 multiply_accumulate loop; late F32 scale',
        'tail_policy': 'valid_rows <=16 -> M16; otherwise existing M32; original M32 jobs/params/grid unchanged',
        'static_full_row_extent_policy': 'unchanged valid_rows == math descriptor M branch within each K128 step',
        'sg4_tail_used': False, 'fma_prefill_composed': False,
        'gpu_executed': False, 'model_payload_bytes_read': 0,
        'raw_f32_scaled_bf16_full_chain_exactness': 'pending root GPU qualification against existing variant7',
    }
    (destination / 'shader-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': str(destination), 'pipelines': len(inventory), 'gpu_executed': False,
                      'model_payload_bytes_read': 0, 'native_job_tile': 32, 'threads': 64, 'k': 128}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--fixed-worker', type=Path, default=FIXED)
    args = parser.parse_args()
    generate(args.destination.resolve(), args.fixed_worker.resolve())
