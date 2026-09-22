#!/usr/bin/env python3
"""CPU-only isolated native R1 traversal shader build; no operand/data reads."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROGRAM = HERE.relative_to(ROOT)
TEACHER = ROOT / 'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5'
ORIGINAL_AIR = ROOT / 'build/gdn-ab-merge-qsa-sep21-worker-v1/reused/115-flash_gathered_mpp.air'
AIR_SHA = 'a0cd35e03daf13324d0308c8b4d8cee1d6d932989be6cbdc0429e6ec471d05c2'
SOURCE_PATH = Path('runtime/metal/kernels/shared/flash_gathered_mpp.metal')
SOURCE_SHA = 'f63efa224f63a3c807fafd699742716d9ff1bac2b2ec08e21cbb13540bc01245'
ABI_PATH = Path('runtime/flash/FlashGatheredMPP.hpp')
ABI_SHA = '2053df6d4e91a0e2a4f73114a1cfef04cbad321c2426ee200e4427dd8747143f'
HELPERS = ('finite', 'error', 'nan', 'sigmoid', 'rank', 'scan', 'execute')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f'Literal native shader anchor drift: {before!r}')
    return text.replace(before, after)


def alias_include(variant):
    # Private user helper definitions must never override weak shipping AIR
    # definitions when the original control and private AIRs are linked.
    return ''.join(f'#define gathered_mpp_{name} r1_duplicate_tid0_{variant}_gathered_mpp_{name}\n'
                   for name in HELPERS) + f'#include "{variant}_helper.metal"\n' + ''.join(f'#undef gathered_mpp_{name}\n' for name in HELPERS)


def entry(name, gate, candidate, tap):
    if gate:
        arguments = '''    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
'''
        if tap:
            arguments += '''    device float *raw_g [[buffer(10)]], device float *scaled_g [[buffer(11)]],
    device float *raw_u [[buffer(12)]], device float *scaled_u [[buffer(13)]],
    device bfloat *bf_g [[buffer(14)]], device bfloat *bf_u [[buffer(15)]],
    device bfloat *bf_silu [[buffer(16)]], device bfloat *bf_activation [[buffer(17)]],
    device uint *completed [[buffer(18)]],
'''
        operands = 'x,g,gs,u,us,ranks,ids,out,diag'
        probes = ',raw_g,scaled_g,raw_u,scaled_u,bf_g,bf_u,bf_silu,bf_activation'
    else:
        arguments = '''    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
'''
        if tap:
            arguments += '''    device float *raw_d [[buffer(8)]], device float *scaled_d [[buffer(9)]],
    device bfloat *bf_d [[buffer(10)]], device uint *completed [[buffer(11)]],
'''
        operands = 'x,w,s,w,s,ranks,ids,out,diag'
        probes = ',raw_d,scaled_d,nullptr,nullptr,bf_d,nullptr,nullptr,nullptr'
    c = 10 if gate else 40
    px, pz = c, 10
    mapping = 'physical'
    helper = 'r1_duplicate_tid0_' + ('candidate_' if candidate else 'native_') + 'gathered_mpp_execute' + ('_tap' if tap else '')
    error_helper = 'r1_duplicate_tid0_' + ('candidate_' if candidate else 'native_') + 'gathered_mpp_error'
    completion = f'''  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (!tid) completed[logical.z * {c}u + logical.x] = 1u;
''' if tap else ''
    return f'''kernel void {name}(
{arguments}    uint3 physical [[threadgroup_position_in_grid]],
    uint3 total [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {{
  threadgroup bfloat safe_a[{2560 if gate else 640}]; threadgroup atomic_uint nonfinite;
  // Reject physical extents before transposition, avoiding wrapped/aliased
  // logical groups. Every participating thread takes the same guard branch.
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      physical.x >= {px}u || physical.y >= 1u || physical.z >= {pz}u ||
      total.x != {px}u || total.y != 1u || total.z != {pz}u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {{
    if (!tid) {error_helper}(diag,2u); return;
  }}
  const uint3 logical = {mapping};
  {helper}<{str(gate).lower()}>({operands},p,logical,threads,tid,safe_a,&nonfinite{probes if tap else ''});
{completion}}}
'''


def tap_helper(execute, name, consumer):
    original_name = 'gathered_mpp_execute'
    tap = replace(execute, 'inline void ' + original_name + '(', 'inline void ' + name + '(')
    ending='threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite' + ') {'
    extended=ending[:-3] + """,
    device float *raw_g, device float *scaled_g, device float *raw_u, device float *scaled_u,
    device bfloat *bf_g, device bfloat *bf_u, device bfloat *bf_silu, device bfloat *bf_activation) {"""
    tap=replace(tap,ending,extended)
    tap=replace(tap,'    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;',"""    const float scale = gs[ulong(rank) * Width + n];
    const float tap_raw_g = gd[i], result = tap_raw_g * scale;""")
    tap=replace(tap,'      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;',"""      const float up_scale = us[ulong(rank) * Width + n];
      const float tap_raw_u = ud[i], up_result = tap_raw_u * up_scale;""")
    tap=replace(tap,'    bfloat value = bfloat(result);',"""    bfloat value = bfloat(result);
    const ulong tap_at = route * Width + n;
    raw_g[tap_at] = tap_raw_g; scaled_g[tap_at] = result; bf_g[tap_at] = value;""")
    tap=replace(tap,'      value = silu * uv;',"""      raw_u[tap_at] = tap_raw_u; scaled_u[tap_at] = up_result;
      bf_u[tap_at] = uv; bf_silu[tap_at] = silu;
      value = silu * uv;
      bf_activation[tap_at] = value;""")
    return tap


def generate(original):
    begin=original.index('template <bool Gate>\ninline void gathered_mpp_execute(')
    end=original.index('kernel void flash_gathered_mpp_gate_up_m16_n64_sg4(')
    prefix=original[:end]
    before="""  for (uint slot = 0; slot < 10; ++slot)
    if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
      gathered_mpp_error(diag, 1u);"""
    after="""  if (!tid)
    for (uint slot = 0; slot < 10; ++slot)
      if (route / 10 * 10 + slot != route && ids[route / 10 * 10 + slot] == id && !tid)
        gathered_mpp_error(diag, 1u);"""
    candidate_prefix=replace(prefix,before,after)
    restored=replace(candidate_prefix,after,before)
    if restored!=prefix:raise ValueError('Duplicate loop journal does not restore original prefix')
    journal={'schema':'duplicate-TID0-literal-prefix-source-journal-v1','original_source_sha256':SOURCE_SHA,
             'original_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),
             'candidate_prefix_sha256':hashlib.sha256(candidate_prefix.encode()).hexdigest(),
             'edit_count':1,'before':before,'after':after,'reverse_restores_original_prefix_exact':restored==prefix,
             'retained_inner_thread0_predicate':True,'all_lane_own_ID_and_rank_checks_unchanged':True,
             'floating_MPP_scan_sanitize_and_output_source_unchanged':True}
    write(HERE/'native_helper.metal',prefix.encode());write(HERE/'candidate_helper.metal',candidate_prefix.encode())
    write(HERE/'SOURCE_JOURNAL.json',(json.dumps(journal,indent=2)+'\n').encode())
    candidate='// Original R1 math; only candidate duplicate loop is thread-zero-owned.\n'+alias_include('native')+alias_include('candidate')
    taps='// Untimed same-SSA native/candidate production taps.\n'+alias_include('native')+alias_include('candidate')
    for variant,p in [('native',prefix),('candidate',candidate_prefix)]:
        execute=p[p.index('template <bool Gate>\ninline void gathered_mpp_execute('):]
        aliases=''.join(f'#define gathered_mpp_{name} r1_duplicate_tid0_{variant}_gathered_mpp_{name}\n' for name in HELPERS)
        cleanup=''.join(f'#undef gathered_mpp_{name}\n' for name in HELPERS)
        taps+=aliases+tap_helper(execute,'r1_duplicate_tid0_'+variant+'_gathered_mpp_execute_tap',False)+cleanup
    inventory=[]
    for gate in (True,False):
        for changed in (False,True):
            name='r1_duplicate_tid0_'+('gu_' if gate else 'down_')+('candidate' if changed else 'native')
            candidate+=entry(name,gate,changed,False);taps+=entry(name+'_tap',gate,changed,True)
            inventory.append({'name':name,'tap':name+'_tap','candidate':changed,'grid':[10 if gate else 40,1,10],
                              'threads':[128,1,1],'completion_index':'route*C+Ntile'})
    write(HERE/'candidate.metal',candidate.encode());write(HERE/'taps.metal',taps.encode())
    return {'native_helper_prefix_byte_exact':True,'reverse_journal_restores_original_prefix_exact':True,
            'source_journal':journal,'pipelines':inventory,'additional_GPU_buffers':0,'additional_dispatches':0,
            'additional_barriers':0,'geometry_permutation':False}


def run(command):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, default=ROOT / 'build/expert-r1-duplicate-tid0-sep22-kernel-v1')
    args = ap.parse_args()
    output = args.build.resolve()
    if ROOT / 'build' not in output.parents or not output.name.startswith('expert-r1-duplicate-tid0-sep22-kernel-'):
        raise ValueError('Kernel output must be a dedicated private kernel build directory')
    if (output / 'CPU_READY.json').exists():
        raise ValueError('A ready kernel closure is immutable; choose a fresh kernel output')
    source, abi = TEACHER / 'source' / SOURCE_PATH, TEACHER / 'source' / ABI_PATH
    manifest = json.loads((TEACHER / 'overlay-manifest.json').read_text())
    records = {r['path']: r['sha256'] for r in manifest['files']}
    if sha(source) != SOURCE_SHA or records[str(SOURCE_PATH)] != SOURCE_SHA or sha(abi) != ABI_SHA or records[str(ABI_PATH)] != ABI_SHA:
        raise ValueError('Qualified Teacher native source/ABI seal differs')
    if sha(ORIGINAL_AIR) != AIR_SHA:
        raise ValueError('Immutable original native AIR seal differs')
    source_audit = generate(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    dest = output / 'source' / PROGRAM
    dest.mkdir(parents=True, exist_ok=True)
    # Source receipts tie every compiled include back to the frozen program.
    sources = []
    for name in ('build.py', 'audit.py', 'original_recipe.json', 'SOURCE_JOURNAL.json', 'native_helper.metal', 'candidate_helper.metal', 'candidate.metal', 'taps.metal'):
        path = HERE / name
        if not path.is_file():
            raise ValueError('Kernel input is not ready: ' + str(path))
        target = dest / name
        write(target, path.read_bytes())
        sources.append({'path': str(target.relative_to(output)), 'program_path': str(PROGRAM / name), 'sha256': sha(target)})
    abi_target = output / 'source' / ABI_PATH
    write(abi_target, abi.read_bytes())
    sources.append({'path': str(abi_target.relative_to(output)), 'program_path': str(ABI_PATH), 'sha256': sha(abi_target)})
    shutil.copy2(ORIGINAL_AIR, output / 'original-gathered.air')
    # Original producer recipe, authenticated by its stored config digest.
    metal = ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.1', '-O3', '-Wall', '-Wextra', '-Werror',
             '-I' + str(output / 'source/runtime'), '-Iruntime', '-mmacosx-version-min=27.0',
             '-DSPLASH_INT8_EXPERIMENT=1', '-I' + str(output / 'source/dev/benchmarks/prefill4k_attention')]
    commands = []
    for name in ('candidate', 'taps'):
        command = metal + ['-c', str(dest / (name + '.metal')), '-o', str(output / (name + '.air'))]
        commands.append(command); run(command)
    for name in ('original-gathered', 'candidate', 'taps'):
        command = ['xcrun', 'air-opt', '-S', str(output / (name + '.air')), '-o', str(output / (name + '.ll'))]
        commands.append(command); run(command)
    command = [sys.executable, '-B', str(dest / 'audit.py'), '--build', str(output)]
    commands.append(command); run(command)
    audit = json.loads((output / 'arithmetic-audit.json').read_text())
    artifacts = [{'path': name, 'sha256': sha(output / name)} for name in
                 ('original-gathered.air', 'candidate.air', 'taps.air', 'original-gathered.ll', 'candidate.ll', 'taps.ll', 'arithmetic-audit.json')]
    passed = bool(audit['candidate_native_FP_tree_match'] and audit['taps_shipping_FP_tree_match'] and audit['duplicate_predicate_equivalence_proved'])
    ready = {'schema': 'native-R1-duplicate-TID0-kernel-CPU-closure-v1', 'pass': passed,
             'GPU_work': False, 'model_or_activation_payload_reads': False,
             'original_AIR_sha256': AIR_SHA,
             'candidate_native_FP_tree_match': audit['candidate_native_FP_tree_match'],
             'taps_shipping_FP_tree_match': audit['taps_shipping_FP_tree_match'],
             'duplicate_predicate_equivalence_proved': audit['duplicate_predicate_equivalence_proved'],
             'sources': sources, 'artifacts': artifacts, 'source_audit': source_audit,
             'original_recipe': json.loads((dest / 'original_recipe.json').read_text()),
             'commands': commands, 'actual_GPU_bit_exactness_proved': False, 'performance_proved': False}
    filename = 'CPU_READY.json' if passed else 'CPU_NOT_READY.json'
    write(output / filename, (json.dumps(ready, indent=2) + '\n').encode())
    print(json.dumps({'kernel_build': str(output), 'pass': passed, 'GPU_work': False,
                      'candidate_native_FP_tree_match': ready['candidate_native_FP_tree_match'],
                      'taps_shipping_FP_tree_match': ready['taps_shipping_FP_tree_match']}))
    if not passed:
        raise SystemExit('Actual AIR arithmetic comparison did not admit this closure')


if __name__ == '__main__':
    main()
