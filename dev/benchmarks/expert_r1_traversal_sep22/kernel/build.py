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


def alias_include():
    # Private user helper definitions must never override weak shipping AIR
    # definitions when the original control and private AIRs are linked.
    return ''.join(f'#define gathered_mpp_{name} r1_traversal_private_gathered_mpp_{name}\n'
                   for name in HELPERS) + '#include "native_helper.metal"\n'


def entry(name, gate, swapped, tap):
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
    px, pz = (10, c) if swapped else (c, 10)
    mapping = 'uint3(physical.z,physical.y,physical.x)' if swapped else 'physical'
    helper = 'r1_traversal_private_gathered_mpp_execute_tap' if tap else 'gathered_mpp_execute'
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
    if (!tid) gathered_mpp_error(diag,2u); return;
  }}
  const uint3 logical = {mapping};
  {helper}<{str(gate).lower()}>({operands},p,logical,threads,tid,safe_a,&nonfinite{probes if tap else ''});
{completion}}}
'''


def generate(original):
    begin = original.index('template <bool Gate>\ninline void gathered_mpp_execute(')
    end = original.index('kernel void flash_gathered_mpp_gate_up_m16_n64_sg4(')
    prefix, execute = original[:end], original[begin:end]
    tap = replace(execute, 'inline void gathered_mpp_execute(',
                  'inline void r1_traversal_private_gathered_mpp_execute_tap(')
    tap = replace(tap, 'threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite) {',
                  '''threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite,
    device float *raw_g, device float *scaled_g, device float *raw_u, device float *scaled_u,
    device bfloat *bf_g, device bfloat *bf_u, device bfloat *bf_silu, device bfloat *bf_activation) {''')
    tap = replace(tap, '    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;',
                  '''    const float scale = gs[ulong(rank) * Width + n];
    const float tap_raw_g = gd[i], result = tap_raw_g * scale;''')
    tap = replace(tap, '      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;',
                  '''      const float up_scale = us[ulong(rank) * Width + n];
      const float tap_raw_u = ud[i], up_result = tap_raw_u * up_scale;''')
    tap = replace(tap, '    bfloat value = bfloat(result);',
                  '''    bfloat value = bfloat(result);
    const ulong tap_at = route * Width + n;
    raw_g[tap_at] = tap_raw_g; scaled_g[tap_at] = result; bf_g[tap_at] = value;''')
    tap = replace(tap, '      value = silu * uv;',
                  '''      raw_u[tap_at] = tap_raw_u; scaled_u[tap_at] = up_result;
      bf_u[tap_at] = uv; bf_silu[tap_at] = silu;
      value = silu * uv;
      bf_activation[tap_at] = value;''')
    # Reverse the stores/signature/name mechanically: the original helper's
    # entire expression/pointer/control body must be restored byte-for-byte.
    restored = replace(tap, 'inline void r1_traversal_private_gathered_mpp_execute_tap(',
                       'inline void gathered_mpp_execute(')
    restored = replace(restored, '''threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite,
    device float *raw_g, device float *scaled_g, device float *raw_u, device float *scaled_u,
    device bfloat *bf_g, device bfloat *bf_u, device bfloat *bf_silu, device bfloat *bf_activation) {''',
                       'threadgroup bfloat *safe_a, threadgroup atomic_uint *nonfinite) {')
    restored = replace(restored, '''    const float scale = gs[ulong(rank) * Width + n];
    const float tap_raw_g = gd[i], result = tap_raw_g * scale;''',
                       '    const float scale = gs[ulong(rank) * Width + n], result = gd[i] * scale;')
    restored = replace(restored, '''      const float up_scale = us[ulong(rank) * Width + n];
      const float tap_raw_u = ud[i], up_result = tap_raw_u * up_scale;''',
                       '      const float up_scale = us[ulong(rank) * Width + n], up_result = ud[i] * up_scale;')
    restored = replace(restored, '''    const ulong tap_at = route * Width + n;
    raw_g[tap_at] = tap_raw_g; scaled_g[tap_at] = result; bf_g[tap_at] = value;\n''', '')
    restored = replace(restored, '''      raw_u[tap_at] = tap_raw_u; scaled_u[tap_at] = up_result;
      bf_u[tap_at] = uv; bf_silu[tap_at] = silu;\n''', '')
    restored = replace(restored, '      bf_activation[tap_at] = value;\n', '')
    if restored != execute:
        raise ValueError('Tap instrumentation changed literal native arithmetic/control body')
    write(HERE / 'native_helper.metal', prefix.encode())
    candidate = '// Private native R1 CTA traversal; no floating arithmetic changes.\n' + alias_include()
    taps = '// Untimed literal native taps and per-logical-CTA completion census.\n' + alias_include() + tap
    inventory = []
    for gate in (True, False):
        for swapped in (False, True):
            base = 'r1_traversal_' + ('gu_' if gate else 'down_') + ('swapped' if swapped else 'native')
            candidate += entry(base, gate, swapped, False)
            taps += entry(base + '_tap', gate, swapped, True)
            inventory.append({'name': base, 'tap': base + '_tap', 'gate': gate,
                              'swapped': swapped, 'grid': ([10, 1, 10] if gate else [10, 1, 40] if swapped else [40, 1, 10]),
                              'threads': [128, 1, 1], 'completion_index': 'logical.z*C+logical.x'})
    write(HERE / 'candidate.metal', candidate.encode())
    write(HERE / 'taps.metal', taps.encode())
    return {'native_helper_prefix_byte_exact': prefix == original[:end],
            'tap_removed_body_byte_exact': restored == execute,
            'helper_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
            'execute_sha256': hashlib.sha256(execute.encode()).hexdigest(), 'pipelines': inventory}


def run(command):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, default=ROOT / 'build/expert-r1-traversal-sep22-kernel-v1')
    args = ap.parse_args()
    output = args.build.resolve()
    if ROOT / 'build' not in output.parents or not output.name.startswith('expert-r1-traversal-sep22-kernel-'):
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
    for name in ('build.py', 'audit.py', 'original_recipe.json', 'native_helper.metal', 'candidate.metal', 'taps.metal'):
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
    passed = bool(audit['candidate_native_FP_tree_match'] and audit['taps_shipping_FP_tree_match'])
    ready = {'schema': 'native-R1-CTA-traversal-kernel-CPU-closure-v1', 'pass': passed,
             'GPU_work': False, 'model_or_activation_payload_reads': False,
             'original_AIR_sha256': AIR_SHA,
             'candidate_native_FP_tree_match': audit['candidate_native_FP_tree_match'],
             'taps_shipping_FP_tree_match': audit['taps_shipping_FP_tree_match'],
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
