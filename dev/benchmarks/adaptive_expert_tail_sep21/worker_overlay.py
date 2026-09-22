#!/usr/bin/env python3
"""Freeze qualified M16 adaptive tails over the sealed exact pointwise worker.

CPU/source preparation only. Copies code, compiler inputs, metadata and root's
benchmark evidence; never reads any model payload or creates a GPU backend.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path('dev/benchmarks/adaptive_expert_tail_sep21')
CHANGED = {'FlashInt8ExpertStore', 'FlashForward', 'FlashWorker'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f'Sealed adaptive-tail source anchor drift: {before!r}')
    return text.replace(before, after)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


HOOKS = '''
// Device-free private policy/ABI probes call the exact runtime helpers.
std::string adaptiveExpertTailPipelineForCPU(const char *phase, uint32_t tileRows, bool enabled) {
  return adaptive_expert_tail_sep21::pipeline(phase, tileRows, enabled);
}
std::string adaptiveExpertTailPipelineRuntimeForCPU(const char *phase, uint32_t tileRows) {
  return pipeline(phase, static_cast<FlashMoEBlockedTile>(tileRows));
}
FlashInt8ExpertStoreParams adaptiveExpertTailParamsForCPU(uint32_t rows, uint32_t selections, uint32_t tileRows) {
  return allRowsParams(rows, selections, static_cast<FlashMoEBlockedTile>(tileRows));
}
uint32_t adaptiveExpertTailLaunchForCPU(const FlashInt8ExpertStoreParams &p) {
  return allRowsLaunch(p);
}

'''


def transform(relative, text):
    touched = ('runtime/flash/FlashInt8ExpertStore.mm', 'runtime/flash/FlashForward.cpp',
               'runtime/flash/FlashWorker.mm', 'dev/benchmarks/prefill4k_attribution.mm')
    if relative in touched:
        text = '#include "dev/benchmarks/adaptive_expert_tail_sep21/worker_bridge.hpp"\n' + text
    if relative == 'runtime/flash/FlashInt8ExpertStore.mm':
        text = replace(text, '''  return std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(m) +
      (m == 64 ? "_n64_sg8" : "_n64");''',
            '''  return adaptive_expert_tail_sep21::pipeline(phase, m, adaptive_expert_tail_sep21::requested());''')
        text = replace(text, '} // namespace\n\nstruct FlashInt8ExpertStore::Impl final {',
                       '} // namespace\n' + HOOKS + 'struct FlashInt8ExpertStore::Impl final {')
    if relative == 'runtime/flash/FlashForward.cpp':
        text = replace(text, '      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +',
            '''      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +
      std::string(adaptive_expert_tail_sep21::marker(adaptive_expert_tail_sep21::requested())) +''')
    if relative == 'runtime/flash/FlashWorker.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      (void)adaptive_expert_tail_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
    if relative == 'dev/benchmarks/prefill4k_attribution.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.',
            '''      (void)adaptive_expert_tail_sep21::requested(); // Freeze adaptive policy before backend creation.
      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.''')
        text = replace(text, '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert")) return "moe";',
            '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert") || name.starts_with("adaptive_expert_tail_sep21_")) return "moe";')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'build/moe-pointwise-sep21-worker-v1')
    parser.add_argument('--qualified', type=Path, default=ROOT / 'build/adaptive-expert-tail-sep21')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/release/flash/sep21-adaptive-tail-r2k-spread-v1.json')
    parser.add_argument('--hot-report', type=Path, default=ROOT / 'build/release/flash/sep21-adaptive-tail-r2k-hot-v1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/adaptive-expert-tail-sep21-worker-v1')
    args = parser.parse_args()
    base, qualified, output = args.base.resolve(), args.qualified.resolve(), args.output.resolve()
    if output in (base, qualified) or ROOT / 'build' not in output.parents:
        raise ValueError('Adaptive worker requires a distinct private build directory')
    parent_path = base / 'overlay-manifest.json'
    parent = json.loads(parent_path.read_text())
    if parent.get('omitted_original_target_tensor_count') != 432 or not parent.get('pointwise_composed'):
        raise ValueError('Expected sealed Full512 gathered+bulk exact pointwise base')
    report_path = args.report.resolve()
    witness_path = Path(str(report_path) + '.invocation.json')
    report, witness = json.loads(report_path.read_text()), json.loads(witness_path.read_text())
    case = next(c for c in report['cases'] if c['rows'] == 2048 and c['pattern'] == 'spread-all')
    winner = next(v for v in case['variants'] if v['variant'] == 1)
    if not report['pass'] or not winner['screen_pass'] or not winner['strict_full_bf16_exact'] or winner['numerical_alternative']:
        raise ValueError('Root M16 adaptive-tail exact qualification failed')
    if any(not p['bit_exact_and_finite'] for p in winner['probes'].values()):
        raise ValueError('Root raw F32/scaled BF16 qualification differs')
    shader = (qualified / 'adaptive.metal').read_bytes()
    if sha(shader) != witness['sha256']['private_shader']:
        raise ValueError('Private adaptive shader differs from root GPU-qualified source')
    for relative in ('runtime/metal/abi/FlashInt8ExpertStore.h', 'runtime/metal/abi/FlashMoEBuckets.h',
                     'runtime/metal/kernels/common/flash_affine_mpp_common.h'):
        if (base / 'source' / relative).read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError(f'Frozen shader include differs from qualified include: {relative}')
    manifest = copy.deepcopy(parent)
    manifest.update({
        'route': 'private-full512-bulk-gathered-pointwise-native-m32-m16-tail-sep21-v1',
        'adaptive_tail_composed': True,
        'adaptive_tail_environment': 'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21=0|1',
        'adaptive_tail_default': 0, 'adaptive_tail_flag0_original_m32_pipelines': True,
        'adaptive_tail_policy': 'original M32 jobs/params/grid/SG4; valid_rows <=16 M16 math; otherwise original M32 math',
        'adaptive_tail_scope': 'original native M32 gate/up and down/scatter hit producers only; M16/M64/gathered/miss paths unchanged',
        'adaptive_tail_added_allocations_bytes': 0, 'adaptive_tail_changes_numerical_derivative': False,
        'adaptive_tail_base_build': str(base), 'adaptive_tail_base_manifest_sha256': sha(parent_path.read_bytes()),
        'adaptive_tail_transform_sha256': sha(Path(__file__).read_bytes()),
        'adaptive_tail_qualified_shader_sha256': sha(shader),
        'adaptive_tail_root_component_report': str(report_path), 'adaptive_tail_root_component_report_sha256': sha(report_path.read_bytes()),
        'adaptive_tail_root_component_witness_sha256': sha(witness_path.read_bytes()),
        'adaptive_tail_exact_qualification': 'root R2048 spread-all, true row RMS: raw F32/scaled BF16/full-chain bit exact; actual whole-model remains pending',
        'gpu_executed': False, 'payload_bytes_read': 0, 'files': [], 'adaptive_tail_link_inputs': [],
    })
    changed = []
    for record in parent['files']:
        relative = record['path']
        original = (base / 'source' / relative).read_bytes()
        if sha(original) != record['overlay_sha256']:
            raise ValueError(f'Sealed pointwise source drift: {relative}')
        data = transform(relative, original.decode()).encode()
        write(output / 'source' / relative, data)
        if data != original:
            changed.append(relative)
        manifest['files'].append({**record, 'adaptive_tail_changed': data != original,
                                  'adaptive_tail_base_overlay_sha256': sha(original), 'overlay_sha256': sha(data)})
    expected = {'runtime/flash/FlashInt8ExpertStore.mm', 'runtime/flash/FlashForward.cpp',
                'runtime/flash/FlashWorker.mm', 'dev/benchmarks/prefill4k_attribution.mm'}
    if set(changed) != expected:
        raise ValueError(f'Expected only four adaptive route/attribution changes, got {changed}')
    for name in ('worker_bridge.hpp', 'worker_policy_cpu.cpp'):
        relative = PRIVATE / name
        data = (ROOT / relative).read_bytes()
        write(output / 'source' / relative, data)
        manifest['files'].append({'path': relative.as_posix(), 'new_adaptive_tail_file': True,
                                  'adaptive_tail_repository_sha256': sha(data), 'overlay_sha256': sha(data)})
    relative = PRIVATE / 'adaptive.metal'
    write(output / 'source' / relative, shader)
    manifest['files'].append({'path': relative.as_posix(), 'new_adaptive_tail_file': True,
                              'qualified_shader_copy': True, 'overlay_sha256': sha(shader)})
    inputs = {'REUSED': [], 'CORE': [], 'AIRS': []}
    def freeze_input(path, relative, category, expected_sha=None):
        data = path.read_bytes()
        if expected_sha and sha(data) != expected_sha:
            raise ValueError(f'Sealed pointwise compiler input drift: {path}')
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest['adaptive_tail_link_inputs'].append({'source_path': str(path), 'private_path': relative.as_posix(),
                                                     'category': category, 'sha256': sha(data)})
    for record in parent['pointwise_link_inputs']:
        freeze_input(base / record['private_path'], Path('reused/pointwise-inputs') / record['private_path'],
                     record['category'], record['sha256'])
    for path in sorted((base / 'host').glob('*.o')):
        if path.stem not in CHANGED:
            freeze_input(path, Path('reused/pointwise-host') / path.name, 'REUSED')
    freeze_input(base / 'pointwise.air', Path('reused/pointwise-metal/pointwise.air'), 'AIRS')
    frozen_make = '\n'.join(f'{key} := ' + ' '.join('$(BUILD)/' + path for path in paths)
                            for key, paths in inputs.items()) + '\n'
    write(output / 'link-inputs.mk', frozen_make.encode())
    manifest['adaptive_tail_link_make_sha256'] = sha(frozen_make.encode())
    manifest['adaptive_tail_changed_files'] = changed
    for path in (report_path, witness_path, args.hot_report.resolve(), Path(str(args.hot_report.resolve()) + '.invocation.json')):
        data = path.read_bytes()
        write(output / 'qualification' / path.name, data)
    manifest['adaptive_tail_root_hot_report_sha256'] = sha(args.hot_report.resolve().read_bytes())
    write(output / 'overlay-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    write(output / 'base-build.txt', (str(base) + '\n').encode())
    for name in ('splash-flash.config', 'splash.metallib.config'):
        write(output / name, (base / name).read_bytes().rstrip(b'\n') + b'-adaptive-native-m32-m16-tail-sep21-v1\n')
    print(json.dumps({'prepared': str(output), 'frozen_sources': len(manifest['files']),
                      'changed_files': changed, 'frozen_link_inputs': len(manifest['adaptive_tail_link_inputs']),
                      'gpu_executed': False, 'payload_bytes_read': 0, 'added_allocations_bytes': 0}))


if __name__ == '__main__':
    main()
