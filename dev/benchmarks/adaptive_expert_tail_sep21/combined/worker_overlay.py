#!/usr/bin/env python3
"""Seal exact SG2/K128 M16 tails plus independently controlled qualified GDN FMA.

The exact tail certificate and the changed-rounding prefill FMA certificate are
kept separate. All inherited sources/compiler inputs come from sealed workers;
only code/metadata/benchmark evidence is read, never any model payload.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[4]
PRIVATE = Path('dev/benchmarks/adaptive_expert_tail_sep21/combined')
SG2_POLICY = Path('dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp')
FMA_PRIVATE = Path('dev/benchmarks/gdn_chunk_sep21')
OWN_NAMES = {'FlashInt8ExpertStore', 'FlashForward', 'FlashWorker', 'FlashGDNStaged', 'FlashBatchPrefill'}
EXPECTED_CHANGED = {'runtime/flash/FlashInt8ExpertStore.mm', 'runtime/flash/FlashForward.cpp',
                    'runtime/flash/FlashWorker.mm', 'runtime/flash/FlashGDNStaged.cpp',
                    'runtime/flash/FlashBatchPrefill.cpp', 'dev/benchmarks/prefill4k_attribution.mm',
                    SG2_POLICY.as_posix()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f'Combined SG2-tail source anchor drift: {before!r}')
    return text.replace(before, after)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


HOOKS = '''
// Device-free probes call actual inherited policy, fallback and ABI helpers.
std::string combinedSG2TailPipelineForCPU(bool gate) {
  return fixed_sg2_prefill_sep21::producerName(gate);
}
std::string combinedSG2TailFallbackPipelineForCPU(const char *phase, uint32_t tileRows) {
  return pipeline(phase, static_cast<FlashMoEBlockedTile>(tileRows));
}
bool combinedSG2TailEligibleForCPU(uint32_t rows, uint32_t tileRows, bool verification) {
  return fixed_sg2_prefill_sep21::eligible(rows, static_cast<FlashMoEBlockedTile>(tileRows), verification);
}
FlashInt8ExpertStoreParams combinedSG2TailParamsForCPU(uint32_t rows, uint32_t selections, uint32_t tileRows) {
  return allRowsParams(rows, selections, static_cast<FlashMoEBlockedTile>(tileRows));
}
uint32_t combinedSG2TailLaunchForCPU(const FlashInt8ExpertStoreParams &p) {
  return allRowsLaunch(p);
}

'''


def tail_transform(relative, text):
    if relative in (SG2_POLICY.as_posix(), 'runtime/flash/FlashForward.cpp',
                    'runtime/flash/FlashWorker.mm', 'dev/benchmarks/prefill4k_attribution.mm'):
        text = '#include "dev/benchmarks/adaptive_expert_tail_sep21/combined/worker_bridge.hpp"\n' + text
    if relative == SG2_POLICY.as_posix():
        text = replace(text,
            '[[nodiscard]] inline const char *producerName(bool gate) {return producerNameFor(gate,selection());}',
            '''[[nodiscard]] inline const char *producerName(bool gate) {
  return adaptive_expert_tail_sg2k128_sep21::producerNameFor(gate,selection(),
      adaptive_expert_tail_sg2k128_sep21::requested());
}''')
    if relative == 'runtime/flash/FlashInt8ExpertStore.mm':
        text = replace(text, '} // namespace\n\nstruct FlashInt8ExpertStore::Impl final {',
                       '} // namespace\n' + HOOKS + 'struct FlashInt8ExpertStore::Impl final {')
    if relative == 'runtime/flash/FlashForward.cpp':
        text = replace(text, '      std::string(fixed_sg2_prefill_sep21::marker()) +',
            '''      std::string(fixed_sg2_prefill_sep21::marker()) +
      std::string(adaptive_expert_tail_sg2k128_sep21::markerFor(
          fixed_sg2_prefill_sep21::selection(),adaptive_expert_tail_sg2k128_sep21::requested())) +''')
    if relative == 'runtime/flash/FlashWorker.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail control before paths/metadata/backend.
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
    if relative == 'dev/benchmarks/prefill4k_attribution.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.',
            '''      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail scheduling before backend creation.
      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.''')
        text = replace(text, '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert")) return "moe";',
            '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert") || name.starts_with("prefill_moe_sep21_memory_fixed_") || name.starts_with("adaptive_expert_tail_sg2k128_sep21_")) return "moe";')
    return text


def fma_module():
    path = ROOT / FMA_PRIVATE / 'worker_overlay.py'
    spec = importlib.util.spec_from_file_location('frozen_fma_transform', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(relative, text):
    return fma_module().transform(relative, tail_transform(relative, text))


def qualify_tail(report_path, expected_pattern, shader_sha):
    report = json.loads(report_path.read_text())
    witness_path = Path(str(report_path) + '.invocation.json')
    witness = json.loads(witness_path.read_text())
    case = next(c for c in report['cases'] if c['rows'] == 2048 and c['pattern'] == expected_pattern)
    variant = next(v for v in case['variants'] if v['variant'] == 1)
    if not report['pass'] or not variant['screen_pass'] or not variant['strict_full_bf16_exact'] or variant['numerical_alternative']:
        raise ValueError(f'Combined exact-tail component qualification failed: {report_path}')
    if any(not p['bit_exact_and_finite'] for p in variant['probes'].values()):
        raise ValueError('Combined raw F32/scaled BF16 tail qualification differs')
    if witness['sha256']['private_shader'] != shader_sha:
        raise ValueError('Combined shader differs from root exact-tail qualification')
    return {'path': str(report_path), 'report_sha256': sha(report_path.read_bytes()),
            'witness_sha256': sha(witness_path.read_bytes()), 'pattern': expected_pattern}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1')
    parser.add_argument('--fma-worker', type=Path, default=ROOT / 'build/gdn-prefill-fma-sep21-worker-v1')
    parser.add_argument('--qualified', type=Path, default=ROOT / 'build/adaptive-expert-tail-sg2k128-sep21')
    parser.add_argument('--spread-report', type=Path, default=ROOT / 'build/release/flash/sep21-adaptive-sg2k128-r2k-spread-v1.json')
    parser.add_argument('--hot-report', type=Path, default=ROOT / 'build/release/flash/sep21-adaptive-sg2k128-r2k-hot-v1.json')
    parser.add_argument('--fma-semantic-report', type=Path, default=ROOT / 'build/release/flash/sep21-fma-pointwise-model-and-quality-v1-3.semantic.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1')
    args = parser.parse_args()
    base, fma, qualified, output = (p.resolve() for p in (args.base, args.fma_worker, args.qualified, args.output))
    if output in (base, fma, qualified) or ROOT / 'build' not in output.parents:
        raise ValueError('Combined worker requires a distinct private directory')
    parent_path, fma_path = base / 'overlay-manifest.json', fma / 'overlay-manifest.json'
    parent, fma_manifest = json.loads(parent_path.read_text()), json.loads(fma_path.read_text())
    if not parent.get('fixed_sg2_scope') or not parent.get('pointwise_composed') or parent.get('omitted_original_target_tensor_count') != 432:
        raise ValueError('Expected sealed main-only SG2/K128 pointwise Full512 base')
    fma_transform_path = ROOT / FMA_PRIVATE / 'worker_overlay.py'
    if sha(fma_transform_path.read_bytes()) != fma_manifest['gdn_fma_transform_sha256']:
        raise ValueError('FMA transform differs from the root-qualified snapshot')
    shader = (qualified / 'adaptive.metal').read_bytes()
    tail_evidence = [qualify_tail(args.spread_report.resolve(), 'spread-all', sha(shader)),
                     qualify_tail(args.hot_report.resolve(), 'hit-concentrated', sha(shader))]
    manifest = copy.deepcopy(parent)
    manifest.update({
        'route': 'private-full512-pointwise-bulk-gathered-sg2-k128-m16-tail-optional-gdn-prefill-fma-sep21-v1',
        'combined_tail_composed': True, 'combined_fma_composed': True,
        'combined_tail_environment': 'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21=0|1',
        'combined_sg2_environment': 'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=0|7',
        'combined_fma_environment': fma_manifest['gdn_fma_required_environment'],
        'combined_controls_independent': True, 'combined_defaults': {'tail': 0, 'sg2': 0, 'fma': 0},
        'combined_flag0_current_sg2_base': 'tail=0,FMA=0,SG2=7 retains original SG2 pointwise execution',
        'combined_tail_scope': 'inherited main nonverification R2048/M32 SG2 methods only; M16/M64/other rows/batch executor/decode/verify unchanged',
        'combined_tail_exact_certificate': tail_evidence,
        'combined_tail_source_sha256': sha(shader), 'combined_tail_numerical_derivative_changed': False,
        'combined_fma_certificate_separate_from_exact_tail': True,
        'combined_fma_scope': fma_manifest['gdn_fma_scope'],
        'combined_fma_numerical_derivative_changed': True,
        'combined_fma_policy': 'existing source-bound FMA identity transform; disabled retains base derivative',
        'combined_fma_source_sha256': fma_manifest['gdn_fma_kernel_source_sha256'],
        'combined_fma_qualified_worker': str(fma), 'combined_fma_worker_manifest_sha256': sha(fma_path.read_bytes()),
        'combined_fma_transform_sha256': sha(fma_transform_path.read_bytes()),
        'combined_fma_model_semantic_report': str(args.fma_semantic_report.resolve()),
        'combined_fma_model_semantic_report_sha256': sha(args.fma_semantic_report.resolve().read_bytes()),
        'combined_whole_model_qualification': 'pending root single-driver 3 warm trials and 22-case suite',
        'combined_added_allocation_bytes': 0, 'combined_base_build': str(base),
        'combined_base_manifest_sha256': sha(parent_path.read_bytes()),
        'combined_transform_sha256': sha(Path(__file__).read_bytes()),
        'gpu_executed': False, 'payload_bytes_read': 0, 'files': [], 'combined_link_inputs': [],
    })
    changed = []
    for record in parent['files']:
        relative = record['path']
        original = (base / 'source' / relative).read_bytes()
        if sha(original) != record['overlay_sha256']:
            raise ValueError(f'Frozen SG2 base source drift: {relative}')
        data = transform(relative, original.decode()).encode()
        write(output / 'source' / relative, data)
        if data != original:
            changed.append(relative)
        manifest['files'].append({**record, 'combined_changed': data != original,
                                  'combined_base_overlay_sha256': sha(original), 'overlay_sha256': sha(data)})
    if set(changed) != EXPECTED_CHANGED:
        raise ValueError(f'Unexpected combined host/policy edits: {changed}')
    for name in ('worker_bridge.hpp', 'worker_policy_cpu.cpp'):
        relative = PRIVATE / name
        data = (ROOT / relative).read_bytes()
        write(output / 'source' / relative, data)
        manifest['files'].append({'path': relative.as_posix(), 'new_combined_tail_file': True,
                                  'repository_sha256': sha(data), 'overlay_sha256': sha(data)})
    relative = PRIVATE / 'adaptive.metal'
    write(output / 'source' / relative, shader)
    manifest['files'].append({'path': relative.as_posix(), 'new_combined_tail_file': True,
                              'qualified_shader_copy': True, 'overlay_sha256': sha(shader)})
    fma_records = {r['path']: r for r in fma_manifest['files']}
    for name in ('worker_bridge.hpp', 'worker_policy_cpu.cpp', 'scalar_fma.metal'):
        relative = FMA_PRIVATE / name
        data = (fma / 'source' / relative).read_bytes()
        if sha(data) != fma_records[relative.as_posix()]['overlay_sha256']:
            raise ValueError(f'Frozen FMA private source drift: {relative}')
        write(output / 'source' / relative, data)
        manifest['files'].append({'path': relative.as_posix(), 'new_combined_fma_file': True,
                                  'fma_snapshot_sha256': sha(data), 'overlay_sha256': sha(data)})
    if sha((output / 'source' / FMA_PRIVATE / 'scalar_fma.metal').read_bytes()) != manifest['combined_fma_source_sha256']:
        raise ValueError('FMA kernel differs from the separate rounding certificate')
    inputs = {'REUSED': [], 'CORE': [], 'AIRS': []}
    def freeze(path, relative, category, expected=None):
        data = path.read_bytes()
        if expected and sha(data) != expected:
            raise ValueError(f'Frozen combined compiler input drift: {path}')
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest['combined_link_inputs'].append({'source_path': str(path), 'private_path': relative.as_posix(),
                                                 'category': category, 'sha256': sha(data)})
    for record in parent['fixed_sg2_link_inputs']:
        if record['category'] == 'REUSED' and Path(record['private_path']).stem in OWN_NAMES:
            continue
        freeze(base / record['private_path'], Path('reused/sg2-base') / record['private_path'],
               record['category'], record['sha256'])
    freeze(base / 'fixed-sg2.air', Path('reused/sg2-base/fixed-sg2.air'), 'AIRS')
    freeze(fma / 'gdn-prefill-fma.air', Path('reused/qualified-fma/gdn-prefill-fma.air'), 'AIRS')
    make = '\n'.join(f'{key} := ' + ' '.join('$(BUILD)/' + p for p in paths) for key, paths in inputs.items()) + '\n'
    write(output / 'link-inputs.mk', make.encode())
    manifest['combined_link_make_sha256'] = sha(make.encode())
    manifest['combined_changed_files'] = changed
    for p in (args.spread_report.resolve(), Path(str(args.spread_report.resolve()) + '.invocation.json'),
              args.hot_report.resolve(), Path(str(args.hot_report.resolve()) + '.invocation.json'), args.fma_semantic_report.resolve()):
        write(output / 'qualification' / p.name, p.read_bytes())
    write(output / 'overlay-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    write(output / 'base-build.txt', (str(base) + '\n').encode())
    for name in ('splash-flash.config', 'splash.metallib.config'):
        write(output / name, (base / name).read_bytes().rstrip(b'\n') + b'-exact-sg2-k128-m16-tail-optional-qualified-fma-sep21-v1\n')
    print(json.dumps({'prepared': str(output), 'changed_sources': changed,
                      'frozen_sources': len(manifest['files']), 'frozen_link_inputs': len(manifest['combined_link_inputs']),
                      'gpu_work': False, 'payload_bytes_read': 0, 'added_allocation_bytes': 0,
                      'fma_certificate_separate_from_exact_tail': True}))


if __name__ == '__main__':
    main()
