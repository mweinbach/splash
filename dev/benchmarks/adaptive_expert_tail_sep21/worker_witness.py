#!/usr/bin/env python3
"""Verify frozen worker inputs, actual compiled policy and pre-backend rejection."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from worker_overlay import ROOT, PRIVATE, sha, transform


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/adaptive-expert-tail-sep21-worker-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh CPU witness output')
    build = args.build.resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    base = Path(manifest['adaptive_tail_base_build'])
    checks = {
        'pointwise_base_manifest_fresh': sha((base / 'overlay-manifest.json').read_bytes()) == manifest['adaptive_tail_base_manifest_sha256'],
        'overlay_transform_fresh': sha(Path(__file__).with_name('worker_overlay.py').read_bytes()) == manifest['adaptive_tail_transform_sha256'],
    }
    mismatches = []
    for record in manifest['files']:
        relative = record['path']
        actual = (build / 'source' / relative).read_bytes()
        if record.get('new_adaptive_tail_file'):
            expected = actual if record.get('qualified_shader_copy') else (ROOT / relative).read_bytes()
            if not record.get('qualified_shader_copy'):
                checks[f'new_source_fresh:{relative}'] = sha(expected) == record['adaptive_tail_repository_sha256']
        else:
            original = (base / 'source' / relative).read_bytes()
            checks[f'base_source_fresh:{relative}'] = sha(original) == record['adaptive_tail_base_overlay_sha256']
            expected = transform(relative, original.decode()).encode()
        if actual != expected or sha(actual) != record['overlay_sha256']:
            mismatches.append(relative)
    checks['frozen_sources_fresh'] = not mismatches
    kernel = (build / 'source' / PRIVATE / 'adaptive.metal').read_bytes()
    report_path = build / 'qualification' / Path(manifest['adaptive_tail_root_component_report']).name
    report = json.loads(report_path.read_text())
    report_witness = json.loads(Path(str(report_path) + '.invocation.json').read_text())
    checks['qualified_shader_identical'] = sha(kernel) == manifest['adaptive_tail_qualified_shader_sha256'] == report_witness['sha256']['private_shader']
    checks['qualified_report_fresh'] = sha(report_path.read_bytes()) == manifest['adaptive_tail_root_component_report_sha256']
    checks['root_exact_component_passed'] = report['pass'] and all(v['screen_pass'] for c in report['cases'] for v in c['variants'] if v['variant'] == 1)
    worker = (build / 'source/runtime/flash/FlashWorker.mm').read_text()
    forward = (build / 'source/runtime/flash/FlashForward.cpp').read_text()
    store = (build / 'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    base_forward = (base / 'source/runtime/flash/FlashForward.cpp').read_text()
    base_store = (base / 'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    checks.update({
        'frozen_before_paths_metadata_backend': worker.index('(void)adaptive_expert_tail_sep21::requested();') < worker.index('std::filesystem::canonical(argv[2])') < worker.index('metal::MetalBackend backend('),
        'route_marker_in_actual_kernel_routes': 'std::string(adaptive_expert_tail_sep21::marker(adaptive_expert_tail_sep21::requested()))' in forward,
        'actual_runtime_uses_cpu_tested_policy': 'return adaptive_expert_tail_sep21::pipeline(phase, m, adaptive_expert_tail_sep21::requested());' in store,
        'workspace_planner_unchanged': extract(forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes') == extract(base_forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes'),
        'numerical_derivative_unchanged': extract(store, '    std::string derivative =', '    numericalIdentity =') == extract(base_store, '    std::string derivative =', '    numericalIdentity ='),
        'allrows_scratch_params_launch_unchanged': extract(store, '// Metadata-only host validation.', '} // namespace\n') == extract(base_store, '// Metadata-only host validation.', '} // namespace\n'),
        'pointwise_and_gathered_unchanged': 'pointwise_sep21::addPoison(graph,' in store and extract(store, 'namespace {\nvoid gatheredMPPViews', '} // namespace splash::flash') == extract(base_store, 'namespace {\nvoid gatheredMPPViews', '} // namespace splash::flash'),
        'no_additional_gpu_allocation_calls': all((build / 'source' / r['path']).read_text().count('allocateBuffer(') == (base / 'source' / r['path']).read_text().count('allocateBuffer(') for r in manifest['files'] if not r.get('new_adaptive_tail_file')),
        'additional_planned_and_actual_buffers_zero': manifest['adaptive_tail_added_allocations_bytes'] == 0,
        'attribution_classifies_adaptive_as_moe': 'name.starts_with("adaptive_expert_tail_sep21_")' in (build / 'source/dev/benchmarks/prefill4k_attribution.mm').read_text(),
        'link_make_fresh': sha((build / 'link-inputs.mk').read_bytes()) == manifest['adaptive_tail_link_make_sha256'],
    })
    input_mismatch = []
    for record in manifest['adaptive_tail_link_inputs']:
        if sha((build / record['private_path']).read_bytes()) != record['sha256']:
            input_mismatch.append(record['private_path'])
    checks['all_reused_host_core_air_inputs_frozen_fresh'] = not input_mismatch
    live_dependencies = []
    for dependency in (build / 'host').glob('*.d'):
        first = dependency.read_text().replace('\\\n', ' ').splitlines()[0]
        for token in first.split(': ', 1)[1].split():
            if token.startswith('runtime/') and token.endswith(('.h', '.hpp')):
                live_dependencies.append(token)
    checks['compiled_hosts_use_only_frozen_runtime_headers'] = not live_dependencies
    probes = {}
    for mode in ([], ['--freeze0'], ['--freeze1'], ['--missing0'], ['--retry0'], ['--retry1']):
        probes[' '.join(mode) or 'default'] = json.loads(subprocess.check_output([str(build / 'policy-cpu'), *mode], text=True))
    checks['compiled_actual_policy_abi_and_freeze_checks_pass'] = all(p['pass'] and not p['gpu_work'] for p in probes.values())
    rejected = {}
    for value in ('', '2', 'true', '01', ' 1', '1 '):
        env = dict(os.environ)
        env['SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21'] = value
        result = subprocess.run([str(build / 'splash-flash'), 'serve-flash-native', '/adaptive-does-not-exist', '16384', 'auto'], env=env, text=True, capture_output=True)
        rejected[value] = {'returncode': result.returncode, 'stderr': result.stderr}
        checks[f'invalid_switch_rejected_before_path:{value!r}'] = result.returncode != 0 and 'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21 must be 0 or 1' in result.stderr
    witness = {
        'schema': 'adaptive-expert-tail-sep21-whole-worker-cpu-witness-v1',
        'pass': all(checks.values()), 'gpu_work': False, 'model_loaded': False, 'payload_bytes_read': 0,
        'checks': checks, 'source_mismatch': mismatches, 'link_input_mismatch': input_mismatch,
        'live_runtime_header_dependencies': live_dependencies, 'compiled_policy_cpu': probes,
        'invalid_switch_probes': rejected, 'frozen_source_count': len(manifest['files']),
        'frozen_link_input_count': len(manifest['adaptive_tail_link_inputs']),
        'runtime_sha256': {name: sha((build / name).read_bytes()) for name in ('splash-flash', 'splash.metallib', 'prefill4k-attribution')},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2) + '\n')
    print(json.dumps({'pass': witness['pass'], 'gpu_work': False, 'payload_bytes_read': 0,
                      'failed_checks': [name for name, passed in checks.items() if not passed],
                      'frozen_sources': len(manifest['files']), 'frozen_link_inputs': len(manifest['adaptive_tail_link_inputs'])}))
    if not witness['pass']:
        raise ValueError('Frozen adaptive worker CPU/source witness failed')


if __name__ == '__main__':
    main()
