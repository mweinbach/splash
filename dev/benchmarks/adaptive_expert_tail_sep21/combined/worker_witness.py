#!/usr/bin/env python3
"""Device-free source closure and independent-control proof for composed worker."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from worker_overlay import ROOT, PRIVATE, FMA_PRIVATE, SG2_POLICY, sha, transform


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh CPU witness output')
    build = args.build.resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    base, fma = Path(manifest['combined_base_build']), Path(manifest['combined_fma_qualified_worker'])
    checks = {
        'base_manifest_fresh': sha((base / 'overlay-manifest.json').read_bytes()) == manifest['combined_base_manifest_sha256'],
        'fma_snapshot_manifest_fresh': sha((fma / 'overlay-manifest.json').read_bytes()) == manifest['combined_fma_worker_manifest_sha256'],
        'combined_transform_fresh': sha(Path(__file__).with_name('worker_overlay.py').read_bytes()) == manifest['combined_transform_sha256'],
        'fma_transform_fresh': sha((ROOT / FMA_PRIVATE / 'worker_overlay.py').read_bytes()) == manifest['combined_fma_transform_sha256'],
        'certificates_kept_separate': manifest['combined_fma_certificate_separate_from_exact_tail'],
    }
    mismatches = []
    for record in manifest['files']:
        relative = record['path']
        actual = (build / 'source' / relative).read_bytes()
        if record.get('new_combined_tail_file'):
            expected = actual if record.get('qualified_shader_copy') else (ROOT / relative).read_bytes()
            if not record.get('qualified_shader_copy'):
                checks[f'new_source_fresh:{relative}'] = sha(expected) == record['repository_sha256']
        elif record.get('new_combined_fma_file'):
            expected = (fma / 'source' / relative).read_bytes()
            checks[f'fma_snapshot_source_fresh:{relative}'] = sha(expected) == record['fma_snapshot_sha256']
        else:
            original = (base / 'source' / relative).read_bytes()
            checks[f'base_source_fresh:{relative}'] = sha(original) == record['combined_base_overlay_sha256']
            expected = transform(relative, original.decode()).encode()
        if actual != expected or sha(actual) != record['overlay_sha256']:
            mismatches.append(relative)
    checks['all_frozen_sources_fresh'] = not mismatches
    shader = (build / 'source' / PRIVATE / 'adaptive.metal').read_bytes()
    checks['exact_tail_shader_identical'] = sha(shader) == manifest['combined_tail_source_sha256']
    for evidence in manifest['combined_tail_exact_certificate']:
        path = build / 'qualification' / Path(evidence['path']).name
        report = json.loads(path.read_text())
        invocation = Path(str(path) + '.invocation.json')
        witness = json.loads(invocation.read_text())
        checks[f'exact_tail_report_fresh:{evidence["pattern"]}'] = sha(path.read_bytes()) == evidence['report_sha256'] and sha(invocation.read_bytes()) == evidence['witness_sha256']
        checks[f'exact_tail_certificate_passed:{evidence["pattern"]}'] = report['pass'] and witness['sha256']['private_shader'] == sha(shader) and all(v['screen_pass'] and v['strict_full_bf16_exact'] and not v['numerical_alternative'] and all(p['bit_exact_and_finite'] for p in v['probes'].values()) for c in report['cases'] for v in c['variants'])
    checks['fma_shader_identical_to_rounding_certificate'] = sha((build / 'source' / FMA_PRIVATE / 'scalar_fma.metal').read_bytes()) == manifest['combined_fma_source_sha256']
    semantic_path = build / 'qualification' / Path(manifest['combined_fma_model_semantic_report']).name
    semantic = json.loads(semantic_path.read_text())
    checks['separate_fma_model_evidence_fresh'] = sha(semantic_path.read_bytes()) == manifest['combined_fma_model_semantic_report_sha256'] and semantic['completed'] and semantic['full_plan_coverage']
    worker = (build / 'source/runtime/flash/FlashWorker.mm').read_text()
    forward = (build / 'source/runtime/flash/FlashForward.cpp').read_text()
    store = (build / 'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    base_store = (base / 'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    base_forward = (base / 'source/runtime/flash/FlashForward.cpp').read_text()
    policy = (build / 'source' / SG2_POLICY).read_text()
    base_policy = (base / 'source' / SG2_POLICY).read_text()
    staged = (build / 'source/runtime/flash/FlashGDNStaged.cpp').read_text()
    checks.update({
        'tail_frozen_before_paths_backend': worker.index('(void)adaptive_expert_tail_sg2k128_sep21::requested();') < worker.index('std::filesystem::canonical(argv[2])') < worker.index('metal::MetalBackend backend('),
        'fma_frozen_before_paths_backend': worker.index('(void)gdn_prefill_fma_sep21::requested();') < worker.index('std::filesystem::canonical(argv[2])') < worker.index('metal::MetalBackend backend('),
        'tail_identity_marker_only_when_sg2_enabled': 'adaptive_expert_tail_sg2k128_sep21::markerFor(\n          fixed_sg2_prefill_sep21::selection(),adaptive_expert_tail_sg2k128_sep21::requested())' in forward,
        'actual_sg2_producer_uses_tail_controller': 'adaptive_expert_tail_sg2k128_sep21::producerNameFor(gate,selection(),' in policy,
        'inherited_sg2_eligibility_unchanged': extract(policy, '[[nodiscard]] constexpr bool eligibleFor', '[[nodiscard]] constexpr const char *markerFor') == extract(base_policy, '[[nodiscard]] constexpr bool eligibleFor', '[[nodiscard]] constexpr const char *markerFor'),
        'inherited_m32_jobs_params_launch_unchanged': extract(store, '// Metadata-only host validation.', '} // namespace\n') == extract(base_store, '// Metadata-only host validation.', '} // namespace\n'),
        'store_numerical_derivative_unchanged': extract(store, '    std::string derivative =', '    numericalIdentity =') == extract(base_store, '    std::string derivative =', '    numericalIdentity ='),
        'workspace_planner_unchanged': extract(forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes') == extract(base_forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes'),
        'main_sg2_selector_unchanged': extract(forward, '        if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification))', '        }\n') == extract(base_forward, '        if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification))', '        }\n'),
        'gathered_methods_unchanged': extract(store, 'namespace {\nvoid gatheredMPPViews', '} // namespace splash::flash') == extract(base_store, 'namespace {\nvoid gatheredMPPViews', '} // namespace splash::flash'),
        'fma_only_in_staged_prefill_helper': staged.count('gdn_prefill_fma_sep21::eligible(') == 1 and 'void addGDNStagedPrefill(' in staged,
        'fma_runtime_identity_source_bound': 'gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),' in worker,
        'base_identity_also_reported': 'target_base_numerical_derivative_sha256' in worker,
        'no_added_gpu_buffer_calls': all((build / 'source' / r['path']).read_text().count('allocateBuffer(') == (base / 'source' / r['path']).read_text().count('allocateBuffer(') for r in manifest['files'] if not r.get('new_combined_tail_file') and not r.get('new_combined_fma_file')),
        'no_added_workspace_or_payload': manifest['combined_added_allocation_bytes'] == 0,
        'frozen_link_make_fresh': sha((build / 'link-inputs.mk').read_bytes()) == manifest['combined_link_make_sha256'],
    })
    input_mismatches = [r['private_path'] for r in manifest['combined_link_inputs'] if sha((build / r['private_path']).read_bytes()) != r['sha256']]
    checks['all_frozen_compiler_inputs_fresh'] = not input_mismatches
    live_dependencies = []
    for p in (build / 'host').glob('*.d'):
        first = p.read_text().replace('\\\n', ' ').splitlines()[0]
        for token in first.split(': ', 1)[1].split():
            if token.startswith('runtime/') and token.endswith(('.h', '.hpp')):
                live_dependencies.append(token)
    checks['compiled_hosts_have_no_live_runtime_headers'] = not live_dependencies
    probes = {}
    for mode in ('', '--freeze00', '--freeze01', '--freeze70', '--freeze71', '--missing00', '--retry00', '--retry71', '--sg2-first', '--tail-first'):
        probes[mode or 'default'] = json.loads(subprocess.check_output([str(build / 'policy-cpu'), *([mode] if mode else [])], text=True))
    checks['actual_sg2_tail_policy_and_abi_cpu_pass'] = all(p['pass'] and not p['gpu_work'] for p in probes.values())
    fma_probes = {'source': json.loads(subprocess.check_output([str(build / 'fma-policy-cpu'), '--source', str(build / 'source' / FMA_PRIVATE / 'scalar_fma.metal')], text=True))}
    for mode in ('--freeze0', '--freeze1'):
        fma_probes[mode] = json.loads(subprocess.check_output([str(build / 'fma-policy-cpu'), mode], text=True))
    checks['separate_fma_control_identity_cpu_pass'] = all(p['pass'] for p in fma_probes.values())
    rejected = {}
    for flag, values, expected in (
        ('SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21', ('', '2', 'true', '01', ' 1'), 'must be 0 or 1'),
        ('SPLASH_FLASH_GDN_PREFILL_FMA_SEP21', ('', '2', 'true', '01', ' 1'), 'must be exactly 0 or 1'),
        ('SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT', ('', '1', '6', '07'), 'must be 0 or 7'),
    ):
        for value in values:
            env = dict(os.environ)
            env.update({'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21': '0', 'SPLASH_FLASH_GDN_PREFILL_FMA_SEP21': '0', 'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT': '0', 'SPLASH_FLASH_MOE_POINTWISE_SEP21': '0'})
            env[flag] = value
            result = subprocess.run([str(build / 'splash-flash'), 'serve-flash-native', '/combined-does-not-exist', '16384', 'auto'], env=env, text=True, capture_output=True)
            key = f'{flag}={value!r}'
            rejected[key] = {'returncode': result.returncode, 'stderr': result.stderr}
            checks[f'invalid_control_before_path:{key}'] = result.returncode != 0 and flag in result.stderr and expected in result.stderr
    env = dict(os.environ)
    env.update({'SPLASH_FLASH_GDN_PREFILL_FMA_SEP21': '1', 'SPLASH_FLASH_GDN_STAGED': '0'})
    result = subprocess.run([str(build / 'splash-flash'), 'serve-flash-native', '/combined-does-not-exist', '16384', 'auto'], env=env, text=True, capture_output=True)
    checks['fma_dependency_rejected_before_path'] = result.returncode != 0 and 'requires SPLASH_FLASH_GDN_STAGED=1' in result.stderr
    witness = {'schema': 'combined-sg2-tail-optional-fma-worker-cpu-witness-v1', 'pass': all(checks.values()),
               'gpu_work': False, 'model_loaded': False, 'payload_bytes_read': 0,
               'checks': checks, 'source_mismatch': mismatches, 'link_input_mismatch': input_mismatches,
               'live_runtime_dependencies': live_dependencies, 'compiled_tail_policy': probes,
               'separate_fma_policy': fma_probes, 'invalid_controls': rejected,
               'fma_model_prior_evidence': {'completed': semantic['completed'], 'valid': semantic['valid'], 'passed_cases': semantic['passed_cases'], 'failed_cases': semantic['failed_cases'], 'combined_model_test_pending': True},
               'frozen_source_count': len(manifest['files']), 'frozen_link_input_count': len(manifest['combined_link_inputs']),
               'runtime_sha256': {name: sha((build / name).read_bytes()) for name in ('splash-flash', 'splash.metallib', 'prefill4k-attribution')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2) + '\n')
    print(json.dumps({'pass': witness['pass'], 'gpu_work': False, 'payload_bytes_read': 0,
                      'failed_checks': [k for k, v in checks.items() if not v],
                      'frozen_sources': len(manifest['files']), 'frozen_link_inputs': len(manifest['combined_link_inputs'])}))
    if not witness['pass']:
        raise ValueError('Combined sealed CPU/source witness failed')


if __name__ == '__main__':
    main()
