#!/usr/bin/env python3
"""Device-free sealed hybrid phase/alias/residency/closure and control proof."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
from hybrid_overlay import ROOT, PRIVATE, FMA_PRIVATE, SG2_POLICY, BASE_IDENTITY, OWN_NAMES, sha, transform


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/hybrid-sg2-tail-fma-sep21-worker-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh hybrid witness path')
    build = args.build.resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    base, certified = Path(manifest['hybrid_combined_base_build']), Path(manifest['hybrid_combined_certificate_build'])
    certificate = json.loads((certified / 'overlay-manifest.json').read_text())
    checks = {
        'hybrid_v4_manifest_fresh': sha((base / 'overlay-manifest.json').read_bytes()) == manifest['hybrid_combined_base_manifest_sha256'],
        'separate_certificate_manifest_fresh': sha((certified / 'overlay-manifest.json').read_bytes()) == manifest['hybrid_combined_certificate_manifest_sha256'],
        'hybrid_transform_fresh': sha(Path(__file__).with_name('hybrid_overlay.py').read_bytes()) == manifest['hybrid_combined_transform_sha256'],
        'fma_and_exact_tail_certificates_separate': manifest['hybrid_combined_fma_certificate_separate'],
        'no_allrows_flag_requirement': not manifest['hybrid_combined_allrows_flags_required'],
    }
    mismatches = []
    for record in manifest['files']:
        relative = record['path']
        actual = (build / 'source' / relative).read_bytes()
        if record.get('new_hybrid_combined_file'):
            expected = (certified / 'source' / relative).read_bytes() if record['certified_source'] else (ROOT / relative).read_bytes()
        else:
            source = (base / 'source' / relative).read_bytes()
            checks[f'base_source_fresh:{relative}'] = sha(source) == record['hybrid_combined_base_sha256']
            expected = transform(relative, source.decode()).encode()
        if actual != expected or sha(actual) != record['overlay_sha256']:
            mismatches.append(relative)
    checks['all_sources_fresh'] = not mismatches
    def source(relative): return (build / 'source' / relative).read_text()
    def ancestor(relative): return (base / 'source' / relative).read_text()
    store, base_store = source('runtime/flash/FlashInt8ExpertStore.mm'), ancestor('runtime/flash/FlashInt8ExpertStore.mm')
    forward, base_forward = source('runtime/flash/FlashForward.cpp'), ancestor('runtime/flash/FlashForward.cpp')
    worker, base_worker = source('runtime/flash/FlashWorker.mm'), ancestor('runtime/flash/FlashWorker.mm')
    original_raw = extract(base_store, 'void FlashInt8ExpertStore::addGateUp(', '} // namespace splash::flash')
    alias_begin = '        layer.source[plane] = weights.projection('
    alias_end = '    const uint64_t after = backend.memoryStats().allocatedBytes;'
    q4_begin = '    } else {\n    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj")'
    q4_end = '    if (!batchSharedExpertFused(graph'
    checks.update({
        'original_q4_validated_i8_methods_unchanged': original_raw in store,
        'q4_alias_array_and_retarget_unchanged': 'std::array<FlashTensor, 9> sourceTensors;' in store and extract(store, alias_begin, alias_end) == extract(base_store, alias_begin, alias_end),
        'immutable_source_alias_disjoint_checks_unchanged': extract(store, '  void immutableDisjoint(', '};\n\nFlashInt8ExpertStore::FlashInt8ExpertStore') == extract(base_store, '  void immutableDisjoint(', '};\n\nFlashInt8ExpertStore::FlashInt8ExpertStore'),
        'immutable_saved_weight_enumeration_unchanged': extract(store, 'std::vector<metal::MetalBuffer> FlashInt8ExpertStore::immutableWeightBuffers()', 'FlashInt8ExpertStoreGraphCounters FlashInt8ExpertStore::graphCounters') == extract(base_store, 'std::vector<metal::MetalBuffer> FlashInt8ExpertStore::immutableWeightBuffers()', 'FlashInt8ExpertStoreGraphCounters FlashInt8ExpertStore::graphCounters'),
        'existing_i8_phase_gate_unchanged': 'const bool blocked = impl_->blockMoE && !verification && rows >= 256;' in forward,
        'new_sg2_calls_only_existing_i8_branch': forward.index('if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification))') > forward.index('const bool blocked = impl_->blockMoE && !verification && rows >= 256;'),
        'q4_small_and_verification_branch_unchanged': extract(forward, q4_begin, q4_end) == extract(base_forward, q4_begin, q4_end),
        'prefill_workspace_planner_unchanged': extract(forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes') == extract(base_forward, 'uint64_t FlashForward::workspacePlannedBytes', 'std::string FlashForward::kernelRoutes'),
        'fixed_r4_dense_cpp_and_header_unchanged': source('runtime/flash/FlashFloatDenseCache.cpp') == ancestor('runtime/flash/FlashFloatDenseCache.cpp') and source('runtime/flash/FlashFloatDenseCache.hpp') == ancestor('runtime/flash/FlashFloatDenseCache.hpp'),
        'weights_residency_getter_and_headers_unchanged': source('runtime/flash/FlashWeights.mm') == ancestor('runtime/flash/FlashWeights.mm') and source('runtime/flash/FlashWeights.hpp') == ancestor('runtime/flash/FlashWeights.hpp'),
        'legacy_startup_guards_unchanged': extract(worker, '      // Private fixed-R4 hybrid:', '      const auto hybridStoreMetadata') == extract(base_worker, '      // Private fixed-R4 hybrid:', '      const auto hybridStoreMetadata'),
        'resident_union_block_unchanged': extract(worker, '      savedResidency.requested =', '      std::unique_ptr<idle_maintenance::Maintenance>') == extract(base_worker, '      savedResidency.requested =', '      std::unique_ptr<idle_maintenance::Maintenance>'),
        'static_base_hybrid_identity_preserved': f'json::quote("{BASE_IDENTITY}")' in worker and manifest['target_numerical_derivative_sha256'] == BASE_IDENTITY,
        'fma_identity_binds_existing_hybrid_fingerprint': f'gdn_prefill_fma_sep21::numericalIdentity("{BASE_IDENTITY}",gdn_prefill_fma_sep21::requested())' in worker,
        'phase_identity_policy_and_no_q4_parity_claim_unchanged': extract(worker, '      << R"(,"target_phase_policy":', '      << R"(,"engine_instance_id":') == extract(base_worker, '      << R"(,"target_phase_policy":', '      << R"(,"engine_instance_id":'),
        'tail_marker_actual_forward': 'adaptive_expert_tail_sg2k128_sep21::markerFor(' in forward,
        'flag_parsing_before_paths_backend': all(worker.index(anchor) < worker.index('std::filesystem::canonical(argv[2])') < worker.index('metal::MetalBackend backend(') for anchor in ('(void)adaptive_expert_tail_sg2k128_sep21::requested();', '(void)fixed_sg2_prefill_sep21::selection();', '(void)gdn_prefill_fma_sep21::requested();')),
        'no_new_gpu_allocations': all(source(r['path']).count('allocateBuffer(') == ancestor(r['path']).count('allocateBuffer(') for r in manifest['files'] if not r.get('new_hybrid_combined_file')),
        'no_added_backing_or_workspace': manifest['hybrid_combined_added_backing_and_workspace_bytes'] == 0,
    })
    census = manifest['expert_residency_metadata_census']
    checks['original_residency_census_preserved'] = census['selected_owner_count'] == 25 and census['selected_owner_bytes'] == 69363302400 and census['composite_owner_count'] == 748 and census['composite_owner_bytes'] == 202252746752
    # Strict source comparison of the new raw helper back to its inherited
    # validator graph catches parameter, alias, sanitizer, grid and miss drift.
    new_methods = extract(store, 'void FlashInt8ExpertStore::addFixedSG2PrefillGateUp(', 'fixed_sg2_prefill_sep21::Counters FlashInt8ExpertStore::fixedSG2PrefillCounters')
    checks['new_raw_hit_param_and_grid_copied_twice'] = new_methods.count('miss.blocked.job_capacity, uint32_t(tile), miss.stored_experts, 0, 0};') == 2 and new_methods.count('miss.blocked.job_capacity, 1}, {threads, 1, 1}') == 4
    checks['new_raw_sg2_threads_and_full512_guards'] = new_methods.count('fixed_sg2_prefill_sep21::producerThreads();') == 2 and new_methods.count('selectedIDs.size() !=512') == 2 and new_methods.count('eligible(rows,tile,false)') == 2
    checks['new_methods_reconstruct_exactly'] = transform('runtime/flash/FlashInt8ExpertStore.mm', base_store) == store
    inputs = manifest['hybrid_combined_link_inputs']
    input_mismatches = [r['private_path'] for r in inputs if sha((build / r['private_path']).read_bytes()) != r['sha256']]
    checks['all_compiler_inputs_frozen_fresh'] = not input_mismatches
    checks['link_make_fresh'] = sha((build / 'link-inputs.mk').read_bytes()) == manifest['hybrid_combined_link_make_sha256']
    weight_inputs = [r for r in inputs if Path(r['private_path']).name == 'FlashWeights.o']
    checks['one_correct_v4_weights_object'] = len(weight_inputs) == 1 and weight_inputs[0]['source_path'] == str(base / 'host/FlashWeights.o')
    checks['no_reused_changed_header_dependents'] = not any(r['category'] == 'REUSED' and Path(r['private_path']).stem in OWN_NAMES for r in inputs)
    air_names = [Path(r['private_path']).name for r in inputs if r['category'] == 'AIRS']
    checks['all_restored_and_control_air_inputs_present_once'] = all(air_names.count(n) == 1 for n in ('pointwise.air', 'bulk-attention.air', 'dense-prefill.air', 'fixed-sg2.air', 'gdn-prefill-fma.air'))
    live_dependencies = []
    for p in (build / 'host').glob('*.d'):
        first = p.read_text().replace('\\\n', ' ').splitlines()[0]
        for token in first.split(': ', 1)[1].split():
            if token.startswith('runtime/') and token.endswith(('.h', '.hpp')):
                live_dependencies.append(token)
    checks['compiled_hosts_only_frozen_runtime_headers'] = not live_dependencies
    probes = {}
    for mode in ('', '--freeze00', '--freeze01', '--freeze70', '--freeze71', '--missing00', '--retry00', '--retry71', '--sg2-first', '--tail-first'):
        probes[mode or 'default'] = json.loads(subprocess.check_output([str(build / 'policy-cpu'), *([mode] if mode else [])], text=True))
    checks['linked_hybrid_policy_capacity_and_freezes_pass'] = all(p['pass'] and not p['gpu_work'] for p in probes.values())
    fma_probes = {'source': json.loads(subprocess.check_output([str(build / 'fma-policy-cpu'), '--source', str(build / 'source' / FMA_PRIVATE / 'scalar_fma.metal')], text=True))}
    for mode in ('--freeze0', '--freeze1'):
        fma_probes[mode] = json.loads(subprocess.check_output([str(build / 'fma-policy-cpu'), mode], text=True))
    checks['independent_fma_identity_policy_pass'] = all(p['pass'] for p in fma_probes.values())
    rejected = {}
    for flag, values, expected in (
        ('SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21', ('', '2', 'true', '01'), 'must be 0 or 1'),
        ('SPLASH_FLASH_GDN_PREFILL_FMA_SEP21', ('', '2', 'true', '01'), 'must be exactly 0 or 1'),
        ('SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT', ('', '1', '6', '07'), 'must be 0 or 7')):
        for value in values:
            env = dict(os.environ)
            env.update({'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21': '0', 'SPLASH_FLASH_GDN_PREFILL_FMA_SEP21': '0', 'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT': '0', 'SPLASH_FLASH_MOE_POINTWISE_SEP21': '0', 'SPLASH_FLASH_ALLROWS_FULL512_TARGET': '0', 'SPLASH_FLASH_ALLROWS_GATHERED_MPP': '0'})
            env[flag] = value
            result = subprocess.run([str(build / 'splash-flash'), 'serve-flash-native', '/hybrid-composite-does-not-exist', '16384', 'auto'], env=env, text=True, capture_output=True)
            key = f'{flag}={value!r}'
            rejected[key] = {'returncode': result.returncode, 'stderr': result.stderr}
            checks[f'invalid_flag_before_path:{key}'] = result.returncode != 0 and flag in result.stderr and expected in result.stderr
    fma_bridge = source((FMA_PRIVATE / 'worker_bridge.hpp').as_posix())
    policy = re.search(r'kPolicy =\s*"([^"]+)";', fma_bridge).group(1)
    kernel_sha = re.search(r'kKernelSourceSHA256 =\s*"([^"]+)";', fma_bridge).group(1)
    enabled_identity = hashlib.sha256((BASE_IDENTITY + '\n' + policy + '\n' + kernel_sha).encode()).hexdigest()
    witness = {'schema': 'resident-hybrid-sg2-tail-optional-fma-cpu-source-witness-v1', 'pass': all(checks.values()),
               'gpu_work': False, 'model_loaded': False, 'payload_bytes_read': 0, 'checks': checks,
               'source_mismatch': mismatches, 'link_input_mismatch': input_mismatches,
               'live_runtime_header_dependencies': live_dependencies, 'compiled_policy': probes,
               'separate_fma_policy': fma_probes, 'invalid_controls': rejected,
               'base_hybrid_identity': BASE_IDENTITY, 'optional_fma_hybrid_identity': enabled_identity,
               'residency_census': {k: census[k] for k in ('selected_owner_count', 'selected_owner_bytes', 'composite_owner_count', 'composite_owner_bytes')},
               'frozen_sources': len(manifest['files']), 'frozen_link_inputs': len(inputs),
               'runtime_sha256': {name: sha((build / name).read_bytes()) for name in ('splash-flash', 'splash.metallib', 'prefill4k-attribution')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2) + '\n')
    print(json.dumps({'pass': witness['pass'], 'gpu_work': False, 'payload_bytes_read': 0,
                      'failed_checks': [k for k, v in checks.items() if not v],
                      'frozen_sources': len(manifest['files']), 'frozen_link_inputs': len(inputs),
                      'optional_fma_hybrid_identity': enabled_identity}))
    if not witness['pass']:
        raise ValueError('Hybrid sealed CPU/source witness failed')


if __name__ == '__main__':
    main()
