#!/usr/bin/env python3
"""Freeze CPU-only semantic source closure and review receipt; no device work."""
from pathlib import Path
import argparse, hashlib, json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path('dev/benchmarks/expert_r4_compact_verify_worker_sep22')
HELPERS = ('dev/benchmarks/prefill4k_attribution_quality.py',
           'dev/benchmarks/prefill4k_attribution_quality_python.py',
           'dev/benchmarks/qualify_flash_http.py', 'dev/benchmarks/flash_precision_quality.py',
           'dev/benchmarks/prefill_decode_phase_quality.py',
           'dev/benchmarks/singleton_teacher_bulk_quality.py',
           'dev/benchmarks/phase_saved_only_resource_quality.py',
           'dev/tests/flash/test_prefill4k_semantic_quality.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/compact-native-r4-verify-teacher-sep22-worker-v1b')
    parser.add_argument('--freeze-semantic', action='store_true')
    parser.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    args = parser.parse_args()
    build = args.build.resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    witness = json.loads((build / 'cpu-source-witness.json').read_text())
    if witness.get('pass') is not True or witness.get('source_identity_sha256') != manifest['source_identity_sha256']:
        raise ValueError('The compiled CPU source witness is unavailable or stale')
    semantic = build / 'semantic-source-seal.json'
    if args.freeze_semantic:
        if semantic.exists():
            raise ValueError('Fresh immutable semantic source seal required')
        records = []
        paths = [Path(path) for path in HELPERS]
        paths += [PRIVATE / name for name in ('semantic_quality.py', 'test_semantic_quality.py', 'final_seal.py',
                                               'INTEGRATION.md', 'R8_R16_PROPOSAL.md')]
        for rel in paths:
            data = (ROOT / rel).read_bytes()
            out = build / 'source' / rel
            if out.exists() and out.read_bytes() != data:
                raise ValueError('Refusing to change an already sealed source: ' + str(rel))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            records.append({'path': str(rel), 'sha256': sha(data), 'bytes': len(data),
                            'original_quality_helper_unchanged': str(rel) in HELPERS})
        semantic.write_text(json.dumps({'schema': 'compact-R4-original22-private-semantic-source-seal-v1',
                'compiled_source_identity_sha256': manifest['source_identity_sha256'], 'sources': records,
                'gpu_work': False, 'model_payload_reads': False, 'capture_payload_reads': False}, indent=2) + '\n')
        print(json.dumps({'semantic_sources_frozen': len(records), 'output': str(semantic), 'gpu_work': False}))
        return
    final = build / 'READY.json'
    if final.exists():
        raise ValueError('Fresh final READY receipt required')
    s = json.loads(semantic.read_text())
    if s.get('compiled_source_identity_sha256') != manifest['source_identity_sha256']:
        raise ValueError('Semantic closure targets a different compiled worker')
    for record in s['sources']:
        if sha((build / 'source' / record['path']).read_bytes()) != record['sha256']:
            raise ValueError('Semantic source drift: ' + record['path'])
    for record in witness['artifacts'] + witness['compiled_objects']:
        if sha((build / record['path']).read_bytes()) != record['sha256']:
            raise ValueError('Compiled CPU artifact drift: ' + record['path'])
    cpu_log = (build / 'cpu-self-test.log').read_text()
    for mode in ('--freeze0', '--freeze1', '--missing', '--retry0', '--retry1'):
        if '"mode":"' + mode + '"' not in cpu_log:
            raise ValueError('Policy CPU mode evidence missing: ' + mode)
    if '"valid":true,"gpu_work":false' not in cpu_log:
        raise ValueError('Original worker CPU self-test evidence missing')
    tests = (build / 'semantic-cpu-tests.log').read_text()
    if not tests.rstrip().endswith('OK'):
        raise ValueError('Sealed-source semantic CPU tests did not pass')
    source_review = json.loads((build / 'independent-review.json').read_text())
    semantic_review = json.loads((build / 'independent-semantic-review.json').read_text())
    if any(review.get('pass') is not True or review.get('compiled_source_identity_sha256') != manifest['source_identity_sha256']
           for review in (source_review, semantic_review)):
        raise ValueError('Independent review is unavailable or targets a different worker')
    files = ('overlay-manifest.json', 'cpu-source-witness.json', 'cpu-self-test.log', 'cpu-build.log',
             'semantic-source-seal.json', 'semantic-cpu-tests.log', 'independent-review.json', 'independent-semantic-review.json')
    receipt = {'schema': 'private-compact-native-R4-verify-worker-READY-v1', 'ready_for_root_qualification': True,
               'build': str(build), 'base': manifest['base'], 'compiled_source_identity_sha256': manifest['source_identity_sha256'],
               'flag': 'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22', 'default_enabled': False,
               'scope': 'singleton main physicalR4 verification only', 'dispatches_per_layer': 6,
               'base_gather_dispatches_per_layer': 2, 'additional_gpu_allocation_bytes': 0, 'planner_threadgroup_bytes': 2432,
               'host_tus_rebuilt': witness['host_tus_rebuilt'], 'actual_new_header_consumers': witness['new_bridge_consumers'],
               'compiled_artifacts': witness['artifacts'],
               'receipts': [{'path': name, 'sha256': sha((build / name).read_bytes())} for name in files],
               'qualified_component_six_patterns': 'build/release/flash/sep22-expert-r4-compact-native-six-pattern-summary-v2.json',
               'root_launch_delta': 'Reuse exact qualified W5 environment, replace worker/library build and set SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22=1',
               'semantic_cli': f'{args.python} {build}/source/{PRIVATE}/semantic_quality.py --build {build} measure [unchanged root frozen22 arguments] --run-root-gpu',
               'semantic_compare_cli': f'{args.python} {build}/source/{PRIVATE}/semantic_quality.py --build {build} compare --reports [baseline] [candidate] --allow-runtime-change --output [fresh]',
               'full_model_quality_qualified': False, 'verification_state_qualified': False,
               'end_to_end_speed_qualified': False, 'gpu_work': False, 'model_payload_reads': False,
               'capture_payload_reads': False, 'production_edits': False,
               'future_extensions': 'R8/R16 proof proposal only; no extension source or executable built'}
    final.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'READY': str(final), 'source_identity_sha256': manifest['source_identity_sha256'], 'gpu_work': False,
                      'full_model_quality_qualified': False}))


if __name__ == '__main__':
    main()
