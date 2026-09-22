#!/usr/bin/env python3
"""Compose two qualified private schedules after the compact worker is sealed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from overlay import transform


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--build', required=True)
    parser.add_argument('--ready-receipt', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    base, build = (root / args.base).resolve(), (root / args.build).resolve()
    receipt = (root / args.ready_receipt).resolve()
    if build.exists():
        raise SystemExit('fresh combined build required')
    ready = json.loads(receipt.read_text())
    if (ready.get('schema') != 'private-compact-native-R4-verify-worker-READY-v1'
            or ready.get('ready_for_root_qualification') is not True
            or Path(ready.get('build', '')).resolve() != base):
        raise SystemExit('compact actor final PASS receipt required')
    parent_path = base / 'overlay-manifest.json'
    parent = json.loads(parent_path.read_text())
    witness_path = base / 'cpu-source-witness.json'
    witness = json.loads(witness_path.read_text())
    if not witness.get('pass'):
        raise SystemExit('compact source witness not PASS')
    runtime = {entry['path']: entry['sha256'] for entry in witness['artifacts']}
    for name in ['splash-flash', 'splash.metallib']:
        if runtime.get(name) != sha(base / name):
            raise SystemExit('compact compiled artifact changed: ' + name)
    for entry in witness['artifacts']:
        if sha(base / entry['path']) != entry['sha256']:
            raise SystemExit('compact compiled object/AIR changed: ' + entry['path'])
    for entry in parent['files']:
        if sha(base / 'source' / entry['path']) != entry['sha256']:
            raise SystemExit('compact source changed: ' + entry['path'])
    for entry in parent['frozen_inputs']:
        if sha(base / entry['path']) != entry['sha256']:
            raise SystemExit('compact frozen input changed: ' + entry['path'])
    component = root / 'build/hc-pad-producer-sep22-v6'
    component_manifest = json.loads((component / 'manifest.json').read_text())
    report = root / 'build/release/flash/sep22-hc-pad-producer-r4-97chain-v2.json'
    qualified = json.loads(report.read_text())
    if not qualified['pass'] or qualified['cases'] != 97 or qualified['library_sha256'] != component_manifest['metallib_sha256']:
        raise SystemExit('Root exact97 HC component proof required')
    # Copy only after explicit stable receipt; the actor's tree stays read-only.
    build.mkdir(parents=True)
    shutil.copytree(base / 'source', build / 'source')
    shutil.copytree(base / 'reused', build / 'reused')
    changed = []
    for relative in ['runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm']:
        path = build / 'source' / relative
        before = path.read_text()
        after = transform(relative, before)
        path.write_text(after)
        changed.append({'path': relative, 'parent_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'sha256': sha(path)})
    private = build / 'source/dev/benchmarks/hc_pad_verify_worker_sep22'
    private.mkdir(parents=True, exist_ok=True)
    for name in ['worker_bridge.hpp', 'policy_cpu.cpp', 'overlay.py', 'prepare.py', 'worker.mk', 'semantic_quality.py']:
        shutil.copyfile(here / name, private / name)
    qualified_dir = component / 'source/dev/benchmarks/hc_pad_producer_sep22'
    hc_dir = build / 'source/dev/benchmarks/hc_pad_producer_sep22'
    hc_dir.mkdir(parents=True, exist_ok=True)
    for name in ['abi.hpp', 'candidate.metal']:
        shutil.copyfile(qualified_dir / name, hc_dir / name)
    full_shader = (hc_dir / 'candidate.metal').read_text()
    probe_boundary = '// Diagnostic-only HC-up probe. Original timed HC-up remains linked unchanged.'
    if full_shader.count(probe_boundary) != 1:
        raise SystemExit('qualified HC-up diagnostic source boundary changed')
    (hc_dir / 'candidate.metal').write_text(full_shader.split(probe_boundary, 1)[0])
    for name in ['flash_hc_fused.metal', 'flash_hc_up_f32_mpp.metal']:
        relative = Path('runtime/metal/kernels/shared') / name
        path = build / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(component / 'source' / relative, path)
    compact_air = build / 'reused/air/compact-qualified.air'
    shutil.copyfile(base / 'compact-plan.air', compact_air)
    link = (base / 'link-inputs.mk').read_text()
    link += 'AIRS += $(BUILD)/reused/air/compact-qualified.air\n'
    (build / 'link-inputs.mk').write_text(link)
    snapshots = ['overlay-manifest.json', 'cpu-source-witness.json', 'independent-review.json']
    for name in ['semantic-source-seal.json', 'adapter-receipt.json', 'ready-closure.json']:
        if (base / name).exists():
            snapshots.append(name)
    (build / 'parent-seals').mkdir()
    for name in snapshots:
        shutil.copyfile(base / name, build / 'parent-seals' / name)
    shutil.copyfile(receipt, build / 'parent-seals/actor-ready-receipt.json')
    sources = [{'path': str(p.relative_to(build / 'source')), 'sha256': sha(p)}
               for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    frozen = [{'path': str(p.relative_to(build)), 'sha256': sha(p)}
              for p in sorted((build / 'reused').rglob('*')) if p.is_file()]
    manifest = {'schema': 'hc-pad-compact-r4-verify-teacher-source-v1', 'base': str(base),
                'base_manifest_sha256': sha(parent_path), 'base_witness_sha256': sha(witness_path),
                'actor_ready_receipt_sha256': sha(receipt), 'base_runtime_sha256': runtime,
                'compact_source_identity_sha256': parent['source_identity_sha256'],
                'HC_component_manifest_sha256': sha(component / 'manifest.json'),
                'HC_Root_component_report_sha256': sha(report), 'files': sources,
                'changed_sources': changed, 'frozen_inputs': frozen,
                'all50_noncore_host_TUs_rebuilt': True, 'public_headers_changed_by_HC': False,
                'Scope': 'HC main VerifyR4 only; compact singleton main VerifyR4; others retained',
                'numerical_derivative_changed_by_HC': False, 'new_GPU_buffers_or_cache_bytes': 0,
                'gpu_executed': False, 'model_payload_bytes_read': 0,
                'source_transform_sha256': sha(here / 'overlay.py'),
                'link_inputs_sha256': sha(build / 'link-inputs.mk')}
    (build / 'overlay-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': str(build), 'sources': len(sources), 'changed': changed,
                      'gpu_executed': False, 'model_payload_bytes_read': 0}))


if __name__ == '__main__':
    main()
