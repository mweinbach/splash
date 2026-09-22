#!/usr/bin/env python3
"""Seal a strict layer or whole oracle over the immutable private worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', default='build/lazy-copy-fusion-sep21-v2')
    parser.add_argument('--build', default='build/lazy-copy-fusion-sep21-layer-oracle-v1')
    parser.add_argument('--kind', choices=['layer', 'whole', 'batch'], default='layer')
    parser.add_argument('--object-witness', help='previous sealed oracle manifest for all nonworker object hashes')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent; root = here.parents[2]
    worker, build = root / args.worker, root / args.build
    if build.exists():
        raise SystemExit('fresh oracle build directory required')
    worker_manifest = json.loads((worker / 'manifest.json').read_text())
    if sha(worker / 'splash-flash') != worker_manifest['worker_sha256'] or sha(worker / 'splash.metallib') != worker_manifest['metallib_sha256']:
        raise SystemExit('immutable worker artifact changed')
    for entry in worker_manifest['sources']:
        if sha(Path(entry['path'])) != entry['sha256']:
            raise SystemExit('immutable worker source changed')
    for entry in worker_manifest['frozen_inputs']:
        if sha(Path(entry['frozen'])) != entry['sha256']:
            raise SystemExit('immutable worker reused input changed')
    witness_path = Path(args.object_witness) if args.object_witness else root / 'build/lazy-copy-fusion-sep21-layer-oracle-v1/manifest.json'
    witness = {}
    if witness_path.exists():
        previous = json.loads(witness_path.read_text())
        if previous['worker_provenance']['worker_sha256'] != worker_manifest['worker_sha256']:
            raise SystemExit('object witness belongs to a different worker')
        witness = {entry['original']: entry['sha256'] for entry in previous['frozen_inputs']}
    build.mkdir(parents=True)
    shutil.copytree(worker / 'source', build / 'source')
    candidate = build / 'source/dev/benchmarks/lazy_copy_fusion_sep21'
    source_name = {'layer': 'oracle.mm', 'whole': 'whole_oracle.mm', 'batch': 'batch_oracle.mm'}[args.kind]
    source_files = [source_name] if args.kind == 'whole' else [source_name, 'gdn_fixture.hpp']
    for name in source_files:
        shutil.copyfile(here / name, candidate / name)
    shutil.copyfile(worker / 'splash.metallib', build / 'splash.metallib')
    linked = json.loads((worker / 'link-inputs.json').read_text())
    objects, closure = [], []
    for i, item in enumerate(linked['nonworker_objects']):
        path = Path(item); frozen = build / 'reused' / f'{i:03d}-{path.name}'
        if witness and (item not in witness or sha(path) != witness[item]):
            raise SystemExit('linked object differs from previous sealed oracle witness')
        frozen.parent.mkdir(exist_ok=True); shutil.copyfile(path, frozen)
        closure.append({'original': str(path), 'frozen': str(frozen), 'path': str(frozen),
                        'sha256': sha(frozen), 'bytes': frozen.stat().st_size})
        objects.append(frozen)
    source_inventory = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
                        for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    provenance = {'schema': 'splash-sep21-exact-lazy-copy-oracle-closure-v1',
                  'kind': args.kind, 'worker_manifest_sha256': sha(worker / 'manifest.json'),
                  'worker_sha256': sha(worker / 'splash-flash'),
                  'metallib_sha256': sha(build / 'splash.metallib'),
                  'abi_header_sha256': sha(build / 'source/runtime/metal/MetalBackend.hpp'),
                  'objects': closure, 'sources': source_inventory, 'command_timing_bytes': 200}
    if witness:
        provenance['previous_object_witness_sha256'] = sha(witness_path)
    header = {'layer': 'LazyCopyFusionOracleBuildProvenance', 'whole': 'LazyCopyFusionWholeBuildProvenance',
              'batch': 'LazyCopyFusionBatchBuildProvenance'}[args.kind]
    constant = {'layer': 'kLazyCopyFusionOracleBuildProvenance', 'whole': 'kLazyCopyFusionWholeBuildProvenance',
                'batch': 'kLazyCopyFusionBatchBuildProvenance'}[args.kind]
    (build / (header + '.hpp')).write_text(
        '#pragma once\ninline constexpr const char* ' + constant + ' = R"CLOSURE('
        + json.dumps(provenance, sort_keys=True) + ')CLOSURE";\n')
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(candidate), '-I' + str(build), '-mmacosx-version-min=27.0',
             '-DSPLASH_INT8_EXPERIMENT=1', '-fobjc-arc']
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, str(candidate / source_name),
               *map(str, objects), *linked['link_flags'], '-o', str(build / 'oracle')]
    subprocess.run(command, cwd=root, check=True)
    result = subprocess.run([str(build / 'oracle'), '--cpu-only'], cwd=root,
                            check=True, capture_output=True, text=True)
    (build / 'cpu-self-test.json').write_text(result.stdout)
    sources = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    manifest = {'schema': 'splash-sep21-exact-lazy-copy-oracle-frozen-v1',
                'kind': args.kind, 'gpu_executed': False, 'model_payloads_read': False,
                'worker_provenance': provenance, 'sources': sources, 'frozen_inputs': closure,
                'compiler_command': command, 'oracle_sha256': sha(build / 'oracle'),
                'metallib_sha256': sha(build / 'splash.metallib'),
                'gpu_qualification_status': 'not run'}
    (build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'pass': True, 'gpu_executed': False, 'kind': args.kind,
                      'build': str(build), 'sources': len(sources), 'frozen_inputs': len(closure)}))


if __name__ == '__main__':
    main()
