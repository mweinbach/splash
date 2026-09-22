#!/usr/bin/env python3
"""Freeze a Root-only TRUNKPREF oracle without loading model payloads or Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('--build', required=True)
    parser.add_argument('--role', choices=['teacher', 'phase'], required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    worker = (root / args.worker).resolve()
    build = (root / args.build).resolve()
    if build.exists():
        raise SystemExit('fresh oracle directory required')
    seal_path = worker / 'compiled-cpu-seal.json'
    seal = json.loads(seal_path.read_text())
    if not seal['pass'] or seal['gpu_executed']:
        raise SystemExit('valid CPU-only worker seal required')
    for relative, digest in seal['source_sha256'].items():
        if sha(worker / 'source' / relative) != digest:
            raise SystemExit('worker source changed: ' + relative)
    for relative, digest in seal['artifact_sha256'].items():
        if sha(worker / relative) != digest:
            raise SystemExit('worker artifact changed: ' + relative)
    # Artifact census is the exact effective make link closure: no parent union.
    object_paths = [(worker / relative, digest) for relative, digest in
                    seal['artifact_sha256'].items() if relative.endswith('.o')]
    if len(object_paths) != 54:
        raise SystemExit('expected exact worker closure of 54 objects')
    mains = [p for p, _ in object_paths if re.sub(r'^\d+-', '', p.stem) == 'FlashWorker']
    if len(mains) != 1:
        raise SystemExit('exactly one Worker main must be excluded')
    object_paths = [(p, digest) for p, digest in object_paths if p not in mains]
    identities = [re.sub(r'^\d+-', '', p.stem) for p, _ in object_paths]
    if len(set(identities)) != len(identities) or 'FlashForward' not in identities or 'teacher_bulk' not in identities:
        raise SystemExit('duplicate/stale/missing nonworker closure')
    build.mkdir(parents=True)
    shutil.copytree(worker / 'source', build / 'source')
    candidate = build / 'source/dev/benchmarks/prefill_exact_phase_sep22'
    candidate.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(here / 'oracle.mm', candidate / 'oracle.mm')
    shutil.copyfile(worker / 'splash.metallib', build / 'splash.metallib')
    objects = []
    for index, (path, digest) in enumerate(object_paths):
        frozen = build / 'objects' / f'{index:03d}-{path.name}'
        frozen.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, frozen)
        assert sha(frozen) == digest
        objects.append({'original': str(path), 'path': str(frozen), 'sha256': digest,
                        'bytes': frozen.stat().st_size})
    sources = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    provenance = {'schema': 'trunkpref-exact-compile-closure-v1', 'role': args.role,
                  'worker': str(worker), 'worker_cpu_seal_sha256': sha(seal_path),
                  'worker_sha256': sha(worker / 'splash-flash'),
                  'metallib_sha256': sha(build / 'splash.metallib'),
                  'worker_header_api_changed': False, 'command_timing_bytes': 200,
                  'scope': 'TRUNKPREF only; no trained head prime or state',
                  'sources': sources, 'objects': objects}
    provenance_path = build / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    header = '#pragma once\ninline constexpr const char *kPrefillExactProvenancePath = '
    header += json.dumps(str(provenance_path)) + ';\n'
    header += 'inline constexpr const char *kPrefillExactBuildProvenance = R"CLOSURE('
    header += json.dumps({key: value for key, value in provenance.items()
                         if key not in ['sources', 'objects']}, sort_keys=True) + ')CLOSURE";\n'
    (build / 'PrefillExactBuildProvenance.hpp').write_text(header)
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror',
             '-Wno-deprecated-declarations', '-fobjc-arc', '-mmacosx-version-min=27.0',
             '-DSPLASH_INT8_EXPERIMENT=1', '-DSPLASH_PHASE_ORACLE=' + str(int(args.role == 'phase')),
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build)]
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, str(candidate / 'oracle.mm'),
               *[p['path'] for p in objects], '-framework', 'Foundation', '-framework',
               'Metal', '-framework', 'IOKit', '-o', str(build / 'oracle')]
    (build / 'compiler-command.json').write_text(json.dumps(command, indent=2) + '\n')
    subprocess.run(command, cwd=root, check=True)
    cpu = subprocess.run([str(build / 'oracle'), '--cpu-only'], cwd=root, check=True,
                         text=True, capture_output=True)
    (build / 'cpu-self-test.json').write_text(cpu.stdout)
    shutil.copyfile(here / 'frame_headers.py', build / 'frame_headers.py')
    parser_cpu = subprocess.run([sys.executable, str(build / 'frame_headers.py'), '--self-test'],
                                cwd=root, check=True, text=True, capture_output=True)
    (build / 'header-parser-cpu-self-test.json').write_text(parser_cpu.stdout)
    manifest = {**provenance, 'gpu_executed': False, 'model_payload_bytes_read': 0,
                'oracle_sha256': sha(build / 'oracle'),
                'provenance_sha256': sha(provenance_path),
                'provenance_header_sha256': sha(build / 'PrefillExactBuildProvenance.hpp'),
                'compiler_command': command, 'cpu': json.loads(cpu.stdout),
                'header_parser_sha256': sha(build / 'frame_headers.py'),
                'header_parser_cpu': json.loads(parser_cpu.stdout),
                'Root_GPU_qualification_complete': False}
    (build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'pass': True, 'role': args.role, 'build': str(build),
                      'oracle_sha256': manifest['oracle_sha256'],
                      'metallib_sha256': manifest['metallib_sha256'],
                      'sources': len(sources), 'nonworker_objects': len(objects),
                      'gpu_executed': False, 'model_payload_bytes_read': 0}))


if __name__ == '__main__':
    main()
