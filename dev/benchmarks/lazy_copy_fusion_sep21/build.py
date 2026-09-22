#!/usr/bin/env python3
"""Freeze/compile an exact private lazy-copy worker; no GPU recipes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='build/lazy-copy-fusion-sep21-v2')
    parser.add_argument('--base', default='build/moe-pointwise-sep21-worker-v1')
    parser.add_argument('--reuse-applied-source', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    build, base = root / args.build, root / args.base
    if (build / 'manifest.json').exists():
        raise SystemExit('sealed build cannot be changed')
    if not args.reuse_applied_source:
        if build.exists():
            raise SystemExit('fresh build required')
        build.mkdir(parents=True)
        shutil.copytree(base / 'source', build / 'source')
        spec = importlib.util.spec_from_file_location('overlay', here / 'overlay.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.patch(build / 'source')
    src = build / 'source/dev/benchmarks/lazy_copy_fusion_sep21'
    if src.exists():
        raise SystemExit('candidate source already copied; choose fresh build')
    shutil.copytree(here, src)
    changed_names = ['FlashGDNLazyRollback', 'FlashForward', 'FlashBatchVerify',
                     'FlashBatchVerifyGDN', 'FlashWorker']
    all_host = sorted((base / 'reused/base-host').glob('*.o'))
    all_host += sorted((base / 'host').glob('*.o'))
    all_host += sorted((base / 'reused/core/engine').rglob('*.o'))
    if len({p.name for p in all_host}) != len(all_host):
        raise SystemExit('unexpected duplicate object name in frozen base')
    host = [p for p in all_host if p.stem not in changed_names]
    airs_line = next(line for line in (base / 'link-inputs.mk').read_text().splitlines()
                    if line.startswith('AIRS := '))
    airs = [base / token.removeprefix('$(BUILD)/')
            for token in airs_line.removeprefix('AIRS := ').split()]
    airs += [base / 'pointwise.air']
    closure = []
    for i, path in enumerate(host + airs):
        copied = build / 'reused' / f'{i:03d}-{path.name}'
        copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, copied)
        closure.append({'original': str(path), 'frozen': str(copied),
                        'sha256': sha(copied), 'bytes': copied.stat().st_size})
    reused_host = [Path(e['frozen']) for e in closure[:len(host)]]
    reused_air = [Path(e['frozen']) for e in closure[len(host):]]
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build / 'source/dev/benchmarks/prefill4k_attention'),
             '-I' + str(src), '-I' + str(build), '-mmacosx-version-min=27.0',
             '-DSPLASH_INT8_EXPERIMENT=1', '-fobjc-arc']
    metal_flags = ['-std=metal4.1', '-O3', '-Wall', '-Wextra', '-Werror',
                   '-I' + str(build / 'source/runtime'), '-mmacosx-version-min=27.0']
    commands = []
    outputs = []
    for name in changed_names:
        source = build / 'source/runtime/flash' / (name + ('.mm' if name == 'FlashWorker' else '.cpp'))
        obj = build / 'host' / (name + '.o'); obj.parent.mkdir(exist_ok=True)
        command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-c', str(source), '-o', str(obj)]
        subprocess.run(command, cwd=root, check=True); commands.append(command); outputs.append(obj)
    command = ['xcrun', '-sdk', 'macosx', 'metal', *metal_flags, '-c',
               str(src / 'candidate.metal'), '-o', str(build / 'candidate.air')]
    subprocess.run(command, cwd=root, check=True); commands.append(command)
    command = ['xcrun', '-sdk', 'macosx', 'metallib', *map(str, reused_air),
               str(build / 'candidate.air'), '-o', str(build / 'splash.metallib')]
    subprocess.run(command, cwd=root, check=True); commands.append(command)
    link_flags = ['-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit']
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, *map(str, outputs + reused_host),
               *link_flags, '-o', str(build / 'splash-flash')]
    subprocess.run(command, cwd=root, check=True); commands.append(command)
    result = subprocess.run([str(build / 'splash-flash'), '--cpu-self-test'],
                            cwd=root, capture_output=True, text=True, check=True)
    (build / 'worker-cpu-self-test.txt').write_text(result.stdout)
    # The strict layer/model oracle may be added in a separate immutable build;
    # never imply its correctness check has run from this CPU worker preparation.
    objects = outputs + reused_host
    (build / 'link-inputs.json').write_text(json.dumps({
        'host_objects': list(map(str, objects)), 'nonworker_objects':
        list(map(str, [p for p in objects if p.name != 'FlashWorker.o'])),
        'flags': flags, 'link_flags': link_flags}, indent=2) + '\n')
    sources = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    manifest = {'schema': 'splash-sep21-exact-lazy-copy-worker-frozen-v1',
                'gpu_executed': False, 'production_sources_modified': False,
                'new_gpu_allocations': 0, 'base_binary_sha256': sha(base / 'splash-flash'),
                'base_numerical_derivative_policy_unchanged': True,
                'required_environment_flag': 'SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21=1',
                'default_flag0_graphs_preserved': True,
                'changed_objects': changed_names, 'frozen_inputs': closure, 'sources': sources,
                'compiler_commands': commands, 'worker_sha256': sha(build / 'splash-flash'),
                'metallib_sha256': sha(build / 'splash.metallib'),
                'strict_gpu_correctness_status': 'not run',
                'timing_projection': '72copy dispatches eliminated at R4/B1; stage1.17ms ceiling is not an unsampled speedup'}
    (build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'pass': True, 'gpu_executed': False, 'build': str(build),
                      'sources': len(sources), 'frozen_inputs': len(closure)}))


if __name__ == '__main__':
    main()
