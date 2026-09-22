#!/usr/bin/env python3
"""Freeze and compile a Root-only exact HC-down row-reuse component screen."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--build', default='build/hc-down-rowreuse-sep21-v3')
    args.add_argument('--base', default='build/moe-pointwise-sep21-worker-v1')
    args = args.parse_args()
    root = Path(__file__).resolve().parents[3]
    here = Path(__file__).resolve().parent
    build, base = root / args.build, root / args.base
    if build.exists():
        raise SystemExit('fresh build directory required')
    build.mkdir(parents=True)
    shutil.copytree(base / 'source', build / 'source')
    src = build / 'source/dev/benchmarks/hc_down_rowreuse_sep21'
    shutil.copytree(here, src)
    names = ['FlashAffine', 'FlashHC', 'FlashHCFused', 'FlashFloatDenseCache',
             'FlashOperandStore', 'FlashDenseCache', 'FlashAffineMPP',
             'FlashDescriptor', 'FlashWeights', 'FlashPLESSDStore',
             'FlashInt8ExpertStoreMetadata']
    objects = [base / 'reused/base-host' / (n + '.o') for n in names]
    objects += [base / 'reused/core/engine/metal' / (n + '.o')
                for n in ['MetalBackend', 'DeviceCapabilities']]
    air_line = next(line for line in (base / 'link-inputs.mk').read_text().splitlines()
                    if line.startswith('AIRS := '))
    airs = [base / token.removeprefix('$(BUILD)/')
            for token in air_line.removeprefix('AIRS := ').split()]
    airs += [base / 'pointwise.air']
    inputs = objects + airs
    sealed = []
    for i, path in enumerate(inputs):
        if not path.is_file():
            raise SystemExit(f'missing frozen input: {path}')
        copied = build / 'reused' / f'{i:03d}-{path.name}'
        copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, copied)
        sealed.append({'original': str(path), 'frozen': str(copied),
                       'sha256': sha(copied), 'bytes': copied.stat().st_size})
    objects = [Path(x['frozen']) for x in sealed[:len(objects)]]
    airs = [Path(x['frozen']) for x in sealed[len(objects):]]
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror',
             '-I' + str(build / 'source/runtime'), '-I' + str(src), '-I' + str(build),
             '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1', '-fobjc-arc']
    metalflags = ['-std=metal4.1', '-O3', '-Wall', '-Wextra', '-Werror',
                  '-I' + str(build / 'source/runtime'), '-I' + str(src),
                  '-mmacosx-version-min=27.0']
    commands = [
        ['xcrun', '-sdk', 'macosx', 'metal', *metalflags, '-c',
         str(src / 'flash_hc_down_packed.metal'), '-o', str(build / 'candidate.air')],
        ['xcrun', '-sdk', 'macosx', 'metallib', *map(str, airs), str(build / 'candidate.air'),
         '-o', str(build / 'splash.metallib')],
        ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-c',
         str(src / 'FlashHCDownPacked.cpp'), '-o', str(build / 'bridge.o')],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    provenance = {'schema': 'splash-sep21-frozen-host-object-closure-v1',
                  'base_binary_sha256': sha(base / 'splash-flash'),
                  'abi_header_sha256': sha(build / 'source/runtime/metal/MetalBackend.hpp'),
                  'objects': sealed[:len(objects)],
                  'candidate_bridge_sha256': sha(build / 'bridge.o')}
    (build / 'FlashHCDownPackedHostProvenance.hpp').write_text(
        '#pragma once\ninline constexpr const char* kFlashOracleHostObjectProvenance = R"CLOSURE('
        + json.dumps(provenance, sort_keys=True) + ')CLOSURE";\n')
    subprocess.run(['xcrun', '-sdk', 'macosx', 'clang++', *flags,
                    str(src / 'flash_hc_down_packed_oracle.mm'), str(build / 'bridge.o'),
                    *map(str, objects), '-framework', 'Foundation', '-framework', 'Metal',
                    '-framework', 'IOKit', '-o', str(build / 'oracle')], cwd=root, check=True)
    result = subprocess.run([str(build / 'oracle'), '--cpu-self-test'],
                            cwd=root, check=True, capture_output=True, text=True)
    (build / 'cpu-self-test.json').write_text(result.stdout)
    sources = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    manifest = {'schema': 'splash-sep21-hc-rowreuse-frozen-build-v2', 'gpu_executed': False,
                'base_binary_sha256': sha(base / 'splash-flash'), 'frozen_inputs': sealed,
                'sources': sources, 'compiler_commands': commands,
                'oracle_sha256': sha(build / 'oracle'), 'metallib_sha256': sha(build / 'splash.metallib'),
                'preserved_previous_result': str(root / 'build/release/flash/hc-down-packed-reuse2-r16-screen.json'),
                'policy': 'Root-only GPU; physical R4/8/16; exact raw F32 and full BF16 before timing; original packed F32 coefficients; no extra weight cache'}
    (build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'pass': True, 'gpu_executed': False, 'build': str(build),
                      'sources': len(sources), 'frozen_inputs': len(sealed)}))


if __name__ == '__main__':
    main()
