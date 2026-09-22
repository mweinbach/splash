#!/usr/bin/env python3
"""Compile/seal an isolated HC component; never open model payloads or Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', default='build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5')
    parser.add_argument('--build', default='build/hc-pad-producer-sep22-v1')
    parser.add_argument('--match-original-math', action='store_true', help='Match original089 AIR Metal4.1/default-fast recipe; requires new GPU proof')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    worker, build = (root / args.worker).resolve(), (root / args.build).resolve()
    if build.exists():
        raise SystemExit('fresh component directory required')
    sealed = json.loads((worker / 'compiled-cpu-seal.json').read_text())
    if not sealed['pass'] or sealed['gpu_executed']:
        raise SystemExit('CPU-sealed current parent required')
    for name, digest in sealed['source_sha256'].items():
        if sha(worker / 'source' / name) != digest:
            raise SystemExit('parent source drift: ' + name)
    for name, digest in sealed['artifact_sha256'].items():
        if sha(worker / name) != digest:
            raise SystemExit('parent artifact drift: ' + name)
    overlay = json.loads((worker / 'overlay-manifest.json').read_text())
    air_names = ['flash_affine', 'flash_float_dense_cache', 'flash_hc_fused', 'flash_hc_up_f32_mpp']
    air_inputs = []
    for name in air_names:
        matches = [(Path(path), digest) for path, digest in overlay['parent_input_seals'].items()
                   if path.endswith('.air') and re.sub(r'^\d+-', '', Path(path).stem) == name]
        if len(matches) != 1:
            raise SystemExit('exact original AIR missing/ambiguous: ' + name)
        path, digest = matches[0]
        if sha(path) != digest:
            raise SystemExit('original AIR drift: ' + name)
        air_inputs.append((path, digest))
    build.mkdir(parents=True)
    shutil.copytree(worker / 'source', build / 'source')
    private = build / 'source/dev/benchmarks/hc_pad_producer_sep22'
    private.mkdir(parents=True, exist_ok=True)
    for name in ['abi.hpp', 'bridge.hpp', 'candidate.metal', 'oracle.mm', 'prepare.py']:
        shutil.copyfile(here / name, private / name)
    for name, expected in [('flash_hc_fused.metal', 'c32cff6a4b4aab8efc8b0cfcc8212c191684a4b5d9d451f77b5c4a19d17d9a16'),
                           ('flash_hc_up_f32_mpp.metal', '6621085583023d0510fa69ca4c09216485ebb3edd65eb6ba70db789849b5e559')]:
        original = root / 'runtime/metal/kernels/shared' / name
        if sha(original) != expected:
            raise SystemExit('original helper differs from qualified shader source: ' + name)
        destination = build / 'source/runtime/metal/kernels/shared' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    objects = []
    for index, (name, digest) in enumerate(sealed['artifact_sha256'].items()):
        if not name.endswith('.o') or re.sub(r'^\d+-', '', Path(name).stem) == 'FlashWorker':
            continue
        original = worker / name
        frozen = build / 'objects' / f'{index:03d}-{original.name}'
        frozen.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, frozen)
        objects.append({'path': str(frozen), 'source': str(original), 'sha256': digest})
    if len(objects) != 53:
        raise SystemExit('parent effective nonworker closure must contain53 objects')
    airs = []
    for index, (path, digest) in enumerate(air_inputs):
        frozen = build / 'air' / f'{index:03d}-{path.name}'
        frozen.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, frozen)
        airs.append({'path': str(frozen), 'source': str(path), 'sha256': digest})
    source = private / 'candidate.metal'
    up_probe_source = None
    if args.match_original_math:
        complete = source.read_text()
        boundary = '// Diagnostic-only HC-up probe. Original timed HC-up remains linked unchanged.'
        if complete.count(boundary) != 1:
            raise SystemExit('Qualified split-HC translation unit boundary changed')
        down, up = complete.split(boundary, 1)
        source.write_text(down)
        up_probe_source = private / 'up_probe.metal'
        up_probe_source.write_text('#include "abi.hpp"\n' + boundary + up)
    metal_command = ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-O3',
                     '-fno-fast-math', '-I' + str(build / 'source/runtime'),
                     '-I' + str(private), '-c', str(source), '-o', str(build / 'candidate.air')]
    if args.match_original_math:
        metal_command = [value for value in metal_command if value != '-fno-fast-math']
        metal_command[metal_command.index('-std=metal4.0')] = '-std=metal4.1'
        metal_command[metal_command.index('-O3') + 1:metal_command.index('-O3') + 1] = ['-mmacosx-version-min=27.0']
    subprocess.run(metal_command, cwd=root, check=True)
    private_airs = [str(build / 'candidate.air')]
    up_probe_command = None
    if up_probe_source is not None:
        up_probe_command = [str(up_probe_source) if value == str(source) else
                            str(build / 'up-probe.air') if value == str(build / 'candidate.air') else value
                            for value in metal_command]
        subprocess.run(up_probe_command, cwd=root, check=True)
        private_airs.append(str(build / 'up-probe.air'))
    link_shader = ['xcrun', '-sdk', 'macosx', 'metallib', *[item['path'] for item in airs],
                   *private_airs, '-o', str(build / 'splash.metallib')]
    subprocess.run(link_shader, cwd=root, check=True)
    sources = [{'path': str(p), 'sha256': sha(p)} for p in sorted((build / 'source').rglob('*')) if p.is_file()]
    provenance = {'schema': 'hc-producer-pad-component-closure-v1', 'scope': 'VerifyR4 component only',
                  'worker': str(worker), 'worker_cpu_seal_sha256': sha(worker / 'compiled-cpu-seal.json'),
                  'host_timing_bytes': 200, 'sources': sources, 'objects': objects, 'original_AIRs': airs,
                  'candidate_AIR_sha256': sha(build / 'candidate.air'),
                  'up_probe_AIR_sha256': sha(build / 'up-probe.air') if up_probe_source else None,
                  'separate_down_and_safe_up_probe_translation_units': bool(up_probe_source),
                  'candidate_source_sha256': sha(source), 'private_ABI_sha256': sha(private / 'abi.hpp'),
                  'metallib_sha256': sha(build / 'splash.metallib'),
                  'private_math_recipe': 'Metal4.1/O3/default-fast/original089-baseline-match' if args.match_original_math else 'Metal4.0/O3/fno-fast-math/unqualified-real-input-policy',
                  'baseline_math_recipe_match_requires_fresh_GPU_proof': bool(args.match_original_math),
                  'original_down_source_sha256': sha(root / 'runtime/metal/kernels/shared/flash_hc_fused.metal'),
                  'original_up_source_sha256': sha(root / 'runtime/metal/kernels/shared/flash_hc_up_f32_mpp.metal')}
    (build / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summary = {key: value for key, value in provenance.items() if key not in ['sources', 'objects']}
    (build / 'HCProducerPadProvenance.hpp').write_text(
        '#pragma once\ninline constexpr const char *kHCProducerPadMetallibSHA=' + json.dumps(provenance['metallib_sha256'])
        + ';\ninline constexpr const char *kHCProducerPadProvenance=R"CLOSURE('
        + json.dumps(summary, sort_keys=True) + ')CLOSURE";\n')
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
             '-fobjc-arc', '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(private), '-I' + str(build)]
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, str(private / 'oracle.mm'),
               *[obj['path'] for obj in objects], '-framework', 'Foundation',
               '-framework', 'Metal', '-framework', 'IOKit', '-o', str(build / 'oracle')]
    subprocess.run(command, cwd=root, check=True)
    result = subprocess.run([str(build / 'oracle'), '--cpu-only'], cwd=root, check=True,
                            capture_output=True, text=True)
    (build / 'cpu-self-test.json').write_text(result.stdout)
    manifest = {**provenance, 'pass': True, 'gpu_executed': False, 'model_payload_bytes_read': 0,
                'oracle_sha256': sha(build / 'oracle'), 'cpu': json.loads(result.stdout),
                'compiler_command': command, 'metal_command': metal_command,
                'up_probe_metal_command': up_probe_command,
                'metallib_link_command': link_shader, 'Root_numerical_timing_qualification_complete': False}
    (build / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ['pass', 'gpu_executed', 'oracle_sha256', 'metallib_sha256']}))


if __name__ == '__main__':
    main()
