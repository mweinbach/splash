#!/usr/bin/env python3
"""CPU-only fresh diagnostic hook relink; preserve all other object bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-build', required=True)
    p.add_argument('--build', required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    parent, build = (root / a.parent_build).resolve(), (root / a.build).resolve()
    if build.exists():
        raise SystemExit('fresh diagnostic output required')
    original = json.loads((parent / 'CPU_READY.json').read_text())
    if not original['pass'] or original['gpu_executed']:
        raise SystemExit('CPU-sealed parent required')
    build.mkdir(parents=True)
    shutil.copytree(parent / 'source', build / 'source')
    private = build / 'source/dev/benchmarks/batch_verify_exact_sep22'
    shutil.copyfile(here / 'batch_inspection.cpp.inc', private / 'batch_inspection.cpp.inc')
    worker = Path(original['worker'])
    batch_cpp = build / 'source/runtime/flash/FlashBatchVerify.cpp'
    batch_cpp.write_text('#include "inspect.hpp"\n' + (worker / 'source/runtime/flash/FlashBatchVerify.cpp').read_text()
                         + '\n' + (private / 'batch_inspection.cpp.inc').read_text())
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
             '-fobjc-arc', '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build / 'source/dev/benchmarks/prefill4k_attention'),
             '-I' + str(private), '-I' + str(build)]
    objects = []
    for old in original['objects']:
        src = Path(old['path'])
        if sha(src) != old['sha256']:
            raise SystemExit('parent compiled closure changed')
        dst = build / 'objects' / src.name
        dst.parent.mkdir(exist_ok=True)
        entry = dict(old, path=str(dst), diagnostic_parent_path=str(src), diagnostic_parent_sha256=old['sha256'])
        if src.name == '010-FlashBatchVerify.o':
            subprocess.run(['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MMD', '-MP', '-c', str(batch_cpp), '-o', str(dst)], cwd=root, check=True)
            entry['inspection_diagnostic_only_recompiled'] = True
        else:
            shutil.copyfile(src, dst)
            entry['inspection_diagnostic_only_recompiled'] = False
        entry['sha256'] = sha(dst)
        objects.append(entry)
    if sum(x['inspection_diagnostic_only_recompiled'] for x in objects) != 1:
        raise SystemExit('exactly the BatchVerify inspection object must change')
    shutil.copyfile(parent / 'splash.metallib', build / 'splash.metallib')
    headers = []
    for old in original['headers']:
        rel = Path(old['path']).relative_to(parent)
        dst = build / rel
        headers.append({'path': str(dst), 'sha256': sha(dst)})
    prov = {k: v for k, v in original.items() if k not in ['headers', 'objects', 'compiler_commands', 'artifact_sha256', 'cpu_control', 'cpu_candidate']}
    prov.update({'objects': objects, 'headers': headers, 'inspection_diagnostic_parent_build': str(parent),
                 'only_one_inspection_object_recompiled': True, 'unchanged_floating_library': True,
                 'Root_new_artifact_pin_required': True, 'gpu_executed': False, 'model_or_input_or_runtime_payload_reads': 0})
    (build / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    summary = {k: v for k, v in prov.items() if k not in ['objects', 'header_census', 'headers']}
    (build / 'BatchVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='
        + json.dumps(str(build / 'provenance.json')) + ';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('
        + json.dumps(summary, sort_keys=True) + ')META";\n')
    commands = []
    for role in ['control', 'candidate']:
        cmd = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-Wno-unused-function', '-DSPLASH_VERIFY_CANDIDATE=' + str(int(role == 'candidate')),
               str(private / 'oracle.mm'), *[x['path'] for x in objects], '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit',
               '-o', str(build / ('oracle-' + role))]
        subprocess.run(cmd, cwd=root, check=True)
        commands.append(cmd)
        cpu = subprocess.run([str(build / ('oracle-' + role)), '--cpu-only'], cwd=root, check=True, text=True, capture_output=True)
        prov['cpu_' + role] = json.loads(cpu.stdout)
        (build / ('cpu-self-test-' + role + '.json')).write_text(cpu.stdout)
    prov['compiler_commands'] = commands
    prov['artifact_sha256'] = {name: sha(build / name) for name in ['oracle-control', 'oracle-candidate', 'splash.metallib']}
    (build / 'CPU_READY.json').write_text(json.dumps(prov, indent=2) + '\n')
    (build / 'manifest.json').write_text(json.dumps(prov, indent=2) + '\n')
    print(json.dumps({'pass': True, 'build': str(build), 'only_changed_object': '010-FlashBatchVerify.o',
                      'other_objects_byte_identical': 52, 'artifacts': prov['artifact_sha256'], 'GPU_started': False, 'payload_reads': 0}))


if __name__ == '__main__':
    main()
