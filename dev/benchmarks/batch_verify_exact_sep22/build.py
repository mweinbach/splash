#!/usr/bin/env python3
"""CPU-only sealed clone, complete actual header-consumer rebuild and link.

Only source, JSON metadata and object/executable/library artifacts are read.
Model, canonical token and runtime export/capture payloads are never opened.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', required=True)
    p.add_argument('--build', required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    worker, build = (root / a.worker).resolve(), (root / a.build).resolve()
    if build.exists():
        raise SystemExit('fresh private oracle build required')
    ready = json.loads((worker / 'READY.json').read_text())
    overlay = json.loads((worker / 'overlay-manifest.json').read_text())
    expected_identity = 'bd89d8b66c88b4559d97faff00fd1a75f4ed7c275fc29a01604179531a482ed7'
    if not ready['ready_for_root_qualification'] or ready['source_identity_sha256'] != expected_identity:
        raise SystemExit('required CPU-qualified v1c source identity missing')
    artifacts = {x['path']: x['sha256'] for x in ready['artifacts']}
    if sha(worker / 'splash.metallib') != artifacts['splash.metallib']:
        raise SystemExit('sealed candidate library changed')
    host = list(overlay['rebuild'])
    # teacher_bulk is an explicitly rebuilt additional header consumer in the
    # generated linker census, recorded separately from the inherited list.
    if len(host) == 49 and (worker / 'host/teacher_bulk.o').is_file():
        host.append({'object': 'teacher_bulk', 'source': 'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'})
    if len(host) != 50 or len({x['object'] for x in host}) != 50:
        raise SystemExit('complete actual 50-TU host census required')
    build.mkdir(parents=True)
    shutil.copytree(worker / 'source', build / 'source')
    private = build / 'source/dev/benchmarks/batch_verify_exact_sep22'
    private.mkdir(parents=True, exist_ok=True)
    names = ['inspect.hpp', 'batch_inspection.cpp.inc', 'forward_inspection.cpp.inc', 'oracle.mm', 'build.py']
    for name in names:
        shutil.copyfile(here / name, private / name)
    metal_h = build / 'source/runtime/metal/MetalBackend.hpp'
    original_metal_h = metal_h.read_text()
    if original_metal_h != (root / 'runtime/metal/MetalBackend.hpp').read_text():
        raise SystemExit('clone MetalBuffer header differs from clean original Core source ABI')
    getter_anchor = '  [[nodiscard]] uint64_t sizeBytes() const noexcept;'
    if original_metal_h.count(getter_anchor) < 1:
        raise SystemExit('MetalBuffer readonly getter declaration anchor drift')
    metal_h.write_text(original_metal_h.replace(getter_anchor, getter_anchor
        + '\n  // Clone-only metadata: native owner charge and opaque owner identity.\n'
        + '  [[nodiscard]] uint64_t oracleChargedBytesSep22() const noexcept;\n'
        + '  [[nodiscard]] uintptr_t oracleOwnerIdentitySep22() const noexcept;', 1))
    core_sources = {'MetalBackend': 'runtime/metal/MetalBackend.mm', 'DeviceCapabilities': 'runtime/metal/DeviceCapabilities.cpp',
                    'Protocol': 'runtime/engine/Protocol.cpp', 'MemoryGovernor': 'runtime/engine/MemoryGovernor.cpp'}
    core_source_pins = []
    for name, relative in core_sources.items():
        src = root / relative
        dst = build / 'source' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        core_source_pins.append({'source': relative, 'original_sha256': sha(src), 'original_git_tracked_source_unmodified_by_oracle': True})
    core_cpp = build / 'source/runtime/metal/MetalBackend.mm'
    original_core_cpp = core_cpp.read_text()
    getters = ('\nnamespace splash::metal {\n'
        'uint64_t MetalBuffer::oracleChargedBytesSep22() const noexcept {\n'
        '  return impl_ && impl_->allocation ? impl_->allocation->bytes : 0;\n}\n'
        'uintptr_t MetalBuffer::oracleOwnerIdentitySep22() const noexcept {\n'
        '  return impl_ && impl_->allocation ? reinterpret_cast<uintptr_t>(impl_->allocation.get()) : 0;\n}\n}\n')
    core_cpp.write_text(original_core_cpp + getters)
    if not core_cpp.read_text().startswith(original_core_cpp):
        raise SystemExit('readonly getter changed original Core methods')
    forward_h = build / 'source/runtime/flash/FlashForward.hpp'
    original_forward_h = forward_h.read_text()
    anchor = '  friend class FlashBatchForward;'
    if original_forward_h.count(anchor) != 2:
        raise SystemExit('actual Forward class friendship anchor changed')
    before, after = original_forward_h.rsplit(anchor, 1)
    forward_h.write_text(before + '  friend class FlashDeepPrefixOracle;\n' + anchor + after)
    batch_h = build / 'source/runtime/flash/FlashBatchVerify.hpp'
    original_batch_h = batch_h.read_text()
    if original_batch_h.count('private:\n  struct Impl;') != 1:
        raise SystemExit('actual BatchVerify class friendship anchor changed')
    batch_h.write_text(original_batch_h.replace('private:\n  struct Impl;', 'private:\n  friend class FlashDeepPrefixOracle;\n  struct Impl;'))
    for rel, inc in [('runtime/flash/FlashForward.cpp', 'forward_inspection.cpp.inc'),
                     ('runtime/flash/FlashBatchVerify.cpp', 'batch_inspection.cpp.inc')]:
        dst = build / 'source' / rel
        original = dst.read_text()
        dst.write_text('#include "inspect.hpp"\n' + original + '\n' + (private / inc).read_text())
        # The original executed source is retained as one exact substring.
        if original not in dst.read_text():
            raise SystemExit('readonly hook altered an original worker method body')
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
             '-fobjc-arc', '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build / 'source/dev/benchmarks/prefill4k_attention'),
             '-I' + str(private), '-I' + str(build)]
    census = []
    objects = []
    tasks = []
    for entry in host:
        name = entry['object']
        src = build / 'source' / entry['source']
        cmd = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MM', str(src)]
        deps = subprocess.run(cmd, cwd=root, check=True, text=True, capture_output=True).stdout.replace('\\\n', ' ').split()
        if any(t.startswith(('runtime/', 'dev/')) for t in deps):
            raise SystemExit('unsealed live source dependency in ' + name)
        affected = any(t.endswith(('/runtime/flash/FlashForward.hpp', '/runtime/flash/FlashBatchVerify.hpp', '/runtime/metal/MetalBackend.hpp')) for t in deps)
        census.append({'object': name, 'source': entry['source'], 'header_consumer': affected,
                       'excluded_worker_main': name == 'FlashWorker', 'dependencies': deps})
        if name == 'FlashWorker':
            continue
        dst = build / 'objects' / (name + '.o')
        dst.parent.mkdir(exist_ok=True)
        inherited = worker / 'host' / (name + '.o')
        if not inherited.is_file():
            raise SystemExit('sealed host object missing: ' + name)
        record = {'path': str(dst), 'inherited_path': str(inherited), 'inherited_sha256': sha(inherited),
                  'recompiled_header_consumer': affected}
        objects.append(record)
        if affected:
            tasks.append((src, dst))
        else:
            shutil.copyfile(inherited, dst)
    def compile_one(task):
        src, dst = task
        subprocess.run(['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MMD', '-MP', '-c', str(src), '-o', str(dst)], cwd=root, check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(compile_one, tasks))
    link = (worker / 'link-inputs.mk').read_text()
    match = re.search(r'^CORE := (.*)$', link, re.MULTILINE)
    if not match:
        raise SystemExit('sealed original Core closure missing')
    core_paths = [Path(x.replace('$(BUILD)', str(worker))) for x in match.group(1).split()]
    if len(core_paths) != 4:
        raise SystemExit('exact original four-Core closure required')
    core_census = []
    for inherited in core_paths:
        dst = build / 'objects' / inherited.name
        name = inherited.stem.split('-', 1)[-1]
        src = build / 'source' / core_sources[name]
        deps = subprocess.run(['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MM', str(src)], cwd=root, check=True, text=True, capture_output=True).stdout.replace('\\\n', ' ').split()
        affected = name == 'MetalBackend' or any(t.endswith('/runtime/metal/MetalBackend.hpp') for t in deps)
        core_census.append({'object': name, 'source': core_sources[name], 'header_consumer': affected, 'dependencies': deps})
        if affected:
            compile_one((src, dst))
        else:
            shutil.copyfile(inherited, dst)
        objects.append({'path': str(dst), 'inherited_path': str(inherited), 'inherited_sha256': sha(inherited), 'recompiled_header_consumer': affected})
    if len(objects) != 53 or len(census) != 50:
        raise SystemExit('exact 53-nonworker/50-source closure not achieved')
    shutil.copyfile(worker / 'splash.metallib', build / 'splash.metallib')
    headers = [{'path': str(path), 'sha256': sha(path)} for path in [forward_h, batch_h, metal_h, core_cpp, *[private / n for n in names if n != 'build.py']]]
    provenance = {'schema': 'batchverify-private-max16-readonly-hook-closure-v1', 'worker': str(worker),
                  'worker_source_identity_sha256': expected_identity, 'metallib_sha256': artifacts['splash.metallib'],
                  'header_census': census, 'objects': objects, 'headers': headers,
                  'core_header_census': core_census, 'core_source_pins': core_source_pins,
                  'clone_only_charge_getters_return_actual_original_owner_fields': True,
                  'original_Core_method_bodies_literal': True,
                  'original_worker_method_bodies_literal': True, 'new_inspection_only_clone_methods': True,
                  'numeric_policy_changed': False, 'production_edits': False, 'gpu_executed': False,
                  'model_or_input_or_runtime_payload_reads': 0, 'Root_new_artifact_pin_required': True}
    (build / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summary = {k: v for k, v in provenance.items() if k not in ['objects', 'header_census', 'headers', 'core_header_census']}
    (build / 'BatchVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='
        + json.dumps(str(build / 'provenance.json')) + ';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('
        + json.dumps(summary, sort_keys=True) + ')META";\n')
    commands = []
    for role in ['control', 'candidate']:
        cmd = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-Wno-unused-function',
               '-DSPLASH_VERIFY_CANDIDATE=' + str(int(role == 'candidate')), str(private / 'oracle.mm'),
               *[x['path'] for x in objects], '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit',
               '-o', str(build / ('oracle-' + role))]
        subprocess.run(cmd, cwd=root, check=True)
        commands.append(cmd)
        cpu = subprocess.run([str(build / ('oracle-' + role)), '--cpu-only'], cwd=root, check=True, text=True, capture_output=True)
        (build / ('cpu-self-test-' + role + '.json')).write_text(cpu.stdout)
        provenance['cpu_' + role] = json.loads(cpu.stdout)
    provenance.update({'pass': True, 'compiler_commands': commands, 'full_model_state_qualified': False,
                       'rebuilt_header_consumers': [x['object'] for x in census if x['header_consumer'] and not x['excluded_worker_main']]})
    (build / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for obj in objects:
        obj['sha256'] = sha(Path(obj['path']))
    provenance['artifact_sha256'] = {name: sha(build / name) for name in ['oracle-control', 'oracle-candidate', 'splash.metallib']}
    (build / 'CPU_READY.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'pass': True, 'build': str(build), 'object_census': len(objects),
                      'rebuilt_header_consumers': provenance['rebuilt_header_consumers'],
                      'GPU_started': False, 'payload_reads': 0, 'Root_artifact_pin_required': True}))


if __name__ == '__main__':
    main()
