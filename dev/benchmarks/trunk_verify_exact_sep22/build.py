#!/usr/bin/env python3
"""CPU prepare only. Copies sealed worker closure, adds readonly oracle hook.
No model, fixture or export data is opened. Only metadata/header digests are
computed here; inherited artifact digests are copied from the worker CPU seal.
Root must independently pin the newly linked oracle before GPU execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def header_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stem(path):
    return re.sub(r'^(?:\d+-)+', '', Path(path).stem)


def source_for(worker, obj):
    deps = (worker / obj).with_suffix('.d')
    if deps.exists():
        tokens = deps.read_text().replace('\\\n', ' ').split()
        for token in tokens[1:]:
            if token.endswith(('.cpp', '.mm')):
                path = Path(token)
                if not path.is_absolute():
                    path = Path.cwd() / path
                if path.exists() and worker / 'source' in path.resolve().parents:
                    return path.resolve().relative_to(worker / 'source')
    name = stem(obj)
    for suffix in ['.cpp', '.mm']:
        path = Path('runtime/flash') / (name + suffix)
        if (worker / 'source' / path).exists():
            return path
    special = {
        'Prefill4kQSABulk': 'dev/benchmarks/prefill4k_attention/bulk.cpp',
        'Prefill4kQSACoalesced': 'dev/benchmarks/prefill4k_attention/coalesced.cpp',
        'teacher_bulk': 'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp',
        'twopass': 'dev/benchmarks/prefill_qsa_twopass_sep21/twopass.cpp',
        'worker_cache': 'dev/benchmarks/dense_w8a8_sep21/worker_cache.cpp',
    }
    if name in special and (worker / 'source' / special[name]).exists():
        return Path(special[name])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', required=True)
    p.add_argument('--build', required=True)
    p.add_argument('--role', choices=['control', 'candidate'], required=True)
    p.add_argument('--variant',choices=['standard','guard-bundle','composite-bundle','raw-q4q5'],default='standard')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    worker, build = (root / a.worker).resolve(), (root / a.build).resolve()
    if build.exists():
        raise SystemExit('fresh oracle output required')
    if (worker/'compiled-cpu-seal.json').exists():
        seal=json.loads((worker/'compiled-cpu-seal.json').read_text())
    else:
        witness=json.loads((worker/'cpu-source-witness.json').read_text());ready=json.loads((worker/'READY.json').read_text());overlay=json.loads((worker/'overlay-manifest.json').read_text())
        if not witness['pass'] or not ready['ready_for_Root_guard_state_qualification']:
            raise SystemExit('registered Guard READY/source witness required')
        artifacts={x['path']:x['sha256'] for x in witness['compiled_objects']+witness['artifacts']}
        artifacts.update({x['path']:x['sha256'] for x in overlay['frozen_inputs'] if x['path'].endswith('.o')})
        seal={'pass':True,'gpu_executed':False,'artifact_sha256':artifacts}

    if not seal['pass'] or seal.get('gpu_executed',seal.get('GPU_work',False)):
        raise SystemExit('CPU sealed worker required')
    artifact_digests = seal.get('artifact_sha256') or {x['path']: x['sha256'] for x in seal.get('compiled_objects',[])+seal['artifacts']}
    if seal.get('compiled_objects'):
        overlay=json.loads((worker/'overlay-manifest.json').read_text())
        artifact_digests.update({x['path']:x['sha256'] for x in overlay.get('frozen_inputs',[]) if x['path'].endswith('.o')})
    inherited = [(path, digest) for path, digest in artifact_digests.items()
                 if path.endswith('.o') and stem(path) != 'FlashWorker']
    names = [stem(path) for path, _ in inherited]
    if len(inherited) != 53 or len(set(names)) != 53 or 'FlashForward' not in names:
        raise SystemExit('exact 53 nonworker closure required')
    build.mkdir(parents=True)
    shutil.copytree(worker / 'source', build / 'source')
    private = build / 'source/dev/benchmarks/trunk_verify_exact_sep22'
    private.mkdir(parents=True, exist_ok=True)
    for name in ['inspect.hpp', 'inspection.cpp.inc', 'oracle.mm', 'build.py']:
        shutil.copyfile(here / name, private / name)
    if a.variant in ['guard-bundle','composite-bundle','raw-q4q5']:
        variant_source={'composite-bundle':'composite_bundle_oracle.mm','guard-bundle':'guard_bundle_oracle.mm','raw-q4q5':'raw_q4q5_oracle.mm'}[a.variant]
        shutil.copyfile(here/variant_source,private/'oracle.mm')
        inspector=private/'inspect.hpp'
        text=inspector.read_text().replace('  static bool rawOwns(const FlashForward &target,const FlashRequestState &request);','  static bool rawOwns(const FlashForward &target,const FlashRequestState &request);\n  static uint32_t actualBundleGuardProbes(FlashForward &target);')
        inspector.write_text(text)
        with (private/'inspection.cpp.inc').open('a') as stream:stream.write('\n'+(here/'guard_bundle_inspection.cpp.inc').read_text())
    forward_h = build / 'source/runtime/flash/FlashForward.hpp'
    h = forward_h.read_text()
    anchor = '  friend class FlashBatchForward;'
    if h.count(anchor) != 2:
        raise SystemExit('private friend anchor drift')
    before, after = h.rsplit(anchor, 1)
    forward_h.write_text(before + '  friend class FlashDeepPrefixOracle;\n' + anchor + after)
    forward = build / 'source/runtime/flash/FlashForward.cpp'
    forward.write_text('#include "inspect.hpp"\n' + forward.read_text()
                       + '\n' + (private / 'inspection.cpp.inc').read_text())
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror',
             '-Wno-deprecated-declarations', '-fobjc-arc',
             '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-DSPLASH_VERIFY_CANDIDATE=' + str(int(a.role == 'candidate')),
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build / 'source/dev/benchmarks/prefill4k_attention'),
             '-I' + str(private), '-I' + str(build)]
    census, objects = [], []
    for index, (path, digest) in enumerate(inherited):
        src = source_for(worker, path)
        changed_header = False
        if src is None and stem(path) not in {'MetalBackend','DeviceCapabilities','Protocol','MemoryGovernor'}:
            raise SystemExit('source census missing noncore TU: ' + stem(path))
        if src is not None:
            command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MM', str(build / 'source' / src)]
            result = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True)
            tokens = result.stdout.replace('\\\n', ' ').split()
            changed_header = any(t.endswith('/runtime/flash/FlashForward.hpp') for t in tokens)
            live = [t for t in tokens if t.startswith(('runtime/', 'dev/'))]
            if live:
                raise SystemExit('live source dependency: ' + str(live))
            census.append({'object': stem(path), 'source': str(src),
                           'modified_header_consumer': changed_header, 'dependencies': tokens})
        dst = build / 'objects' / f'{index:03d}-{stem(path)}.o'
        dst.parent.mkdir(exist_ok=True)
        if changed_header:
            subprocess.run(['xcrun', '-sdk', 'macosx', 'clang++', *flags,
                            '-MMD', '-MP', '-c', str(build / 'source' / src), '-o', str(dst)], cwd=root, check=True)
        else:
            shutil.copyfile(worker / path, dst)
        objects.append({'path': str(dst), 'inherited_path': str(worker / path),
                        'inherited_sha256': digest, 'recompiled_for_header': changed_header})
    worker_source = build / 'source/runtime/flash/FlashWorker.mm'
    result = subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(worker_source)],cwd=root,check=True,text=True,capture_output=True)
    tokens = result.stdout.replace('\\\n',' ').split()
    census.append({'object':'FlashWorker','source':'runtime/flash/FlashWorker.mm','excluded_main':True,
                   'modified_header_consumer':any(t.endswith('/runtime/flash/FlashForward.hpp') for t in tokens),'dependencies':tokens})
    if len(census)!=50:
        raise SystemExit('expected actual 50 source TU census')
    if not any(x['object'] == 'FlashForward' and x['modified_header_consumer'] for x in census):
        raise SystemExit('readonly Forward hook was not compiled')
    shutil.copyfile(worker / 'splash.metallib', build / 'splash.metallib')
    headers = [{'path': str(path), 'sha256': header_sha(path)} for path in
               [forward_h, private / 'inspect.hpp', private / 'inspection.cpp.inc']]
    expected_lib = artifact_digests.get('splash.metallib')
    if not expected_lib:
        raise SystemExit('worker library metadata pin absent')
    provenance = {'schema': 'trunkverify-private-readonly-hook-closure-v1',
                  'role': a.role, 'variant':a.variant,'worker': str(worker), 'metallib_sha256': expected_lib,
                  'objects': objects, 'header_census': census, 'headers': headers,
                  'runtime_worker_math_changed': False, 'production_edits': False,
                  'timing_bytes': 200, 'new_oracle_artifact_Root_pin_required': True,
                  'source_count': sum(p.is_file() for p in (build / 'source').rglob('*'))}
    (build / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summary = {k: v for k, v in provenance.items() if k not in ['objects', 'header_census', 'headers']}
    (build / 'TrunkVerifyBuildProvenance.hpp').write_text(
        '#pragma once\ninline constexpr const char *kPrefillExactProvenancePath=' + json.dumps(str(build / 'provenance.json'))
        + ';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META(' + json.dumps(summary, sort_keys=True) + ')META";\n')
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, str(private / 'oracle.mm'),
               *[x['path'] for x in objects], '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit', '-o', str(build / 'oracle')]
    (build / 'compiler-command.json').write_text(json.dumps(command, indent=2) + '\n')
    subprocess.run(command, cwd=root, check=True)
    cpu = subprocess.run([str(build / 'oracle'), '--cpu-only'], cwd=root, check=True, text=True, capture_output=True)
    (build / 'cpu-self-test.json').write_text(cpu.stdout)
    provenance.update({'pass': True, 'gpu_executed': False, 'model_payload_reads': 0,
                       'input_or_export_payload_reads': 0, 'payload_hashes_by_preparation': 0,
                       'cpu': json.loads(cpu.stdout), 'compiler_command': command,
                       'whole_trunk_verify_qualified': False})
    (build / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'pass': True, 'build': str(build), 'role': a.role,
                      'rebuilt_header_consumers': [x['object'] for x in census if x['modified_header_consumer']],
                      'object_census': len(objects), 'cpu': json.loads(cpu.stdout),
                      'GPU_started': False, 'Root_artifact_pin_still_required': True}))


if __name__ == '__main__':
    main()
