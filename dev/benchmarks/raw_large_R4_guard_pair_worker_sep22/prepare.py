#!/usr/bin/env python3
"""Private CPU Worker composition; program artifacts only, no model payloads."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, re, shutil, subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIVATE = 'dev/benchmarks/raw_large_R4_guard_pair_worker_sep22'
BASE = ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
COMPONENT = ROOT / 'build/raw-large-R4-guard-pair-sep22-component-v1'
PARENT_EXE = '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
PARENT_LIB = '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    if build.exists() or ROOT / 'build' not in build.parents:
        raise ValueError('Fresh private Worker build required')
    if sha(BASE / 'splash-flash') != PARENT_EXE or sha(BASE / 'splash.metallib') != PARENT_LIB:
        raise ValueError('Registered current parent changed')
    seal = json.loads((BASE / 'compiled-cpu-seal.json').read_text())
    parent = json.loads((BASE / 'overlay-manifest.json').read_text())
    kernel = json.loads((COMPONENT / 'build-manifest.json').read_text())
    if seal.get('pass') is not True or json.loads((COMPONENT / 'strict-SSA-audit.json').read_text()).get('pass') is not True:
        raise ValueError('Current compiled closures must pass')
    for item in seal['compiled_objects']:
        if sha(BASE / item['path']) != item['sha256']:
            raise ValueError('Current parent object changed')
    air = COMPONENT / 'candidate.air'
    if sha(air) != kernel['artifacts']['candidate.air']:
        raise ValueError('Sealed R4 shipping AIR changed')
    build.mkdir()
    shutil.copytree(BASE / 'source', build / 'source')
    # Only source, objects and AIR inputs are cloned. Completed Root spill or
    # generation files anywhere else in the parent tree are never accessed.
    link = (BASE / 'link-inputs.mk').read_text()
    (build / 'link-inputs.mk').write_text(link)
    for item in seal['compiled_objects']:
        destination = build / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / item['path'], destination)
    own = build / 'source' / PRIVATE
    shutil.copytree(HERE, own, ignore=shutil.ignore_patterns('__pycache__', 'semantic_quality.py', 'test_semantic_quality.py'))
    spec = importlib.util.spec_from_file_location('rawLargeR4Overlay', HERE / 'overlay.py')
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    changed = ('runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm')
    journal = []
    for name in changed:
        original = (BASE / 'source' / name).read_text()
        (build / 'source' / name).write_text(overlay.transform(name, original))
        journal.append({'path': name, 'parent_sha256': sha(BASE / 'source' / name),
                        'sha256': sha(build / 'source' / name)})
    parts = {name: sha(own / name) for name in ('policy.hpp', 'overlay.py', 'policy_cpu.cpp', 'prepare.py')}
    parts.update(parent_source_identity=parent['source_identity_sha256'],
                 parent_compiled_seal_sha256=sha(BASE / 'compiled-cpu-seal.json'),
                 candidate_AIR_sha256=sha(air), component_math_audit_sha256=sha(COMPONENT / 'strict-SSA-audit.json'))
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (own / 'source_identity.hpp').write_text('#pragma once\nnamespace splash::flash::raw_large_r4_guard_pair_sep22 {inline constexpr char kSourceIdentitySha256[]=' + json.dumps(identity) + ';}\n')
    flags = ['-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
             '-fobjc-arc', '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-I' + str(build / 'source'), '-I' + str(build / 'source/runtime'),
             '-I' + str(build / 'source/dev/benchmarks/prefill4k_attention')]
    names = next(line for line in link.splitlines() if line.startswith('REBUILD_NAMES :=')).split(':=', 1)[1].split()
    sources = {match.group(1): match.group(2) for match in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$', link, re.M)}
    cores = [build / token.removeprefix('$(BUILD)/') for token in next(line for line in link.splitlines() if line.startswith('CORE :=')).split(':=', 1)[1].split()]
    objects = [build / 'host' / (name + '.o') for name in names]
    if len(names) != 50 or len(set(names)) != 50 or len(cores) != 4:
        raise ValueError('Actual fifty-host/Core4 closure required')
    commands, census, object_pins = [], [], []
    for name, obj in zip(names, objects):
        source = build / 'source' / sources[name]
        command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MM', str(source)]
        dependencies = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.replace('\\\n', ' ').split()
        consumer = any(token.endswith('/raw_large_R4_guard_pair_worker_sep22/policy.hpp') for token in dependencies)
        if consumer != (name in ('FlashForward', 'FlashWorker')):
            raise ValueError('Unexpected private header consumer: ' + name)
        census.append({'object': name, 'source': str(source), 'consumer': consumer, 'dependencies': dependencies})
        if consumer:
            command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, '-MMD', '-MP', '-c', str(source), '-o', str(obj)]
            commands.append(command)
            run(command)
        elif sha(obj) != sha(BASE / obj.relative_to(build)):
            raise ValueError('Unchanged object drift')
        object_pins.append({'path': str(obj.relative_to(build)), 'sha256': sha(obj), 'parent_sha256': sha(BASE / obj.relative_to(build)), 'changed_for_private_header': consumer})
    for obj in cores:
        if sha(obj) != sha(BASE / obj.relative_to(build)):
            raise ValueError('Core4 drift')
        object_pins.append({'path': str(obj.relative_to(build)), 'sha256': sha(obj), 'parent_sha256': sha(BASE / obj.relative_to(build)), 'changed_for_private_header': False})
    shipping = [command for command in parent['compiler_commands']
                if 'metallib' in command and command[-1] == str(BASE / 'splash.metallib')]
    if len(shipping) != 1:
        raise ValueError('Exact authoritative parent AIR tuple required')
    airs = []
    for token in shipping[0][shipping[0].index('metallib') + 1:-2]:
        original = Path(token)
        destination = build / original.relative_to(BASE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        airs.append(destination)
    command = ['xcrun', '-sdk', 'macosx', 'metallib', *map(str, airs), '-o', str(build / 'baseline-relinked.metallib')]
    commands.append(command)
    run(command)
    # Publish the complete actual shipping tuple for later isolated oracle
    # builders; inherited link-inputs omitted HC/old-Q4 appended AIRs.
    kept = [line for line in link.splitlines() if not line.startswith(('AIRS :=', 'AIRS +='))]
    kept.append('AIRS := ' + ' '.join('$(BUILD)/' + str(path.relative_to(build)) for path in [*airs, build / 'raw-large-R4.air']))
    (build / 'link-inputs.mk').write_text('\n'.join(kept) + '\n')
    if sha(build / 'baseline-relinked.metallib') != PARENT_LIB:
        raise ValueError('Parent AIR relink did not exactly reproduce754')
    shutil.copy2(air, build / 'raw-large-R4.air')
    command = ['xcrun', '-sdk', 'macosx', 'metallib', *map(str, airs), str(build / 'raw-large-R4.air'), '-o', str(build / 'splash.metallib')]
    commands.append(command)
    run(command)
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, *map(str, objects + cores), '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit', '-o', str(build / 'splash-flash')]
    commands.append(command)
    run(command)
    command = ['xcrun', '-sdk', 'macosx', 'clang++', *flags, str(own / 'policy_cpu.cpp'), *map(str, [obj for name, obj in zip(names, objects) if name != 'FlashWorker'] + cores), '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOKit', '-o', str(build / 'policy-CPU')]
    commands.append(command)
    run(command)
    clean_environment = {key: value for key, value in os.environ.items() if not key.startswith(('SPLASH_', 'FLASH_'))}
    cpu = subprocess.run([str(build / 'policy-CPU')], cwd=ROOT, env=clean_environment, check=True, capture_output=True, text=True)
    worker_cpu = subprocess.run([str(build / 'splash-flash'), '--cpu-self-test'], cwd=ROOT, env=clean_environment, check=True, capture_output=True, text=True)
    (build / 'policy-CPU.json').write_text(cpu.stdout)
    (build / 'Worker-CPU.json').write_text(worker_cpu.stdout)
    for source in (BASE / 'source').rglob('*'):
        if source.is_file() and str(source.relative_to(BASE / 'source')) not in changed:
            if sha(source) != sha(build / 'source' / source.relative_to(BASE / 'source')):
                raise ValueError('Unrelated source drift')
    artifacts = [{'path': name, 'sha256': sha(build / name)} for name in ('splash-flash', 'splash.metallib', 'policy-CPU')]
    receipt = {'schema': 'raw-large-R4-private-worker-CPU-v1', 'pass': True,
               'base': str(BASE), 'source_identity_sha256': identity, 'identity_parts': parts,
               'files': [{'path': str(source.relative_to(build / 'source')), 'sha256': sha(source)} for source in sorted((build / 'source').rglob('*')) if source.is_file()],
               'source_hook_journal': journal, 'actual50TU_header_census': census,
               'compiled_objects': object_pins, 'artifacts': artifacts, 'compiler_commands': commands,
               'private_header_consumers': ['FlashForward', 'FlashWorker'],
               'other48_host_and_Core4_unchanged': True, 'public_headers_changed': False,
               'original_ordered_AIR_relink_exact754': True, 'new_shipping_AIR_count': 1,
               'old_qualified_26Q4_precedence_retained': True, 'new_potential_main_roles': 87,
               'new_weight_cache_or_GPU_owner_bytes': 0, 'Metal_math_compiles': 0,
               'CPU_policy': json.loads(cpu.stdout), 'CPU_Worker': json.loads(worker_cpu.stdout),
               'GPU_work': False, 'model_token_tensor_capture_or_generation_payload_reads': 0,
               'actual_trained_state_or_performance_or_Original22_qualified': False}
    (build / 'overlay-manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (build / 'compiled-cpu-seal.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'build': str(build), 'source_identity_sha256': identity,
                      'worker_sha256': sha(build / 'splash-flash'), 'library_sha256': sha(build / 'splash.metallib'), 'GPU_work': False}))


if __name__ == '__main__':
    main()
