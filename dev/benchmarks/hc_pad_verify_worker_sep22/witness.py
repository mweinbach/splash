#!/usr/bin/env python3
"""Authenticate combined source/compiler closure without opening a model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from overlay import transform


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = (root / args.build).resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    base = Path(manifest['base'])
    checks = []
    def check(ok, label):
        if not ok:
            raise ValueError(label)
        checks.append(label)
    for entry in manifest['files']:
        check(sha(build / 'source' / entry['path']) == entry['sha256'], 'source:' + entry['path'])
    for entry in manifest['frozen_inputs']:
        check(sha(build / entry['path']) == entry['sha256'], 'frozen:' + entry['path'])
    check(sha(build / 'link-inputs.mk') == manifest['link_inputs_sha256'], 'link input declaration')
    for relative in ['runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm']:
        check(transform(relative, (base / 'source' / relative).read_text()) == (build / 'source' / relative).read_text(), 'exact transform:' + relative)
    for path in sorted((base / 'source/runtime/flash').glob('*.hpp')):
        check(path.read_bytes() == (build / 'source/runtime/flash' / path.name).read_bytes(), 'public/internal header unchanged:' + path.name)
    for name in ['FlashMTP.cpp', 'FlashBatchForward.cpp', 'FlashBatchVerify.cpp', 'FlashBatchPrefill.cpp', 'FlashBatchMTPForward.cpp']:
        check((base / 'source/runtime/flash' / name).read_bytes() == (build / 'source/runtime/flash' / name).read_bytes(), 'excluded source unchanged:' + name)
    forward = (build / 'source/runtime/flash/FlashForward.cpp').read_text()
    worker = (build / 'source/runtime/flash/FlashWorker.mm').read_text()
    check('eligible(verification,rows,maximumRows)' in forward, 'VerifyR4-only predicate')
    check(forward.count(', normalizedReady, verification);') == 3, 'three explicit target context calls')
    check('checkHC("language_model.model.hyper_connection_mixer",false)' in forward and 'layer<48' in forward, '97-role startup preflight')
    identity_begin = worker.index('<< R"(,"identity":{"source":)"')
    identity_end = worker.index('<< R"(,"persisted_operands":', identity_begin)
    identity = worker[identity_begin:identity_end]
    check('hc_pad_verify_r4_enabled' in identity and 'hc_pad_verify_r4_policy' in identity, 'HC static identity fields')
    check('hc_pad_verify_sep22::graphCalls' not in identity and 'savedPaddingDispatches' not in identity, 'changing counters outside identity')
    counter_offset = worker.index('hc_pad_verify_r4_route_counters')
    check(counter_offset < identity_begin or counter_offset >= identity_end, 'top-level graph counter placement')
    check('hc_pad_verify_sep22::validateDependencies' in worker[:worker.index('const auto package') if 'const auto package' in worker else len(worker)], 'early strict dependency validation')
    hosts = sorted((build / 'host').glob('*.o'))
    check(len(hosts) == 50, 'all50 noncoreTUs compiled')
    check(len(set(re.sub(r'^\d+-', '', path.stem) for path in hosts)) == 50, 'unique host identities')
    deps = sorted((build / 'host').glob('*.d'))
    check(len(deps) == 50, 'all50 dependency files')
    for path in deps:
        text = path.read_text()
        check(str(root / 'runtime') not in text and re.search(r'(?<!source/)\bruntime/flash/', text) is None, 'sealed header dependencies:' + path.name)
    results = []
    for executable, modes in [('hc-policy-cpu', [None]), ('compact-policy-cpu', [None, '--freeze0', '--freeze1', '--missing', '--retry0', '--retry1']), ('splash-flash', ['--cpu-self-test'])]:
        for mode in modes:
            command = [str(build / executable)] + ([mode] if mode else [])
            result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
            results.append({'command': command, 'result': json.loads(result.stdout)})
    artifacts = [{'path': str(path.relative_to(build)), 'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in [*hosts, *sorted((build / 'reused/core').glob('*.o')), build / 'hc-pad.air',
                              build / 'splash-flash', build / 'splash.metallib', build / 'hc-policy-cpu', build / 'compact-policy-cpu']]
    seal = {'schema': 'hc-pad-compact-verify-worker-cpu-seal-v1', 'pass': True,
            'source_manifest_sha256': sha(build / 'overlay-manifest.json'), 'checks': checks,
            'source_count': len(manifest['files']), 'host_tus_rebuilt': 50, 'Core_objects': 4,
            'original_AIRs': 76, 'qualified_compact_AIRs': 1, 'HC_private_AIRs': 1,
            'artifacts': artifacts, 'cpu': results, 'public_headers_changed_by_HC': False,
            'numerical_derivative_changed_by_HC': False, 'GPU_buffers_cache_growth_bytes': 0,
            'gpu_executed': False, 'model_payload_bytes_read': 0, 'Root_whole_worker_qualified': False}
    (build / 'compiled-cpu-seal.json').write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps({key: seal[key] for key in ['pass', 'source_count', 'host_tus_rebuilt', 'gpu_executed', 'model_payload_bytes_read']}))


if __name__ == '__main__':
    main()
