#!/usr/bin/env python3
"""Seal a Root command from a freshly compiled and reviewed Host V2 oracle."""
import argparse
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path('/Users/mweinbach/Projects/splash')
HERE = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--build', type=pathlib.Path, required=True)
    parser.add_argument('--review', type=pathlib.Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    if ROOT / 'build' not in build.parents:
        raise ValueError('Private build required')
    ready = json.loads((build / 'CPU_READY.json').read_text())
    if (ready.get('schema') != 'current-R5-large-odd-rowpair-bounded-component-CPU-v2'
            or ready.get('pass') is not True
            or ready.get('host_accounting_scope') != 'owned-zero; measured-process-cache-bounded'
            or ready.get('library_sha256') != '93cef9b26e1feaf83df3462973ed3d1b968472159ae1d2ecf2519d2d36b73821'):
        raise ValueError('Fresh Host V2 and identical qualified numerical program required')
    for item in ready['sources']:
        if sha(build / 'source' / item['path']) != item['sha256']:
            raise ValueError('Compiled source pin changed')
    for item in ready['Core4']:
        if sha(build / item['path']) != item['sha256']:
            raise ValueError('Immutable current Core4 pin changed')
    review = json.loads(args.review.read_text())
    if review.get('pass') is not True:
        raise ValueError('Independent source/compiled review must pass')
    sealed = build / 'Root-synthetic-all7-v2'
    sealed.mkdir()
    for name in ('run_root.py', 'prepare.py', 'test_run_root.py'):
        shutil.copy2(HERE / name, sealed / name)
    report = ROOT / 'build/release/flash/sep22-R5-odd-raw-all7-synthetic-component-v2.json'
    command = {
        'schema': 'Root-R5-odd-raw-bounded-synthetic-all7-command-v2',
        'cwd': str(ROOT), 'build': str(build),
        'argv': [str(build / 'oracle'), '--gpu', str(build / 'component.metallib'),
                 str(ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'), str(report),
                 '--shape', 'all', '--synthetic-cases'],
        'report': str(report), 'log': str(report.with_suffix('.log')),
        'input_scope': ('synthetic thirteen robust patterns for each of seven observed '
                        'shape/quant combinations; NOT current model activations'),
        'Root_only_GUV_bytes': 256 << 20,
        'selected_native_coefficients_per_case_max_bytes': 64 << 20,
        'host_accounting_scope': 'owned-zero; measured-process-cache-bounded',
        'baseline_dispatches': 1, 'candidate_dispatches': 1,
        'minimum_warm_GPU_ms_each': 150, 'balanced_pairs': 18,
        'CPU_READY_sha256': sha(build / 'CPU_READY.json'),
        'Source_CPU_review_sha256': sha(args.review),
        'V1_failed_resource_gate_not_regraded': True,
        'Root_only_model_coefficient_reads': True,
        'actual_GPU_or_current_inputs_qualified': False,
    }
    programs = [build / name for name in ('CPU_READY.json', 'oracle', 'component.metallib',
                'native-qmv.air', 'candidate.air', 'control_probe.air', 'candidate_probe.air',
                'strict-SSA-audit.json', 'build-manifest.json', 'Provenance.hpp',
                'CPU-self-test.json')]
    programs += [build / item['path'] for item in ready['Core4']]
    own = build / 'source/dev/benchmarks/R5_raw_odd_rowpair_sep22'
    programs += [own / name for name in ('oracle.mm', 'policy.hpp', 'storage.hpp', 'prepare.py')]
    programs += [sealed / name for name in ('run_root.py', 'prepare.py', 'test_run_root.py')]
    programs += [args.review.resolve()]
    command['program_pins'] = {str(path): sha(path) for path in programs}
    path = sealed / 'root-command.json'
    path.write_text(json.dumps(command, indent=2) + '\n')
    runner = sealed / 'run-root.sh'
    runner.write_text('#!/bin/zsh\nset -euo pipefail\ncd ' + str(ROOT) + '\n'
                      + str(ROOT / '.venv/bin/python') + ' ' + str(sealed / 'run_root.py')
                      + ' --command ' + str(path) + ' --command-sha256 ' + sha(path)
                      + ' --run-root-gpu\n')
    runner.chmod(0o755)
    print(json.dumps({'command': str(path), 'command_sha256': sha(path),
                      'run_root': str(runner), 'pins': len(command['program_pins']),
                      'GPU_work': False}))


if __name__ == '__main__':
    main()
