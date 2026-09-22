#!/usr/bin/env python3
"""Prepare exact matched Q4 singleton commands; no tokenizer/model import."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
MODEL = ROOT / 'build/rawQ4-GDN26-matched-model-sep22-root-v1'
PARENT = ROOT / 'build/guard-HC-fast-composite-sep22-model-v1/root-model-command.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--report-version', default='v1')
    args = parser.parse_args()
    model = args.model_dir.resolve()
    model.mkdir(exist_ok=False)
    parent = json.loads(PARENT.read_text())
    programs = [WORKER / 'splash-flash', WORKER / 'splash.metallib',
                WORKER / 'compiled-cpu-seal.json', WORKER / 'overlay-manifest.json',
                WORKER / 'Root-rawQ4-native-qualified.json',
                WORKER / 'rawQ4-qualified.air',
                WORKER / 'rawQ4-semantic-source-seal.json',
                WORKER / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py',
                WORKER / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py',
                WORKER / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/test_semantic_quality.py',
                ROOT / 'dev/benchmarks/raw_q4_verify_worker_sep22/run_matched_root.py']
    for record in parent['artifact_and_helper_pins']:
        path = Path(record['path'])
        if path not in programs:
            programs.append(path)
    pins = {str(path): sha(path) for path in programs}
    for flag in ('0', '1'):
        argv = list(parent['argv'])
        argv[2] = str(WORKER / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py')
        argv[argv.index('--binary') + 1] = str(WORKER / 'splash-flash')
        report = ROOT / f'build/release/flash/sep22-rawQ4-GDN26-matched-flag{flag}-model-and-quality-{args.report_version}.json'
        argv[argv.index('--output') + 1] = str(report)
        argv[argv.index('--port') + 1] = '8046'
        argv.extend(['--env', 'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22=' + flag])
        environment = dict(parent['environment'])
        environment['SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22'] = flag
        command = {'schema': 'Root-matched-rawQ4-GDN26-model-command-v1',
                   'Root_GPU_only': True, 'cwd': str(ROOT), 'worker': str(WORKER),
                   'flag': flag, 'argv': argv, 'environment': environment,
                   'pins': pins, 'report': str(report),
                   'parent_command': str(PARENT), 'parent_command_sha256': sha(PARENT),
                   'sole_selector_delta': 'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22=' + flag,
                   'original22_plan_graders_budgets_unchanged': True,
                   'actual_full_native_receipt_sha256': sha(WORKER / 'Root-rawQ4-native-qualified.json')}
        path = model / f'root-flag{flag}-command.json'
        path.write_text(json.dumps(command, indent=2) + '\n')
        runner = ROOT / 'dev/benchmarks/raw_q4_verify_worker_sep22/run_matched_root.py'
        script = model / f'run-root-flag{flag}.sh'
        script.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join([
            str(ROOT / '.venv/bin/python'), '-B', str(runner), str(path), sha(path)]) + '\n')
        print(json.dumps({'flag': flag, 'script': str(script), 'command_sha256': sha(path)}))


if __name__ == '__main__':
    main()
