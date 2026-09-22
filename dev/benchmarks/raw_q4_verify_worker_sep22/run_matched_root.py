#!/usr/bin/env python3
"""Execute a frozen exact-runtime matched command after actual state proof."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3 or sha(sys.argv[1]) != sys.argv[2]:
        raise RuntimeError('Externally preregistered matched command digest differs')
    command = json.loads(Path(sys.argv[1]).read_text())
    if command['schema'] != 'Root-matched-rawQ4-GDN26-model-command-v1':
        raise RuntimeError('Unknown matched command')
    for path, expected in command['pins'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen program/runtime/actual-receipt drift: ' + path)
    worker = Path(command['worker'])
    parent_path = Path(command['parent_command'])
    if sha(parent_path) != command['parent_command_sha256']:
        raise RuntimeError('Qualified parent controls changed')
    parent = json.loads(parent_path.read_text())
    expected = list(parent['argv'])
    expected[2] = str(worker / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py')
    expected[expected.index('--binary') + 1] = str(worker / 'splash-flash')
    expected[expected.index('--output') + 1] = command['report']
    expected[expected.index('--port') + 1] = '8046'
    expected.extend(['--env', 'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22=' + command['flag']])
    if expected != command['argv'] or command['flag'] not in ('0', '1'):
        raise RuntimeError('Matched command differs from original controls and sole selector delta')
    if sha(worker / 'Root-rawQ4-native-qualified.json') != command['actual_full_native_receipt_sha256']:
        raise RuntimeError('Actual complete native proof receipt changed')
    report = Path(command['report'])
    for path in (report, report.with_name(report.stem + '-3.semantic.json'),
                 report.with_name(report.stem + '-3.server.log')):
        if path.exists():
            raise RuntimeError('Fresh matched report/semantic/log required')
    helper = worker / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py'
    spec = importlib.util.spec_from_file_location('root_matched_rawQ4', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load(worker, expected=command['flag'] == '1', require_state=True)
    return subprocess.run(expected, cwd=command['cwd']).returncode


if __name__ == '__main__':
    raise SystemExit(main())
