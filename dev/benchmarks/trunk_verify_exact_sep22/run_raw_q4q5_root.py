#!/usr/bin/env python3
"""Root-only execution of the fresh strict raw-Q4/Q5 whole-state comparator."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = '90d42b5ce8edd6c7254b6f9286628a7ffa7a6eede53fa3058eba8b519e9a64d6'
WORKER = '613e4dfe6a9b429cabf8fbd2270b5857dbc601cde6d1b3af10c46fec1b8095c0'
LIBRARY = '8fffe24fd4d99174cc522dcef6090b5b2e72e36f26e5c78afe9df46a118d4246'
FLAGS = (
    'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22',
    'SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22',
    'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22',
    'SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22',
    'SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22',
    'SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(path, expected_digest):
    path = Path(path)
    command = json.loads(path.read_text())
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    if sha(path) != expected_digest or command.get('runner_source_sha256') != sha(__file__):
        raise SystemExit('registered command or runner source drift')
    if command.get('role') != 'compare' or manifest.get('variant') != 'raw-q4q5' or manifest.get('role') != 'candidate':
        raise SystemExit('strict new rawQ4Q5 comparator required')
    if command.get('combined_source_identity_sha256') != SOURCE or command.get('qualified_worker_executable_sha256') != WORKER:
        raise SystemExit('current combined worker identity differs')
    if command.get('metallib_sha256') != LIBRARY or manifest.get('metallib_sha256') != LIBRARY:
        raise SystemExit('current combined library identity differs')
    if command['argv'][1:3] != ['--gpu', 'compare']:
        raise SystemExit('compare only required')
    environment = command['environment']
    if any(environment.get(flag) != '1' for flag in FLAGS):
        raise SystemExit('strict RAWQ4/RAWQ5/COMPACT/HC/BUNDLE/REGISTRATION flags1 required')
    if sha(command['argv'][0]) != command['oracle_sha256'] or sha(command['argv'][3]) != LIBRARY:
        raise SystemExit('current native artifact drift')
    for header in manifest['headers']:
        if sha(header['path']) != header['sha256']:
            raise SystemExit('readonly oracle header drift')
    original = json.loads(Path(command['original_control_command']).read_text())
    expected = dict(original['environment'])
    for flag in FLAGS:
        expected[flag] = '1'
    if expected != environment or any(command['argv'][i] != original['argv'][i] for i in (1, 2, 4, 5, 7)):
        raise SystemExit('original control/input policy changed')
    report_path = Path(command['argv'][6])
    for suffix in ('', '.partial', '.failure.json', '.writing'):
        if Path(str(report_path) + suffix).exists():
            raise SystemExit('fresh rawQ4Q5 report required before native launch')
    return command


def main():
    if len(sys.argv) != 3:
        raise SystemExit('sealed command path and metadata digest required')
    command = validate(sys.argv[1], sys.argv[2])
    # Only Root invokes this entry point and reads actual control metadata.
    control = json.loads(Path(command['control_report']).read_text())
    if not control['pass'] or control['role'] != 'export' or not control['backend_destroyed']:
        raise SystemExit('completed separate-process control required')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('SPLASH_FLASH_')}
    environment.update(command['environment'])
    result = subprocess.run(command['argv'], cwd=command['cwd'], env=environment)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
