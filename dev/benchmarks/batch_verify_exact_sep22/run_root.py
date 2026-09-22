#!/usr/bin/env python3
"""Root-only bounded partition runner. Preparers must not invoke GPU commands."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit('sealed command path and metadata SHA required')
    path = Path(sys.argv[1])
    command = json.loads(path.read_text())
    if sha(path) != sys.argv[2] or not command['Root_GPU_only']:
        raise SystemExit('Root command metadata changed')
    role = command['role']
    if role not in ['export', 'compare'] or command['argv'][1:3] != ['--gpu', role]:
        raise SystemExit('role/argv mismatch')
    if sha(Path(command['argv'][0])) != command['oracle_sha256'] or sha(Path(command['argv'][3])) != command['metallib_sha256']:
        raise SystemExit('Root-pinned artifact changed')
    manifest = json.loads((Path(command['build']) / 'CPU_READY.json').read_text())
    if not manifest['pass'] or manifest['gpu_executed']:
        raise SystemExit('CPU sealed closure required')
    for header in manifest['headers']:
        if sha(Path(header['path'])) != header['sha256']:
            raise SystemExit('private inspection source/header changed')
    for obj in manifest['objects']:
        if sha(Path(obj['path'])) != obj['sha256']:
            raise SystemExit('compiled object closure changed')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SPLASH_FLASH_')}
    env.update(command['environment'])
    if env['SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22'] != str(int(role == 'compare')):
        raise SystemExit('kernel role policy mismatch')
    if env['SPLASH_FLASH_ALLROWS_FULL512_TARGET'] != '1' or env['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21'] != '0':
        raise SystemExit('pure I8 policy required')
    if role == 'compare':
        prior = json.loads(Path(command['control_report']).read_text())
        if not prior['pass'] or prior['role'] != 'export' or not prior['backend_destroyed'] or not prior['partition_complete']:
            raise SystemExit('completed separate-process control export required')
        if prior['common']['selected_checkpoints'] != command['argv'][8].split(','):
            raise SystemExit('selected replay partition differs from control')
    result = subprocess.run(command['argv'], cwd=command['cwd'], env=env)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
