#!/usr/bin/env python3
"""Prepare Root commands; this script never starts an oracle or reads inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--build', required=True)
parser.add_argument('--role', choices=['export', 'compare'], required=True)
parser.add_argument('--report', help='fresh Root report path, relative to repo')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[2]
build = (root / args.build).resolve()
manifest_path = build / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
if manifest['role'] != ('phase' if args.role == 'compare' else 'teacher'):
    raise SystemExit('oracle manifest/command role mismatch')
environment = json.loads((root / 'build/prefill-i8-decode-q4-teacher-sep22-worker-v3/root-mtp-environment.json').read_text())
worker_environment = Path(manifest['worker']) / 'root-mtp-environment.json'
if manifest['role'] == 'phase' and worker_environment.exists():
    environment = json.loads(worker_environment.read_text())
environment['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21'] = '1' if args.role == 'compare' else '0'
environment['SPLASH_FLASH_ALLROWS_FULL512_TARGET'] = '0' if args.role == 'compare' else '1'
if args.role == 'export':
    environment['SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT'] = '0'
    environment['SPLASH_FLASH_DENSE_W8A8_RESIDENCY_PRUNE_SEP21'] = '0'
report = (root / args.report).resolve() if args.report else root / f'build/release/flash/sep22-trunkpref-{args.role}-v1.json'
export_report = root / 'build/release/flash/sep22-trunkpref-export-v1.json'
spill = root / 'build/release/flash/sep22-trunkpref-export-v1'
argv = [str(build / 'oracle'), '--gpu', args.role, str(build / 'splash.metallib'),
        str(root / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
        str(root / 'build/release/flash/prefill4k-fixture/code2048.tokens.json'),
        str(report), str(spill)]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name, key in [('oracle', 'oracle_sha256'), ('splash.metallib', 'metallib_sha256')]:
    if sha(build / name) != manifest[key]:
        raise SystemExit('compiled oracle artifact drift: ' + name)
runner = build / 'run_root.py'
runner_source = (here / 'run_root.py').read_text()
assert runner_source.count('TRUSTED_REPORT = None') == 1
assert runner_source.count('TRUSTED_ROLE = None') == 1
runner.write_text(runner_source.replace('TRUSTED_REPORT = None', 'TRUSTED_REPORT = ' + repr(str(report)))
                  .replace('TRUSTED_ROLE = None', 'TRUSTED_ROLE = ' + repr(args.role)))
pins = [{'path': str(p), 'sha256': sha(p)} for p in
        [build / 'oracle', build / 'splash.metallib', manifest_path,
         build / 'provenance.json', build / 'PrefillExactBuildProvenance.hpp']]
command = {'schema': 'trunkpref-root-command-v1', 'role': args.role,
           'Root_GPU_only': True, 'cwd': str(root), 'argv': argv,
           'environment': environment, 'artifact_pins': pins,
           'export_report': str(export_report), 'runner_sha256': sha(runner),
           'inputs_read_or_hashed_by_preparation': False,
           'port_or_worker_frames_used': False, 'scope': 'TRUNKPREF only'}
path = build / f'root-{args.role}-command.json'
path.write_text(json.dumps(command, indent=2) + '\n')
script = build / f'run-root-{args.role}.sh'
script.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(root / '.venv/bin/python'))
                  + ' -B ' + shlex.quote(str(runner)) + ' ' + shlex.quote(str(path)) + '\n')
script.chmod(0o755)
seal = {'pass': True, 'GPU_execution_started': False,
        'scope': 'TRUNKPREF only', 'command_sha256': sha(path),
        'script_sha256': sha(script), 'runner_sha256': sha(runner),
        'oracle_sha256': manifest['oracle_sha256'],
        'metallib_sha256': manifest['metallib_sha256'],
        'worker_cpu_seal_sha256': manifest['worker_cpu_seal_sha256']}
(build / 'ready-command-seal.json').write_text(json.dumps(seal, indent=2) + '\n')
print(json.dumps({'pass': True, 'Root_command': str(script), **seal}))
