#!/usr/bin/env python3
"""Prepare a Root-only GPU invocation; never execute it or inspect inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

parser = argparse.ArgumentParser()
parser.add_argument('--build', required=True)
parser.add_argument('--report', default='build/release/flash/sep22-hc-pad-producer-r4-97chain-v1.json')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[2]
build = (root / args.build).resolve()
manifest = json.loads((build / 'manifest.json').read_text())
environment = {'SPLASH_FLASH_ALLROWS_FULL512_TARGET': '1',
               'SPLASH_FLASH_INT8_EXPERT_STORE': str(root / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1'),
               'SPLASH_FLASH_PLE_SSD_STREAMING': '1',
               'SPLASH_FLASH_OPERAND_STORE': str(root / 'install/local-models/Flash-Next-operands-v1'),
               'SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT': '0',
               'SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE': '0',
               'HC_PAD_PAIRS': '10'}
argv = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'),
        str(root / 'install/local-models/Flash-Next-oQ4e-mtp-v1'), str((root / args.report).resolve())]
runner = build / 'run_root.py'
runner.write_text('''import hashlib,json,os,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]);command=json.loads(p.read_text())
for entry in command['pins']:
 if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
  raise SystemExit('sealed HC component artifact drift')
env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','HC_PAD_'))}
env.update(command['environment'])
raise SystemExit(subprocess.run(command['argv'],cwd=command['cwd'],env=env).returncode)
''')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
pins = []
for name, expected in [('oracle', manifest['oracle_sha256']), ('splash.metallib', manifest['metallib_sha256'])]:
    if sha(build / name) != expected:
        raise SystemExit('compiled artifact drift')
    pins.append({'path': str(build / name), 'sha256': expected})
pins += [{'path': str(path), 'sha256': sha(path)} for path in [build / 'manifest.json', runner]]
command = {'schema': 'hc-pad-producer-root-command-v1', 'Root_GPU_only': True,
           'cwd': str(root), 'argv': argv, 'environment': environment, 'pins': pins,
           'prefill_AR_head_batch_changed': False, 'coefficient_workspace_growth_bytes': 0,
           'input_files_or_model_payloads_read_during_preparation': False}
path = build / 'root-command.json'
path.write_text(json.dumps(command, indent=2) + '\n')
script = build / 'run-root.sh'
script.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(root / '.venv/bin/python'))
                  + ' -B ' + shlex.quote(str(runner)) + ' ' + shlex.quote(str(path)) + '\n')
script.chmod(0o755)
seal = {'pass': True, 'Root_GPU_not_run': True, 'oracle_sha256': manifest['oracle_sha256'],
        'metallib_sha256': manifest['metallib_sha256'], 'command_sha256': sha(path),
        'script_sha256': sha(script), 'runner_sha256': sha(runner)}
(build / 'ready-command-seal.json').write_text(json.dumps(seal, indent=2) + '\n')
print(json.dumps({'Root_command': str(script), **seal}))
