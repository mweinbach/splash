#!/usr/bin/env python3
"""Metadata-only command preparation; execution is exclusively Root-owned."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

p = argparse.ArgumentParser()
p.add_argument('--build', required=True)
p.add_argument('--role', choices=['export', 'compare'], required=True)
p.add_argument('--environment-command', required=True, help='existing nativeclock shell command metadata')
p.add_argument('--report', required=True)
p.add_argument('--spill', required=True)
p.add_argument('--checkpoints', required=True)
p.add_argument('--control-report')
p.add_argument('--root-oracle-sha256', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
build = (root / a.build).resolve()
manifest = json.loads((build / 'CPU_READY.json').read_text())
if not manifest['pass'] or manifest['gpu_executed']:
    raise SystemExit('CPU-ready private oracle required')
raw = shlex.split((root / a.environment_command).read_text())
env = {}
for i, item in enumerate(raw[:-1]):
    if item == '--env':
        name, value = raw[i + 1].split('=', 1)
        if not name.startswith('SPLASH_FLASH_'):
            raise SystemExit('unexpected environment name')
        env[name] = value
for name in ['SPLASH_FLASH_BATCH', 'SPLASH_FLASH_BATCH_MTP', 'SPLASH_FLASH_BATCH_MTP_PREFILL',
             'SPLASH_FLASH_BATCH_PREFILL', 'SPLASH_FLASH_MTP', 'SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY',
             'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21', 'SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT',
             'SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE', 'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22',
             'SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22', 'SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21',
             'SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22']:
    env[name] = '0'
env.pop('SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22', None)
env['SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS'] = '0'
env['SPLASH_FLASH_ALLROWS_FULL512_TARGET'] = '1'
env['SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22'] = str(int(a.role == 'compare'))
for name in ['SPLASH_FLASH_ALLROWS_GATHERED_MPP', 'SPLASH_FLASH_BLOCKED_MOE', 'SPLASH_FLASH_MOE_DIRECT_A',
             'SPLASH_FLASH_MOE_Q4X8', 'SPLASH_FLASH_MOE_POINTWISE_SEP21', 'SPLASH_FLASH_GDN_LAZY_ROLLBACK',
             'SPLASH_FLASH_GPU_GREEDY']:
    if env.get(name) != '1':
        raise SystemExit('qualified native dependency flag missing: ' + name)
if env.get('SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS') != '4':
    raise SystemExit('exact gathered cap4 required')
selected = a.checkpoints.split(',')
valid = manifest['cpu_control']['checkpoints']
if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected) or any(x not in valid for x in selected):
    raise SystemExit('one or two known unique full checkpoints required')
binary = 'oracle-candidate' if a.role == 'compare' else 'oracle-control'
if a.root_oracle_sha256 != manifest['artifact_sha256'][binary]:
    raise SystemExit('Root independently supplied oracle pin differs from CPU build')
report, spill = (root / a.report).resolve(), (root / a.spill).resolve()
for suffix in ['', '.partial', '.failure.json', '.writing']:
    if Path(str(report) + suffix).exists():
        raise SystemExit('fresh report required')
if a.role == 'compare' and not a.control_report:
    raise SystemExit('separate completed control report required')
command = {'schema': 'batchverify-bounded-partition-root-command-v1', 'role': a.role,
           'Root_GPU_only': True, 'build': str(build), 'cwd': str(root), 'environment': env,
           'argv': [str(build / binary), '--gpu', a.role, str(build / 'splash.metallib'),
                    str(root / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
                    str(root / 'build/release/flash/prefill4k-fixture/code2048.tokens.json'),
                    str(report), str(spill), a.checkpoints],
           'control_report': str((root / a.control_report).resolve()) if a.control_report else None,
           'oracle_sha256': a.root_oracle_sha256, 'metallib_sha256': manifest['metallib_sha256'],
           'payload_reads_or_hashes_by_preparer': 0, 'spill_limit_bytes': 4 << 30,
           'actual_foreign_trunk_or_head_or_worker_deadline_proof_claim': False}
path = build / ('root-' + a.role + '-' + '-'.join(selected) + '-command.json')
if path.exists():
    raise SystemExit('fresh command metadata path required')
path.write_text(json.dumps(command, indent=2) + '\n')
digest = hashlib.sha256(path.read_bytes()).hexdigest()
runner = root / 'dev/benchmarks/batch_verify_exact_sep22/run_root.py'
script = path.with_suffix('.sh')
script.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(root / '.venv/bin/python')) + ' -B '
                  + shlex.quote(str(runner)) + ' ' + shlex.quote(str(path)) + ' ' + shlex.quote(digest) + '\n')
script.chmod(0o755)
print(json.dumps({'prepared': str(script), 'command_metadata_sha256': digest, 'GPU_started': False, 'payload_reads': 0}))
