#!/usr/bin/env python3
"""Prepare metadata-only Root entry points; never execute GPU or read inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

p=argparse.ArgumentParser()
p.add_argument('--build',required=True)
p.add_argument('--role',choices=['export','compare'],required=True)
p.add_argument('--environment',required=True,help='existing worker environment metadata')
p.add_argument('--report',required=True)
p.add_argument('--spill',required=True)
p.add_argument('--control-report')
p.add_argument('--root-oracle-sha256',required=True,help='Root independently supplied new oracle artifact pin')
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
b=(root/a.build).resolve();m=json.loads((b/'manifest.json').read_text())
if m['role']!=('candidate' if a.role=='compare' else 'control'):raise SystemExit('manifest/role mismatch')
env=json.loads((root/a.environment).read_text())
for n in ['SPLASH_FLASH_BATCH','SPLASH_FLASH_BATCH_MTP','SPLASH_FLASH_BATCH_MTP_PREFILL','SPLASH_FLASH_BATCH_PREFILL','SPLASH_FLASH_MTP','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY','SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT','SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE']:
 env[n]='0'
env['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21']='0'
env['SPLASH_FLASH_ALLROWS_FULL512_TARGET']='1'
for n in ['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']:env[n]='1' if a.role=='compare' else '0'
report=(root/a.report).resolve();spill=(root/a.spill).resolve()
for suffix in ['', '.partial', '.failure.json', '.writing']:
 if Path(str(report)+suffix).exists():raise SystemExit('fresh report path required')
command={'schema':'trunkverify-root-command-v1','role':a.role,'Root_GPU_only':True,
 'argv':[str(b/'oracle'),'--gpu',a.role,str(b/'splash.metallib'),str(root/'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(root/'build/release/flash/prefill4k-fixture/code2048.tokens.json'),str(report),str(spill)],
 'cwd':str(root),'environment':env,'control_report':str((root/a.control_report).resolve()) if a.control_report else None,
 'oracle_sha256':a.root_oracle_sha256,'metallib_sha256':m['metallib_sha256'],
 'inputs_or_export_payload_read_or_hashed':False,'scope':'TRUNKVerify/future bounded matrix; no trained head/service claim'}
path=b/('root-'+a.role+'-command.json');path.write_text(json.dumps(command,indent=2)+'\n')
sha=hashlib.sha256(path.read_bytes()).hexdigest()
script=b/('run-root-'+a.role+'.sh')
runner=root/'dev/benchmarks/trunk_verify_exact_sep22/run_root.py'
script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.quote(str(root/'.venv/bin/python'))+' -B '+shlex.quote(str(runner))+' '+shlex.quote(str(path))+' '+shlex.quote(sha)+'\n');script.chmod(0o755)
print(json.dumps({'prepared':str(script),'command_metadata_sha256':sha,'GPU_started':False,'payload_reads':0}))
