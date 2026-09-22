#!/usr/bin/env python3
"""Registered COMPOSITE metadata; original HC qualification runners untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--original-command',required=True);p.add_argument('--oracle-sha256',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];build=(root/a.build).resolve();m=json.loads((build/'manifest.json').read_text());old=json.loads((root/a.original_command).read_text())
if m.get('variant')!='composite-bundle' or m['role']!='candidate' or old['role']!='compare':raise SystemExit('registered Composite variant/control command required')
c=dict(old);c['environment']=dict(old['environment']);c['schema']='trunkverify-registered-composite-root-command-v1';c['registered_composite']=True;c['scope']='COMPACT1 HC1 BUNDLE1 REGISTRATION1 real26frames54repeats plus8actual metadata guards'
c['environment']['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22']='1';c['environment']['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']='1';c['environment']['SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22']='1';c['environment']['SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22']='1'
c['argv']=list(old['argv']);c['argv'][0]=str(build/'oracle');c['argv'][3]=str(build/'splash.metallib');report=root/'build/release/flash/sep22-trunkverify-composite-bundle-hc-fast-compare-v1.json';c['argv'][6]=str(report)
for suffix in ['', '.failure.json', '.partial', '.writing']:
 if Path(str(report)+suffix).exists():raise SystemExit('fresh guard report required')
c['oracle_sha256']=a.oracle_sha256;c['metallib_sha256']=m['metallib_sha256'];c['original_control_command']=str((root/a.original_command).resolve())
runner=build/'run_composite_bundle_root.py';shutil.copyfile(Path(__file__).with_name('run_composite_bundle_root.py'),runner)
c['runner_source_sha256']=hashlib.sha256(runner.read_bytes()).hexdigest()
path=build/'root-composite-compare-command.json';path.write_text(json.dumps(c,indent=2)+'\n');sha=hashlib.sha256(path.read_bytes()).hexdigest()
script=build/'run-root-composite-compare.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.quote(str(root/'.venv/bin/python'))+' -B '+shlex.quote(str(runner))+' '+shlex.quote(str(path))+' '+shlex.quote(sha)+'\n');script.chmod(0o755)
print(json.dumps({'prepared':str(script),'command_metadata_sha256':sha,'GPU_started':False,'input_or_export_payload_reads':0}))
