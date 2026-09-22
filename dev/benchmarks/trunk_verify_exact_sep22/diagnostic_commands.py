#!/usr/bin/env python3
"""Prepare one-feature diagnostic metadata only; never execute child or inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--original-command',required=True);p.add_argument('--oracle-sha256',required=True);p.add_argument('--report-version',default='v1');p.add_argument('--only-feature',choices=['compact_only','hc_only']);a=p.parse_args()
root=Path(__file__).resolve().parents[3];build=(root/a.build).resolve();original=json.loads((root/a.original_command).read_text());m=json.loads((build/'manifest.json').read_text())
if not m.get('diagnostic_only') or original['role']!='compare':raise SystemExit('diagnostic candidate/original comparator required')
runner=build/'run_diagnostic_root.py';shutil.copyfile(Path(__file__).with_name('run_diagnostic_root.py'),runner)
for feature,flags in [('compact_only',('1','0')),('hc_only',('0','1'))]:
 if a.only_feature and a.only_feature!=feature:continue
 c=dict(original);c['schema']='trunkverify-diagnostic-command-v1';c['role']='diagnostic_compare';c['diagnostic_run']=True;c['qualification_complete']=False;c['diagnostic_feature']=feature;c['environment']=dict(original['environment'])
 c['environment']['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22'],c['environment']['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']=flags
 c['argv']=list(original['argv']);c['argv'][0]=str(build/'oracle');c['argv'][3]=str(build/'splash.metallib');report=root/f'build/release/flash/sep22-trunkverify-diagnostic-{feature}-{a.report_version}.json';c['argv'][6]=str(report)
 for suffix in ['', '.failure.json', '.partial', '.writing']:
  if Path(str(report)+suffix).exists():raise SystemExit('fresh diagnostic report required')
 c['oracle_sha256']=a.oracle_sha256;c['metallib_sha256']=m['metallib_sha256'];c['original_full_candidate_command']=str((root/a.original_command).resolve());c['original_controls_preserved']=True
 path=build/f'root-diagnostic-{feature}-command.json';path.write_text(json.dumps(c,indent=2)+'\n');sha=hashlib.sha256(path.read_bytes()).hexdigest()
 script=build/f'run-root-diagnostic-{feature}.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.quote(str(root/'.venv/bin/python'))+' -B '+shlex.quote(str(runner))+' '+shlex.quote(str(path))+' '+shlex.quote(sha)+'\n');script.chmod(0o755)
 print(json.dumps({'prepared':str(script),'command_metadata_sha256':sha,'same_control_spill':c['argv'][7],'GPU_started':False,'diagnostic_only':True}))
