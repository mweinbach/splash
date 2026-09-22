#!/usr/bin/env python3
"""Register concrete two-process current2K qualifier commands; CPU metadata only."""
from pathlib import Path
import hashlib,json,shlex,shutil
ROOT=Path(__file__).resolve().parents[4]
shipping=ROOT/'build/gemv-r1-teacher-sep22-worker-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=json.loads((shipping/'root-matched-command-seal.json').read_text())
environment={};args=seal['root_commands']['0'];i=0
while i<len(args):
 if args[i]=='--env':k,v=args[i+1].split('=',1);environment[k]=v;i+=2
 else:i+=1
tokens=ROOT/'build/release/flash/prefill4k-fixture/code2048.tokens.json'
if not tokens.exists() or sha(tokens)!='4985e55294b83c72cb9e51e00c40f918460b6c4f560cb5b32d4be3662e540b57':raise SystemExit('canonical2K token fixture path/hash required')
runner=Path(__file__).with_name('run_root.py');spill=shipping/'current2k-prefill-control.bin'
controlreport=ROOT/'build/release/flash/sep22-r1-teacher-current2k-control-proof-v1.json'
for role,build,flag,report in [('control',ROOT/'build/gemv-r1-teacher-current2k-control-sep22-v4','0',controlreport),
 ('candidate',ROOT/'build/gemv-r1-teacher-current2k-candidate-sep22-v3','1',ROOT/'build/release/flash/sep22-r1-teacher-current2k-candidate-proof-v1.json')]:
 m=json.loads((build/'manifest.json').read_text())
 if not m['cpu']['pass'] or m['host_object_count']!=53 or len(m['all50_header_census'])!=50:raise SystemExit('compiled53objects/all50 census qualifier required')
 if report.exists():raise SystemExit('fresh registered proof report required')
 pins={**m['sources'],**{x['path']:x['sha256'] for x in m['objects']},str(build/'manifest.json'):sha(build/'manifest.json'),
  str(build/'oracle'):m['oracle_sha256'],str(build/'splash.metallib'):m['metallib_sha256']}
 pins.update({x['path']:x['sha256'] for x in m['original_shipping_air_pins']})
 local={**environment,'SPLASH_FLASH_GEMV_DECODE_R1_SEP21':flag}
 receipt=shipping/f'current2k-{role}-qualified-receipt.json'
 r={'schema':'registered-TeacherV5-current2K-standard-R1-qualifier-v1','role':role,'cwd':str(ROOT),'environment':local,
  'report':str(report),'receipt':str(receipt),'code_pins':pins,'runner_sha256':sha(runner),
  'shipping_cpu_seal_sha256':sha(shipping/'compiled-cpu-seal.json'),'shipping_worker_sha256':sha(shipping/'splash-flash'),'shipping_metallib_sha256':sha(shipping/'splash.metallib'),
  'argv':[str(build/'oracle'),'--root-gpu',str(build/'splash.metallib'),str(ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(tokens),str(report),str(spill),'export' if role=='control' else 'compare'],
  'control_report':str(controlreport),'one_forward_per_process':True,'maximum_spill_bytes':4<<30,'Root_only':True,'numeric_new_gates':False}
 path=shipping/f'root-current2k-{role}-command.json';path.write_text(json.dumps(r,indent=2)+'\n')
 script=shipping/f'run-root-current2k-{role}.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.join([str(ROOT/'.venv/bin/python'),'-B',str(runner),str(path),sha(path)])+'\n')
 print(json.dumps({'registered':str(path),'sha256':sha(path),'gpu_work':False,'model_payload_reads':False}))
