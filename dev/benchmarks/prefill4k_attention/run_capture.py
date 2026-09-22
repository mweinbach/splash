#!/usr/bin/env python3
"""Dry-run by default; root serializes explicit --run GPU/model capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from install import launcher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tokens',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--outdir',type=Path,required=True)
p.add_argument('--binary',type=Path,default=ROOT/'build/prefill4k-attention/capture/prefill4k-attention-capture')
p.add_argument('--library',type=Path,default=ROOT/'build/prefill4k-attribution/splash.metallib')
p.add_argument('--all-rows',action='store_true',help='Capture all 2048 prepared query/gate rows at offset 0 instead of the last 128')
p.add_argument('--run',action='store_true')
a=p.parse_args()
raw_all_rows=os.environ.get('PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS','0')
if raw_all_rows not in ('','0','1'):
 p.error('PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS must be absent/0/1')
all_rows=a.all_rows or raw_all_rows=='1'
capture_rows=2048 if all_rows else 128
query_offset=0 if all_rows else 1920
capture_planned_bytes=3*(capture_rows*12288*2+capture_rows*6144*2+2048*512*2*2)
if a.outdir.exists(): raise RuntimeError('Capture directory must be fresh')
source=json.loads(a.tokens.read_text())
if not isinstance(source,list) or len(source)!=2048 or any(type(x) is not int or x<0 or x>=248320 for x in source):
 raise RuntimeError('Actual capture requires exactly2048 valid token IDs')
package=ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'
environment={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','PREFILL4K_'))}
defaults=launcher._local_profile_defaults(package);environment.update(defaults)
controls={'PREFILL4K_ATTRIBUTION_MODE':'normal','PREFILL4K_ATTRIBUTION_ROWS':'2048',
 'PREFILL4K_ATTRIBUTION_CAPACITY':'8192','PREFILL4K_ATTRIBUTION_WARMUP':'0',
 'PREFILL4K_ATTRIBUTION_REPEATS':'1','PREFILL4K_ATTRIBUTION_TEACHER_PRIME':'0',
 'PREFILL4K_ATTENTION_CAPTURE':str(a.outdir.resolve()),
 'PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS':'1' if all_rows else '0'}
environment.update(controls)
command=[str(a.binary.resolve()),str(a.library.resolve()),str(package),str(a.tokens.resolve()),str(a.report.resolve())]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
witness=a.report.with_suffix(a.report.suffix+'.invocation.json')
if witness.exists() or a.report.exists():raise RuntimeError('Choose fresh capture/report output')
witness.parent.mkdir(parents=True,exist_ok=True)
witness.write_text(json.dumps({'schema':'splash-private-qsa-capture-invocation-v1','gpu_executed':a.run,
 'command':command,'policy_environment':defaults,'controls':controls,'binary_sha256':sha(a.binary),
 'metallib_sha256':sha(a.library),'tokens_sha256':sha(a.tokens),'capture_planned_bytes':capture_planned_bytes,
 'all_rows':all_rows,'rows':capture_rows,'query_offset':query_offset,
 'overlay_source_sha256':sha(ROOT/'build/prefill4k-attention/capture/source/FlashForward.cpp')},indent=2)+'\n')
print(json.dumps({'gpu_executed':a.run,'invocation':str(witness),'command':command,'all_rows':all_rows,'rows':capture_rows,'query_offset':query_offset,'capture_planned_bytes':capture_planned_bytes}))
if a.run:subprocess.run(command,cwd=ROOT,env=environment,check=True)
