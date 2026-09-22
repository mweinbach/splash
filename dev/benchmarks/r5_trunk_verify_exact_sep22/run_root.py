#!/usr/bin/env python3
"""Root-only paired native state run. Preparers never invoke this file."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv)!=3:raise SystemExit('sealed command path and external command SHA required')
    path=Path(sys.argv[1])
    if sha(path)!=sys.argv[2]:raise SystemExit('Root command changed')
    command=json.loads(path.read_text())
    if command['schema']!='R5-current-Q4-main-state-paired-Root-command-v1' or len(command['jobs'])!=2:raise SystemExit('paired Root command required')
    for name,digest in command['pins'].items():
        if sha(Path(name))!=digest:raise SystemExit('frozen source/CPU artifact drift: '+name)
    jobs=command['jobs'];a,b=jobs
    if [a['role'],b['role']]!=['control','candidate'] or a['argv'][1:3]!=['--gpu','export'] or b['argv'][1:3]!=['--gpu','compare']:raise SystemExit('ordered separate role processes required')
    flag='SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'
    if a['environment'][flag]!='0' or b['environment'][flag]!='1' or {k:v for k,v in a['environment'].items() if k!=flag}!={k:v for k,v in b['environment'].items() if k!=flag}:raise SystemExit('only R5 role delta allowed')
    for job in jobs:
        if job['environment'].get('SPLASH_FLASH_MTP_DRAFT_DEPTH')!='4' or job['environment'].get('SPLASH_FLASH_MTP')!='1' or job['environment'].get('SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS')!='4':raise SystemExit('genuine fixed4 and gathered cap4 required')
        report=Path(job['argv'][6])
        for suffix in ('','.writing','.partial','.failure.json'):
            if Path(str(report)+suffix).exists():raise SystemExit('fresh native report required')
    if Path(command['spill']).exists():raise SystemExit('fresh native spill directory required')
    for index,job in enumerate(jobs):
        if index:
            control_path=Path(a['argv'][6]);control=json.loads(control_path.read_text())
            if not control['pass'] or control['role']!='export' or not control['backend_destroyed'] or control['frames']!=29 or control['repeated_frames']!=65 or control['producer']['actual_R5_calls']!=0 or control['allocation']['actual']['after_model_destruction']!=0 or not control['allocation']['actual']['backend_stopped']:raise SystemExit('actual completed zero-owner control export required')
            manifest=Path(command['spill'])/'complete.json';declaration=json.loads(manifest.read_text())
            if not declaration['complete'] or declaration['common']!=control['common'] or len(declaration['frames'])!=29:raise SystemExit('control export and actual manifest mismatch')
        env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_','FLASH_'))};env.update(job['environment'])
        result=subprocess.run(job['argv'],cwd=command['cwd'],env=env)
        if result.returncode:return result.returncode if result.returncode>0 else 128-result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
