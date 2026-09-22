#!/usr/bin/env python3
"""Root only: execute every preregistered R5 union after external pin checks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command',type=Path,required=True)
    parser.add_argument('--command-sha256',required=True)
    parser.add_argument('--run-root-gpu',action='store_true')
    args=parser.parse_args()
    if not args.run_root_gpu:raise ValueError('Explicit Root GPU execution required')
    if sha(args.command)!=args.command_sha256:raise ValueError('Externally preregistered Root command differs')
    command=json.loads(args.command.read_text())
    if command['schema']!='Root-R5-integer-only-U10-U25-U50-component-command-v1':raise ValueError('Unknown component command')
    for path,expected in command['pins'].items():
        if sha(path)!=expected:raise ValueError('Frozen component/runtime/source/CPU receipt drift: '+path)
    root=Path(command['cwd']);build=Path(command['build'])
    if [case['name']for case in command['cases']]!=['u10','u25','u50']:raise ValueError('All three genuine union cases required')
    for case in command['cases']:
        expected=[str(build/'oracle'),'--gpu',str(build/'splash.metallib'),command['expert_store'],'0','5',case['report'],'--case',case['name'],'--pairs','18']
        if case['argv']!=expected:raise ValueError('Primitive control/geometry/case drift')
        for path in (Path(case['report']),Path(case['report']+'.resource.json'),Path(case['log'])):
            if path.exists():raise FileExistsError('Fresh result/log/resource required: '+str(path))
    lockpath=root/'build/splash-tuning-gpu.lock'
    with lockpath.open('a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in command['cases']:
            with Path(case['log']).open('x')as log:
                result=subprocess.run(case['argv'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Primitive failed; retain partial artifacts: '+case['name'])
            # Only Root executes this function and reads actual output evidence.
            report=json.loads(Path(case['report']).read_text())
            resource=json.loads(Path(case['report']+'.resource.json').read_text())
            if (report.get('rows')!=5 or report.get('case')!=case['name']
                    or report.get('actual_route_union_count')!=case['union']
                    or report.get('all_pre_timing_semantic_checks_pass')is not True
                    or report.get('timing_pass')is not True or resource.get('pass')is not True):
                raise ValueError('Incomplete native source/union/safety/timing/resource evidence')
            timings=report.get('timings')
            if not isinstance(timings,list)or len(timings)!=2:raise ValueError('Both full chain controls required')
            for item in timings:
                if item['warm_gpu_ms']<150 or len(item['samples'])!=18:raise ValueError('Each native chain needs150ms warm and18 balanced samples')
            print(json.dumps({'case':case['name'],'source_scope':'R5 primitive; synthetic normalized input; bounded real coefficient slices',
                             'native_pass':True,'report':case['report'],'report_sha256':sha(case['report']),
                             'resource_sha256':sha(case['report']+'.resource.json'),'no_Worker_or_task_qualification':True}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
