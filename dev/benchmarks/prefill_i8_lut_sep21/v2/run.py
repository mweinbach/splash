#!/usr/bin/env python3
"""Prepare a fresh v2 guard/full invocation; only Root uses --run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/prefill-i8-lut-sep21-v2')
    p.add_argument('--guard-only',action='store_true')
    p.add_argument('--pack-dir',type=Path)
    p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--pattern',choices=('both','spread-all','hit-concentrated'),default='both')
    p.add_argument('--variant',type=int,choices=range(1,9))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.pairs<2 or a.pairs>32 or a.pairs%2:raise ValueError('even pairs2..32 required')
    if not a.guard_only and not a.pack_dir:raise ValueError('full oracle requires --pack-dir')
    build=a.build.resolve();report=a.report.resolve();witness=Path(str(report)+'.invocation.json')
    for file in (report,witness,Path(str(report)+'.checkpoints.jsonl')):
        if file.exists():raise ValueError(f'choose fresh path: {file}')
    command=[str(build/'oracle'),'--guards' if a.guard_only else '--gpu',str(build/'splash.metallib')]
    if not a.guard_only:command.append(str(a.pack_dir.resolve()))
    command+=['2048',str(report)]
    controls={'SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','PREFILL_I8_LUT_PAIRS':str(a.pairs)}
    if a.pattern!='both':controls['PREFILL_I8_LUT_PATTERN']=a.pattern
    if a.variant:controls['PREFILL_I8_LUT_VARIANT']=str(a.variant)
    env={k:v for k,v in os.environ.items() if not k.startswith(
        ('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_','PREFILL_I8_LUT_'))}
    env.update(controls)
    old_lib=ROOT/'build/prefill-i8-lut-sep21/splash.metallib'
    lib=build/'splash.metallib';old_sha=sha(old_lib);new_sha=sha(lib)
    if old_sha is None or old_sha!=new_sha:raise ValueError('v2 must reuse exactly the v1 metallib')
    data={'schema':'prefill-i8-lut-invocation-v2','gpu_execution_requested':a.run,
          'command':command,'controls':controls,'guard_only':a.guard_only,
          'source_sha256':{f:sha(HERE/f) for f in ('generate.py','run.py','cache.hpp','guard_reference.hpp','component.mk')},
          'binary_sha256':sha(build/'oracle'),'generated_source_sha256':sha(build/'oracle.mm'),
          'v1_metallib_sha256':old_sha,'v2_metallib_sha256':new_sha,'metallib_byte_identical':True,
          'source_proven_guard_policy':'duplicates diagnosed and retained; only out-of-range IDs excluded',
          'candidate_fidelity_preregistration_changed':False,'negatives_before_timing':True,
          'durable_per_event_checkpoints':str(report)+'.checkpoints.jsonl',
          'coefficient_payload_loading_requested':not a.guard_only,'timing_requested':not a.guard_only,
          'guard_only_patterns':['spread-all','hit-concentrated'],
          'comparison_controls':['whole_sg4','current_best_sg2k128','matched_same_descriptor_uncompressed']}
    if not a.guard_only:
        data['pack_manifest_sha256']=sha(a.pack_dir.resolve()/'manifest.json')
        data['pack_certificate_sha256']=sha(a.pack_dir.resolve()/'certificate.json')
    report.parent.mkdir(parents=True,exist_ok=True);witness.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'gpu_execution_requested':a.run,'command':command,'witness':str(witness)}))
    if a.run:subprocess.run(command,env=env,check=True,cwd=ROOT)

if __name__=='__main__':main()
