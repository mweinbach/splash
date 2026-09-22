#!/usr/bin/env python3
"""Prepare matched standard flag0/1 commands; no GPU/model/capture payload reads."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[3]
FLAG='SPLASH_FLASH_GEMV_DECODE_R1_SEP21'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/gemv-r1-teacher-sep22-worker-v1')
    a=p.parse_args();b=a.build.resolve();seal=json.loads((b/'compiled-cpu-seal.json').read_text())
    if not seal['pass']:raise ValueError('Passing compiled CPU/source seal required')
    manifest=json.loads((b/'overlay-manifest.json').read_text());parent=Path(manifest['parent'])
    parent_command=parent/'root-standard-command.txt';base=shlex.split(parent_command.read_text())
    binary_index=base.index('--binary')+1;base[binary_index]=str(b/'splash-flash')
    if '--run-root-gpu' in base:base.remove('--run-root-gpu')
    env={};without=[];i=0
    while i<len(base):
        if base[i]=='--env':
            key,value=base[i+1].split('=',1);env[key]=value;i+=2
        else:without.append(base[i]);i+=1
    env.update({'SPLASH_FLASH_MTP':'0','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY':'0',
        'SPLASH_FLASH_BATCH_MTP':'0','SPLASH_FLASH_BATCH_MTP_PREFILL':'0',
        'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'0','SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS':'0',
        'SPLASH_FLASH_CAPTURE_EXPERT_IDS':'0'})
    semantic=ROOT/'build/release/flash/prefill4k-semantic-plan-v1.json';semantic_sha=sha(semantic)
    if env.get('SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS')!='4':raise ValueError('Current teacher gathered cap4 must be preserved')
    # Both commands share exact canonical workload/context/out/wire settings.
    required={'--mtp':'standard','--contexts':'2048','--batches':'1','--workloads':'coding',
              '--output-tokens':'256','--warmup':'1','--trials':'3','--max-context':'16384'}
    for key,value in required.items():
        if without[without.index(key)+1]!=value:raise ValueError('Parent canonical geometry differs: '+key)
    commands={};plans={};dry_receipts={}
    for value in ('0','1'):
        report=ROOT/f'build/release/flash/sep22-r1-teacher-standard-flag{value}-model-and-quality-v1.json'
        if report.exists():raise ValueError('Fresh Root report required: '+str(report))
        argv=list(without);argv[argv.index('--output')+1]=str(report)
        local_env={**env,FLAG:value}
        for key,v in local_env.items():argv+=['--env',key+'='+v]
        commands[value]=argv+['--run-root-gpu']
        dry=list(argv);dry[dry.index('--output')+1]=str(b/f'root-standard-flag{value}-dry-plan.json')
        plan_path=b/f'root-standard-flag{value}-dry-plan.json'
        if not plan_path.exists():subprocess.run(dry,cwd=ROOT,check=True)
        plan=json.loads(plan_path.read_text());dry_receipts[value]=plan
        if plan['gpu_executed']:raise ValueError('Command preparation may not execute GPU work')
        resolved=plan['environments']['standard']['resolved_flash_environment']
        if any(resolved.get(k)!='0' for k in ('SPLASH_FLASH_MTP','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY','SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21','SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS')):
            raise ValueError('Standard command must disable MTP and teacher priming')
        plans[value]={'path':str(plan_path),'sha256':sha(plan_path),'wave_count':len(plan.get('waves',[]))}
        script=b/f'run-root-standard-flag{value}.sh'
        script.write_text('#!/bin/sh\nset -eu\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join([str(ROOT/'.venv/bin/python'),'-B',str(Path(__file__).with_name('run_standard_root.py')),str(b),value])+'\n')
    if dry_receipts['0']['plan']!=dry_receipts['1']['plan'] or dry_receipts['0']['settings']!=dry_receipts['1']['settings']:
        raise ValueError('Matched commands must use identical frozen request plans/settings')
    envs=[dry_receipts[x]['environments']['standard']['resolved_flash_environment'] for x in ('0','1')]
    differing={k for k in set(envs[0])|set(envs[1]) if envs[0].get(k)!=envs[1].get(k)}
    if differing!={FLAG}:raise ValueError('Only the vector feature may differ in matched resolved environments')
    receipt={'schema':'TeacherV5-strict-standard-R1-matched-commands-v1','gpu_executed':False,'model_payload_bytes_read':0,
        'compiled_cpu_seal_sha256':sha(b/'compiled-cpu-seal.json'),'binary_sha256':sha(b/'splash-flash'),'metallib_sha256':sha(b/'splash.metallib'),
        'parent_standard_command_sha256':sha(parent_command),'original22_plan_sha256':semantic_sha,
        'root_commands':commands,'dry_plans':plans,'identical_geometry':required,
        'only_feature_difference':FLAG,'standard_explicitly_mtp_teacher_cache_disabled':True,
        'Root_current2K_state_and_numeric_proof_required_before_benchmark':True,
        'Root_order':'flag0 complete/unload then flag1 complete/unload; repetitions only if first pair exposes material uncertainty',
        'semantics_scope':'compare CURRENT flag0/flag1 original22 standard baseline; MTP20 is separate, all mtpState paths excluded',
        'numeric_scope':'inherited sampled RN/RTZ/FTZ/F64 per-route certificate; not MPP raw/BF16 exactness'}
    (b/'root-matched-command-seal.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'prepared_commands':str(b),'gpu_work':False,'original22_plan_sha256':semantic_sha,
                      'flag0':commands['0'],'flag1':commands['1']}))


if __name__=='__main__':main()
