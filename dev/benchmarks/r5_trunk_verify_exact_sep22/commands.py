#!/usr/bin/env python3
"""Prepare CPU/source-only Root launch metadata. Does not execute inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def publish(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',required=True);a=p.parse_args();build=(ROOT/a.build).resolve()
    ready=json.loads((build/'CPU_READY.json').read_text())
    if not ready['pass'] or ready['GPU_work'] or ready['expected_R5_candidate_graph_calls']!=288:raise SystemExit('fresh CPU-ready R5 state build required')
    old=ROOT/'build/trunkverify-rawQ4-GDN26-VerifyR4-sep22-v1/Root-rawQ4-native-command.json'
    environment=dict(json.loads(old.read_text())['environment']);environment['SPLASH_FLASH_MTP']='1';environment['SPLASH_FLASH_MTP_DRAFT_DEPTH']='4'
    report0=ROOT/'build/release/flash/sep22-trunkverify-R5-current-Q4-control-v1.json';report1=ROOT/'build/release/flash/sep22-trunkverify-R5-current-Q4-compare-v1.json'
    spill=ROOT/'build/release/flash/sep22-trunkverify-R5-current-Q4-spill-v1'
    for path in (report0,report1):
        for suffix in ('','.writing','.partial','.failure.json'):
            if Path(str(path)+suffix).exists():raise SystemExit('fresh native report required')
    if spill.exists():raise SystemExit('fresh spill path required')
    runner=build/'run_root.py';shutil.copyfile(HERE/'run_root.py',runner)
    pins={}
    def pin(path):pins[str(path)]=sha(path)
    for path in (build/'CPU_READY.json',old,runner):pin(path)
    for role in ('control','candidate'):
        output=build/role
        for path in output.iterdir():
            if path.is_file():pin(path)
    for path in (build/'source').rglob('*'):
        if path.is_file():pin(path)
    for obj in ready['objects']:pin(Path(obj['path']))
    pin(build/'splash.metallib')
    worker=Path(ready['worker'])
    for name in ('CPU_READY.json','compiled-cpu-seal.json','overlay-manifest.json','splash-flash','splash.metallib','R5-qualified.air'):pin(worker/name)
    for obj in json.loads((worker/'CPU_READY.json').read_text())['compiled_objects']:pin(worker/obj['path'])
    package=ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1';tokens=ROOT/'build/release/flash/prefill4k-fixture/code2048.tokens.json'
    jobs=[]
    for role,flag,report in (('control','0',report0),('candidate','1',report1)):
        env={**environment,'SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22':flag}
        jobs.append({'role':role,'environment':env,'argv':[str(build/role/'oracle'),'--gpu','export' if role=='control' else 'compare',str(build/'splash.metallib'),str(package),str(tokens),str(report),str(spill)]})
    command={'schema':'R5-current-Q4-main-state-paired-Root-command-v1','Root_GPU_only':True,'GPU_executed_by_preparation':False,'cwd':str(ROOT),'build':str(build),'pins':pins,'jobs':jobs,'spill':str(spill),'scope':'same CURRENT R5 source/library flag0 blocked10 versus flag1 integer6; native main state only','expected':{'unique_frames':29,'repeated_frames':65,'state_frames':17,'request_planes':134,'lazy_arenas':216,'candidate_R5_calls':288,'candidate_R5_rows':1440,'R4_only_counters':0,'R5_metadata_guards':9,'preserved_R4_metadata_guards_separate':8,'spill_limit_bytes':4<<30,'source_spill_bound_bytes':4182654976,'host_streaming_metadata_precheck_bytes':64<<20},'payload_reads_or_hashes_by_preparation':0,'teacher_head_worker_tasks_performance_qualified':False,'original_environment_metadata':str(old),'mode_delta_only':['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22']}
    path=build/'Root-command.json';publish(path,command);digest=sha(path)
    script=build/'run-root.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.quote(str(ROOT/'.venv/bin/python'))+' -B '+shlex.quote(str(runner))+' '+shlex.quote(str(path))+' '+shlex.quote(digest)+'\n');script.chmod(0o755)
    print(json.dumps({'prepared':str(script),'command_sha256':digest,'pins':len(pins),'GPU_work':False,'payload_reads':0}))


if __name__=='__main__':main()
