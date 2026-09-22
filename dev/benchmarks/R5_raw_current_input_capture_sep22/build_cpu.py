#!/usr/bin/env python3
"""Authorized isolated CPU closure build; no token/model/capture/report access."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import os

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'build/R5-integer-currentQ4-fixed4-sep22-worker-v2'
EXE='6f7e22a2ca9c0c9728bf356391d2bde17c4e9bc90ab6295e625cac15e7987d68'
LIB='dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6'
SOURCE_PINS='687a7f159c11ea9840550c51fc0c92f269ea7c021b579614b5382f8444bf090a'
EXPECTED={'FlashForward','FlashWorker','FlashBatchForward','FlashBatchPrefill','FlashBatchVerify'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def canonical(name):return re.sub(r'^(?:\d+-)+','',name)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--guarded-friend-source-review-sha256')
    args=parser.parse_args();build=args.build.resolve()
    require(not build.exists() and ROOT/'build' in build.parents,'fresh isolated build required')
    pins=json.loads((HERE/'SOURCE_PINS.json').read_text())
    require(sha(HERE/'SOURCE_PINS.json')==SOURCE_PINS,'independently reviewed capture source pin drift')
    if args.guarded_friend_source_review_sha256:
        require(len(args.guarded_friend_source_review_sha256)==64 and all(c in '0123456789abcdef' for c in args.guarded_friend_source_review_sha256),'guarded-friend source review digest malformed')
    for name,digest in pins['source_files_sha256'].items():require(sha(HERE/name)==digest,'reviewed capture source drift:'+name)
    seal=json.loads((BASE/'compiled-cpu-seal.json').read_text());parent=json.loads((BASE/'overlay-manifest.json').read_text())
    require(seal['pass'] and seal['source_identity_sha256']=='4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a' and sha(BASE/'splash-flash')==EXE and sha(BASE/'splash.metallib')==LIB,'exact current parent required')
    require(len(seal['compiled_objects'])==54,'exact parent54 object closure required')
    for record in seal['compiled_objects']:require(sha(BASE/record['path'])==record['sha256'],'parent object drift:'+record['path'])
    for record in parent['files']:require(sha(BASE/'source'/record['path'])==record['sha256'],'parent source/header drift:'+record['path'])
    build.mkdir(parents=True);shutil.copytree(BASE/'source',build/'source')
    private=build/'source'/HERE.relative_to(ROOT);private.mkdir(parents=True,exist_ok=True)
    for name in [*pins['source_files_sha256'],'SOURCE_PINS.json','SOURCE_V2_WHOLE_BOUND.diff','SOURCE_V3_GUARDED_FRIEND.diff','build_cpu.py']:shutil.copyfile(HERE/name,private/name)
    for record in seal['compiled_objects']:
        dst=build/record['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/record['path'],dst)
    shutil.copyfile(BASE/'splash.metallib',build/'splash.metallib')
    shutil.copyfile(BASE/'link-inputs.mk',build/'link-inputs.mk')
    spec=importlib.util.spec_from_file_location('reviewedR5CaptureOverlay',HERE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
    inspection=(HERE/'inspection.cpp.inc').read_text();journal=[]
    for rel in ['runtime/flash/FlashForward.hpp','runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:
        old=(BASE/'source'/rel).read_text();new=overlay.transform(rel,old,inspection);(build/'source'/rel).write_text(new)
        before,after=old.splitlines(True),new.splitlines(True);edits=[]
        for tag,i,j,x,y in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
            if tag!='equal':edits.append({'old_start':i,'old_end':j,'new_start':x,'new_end':y,'old_lines':before[i:j],'new_lines':after[x:y]})
        restored=list(after)
        for edit in reversed(edits):restored[edit['new_start']:edit['new_end']]=edit['old_lines']
        require(restored==before,'literal parent inverse drift:'+rel)
        journal.append({'path':rel,'parent_sha256':sha(BASE/'source'/rel),'sha256':sha(build/'source'/rel),'inverse_literal_exact':True,'edits':edits})
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(build/'source/runtime'),'-I'+str(build/'source'),'-I'+str(build/'source/dev/benchmarks/prefill4k_attention')]
    link=(build/'link-inputs.mk').read_text();names=next(x for x in link.splitlines() if x.startswith('REBUILD_NAMES :=')).split(':=')[1].split()
    srcs={m.group(1):m.group(2) for m in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$',link,re.M)}
    cores=[build/t.removeprefix('$(BUILD)/') for t in next(x for x in link.splitlines() if x.startswith('CORE :=')).split(':=')[1].split()]
    objects=[build/'host'/(name+'.o') for name in names];require(len(names)==50 and len(set(names))==50 and len(cores)==4,'actual50 host/Core4 required')
    commands=[];census=[];object_pins=[]
    for name,obj in zip(names,objects):
        source=build/'source'/srcs[name]
        deps=subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(source)],cwd=ROOT,capture_output=True,text=True,check=True).stdout.replace('\\\n',' ').split()
        changed=any(t.endswith('/runtime/flash/FlashForward.hpp') for t in deps)
        require(changed==(canonical(name) in EXPECTED),'actual header consumer mismatch:'+name)
        require(all(not t.startswith(('runtime/','dev/')) for t in deps),'live source dependency in isolated closure')
        census.append({'object':name,'canonical_object':canonical(name),'source':str(source.relative_to(build/'source')),'modified_Forward_header_consumer':changed,'dependencies':deps})
        if changed:
            command=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(source),'-o',str(obj)];commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
        else:require(sha(obj)==sha(BASE/obj.relative_to(build)),'unaffected host object changed')
        object_pins.append({'path':str(obj.relative_to(build)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(build)),'recompiled':changed})
    for obj in cores:
        require(sha(obj)==sha(BASE/obj.relative_to(build)),'Core4 byte drift');object_pins.append({'path':str(obj.relative_to(build)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(build)),'recompiled':False})
    command=['xcrun','-sdk','macosx','clang++',*flags,*map(str,objects+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(build/'splash-flash')];commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    require(sha(build/'splash.metallib')==LIB,'original library drift')
    environment={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_','FLASH_'))}
    cpu=subprocess.run([str(build/'splash-flash'),'--cpu-self-test'],cwd=ROOT,env=environment,capture_output=True,text=True,check=True)
    (build/'cpu-self-test.json').write_text(cpu.stdout)
    record={'schema':'current-real-R5-raw-capture-isolated-CPU-closure-v1','pass':True,'GPU_work':False,'compiler_scope':'CPU only five actual modified header consumers','parent':str(BASE),'parent_source_identity_sha256':seal['source_identity_sha256'],'parent_exe_sha256':EXE,'library_sha256':LIB,'exe_sha256':sha(build/'splash-flash'),'reviewed_capture_SOURCE_PINS_sha256':SOURCE_PINS,'independent_capture_Source_GO_sha256':'a5c1eb6a4d2d6006d5c24c2b2507fd4b1fa77b8ce5f0ed830c74ccbe89210780','independent_guarded_friend_source_delta_review_sha256':args.guarded_friend_source_review_sha256,'actual50TU_census':census,'compiled54_objects':object_pins,'private_header_consumers':sorted(EXPECTED),'Core4_unchanged':True,'other45_host_objects_unchanged':True,'all53_nonWorker_objects_current_parent_matched_or_header_rebuilt':True,'new_AIR_or_metallib_compiles':0,'whole_campaign_preflight_retained':True,'model_token_tensor_capture_profile_or_actual_report_reads_or_hashes':0,'whole_model_capture_QA_or_performance_qualified':False,'execution_command_sealed':False,'pending_command_policy_binding':'Root measured winner, not old W8-on policy','CPU_self_test':json.loads(cpu.stdout),'journal':journal,'compiler_commands':commands,'files':[{'path':str(p.relative_to(build/'source')),'sha256':sha(p)} for p in sorted((build/'source').rglob('*')) if p.is_file()]}
    for name in ['compiled-cpu-seal.json','CPU_READY.json','overlay-manifest.json']:(build/name).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'build':str(build),'CPU_READY_sha256':sha(build/'CPU_READY.json'),'exe_sha256':record['exe_sha256'],'library_sha256':LIB,'actual50TU_census':len(census),'compiled54_objects':len(object_pins),'rebuilt':sorted(EXPECTED),'GPU_work':False,'execution_command_sealed':False}))


if __name__=='__main__':main()
