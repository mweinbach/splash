#!/usr/bin/env python3
"""Private current-Q4 R5 setup integration; CPU source/artifact work only."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PRIVATE=HERE.relative_to(ROOT);BASE=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
COMPONENT=ROOT/'build/expert-r5-compact-native-sep22-component-v4'
COMPONENT5=ROOT/'build/expert-r5-compact-native-sep22-component-v5'
PARENT_EXE='663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
PARENT_LIB='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
QUALIFIED_AIR='50002976851cd0bf2cf0f133c0164ccf177dfc1e7493b28563ba8fdbd1bb7ac3'
LOADED_COMPONENT_LIB='bc92f5c869717d6a896ebdc0615f623e0c1c36628ad632f4e716cea352e6baff'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args(argv);b=args.build.resolve()
    if b.exists()or ROOT/'build'not in b.parents:raise ValueError('Fresh private worker output required')
    parent=json.loads((BASE/'overlay-manifest.json').read_text());seal=json.loads((BASE/'compiled-cpu-seal.json').read_text())
    if not seal['pass']or sha(BASE/'splash-flash')!=PARENT_EXE or sha(BASE/'splash.metallib')!=PARENT_LIB:raise ValueError('Qualified current parent drift')
    for record in seal['compiled_objects']+seal['artifacts']:
        if sha(BASE/record['path'])!=record['sha256']:raise ValueError('Current parent compiled artifact drift')
    if sha(COMPONENT/'R5-integer-only.air')!=QUALIFIED_AIR or sha(COMPONENT/'splash.metallib')!=LOADED_COMPONENT_LIB or sha(COMPONENT5/'splash.metallib')!=LOADED_COMPONENT_LIB:raise ValueError('Loaded winning INTEGER leaf/library drift')
    for name in ('abi.hpp','plan.metal'):
        rel=Path('source/dev/benchmarks/expert_r5_compact_native_sep22')/name
        if (COMPONENT/rel).read_bytes()!=(COMPONENT5/rel).read_bytes():raise ValueError('Winning leaf source/ABI differs')
    shutil.copytree(BASE,b)
    for name in ('overlay-manifest.json','compiled-cpu-seal.json','Root-rawQ4-native-qualified.json'):(b/name).rename(b/('inherited-parent-'+name))
    own=b/'source'/PRIVATE;shutil.copytree(HERE,own)
    spec=importlib.util.spec_from_file_location('_current_R5_overlay',HERE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
    journal=[]
    for name in ('runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'):
        old=(BASE/'source'/name).read_text();new=overlay.transform(name,old);(b/'source'/name).write_text(new)
        before,after=old.splitlines(keepends=True),new.splitlines(keepends=True);edits=[]
        for tag,i,j,a,c in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
            if tag!='equal':edits.append({'old_start':i,'old_end':j,'new_start':a,'new_end':c,'old_lines':before[i:j],'new_lines':after[a:c]})
        restored=list(after)
        for edit in reversed(edits):restored[edit['new_start']:edit['new_end']]=edit['old_lines']
        if restored!=before:raise ValueError('Literal parent source restoration failed')
        journal.append({'path':name,'parent_sha256':sha(BASE/'source'/name),'sha256':sha(b/'source'/name),'edits':edits,'inverse_literal_exact':True})
    parts={name:sha(own/name)for name in ('policy.hpp','overlay.py','policy_cpu.cpp','prepare.py')}
    parts.update({'parent_source_identity':parent['source_identity_sha256'],'parent_compiled_seal_sha256':sha(BASE/'compiled-cpu-seal.json'),'planner_AIR_sha256':QUALIFIED_AIR,'scope':'singleton main actual VerifyR5 INTEGER setup only; current Q4 R4 branches/FP/head/batch/pref unchanged-v1'})
    identity=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (own/'source_identity.hpp').write_text('#pragma once\nnamespace splash::flash::compact_native_r5_verify_sep22 {inline constexpr char kSourceIdentitySha256[]="'+identity+'";}\n')
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')]
    link=(b/'link-inputs.mk').read_text();names=next(line for line in link.splitlines()if line.startswith('REBUILD_NAMES :=')).split(':=',1)[1].split();srcs={m.group(1):m.group(2)for m in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$',link,re.M)}
    cores=[b/token.removeprefix('$(BUILD)/')for token in next(line for line in link.splitlines()if line.startswith('CORE :=')).split(':=',1)[1].split()];objects=[b/'host'/(name+'.o')for name in names]
    if len(names)!=50 or len(cores)!=4:raise ValueError('Current50Host/Core4 closure required')
    commands=[];census=[];pins=[]
    def execute(command):commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    for name,obj in zip(names,objects):
        source=b/'source'/srcs[name];command=['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(source)]
        dependencies=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True).stdout.replace('\\\n',' ').split()
        consumer=any(path.endswith('/expert_r5_verify_worker_sep22/policy.hpp')for path in dependencies)
        if consumer!=(name in ('FlashForward','FlashWorker')):raise ValueError('Unexpected private R5 header consumer:'+name)
        census.append({'object':name,'R5_private_header_consumer':consumer,'dependencies':dependencies})
        if consumer:execute(['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(source),'-o',str(obj)])
        elif sha(obj)!=sha(BASE/obj.relative_to(b)):raise ValueError('Unaffected object changed')
        pins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(b)),'changed_for_private_header':consumer})
    for obj in cores:
        if sha(obj)!=sha(BASE/obj.relative_to(b)):raise ValueError('Core4 changed')
        pins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(b)),'changed_for_private_header':False})
    # The exact parent ordered tuple, followed ONLY by qualified INTEGER AIR.
    parent_command=[command for command in parent['compiler_commands']if 'metallib'in command][-1]
    airs=[b/Path(token).relative_to(BASE)for token in parent_command[4:parent_command.index('-o')]]
    execute(['xcrun','-sdk','macosx','metallib',*map(str,airs),'-o',str(b/'baseline-relinked.metallib')])
    if sha(b/'baseline-relinked.metallib')!=PARENT_LIB:raise ValueError('Parent754 exact relink failed')
    # Authenticated recipe witness: same parent tuple plus the existing probe
    # and original500029 leaf reproduces the entire Root-loaded winningbc92.
    probe=ROOT/'build/prefill4k-allrows-qmv-one-layer/probe.air'
    if sha(probe)!='f92914ca3ba6c7c02eb33c5774c17eb693a7136d109b3dc01dc65e255a113c30':raise ValueError('Existing original native observation AIR drift')
    shutil.copy2(COMPONENT/'R5-integer-only.air',b/'R5-qualified.air')
    execute(['xcrun','-sdk','macosx','metallib',*map(str,airs),str(probe),str(b/'R5-qualified.air'),'-o',str(b/'qualified-component-relinked.metallib')])
    if sha(b/'qualified-component-relinked.metallib')!=LOADED_COMPONENT_LIB:raise ValueError('Exact Root-loadedbc92 recipe not reproduced')
    execute(['xcrun','-sdk','macosx','metallib',*map(str,airs),str(b/'R5-qualified.air'),'-o',str(b/'splash.metallib')])
    execute(['xcrun','-sdk','macosx','clang++',*flags,*map(str,objects+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')])
    execute(['xcrun','-sdk','macosx','clang++',*flags,str(own/'policy_cpu.cpp'),*map(str,[obj for obj,name in zip(objects,names)if name!='FlashWorker']+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'R5-policy-CPU')])
    cpu=subprocess.run([str(b/'R5-policy-CPU')],cwd=ROOT,check=True,capture_output=True,text=True);worker=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],cwd=ROOT,check=True,capture_output=True,text=True)
    dependencies=[];env={k:v for k,v in os.environ.items()if not k.startswith(('SPLASH_','FLASH_'))}
    for name in ('SPLASH_FLASH_ALLROWS_FULL512_TARGET','SPLASH_FLASH_BLOCKED_MOE','SPLASH_FLASH_MOE_DIRECT_A','SPLASH_FLASH_MOE_Q4X8','SPLASH_FLASH_ALLROWS_GATHERED_MPP'):env[name]='1'
    env.update({'SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS':'4','SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22':'1','SPLASH_FLASH_MTP':'1','SPLASH_FLASH_MTP_DRAFT_DEPTH':'4'})
    for removed in (None,*[name for name in env if name.startswith('SPLASH_')and name!='SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22']):
        test=dict(env)
        if removed:test.pop(removed)
        result=subprocess.run([str(b/'R5-policy-CPU'),'--depth'],cwd=ROOT,env=test,capture_output=True,text=True)
        if (result.returncode==0)!=(removed is None):raise ValueError('R5 missing dependency/depth accepted:'+str(removed))
        dependencies.append({'removed':removed,'accepted':result.returncode==0})
    for value in ('0','1'):
        test=dict(env);test['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22']=value
        subprocess.run([str(b/'R5-policy-CPU'),'--lifetime'],cwd=ROOT,env=test,check=True,capture_output=True,text=True)
    changed_paths=['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']
    for source in (BASE/'source').rglob('*'):
        if source.is_file()and str(source.relative_to(BASE/'source'))not in changed_paths and sha(source)!=sha(b/'source'/source.relative_to(BASE/'source')):raise ValueError('Unrelated source/public header changed')
    core={'schema':'singleton-R5-integer-current-Q4-worker-compiled-source-v1','pass':True,'source_identity_sha256':identity,'exe_sha256':sha(b/'splash-flash'),'library_sha256':sha(b/'splash.metallib'),'planner_AIR_sha256':QUALIFIED_AIR,'parent_exe_sha256':PARENT_EXE,'parent_library_sha256':PARENT_LIB,'public_headers_changed':False,'new_GPU_allocation_bytes':0,'changed_paths':changed_paths,'GPU_work':False,'whole_state_qualified':False,'maximum_verify_rows':5,'compiled_objects':pins,'actual50TU_header_census':census,'private_header_consumers':['FlashForward','FlashWorker'],'Core4_unchanged':True,'CPU_policy':json.loads(cpu.stdout),'CPU_Worker':json.loads(worker.stdout),'dependency_cases':dependencies,'qualified_loadedbc92_recipe_reproduced':True,'new_floating_AIR_compiles':0,'new_integer_AIR_compiles':0}
    (b/'compiled-cpu-seal.json').write_text(json.dumps(core,indent=2)+'\n');(b/'CPU_READY.json').write_text(json.dumps(core,indent=2)+'\n')
    overlay_manifest={**core,'schema':'singleton-main-R5-integer-only-original-native-M16-worker-source-v1','base':str(BASE),'parent_source_identity':parent['source_identity_sha256'],'identity_parts':parts,'source_hook_journal':journal,'compiler_commands':commands,'files':[{'path':str(path.relative_to(b/'source')),'sha256':sha(path)}for path in sorted((b/'source').rglob('*'))if path.is_file()]}
    (b/'overlay-manifest.json').write_text(json.dumps(overlay_manifest,indent=2)+'\n')
    print(json.dumps({'CPU_ready':True,'build':str(b),'source_identity':identity,'exe_sha256':core['exe_sha256'],'library_sha256':core['library_sha256'],'CPU_READY_sha256':sha(b/'CPU_READY.json'),'GPU_work':False}))
    return 0
if __name__=='__main__':raise SystemExit(main())
