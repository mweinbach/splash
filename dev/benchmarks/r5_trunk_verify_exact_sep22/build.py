#!/usr/bin/env python3
"""Build CPU-only oracle from CURRENT R5 headers and matching object closure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
WORKER=ROOT/'build/R5-integer-currentQ4-fixed4-sep22-worker-v2'
BASE=ROOT/'dev/benchmarks/trunk_verify_exact_sep22'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stem(path):return re.sub(r'^(?:\d+-)+','',Path(path).stem)
def publish(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',required=True);a=p.parse_args()
    build=(ROOT/a.build).resolve()
    if build.exists():raise SystemExit('fresh build required')
    ready=json.loads((WORKER/'CPU_READY.json').read_text())
    if not ready['pass'] or ready['GPU_work'] or ready['maximum_verify_rows']!=5 or len(ready['compiled_objects'])!=54 or len(ready['actual50TU_header_census'])!=50:raise SystemExit('current R5 CPU closure required')
    # Only source and compiled artifact bytes are authenticated by preparation.
    for obj in ready['compiled_objects']:
        if sha(WORKER/obj['path'])!=obj['sha256']:raise SystemExit('current matching object drift: '+obj['path'])
    if sha(WORKER/'splash.metallib')!=ready['library_sha256']:raise SystemExit('current library drift')
    build.mkdir(parents=True);shutil.copytree(WORKER/'source',build/'source')
    private=build/'source/dev/benchmarks/r5_trunk_verify_exact_sep22';private.mkdir(parents=True,exist_ok=True)
    for name in ['oracle.mm','prepare.py','source_delta.json','build.py','r5_inspection.cpp.inc']:shutil.copyfile(HERE/name,private/name)
    inspector=(BASE/'inspect.hpp').read_text()
    anchor='  static bool rawOwns(const FlashForward &target,const FlashRequestState &request);'
    if inspector.count(anchor)!=1:raise SystemExit('inspector anchor changed')
    inspector=inspector.replace(anchor,anchor+'\n  static uint32_t actualBundleGuardProbes(FlashForward &target);\n  static uint32_t actualR5GuardProbes(FlashForward &target);',1)
    (private/'inspect.hpp').write_text(inspector)
    inc=(BASE/'inspection.cpp.inc').read_text()
    if inc.count('r.maximumRows()!=4')!=1:raise SystemExit('tape maximum anchor changed')
    inc=inc.replace('r.maximumRows()!=4','r.maximumRows()!=5',1)+'\n'+(BASE/'guard_bundle_inspection.cpp.inc').read_text()+'\n'+(HERE/'r5_inspection.cpp.inc').read_text()
    (private/'inspection.cpp.inc').write_text(inc)
    header=build/'source/runtime/flash/FlashForward.hpp';original_h=header.read_text();anchor='  friend class FlashBatchForward;'
    if original_h.count(anchor)!=2:raise SystemExit('private friend anchor changed')
    before,after=original_h.rsplit(anchor,1)
    modified_h=before+'  friend class FlashDeepPrefixOracle;\n'+anchor+after;header.write_text(modified_h)
    forward=build/'source/runtime/flash/FlashForward.cpp';original_f=forward.read_text();prefix='#include "inspect.hpp"\n';suffix='\n'+inc
    forward.write_text(prefix+original_f+suffix)
    if forward.read_text()[len(prefix):-len(suffix)]!=original_f or header.read_text().replace('  friend class FlashDeepPrefixOracle;\n'+anchor,anchor,1)!=original_h:raise SystemExit('literal inverse math restoration failed')
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(build/'source'),'-I'+str(build/'source/runtime'),'-I'+str(build/'source/dev/benchmarks/prefill4k_attention'),'-I'+str(private)]
    compiler=['xcrun','-sdk','macosx','clang++']
    census=[];objects=[];records={stem(x['object']):x for x in ready['actual50TU_header_census']}
    for index,obj in enumerate(ready['compiled_objects']):
        name=stem(obj['path']);record=records.get(name);source=None;consumer=False
        if record:
            sources=[Path(d).relative_to(WORKER/'source') for d in record['dependencies'] if d.endswith(('.mm','.cpp')) and str(WORKER/'source')+'/' in d]
            if len(sources)!=1:raise SystemExit('exact source census differs: '+name)
            source=build/'source'/sources[0]
            result=subprocess.run(compiler+flags+['-MM',str(source)],cwd=ROOT,check=True,capture_output=True,text=True)
            deps=result.stdout.replace('\\\n',' ').split()
            consumer=any(d.endswith('/runtime/flash/FlashForward.hpp') for d in deps)
            if any(d.startswith(('runtime/','dev/')) for d in deps):raise SystemExit('live source dependency')
            census.append({'object':name,'source':str(source),'modified_header_consumer':consumer,'excluded_main':name=='FlashWorker','dependencies':deps})
        elif name not in {'MetalBackend','DeviceCapabilities','Protocol','MemoryGovernor'}:raise SystemExit('uncensused object '+name)
        if name=='FlashWorker':continue
        dst=build/'objects'/f'{index:03d}-{name}.o';dst.parent.mkdir(exist_ok=True)
        if consumer:subprocess.run(compiler+flags+['-MMD','-MP','-c',str(source),'-o',str(dst)],cwd=ROOT,check=True)
        else:shutil.copyfile(WORKER/obj['path'],dst)
        objects.append({'path':str(dst),'sha256':sha(dst),'inherited_path':str(WORKER/obj['path']),'inherited_sha256':obj['sha256'],'recompiled_for_header':consumer})
    if len(census)!=50 or len(objects)!=53 or not next(x for x in census if x['object']=='FlashForward')['modified_header_consumer']:raise SystemExit('53 non-main objects and actual50 TU census required')
    shutil.copyfile(WORKER/'splash.metallib',build/'splash.metallib')
    inputs=[WORKER/'CPU_READY.json',WORKER/'compiled-cpu-seal.json',header,forward,*private.glob('*')]
    pin_records=[{'path':str(p),'sha256':sha(p)} for p in inputs if p.is_file()]
    provenance={'schema':'R5-main-native-state-CPU-closure-v1','worker':str(WORKER),'worker_executable_sha256':ready['exe_sha256'],'worker_source_identity_sha256':ready['source_identity_sha256'],'metallib_sha256':ready['library_sha256'],'planner_AIR_sha256':ready['planner_AIR_sha256'],'objects':objects,'header_census':census,'headers':pin_records,'runtime_worker_math_changed':False,'private_friend_only_header_clone':True,'production_edits':False,'GPU_work':False,'model_operand_token_export_payload_reads':0,'physical_state_planes':134,'lazy_tape_arenas':216,'tape_planes_with_defined_PLE':218,'expected_unique_frames':29,'expected_repeated_frames':65,'expected_R5_candidate_graph_calls':288,'expected_R5_candidate_graph_rows':1440,'expected_R4_only_counters':0,'whole_native_state_qualified':False,'head_worker_task_performance_claim':False}
    for role in ['control','candidate']:
        output=build/role;output.mkdir();meta={**provenance,'role':role};publish(output/'provenance.json',meta)
        summary={k:v for k,v in meta.items() if k not in {'objects','header_census','headers'}}
        (output/'TrunkVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char*kPrefillExactProvenancePath='+json.dumps(str(output/'provenance.json'))+';\ninline constexpr const char*kPrefillExactBuildProvenance=R"META('+json.dumps(summary,sort_keys=True)+')META";\n')
        command=compiler+flags+['-I'+str(output),'-DSPLASH_VERIFY_CANDIDATE='+str(int(role=='candidate')),str(private/'oracle.mm'),*[x['path'] for x in objects],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(output/'oracle')]
        publish(output/'compiler-command.json',command);subprocess.run(command,cwd=ROOT,check=True)
        cpu=subprocess.run([str(output/'oracle'),'--cpu-only'],cwd=ROOT,check=True,capture_output=True,text=True);publish(output/'cpu-self-test.json',json.loads(cpu.stdout))
        meta.update({'pass':True,'oracle_sha256':sha(output/'oracle'),'cpu':json.loads(cpu.stdout),'compiler_command':command});publish(output/'manifest.json',meta)
        provenance[role]={'oracle':str(output/'oracle'),'oracle_sha256':meta['oracle_sha256'],'manifest_sha256':sha(output/'manifest.json'),'cpu':meta['cpu']}
    provenance['pass']=True;publish(build/'CPU_READY.json',provenance)
    print(json.dumps({'pass':True,'build':str(build),'CPU_READY_sha256':sha(build/'CPU_READY.json'),'header_consumers':[x['object'] for x in census if x['modified_header_consumer']],'control':provenance['control'],'candidate':provenance['candidate'],'GPU_work':False}))


if __name__=='__main__':main()
