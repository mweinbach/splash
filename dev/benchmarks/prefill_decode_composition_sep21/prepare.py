#!/usr/bin/env python3
"""CPU-only phase worker snapshot from the actual sealed teacher link."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,re,shlex,subprocess
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/prefill_decode_composition_sep21')
HYBRID=ROOT/'build/hybrid-sg2-tail-fma-sep21-worker-v1'
PRUNED=ROOT/'build/dense-w8a8-residency-prune-sep21-worker-v1'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5');ap.add_argument('--output',type=Path,default=ROOT/'build/prefill-i8-decode-q4-teacher-sep22-worker-v1');a=ap.parse_args();base=a.base.resolve();out=a.output.resolve()
    if out.exists() or ROOT/'build'not in out.parents:raise ValueError('Choose fresh private build')
    parentPath=base/'overlay-manifest.json';parent=json.loads(parentPath.read_text());sealPath=base/'compiled-cpu-seal.json';seal=json.loads(sealPath.read_text())
    if not seal['pass'] or seal['effective_objects']!=54 or sha(parentPath.read_bytes())!=seal['source_manifest_sha256']:raise ValueError('Sealed exact teacher parent required')
    for rel,digest in seal['source_sha256'].items():
        if sha((base/'source'/rel).read_bytes())!=digest:raise ValueError('Parent source drift:'+rel)
    for rel,digest in seal['artifact_sha256'].items():
        if sha((base/rel).read_bytes())!=digest:raise ValueError('Parent artifact drift:'+rel)
    result=subprocess.run(['make','-pqn','-rR','-f',str(base/'machinery/worker.mk'),f'BUILD={base}',str(base/'splash-flash')],cwd=ROOT,capture_output=True,text=True,timeout=30)
    if result.returncode not in (0,1):raise ValueError('Parent actual make closure failed')
    target=str(base/'splash-flash')+':';lines=[l for l in result.stdout.splitlines()if l.startswith(target)]
    if len(lines)!=1:raise ValueError('Ambiguous effective parent link')
    objects=[Path(x).resolve()for x in shlex.split(lines[0].split(':',1)[1])if x.endswith('.o')]
    coreLine=re.findall(r'^CORE\s*:=\s*(.*)$',result.stdout,re.M)
    if len(coreLine)!=1:raise ValueError('Parent CORE classification ambiguous')
    core={Path(x).resolve()for x in shlex.split(coreLine[0])}
    if len(objects)!=54 or len(set(objects))!=54 or len(core)!=4:raise ValueError('Effective54-object closure differs')
    if set(str(x.relative_to(base))for x in objects)!={x for x in seal['artifact_sha256']if x.endswith('.o')}:raise ValueError('Actual parent link and sealed objects differ')
    hybrid=json.loads((HYBRID/'overlay-manifest.json').read_text());hybridRecords={r['path']:r for r in hybrid['files']}
    for rel in ('runtime/flash/FlashWeights.mm','runtime/flash/FlashWeights.hpp'):
        if sha((HYBRID/'source'/rel).read_bytes())!=hybridRecords[rel]['overlay_sha256']:raise ValueError('Hybrid loader source drift')
    prune=json.loads((PRUNED/'overlay-manifest.json').read_text());pruneRecords={r['path']:r for r in prune['files']}
    bridge='dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp'
    if sha((PRUNED/'source'/bridge).read_bytes())!=pruneRecords[bridge]['overlay_sha256']:raise ValueError('Residency selector source drift')
    transformer=load(ROOT/PRIVATE/'transform.py','private_phase_transform');files=[]
    for r in parent['files']:
        rel=r['path'];data=(base/'source'/rel).read_bytes();changed=transformer.transform(rel,data.decode(),HYBRID/'source').encode()
        write(out/'source'/rel,changed);files.append({'path':rel,'sha256':sha(changed),'parent_sha256':sha(data),'changed':changed!=data})
    for rel,path in [(PRIVATE/'phase.hpp',ROOT/PRIVATE/'phase.hpp'),(Path(bridge),PRUNED/'source'/bridge)]:
        data=path.read_bytes();write(out/'source'/rel,data);files.append({'path':rel.as_posix(),'sha256':sha(data),'new':True,'origin':str(path)})
    sourceNames={r['path']for r in files};aliases={'teacher_bulk':'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp','Prefill4kQSABulk':'dev/benchmarks/prefill4k_attention/bulk.cpp','Prefill4kQSACoalesced':'dev/benchmarks/prefill4k_attention/coalesced.cpp'}
    rebuild=[];frozen=[];coreInputs=[]
    for obj in objects:
        if obj in core:
            rel=Path('reused/core')/obj.name;data=obj.read_bytes();write(out/rel,data);coreInputs.append(rel.as_posix());frozen.append({'path':rel.as_posix(),'sha256':sha(data),'parent_path':str(obj)});continue
        stem=re.sub(r'^\d{3}-','',obj.stem)
        src=aliases.get(stem)or next((x for x in sourceNames if Path(x).stem==stem and x.endswith(('.cpp','.mm'))),None)
        if not src:raise ValueError('Unresolved actual parent TU:'+str(obj))
        rebuild.append({'object':obj.stem,'source':src,'parent_object_sha256':sha(obj.read_bytes()),'parent_path':str(obj)})
    lib=(base/'splash.metallib').read_bytes();write(out/'splash.metallib',lib)
    link='REBUILD_NAMES := '+' '.join(r['object']for r in rebuild)+'\nCORE := '+' '.join('$(BUILD)/'+x for x in coreInputs)+'\n'
    for r in rebuild:link+='SRC_'+r['object']+' := $(BUILD)/source/'+r['source']+'\n'
    write(out/'link-inputs.mk',link.encode())
    tools={}
    for name in ('prepare.py','transform.py','phase.hpp','worker.mk','plan.md'):
        data=(ROOT/PRIVATE/name).read_bytes();write(out/'machinery'/name,data);tools[name]=sha(data)
    for name in ('splash-flash.config','splash.metallib.config'):
        if(base/name).exists():write(out/name,(base/name).read_bytes())
    manifest={'schema':'private-all-prefill-i8-explicit-decode-verify-q4-teacher-v1','base':str(base),'base_source_manifest_sha256':sha(parentPath.read_bytes()),'base_compiled_cpu_seal_sha256':sha(sealPath.read_bytes()),'base_runtime_sha256':sha((base/'splash-flash').read_bytes()),'base_metallib_sha256':sha(lib),'rebuild':rebuild,'frozen_core':frozen,'files':files,'machinery_sha256':tools,'actual_parent_effective_objects':[str(x)for x in objects],'all50parentHostTUs_rebuilt':True,'new_shader_or_extra_decode_workspace':False,'f32_backing_count':296,'f32_backing_bytes':12097945600,'f32_persistent_count':118,'f32_persistent_bytes':3247964160,'f32_transient_only_count':178,'f32_transient_only_bytes':8849981440,'expected_resident_owner_count_denseW8_on':832,'expected_resident_bytes_denseW8_on':200368848896,'fresh_governor_fit_proven':False,'prefill_math_changed':False,'phase_numerical_derivative_changed':True,'teacher_cache_math_changed':False,'gpu_executed':False,'model_payload_bytes_read':0}
    write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'output':str(out),'sources':len(files),'host_recompiled':len(rebuild),'core_reused':4,'GPU_executed':False}))
if __name__=='__main__':main()
