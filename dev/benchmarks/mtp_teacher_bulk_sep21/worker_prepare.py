#!/usr/bin/env python3
"""Freeze exact-current-kernel teacher bulk component, CPU/source/artifacts only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/mtp_teacher_bulk_sep21')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Frozen teacher source drift: '+before[:90])
    return text.replace(before,after,1)
def transform(relative,text):
    path=Path(__file__).with_name('worker_transform.py')
    spec=importlib.util.spec_from_file_location('teacher_whole_transform',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.transform(relative,text)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=ROOT/'build/gdn-ab-merge-qsa-sep21-worker-v1')
    ap.add_argument('--output',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v2')
    args=ap.parse_args();base=args.base.resolve();out=args.output.resolve()
    if out.exists() or ROOT/'build' not in out.parents:raise ValueError('Choose fresh private output')
    parent=json.loads((base/'overlay-manifest.json').read_text())
    sealPath=base/'cpu-source-witness.json';compiled=json.loads(sealPath.read_text())
    if not compiled.get('pass') or compiled['source_files_verified']!=276:raise ValueError('Exact AB-QSA source witness required')
    expected={'splash-flash':'d4e9efb3249308151d16a3796100421e06b5a888318f97d25ec821a68846ada6',
              'splash.metallib':'bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0'}
    for rel,digest in expected.items():
        if sha((base/rel).read_bytes())!=digest or compiled['runtime_sha256'][rel]!=digest:raise ValueError('Parent compiled artifact differs: '+rel)
    for r in parent['gdn_ab_frozen_inputs']:
        if sha((base/r['private_path']).read_bytes())!=r['sha256']:raise ValueError('Parent frozen input differs')
    if sha((base/'link-inputs.mk').read_bytes())!=parent['gdn_ab_link_make_sha256']:raise ValueError('Parent link closure differs')
    ancestor=Path(parent['gdn_ab_base_build']);hp=ancestor/'machinery/parent_overlay.py'
    spec=importlib.util.spec_from_file_location('teacher_parent',hp);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    closure=helper.effective_closure(base,ROOT/'dev/benchmarks/gdn_ab_merge_sep21/worker.mk')
    seals={str((base/r['private_path']).resolve()):r['sha256']for r in parent['gdn_ab_frozen_inputs']}
    unsealed=[]
    for obj in closure['objects']:
        if str(obj.resolve())not in seals:
            if obj in closure['core']:raise ValueError('Unsealed inherited Core object')
            unsealed.append({'path':str(obj),'sha256':sha(obj.read_bytes()),'scope':'parent-owned host object; fully recompiled from authenticated source'})
    witness={'source_path':str(sealPath),'sha256':sha(sealPath.read_bytes()),'runtime_sha256':expected}
    records={x['path']:x for x in parent['files']}
    deps=helper.frozen_dependency_closure(base,closure['objects'],records)
    files=[]
    for rel,r in records.items():
        data=(base/'source'/rel).read_bytes()
        if sha(data)!=r['overlay_sha256']:raise ValueError('Parent source seal differs: '+rel)
        changed=transform(rel,data.decode()).encode()
        write(out/'source'/rel,changed);files.append({'path':rel,'parent_sha256':sha(data),'sha256':sha(changed),'changed':changed!=data})
    for name in ('bulk.hpp','bulk.cpp','policy_cpu.cpp','policy.hpp'):
        rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data)
        files.append({'path':rel.as_posix(),'sha256':sha(data),'new':True})
    inputs={'REUSED':[],'CORE':[]};frozen=[];rebuild=[]
    aliases={'Prefill4kQSABulk':'dev/benchmarks/prefill4k_attention/bulk.cpp',
             'Prefill4kQSACoalesced':'dev/benchmarks/prefill4k_attention/coalesced.cpp'}
    for obj in closure['objects']:
        # Many inherited objects intentionally have no adjacent.d. Recompile
        # every non-core host TU against the single modified class definition,
        # rather than guessing which header consumers are ABI-safe to reuse.
        if obj not in closure['core']:
            stem=obj.stem.split('-',1)[1]if obj.stem[:3].isdigit()and '-'in obj.stem else obj.stem
            src=aliases.get(stem) or next((x for x in records if Path(x).stem==stem and x.endswith(('.cpp','.mm'))),None)
            if not src:raise ValueError('Cannot resolve affected source '+str(obj))
            rebuild.append({'object':obj.stem,'source':src,'parent_object':str(obj)});continue
        rel=Path('reused')/obj.relative_to(base);data=obj.read_bytes();write(out/rel,data)
        kind='CORE' if obj in closure['core'] else 'REUSED';inputs[kind].append(rel.as_posix())
        frozen.append({'path':rel.as_posix(),'sha256':sha(data),'source':str(obj)})
    lib=(base/'splash.metallib').read_bytes();write(out/'splash.metallib',lib)
    for name in ('worker_prepare.py','prepare.py','worker_transform.py','worker.mk','README.md'):write(out/'machinery'/name,(ROOT/PRIVATE/name).read_bytes())
    link='\n'.join(k+' := '+' '.join('$(BUILD)/'+p for p in v)for k,v in inputs.items())+'\n'
    link+='REBUILD_NAMES := '+' '.join(x['object']for x in rebuild)+'\n'
    for r in rebuild:link+='SRC_'+r['object']+' := $(BUILD)/source/'+r['source']+'\n'
    write(out/'link-inputs.mk',link.encode())
    man={'schema':'singleton-teacher-exact-bulk2048-whole-worker-v1','base':str(base),'parent_compiled_seal_sha256':sha(sealPath.read_bytes()),'parent_source_manifest_sha256':sha((base/'overlay-manifest.json').read_bytes()),'parent_helper_sha256':sha(hp.read_bytes()),'parent_input_seals':seals,'parent_unsealed_owned_inputs':unsealed,'parent_artifact_witness':witness,'parent_dependencies':deps,'rebuild':rebuild,'frozen_objects':frozen,'metallib_sha256':sha(lib),'files':files,'all_noncore_host_sources_recompiled':True,'additional_workspace_planned_bytes':310181888,'gpu_executed':False,'model_payload_bytes_read':0,'head_proposal_and_batch_paths_changed':False,'parent_files':parent['files'],'component_source_manifest_sha256':sha((ROOT/'build/mtp-teacher-bulk-sep21-v4/source-manifest.json').read_bytes()),'component_gpu_report_sha256':sha((ROOT/'build/release/flash/sep21-teacher-bulk2048-cache-future-and-sequence-v1.json').read_bytes()),'numerical_derivative_changed':False,'singleton_teacher_calls_are_actual_API':True,'logical_prefix_windows_separate':True}
    write(out/'overlay-manifest.json',(json.dumps(man,indent=2)+'\n').encode())
    print(json.dumps({'output':str(out),'source_count':len(files),'rebuild':rebuild,'frozen_objects':len(frozen),'payload_bytes_read':0}))
if __name__=='__main__':main()
