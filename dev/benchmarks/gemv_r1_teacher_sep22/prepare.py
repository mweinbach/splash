#!/usr/bin/env python3
"""Seal standard-only existing R1 vector on TeacherV5; CPU/code inputs only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path('dev/benchmarks/gemv_r1_teacher_sep22')
VECTOR_PRIVATE = Path('dev/benchmarks/gemv_decode_r1_worker_sep21')
PARENT_MANIFEST = 'fe362f3cc3c50485c51c01f661946f38b49eb9d5db5be06824b0be5222988599'
PARENT_WORKER = '5229511c716af33d163d2f077411cf54d39241925daf99e156cc7f8be532bebc'
PARENT_LIBRARY = 'bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0'
VECTOR_AIR = '1655c704f6eb44bfe2f7f084de7446fc6faca9816f53871a9d9a92950cc5a6ba'
VECTOR_ID = 'bd384554f00dafdc285bf6df2dd34bebfe2d963fa1c955e0834c8a08e656f29b'
EXPECTED_OWN = {'002-FlashInt8ExpertStore', '004-FlashBatchPrefill', '008-FlashBatchForward',
                '010-FlashBatchVerify', 'FlashForward', 'FlashWorker'}


def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(path.read_bytes())
def write(path, data): path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data)
def once(text, before, after, count=1):
    if text.count(before) != count: raise ValueError('Sealed standard R1 source drift: ' + before[:120])
    return text.replace(before, after)


def transform(relative, text, old):
    text = old.transform(relative, text)
    if relative in old.CHANGED_PATHS:
        text = f'#include "{PRIVATE}/scope.hpp"\n' + text
    if relative in ('runtime/flash/FlashForward.hpp', 'runtime/flash/FlashForward.cpp'):
        text = text.replace('forwardDecode', 'forwardStandardDecode').replace('ordinaryDecode', 'standardDecode')
    if relative == 'runtime/flash/FlashForward.cpp':
        before = '''    const bool vectorR1Decode = standardDecode && gatheredMPP &&
        gemv_decode_r1_sep21::eligible(rows,verification) && impl_->int8ExpertStore->gemvDecodeR1Enabled();'''
        text = once(text, before, '''    const bool vectorR1Decode = gemv_r1_teacher_sep22::selected(
        standardDecode,rows,verification,gatheredMPP) && impl_->int8ExpertStore->gemvDecodeR1Enabled();''')
    if relative == 'runtime/flash/FlashWorker.mm':
        text = once(text, '      const auto result = forward_.forwardDecode(*request.state, token);',
            '''      // MTP-owned terminal AR fallback retains the qualified TeacherV5 route.
      const auto result = request.mtpState ? forward_.forward(*request.state, token)
          : forward_.forwardStandardDecode(*request.state, token);''')
        text = text.replace('ordinary singleton autoregressive call only; graph construction not GPU completion; synthetic R1 component qualified only',
            'standard singleton R1 without mtpState; graph construction not GPU completion; inherited numerical certificate; current input/model qualification pending')
    if relative in ('runtime/flash/FlashForward.cpp', 'runtime/flash/FlashInt8ExpertStore.mm'):
        text = text.replace('gemv_decode_r1_sep21::implementationMarker()', 'gemv_r1_teacher_sep22::implementationMarker()')
    return text


def load_old(vector):
    path = vector / 'source' / VECTOR_PRIVATE / 'worker_prepare.py'
    spec = importlib.util.spec_from_file_location('sealed_r1_transform', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=ROOT / 'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5')
    p.add_argument('--vector', type=Path, default=ROOT / 'build/gemv-r1-ab-qsa-sep21-worker-v1')
    p.add_argument('--build', type=Path, default=ROOT / 'build/gemv-r1-teacher-sep22-worker-v1')
    a=p.parse_args();parent, vector, build=(x.resolve() for x in (a.parent,a.vector,a.build))
    if build.exists() or ROOT/'build' not in build.parents: raise ValueError('Fresh private build required')
    manifest_path=parent/'overlay-manifest.json';pm=json.loads(manifest_path.read_text())
    seal_path=parent/'compiled-cpu-seal.json';seal=json.loads(seal_path.read_text())
    if not seal['pass'] or file_sha(manifest_path)!=PARENT_MANIFEST or seal['source_manifest_sha256']!=PARENT_MANIFEST:
        raise ValueError('Qualified TeacherV5 source seal differs')
    if file_sha(parent/'splash-flash')!=PARENT_WORKER or file_sha(parent/'splash.metallib')!=PARENT_LIBRARY:
        raise ValueError('Qualified TeacherV5 runtime differs')
    vm=json.loads((vector/'overlay-manifest.json').read_text())
    if vm['vector_r1_source_identity_sha256']!=VECTOR_ID or file_sha(vector/'vector-r1.air')!=VECTOR_AIR:
        raise ValueError('Existing qualified R1 implementation differs')
    old=load_old(vector);old_transform=vector/'source'/VECTOR_PRIVATE/'worker_prepare.py'
    old_records={r['path']:r for r in vm['files']}
    if file_sha(old_transform)!=old_records[str(VECTOR_PRIVATE/'worker_prepare.py')]['overlay_sha256']:
        raise ValueError('Old sealed source transform differs')
    records=[];changed=[]
    for row in pm['files']:
        relative=row['path'];original=(parent/'source'/relative).read_bytes()
        if sha(original)!=row['sha256'] or seal['source_sha256'][relative]!=sha(original):
            raise ValueError('Teacher source drift: '+relative)
        data=transform(relative,original.decode(),old).encode();write(build/'source'/relative,data)
        records.append({'path':relative,'parent_sha256':sha(original),'sha256':sha(data),'changed':data!=original})
        if data!=original:changed.append(relative)
    if set(changed)!=old.CHANGED_PATHS:raise ValueError('Unexpected standard R1 source edits')
    for row in vm['files']:
        relative=Path(row['path'])
        if relative.parts[:len(VECTOR_PRIVATE.parts)]!=VECTOR_PRIVATE.parts:continue
        data=(vector/'source'/relative).read_bytes()
        if sha(data)!=row['overlay_sha256']:raise ValueError('R1 source/certificate drift: '+str(relative))
        write(build/'source'/relative,data);records.append({'path':relative.as_posix(),'sha256':sha(data),'vector_snapshot':True})
    for name in ('scope.hpp','policy_cpu.cpp'):
        relative=PRIVATE/name;data=(ROOT/relative).read_bytes();write(build/'source'/relative,data)
        records.append({'path':relative.as_posix(),'sha256':sha(data),'new_standard_scope':True})
    # The CURRENT 49 original TUs plus teacher bulk are the full host source set.
    source_rows=list(pm['rebuild'])+[{'object':'teacher_bulk','source':'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'}]
    if len(source_rows)!=50 or len({r['object'] for r in source_rows})!=50:raise ValueError('Teacher TU inventory differs')
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc',
           '-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1',
           '-I'+str(build/'source'),'-I'+str(build/'source/runtime'),'-I'+str(build/'source/dev/benchmarks/prefill4k_attention')]
    suffixes=('/runtime/flash/FlashForward.hpp','/runtime/flash/FlashInt8ExpertStore.hpp')
    def dependencies(row):
        src=build/'source'/row['source']
        run=subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(src)],cwd=ROOT,text=True,capture_output=True,check=True)
        tokens=run.stdout.replace('\\\n',' ').split()[1:]
        hits=[x for x in tokens if x.endswith(suffixes)]
        return {**row,'modified_header_consumers':hits,'dependencies':tokens}
    with ThreadPoolExecutor(max_workers=6) as pool:census=list(pool.map(dependencies,source_rows))
    own={r['object'] for r in census if r['modified_header_consumers']}
    if own!=EXPECTED_OWN:raise ValueError('Actual modified-header closure differs: '+str(sorted(own)))
    inputs={'REUSED':[],'CORE':[],'AIRS':[]};pins=[]
    def freeze(src,relative,category,expected=None):
        data=src.read_bytes()
        if expected and sha(data)!=expected:raise ValueError('Authenticated input drift: '+str(src))
        write(build/relative,data);inputs[category].append(relative.as_posix())
        pins.append({'source_path':str(src),'private_path':relative.as_posix(),'category':category,'sha256':sha(data)})
    for row in source_rows:
        src=parent/'host'/(row['object']+'.o');expected=seal['artifact_sha256']['host/'+row['object']+'.o']
        if row['object'] not in own:freeze(src,Path('reused/current-host')/src.name,'REUSED',expected)
        elif file_sha(src)!=expected:raise ValueError('Replaced current host input drift')
    for line in (parent/'link-inputs.mk').read_text().splitlines():
        if not line.startswith('CORE := '):continue
        for token in line.split(' := ',1)[1].split():
            relative=Path(token.removeprefix('$(BUILD)/'));src=parent/relative
            freeze(src,Path('reused/current-core')/src.name,'CORE',seal['artifact_sha256'][str(relative)])
    if len(inputs['REUSED'])!=44 or len(inputs['CORE'])!=4:raise ValueError('Current host/Core closure differs')
    # AB's own candidate was linked FIRST in the authenticated BB09 recipe.
    # The old R1 manifest stores it last, so restore the actual original order.
    airs=[r for r in vm['r1_ab_link_inputs'] if r['category']=='AIRS']
    ab=next(r for r in airs if r['original']==str(Path(pm['base'])/'candidate.air'))
    ordered=[ab]+[r for r in airs if r is not ab]
    if len(ordered)!=76:raise ValueError('Exact original AIR set differs')
    for index,row in enumerate(ordered):
        freeze(vector/row['private_path'],Path('reused/original-air')/f'{index:03d}-{Path(row["private_path"]).name}',
               'AIRS',row['sha256'])
    baseline=build/'baseline-original76.metallib'
    subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(build/x) for x in inputs['AIRS']],'-o',str(baseline)],cwd=ROOT,check=True)
    if file_sha(baseline)!=PARENT_LIBRARY:raise ValueError('Original76 ordered relink must reproduce Teacher BB09 exactly')
    freeze(vector/'vector-r1.air',Path('reused/qualified-vector/vector-r1.air'),'AIRS',VECTOR_AIR)
    # Read-only primitive/runtime evidence and original identity/cert snapshot.
    for src,name in [(vector/'qualification/r1-component.json','r1-component.json'),
                    (ROOT/'build/release/flash/sep21-r1-vector-standard-on-off-comparison-v1.json','historical-r1-on-off.json'),
                    (ROOT/'build/release/flash/sep22-ab-qsa-vs-ar1-vector-standard-semantic-comparison-v1.json','historical-standard22-comparison.json')]:
        write(build/'qualification'/name,src.read_bytes())
    make='\n'.join(k+' := '+' '.join('$(BUILD)/'+x for x in vals) for k,vals in inputs.items())+'\n'
    make+='OWN_NAMES := '+' '.join(r['object'] for r in source_rows if r['object'] in own)+'\n'
    for row in source_rows:
        if row['object'] in own:make+='SRC_'+row['object']+' := $(BUILD)/source/'+row['source']+'\n'
    write(build/'link-inputs.mk',make.encode())
    manifest={'schema':'sealed-TeacherV5-strict-standard-R1-numerical-alternative-v1',
        'parent':str(parent),'parent_manifest_sha256':PARENT_MANIFEST,'parent_compiled_seal_sha256':file_sha(seal_path),
        'parent_runtime_sha256':{'splash-flash':PARENT_WORKER,'splash.metallib':PARENT_LIBRARY},
        'vector':str(vector),'vector_manifest_sha256':file_sha(vector/'overlay-manifest.json'),
        'vector_source_identity_sha256':VECTOR_ID,'vector_air_sha256':VECTOR_AIR,
        'scope_schema':'strict-standard-singleton-R1-no-mtpState-no-prefill-no-verify-no-head-no-batch-v1',
        'flag':'SPLASH_FLASH_GEMV_DECODE_R1_SEP21','default0_original_identity_and_routes':True,
        'arithmetic_numerical_alternative':True,'numerical_certificate_scope':'frozen v1b sampled RN/RTZ/FTZ/F64 and per-route stage bounds, no MPP bit parity claim',
        'prepare_sha256':file_sha(Path(__file__)),'old_transform_sha256':file_sha(old_transform),
        'files':records,'changed_paths':changed,'header_dependency_census':census,'rebuild':[r for r in source_rows if r['object'] in own],
        'frozen_inputs':pins,'link_make_sha256':sha(make.encode()),'original76_ordered_relink_sha256':file_sha(baseline),
        'original76_count':76,'additional_vector_air_count':1,'host_count':50,'core_count':4,
        'added_gpu_backing_and_workspace_bytes':0,'teacher_source_and_arena_unchanged':True,
        'GPU_proof_and_current_standard_model_qualification':'pending Root before benchmark/promotion',
        'gpu_executed':False,'model_payload_bytes_read':0,'production_sources_modified':False}
    write(build/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps({'prepared':str(build),'sources':len(records),'header_TUs':len(census),'rebuild':sorted(own),
                      'frozen_current_host':44,'core':4,'original76_baseline_exact':True,'vector_air_unchanged':True,
                      'gpu_work':False,'model_payload_bytes_read':0}))


if __name__=='__main__':main()
