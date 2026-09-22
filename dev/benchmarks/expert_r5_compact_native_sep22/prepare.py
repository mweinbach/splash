#!/usr/bin/env python3
"""CPU preparation only: new INTEGER AIR, authenticated current floating AIR."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.relative_to(ROOT)
PARENT=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
BASE=ROOT/'dev/benchmarks/expert_batch_compact_native_sep22'
PARENT_LIBRARY='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
PROBE=ROOT/'build/prefill4k-allrows-qmv-one-layer/probe.air'
PROBE_SHA='f92914ca3ba6c7c02eb33c5774c17eb693a7136d109b3dc01dc65e255a113c30'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(argv,log):
    log.append(argv)
    subprocess.run(argv,cwd=ROOT,check=True)

def inverse_journal(old,new):
    before,after=old.splitlines(keepends=True),new.splitlines(keepends=True)
    edits=[]
    for tag,i,j,a,b in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
        if tag!='equal':edits.append({'old_start':i,'old_end':j,'new_start':a,'new_end':b,'old_lines':before[i:j],'new_lines':after[a:b]})
    restored=list(after)
    for edit in reversed(edits):
        a,b=edit['new_start'],edit['new_end']
        if restored[a:b]!=edit['new_lines']:raise ValueError('Ambiguous source restoration')
        restored[a:b]=edit['old_lines']
    if restored!=before:raise ValueError('Literal source restoration failed')
    return edits

def body(text,start,end):return text[text.index(start):text.index(end,text.index(start))]

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output',type=Path,default=ROOT/'build/expert-r5-compact-native-sep22-component-v1')
    args=parser.parse_args(argv);out=args.output.resolve()
    if out.exists()or ROOT/'build'not in out.parents:raise ValueError('Fresh private build required')
    manifest=json.loads((PARENT/'overlay-manifest.json').read_text())
    if sha(PARENT/'splash.metallib')!=PARENT_LIBRARY:raise ValueError('Qualified current library drift')
    sources=[];links=[];commands=[]
    def copy_source(path,rel,expected=None):
        if expected and sha(path)!=expected:raise ValueError('Frozen source drift:'+str(path))
        target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        sources.append({'path':str(rel),'sha256':sha(target)})
    for record in manifest['files']:copy_source(PARENT/'source'/record['path'],Path(record['path']),record['sha256'])
    for path in HERE.iterdir():
        if path.is_file():copy_source(path,PRIVATE/path.name)
    for name in ('prefill4k_allrows_qmv_oracle.mm','prefill4k_allrows_qmv_reference.hpp','prefill4k_allrows_qmv_one_layer.hpp','prefill4k_allrows_qmv_probe.h'):
        copy_source(ROOT/'dev/benchmarks'/name,Path('dev/benchmarks')/name)
    copy_source(ROOT/'dev/benchmarks/expert_r4_cohort_sep22/quality.hpp',PRIVATE/'quality.hpp')
    copy_source(ROOT/'build/prefill4k-allrows-qmv-c2-component/source/runtime/flash/FlashGatheredI8QMV.hpp',Path('runtime/flash/FlashGatheredI8QMV.hpp'))
    journals={}
    for name in ('abi.hpp','plan.metal','metadata.hpp','safety.hpp','oracle.mm','probe.metal'):
        journals[name]={'base_path':str(BASE/name),'base_sha256':sha(BASE/name),'edits':inverse_journal((BASE/name).read_text(),(HERE/name).read_text()),'inverse_literal_exact':True}
    old=(BASE/'oracle.mm').read_text();new=(HERE/'oracle.mm').read_text()
    same_float={name:body(old,start,end)==body(new,start,end)for name,start,end in (
        ('gate','void gate(','void downTail('),('down','void downTail(','void nativeGraph('),
        ('native_probe','void nativeProbe(','struct SampleReport'),('F64_reference','SampleReport f64(','struct NativeSnapshot'))}
    if not all(same_float.values()):raise ValueError('Original floating observer/emitter body changed')
    (out/'source-restoration.json').write_text(json.dumps({'files':journals,'original_floating_functions_literal':same_float},indent=2)+'\n')
    identity=hashlib.sha256(json.dumps({'sources':sources,'parent_library':PARENT_LIBRARY,'fixed_rows':5},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    header=out/'source'/PRIVATE/'source_identity.hpp';header.write_text('#pragma once\ninline constexpr char kCompactNativeR5SourceIdentitySha256[]="'+identity+'";\n')
    def freeze(path,kind,expected=None):
        if expected and sha(path)!=expected:raise ValueError('Frozen binary drift:'+str(path))
        target=out/'reused'/kind/f'{len(links):03d}-{path.name}';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        links.append({'path':str(target.relative_to(out)),'source':str(path),'sha256':sha(target),'kind':kind});return target
    frozen={r['source']:r['sha256']for r in manifest['frozen_inputs']}
    objects=[];object_paths=set()
    for record in manifest['compiled_objects']:
        path=PARENT/record['path']
        if path.name!='FlashWorker.o':
            objects.append(freeze(path,'core'if'reused/core'in str(path)else'host',record['sha256']));object_paths.add(path)
    for path in sorted((PARENT/'reused/core').glob('*.o')):
        if path not in object_paths:objects.append(freeze(path,'core',frozen[str(path)]));object_paths.add(path)
    metal=[c for c in manifest['compiler_commands']if 'metallib'in c][-1]
    airs=[Path(path)for path in metal[4:metal.index('-o')]]
    if not airs or any(path.suffix!='.air'for path in airs):raise ValueError('Parent ordered AIR tuple unavailable')
    original_airs=[freeze(path,'air',frozen.get(str(path)))for path in airs]
    baseline=out/'baseline-relinked.metallib';run(['xcrun','-sdk','macosx','metallib',*map(str,original_airs),'-o',str(baseline)],commands)
    if sha(baseline)!=PARENT_LIBRARY:raise ValueError('Literal ordered current AIR relink differs')
    probe=freeze(PROBE,'probe',PROBE_SHA)
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-ffp-contract=off','-fno-fast-math','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-DEXPERT_R5_ROWS=5','-fobjc-arc',
           '-I'+str(out/'source'),'-I'+str(out/'source/runtime'),'-I'+str(out/'source'/PRIVATE),'-I'+str(out/'source/dev/benchmarks'),'-I'+str(out/'source/dev/benchmarks/prefill4k_attention')]
    run(['xcrun','-sdk','macosx','clang++',*flags,str(out/'source'/PRIVATE/'oracle.mm'),*map(str,objects),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(out/'oracle')],commands)
    air=out/'R5-integer-only.air'
    run(['xcrun','-sdk','macosx','metal','-DEXPERT_R5_ROWS=5','-std=metal4.1','-O3','-Wall','-Wextra','-Werror','-I'+str(out/'source'),'-I'+str(out/'source/runtime'),'-I'+str(out/'source'/PRIVATE),'-mmacosx-version-min=27.0','-c',str(out/'source'/PRIVATE/'plan.metal'),'-o',str(air)],commands)
    run(['xcrun','-sdk','macosx','metallib',*map(str,original_airs),str(probe),str(air),'-o',str(out/'splash.metallib')],commands)
    cpu=subprocess.run([str(out/'oracle'),'--cpu-self-test'],cwd=ROOT,check=True,capture_output=True,text=True)
    (out/'CPU-self-test.jsonl').write_text(cpu.stdout)
    cpu_records=[json.loads(line)for line in cpu.stdout.splitlines()if line.strip()]
    seal={'schema':'R5-integer-original-native-component-source-v1','pass':True,'source_identity_sha256':identity,'sources':sources,'generated_header_sha256':sha(header),'link_inputs':links,'parent_library_sha256':PARENT_LIBRARY,'literal_parent_AIR_relink_sha256':sha(baseline),'original_floating_functions_literal':same_float,'source_restoration_sha256':sha(out/'source-restoration.json'),'compiler_commands':commands,'artifacts':{name:sha(out/name)for name in ('oracle','splash.metallib','R5-integer-only.air')},'CPU_self_test':cpu_records,'GPU_work':False,'model_operand_capture_payload_reads':0,'new_floating_AIR_compiles':0,'new_integer_AIR_compiles':1,'fixed_rows':5,'route_count':50,'native_job_capacity':515,'Root_only_selected_coefficient_byte_bound':246528000,'Root_only_reserved_bytes':384<<20,'primitive_actual_qualified':False,'Worker_or_original22_qualified':False,'Root_independent_source_review_pending':True}
    (out/'CPU_READY.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({'CPU_READY':str(out/'CPU_READY.json'),'sha256':sha(out/'CPU_READY.json'),'GPU_work':False,'native':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
