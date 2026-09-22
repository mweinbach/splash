#!/usr/bin/env python3
"""CPU-only current-composite capture preparation. Never opens data payloads."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c):subprocess.run(c,cwd=ROOT,check=True)
def function(s,name):
    m=re.search(r'\b(?:std::string|void|bool)\s+'+name+r'\(',s)
    if not m:raise ValueError('current policy function missing:'+name)
    a=s.index('{',m.start());depth=1;b=a+1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[m.start():b]
def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve()
    if b.exists():raise ValueError('fresh capture build required')
    base=ROOT/'build/trunkverify-exact-guard-HC-fast-composite-sep22-v1';worker=ROOT/'build/guard-HC-fast-composite-sep22-worker-v1';m=json.loads((base/'manifest.json').read_text());receipt=worker/'Root-native-qualified.json';proof=json.loads(receipt.read_text())
    expected='06ccaac045fadf6122bef528544f866d81e000ffdf5fd8a5b58a529e3a212d6b'
    if not(proof['pass']and proof['qualification_complete']and proof['source_identity_sha256']=='38756086d4862cc9a87db70705928cb2a497450e1268bcb251bb0c0b10d8baba'and proof['newlib_sha256']==expected and m['metallib_sha256']==expected and sha(base/'splash.metallib')==expected):raise ValueError('fresh current composite native qualification/closure differs')
    if len(m['objects'])!=53 or len(m['header_census'])!=50:raise ValueError('exact53-object/current50TU census required')
    shutil.copytree(base,b);own=b/'source/dev/benchmarks/raw_q4_capture_sep22';shutil.copytree(HERE,own)
    # Retain copied history under unambiguous names; never expose the base's
    # oracle/manifest as the new capture artifact or proof.
    (b/'manifest.json').rename(b/'inherited-state-oracle-manifest.json')
    (b/'oracle').rename(b/'inherited-state-oracle-binary-NOT-CAPTURE')
    rp=b/'source/dev/benchmarks/raw_q4_rowpair_sep22';rp.mkdir(parents=True,exist_ok=True)
    for n in ['storage.hpp','policy.hpp']:shutil.copy2(ROOT/'dev/benchmarks/raw_q4_rowpair_sep22'/n,rp/n)
    oldForward=base/'source/runtime/flash/FlashForward.cpp';forward=b/'source/runtime/flash/FlashForward.cpp';old=oldForward.read_text()
    anchor='    impl_->project(graph, prefix, input, output, diag, rows, verification, true);'
    if old.count(anchor)!=1:raise ValueError('actual affine-lambda source anchor differs')
    addition='    raw_q4_capture_sep22::before(graph,prefix,input,rows,verification,begin);\n'+anchor+'\n    raw_q4_capture_sep22::after(graph,prefix,rows,verification,begin);'
    changed='#include "capture_hook.hpp"\n'+old.replace(anchor,addition)
    restored=changed.removeprefix('#include "capture_hook.hpp"\n').replace(addition,anchor)
    if restored!=old:raise ValueError('private hook does not restore original Forward bytes')
    forward.write_text(changed)
    originalOracle=(base/'source/dev/benchmarks/trunk_verify_exact_sep22/oracle.mm').read_text();(own/'current_policy.inc').write_text(function(originalOracle,'selected')+'\n'+function(originalOracle,'commonPolicy')+'\n'+function(originalOracle,'early')+'\n')
    oldBase=str(base);newBase=str(b);oldCommand=m['compiler_command'];flags=oldCommand[4:oldCommand.index(str(base/'source/dev/benchmarks/trunk_verify_exact_sep22/oracle.mm'))];flags=[x.replace(oldBase,newBase)for x in flags]+['-I'+str(own)]
    objects=[];rebuild=[];pins=[]
    for x in m['objects']:
        original=Path(x['path']);dst=Path(str(original).replace(oldBase,newBase));pins.append({'source':str(original),'source_sha256':sha(original),'path':str(dst)});objects.append(dst)
        if re.sub(r'^\d+-','',dst.stem)=='FlashForward':rebuild.append(dst)
        elif sha(dst)!=sha(original):raise ValueError('copied nonaffected object differs')
    if len(rebuild)!=1:raise ValueError('one actual Forward object required')
    commands=[['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(forward),'-o',str(rebuild[0])]];run(commands[-1])
    census=[]
    for entry in m['header_census']:
        source=b/'source'/entry['source'];c=['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(source)];r=subprocess.run(c,cwd=ROOT,check=True,text=True,capture_output=True);tokens=r.stdout.replace('\\\n',' ').split();capture=any(t.endswith('/raw_q4_capture_sep22/capture_hook.hpp')for t in tokens)
        if capture!=(entry['object']=='FlashForward'):raise ValueError('unexpected actual capture-header consumer:'+entry['object'])
        census.append({'object':entry['object'],'source':str(source),'capture_header_consumer':capture,'excluded_worker_main':entry.get('excluded_main',False),'dependencies':tokens})
    for path in ['runtime/flash/FlashForward.hpp','runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashRequestStateInternal.hpp']:
        if sha(b/'source'/path)!=sha(base/'source'/path):raise ValueError('public/state/store header changed:'+path)
    provenance={'schema':'raw-q4-current-input-capture-code-provenance-v1','qualified_parent_worker':str(worker),'qualified_parent_source':proof['source_identity_sha256'],'qualified_parent_executable_sha256':proof['exe_sha256'],'qualified_parent_library_sha256':expected,'qualified_parent_native_receipt':{'path':str(receipt),'sha256':sha(receipt)},'inherited_actual53closure':str(base),'base_state_oracle_sha256':sha(base/'oracle'),'source_hook_restores_Forward_byte_exact':True,'public_headers_changed':False,'shaders_changed':False,'Worker_changed':False,'observer_scope':'one original layer1 QKV current VerifyR4/begin2048; ordinary target greedy inputs not trained MTP proposals','capture_input_logical_bytes':20480,'capture_owner_actual_expected_bytes':32768,'normal_rowpair_integration':False,'GPU_work':False,'payload_reads':0}
    (b/'CaptureProvenance.hpp').write_text('#pragma once\ninline constexpr const char*kCaptureLibrarySHA='+json.dumps(expected)+';\ninline constexpr const char*kCaptureBuildProvenance=R"PROV('+json.dumps(provenance,sort_keys=True)+')PROV";\n')
    hcommand=['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(own/'oracle.mm')];hc=subprocess.run(hcommand,cwd=ROOT,check=True,text=True,capture_output=True)
    if 'capture_hook.hpp'not in hc.stdout:raise ValueError('harness did not consume capture header')
    command=['xcrun','-sdk','macosx','clang++',*flags,str(own/'oracle.mm'),*map(str,objects),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'capture-oracle')];commands.append(command);run(command)
    cpu=subprocess.run([str(b/'capture-oracle'),'--cpu-only'],cwd=ROOT,check=True,capture_output=True,text=True);(b/'capture-CPU.json').write_text(cpu.stdout)
    for n in pins:n['frozen_sha256']=sha(Path(n['path']));n['recompiled_for_capture']=Path(n['path'])==rebuild[0]
    tapeProof={'schema':'current-R4-max4-lanes1-known-tape-source-proof-v1','no_artificial_reseed':True,'each_Verify_fully_overwrites216_logical_regions':True,'initial_history':'full original copy flashGDNConvolutionLaneBytes','rawQKV':'full maxRows4 view copy','initial_recurrence':'48heads x128value x128key all snapshot words written','mixed':'all4 x10240 Q/K/V words written by shared key and head value writers','decay_beta':'all4 x48 head words written','allocation_tails':'constructorA5; read only by inspection/canary checks','full4_commit':'noWork graph fastpath','Prefill_and_futureAR':'nonverification GDN branch, never lazy begin or lazy arena writes','scope_excluded_Prefill_byte_compared':False,'observer_census':'encoded capture; data read only after synchronous completed Verify and following future AR','source_files':[{'path':str(b/'source'/p),'sha256':sha(b/'source'/p)}for p in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashGDNLazyRollback.cpp','runtime/metal/kernels/shared/flash_gdn_lazy_rollback.metal']]}
    (b/'known-tape-source-proof.json').write_text(json.dumps(tapeProof,indent=2)+'\n')
    manifest={'schema':'raw-q4-current-input-capture-CPU-closure-v1','CPU_build_complete':True,'independent_review_complete':False,'Root_GPU_release_pending':True,'GPU_work':False,'model_payload_reads':0,'capture_payload_reads':0,'fixture_payload_reads':0,'inherited_objects':pins,'changed_object_count':1,'new_harness_translation_units':1,'capture_header_consumers':['FlashForward','capture-oracle'],'actual50TU_header_census':census,'harness_dependencies':hc.stdout,'source_files':[{'path':str(p),'sha256':sha(p)}for p in sorted((b/'source').rglob('*'))if p.is_file()],'source_hook_journal':{'original_Forward_sha256':sha(oldForward),'instrumented_Forward_sha256':sha(forward),'restores_original_byte_exact':True,'before_after_addition':addition,'floating_math_modified':False},'known_tape_source_proof_sha256':sha(b/'known-tape-source-proof.json'),'actual_owned_capture_guard_cases_mandatory':6,'compiler_commands':commands,'provenance':provenance,'CPU':json.loads(cpu.stdout),'capture_oracle_sha256':sha(b/'capture-oracle'),'library_sha256':sha(b/'splash.metallib'),'public_headers_and_shader_library_unchanged':True,'actual_input_qualified':False,'whole_worker_rowpair_integration_authorized':False}
    (b/'capture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'CPU_build_complete':True,'GPU_work':False,'payload_reads':0,'build':str(b),'changed_object_count':1,'objects':len(objects),'CPU':manifest['CPU'],'oracle_sha256':manifest['capture_oracle_sha256'],'library_sha256':manifest['library_sha256']}))
if __name__=='__main__':main()
