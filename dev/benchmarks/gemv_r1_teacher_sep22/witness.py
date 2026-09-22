#!/usr/bin/env python3
"""Seal actual current-header closure and standard-only R1 refusal/source proof."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from prepare import ROOT, PRIVATE, VECTOR_PRIVATE, PARENT_LIBRARY, VECTOR_AIR, EXPECTED_OWN, sha, file_sha, transform, load_old


def extract(text, begin, end):
    start=text.index(begin);return text[start:text.index(end,start)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/gemv-r1-teacher-sep22-worker-v1')
    a=p.parse_args();build=a.build.resolve();output=build/'compiled-cpu-seal.json'
    if output.exists():raise ValueError('Fresh compiled witness required')
    man=json.loads((build/'overlay-manifest.json').read_text());parent,vector=Path(man['parent']),Path(man['vector']);old=load_old(vector)
    checks={'parent_manifest_fresh':file_sha(parent/'overlay-manifest.json')==man['parent_manifest_sha256'],
            'parent_compiled_seal_fresh':file_sha(parent/'compiled-cpu-seal.json')==man['parent_compiled_seal_sha256'],
            'prepare_recipe_fresh':file_sha(Path(__file__).with_name('prepare.py'))==man['prepare_sha256'],
            'old_transform_fresh':file_sha(vector/'source'/VECTOR_PRIVATE/'worker_prepare.py')==man['old_transform_sha256']}
    source_sha={};mismatches=[]
    for row in man['files']:
        rel=row['path'];actual=(build/'source'/rel).read_bytes();source_sha[rel]=sha(actual)
        if row.get('new_standard_scope'):expected=(ROOT/rel).read_bytes()
        elif row.get('vector_snapshot'):expected=(vector/'source'/rel).read_bytes()
        else:
            original=(parent/'source'/rel).read_bytes()
            if sha(original)!=row['parent_sha256']:mismatches.append(rel+':parent')
            expected=transform(rel,original.decode(),old).encode()
        if actual!=expected or sha(actual)!=row['sha256']:mismatches.append(rel)
    checks['all_frozen_sources_fresh']=not mismatches
    src=lambda r:(build/'source'/r).read_text();before=lambda r:(parent/'source'/r).read_text()
    worker,forward,header,store=(src('runtime/flash/'+r) for r in ('FlashWorker.mm','FlashForward.cpp','FlashForward.hpp','FlashInt8ExpertStore.mm'))
    original_worker=before('runtime/flash/FlashWorker.mm');original_forward=before('runtime/flash/FlashForward.cpp');original_store=before('runtime/flash/FlashInt8ExpertStore.mm')
    checks.update({
        'worker_mtp_owned_AR_fallback_original': 'const auto result = request.mtpState ? forward_.forward(*request.state, token)\n          : forward_.forwardStandardDecode(*request.state, token);' in worker,
        'only_standard_worker_call_site': worker.count('forward_.forwardStandardDecode(')==1,
        'prefill_teacher_code_unchanged': extract(worker,'  void tick(Request &request)', '    } else if (request.pendingToken)')==extract(original_worker,'  void tick(Request &request)', '    } else if (request.pendingToken)'),
        'mtp_tick_and_seed_code_unchanged': extract(worker,'void Worker::tickMTP(Request &request)', 'void Worker::publishStatus()')==extract(original_worker,'void Worker::tickMTP(Request &request)', 'void Worker::publishStatus()'),
        'forward_impl_stdtag_default_false': 'bool verification, bool standardDecode=false);' in header,
        'new_entry_one_token_nonempty_guard_before_delegate': 'token.size()!=1 || !request.logicalLength()' in forward and 'return forwardImpl(request,token,false,false,false,true);' in forward,
        'original_forward_and_verify_entry_unchanged': extract(forward,'FlashForwardResult FlashForward::forward(', 'FlashForwardResult FlashForward::forwardStandardDecode(')==extract(original_forward,'FlashForwardResult FlashForward::forward(', 'FlashForwardResult FlashForward::verify(') and extract(forward,'FlashForwardResult FlashForward::verify(', 'FlashForwardResult FlashForward::forwardImpl(')==extract(original_forward,'FlashForwardResult FlashForward::verify(', 'FlashForwardResult FlashForward::forwardImpl('),
        'vector_selector_complete_scope': 'gemv_r1_teacher_sep22::selected(\n        standardDecode,rows,verification,gatheredMPP) && impl_->int8ExpertStore->gemvDecodeR1Enabled()' in forward,
        'old_gathered_and_canonical_combine_preserved': 'else if (gatheredMPP)' in forward and 'blocked && !gatheredMPP ? impl_->blockedScratch.scatteredDown' in forward,
        'workspace_and_request_planners_unchanged': extract(forward,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes')==extract(original_forward,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes'),
        'original_gathered_store_methods_unchanged': extract(original_store,'void FlashInt8ExpertStore::addGatheredMPPGateUp(', '} // namespace splash::flash') in store,
        'strict_scope_bound_to_store_numeric_identity': 'gemv_r1_teacher_sep22::implementationMarker()' in extract(store,'    std::string derivative =','    numericalIdentity ='),
        'flag0_derivative_original_except_conditional_addition': '    if (gemvDecodeR1)\n      derivative +=' in store,
        'existing_identity_wrappers_unchanged': extract(worker,'      << R"(,"target_numerical_derivative_sha256":)', '      << R"(,"target_base_numerical_derivative_sha256":)')==extract(original_worker,'      << R"(,"target_numerical_derivative_sha256":)', '      << R"(,"target_base_numerical_derivative_sha256":)'),
        'live_gathered_snapshot_getters_unchanged': extract(store,'bool FlashInt8ExpertStore::gatheredMPPEnabled() const','void FlashInt8ExpertStore::addGatheredMPPGateUp(')==extract(original_store,'bool FlashInt8ExpertStore::gatheredMPPEnabled() const','void FlashInt8ExpertStore::addGatheredMPPGateUp('),
        'no_added_backing_or_workspace': man['added_gpu_backing_and_workspace_bytes']==0 and all(src(r['path']).count('allocateBuffer(')==before(r['path']).count('allocateBuffer(') for r in man['files'] if 'parent_sha256' in r),
        'teacher_bulk_source_unchanged': src('dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp')==before('dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'),
        'original76_relink_exact_bb09': file_sha(build/'baseline-original76.metallib')==PARENT_LIBRARY,
        'vector_arithmetic_not_recompiled': file_sha(build/'reused/qualified-vector/vector-r1.air')==VECTOR_AIR,
        'link_make_fresh':file_sha(build/'link-inputs.mk')==man['link_make_sha256'],
        'all50_header_census_and_exact6_consumers':len(man['header_dependency_census'])==50 and {r['object'] for r in man['header_dependency_census'] if r['modified_header_consumers']}==EXPECTED_OWN,
    })
    frozen=[r for r in man['frozen_inputs'] if file_sha(build/r['private_path'])!=r['sha256']]
    checks['all_frozen_current_objects_core_and_air_fresh']=not frozen
    deps=[];live=[]
    for row in man['rebuild']:
        path=build/'host'/(row['object']+'.d');text=path.read_text().replace('\\\n',' ');first=text.splitlines()[0]
        owned=[]
        for token in first.split(': ',1)[1].split():
            f=Path(token)
            if 'build/gemv-r1-teacher-sep22-worker-v1/source/' in token:
                rel=token.split('/source/',1)[1]
                if rel not in source_sha:live.append(token)
                else:owned.append(rel)
            elif token.startswith(('runtime/','dev/benchmarks/')) or ('/Projects/splash/' in token and '/source/' in token):live.append(token)
        deps.append({'object':row['object'],'dependency_sha256':file_sha(path),'owned_inputs':owned})
    checks['fresh_replacement_deps_only_new_frozen_source']=not live
    inputs=man['frozen_inputs'];checks['44host4core76old1vector']=len([r for r in inputs if r['category']=='REUSED'])==44 and len([r for r in inputs if r['category']=='CORE'])==4 and len([r for r in inputs if r['category']=='AIRS'])==77
    checks['teacher_bulk_linked_once']=sum(Path(r['private_path']).name=='teacher_bulk.o' for r in inputs)==1
    policy={}
    for mode in ('','--freeze0','--freeze1','--missing','--retry0','--retry1'):
        policy[mode or 'default']=json.loads(subprocess.check_output([str(build/'policy-cpu'),*([mode] if mode else [])],text=True))
    checks['strict_selector_optional_setting_policy_pass']=all(x['pass'] and not x['gpu_work'] for x in policy.values())
    refusals={}
    for value in ('','2','true','01',' 1','1 '):
        env=dict(os.environ);env['SPLASH_FLASH_GEMV_DECODE_R1_SEP21']=value
        result=subprocess.run([str(build/'splash-flash'),'serve-flash-native','/r1-teacher-does-not-exist','16384','auto'],env=env,text=True,capture_output=True)
        refusals[value]={'returncode':result.returncode,'stderr':result.stderr}
        checks['strict_flag_before_path:'+repr(value)]=result.returncode!=0 and 'SPLASH_FLASH_GEMV_DECODE_R1_SEP21 must be exactly 0 or 1' in result.stderr
    env=dict(os.environ);env.update({'SPLASH_FLASH_GEMV_DECODE_R1_SEP21':'1','SPLASH_FLASH_ALLROWS_GATHERED_MPP':'0'})
    result=subprocess.run([str(build/'splash-flash'),'serve-flash-native','/r1-teacher-does-not-exist','16384','auto'],env=env,text=True,capture_output=True)
    checks['gathered_dependency_before_path']=result.returncode!=0 and 'requires gathered MPP enabled' in result.stderr
    worker_cpu=json.loads(subprocess.check_output([str(build/'splash-flash'),'--cpu-self-test'],text=True));checks['worker_CPU']=worker_cpu['valid'] and not worker_cpu['gpu_work']
    artifacts={name:file_sha(build/name) for name in ['splash-flash','splash.metallib','policy-cpu','baseline-original76.metallib']}
    for row in man['rebuild']:artifacts['host/'+row['object']+'.o']=file_sha(build/'host'/(row['object']+'.o'))
    seal={'schema':'TeacherV5-strict-standard-R1-compiled-CPU-seal-v1','pass':all(checks.values()),'checks':checks,
          'gpu_executed':False,'model_payload_bytes_read':0,'source_manifest_sha256':file_sha(build/'overlay-manifest.json'),
          'source_sha256':source_sha,'artifact_sha256':artifacts,'frozen_inputs':inputs,'compiled_replacement_dependencies':deps,
          'effective_host_count':50,'core_count':4,'policy':policy,'compiled_refusals':refusals,'worker_CPU':worker_cpu,
          'source_mismatch':mismatches,'input_mismatch':frozen,'live_dependencies':live,
          'Root_current2K_state_and_numeric_proof_required_before_benchmark':True,'full_model_qualified':False,
          'witness_source_sha256':file_sha(Path(__file__))}
    output.write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps({'pass':seal['pass'],'failed_checks':[k for k,v in checks.items() if not v],
        'sources':len(source_sha),'host':50,'core':4,'original76_exact':True,'old_vector_air_unchanged':True,'gpu_work':False,'model_payload_bytes_read':0}))
    if not seal['pass']:raise ValueError('Compiled standard R1 closure witness failed')


if __name__=='__main__':main()
