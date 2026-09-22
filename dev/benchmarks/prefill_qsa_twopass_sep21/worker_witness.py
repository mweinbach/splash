#!/usr/bin/env python3
"""Seal fresh packed-V integration with CPU scope/planner/identity/source checks."""
import argparse,json,os,subprocess
from pathlib import Path
from worker_overlay import ROOT,PRIVATE,DEFAULT_BUILD,FLAG,CHANGED,QUALIFIED,sha,load,transform
def section(text,begin,end):
    start=text.index(begin);return text[start:text.index(end,start)]
def cpu(command,env):
    r=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
    return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def cleanEnv(selected='0'):
    env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_FLASH_')};env[FLAG]=selected
    for dependency in ('QSA_F32','QSA_MPP','QSA_ROW_TILES','QSA_BULK_PREFILL','QSA_BULK_PREFILL_SG8'):env['SPLASH_FLASH_'+dependency]='1'
    return env
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=DEFAULT_BUILD);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build=args.build.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Choose fresh witness output')
    m=json.loads((build/'overlay-manifest.json').read_text());base=Path(m['prefill_qsa_base_build']);helper=load(build/'machinery/parent_overlay.py','qsa_witness_parent')
    checks={'only_expected_three_source_transforms':set(m['prefill_qsa_changed_files'])==CHANGED,'zero_GPU_and_payload_work':not m['gpu_executed'] and m['payload_bytes_read']==0,
        'exact_extra_arena_and_reused_prepared_plan':m['prefill_qsa_added_workspace_bytes']==478150656 and m['prefill_qsa_reused_prepared_bytes']==31457280,
        'parent_manifest_fresh':sha((base/'overlay-manifest.json').read_bytes())==m['prefill_qsa_base_manifest_sha256'],
        'parent_make_fresh':sha(Path(m['prefill_qsa_base_make_path']).read_bytes())==m['prefill_qsa_base_make_sha256'],
        'helper_frozen':sha((build/'machinery/parent_overlay.py').read_bytes())==m['prefill_qsa_parent_helper_sha256'],
        'link_make_frozen':sha((build/'link-inputs.mk').read_bytes())==m['prefill_qsa_link_make_sha256']}
    records={r['path']:r for r in m['files']};mismatch=[]
    for relative,r in records.items():
        actual=(build/'source'/relative).read_bytes()
        if r.get('new_prefill_qsa_file'):
            if 'qualified_component_source' in r:expected=Path(r['qualified_component_source']).read_bytes()
            else:
                expected=Path(r['repository_source']).read_bytes()
                if relative.endswith('worker_bridge.hpp'):
                    expected=expected.replace(b'QUALIFIED_SHADER_SHA',sha((QUALIFIED/'source/experiment/candidate.metal').read_bytes()).encode()).replace(b'QUALIFIED_HOST_SHA',sha((QUALIFIED/'source/experiment/twopass.cpp').read_bytes()).encode())
        else:expected=transform(relative,(base/'source'/relative).read_text(),build/'machinery/worker_transform.py').encode()
        if actual!=expected or sha(actual)!=r['overlay_sha256']:mismatch.append(relative)
    checks['all_frozen_sources_and_strict_transform_fresh']=not mismatch
    checks['tools_fresh']=all(sha((build/r['private_path']).read_bytes())==r['sha256'] and sha(Path(r['source_path']).read_bytes())==r['sha256'] for r in m['prefill_qsa_tools'])
    checks['snapshots_and_qualification_origins_fresh']=all(sha((build/r['private_path']).read_bytes())==r['sha256'] and sha(Path(r['source_path']).read_bytes())==r['sha256'] for r in m['prefill_qsa_qualification_snapshots'])
    inputs=m['prefill_qsa_link_inputs'];checks['all_parent_link_and_qualified_AIR_inputs_fresh']=all(sha((build/r['private_path']).read_bytes())==r['sha256'] and sha(Path(r['source_path']).read_bytes())==r['sha256'] for r in inputs)
    parentClosure=helper.effective_closure(base,Path(m['prefill_qsa_base_make_path']));actualClosure=helper.effective_closure(build,build/'machinery/worker.mk')
    checks['actual_parent_closure_preserved']=[str(x) for x in parentClosure['objects']]==m['prefill_qsa_effective_parent_objects'] and [str(x) for x in parentClosure['airs']]==m['prefill_qsa_effective_parent_airs']
    expectedObjects={str(build/'host/FlashForward.o'),str(build/'host/FlashWorker.o'),str(build/'host/twopass.o')}|{str(build/r['private_path']) for r in inputs if r['category'] in ('REUSED','CORE')}
    checks['new53_object_link_is_exact_pinned_closure']={str(x) for x in actualClosure['objects']}==expectedObjects and len(actualClosure['objects'])==len(parentClosure['objects'])+1
    checks['new75_AIR_link_preserves74_parent_AIRs_plus_qualified_component']={str(x) for x in actualClosure['airs']}=={str(build/r['private_path']) for r in inputs if r['category']=='AIRS'} and len(actualClosure['airs'])==len(parentClosure['airs'])+1
    checks['all_parent_objects_except_replaced_pair_retained']={r['source_path'] for r in inputs if r['category'] in ('REUSED','CORE')}=={str(x) for x in parentClosure['objects'] if x.stem not in ('FlashForward','FlashWorker')}
    checks['qualified_header_ABI_crosspins_fresh']=all(records[r['path']]['overlay_sha256']==r['sha256'] for r in m['prefill_qsa_cross_pins'])
    forward=(build/'source/runtime/flash/FlashForward.cpp').read_text();oldForward=(base/'source/runtime/flash/FlashForward.cpp').read_text()
    worker=(build/'source/runtime/flash/FlashWorker.mm').read_text();oldWorker=(base/'source/runtime/flash/FlashWorker.mm').read_text()
    bridge=(build/'source'/PRIVATE/'worker_bridge.hpp').read_text()
    routes=section(forward,'std::string FlashForward::kernelRoutes() const','FlashHCUpEncodedCounters FlashForward::hcUpEncodedCounters() const')
    identity=section(worker,'      << R"(,"identity":{"source":)"','      << R"(,"persisted_operands":')
    checks['all_mutable_counters_outside_entire_identity_and_routes']=all(term not in identity and term not in routes for term in ('encodedCounters','recordForward','recordLayer','constructed_arenas','encoded_forwards','constructed_arena_bytes','encoded_attention_layers'))
    checks['static_source_bound_numerical_wrapper_flag1_only']='prefill_qsa_twopass_sep21::numericalIdentity(dense_w8a8_sep21::numericalIdentity(' in identity and 'if (!selected) return std::string(base);' in bridge
    checks['no_placeholder_source_digest']='QUALIFIED_' not in bridge
    checks['flag0_route_marker_empty']='return selected?' in bridge and ':"";' in bridge
    checks['source_scope_nonverify_singleton_fresh2048']='begin,rows,verification,true,prefill_qsa_twopass_sep21::requested()' in forward
    checks['packed_V_only_call']='begin,rows,verification,true); // Only Root-qualified packed-V route.' in forward
    checks['planner_adds_exact_optional_extra']='total += prefill_qsa_twopass_sep21::plannedExtraBytes(maximumRows,prefill_qsa_twopass_sep21::requested());' in forward
    checks['original_governor_preconstruction_reservation_and_peak_guard_unchanged']=section(worker,'      uint64_t plannedTrunk','      std::optional<FlashMTPForward> head;')==section(oldWorker,'      uint64_t plannedTrunk','      std::optional<FlashMTPForward> head;')
    checks['single_arena_and_three_aligned_views']='backend.allocateBuffer(prefill4k::twoPassExtraBytes()' in bridge and all(term in bridge for term in ('backend.view(arena,0,25165824)','backend.view(arena,25165824,402653184)','backend.view(arena,427819008,50331648)','after-before!=prefill4k::twoPassExtraBytes()'))
    checks['reuses_existing_prepared_planes']='twoPassQSAWorkspace.emplace(backend,bulkQSAWorkspace->prepared);' in forward
    checks['external_feature_destination_rejects_all_arena_planes']=all(term in forward for term in ('two.arena,two.qualified.prepared.queries','two.qualified.prepared.indexQueries,two.qualified.prepared.selectedBlocks','two.qualified.packedQueries,two.qualified.scoresAndProbabilities,two.qualified.rawAttention'))
    checks['all_historical_batch_MTP_decode_headers_unchanged']=all((build/'source'/relative).read_bytes()==(base/'source'/relative).read_bytes() for relative in records if relative.startswith('runtime/') and relative not in CHANGED)
    completion='  metal::CommandTiming timing;\n  try {\n    timing = impl_->backend.submitCommand(graph.dispatches());'
    checks['all_forward_completion_diagnostics_state_poison_blocks_unchanged']=forward[forward.index(completion):]==oldForward[oldForward.index(completion):]
    # New code does not touch Worker dispatch/cancel/deadline work or state cleanup.
    checks['worker_request_scheduling_and_cancellation_unchanged']=section(worker,'class Worker final','void Worker::publishStatus()')==section(oldWorker,'class Worker final','void Worker::publishStatus()')
    checks['strict_selector_frozen_before_path_backend']=worker.index('(void)prefill_qsa_twopass_sep21::requested();')<worker.index('const auto directory = std::filesystem::canonical(argv[2]);')
    compiledDeps=[];live=[]
    for dep in [build/'host/FlashForward.d',build/'host/FlashWorker.d',build/'host/twopass.d',build/'policy-cpu.d',build/'prefill4k-attribution.d']:
        for token in helper.dep_tokens(dep.read_bytes()):
            path=helper.resolve_input(Path(token)).resolve()
            if build/'source' in path.parents:
                relative=path.relative_to(build/'source').as_posix();compiledDeps.append({'path':relative,'sha256':sha(path.read_bytes())})
                if relative not in records or sha(path.read_bytes())!=records[relative]['overlay_sha256']:live.append(str(path))
            elif ROOT/'runtime' in path.parents or ROOT/'dev' in path.parents:live.append(str(path))
    checks['all_actual_compiled_header_dependencies_private_frozen']=not live
    results={mode or 'default':cpu([build/'policy-cpu']+([mode] if mode else []),cleanEnv()) for mode in ('','--freeze0','--freeze1')}
    checks['CPU_flag_scope_planner_source_identity_and_full_byte_witness_pass']=all(r['returncode']==0 for r in results.values())
    if results['--freeze0']['returncode']==0 and results['--freeze1']['returncode']==0:
        off=json.loads(results['--freeze0']['stdout'])['workspace_plans'];on=json.loads(results['--freeze1']['stdout'])['workspace_plans']
        checks['actual_compiled_planner_delta_exact_by_maxRows']=all(on[row]-off[row]==(478150656 if int(row)>=2048 else 0) for row in off)
    else:checks['actual_compiled_planner_delta_exact_by_maxRows']=False
    selfTest=cpu([build/'splash-flash','--cpu-self-test'],cleanEnv());checks['native_worker_CPU_self_test_pass']=selfTest['returncode']==0
    invalid={}
    for value in ('','2','true','01',' 1','1 ','-1'):
        r=cpu([build/'splash-flash','serve-flash-native','/nonexistent-qsa-cpu-only','auto','auto'],cleanEnv(value));invalid[value]=r
        checks[f'invalid_flag_rejected_before_path_backend:{value!r}']=r['returncode']==3 and FLAG+' must be 0 or 1' in r['stderr'] and 'canonical' not in r['stderr']
    missing={}
    for dep in ('QSA_F32','QSA_MPP','QSA_ROW_TILES','QSA_BULK_PREFILL','QSA_BULK_PREFILL_SG8'):
        env=cleanEnv('1');env['SPLASH_FLASH_'+dep]='0';r=cpu([build/'splash-flash','serve-flash-native','/nonexistent-qsa-cpu-only','auto','auto'],env);missing[dep]=r
        checks[f'missing_dependency_rejected_before_path_backend:{dep}']=r['returncode']==3 and 'requires SPLASH_FLASH_'+dep+'=1' in r['stderr'] and 'canonical' not in r['stderr']
    document={'schema':'splash-prefill-qsa-packedV-worker-cpu-witness-v1','pass':all(checks.values()),'checks':checks,'gpu_work':False,'payload_bytes_read':0,
        'whole_model_qualified':False,'source_mismatch':mismatch,'live_project_dependencies':live,'compiled_dependencies':list({r['path']:r for r in compiledDeps}.values()),
        'compiled_policy_CPU':results,'native_CPU_self_test':selfTest,'invalid_flag_probes':invalid,'missing_dependency_probes':missing,
        'frozen_source_count':len(records),'frozen_link_input_count':len(inputs),'actual_worker_objects':len(actualClosure['objects']),'actual_worker_airs':len(actualClosure['airs']),
        'runtime_sha256':{name:sha((build/name).read_bytes()) for name in ('splash-flash','splash.metallib','prefill4k-attribution')}}
    output.write_text(json.dumps(document,indent=2)+'\n');print(json.dumps({'pass':document['pass'],'checks':len(checks),'failures':[k for k,v in checks.items() if not v],'gpu_work':False}))
    if not document['pass']:raise SystemExit(1)
if __name__=='__main__':main()
