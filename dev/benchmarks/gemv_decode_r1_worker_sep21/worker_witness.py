#!/usr/bin/env python3
"""CPU-only source closure, original-routing and planner witness for R1 GEMV."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
from worker_prepare import ROOT,PRIVATE,CHANGED_PATHS,FLAG,shipping_shader,transform

def sha(data):return hashlib.sha256(data).hexdigest()
def extract(text,begin,end):
    start=text.index(begin);return text[start:text.index(end,start)]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('NEW witness output required')
    build=args.build.resolve();m=json.loads((build/'overlay-manifest.json').read_text());base=Path(m['vector_r1_base_build'])
    checks={'base_manifest_fresh':sha((base/'overlay-manifest.json').read_bytes())==m['vector_r1_base_manifest_sha256'],
            'transform_fresh':sha(Path(__file__).with_name('worker_prepare.py').read_bytes())==m['vector_r1_transform_sha256']}
    bad=[]
    for r in m['files']:
        actual=(build/'source'/r['path']).read_bytes()
        if sha(actual)!=r['overlay_sha256']:bad.append(r['path']);continue
        if not r.get('new_vector_r1_file'):
            original=(base/'source'/r['path']).read_bytes()
            if sha(original)!=r['vector_r1_base_sha256'] or actual!=transform(r['path'],original.decode()).encode():bad.append(r['path'])
        elif r.get('repository_sha256') and sha((ROOT/r['path']).read_bytes())!=r['repository_sha256']:bad.append(r['path'])
    checks['all_frozen_sources_fresh']=not bad
    worker=(build/'source/runtime/flash/FlashWorker.mm').read_text()
    forward=(build/'source/runtime/flash/FlashForward.cpp').read_text()
    forward_header=(build/'source/runtime/flash/FlashForward.hpp').read_text()
    store=(build/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    base_forward=(base/'source/runtime/flash/FlashForward.cpp').read_text()
    base_worker=(base/'source/runtime/flash/FlashWorker.mm').read_text()
    base_store=(base/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    shader=(build/'source'/PRIVATE/'candidate.metal').read_text()
    qualified=(build/'source'/PRIVATE/'qualified-source/kernels.metal').read_text()
    checks.update({
      'workspace_planner_identical':extract(forward,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes')==extract(base_forward,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes'),
      'verification_workspace_identical':extract(forward,'uint64_t FlashForward::verificationWorkspaceBytes','FlashRequestState FlashForward::createState')==extract(base_forward,'uint64_t FlashForward::verificationWorkspaceBytes','FlashRequestState FlashForward::createState'),
      'request_state_planner_identical':extract(forward,'uint64_t FlashForward::requestStateBytes','uint64_t FlashForward::workspacePlannedBytes')==extract(base_forward,'uint64_t FlashForward::requestStateBytes','uint64_t FlashForward::workspacePlannedBytes'),
      'old_forward_and_verify_wrappers_identical':extract(base_forward,'FlashForwardResult FlashForward::forward(','FlashForwardResult FlashForward::verify(') in forward and extract(base_forward,'FlashForwardResult FlashForward::verify(','FlashForwardResult FlashForward::forwardImpl(') in forward,
      'ordinary_decode_explicit_selection_only':worker.count('forward_.forwardDecode(')==1 and 'return forwardImpl(request,token,false,false,false,true);' in forward and 'bool ordinaryDecode=false' in forward_header,
      'worker_prompt_and_mtp_forward_calls_unchanged':worker.count('forward_.forward(')==base_worker.count('forward_.forward(')-1 and extract(worker,'      const auto result = forward_.forward(*request.state,\n','      traceRequestCommand("prefill"')==extract(base_worker,'      const auto result = forward_.forward(*request.state,\n','      traceRequestCommand("prefill"'),
      'r1_route_guard_before_vector_graphs':'const bool vectorR1Decode = ordinaryDecode && gatheredMPP &&\n        gemv_decode_r1_sep21::eligible(rows,verification)' in forward,
      'original_gather_and_blocked_paths_identical':extract(base_forward,'    if (gatheredMPP) {','    if (!batchSharedExpertFused') .replace('    if (gatheredMPP) {','    } else if (gatheredMPP) {',1) in forward,
      'original_combine_layout_selector_identical':extract(forward,'    addCombine(graph, blocked && !gatheredMPP','    const bool nextHasPLE')==extract(base_forward,'    addCombine(graph, blocked && !gatheredMPP','    const bool nextHasPLE'),
      'original_store_producers_identical':extract(base_store,'void FlashInt8ExpertStore::addGateUp(','} // namespace splash::flash') in store,
      'no_new_gpu_allocator_calls':all((build/'source'/r).read_text().count(token)==(base/'source'/r).read_text().count(token) for r in CHANGED_PATHS for token in ('allocateBuffer(','wrapSharedMemory(','std::make_shared<Mapping>(')),
      'added_planned_gpu_bytes_zero':m['vector_r1_added_gpu_bytes']==0,
      'shader_exact_qualified_arithmetic_extraction':shader==shipping_shader(qualified),
      'shipping_shader_only_l32_producer_exports':'GEMV_DECODE_GATE(gemv_decode_sep21_v4_l32_o4_gate_up,32)' in shader and 'GEMV_DECODE_DOWN(gemv_decode_sep21_v4_l32_o4_down,32)' in shader and 'GEMV_DECODE_GATE(gemv_decode_sep21_v4_l16' not in shader and 'projection_probe' not in shader,
      'shipping_shader_explicit_r1_geometry':'if (p.rows!=1 || p.selections!=10' in shader,
      'graph_grids_match_qualified_vector':'{160, rows, selections}' in store and '{640, rows, selections}' in store,
      'source_identity_appended_conditionally':'if (gemvDecodeR1)\n      derivative +=' in store and 'ordinary_r1_decode_policy=' in store,
      'static_marker_has_no_graph_counters':'gemvDecodeR1GateCalls' not in extract(store,'    std::string derivative =','    numericalIdentity ='),
      'flag_frozen_before_paths_and_backend':worker.index('if (gemv_decode_r1_sep21::requested()')<worker.index('std::filesystem::canonical(argv[2])')<worker.index('metal::MetalBackend backend('),
      'link_make_fresh':sha((build/'link-inputs.mk').read_bytes())==m['vector_r1_link_make_sha256'],
    })
    input_bad=[r['private_path'] for r in m['vector_r1_link_inputs'] if sha((build/r['private_path']).read_bytes())!=r['sha256']]
    checks['all_link_inputs_frozen_fresh']=not input_bad
    live=[]
    for dep in (build/'host').glob('*.d'):
        line=dep.read_text().replace('\\\n',' ').splitlines()[0]
        live.extend(t for t in line.split(': ',1)[1].split() if t.startswith(('runtime/','dev/benchmarks/')))
    checks['no_live_runtime_or_benchmark_header_dependencies']=not live
    probes={}
    for mode in ([],['--freeze0'],['--freeze1'],['--missing'],['--retry0'],['--retry1']):
        probes[' '.join(mode)or'default']=json.loads(subprocess.check_output([str(build/'policy-cpu'),*mode],text=True))
    checks['compiled_policy_processes_pass']=all(p['pass'] and not p['gpu_work'] for p in probes.values())
    rejected={}
    for value in ('','2','true','01',' 1','1 '):
        env=dict(os.environ);env[FLAG]=value
        run=subprocess.run([str(build/'splash-flash'),'serve-flash-native','/gemv-r1-does-not-exist','16384','auto'],env=env,text=True,capture_output=True)
        rejected[value]={'returncode':run.returncode,'stderr':run.stderr}
        checks[f'bad_flag_rejected_before_path:{value!r}']=run.returncode!=0 and 'must be exactly 0 or 1 when present' in run.stderr
    report=json.loads((build/'qualification/r1-component.json').read_text())
    checks['synthetic_r1_component_evidence_only']=report['rows']==1 and not report['model_quality_qualified'] and next(v for v in report['variants'] if v['name']=='vectorGEMVL32O4')['numerical_alternative_qualified']
    result={'schema':'splash-private-r1-vector-gemv-worker-cpu-source-witness-v1','pass':all(checks.values()),
      'gpu_work':False,'model_payload_reads':False,'added_gpu_allocation_bytes':0,'model_quality_qualified':False,
      'source_identity_sha256':m['vector_r1_source_identity_sha256'],'checks':checks,'source_mismatches':bad,
      'link_mismatches':input_bad,'live_dependencies':live,'policy_processes':probes,'invalid_flag_processes':rejected,
      'artifacts_sha256':{name:sha((build/name).read_bytes()) for name in ('splash-flash','splash.metallib','policy-cpu','prefill4k-attribution')}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'output':str(args.output),'checks':len(checks),'gpu_work':False}))
    if not result['pass']:raise SystemExit(2)

if __name__=='__main__':main()
