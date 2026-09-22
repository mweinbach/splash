"""CPU/source-bound audit for B4 packed-V prefill + physical R8/R16 verify.

No model/tokenizer/device import, no global patch of the frozen ORIGINAL22
suite, and no inherited semantic/performance qualification. Binding metadata
must be produced from the fresh overlay-manifest and compiled CPU seal.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

MODEL_SOURCE='ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e'
MODEL_LAYOUT='edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0'
ORIGINAL22_PLAN='a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac'
TARGET_NUMERIC_PARENT='b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d'
BQSA_SCHEMA='batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2'
BQSA_MARKER=';private-batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2'
BQSA_SCOPE='arena and encoded fields count graph construction; completed_native_forwards counts healthy API completion after diagnostics/state publication'
INTEGER_SCHEMA='parallel-integer-original-M16-six-stage-R8R16-target-verify-v1'
INTEGER_SCOPE='batch target rows4 active lanes2/4 physicalR8/R16 only; graph construction not GPU completion'
ARENA_BYTES=509607936

def require(value,message):
    if not value: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def is_sha(value): return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def u64(value): return type(value) is int and 0<=value<2**64
def get(value,path):
    for key in path.split('.'):
        if not isinstance(value,dict): return None
        value=value.get(key)
    return value

def load_binding(path,expected_sha):
    """Metadata/source files only; caller supplies the external Root digest."""
    require(is_sha(expected_sha) and sha(path)==expected_sha,'external adapter binding digest differs')
    binding=json.loads(Path(path).read_text())
    require(binding.get('schema')=='integer-B4-twopass-adapter-binding-v1','unregistered adapter schema')
    required=('runtime_build','overlay_manifest_sha256','compiled_seal_sha256','worker_sha256',
              'metallib_sha256','policy_source_sha256','policy_text','policy_shader_sha256',
              'policy_host_sha256','integer_source_identity_sha256','target_execution_child_sha256')
    require(all(key in binding for key in required),'incomplete fresh source/runtime binding')
    for key in required:
        if key.endswith('sha256'): require(is_sha(binding[key]),'invalid fresh binding digest:'+key)
    build=Path(binding['runtime_build']).resolve()
    for name,key in [('overlay-manifest.json','overlay_manifest_sha256'),('compiled-cpu-seal.json','compiled_seal_sha256'),
                     ('splash-flash','worker_sha256'),('splash.metallib','metallib_sha256')]:
        require(sha(build/name)==binding[key],'fresh runtime/manifest/seal drift:'+name)
    manifest=json.loads((build/'overlay-manifest.json').read_text())
    require(manifest.get('BQSA_source_policy_sha256')==binding['policy_source_sha256'],'fresh policy provenance mismatch')
    require(manifest.get('source_identity_sha256')==binding['integer_source_identity_sha256'],'fresh integer source provenance mismatch')
    require(binding.get('original22_plan_content_sha256')==ORIGINAL22_PLAN,'ORIGINAL22 inputs/graders cannot change')
    require(binding.get('target_numeric_parent_sha256')==TARGET_NUMERIC_PARENT,'persisted numerical parent changed')
    require(binding.get('policy_schema')==BQSA_SCHEMA,'fresh B4-only policy schema required')
    require(binding.get('qualified_shapes_or_scores_inherited') is False,'no state/task/performance inheritance')
    return binding

def integer_errors(status,binding,enabled,require_constructed=False):
    e=[];section=status.get('compact_native_batch_verify') if isinstance(status,dict) else None
    if not isinstance(section,dict): return ['integer status unavailable']
    fixed={'schema':INTEGER_SCHEMA,'scope':INTEGER_SCOPE,'enabled':enabled,'requested':enabled,
           'source_identity_sha256':binding['integer_source_identity_sha256'],
           'target_numeric_parent_sha256':TARGET_NUMERIC_PARENT,'dispatches_per_layer':6,
           'base_native_dispatches_per_layer':10,'additional_gpu_allocation_bytes':0,
           'full_model_quality_qualified':False}
    for key,wanted in fixed.items():
        value=section.get(key)
        if type(value)is not type(wanted) or value!=wanted: e.append('integer source/contract differs:'+key)
    total=0
    for rows,tg in ((8,2752),(16,3392)):
        node=section.get('r'+str(rows))
        if not isinstance(node,dict): e.append('missing physical integer width');continue
        for key,wanted in (('physical_rows',rows),('planner_threadgroup_bytes',tg)):
            if type(node.get(key))is not int or node[key]!=wanted:e.append('wrong fixed integer geometry:'+key)
        calls=[]
        for role in ('plan','gate','down'):
            n,r=node.get(role+'_graph_calls'),node.get(role+'_graph_rows');calls.append(n)
            if not u64(n)or not u64(r):e.append('invalid integer graph counter');continue
            if n%48 or r!=n*rows:e.append('integer graph count/physical rows mismatch')
        if all(u64(n)for n in calls):
            if len(set(calls))!=1:e.append('partial integer plan/gate/down graph')
            total+=calls[0]
            if not enabled and any(calls):e.append('disabled integer route encoded work')
    if require_constructed and not total:e.append('no integer target construction evidence')
    return e

def status_errors(status,binding,mode,bqsa_enabled=True,integer_enabled=True):
    """Strict initial/per-run source identity; per-run IDs stay unnormalized."""
    e=[]
    if mode not in ('standard','mtp3'): return ['unknown execution mode']
    if mode=='standard' and (bqsa_enabled or integer_enabled):return ['Standard requires both new flags disabled']
    identity=status.get('identity') if isinstance(status,dict) else None
    if not isinstance(identity,dict):return ['complete native identity required']
    common={'identity.source':MODEL_SOURCE,'identity.loaded_model_layout_sha256':MODEL_LAYOUT,
            'maximum_context_tokens':16384,'batch_prefill.maximum_lanes':4,
            'batch_prefill.maximum_real_rows_per_lane':2048,
            'scheduler.maximum_prefill_rows':2048,'scheduler.maximum_batch_prefill_rows_per_lane':2048,
            'memory_governor.host_measurement_valid':True,'memory_governor.growth_allowed':True,
            'memory_governor.denied_reservations':0}
    for path,wanted in common.items():
        value=get(status,path)
        if type(value)is not type(wanted)or value!=wanted:e.append('current native source/policy/resource differs:'+path)
    wanted={'batch_prefill_twopass_requested':bqsa_enabled,
            'batch_prefill_twopass_schema':BQSA_SCHEMA if bqsa_enabled else None,
            'batch_prefill_twopass_policy':binding['policy_text'] if bqsa_enabled else None,
            'batch_prefill_twopass_source_sha256':binding['policy_source_sha256'] if bqsa_enabled else None,
            'batch_prefill_twopass_shader_sha256':binding['policy_shader_sha256'] if bqsa_enabled else None,
            'batch_prefill_twopass_host_sha256':binding['policy_host_sha256'] if bqsa_enabled else None,
            'batch_prefill_twopass_arena_plan_bytes':ARENA_BYTES if bqsa_enabled else 0}
    for key,value in wanted.items():
        if type(identity.get(key))is not type(value)or identity[key]!=value:e.append('fresh B4 policy identity differs:'+key)
    raw=identity.get('batch_prefill_twopass_numerical_parent_routes')
    if not isinstance(raw,str)or not raw:e.append('raw Forward numerical-parent routes unavailable')
    elif bqsa_enabled:
        material='\n'.join((raw,BQSA_SCHEMA,binding['policy_text'],binding['policy_source_sha256'],binding['policy_shader_sha256'],binding['policy_host_sha256']))
        if identity.get('batch_prefill_twopass_numerical_identity')!=hashlib.sha256(material.encode()).hexdigest():e.append('B4 numerical material is not raw Forward parent routes')
    elif identity.get('batch_prefill_twopass_numerical_identity')is not None:e.append('disabled B4 numerical identity must be null')
    routes=identity.get('batch_prefill_kernel_routes')
    if not isinstance(routes,str)or routes.count(BQSA_MARKER)!=int(bqsa_enabled):e.append('constructed B4 numerical route marker mismatch')
    counters=status.get('batch_prefill_twopass_counters')
    if not isinstance(counters,dict):e.append('B4 counters unavailable')
    else:
        for key,value in {'scope':BQSA_SCOPE,'constructed_arenas':int(bqsa_enabled),'constructed_arena_bytes':ARENA_BYTES if bqsa_enabled else 0}.items():
            if type(counters.get(key))is not type(value)or counters[key]!=value:e.append('B4 arena/counter source scope differs:'+key)
        n=counters.get('encoded_QSA_lane_layer_calls');m=counters.get('encoded_QSA_lane_calls');f=counters.get('completed_native_forwards')
        if not all(u64(x)for x in (n,m,f))or n!=m or n!=48*f:e.append('B4-only cumulative48calls/healthy completion mismatch')
        if not bqsa_enabled and any(x for x in (n,m,f)if u64(x)):e.append('disabled B4 route constructed work')
    derivative=identity.get('target_numerical_derivative_sha256')
    if derivative!=(binding['target_execution_child_sha256']if integer_enabled else TARGET_NUMERIC_PARENT):e.append('target execution child vs persisted numerical parent identity mismatch')
    e+=integer_errors(status,binding,integer_enabled)
    return e

def counter_delta(before,after,path,errors):
    a,b=get(before,path),get(after,path)
    if not u64(a)or not u64(b)or b<a:errors.append('missing/invalid/decreasing counter:'+path);return None
    return b-a

def coverage(before,after,case,width,mode,binding,bqsa_enabled=True,integer_enabled=True):
    """Preserve current BQSA suite's common window/teacher/expert ledger logic.

    The frozen caller should run its ORIGINAL22 grading and existing common
    coverage without the old new-profile assumptions, then merge this result.
    This function audits only the new B4 and R8/R16 route-specific evidence.
    """
    errors=[];deltas={}
    if width not in (2,4)or mode not in ('standard','mtp3'):return {},['unregistered native width/mode']
    for side,status in (('before',before),('after',after)):
        errors += [side+': '+x for x in status_errors(status,binding,mode,bqsa_enabled,integer_enabled)]
    if before.get('identity')!=after.get('identity'):errors.append('within-run native identity changed')
    prompt=case.get('prompt_token_count')
    if not u64(prompt)or prompt<256:return {},errors+['invalid original exact prompt count']
    expected=48 if bqsa_enabled and mode=='mtp3'and width==4 and prompt>=2048 else 0
    for path,wanted in [('batch_prefill_twopass_counters.encoded_QSA_lane_calls',expected),('batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls',expected),('batch_prefill_twopass_counters.completed_native_forwards',int(expected>0))]:
        value=counter_delta(before,after,path,errors);deltas[path]=value
        if value is not None and value!=wanted:errors.append('actual B4-only first-window48/completion counter differs:'+path)
    integer_calls=0
    for rows in (8,16):
        values=[]
        for role in ('plan','gate','down'):
            path=f'compact_native_batch_verify.r{rows}.{role}_graph_calls';calls=counter_delta(before,after,path,errors);rpath=f'compact_native_batch_verify.r{rows}.{role}_graph_rows';n=counter_delta(before,after,rpath,errors);deltas[path]=calls;deltas[rpath]=n;values.append(calls)
            if calls is not None and n is not None and (calls%48 or n!=calls*rows):errors.append('actual integer physical-row/48layer delta differs')
        if all(v is not None for v in values):
            if len(set(values))!=1:errors.append('partial integer delta plan/gate/down')
            integer_calls+=values[0]
    if not integer_enabled and integer_calls:errors.append('disabled integer route has graph delta')
    masked=case.get('body',{}).get('response_format',{}).get('type')=='json_schema'
    if (mode=='standard'or masked)and integer_calls:errors.append('non-MTP/constrained task encoded target verify integer graphs')
    return {'new_B4_eligible_first_window':bool(expected),'new_B4_encoded_lane_layer_calls_expected':expected,'integer_target_graph_calls':integer_calls,'integer_graph_scope':'construction, not GPU completion; actual healthy terminals checked by frozen common suite','route_specific_counter_deltas':deltas,'old_SG8_everyrow_equivalence':False,'fullstate_task_perf_scores_inherited':False},errors
