#!/usr/bin/env python3
"""Private fixed4/R5 admission over the literal qualified Q4 original22 runner."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
PARENT_EXE = '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
PARENT_LIB = '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
PARENT_SOURCE = '162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a'
RAW_SHA = 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e'
ORIGINAL_SHA = 'f07f000096d6aa716a8d758d634cdf7d70e54c1e3423d4c377e7deef986aee5b'
NATIVE_PATH = ROOT / 'dev/benchmarks/flash_http_performance.py'
NATIVE_SHA = 'af0c883e1bb5c787e8cabdc004337172397ccd074c8beecbc454e29bac260389'
PLAN_CONTENT = 'a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac'
PLAN_FILE_SHA = '3fd2bbccf372dd78378929015db04ea347c22d4394c1eea55f39cb5f59a8400c'
PLANNER_AIR = '50002976851cd0bf2cf0f133c0164ccf177dfc1e7493b28563ba8fdbd1bb7ac3'
SECTION = 'compact_native_r5_verify'
PROFILE_SCHEMA = 'singleton-main-R5-integer-only-original-native-M16-six-stage-v1'
SCOPE = 'singleton main physical R5 verification only; per-layer full-chain graph construction, not GPU completion'
BINDING_SCHEMA = 'Root-registered-fixed4-R5-exact-runtime-binding-v2'
COMPILED_SCHEMA = 'singleton-R5-integer-current-Q4-worker-compiled-source-v1'
MANIFEST_SCHEMA = 'singleton-main-R5-integer-only-original-native-M16-worker-source-v1'
RECEIPT_SCHEMA = 'singleton-R5-current-Q4-fixed4-native-state-proof-v1'
FLAG = 'SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'
MARKER = ';private-singleton-main-R5-integer-only-original-M16-sourceSha256='
FILES = ('CPU_READY.json', 'compiled-cpu-seal.json', 'overlay-manifest.json', 'Root-r5-native-qualified.json')
ALLOCATION_GUARDS = ('target_ledger_matches_workspace', 'target_fits_category_plan',
    'workspace_plus_state_plan_fits_reservation', 'state_fits_state_plan',
    'workspace_plus_actual_state_fits_reservation', 'live_backend_delta_fits_reservation')
OWNER_LEDGER = {'after_target_equals_mapped':True, 'after_model_destruction_bytes':0,
    'governor_reserved_bytes':0, 'governor_denied_reservations':0,
    'host_measurement_valid':True, 'backend_stopped':True}
BINDING_KEYS = {'schema', 'worker', 'source_identity_sha256', 'exe_sha256', 'library_sha256',
                'metadata_sha256', 'native_state_counts', 'native_oracle_sha256', 'native_report_sha256',
                'native_allocation_guards', 'native_source_scope'}
_native = None

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def u64(x): return type(x) is int and 0 <= x < 2**64
def is_sha(x): return isinstance(x, str) and len(x) == 64 and set(x) <= set('0123456789abcdef') and x != '0'*64
def get(value, path):
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value: return object()
        value = value[part]
    return value
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
def import_exact(path, digest, name):
    if sha(path) != digest: raise ValueError('Pinned source drift: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
def native_metadata():
    # Called only after the inherited runner establishes its frozen namespace.
    global _native
    if sha(NATIVE_PATH) != NATIVE_SHA: raise ValueError('Exact native policy source drift')
    if _native is None: _native = import_exact(NATIVE_PATH, NATIVE_SHA, '_fixed4_exact_native_policy')
    return _native.native_mtp_depth_metadata

def validate_binding(value):
    if not isinstance(value, dict) or set(value) != BINDING_KEYS or value.get('schema') != BINDING_SCHEMA:
        raise ValueError('Unknown/incomplete fixed4 binding')
    for key in ('source_identity_sha256', 'exe_sha256', 'library_sha256', 'native_oracle_sha256', 'native_report_sha256'):
        if not is_sha(value.get(key)): raise ValueError('Binding digest malformed: ' + key)
    if value['exe_sha256'] == PARENT_EXE or value['library_sha256'] == PARENT_LIB:
        raise ValueError('Future R5 worker cannot reuse parent executable/library admission')
    worker = Path(value['worker']).resolve()
    if worker == PARENT.resolve() or ROOT/'build' not in worker.parents: raise ValueError('New private R5 worker required')
    meta = value.get('metadata_sha256')
    if not isinstance(meta, dict) or set(meta) != set(FILES) or not all(is_sha(x) for x in meta.values()):
        raise ValueError('Every exact compiled/state metadata digest is mandatory')
    counts = value.get('native_state_counts')
    if not isinstance(counts, dict) or not counts or not all(isinstance(k, str) and k and u64(v) and v > 0 for k,v in counts.items()):
        raise ValueError('Fresh exact positive native state counts required')
    validate_allocation_guards(value.get('native_allocation_guards'))
    if value.get('native_source_scope')!='MAIN ONLY':raise ValueError('Fresh native proof admits main state only')
    return value

def validate_allocation_guards(value):
    if not isinstance(value,dict) or set(value)!=set(ALLOCATION_GUARDS) or any(v is not True for v in value.values()):
        raise ValueError('All six actual source allocation guards must be explicit true Booleans')
    return value

def validate_native_receipt(r,b):
    wanted = {'schema': RECEIPT_SCHEMA, 'pass': True, 'qualification_complete': True, 'Root_GPU_executed': True,
              'source_identity_sha256': b['source_identity_sha256'], 'exe_sha256': b['exe_sha256'],
              'library_sha256': b['library_sha256'], 'planner_AIR_sha256': PLANNER_AIR,
              'maximum_verify_rows': 5, 'backend_destroyed': True, 'state_counts': b['native_state_counts'],
              'oracle_sha256': b['native_oracle_sha256'], 'root_report_sha256': b['native_report_sha256'],
              'allocation_guards': b['native_allocation_guards'], 'native_source_scope': 'MAIN ONLY',
              'memory_admission_scope':'explicit_governor_reservation_and_six_ledger_guards',
              'teacher_head_proved':False, 'device_allocation_measurements_present':False,
              'allocation_owner_ledger':OWNER_LEDGER, 'original22_qualified': False}
    if not isinstance(r,dict) or any(not same(r.get(k),v) for k,v in wanted.items()):
        raise ValueError('Fresh R5 whole-state receipt does not bind actual source-scoped evidence')
    if 'allocation_axes' in r or any('device_current' in k or 'device_peak' in k or 'physical_device' in k for k in r):
        raise ValueError('Native proof did not measure physical device allocation axes')
    validate_allocation_guards(r['allocation_guards'])
    return r

def authenticate(path, digest):
    if not is_sha(digest) or sha(path) != digest: raise ValueError('Externally registered binding SHA differs')
    b = validate_binding(json.loads(Path(path).read_text())); worker = Path(b['worker']).resolve()
    for name, wanted in b['metadata_sha256'].items():
        if sha(worker/name) != wanted: raise ValueError('Registered compiled/state metadata drift: ' + name)
    for name, wanted in {'splash-flash': b['exe_sha256'], 'splash.metallib': b['library_sha256'], 'R5-qualified.air': PLANNER_AIR}.items():
        if sha(worker/name) != wanted: raise ValueError('Registered R5 artifact drift: ' + name)
    expected = {'pass': True, 'source_identity_sha256': b['source_identity_sha256'],
                'exe_sha256': b['exe_sha256'], 'library_sha256': b['library_sha256'],
                'planner_AIR_sha256': PLANNER_AIR, 'parent_exe_sha256': PARENT_EXE,
                'parent_library_sha256': PARENT_LIB, 'public_headers_changed': False,
                'new_GPU_allocation_bytes': 0, 'GPU_work': False, 'whole_state_qualified': False,
                'changed_paths': ['runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm']}
    for name in ('CPU_READY.json', 'compiled-cpu-seal.json'):
        value = json.loads((worker/name).read_text())
        if value.get('schema') != COMPILED_SCHEMA or any(not same(value.get(k),v) for k,v in expected.items()):
            raise ValueError('Unknown/incomplete current R5 compiled closure: ' + name)
    m = json.loads((worker/'overlay-manifest.json').read_text())
    if (m.get('schema') != MANIFEST_SCHEMA or m.get('source_identity_sha256') != b['source_identity_sha256']
            or m.get('parent_source_identity') != PARENT_SOURCE or m.get('base') != str(PARENT)
            or m.get('public_headers_changed') is not False or not same(m.get('new_GPU_allocation_bytes'),0)
            or m.get('changed_paths') != expected['changed_paths']): raise ValueError('R5 source profile differs')
    parts = m.get('identity_parts')
    if not isinstance(parts, dict) or fingerprint(parts) != b['source_identity_sha256']: raise ValueError('R5 source identity tuple differs')
    for record in m.get('files', []):
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or sha(worker/'source'/relative) != record['sha256']:
            raise ValueError('Compiled R5 source file differs')
    if not m.get('files'): raise ValueError('Compiled R5 source inventory missing')
    r = json.loads((worker/'Root-r5-native-qualified.json').read_text())
    validate_native_receipt(r,b)
    return b

def fixed_policy(depth):
    return (f'singleton fixed cap{depth} bounded by output budget; concurrent ready MTP peers or pending admission cap singleton drafts at 3; '
            'joint fixedcap3; greedy unmasked; AR for ineligible requests')

def policy_errors(status, depth):
    if depth not in (3,4) or type(depth) is not int: raise ValueError('Only registered fixed3/fixed4 supported')
    wanted = {'capabilities.mtp':True, 'capabilities.batch_mtp':True, 'mtp.enabled':True,
              'mtp.maximum_draft_tokens':depth, 'mtp.singleton_maximum_draft_tokens':depth,
              'mtp.singleton_depth_override':depth, 'mtp.singleton_concurrent_draft_cap':3,
              'mtp.joint_maximum_draft_tokens':3, 'mtp.depth_controller_semantics':None,
              'mtp.teacher_cache_only_requested':True, 'mtp.policy':fixed_policy(depth),
              'identity.worker_semantics':'native-worker5-singleton-fixed-depth1..15-fold8-jointcap3-v6',
              'mtp.joint_policy':'fixed cap3 shared-budget/EOS bound; true joint head and target; survivors retain independent prefixes'}
    errors = ['Registered fixed policy differs: '+k for k,v in wanted.items() if not same(get(status,k),v)]
    hist = get(status,'mtp.completed_cycles_by_proposed_depth')
    if not isinstance(hist,list) or len(hist)!=16 or not all(u64(x) for x in hist): errors.append('Sixteen U64 actual-depth counters required')
    elif any(hist[depth+1:]): errors.append('Actual depth exceeds registered fixed maximum')
    return errors

def r5_status_errors(status, b):
    wanted = {'schema':PROFILE_SCHEMA,'requested':True,'enabled':True,'source_identity_sha256':b['source_identity_sha256'],
              'planner_AIR_sha256':PLANNER_AIR,'scope':SCOPE,'dispatches_per_layer':6,'GPU_allocation_bytes_added':0,
              'whole_state_qualified':False,'maximum_verify_rows':5}
    p = status.get(SECTION) if isinstance(status,dict) else None
    if not isinstance(p,dict): return ['Registered R5 profile missing']
    errors = ['Registered R5 field differs: '+k for k,v in wanted.items() if not same(p.get(k),v)]
    if set(p)!=set(wanted)|{'graph_calls','graph_rows'}: errors.append('Unknown/missing registered R5 profile field')
    calls,rows=p.get('graph_calls'),p.get('graph_rows')
    if not u64(calls) or not u64(rows) or rows!=5*calls: errors.append('R5 cumulative graph counters invalid')
    routes=get(status,'identity.kernel_routes')
    if not isinstance(routes,str) or routes.count(MARKER)!=1 or routes.count(MARKER+b['source_identity_sha256'])!=1:
        errors.append('R5 exact compiled execution marker differs')
    return errors

def inherited_status_view(status):
    if not same(get(status,'mtp.singleton_maximum_draft_tokens'),4): raise ValueError('Only exact declared cap4 can normalize for old gate')
    view=copy.deepcopy(status);view['mtp']['singleton_maximum_draft_tokens']=3;return view

def normalize_policy_summary(report):
    value=copy.deepcopy(report['actual_execution_policy'])
    cap=value.get('mtp.singleton_maximum_draft_tokens')
    if type(cap) is not int or cap not in (3,4): raise ValueError('Unknown declared comparison cap')
    value['mtp.singleton_maximum_draft_tokens']=3
    return value

def make_hooks(http,b):
    metadata=native_metadata()
    def status(value,plan,store,execution_mode='mtp3'):
        if execution_mode!='mtp3': return ['Fixed4 requires original MTP semantic mode']
        return policy_errors(value,4)+r5_status_errors(value,b)
    def coverage(before,after,case,require_counters=True,execution_mode='mtp3'):
        errors=status(before,None,None,execution_mode)+status(after,None,None,execution_mode)
        native=metadata(before,after);errors.extend(native['errors'])
        a,z=(get(s,'mtp.completed_cycles_by_proposed_depth') for s in (before,after))
        if not (isinstance(a,list) and isinstance(z,list) and len(a)==len(z)==16 and all(u64(x) for x in a+z) and all(y>=x for x,y in zip(a,z))):
            return {},errors+['Actual depth histogram invalid/decreased']
        caller=case.get('compact_scope','singleton-main')=='singleton-main' and case.get('body',{}).get('response_format',{}).get('type')!='json_schema'
        if not caller and any(y-x for x,y in zip(a,z)):errors.append('Excluded caller advanced singleton MTP cycles')
        wanted=48*(z[4]-a[4]) if caller else 0;deltas={}
        for key in ('graph_calls','graph_rows'):
            old,new=(get(s,SECTION+'.'+key) for s in (before,after))
            if not u64(old) or not u64(new) or new<old:errors.append('R5 counter invalid/decreased: '+key)
            else:deltas[key]=new-old
        if len(deltas)==2 and (deltas['graph_calls']!=wanted or deltas['graph_rows']!=5*wanted):errors.append('R5 counters differ from48 actual H4 cycles')
        if ownership(before)!=ownership(after):errors.append('R5 policy/provenance changed within request')
        return {'R5_graph_counter_deltas':deltas,'R5_expected_calls':wanted,'R5_actual_H4_cycles':z[4]-a[4],
                'fixed4_actual_depth_histogram_delta':[y-x for x,y in zip(a,z)],'fixed4_native_policy':native},errors
    def ownership(value):
        p=copy.deepcopy(value.get(SECTION,{}))
        for key in ('graph_calls','graph_rows'):p.pop(key,None)
        return {SECTION:p, 'registered_singleton_policy':4}
    return status,coverage,ownership

def load_parent():
    raw=import_exact(PARENT/'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py',RAW_SHA,'_fixed4_original_raw')
    runner=raw.load(PARENT,expected=True,require_state=True)
    if sha(runner.__file__)!=ORIGINAL_SHA: raise ValueError('Frozen original22 runner source drift')
    return runner

def install_fixed4(runner,b):
    hooks=make_hooks(runner.http,b);old_gate,old_cov,old_own=runner.gate_status,runner.coverage,runner.ownership_policy
    def gate(value,plan,store,execution_mode='mtp3'):
        errors=hooks[0](value,plan,store,execution_mode)
        view=inherited_status_view(value) if same(get(value,'mtp.singleton_maximum_draft_tokens'),4) else value
        return old_gate(view,plan,store,execution_mode)+errors
    def coverage(*args,**kwargs):
        details,errors=old_cov(*args,**kwargs);extra,added=hooks[1](*args,**kwargs);return {**details,**extra},errors+added
    runner.gate_status=gate;runner.coverage=coverage
    runner.fixed4_policy_status=hooks[0]
    runner.ownership_policy=lambda s:{**old_own(s),**hooks[2](s)}
    return runner

def load(binding,digest):
    b=authenticate(binding,digest);return install_fixed4(load_parent(),b)

def runtime_args(rest,worker):
    explicit=False
    for i,token in enumerate(rest):
        option=token.split('=',1)[0]
        if len(option)>2 and option.startswith('--') and '--runtime-build'.startswith(option) and option!='--runtime-build':raise ValueError('Complete runtime-build option required')
        if token=='--runtime-build':
            if i+1==len(rest):raise ValueError('Missing runtime-build')
            value=rest[i+1]
        elif token.startswith('--runtime-build='):value=token.split('=',1)[1]
        else:continue
        explicit=True
        if not value or Path(value).resolve()!=worker:raise ValueError('Every runtime-build must match registered current worker')
    return rest if explicit else rest+['--runtime-build',str(worker)]

def plan_args(rest):
    wanted=ROOT/'build/release/flash/prefill4k-semantic-plan-v1.json';seen=False
    for i,token in enumerate(rest):
        option=token.split('=',1)[0]
        if len(option)>2 and option.startswith('--') and '--plan'.startswith(option) and option!='--plan':raise ValueError('Complete original --plan option required')
        if token=='--plan':
            if i+1==len(rest):raise ValueError('Missing original plan value')
            value=rest[i+1]
        elif token.startswith('--plan='):value=token.split('=',1)[1]
        else:continue
        seen=True
        if not value or Path(value).resolve()!=wanted:raise ValueError('Every semantic plan must be original22')
    if not seen or sha(wanted)!=PLAN_FILE_SHA:raise ValueError('Exact preregistered original22 plan file required')
    return rest

def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha256',required=True)
    args,rest=parser.parse_known_args(argv);b=authenticate(args.binding,args.binding_sha256)
    runner=install_fixed4(load_parent(),b)
    if rest and rest[0]=='measure':rest=plan_args(runtime_args(rest,Path(b['worker']).resolve()))
    return runner.main(rest)

if __name__=='__main__':raise SystemExit(main())
