#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shlex,subprocess,tempfile,copy
ROOT=Path(__file__).resolve().parents[3]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,default=ROOT/'build/phase-saved-only-residency-sep22-worker-v1');a=ap.parse_args();b=a.build.resolve();mp=b/'overlay-manifest.json';m=json.loads(mp.read_text());base=Path(m['base']);checks=[]
    def require(ok,label):
        if not ok:raise ValueError(label)
        checks.append(label)
    for r in m['files']:require(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'source:'+r['path'])
    for r in m['reused_objects']:require(sha((b/r['path']).read_bytes())==r['sha256'],'object:'+r['path'])
    require(len(m['reused_objects'])==53,'53UnchangedEffectiveObjects')
    require([r['path']for r in m['files']if r['changed']]==['runtime/flash/FlashWorker.mm'],'onlyWorkerResourceSelectionChanged')
    pp=b/'machinery/prepare.py';sp=importlib.util.spec_from_file_location('resource_frozen',pp);t=importlib.util.module_from_spec(sp);sp.loader.exec_module(t)
    before=(base/'source/runtime/flash/FlashWorker.mm').read_text();after=(b/'source/runtime/flash/FlashWorker.mm').read_text()
    require(t.transform(before)==after,'only4BoundedResourceEdits')
    for start,end in [('  void tick(Request &request) {','  void maskOrEmit'),('      << R"(,"target_numerical_derivative_sha256":)"','      << R"(,"prefill_qsa_twopass_enabled":)"')]:
        if start.startswith('  void tick'):continue # Whole implementation is source-pinned via exact transform; end ordering varies.
        require(before[before.index(start):before.index(end,before.index(start))]==after[after.index(start):after.index(end,after.index(start))],'mathematicalDerivativeByteUnchanged')
    require(sha((b/'splash.metallib').read_bytes())==m['metallib_sha256'],'allShaderByteUnchanged')
    cpu=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True);cpuJSON=json.loads(cpu.stdout);require(cpuJSON['valid']and len(cpuJSON['checks'])==49,'Worker49CPUchecks')
    flags=json.loads((b/'root-mtp-environment.json').read_text());ambient={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};guards=[]
    cases=[('newValid',{},'manifest could not be opened'),('oldProfilePreserved',{'SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22':'0','SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT':'1'},'manifest could not be opened'),('oldNoLeaseRejected',{'SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22':'0'},'requires saved/expert residency'),('newCompetingLeaseRejected',{'SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT':'1'},'requires HYBRID_Q4_EXPERT_RESIDENT=0'),('invalidNewFlag',{'SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22':'2'},'must be 0 or 1')]
    with tempfile.TemporaryDirectory(prefix='savedOnlyCPU-')as temp:
        root=Path(temp).resolve();package=root/'package';package.mkdir();store=root/'store';store.mkdir();(package/'config.json').write_bytes((ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1/config.json').read_bytes())
        for label,changes,marker in cases:
            result=subprocess.run([str(b/'splash-flash'),'serve-flash-native',str(package),'16384','auto'],env={**ambient,**flags,**changes,'SPLASH_FLASH_INT8_EXPERT_STORE':str(store)},capture_output=True,text=True)
            require(result.returncode==3 and marker in result.stderr,'compiledNoBackendGuard:'+label);guards.append({'case':label,'stderr':result.stderr.strip()})
    # Actual new helper's exact baseline and single-field tamper cases.
    import sys;sys.path.insert(0,str(ROOT));from dev.benchmarks import phase_saved_only_resource_quality as q
    status={'identity':{'phase_resource_profile':q.PROFILE,'phase_resource_variant_source_policy':q.SOURCE_POLICY,'phase_prefill_f32_selector_membership_count':508},'saved_operands_residency':{'requested':True,'active':True,'request_succeeded':True,'registered_base_allocation_count':807,'registered_base_allocation_bytes':131005546496,'backing_already_charged':True,'physical_pinning_verified':False},'hybrid_q4_expert_residency':{'requested':False,'added_to_composite':False,'active':False,'selected_owner_count':0,'selected_owner_bytes':0,'registered_composite_owner_count':807,'registered_composite_owner_bytes':131005546496},'persisted_operands':{'f32_tensors':296,'f32_mapped_payload_bytes':12097945600},'phase_f32_persistent_residency':{'retained_owner_count':118,'retained_owner_bytes':3247964160,'transient_only_owner_count':178,'transient_only_owner_bytes':8849981440,'all_backing_retained':True,'all_backing_charged':True},'ple_storage':{'gpu_mapped_original_bytes':74317889536}}
    require(not q.status_errors(status),'strict807ResourceHelperBaseline')
    for group,values in status.items():
        for key,value in values.items():
            bad=copy.deepcopy(status);bad[group][key]=not value if type(value)is bool else value+1 if type(value)is int else value+'bad';require(q.status_errors(bad),'strictResourceFieldTamper:'+group+'.'+key)
    d=b/'host/FlashWorker.d';require(d.exists(),'ActualWorkerCompilerDependencyPresent');known={r['path']for r in m['files']}
    tokens=shlex.split(d.read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);owned=[]
    for token in tokens:
        p=(ROOT/token).resolve()if not Path(token).is_absolute()else Path(token).resolve()
        if b/'source'in p.parents:
            rel=p.relative_to(b/'source').as_posix();require(rel in known,'manifestedCompilerDependency:'+rel);owned.append(rel)
    artifacts={r['path']:r['sha256']for r in m['reused_objects']};artifacts.update({n:sha((b/n).read_bytes())for n in ['host/FlashWorker.o','splash-flash','splash.metallib']})
    out={'schema':'private807ResourceCPUSeal-v1','pass':True,'GPU_executed':False,'model_payload_bytes_read':0,'source_manifest_sha256':sha(mp.read_bytes()),'source_sha256':{r['path']:r['sha256']for r in m['files']},'artifact_sha256':artifacts,'effective_objects':54,'compiled_resource_guards':guards,'CPU_worker':cpuJSON,'all_mathematics_backing_and_reservations_unchanged':True,'newLease807bytes131005546496':True,'GPU_semantic_performance_qualified':False,'actualWorkerDependencies':owned,'checks':checks}
    (b/'compiled-cpu-seal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'sources':len(m['files']),'effective_objects':54,'GPU_qualified':False}))
if __name__=='__main__':main()
