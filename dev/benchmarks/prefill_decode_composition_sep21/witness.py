#!/usr/bin/env python3
"""Verify private source/objects/startup with JSON-only metadata; no backend."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,shlex,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[3]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',type=Path,default=ROOT/'build/prefill-i8-decode-q4-teacher-sep22-worker-v3');a=ap.parse_args();b=a.build.resolve();mp=b/'overlay-manifest.json';m=json.loads(mp.read_text());base=Path(m['base']);checks=[]
    def require(value,label):
        if not value:raise ValueError(label)
        checks.append(label)
    for r in m['files']:require(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'source:'+r['path'])
    for r in m['frozen_core']:require(sha((b/r['path']).read_bytes())==r['sha256'],'core:'+r['path'])
    require(len(m['rebuild'])==50 and len(m['frozen_core'])==4,'all50hostConsumersRebuilt_core4')
    require(sha((b/'splash.metallib').read_bytes())==m['base_metallib_sha256']==sha((base/'splash.metallib').read_bytes()),'allShaderMathAndLibraryByteUnchanged')
    path=b/'machinery/transform.py';sp=importlib.util.spec_from_file_location('phase_frozen_transform',path);t=importlib.util.module_from_spec(sp);sp.loader.exec_module(t)
    hybrid=ROOT/'build/hybrid-sg2-tail-fma-sep21-worker-v1/source'
    for r in m['files']:
        if 'parent_sha256'not in r:continue
        original=(base/'source'/r['path']).read_text();candidate=(b/'source'/r['path']).read_text()
        require(t.transform(r['path'],original,hybrid)==candidate,'onlyAuthenticatedBoundedTransform:'+r['path'])
    pf=(base/'source/runtime/flash/FlashForward.cpp').read_text();cf=(b/'source/runtime/flash/FlashForward.cpp').read_text()
    for prefix in ('  void project(','  void hc('):
        before=t.function(pf,prefix);after=t.function(cf,prefix)
        if prefix=='  void project(':
            changed='''    const bool originalPrefillF32Member=phase_q4_sep21::requested() && singletonMain && !verification &&
        prefillF32SelectorPrefixes.contains(prefix);
    if (floatDenseCache && rows >= 2 && rows <= 16 &&
        (floatDenseCache->contains(prefix) || originalPrefillF32Member)) {'''
            original='    if (floatDenseCache && rows >= 2 && rows <= 16 && floatDenseCache->contains(prefix)) {'
            require(after.count(changed)==1,'exact508VirtualSelectorPresenceOnly_project')
            after=after.replace(changed,original)
        else:
            after=after.replace('bool injection, bool normalizedReady = false, bool prefillSelector = false)',
                'bool injection, bool normalizedReady = false)')
            require(after.count('rows, false, prefillSelector);')==3,'all3GenericHCProjectsExplicitPrefillContext')
            after=after.replace('rows, false, prefillSelector);','rows);')
        require(before==after,'prefillArithmeticByteUnchangedExceptDeclaredSelectorPresence:'+prefix.strip())
    start='FlashForwardResult FlashForward::forwardImpl('
    parentFn=t.function(pf,start);candidateFn=t.function(cf,start)
    candidateFn=candidateFn.replace('bool verification, bool decodePhase)','bool verification)')
    new='''    const bool phasePrefillI8=phase_q4_sep21::requested() && !verification && !decodePhase;
    const bool useI8=impl_->allRowsInt8Target || phasePrefillI8;
    const auto phase=verification?phase_q4_sep21::Phase::Verify:decodePhase?phase_q4_sep21::Phase::Decode:phase_q4_sep21::Phase::Prefill;
    const bool gatheredMPP=useI8 && impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredMPPEnabled() &&
        rows<=impl_->int8ExpertStore->gatheredMPPMaximumRows();
    const bool blocked=impl_->blockMoE && (useI8 || (!verification && !decodePhase && rows>=256));'''
    old='''    const bool gatheredMPP = impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        impl_->int8ExpertStore->gatheredMPPEnabled() && rows <= impl_->int8ExpertStore->gatheredMPPMaximumRows();
    const bool blocked = impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);'''
    require(candidateFn.count(new)==1,'exactDeclaredPhaseBooleansOnly')
    candidateFn=candidateFn.replace(new,old).replace('const auto tile = useI8 && rows < 256','const auto tile = impl_->allRowsInt8Target && rows < 256').replace('    phase_q4_sep21::record(phase,useI8,rows);\n','')
    require(candidateFn.count('normalizedReady, phase_q4_sep21::requested() && !verification && !decodePhase);')==3,
        'all3HCEntriesExplicitPrefillContext_NoDecodeVerify')
    candidateFn=candidateFn.replace('normalizedReady, phase_q4_sep21::requested() && !verification && !decodePhase);','normalizedReady);')
    require(candidateFn==parentFn,'entirePrefillArithmeticBodyByteUnchangedExceptPhasePredicatesCounter')
    pc=(base/'source/runtime/flash/FlashFloatDenseCache.cpp').read_text();cc=(b/'source/runtime/flash/FlashFloatDenseCache.cpp').read_text()
    for prefix in ('flashFloatDenseSmallRowsPolicy(','void addFloatDenseSmallRowsImpl('):
        require(t.function(pc,prefix)==t.function(cc,prefix),'allTinyPrefillF32SelectorAndDotMathByteUnchanged:'+prefix)
    sourceNames={r['path']for r in m['files']};objects=[b/'host'/(r['object']+'.o')for r in m['rebuild']]+[b/r['path']for r in m['frozen_core']];dependencies=[]
    require(len(objects)==54 and len(set(objects))==54,'54UniqueEffectiveObjects')
    for obj in objects:
        if obj.parent!=b/'host':continue
        d=obj.with_suffix('.d');require(d.exists(),'actualCompilerDep:'+obj.name)
        tokens=shlex.split(d.read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);owned=[]
        for token in tokens:
            path=(ROOT/token).resolve()if not Path(token).is_absolute()else Path(token).resolve()
            if b/'source'in path.parents:
                rel=path.relative_to(b/'source').as_posix();require(rel in sourceNames,'compilerOwnedInputManifested:'+rel);owned.append(rel)
            elif ROOT in path.parents and '/build/'in str(path):
                require(False,'liveParentBuildDependency:'+str(path))
        dependencies.append({'object':obj.relative_to(b).as_posix(),'dependency_sha256':sha(d.read_bytes()),'owned_inputs':sorted(set(owned))})
    cpu=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True);cpuJSON=json.loads(cpu.stdout);require(cpuJSON['valid']and len(cpuJSON['checks'])==49,'actualWorker49CPUChecks')
    flags=json.loads((b/'root-mtp-environment.json').read_text());ambient={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};guards=[]
    cases=[('validMTP',{},'manifest could not be opened'),('validStandard',{'SPLASH_FLASH_MTP':'0','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY':'0','SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'0'},'manifest could not be opened'),
        ('phaseDisabled',{'SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21':'0'},'requires PREFILL_I8_DECODE_Q4=1'),('phaseMalformed',{'SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21':'2'},'must be 0 or 1'),
        ('allRowsCompetes',{'SPLASH_FLASH_ALLROWS_FULL512_TARGET':'1'},'requires SPLASH_FLASH_ALLROWS_FULL512_TARGET=0'),('wideGatherChangesTinyPrefill',{'SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS':'16'},'requires physical cap4'),
        ('changedFMA',{'SPLASH_FLASH_GDN_PREFILL_FMA_SEP21':'0'},'requires SPLASH_FLASH_GDN_PREFILL_FMA_SEP21=1'),('changedQSA',{'SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21':'0'},'requires SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21=1'),
        ('changedVariant',{'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT':'0'},'requires fixedSG2variant7')]
    for flag in ('SPLASH_FLASH_BATCH','SPLASH_FLASH_BATCH_PREFILL','SPLASH_FLASH_BATCH_MTP','SPLASH_FLASH_BATCH_MTP_PREFILL','SPLASH_FLASH_GDN_BATCH_ILP','SPLASH_FLASH_GPU_PREFILL_COPY','SPLASH_FLASH_MTP_ADAPTIVE','SPLASH_FLASH_DENSE_SMALL_ROWS','SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT'):
        cases.append((flag,{flag:'1'},'requires '+flag+'=0'))
    for flag in ('SPLASH_FLASH_FLOAT_DENSE_CACHE','SPLASH_FLASH_FLOAT_DENSE_SELECTIVE','SPLASH_FLASH_QMV_F32','SPLASH_FLASH_ALLROWS_GATHERED_MPP'):
        marker='GDNABpair requiresactive QMV_F32=1'if flag=='SPLASH_FLASH_QMV_F32'else'requires '+flag+'=1'
        cases.append((flag,{flag:'0'},marker))
    with tempfile.TemporaryDirectory(prefix='phase-startup-CPU-')as temporary:
        root=Path(temporary).resolve();package=root/'package';store=root/'empty-store';package.mkdir();store.mkdir()
        (package/'config.json').write_bytes((ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1/config.json').read_bytes())
        for label,changes,marker in cases:
            env={**ambient,**flags,**changes,'SPLASH_FLASH_INT8_EXPERT_STORE':str(store)}
            result=subprocess.run([str(b/'splash-flash'),'serve-flash-native',str(package),'16384','auto'],env=env,capture_output=True,text=True)
            require(result.returncode==3 and marker in result.stderr,'compiledNoBackendProfileGuard:'+label)
            guards.append({'case':label,'returncode':result.returncode,'stderr':result.stderr.strip(),'backend_created':False})
    artifacts={str(p.relative_to(b)):sha(p.read_bytes())for p in objects+[b/'splash-flash',b/'splash.metallib']}
    out={'schema':'private-phase-explicit-q4-compiled-cpu-seal-v1','pass':True,'gpu_executed':False,'model_payload_bytes_read':0,'source_manifest_sha256':sha(mp.read_bytes()),'source_sha256':{r['path']:r['sha256']for r in m['files']},'artifact_sha256':artifacts,'effective_objects':54,'compiler_dependency_closure':dependencies,'CPU_worker':cpuJSON,'pre_backend_profile_guards':guards,'prefill_arithmetic_body_proof':True,'fresh_governor_fit_proven':False,'runtime_numeric_semantic_performance_qualification_complete':False,'checks':checks,'witness_source_sha256':sha(Path(__file__).read_bytes())}
    (b/'compiled-cpu-seal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'profile_guards':len(guards),'sources':len(m['files']),'effective_objects':54,'GPU_qualified':False}))
if __name__=='__main__':main()
