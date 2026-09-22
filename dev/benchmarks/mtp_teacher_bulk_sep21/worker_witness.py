#!/usr/bin/env python3
"""Authenticate standalone CPU closure and guard preflights; never creates Metal."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[3]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5');a=p.parse_args();b=a.build.resolve()
    mp=b/'overlay-manifest.json';m=json.loads(mp.read_text());base=Path(m['base']);checks=[]
    def require(value,note):
        if not value:raise ValueError(note)
        checks.append(note)
    for r in m['files']:require(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'source:'+r['path'])
    for r in m['frozen_objects']:require(sha((b/r['path']).read_bytes())==r['sha256'],'core:'+r['path'])
    require(len(m['rebuild'])==49 and len(m['frozen_objects'])==4,'all49noncoreTUs_rebuilt_fourCoreOnly')
    require(sha((b/'splash.metallib').read_bytes())==m['metallib_sha256'],'exactParentMetallib')
    pp=b/'machinery/worker_prepare.py';spec=importlib.util.spec_from_file_location('teacher_frozen_transform',pp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    for rel in ('runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTP.hpp','runtime/flash/FlashWorker.mm'):
        require(h.transform(rel,(base/'source'/rel).read_text())==(b/'source'/rel).read_text(),'onlyAuthenticatedPrivateTeacherInsertions:'+rel)
    linked=[b/'host'/('teacher_bulk.o')]+[b/'host'/(r['object']+'.o')for r in m['rebuild']]+[b/r['path']for r in m['frozen_objects']]
    require(len(linked)==54 and len(set(linked))==54,'54UniqueEffectiveObjects_includingWorker')
    sourcePaths={r['path']for r in m['files']};deps=[]
    for obj in linked:
        if obj.parent!=b/'host':continue
        dep=obj.with_suffix('.d');require(dep.exists(),'compilerDependencyPresent:'+obj.name)
        text=dep.read_text().replace('\\\n',' ');tokens=shlex.split(text.splitlines()[0].split(':',1)[1]);owned=[]
        for token in tokens:
            path=(ROOT/token).resolve() if not Path(token).is_absolute() else Path(token).resolve()
            if b/'source' in path.parents:
                rel=path.relative_to(b/'source').as_posix();require(rel in sourcePaths,'ownedCompilerDepManifested:'+rel);owned.append(rel)
        deps.append({'object':obj.relative_to(b).as_posix(),'dependency_sha256':sha(dep.read_bytes()),'owned_inputs':sorted(set(owned))})
    policy=subprocess.run([str(b/'policy-cpu')],capture_output=True,text=True,check=True);policyJSON=json.loads(policy.stdout)
    require(policyJSON['pass'] and policyJSON['planned_bytes']==310181888 and not policyJSON['gpu_executed'],'CPUactualGeometryAndStrictDependencyCases')
    workerCPU=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True)
    workerJSON=json.loads(workerCPU.stdout);require(workerJSON['valid'] and len(workerJSON['checks'])==49,'actualWorker49CPUSuite')
    ambient={k:v for k,v in __import__('os').environ.items()if not k.startswith('SPLASH_FLASH_')}
    refuse=[]
    cases=[('invalidSyntax',{'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'2'},'must be 0 or 1'),
      ('invalidDependency',{'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'1'},'requires original MTP'),
      ('invalidQAPause',{'SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS':'1000'},'pause must be 0 or 500 ms'),
      ('pauseWithoutBulk',{'SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS':'500'},'pause requires singleton bulk')]
    for label,flags,marker in cases:
        result=subprocess.run([str(b/'splash-flash'),'serve-flash-native','/nonexistent/package','16384','auto'],env={**ambient,**flags},capture_output=True,text=True)
        require(result.returncode==3 and marker in result.stderr,'compiledPrePathBackendRefusal:'+label)
        refuse.append({'case':label,'returncode':result.returncode,'stderr':result.stderr.strip()})
    artifacts={str(p.relative_to(b)):sha(p.read_bytes())for p in linked+[b/'splash-flash',b/'policy-cpu',b/'splash.metallib']}
    out={'schema':'singleton-teacher-bulk-whole-worker-compiled-cpu-seal-v1','pass':True,'gpu_executed':False,'model_payload_bytes_read':0,'source_manifest_sha256':sha(mp.read_bytes()),'source_sha256':{r['path']:r['sha256']for r in m['files']},'artifact_sha256':artifacts,'effective_objects':len(linked),'compiled_host_dependency_closure':deps,'cpu_policy':policyJSON,'compiled_refusals':refuse,'witness_source_sha256':sha(Path(__file__).read_bytes()),'checks':checks,'whole_worker_GPU_qualification_complete':False,'component_GPU_proof_report_sha256':m['component_gpu_report_sha256'],'actualWorkerCPU':workerJSON}
    (b/'compiled-cpu-seal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'effective_objects':54,'source_count':len(m['files']),'geometry_checks':policyJSON['geometry_checks'],'GPU_qualification_complete':False}))
if __name__=='__main__':main()
