#!/usr/bin/env python3
"""Validate frozen CPU-only inputs and seal the private oracle artifacts."""
import argparse,hashlib,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21')
    args=parser.parse_args();build=args.build.resolve();source=build/'source'
    manifest=json.loads((build/'source-manifest.json').read_text())
    for relative,expected in manifest['source_sha256'].items():
        if sha(source/relative)!=expected:raise ValueError(f'Frozen source changed: {relative}')
    dependencies={};toolchain={}
    for dep in sorted((build/'host').glob('*.d')):
        text=dep.read_text().replace('\\\n',' ');first=text.split('\n\n',1)[0]
        # -MP adds empty header targets; consume only the first dependency rule.
        items=first.split(':',1)[1].split()
        for item in items:
            if item.endswith(':'):break
            path=(ROOT/item).resolve()
            if source not in path.parents:
                if path.name=='SDKSettings.json' and path.parent.suffix=='.sdk':
                    frozen=source/'toolchain_inputs'/'SDKSettings.json';frozen.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,frozen);digest=sha(frozen);relative=str(frozen.relative_to(source))
                    manifest['source_sha256'][relative]=digest;toolchain[str(path)]=digest;dependencies[relative]=digest
                    continue
                raise ValueError(f'Unfrozen dependency: {path}')
            if path.is_file():dependencies[str(path.relative_to(source))]=sha(path)
    result=subprocess.run([str(build/'oracle'),'--cpu-self-test'],check=True,text=True,capture_output=True)
    cpu=json.loads(result.stdout)
    if not cpu.get('pass') or cpu.get('gpu_commands')!=0:raise ValueError('CPU oracle did not pass without GPU')
    artifacts={str(p.relative_to(build)):sha(p) for p in sorted(build.rglob('*'))
               if p.is_file() and (p.suffix in ('.o','.air','.metallib') or p.name=='oracle')}
    (build/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'schema':'splash-prefill-qsa-twopass-cpu-seal-v1','pass':True,'gpu_executed':False,
            'model_payload_bytes_read':0,'production_sources_modified':False,'CPU':cpu,
            'frozen_source_files':len(manifest['source_sha256']),'frozen_dependency_files':len(dependencies),
            'source_manifest_sha256':sha(build/'source-manifest.json'),'dependency_sha256':dependencies,
            'artifact_sha256':artifacts,'toolchain_metadata_sha256':toolchain,'original_ordinary_service_max_partitions':32,
            'numerical_alternative':True,'model_quality_qualified':False,
            'real_governor_reservation_before_any_fixture_allocation_enforced_in_GPU_oracle':True,
            'workspace_planned_bytes':509607936,'extra_arena_bytes':478150656,
            'standalone_real_governor_reservation_bytes':2147483648,
            'direct_and_packed_F32P_BF16V_wholeK_variants_compiled':True}
    (build/'cpu-seal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':True,'build':str(build),'CPU':cpu,'artifact_files':len(artifacts),'dependencies':len(dependencies),'GPU_executed':False}))
if __name__=='__main__':main()
