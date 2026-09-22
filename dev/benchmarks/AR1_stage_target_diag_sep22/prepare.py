#!/usr/bin/env python3
"""CPU closure only: Worker1,49 host imports,Core4,authentic ordered754 AIR."""
import argparse,difflib,hashlib,importlib.util,json,pathlib,re,shutil,subprocess

ROOT=pathlib.Path(__file__).resolve().parents[3]
HERE=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
EXE='663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
LIB='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
SOURCE='162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a'

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--build',type=pathlib.Path,required=True)
    a=p.parse_args();b=a.build.resolve()
    need(not b.exists() and ROOT/'build' in b.parents,'Fresh private build required')
    seal=json.loads((BASE/'compiled-cpu-seal.json').read_text())
    parent=json.loads((BASE/'overlay-manifest.json').read_text())
    need(seal['pass'] and seal['source_identity_sha256']==SOURCE,'Current source seal drift')
    need(sha(BASE/'splash-flash')==EXE and sha(BASE/'splash.metallib')==LIB,'Current663/754 drift')
    for record in seal['compiled_objects']:
        need(sha(BASE/record['path'])==record['sha256'],'Current54 object drift:'+record['path'])
    for record in parent['files']:
        need(sha(BASE/'source'/record['path'])==record['sha256'],'Current source/header drift:'+record['path'])
    shutil.copytree(BASE,b)
    for name in ('compiled-cpu-seal.json','overlay-manifest.json'):
        (b/name).rename(b/('normal-parent-'+name))
    private=HERE.relative_to(ROOT)
    shutil.copytree(HERE,b/'source'/private)
    spec=importlib.util.spec_from_file_location('_privateAR1overlay',HERE/'overlay.py')
    overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
    rel='runtime/flash/FlashWorker.mm';original=(BASE/'source'/rel).read_text();changed=overlay.transform(rel,original)
    (b/'source'/rel).write_text(changed)
    before,after=original.splitlines(keepends=True),changed.splitlines(keepends=True)
    edits=[]
    for tag,i,j,x,y in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
        if tag!='equal':edits.append(dict(old_start=i,old_end=j,new_start=x,new_end=y,old_lines=before[i:j],new_lines=after[x:y]))
    restored=list(after)
    for edit in reversed(edits):restored[edit['new_start']:edit['new_end']]=edit['old_lines']
    need(restored==before,'Worker inverse differs from original normal source')
    for record in parent['files']:
        if record['path']!=rel:
            need(sha(b/'source'/record['path'])==record['sha256'],'Unaffected source/header changed')
    diag_id=hashlib.sha256((SOURCE+sha(b/'source'/rel)+sha(HERE/'bridge.hpp')+sha(HERE/'overlay.py')).encode()).hexdigest()
    flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc',
        '-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-DSPLASH_AR1_DIAGNOSTIC_SOURCE_ID="'+diag_id+'"',
        '-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')]
    link=(b/'link-inputs.mk').read_text()
    names=next(x for x in link.splitlines() if x.startswith('REBUILD_NAMES :=')).split(':=')[1].split()
    sources={m.group(1):m.group(2) for m in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$',link,re.M)}
    cores=[b/x.removeprefix('$(BUILD)/') for x in next(y for y in link.splitlines() if y.startswith('CORE :=')).split(':=')[1].split()]
    objects=[b/'host'/(name+'.o') for name in names]
    need(len(names)==50 and len(cores)==4,'Authentic50 host/Core4 closure required')
    census=[];pins=[];commands=[]
    for name,obj in zip(names,objects):
        source=b/'source'/sources[name]
        dep=subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(source)],cwd=ROOT,capture_output=True,text=True,check=True).stdout.replace('\\\n',' ').split()
        consumer=any(x.endswith('/AR1_stage_target_diag_sep22/bridge.hpp') for x in dep)
        need(consumer==(name=='FlashWorker'),'Private diagnostic consumer drift:'+name)
        census.append(dict(object=name,diagnostic_consumer=consumer,dependencies=dep))
        if consumer:
            command=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(source),'-o',str(obj)]
            commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
        else:need(sha(obj)==sha(BASE/obj.relative_to(b)),'Imported49 host object changed')
        pins.append(dict(path=str(obj.relative_to(b)),sha256=sha(obj),parent_sha256=sha(BASE/obj.relative_to(b)),changed=consumer))
    for obj in cores:
        need(sha(obj)==sha(BASE/obj.relative_to(b)),'Core4 object changed')
        pins.append(dict(path=str(obj.relative_to(b)),sha256=sha(obj),parent_sha256=sha(BASE/obj.relative_to(b)),changed=False))
    library_command=next(c for c in parent['compiler_commands'] if len(c)>3 and c[3]=='metallib' and c[-1]==str(BASE/'splash.metallib'))
    authentic_air=[pathlib.Path(x) for x in library_command[4:-2]]
    air_pins=[];new_air=[]
    for old_air in authentic_air:
        rel_air=old_air.relative_to(BASE);new=b/rel_air
        need(sha(old_air)==sha(new),'Authentic ordered AIR changed')
        air_pins.append(dict(path=str(rel_air),sha256=sha(new)));new_air.append(str(new))
    command=['xcrun','-sdk','macosx','metallib',*new_air,'-o',str(b/'splash.metallib')]
    commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    need(sha(b/'splash.metallib')==LIB,'Authentic754 reconstruction differs')
    command=['xcrun','-sdk','macosx','clang++',*flags,*map(str,objects+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')]
    commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    result=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],cwd=ROOT,capture_output=True,text=True,check=True)
    ready=dict(schema='current-standard-AR1-Worker-only-stage-diagnostic-CPU-v1',pass_=True,
        GPU_work=False,model_tensor_token_file_or_response_payload_reads=0,
        parent=str(BASE),parent_exe_sha256=EXE,parent_source_identity_sha256=SOURCE,source_identity_sha256=SOURCE,
        diagnostic_source_id=diag_id,exe_sha256=sha(b/'splash-flash'),library_sha256=LIB,
        changed_paths=[rel],private_header_consumers=['FlashWorker'],actual50TU_census=census,
        compiled54_objects=pins,host_rebuilds=1,authenticated_host_imports=49,current_Core4_unchanged=True,
        authentic_ordered_AIR=air_pins,authentic754_reconstructed=True,new_shader_or_FP_compiles=0,
        Forward_header_or_Graph_changes=0,normal_source_identity_retained=True,
        canonical_throughput_claim=False,SourceWorld_qualification_claim=False,
        CPU_self_test=json.loads(result.stdout),journal=[dict(path=rel,parent_sha256=sha(BASE/'source'/rel),sha256=sha(b/'source'/rel),inverse_literal_exact=True,edits=edits)],
        compiler_commands=commands,files=[dict(path=str(x.relative_to(b/'source')),sha256=sha(x)) for x in sorted((b/'source').rglob('*')) if x.is_file()])
    ready['pass']=ready.pop('pass_')
    for name in ('compiled-cpu-seal.json','overlay-manifest.json','CPU_READY.json'):
        (b/name).write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(dict(build=str(b),CPU_READY_sha256=sha(b/'CPU_READY.json'),exe_sha256=ready['exe_sha256'],library_sha256=LIB,diagnostic_source_id=diag_id,GPU_work=False)))

if __name__=='__main__':main()
