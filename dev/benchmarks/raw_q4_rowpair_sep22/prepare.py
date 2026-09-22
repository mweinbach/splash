#!/usr/bin/env python3
"""Build/freeze the single raw-Q4 rowpair component without GPU/payload access."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c):subprocess.run(c,cwd=ROOT,check=True)
def read_ir(p,out):run(['xcrun','air-opt','-S',str(p),'-o',str(out)])
def bodies(ir):
    return {m.group(1).strip('"'):m.group(2)for m in re.finditer(r'^define\b[^\n]*?@([^\n]*?)\([^\n]*\)[^\n]*\{\n(.*?)^\}',ir,re.M|re.S)}
def floating(ir):
    functions={}
    for match in re.finditer(r'^define\b[^\n]*?@([^\n]*?)\([^\n]*\)[^\n]*\{\n(.*?)^\}',ir,re.M|re.S):
        name,body=match.groups();ops=[]
        for line in body.splitlines():
            if re.search(r'\b(fadd|fsub|fmul|fdiv|frem|fptrunc|fpext|uitofp|sitofp|fcmp)\b',line) or re.search(r'call.*@(?:air|llvm)\.[^ (]*(?:f32|bf16|f16)',line):
                line=re.sub(r'%[A-Za-z0-9._]+','%V',line.strip());line=re.sub(r'!\w+ !\d+','',line);ops.append(line)
        functions[name.strip('"')]=ops
    return functions
def kernel_ops(ir,name):
    funcs=bodies(ir)
    if name not in funcs:raise ValueError('required kernel LLVM body missing:'+name)
    def walk(fn,path):
        if fn in path:raise ValueError('recursive floating-prefix call graph')
        ops=[]
        for line in funcs[fn].splitlines():
            calls=re.findall(r'\bcall\b[^@]*@([^ (]+)\(',line)
            if calls and calls[0].strip('"')in funcs:ops+=walk(calls[0].strip('"'),path+[fn])
            elif re.search(r'\b(fadd|fsub|fmul|fdiv|frem|fptrunc|fpext|uitofp|sitofp|fcmp)\b',line)or re.search(r'call.*@(?:air|llvm)\.[^ (]*(?:f32|bf16|f16)',line):
                line=re.sub(r'%[A-Za-z0-9._]+','%V',line.strip());line=re.sub(r'!\w+ !\d+','',line);line=re.sub(r'#\d+','',line);ops.append(line)
        return ops
    result=walk(name,[])
    if not result:raise ValueError('vacuous floating-prefix witness')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=ROOT/'build/raw-q4-rowpair-sep22-component-v1');a=p.parse_args();b=a.build.resolve()
    if b.exists():raise ValueError('fresh rowpair build directory required')
    parent=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5';airParent=ROOT/'build/compact-native-r4-verify-teacher-sep22-worker-v1b'
    pm=json.loads((parent/'overlay-manifest.json').read_text());am=json.loads((airParent/'overlay-manifest.json').read_text())
    if sha(parent/'splash.metallib')!=pm['metallib_sha256'] or pm['metallib_sha256']!='bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0':raise ValueError('qualified native library differs')
    ar=[x for x in am['frozen_inputs'] if x['path'].endswith('/074-flash_affine_qmv_f32.air')]
    if len(ar)!=1:raise ValueError('exact native QMV AIR census differs')
    oldAir=airParent/ar[0]['path']
    if sha(oldAir)!=ar[0]['sha256'] or ar[0]['sha256']!='efec83efc7c8aa89c69d1c322c91848bd446c434bd5dde3ca430e1fae544f5c4':raise ValueError('immutable native QMV AIR differs')
    # Copy only source/code/object artifacts. This source tree contains no data.
    b.mkdir(parents=True);shutil.copytree(parent/'source',b/'source');own=b/'source/dev/benchmarks/raw_q4_rowpair_sep22';shutil.copytree(HERE,own)
    audit=b/'source/dev/benchmarks/FlashFloatBoundaryAudit.hpp';shutil.copy2(ROOT/'dev/benchmarks/FlashFloatBoundaryAudit.hpp',audit)
    frozen=b/'native-qmv.air';shutil.copy2(oldAir,frozen)
    cores=[];pins=[]
    for record in pm['frozen_objects'][:4]:
        src=parent/record['path']
        if sha(src)!=record['sha256']:raise ValueError('parent native Core4 object differs')
        dst=b/'core'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst);cores.append(dst);pins.append({'source':str(src),'path':str(dst),'sha256':sha(dst)})
    original=b/'source/runtime/metal/kernels/shared/flash_affine_qmv_f32.metal'
    commands=[[sys.executable,str(own/'generate.py'),'--original',str(original),'--source',str(own),'--output',str(b/'generated')]]
    run(commands[-1]);shutil.copy2(own/'pair.metalh',b/'generated/pair.metalh');shutil.copy2(own/'candidate.metal',b/'generated/candidate.metal')
    metal=['xcrun','-sdk','macosx','metal','-std=metal4.1','-O3','-Wall','-Wextra','-Werror','-mmacosx-version-min=27.0','-I'+str(b/'source/runtime'),'-I'+str(b/'generated')]
    private=[]
    for name in ['candidate','control_probe','candidate_probe']:
        out=b/(name+'.air');c=metal+['-c',str(b/'generated'/(name+'.metal')),'-o',str(out)];commands.append(c);run(c);private.append(out)
    # Scoped primitive library: eight immutable shipping-QMV exports plus
    # three new private exports. It is intentionally not a Worker library.
    c=['xcrun','-sdk','macosx','metallib',str(frozen),*map(str,private),'-o',str(b/'component.metallib')];commands.append(c);run(c)
    for name,path in [('native-qmv',frozen),*[(p.stem,p)for p in private]]:read_ir(path,b/(name+'.ll'))
    ni=(b/'native-qmv.ll').read_text();ci=(b/'control_probe.ll').read_text();ti=(b/'candidate.ll').read_text();pi=(b/'candidate_probe.ll').read_text()
    old=kernel_ops(ni,'flash_affine_mlx_qmv_f32xsum_v1_q4_g64');control=kernel_ops(ci,'raw_q4_rowpair_sep22_control_probe');timed=kernel_ops(ti,'raw_q4_rowpair_sep22_timed');probe=kernel_ops(pi,'raw_q4_rowpair_sep22_candidate_probe')
    policy=lambda s:sorted(set(re.findall(r'(?:fast_math_(?:enable|disable)|denorms_(?:enable|disable)|"denormal-fp-math"="[^"]+"|"unsafe-fp-math"="[^"]+")',s)))
    llvm={'schema':'splash-raw-q4-rowpair-actual-AIR-opcode-census-v2','native_control_actual_AIR_sha256':sha(frozen),'original_source_sha256':sha(original),
        'native_vs_control_ordered_opcode_intrinsic_attributes_equal':old==control,'timed_vs_candidate_probe_ordered_opcode_intrinsic_attributes_equal':timed==probe,
        'SSA_operand_dependency_equivalence_proved':False,'GPU_register_equality_proved':False,
        'native_kernel_FP_ops':old,'control_probe_kernel_FP_ops':control,'candidate_timed_kernel_FP_ops':timed,'candidate_probe_kernel_FP_ops':probe,
        'module_FP_policy':{'native':policy(ni),'control':policy(ci),'candidate':policy(ti),'candidate_probe':policy(pi)},
        'native_alloca_present':bool(re.search(r'\balloca\b',ni)),'candidate_alloca_present':bool(re.search(r'\balloca\b',ti)),
        'candidate_alloca_declarations':re.findall(r'^.*\balloca\b.*$',ti,re.M),
        'native_Q4_fast_alloca_declarations':re.findall(r'^.*\balloca\b.*$',next(v for k,v in bodies(ni).items()if 'project_mathILt4ELt64ELb1' in k),re.M),
        'scope':'nonempty transitive outlined-call-expanded ordered floating opcode/intrinsic/attribute census AFTER ERASING SSA operand identities. Equal census is NOT a dependency-tree or register-equality proof. Literal source-byte restoration and mandatory actual shipping/probe GPU equality are separate gates. Original operand identities remain in retained LL files.'}
    (b/'LLVM-prefix.json').write_text(json.dumps(llvm,indent=2)+'\n')
    provenance={'schema':'splash-raw-q4-rowpair-code-provenance-v1','qualified_parent':str(parent),'qualified_parent_library_sha256':pm['metallib_sha256'],'native_QMV_AIR_sha256':sha(frozen),'native_QMV_source_sha256':sha(original),'original_QMV_AIR_source':str(oldAir),'core4':pins,'component_library_sha256':sha(b/'component.metallib'),'component_library_scope':'eight native QMV entries plus3 private entries; no Worker/whole-model library claim','LLVM_witness_sha256':sha(b/'LLVM-prefix.json'),'source_journal_sha256':sha(b/'generated/source-journal.json')}
    (b/'Provenance.hpp').write_text('#pragma once\ninline constexpr const char*kRawQ4RowpairProvenance=R"PROV('+json.dumps(provenance,sort_keys=True)+')PROV";\ninline constexpr const char*kRawQ4RowpairLibrarySHA='+json.dumps(provenance['component_library_sha256'])+';\n')
    cpp=['xcrun','-sdk','macosx','clang++','-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(own),'-I'+str(b),'-I'+str(b/'source'),'-I'+str(b/'source/runtime')]
    c=cpp+[str(own/'oracle.mm'),*map(str,cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'oracle')];commands.append(c);run(c)
    cpu=subprocess.run([str(b/'oracle'),'--cpu-self-test'],cwd=ROOT,check=True,capture_output=True,text=True);(b/'cpu-self-test.json').write_text(cpu.stdout)
    help_result=subprocess.run([str(b/'oracle'),'--help'],cwd=ROOT,check=True,capture_output=True,text=True);(b/'help.txt').write_text(help_result.stdout)
    shader_meta={'original_AIR':str(frozen),'original_AIR_sha256':sha(frozen),'actual_native_QMV_compiler_policy':llvm['module_FP_policy']['native'],'new_Metal_recipe':metal,'new_AIRs':[{'path':str(p),'sha256':sha(p)}for p in private]}
    manifest={'schema':'splash-raw-q4-rowpair-CPU-component-v1','CPU_build_complete':True,'GPU_executed':False,'model_payload_reads':0,'capture_payload_reads':0,'operand_payload_reads':0,'GPU_qualification_complete':False,'source_files':[{'path':str(p),'sha256':sha(p)}for p in sorted((b/'source').rglob('*'))if p.is_file()],
        'owned_source_files':[{'path':str(p),'sha256':sha(p)}for p in sorted(own.rglob('*'))if p.is_file()], 'core4':pins,'shader':shader_meta,'compiler_commands':commands,'provenance':provenance,'oracle_sha256':sha(b/'oracle'),'component_library_sha256':sha(b/'component.metallib'),'CPU_report_sha256':sha(b/'cpu-self-test.json'),
        'LLVM_prefix_witness':llvm,'independent_review_complete':False,'Root_run_authorized':False}
    (b/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'CPU_build_complete':True,'GPU_work':False,'payload_reads':0,'build':str(b),'LLVM_native_control_kernel_prefix_equal':old==control,'LLVM_candidate_probe_kernel_prefix_equal':timed==probe,'candidate_alloca':llvm['candidate_alloca_present']}))
if __name__=='__main__':main()
