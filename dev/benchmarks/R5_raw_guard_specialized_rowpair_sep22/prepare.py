#!/usr/bin/env python3
"""CPU-only bounded dynamic-shape oracle composition; no model/GPU reads."""
import argparse,hashlib,json,pathlib,shutil,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'build/R5-integer-currentQ4-fixed4-sep22-worker-v2';HOST=ROOT/'build/R5-raw-large-odd-rowpair-sep22-component-v2/source/dev/benchmarks/R5_raw_odd_rowpair_sep22';KERNEL=HERE/'kernel/_cpu_build_v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--build',type=pathlib.Path,required=True);a=p.parse_args();b=a.build.resolve()
 if b.exists() or ROOT/'build' not in b.parents:raise ValueError('Fresh private component build required')
 host=(HOST/'oracle.mm').read_text()
 if sha(HOST/'oracle.mm')!='d13b7328f77ebf0b32fd3aa02a773e27ad7e65972a23c173b3cb02f315cbfae2':raise ValueError('Frozen HostV2 source changed')
 if 'resource_gate_pass' not in host or 'sampled_process_cache_within_budget' not in host:
  raise ValueError('Reviewed Host V2 owned-zero/bounded-process resource source required')
 if sha(BASE/'splash.metallib')!='dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6':raise ValueError('Current library changed')
 parent=json.loads((BASE/'compiled-cpu-seal.json').read_text());km=json.loads((KERNEL/'build-manifest.json').read_text());audit=KERNEL/'strict-SSA-audit.json'
 if not audit.exists():raise ValueError('Current exactkernel audit notready')
 ar=json.loads(audit.read_text())
 if ar.get('pass') is not True:raise ValueError('Strict actual perrow arithmetic audit mustpass')
 if km.get('shipping_native_control_AIR_sha256')!='efec83efc7c8aa89c69d1c322c91848bd446c434bd5dde3ca430e1fae544f5c4':raise ValueError('Immutable original AIR must remain exact')
 for n,digest in km['artifacts'].items():
  if sha(KERNEL/n)!=digest:raise ValueError('Kernel artifact changed:'+n)
 b.mkdir();shutil.copytree(BASE/'source',b/'source');own=b/'source/dev/benchmarks/R5_raw_odd_rowpair_sep22';shutil.copytree(HOST,own);shutil.copytree(HERE,b/'source'/HERE.relative_to(ROOT),ignore=shutil.ignore_patterns('_cpu_build*','__pycache__'))
 core=[];pins=[]
 for item in parent['compiled_objects']:
  if '/core/' not in item['path']:continue
  src=BASE/item['path'];dst=b/'core'/src.name;dst.parent.mkdir(exist_ok=True)
  if sha(src)!=item['sha256']:raise ValueError('CurrentCore4 changed')
  shutil.copy2(src,dst);core.append(dst);pins.append({'path':str(dst.relative_to(b)),'source':str(src),'sha256':sha(dst)})
 if len(core)!=4:raise ValueError('Actual currentCore4 census differs')
 for n in ['component.metallib','native-qmv.air','candidate.air','control_probe.air','candidate_probe.air','build-manifest.json','strict-SSA-audit.json']:
  shutil.copy2(KERNEL/n,b/n)
 provenance={'schema':'current-R5-odd-rowpair-guard-specialized-program-provenance-v1','host_accounting_scope':'owned-zero; measured-process-cache-bounded','parent':str(BASE),'parent_source_identity':'4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a','library_scope':'immutableoriginal8exports+4shipping+8probe private exports; noWorkerlibrary','library_sha256':sha(b/'component.metallib'),'kernel_audit_sha256':sha(b/'strict-SSA-audit.json'),'kernel_manifest_sha256':sha(b/'build-manifest.json'),'native_AIR_sha256':sha(b/'native-qmv.air'),'Core4':pins}
 (b/'Provenance.hpp').write_text('#pragma once\ninline constexpr const char*kR5RawOddRowpairProvenance=R"PROV('+json.dumps(provenance,sort_keys=True)+')PROV";\ninline constexpr const char*kR5RawOddRowpairLibrarySHA="'+provenance['library_sha256']+'";\ninline constexpr const char*kR5RawOddRowpairKernelAuditSHA="'+provenance['kernel_audit_sha256']+'";\n')
 boundary=b/'source/dev/benchmarks/FlashFloatBoundaryAudit.hpp'
 if not boundary.exists():shutil.copy2(ROOT/'dev/benchmarks/FlashFloatBoundaryAudit.hpp',boundary)
 cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(own),'-I'+str(b),'-I'+str(b/'source/runtime'),'-I'+str(b/'source'),'-MMD','-MP',str(own/'oracle.mm'),*map(str,core),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'oracle')];subprocess.run(cmd,cwd=ROOT,check=True)
 cpu=subprocess.run([str(b/'oracle'),'--cpu-self-test'],cwd=ROOT,capture_output=True,text=True,check=True);help=subprocess.run([str(b/'oracle'),'--help'],cwd=ROOT,capture_output=True,text=True,check=True);(b/'CPU-self-test.json').write_text(cpu.stdout);(b/'help.txt').write_text(help.stdout)
 r={'schema':'current-R5-large-odd-rowpair-guard-specialized-CPU-v1','host_accounting_scope':'owned-zero; measured-process-cache-bounded','previous_all7_performance_loss_preserved_and_not_regraded':True,'guard_only_specialization':True,'pass':True,'GPU_work':False,'model_tensor_token_capture_profile_or_generation_payload_reads':0,'no_worker_fastpath_integration':True,'Core4':pins,'kernel_program_manifest_sha256':sha(b/'build-manifest.json'),'strict_kernel_audit_sha256':sha(b/'strict-SSA-audit.json'),'provenance':provenance,'oracle_sha256':sha(b/'oracle'),'library_sha256':sha(b/'component.metallib'),'CPU_selftest_sha256':sha(b/'CPU-self-test.json'),'compiler_command':cmd,'sources':[{'path':str(x.relative_to(b/'source')),'sha256':sha(x)} for x in sorted((b/'source').rglob('*')) if x.is_file()],'independent_review_pending':True,'actual_current_input_or_GPU_qualified':False}
 (b/'CPU_READY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'build':str(b),'CPU_READY_sha256':sha(b/'CPU_READY.json'),'oracle_sha256':r['oracle_sha256'],'library_sha256':r['library_sha256'],'GPU_work':False}))
if __name__=='__main__':main()
