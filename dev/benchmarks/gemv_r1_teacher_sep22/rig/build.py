#!/usr/bin/env python3
"""CPU-build current2K capture/state qualifier with private friends; Root GPU only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PRIVATE=Path('dev/benchmarks/gemv_r1_teacher_sep22/rig')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('diagnostic source anchor drift: '+a[:80])
 return s.replace(a,b,1)
def symbols(p):
 s=subprocess.check_output(['xcrun','-sdk','macosx','metal-nm','--defined-only',str(p)],text=True)
 return {line.split()[-1] for line in s.splitlines() if len(line.split())>=3 and line.split()[-2]=='T'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--role',choices=['control','candidate'],required=True)
 a=p.parse_args();worker,build=a.worker.resolve(),a.build.resolve()
 if build.exists():raise ValueError('fresh private diagnostic build required')
 seal=json.loads((worker/'compiled-cpu-seal.json').read_text());man=json.loads((worker/'overlay-manifest.json').read_text())
 if not seal['pass']:raise ValueError('sealed CPU worker required')
 artifact=dict(seal['artifact_sha256'])
 for r in man.get('frozen_inputs',[]):artifact[r['private_path']]=r['sha256']
 inherited=[(worker/k,v) for k,v in artifact.items() if k.endswith('.o') and Path(k).stem!='FlashWorker']
 if len(inherited)!=53:raise ValueError('exact53 nonworkerobjects required, observed'+str(len(inherited)))
 shutil.copytree(worker/'source',build/'source');private=build/'source'/PRIVATE;private.mkdir(parents=True,exist_ok=True)
 for name in ('inspect.hpp','capture.hpp','oracle.mm','build.py'):shutil.copyfile(HERE/name,private/name)
 # Freeze old qualified certificate source, and its diagnostic-only literal taps.
 vector=ROOT/'build/gemv-r1-ab-qsa-sep21-worker-v1';vm=json.loads((vector/'overlay-manifest.json').read_text())
 original=vector/'source/dev/benchmarks/gemv_decode_r1_worker_sep21/qualified-source'
 vector_abi=build/'source/dev/benchmarks/gemv_decode_r1_worker_sep21/abi.hpp'
 vector_abi.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vector/'source/dev/benchmarks/gemv_decode_r1_worker_sep21/abi.hpp',vector_abi)
 target=build/'source/dev/benchmarks/gemv_decode_sep21_v1b';target.mkdir(parents=True,exist_ok=True)
 for name in ('quality.hpp','kernels.metal','PREREGISTRATION.md','FTZ_CERTIFICATE.md'):shutil.copyfile(original/name,target/name)
 oldabi=ROOT/'build/prefill4k-allrows-qmv-c2-component/source/runtime/flash/FlashGatheredI8QMV.hpp'
 shutil.copyfile(oldabi,build/'source/runtime/flash/FlashGatheredI8QMV.hpp')
 for name in ('prefill4k_allrows_qmv_reference.hpp','prefill4k_allrows_qmv_probe.h','prefill4k_allrows_gathered_mpp.metal','prefill4k_allrows_gathered_mpp_probe.metal'):
  dest=build/'source/dev/benchmarks'/name
  if not dest.exists():shutil.copyfile(ROOT/'dev/benchmarks'/name,dest)
 # Some old oracle reference headers are not worker consumers; source only.
 for source in (ROOT/'dev/benchmarks').glob('*reference*.hpp'):
  dest=build/'source/dev/benchmarks'/source.name
  if not dest.exists():shutil.copyfile(source,dest)
 for header in ['FlashForward.hpp','FlashInt8ExpertStore.hpp']:
  f=build/'source/runtime/flash'/header;s=f.read_text();anchor='private:\n'
  # Forward has RequestState private plus Forward private; insert in final class.
  before,after=s.rsplit(anchor,1);f.write_text(before+anchor+'  friend class FlashDeepPrefixOracle;\n'+after)
 forward=build/'source/runtime/flash/FlashForward.cpp';s=forward.read_text();s='#include "inspect.hpp"\n#include "capture.hpp"\n'+s
 anchor='    const bool gatheredMPP = impl_->allRowsInt8Target && impl_->int8ExpertStore &&'
 s=once(s,anchor,'''    if (rows==1 && r1_capture::active()) {
      auto &capture=r1_capture::get(layer);
      impl_->copy(graph,mixed,capture.hidden,5120);
      impl_->copy(graph,ids,capture.ids,80);
    }
'''+anchor)
 anchor='    if (!batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,'
 s=once(s,anchor,'''    if (rows==1 && r1_capture::active()) {
      auto &capture=r1_capture::get(layer);
      impl_->copy(graph,bf(Scratch::ExpertIntermediate,kSelections*640),capture.activated,12800);
      impl_->copy(graph,bf(Scratch::ExpertDown,kSelections*kWidth),capture.down,51200);
    }
'''+anchor)
 s+='''
namespace splash::flash {
metal::MetalBuffer FlashDeepPrefixOracle::hidden(const FlashForward &target,uint32_t rows) {
  if(!target.impl_)throw std::logic_error("diagnostic target unavailable");
  return target.impl_->bf(Scratch::Hyper,rows,kHyper);
}
}
''';forward.write_text(s)
 store=build/'source/runtime/flash/FlashInt8ExpertStore.mm';s='#include "inspect.hpp"\n'+store.read_text()
 s+='''
namespace splash::flash {
std::array<metal::MetalBuffer,7> FlashDeepPrefixOracle::coefficients(const FlashInt8ExpertStore &store,uint32_t index) {
  if(!store.impl_)throw std::logic_error("diagnostic Store unavailable");
  const auto &layer=store.impl_->layer(index);
  if(store.impl_->metadata.layers[index].selectedIDs.size()!=512)throw std::logic_error("Full512 actual coefficient provenance required");
  return {layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],layer.codes[2],layer.scales[2],layer.ranks};
}
}
''';store.write_text(s)
 # Select source identity from current worker inventory, never guess ancestors.
 source_rows=list(man.get('header_dependency_census',[])) or list(man['rebuild'])+[{'object':'teacher_bulk','source':'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'}]
 byname={r['object']:r['source'] for r in source_rows}
 def name(path):
  stem=path.stem
  # CURRENT numbered names are meaningful identifiers; match exact first.
  if stem in byname:return stem
  if len(stem)>4 and stem[:3].isdigit() and stem[3]=='-':
   tail=stem[4:]
   if tail in byname:return tail
  return stem
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-ffp-contract=off','-fno-fast-math','-fobjc-arc',
        '-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-DSPLASH_R1_CANDIDATE='+str(int(a.role=='candidate')),
        '-I'+str(build/'source'),'-I'+str(build/'source/runtime'),'-I'+str(build/'source/dev/benchmarks'),'-I'+str(build/'source/dev/benchmarks/prefill4k_attention'),'-I'+str(private)]
 # Compiler flags for diagnostic arithmetic references are strict. Host worker
 # implementations retain their exact original O3/fast host recipe.
 hostflags=[x for x in flags if x not in ('-ffp-contract=off','-fno-fast-math')]
 census=[];objects=[]
 def prepare_object(pair):
  path,digest=pair
  if sha(path)!=digest:raise ValueError('currentobjectauthenticationdiffers: '+str(path))
  key=name(path);src=byname.get(key);destination=build/'objects'/path.name;destination.parent.mkdir(exist_ok=True)
  dependent=False;row=None
  if src:
   result=subprocess.run(['xcrun','-sdk','macosx','clang++',*hostflags,'-MM',str(build/'source'/src)],cwd=ROOT,text=True,capture_output=True,check=True)
   tokens=result.stdout.replace('\\\n',' ').split();dependent=any(t.endswith(('/runtime/flash/FlashForward.hpp','/runtime/flash/FlashInt8ExpertStore.hpp')) for t in tokens)
   row={'object':key,'source':src,'private_friend_consumer':dependent,'dependencies':tokens}
  if dependent:subprocess.run(['xcrun','-sdk','macosx','clang++',*hostflags,'-MMD','-MP','-c',str(build/'source'/src),'-o',str(destination)],cwd=ROOT,check=True)
  else:shutil.copyfile(path,destination)
  return row,{'path':str(destination),'sha256':sha(destination),'current_worker_path':str(path),'current_worker_sha256':digest,'private_friend_recompiled':dependent}
 with ThreadPoolExecutor(max_workers=5) as pool:
  for row,obj in pool.map(prepare_object,inherited):
   if row:census.append(row)
   objects.append(obj)
 if len(census)!=49:raise ValueError('all49 nonworkerhost sourcecensusrequired')
 # Also census (but never link) Worker, completing exact50-TU proof.
 result=subprocess.run(['xcrun','-sdk','macosx','clang++',*hostflags,'-MM',str(build/'source/runtime/flash/FlashWorker.mm')],cwd=ROOT,text=True,capture_output=True,check=True)
 census.append({'object':'FlashWorker','excluded_main':True,'dependencies':result.stdout.replace('\\\n',' ').split()})
 # Only tap exports are private: original shipping77 AIR source is never rebuilt.
 tap_source=private/'tap.metal';tap_source.write_text('''
#define flash_gathered_mpp_gate_up_m16_n64_sg4 r1diag_unused_gathered_gate
#define flash_gathered_mpp_down_m16_n64_sg4 r1diag_unused_gathered_down
#define flash_gathered_mpp_projection_probe r1diag_gathered_projection
#define gemv_decode_sep21_v4_l32_o4_gate_up r1diag_unused_vector_gate
#define gemv_decode_sep21_v4_l32_o4_down r1diag_unused_vector_down
#define gemv_decode_sep21_v4_l16_o8_gate_up r1diag_unused_l16_gate
#define gemv_decode_sep21_v4_l16_o8_down r1diag_unused_l16_down
#define gemv_decode_sep21_v4_l32_o4_projection_probe r1diag_vector_projection
#include "dev/benchmarks/prefill4k_allrows_gathered_mpp_probe.metal"
#include "dev/benchmarks/gemv_decode_sep21_v1b/kernels.metal"
''')
 tap_air=build/'tap.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.1','-O3','-Wall','-Wextra','-Werror','-mmacosx-version-min=27.0','-I'+str(build/'source'),'-I'+str(build/'source/runtime'),'-I'+str(build/'source/dev/benchmarks'),'-c',str(tap_source),'-o',str(tap_air)],cwd=ROOT,check=True)
 shipping=worker/'splash.metallib';library=build/'splash.metallib'
 # Parent77 inputs from approvedshippingmanifest; nativeTeacher control has only
 # old76. Add immutable oldvectorAIR in control too for same-plane standalonereplay.
 if a.role=='candidate':air_records=[r for r in man['frozen_inputs'] if r['category']=='AIRS'];airs=[worker/r['private_path'] for r in air_records]
 else:
  candidate=ROOT/'build/gemv-r1-teacher-sep22-worker-v1';cm=json.loads((candidate/'overlay-manifest.json').read_text())
  air_records=[r for r in cm['frozen_inputs'] if r['category']=='AIRS'];airs=[candidate/r['private_path'] for r in air_records]
  check=build/'baseline-control76.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*map(str,airs[:76]),'-o',str(check)],cwd=ROOT,check=True)
  if sha(check)!=sha(shipping):raise ValueError('controloriginal76 baseline mustequalnativeTeacher')
 frozenairs=[]
 for i,path in enumerate(airs):
  dest=build/'air'/f'{i:03d}-{path.name}';dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest);frozenairs.append(dest)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*map(str,frozenairs),str(tap_air),'-o',str(library)],cwd=ROOT,check=True)
 before=symbols(shipping);after=symbols(library)
 if not before<=after or 'r1diag_vector_projection' not in after:raise ValueError('diagnostic library mustretainallshippingexports')
 command=['xcrun','-sdk','macosx','clang++',*flags,str(private/'oracle.mm'),*[r['path'] for r in objects],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(build/'oracle')]
 subprocess.run(command,cwd=ROOT,check=True);cpu=json.loads(subprocess.check_output([str(build/'oracle'),'--cpu-only'],text=True))
 sourcepins={str(p):sha(p) for p in (build/'source').rglob('*') if p.is_file()}
 metadata={'schema':'current2K-standard-R1-diagnostic-only-build-v1','role':a.role,'worker':str(worker),'worker_cpu_seal_sha256':sha(worker/'compiled-cpu-seal.json'),
           'sources':sourcepins,'objects':objects,'all50_header_census':census,'metallib_sha256':sha(library),'oracle_sha256':sha(build/'oracle'),
           'shipping_arithmetic_recompiled':False,'diagnostic_taps_only_compiled':True,'shipping_export_count':len(before),'retained_shipping_export_count':len(before&after),
           'original_shipping_air_pins':[{'path':str(p),'sha256':sha(p)} for p in frozenairs],
           'host_object_count':53,'one_forward_per_process':True,'spill_bound_bytes':4<<30,'cpu':cpu,
           'gpu_work':False,'model_payload_bytes_read':0,'compiler_command':command}
 (build/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'prepared':str(build),'role':a.role,'cpu_pass':cpu['pass'],'objects':len(objects),'census':len(census),'shipping_arithmetic_recompiled':False,'gpu_work':False}))


if __name__=='__main__':main()
