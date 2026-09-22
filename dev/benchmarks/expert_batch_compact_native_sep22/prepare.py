#!/usr/bin/env python3
"""Seal FIRST compact-integer/native M16 sources and existing link inputs only."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_batch_compact_native_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(d)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21-worker-v1');p.add_argument('--rows',type=int,choices=(8,16),required=True);p.add_argument('--output',type=Path,default=ROOT/'build/expert-batch-r8-compact-native-sep22-component-v1');a=p.parse_args();parent=a.parent.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('NEW private component output required')
 pm=parent/'overlay-manifest.json';m=json.loads(pm.read_text());records=[]
 def source(rel,data,**meta):write(out/'source'/rel,data);records.append({'path':str(rel),'sha256':sha(data),'bytes':len(data),**meta})
 for r in m['files']:
  data=(parent/'source'/r['path']).read_bytes()
  if sha(data)!=r['overlay_sha256']:raise ValueError('Parent source drift:'+r['path'])
  source(Path(r['path']),data,parent_source=True)
 for path in sorted((ROOT/PRIVATE).iterdir()):
  if path.is_file()and path.suffix in('.hpp','.h','.metal','.mm','.mk','.py','.md','.json'):source(PRIVATE/path.name,path.read_bytes(),component_source=True)
 for name in('prefill4k_allrows_qmv_oracle.mm','prefill4k_allrows_qmv_reference.hpp','prefill4k_allrows_qmv_one_layer.hpp','prefill4k_allrows_qmv_probe.h','prefill4k_allrows_gathered_mpp_probe.metal'):
  source(Path('dev/benchmarks')/name,(ROOT/'dev/benchmarks'/name).read_bytes(),frozen_helper=True)
 source(Path('runtime/flash/FlashGatheredI8QMV.hpp'),(ROOT/'build/prefill4k-allrows-qmv-c2-component/source/runtime/flash/FlashGatheredI8QMV.hpp').read_bytes(),frozen_helper=True)
 source(Path('runtime/metal/MetalBackend.mm'),(ROOT/'runtime/metal/MetalBackend.mm').read_bytes(),fresh_unmodified_backend=True)
 # A math-report helper only; native M16 producer source/AIR is untouched.
 quality=(ROOT/'dev/benchmarks/expert_r4_cohort_sep22/quality.hpp').read_bytes()
 source(PRIVATE/'quality.hpp',quality,frozen_quality_report_helper=True)
 probe=(ROOT/'dev/benchmarks/prefill4k_allrows_gathered_mpp_probe.metal').read_text().replace('"prefill4k_allrows_gathered_mpp.metal"','"metal/kernels/shared/flash_gathered_mpp.metal"');source(PRIVATE/'parent_gathered_probe.metal',probe.encode(),current_gathered_probe=True)
 parts={n:sha((ROOT/PRIVATE/n).read_bytes())for n in('abi.hpp','plan.metal','metadata.hpp','safety.hpp','oracle.mm','oracle.mk','prepare.py','probe.metal','witness.py','geometry-ledger.json','PREREGISTRATION.md')};parts.update({'parent_manifest':sha(pm.read_bytes()),'fixed_rows':a.rows,'schema':'fixedR8R16-parallel-integer-SIMDprefix-native-M16-original-fp-sixdispatch-v1'});identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode())
 source(PRIVATE/'source_identity.hpp',f'#pragma once\ninline constexpr char kCompactNativeBatchSourceIdentitySha256[]="{identity}";\n'.encode(),generated_identity=True)
 inputs={'COMPACT_FROZEN_HOST':[],'COMPACT_FROZEN_AIRS':[]};links=[]
 def freeze(path,rel,cat):
  data=path.read_bytes();write(out/rel,data);inputs[cat].append(rel.as_posix());links.append({'private_path':rel.as_posix(),'source_path':str(path),'sha256':sha(data),'bytes':len(data),'category':cat})
 for line in(parent/'link-inputs.mk').read_text().splitlines():
  cat,raw=line.split(' := ',1)
  for token in raw.split():
   if not token.startswith('$(BUILD)/'):raise ValueError('Unsealed link token')
   rel=Path(token[9:]);path=parent/rel
   if path.name in('FlashWorker.o','MetalBackend.o','flash_gathered_mpp.air'):continue
   freeze(path,Path('reused/parent')/rel,'COMPACT_FROZEN_AIRS'if cat=='AIRS'else'COMPACT_FROZEN_HOST')
 for path in sorted((parent/'host').glob('*.o')):
  if path.name!='FlashWorker.o':freeze(path,Path('reused/parent-host')/path.name,'COMPACT_FROZEN_HOST')
 for path in sorted(parent.glob('*.air')):
  if path.name!='flash_gathered_mpp.air':freeze(path,Path('reused/parent-air')/path.name,'COMPACT_FROZEN_AIRS')
 freeze(ROOT/'build/prefill4k-allrows-qmv-one-layer/probe.air',Path('reused/old-bucket-probe.air'),'COMPACT_FROZEN_AIRS')
 mk='COMPACT_ROWS := '+str(a.rows)+'\n'+'\n'.join(cat+' := '+' '.join('$(COMPACT_BUILD)/'+n for n in names)for cat,names in inputs.items())+'\n';write(out/'component-inputs.mk',mk.encode())
 seal={'schema':'splash-fixed-batch-compact-native-integer-source-seal-v1','fixed_rows':a.rows,'parent_build':str(parent),'parent_manifest_sha256':sha(pm.read_bytes()),'source_identity_sha256':identity,'identity_parts':parts,'sources':records,'link_inputs':links,'link_make_sha256':sha(mk.encode()),'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False,'model_or_capture_hashes':False,'dot_association_changed':False,'new_graph_allocations':0}
 write(out/'source-seal.json',(json.dumps(seal,indent=2)+'\n').encode());print(json.dumps({'sealed':str(out),'sources':len(records),'binary_inputs':len(links),'source_identity_sha256':identity,'gpu_work':False}))
if __name__=='__main__':main()
