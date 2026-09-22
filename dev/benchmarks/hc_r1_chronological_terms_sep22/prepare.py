#!/usr/bin/env python3
"""Prepare standalone HC-R1 exact current source/Core4/original089 component only."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/hc_r1_chronological_terms_sep22')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2');p.add_argument('--output',type=Path,required=True);a=p.parse_args();parent=a.parent.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('NEW private component required')
 m=json.loads((parent/'overlay-manifest.json').read_text());sealed=json.loads((parent/'compiled-cpu-seal.json').read_text());art=sealed['artifacts']
 for r in art:
  if sha(parent/r['path'])!=r['sha256']:raise ValueError('currentparentartifact drift')
 binary=next(r['sha256']for r in art if r['path']=='splash-flash');library=next(r['sha256']for r in art if r['path']=='splash.metallib')
 if binary!='663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'or library!='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8':raise ValueError('exact current663/754 parent required')
 records=[]
 for r in m['files']:
  src=parent/'source'/r['path'];raw=src.read_bytes()
  if sha(src)!=r['sha256']:raise ValueError('currentparent source drift:'+r['path'])
  write(out/'source'/r['path'],raw);records.append({'path':r['path'],'sha256':sha(src),'parent_current':True})
 original=ROOT/'build/hc-pad-producer-sep22-v8/source/runtime/metal/kernels/shared/flash_hc_fused.metal'
 if sha(original)!='c32cff6a4b4aab8efc8b0cfcc8212c191684a4b5d9d451f77b5c4a19d17d9a16':raise ValueError('originalHC Source/recipe drift')
 rel=Path('runtime/metal/kernels/shared/flash_hc_fused.metal');write(out/'source'/rel,original.read_bytes());records.append({'path':str(rel),'sha256':sha(original),'authenticated_original_HC':True})
 for n in ['abi.hpp','candidate.metal','oracle.mm','storage.hpp','prepare.py','build.py','audit.py','PREREGISTRATION.md']:
  rel=PRIVATE/n;src=ROOT/rel;write(out/'source'/rel,src.read_bytes());records.append({'path':str(rel),'sha256':sha(src),'private_new':True})
 core=[r for r in m['frozen_inputs']if r['path'].endswith('.o')and'/core/'in r['path']]
 if len(core)!=4:raise ValueError('exact currentCore4 required')
 frozen=[]
 for r in core:
  src=parent/r['path'];dest=Path('core')/src.name
  if sha(src)!=r['sha256']:raise ValueError('currentCoreartifact drift')
  write(out/dest,src.read_bytes());frozen.append({'path':str(dest),'sha256':sha(src),'category':'Core4'})
 ancestor=Path(m['base']);am=json.loads((ancestor/'overlay-manifest.json').read_text());air=next(r for r in am['frozen_inputs']if r['path'].endswith('089-flash_hc_fused.air'));src=ancestor/air['path']
 if sha(src)!=air['sha256']or air['sha256']!='25a067444dd118a4a2799a4e4dd128c0d33bf1cc18a554885d2d6ec04036f69c':raise ValueError('currentoriginal089AIR drift')
 write(out/'air/original089-HC.air',src.read_bytes());frozen.append({'path':'air/original089-HC.air','sha256':sha(src),'category':'OriginalHC_AIR'})
 identity=hashlib.sha256(json.dumps({'parent':sha(parent/'compiled-cpu-seal.json'),'originalHCsource':sha(original),'originalHCAIR':sha(src),'privateSources':{r['path']:r['sha256']for r in records if r.get('private_new')},'scope':'standaloneR1HCmode0-separateroundedterms-chronological320adds-sourceScope-v1'},sort_keys=True,separators=(',',':')).encode()).hexdigest();provenance={'source_identity_sha256':identity,'current_parent':str(parent),'parent_executable_sha256':binary,'parent_library_sha256':library,'originalHCsource_sha256':sha(original),'originalHCAIR_sha256':sha(src),'Core4_only_no_opaquehostReuse':True,'shipping_scratch_bytes':13271040,'admission_bytes':256<<20,'selectedcoefs_limit':64<<20,'GPU_executed':False,'whole_state_or_model_qualified':False};write(out/'Provenance.hpp',('#pragma once\ninline constexpr const char*kHCChronologyProvenance=R"META('+json.dumps(provenance,sort_keys=True)+')META";\n').encode());manifest={'schema':'HC-R1chronological-term-standalone-source-preparation-v1','pass':True,'sources':records,'frozen':frozen,'provenance':provenance,'source_identity_sha256':identity,'private_dir':str(PRIVATE),'SourceAIR_independent_review_required_before_compile':True,'Root_GPU_only':True,'payload_reads':False};write(out/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'prepared':str(out),'sources':len(records),'Core4':len(core),'originalAIR':air['sha256'],'source_identity_sha256':identity,'GPU_started':False,'compile_started':False}))
if __name__=='__main__':main()
