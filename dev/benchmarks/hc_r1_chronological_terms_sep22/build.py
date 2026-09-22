#!/usr/bin/env python3
"""Compile only AFTER independent narrow Source/AIR GO, no GPU/model access."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--Source-review',type=Path,required=True);p.add_argument('--Source-review-sha256',required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'manifest.json').read_text());review=json.loads(a.Source_review.read_text())
 if sha(a.Source_review)!=a.Source_review_sha256 or not review.get('pass')or review.get('source_identity_sha256')!=m['source_identity_sha256']:raise ValueError('exact independent SOURCE/AIR review required before compile')
 for r in m['sources']:
  if sha(b/'source'/r['path'])!=r['sha256']:raise ValueError('frozencomponent source drift:'+r['path'])
 for r in m['frozen']:
  if sha(b/r['path'])!=r['sha256']:raise ValueError('currentCore/originalHC AIR drift')
 private=b/'source'/m['private_dir'];metal=['xcrun','-sdk','macosx','metal','-std=metal4.1','-O3','-Wall','-Wextra','-Werror','-mmacosx-version-min=27.0','-I'+str(b/'source/runtime'),'-I'+str(private),'-c',str(private/'candidate.metal'),'-o',str(b/'candidate.air')];subprocess.run(metal,cwd=ROOT,check=True);linkmetal=['xcrun','-sdk','macosx','metallib',str(b/'air/original089-HC.air'),str(b/'candidate.air'),'-o',str(b/'splash.metallib')];subprocess.run(linkmetal,cwd=ROOT,check=True);flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b),'-I'+str(b/'source/runtime'),'-I'+str(private)];cpp=['xcrun','-sdk','macosx','clang++',*flags,str(private/'oracle.mm'),*[str(b/r['path'])for r in m['frozen']if r['category']=='Core4'],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'oracle')];subprocess.run(cpp,cwd=ROOT,check=True);cpu=subprocess.run([str(b/'oracle'),'--cpu-only'],capture_output=True,text=True,check=True);(b/'cpu-self-test.json').write_text(cpu.stdout);result={'schema':'HC-R1termcomponent-compiled-CPU-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'compiler_commands':[metal,linkmetal,cpp],'artifact_sha256':{n:sha(b/n)for n in ['candidate.air','splash.metallib','oracle']},'GPU_started':False,'payload_reads':False,'actualFP_AIR_independent_review_required_before_RootGPU':True};(b/'compiled-cpu-seal.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'CPUbuild_pass':True,'GPU_started':False,'source_identity_sha256':m['source_identity_sha256']}))
if __name__=='__main__':main()
