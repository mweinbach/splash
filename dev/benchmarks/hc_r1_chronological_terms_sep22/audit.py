#!/usr/bin/env python3
"""CPU-only current source/Core/AIR pins and independent compiled FP attr witness."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'manifest.json').read_text());s=json.loads((b/'compiled-cpu-seal.json').read_text());checks=[]
 def check(v,k):
  checks.append({'check':k,'pass':bool(v)})
  if not v:raise ValueError(k)
 for r in m['sources']:check(sha(b/'source'/r['path'])==r['sha256'],'current frozenSource:'+r['path'])
 for r in m['frozen']:check(sha(b/r['path'])==r['sha256'],'current frozenCoreAIR:'+r['path'])
 for n,v in s['artifact_sha256'].items():check(sha(b/n)==v,'compiledartifact:'+n)
 private=b/'source'/m['private_dir'];k=(private/'candidate.metal').read_text();o=(private/'oracle.mm').read_text();check('#pragma clang fp contract(off)'in k and'#pragma clang fp reassociate(off)'in k,'newbothstageFPpragmas');check('320'in k and'partialsum'not in k.lower(),'fixed chronology no partialsum source');check('warm[0]<.150||warm[1]<.150'in o and'pair<18'in o,'inclusive150mseach18balanced source');start=o.index('while(warm[0]');end=o.index('require(outputsEqual',start);window=o[start:end];check(not any(x in window for x in ['contents()','memcpy','memset','immutable()','hash(']),'noCPUbuffertoucheswarm/samples')
 result={'schema':'HC-R1termcomponent-CPU-source-artifact-audit-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'checks':checks,'GPU_started':False,'payload_reads':False,'compiled_FP_attributes_review_pending':True,'Root_bitqualification_pending':True};(b/'cpu-source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'CPUauditpass':True,'checks':len(checks),'actualFPattrs_inclusive_bitGates_pending':True,'GPU_started':False}))
if __name__=='__main__':main()
