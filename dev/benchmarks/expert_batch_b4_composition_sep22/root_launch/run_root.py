#!/usr/bin/env python3
"""Root-only QA controller: authenticate immutable source/seals before any GPU child."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def digest(value):return isinstance(value,str)and re.fullmatch(r'[0-9a-f]{64}',value)is not None
def read_pinned(path,expected,label):
 require(digest(expected),label+' external digest format')
 require(sha(path)==expected,label+' external digest mismatch')
 return json.loads(Path(path).read_text())
def validate(command_path,command_sha,native_seal_path,native_seal_sha,runner_sha):
 require(digest(runner_sha)and sha(Path(__file__))==runner_sha,'executed sealed machinery runner differs from external pin')
 seal=read_pinned(native_seal_path,native_seal_sha,'native final seal')
 require(seal.get('pass')is True and seal.get('GPU_executed')is False,'native CPU-only seal unavailable')
 c=read_pinned(command_path,command_sha,'launch command');b=Path(c['build']).resolve()
 require(Path(native_seal_path).resolve()==b/'final-cpu-seal.json','wrong native seal location')
 require(c.get('schema')=='current-BQSA4-integer-batchverify-pinned-Root-command-v2'and c.get('Root_GPU_only')is True,'unregistered command/source role')
 require(c.get('native_final_seal_sha256')==native_seal_sha and c.get('sealed_runner_sha256')==runner_sha,'command expected seal/runner pins differ')
 require(Path(c.get('sealed_runner_path','')).resolve()==Path(__file__).resolve(),'command executes another runner')
 native_ready=b/'CPU_READY.json';require(sha(native_ready)==seal['CPU_READY_SHA'],'native CPU_READY seal drift');m=json.loads(native_ready.read_text())
 require(m.get('pass')is True and m.get('GPU_executed')is False,'native current CPU closure unavailable')
 for name,expected in seal['artifact_sha256'].items():require(sha(b/name)==expected,'sealed native artifact drift:'+name)
 for name,expected in seal['machinery_source_SHA'].items():require(sha(b/'machinery'/name)==expected,'sealed native machinery drift:'+name)
 for r in m['sources']:require(sha(b/'source'/r['path'])==r['sha256'],'frozen native source drift:'+r['path'])
 for r in m['objects']:require(sha(b/r['path'])==r['sha256'],'current53 native object drift:'+r['path'])
 require(c['role']in ('export','compare')and c['argv'][1:3]==['--gpu',c['role']],'role argv mismatch')
 require(len(c['argv'])==9,'native oracle argv geometry mismatch')
 require(Path(c['argv'][0]).resolve()==b/('oracle-control'if c['role']=='export'else'oracle-candidate'),'different native binary path')
 require(Path(c['argv'][3]).resolve()==b/'splash.metallib','different native library path')
 require(sha(c['argv'][0])==c['oracle_sha256']and sha(c['argv'][3])==c['metallib_sha256'],'native pinned executable/library drift')
 labels=c['argv'][8].split(',');require(1<=len(labels)<=2 and len(set(labels))==len(labels),'bounded selected partition mismatch')
 require(all(x in seal['fullphysical_selected7']for x in labels),'unregistered checkpoint label')
 env=c['environment'];require(type(env)is dict and all(type(k)is str and type(v)is str for k,v in env.items()),'literal environment metadata required')
 require(env.get('SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22')==str(int(c['role']=='compare')),'integer source role mismatch')
 require(env.get('SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22')=='1'and env.get('SPLASH_FLASH_MTP_DRAFT_DEPTH')=='3','narrowed BQSA4/MTP3 profile mismatch')
 require(c.get('spill_limit_bytes')==4<<30 and c.get('fullphysical_all18_inherited')is False,'native bounded/current proof scope mismatch')
 return c

def main():
 p=argparse.ArgumentParser();p.add_argument('--command',type=Path,required=True);p.add_argument('--command-sha256',required=True);p.add_argument('--native-seal',type=Path,required=True);p.add_argument('--native-seal-sha256',required=True);p.add_argument('--runner-sha256',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 c=validate(a.command,a.command_sha256,a.native_seal,a.native_seal_sha256,a.runner_sha256)
 if a.validate_only:
  print(json.dumps({'pass':True,'validation_only':True,'GPU_started':False,'command':str(a.command),'native_seal_sha256':a.native_seal_sha256,'runner_sha256':a.runner_sha256}));return 0
 if c['role']=='compare':
  prior=json.loads(Path(c['control_report']).read_text())
  require(prior.get('pass')is True and prior.get('backend_destroyed')is True and prior.get('partition_complete')is True and prior.get('role')=='export','complete separate-process control export required')
  require(prior['common']['selected_checkpoints']==c['argv'][8].split(','),'comparison replay full-history selected partition mismatch')
 env={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};env.update(c['environment'])
 result=subprocess.run(c['argv'],cwd=c['cwd'],env=env);return result.returncode if result.returncode>=0 else 128-result.returncode
if __name__=='__main__':raise SystemExit(main())
