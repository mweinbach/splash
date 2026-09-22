#!/usr/bin/env python3
"""Root-only launch after the independent source/AIR/oracle review passes."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--run-root-gpu',action='store_true');p.add_argument('--real-input',type=Path);a=p.parse_args()
 if not a.run_root_gpu:raise ValueError('Root GPU execution requires explicit --run-root-gpu')
 b=a.build.resolve();s=json.loads((b/'CPU_READY.json').read_text());c=json.loads((b/'root-command.json').read_text());review=json.loads((b/'independent-READY.json').read_text())
 if (review.get('pass') is not True or review.get('source_identity_sha256')!=s['source_identity_sha256']
     or review.get('CPU_READY_sha256')!=sha(b/'CPU_READY.json') or review.get('command_sha256')!=sha(b/'root-command.json')):raise ValueError('Fresh independent source/AIR/oracle approval required')
 for r in s['sources']:
  if sha(b/'source'/r['path'])!=r['sha256']:raise ValueError('Program source drift:'+r['path'])
 for r in s['reused53']+s['artifacts']:
  if sha(b/r['path'])!=r['sha256']:raise ValueError('Compiled artifact drift:'+r['path'])
 argv=c['argv'][:]
 if a.real_input:argv.append(str(a.real_input.resolve()))
 if Path(argv[3]).exists() or Path(argv[3]+'.writing').exists():raise ValueError('Fresh component report required')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_')}
 return subprocess.run(argv,cwd=b,env=env,check=False).returncode
if __name__=='__main__':raise SystemExit(main())
