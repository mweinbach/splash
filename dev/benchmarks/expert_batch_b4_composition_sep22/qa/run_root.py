#!/usr/bin/env python3
"""Root-exclusive serial oracle runner. Preparers never invoke this GPU entry."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=3:raise ValueError('metadata path and Root pin required')
 p=Path(sys.argv[1]);c=json.loads(p.read_text());b=Path(c['build']);m=json.loads((b/'CPU_READY.json').read_text())
 if sha(p)!=sys.argv[2]or not c['Root_GPU_only']or not m['pass']or m['GPU_executed']:raise ValueError('Root-pinned current CPU closure required')
 for r in m['sources']:
  if sha(b/'source'/r['path'])!=r['sha256']:raise ValueError('frozen source drift:'+r['path'])
 for r in m['objects']:
  if sha(b/r['path'])!=r['sha256']:raise ValueError('current53-object drift')
 if sha(Path(c['argv'][0]))!=c['oracle_sha256']or sha(Path(c['argv'][3]))!=c['metallib_sha256']:raise ValueError('Root-pinned artifacts changed')
 if c['role']not in ['export','compare']or c['argv'][1:3]!=['--gpu',c['role']]:raise ValueError('role argv mismatch')
 env={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};env.update(c['environment'])
 if env['SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22']!=str(int(c['role']=='compare'))or env['SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22']!='1'or env['SPLASH_FLASH_MTP_DRAFT_DEPTH']!='3':raise ValueError('role/narrowedBQSA4 mismatch')
 if c['role']=='compare':
  prior=json.loads(Path(c['control_report']).read_text())
  if not prior['pass']or not prior['backend_destroyed']or not prior['partition_complete']or prior['role']!='export':raise ValueError('complete separate-process export required')
  if prior['common']['selected_checkpoints']!=c['argv'][8].split(','):raise ValueError('fullhistory replay selected partition mismatch')
 r=subprocess.run(c['argv'],cwd=c['cwd'],env=env);return r.returncode if r.returncode>=0 else 128-r.returncode
if __name__=='__main__':raise SystemExit(main())
