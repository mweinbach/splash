#!/usr/bin/env python3
"""Root-owned strict registered composite-bundle execution only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 if len(sys.argv)!=3:raise SystemExit('sealed command and metadata digest required')
 path=Path(sys.argv[1]);c=json.loads(path.read_text());m=json.loads((path.parent/'manifest.json').read_text())
 if c.get('runner_source_sha256')!=sha(Path(__file__)):raise SystemExit('registered composite runner source drift')
 if sha(path)!=sys.argv[2] or c['role']!='compare' or not c['registered_composite'] or m.get('variant')!='composite-bundle':raise SystemExit('registered composite role mismatch')
 if c['argv'][1:3]!=['--gpu','compare']:raise SystemExit('compare only required')
 env=c['environment'];flags=tuple(env[n] for n in ['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22','SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22'])
 if flags!=('1','1','1','1'):raise SystemExit('strict COMPACT1/HC1/BUNDLE1/REGISTRATION1 required')
 if sha(c['argv'][0])!=c['oracle_sha256'] or sha(c['argv'][3])!=c['metallib_sha256']:raise SystemExit('artifact drift')
 for h in m['headers']:
  if sha(h['path'])!=h['sha256']:raise SystemExit('oracle header drift')
 original=json.loads(Path(c['original_control_command']).read_text());expected=dict(original['environment']);expected['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22']='1';expected['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']='1';expected['SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22']='1';expected['SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22']='1'
 if expected!=env or any(c['argv'][i]!=original['argv'][i] for i in [1,2,4,5,7]):raise SystemExit('control/input policy changed')
 report=json.loads(Path(c['control_report']).read_text())
 if not report['pass'] or report['role']!='export' or not report['backend_destroyed']:raise SystemExit('completed separate-process control required')
 report_path=Path(c['argv'][6])
 for suffix in ['', '.partial', '.failure.json', '.writing']:
  if Path(str(report_path)+suffix).exists():raise SystemExit('fresh composite report path required before native launch')
 child_env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_FLASH_')};child_env.update(env)
 r=subprocess.run(c['argv'],cwd=c['cwd'],env=child_env)
 return r.returncode if r.returncode>=0 else 128-r.returncode


if __name__=='__main__':raise SystemExit(main())
