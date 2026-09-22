#!/usr/bin/env python3
"""Root-only explicit execution. Preparers do not invoke this runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 if len(sys.argv)!=3:raise SystemExit('sealed command path and metadata SHA required')
 path=Path(sys.argv[1]);c=json.loads(path.read_text())
 if sha(path)!=sys.argv[2]:raise SystemExit('command metadata changed')
 if c['role'] not in ['export','compare'] or c['argv'][1:3]!=['--gpu',c['role']]:raise SystemExit('role/argv mismatch')
 if sha(Path(c['argv'][0]))!=c['oracle_sha256'] or sha(Path(c['argv'][3]))!=c['metallib_sha256']:raise SystemExit('artifact changed')
 m=json.loads((path.parent/'manifest.json').read_text())
 if m['role']!=('candidate' if c['role']=='compare' else 'control'):raise SystemExit('manifest/role mismatch')
 for h in m['headers']:
  if sha(Path(h['path']))!=h['sha256']:raise SystemExit('private inspection header changed')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_FLASH_')};env.update(c['environment'])
 expected='1' if c['role']=='compare' else '0'
 if any(env[n]!=expected for n in ['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']):raise SystemExit('kernel-role policy mismatch')
 if env['SPLASH_FLASH_ALLROWS_FULL512_TARGET']!='1' or env['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21']!='0':raise SystemExit('pure I8 policy required')
 if c['role']=='compare':
  control=json.loads(Path(c['control_report']).read_text())
  if not control['pass'] or control['role']!='export' or not control['backend_destroyed']:raise SystemExit('completed separate-process control export required')
 result=subprocess.run(c['argv'],cwd=c['cwd'],env=env)
 return result.returncode if result.returncode>=0 else 128-result.returncode


if __name__=='__main__':raise SystemExit(main())
