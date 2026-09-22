#!/usr/bin/env python3
"""Explicit Root diagnostic execution; original qualification runner untouched."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 if len(sys.argv)!=3:raise SystemExit('sealed diagnostic command and metadata digest required')
 path=Path(sys.argv[1]);c=json.loads(path.read_text())
 if sha(path)!=sys.argv[2] or c['role']!='diagnostic_compare' or not c['diagnostic_run'] or c['qualification_complete']:raise SystemExit('diagnostic metadata/role invalid')
 if c['argv'][1:3]!=['--gpu','compare']:raise SystemExit('diagnostic compare-only argv required')
 m=json.loads((path.parent/'manifest.json').read_text())
 if m['role']!='diagnostic' or not m['diagnostic_only']:raise SystemExit('diagnostic manifest required')
 flags=(c['environment']['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22'],c['environment']['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22'])
 if flags!={'compact_only':('1','0'),'hc_only':('0','1')}[c['diagnostic_feature']]:raise SystemExit('one-feature diagnostic mismatch')
 if sha(c['argv'][0])!=c['oracle_sha256'] or sha(c['argv'][3])!=c['metallib_sha256']:raise SystemExit('diagnostic artifact drift')
 for h in m['headers']:
  if sha(h['path'])!=h['sha256']:raise SystemExit('inspection header drift')
 original=json.loads(Path(c['original_full_candidate_command']).read_text())
 for i in [1,2,4,5,7]:
  if c['argv'][i]!=original['argv'][i]:raise SystemExit('original input/spill/role changed')
 expected=dict(original['environment']);expected['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22'],expected['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22']=flags
 if expected!=c['environment']:raise SystemExit('non-feature control changed')
 control=json.loads(Path(c['control_report']).read_text())
 if not control['pass'] or control['role']!='export' or not control['backend_destroyed']:raise SystemExit('completed original control export required')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_FLASH_')};env.update(c['environment'])
 r=subprocess.run(c['argv'],cwd=c['cwd'],env=env)
 return r.returncode if r.returncode>=0 else 128-r.returncode


if __name__=='__main__':raise SystemExit(main())
