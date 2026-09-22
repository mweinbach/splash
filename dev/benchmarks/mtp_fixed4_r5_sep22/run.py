#!/usr/bin/env python3
"""Root executes only a fully authenticated canonical fixed4/R5 command."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from dev.benchmarks.mtp_fixed4_r5_sep22 import policy_quality as p,prepare

def validate(c):
    if c.get('schema')!='Root-fixed4-R5-canonical-command-v1' or c.get('cwd')!=str(p.ROOT):raise ValueError('Unknown command/cwd')
    if c.get('base_command')!=str(prepare.BASE) or p.sha(prepare.BASE)!=c['base_command_sha256']:raise ValueError('Canonical base changed')
    for path,wanted in c['pins'].items():
        if not p.is_sha(wanted) or p.sha(path)!=wanted:raise ValueError('Preregistered program/runtime/receipt/plan pin changed: '+path)
    b=p.authenticate(c['binding'],c['binding_sha256']);base=json.loads(prepare.BASE.read_text())
    driver=Path(c['argv'][2]);text,journal=prepare.transform(prepare.DRIVER.read_text())
    if driver.read_text()!=text:raise ValueError('Driver source inverse differs')
    argv,env=prepare.derive(base,driver.parent,b,Path(c['binding']),c['binding_sha256'],Path(c['report']))
    if c['argv']!=argv or c['environment']!=env:raise ValueError('Canonical controls/unrelated flags changed')
    if '--run-root-gpu' not in argv:raise ValueError('Root inference authorization missing')
    for option,wanted in [('contexts','2048'),('output-tokens','256'),('batches','1'),('warmup','1'),('trials','3'),('max-context','16384')]:
        if argv[argv.index('--'+option)+1]!=wanted:raise ValueError('Canonical benchmark control differs: '+option)
    if c['pins'].get(str(p.ROOT/'build/release/flash/prefill4k-semantic-plan-v1.json'))!=p.PLAN_FILE_SHA:raise ValueError('Exact original plan pin required')
    report=Path(c['report'])
    for path in (report,report.with_name(report.stem+'-4.semantic.json'),report.with_name(report.stem+'-4.server.log')):
        if path.exists():raise FileExistsError('Fresh report/semantic/log required')
    p.load(c['binding'],c['binding_sha256']);return argv

def main():
    if len(sys.argv)!=3 or not p.is_sha(sys.argv[2]) or p.sha(sys.argv[1])!=sys.argv[2]:raise ValueError('Externally registered command SHA differs')
    c=json.loads(Path(sys.argv[1]).read_text());argv=validate(c)
    return subprocess.run(argv,cwd=c['cwd']).returncode

if __name__=='__main__':raise SystemExit(main())
