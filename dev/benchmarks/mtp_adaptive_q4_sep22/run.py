#!/usr/bin/env python3
"""Root-only execution after exact private source/control pin checks."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(command, prepare):
    if command.get('schema') != 'qualified-Q4-existing-adaptive-depth-command-v1': raise ValueError('Unknown adaptive command')
    if command.get('policy') not in ('fixed3','adaptive'): raise ValueError('Unknown adaptive policy')
    if command.get('worker') != str(prepare.WORKER) or command.get('cwd') != str(prepare.ROOT): raise ValueError('Runtime/cwd differs')
    base_path=Path(command['base_command'])
    if base_path != prepare.BASE or sha(base_path) != command['base_command_sha256']: raise ValueError('Canonical parent command differs')
    for path, expected in command['pins'].items():
        if sha(path) != expected: raise ValueError('Source/runtime/helper/receipt pin drift: '+path)
    base=json.loads(base_path.read_text())
    driver=Path(command['argv'][2]); text,journal=prepare.transform(prepare.DRIVER.read_text())
    if driver.read_text() != text: raise ValueError('Adaptive driver differs from literal source edits')
    expected_argv, expected_env=prepare.derive(base,driver.parent,command['policy'],Path(command['report']))
    if command['argv'] != expected_argv or command['environment'] != expected_env: raise ValueError('Adaptive command changes canonical controls or unrelated flags')
    if '--run-root-gpu' not in expected_argv: raise ValueError('Explicit Root inference authorization required')
    for suffix in ('','.semantic.json','.server.log'):
        report=Path(command['report'])
        path=report if not suffix else report.with_name(report.stem+'-'+('adaptive' if command['policy']=='adaptive' else '3')+suffix)
        if path.exists(): raise FileExistsError('Fresh adaptive report/semantic/log required: '+str(path))
    return expected_argv

def main():
    if len(sys.argv) != 3 or sha(sys.argv[1]) != sys.argv[2]: raise ValueError('Externally preregistered command SHA differs')
    command=json.loads(Path(sys.argv[1]).read_text())
    path=Path(__file__).with_name('prepare.py')
    spec=importlib.util.spec_from_file_location('_adaptive_prepare',path);prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    argv=validate(command,prepare)
    from dev.benchmarks.mtp_adaptive_q4_sep22.policy_quality import load
    load(command['worker'],command['policy'])
    return subprocess.run(argv,cwd=command['cwd']).returncode

if __name__=='__main__': raise SystemExit(main())
