#!/usr/bin/env python3
"""Registered Root control diagnostic only; never grants a benchmark receipt."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('registration',type=Path);p.add_argument('sha256');a=p.parse_args()
 if sha(a.registration)!=a.sha256:raise SystemExit('diagnostic registration changed')
 r=json.loads(a.registration.read_text())
 if r['schema']!='registered-TeacherV5-current2K-standard-R1-control-diagnostic-v1' or r['role']!='control' or not r['diagnostic_only']:raise SystemExit('registered control-only diagnostic required')
 if sha(__file__)!=r['runner_sha256']:raise SystemExit('diagnostic runner changed')
 for path,digest in r['code_pins'].items():
  if sha(path)!=digest:raise SystemExit('diagnostic code/artifact changed: '+path)
 if Path(r['report']).exists():raise SystemExit('fresh diagnostic report required')
 env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_','PREFILL4K_'))};env.update(r['environment'])
 result=subprocess.run(r['argv'],env=env,cwd=r['cwd'])
 for path,digest in r['code_pins'].items():
  if sha(path)!=digest:raise SystemExit('diagnostic code/artifact changed during Root run')
 if result.returncode not in (0,2):raise SystemExit(result.returncode)
 report=json.loads(Path(r['report']).read_text())
 if not report.get('diagnostic_completed') or report.get('captured_actual_layers')!=48 or report.get('qualified_for_standard_benchmark') or report.get('numeric_qualification') or report.get('timing_attempted'):raise SystemExit('all48-layer unqualified untimed diagnostic required')
 print(json.dumps({'diagnostic_report':r['report'],'report_sha256':sha(r['report']),'failed_original_gates':not report['pass'],'Root_GPU_work':True,'benchmark_receipt_written':False}))
if __name__=='__main__':main()
