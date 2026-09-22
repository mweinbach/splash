#!/usr/bin/env python3
"""Execute only a registered Root current2K qualifier; code pins before/after."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(registration):
 for name,digest in registration['code_pins'].items():
  if sha(name)!=digest:raise SystemExit('registered qualifier code/artifact changed: '+name)

def main():
 p=argparse.ArgumentParser();p.add_argument('registration',type=Path);p.add_argument('registration_sha256');a=p.parse_args()
 if sha(a.registration)!=a.registration_sha256:raise SystemExit('qualifier registration changed')
 r=json.loads(a.registration.read_text());verify(r)
 if sha(__file__)!=r['runner_sha256']:raise SystemExit('Root qualifier runner changed')
 report=Path(r['report']);receipt=Path(r['receipt'])
 if report.exists() or receipt.exists():raise SystemExit('fresh Root proof report/receipt required')
 if r['role']=='candidate':
  control=json.loads(Path(r['control_report']).read_text())
  if not control['pass'] or control.get('captured_layers')!=48 or not control.get('clean_teardown'):raise SystemExit('successful current control proof/clean teardown required before candidate')
 env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_','PREFILL4K_'))};env.update(r['environment'])
 result=subprocess.run(r['argv'],env=env,cwd=r['cwd'])
 verify(r)
 if result.returncode:raise SystemExit(result.returncode)
 proof=json.loads(report.read_text())
 required=['pass','one_forward_per_process','prefill134state_outputs_exact','vector_same_plane_shipping_replay_and_tap_bits_exact',
  'perroute_global_frozen_stage_gates_pass','future_and_same_owner_replay_exact','exceptional_ID_rank_duplicate_nonfinite_safety_pass','ownership_stable','canaries_clean','clean_teardown',
  'backend_created','backend_destroyed_before_publication','normal_Governor_growth_allowed','normal_Governor_host_measurement_valid']
 if any(not proof.get(k) for k in required) or proof.get('captured_layers')!=48 or proof.get('F64_certified_failures')!=0:raise SystemExit('actual current2K qualifier did not satisfy registered unchanged gates')
 if proof.get('denied_reservations')!=0 or proof.get('actual_current_delta_bytes',-1)<0 or proof.get('actual_peak_delta_bytes',-1)<proof.get('actual_current_delta_bytes',0) or proof['actual_peak_delta_bytes']>proof['planned_Governor_bytes']:
  raise SystemExit('actual measured peak/current allocation exceeds normal admission')
 if proof.get('actual_peak_diagnostic_delta_bytes',-1)<0 or proof['actual_peak_diagnostic_delta_bytes']>96<<20 or proof.get('after_target_owner_release_bytes')!=proof.get('mapped_model_baseline_bytes') or proof.get('after_weights_owner_release_bytes')!=0 or proof.get('after_backend_stop_bytes')!=0:
  raise SystemExit('measured diagnostic bound/owner-release-to-model/zero teardown is incomplete')
 if r['role']=='candidate' and (not proof['qualified_for_standard_benchmark'] or proof.get('actual_guard_rejections',0)<9):raise SystemExit('candidate current2K guard proof incomplete')
 qualified={'schema':'TeacherV5-current2K-standard-R1-qualified-receipt-v1','role':r['role'],'pass':True,
  'qualified_for_standard_benchmark':r['role']=='candidate','report':str(report),'report_sha256':sha(report),
  'registration_sha256':sha(a.registration),'shipping_cpu_seal_sha256':r['shipping_cpu_seal_sha256'],
  'shipping_worker_sha256':r['shipping_worker_sha256'],'shipping_metallib_sha256':r['shipping_metallib_sha256'],
  'Root_GPU_executed':True,'MPP_bit_parity_claim':False,'numeric_certificate':'unchanged perroute1e-4/.999999 and sampledRN/RTZ/FTZ/F64; strict-sensitive results separate',
  'control_report':r.get('control_report'),'code_pins':r['code_pins']}
 receipt.write_text(json.dumps(qualified,indent=2)+'\n');print(json.dumps({'qualified_receipt':str(receipt),'sha256':sha(receipt),'Root_GPU_work':True}))
if __name__=='__main__':main()
