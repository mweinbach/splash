#!/usr/bin/env python3
"""Fresh diagnostic-oracle-only relink for measured allocation/owner teardown."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--diagnostic-all-layers',action='store_true');a=p.parse_args();old,build=a.old.resolve(),a.build.resolve()
 if build.exists():raise ValueError('fresh ledger-only diagnostic build required')
 m=json.loads((old/'manifest.json').read_text());shutil.copytree(old,build)
 old_s,new_s=str(old),str(build)
 source=build/'source/dev/benchmarks/gemv_r1_teacher_sep22/rig/oracle.mm';shutil.copyfile(HERE/'oracle.mm',source)
 command=[x.replace(old_s,new_s) for x in m['compiler_command']]
 if a.diagnostic_all_layers:command.insert(command.index('-I'+str(build/'source')),'-DSPLASH_R1_DIAGNOSTIC_ALL_LAYERS=1')
 subprocess.run(command,cwd=ROOT,check=True)
 cpu=json.loads(subprocess.check_output([str(build/'oracle'),'--cpu-only'],text=True))
 for obj in m['objects']:
  copied=Path(obj['path'].replace(old_s,new_s))
  if sha(copied)!=obj['sha256']:raise ValueError('ledger patch maynotmodify inherited/private hostobjects')
 if sha(build/'splash.metallib')!=m['metallib_sha256'] or sha(build/'tap.air')!=sha(old/'tap.air'):raise ValueError('ledger patch maynotmodify taps/library math')
 m['sources']={name.replace(old_s,new_s):digest for name,digest in m['sources'].items()};m['sources'][str(source)]=sha(source)
 m['objects']=[{**x,'path':x['path'].replace(old_s,new_s)} for x in m['objects']]
 m['original_shipping_air_pins']=[{**x,'path':x['path'].replace(old_s,new_s)} for x in m['original_shipping_air_pins']]
 m['oracle_sha256']=sha(build/'oracle');m['compiler_command']=command;m['cpu']=cpu
 if a.diagnostic_all_layers:m['diagnostic_all48_layers']=True;m['qualified_for_standard_benchmark']=False;m['timing_attempted']=False
 m['ledger_only_patch']={'source':str(HERE/'oracle.mm'),'source_sha256':sha(HERE/'oracle.mm'),'old_build':str(old),
  'all53host_objects_identical':True,'tap_and_library_identical':True,'real_peakAllocatedBytes':True,
  'actual_target_state_diagnostic_plan_bounds_enforced':True,'target_release_to_model_weights_release_to_zero_and_backend_destroyed_before_publication':True}
 (build/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'prepared':str(build),'cpu_pass':cpu['pass'],'ledger_only_patch':True,'all53objects_tap_library_unchanged':True,'gpu_work':False,'model_payload_reads':False}))
if __name__=='__main__':main()
