#!/usr/bin/env python3
"""Root standard benchmark gate: current qualified numerical/state receipt required."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('flag',choices=['0','1']);a=p.parse_args();b=a.build.resolve()
 receipt_path=b/'current2k-candidate-qualified-receipt.json'
 if not receipt_path.exists():raise SystemExit('Root current2K capture/F64/perroute/state qualifier receipt is required before standard timing')
 q=json.loads(receipt_path.read_text())
 if not q.get('pass') or not q.get('qualified_for_standard_benchmark') or not q.get('Root_GPU_executed'):raise SystemExit('actual qualified candidate receipt required')
 for name,k in [('compiled-cpu-seal.json','shipping_cpu_seal_sha256'),('splash-flash','shipping_worker_sha256'),('splash.metallib','shipping_metallib_sha256')]:
  if sha(b/name)!=q[k]:raise SystemExit('qualified shipping artifact changed')
 if sha(q['report'])!=q['report_sha256']:raise SystemExit('actual numeric/state report changed')
 for name,digest in q['code_pins'].items():
  if sha(name)!=digest:raise SystemExit('qualified diagnostic proof input changed: '+name)
 command=json.loads((b/'root-matched-command-seal.json').read_text())
 if command['binary_sha256']!=q['shipping_worker_sha256'] or command['metallib_sha256']!=q['shipping_metallib_sha256']:raise SystemExit('matched command uses another qualified runtime')
 subprocess.run(command['root_commands'][a.flag],cwd=ROOT,check=True)
if __name__=='__main__':main()
