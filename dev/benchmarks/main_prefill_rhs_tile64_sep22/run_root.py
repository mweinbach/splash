#!/usr/bin/env python3
"""Root-only source-reviewed invocation; no data or device use during import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command',type=Path,required=True);parser.add_argument('--command-sha256',required=True)
    parser.add_argument('--independent-review',type=Path,required=True);parser.add_argument('--independent-review-sha256',required=True)
    parser.add_argument('--run-root-gpu',action='store_true');args=parser.parse_args()
    require(args.run_root_gpu,'Root-exclusive GPU action required')
    require(sha(args.command)==args.command_sha256,'Exact registered component invocation')
    command=json.loads(args.command.read_text())
    require(command.get('schema')=='Root-main-prefill-RHS-tile64-synthetic-ROI-invocation-v1','Only registered synthetic ROI scope')
    for file,digest in command['artifact_pins'].items():require(sha(file)==digest,'Component source/binary/ABI/closure drift:'+file)
    require(sha(args.independent_review)==args.independent_review_sha256,'Independent source+compiled receipt external pin')
    review=json.loads(args.independent_review.read_text())
    require(review.get('pass') is True and review.get('SOURCE_READY_sha256')==command['SOURCE_READY_sha256'] and
            review.get('oracle_sha256')==command['oracle_sha256'] and review.get('metallib_sha256')==command['metallib_sha256'],
            'Actual independent review must bind these exact source and compiled artifacts')
    require(not Path(command['report']).exists(),'Fresh component report')
    environment={key:value for key,value in os.environ.items() if not key.startswith(('SPLASH_FLASH_','FLASH_'))}
    environment.update(command['environment'])
    return subprocess.run(command['argv'],cwd=command['cwd'],env=environment).returncode
if __name__=='__main__':raise SystemExit(main())
