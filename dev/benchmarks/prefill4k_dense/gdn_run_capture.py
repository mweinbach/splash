#!/usr/bin/env python3
"""Root-exclusive actual GDN capture; dry-run unless --run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from install import launcher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tokens",type=Path,required=True)
parser.add_argument("--report",type=Path,required=True)
parser.add_argument("--outdir",type=Path,required=True)
parser.add_argument("--binary",type=Path,default=ROOT/"build/prefill4k-gdn-capture/capture")
parser.add_argument("--library",type=Path,default=ROOT/"build/prefill4k-dense-runtime/splash.metallib")
parser.add_argument("--run",action="store_true")
args=parser.parse_args()
if args.outdir.exists() or args.report.exists():raise RuntimeError("choose fresh GDN capture paths")
package=ROOT/"install/local-models/Flash-Next-oQ4e-mtp-v1"
environment={k:v for k,v in os.environ.items() if not k.startswith(("SPLASH_FLASH_","PREFILL4K_"))}
profile=launcher._local_profile_defaults(package);environment.update(profile)
controls={"PREFILL4K_ATTRIBUTION_MODE":"normal","PREFILL4K_ATTRIBUTION_ROWS":"2048",
    "PREFILL4K_ATTRIBUTION_CAPACITY":"8192","PREFILL4K_ATTRIBUTION_WARMUP":"0",
    "PREFILL4K_ATTRIBUTION_REPEATS":"1","PREFILL4K_ATTRIBUTION_TEACHER_PRIME":"1",
    "PREFILL4K_GDN_CAPTURE":str(args.outdir.resolve())}
environment.update(controls)
command=[str(args.binary.resolve()),str(args.library.resolve()),str(package),
         str(args.tokens.resolve()),str(args.report.resolve())]
witness=args.report.with_suffix(args.report.suffix+".invocation.json")
if witness.exists():raise RuntimeError("choose fresh GDN invocation")
witness.parent.mkdir(parents=True,exist_ok=True)
witness.write_text(json.dumps({"schema":"splash-actual-gdn-capture-invocation-v1","gpu_executed":args.run,
    "command":command,"profile_environment":profile,"controls":controls},indent=2)+"\n")
print(json.dumps({"gpu_executed":args.run,"command":command,"invocation":str(witness)}))
if args.run:subprocess.run(command,cwd=ROOT,env=environment,check=True)
