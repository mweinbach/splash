#!/usr/bin/env python3
"""Dry-run by default; root exclusively coordinates --run GPU capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from install import launcher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tokens",type=Path,required=True)
parser.add_argument("--report",type=Path,required=True)
parser.add_argument("--outdir",type=Path,required=True)
parser.add_argument("--binary",type=Path,default=ROOT / "build/prefill4k-dense/capture/prefill4k-dense-capture")
parser.add_argument("--library",type=Path,default=ROOT / "build/prefill4k-attribution/splash.metallib")
parser.add_argument("--run",action="store_true")
args = parser.parse_args()
if args.outdir.exists():
    raise RuntimeError("choose a fresh capture output directory")
package = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
environment = {k:v for k,v in os.environ.items() if not k.startswith(("SPLASH_FLASH_","PREFILL4K_"))}
defaults = launcher._local_profile_defaults(package)
environment.update(defaults)
controls = {
    "PREFILL4K_ATTRIBUTION_MODE":"normal",
    "PREFILL4K_ATTRIBUTION_ROWS":"2048",
    "PREFILL4K_ATTRIBUTION_CAPACITY":"8192",
    "PREFILL4K_ATTRIBUTION_WARMUP":"0",
    "PREFILL4K_ATTRIBUTION_REPEATS":"1",
    "PREFILL4K_ATTRIBUTION_TEACHER_PRIME":"0",
    "PREFILL4K_DENSE_CAPTURE":str(args.outdir.resolve()),
}
environment.update(controls)
command = [str(args.binary.resolve()),str(args.library.resolve()),str(package),
           str(args.tokens.resolve()),str(args.report.resolve())]
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
witness = args.report.with_suffix(args.report.suffix + ".invocation.json")
if witness.exists():
    raise RuntimeError("choose fresh capture output/report")
witness.parent.mkdir(parents=True,exist_ok=True)
witness.write_text(json.dumps({
    "schema":"splash-prefill4k-dense-capture-invocation-v1",
    "gpu_executed":args.run,"command":command,"policy_environment":defaults,
    "controls":controls,"binary_sha256":sha(args.binary),"metallib_sha256":sha(args.library),
    "tokens_sha256":sha(args.tokens),
    "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT / "dev/benchmarks/prefill4k_dense/capture.hpp",
        ROOT / "dev/benchmarks/prefill4k_dense/capture_main.mm",
        ROOT / "build/prefill4k-dense/capture/source/FlashForward.cpp"]},
},indent=2)+"\n")
print(json.dumps({"gpu_executed":args.run,"invocation":str(witness),"command":command}))
if args.run:
    subprocess.run(command,cwd=ROOT,env=environment,check=True)
