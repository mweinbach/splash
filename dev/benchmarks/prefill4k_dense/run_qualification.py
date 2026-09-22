#!/usr/bin/env python3
"""Root-controlled native off/on qualification; dry-run unless --run."""
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
parser.add_argument("--build",type=Path,default=ROOT / "build/prefill4k-dense-runtime")
parser.add_argument("--dense-tiles",type=int,choices=[0,1],required=True)
parser.add_argument("--teacher-cache-only",type=int,choices=[0,1],required=True)
parser.add_argument("--mode",choices=["normal","command","stage","dispatch"],default="normal")
parser.add_argument("--warmup",type=int,default=1)
parser.add_argument("--repeats",type=int,default=3)
parser.add_argument("--run",action="store_true")
args = parser.parse_args()
package = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
environment = {k:v for k,v in os.environ.items() if not k.startswith(("SPLASH_FLASH_","PREFILL4K_"))}
defaults = launcher._local_profile_defaults(package)
overrides = {
    "SPLASH_FLASH_PREFILL_DENSE_TILES":str(args.dense_tiles),
    "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY":str(args.teacher_cache_only),
}
environment.update(defaults)
environment.update(overrides)
controls = {
    "PREFILL4K_ATTRIBUTION_MODE":args.mode,
    "PREFILL4K_ATTRIBUTION_ROWS":"2048",
    "PREFILL4K_ATTRIBUTION_CAPACITY":"8192",
    "PREFILL4K_ATTRIBUTION_WARMUP":str(args.warmup),
    "PREFILL4K_ATTRIBUTION_REPEATS":str(args.repeats),
    "PREFILL4K_ATTRIBUTION_TEACHER_PRIME":"1",
}
environment.update(controls)
binary,library = args.build.resolve() / "prefill4k-attribution",args.build.resolve() / "splash.metallib"
command = [str(binary),str(library),str(package),str(args.tokens.resolve()),str(args.report.resolve())]
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
witness = args.report.with_suffix(args.report.suffix + ".invocation.json")
if witness.exists() or args.report.exists():
    raise RuntimeError("choose fresh qualification report output")
witness.parent.mkdir(parents=True,exist_ok=True)
witness.write_text(json.dumps({
    "schema":"splash-prefill4k-dense-qualification-invocation-v1",
    "gpu_executed":args.run,"command":command,"policy_environment":defaults,
    "overrides":overrides,"controls":controls,"binary_sha256":sha(binary),"metallib_sha256":sha(library),
    "tokens_sha256":sha(args.tokens),"profile_sha256":sha(ROOT / ".splash-local-profile.json"),
    "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT / "runtime/flash/FlashDenseCache.cpp",ROOT / "runtime/flash/FlashPrefillDenseTiles.hpp",
        ROOT / "runtime/metal/kernels/shared/flash_dense_cache_prefill.metal",
        ROOT / "runtime/flash/FlashMTP.cpp",ROOT / "dev/benchmarks/prefill4k_attribution.mm"]},
},indent=2)+"\n")
print(json.dumps({"gpu_executed":args.run,"invocation":str(witness),"command":command}))
if args.run:
    subprocess.run(command,cwd=ROOT,env=environment,check=True)
