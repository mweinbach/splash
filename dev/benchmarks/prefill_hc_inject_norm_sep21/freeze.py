#!/usr/bin/env python3
"""Rebuild from a sealed source tree. This performs CPU compilation only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path("dev/benchmarks/prefill_hc_inject_norm_sep21")
p = argparse.ArgumentParser()
p.add_argument("--build", default="build/prefill-hc-inject-norm-sep21")
args = p.parse_args()
build = Path(args.build)
source_root = build / "frozen-src"
if source_root.exists():
    raise SystemExit("Refusing to overwrite an existing frozen source tree")
sources = {f for f in HERE.iterdir() if f.is_file() and f.suffix in {".hpp", ".mm", ".metal", ".py", ".md"}}
sources.add(HERE / "Makefile")
for dep in build.glob("*.d"):
    words = shlex.split(dep.read_text().replace("\\\n", " "))
    sources.update(Path(w) for w in words[1:] if not w.endswith(":"))
sources.update({Path("runtime/metal/kernels/shared/flash_hc.metal"),
                Path("runtime/metal/abi/FlashHC.h"), Path("runtime/metal/abi/FlashHCFused.h")})
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
entries = []
sdk_inputs = []
sdk = Path(subprocess.check_output(["xcrun", "-sdk", "macosx", "--show-sdk-path"], text=True).strip())
for source in sorted(sources):
    if not source.is_file():
        continue
    if source.is_absolute() and source.is_relative_to(sdk):
        sdk_inputs.append({"path": str(source), "bytes": source.stat().st_size, "sha256": digest(source)})
        continue
    if source.is_absolute() or ".." in source.parts:
        raise SystemExit(f"Unexpected non-project source: {source}")
    frozen = source_root / source
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, frozen)
    entries.append({"source": str(source), "frozen": str(frozen), "bytes": source.stat().st_size,
                    "sha256": digest(source)})
# New build directory prevents pre-snapshot objects and dependency files entering
# the Root executable. The only include roots refer to this frozen source tree.
compiled = build / "sealed"
command = ["make", "-f", str(source_root / HERE / "Makefile"), f"ROOT={source_root}",
           f"BUILD={compiled}", "-j4", "all", "cpu-self-test"]
result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(build / "sealed-build.log").write_text(result.stdout)
if result.returncode:
    print(result.stdout)
    raise SystemExit(result.returncode)
for dep in compiled.glob("*.d"):
    if any(word.startswith("runtime/") or word.startswith("dev/")
           for word in shlex.split(dep.read_text().replace("\\\n", " "))[1:]):
        raise SystemExit(f"Live project dependency in frozen build: {dep}")
artifacts = [{"path": str(a), "bytes": a.stat().st_size, "sha256": digest(a)}
             for a in sorted(compiled.iterdir()) if a.suffix in {".air", ".metallib", ".o"} or a.name == "oracle"]
cpu = subprocess.check_output([str(compiled / "oracle"), "--cpu-self-test"], text=True)
document = {"schema": "splash-prefill-hc-inject-norm-sourcecompile-seal-v1", "gpu_executed": False,
            "model_payloads_read": False, "public_sources_mutated": False,
            "sdk": str(sdk), "sdk_inputs": sdk_inputs,
            "cpu_self_test": json.loads(cpu), "sources": entries, "artifacts": artifacts,
            "oracle": str(compiled / "oracle"), "metallib": str(compiled / "splash.metallib")}
destination = build / "source-manifest.json"
destination.write_text(json.dumps(document, indent=2) + "\n")
print(json.dumps({"manifest": str(destination), "sources": len(entries), "artifacts": len(artifacts),
                  "gpu_executed": False, "cpu_self_test": document["cpu_self_test"]}))
