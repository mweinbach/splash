#!/usr/bin/env python3
"""Freeze compiled source and artifacts, never captured/model operands."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--build", default="build/dense-kloop-sync-sep21")
args = p.parse_args()
build = Path(args.build)
sources = set(Path("dev/benchmarks/dense_kloop_sync_sep21").glob("*"))
for dep in build.glob("*.d"):
    words = shlex.split(dep.read_text().replace("\\\n", " "))
    sources.update(Path(w) for w in words[1:] if not w.endswith(":"))
sources.update(Path("runtime/metal/kernels/shared") / (name + ".metal") for name in
               ["flash_dense_cache", "flash_dense_cache_m64", "flash_dense_cache_prefill"])
sources.update(Path("runtime/metal/kernels/common").glob("*.h"))
sources.add(Path("runtime/metal/abi/FlashDenseCache.h"))
def entry(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
document = {
    "schema": "splash-dense-kloop-sourcecompile-freeze-v1",
    "gpu_executed": False,
    "model_payloads_read": False,
    "sdk": subprocess.check_output(["xcrun", "-sdk", "macosx", "--show-sdk-path"], text=True).strip(),
    "sources": [entry(s) for s in sorted(sources) if s.is_file()],
    "artifacts": [entry(a) for a in sorted(build.iterdir()) if a.suffix in {".air", ".metallib", ".o"} or a.name == "oracle"],
}
destination = build / "source-manifest.json"
destination.write_text(json.dumps(document, indent=2) + "\n")
print(json.dumps({"manifest": str(destination), "sources": len(document["sources"]), "artifacts": len(document["artifacts"])}))
