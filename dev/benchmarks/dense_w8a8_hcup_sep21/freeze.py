#!/usr/bin/env python3
"""Source/artifact hashes only; never reads model/captured payloads."""
import hashlib
import json
from pathlib import Path

source = Path("dev/benchmarks/dense_w8a8_hcup_sep21")
build = Path("build/dense-w8a8-hcup-sep21")
paths = list(source.glob("*")) + [
    Path("dev/benchmarks/dense_w8a8_sep21/quantization.hpp"),
    Path("runtime/metal/kernels/shared/flash_dense_cache.metal"),
    Path("runtime/metal/kernels/shared/flash_hc.metal"),
    Path("runtime/metal/abi/FlashDenseCache.h"),
    Path("runtime/metal/abi/FlashHC.h"),
    Path("runtime/metal/kernels/common/flash_affine_mpp_common.h"),
    Path("runtime/metal/kernels/common/flash_dense_traversal.h"),
    build / "oracle", build / "hcup.metallib", build / "candidate.ll",
]
entries = []
for path in sorted(paths):
    if not path.is_file():
        raise SystemExit(f"missing frozen source/artifact: {path}")
    entries.append({"path":str(path),"bytes":path.stat().st_size,
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
(build / "source-manifest.json").write_text(json.dumps({
    "schema":"splash-hcup-w8a8-source-only-freeze-v1",
    "gpu_executed":False,"model_payloads_read":False,
    "captured_payloads_read":False,"production_overlay":False,
    "model_qualified":False,"files":entries,
},indent=2)+"\n")
print(f"froze {len(entries)} source/artifact files; no model or captured payload reads")
