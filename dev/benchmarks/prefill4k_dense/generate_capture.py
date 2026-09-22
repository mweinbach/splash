#!/usr/bin/env python3
"""CPU-only private source overlay generation; normal source is untouched."""
import hashlib
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
original = Path("runtime/flash/FlashForward.cpp").read_text()
signature = """  void project(metal::CommandGraph &graph, const std::string &prefix,
               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
               const metal::MetalBuffer &diagnostics, uint32_t rows) {"""
if original.count(signature) != 1:
    raise SystemExit("private capture project signature drifted")
modified = '#include "capture.hpp"\n' + original.replace(signature,signature + """
    prefill4k_dense::CaptureScope privateCapture(backend,graph,prefix,input,output,
        denseCache && denseCache->contains(prefix) ? &denseCache->tensor(prefix) : nullptr,rows);""")
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(modified)
(destination.parent / "manifest.json").write_text(json.dumps({
    "schema":"splash-private-dense-capture-overlay-v1",
    "gpu_executed":False,
    "source_sha256":hashlib.sha256(original.encode()).hexdigest(),
    "overlay_sha256":hashlib.sha256(modified.encode()).hexdigest(),
    "captures":"first exact BF16 input and output for10 measured projection roles",
},indent=2)+"\n")
