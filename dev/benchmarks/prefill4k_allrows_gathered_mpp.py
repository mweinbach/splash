#!/usr/bin/env python3
"""Independent private gathered-MPP Store transform, no inference on import.

Compose after original allrows Store transform, NEVER after C1/C2 transforms.
Reuses their source-only view guard generator; existing QMV/C2 files are read
only. Original MPP methods remain intact. No allocations/mappings are added.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FLAG = "SPLASH_FLASH_ALLROWS_GATHERED_MPP"
POLICY = "private-direct-gathered-signed-i8-bf16-mpp-m16n64-dynamicK-multiply-sg4-validRows1-late-row-scale-bf16-swiglu-finite-deviceA-nonfinite-localA-rows1to16-v1"
MARKER = "// Independent private gathered MPP Store overlay v1."
HEADER_RELATIVE = "runtime/flash/FlashGatheredMPP.hpp"
METAL_RELATIVE = "runtime/metal/kernels/shared/flash_gathered_mpp.metal"

def qmv_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("private_qmv_guard_generator", ROOT / "dev/benchmarks/prefill4k_allrows_qmv.py")
    if spec is None or spec.loader is None: raise RuntimeError("missing view guard generator")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def extra_files() -> dict[str, str]:
    root = ROOT / "dev/benchmarks"
    return {HEADER_RELATIVE: (root / "prefill4k_allrows_gathered_mpp.hpp").read_text(),
            METAL_RELATIVE: (root / "prefill4k_allrows_gathered_mpp.metal").read_text(),
            "runtime/flash/prefill4k_gathered_mpp_views.hpp": (root / "prefill4k_allrows_qmv.hpp").read_text().replace("FlashGatheredI8QMVParams", "FlashGatheredMPPViewParams").replace("gathered_i8_qmv", "gathered_mpp_view").replace("kInvalidID", "kInvalidOriginalExpertID")}

def transform(relative: str, text: str) -> str:
    if relative not in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"): return text
    if MARKER in text or "gatheredQMV" in text: raise RuntimeError("gathered MPP requires original untransformed allrows Store")
    module = qmv_module()
    result = module.transform(relative, text)
    for before, after in (("flash_gathered_i8_qmv_gate_up_sg4_c1", "flash_gathered_mpp_gate_up_m16_n64_sg4"),
        ("flash_gathered_i8_qmv_down_sg4_c1", "flash_gathered_mpp_down_m16_n64_sg4"),
        (module.MARKER, MARKER), ("FlashGatheredI8QMV.hpp", "FlashGatheredMPP.hpp"),
        ("FlashGatheredI8QMVParams", "FlashGatheredMPPParams"), ("gathered_i8_qmv", "gathered_mpp"),
        ("addGatheredQMV", "addGatheredMPP"), ("gatheredQMV", "gatheredMPP"),
        ("gatheredQMVViews", "gatheredMPPViews"), ("gathered_qmv", "gathered_mpp"),
        ("qmvGate", "gatheredMPPGate"), ("qmvDown", "gatheredMPPDown"),
        ("private gathered I8 QMV", "private gathered I8 MPP"), ("QMV only physical", "gathered MPP only physical")):
        result = result.replace(before, after)
    return result

def cpu_self_test(source: Path) -> dict:
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        before = (source / relative).read_text(); after = transform(relative, before)
        try: transform(relative, after)
        except RuntimeError: pass
        else: raise AssertionError("duplicate transform accepted")
        for token in ("backend.allocateBuffer(", "backend.wrapSharedMemory(", "std::make_shared<Mapping>("):
            if after.count(token) != before.count(token): raise AssertionError("allocation/mapping introduced")
        if relative.endswith(".mm"):
            start = before.index("void FlashInt8ExpertStore::addGateUp("); end = before.index("} // namespace splash::flash", start)
            if before[start:end] not in after: raise AssertionError("old MPP control changed")
            if "if (gatheredMPP)\n      derivative +=" not in after: raise AssertionError("conditional derivative missing")
            if "flash_gathered_i8_qmv_" in after: raise AssertionError("QMV kernel accidentally routed")
            shipping = extra_files()[METAL_RELATIVE]
            for kernel in ("flash_gathered_mpp_gate_up_m16_n64_sg4", "flash_gathered_mpp_down_m16_n64_sg4"):
                if after.count('graph.add("' + kernel + '",') != 1 or shipping.count('kernel void ' + kernel + '(') != 1:
                    raise AssertionError("Store graph kernel is absent from shipping shader")
            if "flash_gathered_mpp_gate_up_sg4_c1" in after or "flash_gathered_mpp_down_sg4_c1" in after:
                raise AssertionError("namespace rename corrupted kernel names")
    if POLICY not in extra_files()[HEADER_RELATIVE]: raise AssertionError("policy mismatch")
    return {"cpu_source_checks":"passed","gpu_work":False,"old_mpp_unchanged":True,"extra_allocations":0,"exact_gpu_parity":"pending"}

def stage(source: Path, destination: Path) -> dict:
    if destination.exists() or source.resolve() == destination.resolve(): raise ValueError("requires NEW private destination")
    destination.mkdir(parents=True); witness = {}
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        before = (source / relative).read_text(); after = transform(relative, before)
        target = destination / relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(after)
        witness[relative] = {"input_bytes":len(before.encode()),"output_bytes":len(after.encode())}
    for relative, content in extra_files().items():
        target = destination / relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(content)
        witness[relative] = {"bytes":len(content.encode())}
    result = {"policy":POLICY,"flag":FLAG,"gpu_work":False,"hashing_performed":False,"files":witness}
    (destination / "gathered-mpp-manifest.json").write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--destination",type=Path); parser.add_argument("--cpu-self-test",action="store_true"); args=parser.parse_args()
    if args.cpu_self_test: print(json.dumps(cpu_self_test(args.source),indent=2))
    else:
        if args.destination is None: parser.error("--destination required")
        print(json.dumps(stage(args.source,args.destination),indent=2))
