#!/usr/bin/env python3
"""Correct mutable BF16 counters outside immutable identity; source/CPU only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
WORKER = "runtime/flash/FlashWorker.mm"
CALLS = '      << R"(,"target_decode_bf16_dense_graph_calls":)" << forward_.decodeBF16DenseEncodedCalls()\n'
ROWS = '      << R"(,"target_decode_bf16_dense_graph_rows":)" << forward_.decodeBF16DenseEncodedRows()\n'
SCOPE = '      << R"(,"target_decode_bf16_dense_counter_scope":"graph construction for known main attention/shared/PLE roles at physical rows1,2,4,8,16 and HCup1/2 only; HCdown, HCup4plus, trained MTP and vocabulary excluded")"\n'
COUNTERS = '''  out << R"(,"decode_bf16_dense_route_counters":{"scope":"graph construction for known main attention/shared/PLE roles at physical rows1,2,4,8,16 and HCup1/2 only; HCdown, HCup4plus, trained MTP and vocabulary excluded","graph_calls":)"
      << forward_.decodeBF16DenseEncodedCalls()
      << R"(,"graph_rows":)" << forward_.decodeBF16DenseEncodedRows() << '}';
'''

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def identity(text: str) -> str:
    begin = text.index('      << R"(,"identity":{"source":)"')
    end = text.index('      << R"(,"weight_format":"mlx_affine_preconverted"})"', begin)
    return text[begin:end]
def assert_immutable_identity(text: str) -> None:
    section = identity(text)
    for token in ["decodeBF16DenseEncodedCalls", "decodeBF16DenseEncodedRows", "graph_calls", "graph_rows",
                  "graph_counters", "Commands()", "Completed", "teacherCachePrimeCalls_", "mtpCommittedHeadCalls_"]:
        if token in section: raise AssertionError(f"Mutable counter in immutable identity: {token}")

def transform(text: str) -> str:
    for line in [CALLS, ROWS, SCOPE]:
        if text.count(line) != 1: raise ValueError("BF16 v2 mutable status anchor drift")
        text = text.replace(line, "")
    anchor = "  const auto requestTrace = requestCommandTraceInfo();"
    if text.count(anchor) != 1: raise ValueError("Top-level counter insertion anchor drift")
    text = text.replace(anchor, COUNTERS + anchor)
    assert_immutable_identity(text)
    return text

def generate(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if output.exists() or ROOT / "build" not in output.parents or output == base:
        raise ValueError("choose fresh separate private build path")
    parent_bytes = (base / "overlay-manifest.json").read_bytes(); parent = json.loads(parent_bytes)
    if not parent.get("bf16_decode_composed"): raise ValueError("requires BF16 v2 source parent")
    records = []
    for entry in parent["files"]:
        relative = entry["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts: raise ValueError("unsafe source path")
        original = (base / "source" / relative).read_bytes()
        if sha(original) != entry["overlay_sha256"]: raise ValueError(f"sealed parent drift: {relative}")
        data = transform(original.decode()).encode() if relative == WORKER else original
        destination = output / "source" / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append({**entry, "status_v3_parent_sha256": sha(original), "status_v3_changed": data != original,
                        "overlay_sha256": sha(data)})
    changed = [entry["path"] for entry in records if entry["status_v3_changed"]]
    if changed != [WORKER]: raise AssertionError("v3 changed non-Worker source")
    objects = []
    for path in sorted((base / "host").glob("*.o")):
        if path.name == "FlashWorker.o": continue
        destination = output / "host" / path.name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        objects.append({"name": path.name, "sha256": sha(path.read_bytes())})
    shutil.copy2(base / "splash.metallib", output / "splash.metallib")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-bulk-gathered-bf16-hcup12-immutable-status-sep21-v3", "status_v3_composed": True,
                     "status_v3_base_build": str(base), "status_v3_base_manifest_sha256": sha(parent_bytes),
                     "status_v3_transform_sha256": sha(Path(__file__).read_bytes()),
                     "status_v3_only_changed_source": WORKER, "status_v3_non_worker_objects_reused": objects,
                     "status_v3_metallib_sha256": sha((base / "splash.metallib").read_bytes()),
                     "immutable_identity_cpu_source_guard_pass": True, "gpu_executed": False, "payload_bytes_read": 0,
                     "files": records})
    (output / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"prepared": str(output), "only_changed_source": WORKER, "reused_non_worker_objects": len(objects),
            "kernels_and_numerical_policy_unchanged": True, "immutable_identity_source_guard_pass": True,
            "gpu_work": False, "payload_bytes_read": 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-bf16-sep21-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-bf16-sep21-v3")
    args = parser.parse_args(); print(json.dumps(generate(args.base, args.output)))
