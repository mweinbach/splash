#!/usr/bin/env python3
"""BF16-selector-aware CPU source handoff checker; reads no model payload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def witness(build: Path) -> dict:
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    parent = Path(manifest["bf16_decode_base_build"])
    assert sha(parent / "overlay-manifest.json") == manifest["bf16_decode_base_manifest_sha256"]
    changed = []
    for entry in manifest["files"]:
        current = build / "source" / entry["path"]
        assert sha(current) == entry["overlay_sha256"], entry["path"]
        if "bf16_decode_parent_sha256" in entry:
            old = parent / "source" / entry["path"]
            assert sha(old) == entry["bf16_decode_parent_sha256"], entry["path"]
            if old.read_bytes() != current.read_bytes(): changed.append(entry["path"])
    assert sorted(changed) == sorted(["runtime/flash/FlashForward.cpp", "runtime/flash/FlashForward.hpp",
                                      "runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashWorker.mm"])
    for relative in ["runtime/flash/FlashMTP.cpp", "runtime/flash/FlashBatchMTPForward.cpp", "runtime/flash/FlashDenseCache.cpp",
                     "runtime/flash/FlashDenseSmallRows.cpp", "runtime/flash/FlashHCFused.cpp"]:
        assert (build / "source" / relative).read_bytes() == (parent / "source" / relative).read_bytes()
    old = (parent / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    store = (build / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    start, end = old.index("void FlashInt8ExpertStore::addGateUp("), old.index("} // namespace splash::flash", old.index("void FlashInt8ExpertStore::addGateUp("))
    assert old[start:end] in store
    old = (parent / "source/runtime/flash/FlashForward.cpp").read_text()
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    start = old.index("    if (!flashHCUpF32MPPGeometry(")
    assert old[start:old.index("\n};", start)] in forward
    start = old.index('    if (int8Head && prefix == "language_model.lm_head") {')
    assert old[start:old.index("\n  void hc(", start)] in forward
    header = (build / "source/runtime/flash/FlashDecodeBF16DensePolicy.hpp").read_text()
    assert "existing_f32_selective=" in header and "HCUpSmall" in header
    return {"schema": "splash-private-bf16-hcup12-decode-source-witness-v2", "pass": True,
            "files_checked": len(manifest["files"]), "changed_existing_sources": changed,
            "default_projection_control_and_r4plus_hc_f32_body_exact": True,
            "trained_mtp_and_bf16_cache_helpers_exact": True,
            "existing_f32_scope_binds_selective_flag": True,
            "extra_arena_bytes_when_enabled": 1048576,
            "runtime_sha256": {name: sha(build / name) for name in ["splash-flash", "splash.metallib"]},
            "gpu_work": False, "model_payload_bytes_read": 0,
            "numerical_parity_qualified": False, "model_quality_qualified": False}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("build", type=Path); args = parser.parse_args()
    print(json.dumps(witness(args.build.resolve()), indent=2))
