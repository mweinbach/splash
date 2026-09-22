#!/usr/bin/env python3
"""CPU-only closure/contract seal; never reads capture tensor paths or runs GPU."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "dev/benchmarks/gdn_tile_actual_capture_sep21"
BUILD = ROOT / "build/gdn-tile-actual-capture-sep21-v1"
COMPONENT_SEAL = ROOT / "build/gdn-tile-chunk-sep21/cpu-seal-v1.json"
MANIFEST = ROOT / "build/prefill4k-gdn-pair/actual-layer0-v1/manifest.json"
LIBRARY_SHA = "1f71721efbd672d98828092997bcfd65aa89ec7d78a4086f9f627526dda9457e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha(path)}


component = json.loads(COMPONENT_SEAL.read_text())
for item in component["sources"]:
    path = ROOT / item["path"]
    assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
library = ROOT / "build/gdn-tile-chunk-sep21/gdn-tile.metallib"
assert sha(library) == LIBRARY_SHA
binary = BUILD / "metal-actual-tile"
assert subprocess.run([str(binary)], check=True, capture_output=True).stdout.startswith(b"Usage:")
contract = subprocess.run([str(binary), "--cpu-contract", str(MANIFEST)],
                          check=True, capture_output=True, text=True)
contract_path = BUILD / "cpu-manifest-contract.jsonl"
contract_path.write_text(contract.stdout)
records = [json.loads(line) for line in contract.stdout.splitlines()]
assert len(records) == 1 and not records[0]["payload_read"] and not records[0]["metal_device_created"]
assert all(not item["content_verified"] for item in records[0]["files"])

# Exhaustive adapter address hulls over the frozen actual capture dimensions.
rows, heads, k, values, tile, time = 2048, 48, 128, 128, 32, 32
chunks = rows // time
state = heads * values * k
tape_elements = chunks * state
checked = 0
for c in range(chunks):
    begin = c * time
    count = min(time, rows - begin)
    for row_bytes in (10240 * 2, heads * 4, heads * 2):
        assert (begin + count) * row_bytes <= rows * row_bytes
        checked += 1
    assert (c + 1) * state <= tape_elements
    for h in range(heads):
        for t in range(values // tile):
            state_begin = (h * values + t * tile) * k
            assert state_begin + tile * k <= state
            assert (c * heads + h) * 4 + t < chunks * heads * 4
            assert ((c + 1) * state if c + 1 < chunks else 0) + state_begin + tile * k <= (
                tape_elements if c + 1 < chunks else state)
            for token in range(count):
                actual = (begin + token) * 6144 + h * values + t * tile
                control = token * 6144 + h * values + t * tile
                assert actual + tile <= rows * 6144 and control + tile <= count * 6144
                checked += 1
            checked += 3
layout = {"pass": True, "checked_address_hulls": checked,
          "rows": rows, "heads": heads, "time": time, "value_tile": tile,
          "probe_tape_bytes": tape_elements * 4,
          "prepared_bytes": 164757504, "range_bytes": 12288,
          "decision_bytes": 49152, "component_planned_arena_physical_bytes": 164823040,
          "metal_device_created": False, "capture_tensor_payload_read": False}
layout_path = BUILD / "cpu-adapter-address-contract.json"
layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")
manifest = json.loads(MANIFEST.read_text())  # Text only; payload paths never opened here.
seal = {"schema": "splash.gdn-tile-actual-capture-private-cpu-seal.v1",
        "sources": [record(p) for p in sorted(SRC.iterdir()) if p.is_file()],
        "included_component_sources": component["sources"],
        "included_component_sources_match_frozen_seal": True,
        "component_cpu_seal": record(COMPONENT_SEAL),
        "artifacts": [record(binary), record(BUILD / "cpu-oracle.o"), record(library)],
        "reports": [record(contract_path), record(layout_path)],
        "manifest_text": record(MANIFEST), "declared_tensor_identities": manifest["files"],
        "capture_payload_opened_by_preparation_agent": False,
        "gpu_executed_by_preparation_agent": False,
        "cpu_manifest_and_adapter_contract_pass": True,
        "full_sequence_bit_identity_claim": False, "whole_history_accuracy_certificate": False,
        "native_branch_reference": "original V16/Time16 from identical actual hybrid incoming chunk state",
        "numerical_gates_unchanged": {"f32_relative_rms": 1e-4,
                                     "f32_max_abs_reference_peak_scale": 5e-4,
                                     "bf16_relative_rms": .003,
                                     "bf16_max_abs_reference_peak_scale": .004},
        "promotion_allowed": False, "whole_model_qualified": False}
(BUILD / "cpu-seal-v1.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
print(json.dumps({"cpu_manifest_and_adapter_contract_pass": True,
                  "included_component_sources_match": True, "binary": record(binary),
                  "payload_read": False, "gpu_executed": False}, sort_keys=True))
