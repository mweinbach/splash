#!/usr/bin/env python3
"""CPU-only seal. Never opens captured tensor paths or runs GPU modes."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "dev/benchmarks/gdn_actual_capture_sep21"
BUILD = ROOT / "build/gdn-actual-capture-sep21-v1"
V6_SEAL = ROOT / "build/gdn-nax-chunks-sep21-v6/cpu-seal-v6.json"
MANIFEST = ROOT / "build/prefill4k-gdn-pair/actual-layer0-v1/manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha(path)}


v6 = json.loads(V6_SEAL.read_text())
for source in v6["sources"]:
    path = ROOT / source["path"]
    assert path.stat().st_size == source["bytes"] and sha(path) == source["sha256"], path
library = ROOT / "build/gdn-nax-chunks-sep21-v6/gdn-wy.metallib"
assert sha(library) == "c58a0344a1cb2e6a29d4aef1a81c4eff2f01191a4380321e48323658a867b709"
binary = BUILD / "metal-actual-capture"
assert subprocess.run([str(binary)], check=True, capture_output=True).stdout.startswith(b"Usage:")
contract = subprocess.run([str(binary), "--cpu-contract", str(MANIFEST)],
                          check=True, capture_output=True, text=True)
contract_path = BUILD / "cpu-manifest-contract.jsonl"
contract_path.write_text(contract.stdout)
records = [json.loads(line) for line in contract.stdout.splitlines()]
assert len(records) == 1 and records[0]["payload_read"] is False
assert records[0]["metal_device_created"] is False
assert all(item["content_verified"] is False for item in records[0]["files"])
manifest = json.loads(MANIFEST.read_text())  # Only allowed manifest text.
sources = [record(path) for path in sorted(SRC.iterdir()) if path.is_file()]
seal = {
    "schema": "splash.gdn-actual-capture-private-cpu-seal.v1",
    "sources": sources,
    "included_v6_sources": v6["sources"],
    "included_v6_sources_match_frozen_seal": True,
    "v6_cpu_seal": record(V6_SEAL),
    "artifacts": [record(binary), record(BUILD / "cpu-oracle.o"), record(library)],
    "reports": [record(contract_path)],
    "manifest_text": record(MANIFEST),
    "declared_tensor_identities": manifest["files"],
    "capture_payload_opened_by_preparation_agent": False,
    "gpu_executed_by_preparation_agent": False,
    "cpu_manifest_contract_pass": True,
    "variant": "private_gdn_wy_v32_t32_sg8",
    "arena_logical_bytes": v6["cpu_layout"]["arena_logical_bytes_2k_singleton"],
    "arena_planned_physical_bytes": v6["cpu_layout"]["arena_planned_physical_bytes_2k_singleton"],
    "numerical_gates_unchanged": {"f32_relative_rms": 1e-4,
                                 "f32_max_abs_reference_peak_scale": 5e-4,
                                 "bf16_relative_rms": .003,
                                 "bf16_max_abs_reference_peak_scale": .004},
    "promotion_allowed": False,
    "whole_model_qualified": False,
}
(BUILD / "cpu-seal-v1.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
print(json.dumps({"cpu_manifest_contract_pass": True,
                  "included_v6_sources_match": True,
                  "binary": record(binary),
                  "payload_read": False,
                  "gpu_executed": False}, sort_keys=True))
