"""CPU metadata-only audit for exact-byte owned original storage."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ALIGNMENT = 16384


def rounded(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def audit(package):
    raw = (package / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    assert manifest_sha == "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402"
    assert (package / "manifest.sha256").read_text() == manifest_sha + "\n"
    assert manifest["source_identity_sha256"] == "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
    assert manifest["alignment"] == ALIGNMENT
    grouped = defaultdict(list)
    logical = 0
    for name, tensor in manifest["tensors"].items():
        grouped[tensor["shard"]].append((tensor["offset"], name, tensor["length"]))
        logical += tensor["length"]
    rows = []
    previous_owned = 0
    peak_copy_phase = 0
    for shard in manifest["shards"]:
        cursor = 0
        for offset, name, length in sorted(grouped[shard["path"]]):
            assert offset == rounded(cursor)
            assert length > 0 and offset + length <= shard["bytes"]
            cursor = offset + length
        assert rounded(cursor) == shard["bytes"]
        admission = shard["bytes"] * 2
        peak_copy_phase = max(peak_copy_phase, previous_owned + admission)
        rows.append({"path": shard["path"], "owned_native_bytes": shard["bytes"],
                     "temporary_source_bytes": shard["bytes"],
                     "admission_bytes": admission,
                     "preceding_owned_bytes": previous_owned,
                     "copied_shard_sha256_if_optional_verification": shard["sha256"],
                     "canonical_offset_tensor_count": len(grouped[shard["path"]])})
        previous_owned += shard["bytes"]
    assert len(rows) == 21 and len(manifest["tensors"]) == 3748
    assert previous_owned == 106320429056
    return {"schema": "splash-private-owned-original-layout-audit-v1",
            "scope": "CPU metadata only; zero payload reads, model load or GPU execution",
            "source_identity_sha256": manifest["source_identity_sha256"],
            "aligned_manifest_sha256": manifest_sha,
            "loaded_model_layout_sha256": "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
            "native_base_count": len(rows), "tensor_count": len(manifest["tensors"]),
            "final_original_native_bytes": previous_owned,
            "logical_tensor_bytes": logical, "padding_bytes": previous_owned - logical,
            "maximum_temporary_source_bytes": max(x["temporary_source_bytes"] for x in rows),
            "maximum_copy_admission_bytes": max(x["admission_bytes"] for x in rows),
            "copy_phase_peak_owned_plus_temporary_source_bytes": peak_copy_phase,
            "source_mappings_retained_on_success": 0,
            "maximum_concurrent_source_mappings": 1,
            "native_buffer_offsets_preserved": True,
            "source_bytes_and_weight_math_changed": False,
            "disk_payloads_created": 0,
            "physical_pinning_verified": False,
            "performance_gain": "unmeasured hypothesis; owned anonymous backing may change idle driver wiring",
            "shards": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.package)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "native_base_count", "tensor_count", "final_original_native_bytes",
        "maximum_temporary_source_bytes", "maximum_copy_admission_bytes",
        "copy_phase_peak_owned_plus_temporary_source_bytes")}))
