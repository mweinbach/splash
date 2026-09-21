"""CPU manifest-only audit of native original-weight residency candidates.

No Metal/MLX imports, payload reads, inference, leases, or model mutations.
The native backend wraps one entire manifest shard; supplying a tensor view
would request residency of that complete base, never just the view's range.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ALIGNMENT = 16384
TYPE_BYTES = {"U32": 4, "BF16": 2, "I64": 8, "F32": 4}
KNOWN_SOURCE = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
KNOWN_LAYOUT = "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402"
KNOWN_FINGERPRINT = "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"


def category(name):
    if name.startswith("vision_tower.") or "vision" in name.lower():
        return "vision"
    if ".ple." in name or "ngram" in name.lower():
        return "ple"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("language_model."):
        return "text"
    return "unknown"


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is not a nonnegative integer")
    return value


def audit(package):
    package = Path(package).resolve()
    raw = (package / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    layout = hashlib.sha256(raw).hexdigest()
    if (
        manifest.get("schema") != "splash-local-qwen4-affine-v1"
        or manifest.get("alignment") != ALIGNMENT
        or manifest.get("source_identity_sha256") != KNOWN_SOURCE
        or layout != KNOWN_LAYOUT
        or (package / "manifest.sha256").read_bytes() != (layout + "\n").encode()
    ):
        raise ValueError("audit input is not the qualified original aligned bundle")
    identity = (
        "splash.native-flash-weights-v1\nsource=" + KNOWN_SOURCE
        + "\nmanifest=" + layout + "\nnorm=one-plus-weight\n"
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    if fingerprint != KNOWN_FINGERPRINT:
        raise ValueError("effective source/layout fingerprint is unqualified")

    shards = {record["path"]: record for record in manifest["shards"]}
    if len(shards) != len(manifest["shards"]) or len(shards) != 21:
        raise ValueError("original shard inventory is not the qualified 21-base layout")
    grouped = defaultdict(list)
    for name, tensor in manifest["tensors"].items():
        if tensor["shard"] not in shards:
            raise ValueError("tensor refers to an unknown base")
        offset = integer(tensor["offset"], "tensor offset")
        length = integer(tensor["length"], "tensor length")
        count = 1
        for dimension in tensor["shape"]:
            dimension = integer(dimension, "tensor dimension")
            if not dimension:
                raise ValueError("zero tensor dimension")
            count *= dimension
        if tensor["dtype"] not in TYPE_BYTES or count * TYPE_BYTES[tensor["dtype"]] != length:
            raise ValueError("tensor logical byte extent does not match dtype/shape")
        if not length or offset % ALIGNMENT or offset + length > shards[tensor["shard"]]["bytes"]:
            raise ValueError("tensor alignment or base range is invalid")
        grouped[tensor["shard"]].append((offset, offset + length, name, tensor, category(name)))

    records = []
    total_category_bytes = Counter()
    total_category_count = Counter()
    overlaps = 0
    for filename, shard in sorted(shards.items()):
        length = integer(shard["bytes"], "base bytes")
        if not length or length % ALIGNMENT or (package / filename).stat().st_size != length:
            raise ValueError("base file size/alignment does not match manifest")
        tensors = sorted(grouped[filename])
        previous_end = 0
        by_bytes, by_count = Counter(), Counter()
        for offset, end, name, tensor, kind in tensors:
            if offset < previous_end:
                overlaps += 1
                raise ValueError(f"logical tensor ranges overlap in {filename}: {name}")
            previous_end = end
            by_bytes[kind] += tensor["length"]
            by_count[kind] += 1
        total_category_bytes.update(by_bytes)
        total_category_count.update(by_count)
        reasons = [kind for kind in ("ple", "vision", "unknown") if by_count[kind]]
        records.append({
            "path": filename,
            "native_base_index": list(record["path"] for record in manifest["shards"]).index(filename),
            "source_path": shard["source_path"],
            "source_sha256": shard["source_sha256"],
            "derived_payload_sha256": shard["sha256"],
            "mapped_base_bytes": length,
            "mapped_base_gib": length / 1024**3,
            "tensor_count": len(tensors),
            "category_tensor_count": dict(by_count),
            "category_logical_bytes": dict(by_bytes),
            "padding_bytes": length - sum(by_bytes.values()),
            "eligible": not reasons,
            "exclusion_reasons": reasons,
            "first_text_tensor": next((name for _, _, name, _, kind in tensors if kind in ("text", "mtp")), None),
            "last_text_tensor": next((name for _, _, name, _, kind in reversed(tensors) if kind in ("text", "mtp")), None),
        })
    selected = [record for record in records if record["eligible"]]
    excluded = [record for record in records if not record["eligible"]]
    selected_bytes = sum(record["mapped_base_bytes"] for record in selected)
    excluded_text = sum(record["category_logical_bytes"].get(kind, 0)
                        for record in excluded for kind in ("text", "mtp"))
    result = {
        "schema": "splash-original-residency-manifest-audit-v1",
        "scope": "CPU manifest and file-size metadata only; no payload reads or Metal device",
        "source_identity_sha256": KNOWN_SOURCE,
        "aligned_manifest_sha256": layout,
        "loaded_model_layout_sha256": fingerprint,
        "actual_native_allocated_size_measured": False,
        "size_semantics": "mapped file/base length; actual MTLBuffer.allocatedSize needs runtime observation",
        "payload_hashes_reverified": False,
        "logical_tensor_overlap_count": overlaps,
        "original_base_count": len(records),
        "original_mapped_bytes": sum(record["mapped_base_bytes"] for record in records),
        "category_logical_bytes": dict(total_category_bytes),
        "category_tensor_count": dict(total_category_count),
        "eligible_base_count": len(selected),
        "eligible_mapped_bytes": selected_bytes,
        "eligible_mapped_gib": selected_bytes / 1024**3,
        "eligible_paths": [record["path"] for record in selected],
        "selected_original_tensor_count": sum(record["tensor_count"] for record in selected),
        "excluded_base_count": len(excluded),
        "excluded_text_or_mtp_logical_bytes": excluded_text,
        "excluded_text_or_mtp_logical_gib": excluded_text / 1024**3,
        "policy": "exclude every entire original base containing any PLE, vision or unknown tensor",
        "proposed_flag": "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT=1 (off by default, review only)",
        "existing_lease_count_constraint": "combine original and derived buffers into the one startup lease; backend rejects a second lease",
        "second_weight_ledger_charge": False,
        "performance_effect": "unmeasured hypothesis; safe-point profile must compare scheduled/callback and GPU times",
        "bases": records,
    }
    lookup = sum(tensor["length"] for name, tensor in manifest["tensors"].items()
                 if ".ngram_embedding.shards." in name)
    result["original_ple_lookup_logical_bytes"] = lookup
    result["original_ple_lookup_logical_gib"] = lookup / 1024**3
    # Existing sidecars are metadata inputs only. This nominal union includes
    # 48 already-allocated rank maps; the backend will expose actual driver
    # allocatedSize once the single combined registration is requested.
    dense = json.loads((package.parent / "Flash-Next-operands-v1" / "manifest.json").read_bytes())
    experts = json.loads((package.parent / "Flash-Next-int8-experts-top64-v1" / "manifest.json").read_bytes())
    derived = sum(entry["allocated_bytes"] for entry in dense["entries"]) + experts["total_bytes"] + 48 * ALIGNMENT
    result["existing_derived_nominal_bytes"] = derived
    result["combined_lease_nominal_bytes"] = derived + selected_bytes
    result["combined_lease_nominal_gib"] = (derived + selected_bytes) / 1024**3
    result["source_evidence"] = {
        "wrap_one_native_base_per_manifest_shard": "runtime/flash/FlashWeights.mm:324",
        "tensor_views_share_native_allocation": "runtime/flash/FlashWeights.mm:368",
        "existing_original_base_getter": "runtime/flash/FlashWeights.mm:605",
        "driver_allocated_size_ledger": "runtime/metal/MetalBackend.mm:923",
        "single_registration_constraint": "runtime/metal/MetalBackend.mm:1853",
        "native_buffer_dedup_identity": "runtime/metal/MetalBackend.mm:1881",
        "whole_base_byte_sum": "runtime/metal/MetalBackend.mm:1886",
        "no_second_backing_charge": "runtime/metal/MetalBackend.mm:1924",
        "current_derived_only_getter": "runtime/flash/FlashForward.cpp:565",
        "current_single_startup_union": "runtime/flash/FlashWorker.mm:2871",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.package)
    arguments.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "eligible_base_count", "eligible_mapped_bytes", "eligible_mapped_gib",
        "excluded_text_or_mtp_logical_gib", "logical_tensor_overlap_count",
        "actual_native_allocated_size_measured",
    )}))


if __name__ == "__main__":
    main()
