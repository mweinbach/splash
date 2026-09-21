#!/usr/bin/env python3
"""Private metadata-only inventory for trained MTP original-F32 proposals.

Opens JSON manifests and checksum sidecars only. Does not import MLX, open tensor
payloads, create a Metal device, build caches, or execute any proposed GPU job.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def metadata(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"metadata exceeds inventory bound: {path}")
    data = path.read_bytes()
    checksum = path.with_name("manifest.sha256").read_text()
    digest = hashlib.sha256(data).hexdigest()
    if checksum != digest + "\n":
        raise ValueError(f"manifest checksum mismatch: {path}")
    return json.loads(data), digest


def trained_prefixes() -> list[str]:
    result = ["mtp.fc_embedding", "mtp.fc_hidden"]
    for role in ("mtp.layers.0.attn_hyper_connection",
                 "mtp.layers.0.mlp_hyper_connection", "mtp.hyper_connection_mixer"):
        result.extend(role + suffix for suffix in
                      (".input_mix_weight_down", ".input_mix_weight_up"))
    result.extend("mtp.layers.0.self_attn." + role for role in
                  ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"))
    result.extend("mtp.layers.0.mlp.shared_expert." + role for role in
                  ("gate_proj", "up_proj", "down_proj"))
    return result


def inventory(package: Path, store: Path) -> dict:
    source, source_digest = metadata(package / "manifest.json")
    saved, saved_digest = metadata(store / "manifest.json")
    if source["source_identity_sha256"] != saved["source_identity_sha256"]:
        raise ValueError("saved/source model identity mismatch")
    if saved["schema"] != "splash-local-affine-operands-v1":
        raise ValueError("unexpected saved operand schema")
    tensors, quantization = source["tensors"], source["quantization"]
    saved_index = {(entry["projection"], entry["format"]): entry
                   for entry in saved["entries"]}
    records = []
    for prefix in trained_prefixes():
        quant = quantization.get(prefix, {})
        bits = quant.get("bits", quantization["bits"])
        group = quant.get("group_size", quantization["group_size"])
        mode = quant.get("mode", quantization.get("mode", "affine"))
        weight = tensors[prefix + ".weight"]
        scale, bias = tensors[prefix + ".scales"], tensors[prefix + ".biases"]
        n, k = weight["shape"][0], scale["shape"][1] * group
        assert mode == "affine" and bits in (4, 5, 6, 8)
        assert weight["dtype"] == "U32" and scale["dtype"] == bias["dtype"] == "BF16"
        assert weight["shape"] == [n, k * bits // 32]
        assert scale["shape"] == bias["shape"] == [n, k // group]
        assert weight["length"] == n * k * bits // 8
        assert scale["length"] == bias["length"] == n * (k // group) * 2
        f32_bytes, bf16_bytes = n * k * 4, n * k * 2
        coverage = {}
        for fmt, size in (("F32", f32_bytes), ("BF16", bf16_bytes)):
            entry = saved_index.get((prefix, fmt))
            if entry:
                assert entry["shape"] == [n, k] and entry["logical_bytes"] == size
            coverage[fmt] = bool(entry)
        records.append({"prefix": prefix, "N": n, "K": k, "bits": bits,
                        "group_size": group, "packed_bytes": weight["length"],
                        "scale_bytes": scale["length"], "bias_bytes": bias["length"],
                        "f32_bytes": f32_bytes, "bf16_bytes": bf16_bytes,
                        "saved_formats_present": coverage,
                        "bypassed_when_hc_fusion_selected": "input_mix_weight" in prefix})

    representative = ("mtp.fc_embedding", "mtp.layers.0.self_attn.q_proj",
                      "mtp.layers.0.self_attn.o_proj",
                      "mtp.layers.0.mlp.shared_expert.down_proj")
    by_prefix = {record["prefix"]: record for record in records}
    plans = []
    for prefix in representative:
        record = by_prefix[prefix]
        plans.append({"prefix": prefix, "N": record["N"], "K": record["K"],
                      "bits": record["bits"], "group_size": record["group_size"],
                      "physical_rows": [1, 4, 8], "tiles": [0, 2],
                      "f32_operand_bytes": record["f32_bytes"],
                      "requires_fresh_f32_source_reconstruction":
                      not record["saved_formats_present"]["F32"]})

    f32_total = sum(record["f32_bytes"] for record in records)
    bf16_total = sum(record["bf16_bytes"] for record in records)
    return {"metadata_only": True, "gpu_commands": 0, "tensor_payloads_opened": 0,
            "model_tensors_loaded": 0, "source_manifest": str(package / "manifest.json"),
            "source_manifest_sha256": source_digest,
            "operand_store_manifest": str(store / "manifest.json"),
            "operand_store_manifest_sha256": saved_digest,
            "saved_weights_manifest_fingerprint": saved["weights_manifest_fingerprint"],
            "saved_entry_counts": dict(Counter(entry["format"] for entry in saved["entries"])),
            "saved_mtp_entries": sum(entry["projection"].startswith("mtp.")
                                     for entry in saved["entries"]),
            "trained_dense_matrices": records,
            "total_f32_operand_bytes": f32_total,
            "total_bf16_operand_bytes": bf16_total,
            "all16_f32_cache_planned_bytes": f32_total + 16384,
            "private_primitive_plan": plans, "primitive_variants": len(plans) * 3 * 2,
            "benchmark_environment": {"SPLASH_FLASH_QMV_F32": "1",
                                      "FLASH_FLOAT_CACHE_ROWS": "1,4,8",
                                      "FLASH_FLOAT_CACHE_TILES": "0,2",
                                      "FLASH_FLOAT_CACHE_REPEATS": "6",
                                      "FLASH_FLOAT_CACHE_CONTINUE_ON_ACCURACY_FAILURE": "1"},
            "validation_contract": [
                "Preserve BF16 input words and independently verify original F32 coefficient bits.",
                "Compare against actual raw projection pipelines and include padding dispatch cost.",
                "Use strict relative L2 below 1e-4; retain failures and BF16 ULP/absolute errors.",
                "Any double boundary exception requires a separate trained-MTP declaration.",
                "Primitive qualification does not establish committed-fold or joint-draft parity."],
            "integration_limits": {
                "existing_bf16_cache_minimum_logical_rows": 16,
                "existing_f32_selective_policy_accepts_mtp": False,
                "fc_hidden_logical_to_physical_rows": {"1": 4, "4": 16, "8": 32},
                "existing_f32_small_primitive_maximum_physical_rows": 16,
                "hc_fusion_may_bypass_six_projection_cache_entries": True},
            "payload_hashes_verified": False, "numerically_qualified": False}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=root / "install/local-models/Flash-Next-oQ4e-mtp-v1")
    parser.add_argument("--operand-store", type=Path,
                        default=root / "install/local-models/Flash-Next-operands-v1")
    args = parser.parse_args()
    print(json.dumps(inventory(args.package.resolve(), args.operand_store.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
