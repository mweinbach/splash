#!/usr/bin/env python3
"""Bounded source/ownership witness for the private all-row Full512 build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from prefill4k_allrows_overlay import ROOT, module
from prefill4k_allrows_store import POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-allrows-full512")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh source witness")
    manifest = json.loads((args.build / "overlay-manifest.json").read_text())
    names = list(manifest["transform_sha256"])
    transforms = [module(name).transform for name in names]
    mismatches = []
    for record in manifest["files"]:
        if record.get("new_private_file"):
            copied = (args.build / "source" / record["path"]).read_bytes()
            extras = module("allrows_qmv").extra_files()
            if manifest.get("gathered_qmv_c2_composed"):
                extras.update(module("allrows_qmv_c2").extra_files())
            expected = extras[record["path"]].encode()
            if copied != expected or hashlib.sha256(copied).hexdigest() != record["overlay_sha256"]:
                mismatches.append(record["path"])
            continue
        original = (ROOT / record["path"]).read_bytes()
        copied = (args.build / "source" / record["path"]).read_bytes()
        expected = original.decode()
        for transform in transforms:
            expected = transform(record["path"], expected)
        if manifest.get("gathered_qmv_c2_composed") and record["path"] == "runtime/flash/FlashWorker.mm":
            expected = expected.replace("      (void)gathered_i8_qmv::requested();", "      (void)gathered_i8_qmv::requestedColumns();\n      (void)gathered_i8_qmv::requested();")
        if (copied != expected.encode() or hashlib.sha256(original).hexdigest() != record["original_sha256"]
                or hashlib.sha256(copied).hexdigest() != record["overlay_sha256"]):
            mismatches.append(record["path"])
    store = (args.build / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    shader = (args.build / "source/runtime/metal/kernels/shared/flash_int8_expert_store.metal").read_text()
    worker = (args.build / "source/runtime/flash/FlashWorker.mm").read_text()
    # Independent partition proof: each nonempty expert owns at least one
    # route. Test concentrated, spread and shuffled arbitrary populations.
    rng = random.Random(4217)
    checked = 0
    for rows in [1, 4, 16, 17, 255, 256, 1024, 4096, 8192]:
        routes = rows * 10
        for trial in range(100):
            counts = [0] * 512
            if trial == 0:
                counts[0] = routes
            elif trial == 1:
                counts = [routes // 512 + int(i < routes % 512) for i in range(512)]
            else:
                remaining = routes
                for _ in range(511):
                    count = rng.randrange(remaining + 1)
                    counts[rng.randrange(512)] += count
                    remaining -= count
                counts[rng.randrange(512)] += remaining
            for tile in [16, 32, 64]:
                active = sum((count + tile - 1) // tile for count in counts)
                declared = (routes + tile - 1) // tile + 511
                assert active <= min(routes, declared)
                checked += 1
    checks = {
        "source_fresh": not mismatches,
        "no_original_tensor_strong_copies": "sourceTensors" not in store and "layer.source[" not in store,
        "no_original_validation_graph": "addMoEBlockedGateUp" not in store and "addMoEBlockedDownScatter" not in store,
        "no_gpu_count_readback": "memcpy(&miss.blocked" not in store and "jobCount.contents()" not in store,
        "preserves_direct_down_padding": 'graph.add("flash_moe_direct_a_prepare_down"' in store,
        "preserves_invalid_route_poison": 'graph.add("flash_moe_blocked_poison_excluded_routes"' in store,
        "tight_active_count_shader_guard": "active > min(p.job_capacity, p.route_capacity)" in shader,
        "missing_rank_full512_sets_diagnostic": "if (p.stored_experts == 512 && !tid) flash_mpp_error(diag, 1u)" in shader,
        "early_preflight_before_backend": worker.index("const auto earlyFull") < worker.index("metal::MetalBackend backend("),
        "trained_mtp_cpp_untouched": next(r for r in manifest["files"] if r["path"] == "runtime/flash/FlashMTP.cpp")["patched"] is False,
        "trained_mtp_header_untouched": next(r for r in manifest["files"] if r["path"] == "runtime/flash/FlashMTP.hpp")["patched"] is False,
    }
    derivative = ("splash.private-allrows-target-v1\nsource=edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
                  "\nstore=ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1"
                  f"\npolicy={POLICY}\nmtp=original-trained-bank\n")
    qmv_derivative = derivative + f"small_row_policy={module('allrows_qmv').POLICY}\n" if manifest.get("gathered_qmv_composed") else None
    c2_derivative = derivative + f"small_row_policy={module('allrows_qmv_c2').POLICY}\n" if manifest.get("gathered_qmv_c2_composed") else None
    result = {"schema": 1, "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0,
              "files_checked": len(manifest["files"]), "source_mismatches": mismatches,
              "checks": checks, "active_job_partition_proof_cases": checked,
              "expected_target_numerical_derivative_sha256": hashlib.sha256(derivative.encode()).hexdigest(),
              "expected_gathered_qmv_target_numerical_derivative_sha256": hashlib.sha256(qmv_derivative.encode()).hexdigest() if qmv_derivative else None,
              "expected_gathered_qmv_c2_target_numerical_derivative_sha256": hashlib.sha256(c2_derivative.encode()).hexdigest() if c2_derivative else None,
              "binary_sha256": hashlib.sha256((args.build / "splash-flash").read_bytes()).hexdigest(),
              "metallib_sha256": hashlib.sha256((args.build / "splash.metallib").read_bytes()).hexdigest(),
              "pass": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "files_checked": result["files_checked"], "active_job_partition_proof_cases": checked, "gpu_work": False}))
    if not result["pass"]:
        raise ValueError("Private all-row ownership/source witness failed")


if __name__ == "__main__":
    main()
