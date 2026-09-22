#!/usr/bin/env python3
"""Copy sealed wide v1 and privately repair its dense shader row guard.

CPU-only source/metadata preparation. No model payloads or GPU execution.
The sealed v1 snapshot and production sources remain untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
SHADER = Path("runtime/metal/kernels/shared/flash_dense_cache_prefill.metal")
OLD_GUARD = "p.rows != 2048"
NEW_GUARD = "(p.rows != 2048 && p.rows != 4096 && p.rows != 8192)"
DESCRIPTOR = (
    "constexpr auto descriptor=matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),\n"
    "      false,true,false,matmul2d_descriptor::mode::multiply);"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if ROOT / "build" not in output.parents or output == base or output.exists():
        raise ValueError("Choose a fresh, separate private build directory")
    parent_bytes = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(parent_bytes)
    if not parent.get("wide_prefix_composed") or not parent.get("qsa_bulk_sg8_available"):
        raise ValueError("The sealed wide v1 Full512 bulk source snapshot is required")
    if parent.get("route") != "private-full512-exact-bulk-first2048-wide-cached-dense-m128-v1":
        raise ValueError("Unexpected sealed wide v1 route")
    source_bytes, records = {}, []
    for record in parent["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative in source_bytes:
            raise ValueError("Unsafe or duplicate source manifest path")
        original = (base / "source" / relative).read_bytes()
        if digest(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed v1 source drift: {relative}")
        source_bytes[relative] = original
        result = dict(record)
        result.update(wide_guard_v2_input_sha256=record["overlay_sha256"],
                      wide_guard_v2_changed=False)
        records.append(result)
    actual_files = {p.relative_to(base / "source")
                    for p in (base / "source").rglob("*") if p.is_file()}
    if actual_files != set(source_bytes):
        raise ValueError("The sealed source tree and manifest file set differ")
    if SHADER in source_bytes:
        raise ValueError("v1 unexpectedly already contains a private dense-prefill shader")
    dense_policy = source_bytes[Path("runtime/flash/FlashPrefillDenseTiles.hpp")].decode()
    if dense_policy.count("if (rows != 2048 && rows != 4096 && rows != 8192) return {};") != 2:
        raise ValueError("Both sealed host row policies must already allow 2K/4K/8K")

    original_shader = (ROOT / SHADER).read_bytes()
    shader_text = original_shader.decode()
    if shader_text.count(OLD_GUARD) != 1 or shader_text.count(DESCRIPTOR) != 1:
        raise ValueError("Production dense-prefill shader guard/descriptor anchor drift")
    private_text = shader_text.replace(OLD_GUARD, NEW_GUARD, 1)
    if private_text.replace(NEW_GUARD, OLD_GUARD, 1) != shader_text:
        raise AssertionError("The shader change is not exactly the row guard")
    private_shader = private_text.encode()
    source_bytes[SHADER] = private_shader
    records.append({"path": str(SHADER), "patched": True, "new_private_file": True,
                    "original_sha256": digest(original_shader),
                    "overlay_sha256": digest(private_shader),
                    "wide_guard_v2_changed": True,
                    "wide_guard_v2_only_change": "row eligibility 2048 -> 2048/4096/8192"})
    eligible = [rows for rows in range(16385)
                if not (rows != 2048 and rows != 4096 and rows != 8192)]
    if eligible != [2048, 4096, 8192]:
        raise AssertionError("The new shader guard eligibility is incorrect")
    build_command = ["make", "-f", "Makefile", "-f", "dev/benchmarks/prefill4k_wide.mk",
                     "-f", "dev/benchmarks/prefill4k_attention/bulk_runtime.mk",
                     "BUILD=build/flash-next", "SPLASH_PRECISION=hybrid",
                     f"PREFILL4K_WIDE_BUILD={output.relative_to(ROOT)}",
                     "PREFILL4K_WIDE_FULLCACHE=1",
                     "PREFILL4K_WIDE_EXTRA_SHADER_NAMES=flash_dense_cache_prefill",
                     "-j", "6", "prefill4k-qsa-bulk-runtime", "prefill4k-wide-cpu"]
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-full512-exact-bulk-first2048-wide-cached-dense-m128-v2",
                     "wide_guard_v2_composed": True, "wide_guard_v2_base_build": str(base),
                     "wide_guard_v2_parent_manifest_sha256": digest(parent_bytes),
                     "wide_guard_v2_generator_sha256": digest(Path(__file__).read_bytes()),
                     "wide_guard_v2_changed_paths": [str(SHADER)],
                     "wide_guard_v2_arithmetic_change": False,
                     "wide_guard_v2_eligible_shader_rows": eligible,
                     "wide_guard_v2_extra_shader_names": ["flash_dense_cache_prefill"],
                     "gpu_executed": False, "payload_bytes_read": 0,
                     "normal_sources_modified": False, "wide_model_parity_qualified": False,
                     "files": records})
    audit = {"schema": "splash-prefill4k-wide-guard-v2-cpu-audit-v1",
             "base_build": str(base), "base_manifest_sha256": digest(parent_bytes),
             "sealed_source_files_verified_and_copied_byte_exact": len(parent["files"]),
             "source_files_total": len(records), "changed_paths": [str(SHADER)],
             "production_shader_sha256": digest(original_shader),
             "private_shader_sha256": digest(private_shader),
             "restoring_only_row_guard_recovers_production_shader_byte_exact": True,
             "shader_2048_guard_result_unchanged": True,
             "shader_row_guard_eligible_rows": eligible,
             "shader_row_guard_cases_checked": 16385,
             "shader_numerical_body_unchanged": True,
             "descriptor": "M128N64 whole-K BF16/BF16 F32 destination strict SG4/SG8",
             "all_v1_bulk_mtp_loader_worker_source_bytes_unchanged": True,
             "bulk_workspace_extra_bytes_unchanged": parent["qsa_bulk_workspace_extra_bytes"],
             "extra_shader_names_required": ["flash_dense_cache_prefill"],
             "build_command": build_command,
             "build_completed": False, "worker_cpu_self_test_valid": None,
             "gpu_executed": False, "payload_bytes_read": 0,
             "normal_sources_modified": False, "wide_model_parity_qualified": False}
    parent_audit_bytes = (base / "wide-prefix-cpu-audit.json").read_bytes()
    policy_test_bytes = (base / "dense-policy-cpu.cpp").read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prefill4k-wide-guard-v2-", dir=output.parent) as tmp:
        staged = Path(tmp) / "output"
        for relative, data in source_bytes.items():
            destination = staged / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staged / "wide-guard-v2-parent-manifest.json").write_bytes(parent_bytes)
        (staged / "wide-prefix-v1-cpu-audit.json").write_bytes(parent_audit_bytes)
        (staged / "dense-policy-cpu.cpp").write_bytes(policy_test_bytes)
        (staged / "wide-guard-v2-cpu-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        staged.rename(output)
    return {"prepared": str(output), **audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-wide-prefix-full512-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-wide-prefix-full512-v2")
    args = parser.parse_args()
    print(json.dumps(generate(args.base, args.output)))


if __name__ == "__main__":
    main()
