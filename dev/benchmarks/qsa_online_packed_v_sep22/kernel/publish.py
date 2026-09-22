#!/usr/bin/env python3
"""Seal quiescent actual CPU proof; no shader compiler or device invocation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "build/qsa-online-packed-v-sep22-kernel-v1"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (BUILD / "CPU_READY.json").exists(): raise ValueError("Sealed kernel is immutable")
    interim = json.loads((BUILD / "CPU_INTERIM.json").read_text())
    compiled = json.loads((BUILD / "compile-receipt.json").read_text())
    # Verify every compiled source snapshot before adding final proof scripts.
    for row in compiled["source"]:
        snapshot = BUILD / row["path"]
        if sha(snapshot) != row["sha256"]: raise ValueError("Compiled source snapshot changed")
        if row["program_path"].startswith("dev/benchmarks/") and sha(ROOT / row["program_path"]) != row["sha256"]:
            raise ValueError("Program source differs from compiled bytes")
    for row in compiled["units"]:
        for suffix, key in (("air", "AIR_sha256"), ("ll", "IR_sha256")):
            if sha(BUILD / (row["unit"] + "." + suffix)) != row[key]: raise ValueError("Compiled arithmetic artifact changed")
    snapshots = BUILD / "source/dev/benchmarks/qsa_online_packed_v_sep22/kernel"
    for source in HERE.iterdir():
        if source.is_file(): shutil.copy2(source, snapshots / source.name)
    result = subprocess.run([sys.executable, str(snapshots / "audit.py"), "--build", str(BUILD)], capture_output=True, text=True)
    if result.returncode: raise ValueError(result.stdout + result.stderr)
    audit = json.loads((BUILD / "arithmetic-audit.json").read_text())
    required = ("candidate_native_FP_tree_match", "taps_shipping_FP_tree_match", "V_bit_permutation_proved")
    if audit.get("pass") is not True or audit["failures"] or not all(audit[key] is True for key in required):
        raise ValueError("Actual CPU proof does not admit this closure")
    sources = []
    for source in sorted(snapshots.iterdir()):
        if source.is_file(): sources.append({"path": str(source.relative_to(BUILD)), "program_path": str((HERE / source.name).relative_to(ROOT)), "sha256": sha(source)})
    sources.extend(row for row in compiled["source"] if row["program_path"].startswith("runtime/"))
    artifact_paths = [Path(unit + "." + suffix) for unit in ("original", "native", "candidate", "pack", "native_reduce_tap", "candidate_reduce_tap") for suffix in ("air", "ll")]
    artifact_paths += [Path(name) for name in ("compile-receipt.json", "independent-source-review.json", "V-permutation-proof.json", "arithmetic-audit.json")]
    artifacts = [{"path": str(path), "sha256": sha(BUILD / path)} for path in artifact_paths]
    ready = {"schema": "online-packed-V-QSA-kernel-CPU-closure-v1", "pass": True,
             "GPU_work": False, "model_input_capture_tensor_generation_payload_reads": False,
             "original_AIR_sha256": sha(BUILD / "original.air"),
             **{key: audit[key] for key in required}, "sources": sources, "artifacts": artifacts,
             "standalone_AIRs": interim["standalone_AIRs"],
             "source_review_sha256": interim["source_review_sha256"],
             "separate_TUs_preserve_original_FP_pragma_context": True,
             "source_quiescent": True, "shader_or_IR_recompile_during_publication": False,
             "original_AIR_is_only_performance_control": True,
             "private_native_aliases_are_untimed_authentication": True,
             "whole_worker_integration": False, "GPU_bit_parity_proved": False,
             "performance_proved": False, "opaque_MPP_order_universal_proof": False,
             "allowed_actual_graph_deltas": ["Two certified integer global V GEP maps",
                 "Two untimed live quotient/rounded attention output sinks with exact builtin ABI mapping",
                 "Only the two new disjoint-output alias-scope memberships; all original scopes/attributes retained"],
             "mandatory_Root_GPU_gate": "Full V bit transpose, active F32 stats/numerators/raw quotient/BF boundaries/diagnostics/cache/future/canaries before pack-inclusive4 vs original3 timing"}
    temporary = BUILD / "CPU_READY.tmp"
    temporary.write_text(json.dumps(ready, indent=2) + "\n")
    temporary.replace(BUILD / "CPU_READY.json")
    print(json.dumps({"pass": True, "GPU_work": False, "CPU_READY": str(BUILD / "CPU_READY.json"),
                      "sha256": sha(BUILD / "CPU_READY.json"), "sources": len(sources), "artifacts": len(artifacts),
                      "AIR_count": len(ready["standalone_AIRs"])}))

if __name__ == "__main__": main()
