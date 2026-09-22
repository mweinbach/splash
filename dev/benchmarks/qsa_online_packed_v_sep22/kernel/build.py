#!/usr/bin/env python3
"""Bounded source-reviewed CPU shader compile; never executes Metal work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
REVIEW = ROOT / "build/release/flash/sep22-online-QSA-packedV-precompile-independent-source-review-v1.json"
REVIEW_SHA = "736bca64f65e9c7e6faddc7c6e0bcc2a7d3a61a74ac6e9ec301b03841a21d87c"
UNITS = ("native", "candidate", "pack", "native_reduce_tap", "candidate_reduce_tap")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if result.returncode: raise RuntimeError(" ".join(argv) + "\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "build/qsa-online-packed-v-sep22-kernel-v1")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists(): raise ValueError("Choose a fresh standalone kernel build")
    if ROOT / "build" not in out.parents: raise ValueError("Output must be private repository build")
    if sha(REVIEW) != REVIEW_SHA: raise ValueError("Independent source GO changed")
    proposed = json.loads((HERE / "PROPOSED_SOURCE_SEAL.json").read_text())
    for row in proposed["files"]:
        if sha(HERE / row["path"]) != row["sha256"]: raise ValueError("Reviewed source changed: " + row["path"])
    recipe = json.loads((HERE / "original_recipe.json").read_text())
    original = ROOT / recipe["current_parent_AIR"]["path"]
    if sha(original) != recipe["current_parent_AIR"]["sha256"]: raise ValueError("Shipping control AIR changed")
    source = out / "source/dev/benchmarks/qsa_online_packed_v_sep22/kernel"
    source.mkdir(parents=True)
    for q in HERE.iterdir():
        if q.is_file(): shutil.copy2(q, source / q.name)
    sources = []
    for q in sorted(source.iterdir()):
        if q.is_file():
            sources.append({"path": str(q.relative_to(out)), "program_path": str((HERE / q.name).relative_to(ROOT)), "sha256": sha(q)})
    for row in recipe["private_ABI_headers"]:
        old = ROOT / row["path"]
        if sha(old) != row["sha256"]: raise ValueError("Frozen original ABI header changed")
        relative = Path("runtime/metal/abi") / old.name
        q = out / "source" / relative
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, q)
        sources.append({"path": str(q.relative_to(out)), "program_path": str(relative), "sha256": sha(q)})
    shutil.copy2(original, out / "original.air")
    shutil.copy2(REVIEW, out / "independent-source-review.json")
    flags = ["-I" + str(out / "source/runtime"),
             "-I" + str(out / "source/dev/benchmarks/prefill4k_attention"),
             "-std=metal4.1", "-O3", "-Wall", "-Wextra", "-Werror", "-Iruntime",
             "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    def compile_unit(unit):
        argv = ["xcrun", "-sdk", "macosx", "metal", *flags, "-c", str(source / (unit + ".metal")), "-o", str(out / (unit + ".air"))]
        output = run(argv)
        run(["xcrun", "air-opt", "-S", str(out / (unit + ".air")), "-o", str(out / (unit + ".ll"))])
        return {"unit": unit, "argv": argv, "output": output, "AIR_sha256": sha(out / (unit + ".air")), "IR_sha256": sha(out / (unit + ".ll"))}
    with ThreadPoolExecutor(max_workers=5) as pool: compiled = list(pool.map(compile_unit, UNITS))
    run(["xcrun", "air-opt", "-S", str(out / "original.air"), "-o", str(out / "original.ll")])
    permutation = json.loads(run([sys.executable, str(source / "check_transpose.py")]))
    (out / "V-permutation-proof.json").write_text(json.dumps(permutation, indent=2) + "\n")
    compile_receipt = {"schema": "online-packed-V-QSA-CPU-compile-v1", "compiler_work": True, "GPU_work": False,
                       "source_review_sha256": REVIEW_SHA, "original_AIR_sha256": sha(out / "original.air"),
                       "units": compiled, "source": sources,
                       "compiler_version": run(["xcrun", "-sdk", "macosx", "metal", "--version"]),
                       "SDK_version": run(["xcrun", "-sdk", "macosx", "--show-sdk-version"]).strip(),
                       "actual_IR_arithmetic_admission": "pending"}
    (out / "compile-receipt.json").write_text(json.dumps(compile_receipt, indent=2) + "\n")
    interim = {"schema": "online-packed-V-QSA-kernel-CPU-closure-v1", "pass": False, "GPU_work": False,
               "original_AIR_sha256": sha(out / "original.air"), "candidate_native_FP_tree_match": False,
               "taps_shipping_FP_tree_match": False, "V_bit_permutation_proved": permutation["pass"],
               "sources": sources, "standalone_AIRs": [unit + ".air" for unit in UNITS],
               "actual_IR_arithmetic_admission": "pending", "source_review_sha256": REVIEW_SHA}
    (out / "CPU_INTERIM.json").write_text(json.dumps(interim, indent=2) + "\n")
    print(json.dumps({"CPU_compiles_pass": True, "GPU_work": False, "output": str(out),
                      "AIR_count": len(compiled), "V_permutation_proof": permutation["pass"], "arithmetic_admission": "pending"}))

if __name__ == "__main__": main()
