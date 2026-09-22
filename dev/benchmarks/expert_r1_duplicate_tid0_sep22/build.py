#!/usr/bin/env python3
"""Build a private R1 duplicate thread0 validation component without GPU or model payload access."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIVATE = Path("dev/benchmarks/expert_r1_duplicate_tid0_sep22")
PARENT = ROOT / "build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5"
ORIGINAL_AIR_SHA = "a0cd35e03daf13324d0308c8b4d8cee1d6d932989be6cbdc0429e6ec471d05c2"
FROZEN_QUALITY_SHA = "102f605a8f9ff4b8bd6552e62009ccf4246af4524bcd5589d5176bd898df5f08"
FROZEN_REFERENCE_SHA = "0be02b3274b21ba7b2d90965537c4ae028f15e96c83afa21cc41a7f9cd51a595"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, cwd=ROOT):
    result = subprocess.run(list(map(str, argv)), cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-build", type=Path, required=True)
    args = parser.parse_args()
    out, kernel = args.output.resolve(), args.kernel_build.resolve()
    if out.exists() or ROOT / "build" not in out.parents:
        raise ValueError("A fresh private build directory is required")
    manifest = json.loads((PARENT / "overlay-manifest.json").read_text())
    parent_seal = json.loads((PARENT / "compiled-cpu-seal.json").read_text())
    if parent_seal.get("pass") is not True:
        raise ValueError("The frozen TeacherV5 CPU seal must pass")
    shutil.copytree(PARENT / "source", out / "source")
    for relative, digest in parent_seal["source_sha256"].items():
        if sha(out / "source" / relative) != digest:
            raise ValueError("Frozen parent source differs: " + relative)
    destination = out / "source" / PRIVATE
    shutil.copytree(HERE, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    journal = json.loads((destination / "HOST_TRANSFORM.json").read_text())
    base_oracle = ROOT / journal["base_oracle_path"]
    base_receipt = ROOT / journal["base_CPU_READY_path"]
    if (sha(base_oracle) != journal["base_oracle_sha256"]
            or sha(base_receipt) != journal["base_CPU_READY_sha256"]):
        raise ValueError("The explicit host transformation base differs")
    replay = base_oracle.read_text()
    for category in ("ordered_exact_substitutions", "guard_proof_extensions", "explicit_component_profile_removals"):
        for edit in journal[category]:
            if replay.count(edit["before"]) != edit["exact_occurrences"]:
                raise ValueError("Host restoration journal anchor differs")
            replay = replay.replace(edit["before"], edit["after"])
    if (replay != (destination / "oracle.mm").read_text()
            or sha(destination / "oracle.mm") != journal["derived_oracle_sha256"]):
        raise ValueError("The explicit host restoration journal does not reproduce the harness")
    helper = Path("dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp")
    shutil.copyfile(ROOT / helper, out / "source" / helper)
    old_vector = ROOT / "build/gemv-r1-ab-qsa-sep21-worker-v1"
    quality = old_vector / "source/dev/benchmarks/gemv_decode_r1_worker_sep21/qualified-source/quality.hpp"
    if sha(quality) != FROZEN_QUALITY_SHA:
        raise ValueError("The original frozen per-route/global metric source differs")
    quality_destination = out / "source/dev/benchmarks/gemv_decode_sep21_v1b/quality.hpp"
    quality_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(quality, quality_destination)
    reference = Path("dev/benchmarks/prefill4k_allrows_qmv_reference.hpp")
    frozen_reference = ROOT / "build/gemv-r1-teacher-current2k-control-diagnostic-sep22-v4/source" / reference
    if sha(frozen_reference) != FROZEN_REFERENCE_SHA:
        raise ValueError("The original registered reference header differs")
    shutil.copyfile(frozen_reference, out / "source" / reference)
    objects = []
    for relative, digest in parent_seal["artifact_sha256"].items():
        if not relative.endswith(".o") or Path(relative).stem == "FlashWorker":
            continue
        source = PARENT / relative
        if sha(source) != digest:
            raise ValueError("Frozen host object differs: " + relative)
        target = out / "objects" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        objects.append({"path": str(target.relative_to(out)), "sha256": digest,
                        "parent_path": str(source), "recompiled": False})
    if len(objects) != 53:
        raise ValueError("The component must reuse exactly 53 non-Worker host objects")
    original_records = [(Path(path), digest) for path, digest in manifest["parent_input_seals"].items()
                        if path.endswith("flash_gathered_mpp.air")]
    if len(original_records) != 1 or original_records[0][1] != ORIGINAL_AIR_SHA:
        raise ValueError("The original gathered AIR pin is missing or ambiguous")
    original = original_records[0][0]
    if sha(original) != ORIGINAL_AIR_SHA:
        raise ValueError("Original native gathered AIR differs")
    shutil.copyfile(original, out / "original-gathered.air")
    # This receipt authenticates the single separately compiled candidate/tap
    # pair and its source/IR comparison. Reusing it prevents another math build.
    kernel_seal = json.loads((kernel / "CPU_READY.json").read_text())
    if kernel_seal.get("pass") is not True or kernel_seal.get("GPU_work") is not False:
        raise ValueError("A passing CPU-only kernel receipt is required")
    if kernel_seal.get("original_AIR_sha256") != ORIGINAL_AIR_SHA:
        raise ValueError("The kernel audit must bind the actual original native AIR")
    if (kernel_seal.get("candidate_native_FP_tree_match") is not True
            or kernel_seal.get("taps_shipping_FP_tree_match") is not True):
        raise ValueError("The original native floating-point trees must match before admission")
    if kernel_seal.get("duplicate_predicate_equivalence_proved") is not True:
        raise ValueError("Thread-zero duplicate validation must retain every rank result and diagnostic")
    for record in kernel_seal["sources"]:
        source = kernel / record["path"]
        if sha(source) != record["sha256"]:
            raise ValueError("Kernel source drift: " + record["path"])
        relative = Path(record["program_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen kernel program path")
        if sha(out / "source" / relative) != record["sha256"]:
            raise ValueError("The component kernel source differs from the compiled input: " + str(relative))
    kernel_evidence = []
    for record in kernel_seal["artifacts"]:
        source = kernel / record["path"]
        if sha(source) != record["sha256"]:
            raise ValueError("Kernel artifact drift: " + record["path"])
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen kernel evidence path")
        target = out / "kernel-proof" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        kernel_evidence.append({"path": str(target.relative_to(out)), "sha256": record["sha256"]})
    for name in ("candidate.air", "taps.air"):
        shutil.copyfile(kernel / name, out / name)
    shutil.copyfile(kernel / "CPU_READY.json", out / "kernel-CPU_READY.json")
    link_metal = ["xcrun", "-sdk", "macosx", "metallib", out / "original-gathered.air",
                  out / "candidate.air", out / "taps.air", "-o", out / "component.metallib"]
    run(link_metal)
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror",
             "-Wno-deprecated-declarations", "-ffp-contract=off", "-fno-fast-math",
             "-fobjc-arc", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1",
             "-I" + str(out / "source"), "-I" + str(out / "source/runtime"),
             "-I" + str(destination)]
    cc = ["xcrun", "-sdk", "macosx", "clang++", *flags, "-MMD", "-MP", "-c",
          destination / "oracle.mm", "-o", out / "oracle.o"]
    run(cc)
    link = ["xcrun", "-sdk", "macosx", "clang++", *flags, out / "oracle.o",
            *[out / record["path"] for record in objects], "-framework", "Foundation",
            "-framework", "Metal", "-framework", "IOKit", "-o", out / "oracle"]
    run(link)
    cpu = json.loads(run([out / "oracle", "--cpu-self-test"]).stdout)
    if (cpu.get("valid") is not True or not isinstance(cpu.get("checks"), int)
            or cpu["checks"] <= 0 or cpu.get("GPU_work") is not False
            or cpu.get("payload_reads") is not False):
        raise ValueError("The component self-test must remain CPU-only")
    dependency = (out / "oracle.d").read_text().replace("\\\n", " ")
    dependencies = []
    for token in shlex.split(dependency.splitlines()[0].split(":", 1)[1]):
        path = Path(token).resolve()
        if ROOT in path.parents:
            if out / "source" not in path.parents:
                raise ValueError("Oracle source escaped the frozen closure: " + str(path))
            dependencies.append({"path": str(path.relative_to(out)), "sha256": sha(path)})
    sources = [{"path": str(path.relative_to(out / "source")), "sha256": sha(path)}
               for path in sorted((out / "source").rglob("*")) if path.is_file()]
    artifacts = [{"path": name, "sha256": sha(out / name)} for name in
                 ("oracle", "oracle.o", "original-gathered.air", "candidate.air", "taps.air",
                  "component.metallib", "kernel-CPU_READY.json")]
    artifacts += kernel_evidence
    parts = {"parent_manifest_sha256": sha(PARENT / "overlay-manifest.json"),
             "parent_cpu_seal_sha256": sha(PARENT / "compiled-cpu-seal.json"),
             "original_AIR_sha256": ORIGINAL_AIR_SHA,
             "original_numeric_metric_source_sha256": FROZEN_QUALITY_SHA,
             "original_numeric_reference_source_sha256": sha(frozen_reference),
             "kernel_receipt_sha256": sha(kernel / "CPU_READY.json"),
             "scope": "R1-only-duplicate-validation-thread0-original-MPP-ten-expert-component-v1",
             "oracle_sha256": sha(destination / "oracle.mm"),
             "builder_sha256": sha(destination / "build.py")}
    parts["host_transform_journal_sha256"] = sha(destination / "HOST_TRANSFORM.json")
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema": "R1-duplicate-tid0-bounded-component-CPU-v1", "pass": True,
               "source_identity_sha256": identity, "identity_parts": parts, "parent": str(PARENT),
               "sources": sources, "reused53": objects, "compiler_dependencies": dependencies,
               "artifacts": artifacts, "kernel_build": str(kernel),
               "kernel_receipt_sha256": sha(kernel / "CPU_READY.json"),
               "compiler_command": list(map(str, cc)), "link_command": list(map(str, link)),
               "metallib_command": list(map(str, link_metal)), "CPU_selftest": cpu,
               "GPU_work": False, "model_capture_payload_reads": 0,
               "Root_coefficient_fixture_bytes": 49305600, "planned_Governor_bytes": 128 << 20,
               "whole_model_qualification": False, "Root_GPU_component_proof_pending": True}
    (out / "CPU_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"built": str(out), "identity": identity, "GPU_work": False,
                      "CPU_READY_sha256": sha(out / "CPU_READY.json"), "artifacts": artifacts}))


if __name__ == "__main__":
    main()
