#!/usr/bin/env python3
"""CPU-only private online QSA packed-V component build; reads program bytes only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIVATE = Path("dev/benchmarks/qsa_online_packed_v_sep22")
PARENT = ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
PARENT_EXE = "663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438"
PARENT_LIB = "7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8"
ORIGINAL_AIR = "7e560b9b3e4c598fa04d3d061ec88b93ca20d705849b550e6390158d97a352d2"
SOURCE_REVIEW = ROOT / "build/release/flash/sep22-online-QSA-packedV-precompile-independent-source-review-v1.json"
SOURCE_REVIEW_SHA = "736bca64f65e9c7e6faddc7c6e0bcc2a7d3a61a74ac6e9ec301b03841a21d87c"
OBJECTS = (
    "host/037-FlashQSA.o", "host/038-FlashQSAFast.o", "host/039-FlashQSAMPP.o",
    "host/042-Prefill4kQSABulk.o", "host/043-Prefill4kQSACoalesced.o",
    "reused/core/047-MetalBackend.o", "reused/core/048-DeviceCapabilities.o",
    "reused/core/050-MemoryGovernor.o",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv):
    result = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
    return result


def safe_relative(value):
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe program-relative path")
    return relative


def copy_authenticated(source, destination, expected):
    if sha(source) != expected:
        raise ValueError("Frozen program input differs: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(destination) != expected:
        raise ValueError("Program input copy differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-build", type=Path, required=True)
    args = parser.parse_args()
    out, kernel = args.output.resolve(), args.kernel_build.resolve()
    if out.exists() or ROOT / "build" not in out.parents:
        raise ValueError("Choose a fresh private build directory")
    if sha(SOURCE_REVIEW) != SOURCE_REVIEW_SHA:
        raise ValueError("The independent precompile source review differs")
    review = json.loads(SOURCE_REVIEW.read_text())
    if (review.get("pass") is not True
            or review.get("GO_for_Root_private_standalone_Metal_CPU_compile_decision") is not True):
        raise ValueError("The independent source review has not admitted this bounded compile")
    manifest_path = PARENT / "overlay-manifest.json"
    parent = json.loads(manifest_path.read_text())
    pins = {item["path"]: item["sha256"] for item in parent["artifacts"]}
    if pins.get("splash-flash") != PARENT_EXE or pins.get("splash.metallib") != PARENT_LIB:
        raise ValueError("The approved current parent identity differs")
    if sha(PARENT / "splash-flash") != PARENT_EXE or sha(PARENT / "splash.metallib") != PARENT_LIB:
        raise ValueError("The actual approved parent program differs")
    shutil.copytree(PARENT / "source", out / "source")
    for record in parent["files"]:
        path = safe_relative(record["path"])
        if sha(out / "source" / path) != record["sha256"]:
            raise ValueError("Frozen parent source differs: " + str(path))
    destination = out / "source" / PRIVATE
    shutil.copytree(HERE, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(manifest_path, out / "parent-overlay-manifest.json")
    shutil.copyfile(SOURCE_REVIEW, out / "independent-precompile-source-review.json")

    object_pins = {r["path"]: r["sha256"] for r in parent["compiled_objects"] + parent["frozen_inputs"]}
    objects = []
    for name in OBJECTS:
        target = out / "objects" / name
        copy_authenticated(PARENT / name, target, object_pins[name])
        objects.append({"path": str(target.relative_to(out)), "sha256": object_pins[name],
                        "parent_path": str(PARENT / name), "recompiled": False})

    # Reconstruct the actual complete current library before adding private names.
    # This proves the shipping arithmetic bytes have not been rebuilt or replaced.
    commands = [a for a in parent["compiler_commands"]
                if a[:4] == ["xcrun", "-sdk", "macosx", "metallib"]
                and a[-2:] == ["-o", str(PARENT / "splash.metallib")]]
    if len(commands) != 1:
        raise ValueError("Missing unique current-parent library link recipe")
    original_airs, air_records = [], []
    for value in commands[0][4:-2]:
        path = Path(value)
        if path.suffix != ".air" or PARENT not in path.parents:
            raise ValueError("Original AIR escaped the current parent")
        target = out / "original-air" / path.relative_to(PARENT)
        digest = sha(path)
        copy_authenticated(path, target, digest)
        original_airs.append(target)
        air_records.append({"path": str(target.relative_to(out)), "sha256": digest,
                            "parent_path": str(path), "recompiled": False})
    matched = [r for r in air_records if r["parent_path"].endswith("/120-prefill4k_bulk_attention_sg8.air")]
    if len(matched) != 1 or matched[0]["sha256"] != ORIGINAL_AIR:
        raise ValueError("The original online attention AIR differs")
    baseline_command = ["xcrun", "-sdk", "macosx", "metallib", *original_airs,
                        "-o", out / "baseline-relinked.metallib"]
    run(baseline_command)
    if sha(out / "baseline-relinked.metallib") != PARENT_LIB:
        raise ValueError("Ordered current-parent AIR reconstruction differs")

    seal = json.loads((kernel / "CPU_READY.json").read_text())
    for field in ("pass", "candidate_native_FP_tree_match", "taps_shipping_FP_tree_match",
                  "V_bit_permutation_proved"):
        if seal.get(field) is not True:
            raise ValueError("The bounded compiled kernel proof is not complete: " + field)
    if seal.get("GPU_work") is not False or seal.get("original_AIR_sha256") != ORIGINAL_AIR:
        raise ValueError("The kernel proof must bind the original AIR and remain CPU-only")
    for record in seal["sources"]:
        if sha(kernel / safe_relative(record["path"])) != record["sha256"]:
            raise ValueError("Kernel source differs")
        if sha(out / "source" / safe_relative(record["program_path"])) != record["sha256"]:
            raise ValueError("The component source differs from the compiled kernel input")
    kernel_artifacts = []
    for record in seal["artifacts"]:
        target = out / "kernel-proof" / safe_relative(record["path"])
        copy_authenticated(kernel / safe_relative(record["path"]), target, record["sha256"])
        kernel_artifacts.append({"path": str(target.relative_to(out)), "sha256": record["sha256"]})
    extra_airs = []
    for value in seal["standalone_AIRs"]:
        relative = safe_relative(value)
        if relative.suffix != ".air" or not (out / "kernel-proof" / relative).is_file():
            raise ValueError("Unregistered standalone private AIR")
        extra_airs.append(out / "kernel-proof" / relative)
    if len(extra_airs) != 5 or len(set(extra_airs)) != 5:
        raise ValueError("Exactly five separate private translation units are required")
    shutil.copyfile(kernel / "CPU_READY.json", out / "kernel-CPU_READY.json")
    metal_link = ["xcrun", "-sdk", "macosx", "metallib", *original_airs, *extra_airs,
                  "-o", out / "component.metallib"]
    run(metal_link)
    if sha(out / "component.metallib") != "9a0207a7961c4d0126114a3eac73a352831a9fd8c7275aafa336e5d353f6f3d1":
        raise ValueError("The HostV3 accounting fix must retain the exact original component math library")

    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror",
             "-Wno-deprecated-declarations", "-ffp-contract=off", "-fno-fast-math",
             "-fobjc-arc", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1",
             "-I" + str(out / "source"), "-I" + str(out / "source/runtime"),
             "-I" + str(out / "source/dev/benchmarks/prefill4k_attention"),
             "-I" + str(destination / "host"), "-I" + str(destination / "kernel")]
    cc = ["xcrun", "-sdk", "macosx", "clang++", *flags, "-MMD", "-MP", "-c",
          destination / "host/oracle.mm", "-o", out / "oracle.o"]
    run(cc)
    link = ["xcrun", "-sdk", "macosx", "clang++", *flags, out / "oracle.o",
            *[out / r["path"] for r in objects], "-framework", "Foundation",
            "-framework", "Metal", "-framework", "IOKit", "-o", out / "oracle"]
    run(link)
    cpu = json.loads(run([out / "oracle", "--cpu-self-test"]).stdout)
    if (cpu.get("valid") is not True or type(cpu.get("checks")) is not int or cpu["checks"] <= 0
            or cpu.get("GPU_work") is not False or cpu.get("payload_reads") is not False):
        raise ValueError("The host self-test must be meaningful and CPU-only")
    dependencies = []
    dependency_text = (out / "oracle.d").read_text().replace("\\\n", " ")
    for token in shlex.split(dependency_text.splitlines()[0].split(":", 1)[1]):
        path = Path(token).resolve()
        if ROOT in path.parents:
            if out / "source" not in path.parents:
                raise ValueError("Host source dependency escaped the sealed source closure")
            dependencies.append({"path": str(path.relative_to(out)), "sha256": sha(path)})
    sources = [{"path": str(p.relative_to(out / "source")), "sha256": sha(p)}
               for p in sorted((out / "source").rglob("*")) if p.is_file()]
    artifacts = [{"path": name, "sha256": sha(out / name)} for name in
                 ("oracle", "oracle.o", "component.metallib", "baseline-relinked.metallib",
                  "kernel-CPU_READY.json", "parent-overlay-manifest.json",
                  "independent-precompile-source-review.json")]
    artifacts += kernel_artifacts
    parts = {"parent_manifest_sha256": sha(manifest_path), "parent_executable_sha256": PARENT_EXE,
             "parent_library_sha256": PARENT_LIB, "original_AIR_sha256": ORIGINAL_AIR,
             "kernel_receipt_sha256": sha(kernel / "CPU_READY.json"),
             "independent_precompile_source_review_sha256": SOURCE_REVIEW_SHA,
             "builder_sha256": sha(destination / "build.py"),
             "oracle_source_sha256": sha(destination / "host/oracle.mm"),
             "scope": "standalone-fresh2K-original-online-SG8-packed-V-v1"}
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"schema": "online-QSA-packed-V-standalone-CPU-v1", "pass": True,
               "source_identity_sha256": identity, "identity_parts": parts, "parent": str(PARENT),
               "sources": sources, "reused_host_objects": objects, "original_ordered_AIRs": air_records,
               "artifacts": artifacts, "compiler_dependencies": dependencies,
               "compiler_command": list(map(str, cc)), "host_link_command": list(map(str, link)),
               "baseline_metallib_command": list(map(str, baseline_command)),
               "component_metallib_command": list(map(str, metal_link)), "CPU_selftest": cpu,
               "GPU_work": False, "model_capture_payload_reads": 0,
               "planned_native_ceiling_bytes": 1 << 30,
               "planned_Host_allowance_bytes": 512 << 20,
               "combined_planned_Governor_admission_bytes": (1 << 30) + (512 << 20),
               "extra_logical_V_bytes": 2097152,
               "worker_integration": False, "whole_model_qualification": False,
               "Root_GPU_component_proof_pending": True}
    (out / "CPU_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"build": str(out), "CPU_READY_sha256": sha(out / "CPU_READY.json"),
                      "identity": identity, "GPU_work": False}))


if __name__ == "__main__":
    main()
