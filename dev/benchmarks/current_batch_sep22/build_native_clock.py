#!/usr/bin/env python3
"""CPU-build/seal optional native clocks over authenticated parent objects."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/current-batch-native-clock-sep22-v4")
    args = parser.parse_args()
    output = args.build.resolve()
    source_plan = json.loads((output / "source-plan.json").read_text())
    parent = Path(source_plan["parent"])
    parent_seal = json.loads((parent / "compiled-cpu-seal.json").read_text())
    if not parent_seal.get("pass") or parent_seal.get("effective_objects") != 54:
        raise ValueError("authenticated qualified parent object closure required")
    artifacts = parent_seal["artifact_sha256"]
    objects = []
    imported = []
    for relative, expected in artifacts.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe parent artifact path")
        if relative == "host/FlashWorker.o" or not relative.endswith(".o"):
            continue
        source = parent / relative
        if sha(source) != expected:
            raise ValueError("authenticated parent object drift: " + relative)
        destination = output / "objects" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        objects.append(destination)
        imported.append({"relative": relative, "parent_sha256": expected,
            "snapshot_sha256": sha(destination)})
    if len(objects) != 53:
        raise ValueError("expected exactly 53 unchanged objects beside Worker")
    for relative, expected in parent_seal["source_sha256"].items():
        if sha(parent / "source" / relative) != expected:
            raise ValueError("authenticated parent source drift: " + relative)
    library = output / "splash.metallib"
    if sha(parent / "splash.metallib") != artifacts["splash.metallib"]:
        raise ValueError("authenticated parent metallib drift")
    shutil.copy2(parent / "splash.metallib", library)
    includes = [f"-I{output / 'source'}", f"-I{parent / 'source'}",
        f"-I{parent / 'source/runtime'}", f"-I{parent / 'source/dev/benchmarks/prefill4k_attention'}"]
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror",
        "-Wno-deprecated-declarations", "-fobjc-arc", "-mmacosx-version-min=27.0",
        "-DSPLASH_INT8_EXPERIMENT=1"]
    command = ["xcrun", "-sdk", "macosx", "clang++", *includes, *flags]
    worker = output / "source/runtime/flash/FlashWorker.mm"
    original_worker = parent / "source/runtime/flash/FlashWorker.mm"
    from prepare_native_clock import transform
    if worker.read_text() != transform(original_worker.read_text()):
        raise ValueError("private Worker differs beyond authenticated metadata insertions")
    obj = output / "FlashWorker.o"
    compile_command = [*command, "-MMD", "-MP", "-c", str(worker), "-o", str(obj)]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    link_command = [*command, str(obj), *map(str, objects), "-framework", "Foundation",
        "-framework", "Metal", "-framework", "IOKit", "-o", str(output / "splash-flash")]
    subprocess.run(link_command, cwd=ROOT, check=True)
    cpu = subprocess.run([str(output / "splash-flash"), "--cpu-self-test"],
        capture_output=True, text=True, check=True)
    test = output / "trace-cpu"
    subprocess.run([*command, str(ROOT / "dev/benchmarks/current_batch_sep22/trace_cpu.cpp"),
        "-o", str(test)], cwd=ROOT, check=True)
    trace = subprocess.run([str(test), str(output / "cpu-trace.jsonl")],
        capture_output=True, text=True, check=True)
    seal = {"schema": "splash-native-lifecycle-clock-private-cpu-seal-sep22-v1",
        "pass": True, "metadata_only": True, "GPU_qualification_complete": False,
        "gpu_executed": False, "model_payload_bytes_read": 0,
        "parent": str(parent), "parent_compiled_cpu_seal_sha256": sha(parent / "compiled-cpu-seal.json"),
        "parent_all_source_hashes_verified": len(parent_seal["source_sha256"]),
        "changed_parent_sources": ["runtime/flash/FlashWorker.mm"],
        "new_header": "dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp",
        "worker_source_sha256": sha(worker), "worker_object_sha256": sha(obj),
        "native_trace_header_sha256": sha(output / "source/dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp"),
        "binary_sha256": sha(output / "splash-flash"), "metallib_sha256": sha(library),
        "parent_metallib_identical": sha(library) == artifacts["splash.metallib"],
        "imported_unchanged_objects": imported, "effective_object_count": 54,
        "kernel_changes": [], "protocol_header_changes": [], "numerical_route_changes": [],
        "compiler_flags": flags, "compile_command": compile_command, "link_command": link_command,
        "worker_cpu_suite": json.loads(cpu.stdout), "native_clock_cpu_policy": json.loads(trace.stdout)}
    (output / "compiled-cpu-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps({key: value for key, value in seal.items() if key not in
        ("imported_unchanged_objects", "compile_command", "link_command", "worker_cpu_suite")}))


if __name__ == "__main__":
    main()
