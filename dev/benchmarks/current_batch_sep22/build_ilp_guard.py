#!/usr/bin/env python3
"""CPU-only guard-TU rebuild retaining the exact clock Worker and other inputs."""
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
    parser.add_argument("--build", type=Path, default=ROOT / "build/current-batch-native-clock-sep22-v5")
    args = parser.parse_args()
    output = args.build.resolve()
    plan = json.loads((output / "source-plan.json").read_text())
    clock = Path(plan["clock_parent"])
    parent = Path(plan["qualified_math_parent"])
    prior = json.loads((clock / "compiled-cpu-seal.json").read_text())
    objects, imported = [], []
    replaced = "host/018-FlashGDNBatchILP.o"
    for record in prior["imported_unchanged_objects"]:
        if record["relative"] == replaced:
            continue
        source = clock / "objects" / record["relative"]
        if sha(source) != record["snapshot_sha256"]:
            raise ValueError("clock parent retained object drift: " + record["relative"])
        destination = output / "objects" / record["relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        objects.append(destination)
        imported.append({**record, "snapshot_sha256": sha(destination)})
    if len(objects) != 52:
        raise ValueError("expected 52 unchanged non-Worker/non-ILP objects")
    worker = output / "FlashWorker.o"
    if sha(clock / "FlashWorker.o") != prior["worker_object_sha256"]:
        raise ValueError("clock Worker object drift")
    shutil.copy2(clock / "FlashWorker.o", worker)
    if sha(output / "source/runtime/flash/FlashWorker.mm") != prior["worker_source_sha256"]:
        raise ValueError("clock Worker source changed")
    header = output / "source/dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp"
    if sha(header) != prior["native_trace_header_sha256"]:
        raise ValueError("clock trace header changed")
    library = output / "splash.metallib"
    if sha(clock / "splash.metallib") != prior["metallib_sha256"]:
        raise ValueError("clock metallib drift")
    shutil.copy2(clock / "splash.metallib", library)
    includes = [f"-I{output / 'source'}", f"-I{parent / 'source'}",
        f"-I{parent / 'source/runtime'}", f"-I{parent / 'source/runtime/flash'}",
        f"-I{parent / 'source/dev/benchmarks/prefill4k_attention'}"]
    flags = prior["compiler_flags"]
    command = ["xcrun", "-sdk", "macosx", "clang++", *includes, *flags]
    cpp = output / "source/runtime/flash/FlashGDNBatchILP.cpp"
    from prepare_ilp_guard import transform
    if cpp.read_text() != transform((parent / "source/runtime/flash/FlashGDNBatchILP.cpp").read_text()):
        raise ValueError("GDN correction differs beyond exact discarded-producer selector")
    ilp = output / "FlashGDNBatchILP.o"
    subprocess.run([*command, "-MMD", "-MP", "-c", str(cpp), "-o", str(ilp)], cwd=ROOT, check=True)
    linked = [worker, ilp, *objects]
    frameworks = ["-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit"]
    subprocess.run([*command, *map(str, linked), *frameworks, "-o", str(output / "splash-flash")], cwd=ROOT, check=True)
    probe = output / "guard-cpu"
    subprocess.run([*command, str(output / "machinery/guard_cpu.cpp"), *map(str, linked[1:]),
        *frameworks, "-o", str(probe)], cwd=ROOT, check=True)
    tests = []
    for value in ("0", "1"):
        result = subprocess.run([str(probe), value], capture_output=True, text=True, check=True)
        tests.append(json.loads(result.stdout))
    worker_cpu = subprocess.run([str(output / "splash-flash"), "--cpu-self-test"],
        capture_output=True, text=True, check=True)
    seal = {"schema": "splash-native-clock-gdn-discarded-producer-guard-cpu-seal-v1",
        "pass": True, "gpu_executed": False, "model_payload_bytes_read": 0,
        "clock_parent": str(clock), "qualified_math_parent": str(parent),
        "clock_parent_seal_sha256": sha(clock / "compiled-cpu-seal.json"),
        "rebuilt_TUs": ["runtime/flash/FlashGDNBatchILP.cpp"],
        "retained_clock_Worker_source_and_object_identical": True,
        "worker_source_sha256": prior["worker_source_sha256"],
        "worker_object_sha256": sha(worker), "native_trace_header_sha256": sha(header),
        "guard_source_sha256": sha(cpp), "guard_object_sha256": sha(ilp),
        "binary_sha256": sha(output / "splash-flash"), "metallib_sha256": sha(library),
        "exact_clock_and_math_parent_metallib_retained": True,
        "effective_object_count": len(linked), "unchanged_nonWorker_nonILP_objects": imported,
        "guard_probe_generated_from_actual_source": True,
        "guard_probe_source_sha256": sha(output / "machinery/guard_cpu.cpp"),
        "guard_probe_binary_sha256": sha(probe), "guard_CPU_tests": tests,
        "worker_cpu_suite": json.loads(worker_cpu.stdout),
        "merged_ILP_recurrence_FMA": False,
        "discarded_phase1_selector_matches_exact_existing_staged_helper": True,
        "all_other_producer_guards_and_appended_graph_source_unchanged": True,
        "kernel_changes": [], "header_changes": [], "arithmetic_changes": [],
        "GPU_qualification_complete": False, "compiler_flags": flags}
    (output / "compiled-cpu-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps({key: value for key, value in seal.items() if key not in
        ("unchanged_nonWorker_nonILP_objects", "worker_cpu_suite")}))


if __name__ == "__main__":
    main()
