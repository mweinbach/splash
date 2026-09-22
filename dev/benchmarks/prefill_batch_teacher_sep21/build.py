#!/usr/bin/env python3
"""Compile isolated host changes and retain byte-identical sealed parent inputs.

No backend is constructed, model is loaded, or GPU command is submitted.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
FLAGS = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", "-Iruntime",
    "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-batch-teacher-gathered-sep21-v1")
    args = parser.parse_args()
    output = args.build.resolve()
    manifest_path = output / "overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent = Path(manifest["batch_teacher_base_build"])
    if not manifest.get("batch_teacher_composed") or manifest.get("batch_teacher_extra_workspace_bytes") != 0:
        raise ValueError("Sealed teacher cache snapshot required")
    for record in manifest["files"]:
        if sha(output / "source" / record["path"]) != record["overlay_sha256"]:
            raise ValueError(f"Snapshot drift: {record['path']}")
    if sha(parent / "overlay-manifest.json") != manifest["batch_teacher_input_manifest_sha256"]:
        raise ValueError("Parent manifest drift")
    includes = [f"-I{output / 'source/runtime'}",
        f"-I{output / 'source/dev/benchmarks/prefill4k_attention'}"]
    command = ["xcrun", "-sdk", "macosx", "clang++", *includes, *FLAGS]
    config = "sealed-batch-teacher-" + sha(manifest_path)[:16]
    provenance = []
    host = output / "host"
    core = output / "core"
    host.mkdir(exist_ok=True)
    core.mkdir(exist_ok=True)
    changed = {"FlashMTP", "FlashBatchMTPForward", "FlashWorker"}
    for source in sorted((parent / "host").glob("*.o")):
        if source.stem in changed:
            continue
        destination = host / source.name
        shutil.copy2(source, destination)
        provenance.append({"output": str(destination), "reused": str(source), "sha256": sha(source)})
    for relative in ("metal/MetalBackend.o", "metal/DeviceCapabilities.o",
        "engine/Protocol.o", "engine/MemoryGovernor.o"):
        source = ROOT / "build/flash-next/engine" / relative
        destination = core / source.name
        shutil.copy2(source, destination)
        provenance.append({"output": str(destination), "reused": str(source), "sha256": sha(source)})
    library = output / "splash.metallib"
    shutil.copy2(parent / "splash.metallib", library)
    provenance.append({"output": str(library), "reused": str(parent / "splash.metallib"), "sha256": sha(library)})

    def compile_one(name: str) -> dict:
        extension = ".mm" if name == "FlashWorker" else ".cpp"
        source = output / "source/runtime/flash" / (name + extension)
        destination = host / (name + ".o")
        invocation = [*command, *( ["-fobjc-arc"] if extension == ".mm" else []),
            "-MMD", "-MP", "-c", str(source), "-o", str(destination)]
        subprocess.run(invocation, cwd=ROOT, check=True)
        destination.with_suffix(destination.suffix + ".config").write_text(config + "\n")
        return {"output": str(destination), "source": str(source), "source_sha256": sha(source),
            "sha256": sha(destination), "command": invocation}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        provenance.extend(pool.map(compile_one, sorted(changed)))
    objects = sorted(host.glob("*.o")) + sorted(core.glob("*.o"))
    frameworks = ["-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit"]
    subprocess.run([*command, "-fobjc-arc", *map(str, objects), *frameworks,
        "-o", str(output / "splash-flash")], cwd=ROOT, check=True)
    nonworker = [obj for obj in objects if obj.name != "FlashWorker.o"]
    support_source = lambda relative: output / "source" / relative if (output / "source" / relative).exists() else ROOT / relative
    for name, source in (("policy-cpu", support_source("dev/benchmarks/prefill_batch_teacher_sep21/policy_cpu.cpp")),
        ("memory-cpu", ROOT / "dev/benchmarks/prefill4k_wide_memory.mm"),
        ("batch-teacher-oracle", support_source("dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm"))):
        if source.exists():
            subprocess.run([*command, "-fobjc-arc", str(source), *map(str, nonworker),
                *frameworks, "-o", str(output / name)], cwd=ROOT, check=True)
    linked_artifacts = {name: {"binary_sha256": sha(output / name), "source_sha256": sha(source)}
        for name, source in (("policy-cpu", support_source("dev/benchmarks/prefill_batch_teacher_sep21/policy_cpu.cpp")),
            ("memory-cpu", ROOT / "dev/benchmarks/prefill4k_wide_memory.mm"),
            ("batch-teacher-oracle", support_source("dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm")))
        if (output / name).exists() and source.exists()}
    report = {"schema": "splash-batch-teacher-isolated-host-build-v1", "gpu_executed": False,
        "payload_bytes_read": 0, "parent_sha256": sha(parent / "splash-flash"),
        "binary_sha256": sha(output / "splash-flash"), "metallib_sha256": sha(library),
        "flags": FLAGS, "manifest_sha256": sha(manifest_path), "linked_artifacts": linked_artifacts,
        "inputs": provenance}
    (output / "host-build-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "inputs"}))


if __name__ == "__main__":
    main()
