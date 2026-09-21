"""Freeze an ABI-preserving private backend diagnostic; never submit GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=PROJECT / "build/flash-next")
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    base, build = args.base.resolve(), args.build.resolve()
    build.mkdir(parents=True, exist_ok=False)
    snapshot = build / "snapshot"
    snapshot.mkdir()
    for name in ("MetalBackend.mm", "Policy.hpp", "policy_test.cpp"):
        shutil.copy2(SOURCE / name, snapshot / name)
    objects = sorted((base / "flash").glob("*.o"))
    core = [base / "engine/metal/DeviceCapabilities.o",
            base / "engine/engine/Protocol.o",
            base / "engine/engine/MemoryGovernor.o"]
    if len(objects) != 42 or not all(path.is_file() for path in core):
        raise RuntimeError(f"Expected 42 frozen Flash objects, got {len(objects)}")
    for path in objects + core:
        target = build / "objects" / path.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    frozen_objects = sorted((build / "objects").rglob("*.o"))
    library = build / "splash.metallib"
    shutil.copy2(base / "splash.metallib", library)
    shutil.copy2(base / "splash-flash.config", build / "splash-flash.config")
    shutil.copy2(base / "splash.metallib.config", build / "splash.metallib.config")
    commands = []

    def run(command: list[str]) -> str:
        commands.append(command)
        (build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    compiler = ["xcrun", "-sdk", "macosx", "clang++", "-std=c++20",
                "-Wall", "-Wextra", "-Werror", "-Iruntime", "-Iruntime/metal",
                "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    backend = build / "MetalBackend.o"
    run([*compiler, "-O3", "-fobjc-arc", "-c", str(snapshot / "MetalBackend.mm"),
         "-o", str(backend)])
    binary = build / "splash-flash"
    run([*compiler, "-O3", "-fobjc-arc", str(backend),
         *(str(path) for path in frozen_objects), "-framework", "Foundation",
         "-framework", "Metal", "-framework", "IOKit", "-o", str(binary)])
    policy = build / "policy-test"
    run([*compiler, "-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
         str(snapshot / "policy_test.cpp"), "-o", str(policy)])
    qualification = json.loads(run([str(policy)]))
    qualification["production_modified"] = False
    qualification["boundary_order"] = ["precommit", "postcommit", "scheduled", "completed"]
    qualification["gpu_executed"] = False
    (build / "cpu-qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "splash-private-command-boundary-memory-v10-build",
        "gpu_executed": False,
        "public_command_timing_abi_bytes": 200,
        "production_modified": False,
        "base": str(base),
        "base_binary_sha256": sha(base / "splash-flash"),
        "binary_sha256": sha(binary),
        "metallib_sha256": sha(library),
        "source_sha256": {path.name: sha(path) for path in snapshot.iterdir()},
        "host_objects_sha256": {str(path.relative_to(build)): sha(path)
                               for path in frozen_objects},
        "backend_sha256": sha(backend),
    }
    (build / "frozen-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"valid": True, "gpu_executed": False, "binary": str(binary),
                      "cpu_checks": qualification["checks"]}))


if __name__ == "__main__":
    main()
