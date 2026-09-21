"""CPU-only frozen primitive builder; this never submits a Metal command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "build/flash-trained-head-joint-q8-v8-v2"
SHADERS = PROJECT / "build/flash-default-v7-instrumented/metal"
MODES = {
    "decode": ("flash_joint_q8_exact_bf16_v9_primitive.mm", "flash_joint_q8_exact_bf16_v9.metal"),
    "lookup": ("flash_joint_q8_exact_bf16_v9_lut_primitive.mm", "flash_joint_q8_exact_bf16_v9.metal"),
    "lookup-geometry": ("flash_joint_q8_exact_bf16_v9_lut_geometry_primitive.mm", "flash_joint_q8_exact_bf16_v9_validated_lut.metal"),
    "register": ("flash_joint_q8_exact_bf16_v9_register_primitive.mm", "flash_joint_q8_exact_bf16_v9_register.metal"),
    "register-rows": ("flash_joint_q8_exact_bf16_v9_register_rows_primitive.mm", "flash_joint_q8_exact_bf16_v9_register.metal"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=False)
    source_name, shader_name = MODES[args.mode]
    snapshot = build / "snapshot"
    snapshot.mkdir()
    for name in (source_name, shader_name):
        shutil.copy2(PROJECT / "dev/benchmarks" / name, snapshot / name)
    shutil.copy2(BASE / "frozen-environment.json", build / "frozen-environment.json")
    flags = ["-O3", "-Wall", "-Wextra", "-Werror", "-Iruntime", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    commands = []

    def run(command: list[str]) -> None:
        commands.append(command)
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        (build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(result.stderr)

    air = build / "exact.air"
    library = build / "splash.metallib"
    executable = build / "oracle"
    run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", *flags, "-c", str(snapshot / shader_name), "-o", str(air)])
    airs = sorted(SHADERS.glob("*/*.air"))
    objects = sorted((BASE / "objects").rglob("*.o"))
    if not airs or len(objects) < 30:
        raise RuntimeError("Missing frozen ABI200 v7 shaders or fresh v8 host objects")
    run(["xcrun", "-sdk", "macosx", "metallib", *(str(path) for path in airs), str(air), "-o", str(library)])
    run(["xcrun", "-sdk", "macosx", "clang++", "-std=c++20", *flags, "-fobjc-arc", str(snapshot / source_name), *(str(path) for path in objects), "-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit", "-o", str(executable)])
    self_test = subprocess.check_output([str(executable), "--cpu-self-test"], text=True)
    (build / "cpu-self-test.json").write_text(self_test)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "splash-private-exact-q8-bf16-v9-primitive-build",
        "mode": args.mode,
        "gpu_executed": False,
        "command_timing_abi_bytes": 200,
        "production_modified": False,
        "source_sha256": sha(snapshot / source_name),
        "shader_sha256": sha(snapshot / shader_name),
        "binary_sha256": sha(executable),
        "metallib_sha256": sha(library),
        "frozen_environment_sha256": sha(build / "frozen-environment.json"),
        "fresh_v8_host_objects_from": str(BASE),
        "host_object_count": len(objects),
        "host_objects_sha256": {str(path): sha(path) for path in objects},
        "base_build_manifest_sha256": sha(BASE / "frozen-build-source-manifest.json"),
    }
    (build / "frozen-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"CPU-only {args.mode} primitive built: {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
