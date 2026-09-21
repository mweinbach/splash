#!/usr/bin/env python3
"""Compile/link private proposal experiments; never create a Metal device."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-build", type=Path, default=Path("build/flash-default-v5"))
    parser.add_argument("--build", type=Path, default=Path("build/flash-mtp-float-candidate"))
    arguments = parser.parse_args()
    source = arguments.production_build.resolve(strict=True)
    if source.name == "flash-next":
        raise ValueError("flash-next objects are stale; use a fresh complete build with current CommandTiming ABI")
    private = arguments.build.resolve()
    if source == private or source in private.parents:
        raise ValueError("private build must be separate from production outputs")
    private.mkdir(parents=True, exist_ok=True)
    objects = sorted(source.joinpath("flash").glob("*.o"))
    objects = [path for path in objects if path.name != "FlashWorker.o"]
    objects += [source / name for name in (
        "engine/metal/MetalBackend.o", "engine/metal/DeviceCapabilities.o",
        "engine/engine/Protocol.o", "engine/engine/MemoryGovernor.o")]
    if not objects or any(not path.is_file() for path in objects):
        raise ValueError("production objects are incomplete")
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", "-Iruntime",
             "-Idev/benchmarks", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    compiler = ["xcrun", "-sdk", "macosx", "clang++"]
    abi_source = private / "host-abi-size.cpp"
    abi_source.write_text(
        '#include "metal/MetalBackend.hpp"\n#include "flash/FlashForward.hpp"\n'
        '#include "flash/FlashMTP.hpp"\n#include <iostream>\nint main(){std::cout'
        '<<"{\\\"CommandTiming\\\":"<<sizeof(splash::metal::CommandTiming)'
        '<<",\\\"FlashForwardResult\\\":"<<sizeof(splash::flash::FlashForwardResult)'
        '<<",\\\"FlashMTPResult\\\":"<<sizeof(splash::flash::FlashMTPResult)<<"}";}\n')
    abi_binary = private / "host-abi-size"
    subprocess.run([*compiler, *flags, str(abi_source), "-o", str(abi_binary)], check=True)
    host_abi = json.loads(subprocess.run([str(abi_binary)], text=True, capture_output=True, check=True).stdout)
    for stem in ("FlashMTPFloatCandidate", "FlashBatchMTPFloatCandidate"):
        output = private / f"{stem}.o"
        subprocess.run([*compiler, *flags, "-c", f"dev/benchmarks/{stem}.cpp", "-o", str(output)], check=True)
        objects.append(output)
    built = []
    for stem in ("flash_mtp_float_probe", "flash_mtp_float_candidate_oracle"):
        file = Path("dev/benchmarks") / f"{stem}.mm"
        if not file.is_file():
            continue
        binary = private / stem.replace("_", "-")
        subprocess.run([*compiler, *flags, "-fobjc-arc", str(file), *map(str, objects),
                        "-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit",
                        "-o", str(binary)], check=True)
        built.append(binary.name)
    shutil.copyfile(source / "splash.metallib", private / "splash.metallib")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"pass": True, "gpu_commands": 0, "model_packages_loaded": 0,
              "production_build_read_only": str(source), "private_build": str(private),
              "production_objects": len(objects) - 2, "binaries": built,
              "production_object_sha256": {str(path): digest(path) for path in objects[:-2]},
              "command_timing_header_sha256": digest(Path("runtime/metal/MetalBackend.hpp")),
              "current_header_host_abi_sizes": host_abi,
              "metallib_sha256": digest(private / "splash.metallib"),
              "source_hashes": {str(path): digest(path) for path in (
                  Path("dev/benchmarks/FlashMTPFloatCandidate.cpp"),
                  Path("dev/benchmarks/FlashBatchMTPFloatCandidate.cpp"))},
              "production_sources_changed": False, "gpu_qualification_complete": False}
    private.joinpath("cpu-build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
