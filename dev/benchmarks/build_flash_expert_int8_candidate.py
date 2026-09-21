#!/usr/bin/env python3
"""Build the isolated saved-INT8 expert oracle without executing GPU work."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-build", type=Path, default=Path("build/flash-next"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group-size", type=int, choices=(0, 64), default=0)
    args = parser.parse_args()
    output = args.output or Path("build/flash-expert-int8" / Path("candidate-g64" if args.group_size else "candidate"))
    stem = "flash_expert_int8_g64" if args.group_size else "flash_expert_int8"
    output.mkdir(parents=True, exist_ok=True)
    common = ["-O3", "-Wall", "-Wextra", "-Werror", "-Iruntime", "-Idev/benchmarks",
              "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", *common, "-c",
         f"dev/benchmarks/{stem}_candidate.metal", "-o",
         str(output / f"{stem}_candidate.air")])
    run(["xcrun", "-sdk", "macosx", "clang++", "-std=c++20", *common, "-fobjc-arc", "-c",
         f"dev/benchmarks/{stem}_oracle.mm", "-o",
         str(output / f"{stem}_oracle.o")])
    objects = sorted((args.base_build / "flash").glob("*.o"))
    objects = [path for path in objects if path.name != "FlashWorker.o"]
    if not objects:
        raise ValueError("base build contains no Flash objects; build flash-next first")
    core = [args.base_build / "engine" / relative for relative in
            ["metal/MetalBackend.o", "metal/DeviceCapabilities.o", "engine/Protocol.o",
             "engine/MemoryGovernor.o"]]
    sources = sorted(Path("runtime/metal/kernels").glob("*/*.metal"))
    airs = [args.base_build / "metal" / source.relative_to("runtime/metal/kernels").with_suffix(".air")
            for source in sources]
    for path in objects + core + airs:
        if not path.is_file():
            raise ValueError(f"base build is missing {path}")
    run(["xcrun", "-sdk", "macosx", "clang++", "-std=c++20", "-O3",
         "-mmacosx-version-min=27.0", "-fobjc-arc", str(output / f"{stem}_oracle.o"),
         *map(str, objects + core), "-framework", "Foundation", "-framework", "Metal",
         "-framework", "IOKit", "-o", str(output / "flash-expert-int8-oracle")])
    run(["xcrun", "-sdk", "macosx", "metallib", *map(str, airs),
         str(output / f"{stem}_candidate.air"), "-o", str(output / "splash.metallib")])
    run([str(output / "flash-expert-int8-oracle"), "--cpu-self-test"])
    print(f"Built {output}; no GPU commands submitted.")


if __name__ == "__main__":
    main()
