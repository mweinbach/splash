#!/usr/bin/env python3
"""Compile LLVM only and reject versioned Metal helper ODR collisions."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def compile_ir(source, output):
    command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", "-O3",
               "-I" + str(SOURCE / "frozen"), "-mmacosx-version-min=27.0",
               "-S", "-emit-llvm", str(source), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    text = output.read_text()
    names = {m.group(1) for m in re.finditer(r"^define linkonce_odr.*?@([^ (]+)\(", text, re.M)
             if "gwy_" in m.group(1) or "wys_" in m.group(1)}
    return names, {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "ir": str(output.relative_to(ROOT)), "ir_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "command": command, "versioned_helpers": sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite symbol proof")
    build = args.output.resolve().parent
    build.mkdir(parents=True, exist_ok=True)
    candidate, c = compile_ir(SOURCE / "candidate.metal", build / "candidate.ll")
    control, n = compile_ir(SOURCE / "frozen/v3_candidate.metal", build / "v3-control.ll")
    old_control, o = compile_ir(ROOT / "dev/benchmarks/gdn_nax_chunks_sep21_v4/frozen/v3_candidate.metal", build / "v4-colliding-control.ll")
    old = sorted(candidate & old_control)
    fresh = sorted(candidate & control)
    assert len(old) == 6, "Did not reproduce the six incompatible v4 helper symbols"
    assert not fresh, "V5 still shares version-specific custom Metal helper symbols"
    snapshot, s = compile_ir(SOURCE / "snapshot.metal", build / "snapshot.ll")
    old_snapshot, os = compile_ir(SOURCE / "frozen/v5_snapshot.metal", build / "v5-snapshot-control.ll")
    snapshot_shared = sorted(snapshot & old_snapshot)
    assert not snapshot_shared, "V6 shares version-specific snapshot/restore helpers with v5 control"
    record = {"schema": "splash.gdn-wy-metal-symbol-proof.v1", "pass": True,
              "old_incompatible_shared_helpers": old, "new_shared_versioned_helpers": fresh,
              "v6_snapshot_control_shared_helpers": snapshot_shared,
              "candidate_stride_floats": 13408, "control_stride_floats": 13344,
              "v4_first_case_control_payload_bytes": 7686144,
              "v4_writer_hull_if_candidate_helper_selected_bytes": 7723008,
              "v4_possible_overrun_bytes": 36864,
              "cause": "Kernel entrypoints were renamed but outlined linkonce_odr template helper symbols were shared for incompatible layouts.",
              "gpu_cause_confirmation": False, "gpu_executed": False,
              "modules": [c, n, o, s, os]}
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "pass": True, "old_collisions": len(old), "new_collisions": len(fresh)}))


if __name__ == "__main__":
    main()
