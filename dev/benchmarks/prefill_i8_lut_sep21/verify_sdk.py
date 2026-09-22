#!/usr/bin/env python3
"""Compile-only SDK evidence; never opens model operands or creates a GPU device."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill-i8-lut-sep21")
    args = parser.parse_args()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), str(HERE / "prepare.py")], check=True, cwd=ROOT)
    compiler = ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", "-O3", "-Wall",
                "-Wextra", "-Werror", "-Iruntime", "-I.", "-mmacosx-version-min=27.0", "-c"]
    results = {}
    for label, filename, expected_pass in (("sg2_sg4_cooperative_input", "sdk_probe.metal", False),
                                          ("sg2_sg4_threadgroup_input", "candidate.metal", True)):
        command = compiler + [str(HERE / filename), "-o", str(build / (label + ".air"))]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (build / (label + ".compiler.txt")).write_text(result.stdout + result.stderr)
        observed_pass = result.returncode == 0
        rejected_scope = "Input cooperative tensors require a single SIMD group" in result.stderr
        results[label] = {"command": command, "returncode": result.returncode,
                          "expected_compile_pass": expected_pass, "observed_compile_pass": observed_pass,
                          "scope_static_assert_observed": rejected_scope,
                          "pass": observed_pass == expected_pass and (expected_pass or rejected_scope)}
    report = {"schema": "prefill-i8-lut-sep21-sdk-compile-v1", "gpu_execution": False,
              "model_payload_reads": False, "model_payload_hashing": False,
              "descriptor_m": 32, "descriptor_n": 64, "descriptor_k": [128, 256],
              "simd_groups": [2, 4], "coefficient_types": ["signed I8", "BF16"],
              "candidate_b_storage": "one cooperative threadgroup tile, sequential gate/up reuse",
              "candidate_b_bytes": {"i8_k128": 8192, "i8_k256": 16384,
                                    "bf16_k128": 16384, "bf16_k256": 32768},
              "candidate_entries": 32, "control_entries": 8,
              "matched_uncompressed_entries": 32, "cases": results,
              "pass": all(item["pass"] for item in results.values()),
              "raw_f32_parity": "unmeasured", "scaled_bf16_parity": "unmeasured",
              "full_chain_parity": "unmeasured", "performance": "unmeasured"}
    destination = build / "sdk-compile.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "gpu_execution": False,
                      "report": str(destination), "candidate_b_storage": report["candidate_b_storage"]}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
