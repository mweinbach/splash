#!/usr/bin/env python3
"""Freeze a Root-only bounded component command; never execute GPU or read operands."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/qsa_online_packed_v_sep22")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--actual-manifest", type=Path)
    args = parser.parse_args()
    build, report = args.build.resolve(), args.report.resolve()
    mode = "actual" if args.actual_manifest else "synthetic"
    registration = build / ("root-" + mode + "-registration.json")
    if registration.exists() or any(Path(str(report) + suffix).exists()
                                    for suffix in ("", ".writing", ".final-writing")):
        raise ValueError("Fresh command and report paths are required")
    receipt_path = build / "CPU_READY.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "online-QSA-packed-V-standalone-CPU-v1"
            or receipt.get("pass") is not True or receipt.get("GPU_work") is not False
            or receipt.get("model_capture_payload_reads") != 0
            or receipt.get("worker_integration") is not False):
        raise ValueError("A passing CPU-only standalone component is required")
    runner = build / "source" / PRIVATE / "run_root.py"
    argv = [str(build / "oracle"), str(build / "component.metallib"), str(report), "--" + mode]
    fixture = None
    if args.actual_manifest:
        manifest = args.actual_manifest.resolve()
        # Metadata pin only. Operand files are opened exclusively by the Root oracle.
        fixture = {"manifest": str(manifest), "manifest_sha256": sha(manifest),
                   "scope": "old-B4-lane0-layer3-fresh2K-projected-inputs-U32-0a383"}
        argv.append(str(manifest))
    code = [{"path": str(build / "source" / r["path"]), "sha256": r["sha256"]}
            for r in receipt["sources"]]
    spec = {"schema": "online-QSA-packed-V-Root-registration-v1", "role": "bounded-component",
            "Root_GPU_only": True, "whole_model_qualified": False, "build": str(build),
            "CPU_READY_sha256": sha(receipt_path), "source_identity_sha256": receipt["source_identity_sha256"],
            "runner": str(runner), "runner_sha256": sha(runner), "argv": argv,
            "report": str(report), "mode": mode, "fixture": fixture, "code_sources": code,
            "environment": {"QSA_ONLINE_PACKED_V_ROOT_GPU": "1"},
            "native_ceiling_bytes": 1 << 30, "Host_allowance_bytes": 512 << 20,
            "combined_component_admission_bytes": (1 << 30) + (512 << 20),
            "warm_GPU_ms_each_minimum": 150, "balanced_pairs": 18,
            "timing_scope": "immutable-prepared-attention;original3-dispatches-vs-pack-plus3-inclusive",
            "normal_standard_or_MTP_permission_created": False}
    registration.write_text(json.dumps(spec, indent=2) + "\n")
    invocation = [str(ROOT / ".venv/bin/python"), "-B", str(runner), "--registration", str(registration),
                  "--expected-registration-sha256", sha(registration), "--run-root-gpu"]
    command = build / ("run-root-" + mode + ".sh")
    command.write_text("#!/bin/sh\nset -eu\nexec " + shlex.join(invocation) + "\n")
    print(json.dumps({"registration": str(registration), "registration_sha256": sha(registration),
                      "command": str(command), "command_sha256": sha(command), "GPU_work": False}))


if __name__ == "__main__":
    main()
