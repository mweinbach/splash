#!/usr/bin/env python3
"""Prepare one frozen Root component command; never execute GPU work."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/expert_r1_finite_summary_sep22")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-base", type=int, default=0)
    args = parser.parse_args()
    build, report = args.build.resolve(), args.report.resolve()
    if not 0 <= args.layer < 48 or not 0 <= args.expert_base <= 502:
        raise ValueError("The selected layer or ten-expert range is out of bounds")
    if report.exists() or (build / "root-registration.json").exists():
        raise ValueError("Fresh command and report paths are required")
    receipt = json.loads((build / "CPU_READY.json").read_text())
    if receipt.get("pass") is not True or receipt.get("GPU_work") is not False:
        raise ValueError("A passing CPU-only component is required")
    runner = build / "source" / PRIVATE / "run_root.py"
    command = [str(build / "oracle"), str(build / "component.metallib"),
               str(ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1"),
               str(report), str(args.layer), str(args.expert_base)]
    code = [{"path": str(build / "source" / row["path"]), "sha256": row["sha256"]}
            for row in receipt["sources"]]
    registration = {"schema": "R1-native-finite-summary-Root-component-registration-v1",
                    "role": "bounded-component", "Root_GPU_only": True,
                    "whole_model_qualified": False, "build": str(build),
                    "CPU_READY_sha256": sha(build / "CPU_READY.json"),
                    "source_identity_sha256": receipt["source_identity_sha256"],
                    "runner": str(runner), "runner_sha256": sha(runner),
                    "argv": command, "report": str(report), "code_sources": code,
                    "coefficient_payload_bytes_read_only_by_Root": 49305600,
                    "fixture": "synthetic finite normalized BF16 R1; ten distinct original experts",
                    "scope": "Original two-dispatch versus finite-summary four-dispatch own chain; no native DRAM or whole-model speed claim"}
    path = build / "root-registration.json"
    path.write_text(json.dumps(registration, indent=2) + "\n")
    invocation = ["python3", str(runner), "--registration", str(path),
                  "--expected-registration-sha256", sha(path), "--run-root-gpu"]
    shell = "#!/bin/sh\nset -eu\nexec " + shlex.join(invocation) + "\n"
    (build / "run-root-component.sh").write_text(shell)
    print(json.dumps({"registration": str(path), "registration_sha256": sha(path),
                      "command": str(build / "run-root-component.sh"),
                      "command_sha256": sha(build / "run-root-component.sh"), "GPU_work": False}))


if __name__ == "__main__":
    main()
