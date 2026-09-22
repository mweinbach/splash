#!/usr/bin/env python3
"""Root-only launch of one externally pinned bounded component registration."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args()
    if not args.run_root_gpu:
        raise ValueError("Root GPU execution requires explicit --run-root-gpu")
    registration = args.registration.resolve()
    if sha(registration) != args.expected_registration_sha256:
        raise ValueError("The registration differs from the external Root pin")
    spec = json.loads(registration.read_text())
    if spec.get("schema") != "R1-axis-traversal-Root-component-registration-v1":
        raise ValueError("Unexpected registration schema")
    if spec.get("role") != "bounded-component" or spec.get("Root_GPU_only") is not True:
        raise ValueError("Unexpected registration role or owner")
    if spec.get("whole_model_qualified") is not False:
        raise ValueError("A component registration cannot qualify a whole model")
    if spec["runner_sha256"] != sha(__file__) or Path(spec["runner"]).resolve() != Path(__file__).resolve():
        raise ValueError("The current launcher is not the registered runner")
    build = Path(spec["build"]).resolve()
    receipt_path = build / "CPU_READY.json"
    if sha(receipt_path) != spec["CPU_READY_sha256"]:
        raise ValueError("The compiled CPU receipt differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "R1-axis-traversal-bounded-component-CPU-v1" or receipt.get("pass") is not True:
        raise ValueError("A passing component CPU receipt is required")
    if receipt.get("GPU_work") is not False or receipt.get("model_capture_payload_reads") != 0:
        raise ValueError("The preparation receipt does not describe the registered CPU-only build")
    if spec["source_identity_sha256"] != receipt["source_identity_sha256"]:
        raise ValueError("The registered program identity differs")
    pins = [(build / "source" / row["path"], row["sha256"]) for row in receipt["sources"]]
    pins += [(build / row["path"], row["sha256"]) for row in receipt["reused53"] + receipt["artifacts"]]
    pins += [(Path(row["path"]), row["sha256"]) for row in spec["code_sources"]]
    def verify():
        for path, digest in pins:
            if sha(path) != digest:
                raise ValueError("Registered code or artifact drift: " + str(path))
    verify()
    argv = spec["argv"]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError("The registered argv must contain only strings")
    if len(argv) < 4 or Path(argv[0]).resolve() != build / "oracle" or Path(argv[1]).resolve() != build / "component.metallib":
        raise ValueError("The registered executable or library differs")
    report = Path(spec["report"]).resolve()
    if Path(argv[3]).resolve() != report:
        raise ValueError("The report path differs from registered argv")
    if any(Path(str(report) + suffix).exists() for suffix in ("", ".writing", ".final-writing")):
        raise ValueError("The component requires a fresh report")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SPLASH_")}
    result = subprocess.run(argv, cwd=build, env=environment, check=False)
    verify()
    # Qualification and timing are reported by the measured oracle only. This
    # runner creates no standard-model permission receipt or benchmark gate.
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
