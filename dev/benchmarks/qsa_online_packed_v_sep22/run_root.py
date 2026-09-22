#!/usr/bin/env python3
"""Root-only launch of one externally pinned standalone online QSA component."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args()
    if not args.run_root_gpu:
        raise ValueError("GPU execution requires explicit Root launch")
    path = args.registration.resolve()
    if sha(path) != args.expected_registration_sha256:
        raise ValueError("Registration differs from the external Root pin")
    spec = json.loads(path.read_text())
    if (spec.get("schema") != "online-QSA-packed-V-Root-registration-v1"
            or spec.get("role") != "bounded-component" or spec.get("Root_GPU_only") is not True
            or spec.get("whole_model_qualified") is not False):
        raise ValueError("Unexpected registration schema, owner, or role")
    if Path(spec["runner"]).resolve() != Path(__file__).resolve() or sha(__file__) != spec["runner_sha256"]:
        raise ValueError("Launcher differs from the registered code")
    build = Path(spec["build"]).resolve()
    ready = build / "CPU_READY.json"
    if sha(ready) != spec["CPU_READY_sha256"]:
        raise ValueError("The compiled CPU receipt differs")
    receipt = json.loads(ready.read_text())
    if (receipt.get("schema") != "online-QSA-packed-V-standalone-CPU-v1"
            or receipt.get("pass") is not True or receipt.get("GPU_work") is not False
            or receipt.get("model_capture_payload_reads") != 0
            or receipt.get("worker_integration") is not False):
        raise ValueError("A passing CPU-only standalone receipt is required")
    if receipt["source_identity_sha256"] != spec["source_identity_sha256"]:
        raise ValueError("Registered program identity differs")
    pins = [(build / "source" / r["path"], r["sha256"]) for r in receipt["sources"]]
    pins += [(build / r["path"], r["sha256"]) for r in
             receipt["reused_host_objects"] + receipt["original_ordered_AIRs"] + receipt["artifacts"]]
    pins += [(Path(r["path"]), r["sha256"]) for r in spec["code_sources"]]
    def verify():
        for filename, digest in pins:
            if sha(filename) != digest:
                raise ValueError("Registered program drift: " + str(filename))
    verify()
    argv = spec["argv"]
    if not isinstance(argv, list) or not all(type(x) is str for x in argv):
        raise ValueError("Registered argv must contain only strings")
    mode = spec["mode"]
    expected = [str(build / "oracle"), str(build / "component.metallib"), spec["report"], "--" + mode]
    if mode == "actual":
        fixture = spec["fixture"]
        if sha(fixture["manifest"]) != fixture["manifest_sha256"]:
            raise ValueError("The registered actual metadata manifest differs")
        expected.append(fixture["manifest"])
    elif mode != "synthetic" or spec["fixture"] is not None:
        raise ValueError("Unknown component input mode")
    if argv != expected:
        raise ValueError("Executable, library, fixture, or report differs from registration")
    if spec["environment"] != {"QSA_ONLINE_PACKED_V_ROOT_GPU": "1"}:
        raise ValueError("Unexpected registered environment")
    if spec["warm_GPU_ms_each_minimum"] != 150 or spec["balanced_pairs"] != 18:
        raise ValueError("The fixed balanced inclusive timing policy differs")
    if (spec["native_ceiling_bytes"] != 1 << 30 or spec["Host_allowance_bytes"] != 512 << 20
            or spec["combined_component_admission_bytes"] != (1 << 30) + (512 << 20)
            or receipt["planned_native_ceiling_bytes"] != spec["native_ceiling_bytes"]
            or receipt["planned_Host_allowance_bytes"] != spec["Host_allowance_bytes"]
            or receipt["combined_planned_Governor_admission_bytes"] != spec["combined_component_admission_bytes"]):
        raise ValueError("The separately admitted native and Host component budgets differ")
    report = Path(spec["report"]).resolve()
    if any(Path(str(report) + suffix).exists() for suffix in ("", ".writing", ".final-writing")):
        raise ValueError("The component requires a fresh report")
    prefixes = ("SPLASH_", "FLASH_", "PREFILL_", "QSA_")
    environment = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    environment.update(spec["environment"])
    result = subprocess.run(argv, cwd=build, env=environment, check=False)
    verify()
    # Only the measured oracle can report numerical qualification and timing.
    # This launcher creates no whole-model benchmark or promotion permission.
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
