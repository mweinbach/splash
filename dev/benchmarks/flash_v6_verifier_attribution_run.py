"""Prepare a v6 target-verifier diagnostic invocation; --run is Root-only GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
BUILD = PROJECT / "build/flash-v6-verifier-attribution"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "command", "stage", "dispatch"), default="normal")
    parser.add_argument("--context", type=int, choices=(128, 2048))
    parser.add_argument("--physical-rows", type=int, choices=(4, 8, 16))
    parser.add_argument("--executor", choices=("singleton", "joint"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--state-hashes", type=int, choices=(0, 1), default=1)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Root only: execute actual Metal target/head inference")
    args = parser.parse_args()
    if not 0 <= args.warmup <= 2 or not 1 <= args.repeats <= 8:
        parser.error("warmup/repeats exceed the diagnostic bounds")
    fixture = BUILD / "fixture"
    environment_file = fixture / "v6-environment.json"
    settings = json.loads(environment_file.read_text())
    if len(settings) != 36 or any(not key.startswith("SPLASH_FLASH_") or not isinstance(value, str) for key, value in settings.items()):
        raise ValueError("The frozen v6 environment must contain34 flags and both saved-store directories")
    settings["SPLASH_FLASH_CAPTURE_EXPERT_IDS"] = "1"
    controls = {"FLASH_V6_VERIFY_MODE": args.mode, "FLASH_V6_VERIFY_WARMUP": str(args.warmup),
                "FLASH_V6_VERIFY_REPEATS": str(args.repeats), "FLASH_V6_VERIFY_STATE_HASHES": str(args.state_hashes)}
    if args.context is not None:
        controls["FLASH_V6_VERIFY_CONTEXT"] = str(args.context)
    if args.physical_rows is not None:
        controls["FLASH_V6_VERIFY_PHYSICAL_ROWS"] = str(args.physical_rows)
    if args.executor:
        controls["FLASH_V6_VERIFY_EXECUTOR"] = args.executor
    executable, library = BUILD / "flash-v6-verifier-attribution-oracle", BUILD / "splash.metallib"
    package = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    report = args.report.resolve()
    for path in (executable, library, fixture, package):
        if not path.exists():
            raise FileNotFoundError(path)
    for suffix in ("", ".commands.jsonl", ".trace.jsonl", ".routes.jsonl", ".invocation.json"):
        if Path(str(report) + suffix).exists():
            raise FileExistsError(f"Choose a fresh report name: {report}{suffix}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    command = [str(executable), str(library), str(package), str(fixture), str(report)]
    source_paths = [PROJECT / "runtime/metal/MetalBackend.hpp", PROJECT / "runtime/metal/MetalBackend.mm",
                    PROJECT / "runtime/flash/FlashForward.cpp", PROJECT / "runtime/flash/FlashBatchVerify.cpp",
                    PROJECT / "runtime/flash/FlashMTP.cpp", PROJECT / "runtime/flash/FlashMTP.hpp",
                    BUILD / "snapshot/flash/FlashBatchVerify.hpp", BUILD / "snapshot/flash/FlashBatchVerify.cpp",
                    PROJECT / "dev/benchmarks/flash_v6_verifier_attribution_oracle.mm"]
    record = {"schema": "splash-v6-verifier-attribution-invocation-v1", "preparation_gpu_work": False,
              "execution_requested": args.run, "gpu_execution_owner": "root", "command_timing_abi_bytes": 200,
              "inherited_flash_settings_removed": True, "command": command,
              "environment": settings, "attribution_controls": controls,
              "binary_sha256": sha(executable), "metallib_sha256": sha(library),
              "fixture_provenance_sha256": sha(fixture / "fixture-provenance.json"),
              "frozen_environment_json_sha256": sha(environment_file),
              "source_sha256": {str(path.relative_to(PROJECT)): sha(path) for path in source_paths},
              "timing_scope": "actual initial target verification cycles, exact model features and trained drafts; extra48 capture copies and optional pre-call state reads; no HTTP score"}
    print(json.dumps(record, indent=2))
    if args.run:
        report.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(report) + ".invocation.json").open("x") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        launch = {key: value for key, value in os.environ.items()
                  if not key.startswith("SPLASH_FLASH_") and not key.startswith("FLASH_V6_VERIFY_")}
        launch.update(settings)
        launch.update(controls)
        return subprocess.call(command, cwd=PROJECT, env=launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
