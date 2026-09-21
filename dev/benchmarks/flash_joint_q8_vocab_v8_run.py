"""Freeze the bounded joint Q8 vocabulary screen; --run is Root-only GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = PROJECT / "build/flash-trained-head-joint-q8-v8-v2"
FIXTURE = PROJECT / "build/flash-v6-verifier-attribution/fixture"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--context", type=int, choices=(128, 2048))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Root alone: execute actual target/head Metal commands")
    args = parser.parse_args()
    build = args.build.resolve()
    settings_file = build / "frozen-environment.json"
    settings = json.loads(settings_file.read_text())
    if len(settings) != 38 or any(not key.startswith("SPLASH_FLASH_") or not isinstance(value, str) for key, value in settings.items()):
        raise ValueError("Frozen v7 environment must contain 36 flags and both saved stores")
    controls = {} if args.context is None else {"FLASH_JOINT_Q8_CONTEXT": str(args.context)}
    executable = build / "flash-joint-q8-vocab-v8-oracle"
    library = build / "splash.metallib"
    package = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    report = args.report.resolve()
    for path in (executable, library, package, FIXTURE):
        if not path.exists():
            raise FileNotFoundError(path)
    for suffix in ("", ".invocation.json"):
        if Path(str(report) + suffix).exists():
            raise FileExistsError(f"Choose fresh output: {report}{suffix}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    command = [str(executable), str(library), str(package), str(FIXTURE), str(report)]
    record = {
        "schema": "splash-private-joint-q8-vocabulary-screen-invocation-v8",
        "preparation_gpu_work": False,
        "execution_requested": args.run,
        "gpu_execution_owner": "root",
        "command_timing_abi_bytes": 200,
        "command": command,
        "environment": settings,
        "controls": controls,
        "binary_sha256": sha(executable),
        "metallib_sha256": sha(library),
        "frozen_environment_json_sha256": sha(settings_file),
        "frozen_build_source_manifest_sha256": sha(build / "frozen-build-source-manifest.json"),
        "fixture_provenance_sha256": sha(FIXTURE / "fixture-provenance.json"),
        "scope": "initial B4R1 Last, true target last premixer features and exact teacher-primed QSA; six AB/BA normal pairs per context",
        "strict_contract": "each full-vocabulary lane relative-L2<=1e-4, full premixer BF16 exact, full QSA state exact, exact per-lane GPU/CPU greedy",
        "numerical_rejection_exit_code": 2,
        "execution_failure_exit_code": 1,
        "deferred": ["future continuation1..4", "truncate-overwrite proof"],
        "no_service_claim": True,
    }
    print(json.dumps(record, indent=2))
    if not args.run:
        return 0
    report.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(report) + ".invocation.json").open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    launch = {key: value for key, value in os.environ.items() if not key.startswith(("SPLASH_FLASH_", "FLASH_HEAD_PROFILE_", "FLASH_JOINT_Q8_", "SPLASH_PRIVATE_JOINT_Q8_"))}
    launch.update(settings)
    launch.update(controls)
    return subprocess.call(command, cwd=PROJECT, env=launch)


if __name__ == "__main__":
    raise SystemExit(main())
