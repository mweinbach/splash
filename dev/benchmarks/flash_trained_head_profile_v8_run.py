"""Freeze a bounded original trained-head invocation; --run is Root-only GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = PROJECT / "build/flash-trained-head-profile-v8-v2"
FIXTURE = PROJECT / "build/flash-v6-verifier-attribution/fixture"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "command", "stage", "dispatch"), default="normal")
    parser.add_argument("--context", type=int, choices=(128, 2048))
    parser.add_argument("--phase", choices=("proposal", "committed_fold", "joint_proposal"))
    parser.add_argument("--rows", type=int, choices=(1, 4, 8))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD, help="Frozen private artifact directory")
    parser.add_argument("--run", action="store_true", help="Root alone: execute actual Metal target/head inference")
    args = parser.parse_args()
    build = args.build.resolve()
    if not 0 <= args.warmup <= 2 or not 1 <= args.repeats <= 8:
        parser.error("warmup/repeats exceed the diagnostic bounds")
    if (args.rows == 1 and args.phase == "committed_fold") or (
        args.rows in (4, 8) and args.phase in ("proposal", "joint_proposal")
    ):
        parser.error("phase/rows selection has no supported geometry")
    environment_file = build / "frozen-environment.json"
    settings = json.loads(environment_file.read_text())
    if len(settings) != 38 or any(
        not key.startswith("SPLASH_FLASH_") or not isinstance(value, str)
        for key, value in settings.items()
    ):
        raise ValueError("Frozen v7 environment must contain 36 flags and both saved-store directories")
    controls = {
        "FLASH_HEAD_PROFILE_MODE": args.mode,
        "FLASH_HEAD_PROFILE_WARMUP": str(args.warmup),
        "FLASH_HEAD_PROFILE_REPEATS": str(args.repeats),
    }
    for name, value in (("CONTEXT", args.context), ("PHASE", args.phase), ("ROWS", args.rows)):
        if value is not None:
            controls[f"FLASH_HEAD_PROFILE_{name}"] = str(value)
    executable = build / "flash-trained-head-profile-v8-oracle"
    library = build / "splash.metallib"
    package = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    report = args.report.resolve()
    for path in (executable, library, FIXTURE, package):
        if not path.exists():
            raise FileNotFoundError(path)
    for suffix in ("", ".commands.jsonl", ".trace.jsonl", ".invocation.json"):
        if Path(str(report) + suffix).exists():
            raise FileExistsError(f"Choose a fresh report name: {report}{suffix}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    command = [str(executable), str(library), str(package), str(FIXTURE), str(report)]
    record = {
        "schema": "splash-original-trained-head-attribution-v8-invocation-v1",
        "preparation_gpu_work": False,
        "execution_requested": args.run,
        "gpu_execution_owner": "root",
        "command_timing_abi_bytes": 200,
        "inherited_flash_settings_removed": True,
        "command": command,
        "environment": settings,
        "attribution_controls": controls,
        "binary_sha256": sha(executable),
        "metallib_sha256": sha(library),
        "fixture_provenance_sha256": sha(FIXTURE / "fixture-provenance.json"),
        "frozen_environment_json_sha256": sha(environment_file),
        "build_source_manifest_sha256": sha(build / "frozen-build-source-manifest.json"),
        "timing_scope": "One original trained-head body, true target features, teacher primed state; no extra dispatches; no HTTP performance claim",
        "perturbation": "Stage changes encoder boundaries; dispatch adds timestamp barriers; post-command hidden/logit hashes occur outside measured call",
        "committed_fold_scope": "R4/R8 true greedy target continuation folds, Last vocabulary output; every pair uses a real previous target premixer feature and next committed target token",
    }
    print(json.dumps(record, indent=2))
    if args.run:
        report.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(report) + ".invocation.json").open("x") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        launch = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SPLASH_FLASH_") and not key.startswith("FLASH_HEAD_PROFILE_")
        }
        launch.update(settings)
        launch.update(controls)
        return subprocess.call(command, cwd=PROJECT, env=launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
