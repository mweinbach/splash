"""Describe a fresh CPU-prepared v5 attribution invocation; --run is Root-only GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
BUILD = PROJECT / "build/flash-v5-attribution"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "command", "stage", "dispatch"), default="normal")
    parser.add_argument("--lanes", type=int, choices=(1, 4), default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Root only: execute actual Metal inference after stopping the model server")
    args = parser.parse_args()
    if not (0 <= args.warmup <= 4 and 1 <= args.repeats <= 8 and 0 <= args.decode_steps <= 128):
        parser.error("warmup/repeat/decode settings are out of bounds")
    fixture = BUILD / "fixture"
    environment_file = fixture / "v5-environment.json"
    environment = json.loads(environment_file.read_text())
    if len(environment) != 32 or any(not key.startswith("SPLASH_FLASH_") or not isinstance(value, str) for key, value in environment.items()):
        raise ValueError("the frozen v5 environment must contain all 30 flags and both saved-store paths")
    token_file = fixture / f"ctx2048-sample0-width{args.lanes}-lane0.tokens.json"
    provenance = fixture / "fixture-provenance.json"
    executable, library = BUILD / "flash-v5-attribution-oracle", BUILD / "splash.metallib"
    package = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    for path in (token_file, provenance, executable, library, package):
        if not path.exists():
            raise FileNotFoundError(path)
    report = args.report.resolve()
    for path in (report, Path(str(report) + ".trace.jsonl"), Path(str(report) + ".commands.jsonl"), Path(str(report) + ".invocation.json")):
        if path.exists():
            raise FileExistsError(f"Use a fresh report name: {path}")
    controls = {
        "FLASH_V5_ATTRIBUTION_MODE": args.mode,
        "FLASH_V5_ATTRIBUTION_LANES": str(args.lanes),
        "FLASH_V5_ATTRIBUTION_PREFILL_ROWS": "2048",
        "FLASH_V5_ATTRIBUTION_CAPACITY": "8192",
        "FLASH_V5_ATTRIBUTION_VERIFY_ROWS": "16",
        "FLASH_V5_ATTRIBUTION_WARMUP": str(args.warmup),
        "FLASH_V5_ATTRIBUTION_REPEATS": str(args.repeats),
        "FLASH_V5_ATTRIBUTION_DECODE_STEPS": str(args.decode_steps),
    }
    if args.lanes == 4:
        controls["FLASH_V5_ATTRIBUTION_LANE_TOKEN_PREFIX"] = str(fixture / "ctx2048-sample0-width4-lane")
        # Native prefix convention is PREFIX0.json, not the descriptive fixture suffix.
        controls["FLASH_V5_ATTRIBUTION_LANE_TOKEN_PREFIX"] = str(fixture / "native-width4-lane")
    command = [str(executable), str(library), str(package), str(token_file), str(report), str(provenance)]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "schema": "splash-v5-attribution-invocation-v1", "gpu_executed": False,
        "root_only_gpu_authorization_required_for_run": True,
        "inherited_flash_settings_removed": True, "command": command,
        "environment": environment, "attribution_controls": controls,
        "source_oracle_sha256": sha(PROJECT / "dev/benchmarks/flash_v5_attribution_oracle.mm"),
        "source_forward_sha256": sha(PROJECT / "runtime/flash/FlashForward.cpp"),
        "source_backend_sha256": sha(PROJECT / "runtime/metal/MetalBackend.mm"),
        "source_host_abi_sha256": sha(PROJECT / "runtime/metal/MetalBackend.hpp"),
        "binary_sha256": sha(executable), "metallib_sha256": sha(library),
        "environment_json_sha256": sha(environment_file), "provenance_json_sha256": sha(provenance),
        "token_json_sha256": sha(token_file),
        "timing_scope": "normal/command are diagnostic; stage/dispatch alter encoding and are never HTTP throughput evidence",
    }
    print(json.dumps(record, indent=2))
    if args.run:
        report.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(report) + ".invocation.json").open("x") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
        launch = {key: value for key, value in os.environ.items()
                  if not key.startswith("SPLASH_FLASH_") and not key.startswith("FLASH_V5_ATTRIBUTION_")}
        launch.update(environment)
        launch.update(controls)
        return subprocess.call(command, cwd=PROJECT, env=launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
