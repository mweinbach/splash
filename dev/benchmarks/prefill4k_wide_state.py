#!/usr/bin/env python3
"""Freeze current Top64 flags and launch the private full-state oracle; dry-run default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from install import launcher


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-wide")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=[4096, 8192], default=8192)
    parser.add_argument("--dense-tiles", choices=["off", "on"], default="off",
                        help="Default off isolates wide geometry from the separate dense tile candidate")
    parser.add_argument("--run", action="store_true", help="Submit GPU work; root serializes all GPU activity")
    args = parser.parse_args()
    package = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SPLASH_FLASH_", "PREFILL4K_"))}
    defaults = launcher._local_profile_defaults(package)
    profile = json.loads((ROOT / ".splash-local-profile.json").read_text())
    if len(defaults) != len(profile["environment"]) + 2 or defaults.get("SPLASH_FLASH_MTP_DRAFT_DEPTH") != "3":
        raise RuntimeError("Current depth3 policy must resolve with its 2 saved paths")
    environment.update(defaults)
    controls = {
        "PREFILL4K_WIDE_STATE_ROWS": str(args.rows),
        "SPLASH_FLASH_PREFILL_DENSE_TILES": "1" if args.dense_tiles == "on" else "0",
    }
    environment.update(controls)
    build, tokens, report = args.build.resolve(), args.tokens.resolve(), args.report.resolve()
    binary, library = build / "state-oracle", build / "splash.metallib"
    command = [str(binary), str(library), str(package), str(tokens), str(report)]
    witness = report.with_suffix(report.suffix + ".invocation.json")
    for path in (report, Path(str(report) + ".checkpoints.jsonl"), witness):
        if path.exists():
            raise RuntimeError(f"Choose a fresh output; {path} exists")
    if len(json.loads(tokens.read_text())) not in (2048, 4096, 8192):
        raise RuntimeError("Use exact frozen code2048/code4096/code8192 tokens")
    provenance = {
        "schema": "splash-private-wide-state-invocation-v1",
        "gpu_requested": args.run,
        "gpu_process_launched": False,
        "gpu_executed_by_this_driver": False,
        "command": command,
        "policy_environment": defaults,
        "controls": controls,
        "inherited_flash_flags_removed": True,
        "profile_sha256": sha(ROOT / ".splash-local-profile.json"),
        "binary_sha256": sha(binary),
        "metallib_sha256": sha(library),
        "tokens_json_sha256": sha(tokens),
        "private_overlay_manifest_sha256": sha(build / "overlay-manifest.json"),
        "private_state_access_manifest_sha256": sha(build / "state-source/manifest.json"),
        "oracle_source_sha256": sha(ROOT / "dev/benchmarks/prefill4k_wide_state.mm"),
        "capacity": 16384,
        "baseline_arena_rows": 2048,
        "candidate_arena_rows": args.rows,
        "claims": "trunk persistent state and final output diagnostics only; no teacher/HTTP/performance qualification",
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"gpu_requested": args.run, "invocation": str(witness), "command": command}))
    if not args.run:
        return 0
    provenance["gpu_process_launched"] = True
    provenance["gpu_executed_by_this_driver"] = None
    witness.write_text(json.dumps(provenance, indent=2) + "\n")
    result = subprocess.run(command, env=environment, cwd=ROOT, check=False)
    provenance["driver_exit_code"] = result.returncode
    provenance["gpu_executed_by_this_driver"] = (
        bool(json.loads(report.read_text()).get("gpu_executed")) if report.exists() else None
    )
    provenance["checkpoint_trace_exists"] = Path(str(report) + ".checkpoints.jsonl").exists()
    witness.write_text(json.dumps(provenance, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
