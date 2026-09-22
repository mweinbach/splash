#!/usr/bin/env python3
"""Prepare the one-trunk allrow rollback oracle invocation; dry-run default."""
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

DERIVATIVE = "2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-allrows-rollback")
    parser.add_argument("--tokens", type=Path, default=ROOT / "build/release/flash/prefill4k-fixture/code2048.tokens.json")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Root-exclusive model load/GPU execution")
    args = parser.parse_args()
    package = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SPLASH_FLASH_", "PREFILL4K_"))}
    defaults = launcher._local_profile_defaults(package)
    controls = {
        "SPLASH_FLASH_ALLROWS_FULL512_TARGET": "1",
        "SPLASH_FLASH_INT8_EXPERT_STORE": str(ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1"),
        "SPLASH_FLASH_MTP_DRAFT_DEPTH": "3",
        "SPLASH_FLASH_PLE_SSD_STREAMING": "1",
        "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "0",
        "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT": "0",
        "SPLASH_FLASH_PREFILL_DENSE_TILES": "1",
    }
    environment.update(defaults)
    environment.update(controls)
    build, tokens, report = args.build.resolve(), args.tokens.resolve(), args.report.resolve()
    binary, library = build / "rollback-oracle", build / "splash.metallib"
    command = [str(binary), str(library), str(package), str(tokens), str(report)]
    witness = report.with_suffix(report.suffix + ".invocation.json")
    for path in (report, Path(str(report) + ".checkpoints.jsonl"), witness):
        if path.exists():
            raise RuntimeError(f"Choose a fresh output; {path} exists")
    if len(json.loads(tokens.read_text())) != 2048:
        raise RuntimeError("Use the exact frozen code2048 fixture")
    profile = json.loads((ROOT / ".splash-local-profile.json").read_text())
    if len(defaults) != len(profile["environment"]) + 2:
        raise RuntimeError("Current local profile and its saved paths failed to resolve")
    manifest_path = build / "overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rollback_read_only_friend") != "FlashAllrowsRollbackOracle":
        raise RuntimeError("Build is not the isolated rollback source copy")
    stale = [record["path"] for record in manifest["files"]
             if sha(build / "source" / record["path"]) != record["overlay_sha256"]]
    if stale:
        raise RuntimeError(f"Rebuild a fresh private source copy: {stale}")
    provenance = {
        "schema": "splash-private-allrows-rollback-invocation-v1",
        "gpu_requested": args.run,
        "gpu_process_launched": False,
        "gpu_executed_by_this_driver": False,
        "model_loaded_by_dry_run": False,
        "model_payload_bytes_read_by_driver": 0,
        "command": command,
        "policy_environment": defaults,
        "controls": controls,
        "inherited_experiment_flags_removed": True,
        "profile_sha256": sha(ROOT / ".splash-local-profile.json"),
        "binary_sha256": sha(binary),
        "metallib_sha256": sha(library),
        "tokens_json_sha256": sha(tokens),
        "private_overlay_manifest_sha256": sha(manifest_path),
        "private_source_files_checked": len(manifest["files"]),
        "oracle_source_sha256": sha(ROOT / "dev/benchmarks/prefill4k_allrows_rollback.mm"),
        "metrics_helper_source_sha256": sha(ROOT / "dev/benchmarks/prefill4k_wide_state.mm"),
        "expected_target_numerical_derivative_sha256": DERIVATIVE,
        "shared_flash_forward_count": 1,
        "shared_full512_store_count": 1,
        "request_state_count": 3,
        "capacity": 16384,
        "maximum_rows": 2048,
        "verify_rows": 4,
        "claims": "same-shape prefix contamination/rollback plus producer diagnostics only; no HTTP, teacher or performance qualification",
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"gpu_requested": args.run, "payload_bytes_read": 0, "invocation": str(witness), "command": command}))
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
