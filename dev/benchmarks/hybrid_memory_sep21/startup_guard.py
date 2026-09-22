#!/usr/bin/env python3
"""CPU-only compiled-worker startup profile guard, with no usable model.

The temporary package has configuration JSON but no manifest or weight payload.
The temporary Full512 directory has no manifest. A valid profile reaches that
deliberate refusal before MetalBackend construction. Earlier refusals identify
profile dependency mistakes. No model or saved operand payload is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
MISSING_MANIFEST = "Flash INT8 expert store: manifest could not be opened without following symlinks"


def run(build: Path) -> dict:
    binary = build / "splash-flash"
    source = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    # The intentional failure site must remain statically before backend.
    metadata_site = source.index("const auto hybridStoreMetadata = loadFlashInt8ExpertStoreMetadata")
    backend_site = source.index("      metal::MetalBackend backend(", metadata_site)
    assert metadata_site < backend_site
    config_path = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1/config.json"
    config_bytes = config_path.read_bytes()
    profile = json.loads((build / "environment.json").read_text())
    assert profile["SPLASH_FLASH_GDN_BATCH_ILP"] == "0", "fixed singleton profile must disable batch-only GDN ILP"
    cases = [
        ("valid_fixed_mtp_profile", {}, MISSING_MANIFEST),
        ("valid_standard_profile", {"SPLASH_FLASH_MTP": "0", "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY": "0"}, MISSING_MANIFEST),
        ("batch_only_ilp_rejected", {"SPLASH_FLASH_GDN_BATCH_ILP": "1"}, "SPLASH_FLASH_GDN_BATCH_ILP requires GDN_STAGED=1 and BATCH_PREFILL=1"),
        ("batch_prefill_rejected", {"SPLASH_FLASH_BATCH_PREFILL": "1"}, "private fixed-R4 hybrid forbids true batch routes"),
        ("deeper_verification_rejected", {"SPLASH_FLASH_MTP_DRAFT_DEPTH": "7"}, "private hybrid requires arena2048, fixed depth3"),
        ("teacher_without_mtp_rejected", {"SPLASH_FLASH_MTP": "0"}, "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY=1 requires SPLASH_FLASH_MTP=1"),
        ("missing_bulk_prerequisite_rejected", {"SPLASH_FLASH_QSA_MPP": "0"}, "private hybrid requires SPLASH_FLASH_QSA_MPP=1"),
        ("alternative_bf16_cache_rejected", {"SPLASH_FLASH_DENSE_SMALL_ROWS": "1"}, "private fixed-R4 hybrid requires DENSE_SMALL_ROWS=0"),
    ]
    if "SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT" in profile:
        cases.extend([
            ("composite_flag_off_accepted", {"SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT": "0"}, MISSING_MANIFEST),
            ("composite_flag_malformed_rejected", {"SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT": "invalid"}, "SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT must be 0 or 1"),
            ("composite_without_saved_rejected", {"SPLASH_FLASH_SAVED_OPERANDS_RESIDENT": "0"}, "private hybrid Q4 expert residency requires saved composite operands"),
            ("legacy_original_text_flag_still_rejected", {"SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT": "1"}, "private fixed-R4 hybrid forbids true batch routes"),
        ])
    results = []
    with tempfile.TemporaryDirectory(prefix="hybrid-cpu-profile-guard-") as temporary:
        folder = Path(temporary).resolve()
        package = folder / "config-only-package"
        empty_store = folder / "empty-full512-store"
        package.mkdir()
        empty_store.mkdir()
        (package / "config.json").write_bytes(config_bytes)
        for name, overrides, expected in cases:
            env = {k: v for k, v in os.environ.items() if not k.startswith("SPLASH_FLASH_")}
            env.update(profile)
            env.update(overrides)
            # Both paths are metadata-only traps; no existing model store is
            # available to this worker, even if a preflight regresses.
            env["SPLASH_FLASH_INT8_EXPERT_STORE"] = str(empty_store)
            env["SPLASH_FLASH_OPERAND_STORE"] = str(folder / "missing-dense-store")
            complete = subprocess.run([str(binary.resolve()), "serve-flash-native", str(package), "16384", "auto"],
                                      env=env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, timeout=10)
            passed = complete.returncode != 0 and expected in complete.stderr
            results.append(dict(name=name, exit_code=complete.returncode, expected_refusal=expected,
                                stdout=complete.stdout, stderr=complete.stderr, passed=passed))
    return dict(schema="splash-hybrid-fixed-r4-compiled-startup-profile-guard-v1",
                passed=all(c["passed"] for c in results), gpu_execution=False,
                model_loaded=False, model_payload_bytes_read=0,
                binary=str(binary.resolve()), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                configuration_metadata_sha256=hashlib.sha256(config_bytes).hexdigest(),
                intentional_manifest_failure_source_site_before_backend=True,
                cases=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.build.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], cases=len(result["cases"]), gpu_execution=False,
                         model_payload_bytes_read=0, failed=[c["name"] for c in result["cases"] if not c["passed"]])))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
