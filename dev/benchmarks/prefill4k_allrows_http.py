#!/usr/bin/env python3
"""Write a private all-row HTTP launch witness; root-run only with --run."""
from __future__ import annotations

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--gathered-qmv", action="store_true", help="Select separately identified rows1..16 scalar reduction")
    parser.add_argument("--gathered-mpp", action="store_true", help="Select independently qualified direct M16/N64 gathered producer")
    parser.add_argument("--qmv-columns", type=int, choices=[1, 2], default=1)
    parser.add_argument("--qsa-bulk", action="store_true", help="Select exact bulk SG8 prefill in a compiled combined snapshot")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--rows", type=int, choices=[2048, 4096, 8192], default=2048)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.gathered_mpp and (args.gathered_qmv or args.qmv_columns != 1 or args.qsa_bulk):
        raise ValueError("Direct gathered MPP requires its independent snapshot and no C2/bulk selection")
    if args.qmv_columns == 2 and not args.gathered_qmv:
        raise ValueError("Columns2 requires --gathered-qmv")
    qmv_build = "build/prefill4k-allrows-qmv-c2" if args.qmv_columns == 2 else "build/prefill4k-allrows-qmv"
    if args.qsa_bulk:
        qmv_build = "build/prefill4k-qsa-bulk-allrows-qmv-c2"
    args.build = args.build or ROOT / (qmv_build if args.gathered_qmv or args.qsa_bulk else "build/prefill4k-allrows-full512")
    if args.gathered_mpp and args.build == ROOT / "build/prefill4k-allrows-full512":
        args.build = ROOT / "build/prefill4k-allrows-gathered-mpp"
    if args.witness.exists() or args.admission_report.exists():
        raise ValueError("Choose fresh launch witness and admission report paths")
    package = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    flags = launcher._local_profile_defaults(package)
    flags.update({
        "SPLASH_FLASH_ALLROWS_FULL512_TARGET": "1",
        "SPLASH_FLASH_INT8_EXPERT_STORE": str(ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1"),
        "SPLASH_FLASH_PREFILL_ROWS": str(args.rows),
        "SPLASH_FLASH_MTP_DRAFT_DEPTH": "3",
        "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY": "1",
        "SPLASH_FLASH_PLE_SSD_STREAMING": "1",
        "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "0",
        "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT": "0",
        "SPLASH_FLASH_PREFILL_DENSE_TILES": "1",
        "SPLASH_FLASH_PRIVATE_ADMISSION_REPORT": str(args.admission_report.resolve()),
        "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV": "1" if args.gathered_qmv else "0",
        "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS": str(args.qmv_columns),
        "SPLASH_FLASH_QSA_BULK_PREFILL": "1" if args.qsa_bulk else "0",
        "SPLASH_FLASH_QSA_BULK_PREFILL_SG8": "1" if args.qsa_bulk else "0",
        "SPLASH_FLASH_ALLROWS_GATHERED_MPP": "1" if args.gathered_mpp else "0",
    })
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SPLASH_FLASH_")}
    environment.update(flags)
    command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "server/server.py"),
               "--local-package", str(package), "--model", "local/Qwen3.8-Flash-Next-oQ4e-mtp",
               "--binary", str(args.build.resolve() / "splash-flash"), "--port", str(args.port),
               "--max-context", "16384", "--max-memory", "auto", "--no-webui"]
    import prefill4k_allrows_store
    import prefill4k_allrows_qmv
    derivative = ("splash.private-allrows-target-v1\nsource=edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
                  "\nstore=ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1"
                  f"\npolicy={prefill4k_allrows_store.POLICY}\nmtp=original-trained-bank\n")
    if args.gathered_qmv:
        if args.qmv_columns == 2:
            import prefill4k_allrows_qmv_c2
            policy = prefill4k_allrows_qmv_c2.POLICY
        else:
            policy = prefill4k_allrows_qmv.POLICY
        derivative += f"small_row_policy={policy}\n"
    if args.gathered_mpp:
        import prefill4k_allrows_gathered_mpp
        derivative += f"small_row_policy={prefill4k_allrows_gathered_mpp.POLICY}\n"
    witness = {"schema": 1, "root_run": args.run, "arithmetic_change": True,
               "command": command, "policy_environment": flags,
               "binary_sha256": hashlib.sha256((args.build / "splash-flash").read_bytes()).hexdigest(),
               "metallib_sha256": hashlib.sha256((args.build / "splash.metallib").read_bytes()).hexdigest(),
               "expected_original_gpu_bytes": 6370164736,
               "expected_omitted_original_target_tensors": 432,
               "expected_numerical_derivative_sha256": hashlib.sha256(derivative.encode()).hexdigest()}
    args.witness.parent.mkdir(parents=True, exist_ok=True)
    args.admission_report.parent.mkdir(parents=True, exist_ok=True)
    with args.witness.open("x") as file:
        json.dump(witness, file, indent=2)
        file.write("\n")
    print(json.dumps({"gpu_executed": args.run, "witness": str(args.witness), "command": command}))
    if args.run:
        subprocess.run(command, env=environment, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
