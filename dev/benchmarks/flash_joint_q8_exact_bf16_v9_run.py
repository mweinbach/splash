"""Freeze exact Q8 BF16 primitive invocation; --run is Root-only GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
DICTIONARY = PROJECT / "build/flash-vocab-bf16-dictionary-v9"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variants")
    parser.add_argument("--patterns", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--rows", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--pairs", type=int, choices=range(1, 13), default=4)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--shader-validation", action="store_true")
    parser.add_argument("--run", action="store_true", help="Root alone: execute actual vocabulary Metal commands")
    args = parser.parse_args()
    build, report = args.build.resolve(), args.report.resolve()
    manifest = json.loads((build / "frozen-source-manifest.json").read_text())
    mode = manifest["mode"]
    executable, library = build / "oracle", build / "splash.metallib"
    settings = json.loads((build / "frozen-environment.json").read_text())
    if len(settings) != 38 or any(not key.startswith("SPLASH_FLASH_") for key in settings):
        raise ValueError("Frozen v7 environment must contain 36 switches and both saved stores")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for path, key in ((executable, "binary_sha256"), (library, "metallib_sha256")):
        if sha(path) != manifest[key]:
            raise ValueError(f"Frozen artifact changed: {path}")
    controls = {"FLASH_Q8_EXACT_PATTERNS": str(args.patterns), "FLASH_Q8_EXACT_PAIRS": str(args.pairs)}
    if mode == "register-rows":
        controls["FLASH_Q8_EXACT_ROWS"] = str(args.rows)
    elif args.rows != 4 or args.patterns == 5:
        raise ValueError("Only register-rows mode supports rows2/3 or nonfinite pattern5")
    if args.variants:
        controls["FLASH_Q8_EXACT_VARIANTS"] = args.variants
    input_sha = None
    if args.input:
        input_file = args.input.resolve()
        if input_file.stat().st_size not in (2560 * 2, 4 * 2560 * 2):
            raise ValueError("Captured input must contain one or four BF16[2560] rows")
        input_sha = sha(input_file)
        controls["FLASH_Q8_EXACT_INPUT_BF16"] = str(input_file)
    dictionary_sha = None
    if mode.startswith("lookup"):
        dictionary_manifest = json.loads((DICTIONARY / "manifest.json").read_text())
        # This private package has independently verified finite words and pair
        # indices. The primitive still audits all actual selected coefficients.
        if sha(DICTIONARY / "dictionary.bf16") != "9a8e340291a3764f3457d393bce600c8cfe1197f58def1f3223e8e2cef11c81a" or sha(DICTIONARY / "group_indices.u16") != "48eb7e9de6252c7f76756353a2911b1139f1f6bb7518b37b7fc3a49510c526c4":
            raise ValueError("Frozen exact dictionary payload changed")
        if dictionary_manifest["schema"] != "splash-private-exact-bf16-q8-vocabulary-dictionary-v1":
            raise ValueError("Unexpected dictionary schema")
        dictionary_sha = sha(DICTIONARY / "manifest.json")
        controls["FLASH_Q8_EXACT_DICTIONARY"] = str(DICTIONARY)
    if args.shader_validation:
        controls.update({"MTL_SHADER_VALIDATION": "1", "MTL_DEBUG_LAYER": "1"})
    command = [str(executable), str(library), str(PACKAGE), str(report)]
    record = {
        "schema": "splash-private-exact-q8-bf16-v9-primitive-invocation",
        "execution_requested": args.run,
        "gpu_execution_owner": "root",
        "mode": mode,
        "command": command,
        "environment": settings,
        "controls": controls,
        "captured_input_sha256": input_sha,
        "dictionary_manifest_sha256": dictionary_sha,
        "frozen_build_manifest_sha256": sha(build / "frozen-source-manifest.json"),
        "mandatory_coefficient_audit_words": 635699200,
        "strict_contract": "every full-vocabulary lane relative-L2<=1e-4; GPU greedy/diagnostics exact; guards unchanged",
        "strict_bound_relaxed": False,
        "no_service_claim": True,
    }
    print(json.dumps(record, indent=2))
    if not args.run:
        return 0
    if report.exists() or Path(str(report) + ".invocation.json").exists():
        raise FileExistsError("Choose a fresh report")
    report.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(report) + ".invocation.json").open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    launch = {key: value for key, value in os.environ.items() if not key.startswith(("SPLASH_FLASH_", "FLASH_Q8_EXACT_"))}
    launch.update(settings)
    launch.update(controls)
    return subprocess.call(command, cwd=PROJECT, env=launch)


if __name__ == "__main__":
    raise SystemExit(main())
