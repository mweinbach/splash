#!/usr/bin/env python3
"""CPU-only Metal compiler probes. Does not import or instantiate Metal."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("matmul_probe.metal")
OUT = ROOT / "build/gdn-nax-chunks-sep21/sdk-probe"
OUT.mkdir(parents=True, exist_ok=True)
SHAPES = [(16, 128, 128), (32, 128, 128), (128, 128, 16),
          (128, 128, 32), (128, 128, 128), (16, 16, 128), (32, 32, 128),
          (128, 16, 128), (128, 32, 128)]
DTYPES = [("float", "float"), ("bfloat", "float"), ("float", "bfloat")]
CASES = []
for m, n, k in SHAPES:
    for sg in (4, 8):
        for left, right in DTYPES:
            CASES.append(dict(name=f"m{m}n{n}k{k}-sg{sg}-{left}-{right}",
                              m=m, n=n, k=k, sg=sg, left=left, right=right))
for sg in (4, 8):
    for left, right in DTYPES:
        for variant, defs in (
            ("threadgroup", {"PROBE_ADDRESS": "threadgroup"}),
            ("transpose-right", {"PROBE_TRANSPOSE_RIGHT": "true"}),
            ("relaxed", {"PROBE_RELAXED": "true"}),
            ("cooperative-input", {"PROBE_COOPERATIVE_INPUT": "1"}),
            ("const-input", {"PROBE_CONST": "const"}),
            ("multiply-accumulate", {"PROBE_ACCUMULATE": "1"}),
        ):
            CASES.append(dict(name=f"m32n128k128-sg{sg}-{left}-{right}-{variant}",
                              m=32, n=128, k=128, sg=sg, left=left, right=right,
                              extra=defs))
for left, right in DTYPES:
    CASES.append(dict(name=f"m32n128k128-sg1-{left}-{right}-cooperative-input",
                      m=32, n=128, k=128, sg=1, left=left, right=right,
                      extra={"PROBE_COOPERATIVE_INPUT": "1"}))
for case in CASES:
    case["expected_pass"] = not (case.get("extra", {}).get("PROBE_CONST") or
        (case.get("extra", {}).get("PROBE_COOPERATIVE_INPUT") and case["sg"] > 1))

def compile_case(case):
    defs = {f"PROBE_{key.upper()}": str(case[key])
            for key in ("m", "n", "k", "sg", "left", "right")}
    defs.update(case.get("extra", {}))
    cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.1", "-O3",
           "-Wall", "-Wextra", "-Werror", "-mmacosx-version-min=27.0"]
    cmd += [f"-D{key}={value}" for key, value in defs.items()]
    cmd += ["-c", str(SOURCE), "-o", str(OUT / f"{case['name']}.air")]
    run = subprocess.run(cmd, text=True, capture_output=True)
    (OUT / f"{case['name']}.log").write_text(run.stdout + run.stderr)
    return {**case, "command": cmd, "exit_code": run.returncode,
            "diagnostic": run.stdout + run.stderr}

selected = [case for case in CASES
            if not sys.argv[1:] or any(pattern in case["name"] for pattern in sys.argv[1:])]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    updated = {result["name"]: result for result in executor.map(compile_case, selected)}
existing_path = OUT / "results.json"
existing = ({result["name"]: result for result in json.loads(existing_path.read_text())}
            if existing_path.exists() else {})
results = [{**existing[case["name"]], **case} if case["name"] not in updated
           else updated[case["name"]] for case in CASES
           if case["name"] in existing or case["name"] in updated]
(OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
passed = sum(result["exit_code"] == 0 for result in results)
print(f"Compile only: {passed}/{len(results)} passed")
matched = sum((result["exit_code"] == 0) == result["expected_pass"] for result in results)
print(f"Expected outcome: {matched}/{len(results)} matched")
for result in results:
    if result["exit_code"]:
        errors = [line for line in result["diagnostic"].splitlines()
                  if "error:" in line]
        print(result["name"] + ": " + " | ".join(errors[:1]))
if matched != len(results):
    raise SystemExit(1)
