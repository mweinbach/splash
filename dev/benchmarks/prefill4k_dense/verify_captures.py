#!/usr/bin/env python3
"""CPU-only capture hashes/coefficient identity/finite-word checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare import ROLES

parser = argparse.ArgumentParser()
parser.add_argument("manifest",type=Path)
parser.add_argument("--operands",type=Path,default=Path("install/local-models/Flash-Next-operands-v1/manifest.json"))
parser.add_argument("--out",type=Path)
args = parser.parse_args()
captures = json.loads(args.manifest.read_text())
roles = [case["projection"] for case in captures["cases"]]
if len(roles) != len(ROLES) or set(roles) != set(ROLES):
    raise ValueError("capture must contain all10 unique measured roles")
operands = json.loads(args.operands.read_text())
expected = {e["projection"]:e for e in operands["entries"] if e["format"] == "BF16"}
results = []
for case in captures["cases"]:
    original = expected[case["projection"]]
    if case["weights_sha256"] != original["payload_sha256"]:
        raise ValueError("captured BF16 coefficient bits differ from verified saved operand")
    lengths = {
        "weights":case["input_size"]*case["output_size"]*2,
        "input":case["rows"]*case["input_size"]*2,
        "expected":case["rows"]*case["output_size"]*2,
    }
    checks = {}
    for field,length in lengths.items():
        path = Path(case[field + "_file"])
        payload = path.read_bytes()
        if len(payload) != length or hashlib.sha256(payload).hexdigest() != case[field + "_sha256"]:
            raise ValueError("capture file hash/extent invalid")
        words = np.frombuffer(payload,dtype="<u2")
        nonfinite = int(np.count_nonzero((words & 0x7F80) == 0x7F80))
        if nonfinite:
            raise ValueError("nonfinite captured BF16 value")
        checks[field] = {"bytes":length,"nonfinite":nonfinite,"sha256_exact":True}
    results.append({"projection":case["projection"],"saved_coefficients_exact":True,"files":checks})
report = {"schema":"splash-prefill4k-dense-capture-verification-v1","valid":True,
          "gpu_executed":False,"cases":results}
if args.out:
    args.out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"capture_cases":len(results),"valid":True,"gpu_executed":False}))
