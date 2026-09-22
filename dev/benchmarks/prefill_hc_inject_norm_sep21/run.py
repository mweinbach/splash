#!/usr/bin/env python3
"""Preview by default; Root supplies --run to enter the serialized GPU queue."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--build", default="build/prefill-hc-inject-norm-sep21")
p.add_argument("--report", required=True)
p.add_argument("--pairs", default=8, type=int)
p.add_argument("--run", action="store_true")
args = p.parse_args()
if args.pairs < 4 or args.pairs > 64 or args.pairs % 2:
    raise SystemExit("--pairs must be even and 4..64")
manifest = Path(args.build) / "source-manifest.json"
document = json.loads(manifest.read_text())
for entry in document["sources"]:
    path = Path(entry["frozen"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"Frozen source digest mismatch: {path}")
for entry in document["artifacts"]:
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"Artifact digest mismatch: {path}")
for entry in document.get("sdk_inputs", []):
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"SDK metadata digest mismatch: {path}")
report = Path(args.report)
if report.exists():
    raise SystemExit(f"Refusing to overwrite report: {report}")
command = [document["oracle"], "--gpu", document["metallib"], str(report), str(args.pairs)]
print(json.dumps({"command": command, "source_manifest": str(manifest),
                  "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  "run_requested": args.run, "model_payloads_read": False}), flush=True)
if not args.run:
    raise SystemExit(0)
report.parent.mkdir(parents=True, exist_ok=True)
result = subprocess.run(command)
provenance = {"source_manifest": str(manifest), "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "command": command, "gpu_executed": True, "exit_code": result.returncode}
Path(str(report) + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
raise SystemExit(result.returncode)
