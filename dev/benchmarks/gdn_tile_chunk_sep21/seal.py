#!/usr/bin/env python3
"""CPU-only standalone source/object witness; never runs a Metal mode."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).resolve().parent

def item(path):
    path=path.resolve()
    data=path.read_bytes()
    return {"path":str(path.relative_to(ROOT)),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",type=Path,default=ROOT/"build/gdn-tile-chunk-sep21")
    p.add_argument("--source-witness",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit("Refusing to replace a seal")
    witness=json.loads(a.source_witness.read_text())
    if not witness.get("pass"):raise SystemExit("Source witness failed")
    build=a.build.resolve()
    sources=[item(f) for f in sorted(SOURCE.rglob("*")) if f.is_file() and "__pycache__" not in f.parts]
    blob="\n".join(f"{x['path']} {x['sha256']}" for x in sources).encode()
    artifact_names=["metal-oracle-tile","gdn-tile.metallib","candidate.air","canonical.air","v6-math-control.air","cpu-oracle.o","cpu-wy-reference"]
    reports=[a.source_witness,build/"cpu-layout.json",build/"cpu-reference.json"]
    record={"schema":"splash.gdn-tile-chunk-cpu-seal.v1","sources":sources,
        "source_tree_sha256":hashlib.sha256(blob).hexdigest(),
        "artifacts":[item(build/n) for n in artifact_names],"reports":[item(f) for f in reports],
        "numerical_policy":"T32/V32/SG8 local reasons, unchanged .095 predicates, deferred state commit, native V8/T16 four waves from hybrid incoming F32 seed",
        "gpu_executed_by_agents":False,"native_chunk_byte_equivalence_proven":False,
        "safe_v6_math_gpu_equivalence_proven":False,"whole_history_f64_accuracy_certificate":False,
        "actual_input_speed_qualified":False,"whole_worker_composition_allowed":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(a.output),"sources":len(sources),"source_tree_sha256":record["source_tree_sha256"]}))

if __name__=="__main__":main()
