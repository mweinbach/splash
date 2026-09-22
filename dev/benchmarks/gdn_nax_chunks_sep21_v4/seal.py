#!/usr/bin/env python3
"""CPU-only source/artifact witness; never imports Metal or runs the oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def entry(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "dev/benchmarks/gdn_nax_chunks_sep21_v4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=root / "build/gdn-nax-chunks-sep21-v4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing seal witness")
    # Whole-worker preparation is concurrent and has its own witness. Seal only
    # this standalone's explicit source/proof closure, never its mutable scripts.
    names = {"candidate.metal", "snapshot.metal", "native_fallback.metal",
             "native_fallback_audit.metal", "norm_reference.metal", "metal_oracle.mm", "wy_cpu.hpp",
             "wy_cpu_test.cpp", "Makefile", "source_proof.py", "seal.py",
             "README.md", "fallback_review.md", "correction_audit.md", "source_analysis.md"}
    files = sorted(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and (p.relative_to(source).parts[0] in {"frozen", "sdk_probe"}
                        or p.relative_to(source).as_posix() in names))
    artifacts = [build / name for name in ["metal-oracle-wy", "gdn-wy.metallib",
                 "cpu-wy-reference", "cpu-oracle.o", "candidate.air", "canonical.air",
                 "snapshot.air", "native-fallback.air", "native-fallback-audit.air",
                 "v3-control.air", "norm-reference.air"]]
    layout = build / "cpu-layout.json"
    reference = build / "cpu-reference-frozen.json"
    layout_data = json.loads(layout.read_text())
    reference_data = json.loads(reference.read_text())["summary"]
    if not layout_data["pass"] or not reference_data["reference_test_pass"]:
        raise SystemExit("Physical layout or float64 equivalence witness failed")
    entries = [entry(p, root) for p in files]
    ordered = "\n".join(f"{e['path']} {e['sha256']}" for e in entries).encode()
    witness = {
        "schema": "splash.gdn-wy-private-seal.v1",
        "source_tree_sha256": hashlib.sha256(ordered).hexdigest(),
        "sources": entries, "artifacts": [entry(p, root) for p in artifacts],
        "reports": [entry(layout, root), entry(reference, root),
                    entry(build / "replay-source-proof-v2.json", root)],
        "cpu_layout": layout_data, "cpu_reference_summary": reference_data,
        "numerical_policy": "BF16 source/output; F32 state/transforms; relaxed precision disabled; guarded exact native replay from immutable incoming F32 state",
        "strict_f32_qualification": reference_data["f32_quality_qualification_pass"],
        "gpu_executed_by_preparation_agents": False,
        "whole_model_qualified": False,
        "gpu_replay_reference_equivalence_qualified": False,
        "selector_is_whole_history_certificate": False,
        "promotion_allowed": False,
        "caveat": "A CPU seal permits Root diagnostics; it does not override the unchanged numerical gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sources": len(entries),
                     "source_tree_sha256": witness["source_tree_sha256"],
                     "strict_f32_qualification": witness["strict_f32_qualification"]}))


if __name__ == "__main__":
    main()
