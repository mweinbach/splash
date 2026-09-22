#!/usr/bin/env python3
"""Seal a fresh normal-source control snapshot and a separate exact-bulk build.

This prepares the existing Top64 model route without changing coefficients,
teacher priming, numerical identity, normal source, or installed defaults.
No model payload, GPU or network work occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import bulk_runtime_overlay

ROOT = bulk_runtime_overlay.ROOT


def prepare(output: Path, control: Path) -> dict:
    output, control = output.resolve(), control.resolve()
    for path in (output, control):
        if ROOT / "build" not in path.parents or path.exists():
            raise ValueError("Choose fresh separate build outputs beneath repository build")
    if output == control:
        raise ValueError("Control and candidate outputs must differ")
    relatives = [path.relative_to(ROOT) for path in sorted((ROOT / "runtime/flash").glob("*")) if path.is_file()]
    relatives += [Path(f"runtime/metal/kernels/shared/{name}.metal")
                  for name in ("flash_gdn", "flash_gdn_fused", "flash_gdn_staged")]
    records, sources = [], {}
    for relative in relatives:
        data = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        records.append({"path": str(relative), "patched": False,
                        "original_sha256": digest, "overlay_sha256": digest})
        sources[relative] = data
    manifest = {"schema": 1, "route": "private-normal-source-snapshot-top64-control-v1",
                "normal_source_snapshot": True, "normal_sources_modified": False,
                "arithmetic_change": False, "trained_mtp_source_changed": False,
                "payload_bytes_read": 0, "gpu_work": False, "files": records}
    control.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="private-normal-control-", dir=control.parent) as temporary:
        staged = Path(temporary) / "output"
        for relative, data in sources.items():
            destination = staged / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(control)
    result = bulk_runtime_overlay.generate(control, output)
    result["control_source_snapshot"] = str(control)
    result["control_uses_original_numerical_model_identity"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-top64")
    parser.add_argument("--control", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-top64-control")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.control)))


if __name__ == "__main__":
    main()
