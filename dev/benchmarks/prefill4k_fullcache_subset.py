#!/usr/bin/env python3
"""Repack a frequency-selected expert subset from the certified full store.

CPU-only. Heavy copy/hash work requires Root's shared-memory slot. Coefficients
are copied as exact I8/F32 bytes; no source checkpoint read or requantization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from prefill4k_fullcache_convert import ROOT, private_converter, sha


def inspect(source, plan_path, output):
    base = private_converter()
    source, plan_path, output = (Path(p).resolve() for p in (source, plan_path, output))
    source_manifest = source / "manifest.json"
    raw = source_manifest.read_bytes()
    full = base._json(raw)
    certificate = base._json((source / "coefficient-certificate.json").read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if (not certificate["pass"] or certificate["gpu_work"] or not certificate["certificate_completed_before_atomic_publication"]
            or certificate["full_manifest_sha256"] != digest or full["total_bytes"] != 121173442560
            or full["planned_allocation_bytes"] != 121174228992
            or not all(ids == list(range(512)) for ids in full["selected_experts"])):
        raise ValueError("Subset requires the completed certified full512 source")
    plan_raw = plan_path.read_bytes()
    plan = base._json(plan_raw)
    selected = base.validate_plan(plan, full["source_identity_sha256"])
    if plan["requested_limit"] != 256 or not all(len(ids) ==256 for ids in selected):
        raise ValueError("Private subset requires uniform observed Top256 inventories")
    if plan["scope"]["selection_policy"] != "positive counts only; count descending then ID ascending; selected maps sorted by ID":
        raise ValueError("Selection must preserve the validated observed-frequency policy")
    if ROOT / "build/prefill4k-fullcache-artifacts" not in output.parents or output.exists():
        raise ValueError("Subset destination must be NEW inside the private artifacts directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    layers = [base.layout_layer(i, len(ids)) for i, ids in enumerate(selected)]
    total = sum(layer["bytes"] for layer in layers)
    base.check_destination(source, output, plan_path, total)
    for old in full["layers"]:
        p = source / old["path"]
        if p.stat().st_size != old["bytes"] or p.stat().st_mode & 0o222:
            raise ValueError("Certified full source layer is not a readonly exact-size file")
    for count in (64,128):
        prior = base._json((ROOT / f"build/release/flash/hot-expert-plan{count}.json").read_bytes())
        if plan["captures"] != prior["captures"] or not all(set(a).issubset(b) for a,b in zip(prior["selected_experts"], selected)):
            raise ValueError("Top256 must use matching captures and contain both qualified smaller inventories")
    report = {
        "gpu_work": False, "heavy_copy_started": False,
        "source": str(source), "source_manifest_sha256": digest,
        "plan": str(plan_path), "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "output": str(output), "count": 256, "target_layers": 48,
        "mapped_bytes": total, "planned_allocation_bytes": total +48 * base.ALIGNMENT,
        "frequency_coverage_estimate": plan["aggregate"]["hit_rate"],
        "coverage_scope": "observed supplied captures; not general workload coverage or measured speed",
        "selection": plan["scope"]["selection_policy"], "contains_top64_and_top128": True,
        "arithmetic": "Exact byte subset of certified full512; inherited numerical alternative; no new rounding",
    }
    return base, full, certificate, selected, layers, report


def repack(source, plan_path, output):
    base, full, inherited, selected, layers, report = inspect(source, plan_path, output)
    source, plan_path, output = (Path(p).resolve() for p in (source, plan_path, output))
    snapshots = {source / "manifest.json": base._snapshot(source / "manifest.json"),
                 source / "coefficient-certificate.json": base._snapshot(source / "coefficient-certificate.json"),
                 plan_path: base._snapshot(plan_path)}
    for old in full["layers"]:
        snapshots[source / old["path"]] = base._snapshot(source / old["path"])
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    copied = {"codes": 0, "scales": 0}
    try:
        for layer in layers:
            index = layer["layer_index"]
            old = full["layers"][index]
            old_path = source / old["path"]
            if sha(old_path) != old["sha256"]:
                raise ValueError("Full source layer differs from certified SHA256")
            path = temporary / layer["path"]
            with old_path.open("rb") as original, path.open("w+b") as destination:
                destination.truncate(layer["bytes"])
                for role,(n,k) in base.PROJECTIONS.items():
                    for plane,per_expert in (("codes", n*k),("scales",n*4)):
                        out_layout = layer["projections"][role][plane]
                        in_layout = old["projections"][role][plane]
                        digest = hashlib.sha256()
                        for rank,expert in enumerate(selected[index]):
                            original.seek(in_layout["offset"] +expert *per_expert)
                            raw = original.read(per_expert)
                            if len(raw) != per_expert:
                                raise ValueError("Short certified-source expert read")
                            destination.seek(out_layout["offset"] +rank *per_expert)
                            if destination.write(raw) != len(raw):
                                raise ValueError("Short subset expert write")
                            digest.update(raw)
                            copied[plane] += len(raw)
                        out_layout["sha256"] = digest.hexdigest()
                destination.flush(); os.fsync(destination.fileno())
            path.chmod(0o444)
            # Read output once and independently check each plane against the
            # copied source digest, while also computing its whole-file SHA.
            whole = hashlib.sha256()
            cursor = 0
            with path.open("rb") as verified:
                for role in base.PROJECTIONS:
                    for plane in ("codes","scales"):
                        layout = layer["projections"][role][plane]
                        padding = verified.read(layout["offset"] -cursor)
                        if any(padding):
                            raise ValueError("Subset alignment padding is nonzero")
                        whole.update(padding)
                        remaining = layout["length"]
                        plane_digest = hashlib.sha256()
                        while remaining:
                            raw = verified.read(min(8 <<20,remaining))
                            if not raw:
                                raise ValueError("Subset readback is truncated")
                            plane_digest.update(raw); whole.update(raw)
                            remaining -=len(raw)
                        if plane_digest.hexdigest() !=layout["sha256"]:
                            raise ValueError("Subset plane readback differs from exact certified-source bytes")
                        cursor = layout["offset"] +layout["length"]
                tail = verified.read()
                if any(tail) or cursor +len(tail) !=layer["bytes"]:
                    raise ValueError("Subset tail or length differs")
                whole.update(tail)
            layer["sha256"] = whole.hexdigest()
            print(json.dumps({"phase":"subset_layer_stored", "layer":index, "selected":256, "bytes":layer["bytes"], "sha256":layer["sha256"]}), flush=True)
        if copied != {"codes":60397977600,"scales":188743680}:
            raise ValueError("Subset copied-byte cardinality mismatch")
        record = {key:full[key] for key in base.MANIFEST_KEYS if key not in {"plan_sha256","selected_experts","layers","total_bytes","planned_allocation_bytes"}}
        record.update({"plan_sha256":report["plan_sha256"], "selected_experts":selected,
                       "layers":layers, "total_bytes":report["mapped_bytes"],
                       "planned_allocation_bytes":report["planned_allocation_bytes"]})
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
        manifest.chmod(0o444)
        for path,snapshot in snapshots.items():
            if base._snapshot(path) !=snapshot:
                raise ValueError("Certified source or frequency plan changed during subset copy")
        certificate = {**report,"pass":True,"heavy_copy_started":True,"gpu_work":False,
                       "model_quality_qualified":False,"copied_code_bytes":copied["codes"],
                       "copied_scale_bytes":copied["scales"],
                       "full_manifest_sha256":sha(manifest),
                       "coefficient_certificate_inherited":True,"coefficient_source_elements":inherited["coefficient_elements"],
                       "source_layer_sha_verified_before_copy":True,"output_plane_source_digests_verified_on_readback":True,
                       "coefficient_elements_in_subset":60397977600,"rows_in_subset":47185920,
                       "certified_source_snapshots_unchanged":True,
                       "certificate_completed_before_atomic_publication":True}
        temporary.joinpath("coefficient-certificate.json").write_text(json.dumps(certificate,indent=2,sort_keys=True)+'\n')
        if output.exists():
            raise ValueError("Subset destination appeared; refusing replacement")
        temporary.rename(output)
        print(json.dumps({"phase":"subset_complete",**certificate}),flush=True)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1")
    parser.add_argument("--plan",type=Path,default=ROOT / "build/release/flash/prefill4k-fullcache-plan256.json")
    parser.add_argument("--output",type=Path,default=ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1")
    parser.add_argument("--inspect-only",action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        *_,report = inspect(args.source,args.plan,args.output)
        print(json.dumps(report,indent=2,sort_keys=True))
    else:
        repack(args.source,args.plan,args.output)


if __name__ == "__main__":
    main()
