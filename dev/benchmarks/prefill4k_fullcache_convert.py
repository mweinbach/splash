#!/usr/bin/env python3
"""CPU-only full target-expert I8 candidate with coefficient certificates.

Writes a NEW private sidecar atomically. The source checkpoint and qualified
top64/top128 stores remain untouched. No GPU or inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dev/tools/flash_int8_expert_store_convert.py"


def replacement(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Private converter source contract drift: {before!r}")
    return text.replace(before, after)


def private_converter():
    source = BASE.read_text()
    source = replacement(source, 'not 1 <= plan["requested_limit"] <= 128', 'not 1 <= plan["requested_limit"] <= 512')
    source = replacement(source, 'not 1 <= selected_count <= 128', 'not 1 <= selected_count <= 512')
    source = replacement(source, '        temporary.rename(output)', '        private_before_publish(temporary, record)\n        temporary.rename(output)')
    source = source.replace("[1,128]", "[1,512]")
    module = types.ModuleType("prefill4k_fullcache_private_converter")
    module.__file__ = str(BASE)
    exec(compile(source, str(BASE) + " [PRIVATE fullcache]", "exec"), module.__dict__)
    module.private_before_publish = lambda *_: (_ for _ in ()).throw(ValueError("Missing private coefficient publication certificate"))
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(base, package, plan, output, prior, budget_evidence):
    package, plan, output, prior = map(lambda p: Path(p).resolve(), (package, plan, output, prior))
    private_parent = ROOT / "build/prefill4k-fullcache-artifacts"
    if private_parent not in output.parents:
        raise ValueError("New fullcache sidecar must be inside build/prefill4k-fullcache-artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Refusing existing fullcache destination")
    record = {
        "schema": base.PLAN_SCHEMA,
        "source_identity": base.EXPECTED_SOURCE_IDENTITY,
        "requested_limit": 512,
        "selected_experts": [list(range(512)) for _ in range(48)],
        "scope": "PRIVATE full target large-row prefill only; small decode and MTP source weights retained",
    }
    if plan.exists():
        if json.loads(plan.read_text()) != record:
            raise ValueError("Existing private plan differs; refusing replacement")
    else:
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    inspected = base.inspect_plan(package, plan)
    base.check_destination(package, output, plan, inspected["total_bytes"])
    manifest = json.loads((package / "manifest.json").read_text())
    selected = base.validate_plan(record, base.EXPECTED_SOURCE_IDENTITY)
    snapshots = {package / "manifest.json": base._snapshot(package / "manifest.json")}
    base.prepare_source(package, manifest, selected, snapshots)
    prior_manifest = prior / "manifest.json"
    prior_raw = prior_manifest.read_bytes()
    prior_record = base._json(prior_raw)
    if len(prior_record["layers"]) != 48 or any(len(ids) != 64 for ids in prior_record["selected_experts"]):
        raise ValueError("Preserved subset comparison requires the qualified uniform top64 store")
    if prior_record["source_manifest_sha256"] != inspected["source_manifest_sha256"]:
        raise ValueError("Preserved top64 store does not derive from this source manifest")
    evidence = json.loads(Path(budget_evidence).read_text())
    final = evidence["final_status"]
    persisted = final["persisted_experts"]
    prior_digest = hashlib.sha256(prior_raw).hexdigest()
    if (not persisted["enabled"] or persisted["store_manifest_sha256"] != prior_digest
            or persisted["expert_count"] != 3072 or persisted["mapped_bytes"] != prior_record["total_bytes"]
            or final["identity"]["source"] != base.EXPECTED_SOURCE_IDENTITY
            or prior_digest not in final["identity"]["kernel_routes"]
            or not final["ple_storage"]["ssd_streaming_enabled"]):
        raise ValueError("Budget evidence does not match the same-source qualified top64/PLE-SSD baseline")
    physical = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True))
    reserve = max(16 << 30, physical // 10)
    baseline_peak = final["memory_actual"]["peak_bytes"]
    candidate_peak_estimate = baseline_peak - prior_record["planned_allocation_bytes"] + inspected["planned_allocation_bytes"]
    limit = physical - reserve
    if candidate_peak_estimate + (2 << 30) > limit:
        raise ValueError("Offline fullcache estimate fails engine limit and2GiB experimental margin")
    inspected.update({
        "gpu_work": False,
        "arithmetic_change": True,
        "source_contract_prepared": True,
        "source_shards_sha_verification": "performed by conversion before any output payload is written",
        "budget_evidence": str(Path(budget_evidence).resolve()),
        "budget_evidence_sha256": sha(budget_evidence),
        "baseline_top64_peak_bytes": baseline_peak,
        "physical_bytes": physical,
        "host_reserve_bytes": reserve,
        "engine_limit_bytes": limit,
        "candidate_peak_estimate_bytes": candidate_peak_estimate,
        "remaining_estimated_engine_headroom_bytes": limit - candidate_peak_estimate,
        "runtime_admission": "real MemoryGovernor reservation still required before mapping/allocation; this estimate is not admission",
        "preserved_top64_manifest_sha256": hashlib.sha256(prior_raw).hexdigest(),
    })
    return inspected, prior_record


def certify(base, prior, prior_record):
    audit = {"coefficient_elements": 0, "rows": 0, "zero_rows": 0,
             "maximum_absolute_error": 0.0, "maximum_error_divided_by_bound": 0.0,
             "error_square_sum": 0.0, "reference_square_sum": 0.0,
             "top64_code_bytes_exact": 0, "top64_scale_bytes_exact": 0,
             "coefficient_bound": "abs(F64(code)*F64(stored_scale)-F64(source_BF16)) <=0.5*stored_scale +32*2^-24*row_absmax",
             "bound_scope": "coefficient error only; no universal relative dot or whole-model bound"}
    original_quantize = base._PILOT.symmetric_int8
    original_writer = base.write_projection

    def checked_quantize(reference):
        codes, scales = original_quantize(reference)
        source = base._PILOT.bf16_to_f32(reference)
        maximum = np.max(np.abs(source), axis=-1)
        expected_scale = np.divide(maximum, np.float32(127), dtype=np.float32)
        expected_scale = np.where(maximum == np.float32(0), np.float32(1), expected_scale).astype(np.float32)
        stored_scale = scales.reshape(maximum.shape)
        if not np.array_equal(stored_scale.view(np.uint32), expected_scale.view(np.uint32)):
            raise ValueError("Stored F32 row scale differs from specified absmax/127 policy")
        if not np.isfinite(stored_scale).all() or (stored_scale <= 0).any() or (codes == -128).any():
            raise ValueError("Invalid signed code or row scale")
        zero = maximum == 0
        if np.any(codes[zero] != 0) or np.any(stored_scale[zero] != np.float32(1)):
            raise ValueError("All-zero row policy differs")
        source64 = source.astype(np.float64)
        reconstructed = codes.astype(np.float64) * stored_scale.astype(np.float64)[..., None]
        error = np.abs(reconstructed - source64)
        bound = np.float64(0.5) * stored_scale.astype(np.float64) + np.float64(32 * 2**-24) * maximum.astype(np.float64)
        if np.any(error > bound[..., None]):
            raise ValueError("Independent coefficient absolute-error certificate failed")
        audit["coefficient_elements"] += reference.size
        audit["rows"] += maximum.size
        audit["zero_rows"] += int(np.count_nonzero(zero))
        audit["maximum_absolute_error"] = max(audit["maximum_absolute_error"], float(np.max(error)))
        audit["maximum_error_divided_by_bound"] = max(audit["maximum_error_divided_by_bound"], float(np.max(np.max(error, axis=-1) / bound)))
        audit["error_square_sum"] += float(np.sum(error * error, dtype=np.float64))
        audit["reference_square_sum"] += float(np.sum(source64 * source64, dtype=np.float64))
        return codes, scales

    def checked_writer(stream, layout, planes, experts, **kwargs):
        result = original_writer(stream, layout, planes, experts, **kwargs)
        stream.flush()
        layer = int(layout["source_prefix"].split(".")[3])
        role = layout["source_prefix"].split(".")[-1]
        old_layout = prior_record["layers"][layer]["projections"][role]
        old_path = prior / prior_record["layers"][layer]["path"]
        h, n, k = layout["dimensions"]
        for plane, dtype, shape, old_shape, element_bytes in (
            ("codes", np.int8, (h, n, k), (64, n, k), 1),
            ("scales", np.uint32, (h, n), (64, n), 4),
        ):
            new = np.memmap(stream.name, dtype=dtype, mode="r", offset=layout[plane]["offset"], shape=shape)
            old = np.memmap(old_path, dtype=dtype, mode="r", offset=old_layout[plane]["offset"], shape=old_shape)
            for rank, expert in enumerate(prior_record["selected_experts"][layer]):
                if not np.array_equal(new[expert], old[rank]):
                    raise ValueError(f"Preserved top64 payload differs at{layer}/{role}/{plane}/{expert}")
                audit[f"top64_{'code' if plane == 'codes' else 'scale'}_bytes_exact"] += old[rank].size * element_bytes
            del new, old
        return result

    base._PILOT.symmetric_int8 = checked_quantize
    base.write_projection = checked_writer
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1")
    parser.add_argument("--plan", type=Path, default=ROOT / "build/release/flash/prefill4k-fullcache-plan512.json")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1")
    parser.add_argument("--preserved-top64", type=Path, default=ROOT / "install/local-models/Flash-Next-int8-experts-top64-v1")
    parser.add_argument("--budget-evidence", type=Path, default=ROOT / "build/release/flash/prefill4k-current-context-baseline.json")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--cpu-self-test", action="store_true")
    args = parser.parse_args()
    base = private_converter()
    if args.cpu_self_test:
        audit = certify(base, Path("unused"), {"layers": []})
        bits = np.zeros((3, 2, 2560), dtype=np.uint16)
        bits[1, :, :64] = 0x3F80
        bits[1, :, 64:128] = 0x3C00
        bits[2, :, :] = 0x0001
        codes, scales = base._PILOT.symmetric_int8(bits)
        if audit["coefficient_elements"] != bits.size or audit["zero_rows"] != 2:
            raise ValueError("Coefficient certificate CPU extent or zero-row fixture failed")
        cancellation_source = np.float64(1) - np.float64(128) / np.float64(128)
        cancellation_candidate = np.float64(scales[1, 0, 0]) * (np.float64(codes[1, 0, 0]) - 128 * np.float64(codes[1, 0, 64]))
        if cancellation_source != 0 or cancellation_candidate == 0:
            raise ValueError("Adversarial cancellation must expose declared quantization arithmetic change")
        plan = {"schema": base.PLAN_SCHEMA, "source_identity": base.EXPECTED_SOURCE_IDENTITY,
                "requested_limit": 512, "selected_experts": [list(range(512)) for _ in range(48)]}
        selected = base.validate_plan(plan, base.EXPECTED_SOURCE_IDENTITY)
        if sum(base.layout_layer(i, len(ids))["bytes"] for i, ids in enumerate(selected)) != 121173442560:
            raise ValueError("Private full512 geometry/accounting differs")
        print(json.dumps({"pass": True, "gpu_work": False, "checks": ["full512_source_geometry", "all_zero_rows", "BF16_min_subnormal", "coefficient_absolute_error_bound", "scale_bits", "cancellation_declared_nonexact"], "cancellation_source_dot": float(cancellation_source), "cancellation_candidate_dot": float(cancellation_candidate)}))
        return
    inspected, prior_record = prepare(base, args.package, args.plan, args.output, args.preserved_top64, args.budget_evidence)
    print(json.dumps({"phase": "private_preflight", **inspected}, sort_keys=True), flush=True)
    preflight_path = args.plan.with_name("prefill4k-fullcache-preflight512.json")
    preflight_path.write_text(json.dumps(inspected, indent=2, sort_keys=True) + "\n")
    if args.inspect_only:
        return
    prior_snapshots = {args.preserved_top64 / layer["path"]: base._snapshot(args.preserved_top64 / layer["path"]) for layer in prior_record["layers"]}
    prior_manifest_path = args.preserved_top64 / "manifest.json"
    prior_snapshots[prior_manifest_path] = base._snapshot(prior_manifest_path)
    for layer in prior_record["layers"]:
        if sha(args.preserved_top64 / layer["path"]) != layer["sha256"]:
            raise ValueError("Preserved top64 payload does not match its qualified manifest")
    audit = certify(base, args.preserved_top64, prior_record)
    def certificate_before_publish(temporary, converted):
        for name, expected in {"coefficient_elements": 120795955200, "rows": 94371840,
                               "top64_code_bytes_exact": 15099494400, "top64_scale_bytes_exact": 47185920}.items():
            if audit[name] != expected:
                raise ValueError(f"Fullcache certificate skipped elements: {name}={audit[name]}, expected{expected}")
        for path, snapshot in prior_snapshots.items():
            if base._snapshot(path) != snapshot:
                raise ValueError("Preserved top64 payload/manifest changed during CPU conversion")
        audit.update({"pass": True, "gpu_work": False, "model_quality_qualified": False,
                      "source_converter_sha256": sha(BASE), "output": str(args.output.resolve()),
                      "full_manifest_sha256": sha(temporary / "manifest.json"),
                      "total_bytes": converted["total_bytes"], "planned_allocation_bytes": converted["planned_allocation_bytes"],
                      "preserved_top64_payload_snapshots_unchanged": True,
                      "certificate_completed_before_atomic_publication": True})
        temporary.joinpath("coefficient-certificate.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    base.private_before_publish = certificate_before_publish
    base.convert(args.package, args.plan, args.output, emit=lambda line: print(line, flush=True))
    print(json.dumps({"phase": "private_certificate_complete", **audit}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
