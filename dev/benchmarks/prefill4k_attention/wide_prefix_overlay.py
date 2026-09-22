#!/usr/bin/env python3
"""Extend the sealed Full512 bulk snapshot to wide fresh prefill calls.

Only the first 2048 QSA rows use dense exact bulk. Later rows keep the
authoritative chronological 128-row sparse path. Cached dense projections
retain their qualified M128 whole-K descriptors at 4096/8192 rows.
This generator reads source and report metadata only, never model payloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FORWARD = "runtime/flash/FlashForward.cpp"
DENSE = "runtime/flash/FlashPrefillDenseTiles.hpp"
PRIVATE = "dev/benchmarks/prefill4k_attention"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Sealed source anchor drift: {before[:100]}")
    return text.replace(before, after, 1)


def transform_forward(text: str) -> str:
    text = replace_once(text,
        "      if (impl_->bulkQSAWorkspace && !verification && begin == 0 && rows == 2048) {\n"
        "        const FlashQSAFastInputs bulkInputs{q, k, v, index,",
        "      uint32_t preparedQSARows = 0;\n"
        "      if (impl_->bulkQSAWorkspace && !verification && begin == 0 &&\n"
        "          (rows == 2048 || rows == 4096 || rows == 8192)) {\n"
        "        const FlashQSAFastInputs bulkInputs{slice(q, 0, 2048, 12288),\n"
        "            slice(k, 0, 2048, 512), slice(v, 0, 2048, 512),\n"
        "            slice(index, 0, 2048, 640),")
    text = replace_once(text,
        "            attentionOutput, diag, {},\n",
        "            slice(attentionOutput, 0, 2048, 6144), diag, {},\n")
    text = replace_once(text,
        "            impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,\n"
        "            impl_->bulkQSAPrefillSG8);\n"
        "      } else {\n"
        "      for (uint32_t offset = 0; offset < rows;) {",
        "            impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, 2048,\n"
        "            impl_->bulkQSAPrefillSG8);\n"
        "        preparedQSARows = 2048;\n"
        "      }\n"
        "      for (uint32_t offset = preparedQSARows; offset < rows;) {")
    text = replace_once(text,
        "        offset += count;\n      }\n      }\n      affine(graph, attention + \".o_proj\", attentionOutput, branch);",
        "        offset += count;\n      }\n      affine(graph, attention + \".o_proj\", attentionOutput, branch);")
    return replace_once(text,
        ";private-qsa-bulk-prefill-begin0-r2048-w128-m16p1-p4-m32p4-v2",
        ";private-qsa-bulk-first2048-begin0-r2048or4096or8192-chronological-w128-tail-v1")


def transform_dense(text: str) -> str:
    anchor = "  if (rows != 2048) return {};"
    if text.count(anchor) != 2:
        raise ValueError("Both sealed cached dense role/geometry guards are required")
    text = text.replace(anchor,
        "  if (rows != 2048 && rows != 4096 && rows != 8192) return {};")
    text = replace_once(text,
        ";dense-bf16-cached-r2048-inspected-shapes-m128n64-simd4or8-whole-k-v1",
        ";private-dense-bf16-cached-r2048or4096or8192-inspected-shapes-m128n64-simd4or8-whole-k-v1")
    text = replace_once(text,
        "// Actual2048-row model input/output fixtures were byte-exact for these whole-K\n"
        "// variants. Selection is restricted by the caller to cached affine projections;",
        "// Actual2048-row fixtures were byte-exact. Wider4096/8192-row eligibility\n"
        "// retains each qualified M128 whole-K descriptor; model parity at those\n"
        "// wider shapes remains unqualified. Only cached affine projections are eligible;")
    return text


def validate_partitions() -> dict:
    checks = 0
    for rows in (1, 127, 128, 129, 2047, 2048, 2049, 4095, 4096, 8191, 8192):
        for begin in (0, 1, 2048):
            for verification in (False, True):
                for bulk in (False, True):
                    prepared = 2048 if bulk and not verification and begin == 0 and rows in (2048,4096,8192) else 0
                    windows = [(begin,prepared)] if prepared else []
                    windows += [(begin+offset,min(128,rows-offset)) for offset in range(prepared,rows,128)]
                    covered = [r for offset,count in windows for r in range(offset,offset+count)]
                    if covered != list(range(begin,begin+rows)):
                        raise AssertionError("QSA row coverage has a missing/duplicate/reordered row")
                    if any(count > 128 for offset,count in windows if offset != begin or not prepared):
                        raise AssertionError("Sparse tail window widened")
                    if prepared and rows > 2048 and windows[1][0] != 2048:
                        raise AssertionError("Sparse tail does not begin at the first uncached row")
                    checks += 1
    return {"partition_combinations_checked": checks, "all_rows_once_in_chronological_order": True,
            "ordinary_sparse_tail_maximum_rows": 128, "bulk_prefix_rows": 2048,
            "nonfresh_or_verification_falls_back": True}


def generate(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if ROOT / "build" not in output.parents or output == base or output.exists():
        raise ValueError("Choose a fresh separate private build directory")
    manifest_bytes = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(manifest_bytes)
    if not parent.get("qsa_bulk_composed") or not parent.get("qsa_bulk_sg8_available"):
        raise ValueError("The sealed exact bulk SG8 Full512 source snapshot is required")
    if parent.get("gathered_qmv_composed") or parent.get("gathered_qmv_c2_composed"):
        raise ValueError("Rejected C2 producer cannot be the source baseline")
    files, records, changed = {}, [], []
    for record in parent["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source manifest path")
        original = (base / "source" / relative).read_bytes()
        if digest(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed source drift: {relative}")
        transform = {FORWARD: transform_forward, DENSE: transform_dense}.get(str(relative))
        data = transform(original.decode()).encode() if transform else original
        files[relative] = data
        result = dict(record)
        result["wide_prefix_input_sha256"] = record["overlay_sha256"]
        result["overlay_sha256"] = digest(data)
        result["wide_prefix_changed"] = data != original
        if data != original:
            result["patched"] = True
            changed.append(str(relative))
        records.append(result)
    if set(changed) != {FORWARD, DENSE}:
        raise AssertionError("Wide prefix modifies exactly Forward scheduling and cached dense role policy")
    checks = validate_partitions()
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-full512-exact-bulk-first2048-wide-cached-dense-m128-v1",
        "wide_prefix_composed": True, "wide_prefix_base_build": str(base),
        "wide_prefix_input_manifest_sha256": digest(manifest_bytes),
        "wide_prefix_generator_sha256": digest(Path(__file__).read_bytes()),
        "wide_prefix_changed_paths": changed, "qsa_bulk_eligible_call": "fresh main rows2048/4096/8192 first2048; chronological authoritative128-row sparse tail",
        "qsa_bulk_workspace_extra_bytes": 234356736, "gpu_executed": False,
        "payload_bytes_read": 0, "normal_sources_modified": False,
        "wide_model_parity_qualified": False, "files": records})
    audit = {"schema": "splash-prefill4k-wide-prefix-cpu-audit-v1", "source_files_verified": len(records),
        "changed_paths": changed, "all_bulk_kernel_and_helper_bytes_unchanged": all(not r["wide_prefix_changed"] for r in records if r["path"].startswith(PRIVATE)),
        "original_trained_mtp_source_unchanged": all(not r["wide_prefix_changed"] for r in records if "MTP" in r["path"]),
        "int8_loader_store_workers_unchanged": all(not r["wide_prefix_changed"] for r in records if any(n in r["path"] for n in ("FlashWeights", "FlashInt8ExpertStore", "FlashWorker"))),
        "bulk_workspace_extra_bytes_unchanged": 234356736,
        "2048_call_same_bulk_descriptor_and_bindings_extents": True,
        "wider_dense_role_policy_needs_actual_model_parity": True,
        "gpu_executed": False, "payload_bytes_read": 0, **checks}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prefill4k-wide-prefix-",dir=output.parent) as temporary:
        staged = Path(temporary)/"output"
        for relative,data in files.items():
            destination = staged/"source"/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
        (staged/"overlay-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (staged/"wide-prefix-cpu-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
        staged.rename(output)
    return {"prepared":str(output),**audit}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=ROOT/"build/prefill4k-qsa-bulk-allrows-full512")
    parser.add_argument("--output",type=Path,default=ROOT/"build/prefill4k-wide-prefix-full512-v1")
    args=parser.parse_args()
    print(json.dumps(generate(args.base,args.output)))


if __name__ == "__main__":
    main()
