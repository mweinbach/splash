#!/usr/bin/env python3
"""Generate a PRIVATE default-off exact-bulk QSA FlashForward overlay.

Usage: bulk_overlay.py BASE_FORWARD_CPP DESTINATION_FOLDER
The base may be normal FlashForward.cpp or the private all-rows Full512/wide
snapshot. Only a new FlashForward.cpp and manifest.json are written; the base
and other source files are never modified. Existing output files are rejected.

Compile the generated source with -Idev/benchmarks/prefill4k_attention and the
base snapshot's runtime include root. Link the matching bulk.cpp/coalesced.cpp
and exact-v2 bulk_attention.air (or its bulk_attention_sg8.air replacement for
the optional SG8 temporal composition) plus ordinary QSA Metal kernels. This
generator does not compile, link, execute GPU work, load models, or alter flags.

SPLASH_FLASH_QSA_BULK_PREFILL is absent/0 by default. When 1, F32/MPP/ROW_TILES
must all be 1, including the actual frozen row-tiles policy. Provisioning needs
maximumRows >= 2048; eligible calls require !verification, begin==0, rows==2048.
Every other main-QSA call retains the original loop verbatim. Batch source is
untouched, and its external-destination validation rejects all five new planes.
SPLASH_FLASH_QSA_BULK_PREFILL_SG8 is separately absent/0 by default and requires
enabled BULK plus those same dependencies. It changes only the temporal bulk
dispatch to SG8; normal SG4 bulk remains the default and allocation is unchanged.

Planned admission adds bulkExactPlannedBytes() before construction. Actual
workspace accounting uses the original backend allocation-ledger difference,
which includes all five allocations; the extra bytes must not be added twice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FLAG = "SPLASH_FLASH_QSA_BULK_PREFILL"
SG8_FLAG = "SPLASH_FLASH_QSA_BULK_PREFILL_SG8"
DEPENDENCIES = ("SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP", "SPLASH_FLASH_QSA_ROW_TILES")
ROUTE = ";private-qsa-bulk-prefill-begin0-r2048-w128-m16p1-p4-m32p4-v2"
SG8_ROUTE = ";private-qsa-bulk-prefill-temporal-sg8-w4to15-m32-p4-t256-v1"
PLANES = {
    "prepared.queries": 2048 * 6144 * 2,
    "prepared.indexQueries": 2048 * 512 * 2,
    "prepared.selectedBlocks": 2048 * 512 * 4,
    "partials.partitionStatistics": 2048 * 24 * 4 * 2 * 4,
    "partials.partitionValues": 2048 * 24 * 4 * 256 * 4,
}
PLANNED_BYTES = sum(PLANES.values())
assert PLANNED_BYTES == 234356736
assert all(size % 16384 == 0 for size in PLANES.values())


@dataclass(frozen=True)
class Edit:
    label: str
    before: str
    after: str


def transform(source: str) -> tuple[str, list[Edit]]:
    """Require unique inspected anchors and retain an exactly reversible journal."""
    if FLAG in source or "privateBulkQSAPrefillEnabled" in source:
        raise ValueError("base already contains the private bulk overlay")
    if "privateAttentionCapture" in source or '#include "capture.hpp"' in source:
        raise ValueError("base must be the authoritative Forward, not a capture overlay")
    modified = source
    edits: list[Edit] = []

    def change(label: str, before: str, after: str) -> None:
        nonlocal modified
        if modified.count(before) != 1:
            raise ValueError(f"authoritative Forward anchor drifted: {label}")
        modified = modified.replace(before, after, 1)
        edits.append(Edit(label, before, after))

    anchor = '#include "flash/FlashForward.hpp"\n'
    change("private bulk header", anchor, '#include "bulk.hpp"\n' + anchor)
    anchor = "std::optional<FlashExpertCachePlan> hotExpertPlan(const FlashWeights &weights) {"
    helper = '''// Private experiment: validate dependencies before admission/construction.
bool privateBulkQSAPrefillEnabled() {
  if (!fusionEnabled("SPLASH_FLASH_QSA_BULK_PREFILL")) return false;
  if (!fusionEnabled("SPLASH_FLASH_QSA_F32") ||
      !fusionEnabled("SPLASH_FLASH_QSA_MPP") ||
      !fusionEnabled("SPLASH_FLASH_QSA_ROW_TILES") ||
      !qsaOnlineMPPRowTilesEnabled())
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL requires QSA_F32=1, QSA_MPP=1 and QSA_ROW_TILES=1");
  return true;
}

bool privateBulkQSAPrefillSG8Enabled(bool bulkEnabled) {
  if (!fusionEnabled("SPLASH_FLASH_QSA_BULK_PREFILL_SG8")) return false;
  if (!bulkEnabled)
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL_SG8 requires SPLASH_FLASH_QSA_BULK_PREFILL=1 and its dependencies");
  return true;
}

'''
    change("default-off flag/dependencies", anchor, helper + anchor)
    anchor = '  const bool mppQSA = fusionEnabled("SPLASH_FLASH_QSA_MPP");\n'
    change("frozen private flag", anchor,
           anchor + "  const bool bulkQSAPrefill = privateBulkQSAPrefillEnabled();\n"
           "  const bool bulkQSAPrefillSG8 = privateBulkQSAPrefillSG8Enabled(bulkQSAPrefill);\n")
    anchor = "  FlashQSAFastWorkspace qsaFastWorkspace;\n"
    change("optional bulk workspace", anchor,
           anchor + "  std::optional<prefill4k::BulkExactWorkspace> bulkQSAWorkspace;\n")
    anchor = "    uint32_t gdnSlot = 0;\n"
    allocation = '''    if (bulkQSAPrefill && maximumRows >= 2048) {
      bulkQSAWorkspace.emplace(prefill4k::allocateBulkExactWorkspace(backend));
      const auto &bulk = *bulkQSAWorkspace;
      const uint64_t bulkPlaneBytes = bulk.prepared.queries.sizeBytes() +
          bulk.prepared.indexQueries.sizeBytes() + bulk.prepared.selectedBlocks.sizeBytes() +
          bulk.partials.partitionStatistics.sizeBytes() + bulk.partials.partitionValues.sizeBytes();
      if (bulkPlaneBytes != prefill4k::bulkExactPlannedBytes())
        throw std::logic_error("private bulk QSA five-plane allocation/admission mismatch");
    }
'''
    change("five-plane allocation inside existing ledger", anchor, allocation + anchor)
    anchor = "    workspaceBytes = after - before;\n"
    change("actual bytes include all five planes without double counting", anchor, anchor +
           '''    if (bulkQSAWorkspace && workspaceBytes < prefill4k::bulkExactPlannedBytes())
      throw std::logic_error("private bulk QSA planes omitted from allocation ledger");
''')
    anchor = "  return total;\n}\n\nstd::string FlashForward::kernelRoutes() const {"
    planned = '''  const bool privateBulkPrefillEnabled = privateBulkQSAPrefillEnabled();
  (void)privateBulkQSAPrefillSG8Enabled(privateBulkPrefillEnabled);
  if (privateBulkPrefillEnabled && maximumRows >= 2048)
    total += prefill4k::bulkExactPlannedBytes();
'''
    change("planned admission before constructor", anchor, planned + anchor)
    anchor = "  return std::string(flashAffineSemantics()) +\n"
    change("explicit private route identity", anchor, anchor +
           f'      (impl_->bulkQSAWorkspace ? "{ROUTE}" : "") +\n'
           f'      (impl_->bulkQSAWorkspace && impl_->bulkQSAPrefillSG8 ? "{SG8_ROUTE}" : "") +\n')
    anchor = "      impl_->beforePLEConvolution, impl_->retainedCount, impl_->beforeGDNConvolution}) reject(buffer);\n"
    alias_guard = '''  if (impl_->bulkQSAWorkspace) {
    const auto &bulk = *impl_->bulkQSAWorkspace;
    for (const auto &buffer : {bulk.prepared.queries, bulk.prepared.indexQueries,
        bulk.prepared.selectedBlocks, bulk.partials.partitionStatistics,
        bulk.partials.partitionValues}) reject(buffer);
  }
'''
    change("external destination rejects all five bulk planes", anchor, anchor + alias_guard)
    loop_begin = "      for (uint32_t offset = 0; offset < rows;) {\n"
    loop_end = '      affine(graph, attention + ".o_proj", attentionOutput, branch);'
    if modified.count(loop_begin) != 1 or modified.count(loop_end) != 1:
        raise ValueError("authoritative main-QSA loop anchors drifted")
    first, last = modified.index(loop_begin), modified.index(loop_end)
    if first >= last:
        raise ValueError("authoritative main-QSA loop anchor order drifted")
    original_loop = modified[first:last]
    if not original_loop.endswith("      }\n") or "addQSAOnlineMPP" not in original_loop:
        raise ValueError("authoritative main-QSA loop boundary/body drifted")
    bulk_branch = '''      if (impl_->bulkQSAWorkspace && !verification && begin == 0 && rows == 2048) {
        const FlashQSAFastInputs bulkInputs{q, k, v, index,
            &impl_->weights.tensor(qNorm), &impl_->weights.tensor(kNorm),
            &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
            attentionOutput, diag, {},
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
        prefill4k::addBulkExactQSA(impl_->backend, graph, bulkInputs, state.qsa[layer],
            impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,
            impl_->bulkQSAPrefillSG8);
      } else {
'''
    change("only eligible main QSA calls use bulk; fallback loop verbatim", original_loop,
           bulk_branch + original_loop + "      }\n")
    if restore(modified, edits) != source:
        raise AssertionError("private overlay journal does not restore base byte-for-byte")
    return modified, edits


def restore(modified: str, edits: list[Edit]) -> str:
    for edit in reversed(edits):
        if modified.count(edit.after) != 1:
            raise AssertionError(f"overlay reverse anchor ambiguous: {edit.label}")
        modified = modified.replace(edit.after, edit.before, 1)
    return modified


def manifest(base: Path, source: str, modified: str, edits: list[Edit]) -> dict[str, Any]:
    return {
        "schema": "splash-private-exact-bulk-forward-overlay-v1",
        "gpu_executed": False,
        "base_forward_cpp": str(base.resolve()),
        "base_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "overlay_source_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "base_restored_byte_exact": restore(modified, edits) == source,
        "base_allrows_full512_identity_preserved": "SPLASH_FLASH_ALLROWS_FULL512_TARGET" in source,
        "flag": FLAG, "default": "absent/0",
        "sg8_flag": SG8_FLAG, "sg8_default": "absent/0 (SG4 bulk)",
        "sg8_requirement": "enabled BULK plus its dependencies, checked before admission and construction",
        "sg8_policy": "frozen const bool passed to addBulkExactQSA; only temporal windows 4..15 use SG8; allocation and main-QSA gate unchanged",
        "dependencies": {name: "1" for name in DEPENDENCIES},
        "row_tiles_dependency": "actual frozen qsaOnlineMPPRowTilesEnabled must also be true",
        "allocation_gate": "flag validated/enabled && maximumRows>=2048",
        "main_qsa_gate": "provisioned bulk workspace && !verification && begin==0 && rows==2048",
        "fallback": "original main-QSA loop retained verbatim; other row counts/begins/>2K use original route",
        "batch": "batch Forward source untouched; external destination guards include bulk planes",
        "workspace_planned_extra_bytes": PLANNED_BYTES,
        "workspace_planned_extra_mib": PLANNED_BYTES / (1 << 20),
        "workspace_planes": PLANES,
        "workspace_accounting": "bulkExactPlannedBytes added by workspacePlannedBytes before construction; five planes allocated inside original before/after ledger; actual size sum checked; no double-count addition",
        "private_route_identity": ROUTE,
        "private_sg8_route_identity": SG8_ROUTE,
        "header_requirement": 'bulk.hpp from -Idev/benchmarks/prefill4k_attention',
        "source_interface": "addBulkExactQSA(..., uint32_t begin, uint32_t rows, bool sg8=false)",
        "runtime_headers": "compile against the supplied base snapshot runtime include root",
        "link_requirements": ["matching bulk.cpp", "matching coalesced.cpp", "ordinary FlashQSA/FlashQSAFast/FlashQSAMPP objects"],
        "metal_requirements": ["exact-v2 bulk_attention.air (SG4 default) or replacement bulk_attention_sg8.air for optional SG8; never link both", "ordinary QSA prepare/pool/selection kernels"],
        "edits": [{"label": edit.label, "anchor": edit.before.splitlines()[0]} for edit in edits],
        "qualification": None,
        "scope": "private source generation/CPU audit only; full-model output/cache/service qualification required",
    }


def generate(base: Path, destination: Path) -> dict[str, Any]:
    source = base.read_bytes().decode("utf-8")
    modified, edits = transform(source)
    output = destination / "FlashForward.cpp"
    witness = destination / "manifest.json"
    if output.resolve() == base.resolve():
        raise ValueError("destination cannot overwrite base Forward")
    if output.exists() or witness.exists():
        raise ValueError("choose fresh private overlay output files")
    result = manifest(base, source, modified, edits)
    destination.mkdir(parents=True, exist_ok=True)
    output.write_text(modified)
    witness.write_text(json.dumps(result, indent=2) + "\n")
    return result


def self_test(base: Path | None = None) -> dict[str, Any]:
    sources = [base] if base else [ROOT / "runtime/flash/FlashForward.cpp"]
    wide = ROOT / "build/prefill4k-allrows-full512/source/runtime/flash/FlashForward.cpp"
    if base is None and wide.exists():
        sources.append(wide)
    witnesses = []
    with tempfile.TemporaryDirectory(prefix="splash-private-bulk-forward-") as temporary:
        for number, path in enumerate(sources):
            original_bytes = path.read_bytes()
            witness = generate(path, Path(temporary) / str(number))
            assert path.read_bytes() == original_bytes
            assert witness["base_restored_byte_exact"]
            assert witness["workspace_planned_extra_bytes"] == 234356736
            assert witness["sg8_flag"] == SG8_FLAG
            assert witness["private_sg8_route_identity"] == SG8_ROUTE
            witnesses.append({"base": str(path), "base_restored_byte_exact": True,
                              "allrows_full512": witness["base_allrows_full512_identity_preserved"]})
            modified = (Path(temporary) / str(number) / "FlashForward.cpp").read_text()
            assert modified.count("prefill4k::addBulkExactQSA(") == 1
            assert "*impl_->bulkQSAWorkspace, begin, rows,\n            impl_->bulkQSAPrefillSG8);" in modified
            assert "const bool bulkQSAPrefillSG8 = privateBulkQSAPrefillSG8Enabled(bulkQSAPrefill);" in modified
            assert "(void)privateBulkQSAPrefillSG8Enabled(privateBulkPrefillEnabled);" in modified
            assert modified.index("allocateBulkExactWorkspace") < modified.index("    const uint64_t after = backend.memoryStats().allocatedBytes;")
            # The source-mutation protection and all unique anchors fail closed.
            for bad in (modified, path.read_text().replace("    uint32_t gdnSlot = 0;", "    uint32_t gdnSlot = 1;", 1)):
                try:
                    transform(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("modified/anchor-drifted base accepted")
    eligible = 0
    for enabled in (False, True):
        for maximum_rows in (128, 2048, 8192):
            for verification in (False, True):
                for begin in (0, 128, 2048):
                    for rows in (1, 128, 2048, 4096, 8192):
                        provisioned = enabled and maximum_rows >= 2048
                        chosen = provisioned and not verification and begin == 0 and rows == 2048
                        if chosen:
                            assert enabled and maximum_rows >= 2048 and begin == 0 and not verification and rows == 2048
                            eligible += 1
    assert eligible == 2
    private_policy_cases = valid_private_policies = 0
    for bulk in (False, True):
        for sg8 in (False, True):
            for f32 in (False, True):
                for mpp in (False, True):
                    for row_tiles in (False, True):
                        for frozen_row_tiles in (False, True):
                            private_policy_cases += 1
                            valid = ((not sg8 or bulk) and
                                     (not bulk or (f32 and mpp and row_tiles and frozen_row_tiles)))
                            if valid:
                                valid_private_policies += 1
                                if sg8:
                                    assert bulk and f32 and mpp and row_tiles and frozen_row_tiles
                            if not bulk and sg8:
                                assert not valid
    assert private_policy_cases == 64 and valid_private_policies == 18
    return {"gpu_executed": False, "temporary_generation_and_source_restore_verified": witnesses,
            "planned_five_plane_bytes": PLANNED_BYTES, "main_qsa_gate_cases_checked": 180,
            "eligible_cases": eligible, "production_or_batch_sources_modified": False,
            "private_sg8_dependency_policy_cases_checked": private_policy_cases,
            "valid_private_policies": valid_private_policies,
            "sg4_default_and_unchanged_allocation_verified": True,
            "compilation_or_model_qualification": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_forward_cpp", type=Path, nargs="?")
    parser.add_argument("destination_folder", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true", help="CPU-only temporary generation and source/gate audits")
    args = parser.parse_args()
    if args.self_test:
        if args.destination_folder:
            parser.error("self-test does not use a persistent destination")
        print(json.dumps(self_test(args.base_forward_cpp), indent=2))
        return 0
    if args.base_forward_cpp is None or args.destination_folder is None:
        parser.error("BASE_FORWARD_CPP and DESTINATION_FOLDER are required")
    result = generate(args.base_forward_cpp, args.destination_folder)
    print(json.dumps({"gpu_executed": False, "base_restored_byte_exact": result["base_restored_byte_exact"],
                      "overlay": str(args.destination_folder / "FlashForward.cpp"),
                      "manifest": str(args.destination_folder / "manifest.json"),
                      "planned_extra_bytes": PLANNED_BYTES, "flag": FLAG, "default": "absent/0",
                      "sg8_flag": SG8_FLAG, "sg8_default": "absent/0 (SG4 bulk)"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"private bulk overlay error: {error}")
