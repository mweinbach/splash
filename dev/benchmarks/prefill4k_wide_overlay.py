#!/usr/bin/env python3
"""Prepare a private, bounded singleton-prefill overlay; never execute a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def replacement(text: str, before: str, after: str, expected: int = 1) -> str:
    actual = text.count(before)
    if actual != expected:
        raise RuntimeError(f"Overlay source drift: expected {expected} occurrences, got {actual}: {before!r}")
    return text.replace(before, after)


def changed_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashGDN.hpp":
        return replacement(text, "kFlashGDNMaximumRows = 2048", "kFlashGDNMaximumRows = 8192")
    if relative == "runtime/flash/FlashGDNBatchILP.cpp":
        # The independent multi-lane candidate retains its qualified limits.
        text = replacement(text, "rows <= kFlashGDNMaximumRows", "rows <= 2048")
        return replacement(text, "lane.rows > kFlashGDNMaximumRows", "lane.rows > 2048")
    if relative == "runtime/flash/FlashForward.cpp":
        text = replacement(text, "maximumRows > 2048", "maximumRows > kFlashGDNMaximumRows", expected=2)
        text = replacement(text, "maximumRows > kFlashGDNMaximumRows || maximumRows > kFlashGDNMaximumRows", "maximumRows > kFlashGDNMaximumRows")
        text = replacement(text, "Flash forward maximum rows must be 1..2048", "PRIVATE Flash forward maximum rows must be 1..8192")
        return replacement(text, "return std::string(flashAffineSemantics()) +", 'return std::string(flashAffineSemantics()) +\n      ";private-singleton-prefill-arena-max8192-causal-qsa128-v1" +')
    if relative == "runtime/flash/FlashPLESSD.cpp":
        text = replacement(text, "rows > 2048", "rows > (lanes == 1 ? 8192u : 2048u)")
        return replacement(text, "Flash PLE SSD staging requires lanes1..4 and rows1..2048", "PRIVATE Flash PLE SSD staging requires singleton rows1..8192 or lanes2..4 rows1..2048")
    if relative == "runtime/flash/FlashWorker.mm":
        text = replacement(text, "constexpr std::array<uint32_t, 7> choices{32, 64, 128, 256, 512, 1024, 2048};", "constexpr std::array<uint32_t, 9> choices{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};")
        text = replacement(text, "SPLASH_FLASH_PREFILL_ROWS must be 32,64,128,256,512,1024 or 2048", "PRIVATE SPLASH_FLASH_PREFILL_ROWS must be 32,64,128,256,512,1024,2048,4096 or 8192")
        text = replacement(text, "for (const uint32_t rows : {32u, 64u, 128u, 256u, 512u, 1024u, 2048u}) {", "for (const uint32_t rows : {32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u}) {")
        text = replacement(text, "511u, 2047u, 2048u, 2049u, 4097u}) {", "511u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 8191u, 8192u, 8193u}) {")
        text = replacement(text, '\"16\", \"96\", \"129\", \"4096\", \"128x\"', '\"16\", \"96\", \"129\", \"4095\", \"8193\", \"16384\", \"128x\"')
        extra = '''  {
    const auto p2 = FlashForward::workspacePlannedBytes(16384, 2048, 4);
    const auto p4 = FlashForward::workspacePlannedBytes(16384, 4096, 4);
    const auto p8 = FlashForward::workspacePlannedBytes(16384, 8192, 4);
    require(p2 < p4 && p4 < p8, "private wide admission must scale actual row scratch");
    for (uint32_t rows : {0u, 8193u, 16384u, UINT32_MAX}) {
      bool rejected = false;
      try { (void)FlashForward::workspacePlannedBytes(16384, rows, 4); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "private wide admission must reject unsupported row extents before allocation");
    }
    require(FlashPLESSD::plannedBytes(1, 8192) == 4 * FlashPLESSD::plannedBytes(1, 2048),
        "private PLE singleton staging must scale true canonical IDs and rows");
    for (auto geometry : {std::pair{1u, 8193u}, std::pair{2u, 2049u}, std::pair{4u, 8192u}}) {
      bool rejected = false;
      try { (void)FlashPLESSD::plannedBytes(geometry.first, geometry.second); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "private PLE staging must preserve batch limits and reject overwide singleton");
    }
    for (uint32_t rows : {2048u, 4096u, 8192u}) {
      metal::CommandGraph graph;
      try { addGDN(graph, {}, {}, {}, rows); }
      catch (const std::invalid_argument &error) {
        require(std::string(error.what()).find("row/lane geometry") == std::string::npos,
            "private GDN positive row range must reach weight validation without GPU work");
      }
    }
    for (uint32_t rows : {0u, 8193u, UINT32_MAX}) {
      metal::CommandGraph graph;
      bool rejected = false;
      try { addGDN(graph, {}, {}, {}, rows); }
      catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("row/lane geometry") != std::string::npos;
      }
      require(rejected, "private GDN overwide row extent must reject before buffer work");
    }
  }
'''
        text = replacement(text, '#include "flash/FlashPLESSDStore.hpp"', '#include "flash/FlashPLESSDStore.hpp"\n#include "flash/FlashPLESSD.hpp"')
        text = replacement(text, "#include <filesystem>", "#include <filesystem>\n#include <fstream>")
        admission = '''      if (const char *reportPath = std::getenv("SPLASH_FLASH_PRIVATE_ADMISSION_REPORT")) {
        if (!*reportPath || std::filesystem::exists(reportPath))
          throw std::invalid_argument("private admission report requires a fresh nonempty path");
        governor.setPressure(pressure.value());
        const auto snapshot = governor.snapshot();
        std::ofstream report(reportPath);
        if (!report) throw std::runtime_error("cannot create private startup admission report");
        const bool engineFits = snapshot.observedResidentBytes <= snapshot.limitBytes &&
            snapshot.reservedBytes <= snapshot.limitBytes - snapshot.observedResidentBytes &&
            plannedTrunk <= snapshot.limitBytes - snapshot.observedResidentBytes - snapshot.reservedBytes;
        const bool hostFits = snapshot.hostMeasurementValid && snapshot.growthAllowed &&
            snapshot.pressure == engine::MemoryPressure::Normal &&
            snapshot.hostHeadroomBytes >= plannedTrunk &&
            snapshot.hostHeadroomBytes - plannedTrunk >= engine::kHostWarningMarginBytes;
        report << "{\\\"schema\\\":1,\\\"stage\\\":\\\"after original weight mapping; before trunk/cache construction\\\","
            << "\\\"capacity\\\":" << capacity << ",\\\"maximum_prefill_rows\\\":" << prefillRows
            << ",\\\"original_weight_bytes\\\":" << weights.actualAllocatedBytes()
            << ",\\\"planned_trunk_bytes\\\":" << plannedTrunk
            << ",\\\"engine_limit_bytes\\\":" << snapshot.limitBytes
            << ",\\\"observed_resident_bytes\\\":" << snapshot.observedResidentBytes
            << ",\\\"reserved_bytes\\\":" << snapshot.reservedBytes
            << ",\\\"engine_headroom_bytes\\\":" << snapshot.headroomBytes
            << ",\\\"host_measurement_valid\\\":" << (snapshot.hostMeasurementValid ? "true" : "false")
            << ",\\\"host_available_bytes\\\":" << snapshot.hostAvailableBytes
            << ",\\\"host_reserve_bytes\\\":" << snapshot.hostReserveBytes
            << ",\\\"host_headroom_bytes\\\":" << snapshot.hostHeadroomBytes
            << ",\\\"host_warning_margin_bytes\\\":" << engine::kHostWarningMarginBytes
            << ",\\\"effective_pressure\\\":" << json::quote(pressureName(snapshot.pressure))
            << ",\\\"stage_engine_fits\\\":" << (engineFits ? "true" : "false")
            << ",\\\"stage_host_fits\\\":" << (hostFits ? "true" : "false")
            << ",\\\"admission_scope\\\":\\\"sampled necessary stage bounds; subsequent tryReserve refreshes and remains authoritative\\\"}\\n";
        report.flush();
        if (!report) throw std::runtime_error("cannot finish private startup admission report");
      }
'''
        text = replacement(text, "      auto trunkReservation = governor.tryReserve(plannedTrunk, &trunkFailure);", admission + "      auto trunkReservation = governor.tryReserve(plannedTrunk, &trunkFailure);")
        text = replacement(text, '"prefill_geometry","sliced_mtp_priming"', '"prefill_geometry","private_wide_row_ranges","private_wide_workspace_admission","private_wide_gdn_host_geometry","private_wide_ple_staging","sliced_mtp_priming"')
        return replacement(text, "  require(jointDepth(std::array<uint32_t, 4>{128, 4, 9, 7})", extra + "  require(jointDepth(std::array<uint32_t, 4>{128, 4, 9, 7})")
    if relative.startswith("runtime/metal/kernels/shared/flash_gdn"):
        # Replace the parameter guard only: 2048 is also a genuine Q/K width.
        return replacement(text, "p.rows <= 2048", "p.rows <= 8192")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("build/prefill4k-wide"))
    parser.add_argument("--fullcache", action="store_true", help="Compose the private Full512 loader/shader transforms")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    # Keep this helper incapable of overwriting production sources.
    output = output.resolve()
    if root / "build" not in output.parents:
        raise ValueError("Private overlay output must be below this repository's build directory")
    shader_paths = [f"runtime/metal/kernels/shared/{name}.metal" for name in ("flash_gdn", "flash_gdn_fused", "flash_gdn_staged")]
    paths = [str(path.relative_to(root)) for path in sorted((root / "runtime/flash").glob("*")) if path.is_file()]
    paths += shader_paths
    fullcache = None
    if args.fullcache:
        spec = importlib.util.spec_from_file_location("prefill4k_fullcache_transform", root / "dev/benchmarks/prefill4k_fullcache_overlay.py")
        if not spec or not spec.loader:
            raise RuntimeError("Cannot load the existing private Full512 source transform")
        fullcache = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fullcache)
        paths += [
            "runtime/metal/kernels/shared/flash_int8_expert_store.metal",
            "dev/benchmarks/flash_int8_expert_store_metadata_cpu.mm",
            "dev/benchmarks/flash_int8_expert_store_oracle.mm",
            "dev/benchmarks/flash_expert_int8_bucket_reference.hpp",
        ]
    manifest = {
        "schema": 1,
        "route": "private-singleton-prefill-arena-max8192-causal-qsa128-v1",
        "normal_sources_modified": False,
        "maximum_singleton_rows": 8192,
        "maximum_batch_rows_per_lane": 2048,
        "maximum_causal_qsa_rows": 128,
        "maximum_verification_rows": 16,
        "fullcache512_supported": args.fullcache,
        "arithmetic_change": args.fullcache,
        "fullcache_planned_payload_and_ranks_bytes": 121174228992 if args.fullcache else None,
        "files": [],
    }
    for relative in paths:
        original = (root / relative).read_bytes()
        text = transform(relative, original.decode())
        if fullcache:
            text = fullcache.transform(relative, text)
        modified = text.encode()
        changed_bytes(output / "source" / relative, modified)
        manifest["files"].append({
            "path": relative,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "overlay_sha256": hashlib.sha256(modified).hexdigest(),
            "patched": original != modified,
        })
    changed_bytes(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    patched = sum(item["patched"] for item in manifest["files"])
    print(f"Prepared {len(paths)} private source files ({patched} patched) in {output}")


if __name__ == "__main__":
    main()
