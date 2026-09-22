#!/usr/bin/env python3
"""Compose unchanged direct gathered MPP on sealed Full512 + exact SG8 bulk.

Only private source snapshots are written. No device, model metadata or model
payload is touched. The maximum gathered physical row count is frozen by Store
construction and included in its enabled numerical derivative.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dev/benchmarks"))
sys.dont_write_bytecode = True
from prefill4k_allrows_overlay import module

MAX_FLAG = "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS"
ROUTE_POLICY = "private-gathered-mpp-physical-row-cap-frozen1or2or4or8or16-default4-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    if text.count(before) != count:
        raise ValueError(f"Capped gathered MPP source drift: {before!r}")
    return text.replace(before, after)


def cap_transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashGatheredMPP.hpp":
        return replace(text, "using Geometry = gathered_mpp_view::Geometry;", r'''inline constexpr const char *kMaximumRowsFlag = "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS";
inline constexpr std::string_view kRowCapPolicy =
    "private-gathered-mpp-physical-row-cap-frozen1or2or4or8or16-default4-v1";
inline uint32_t requestedMaximumRows() {
  const char *raw = std::getenv(kMaximumRowsFlag);
  if (!raw) return 4;
  const std::string_view value(raw);
  if (value == "1") return 1;
  if (value == "2") return 2;
  if (value == "4") return 4;
  if (value == "8") return 8;
  if (value == "16") return 16;
  throw std::invalid_argument("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS must be exactly1,2,4,8 or16");
}
using Geometry = gathered_mpp_view::Geometry;''')
    if relative == "runtime/flash/FlashInt8ExpertStore.hpp":
        return replace(text, "  [[nodiscard]] bool gatheredMPPEnabled() const;",
                       "  [[nodiscard]] bool gatheredMPPEnabled() const;\n  [[nodiscard]] uint32_t gatheredMPPMaximumRows() const;")
    if relative == "runtime/flash/FlashInt8ExpertStore.mm":
        text = replace(text, "  const bool gatheredMPP = gathered_mpp::requested();",
                       "  const bool gatheredMPP = gathered_mpp::requested();\n  const uint32_t gatheredMPPMaximumRows = gathered_mpp::requestedMaximumRows();")
        text = replace(text, '      derivative += std::string("small_row_policy=") + std::string(gathered_mpp::kPolicy) + "\\n";',
                       '      derivative += std::string("small_row_policy=") + std::string(gathered_mpp::kPolicy) + "\\nsmall_row_cap_policy=" +\n          std::string(gathered_mpp::kRowCapPolicy) + "\\nsmall_row_max_physical_rows=" +\n          std::to_string(gatheredMPPMaximumRows) + "\\n";')
        text = replace(text, "  return impl_->gatheredMPP;\n}", """  if (gathered_mpp::requestedMaximumRows() != impl_->gatheredMPPMaximumRows)
    fail("private gathered I8 MPP maximum rows changed after Store construction");
  return impl_->gatheredMPP;
}
uint32_t FlashInt8ExpertStore::gatheredMPPMaximumRows() const {
  (void)gatheredMPPEnabled();
  return impl_->gatheredMPPMaximumRows;
}""")
        return replace(text, "  const auto g = gathered_mpp::geometry(rows, selections);",
                       "  if (rows > gatheredMPPMaximumRows()) fail(\"private gathered MPP rows exceed frozen route cap\");\n  const auto g = gathered_mpp::geometry(rows, selections);", count=2)
    if relative in ("runtime/flash/FlashForward.cpp", "runtime/flash/FlashBatchForward.cpp", "runtime/flash/FlashBatchVerify.cpp"):
        # The cap getter must be after the pointer guard, preserving short circuit.
        rows = "rows" if relative.endswith("FlashForward.cpp") else "lanes" if relative.endswith("FlashBatchForward.cpp") else "flattened"
        text = replace(text,
                       f"impl_->allRowsInt8Target && {rows} <= 16 &&\n        impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredMPPEnabled();",
                       f"impl_->allRowsInt8Target && impl_->int8ExpertStore &&\n        impl_->int8ExpertStore->gatheredMPPEnabled() && {rows} <= impl_->int8ExpertStore->gatheredMPPMaximumRows();")
        return text
    if relative == "runtime/flash/FlashBatchPrefill.cpp":
        return replace(text, "impl_->allRowsInt8Target && flat <= 16 && qmvStore &&\n        qmvStore->gatheredMPPEnabled();",
                       "impl_->allRowsInt8Target && qmvStore &&\n        qmvStore->gatheredMPPEnabled() && flat <= qmvStore->gatheredMPPMaximumRows();")
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, "      (void)gathered_mpp::requested();",
                       "      (void)gathered_mpp::requestedMaximumRows();\n      (void)gathered_mpp::requested();")
        marker = '      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"'
        return replace(text, marker, marker + '''
      << R"(,"target_gathered_mpp_enabled":)" << (persistedExperts && persistedExperts->gatheredMPPEnabled() ? "true" : "false")
      << R"(,"target_gathered_mpp_max_physical_rows":)" << (persistedExperts ? persistedExperts->gatheredMPPMaximumRows() : 4)
      << R"(,"target_gathered_mpp_row_cap_policy":)" << json::quote(std::string(gathered_mpp::kRowCapPolicy))
      << R"(,"target_gathered_mpp_numerical_policy":)" << json::quote(std::string(gathered_mpp::kPolicy))''')
    return text


def generate(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if ROOT / "build" not in output.parents or output == base or output.exists():
        raise ValueError("Choose a fresh, separate output beneath repository build")
    manifest_bytes = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(manifest_bytes)
    if not parent.get("qsa_bulk_composed") or not parent.get("qsa_bulk_sg8_available"):
        raise ValueError("Use the sealed exact bulk SG8 source snapshot")
    if parent.get("gathered_qmv_composed") or parent.get("gathered_mpp_composed"):
        raise ValueError("Use original MPP Full512 control as input")
    transforms = [module(name).transform for name in ("allrows_gathered_mpp", "allrows_gathered_mpp_routes")]
    records, content = [], {}
    for record in parent["files"]:
        relative = record["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe source-manifest path")
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed source drift: {relative}")
        text = original.decode()
        for transform in transforms:
            text = transform(relative, text)
        text = cap_transform(relative, text)
        modified = text.encode()
        content[relative] = modified
        records.append({**record, "base_overlay_sha256": sha(original),
                        "combined_changed": original != modified, "overlay_sha256": sha(modified)})
    for relative, text in module("allrows_gathered_mpp").extra_files().items():
        data = cap_transform(relative, text).encode()
        content[relative] = data
        records.append({"path": relative, "new_private_file": True,
                        "patched": True, "overlay_sha256": sha(data)})
    # These independent invariants prevent accidental arithmetic or lifecycle edits.
    for relative in ("runtime/flash/FlashMTP.cpp", "runtime/flash/FlashBatchMTPForward.cpp",
                     "runtime/metal/kernels/shared/flash_int8_expert_store.metal"):
        if content[relative] != (base / "source" / relative).read_bytes():
            raise AssertionError(f"Unchanged arithmetic changed: {relative}")
    source_store = (base / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    old_start = source_store.index("void FlashInt8ExpertStore::addGateUp(")
    old_end = source_store.index("} // namespace splash::flash", old_start)
    if source_store[old_start:old_end] not in content["runtime/flash/FlashInt8ExpertStore.mm"].decode():
        raise AssertionError("Original bucketed MPP Store methods changed")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-allrows-full512-exact-bulk-sg8-capped-gathered-mpp-sep21-v1",
                     "gathered_mpp_composed": True, "gathered_mpp_base_build": str(base),
                     "gathered_mpp_base_manifest_sha256": sha(manifest_bytes),
                     "gathered_mpp_cap_overlay_sha256": sha(Path(__file__).read_bytes()),
                     "gathered_mpp_row_cap_flag": MAX_FLAG,
                     "gathered_mpp_row_cap_policy": ROUTE_POLICY,
                     "gathered_mpp_default_maximum_rows": 4,
                     "gathered_mpp_shader_unchanged": True, "old_mpp_methods_unchanged": True,
                     "trained_mtp_source_changed": False,
                     "gpu_executed": False, "payload_bytes_read": 0, "files": records})
    manifest["transform_sha256"].update({name: sha((ROOT / "dev/benchmarks" / f"prefill4k_{name}.py").read_bytes())
                                          for name in ("allrows_gathered_mpp", "allrows_gathered_mpp_routes")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="capped-gathered-sep21-", dir=output.parent) as temp:
        staged = Path(temp) / "output"
        for relative, data in content.items():
            path = staged / "source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return {"prepared": str(output), "files": len(records), "source_hashes_verified": True,
            "gpu_executed": False, "payload_bytes_read": 0, "default_maximum_rows": 4,
            "mtp_unchanged": True, "old_mpp_methods_unchanged": True, "gathered_shader_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-allrows-full512")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1")
    args = parser.parse_args()
    print(json.dumps(generate(args.base, args.output)))
