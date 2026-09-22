#!/usr/bin/env python3
"""Freeze prefill-only explicit-FMA GDN over the sealed pointwise worker."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/gdn_chunk_sep21")
QUALIFIED_KERNEL_SHA256 = "166381991f5444585e7db3634f41432322efdd66bf08a8c402d5fed49e58c699"
CHANGED_NAMES = {"FlashForward", "FlashWorker", "FlashGDNStaged", "FlashBatchPrefill"}
CHANGED_SOURCE = {
    "runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm",
    "runtime/flash/FlashGDNStaged.cpp", "runtime/flash/FlashBatchPrefill.cpp",
    "dev/benchmarks/prefill4k_attribution.mm",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"FMA sealed-source anchor drift: {before!r}")
    return text.replace(before, after)


def transform(relative, text):
    if relative not in CHANGED_SOURCE:
        return text
    text = '#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"\n' + text
    if relative == "runtime/flash/FlashGDNStaged.cpp":
        text = replace(text, '  metal::CommandGraph validated;', '''  // This helper is called only for prefill. Decode, verification capture,
  // lazy rollback and replay continue to use their frozen native entrypoints.
  if (gdn_prefill_fma_sep21::eligible(rows, lanes, gdn_prefill_fma_sep21::requested())) {
    pipeline = gdn_prefill_fma_sep21::kPipeline;
    values = 16;
  }
  metal::CommandGraph validated;''')
    if relative == "runtime/flash/FlashForward.cpp":
        text = replace(text, '(impl_->stagedGDN ? ";gdn-prefill-staged-v16-t16" : "") +',
                       '''(impl_->stagedGDN
          ? (gdn_prefill_fma_sep21::requested()
              ? std::string(gdn_prefill_fma_sep21::marker(true))
              : std::string(";gdn-prefill-staged-v16-t16"))
          : std::string{}) +''')
    if relative == "runtime/flash/FlashBatchPrefill.cpp":
        text = replace(text, 'stagedGDN != route(";gdn-prefill-staged-v16-t16") ||',
                       '''stagedGDN != (route(";gdn-prefill-staged-v16-t16") ||
            route(gdn_prefill_fma_sep21::marker(true))) ||''')
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
                       '''      (void)gdn_prefill_fma_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
        before = '''      << R"(,"target_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(persistedExperts->numericalIdentitySha256()) : "null")'''
        after = '''      << R"(,"target_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(
          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),
              gdn_prefill_fma_sep21::requested())) : "null")
      << R"(,"target_base_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(persistedExperts->numericalIdentitySha256()) : "null")
      << R"(,"gdn_prefill_fma_enabled":)" << (gdn_prefill_fma_sep21::requested() ? "true" : "false")
      << R"(,"gdn_prefill_fma_numerical_policy":)" << (gdn_prefill_fma_sep21::requested()
          ? json::quote(std::string(gdn_prefill_fma_sep21::kPolicy)) : "null")
      << R"(,"gdn_prefill_fma_kernel_sha256":)" << (gdn_prefill_fma_sep21::requested()
          ? json::quote(std::string(gdn_prefill_fma_sep21::kKernelSourceSHA256)) : "null")
      << R"(,"gdn_prefill_fma_scope":)" << json::quote("staged prefill rows64..2048, lanes1..32; decode/verify/replay unchanged")'''
        text = replace(text, before, after)
    if relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.',
                       '''      (void)gdn_prefill_fma_sep21::requested(); // Freeze numerical prefill policy before backend creation.
      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.''')
        # Family matching is explicit: the private FMA recurrence belongs to
        # GDN even though its shader name no longer starts with flash_gdn.
        text = replace(text, 'if (name.starts_with("flash_gdn")) return "gdn";',
                       'if (name.starts_with("flash_gdn") || name.starts_with("private_gdn_scalar_fma")) return "gdn";')
    return text


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/moe-pointwise-sep21-worker-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gdn-prefill-fma-sep21-worker-v1")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or ROOT / "build" not in output.parents:
        raise ValueError("FMA worker requires a distinct private build directory")
    parent_path = base / "overlay-manifest.json"
    parent = json.loads(parent_path.read_text())
    if (not parent.get("pointwise_composed") or not parent.get("gathered_mpp_composed")
            or not parent.get("qsa_bulk_composed") or parent.get("omitted_original_target_tensor_count") != 432):
        raise ValueError("Expected sealed pointwise + Full512 + gathered-MPP + exact bulk-SG8 worker")
    kernel = (ROOT / PRIVATE / "scalar_fma.metal").read_bytes()
    if sha(kernel) != QUALIFIED_KERNEL_SHA256:
        raise ValueError("FMA source differs from Root GPU qualified kernel")
    manifest = copy.deepcopy(parent)
    manifest.update({
        "route": "private-full512-bulk-gathered-pointwise-gdn-prefill-fma-sep21-v1",
        "gdn_fma_composed": True,
        "gdn_fma_required_environment": "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21=0|1; flag1 requires SPLASH_FLASH_GDN_STAGED=1",
        "gdn_fma_flag0_original_graphs": True,
        "gdn_fma_added_allocations_bytes": 0,
        "gdn_fma_changes_numerical_derivative": True,
        "gdn_fma_scope": {"function": "addGDNStagedPrefill", "minimum_rows": 64, "maximum_rows": 2048,
                          "minimum_lanes": 1, "maximum_lanes": 32,
                          "kernel": "private_gdn_scalar_fma_v16_t32", "decode_verify_replay_changed": False},
        "gdn_fma_qualification": "Root isolated GPU F64 full state/history/output +23fixtures+continuations/canaries; delta-only relative cancellation gate fails identically for native canonical and FMA; full-model logits/state/MTP acceptance remains pending",
        "gdn_fma_base_build": str(base), "gdn_fma_base_manifest_sha256": sha(parent_path.read_bytes()),
        "gdn_fma_transform_sha256": sha(Path(__file__).read_bytes()),
        "gdn_fma_kernel_source_sha256": QUALIFIED_KERNEL_SHA256,
        "gdn_fma_new_target_identity": "SHA256(base numerical derivative + newline + FMA numerical policy + newline + qualified kernel source SHA)",
        "gpu_executed": False, "payload_bytes_read": 0, "files": [],
    })
    changed = []
    for record in parent["files"]:
        relative = record["path"]
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed base source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        write(output / "source" / relative, data)
        if data != original:
            changed.append(relative)
        manifest["files"].append({**record, "gdn_fma_changed": data != original,
                                 "gdn_fma_base_overlay_sha256": sha(original), "overlay_sha256": sha(data)})
    # Never reconstruct inherited sources from the live repository: all five
    # changed sources must already be sealed by the certified parent.
    have = {record["path"] for record in manifest["files"]}
    if CHANGED_SOURCE - have:
        raise ValueError(f"Required sealed parent source missing: {sorted(CHANGED_SOURCE - have)}")
    if set(changed) != CHANGED_SOURCE:
        raise ValueError(f"Expected precisely five isolated host edits, got {changed}")
    for relative in (PRIVATE / "worker_bridge.hpp", PRIVATE / "worker_policy_cpu.cpp", PRIVATE / "scalar_fma.metal"):
        data = (ROOT / relative).read_bytes()
        write(output / "source" / relative, data)
        manifest["files"].append({"path": str(relative), "new_gdn_fma_file": True,
                                 "gdn_fma_repository_sha256": sha(data), "overlay_sha256": sha(data)})
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    manifest["gdn_fma_link_inputs"] = []
    def freeze_input(path, relative, category, expected=None):
        data = path.read_bytes()
        if expected and sha(data) != expected:
            raise ValueError(f"Sealed base link input drift: {path}")
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["gdn_fma_link_inputs"].append({"source_path": str(path), "private_path": relative.as_posix(),
                                               "category": category, "sha256": sha(data)})
    for record in parent["pointwise_link_inputs"]:
        path = base / record["private_path"]
        if record["category"] == "REUSED" and path.stem in CHANGED_NAMES:
            continue
        relative = Path("reused/pointwise") / record["private_path"]
        freeze_input(path, relative, record["category"], record["sha256"])
    for path in sorted((base / "host").glob("*.o")):
        if path.stem not in CHANGED_NAMES:
            freeze_input(path, Path("reused/pointwise-host") / path.name, "REUSED")
    freeze_input(base / "pointwise.air", Path("reused/pointwise-metal/pointwise.air"), "AIRS")
    frozen_make = "\n".join(f"{category} := " + " ".join("$(BUILD)/" + p for p in paths)
                            for category, paths in inputs.items()) + "\n"
    write(output / "link-inputs.mk", frozen_make.encode())
    manifest["gdn_fma_link_inputs_make_sha256"] = sha(frozen_make.encode())
    manifest["gdn_fma_changed_files"] = sorted(changed)
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    write(output / "base-build.txt", (str(base) + "\n").encode())
    for name in ("splash-flash.config", "splash.metallib.config"):
        write(output / name, (base / name).read_bytes().rstrip(b"\n") + b"-gdn-prefill-fma-v16-t32-sep21-v1\n")
    print(json.dumps({"prepared": str(output), "changed_files": sorted(changed), "frozen_sources": len(manifest["files"]),
                      "frozen_link_inputs": len(manifest["gdn_fma_link_inputs"]), "added_gpu_allocation_bytes": 0,
                      "gpu_executed": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
