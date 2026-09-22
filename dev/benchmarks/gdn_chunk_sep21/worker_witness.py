#!/usr/bin/env python3
"""Device-free witness for a sealed, prefill-only explicit-FMA worker overlay.

This checks source/link integrity and routing scope only. It never executes a
worker, creates a Metal backend, or reads model payloads. A passing result does
not establish full-model fidelity, MTP fidelity, or a performance improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/gdn_chunk_sep21")
PARENT_MANIFEST_SHA256 = "cb836db216347c488c26ef80bdf1b78db1b485105f26afd9829a91ca5f2c1696"
KERNEL_SOURCE_SHA256 = "166381991f5444585e7db3634f41432322efdd66bf08a8c402d5fed49e58c699"
ALLOWED_CHANGED = {
    "runtime/flash/FlashGDNStaged.cpp",
    "runtime/flash/FlashForward.cpp",
    "runtime/flash/FlashBatchPrefill.cpp",
    "runtime/flash/FlashWorker.mm",
    "dev/benchmarks/prefill4k_attribution.mm",
}
PROTECTED_MODULES = (
    "FlashBatchForward", "FlashBatchMTPForward", "FlashBatchVerify",
    "FlashBatchVerifyGDN", "FlashGDNFused", "FlashGDNLazyRollback",
    "FlashGDNBatchILP", "FlashMTP", "FlashMTPDepth",
)
EXPECTED_SCOPE = {
    "function": "addGDNStagedPrefill", "minimum_rows": 64, "maximum_rows": 2048,
    "minimum_lanes": 1, "maximum_lanes": 32,
    "kernel": "private_gdn_scalar_fma_v16_t32", "decode_verify_replay_changed": False,
}
BRIDGE_INCLUDE = '#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"\n'
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".mm", ".metal"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Unsealed path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsealed symlink: {relative}")
    return path


def unique_index(text: str, anchor: str) -> int:
    if text.count(anchor) != 1:
        raise ValueError(f"Source anchor must be unique: {anchor!r}")
    return text.index(anchor)


def body_span(text: str, anchor: str) -> tuple[int, int]:
    """Match a qualified function/whole struct, skipping C++ quoted content."""
    begin = unique_index(text, anchor)
    cursor = text.index("{", begin)
    depth = 0
    while cursor < len(text):
        if text.startswith("//", cursor):
            newline = text.find("\n", cursor + 2)
            cursor = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", cursor):
            end = text.find("*/", cursor + 2)
            if end < 0:
                raise ValueError("Unclosed source comment")
            cursor = end + 2
            continue
        raw = re.match(r'R"([^\s()\\]{0,16})\(', text[cursor:])
        if raw:
            end_token = ")" + raw.group(1) + '"'
            end = text.find(end_token, cursor + raw.end())
            if end < 0:
                raise ValueError("Unclosed raw source string")
            cursor = end + len(end_token)
            continue
        if text[cursor] in ('"', "'"):
            quote = text[cursor]
            cursor += 1
            while cursor < len(text):
                if text[cursor] == "\\":
                    cursor += 2
                elif text[cursor] == quote:
                    cursor += 1
                    break
                else:
                    cursor += 1
            continue
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if not depth:
                return begin, cursor + 1
        cursor += 1
    raise ValueError(f"Unclosed source body: {anchor!r}")


def extract_body(text: str, anchor: str) -> str:
    begin, end = body_span(text, anchor)
    return text[begin:end]


def erase_body(text: str, anchor: str) -> str:
    begin, end = body_span(text, anchor)
    return text[:begin] + "<sealed-route-body>" + text[end:]


def without_bridge(text: str) -> str:
    if text.count(BRIDGE_INCLUDE) != 1:
        raise ValueError("Expected exactly one private bridge include")
    return text.replace(BRIDGE_INCLUDE, "", 1)


def strip_comments_and_space(text: str) -> str:
    # Used only for narrow policy expressions with no quoted comment markers.
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/gdn-prefill-fma-sep21-worker-v1")
    parser.add_argument("--base", type=Path, default=ROOT / "build/moe-pointwise-sep21-worker-v1")
    parser.add_argument("--output", type=Path, help="Optional fresh JSON witness file; stdout always has the result")
    args = parser.parse_args()
    build, base = args.build.resolve(), args.base.resolve()
    if args.output and args.output.exists():
        parser.error("Choose a fresh witness output")
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    details: dict[str, Any] = {}

    def check(name: str, operation) -> None:
        try:
            checks[name] = bool(operation())
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, ImportError) as error:
            checks[name] = False
            errors[name] = f"{type(error).__name__}: {error}"

    try:
        parent_bytes = (base / "overlay-manifest.json").read_bytes()
        parent = json.loads(parent_bytes)
        manifest = json.loads((build / "overlay-manifest.json").read_text())
        overlay_path = ROOT / PRIVATE / "worker_overlay.py"
        spec = importlib.util.spec_from_file_location("gdn_fma_sealed_worker_overlay", overlay_path)
        if not spec or not spec.loader:
            raise ValueError("Private overlay transformer unavailable")
        overlay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(overlay)
    except (OSError, ValueError, ImportError) as error:
        result = {
            "schema": "splash-gdn-prefill-fma-worker-source-witness-v1", "pass": False,
            "source_integrity_pass": False, "gpu_work": False, "model_loaded": False,
            "payload_bytes_read": 0, "model_fidelity_pass": None, "mtp_fidelity_pass": None,
            "fidelity_qualification": "pending full-model and MTP gates",
            "error": f"{type(error).__name__}: {error}",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 1

    check("sealed_parent_manifest_sha256", lambda:
          sha(parent_bytes) == manifest["gdn_fma_base_manifest_sha256"] == PARENT_MANIFEST_SHA256)
    check("parent_build_matches_argument", lambda: Path(manifest["gdn_fma_base_build"]).resolve() == base)
    check("parent_is_pointwise_gathered_bulk_full512", lambda:
          parent["pointwise_composed"] and parent["gathered_mpp_composed"] and parent["qsa_bulk_composed"]
          and parent["omitted_original_target_tensor_count"] == 432)
    check("transformer_sha256_sealed", lambda:
          file_sha(overlay_path) == manifest["gdn_fma_transform_sha256"])
    check("source_only_manifest_scope", lambda:
          manifest["gdn_fma_composed"] and manifest["gdn_fma_flag0_original_graphs"]
          and manifest["gdn_fma_scope"] == EXPECTED_SCOPE
          and manifest["gdn_fma_added_allocations_bytes"] == 0
          and manifest["gpu_executed"] is False and manifest["payload_bytes_read"] == 0)

    parent_records = {record["path"]: record for record in parent["files"]}
    records = {record["path"]: record for record in manifest["files"]}
    check("unique_source_records", lambda:
          len(parent_records) == len(parent["files"]) and len(records) == len(manifest["files"]))
    check("all_parent_source_files_retained", lambda: set(parent_records) <= set(records))
    parent_mismatch, source_mismatch, digest_mismatch = [], [], []
    changed, added = [], []
    for relative, record in records.items():
        try:
            if Path(relative).suffix not in SOURCE_SUFFIXES:
                raise ValueError(f"Non-source manifest file: {relative}")
            actual = sealed_path(build / "source", relative).read_bytes()
            if relative in parent_records:
                original = sealed_path(base / "source", relative).read_bytes()
                if sha(original) != parent_records[relative]["overlay_sha256"]:
                    parent_mismatch.append(relative)
                if actual != original:
                    changed.append(relative)
                if relative not in ALLOWED_CHANGED and actual != original:
                    source_mismatch.append(relative)
            else:
                added.append(relative)
                relative_path = Path(relative)
                is_cpu_source = relative_path.suffix == ".cpp" and "policy" in relative_path.name
                if relative_path.parent != PRIVATE or not (
                    relative_path.name in {"worker_bridge.hpp", "scalar_fma.metal"} or is_cpu_source
                ):
                    raise ValueError(f"Unapproved new source: {relative}")
                original = sealed_path(ROOT, relative).read_bytes()
            expected = overlay.transform(relative, original.decode()).encode()
            if actual != expected:
                source_mismatch.append(relative)
            if sha(actual) != record["overlay_sha256"]:
                digest_mismatch.append(relative)
        except (OSError, ValueError, KeyError, UnicodeError) as error:
            source_mismatch.append(relative)
            errors[f"source:{relative}"] = f"{type(error).__name__}: {error}"
    details.update({"files_checked": len(records), "parent_source_mismatch": parent_mismatch,
                    "source_mismatch": sorted(set(source_mismatch)), "source_digest_mismatch": digest_mismatch,
                    "changed_files": sorted(changed), "added_files": sorted(added)})
    check("all_sealed_parent_source_digests_fresh", lambda: not parent_mismatch)
    check("all_output_source_digests_and_transforms_fresh", lambda: not source_mismatch and not digest_mismatch)
    check("only_five_allowed_host_sources_changed", lambda:
          set(changed) == ALLOWED_CHANGED and set(manifest["gdn_fma_changed_files"]) == ALLOWED_CHANGED)
    check("source_tree_has_no_unmanifested_files", lambda:
          {path.relative_to(build / "source").as_posix() for path in (build / "source").rglob("*")
           if path.is_file()} == set(records))

    protected_source = [f"runtime/flash/{name}{suffix}" for name in PROTECTED_MODULES for suffix in (".cpp", ".hpp")]
    check("all_decode_verify_replay_lazy_mtp_modules_identical", lambda:
          all(relative in parent_records and relative in records
              and file_sha(sealed_path(base / "source", relative)) == parent_records[relative]["overlay_sha256"]
              == file_sha(sealed_path(build / "source", relative)) for relative in protected_source))
    protected_headers = [relative for relative in parent_records
                         if relative.startswith("runtime/") and Path(relative).suffix in {".h", ".hpp"}]
    check("all_runtime_headers_and_abi_identical", lambda:
          all(file_sha(sealed_path(build / "source", relative)) == parent_records[relative]["overlay_sha256"]
              for relative in protected_headers))
    details["protected_sources_checked"] = len(protected_source)
    details["protected_runtime_headers_checked"] = len(protected_headers)

    def source(relative: str, old: bool = False) -> str:
        return sealed_path((base if old else build) / "source", relative).read_text()

    forward_relative = "runtime/flash/FlashForward.cpp"
    route_anchor = "std::string FlashForward::kernelRoutes() const"
    check("forward_execution_workspace_verify_replay_entire_file_identical", lambda:
          erase_body(without_bridge(source(forward_relative)), route_anchor)
          == erase_body(source(forward_relative, True), route_anchor))
    forward_anchors = (
        "struct FlashForward::Impl final", "uint64_t FlashForward::requestStateBytes(",
        "uint64_t FlashForward::workspacePlannedBytes(", "uint64_t FlashForward::verificationWorkspaceBytes(",
        "FlashRequestState FlashForward::createState()", "FlashForwardResult FlashForward::forward(",
        "FlashForwardResult FlashForward::verify(", "FlashForwardResult FlashForward::forwardImpl(",
        "metal::CommandTiming FlashForward::commitVerify(", "void FlashForward::abortVerify(",
    )
    check("forward_protected_function_ranges_identical", lambda:
          all(extract_body(source(forward_relative), anchor) == extract_body(source(forward_relative, True), anchor)
              for anchor in forward_anchors))

    batch_relative = "runtime/flash/FlashBatchPrefill.cpp"
    def batch_without_marker_guard(text: str) -> str:
        begin = unique_index(text, "        stagedGDN != ")
        end = unique_index(text, "        cachedDense != ")
        if end <= begin:
            raise ValueError("Staged marker guard ordering changed")
        return text[:begin] + "<sealed-staged-marker-guard>\n" + text[end:]
    check("batch_prefill_execution_workspace_identical_outside_marker_guard", lambda:
          batch_without_marker_guard(without_bridge(source(batch_relative)))
          == batch_without_marker_guard(source(batch_relative, True)))
    batch_anchors = (
        "std::array<uint64_t, kSlots> sizes(", "uint64_t FlashBatchPrefill::workspacePlannedBytes(",
        "FlashBatchPrefillResult FlashBatchPrefill::forwardBatch(",
        "std::vector<FlashBatchPrefillStatePlane> FlashBatchPrefill::inspectState(",
        "bool FlashBatchPrefill::canariesIntact() const noexcept",
    )
    check("batch_prefill_protected_function_ranges_identical", lambda:
          all(extract_body(source(batch_relative), anchor) == extract_body(source(batch_relative, True), anchor)
              for anchor in batch_anchors))

    staged_relative = "runtime/flash/FlashGDNStaged.cpp"
    def staged_conditional() -> str:
        native = source(staged_relative, True)
        actual = without_bridge(source(staged_relative))
        anchor = "  metal::CommandGraph validated;"
        native_split = unique_index(native, anchor)
        prefix, suffix = native[:native_split], native[native_split:]
        if not actual.startswith(prefix) or not actual.endswith(suffix):
            raise ValueError("Native staged case/switch/validation/graph order changed")
        return actual[len(prefix):len(actual) - len(suffix)]
    check("flag0_exact_native_stage_case_switch_graph_order", lambda: bool(staged_conditional().strip()))
    def eligible_conditional_exact() -> bool:
        normalized = strip_comments_and_space(staged_conditional())
        accepted = {
            "if(gdn_prefill_fma_sep21::eligible(rows,lanes,gdn_prefill_fma_sep21::requested()))"
            "{pipeline=gdn_prefill_fma_sep21::kPipeline;values=" + values + ";}"
            for values in ("gdn_prefill_fma_sep21::kValues", "16")
        }
        return normalized in accepted
    check("flag1_changes_only_pipeline_and_values_under_eligible_guard", eligible_conditional_exact)
    check("no_alternative_decode_verify_or_step_fma_call_sites", lambda:
          [relative for relative in records if relative.startswith("runtime/")
           and "gdn_prefill_fma_sep21::eligible(" in source(relative)] == [staged_relative]
          and sum(source(relative).count("gdn_prefill_fma_sep21::kPipeline") for relative in records
                  if relative.startswith("runtime/")) == 1)

    bridge_relative = (PRIVATE / "worker_bridge.hpp").as_posix()
    kernel_relative = (PRIVATE / "scalar_fma.metal").as_posix()
    check("kernel_identical_to_root_qualified_source", lambda:
          file_sha(ROOT / kernel_relative) == file_sha(build / "source" / kernel_relative)
          == manifest["gdn_fma_kernel_source_sha256"] == overlay.QUALIFIED_KERNEL_SHA256 == KERNEL_SOURCE_SHA256)
    check("bridge_kernel_identity_and_zero_allocation_declaration", lambda:
          KERNEL_SOURCE_SHA256 in source(bridge_relative)
          and 'kPipeline = "private_gdn_scalar_fma_v16_t32"' in source(bridge_relative)
          and "kGPUAddedAllocations = 0" in source(bridge_relative))
    check("bridge_eligibility_exact64to2048_lanes1to32", lambda:
          strip_comments_and_space(extract_body(source(bridge_relative), "constexpr bool eligible("))
          .split("{", 1)[1] == "returnenabled&&rows>=64&&rows<=2048&&lanes>=1&&lanes<=32;}")
    check("bridge_requested_policy_process_frozen", lambda:
          "static const bool enabled = detail::requestedFromEnvironment();" in
          extract_body(source(bridge_relative), "inline bool requested()"))
    check("bridge_disabled_route_and_identity_preserve_native_inputs", lambda:
          "Route{nativePipeline, nativeValues, nativeTime, false}" in source(bridge_relative)
          and "if (!enabled) return std::string(base);" in source(bridge_relative)
          and 'return enabled ? kMarker : "";' in source(bridge_relative))

    check("no_additional_gpu_buffer_allocation_calls", lambda:
          all(source(relative).count("allocateBuffer(") == source(relative, True).count("allocateBuffer(")
              for relative in parent_records)
          and all("allocateBuffer(" not in source(relative) for relative in added))
    def freezes_before_backend() -> bool:
        worker = source("runtime/flash/FlashWorker.mm")
        attribution = source("dev/benchmarks/prefill4k_attribution.mm")
        freeze = "(void)gdn_prefill_fma_sep21::requested();"
        return (worker.index(freeze) < worker.index("std::filesystem::canonical(argv[2])")
                < worker.index("metal::MetalBackend backend(")
                and attribution.index(freeze) < attribution.index("metal::MetalBackend backend("))
    check("worker_and_attribution_freeze_before_paths_and_backend", freezes_before_backend)
    check("worker_status_binds_numerical_identity_to_fma_source", lambda:
          "gdn_prefill_fma_sep21::numericalIdentity(" in source("runtime/flash/FlashWorker.mm")
          and "gdn_prefill_fma_sep21::kPolicy" in source("runtime/flash/FlashWorker.mm"))

    parent_inputs = parent["pointwise_link_inputs"]
    frozen_inputs = manifest["gdn_fma_link_inputs"]
    link_mismatch = []
    actual_input_counter: Counter = Counter()
    for record in frozen_inputs:
        try:
            relative = record["private_path"]
            if Path(relative).suffix not in {".o", ".air"}:
                raise ValueError(f"Non-link-input frozen record: {relative}")
            digest = file_sha(sealed_path(build, relative))
            if digest != record["sha256"]:
                link_mismatch.append(relative)
            actual_input_counter[(record["category"], digest)] += 1
        except (OSError, ValueError, KeyError) as error:
            link_mismatch.append(record.get("private_path", "<missing-private-path>"))
            errors[f"link:{record.get('private_path')}"] = f"{type(error).__name__}: {error}"
    expected_input_counter: Counter = Counter()
    parent_input_mismatch = []
    for record in parent_inputs:
        try:
            digest = file_sha(sealed_path(base, record["private_path"]))
            if digest != record["sha256"]:
                parent_input_mismatch.append(record["private_path"])
            if Path(record["private_path"]).name not in {"FlashGDNStaged.o", "FlashBatchPrefill.o"}:
                expected_input_counter[(record["category"], digest)] += 1
        except (OSError, ValueError, KeyError) as error:
            parent_input_mismatch.append(record.get("private_path", "<missing-private-path>"))
            errors[f"parent-link:{record.get('private_path')}"] = f"{type(error).__name__}: {error}"
    for name in sorted({Path(relative).stem for relative in parent["pointwise_changed_files"]}
                       - {"FlashForward", "FlashWorker"}):
        check(f"pointwise_frozen_host_{name}_available", lambda name=name:
              (base / "host" / (name + ".o")).is_file())
        if (base / "host" / (name + ".o")).is_file():
            expected_input_counter[("REUSED", file_sha(base / "host" / (name + ".o")))] += 1
    check("pointwise_frozen_air_available", lambda: (base / "pointwise.air").is_file())
    if (base / "pointwise.air").is_file():
        expected_input_counter[("AIRS", file_sha(base / "pointwise.air"))] += 1
    check("all_parent_frozen_link_digests_fresh", lambda: not parent_input_mismatch)
    check("all_output_frozen_link_digests_fresh", lambda: not link_mismatch)
    check("frozen_link_graph_exact_parent_with_only_two_host_replacements", lambda:
          actual_input_counter == expected_input_counter)
    check("protected_decode_verify_replay_lazy_mtp_objects_present_identical", lambda:
          all(any(Path(record["private_path"]).name == name + ".o"
                  and actual_input_counter[(record["category"], record["sha256"])] > 0
                  for record in parent_inputs) for name in PROTECTED_MODULES))
    check("all_parent_air_inputs_identical_and_retained", lambda:
          all(actual_input_counter[(record["category"], record["sha256"])]
              >= Counter((item["category"], item["sha256"]) for item in parent_inputs
                         if item["category"] == "AIRS")[(record["category"], record["sha256"])]
              for record in parent_inputs if record["category"] == "AIRS"))
    check("frozen_link_make_list_digest_fresh", lambda:
          file_sha(build / "link-inputs.mk") == manifest["gdn_fma_link_inputs_make_sha256"])
    def make_references_match_inputs() -> bool:
        references = re.findall(r"\$\(BUILD\)/([^\s]+)", (build / "link-inputs.mk").read_text())
        return Counter(references) == Counter(record["private_path"] for record in frozen_inputs)
    check("frozen_make_list_references_exact_manifest_inputs", make_references_match_inputs)
    details.update({"frozen_link_inputs_checked": len(frozen_inputs), "link_input_mismatch": link_mismatch,
                    "parent_link_input_mismatch": parent_input_mismatch})

    live_headers = []
    dependency_files = list((build / "host").glob("*.d"))
    for path in dependency_files:
        for token in path.read_text().replace("\\\n", " ").split():
            if token.endswith((".h", ".hpp")) and (
                token.startswith("runtime/") or token.startswith(str(ROOT / "runtime") + "/")
            ):
                live_headers.append(token)
    check("compiled_changed_hosts_have_no_live_runtime_header_dependencies", lambda: not live_headers)
    details["compiled_host_dependency_files_checked"] = len(dependency_files)
    details["live_runtime_header_dependencies"] = sorted(set(live_headers))
    details["artifact_sha256"] = {name: file_sha(build / name) for name in
                                   ("splash-flash", "splash.metallib", "prefill4k-attribution", "policy-cpu")
                                   if (build / name).is_file()}
    result = {
        "schema": "splash-gdn-prefill-fma-worker-source-witness-v1", "pass": all(checks.values()),
        "source_integrity_pass": all(checks.values()), "gpu_work": False, "model_loaded": False,
        "payload_bytes_read": 0, "model_fidelity_pass": None, "mtp_fidelity_pass": None,
        "fidelity_qualification": "pending full-model and MTP gates", "checks": checks, "errors": errors,
        "kernel_source_sha256": KERNEL_SOURCE_SHA256, "base_manifest_sha256": sha(parent_bytes),
        **details,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
