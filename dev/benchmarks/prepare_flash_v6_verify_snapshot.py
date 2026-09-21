#!/usr/bin/env python3
"""Prepare a private joint-verifier expert-ID capture snapshot, without execution.

The generated class keeps the FlashBatchVerify name so the existing private
friendships remain valid. Put the snapshot root before runtime in the include
path and compile the generated CPP in place of the production verifier object.
This script never changes production sources or submits GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transformed_sources(header: str, source: str) -> tuple[str, str, list[dict[str, object]]]:
    transformations: list[dict[str, object]] = []
    inverse: list[tuple[str, str, str]] = []

    def replace(text: str, filename: str, label: str, anchor: str, replacement: str) -> str:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(f"{filename}: {label} expected one anchor, found {count}")
        transformations.append({"file": filename, "label": label, "anchor_count": count})
        inverse.append((filename, replacement, anchor))
        return text.replace(anchor, replacement, 1)

    original_header, original_source = header, source
    header = replace(
        header, "FlashBatchVerify.hpp", "metadata_getter_declaration",
        "  [[nodiscard]] uint64_t workspaceBytes() const noexcept;\n",
        "  [[nodiscard]] uint64_t workspaceBytes() const noexcept;\n"
        "  // Private diagnostic I64[layer,maximumLanes*maximumRows,10]. Only the\n"
        "  // first physicalRows entries per layer are valid after a successful\n"
        "  // verifyBatch(), through the next verifyBatch(). Metadata only; no work.\n"
        "  [[nodiscard]] metal::MetalBuffer capturedDiagnosticExpertIDs(\n"
        "      uint32_t &physicalRows, uint32_t &rowStride) const;\n",
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "retained_shared_capture_fields",
        "  metal::MetalBuffer greedyResults;\n",
        "  metal::MetalBuffer greedyResults;\n"
        "  const bool captureDiagnosticRoutes = fusionEnabled(\n"
        "      \"SPLASH_FLASH_CAPTURE_EXPERT_IDS\");\n"
        "  metal::MetalBuffer capturedDiagnosticRoutes;\n"
        "  uint32_t capturedRows = 0;\n",
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "bounded_shared_capture_allocation",
        "    const uint64_t before = backend.memoryStats().allocatedBytes;\n",
        "    const uint64_t before = backend.memoryStats().allocatedBytes;\n"
        "    if (captureDiagnosticRoutes)\n"
        "      capturedDiagnosticRoutes = backend.allocateBuffer(\n"
        "          rounded(uint64_t{48} * maximumLanes * maximumRows * kSelections * 8),\n"
        "          metal::BufferStorage::Shared, \"flash-batch-verify diagnostic expert IDs\");\n",
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "private_capture_workspace_accounting",
        "  if (fusionEnabled(\"SPLASH_FLASH_GPU_GREEDY\"))\n"
        "    total += greedyGPUWorkspacePlannedBytes(maximumLanes * maximumRows, 248320);\n",
        "  if (fusionEnabled(\"SPLASH_FLASH_CAPTURE_EXPERT_IDS\"))\n"
        "    total += rounded(uint64_t{48} * maximumLanes * maximumRows * kSelections * 8);\n"
        "  if (fusionEnabled(\"SPLASH_FLASH_GPU_GREEDY\"))\n"
        "    total += greedyGPUWorkspacePlannedBytes(maximumLanes * maximumRows, 248320);\n",
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "clear_capture_metadata_at_verify_start",
        "  if (!impl_) throw std::logic_error(\"Flash batch is not initialized\");\n"
        "  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());\n",
        "  if (!impl_) throw std::logic_error(\"Flash batch is not initialized\");\n"
        "  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());\n"
        "  impl_->capturedRows = 0;\n",
    )
    route = "    addRoute(graph, bf(Slot::Router, 512), expertIDs, routes, diag, flattened, 512, kSelections);\n"
    source = replace(
        source, "FlashBatchVerify.cpp", "capture_after_route_before_scratch_reuse",
        route,
        route + "    if (impl_->capturedDiagnosticRoutes)\n"
        "      impl_->copy(graph, expertIDs, impl_->backend.view(impl_->capturedDiagnosticRoutes,\n"
        "          uint64_t{layer} * impl_->maximumLanes * impl_->maximumRows * kSelections * 8,\n"
        "          uint64_t{flattened} * kSelections * 8), uint64_t{flattened} * kSelections * 8);\n",
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "publish_only_successful_submission_metadata",
        "  for (uint32_t lane = 0; lane < lanes; ++lane) {\n"
        "    lengths[lane] = states[lane]->length += rows;\n",
        "  impl_->capturedRows = impl_->capturedDiagnosticRoutes ? flattened : 0;\n"
        "  for (uint32_t lane = 0; lane < lanes; ++lane) {\n"
        "    lengths[lane] = states[lane]->length += rows;\n",
    )
    workspace_getter = (
        "uint64_t FlashBatchVerify::workspaceBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }\n"
    )
    source = replace(
        source, "FlashBatchVerify.cpp", "metadata_getter_definition",
        workspace_getter,
        workspace_getter + "\n"
        "metal::MetalBuffer FlashBatchVerify::capturedDiagnosticExpertIDs(\n"
        "    uint32_t &physicalRows, uint32_t &rowStride) const {\n"
        "  physicalRows = impl_ ? impl_->capturedRows : 0;\n"
        "  rowStride = impl_ ? impl_->maximumLanes * impl_->maximumRows : 0;\n"
        "  return impl_ ? impl_->capturedDiagnosticRoutes : metal::MetalBuffer{};\n"
        "}\n",
    )

    # Removing only the known additive diagnostic transformations must reproduce
    # each original byte. This also detects an accidentally destructive edit.
    recovered = {"FlashBatchVerify.hpp": header, "FlashBatchVerify.cpp": source}
    for filename, replacement, anchor in reversed(inverse):
        if recovered[filename].count(replacement) != 1:
            raise ValueError(f"{filename}: inverse transformation is not unique")
        recovered[filename] = recovered[filename].replace(replacement, anchor, 1)
    if recovered["FlashBatchVerify.hpp"] != original_header or recovered["FlashBatchVerify.cpp"] != original_source:
        raise ValueError("Private snapshot changed production code beyond diagnostic additions")
    if header.count("class FlashBatchVerify final") != 1:
        raise ValueError("Private snapshot no longer preserves the existing class identity")
    if source.count("impl_->copy(graph, expertIDs, impl_->backend.view(impl_->capturedDiagnosticRoutes,") != 1:
        raise ValueError("Private snapshot must have exactly one per-layer capture site")
    return header, source, transformations


def prepare(repo: Path, snapshot_root: Path, check: bool) -> dict[str, object]:
    originals: dict[str, bytes] = {}
    for filename in ("FlashBatchVerify.hpp", "FlashBatchVerify.cpp"):
        originals[filename] = (repo / "runtime" / "flash" / filename).read_bytes()
    header, source, transformations = transformed_sources(
        originals["FlashBatchVerify.hpp"].decode("utf-8"),
        originals["FlashBatchVerify.cpp"].decode("utf-8"),
    )
    generated = {"FlashBatchVerify.hpp": header.encode("utf-8"), "FlashBatchVerify.cpp": source.encode("utf-8")}
    output_directory = snapshot_root / "flash"
    original_directory = (repo / "runtime" / "flash").resolve()
    if output_directory.resolve() == original_directory or original_directory in output_directory.resolve().parents:
        raise ValueError("Snapshot output cannot overwrite production sources")
    if not check:
        output_directory.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for filename, data in generated.items():
        destination = output_directory / filename
        if check:
            if not destination.exists() or destination.read_bytes() != data:
                raise ValueError(f"Snapshot does not match deterministic transformations: {destination}")
        elif not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)
        files.append({
            "original": str(repo / "runtime" / "flash" / filename),
            "original_sha256": sha256_bytes(originals[filename]),
            "snapshot": str(destination),
            "snapshot_sha256": sha256_bytes(data),
            "original_bytes": len(originals[filename]),
            "snapshot_bytes": len(data),
        })
    # Verify no production file changed while preparing this snapshot.
    for filename, data in originals.items():
        if (repo / "runtime" / "flash" / filename).read_bytes() != data:
            raise ValueError(f"Production source changed during preparation: {filename}")
    return {
        "schema": "splash-private-flash-v6-joint-verifier-capture-v1",
        "mode": "check" if check else "prepare",
        "cpu_only": True,
        "class_identity_preserved": True,
        "original_code_recovered_exactly": True,
        "transformation_count": len(transformations),
        "transformations": transformations,
        "files": files,
        "capture": {
            "enable_environment": "SPLASH_FLASH_CAPTURE_EXPERT_IDS=1",
            "layers": 48,
            "maximum_physical_rows": 16,
            "selections_per_row": 10,
            "element_type": "I64",
            "maximum_payload_bytes": 48 * 16 * 10 * 8,
            "maximum_rounded_allocation_bytes": 65536,
            "getter_submits_work": False,
            "capture_dispatch_sites": 1,
            "dispatches_per_enabled_full_verify": 48,
            "math_and_request_state_changes": False,
        },
    }


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify existing snapshot bytes without rewriting")
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    snapshot_root = (arguments.snapshot_root or repo / "build/flash-v6-verifier-attribution/snapshot").resolve()
    print(json.dumps(prepare(repo, snapshot_root, arguments.check), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
