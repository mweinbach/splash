#!/usr/bin/env python3
"""Compose only the exact 2K QSA bulk overlay over source text in memory.

The caller owns source provenance, private snapshot writing, worker policy,
F32 pruning, builds, and qualification. ``compose`` returns the changed source
mapping, five helper source files, and a byte-level preservation witness. It
reads only the existing generator and those five source files, never a model
payload, and writes no files. Inputs must retain the Original-Q4 small-row
expert branch with the existing rows>=256 blocked-prefill boundary.

Enable the retained optional SG8 route with QSA_F32/MPP/ROW_TILES=1 and both
SPLASH_FLASH_QSA_BULK_PREFILL=1 / SPLASH_FLASH_QSA_BULK_PREFILL_SG8=1. Its
eligibility remains main !verification, begin==0, rows==2048; other QSA calls
retain their exact original loop.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/prefill4k_attention")
FORWARD = "runtime/flash/FlashForward.cpp"
HELPER_NAMES = ("bulk.hpp", "bulk.cpp", "coalesced.hpp", "coalesced.cpp",
                "bulk_attention_sg8.metal")
POINTWISE_PRIVATE = Path("dev/benchmarks/moe_pointwise_sep21")
POINTWISE_PATHS = (
    "runtime/flash/FlashMoE.cpp", "runtime/flash/FlashMoEBlocked.cpp",
    "runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashExpertDenseCache.cpp",
    FORWARD, "runtime/flash/FlashWorker.mm",
)
BLOCKED_BOUNDARY = "    const bool blocked = impl_->blockMoE && rows >= 256;\n"
Q4_BRANCH_END = "    if (!batchSharedExpertFused(graph, mlp + \".shared_expert\", mixed,\n"
QSA_LOOP_START = "      for (uint32_t offset = 0; offset < rows;) {\n"
QSA_LOOP_END = "      affine(graph, attention + \".o_proj\", attentionOutput, branch);"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bulk_module():
    # Register before executing: bulk_overlay.Edit is a dataclass and resolves
    # postponed annotations against its module during decoration.
    name = "splash_hybrid_memory_sep21_bulk_overlay"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ROOT / PRIVATE / "bulk_overlay.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source-only bulk transform: {path}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        del sys.modules[name]
        raise
    return result


def _section(text: str, start: str, end: str, label: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Hybrid bulk preservation anchor drifted: {label}")
    first, last = text.index(start), text.index(end)
    if first >= last:
        raise ValueError(f"Hybrid bulk preservation anchor order drifted: {label}")
    return text[first:last]


def _q4_section(text: str) -> str:
    if "SPLASH_FLASH_ALLROWS_FULL512_TARGET" in text:
        raise ValueError("Hybrid bulk input must retain Original-Q4 small-row target routes")
    section = _section(text, BLOCKED_BOUNDARY, Q4_BRANCH_END, "Original-Q4 expert branch")
    for suffix in ("gate_proj", "up_proj", "down_proj"):
        anchor = f'impl_->weights.projection(mlp + ".switch_mlp.{suffix}")'
        if section.count(anchor) != 2:
            raise ValueError(f"Original-Q4 blocked and gathered source contract drifted: {suffix}")
    if section.count("addGatheredAffine(") != 3 or section.count("addSiLUMultiply(") != 1:
        raise ValueError("Original-Q4 small-row affine/activation contract drifted")
    return section


def transform(relative: str | Path, text: str) -> str:
    """Per-file convenience API; non-Forward text is returned byte-identical."""
    if str(relative) != FORWARD:
        return text
    original_q4 = _q4_section(text)
    bulk = _bulk_module()
    modified, edits = bulk.transform(text)
    if bulk.restore(modified, edits) != text or _q4_section(modified) != original_q4:
        raise AssertionError("Bulk-only composition changed an Original-Q4 route")
    return modified


def extra_files() -> dict[str, str]:
    """Return the five existing helper sources without changing their bytes."""
    return {str(PRIVATE / name): (ROOT / PRIVATE / name).read_bytes().decode("utf-8")
            for name in HELPER_NAMES}


def compose(original_sources: Mapping[str | Path, str]) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Return (composed input mapping, helper mapping, preservation witness).

    Helper files are separate so the caller can record them as new private
    inputs. No snapshot directory or manifest is created by this module.
    ``original_sources`` is not mutated. An existing bulk overlay is rejected
    by the authoritative generator; duplicate normalized paths are rejected.
    """
    sources: dict[str, str] = {}
    for relative, text in original_sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or str(path) == ".":
            raise ValueError(f"Unsafe source path: {relative}")
        key = str(path)
        if key in sources:
            raise ValueError(f"Duplicate normalized source path: {relative}")
        if not isinstance(text, str):
            raise TypeError(f"Source mapping values must be UTF-8 text: {relative}")
        sources[key] = text
    if FORWARD not in sources:
        raise ValueError(f"Source mapping must include {FORWARD}")
    original = sources[FORWARD]
    original_q4 = _q4_section(original)
    original_loop = _section(original, QSA_LOOP_START, QSA_LOOP_END, "original main-QSA loop")
    bulk = _bulk_module()
    modified, edits = bulk.transform(original)
    restored_exact = bulk.restore(modified, edits).encode("utf-8") == original.encode("utf-8")
    q4_exact = _q4_section(modified).encode("utf-8") == original_q4.encode("utf-8")
    qsa_fallback_exact = modified.count(original_loop) == 1
    if not restored_exact or not q4_exact or not qsa_fallback_exact:
        raise AssertionError("Bulk overlay failed its byte preservation witness")
    composed = dict(sources)
    composed[FORWARD] = modified
    helpers = extra_files()
    for relative, helper in helpers.items():
        if relative in sources and sources[relative] != helper:
            raise ValueError(f"Input has conflicting bulk helper source: {relative}")
    changed = [relative for relative in sources if composed[relative] != sources[relative]]
    if changed != [FORWARD]:
        raise AssertionError("Bulk composition changed an unexpected source path")
    witness: dict[str, Any] = {
        "schema": "splash-hybrid-original-q4-decode-exact2k-bulk-source-compose-v1",
        "gpu_executed": False,
        "payload_bytes_read": 0,
        "files_written": 0,
        "input_source_count": len(sources),
        "changed_source_paths": changed,
        "input_forward_sha256": sha(original),
        "output_forward_sha256": sha(modified),
        "reversible_forward_byte_exact": restored_exact,
        "original_q4_small_and_blocked_routes_byte_exact": q4_exact,
        "original_q4_expert_section_sha256": sha(original_q4),
        "original_q4_expert_section_bytes": len(original_q4.encode("utf-8")),
        "original_qsa_fallback_loop_byte_exact": qsa_fallback_exact,
        "original_qsa_fallback_loop_sha256": sha(original_loop),
        "original_qsa_fallback_loop_bytes": len(original_loop.encode("utf-8")),
        "all_nonforward_input_sources_byte_exact": all(composed[p] == sources[p] for p in sources if p != FORWARD),
        "original_q4_main_row_boundary": "blocked iff flag && rows>=256; original gathered Q4 otherwise",
        "qsa_bulk_eligible_call": "main !verification && begin==0 && rows==2048",
        "qsa_bulk_workspace_extra_bytes": bulk.PLANNED_BYTES,
        "qsa_bulk_sg8_available": True,
        "qsa_bulk_dependency_flags": list(bulk.DEPENDENCIES),
        "qsa_bulk_required_flags": [bulk.FLAG, bulk.SG8_FLAG],
        "qsa_bulk_coefficients_changed": False,
        "worker_and_loader_changed": False,
        "helpers": {p: {"sha256": sha(t), "bytes": len(t.encode("utf-8"))} for p, t in helpers.items()},
        "transform_sha256": sha((ROOT / PRIVATE / "bulk_overlay.py").read_bytes().decode("utf-8")),
        "qualification": None,
    }
    return composed, helpers, witness


compose_bulk = compose


def _pointwise_module():
    name = "splash_hybrid_memory_sep21_pointwise_overlay"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ROOT / POINTWISE_PRIVATE / "worker_overlay.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source-only pointwise transform: {path}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        del sys.modules[name]
        raise
    return result


def transform_store_pointwise(text: str) -> str:
    """Adapt the raw parent's validated-dispatch copier to exact poison SG1.

    The raw store gets its poison dispatch from addMoEBlockedDownScatter, which
    the normal pointwise transform already changes in FlashMoEBlocked.cpp.
    Preserve the copied name, buffers, inline params, and launch dimensions;
    only permit that qualified private name through the existing poison ABI
    branch. The allrows Store's direct graph.add anchor is deliberately unused.
    """
    include = '#include "dev/benchmarks/moe_pointwise_sep21/bridge.hpp"\n'
    if include in text or "private_moe_poison_route32" in text:
        raise ValueError("Raw Store already contains the pointwise adaptation")
    before = '  if (d.pipelineName == "flash_moe_blocked_poison_excluded_routes") {\n'
    after = ('  if (d.pipelineName == "flash_moe_blocked_poison_excluded_routes" ||\n'
             '      d.pipelineName == "private_moe_poison_route32") {\n')
    copy = '    graph.add(d.pipelineName, std::move(buffers), params, d.threadgroups, d.threadsPerThreadgroup);'
    if text.count(before) != 1 or text.count(copy) != 2:
        raise ValueError("Raw Store validated-dispatch copy contract drifted")
    source = '    std::array<FlashTensor, 9> sourceTensors;\n'
    if text.count(source) != 1:
        raise ValueError("Raw Store must retain Original-Q4 source tensor aliases")
    modified = include + text.replace(before, after, 1)
    if modified.removeprefix(include).replace(after, before, 1).encode() != text.encode():
        raise AssertionError("Raw Store pointwise change is not byte reversible")
    if modified.count(copy) != 2 or modified.count(source) != 1:
        raise AssertionError("Raw Store pointwise composition changed copied dispatch or Q4 aliases")
    return modified


def transform_pointwise(relative: str | Path, text: str) -> str:
    """Compose the existing pointwise transform with the raw-Store exception."""
    relative = str(relative)
    if relative == "runtime/flash/FlashInt8ExpertStore.mm":
        return transform_store_pointwise(text)
    return _pointwise_module().transform(relative, text)


def pointwise_extra_files() -> dict[str, str]:
    pointwise = _pointwise_module()
    files = {str(POINTWISE_PRIVATE / name): (ROOT / POINTWISE_PRIVATE / name).read_bytes().decode("utf-8")
             for name in ("bridge.hpp", "candidate.metal", "policy_cpu.cpp")}
    kernel = str(POINTWISE_PRIVATE / "candidate.metal")
    if sha(files[kernel]) != pointwise.QUALIFIED_KERNEL_SHA256:
        raise ValueError("Pointwise helper kernel differs from the qualified source")
    attribution = "dev/benchmarks/prefill4k_attribution.mm"
    files[attribution] = pointwise.transform(attribution, (ROOT / attribution).read_bytes().decode("utf-8"))
    return files


def compose_pointwise(original_sources: Mapping[str | Path, str]) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Return six host route changes, unchanged kernel helpers, and CPU witness.

    This function expects the raw Original-Q4 parent, optionally after
    ``compose_bulk``. The caller still owns Worker status/numerical phase policy.
    Pointwise itself changes only exact combine/poison dispatch and early flag
    validation, adds no workspace, and preserves source coefficient bindings.
    """
    sources = {str(Path(p)): t for p, t in original_sources.items()}
    if len(sources) != len(original_sources):
        raise ValueError("Duplicate normalized pointwise source path")
    for relative, text in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not isinstance(text, str):
            raise ValueError(f"Invalid pointwise source input: {relative}")
    if any(p not in sources for p in POINTWISE_PATHS):
        raise ValueError("Pointwise source mapping must include all six host paths")
    original_q4 = _q4_section(sources[FORWARD])
    composed = {relative: transform_pointwise(relative, text)
                if relative in POINTWISE_PATHS else text for relative, text in sources.items()}
    changed = [p for p in sources if composed[p] != sources[p]]
    if set(changed) != set(POINTWISE_PATHS) or _q4_section(composed[FORWARD]) != original_q4:
        raise AssertionError("Pointwise composition changed an unexpected host source or Q4 route")
    helpers = pointwise_extra_files()
    for p, text in helpers.items():
        if p in sources and sources[p] != text:
            raise ValueError(f"Pointwise helper conflicts with source mapping: {p}")
    store = "runtime/flash/FlashInt8ExpertStore.mm"
    witness: dict[str, Any] = {
        "schema": "splash-hybrid-original-q4-exact-pointwise-source-compose-v1",
        "gpu_executed": False, "payload_bytes_read": 0, "files_written": 0,
        "changed_source_paths": changed,
        "source_pairs": {p: {"input_sha256": sha(sources[p]), "output_sha256": sha(composed[p])}
                         for p in changed},
        "original_q4_expert_routes_byte_exact": _q4_section(composed[FORWARD]) == original_q4,
        "original_q4_expert_section_sha256": sha(original_q4),
        "raw_store_source_tensor_aliases_retained": 'std::array<FlashTensor, 9> sourceTensors;' in composed[store],
        "raw_store_poison_abi_copy_retains_dispatch_name_and_geometry": True,
        "raw_store_private_poison_name_recognized": 'd.pipelineName == "private_moe_poison_route32"' in composed[store],
        "pointwise_added_workspace_bytes": 0,
        "pointwise_required_environment": "SPLASH_FLASH_MOE_POINTWISE_SEP21=1",
        "pointwise_numerical_coefficients_changed": False,
        "helpers": {p: {"sha256": sha(t), "bytes": len(t.encode())} for p, t in helpers.items()},
        "qualification": "inherited exact synthetic kernel qualification; hybrid whole-model qualification pending",
    }
    return composed, helpers, witness
