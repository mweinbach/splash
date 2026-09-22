#!/usr/bin/env python3
"""Private coexisting BF16 small-row selector; source-only/CPU preparation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HEADER = "runtime/flash/FlashDecodeBF16DensePolicy.hpp"

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def replace(text: str, before: str, after: str, count: int = 1) -> str:
    if text.count(before) != count: raise ValueError(f"BF16 selector source drift: {before!r}")
    return text.replace(before, after)

def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashForward.hpp":
        return replace(text, "  [[nodiscard]] uint64_t qsaOutF32N32EncodedCalls() const;", """  [[nodiscard]] bool decodeBF16DenseEnabled() const;
  [[nodiscard]] std::string decodeBF16DensePolicy() const;
  [[nodiscard]] uint64_t decodeBF16DenseEncodedCalls() const;
  [[nodiscard]] uint64_t decodeBF16DenseEncodedRows() const;
  [[nodiscard]] uint64_t qsaOutF32N32EncodedCalls() const;""")
    if relative == "runtime/flash/FlashForward.cpp":
        text = '#include "flash/FlashDecodeBF16DensePolicy.hpp"\n' + text
        text = replace(text, "  const bool cacheFloat = fusionEnabled(\"SPLASH_FLASH_FLOAT_DENSE_CACHE\");", """  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");
  const bf16_decode::Policy decodeBF16Policy = bf16_decode::requested();
  std::unique_ptr<FlashDenseSmallRowsWorkspace> decodeBF16Workspace;
  uint64_t decodeBF16Calls = 0, decodeBF16Rows = 0;""")
        text = replace(text, "    descriptor.validate();", "    bf16_decode::validateDependencies(decodeBF16Policy);\n    descriptor.validate();")
        text = replace(text, "    if (captureRoutes)\n", "    if (decodeBF16Policy.enabled)\n      decodeBF16Workspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend);\n    if (captureRoutes)\n")
        anchor = "    if (int8Head && prefix == \"language_model.lm_head\") {"
        branch = '''    bf16_decode::validateFrozen(decodeBF16Policy);
    if (decodeBF16Policy.enabled && denseCache && decodeBF16Workspace && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      bool selected = bf16_decode::geometry(decodeBF16Policy, prefix, rows, projection.outputSize, projection.inputSize);
      if (selected && decodeBF16Policy.scope == bf16_decode::Scope::ExistingF32Route) {
        selected = floatDenseCache && rows >= 2 && rows <= 16 && floatDenseCache->contains(prefix) &&
            (!selectiveFloat || bool(flashFloatDenseSmallRowsPolicy(prefix, rows,
                projection.outputSize, projection.inputSize, projection.bits, projection.groupSize)));
      }
      if (selected) {
        const auto tile = rows <= 8
            ? (projection.outputSize >= 1024 ? FlashDenseSmallRowsTile::M8N128 : FlashDenseSmallRowsTile::M8N64)
            : (projection.outputSize >= 1024 ? FlashDenseSmallRowsTile::M16N128 : FlashDenseSmallRowsTile::M16N64);
        addDenseBF16SmallRows(backend, graph, input, denseCache->tensor(prefix), output,
            diagnostics, rows, *decodeBF16Workspace, tile);
        ++decodeBF16Calls; decodeBF16Rows += rows;
        return;
      }
    }
'''
        text = replace(text, anchor, branch + anchor)
        text = replace(text, "    hcUpCounters.recordAttempt(rows);", '''    hcUpCounters.recordAttempt(rows);
    bf16_decode::validateFrozen(decodeBF16Policy);
    if (decodeBF16Workspace && denseCache && denseCache->contains(prefix) &&
        bf16_decode::geometry(decodeBF16Policy, prefix, rows, 10240, 320)) {
      // R1/R2 have no eligible original F32 HC-up tile. Preserve that counter
      // classification while the independent BF16 counters record this route.
      ++hcUpCounters.skippedUnsupportedGeometry;
      const auto rawUp = bf(Scratch::HCUp, rows, kHyper);
      addDenseBF16SmallRows(backend, graph, activated, denseCache->tensor(prefix), rawUp,
          diagnostics, rows, *decodeBF16Workspace, FlashDenseSmallRowsTile::M8N128);
      addHCMix(graph, normalized, rawUp, mixed,
          {rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)});
      ++decodeBF16Calls; decodeBF16Rows += rows;
      return true;
    }''')
        marker = "  total += roundAllocation(uint64_t{16} * 32768 * 2);"
        text = replace(text, marker, marker + "\n  const auto decodePolicy = bf16_decode::requested();\n  bf16_decode::validateDependencies(decodePolicy);\n  total += bf16_decode::workspaceBytes(decodePolicy);")
        text = replace(text, "      (impl_->smallDenseWorkspace ? \";dense-all-rows-bf16-static-operands-padded-m8\" : \"\") +", """      (impl_->decodeBF16Policy.enabled ? std::string(";") + bf16_decode::identity(impl_->decodeBF16Policy) : "") +
      (impl_->smallDenseWorkspace ? ";dense-all-rows-bf16-static-operands-padded-m8" : "") +""")
        text = replace(text, "uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {", '''bool FlashForward::decodeBF16DenseEnabled() const {
  if (!impl_) throw std::logic_error("BF16 decode selector uninitialized");
  bf16_decode::validateFrozen(impl_->decodeBF16Policy);
  return impl_->decodeBF16Policy.enabled;
}
std::string FlashForward::decodeBF16DensePolicy() const {
  (void)decodeBF16DenseEnabled();
  return bf16_decode::identity(impl_->decodeBF16Policy);
}
uint64_t FlashForward::decodeBF16DenseEncodedCalls() const { return impl_ ? impl_->decodeBF16Calls : 0; }
uint64_t FlashForward::decodeBF16DenseEncodedRows() const { return impl_ ? impl_->decodeBF16Rows : 0; }
uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {''')
        return replace(text, "  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());",
                       "  if (impl_->decodeBF16Workspace) reject(impl_->decodeBF16Workspace->paddedInput());\n  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());")
    if relative == "runtime/flash/FlashInt8ExpertStore.mm":
        text = '#include "flash/FlashDecodeBF16DensePolicy.hpp"\n' + text
        return replace(text, "    numericalIdentity = hash(derivative.data(), derivative.size());", """    const auto denseDecodePolicy = bf16_decode::requested();
    bf16_decode::validateDependencies(denseDecodePolicy);
    if (denseDecodePolicy.enabled)
      derivative += std::string("target_dense_decode_policy=") + bf16_decode::identity(denseDecodePolicy) + "\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());""")
    if relative == "runtime/flash/FlashWorker.mm":
        text = '#include "flash/FlashDecodeBF16DensePolicy.hpp"\n' + text
        text = replace(text, "      (void)gathered_mpp::requestedMaximumRows();",
                       "      bf16_decode::validateDependencies(bf16_decode::requested());\n      (void)gathered_mpp::requestedMaximumRows();")
        marker = '      << R"(,"target_gathered_mpp_enabled":)"'
        return replace(text, marker, '''      << R"(,"target_decode_bf16_dense_enabled":)" << (forward_.decodeBF16DenseEnabled() ? "true" : "false")
      << R"(,"target_decode_bf16_dense_numerical_policy":)" << json::quote(forward_.decodeBF16DensePolicy())
      << R"(,"target_decode_bf16_dense_graph_calls":)" << forward_.decodeBF16DenseEncodedCalls()
      << R"(,"target_decode_bf16_dense_graph_rows":)" << forward_.decodeBF16DenseEncodedRows()
      << R"(,"target_decode_bf16_dense_counter_scope":"graph construction for known main attention/shared/PLE roles at physical rows1,2,4,8,16 and HCup1/2 only; HCdown, HCup4plus, trained MTP and vocabulary excluded")"
''' + marker)
    return text

def generate(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if ROOT / "build" not in output.parents or output == base or output.exists():
        raise ValueError("requires fresh private output beneath build")
    parent_bytes = (base / "overlay-manifest.json").read_bytes(); parent = json.loads(parent_bytes)
    if not parent.get("gathered_mpp_composed") or not parent.get("qsa_bulk_sg8_available"):
        raise ValueError("requires combined capped gathered MPP + exact SG8 bulk parent")
    content, records = {}, []
    for entry in parent["files"]:
        relative = entry["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts: raise ValueError("unsafe source path")
        original = (base / "source" / relative).read_bytes()
        if sha(original) != entry["overlay_sha256"]: raise ValueError(f"sealed source drift: {relative}")
        modified = transform(relative, original.decode()).encode(); content[relative] = modified
        records.append({**entry, "bf16_decode_parent_sha256": sha(original), "bf16_decode_changed": original != modified,
                        "overlay_sha256": sha(modified)})
    for relative in ["runtime/flash/FlashMTP.cpp", "runtime/flash/FlashBatchMTPForward.cpp", "runtime/flash/FlashDenseCache.cpp",
                     "runtime/flash/FlashDenseSmallRows.cpp", "runtime/metal/kernels/shared/flash_gathered_mpp.metal"]:
        if content[relative] != (base / "source" / relative).read_bytes(): raise AssertionError(f"protected source changed: {relative}")
    content[HEADER] = (Path(__file__).parent / "FlashDecodeBF16DensePolicy.hpp").read_bytes()
    records.append({"path": HEADER, "new_private_file": True, "patched": True, "overlay_sha256": sha(content[HEADER])})
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-bulk-gathered-mpp-bf16-main-smallrows-hcup1_2-selector-sep21-v2",
                     "bf16_decode_composed": True, "bf16_decode_base_build": str(base),
                     "bf16_decode_base_manifest_sha256": sha(parent_bytes), "bf16_decode_overlay_sha256": sha(Path(__file__).read_bytes()),
                     "bf16_decode_extra_workspace_bytes_when_enabled": 1048576, "gpu_executed": False,
                     "payload_bytes_read": 0, "files": records})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bf16-decode-sep21-", dir=output.parent) as temp:
        staged = Path(temp) / "output"
        for relative, data in content.items():
            path = staged / "source" / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n"); staged.rename(output)
    return {"prepared": str(output), "files": len(records), "gpu_work": False, "payload_bytes_read": 0,
            "hc_fastpath_and_head_and_prefill_kernels_unchanged": True, "numerical_alternative": "cached BF16 coefficients"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-bf16-sep21-v2")
    args = parser.parse_args(); print(json.dumps(generate(args.base, args.output)))
