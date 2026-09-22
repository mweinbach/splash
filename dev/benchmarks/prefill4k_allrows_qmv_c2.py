#!/usr/bin/env python3
"""Optional C2 private follow-on transform; C1 files remain immutable.

Compose after prefill4k_allrows_qmv.transform and stage extra_files AFTER C1's
extra_files (the augmented header replaces C1's header). No inference on import.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FLAG = "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS"
POLICY = "private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c2-rows1to16-v1"
MARKER = "// Private gathered I8 QMV optional C2 Store overlay v1."
HEADER_RELATIVE = "runtime/flash/FlashGatheredI8QMV.hpp"
METAL_RELATIVE = "runtime/metal/kernels/shared/flash_gathered_i8_qmv_c2.metal"


def c1_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("prefill4k_qmv_c1_component", ROOT / "dev/benchmarks/prefill4k_allrows_qmv.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import private C1 component")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    actual = text.count(before)
    if actual != count:
        raise RuntimeError(f"Private QMV C2 source drift: expected{count}, got{actual}: {before!r}")
    return text.replace(before, after)


HEADER = r'''
#ifndef __METAL_VERSION__
namespace splash::flash::gathered_i8_qmv {
inline constexpr const char *kColumnsFlag = "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS";
inline constexpr std::string_view kC2Policy =
    "private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c2-rows1to16-v1";
inline uint32_t requestedColumns() {
  const char *raw = std::getenv(kColumnsFlag);
  uint32_t columns = 1;
  if (raw) {
    const std::string_view value(raw);
    if (value == "1") columns = 1;
    else if (value == "2") columns = 2;
    else throw std::invalid_argument("SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS must be exactly1 or2");
  }
  if (columns == 2 && !requested())
    throw std::invalid_argument("private QMV columns2 requires SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV=1");
  return columns;
}
inline std::string_view numericalPolicy(uint32_t columns) {
  if (columns == 1) return kPolicy;
  if (columns == 2) return kC2Policy;
  throw std::invalid_argument("private QMV columns policy outside1 or2");
}
} // namespace splash::flash::gathered_i8_qmv
#endif
'''


def extra_files() -> dict[str, str]:
    c1 = c1_module().extra_files()
    return {HEADER_RELATIVE: c1[HEADER_RELATIVE] + HEADER,
            METAL_RELATIVE: (ROOT / "dev/benchmarks/prefill4k_allrows_qmv_c2.metal").read_text()}


def transform(relative: str, text: str) -> str:
    if relative not in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        return text
    if c1_module().MARKER not in text:
        raise RuntimeError("C2 transform requires the C1 Store transform first")
    if MARKER in text:
        raise RuntimeError("private QMV C2 transform already applied")
    text = MARKER + "\n" + text
    if relative.endswith(".hpp"):
        return replace(text, "  [[nodiscard]] bool gatheredQMVEnabled() const;",
            "  [[nodiscard]] bool gatheredQMVEnabled() const;\n  [[nodiscard]] uint32_t gatheredQMVColumns() const;")
    text = replace(text, "  const bool gatheredQMV = gathered_i8_qmv::requested();",
        "  const bool gatheredQMV = gathered_i8_qmv::requested();\n  const uint32_t gatheredColumns = gathered_i8_qmv::requestedColumns();")
    text = replace(text, "std::string(gathered_i8_qmv::kPolicy)",
        "std::string(gathered_i8_qmv::numericalPolicy(gatheredColumns))")
    text = replace(text, "  return impl_->gatheredQMV;", """  if (gathered_i8_qmv::requestedColumns() != impl_->gatheredColumns)
    fail("private gathered I8 QMV columns changed after Store construction");
  return impl_->gatheredQMV;""")
    text = replace(text, "void FlashInt8ExpertStore::addGatheredQMVGateUp(", """uint32_t FlashInt8ExpertStore::gatheredQMVColumns() const {
  (void)gatheredQMVEnabled();
  return impl_->gatheredColumns;
}
void FlashInt8ExpertStore::addGatheredQMVGateUp(""")
    text = replace(text, '  graph.add("flash_gathered_i8_qmv_gate_up_sg4_c1",',
        '  graph.add(impl_->gatheredColumns == 2 ? "flash_gathered_i8_qmv_gate_up_sg4_c2" :\n      "flash_gathered_i8_qmv_gate_up_sg4_c1",')
    text = replace(text, "{g.gateColumnGroups, rows, selections}",
        "{g.gateColumnGroups / impl_->gatheredColumns, rows, selections}")
    text = replace(text, '  graph.add("flash_gathered_i8_qmv_down_sg4_c1",',
        '  graph.add(impl_->gatheredColumns == 2 ? "flash_gathered_i8_qmv_down_sg4_c2" :\n      "flash_gathered_i8_qmv_down_sg4_c1",')
    return replace(text, "{g.downColumnGroups, rows, selections}",
        "{g.downColumnGroups / impl_->gatheredColumns, rows, selections}")


def cpu_self_test(source: Path) -> dict:
    c1 = c1_module()
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        original = (source / relative).read_text()
        c1_text = c1.transform(relative, original)
        c2_text = transform(relative, c1_text)
        try:
            transform(relative, c2_text)
        except RuntimeError:
            pass
        else:
            raise AssertionError("C2 duplicate transform admitted")
        if relative.endswith(".mm"):
            for primitive in ("backend.allocateBuffer(", "backend.wrapSharedMemory(", "std::make_shared<Mapping>("):
                if c2_text.count(primitive) != original.count(primitive):
                    raise AssertionError("C2 added mapping/allocation")
            if c2_text.count("/ impl_->gatheredColumns") != 2:
                raise AssertionError("C2 exact half-grid dispatch missing")
            start = original.index("void FlashInt8ExpertStore::addGateUp(")
            end = original.index("} // namespace splash::flash", start)
            if original[start:end] not in c2_text:
                raise AssertionError("original MPP methods changed")
    files = extra_files()
    if POLICY not in files[HEADER_RELATIVE]:
        raise AssertionError("C2 helper/Python policy differs")
    c1_shader = c1.extra_files()[c1.METAL_RELATIVE]
    c2_shader = files[METAL_RELATIVE]
    for shader in (c1_shader, c2_shader):
        if shader.count("k += 32") != 2 or shader.count("#pragma clang fp contract(off)") != 3:
            raise AssertionError("per-output laneK/F32 contract policy differs")
    seed = "splash.private-allrows-target-v1\nsource=s\nstore=t\npolicy=mpp\nmtp=original-trained-bank\n"
    if len({hashlib.sha256((seed + suffix).encode()).hexdigest() for suffix in
            ("", f"small_row_policy={c1.POLICY}\n", f"small_row_policy={POLICY}\n")}) != 3:
        raise AssertionError("MPP/C1/C2 derivative identity collision")
    return {"cpu_source_checks": "passed", "gpu_work": False, "extra_allocations": 0,
            "c1_default_preserved": True, "c2_groups": {"gate": [80, "rows", 10], "down": [320, "rows", 10]},
            "paired_gpu_byte_qualification": "pending"}


def stage(source: Path, destination: Path) -> dict:
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("C2 staging requires a NEW destination")
    destination.mkdir(parents=True)
    c1 = c1_module()
    witness = {}
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        before = (source / relative).read_text()
        after = transform(relative, c1.transform(relative, before))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
        witness[relative] = {"input_sha256": hashlib.sha256(before.encode()).hexdigest(),
                             "output_sha256": hashlib.sha256(after.encode()).hexdigest()}
    files = c1.extra_files() | extra_files()
    for relative, content in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        witness[relative] = {"output_sha256": hashlib.sha256(content.encode()).hexdigest()}
    result = {"c1_policy": c1.POLICY, "c2_policy": POLICY, "flag": c1.FLAG, "columns_flag": FLAG,
              "gpu_work": False, "files": witness}
    (destination / "qmv-overlay-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--cpu-self-test", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_test:
        print(json.dumps(cpu_self_test(args.source), indent=2))
    else:
        if args.destination is None:
            parser.error("staging requires --destination")
        print(json.dumps(stage(args.source, args.destination), indent=2))
