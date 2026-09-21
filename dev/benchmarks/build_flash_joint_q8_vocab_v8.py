"""Fresh CPU-only private host build for the joint Q8 vocabulary screen."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = PROJECT / "build/flash-trained-head-joint-q8-v8-v2"


def replacement(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Private snapshot edit requires exactly one match: {before[:100]}")
    return text.replace(before, after)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    build = args.build.resolve()
    if build.exists():
        raise FileExistsError(f"Choose a fresh private build directory: {build}")
    if not 1 <= args.jobs <= 8:
        parser.error("CPU compiler jobs must be 1..8")
    snapshot = build / "snapshot"
    shutil.copytree(PROJECT / "runtime", snapshot / "runtime")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    original_hashes = {str(path.relative_to(snapshot)): sha(path) for path in (snapshot / "runtime").rglob("*") if path.is_file()}
    source_dir = snapshot / "benchmarks"
    source_dir.mkdir()
    for name in ("flash_trained_head_profile_v8_oracle.mm", "flash_joint_q8_vocab_v8_oracle.mm"):
        shutil.copy2(PROJECT / "dev/benchmarks" / name, source_dir / name)
    shutil.copy2(PROJECT / "build/flash-trained-head-profile-v8-v2/frozen-environment.json", build / "frozen-environment.json")
    shutil.copy2(PROJECT / "build/flash-trained-head-profile-v8-v2/splash.metallib", build / "splash.metallib")
    mtp_header = snapshot / "runtime/flash/FlashMTP.hpp"
    mtp_header.write_text(replacement(mtp_header.read_text(),
        "  void truncate(FlashMTPState &state, uint64_t retainedLength);",
        "  void truncate(FlashMTPState &state, uint64_t retainedLength);\n"
        "  // Private completed-command CPU diagnostics; no production API or GPU work.\n"
        "  [[nodiscard]] std::vector<metal::MetalBuffer> privateQSAStateBuffersV8(const FlashMTPState &state) const;\n"
        "  void privateCopyStateV8(FlashMTPState &destination, const FlashMTPState &source);"))
    mtp = snapshot / "runtime/flash/FlashMTP.cpp"
    mtp.write_text(replacement(mtp.read_text(),
        "void FlashMTPForward::truncate(FlashMTPState &request, uint64_t retainedLength) {",
        "std::vector<metal::MetalBuffer> FlashMTPForward::privateQSAStateBuffersV8(const FlashMTPState &state) const {\n"
        "  if (!impl_) throw std::logic_error(\"private state diagnostics require initialized head\");\n"
        "  std::lock_guard lock(impl_->mutex);\n"
        "  if (!ownsState(state)) throw std::invalid_argument(\"private state diagnostics require healthy owned state\");\n"
        "  const auto &q = state.impl_->qsa;\n"
        "  return {q.keys, q.values, q.rawIndexKeys, q.pooledKeys, q.indexPositions};\n"
        "}\n"
        "void FlashMTPForward::privateCopyStateV8(FlashMTPState &destination, const FlashMTPState &source) {\n"
        "  if (!impl_) throw std::logic_error(\"private state copy requires initialized head\");\n"
        "  std::lock_guard lock(impl_->mutex);\n"
        "  if (!ownsState(destination) || !ownsState(source)) throw std::invalid_argument(\"private state copy requires healthy same-owner states\");\n"
        "  const auto &a=source.impl_->qsa; auto &b=destination.impl_->qsa;\n"
        "  const std::array from{a.keys,a.values,a.rawIndexKeys,a.pooledKeys,a.indexPositions};\n"
        "  const std::array to{b.keys,b.values,b.rawIndexKeys,b.pooledKeys,b.indexPositions};\n"
        "  for (size_t index=0;index<from.size();++index) {\n"
        "    if (!from[index].contents() || !to[index].contents() || from[index].sizeBytes()!=to[index].sizeBytes()) throw std::logic_error(\"private QSA state copy extent differs\");\n"
        "    if (from[index].contents()!=to[index].contents()) std::memcpy(to[index].contents(),from[index].contents(),from[index].sizeBytes());\n"
        "  }\n"
        "  destination.impl_->length=source.impl_->length;\n"
        "}\n\n"
        "void FlashMTPForward::truncate(FlashMTPState &request, uint64_t retainedLength) {"))
    joint_header = snapshot / "runtime/flash/FlashBatchMTPForward.hpp"
    joint_header.write_text(replacement(joint_header.read_text(),
        "  [[nodiscard]] const char *projectionRouteSemantics() const noexcept;",
        "  [[nodiscard]] const char *projectionRouteSemantics() const noexcept;\n"
        "  [[nodiscard]] bool privateQ8VocabularyEnabledV8() const noexcept;\n"
        "  [[nodiscard]] const char *privateVocabularyRouteV8() const noexcept;\n"
        "  [[nodiscard]] std::string privateVocabularyIdentityV8() const;"))
    joint = snapshot / "runtime/flash/FlashBatchMTPForward.cpp"
    text = replacement(joint.read_text(), '#include "flash/FlashDenseSmallRows.hpp"',
        '#include "flash/FlashDenseSmallRows.hpp"\n#include "flash/FlashInt8Head.hpp"')
    text = replacement(text, "  std::unique_ptr<FlashDenseSmallRowsWorkspace> vocabularyWorkspace;",
        "  std::unique_ptr<FlashDenseSmallRowsWorkspace> vocabularyWorkspace;\n"
        "  const bool privateQ8Vocabulary;\n"
        "  std::unique_ptr<FlashInt8Head> privateQ8Head;")
    text = replacement(text, "        denseCache(head.batchDenseCache()) {",
        "        denseCache(head.batchDenseCache()),\n"
        '        privateQ8Vocabulary(enabled("SPLASH_PRIVATE_JOINT_Q8_VOCAB")) {')
    text = replacement(text, "    if (cachedVocabulary && maximumLanes > 1)\n      vocabularyWorkspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend, kWidth);",
        "    if (privateQ8Vocabulary && maximumLanes > 1) {\n"
        "      privateQ8Head = std::make_unique<FlashInt8Head>(backend, weights);\n"
        "      if (!privateQ8Head->usesOriginalCodeStorage()) throw std::invalid_argument(\"private joint candidate requires original zero-copy UINT8 codes\");\n"
        "    } else if (cachedVocabulary && maximumLanes > 1)\n"
        "      vocabularyWorkspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend, kWidth);")
    text = replacement(text, "uint64_t FlashBatchMTPForward::workspacePlannedBytes(uint32_t capacity,",
        "bool FlashBatchMTPForward::privateQ8VocabularyEnabledV8() const noexcept { return impl_ && impl_->privateQ8Vocabulary; }\n"
        "const char *FlashBatchMTPForward::privateVocabularyRouteV8() const noexcept {\n"
        "  return impl_ && impl_->privateQ8Head ? impl_->privateQ8Head->codeStorageSemantics() : \"original-bf16-cached-vocabulary-m8-n128-v1\";\n"
        "}\n"
        "std::string FlashBatchMTPForward::privateVocabularyIdentityV8() const {\n"
        "  return impl_ && impl_->privateQ8Head ? impl_->privateQ8Head->identitySha256() : std::string{};\n"
        "}\n"
        "uint64_t FlashBatchMTPForward::workspacePlannedBytes(uint32_t capacity,")
    text = replacement(text, "    if (mode == FlashMTPLogits::Last && logitRows >= 2 &&\n        impl_->cachedVocabulary && impl_->vocabularyWorkspace) {",
        "    if (mode == FlashMTPLogits::Last && logitRows >= 2 && logitRows <= 4 && impl_->privateQ8Head) {\n"
        "      impl_->privateQ8Head->addProjection(graph, headInput, logits, diag, logitRows);\n"
        "    } else if (mode == FlashMTPLogits::Last && logitRows >= 2 &&\n"
        "        impl_->cachedVocabulary && impl_->vocabularyWorkspace) {")
    joint.write_text(text)
    flags = ["-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", f"-I{snapshot / 'runtime'}", "-mmacosx-version-min=27.0", "-DSPLASH_INT8_EXPERIMENT=1"]
    dev_mk = (PROJECT / "dev/flash.mk").read_text()
    def source_set(variable: str) -> list[str]:
        match = re.search(rf"^{variable} := (.*?)(?=\n[A-Z_]+\s*:?=)", dev_mk, re.M | re.S)
        if not match:
            raise ValueError(variable)
        return match.group(1).replace("\\\n", " ").split()
    sources = [name for name in source_set("FLASH_CPP_SOURCES") + source_set("FLASH_MM_SOURCES") if not name.endswith("FlashWorker.mm")]
    sources += ["runtime/metal/MetalBackend.mm", "runtime/metal/DeviceCapabilities.cpp", "runtime/engine/Protocol.cpp", "runtime/engine/MemoryGovernor.cpp"]
    commands = []
    objects = []
    for name in sources:
        source = snapshot / name
        output = build / "objects" / (name.removeprefix("runtime/").rsplit(".", 1)[0] + ".o")
        output.parent.mkdir(parents=True, exist_ok=True)
        objects.append(output)
        commands.append(["xcrun", "-sdk", "macosx", "clang++", *flags, *(["-fobjc-arc"] if name.endswith(".mm") else []), "-MMD", "-MP", "-c", str(source), "-o", str(output)])
    (build / "host-build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    def compile_one(command: list[str]) -> dict:
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_one, commands))
    (build / "host-build-results.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = [result for result in results if result["returncode"]]
    if failed:
        for result in failed:
            print(result["stderr"])
        return 1
    executable = build / "flash-joint-q8-vocab-v8-oracle"
    link = ["xcrun", "-sdk", "macosx", "clang++", *flags, "-fobjc-arc", str(source_dir / "flash_joint_q8_vocab_v8_oracle.mm"), *(str(path) for path in objects), "-framework", "Foundation", "-framework", "Metal", "-framework", "IOKit", "-o", str(executable)]
    (build / "host-link-command.json").write_text(json.dumps(link, indent=2) + "\n")
    result = subprocess.run(link, cwd=PROJECT, capture_output=True, text=True)
    (build / "host-link-result.json").write_text(json.dumps({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2) + "\n")
    if result.returncode:
        print(result.stderr)
        return result.returncode
    help_result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=True)
    (build / "help.txt").write_text(help_result.stdout)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot_hashes = {str(path.relative_to(snapshot)): sha(path) for path in snapshot.rglob("*") if path.is_file()}
    modified = [name for name, value in original_hashes.items() if snapshot_hashes[name] != value]
    expected_modified = ["runtime/flash/FlashMTP.hpp", "runtime/flash/FlashMTP.cpp", "runtime/flash/FlashBatchMTPForward.hpp", "runtime/flash/FlashBatchMTPForward.cpp"]
    if sorted(modified) != sorted(expected_modified):
        raise ValueError(f"Unexpected private source drift: {modified}")
    manifest = {"schema": "splash-private-joint-q8-vocabulary-build-v8", "gpu_executed": False, "all_host_objects_fresh": True, "host_object_count": len(objects), "command_timing_abi_bytes": 200, "production_source_modified": False, "private_modified_source_paths": modified, "candidate_switch": "SPLASH_PRIVATE_JOINT_Q8_VOCAB=0|1 captured per instance", "candidate_rows": "Last real logitRows2..4 only", "operand_semantics": "unchanged original Q8 UINT8 zero-copy codes, original BF16 scale/bias, group-factored F32 MPP, BF16 output", "numerical_contract": "declared numerical alternative to original BF16 dequantized cached vocabulary; strict full-vocabulary relative-L2<=1e-4; never relaxed", "binary_sha256": sha(executable), "metallib_sha256": sha(build / "splash.metallib"), "frozen_environment_json_sha256": sha(build / "frozen-environment.json"), "original_source_sha256": original_hashes, "snapshot_source_sha256": snapshot_hashes, "object_sha256": {str(path.relative_to(build)): sha(path) for path in objects}}
    (build / "frozen-build-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"CPU-only private build complete: {executable}; {len(objects)} fresh host objects; no GPU execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
