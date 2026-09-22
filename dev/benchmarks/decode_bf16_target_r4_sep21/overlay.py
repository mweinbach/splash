#!/usr/bin/env python3
"""Seal measured physical-R4 BF16 producers over the pointwise worker."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path("dev/benchmarks/decode_bf16_target_r4_sep21")
NARROW=Path("dev/benchmarks/decode_bf16_narrow_sep21")
CHANGED={"FlashForward","FlashInt8ExpertStore","FlashWorker","FlashBatchForward","FlashBatchPrefill","FlashBatchVerify"}
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def replace(text:str,before:str,after:str)->str:
    if text.count(before)!=1:raise ValueError(f"targetedBF16 sourceanchor drift: {before!r}")
    return text.replace(before,after)
def immutable_identity_guard(worker:str)->None:
    a=worker.index('      << R"(,"identity":{"source":)"');z=worker.index('      << R"(,"weight_format":"mlx_affine_preconverted"})"',a)
    for marker in ["targetBF16R4EncodedCalls", "targetBF16R4EncodedRows", "graph_calls", "graph_rows", "route_counters"]:
        if marker in worker[a:z]:raise AssertionError("mutabletargetcounter insideidentity")

def transform(relative:str,text:str)->str:
    if relative=="runtime/flash/FlashForward.hpp":
        return replace(text,"  [[nodiscard]] uint64_t qsaOutF32N32EncodedCalls() const;","""  [[nodiscard]] bool targetBF16R4Enabled() const;
  [[nodiscard]] uint64_t targetBF16R4EncodedCalls() const;
  [[nodiscard]] uint64_t targetBF16R4EncodedRows() const;
  [[nodiscard]] uint64_t qsaOutF32N32EncodedCalls() const;""")
    if relative not in ["runtime/flash/FlashForward.cpp","runtime/flash/FlashInt8ExpertStore.mm","runtime/flash/FlashWorker.mm"]:return text
    text='#include "dev/benchmarks/decode_bf16_target_r4_sep21/bridge.hpp"\n'+text
    if relative=="runtime/flash/FlashForward.cpp":
        text=replace(text,'  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");', '''  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");
  const bool targetBF16R4 = bf16_target_r4::requested();
  std::unique_ptr<FlashDenseSmallRowsWorkspace> targetBF16R4Workspace;
  uint64_t targetBF16R4Calls = 0, targetBF16R4Rows = 0;''')
        text=replace(text,"    descriptor.validate();","    bf16_target_r4::validateDependencies(targetBF16R4);\n    descriptor.validate();")
        text=replace(text,"    if (captureRoutes)\n","    if (targetBF16R4)\n      targetBF16R4Workspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend);\n    if (captureRoutes)\n")
        text=replace(text,'    if (int8Head && prefix == "language_model.lm_head") {','''    bf16_target_r4::validateFrozen(targetBF16R4);
    if (targetBF16R4 && rows == 4 && denseCache && targetBF16R4Workspace && denseCache->contains(prefix)) {
      const auto &source = weights.projection(prefix);
      const auto selected = source.weights && source.scales && source.biases
          ? bf16_target_r4::select(prefix, rows, source.outputSize, source.inputSize, source.bits,
              source.groupSize, source.experts, source.weights->dtype, source.scales->dtype, source.biases->dtype)
          : std::nullopt;
      if (selected) {
        const auto &cached = denseCache->tensor(prefix);
        if (cached.dtype != FlashDType::BF16 || cached.shape != std::vector<uint64_t>{source.outputSize, source.inputSize})
          throw std::logic_error("targeted R4 BF16 cachedshape/dtype differs from admitted source");
        bf16_target_r4::addProjection(backend, graph, input, cached, output, diagnostics,
            rows, *targetBF16R4Workspace, *selected);
        ++targetBF16R4Calls; targetBF16R4Rows += rows;
        return;
      }
    }
    if (int8Head && prefix == "language_model.lm_head") {''')
        marker="  total += roundAllocation(uint64_t{16} * 32768 * 2);"
        text=replace(text,marker,marker+"\n  const bool targetR4 = bf16_target_r4::requested();\n  bf16_target_r4::validateDependencies(targetR4);\n  total += bf16_target_r4::workspaceBytes(targetR4);")
        text=replace(text,'      (impl_->smallDenseWorkspace ? ";dense-all-rows-bf16-static-operands-padded-m8" : "") +','''      (impl_->targetBF16R4 ? std::string(";") + std::string(bf16_target_r4::kSemantics) : "") +
      (impl_->smallDenseWorkspace ? ";dense-all-rows-bf16-static-operands-padded-m8" : "") +''')
        text=replace(text,"uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {",'''bool FlashForward::targetBF16R4Enabled() const {
  if (!impl_) throw std::logic_error("targeted R4 BF16 uninitialized");
  bf16_target_r4::validateFrozen(impl_->targetBF16R4);
  return impl_->targetBF16R4;
}
uint64_t FlashForward::targetBF16R4EncodedCalls() const { return impl_ ? impl_->targetBF16R4Calls : 0; }
uint64_t FlashForward::targetBF16R4EncodedRows() const { return impl_ ? impl_->targetBF16R4Rows : 0; }
uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {''')
        return replace(text,"  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());","  if (impl_->targetBF16R4Workspace) reject(impl_->targetBF16R4Workspace->paddedInput());\n  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());")
    if relative=="runtime/flash/FlashInt8ExpertStore.mm":
        return replace(text,"    numericalIdentity = hash(derivative.data(), derivative.size());",'''    if (bf16_target_r4::requested())
      derivative += std::string("target_physicalR4_dense_precision_policy=") + std::string(bf16_target_r4::kSemantics) + "\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());''')
    text=replace(text,"      (void)pointwise_sep21::requested();", "      bf16_target_r4::validateDependencies(bf16_target_r4::requested());\n      (void)pointwise_sep21::requested();")
    identity='      << R"(,"target_gathered_mpp_enabled":)"'
    text=replace(text,identity,'''      << R"(,"target_bf16_r4_enabled":)" << (forward_.targetBF16R4Enabled() ? "true" : "false")
      << R"(,"target_bf16_r4_numerical_policy":)" << json::quote(std::string(bf16_target_r4::kSemantics))
'''+identity)
    anchor="  const auto requestTrace = requestCommandTraceInfo();"
    text=replace(text,anchor,'''  out << R"(,"target_bf16_r4_route_counters":{"scope":"graph construction only; physicalR4 with four exact measured role/shape/Q4orQ6G64 tuples; all other rows/roles/sourceformats retain priorroute","graph_calls":)"
      << forward_.targetBF16R4EncodedCalls()
      << R"(,"graph_rows":)" << forward_.targetBF16R4EncodedRows() << '}';
'''+anchor)
    immutable_identity_guard(text);return text

def generate(base:Path,output:Path)->dict:
    base,output=base.resolve(),output.resolve()
    if output.exists() or output==base or ROOT/"build" not in output.parents:raise ValueError("fresh separate private outputrequired")
    parent_bytes=(base/"overlay-manifest.json").read_bytes();parent=json.loads(parent_bytes)
    if not parent.get("pointwise_composed") or parent.get("bf16_decode_composed"):raise ValueError("sealedpointwisebase required; nolegacyBF16 overlay")
    content,records={},[]
    for entry in parent["files"]:
        relative=entry["path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:raise ValueError("unsafe sourcepath")
        original=(base/"source"/relative).read_bytes()
        if sha(original)!=entry["overlay_sha256"]:raise ValueError(f"sealedparentdrift: {relative}")
        changed=transform(relative,original.decode()).encode();content[relative]=changed
        records.append({**entry,"target_r4_parent_sha256":sha(original),"target_r4_changed":changed!=original,"overlay_sha256":sha(changed)})
    for relative in [PRIVATE/"bridge.hpp",PRIVATE/"policy_cpu.cpp",NARROW/"KernelParams.hpp",NARROW/"kernels.metal"]:
        content[relative.as_posix()]=(ROOT/relative).read_bytes();records.append({"path":relative.as_posix(),"new_target_r4_file":True,"overlay_sha256":sha(content[relative.as_posix()])})
    # Rebuild changed implementations and all three batch implementations that
    # consume the additive Forward.hpp getters; other header closures match.
    manifest=copy.deepcopy(parent);manifest.update({"route":"private-pointwise-gathered-exactbulk-targetedBF16-fourroles-physicalR4-full-header-closure-v2",
      "target_bf16_r4_composed":True,"target_r4_base_build":str(base),"target_r4_base_manifest_sha256":sha(parent_bytes),
      "target_r4_transform_sha256":sha(Path(__file__).read_bytes()),"target_r4_enabled_extra_workspace_bytes":1048576,
      "gpu_executed":False,"payload_bytes_read":0,"files":records,"target_r4_frozen_link_inputs":[]})
    paths={"REUSED":[],"CORE":[],"AIRS":[]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="targetbf16r4-",dir=output.parent) as temp:
        staged=Path(temp)/"output"
        for relative,data in content.items():
            path=staged/"source"/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        def freeze(src:Path,destination:Path,kind:str):
            data=src.read_bytes();path=staged/destination;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            paths[kind].append(destination.as_posix());manifest["target_r4_frozen_link_inputs"].append({"private_path":destination.as_posix(),"source_path":str(src),"sha256":sha(data),"category":kind})
        for path in sorted((base/"host").glob("*.o")):
            if path.stem not in CHANGED:freeze(path,Path("reused/parent-host")/path.name,"REUSED")
        for entry in parent["pointwise_link_inputs"]:
            src=base/entry["private_path"]
            if sha(src.read_bytes())!=entry["sha256"]:raise ValueError("pointwisefrozenlinkinput drift")
            if entry["category"]=="REUSED" and src.stem in CHANGED:continue
            freeze(src,Path("reused/ancestor")/entry["private_path"],entry["category"])
        freeze(base/"pointwise.air",Path("reused/pointwise.air"),"AIRS")
        make="\n".join(f"{key} := "+" ".join("$(BUILD)/"+p for p in values) for key,values in paths.items())+"\n"
        (staged/"link-inputs.mk").write_text(make);manifest["target_r4_link_make_sha256"]=sha(make.encode())
        (staged/"overlay-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");staged.rename(output)
    return {"prepared":str(output),"frozen_sources":len(records),"frozen_link_inputs":len(manifest["target_r4_frozen_link_inputs"]),"rebuilt_host_names":sorted(CHANGED),"gpu_work":False,"model_payload_bytes_read":0}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--base",type=Path,default=ROOT/"build/moe-pointwise-sep21-worker-v1")
    parser.add_argument("--output",type=Path,default=ROOT/"build/bf16-target-r4-pointwise-sep21-v2");args=parser.parse_args();print(json.dumps(generate(args.base,args.output)))
