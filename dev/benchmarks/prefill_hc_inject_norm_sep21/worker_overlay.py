#!/usr/bin/env python3
"""CPU/source-only HC overlay using the parent's actual expanded link closure."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/prefill_hc_inject_norm_sep21")
DEFAULT_BASE = ROOT / "build/dense-w8a8-sep21-worker-v3"
DEFAULT_BUILD = ROOT / "build/prefill-hc-inject-norm-sep21-worker-v3"
QUALIFIED = ROOT / "build/prefill-hc-inject-norm-sep21"
FLAG = "SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21"
CHANGED = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm", "dev/benchmarks/prefill4k_attribution.mm"}
TOOLS = ("worker_overlay.py", "worker_transform.py", "worker_witness.py", "worker.mk", "worker_README.md")

def sha(data):
    return hashlib.sha256(data).hexdigest()
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def transform(relative, text, path=None):
    return load(path or Path(__file__).with_name("worker_transform.py"),"private_hc_transform").transform(relative,text)
def tensor_declaration(text):
    begin=text.index("enum class FlashDType")
    return text[begin:text.index("// Raw, aligned",begin)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=DEFAULT_BASE)
    p.add_argument("--output",type=Path,default=DEFAULT_BUILD)
    p.add_argument("--qualified-report",type=Path,default=ROOT/"build/release/flash/sep21-prefill-hc-inject-norm-v1.json")
    args=p.parse_args();base=args.base.resolve();output=args.output.resolve()
    if base==output or ROOT/"build" not in output.parents or output.exists():
        raise ValueError("Choose a fresh, distinct private build directory")
    parent_path=base/"overlay-manifest.json";parent=json.loads(parent_path.read_text())
    if not parent.get("dense_w8a8_composed") or parent.get("dense_w8a8_hybrid_parent"):
        raise ValueError("Expected the frozen pure Full512 dense-v3 parent")
    parent_helper=base/"machinery/worker_overlay.py";parent_make=base/"machinery/worker.mk"
    tools={Path(r["private_path"]).name:r for r in parent["dense_w8a8_tools"]}
    for name,path in [("worker_overlay.py",parent_helper),("worker.mk",parent_make)]:
        if sha(path.read_bytes())!=tools[name]["sha256"]:raise ValueError(f"Parent machinery drift: {name}")
    helper=load(parent_helper,"authenticated_dense_parent_overlay")
    closure=helper.effective_closure(base,parent_make)
    seals,unsealed=helper.parent_input_seals(base,parent,closure)
    witness=helper.parent_artifact_witness(base)
    records={r["path"]:r for r in parent["files"]}
    if len(records)!=len(parent["files"]):raise ValueError("Duplicate parent source paths")
    deps=helper.frozen_dependency_closure(base,closure["objects"],records)
    qualified_path=QUALIFIED/"source-manifest.json";qualified=json.loads(qualified_path.read_text())
    for r in qualified["sources"]:
        if sha(Path(r["frozen"]).read_bytes())!=r["sha256"]:raise ValueError(f"Qualified frozen source drift: {r['frozen']}")
    for r in qualified["artifacts"]:
        if sha(Path(r["path"]).read_bytes())!=r["sha256"]:raise ValueError(f"Qualified artifact drift: {r['path']}")
    report_path=args.qualified_report.resolve();report=json.loads(report_path.read_text())
    provenance_path=Path(str(report_path)+".provenance.json");provenance=json.loads(provenance_path.read_text())
    expected={(r,d,c) for r in (512,1024,2048) for d in ("BF16","F32") for c in ("one_plus_weight","direct_gamma")}
    observed={(t["rows"],t["weight_dtype"],t["norm_convention"]) for t in report["timings"]}
    if (report.get("pass") is not True or report["qualified_fixtures"]!=60 or not report["in_place_and_out_of_place"] or
        report["host_guard_cases"]!=7 or report["shader_guard_cases"]!=19 or report["timing_size_bytes"]!=200 or
        report["matched_pairs"]%2 or observed!=expected or len(report["timings"])!=12 or
        any(t["warm_gpu_ms"]<150 for t in report["timings"]) or provenance["exit_code"]!=0 or
        provenance["source_manifest_sha256"]!=sha(qualified_path.read_bytes())):
        raise ValueError("Incomplete strict Root HC component qualification")
    cross=[]
    for relative in ("runtime/flash/FlashHC.cpp","runtime/flash/FlashHC.hpp","runtime/flash/FlashDescriptor.hpp",
                     "runtime/metal/CommandGraph.hpp","runtime/metal/MetalBackend.hpp","runtime/metal/abi/FlashHC.h","runtime/metal/abi/FlashHCFused.h"):
        record=next(r for r in qualified["sources"] if r["source"]==relative)
        if records[relative]["overlay_sha256"]!=record["sha256"]:raise ValueError(f"Qualified header/HC host mismatch: {relative}")
        cross.append({"path":relative,"sha256":record["sha256"]})
    component_weights=next(r for r in qualified["sources"] if r["source"]=="runtime/flash/FlashWeights.hpp")
    original_tensor=tensor_declaration(Path(component_weights["frozen"]).read_text())
    if original_tensor!=tensor_declaration((base/"source/runtime/flash/FlashWeights.hpp").read_text()):
        raise ValueError("FlashTensor/dtype declaration differs from qualified component")
    manifest=copy.deepcopy(parent)
    manifest.update({"route":"private-pure-full512-sg2tail-fma-optional-densew8-exact-prefill-hc-inject-norm-terminal-excluded-stable-identity-v3",
        "prefill_hc_composed":True,"prefill_hc_base_build":str(base),"prefill_hc_base_manifest_sha256":sha(parent_path.read_bytes()),
        "prefill_hc_base_make_path":str(parent_make),"prefill_hc_base_make_sha256":sha(parent_make.read_bytes()),
        "prefill_hc_parent_helper_sha256":sha(parent_helper.read_bytes()),"prefill_hc_parent_input_seals":seals,
        "prefill_hc_parent_owned_inputs_without_prior_individual_seal":unsealed,"prefill_hc_parent_artifact_witness":witness,
        "prefill_hc_effective_parent_objects":[str(x) for x in closure["objects"]],
        "prefill_hc_effective_parent_airs":[str(x) for x in closure["airs"]],"prefill_hc_parent_dependencies":deps,
        "prefill_hc_qualified_manifest_sha256":sha(qualified_path.read_bytes()),
        "prefill_hc_component_report_sha256":sha(report_path.read_bytes()),"prefill_hc_component_provenance_sha256":sha(provenance_path.read_bytes()),
        "prefill_hc_component_actual_model_inputs":False,"prefill_hc_whole_model_qualified":False,
        "prefill_hc_cross_pins":cross,"prefill_hc_flash_tensor_declaration_sha256":sha(original_tensor.encode()),
        "prefill_hc_flag":FLAG,"prefill_hc_default":0,"prefill_hc_numerical_change":False,
        "prefill_hc_added_gpu_workspace_bytes":0,"prefill_hc_added_immutable_buffers":0,
        "prefill_hc_scope":"singleton main nonverification rows512..2048; injection immediately followed by audited norm; PLE-between and terminal mixer excluded",
        "prefill_hc_diagnostic_delta":"native qualified sticky OR4 for nonfinite stored updated/normalized; no changed finite arithmetic",
        "prefill_hc_counters_are_encoding_only":True,"prefill_hc_mutable_counters_outside_identity":True,
        "prefill_hc_link_inputs":[],"prefill_hc_tools":[],"files":[],"gpu_executed":False,"payload_bytes_read":0})
    changed=[]
    for relative,record in records.items():
        original=(base/"source"/relative).read_bytes()
        if sha(original)!=record["overlay_sha256"]:raise ValueError(f"Parent source drift: {relative}")
        data=transform(relative,original.decode()).encode()
        if data!=original:changed.append(relative)
        write(output/"source"/relative,data)
        manifest["files"].append({**record,"prefill_hc_changed":data!=original,
            "prefill_hc_base_overlay_sha256":sha(original),"overlay_sha256":sha(data)})
    if set(changed)!=CHANGED:raise ValueError(f"Unexpected transformed sources: {changed}")
    for name in ("bridge.hpp","candidate.metal"):
        relative=PRIVATE/name;record=next(r for r in qualified["sources"] if r["source"]==relative.as_posix())
        data=Path(record["frozen"]).read_bytes();write(output/"source"/relative,data)
        manifest["files"].append({"path":relative.as_posix(),"new_prefill_hc_file":True,
            "qualified_component_source":record["frozen"],"overlay_sha256":sha(data)})
    for name in ("worker_bridge.hpp","worker_policy_cpu.cpp"):
        relative=PRIVATE/name;data=(ROOT/relative).read_bytes();write(output/"source"/relative,data)
        manifest["files"].append({"path":relative.as_posix(),"new_prefill_hc_file":True,
            "repository_sha256":sha(data),"overlay_sha256":sha(data)})
    inputs={"REUSED":[],"CORE":[],"AIRS":[]}
    def freeze(path,category,relative):
        data=path.read_bytes();write(output/relative,data);inputs[category].append(relative.as_posix())
        manifest["prefill_hc_link_inputs"].append({"source_path":str(path),"private_path":relative.as_posix(),"category":category,"sha256":sha(data)})
    for path in closure["objects"]:
        if path.stem not in {"FlashForward","FlashWorker"}:
            freeze(path,"CORE" if path in closure["core"] else "REUSED",Path("reused/base")/path.relative_to(base))
    for path in closure["airs"]:freeze(path,"AIRS",Path("reused/base")/path.relative_to(base))
    candidate_air=next(Path(r["path"]) for r in qualified["artifacts"] if Path(r["path"]).name=="candidate.air")
    freeze(candidate_air,"AIRS",Path("reused/qualified-hc/candidate.air"))
    for record in deps:
        if record["dependency_metadata_present"]:
            path=Path(record["source_path"]);relative=Path("qualification/parent-dependencies")/path.relative_to(base)
            write(output/relative,path.read_bytes());record["private_path"]=relative.as_posix()
    make="\n".join(f"{key} := "+" ".join("$(BUILD)/"+v for v in values) for key,values in inputs.items())+"\n"
    write(output/"link-inputs.mk",make.encode());manifest["prefill_hc_link_make_sha256"]=sha(make.encode())
    manifest["prefill_hc_changed_files"]=changed
    for name in TOOLS:
        source=ROOT/PRIVATE/name;relative=Path("machinery")/name;data=source.read_bytes();write(output/relative,data)
        manifest["prefill_hc_tools"].append({"source_path":str(source),"private_path":relative.as_posix(),"sha256":sha(data)})
    write(output/"machinery/parent_overlay.py",parent_helper.read_bytes())
    snapshots={"base-overlay-manifest.json":parent_path,"base-worker.mk":parent_make,
        "base-cpu-witness-v1.json":Path(witness["source_path"]),"source-manifest.json":qualified_path,
        "hc-component-report.json":report_path,"hc-component-provenance.json":provenance_path,
        "base-link-inputs.mk":base/"link-inputs.mk"}
    manifest["prefill_hc_qualification_snapshots"]=[]
    for name,path in snapshots.items():
        relative=Path("qualification")/name;data=path.read_bytes();write(output/relative,data)
        manifest["prefill_hc_qualification_snapshots"].append({"private_path":relative.as_posix(),"source_path":str(path),"sha256":sha(data)})
    for name in ("splash-flash.config","splash.metallib.config"):
        write(output/name,(base/name).read_bytes().rstrip(b"\n")+b"-exact-prefill-hc-inject-norm-terminal-excluded-stable-identity-sep21-v3\n")
    write(output/"overlay-manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
    write(output/"base-build.txt",(str(base)+"\n").encode())
    print(json.dumps({"prepared":str(output),"frozen_sources":len(manifest["files"]),
        "parent_objects":len(closure["objects"]),"parent_airs":len(closure["airs"]),
        "frozen_link_inputs":len(manifest["prefill_hc_link_inputs"]),"gpu_work":False,"payload_bytes_read":0,"added_gpu_workspace_bytes":0}))
if __name__=="__main__":main()
