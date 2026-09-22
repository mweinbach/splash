#!/usr/bin/env python3
"""CPU-only source, linked-object and immutable-status audit for targeted R4."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("build",type=Path);parser.add_argument("--report",type=Path);args=parser.parse_args()
    build=args.build.resolve();manifest=json.loads((build/"overlay-manifest.json").read_text());parent=Path(manifest["target_r4_base_build"])
    assert sha(parent/"overlay-manifest.json")==manifest["target_r4_base_manifest_sha256"]
    spec=importlib.util.spec_from_file_location("target_overlay",Path(__file__).with_name("overlay.py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    changed=[]
    for entry in manifest["files"]:
        actual=build/"source"/entry["path"];assert sha(actual)==entry["overlay_sha256"]
        if "target_r4_parent_sha256" in entry:
            original=parent/"source"/entry["path"];assert sha(original)==entry["target_r4_parent_sha256"]
            assert module.transform(entry["path"],original.read_text()).encode()==actual.read_bytes()
            if actual.read_bytes()!=original.read_bytes():changed.append(entry["path"])
    assert sorted(changed)==sorted(["runtime/flash/FlashForward.cpp","runtime/flash/FlashForward.hpp","runtime/flash/FlashInt8ExpertStore.mm","runtime/flash/FlashWorker.mm"])
    for entry in manifest["target_r4_frozen_link_inputs"]:
        assert sha(build/entry["private_path"])==entry["sha256"]==sha(Path(entry["source_path"]))
    assert sha(build/"link-inputs.mk")==manifest["target_r4_link_make_sha256"]
    rebuilt={"FlashForward","FlashInt8ExpertStore","FlashWorker","FlashBatchForward","FlashBatchPrefill","FlashBatchVerify"}
    assert {path.stem for path in (build/"host").glob("*.o")}==rebuilt
    reused={Path(entry["private_path"]).stem for entry in manifest["target_r4_frozen_link_inputs"] if entry["category"]=="REUSED"}
    assert not rebuilt.intersection(reused)
    worker=(build/"source/runtime/flash/FlashWorker.mm").read_text();module.immutable_identity_guard(worker)
    assert worker.count('"target_bf16_r4_route_counters"')==1
    old=(parent/"source/runtime/flash/FlashForward.cpp").read_text();new=(build/"source/runtime/flash/FlashForward.cpp").read_text()
    a=old.index('    if (int8Head && prefix == "language_model.lm_head") {');z=old.index("\n  void hc(",a);assert old[a:z] in new
    a=old.index("  bool cachedHCUp(");z=old.index("\n};",a);assert old[a:z] in new
    for relative in ["runtime/flash/FlashMTP.cpp","runtime/flash/FlashBatchMTPForward.cpp","runtime/flash/FlashDenseCache.cpp","runtime/flash/FlashDenseSmallRows.cpp"]:
        assert (build/"source"/relative).read_bytes()==(parent/"source"/relative).read_bytes()
    live=[]
    for dep in (build/"host").glob("*.d"):
        for token in dep.read_text().replace("\\\n"," ").split():
            if token.startswith("runtime/") and token.endswith((".h",".hpp")):live.append(token)
    assert not live
    cpu=json.loads(subprocess.check_output([str(build/"policy-cpu")],text=True));assert cpu["pass"] and not cpu["gpu_work"]
    rejected={}
    for value in ["","2","true","01"," 1","1 "]:
        env=dict(os.environ);env["SPLASH_FLASH_DECODE_BF16_DENSE"]="0";env["SPLASH_FLASH_DECODE_BF16_R4_TARGETED"]=value
        run=subprocess.run([str(build/"splash-flash"),"serve-flash-native","/target-r4-does-not-exist","16384","auto"],env=env,text=True,capture_output=True)
        assert run.returncode and "SPLASH_FLASH_DECODE_BF16_R4_TARGETED must be exactly0 or1" in run.stderr
        rejected[value]={"returncode":run.returncode,"stderr":run.stderr.strip()}
    result={"schema":"splash-private-targeted-BF16-R4-pointwise-full-closure-source-cpu-witness-v2","pass":True,
      "gpu_work":False,"model_payload_bytes_read":0,"source_files_verified":len(manifest["files"]),
      "frozen_link_inputs_verified":len(manifest["target_r4_frozen_link_inputs"]),"changed_existing_sources":changed,
      "default_projection_control_and_hc_head_mtp_sources_preserved":True,"immutable_identity_counter_placement_pass":True,
      "all_six_header_dependent_hosts_rebuilt_without_duplicate_ancestor_objects":True,
      "no_live_runtime_header_dependencies":True,"compiled_policy_cpu":cpu,"invalid_flags_rejected_before_paths_or_backend":rejected,
      "enabled_extra_admitted_workspace_bytes":1048576,"numerical_alternative":"cached roundedBF16 coefficient precision at four exactphysicalR4 tuples",
      "runtime_sha256":{name:sha(build/name) for name in ["splash-flash","splash.metallib"]},"model_quality_qualified":False}
    if args.report:
        assert not args.report.exists();args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
