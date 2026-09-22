#!/usr/bin/env python3
"""CPU-only immutable closure and scoped exact-worker witness."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from worker_overlay import ROOT, PRIVATE, DEFAULT_BUILD, FLAG, CHANGED, sha, load, transform, tensor_declaration

def section(text,begin,end):
    start=text.index(begin);return text[start:text.index(end,start)]
def identity_expression(text):
    begin='      << R"(,"target_numerical_derivative_sha256":)"'
    start=text.index(begin);return text[start:text.index('\n      << R"(',start+len(begin))]
def cpu(command,env=None):
    result=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
    return {"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
def clean_env():
    env={k:v for k,v in os.environ.items() if not k.startswith("SPLASH_FLASH_")}
    env[FLAG]="0";return env
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--build",type=Path,default=DEFAULT_BUILD);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();build=args.build.resolve()
    if args.output.exists():raise ValueError("Choose a fresh witness output")
    m=json.loads((build/"overlay-manifest.json").read_text());base=Path(m["prefill_hc_base_build"])
    helper=load(build/"machinery/parent_overlay.py","frozen_authenticated_parent_helper")
    checks={"gpu_work_and_payload_reads_zero":not m["gpu_executed"] and m["payload_bytes_read"]==0,
        "new_route_exact_and_workspace_zero":not m["prefill_hc_numerical_change"] and m["prefill_hc_added_gpu_workspace_bytes"]==0 and m["prefill_hc_added_immutable_buffers"]==0,
        "only_three_expected_sources_transformed":set(m["prefill_hc_changed_files"])==CHANGED,
        "parent_helper_frozen_and_fresh":sha((build/"machinery/parent_overlay.py").read_bytes())==m["prefill_hc_parent_helper_sha256"],
        "parent_manifest_fresh":sha((base/"overlay-manifest.json").read_bytes())==m["prefill_hc_base_manifest_sha256"],
        "parent_make_fresh":sha(Path(m["prefill_hc_base_make_path"]).read_bytes())==m["prefill_hc_base_make_sha256"],
        "private_link_make_fresh":sha((build/"link-inputs.mk").read_bytes())==m["prefill_hc_link_make_sha256"]}
    tools_mismatch=[r["private_path"] for r in m["prefill_hc_tools"] if sha((build/r["private_path"]).read_bytes())!=r["sha256"]]
    checks["worker_machinery_frozen_fresh"]=not tools_mismatch
    tools={Path(r["private_path"]).name:r for r in m["prefill_hc_tools"]}
    checks["executed_witness_and_generator_pinned"]=sha(Path(__file__).read_bytes())==tools["worker_witness.py"]["sha256"] and sha(Path(transform.__code__.co_filename).read_bytes())==tools["worker_overlay.py"]["sha256"]
    records={r["path"]:r for r in m["files"]};checks["source_paths_unique"]=len(records)==len(m["files"])
    mismatch=[]
    for relative,r in records.items():
        actual=(build/"source"/relative).read_bytes()
        if r.get("new_prefill_hc_file"):
            expected=actual
            origin=Path(r["qualified_component_source"]) if "qualified_component_source" in r else ROOT/relative
            fresh=sha(origin.read_bytes())==r["overlay_sha256"]
        else:
            original=(base/"source"/relative).read_bytes();fresh=sha(original)==r["prefill_hc_base_overlay_sha256"]
            expected=transform(relative,original.decode(),build/"machinery/worker_transform.py").encode()
        if not fresh or expected!=actual or sha(actual)!=r["overlay_sha256"]:mismatch.append(relative)
    checks["all_sources_and_strict_transform_fresh"]=not mismatch
    snapshots=m["prefill_hc_qualification_snapshots"]
    checks["all_qualification_snapshots_and_origins_fresh"]=all(sha((build/r["private_path"]).read_bytes())==r["sha256"] and sha(Path(r["source_path"]).read_bytes())==r["sha256"] for r in snapshots)
    checks["qualified_header_and_hc_host_cross_pins_fresh"]=all(records[r["path"]]["overlay_sha256"]==r["sha256"] for r in m["prefill_hc_cross_pins"])
    checks["qualified_flash_tensor_declaration_fresh"]=sha(tensor_declaration((build/"source/runtime/flash/FlashWeights.hpp").read_text()).encode())==m["prefill_hc_flash_tensor_declaration_sha256"]
    parent_closure=helper.effective_closure(base,Path(m["prefill_hc_base_make_path"]))
    checks["actual_parent_object_and_air_closure_unchanged"]=[str(x) for x in parent_closure["objects"]]==m["prefill_hc_effective_parent_objects"] and [str(x) for x in parent_closure["airs"]]==m["prefill_hc_effective_parent_airs"]
    inputs=m["prefill_hc_link_inputs"]
    input_mismatch=[r["private_path"] for r in inputs if sha((build/r["private_path"]).read_bytes())!=r["sha256"] or sha(Path(r["source_path"]).read_bytes())!=r["sha256"]]
    checks["all_frozen_link_inputs_and_origins_fresh"]=not input_mismatch
    checks["every_parent_object_retained_except_replaced_pair"]={r["source_path"] for r in inputs if r["category"] in ("REUSED","CORE")}=={str(x) for x in parent_closure["objects"] if x.stem not in ("FlashForward","FlashWorker")}
    parent_airs=[r for r in inputs if r["category"]=="AIRS" and "qualified-hc" not in r["private_path"]]
    checks["every_parent_air_retained_once"]={r["source_path"] for r in parent_airs}=={str(x) for x in parent_closure["airs"]} and len(parent_airs)==len(parent_closure["airs"])
    checks["one_root_qualified_candidate_air_added"]=len([r for r in inputs if r["category"]=="AIRS" and "qualified-hc" in r["private_path"]])==1
    actual=helper.effective_closure(build,build/"machinery/worker.mk")
    expected_objects={str(build/"host/FlashForward.o"),str(build/"host/FlashWorker.o")} | {str(build/r["private_path"]) for r in inputs if r["category"] in ("REUSED","CORE")}
    checks["actual_private_object_link_matches_pinned_closure"]={str(x) for x in actual["objects"]}==expected_objects
    checks["actual_private_air_link_matches_pinned_closure"]={str(x) for x in actual["airs"]}=={str(build/r["private_path"]) for r in inputs if r["category"]=="AIRS"}
    checks["dense_cache_object_and_air_preserved"]=sum(x.name=="worker_cache.o" for x in actual["objects"])==1 and sum(x.name=="dense-w8a8.air" for x in actual["airs"])==1
    parent_witness=m["prefill_hc_parent_artifact_witness"]
    checks["parent_published_runtime_artifacts_still_match_witness"]=all(sha((base/name).read_bytes())==digest for name,digest in parent_witness["artifacts"].items())
    checks["all_parent_compiler_input_seals_fresh"]=all(sha(Path(path).read_bytes())==r["sha256"] for path,r in m["prefill_hc_parent_input_seals"].items())
    dep_mismatch=[]
    for r in m["prefill_hc_parent_dependencies"]:
        if not r["dependency_metadata_present"]:continue
        if sha(Path(r["source_path"]).read_bytes())!=r["sha256"] or sha((build/r["private_path"]).read_bytes())!=r["sha256"]:dep_mismatch.append(r["source_path"])
        for dep in r["dependencies"]:
            if sha(Path(dep["source_path"]).read_bytes())!=dep.get("sha256",dep.get("external_source_sha256")):dep_mismatch.append(dep["source_path"])
    checks["parent_dependency_and_header_closure_fresh"]=not dep_mismatch
    new_deps=[];live=[];compiled_deps=[]
    for path in [build/"host/FlashForward.d",build/"host/FlashWorker.d",build/"policy-cpu.d",build/"prefill4k-attribution.d"]:
        for token in helper.dep_tokens(path.read_bytes()):
            dep=helper.resolve_input(Path(token)).resolve()
            if build/"source" in dep.parents:
                relative=dep.relative_to(build/"source").as_posix()
                if relative not in records or sha(dep.read_bytes())!=records[relative]["overlay_sha256"]:new_deps.append(str(dep))
                compiled_deps.append({"path":relative,"sha256":sha(dep.read_bytes())})
            elif ROOT/"runtime" in dep.parents or ROOT/"dev" in dep.parents:live.append(str(dep))
    checks["all_new_compiled_sources_and_headers_private_and_fresh"]=not new_deps and not live
    forward=(build/"source/runtime/flash/FlashForward.cpp").read_text();old_forward=(base/"source/runtime/flash/FlashForward.cpp").read_text()
    worker=(build/"source/runtime/flash/FlashWorker.mm").read_text();old_worker=(base/"source/runtime/flash/FlashWorker.mm").read_text()
    attribution=(build/"source/dev/benchmarks/prefill4k_attribution.mm").read_text()
    checks["target_numerical_identity_expression_byte_identical"]=identity_expression(worker)==identity_expression(old_worker)
    routes=section(forward,"std::string FlashForward::kernelRoutes() const", "FlashHCUpEncodedCounters FlashForward::hcUpEncodedCounters() const")
    checks["new_static_selection_marker_in_routes"]= "prefill_hc_inject_norm_sep21::selectionMarker(prefill_hc_inject_norm_sep21::requested())" in routes
    checks["mutable_counters_outside_all_kernel_and_numerical_identity"]= "encodedCounters" not in routes and "record" not in routes and "encodedCounters" not in identity_expression(worker)
    json_identity=section(worker,'      << R"(,"identity":{"source":)"','      << R"(,"persisted_operands":')
    checks["mutable_counters_outside_entire_json_identity_object"]= "encodedCounters" not in json_identity and "encoded_" not in json_identity and worker.index('      << R"(,"prefill_hc_inject_norm":{"enabled":)"')<worker.index('      << R"(,"identity":{"source":)"')
    checks["selector_frozen_before_paths_and_backend"]=worker.index("(void)prefill_hc_inject_norm_sep21::requested();")<worker.index("const auto directory = std::filesystem::canonical(argv[2]);")
    checks["attribution_selector_frozen_before_backend"]=attribution.index("(void)prefill_hc_inject_norm_sep21::requested();")<attribution.index("metal::MetalBackend backend(argv[1]);")
    checks["main_scope_explicit_nonverification_singleton"]= "rows, verification, true, prefill_hc_inject_norm_sep21::requested()" in forward
    checks["ple_exclusion_applies_to_both_old_and_new_fusion"]= "normalizedReady = (impl_->fuseHC && rows <= 32 && !nextHasPLE) || privatePrefillNextNorm;" in forward and "nextNormEligible(privatePrefillHC, nextHasPLE, nextIsTerminalMixer)" in forward
    checks["new_prefill_fusion_explicitly_excludes_terminal_mixer"]= "const bool nextIsTerminalMixer = layer + 1 == impl_->descriptor.layers;" in forward and "nextNormEligible(privatePrefillHC, nextHasPLE, nextIsTerminalMixer)" in forward
    checks["terminal_mixer_and_audited_norm_selection_preserved"]='layer + 1 == impl_->descriptor.layers\n          ? "language_model.model.hyper_connection_mixer"' in forward and "impl_->weights.normConvention(nextNorm)" in forward
    checks["workspace_planner_byte_identical"]=section(forward,"uint64_t FlashForward::workspaceBytes", "std::string FlashForward::kernelRoutes() const")==section(old_forward,"uint64_t FlashForward::workspaceBytes", "std::string FlashForward::kernelRoutes() const")
    checks["all_public_runtime_headers_and_hc_down_up_unchanged"]=all((build/"source"/relative).read_bytes()==(base/"source"/relative).read_bytes() for relative in records if relative.startswith("runtime/") and relative not in CHANGED)
    checks["dense_projection_control_and_allocation_logic_preserved"]=section(forward,"struct FlashForward::Impl", "uint64_t FlashForward::workspaceBytes")==section(old_forward,"struct FlashForward::Impl", "uint64_t FlashForward::workspaceBytes")
    results={mode or "default":cpu([build/"policy-cpu"]+([mode] if mode else []),clean_env()) for mode in ("","--freeze0","--freeze1")}
    checks["compiled_scope_flags_counter_and_identity_policy_pass"]=all(r["returncode"]==0 for r in results.values())
    self_test=cpu([build/"splash-flash","--cpu-self-test"],clean_env());checks["main_worker_cpu_self_test_pass"]=self_test["returncode"]==0
    probes={}
    for value in ("","2","true","01"," 1","1 ","-1"):
        env=clean_env();env[FLAG]=value;r=cpu([build/"splash-flash","serve-flash-native","/nonexistent-prefill-hc-cpu-only", "auto","auto"],env);probes[value]=r
        checks[f"invalid_flag_rejected_before_path_backend:{value!r}"]=r["returncode"]==3 and FLAG+" must be 0 or 1" in r["stderr"] and "canonical" not in r["stderr"]
    document={"schema":"splash-prefill-hc-worker-cpu-witness-v1","pass":all(checks.values()),"checks":checks,
        "gpu_work":False,"model_loaded":False,"payload_bytes_read":0,"whole_model_qualified":False,
        "source_mismatch":mismatch,"tool_mismatch":tools_mismatch,"link_input_mismatch":input_mismatch,
        "parent_dependency_mismatch":dep_mismatch,"new_dependency_mismatch":new_deps,"live_project_dependencies":live,
        "compiled_dependencies":list({r["path"]:r for r in compiled_deps}.values()),"compiled_policy_cpu":results,
        "main_worker_cpu_self_test":self_test,"invalid_flag_probes":probes,
        "frozen_source_count":len(records),"frozen_link_input_count":len(inputs),"actual_worker_objects":len(actual["objects"]),"actual_worker_airs":len(actual["airs"]),
        "runtime_sha256":{name:sha((build/name).read_bytes()) for name in ("splash-flash","splash.metallib","prefill4k-attribution")}}
    args.output.write_text(json.dumps(document,indent=2)+"\n")
    print(json.dumps({"pass":document["pass"],"checks":len(checks),"failures":[k for k,v in checks.items() if not v],"gpu_work":False,"payload_bytes_read":0}))
    if not document["pass"]:raise SystemExit(1)
if __name__=="__main__":main()
