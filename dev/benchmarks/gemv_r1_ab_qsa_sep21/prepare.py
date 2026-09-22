#!/usr/bin/env python3
"""Compose sealed AR1 vector implementation onto sealed AB/QSA source closure."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3];VECTOR_PRIVATE=Path("dev/benchmarks/gemv_decode_r1_worker_sep21")
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=ROOT/"build/gdn-ab-merge-qsa-sep21-worker-v1");p.add_argument("--vector",type=Path,default=ROOT/"build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1");p.add_argument("--build",type=Path,default=ROOT/"build/gemv-r1-ab-qsa-sep21-worker-v1");args=p.parse_args();base,old,build=args.base.resolve(),args.vector.resolve(),args.build.resolve()
    if build.exists():raise ValueError("freshAR1composition outputrequired")
    bmBytes=(base/"overlay-manifest.json").read_bytes();bm=json.loads(bmBytes);vmBytes=(old/"overlay-manifest.json").read_bytes();vm=json.loads(vmBytes)
    assert bm.get("gdn_ab_composed") and vm.get("vector_r1_source_identity_sha256")
    transformPath=old/"source"/VECTOR_PRIVATE/"worker_prepare.py";assert sha(transformPath)==vm["vector_r1_transform_sha256"]
    spec=importlib.util.spec_from_file_location("sealed_vector_transform",transformPath);transform=importlib.util.module_from_spec(spec);spec.loader.exec_module(transform)
    build.mkdir(parents=True);shutil.copytree(base/"source",build/"source");changed=[];records=[]
    for entry in bm["files"]:
        relative=entry["path"];source=base/"source"/relative;assert sha(source)==entry["overlay_sha256"]
        data=transform.transform(relative,source.read_text()).encode();dest=build/"source"/relative;dest.write_bytes(data)
        if data!=source.read_bytes():changed.append(relative)
        records.append({**entry,"r1_ab_base_overlay_sha256":sha(source),"r1_ab_changed":data!=source.read_bytes(),"overlay_sha256":hashlib.sha256(data).hexdigest()})
    assert set(changed)==transform.CHANGED_PATHS
    for entry in vm["files"]:
        relative=Path(entry["path"])
        if relative.parts[:len(VECTOR_PRIVATE.parts)]==VECTOR_PRIVATE.parts:
            source=old/"source"/relative;assert sha(source)==entry["overlay_sha256"]
            dest=build/"source"/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            records.append({"path":relative.as_posix(),"new_vector_r1_file":True,"sealed_vector_sha256":sha(source),"overlay_sha256":sha(source)})
    qualification=build/"qualification";qualification.mkdir();shutil.copy2(old/"qualification/r1-component.json",qualification/"r1-component.json");shutil.copy2(old/"qualified-source-closure.json",qualification/"r1-qualified-source-closure.json")
    # Dependency census uses the exact sealed source/header tree and compiler
    # switches. Rebuild every TU consuming either changed public class header.
    cppFlags=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-Wno-deprecated-declarations","-fobjc-arc","-mmacosx-version-min=27.0","-DSPLASH_INT8_EXPERIMENT=1","-I"+str(build/"source"),"-I"+str(build/"source/runtime"),"-I"+str(build/"source/dev/benchmarks/prefill4k_attention")]
    runtimeSources=sorted(x for x in (build/"source/runtime/flash").glob("*") if x.suffix in [".cpp",".mm"])
    extraSources=[build/"source/dev/benchmarks/prefill4k_attention"/(n+".cpp") for n in ["bulk","coalesced"]]
    headerSuffixes=["/runtime/flash/FlashForward.hpp","/runtime/flash/FlashInt8ExpertStore.hpp"]
    def dependencies(source:Path):
        result=subprocess.run(["xcrun","-sdk","macosx","clang++",*cppFlags,"-MM",str(source)],cwd=ROOT,text=True,capture_output=True,check=True)
        tokens=result.stdout.replace("\\\n"," ").split();hits=[x for x in tokens if any(x.endswith(suffix) for suffix in headerSuffixes)]
        return {"source":str(source.relative_to(build/"source")),"consume_modified_headers":hits,"dependency_tokens":tokens}
    with ThreadPoolExecutor(max_workers=8) as pool:census=list(pool.map(dependencies,runtimeSources+extraSources))
    own={Path(row["source"]).stem for row in census if row["consume_modified_headers"]};own|={"FlashForward","FlashInt8ExpertStore","FlashWorker"}
    inputs={"REUSED":[],"CORE":[],"AIRS":[]};frozen=[]
    def freeze(path:Path,kind:str):
        dst=build/"reused"/f"{len(frozen):03d}-{path.name}";dst.parent.mkdir(exist_ok=True);shutil.copy2(path,dst);inputs[kind].append(dst.relative_to(build).as_posix());frozen.append({"original":str(path),"private_path":dst.relative_to(build).as_posix(),"category":kind,"sha256":sha(path)})
    for path in sorted((base/"host").glob("*.o")):
        if path.stem not in own:freeze(path,"REUSED")
    for line in (base/"link-inputs.mk").read_text().splitlines():
        kind,raw=line.split(" := ",1)
        for token in raw.split():
            assert token.startswith("$(BUILD)/");path=base/token.removeprefix("$(BUILD)/")
            # Number-prefixed frozen names retain the original object stem.
            originalStem=path.stem.split("-",1)[-1] if path.stem[:1].isdigit() else path.stem
            if kind=="REUSED" and originalStem in own:continue
            freeze(path,kind)
    for path in sorted(base.glob("*.air")):freeze(path,"AIRS")
    make="\n".join(f"{kind} := "+" ".join("$(BUILD)/"+path for path in paths) for kind,paths in inputs.items())+"\n"
    (build/"link-inputs.mk").write_text(make)
    manifest=copy.deepcopy(bm);manifest.update({"route":"sealed-ordinaryAR1-vector-numerical-alt-over-exactAB-and-packedQSA-v1","vector_r1_source_identity_sha256":vm["vector_r1_source_identity_sha256"],"vector_r1_identity_parts":vm["vector_r1_identity_parts"],"vector_r1_flag":vm["vector_r1_flag"],"vector_r1_added_gpu_bytes":0,"r1_ab_base_build":str(base),"r1_ab_base_manifest_sha256":hashlib.sha256(bmBytes).hexdigest(),"r1_ab_vector_build":str(old),"r1_ab_vector_manifest_sha256":hashlib.sha256(vmBytes).hexdigest(),"r1_ab_transform_sha256":sha(Path(__file__)),"r1_ab_changed_files":changed,"r1_ab_header_dependency_census":census,"r1_ab_rebuilt_host_names":sorted(own),"r1_ab_link_inputs":frozen,"r1_ab_link_make_sha256":sha(build/"link-inputs.mk"),"gpu_work":False,"model_payload_reads":0,"files":records})
    (build/"overlay-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    ownText="OWN_NAMES := "+" ".join(sorted(own))+"\n";(build/"own-inputs.mk").write_text(ownText)
    print(json.dumps({"prepared":str(build),"rebuild_headers_actual_consumers":sorted(own),"dependency_TUs":len(census),"sources":len(records),"frozen_inputs":len(frozen),"source_identity":vm["vector_r1_source_identity_sha256"],"gpu_work":False,"model_payload_reads":0}))
if __name__=="__main__":main()
