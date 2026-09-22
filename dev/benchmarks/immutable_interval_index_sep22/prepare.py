#!/usr/bin/env python3
"""CPU-only literal old-source guard versus portable shipping-header component."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path("/Users/mweinbach/Projects/splash")
HERE=Path(__file__).resolve().parent
BASE=ROOT/"build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def extract(text,signature):
    start=text.index(signature);begin=text.index("{",start);depth=1;stop=begin+1
    while depth:
        depth+=(text[stop]=="{")-(text[stop]=="}");stop+=1
    return text[start:stop]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--build",type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve()
    if build.exists():raise ValueError("fresh interval component required")
    manifest=json.loads((BASE/"overlay-manifest.json").read_text())
    records={r["path"]:r["sha256"] for r in manifest["files"]}
    names=["runtime/flash/FlashInt8ExpertStore.mm","runtime/flash/FlashMoEBuckets.cpp",
           "dev/benchmarks/expert_r4_preflight_bundle_sep22/guard.hpp"]
    for name in names:
        if sha(BASE/"source"/name)!=records[name]:raise ValueError("sealed original source drift:"+name)
    store=(BASE/"source"/names[0]).read_text();bucket=(BASE/"source"/names[1]).read_text()
    functions={name:extract(store,name) for name in [
        "[[noreturn]] void fail(const char *reason)","void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes)",
        "void disjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b)",
        "void allRowsScratch(const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,",
        "void immutableDisjoint(const metal::MetalBuffer &output) const"]}
    functions["void requireGeometry(uint32_t rows, uint32_t selections)"]=extract(bucket,"void requireGeometry(uint32_t rows, uint32_t selections)")
    functions["uint32_t moEBucketJobCapacity(uint32_t rows, uint32_t selections,"]=extract(bucket,"uint32_t moEBucketJobCapacity(uint32_t rows, uint32_t selections,")
    build.mkdir(parents=True);source=build/"source";source.mkdir()
    for name in ["index.hpp","metadata.hpp","metadata.cpp","oracle.cpp","prepare.py"]:shutil.copy2(HERE/name,source/name)
    shutil.copy2(BASE/"source"/names[2],source/"original_guard.hpp")
    declarations="\n".join(value for key,value in functions.items() if "immutableDisjoint" not in key)
    immutable=functions["void immutableDisjoint(const metal::MetalBuffer &output) const"].replace(
        "void immutableDisjoint(","void Original::immutableDisjoint(",1)
    original='#include "metadata.hpp"\n#include <algorithm>\n#include <array>\nnamespace guard_metadata {\nconstexpr uint32_t kExperts=512,kFlashMoEBucketMaximumRows=8192,kFlashMoEBucketMaximumSelections=10;\n'+declarations+"\n"+immutable+"\n}\n"
    (source/"original.cpp").write_text(original)
    journal={"schema":"literal-original-guard-CPU-extraction-v1","sealed_parent":str(BASE),
        "parent_source_identity":manifest["source_identity_sha256"],"parent_manifest_sha256":sha(BASE/"overlay-manifest.json"),
        "authoritative_sources":[{"path":str(BASE/"source"/name),"sha256":records[name]} for name in names],
        "functions":[{"signature":key,"literal":value,"sha256":hashlib.sha256(value.encode()).hexdigest()}for key,value in functions.items()],
        "only_owner_qualification_change":"immutableDisjoint signature becomes Original::immutableDisjoint; BODY BYTE IDENTICAL",
        "metadata_adapters_separate_TU":True,"actual_shipping_index_header":True,
        "no_Store_Worker_Core_shader_or_GPU_changes":True,"model_data_payload_access":False}
    (build/"source-journal.json").write_text(json.dumps(journal,indent=2)+"\n")
    flags=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-I"+str(source)]
    commands=[];objects=[]
    for name in ["metadata.cpp","original.cpp","oracle.cpp"]:
        obj=build/(Path(name).stem+".o");command=["xcrun","-sdk","macosx","clang++",*flags,"-MMD","-MP","-c",str(source/name),"-o",str(obj)]
        commands.append(command);subprocess.run(command,cwd=ROOT,check=True);objects.append(obj)
    command=["xcrun","-sdk","macosx","clang++",*flags,*map(str,objects),"-o",str(build/"oracle")]
    commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
    proof=subprocess.run([str(build/"oracle")],cwd=ROOT,check=True,capture_output=True,text=True)
    (build/"CPU-proof.json").write_text(proof.stdout);result=json.loads(proof.stdout)
    if not result["pass"] or result["model_or_device_or_data_payload_access"]:raise ValueError("CPU source parity failure")
    bench=subprocess.run([str(build/"oracle"),"--benchmark"],cwd=ROOT,check=True,capture_output=True,text=True)
    (build/"CPU-benchmark.json").write_text(bench.stdout)
    ready={"schema":"immutable96-index-portable-shipping-header-CPU-component-v1","pass":True,
        "original_source_journal_sha256":sha(build/"source-journal.json"),"actual_shipping_header_sha256":sha(source/"index.hpp"),
        "source_files":[{"path":str(p),"sha256":sha(p)}for p in sorted(source.glob("*"))if p.is_file()],
        "objects":[{"path":str(p),"sha256":sha(p)}for p in objects],"oracle_sha256":sha(build/"oracle"),
        "compiler_commands":commands,"CPU_proof":result,"CPU_benchmark":json.loads(bench.stdout),
        "model_capture_response_operand_payload_read_or_hashed":False,"GPU_executed":False,
        "Store_or_Worker_integration":False,"native_ObjC_or_whole_decode_speedup_claimed":False,
        "Root_worker_build_decision_pending":True,"independent_shipping_source_review_pending":True}
    (build/"CPU_READY.json").write_text(json.dumps(ready,indent=2)+"\n")
    print(json.dumps({"CPU_READY":str(build/"CPU_READY.json"),"proof_checks":result["decision_checks"],
        "shipping_header_sha256":ready["actual_shipping_header_sha256"],"metadata_benchmark":ready["CPU_benchmark"]["metadata_benchmark"]}))
if __name__=="__main__":main()
