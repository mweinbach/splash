#!/usr/bin/env python3
"""CPU-only isolated actual-capture build; no GPU/data payload operations."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PARENT=ROOT/"build/batch-prefill-restored-teacher-clock-sep22-worker-v1"
HOOK=ROOT/"build/batch-qsa-actual-capture-sep22-source-v2"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(command):subprocess.run(command,cwd=ROOT,check=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve()
    if build.exists():raise ValueError("fresh capture build required")
    seal=json.loads((PARENT/"compiled-cpu-seal.json").read_text())
    receipt=json.loads((HOOK/"capture-source-receipt.json").read_text())
    if not seal["pass"] or seal["effective_objects"]!=54 or seal["host_TUs_rebuilt"]!=50:
        raise ValueError("complete current restored parent closure required")
    if not receipt["pass"] or receipt["parent_seal_sha256"]!=sha(PARENT/"compiled-cpu-seal.json"):
        raise ValueError("frozen hook current-parent admission differs")
    sources={}
    for record in seal["source_files"]:
        path=(PARENT/"source"/record["path"]).resolve()
        if sha(path)!=record["sha256"]:raise ValueError("parent source drift:"+str(path))
        sources[str(path)]=record["sha256"]
    for record in receipt["files"]:
        if sha(Path(record["path"]))!=record["sha256"]:raise ValueError("frozen v2 hook source drift")
    objects=[];replaced=[]
    for record in seal["compiled_objects"]:
        path=PARENT/record["object"]
        if sha(path)!=record["sha256"]:raise ValueError("parent host object drift")
        if path.name=="FlashWorker.o":continue
        if Path(record["source"]).name=="FlashBatchPrefill.cpp":replaced.append(path);continue
        objects.append({"path":str(path),"sha256":record["sha256"],"kind":"exact-current-parent-host"})
    for record in seal["core_objects"]:
        path=PARENT/record["path"]
        if sha(path)!=record["sha256"]:raise ValueError("parent Core object drift")
        objects.append({"path":str(path),"sha256":record["sha256"],"kind":"exact-current-parent-Core"})
    if len(objects)!=52 or len(replaced)!=1:raise ValueError("52imports+one replaced BatchPrefill required")
    if sha(PARENT/"splash.metallib")!="1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf":
        raise ValueError("exact qualified current restored library required")
    build.mkdir(parents=True);own=build/"source";own.mkdir()
    for name in ["capture.hpp","FlashBatchPrefill_capture.cpp"]:shutil.copy2(HOOK/name,own/name)
    for name in ["oracle.mm","build_capture.py"]:shutil.copy2(HERE/name,own/name)
    provenance={
        "schema":"actual-batch-QSA-capture-code-provenance-v1",
        "parent":str(PARENT),"parent_seal_sha256":sha(PARENT/"compiled-cpu-seal.json"),
        "parent_Worker_sha256":seal["binary_sha256"],
        "metallib_sha256":seal["metallib_sha256"],
        "capture_source_receipt_sha256":sha(HOOK/"capture-source-receipt.json"),
        "capture_header_sha256":sha(own/"capture.hpp"),
        "capture_modified_BatchPrefill_sha256":sha(own/"FlashBatchPrefill_capture.cpp"),
        "capture_harness_sha256":sha(own/"oracle.mm"),
        "original_BatchPrefill_sha256":receipt["parent_source_sha256"],
        "original_body_after_debug_only_normalization_exact":True,
        "public_header_layout_or_Metal_math_changes":False,
        "scope":"one Model/Forward/Batch;two sequential fresh oldbulk1 cohorts;all134physical state andalloutput bytes;onecohortlive;lane0layer3 actual owned capture",
        "GPU_executed_by_preparation":False,"model_operand_capture_fixture_payload_bytes_read_or_hashed":0,
    }
    (own/"CaptureBuildProvenance.hpp").write_text("#pragma once\ninline constexpr const char*kCaptureBuildProvenance=R\"PROV("+json.dumps(provenance,sort_keys=True)+")PROV\";\n")
    shutil.copy2(PARENT/"splash.metallib",build/"splash.metallib")
    flags=[*seal["compiler_flags"],"-I"+str(own)]
    cpp=own/"FlashBatchPrefill_capture.cpp";obj=build/"FlashBatchPrefill_capture.o";hobj=build/"capture-oracle.o";binary=build/"capture-oracle"
    commands=[]
    for source,output in [(cpp,obj),(own/"oracle.mm",hobj)]:
        command=["xcrun","-sdk","macosx","clang++",*flags,"-MMD","-MP","-c",str(source),"-o",str(output)]
        commands.append(command);run(command)
    link=["xcrun","-sdk","macosx","clang++",*flags,str(hobj),str(obj),
          *[record["path"] for record in objects],"-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(binary)]
    commands.append(link);run(link)
    dependencies=[]
    for dep in [build/"FlashBatchPrefill_capture.d",build/"capture-oracle.d"]:
        paths=shlex.split(dep.read_text().replace("\\\n"," ").splitlines()[0].split(":",1)[1])
        for token in paths:
            path=Path(token).resolve()
            if str(path) in sources:
                if sha(path)!=sources[str(path)]:raise ValueError("current parent header source drift")
                dependencies.append({"path":str(path),"sha256":sources[str(path)],"kind":"sealed-current-parent-source"})
            elif own in path.parents:
                dependencies.append({"path":str(path),"sha256":sha(path),"kind":"capture-owned-source"})
            elif ROOT in path.parents:raise ValueError("unsealed workspace dependency:"+str(path))
    census=[]
    for record in seal["compiled_objects"]:
        source=cpp if Path(record["source"]).name=="FlashBatchPrefill.cpp" else PARENT/"source"/record["source"]
        r=subprocess.run(["xcrun","-sdk","macosx","clang++",*flags,"-MM",str(source)],cwd=ROOT,check=True,capture_output=True,text=True)
        consumer="capture.hpp" in r.stdout
        if consumer!=(source==cpp):raise ValueError("unexpected private capture header consumer")
        census.append({"source":str(source),"private_capture_header_consumer":consumer,"excluded_Worker_Main":Path(record["source"]).name=="FlashWorker.mm"})
    cpu=json.loads(subprocess.run([str(binary),"--cpu-self-test"],cwd=ROOT,check=True,capture_output=True,text=True).stdout)
    helptext=subprocess.run([str(binary),"--help"],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    if not cpu["pass"] or cpu["device_created"] or cpu["metadata_or_payload_reads"]:raise ValueError("CPU-only capture path failed")
    for record in objects:
        if sha(Path(record["path"]))!=record["sha256"]:raise ValueError("parent import drift during build")
    result={
        "schema":"actual-batch-QSA-capture-isolated-CPU-closure-v1","pass":True,
        "GPU_executed":False,"model_operand_capture_fixture_export_payload_bytes_read_or_hashed":0,
        "parent":str(PARENT),"parent_seal_sha256":provenance["parent_seal_sha256"],
        "capture_source_receipt_sha256":provenance["capture_source_receipt_sha256"],
        "current_nonWorker_runtime_objects":53,"imported_objects":objects,
        "rebuilt_BatchPrefill_object":{"path":str(obj),"sha256":sha(obj)},
        "new_harness_object":{"path":str(hobj),"sha256":sha(hobj)},
        "actual50TU_header_census":census,"compiler_dependencies":dependencies,
        "source_files":[{"path":str(p),"sha256":sha(p)}for p in sorted(own.glob("*"))if p.is_file()],
        "compiler_commands":commands,"CPU":cpu,"help_no_device":helptext,
        "oracle_sha256":sha(binary),"metallib_sha256":sha(build/"splash.metallib"),
        "provenance":provenance,"Root_independent_review_and_GPU_execution_pending":True,
        "new_batch_twoPass_qualified":False,"capture_preservation_GPU_proved":False,
    }
    (build/"CPU_READY.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"CPU_READY":str(build/"CPU_READY.json"),"oracle_sha256":result["oracle_sha256"],"CPU":cpu}))
if __name__=="__main__":main()
