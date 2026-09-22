#!/usr/bin/env python3
"""CPU-only two-role main/state oracle; authentic current objects, no data reads."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"build/batch-prefill-restored-teacher-clock-sep22-worker-v1"
CANDIDATE=ROOT/"build/batch-prefill-twopass-restored-sep22-worker-v3"
REFERENCE=ROOT/"build/batch-qsa-intended-reference-sep22-host-v1"
REFERENCE_SOURCE=ROOT/"build/batch-qsa-intended-reference-sep22-source-v2"
CHECKS=ROOT/"build/batch-prefill-restored-main-proof-sep22-v3/source/MainProofChecks.hpp"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve()
    if build.exists():raise ValueError("fresh main alternative build required")
    old=json.loads((OLD/"compiled-cpu-seal.json").read_text())
    candidate=json.loads((CANDIDATE/"compiled-cpu-seal.json").read_text())
    compiled=json.loads((CANDIDATE/"compiled-build.json").read_text())
    reference=json.loads((REFERENCE/"CPU_HOST_READY.json").read_text())
    if not old["pass"] or not candidate["pass"] or not reference["pass"]:raise ValueError("sealed roles required")
    build.mkdir(parents=True);common=build/"common";common.mkdir()
    for source,name in [(HERE/"intended_main_oracle.mm","intended_main_oracle.mm"),(CHECKS,"MainProofChecks.hpp"),
                        (HERE/"ALTERNATIVE_QUALIFICATION.md","ALTERNATIVE_QUALIFICATION.md"),(Path(__file__),"build_intended_main.py")]:
        shutil.copy2(source,common/name)
    results={}
    for role in ["reference","candidate"]:
        out=build/role;out.mkdir();objects=[];sources={};flags=[]
        if role=="reference":
            for record in old["source_files"]:
                path=OLD/"source"/record["path"]
                if sha(path)!=record["sha256"]:raise ValueError("old current source drift")
                sources[str(path.resolve())]=record["sha256"]
            for record in reference["runtime_objects_for_private_oracle"]:
                path=Path(record["path"])
                if sha(path)!=record["sha256"]:raise ValueError("reference object drift")
                objects.append({"path":str(path),"sha256":record["sha256"]})
            record=reference["reference_Batch_object"];path=Path(record["path"])
            if sha(path)!=record["sha256"]:raise ValueError("fresh reference Batch object drift")
            objects.append({"path":str(path),"sha256":record["sha256"]})
            if len(objects)!=53:raise ValueError("exact53 reference runtime objects")
            header=REFERENCE_SOURCE/"reference_qsa.hpp";shutil.copy2(header,out/"reference_qsa.hpp")
            flags=[*old["compiler_flags"],"-DSPLASH_BATCH_INTENDED_REFERENCE=1","-I"+str(out)]
            parent=OLD
        else:
            for path,digest in candidate["source_sha256"].items():
                source=CANDIDATE/"source"/path
                if sha(source)!=digest:raise ValueError("candidate current source drift")
                sources[str(source.resolve())]=digest
            for record in compiled["objects"]:
                path=CANDIDATE/record["path"]
                if path.name=="FlashWorker.o":continue
                if sha(path)!=record["sha256"]:raise ValueError("candidate runtime object drift")
                objects.append({"path":str(path),"sha256":record["sha256"]})
            for path,digest in candidate["artifact_sha256"].items():
                if path.startswith("core/") and path.endswith(".o"):
                    if sha(CANDIDATE/path)!=digest:raise ValueError("candidate Core drift")
                    objects.append({"path":str(CANDIDATE/path),"sha256":digest})
            if len(objects)!=53:raise ValueError("exact53 current candidate runtime objects")
            command=compiled["objects"][0]["command"];flags=command[4:command.index("-MMD")]
            parent=CANDIDATE
        if sha(parent/"splash.metallib")!="1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf":
            raise ValueError("both roles exact current1e34 library required")
        provenance={"schema":"batch-alternative-main-code-provenance-v1","role":role,
                    "parent":str(parent),"parent_seal_sha256":sha(parent/"compiled-cpu-seal.json"),
                    "reference_host_receipt_sha256":sha(REFERENCE/"CPU_HOST_READY.json"),
                    "candidate_source_policy_sha256":candidate["source_policy_sha256"],
                    "same_input_intended_existing_packedV_math":True,"old_EVERYROW_equivalence":False,
                    "old_extra_everyrow_failed_rows":116,"original_global_F64_constants_changed":False,
                    "harness_sha256":sha(common/"intended_main_oracle.mm"),
                    "metadata_checks_sha256":sha(common/"MainProofChecks.hpp"),
                    "alternative_plan_sha256":sha(common/"ALTERNATIVE_QUALIFICATION.md"),
                    "runtime_objects":objects,"capacity_scope":4096,"GPU_executed_by_builder":False}
        (out/"MainAlternativeProvenance.hpp").write_text("#pragma once\ninline constexpr const char*kMainAlternativeProvenance=R\"PROV("+json.dumps(provenance,sort_keys=True)+")PROV\";\n")
        flags=[*flags,"-I"+str(common),"-I"+str(out)]
        source=common/"intended_main_oracle.mm";obj=out/"oracle.o";binary=out/"oracle"
        compile_command=["xcrun","-sdk","macosx","clang++",*flags,"-MMD","-MP","-c",str(source),"-o",str(obj)]
        subprocess.run(compile_command,cwd=ROOT,check=True)
        link=["xcrun","-sdk","macosx","clang++",*flags,str(obj),*[r["path"] for r in objects],
              "-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(binary)]
        subprocess.run(link,cwd=ROOT,check=True)
        tokens=shlex.split((out/"oracle.d").read_text().replace("\\\n"," ").splitlines()[0].split(":",1)[1]);dependencies=[]
        for token in tokens:
            path=Path(token).resolve()
            if str(path) in sources:
                if sha(path)!=sources[str(path)]:raise ValueError("role header source drift")
                dependencies.append({"path":str(path),"sha256":sources[str(path)],"kind":"exact-role-current-source"})
            elif common in path.parents or out in path.parents:
                dependencies.append({"path":str(path),"sha256":sha(path),"kind":"oracle-private-source"})
            elif ROOT in path.parents:raise ValueError("unsealed workspace header:"+str(path))
        cpu=json.loads(subprocess.run([str(binary),"--cpu-self-test"],cwd=ROOT,check=True,capture_output=True,text=True).stdout)
        helptext=subprocess.run([str(binary),"--help"],cwd=ROOT,check=True,capture_output=True,text=True).stdout
        if not cpu["pass"] or cpu["GPU_executed"] or cpu["data_payload_reads"] or cpu["B4_spill_bound"]>=4<<30:
            raise ValueError("no-data metadata/campaign CPU gate failed")
        for record in objects:
            if sha(Path(record["path"]))!=record["sha256"]:raise ValueError("runtime import changed during CPU build")
        results[role]={"parent":str(parent),"runtime_objects":objects,"runtime_object_count":53,
                       "oracle_sha256":sha(binary),"oracle_object_sha256":sha(obj),
                       "metallib_sha256":sha(parent/"splash.metallib"),"compiler_dependencies":dependencies,
                       "compile_command":compile_command,"link_command":link,"CPU":cpu,
                       "help_no_device":helptext,"provenance":provenance}
    receipt={"schema":"batch-intended-alternative-main-two-role-CPU-closure-v1","pass":True,
             "GPU_executed":False,"model_capture_fixture_operand_export_payload_read_or_hashed":False,
             "roles":results,"source_files":[{"path":str(p),"sha256":sha(p)}for p in sorted(build.rglob("*"))if p.is_file() and p.suffix in [".mm",".hpp",".py",".md"]],
             "public_role_headers_changed":False,"new_API_inspection_friends":0,
             "one_Model_Forward_Batch_per_Root_process":True,"exact_all134_full_state_each4checkpoint":True,
             "genuine_greedy_AR_each3steps":True,"whole_campaign_exact_actual_view_preflight_before_first_payload":True,
             "old_extra_EVERYROW_116_FAIL_visible":True,"Root_GPU_qualification_pending":True,
             "original22_Head_Worker_or_service16K_qualified":False}
    (build/"CPU_READY.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"CPU_READY":str(build/"CPU_READY.json"),"roles":{k:v["oracle_sha256"] for k,v in results.items()}}))
if __name__=="__main__":main()
