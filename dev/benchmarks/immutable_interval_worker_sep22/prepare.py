#!/usr/bin/env python3
"""CPU-only private two-TU worker; current public headers/Core/library untouched."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess
ROOT=Path("/Users/mweinbach/Projects/splash")
HERE=Path(__file__).resolve().parent
BASE=ROOT/"build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
COMPONENT=ROOT/"build/immutable96-interval-index-sep22-component-v4"
PRIVATE="dev/benchmarks/immutable_interval_worker_sep22"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--build",type=Path,required=True);a=p.parse_args();out=a.build.resolve()
 if out.exists():raise ValueError("fresh private worker required")
 m=json.loads((BASE/"overlay-manifest.json").read_text());seal=json.loads((BASE/"compiled-cpu-seal.json").read_text())
 if not seal["pass"] or sha(BASE/"splash-flash")!="663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438" or sha(BASE/"splash.metallib")!="7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8":
  raise ValueError("qualified exact current Q4 parent required")
 for r in m["files"]:
  if sha(BASE/"source"/r["path"])!=r["sha256"]:raise ValueError("parent source drift:"+r["path"])
 for r in seal["compiled_objects"]+seal["artifacts"]:
  if sha(BASE/r["path"])!=r["sha256"]:raise ValueError("parent object/artifact drift")
 if sha(COMPONENT/"source/index.hpp")!="5acc0147351516e3a6607070ac3151fd8bae34e43d05852576366f087aa73b9f":
  raise ValueError("actual independently reviewed shipping header differs")
 shutil.copytree(BASE,out)
 for name in ["compiled-cpu-seal.json","overlay-manifest.json"]:(out/name).rename(out/("inherited-Q4-"+name))
 own=out/"source"/PRIVATE;own.mkdir(parents=True)
 for name in ["policy.hpp","overlay.py","policy_cpu.cpp","prepare.py"]:shutil.copy2(HERE/name,own/name)
 shutil.copy2(COMPONENT/"source/index.hpp",own/"index.hpp")
 parts={name:sha(own/name)for name in ["policy.hpp","overlay.py","policy_cpu.cpp","prepare.py","index.hpp"]}
 parts.update(parent_source=m["source_identity_sha256"],parent_seal=sha(BASE/"compiled-cpu-seal.json"),
              source_store=sha(BASE/"source/runtime/flash/FlashInt8ExpertStore.mm"),
              source_worker=sha(BASE/"source/runtime/flash/FlashWorker.mm"),component=sha(COMPONENT/"CPU_READY.json"))
 identity=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 (own/"source_identity.hpp").write_text('#pragma once\nnamespace splash::flash::immutable_interval_worker_sep22 {inline constexpr char sourcePolicySHA[]='+json.dumps(identity)+';}\n')
 spec=importlib.util.spec_from_file_location("ownedIndexOverlay",own/"overlay.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 changed=["runtime/flash/FlashInt8ExpertStore.mm","runtime/flash/FlashWorker.mm"]
 journal=[]
 for name in changed:
  old=(BASE/"source"/name).read_text();new=mod.transform(name,old);(out/"source"/name).write_text(new)
  journal.append({"path":name,"parent_sha256":sha(BASE/"source"/name),"changed_sha256":sha(out/"source"/name)})
 flags=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-Wno-deprecated-declarations","-fobjc-arc","-mmacosx-version-min=27.0","-DSPLASH_INT8_EXPERIMENT=1",
        "-I"+str(out/"source"),"-I"+str(out/"source/runtime"),"-I"+str(out/"source/dev/benchmarks/prefill4k_attention")]
 link=(out/"link-inputs.mk").read_text();names=next(x for x in link.splitlines()if x.startswith("REBUILD_NAMES :=")).split(":=",1)[1].split()
 sources={x.group(1):x.group(2)for x in re.finditer(r"^SRC_(\S+) := \$\(BUILD\)/source/(.*)$",link,re.M)}
 core=[out/t.removeprefix("$(BUILD)/")for t in next(x for x in link.splitlines()if x.startswith("CORE :=")).split(":=",1)[1].split()]
 objects=[];commands=[];census=[]
 for name in names:
  source=out/"source"/sources[name];obj=out/"host"/(name+".o");old=BASE/"host"/(name+".o")
  dep=subprocess.run(["xcrun","-sdk","macosx","clang++",*flags,"-MM",str(source)],cwd=ROOT,check=True,capture_output=True,text=True).stdout
  consumer="/immutable_interval_worker_sep22/policy.hpp" in dep
  if consumer!=(str(source.relative_to(out/"source"))in changed):raise ValueError("unexpected private header consumer")
  census.append({"object":name,"source":str(source),"new_private_header_consumer":consumer,"dependencies":dep})
  if consumer:
   command=["xcrun","-sdk","macosx","clang++",*flags,"-MMD","-MP","-c",str(source),"-o",str(obj)];commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
  elif sha(obj)!=sha(old):raise ValueError("unchanged current object drift")
  objects.append({"path":str(obj.relative_to(out)),"sha256":sha(obj),"parent_sha256":sha(old),"recompiled":consumer})
 for obj in core:
  if sha(obj)!=sha(BASE/obj.relative_to(out)):raise ValueError("Core changed")
  objects.append({"path":str(obj.relative_to(out)),"sha256":sha(obj),"parent_sha256":sha(obj),"recompiled":False})
 if len(objects)!=54 or sum(x["recompiled"]for x in objects)!=2 or len(census)!=50:raise ValueError("exact2new52same/50census required")
 command=["xcrun","-sdk","macosx","clang++",*flags,*[str(out/x["path"])for x in objects],"-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(out/"splash-flash")]
 commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
 command=["xcrun","-sdk","macosx","clang++",*flags,str(own/"policy_cpu.cpp"),"-o",str(out/"policy-CPU")]
 commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
 cpu=json.loads(subprocess.run([str(out/"policy-CPU")],cwd=ROOT,check=True,capture_output=True,text=True).stdout)
 worker=json.loads(subprocess.run([str(out/"splash-flash"),"--cpu-self-test"],cwd=ROOT,check=True,capture_output=True,text=True).stdout)
 env={k:v for k,v in os.environ.items()if not k.startswith("SPLASH_")};cases=[]
 for value,expected in [("2","must be exactly0 or1"),("1","immutable interval index requires SPLASH_FLASH_ALLROWS_FULL512_TARGET=1")]:
  e=dict(env);e["SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22"]=value
  r=subprocess.run([str(out/"splash-flash"),"serve-flash-native","/nonexistent-index-CPU-no-model","16384","auto"],env=e,capture_output=True,text=True)
  if not r.returncode or expected not in r.stderr or "filesystem" in r.stderr:raise ValueError("compiled strict prepath refusal failed")
  cases.append({"value":value,"exit":r.returncode,"stderr":r.stderr.strip(),"beforepath_device_model":True})
 for value in ("0","1"):
  e=dict(env);e["SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22"]=value
  subprocess.run([str(out/"policy-CPU"),"--lifetime"],env=e,cwd=ROOT,check=True,capture_output=True,text=True)
 unrelated=[]
 for p in (BASE/"source").rglob("*"):
  if p.is_file()and str(p.relative_to(BASE/"source"))not in changed:
   if sha(p)!=sha(out/"source"/p.relative_to(BASE/"source")):raise ValueError("public header/other source drift")
   unrelated.append(str(p.relative_to(BASE/"source")))
 if sha(out/"splash.metallib")!=sha(BASE/"splash.metallib"):raise ValueError("library changed")
 artifact=[{"path":n,"sha256":sha(out/n)}for n in ["splash-flash","splash.metallib","policy-CPU"]]
 ready={"schema":"immutable96-index-two-private-TU-worker-CPU-closure-v1","pass":True,"GPU_executed":False,
        "model_capture_response_operand_payload_read_or_hashed":False,"base":str(BASE),"source_policy_sha256":identity,"identity_parts":parts,
        "changed_TUs":changed,"source_journal":journal,"public_headers_Core_or_Metal_library_changed":False,
        "FP_geometry_original_guard_graph_suffix_unchanged":True,"GPU_allocations_added":0,"CPU_metadata_index_only":True,
        "objects":objects,"actual50TU_private_header_census":census,"compiler_commands":commands,
        "source_files":[{"path":str(p.relative_to(out/"source")),"sha256":sha(p)}for p in sorted((out/"source").rglob("*"))if p.is_file()],
        "artifacts":artifact,"CPU_policy":cpu,"CPU_Worker":worker,"compiled_prepath_refusals":cases,
        "Root_actual_metadata_bounded_native_QA_and_service_pending":True,"whole_model_or_performance_qualified":False}
 (out/"compiled-cpu-seal.json").write_text(json.dumps(ready,indent=2)+"\n");(out/"overlay-manifest.json").write_text(json.dumps(ready,indent=2)+"\n")
 print(json.dumps({"CPU_READY":str(out/"compiled-cpu-seal.json"),"source_policy_sha256":identity,"worker_sha256":sha(out/"splash-flash"),"CPU_policy":cpu}))
if __name__=="__main__":main()
