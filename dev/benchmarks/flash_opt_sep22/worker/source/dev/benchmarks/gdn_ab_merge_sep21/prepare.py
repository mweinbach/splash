#!/usr/bin/env python3
"""Freeze and CPU-compile exact paired GDN A/B projection component."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=ROOT/"build/moe-pointwise-sep21-worker-v1");p.add_argument("--build",type=Path,default=ROOT/"build/gdn-ab-merge-sep21-oracle-v1");args=p.parse_args();base,build=args.base.resolve(),args.build.resolve()
    if build.exists():raise ValueError("freshGDNABbuildrequired")
    build.mkdir(parents=True);shutil.copytree(base/"source",build/"source");src=build/"source/dev/benchmarks/gdn_ab_merge_sep21";shutil.copytree(HERE,src)
    reference=build/"source/runtime/metal/kernels/shared/flash_affine_qmv_f32.metal";reference.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/"runtime/metal/kernels/shared/flash_affine_qmv_f32.metal",reference)
    subprocess.run([str(ROOT/".venv/bin/python"),str(src/"shader_generate.py"),"--original",str(reference),"--output",str(src/"gdn_ab_qmv_probe_generated.metal"),"--journal",str(build/"probe-source-journal.json")],check=True,cwd=ROOT)
    names=["FlashAffine","FlashHC","FlashHCFused","FlashFloatDenseCache","FlashOperandStore","FlashDenseCache","FlashAffineMPP","FlashDescriptor","FlashWeights","FlashPLESSDStore","FlashInt8ExpertStoreMetadata"]
    objects=[base/"reused/base-host"/(n+".o") for n in names]+[base/"reused/core/engine/metal"/(n+".o") for n in ["MetalBackend","DeviceCapabilities"]]
    airLine=next(line for line in (base/"link-inputs.mk").read_text().splitlines() if line.startswith("AIRS := "));airs=[base/t.removeprefix("$(BUILD)/") for t in airLine.removeprefix("AIRS := ").split()]+[base/"pointwise.air"]
    links=[];fObjects=[];fAirs=[]
    for index,path in enumerate(objects+airs):
        dest=build/"reused"/f"{index:03d}-{path.name}";dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);links.append({"original":str(path),"frozen":str(dest),"sha256":sha(dest)});(fObjects if index<len(objects) else fAirs).append(dest)
    provenance={"base_sha256":sha(base/"splash-flash"),"reference_qmv_source_sha256":sha(reference),"objects":links[:len(objects)]}
    (build/"Provenance.hpp").write_text('#pragma once\ninline constexpr const char *kGDNABProvenance=R"CLOSURE('+json.dumps(provenance,sort_keys=True)+')CLOSURE";\n')
    cpp=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-Wno-deprecated-declarations","-mmacosx-version-min=27.0","-DSPLASH_INT8_EXPERIMENT=1","-fobjc-arc","-I"+str(build/"source/runtime"),"-I"+str(build/"source"),"-I"+str(src),"-I"+str(build)]
    # Numerical audit header is source only and absent from parent's complete
    # runtime closure; freeze it alongside other component instructions.
    audit=build/"source/dev/benchmarks/FlashFloatBoundaryAudit.hpp";audit.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/"dev/benchmarks/FlashFloatBoundaryAudit.hpp",audit)
    commands=[["xcrun","-sdk","macosx","metal","-std=metal4.1","-O3","-Wall","-Wextra","-Werror","-mmacosx-version-min=27.0","-I"+str(build/"source/runtime"),"-I"+str(src),"-c",str(src/"candidate.metal"),"-o",str(build/"candidate.air")],
      ["xcrun","-sdk","macosx","metallib",*map(str,fAirs),str(build/"candidate.air"),"-o",str(build/"splash.metallib")],
      ["xcrun","-sdk","macosx","clang++",*cpp,str(src/"oracle.mm"),*map(str,fObjects),"-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(build/"oracle")]]
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)
    cpu=subprocess.run([str(build/"oracle"),"--cpu-self-test"],cwd=ROOT,check=True,text=True,capture_output=True);(build/"cpu-self-test.json").write_text(cpu.stdout)
    manifest={"schema":"splash-gdn-ab-merge-frozen-cpu-preparation-v1","gpu_work":False,"model_payload_bytes_read":0,"source_files":[{"path":str(path),"sha256":sha(path)} for path in sorted((build/"source").rglob("*")) if path.is_file()],"frozen_inputs":links,"compiler_commands":commands,"runtime_sha256":{"oracle":sha(build/"oracle"),"splash.metallib":sha(build/"splash.metallib")},"original_qmv_source_sha256":sha(reference)}
    (build/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"pass":True,"build":str(build),"gpu_work":False,"model_payload_bytes_read":0,"sources":len(manifest["source_files"]),"frozen_inputs":len(links)}))
if __name__=="__main__":main()
