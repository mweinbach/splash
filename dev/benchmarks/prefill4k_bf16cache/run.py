#!/usr/bin/env python3
"""Prepare/run a serial Root-only BF16 coefficient-cache primitive screen."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hot",type=int,choices=(64,128),default=128)
    p.add_argument("--layer",type=int,choices=(0,24,47),default=0)
    p.add_argument("--rows",type=int,default=2048)
    p.add_argument("--tile",type=int,choices=(16,32,64),default=32)
    p.add_argument("--pairs",type=int,default=4)
    p.add_argument("--pattern",choices=("all-hot","mixed","all-cold","spread"))
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--run",action="store_true",help="Root serializes GPU/model work; omission creates provenance only")
    a = p.parse_args()
    if not 1 <=a.rows <=8192 or not 1 <=a.pairs <=32 or (a.tile ==64 and a.rows <1024):
        raise ValueError("Unsupported primitive row/pair/tile geometry")
    build = ROOT /"build/prefill4k-bf16cache"
    plan_path = ROOT /f"build/release/flash/hot-expert-plan{a.hot}.json"
    plan = json.loads(plan_path.read_text())
    ids = plan["selected_experts"][a.layer]
    if len(ids) !=a.hot or ids !=sorted(set(ids)):
        raise ValueError("Observed hot selection differs")
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SPLASH_FLASH_","FLASH_EXPERT_CACHE_"))}
    controls = {"SPLASH_FLASH_PLE_SSD_STREAMING":"1","SPLASH_FLASH_MOE_Q4X8":"1",
                "SPLASH_FLASH_MOE_DIRECT_A":"1","SPLASH_FLASH_MOE_M64":"1",
                "FLASH_EXPERT_CACHE_HOT":str(a.hot),"FLASH_EXPERT_CACHE_HOT_IDS":','.join(map(str,ids)),
                "FLASH_EXPERT_CACHE_PREFIX":f"language_model.model.layers.{a.layer}.mlp.switch_mlp",
                "FLASH_EXPERT_CACHE_ROWS":str(a.rows),"FLASH_EXPERT_CACHE_TILE":str(a.tile),
                "FLASH_EXPERT_CACHE_PAIRS":str(a.pairs),"FLASH_EXPERT_CACHE_REQUIRE_EXACT":"1"}
    if a.pattern: controls["FLASH_EXPERT_CACHE_PATTERN"] =a.pattern
    env.update(controls)
    report = a.report.resolve()
    command = [str(build /"oracle"),str(build /"splash.metallib"),str(ROOT /"install/local-models/Flash-Next-oQ4e-mtp-v1"),str(report)]
    witness = report.with_suffix(report.suffix +".invocation.json")
    if report.exists() or witness.exists(): raise ValueError("Choose fresh primitive report/provenance")
    witness.parent.mkdir(parents=True,exist_ok=True)
    witness.write_text(json.dumps({"scope":"isolated lossless BF16 coefficient cache; whole-K reduction/output exactness requires qualification",
        "gpu_executed":a.run,"command":command,"controls":controls,"binary_sha256":sha(build /"oracle"),
        "metallib_sha256":sha(build /"splash.metallib"),"frequency_plan_sha256":sha(plan_path),"selected_original_expert_ids":ids,
        "current_direct_a_control":True,"stored_source_coefficients_lossless":True,"strict_bf16_output_equality_required":True,
        "generation_qualified":False},indent=2)+'\n')
    print(json.dumps({"gpu_executed":a.run,"command":command,"witness":str(witness)}))
    if a.run: subprocess.run(command,env=env,check=True,cwd=ROOT)

if __name__ =="__main__": main()
