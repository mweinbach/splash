#!/usr/bin/env python3
"""Root-only original driver invocation after actual native and task proof."""
import argparse
from pathlib import Path
import subprocess
from launch_common import ROOT,load_pinned,canonical,validate_binding,require

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role",choices=("old","new"),required=True)
    parser.add_argument("--launch-witness-sha256",required=True)
    parser.add_argument("--proof-binding-sha256",required=True)
    parser.add_argument("--run-root-gpu",action="store_true")
    args=parser.parse_args();require(args.run_root_gpu,"Root-exclusive inference action required")
    plan=load_pinned(args.launch_witness_sha256)
    argv=canonical(plan,args.role)
    validate_binding(plan,args.launch_witness_sha256,args.proof_binding_sha256)
    for key in ("report","trace","native_audit","profile_audit"):
        require(not Path(plan["profiles"][args.role][key]).exists(),"Fresh normal output required: "+key)
    return subprocess.run(argv,cwd=ROOT).returncode

if __name__=="__main__":raise SystemExit(main())
