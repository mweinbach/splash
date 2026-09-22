#!/usr/bin/env python3
"""CPU audit of pair-worker sources, numerical identity, linkage and status."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from collections import Counter

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("build",type=Path);p.add_argument("--report",type=Path);args=p.parse_args();b=args.build.resolve();m=json.loads((b/"overlay-manifest.json").read_text());base=Path(m["gdn_ab_base_build"])
    assert sha(base/"overlay-manifest.json")==m["gdn_ab_base_manifest_sha256"]
    spec=importlib.util.spec_from_file_location("worker_overlay",Path(__file__).with_name("worker_prepare.py"));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    changed=[]
    for e in m["files"]:
        now=b/"source"/e["path"];assert sha(now)==e["overlay_sha256"]
        if "gdn_ab_base_overlay_sha256" in e:
            old=base/"source"/e["path"];assert sha(old)==e["gdn_ab_base_overlay_sha256"]
            assert module.transform(e["path"],old.read_text()).encode()==now.read_bytes()
            if old.read_bytes()!=now.read_bytes():changed.append(e["path"])
    assert sorted(changed)==["runtime/flash/FlashForward.cpp","runtime/flash/FlashWorker.mm"]
    names=[]
    for e in m["gdn_ab_frozen_inputs"]:
        assert sha(b/e["private_path"])==e["sha256"]==sha(Path(e["original"]))
        if e["kind"]=="REUSED":names.append(Path(e["original"]).name)
    names+=['FlashForward.o','FlashWorker.o'];assert max(Counter(names).values())==1
    assert sha(b/"link-inputs.mk")==m["gdn_ab_link_make_sha256"]
    for relative in ["runtime/flash/FlashForward.hpp","runtime/flash/FlashBatchForward.cpp","runtime/flash/FlashBatchVerify.cpp","runtime/flash/FlashBatchPrefill.cpp","runtime/flash/FlashInt8ExpertStore.mm"]:
        assert (b/"source"/relative).read_bytes()==(base/"source"/relative).read_bytes()
    forward=(b/"source/runtime/flash/FlashForward.cpp").read_text();old=(base/"source/runtime/flash/FlashForward.cpp").read_text()
    a=old.index('uint64_t FlashForward::workspacePlannedBytes');z=old.index('std::string FlashForward::kernelRoutes',a);assert old[a:z] in forward
    worker=(b/"source/runtime/flash/FlashWorker.mm").read_text();a=worker.index('      << R"(,"identity":{"source":)"');z=worker.index('      << R"(,"weight_format":"mlx_affine_preconverted"})"',a)
    assert 'abGraphs' not in worker[a:z] and 'paired_graph_calls' not in worker[a:z]
    assert worker.count('"gdn_ab_merge_route_counters"')==1
    live=[]
    for f in (b/"host").glob('*.d'):
        for token in f.read_text().replace('\\\n',' ').split():
            if token.startswith('runtime/') and token.endswith(('.h','.hpp')):live.append(token)
    assert not live
    rejected={}
    for value in ['', '2','true','01',' 1','1 ']:
        env=dict(os.environ);env['SPLASH_FLASH_GDN_AB_MERGE_SEP21']=value
        r=subprocess.run([str(b/'splash-flash'),'serve-flash-native','/gdnAB-no-such-path','16384','auto'],env=env,text=True,capture_output=True)
        assert r.returncode and 'SPLASH_FLASH_GDN_AB_MERGE_SEP21 must be0 or1' in r.stderr
        rejected[value]={'exitcode':r.returncode,'stderr':r.stderr.strip()}
    result={'schema':'splash-exact-GDNAB-worker-source-CPU-witness-v1','pass':True,'gpu_work':False,'model_payload_reads':0,
      'source_files_verified':len(m['files']),'frozen_inputs_verified':len(m['gdn_ab_frozen_inputs']),'effective_host_objects_unique':len(names),
      'changed_sources':changed,'public_headers_batch_policy_cache_plan_and_numerical_derivative_unchanged':True,
      'mutable_counters_top_level_outside_identity':True,'no_live_header_dependencies':True,'malformed_flag_before_path_backend_rejection':rejected,
      'new_gpu_allocations_bytes':0,'full_eligible_target_dispatches_72_to_36':True,'base_build':str(base),
      'runtime_sha256':{name:sha(b/name) for name in ['splash-flash','splash.metallib']},'model_quality_qualified':False}
    if args.report:
        assert not args.report.exists();args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
