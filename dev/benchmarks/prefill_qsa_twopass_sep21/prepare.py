#!/usr/bin/env python3
"""Freeze CPU-only private QSA prototype inputs; reads no model payloads."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OWN=ROOT/'dev/benchmarks/prefill_qsa_twopass_sep21'
BASE=ROOT/'dev/benchmarks/prefill4k_attention'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21')
    args=parser.parse_args();out=args.build.resolve()
    if ROOT/'build' not in out.parents:raise ValueError('Private output must be beneath build')
    if out.exists():raise ValueError('Choose a fresh build directory')
    source=out/'source';source.mkdir(parents=True)
    shutil.copytree(ROOT/'runtime',source/'runtime')
    shutil.copytree(BASE,source/'base')
    shutil.copytree(OWN,source/'experiment')
    shutil.copyfile(BASE/'oracle.mm',source/'experiment/base_oracle.mm')
    evidence={}
    for relative in ('build/release/flash/sep21-qsa-twopass-independent-cpu-proof-v1.json',
                     'dev/benchmarks/prefill_qsa_twopass_sep21_audit/audit.py',
                     'build/sep21-qsa-global-gemm-audit-v1/schedule-cost.json'):
        path=ROOT/relative
        if path.exists():
            destination=source/'evidence'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);evidence[relative]=sha(path)
    hashes={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    report={'schema':'splash-prefill-qsa-twopass-source-v1','gpu_executed':False,'model_payload_bytes_read':0,
            'production_sources_modified':False,'source_sha256':hashes,'evidence_sha256':evidence,
            'numerical_alternative':True,'probability_dtype':'F32','workspace_planned_bytes':509607936,
            'extra_arena_bytes':478150656,'scope':'fresh_begin0_rows2048_nonverification_only',
            'stage_order_direct':['source_prefix','packQ_with_selection_identity_guard','wholeK_QK','global_causal_F32_softmax','wholeK_F32P_BF16V_PV','source_BF16_attention_gate'],
            'stage_order_packed':['source_prefix','packQ_with_selection_identity_guard','wholeK_QK','global_causal_F32_softmax','packV_reuses_dead_Qpack','wholeK_F32P_BF16V_PV','source_BF16_attention_gate'],
            'preregistered':{'raw_source_relative_l2_max':2e-5,'BF16_source_relative_l2_max':1e-4,
                            'source_cosine_min':.99999999,'f64_QK_abs':2e-5,'f64_QK_relative':2e-6,
                            'f64_raw_abs':2e-6,'f64_raw_relative':2e-5,'global_probability_sum_abs_max':2e-6}}
    (out/'source-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'prepared':str(out),'source_files':len(hashes),'gpu_executed':False}))

if __name__=='__main__':main()
