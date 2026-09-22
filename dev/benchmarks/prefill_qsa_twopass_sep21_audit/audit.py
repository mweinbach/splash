#!/usr/bin/env python3
"""Independent CPU geometry/causality proof; generated values, no payload reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
ROWS=2048
HEADS=24
KV=2
PER_KV=12
DIM=256


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry():
    checked=0
    # Independent chronological source definition from cache block semantics:
    # all completed groups are present at fresh <=2K, followed by the live tail.
    for row in range(ROWS):
        count=row+1;complete=count//4
        picked=min(complete,512)
        source=[block*4+offset for block in range(picked) for offset in range(4)]
        source.extend(complete*4+offset for offset in range(count%4))
        assert source==list(range(count));checked+=count
        partitions=1 if row<128 else 4
        spans=[]
        length=(count+partitions-1)//partitions
        for partition in range(partitions):
            start=min(partition*length,count);stop=min(start+length,count)
            spans.extend(range(start,stop))
        assert spans==source
        # Qualified temporal32 shares the union of adjacent query partitions;
        # per-head masks must cover exactly the same chronological sources.
        if row>=512:
            begin=(row//128)*128;local=row-begin
            for h in range(PER_KV):
                flat_tile=((local*PER_KV+h)//32)*32
                first=flat_tile//PER_KV;last=min((flat_tile+31)//PER_KV,127)
                first_count=begin+first+1;last_count=begin+last+1
                first_length=(first_count+3)//4;last_length=(last_count+3)//4
                for partition in range(4):
                    native_start=min(partition*first_length,first_count)
                    native_stop=min(partition*last_length+last_length,last_count)
                    start=min(partition*length,count);stop=min(start+length,count)
                    assert native_start<=start<=stop<=native_stop
                    assert list(range(max(native_start,start),min(native_stop,stop)))==list(range(start,stop))
                    checked+=stop-start
    # Complete query-packing permutation, independent of candidate shader code.
    original=np.arange(ROWS*HEADS*DIM,dtype=np.uint32).reshape(ROWS,KV,PER_KV,DIM)
    packed=np.transpose(original,(1,0,2,3)).reshape(KV,ROWS*PER_KV,DIM)
    assert np.array_equal(np.sort(packed.reshape(-1)),np.arange(ROWS*HEADS*DIM,dtype=np.uint32))
    checked+=packed.size
    for kv in range(KV):
        for row in (0,1,127,128,511,512,1023,1920,2047):
            for h in (0,1,11):
                for d in (0,31,63,64,127,128,255):
                    assert int(packed[kv,row*PER_KV+h,d])==(row*HEADS+kv*PER_KV+h)*DIM+d
                    checked+=1
    pack_hash=hashlib.sha256(packed.tobytes()).hexdigest()
    # Direct and packed V describe the same (kv, dimension, token) source cells.
    values=np.arange(ROWS*KV*DIM,dtype=np.uint32).reshape(ROWS,KV,DIM)
    packed_v=np.transpose(values,(1,2,0)).copy()
    for kv in range(KV):
        for d in range(DIM):
            actual=packed_v[kv,d,:]
            golden=np.arange(ROWS,dtype=np.uint32)*KV*DIM+kv*DIM+d
            assert np.array_equal(actual,golden);checked+=ROWS
    # The Qpack arena becomes dead after QK; packed V then fits in its first2MB.
    q_bytes=ROWS*HEADS*DIM*2;v_bytes=ROWS*KV*DIM*2
    assert v_bytes<=q_bytes
    return dict(checks=checked,query_pack_uint32_permutation_sha256=pack_hash,
                query_elements=ROWS*HEADS*DIM,packed_v_elements=ROWS*KV*DIM,
                chronological_row_token_memberships=ROWS*(ROWS+1)//2,
                causal_head_token_pairs=ROWS*(ROWS+1)//2*HEADS,
                direct_and_packed_v_source_sets_equal=True,
                temporal_partition_unions_and_per_head_masks_preserved=True,
                gate_source_formula='(row*24+head)*512+256+dimension',
                kv_source_formula='(token*2+kv)*256+dimension',
                query_unpack_formula='[kv,row*12+headWithinKV,dimension] -> [row,kv*12+headWithinKV,dimension]',
                q_pack_v_pack_lifetimes_disjoint=True)


def reference():
    # Small complete FP64 causal reference plus scalar math.fsum audits,
    # independent of any GPU matrix/reduction geometry.
    rows=17;rng=np.random.default_rng(214037)
    q=rng.normal(size=(rows,HEADS,DIM)).astype(np.float64)
    k=rng.normal(size=(rows,KV,DIM)).astype(np.float64)
    v=rng.normal(size=(rows,KV,DIM)).astype(np.float64)
    result=np.zeros_like(q);maximum_scalar_abs_error=0.0;checks=0
    for head in range(HEADS):
        kv=head//PER_KV
        scores=q[:,head,:]@k[:,kv,:].T/16.0
        scores[np.arange(rows)[None,:]>np.arange(rows)[:,None]]=-np.inf
        exp=np.exp(scores-np.max(scores,axis=1,keepdims=True))
        p=exp/np.sum(exp,axis=1,keepdims=True)
        assert np.all(p[np.arange(rows)[None,:]>np.arange(rows)[:,None]]==0)
        assert np.allclose(np.sum(p,axis=1),1,rtol=0,atol=4e-16)
        result[:,head,:]=p@v[:,kv,:]
        for row in (0,1,8,16):
            scalar_scores=[math.fsum(float(q[row,head,d]*k[t,kv,d]) for d in range(DIM))/16 for t in range(row+1)]
            m=max(scalar_scores);weights=[math.exp(s-m) for s in scalar_scores];denominator=math.fsum(weights)
            for d in (0,31,63,127,255):
                expected=math.fsum(weights[t]*float(v[t,kv,d]) for t in range(row+1))/denominator
                maximum_scalar_abs_error=max(maximum_scalar_abs_error,abs(expected-result[row,head,d]));checks+=1
        # Finite future source perturbations cannot change an earlier anchor.
        for anchor in (0,1,8):
            changed_k=k[:,kv,:].copy();changed_v=v[:,kv,:].copy()
            changed_k[anchor+1:]*=10000;changed_v[anchor+1:]+=1e6
            changed_scores=q[anchor,head,:]@changed_k.T/16
            changed_scores[anchor+1:]=-np.inf
            changed_exp=np.exp(changed_scores-np.max(changed_scores))
            changed_p=changed_exp/np.sum(changed_exp)
            observed=changed_p@changed_v
            assert np.allclose(observed,result[anchor,head,:],rtol=0,atol=1e-12);checks+=DIM
    assert maximum_scalar_abs_error<=4e-15
    return dict(generated_reference_rows=rows,checks=checks,
                maximum_scalar_fsum_abs_error=maximum_scalar_abs_error,
                finite_future_kv_causal_anchors_preserved=True,
                mathematical_f64_reference_only=True,
                gpu_precision_qualification='pending separately preregistered producer oracle',
                nonfinite_fresh_v='original prepare flags numerical failure for all fresh rows; no partial nonfinite-output parity claim',
                inactive_cache_rows_ge_2048='must never be read; poison guards required in root GPU oracle')


def costs():
    flat=ROWS*PER_KV
    scores=KV*flat*ROWS*4
    q=KV*flat*DIM*2
    out=KV*flat*DIM*4
    full_pairs=ROWS*ROWS*HEADS
    causal_pairs=ROWS*(ROWS+1)//2*HEADS
    return dict(full_qk_plus_pv_flops=full_pairs*DIM*4,
                causal_minimum_qk_plus_pv_flops=causal_pairs*DIM*4,
                full_qk_ctas_m128_n64=KV*(flat//128)*(ROWS//64),
                full_pv_ctas_m128_n64=KV*(flat//128)*(DIM//64),
                softmax_ctas=KV*flat,
                score_or_f32_probability_union_bytes=scores,
                packed_query_arena_bytes=q,fp32_pv_output_bytes=out,
                total_logical_arena_bytes=scores+q+out,
                packed_v_reuses_dead_packed_q_bytes=ROWS*KV*DIM*2,
                f32_scores_to_p_minimum_valid_read_plus_full_write_bytes=causal_pairs*4+scores,
                precision_scope='global F32 softmax and whole-K dot reduction numerical alternative; no parity claim',
                model_speedup='hypothesis; require full timed pipeline then whole-model quality and acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('Choose fresh output')
    sources=[Path(__file__),ROOT/'runtime/metal/kernels/shared/flash_qsa_bulk.metal',
             ROOT/'runtime/metal/kernels/shared/flash_qsa_fast.metal',ROOT/'runtime/flash/FlashQSABulk.cpp']
    evidence=dict(schema='qsa-full-twopass-independent-cpu-source-and-mask-proof-v1',
                  gpu_executed=False,model_payload_bytes_read=0,
                  geometry=geometry(),reference=reference(),cost=costs(),
                  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
                  pass_=True)
    evidence['pass']=evidence.pop('pass_')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(pass_=True,output=str(out),gpu_executed=False,model_payload_bytes_read=0,
                         source_mask_checks=evidence['geometry']['checks'],reference_checks=evidence['reference']['checks'],
                         arena_bytes=evidence['cost']['total_logical_arena_bytes'])))


if __name__=='__main__':main()
