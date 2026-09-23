import json,sys
def load(p):
    r=json.load(open(p))
    s=r['summary'][0] if r['summary'] else {}
    waves=[w for w in r['waves'] if not w['warmup']]
    texts=[]
    cyc=[];acc=[];vg=[];dg=[]
    for w in waves:
        rec=w['records'][0]
        texts.append(rec.get('content') or rec.get('text') or rec.get('content_text') or '')
        m=w['mtp_counters']; d=w['native_counter_delta']
        cyc.append(m.get('mtp.verification_cycles')); acc.append(m.get('mtp.accepted_committed_drafts'))
        vg.append(d.get('mtp.target_verify.total_gpu_ms',0)/max(1,m.get('mtp.verification_cycles') or 1))
        dg.append(d.get('mtp.head_decode.total_gpu_ms',0)/max(1,m.get('mtp.verification_cycles') or 1))
    return s,texts,cyc,acc,vg,dg,waves
for p in sys.argv[1:]:
    s,texts,cyc,acc,vg,dg,waves=load(p)
    print(p.split('/')[-1], 'prefill %.1f decode %.2f'%(s.get('native_prefill_median_tokens_per_second',0), s.get('native_exact_request_decode_median_tokens_per_second',0)),
          'cycles',cyc,'accepted',acc,'verify_gpu_ms/cycle %.2f'%(sum(vg)/len(vg)),'draft_gpu_ms/cycle %.2f'%(sum(dg)/len(dg)))
