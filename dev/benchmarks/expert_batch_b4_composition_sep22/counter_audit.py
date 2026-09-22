#!/usr/bin/env python3
"""Strict CPU-only default-off compact batch target route/status coverage audit."""
import argparse,json,re
from pathlib import Path
SECTION='compact_native_batch_verify'
SCHEMA='parallel-integer-original-M16-six-stage-R8R16-target-verify-v1'
SCOPE='batch target rows4 active lanes2/4 physicalR8/R16 only; graph construction not GPU completion'
def errors(status,identity,require_constructed=False):
 e=[];s=status.get(SECTION) if isinstance(status,dict) else None
 if not isinstance(s,dict):return ['compact batch status unavailable']
 if s.get('schema')!=SCHEMA or s.get('scope')!=SCOPE:e.append('unknown scope/schema')
 if type(s.get('enabled'))is not bool or s.get('requested')is not s.get('enabled'):e.append('nonliteral requested/enabled or mismatch')
 if s.get('source_identity_sha256')!=identity:e.append('compiled identity mismatch')
 for k,v in [('dispatches_per_layer',6),('base_native_dispatches_per_layer',10),('additional_gpu_allocation_bytes',0),('full_model_quality_qualified',False)]:
  if type(s.get(k))is not type(v) or s[k]!=v:e.append('invalid '+k)
 total=0
 for rows,tg in [(8,2752),(16,3392)]:
  w=s.get('r'+str(rows))
  if not isinstance(w,dict):e.append('missing actual width ledger');continue
  if type(w.get('physical_rows'))is not int or w['physical_rows']!=rows or type(w.get('planner_threadgroup_bytes'))is not int or w['planner_threadgroup_bytes']!=tg:e.append('wrong fixed geometry')
  vals=[]
  for role in ['plan','gate','down']:
   calls=w.get(role+'_graph_calls');n=w.get(role+'_graph_rows');vals.append(calls)
   if type(calls)is not int or type(n)is not int or calls<0 or n<0 or calls>=2**64 or n>=2**64:e.append('invalid graph counter');continue
   if n!=calls*rows:e.append('counter does not count actual physical rows')
   if calls%48:e.append('incomplete 48 layer graph construction')
  if not all(type(x)is int for x in vals):continue
  if len(set(vals))!=1:e.append('partial plan/gate/down construction')
  total+=vals[0]
  if not s.get('enabled') and any(vals):e.append('disabled route constructed graph')
 if require_constructed and not total:e.append('no compact target graph construction evidence')
 return e

def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--require-constructed',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('NEW audit output required')
 text=(a.build/'source/dev/benchmarks/expert_batch_compact_verify_worker_sep22/source_identity.hpp').read_text();identity=re.search(r'kSourceIdentitySha256\[\]="([0-9a-f]{64})"',text).group(1)
 status=json.loads(a.status.read_text());err=errors(status,identity,a.require_constructed);result={'schema':'compact-native-batch-target-coverage-cpu-audit-v1','pass':not err,'errors':err,'compiled_source_identity_sha256':identity,'graph_construction_not_completion':True,'GPU_work':False,'payload_reads':False};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if err:raise SystemExit(2)
if __name__=='__main__':main()
