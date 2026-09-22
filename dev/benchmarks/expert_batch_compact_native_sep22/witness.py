#!/usr/bin/env python3
"""CPU-only sealed fixed-batch closure and native/parallel integer metadata witness."""
from pathlib import Path
import argparse,hashlib,json,random
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_batch_compact_native_sep22')
def sha(b): return hashlib.sha256(b).hexdigest()
def native(ids):
 counts=[ids.count(e) for e in range(512)]
 offsets=[0];joboffs=[0];mapping=[];jobs=[]
 for e,c in enumerate(counts):
  offsets.append(offsets[-1]+c);joboffs.append(joboffs[-1]+(c+15)//16)
  mapping.extend(r for r,x in enumerate(ids) if x==e)
  jobs.extend((e,x) for x in range(offsets[e],offsets[e+1],16))
 inverse=[2**32-1]*len(ids)
 for p,r in enumerate(mapping):inverse[r]=p
 return counts,offsets,joboffs,mapping+[2**32-1]*(len(ids)-len(mapping)),inverse,jobs

def parallel(ids):
 counts=[0]*512
 for tid in range(256):
  for x in ids:counts[tid*2]+=x==tid*2;counts[tid*2+1]+=x==tid*2+1
 offsets=[0]*513;joboffs=[0]*513
 for source,dest in ((counts,offsets),([(c+15)//16 for c in counts],joboffs)):
  pair=[source[2*t]+source[2*t+1] for t in range(256)]
  totals=[sum(pair[32*g:32*g+32]) for g in range(8)]
  prefixes=[sum(totals[:g]) for g in range(8)]
  for t in range(256):
   first=prefixes[t//32]+sum(pair[32*(t//32):t]);dest[t*2]=first;dest[t*2+1]=first+source[t*2]
  dest[512]=sum(totals)
 mapping=[2**32-1]*len(ids);inverse=mapping.copy()
 for r,x in enumerate(ids):
  if 0<=x<512:
   p=sum(0<=v<512 and (v<x or (v==x and q<r)) for q,v in enumerate(ids));mapping[p]=r;inverse[r]=p
 jobs=[None]*joboffs[512]
 for e,c in enumerate(counts):
  for j in range((c+15)//16):jobs[joboffs[e]+j]=(e,offsets[e]+16*j)
 return counts,offsets,joboffs,mapping,inverse,jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.build.resolve()
 if a.output.exists():raise ValueError('NEW witness output required')
 seal=json.loads((b/'source-seal.json').read_text());rows=seal['fixed_rows'];routes=rows*10;jobs=(routes+15)//16+511
 bad=[r['path'] for r in seal['sources'] if sha((b/'source'/r['path']).read_bytes())!=r['sha256']]
 linkbad=[r['private_path'] for r in seal['link_inputs'] if sha((b/r['private_path']).read_bytes())!=r['sha256']]
 k=(b/'source'/PRIVATE/'plan.metal').read_text();o=(b/'source'/PRIVATE/'oracle.mm').read_text();s=(b/'source'/PRIVATE/'safety.hpp').read_text();backend=(b/'source/runtime/metal/MetalBackend.mm').read_text();parent=Path(seal['parent_build']);native_src=Path('runtime/metal/kernels/shared/flash_int8_expert_store.metal');code='\n'.join(x.split('//',1)[0] for x in k.splitlines())
 checks={'sources_match_seal':not bad,'links_match_seal':not linkbad,'link_make_matches':sha((b/'component-inputs.mk').read_bytes())==seal['link_make_sha256'],
 'rows_fixed8or16':rows in (8,16) and f'COMPACT_ROWS := {rows}' in (b/'component-inputs.mk').read_text(),
 'native_float_source_literal':(b/'source'/native_src).read_bytes()==(parent/'source'/native_src).read_bytes(),
 'planner_integer_only':all(x not in code for x in ('float','bfloat','ranks','codes')),'one_new_export':k.count('kernel void ')==1,
 'exact_full_cta':'all(grid==uint3(1))&&all(threads==uint3(256,1,1))' in k,'uniform_bad_geometry_return':k.index('return;')<k.index('threadgroup long cachedIDs'),
 'stable_original_route_order':'other<id||(other==id&&r<tid)' in k,'original_duplicate_row_ownership':'r>=tid/10*10&&r<tid&&other==id' in k,
 'all_native_sentinels':'j<kCompactNativeBatchJobCapacity;j+=256)jobs[j]={UINT_MAX,0}' in k,'step16_jobs':all(x in k for x in ('first+j*16','first+c0+j*16')),
 'sticky_never_cleared':all(x not in k for x in ('atomic_store','*diag=')),
 'prefix_scratch_reader_barrier':k.index('const uint first=groupPrefixes')<k.index('if(!lane)groupTotals[simd]=jobGroupTotal;')<k.index('threadgroup_barrier(mem_flags::mem_threadgroup);',k.index('if(!lane)groupTotals[simd]=jobGroupTotal;'))<k.index('if(!simd)',k.index('if(!lane)groupTotals[simd]=jobGroupTotal;')),
 'native_original_float_consumers':all(x in o for x in ('flash_moe_direct_a_pack','flash_int8_expert_store_gate_up_m16_n64','flash_moe_blocked_poison_excluded_routes','flash_moe_direct_a_prepare_down','flash_int8_expert_store_down_scatter_m16_n64')),
 'same_scratch_and_probe_bases':'same_native_scratch_and_diag_bases' in o and 'commonCapture' in o,'raw_prepared_distinct':'rawGU' in o and 'preparedGU' in o,
 'unsupported_gather_not_executed':'gatheredGraph' not in o and 'flash_gathered_mpp_' not in o,'native_norm_and_perrow_gates':'nativeQuality[plane].pass()' in o and 'activatedQuality.pass()&&downQuality.pass()' in o,
 'exact_gate_before_warm':o.index('bool prequalified=')<o.index('while(std::any_of(warmGPU'),
 '150ms_each_two_variants':'variants=2' in o and 'return x<.150' in o,'balanced18':'pairs=18' in o and 'pairs=(pairs+1)/2*2' in o,
 'partial_bad_cta_case':'malformed_partial_cta_clears_stale_ranges' in s,'large_i64_cases':'INT64_MIN' in s and 'INT64_MAX' in s,
 'alias_rejected':'host_metadata_operand_alias_rejected' in s,'last_hidden_row_nonfinite_even_no_routes':'(rows-1)*2560' in s and 'original_pack_all_hidden_nonfinite_evidence' in s,
 'bad_rank_parity_executed':'executed_native_bad_rank_parity' in o and 'badRankNativeSkip' in o,
 'post_timing_immutable_weights':o.rindex('layer.checkImmutableHashes(immutable)')>o.index('while(std::any_of(warmGPU'),
 'serial_backend':'descriptor.dispatchType = MTLDispatchTypeSerial;' in backend and 'MTLDispatchTypeConcurrent' not in backend}
 rng=random.Random(0x516521);patterns=[]
 patterns.extend(([7]*routes,[-1]*routes,list(range(routes)),[511]*routes))
 for _ in range(256):patterns.append([rng.choice([-2**63,-1,0,7,511,512,2**63-1,rng.randrange(512)]) for _ in range(routes)])
 mismatches=sum(native(x)!=parallel(x) for x in patterns);checks['distinct_native_vs_parallel_integer_patterns']=not mismatches
 artifacts={n:sha((b/n).read_bytes()) for n in ('oracle','splash.metallib','compact-probe.air','host/MetalBackend.o')}
 result={'schema':'splash-fixed-batch-integer-native-source-witness-v1','pass':all(checks.values()),'rows':rows,'routes':routes,'native_job_capacity':jobs,'checks':checks,'failed_checks':[x for x,v in checks.items() if not v],'source_mismatches':bad,'link_mismatches':linkbad,'source_identity_sha256':seal['source_identity_sha256'],'artifacts_sha256':artifacts,'integer_cpu_patterns':len(patterns),'integer_cpu_mismatches':mismatches,'gpu_work':False,'model_or_capture_payload_reads':False,'model_or_capture_payload_hashes':False,'new_global_allocations':0,'whole_model_qualified':False}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':result['pass'],'checks':len(checks),'failed':result['failed_checks'],'rows':rows,'patterns':len(patterns),'output':str(a.output)}))
 if not result['pass']:raise SystemExit(2)
if __name__=='__main__':main()
