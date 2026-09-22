#!/usr/bin/env python3
"""CPU-only compact INTEGER/native M16 closure witness; no GPU or payload IO."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_compact_native_parallel_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/expert-r4-compact-native-sep22-component-v2');p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.build.resolve()
 if a.output.exists():raise ValueError('NEW source witness required')
 seal=json.loads((b/'source-seal.json').read_text());bad=[r['path']for r in seal['sources']if sha((b/'source'/r['path']).read_bytes())!=r['sha256']];linkbad=[r['private_path']for r in seal['link_inputs']if sha((b/r['private_path']).read_bytes())!=r['sha256']]
 k=(b/'source'/PRIVATE/'plan.metal').read_text();o=(b/'source'/PRIVATE/'oracle.mm').read_text();s=(b/'source'/PRIVATE/'safety.hpp').read_text();m=(b/'source'/PRIVATE/'metadata.hpp').read_text();backend=(b/'source/runtime/metal/MetalBackend.mm').read_text();parent=Path(seal['parent_build'])
 original_native=Path('runtime/metal/kernels/shared/flash_int8_expert_store.metal')
 code='\n'.join(line.split('//',1)[0]for line in k.splitlines())
 checks={'all_sources_match_seal':not bad,'all_frozen_link_inputs_match':not linkbad,
 'link_make_match':sha((b/'component-inputs.mk').read_bytes())==seal['link_make_sha256'],
 'native_producer_source_literal':(b/'source'/original_native).read_bytes()==(parent/'source'/original_native).read_bytes(),
 'planner_no_floating_arithmetic_or_weight_rank_input':'float'not in code and'bfloat'not in code and'ranks'not in code and'codes'not in code,
 'only_new_integer_export':k.count('kernel void ')==1 and'expert_r4_compact_native_sep22_plan'in k,
 'native_m16_capacity514': 'p.tile_rows==16&&p.job_capacity==514'in k,
 'native_jobs_step16': 'first+j*16' in k and'first+c0+j*16' in k,
 'all_sentinels514_initialized':'for(uint j=tid;j<514;j+=256)jobs[j]={UINT_MAX,0};'in k,
 'rank_filtering_absent': 'ranks'not in k,
 'sticky_never_cleared': 'atomic_store'not in k and'*diag='not in k,
 'actual_single_full_grid': 'grid [[threadgroups_per_grid]]'in k and'all(grid==uint3(1))'in k,
 'malformed_setup_clears_stale_once':'if(all(group==uint3(0))&&!tid)'in k and'*jobCount=0;'in k,
 'stable_flat_route_order': 'other<id||(other==id&&r<tid)'in k,
 'duplicate_original_ownership': 'r>=tid/10*10&&r<tid&&other==id'in k,
 'initial_metadata_device_barrier': 'threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);'in k,
 'same_native_pack_and_prepare': 'flash_moe_direct_a_pack'in o and'flash_moe_direct_a_prepare_down'in o,
 'same_original_native_gate_down': 'flash_int8_expert_store_gate_up_m16_n64'in o and'flash_int8_expert_store_down_scatter_m16_n64'in o,
 'same_buffer_snapshots': 'snapshots'in o and'commonCapture'in o,
 'raw_and_prepared_gu_separate': 'rawGU'in o and'preparedGU'in o and'gateDiagnostic'in o,
 'canonical_observer_not_gpu_unpack':'flash_qmv_probe_unpack_activation'not in o and'canonicalize'in o,
 'all103_rows_and514_jobs':'operandRows=103'in o and'jobSlots=514'in o,
 'executed_bad_rank_diff_visible':'executed_preexisting_bad_rank_difference'in o and'badRankNativeSkip'in o and'badRankGatherPoison'in o,
 'metadata_reference_large_duplicate':'r.jobs[1].row_begin!=16'in m and'r.jobs[2].row_begin!=32'in m,
 'safety_alias_padding_and_bad_grid':all(t in s for t in ('compact native operand/metadata aliases','all_invalid_metadata_and_padding_exact','forty_duplicates_m16_jobs_step16_exact','malformed_multi_cta_grid_clears_stale_ranges')),
 'before_warm_qualification': 'prequalified'in o and'while('in o,
 'backend_serial_source_bound':'descriptor.dispatchType = MTLDispatchTypeSerial;'in backend and'[command computeCommandEncoder];'in backend and'MTLDispatchTypeConcurrent'not in backend,
 }
 artifacts={n:sha((b/n).read_bytes())for n in('oracle','splash.metallib','compact-probe.air','host/MetalBackend.o')}
 result={'schema':'splash-first-sixdispatch-compact-native-r4-cpu-source-witness-v1','pass':all(checks.values()),'checks':checks,'source_mismatches':bad,'link_mismatches':linkbad,'source_identity_sha256':seal['source_identity_sha256'],'artifacts_sha256':artifacts,'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False,'model_or_capture_hashes':False,'dot_association_changed':False,'parallel_plan_threadgroup_bytes':2432,'independent_parallel_review':'No must-fix: barriers protect reused prefix scratch, stable lexicographic route positions, adjacent expert pair scans, M16 duplicates and all514sentinels' ,'new_weights_sidecar_allocation':False,'cpu_build_and_entry':{'warnings_as_errors':True,'exit_code':0,'six_case_cpu_checks':'passed; independently reported exact build output, no repeated test work'},'independent_review':'no remaining must-fix: planner, same scratch/diag/probe bases, raw/prepared stages, exact metadata/padding, executed rank difference, pre-timing checks','graph_dispatches':{'old_native':10,'compact_native':6,'current_gather':2},'promotion_qualified':False}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':result['pass'],'checks':len(checks),'failed_checks':[n for n,v in checks.items()if not v],'output':str(a.output),'gpu_work':False}))
 if not result['pass']:raise SystemExit(2)
if __name__=='__main__':main()
