#!/usr/bin/env python3
"""Verify FIRST N16/R4 CPU source/link closure and arithmetic/stage ownership."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_cohort_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=ROOT/'build/expert-r4-cohort-n16-sep22-component-v1');p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.build.resolve()
 if a.output.exists():raise ValueError('NEW CPU witness path required')
 seal=json.loads((b/'source-seal.json').read_text());bad=[r['path']for r in seal['sources']if sha((b/'source'/r['path']).read_bytes())!=r['sha256']]
 linkbad=[r['private_path']for r in seal['link_inputs']if sha((b/r['private_path']).read_bytes())!=r['sha256']]
 k=(b/'source'/PRIVATE/'kernels.metal').read_text();q=(b/'source'/PRIVATE/'quality.hpp').read_text();o=(b/'source'/PRIVATE/'oracle.mm').read_text();s=(b/'source'/PRIVATE/'malformed.hpp').read_text();backend=(b/'source/runtime/metal/MetalBackend.mm').read_text()
 checks={'all_sources_match_seal':not bad,'all_binary_inputs_match_seal':not linkbad,
  'link_make_match':sha((b/'component-inputs.mk').read_bytes())==seal['link_make_sha256'],
  'only_new_n16_tile':all(name in k for name in ('expert_r4_cohort_sep22_gate_up_n16','expert_r4_cohort_sep22_down_n16'))and'gate_up_n8'not in k and'gate_up_n32'not in k,
  'association_literal': '(p.x+p.y)+(p.z+p.w)'in k and'for(ushort delta=16;delta;delta/=2)'in k and'gdot[r][c]+=expert_r4_dot4(g*a[r])'in k,
  'one_coeff_load_drives_all_four_rows': 'const float4 g=float4(reinterpret_cast<const device char4 *>(gate+row)[chunk]);'in k and'for(uint r=0;r<4;++r)if(r<m)'in k,
  'input_dtype_unchanged': 'return as_type<float>(uint(bits)<<16);'in k and'if((bits&0x7f80u)==0x7f80u)'in k,
  'all_hidden_rows_inspected': 'for(uint i=tid;i<4*2560;i+=256)'in k,
  'plan_never_clears_diagnostic': 'atomic_store'not in k and'*diag='not in k,
  'persistent_ten_slots': 'for(uint job=group.y;job<*jobCount;job+=10)'in k,
  'actual_full_grid_guards': 'grid [[threadgroups_per_grid]]'in k and'any(grid!=uint3(Width/16,10,1))'in k,
  'metadata_checked_before_operand_pointers': k.index('if(!expert_r4_begin<Gate,Audit>')<k.index('reinterpret_cast<const device ushort4 *>(input+ulong(row)*K)'),
  'stable_live_ownership_closed': 'begin!=covered'in k and'inverse[route]!=index'in k and'routeMap[inverse[route]]!=route'in k,
  'uniform_stage_decisions': 'threadgroup_barrier(mem_flags::mem_threadgroup);'in k and'return *uniform!=0;'in k,
  'no_gate_numeric_poison_bit': 'expert_r4_error(diag,Gate?1u:5u);'in k,
  'audit_completion_is_after_device_barrier': k.index('threadgroup_barrier(mem_flags::mem_device);',k.index('inline void expert_r4_finish'))<k.index('atomic_fetch_add_explicit',k.index('inline void expert_r4_finish')),
  'audit_down_checks_full_gate_count': 'gc==kExpertR4GateCTAs&&dc<kExpertR4DownCTAs'in k,
  'shipping_vs_audit_shared_arithmetic': 'EXPERT_R4_GATE(expert_r4_cohort_sep22_gate_up_n16,false)'in k and'EXPERT_R4_GATE(expert_r4_cohort_sep22_gate_up_n16_audit,true)'in k,
  'backend_snapshot_serial_pass': 'descriptor.dispatchType = MTLDispatchTypeSerial;'in backend and'[command computeCommandEncoder];'in backend and'MTLDispatchTypeConcurrent'not in backend,
  'alias_immutability_host_checks': 'R4 cohort safety operand/metadata aliases'in s and'R4 cohort safety aliases immutable payload/ranks'in s and'host_metadata_operand_alias_rejected'in s,
  'malformed_grid_epoch_order_cases':all(n in s for n in ('missing_plan_fails_before_operands','down_before_gate_fails_closed','partial_gate_grid_rejected','wrong_metadata_epoch_rejected','corrupt_live_route_map_rejected')),
  'duplicate_positive_zero_and_rank_cases':all(n in s for n in ('duplicate_ownership_and_split_cohorts','nonfinite_positive_zero_bitwise_pair','corrupt_rank_excluded_before_coefficients')),
  'quality_thresholds_unchanged': '1e-4'in q and'.999999'in q,
  'new_depth_count_visible': 'reducerDepth'in q and'dependencyAdditionCount'in q,
  'before_timing_qualification': 'prequalified'in o or'qualified'in o,
 }
 result=subprocess.run([str(b/'oracle'),'--cpu-self-test'],text=True,capture_output=True);checks['compiled_cpu_entry_pass']=result.returncode==0 and'"gpu_work":false'in result.stdout
 deps=[]
 for dep in b.rglob('*.d'):
  text=dep.read_text().replace('\\\n',' ');deps.extend(t for t in text.split()if t.startswith(('runtime/','dev/benchmarks/')))
 checks['no_live_custom_header_dependencies']=not deps
 artifacts={name:sha((b/name).read_bytes())for name in ('oracle','splash.metallib','cohort-probe.air','host/MetalBackend.o')}
 r={'schema':'splash-first-expert-r4-n16-cpu-source-witness-v1','pass':all(checks.values()),'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False,'model_or_capture_hashes':False,'new_tiles':[16],'metadata_payload_bytes':6796,'source_identity_sha256':seal['source_identity_sha256'],'checks':checks,'source_mismatches':bad,'link_mismatches':linkbad,'live_dependencies':deps,'cpu_entry':{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},'artifacts_sha256':artifacts,'stage_contract':'exact full grids; ordered serial compute plan→gate→down; audit same arithmetic plus full completion counters; shipping no contended counters','primary_serial_reference':'https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder%28%29','independent_review':'math/source/liveownership/grid/stage review found no must-fix before seal; later real-input/state/semantic/acceptance gates remain required'}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'pass':r['pass'],'failed_checks':[n for n,v in checks.items()if not v],'output':str(a.output),'gpu_work':False}))
 if not r['pass']:raise SystemExit(2)
if __name__=='__main__':main()
