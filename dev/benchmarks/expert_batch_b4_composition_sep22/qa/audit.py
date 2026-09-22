#!/usr/bin/env python3
"""CPU source/dependency/control-plan audit; no model or capture data opened."""
import argparse,hashlib,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True);a=ap.parse_args();b=a.build.resolve();m=json.loads((b/'CPU_READY.json').read_text());worker=Path(m['current_worker']);wm=json.loads((worker/'overlay-manifest.json').read_text());checks=[]
 def require(ok,n):
  if not ok:raise ValueError(n)
  checks.append(n)
 require(m['pass']and not m['GPU_executed']and m['model_operand_response_capture_payload_reads']==0,'CPUOnlyCurrentSource')
 for r in m['sources']:require(sha(b/'source'/r['path'])==r['sha256'],'frozenSource:'+r['path'])
 require(len(m['objects'])==53 and len(m['header_census'])==50,'Actual53Object50TUCensus')
 for r in m['objects']:
  require(sha(b/r['path'])==r['sha256'],'object:'+r['path'])
  require(bool(r['dependencies']),'actualPrivateCompilerDependencies:'+r['path'])
 for role in ['control','candidate']:
  require(m['CPU_'+role]['pass']and not m['CPU_'+role]['gpu_executed'],'CompiledCPUFixture:'+role)
  require(m['CPU_'+role]['selected2_spill_plan_upper']==3991830528 and m['CPU_'+role]['selected2_spill_plan_upper']<4<<30,'Selected2Bound:'+role)
 require(sha(b/'splash.metallib')==m['metallib_sha256']==sha(worker/'splash.metallib'),'ExactCurrent78AIRLibrary')
 # Every original current executed method remains a literal contiguous body.
 for rec in wm['files']:
  rel=rec['path'];orig=(worker/'source'/rel).read_bytes();clone=(b/'source'/rel).read_bytes()
  if rel in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashBatchVerify.cpp']:require(orig in clone,'OriginalEntireCPPMethodBodiesLiteral:'+rel)
  elif rel in ['runtime/flash/FlashForward.hpp','runtime/flash/FlashBatchVerify.hpp','runtime/metal/MetalBackend.hpp']:continue
  else:require(orig==clone,'AllOtherCurrentSourceByteIdentical:'+rel)
 text=(b/'source/dev/benchmarks/expert_batch_b4_composition_sep22/qa/oracle.mm').read_text()
 require('stateWitness'not in text,'NoHashOnlyRejectedStateWitness')
 require('std::memcpy(hostSnapshot_.data()+out.bytes,data,buffer.sizeBytes())'in text and 'std::memcmp(p.data,arena.data()+p.offset,p.bytes)'in text,'ExactFullPhysicalBeforeAfterRejectedBytes')
 require('for(const auto &p:Access::batchPlanes(batch_))'in text and 'for(const auto &p:Access::planes(*slots_[lane].state))'in text,'RejectedSnapshotWhole134RequestAndAllBatchBacking')
 require('store_.activate();groupedFresh({0,1,2,3}'in text,'DeferredSinkAfterNativeHostAndSpillGuards')
 require('verify({0,1,2,3},4,"fresh-r16");commit({0,1,2,3},{0,1,2,4}'in text,'ActualBQSA4R16MixedPrefixSource')
 require('verify({1,2},4,"same-r8")'in text and 'allocateOwner(0);allocateOwner(1);groupedFresh({0,1}'in text,'ValidTerminal0SurvivorR8AndIndependentB2Recreation')
 require('after==before+expected'in text and 'lanes.size()==4?48:0'in text,'ActualNarrowedBQSA4CounterDelta48B2Zero')
 require('BQSA_numeric_parent_routes_raw'in text and 'compactNativeBatchVerifyNumericParent()'in text,'RawPublicRoutesAndNumericParentNoSuffixStrip')
 require('erase('not in text and '.replace('not in text,'NoRuntimeMarkerStripping')
 plan=json.loads((b/'partition-plan.json').read_text());require(len(plan['commands'])==8 and len(plan['partition_order'])==4,'FourSerializedReplayPartitionsEightRootProcesses')
 require([x for pair in plan['partition_order']for x in pair]==m['full_physical_selected_labels'],'ExactRootSelected7Once')
 for i in range(4):
  pair=plan['commands'][i*2:i*2+2];commands=[json.loads(Path(r['command']).read_text())for r in pair]
  for r,c in zip(pair,commands):require(sha(Path(r['command']))==r['command_SHA'],'CommandPin:'+str(i)+c['role'])
  e0,e1=[dict(c['environment'])for c in commands];diff={k:[e0.get(k),e1.get(k)]for k in set(e0)|set(e1)if e0.get(k)!=e1.get(k)}
  require(diff=={'SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22':['0','1']},'OnlyTargetIntegerFlagDiff:'+str(i))
  require(e0['SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22']=='1'and e1['SPLASH_FLASH_MTP_DRAFT_DEPTH']=='3','SameNarrowedBQSA4MTP3BothRoles:'+str(i))
  require(commands[0]['argv'][8]==commands[1]['argv'][8], 'SameEntireReplaySelectedPartition:'+str(i))
  require(not Path(commands[0]['argv'][6]).exists()and not Path(commands[1]['argv'][6]).exists(),'FreshRootReportsNotRun:'+str(i))
 sources={}
 for p in sorted(Path(__file__).parent.glob('*')):
  if p.is_file():compile(p.read_text(),str(p),'exec')if p.suffix=='.py'else None;sources[p.name]=sha(p)
 # Freeze own remaining preparer/runner/audit sources into the reviewable build.
 dst=b/'machinery';dst.mkdir(exist_ok=True)
 for p in Path(__file__).parent.iterdir():
  if p.is_file():(dst/p.name).write_bytes(p.read_bytes())
 out={'schema':'current-BQSA4-integer-batchverify-final-CPU-source-seal-v1','pass':True,'GPU_executed':False,'model_input_operand_response_export_payload_reads':0,'CPU_READY_SHA':sha(b/'CPU_READY.json'),'current_worker_seal_SHA':sha(worker/'compiled-cpu-seal.json'),'source_identity':m['current_source_identity'],'native_prefill_plan':5093900288,'native_max16_verify_plan':524435456,'host_exact_rejection_copy_upper':1704230912,'selected2_spill_upper':3991830528,'fullphysical_selected7':m['full_physical_selected_labels'],'all18_legacy_campaign_logic_output_max16_checks_retained':True,'fullphysical_all18_inherited':False,'actual_FP_body_changes':False,'actual_new_GPUMath_or_ModelState_proved':False,'Root_GPU_required':True,'independent_review_required_before_GPU':True,'artifact_sha256':{n:sha(b/n)for n in ['oracle-control','oracle-candidate','splash.metallib','CPU_READY.json','partition-plan.json']},'machinery_source_SHA':sources,'checks':checks}
 p=b/'final-cpu-seal.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'seal':str(p),'sealSHA':sha(p),'GPU_started':False}))
if __name__=='__main__':main()
