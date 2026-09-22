#!/usr/bin/env python3
"""Prepare only current Root partition commands; no inputs/model/captures opened."""
import argparse,hashlib,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True);ap.add_argument('--environment-command',type=Path,default=ROOT/'build/current-batch-native-clock-sep22-v5/root-3-command.txt');a=ap.parse_args();b=a.build.resolve();m=json.loads((b/'CPU_READY.json').read_text());raw=shlex.split(a.environment_command.read_text());env={}
 if not m['pass']or m['GPU_executed']:raise ValueError('CPU-sealed current oracle required')
 for i,t in enumerate(raw[:-1]):
  if t=='--env':k,v=raw[i+1].split('=',1);env[k]=v
 for k in ['SPLASH_FLASH_BATCH','SPLASH_FLASH_BATCH_MTP','SPLASH_FLASH_BATCH_MTP_PREFILL','SPLASH_FLASH_BATCH_PREFILL','SPLASH_FLASH_MTP','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY','SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY','SPLASH_FLASH_BATCH_QSA_BULK_PREFILL','SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22']:env[k]='1'
 for k in ['SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21','SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT','SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE','SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21','SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22']:env[k]='0'
 env.pop('SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22',None);env['SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS']='0';env['SPLASH_FLASH_MTP_DRAFT_DEPTH']='3';env['SPLASH_FLASH_ALLROWS_FULL512_TARGET']='1'
 order=[['initial','fresh-r16.pending'],['fresh-r16.committed','same-r8.future.completed'],['b2.initial','fresh-r8.pending'],['fresh-r8.future.completed']];records=[]
 for i,labels in enumerate(order,1):
  stem='sep22-current-BQSA4-integer-batchverify-v5-part'+str(i);spill=ROOT/'build/release/flash'/(stem+'-spill');control_report=ROOT/'build/release/flash'/(stem+'-export.json')
  for role in ['export','compare']:
   binary='oracle-control'if role=='export'else'oracle-candidate';report=ROOT/'build/release/flash'/(stem+'-'+role+'.json');p=b/(f'root-part{i}-{role}-command.json');new=dict(env);new['SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22']=str(int(role=='compare'))
   if report.exists()or p.exists()or(role=='export'and spill.exists()):raise ValueError('fresh partition/command output required')
   cmd={'schema':'current-BQSA4-integer-batchverify-bounded-Root-command-v1','Root_GPU_only':True,'build':str(b),'cwd':str(ROOT),'role':role,'environment':new,'argv':[str(b/binary),'--gpu',role,str(b/'splash.metallib'),str(ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(ROOT/'build/release/flash/prefill4k-fixture/code2048.tokens.json'),str(report),str(spill),','.join(labels)],'control_report':str(control_report)if role=='compare'else None,'oracle_sha256':m['artifact_sha256'][binary],'metallib_sha256':m['metallib_sha256'],'payload_reads_or_hashes_by_preparer':0,'spill_limit_bytes':4<<30,'fullphysical_all18_inherited':False,'head_or_service_lifecycle_or_quality_or_timing_qualified':False}
   p.write_text(json.dumps(cmd,indent=2)+'\n');digest=sha(p);runner=ROOT/'dev/benchmarks/expert_batch_b4_composition_sep22/qa/run_root.py';sh=p.with_suffix('.sh');sh.write_text('#!/bin/sh\nset -eu\nexec '+shlex.quote(str(ROOT/'.venv/bin/python'))+' -B '+shlex.quote(str(runner))+' '+shlex.quote(str(p))+' '+shlex.quote(digest)+'\n');sh.chmod(0o755);records.append({'ordinal':i,'role':role,'labels':labels,'command':str(p),'command_SHA':digest,'script':str(sh),'report':str(report),'spill':str(spill)})
 plan={'schema':'current-BQSA4-integer-selected7-replay-partition-plan-v1','required_fullphysical_checkpoints':m['full_physical_selected_labels'],'all18_legacy_logical_output_max16_campaign_retained':True,'all18_fullphysical_inherited':False,'partition_order':order,'commands':records,'GPU_started':False,'model_token_operand_response_capture_payload_reads':0};p=b/'partition-plan.json';p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'plan':str(p),'commands':8,'GPU_started':False,'payload_reads':0}))
if __name__=='__main__':main()
