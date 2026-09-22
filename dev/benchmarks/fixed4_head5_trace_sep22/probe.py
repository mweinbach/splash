#!/usr/bin/env python3
"""Root-only single diagnostic request; timestamps never become throughput."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
TRACE_FLAG='SPLASH_FLASH_REQUEST_COMMAND_TRACE'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.writing');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def import_exact(path,digest,name):
    if sha(path)!=digest:raise ValueError('Frozen dependency drift:'+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def u64(value):return type(value)is int and 0<=value<2**64

def audit_trace(records,instance):
    if not isinstance(records,list)or not records:raise ValueError('Completed trace records required')
    identities=set();sequences=[]
    for record in records:
        if (record.get('schema')!='splash-request-command-trace-v1' or record.get('instrumentation_on')is not True
                or record.get('instance_id')!=instance or record.get('lanes')!=1
                or type(record.get('submission_expected'))is not bool or type(record.get('profile_present'))is not bool):
            raise ValueError('Actual trace source/profile/timestamp contract differs')
        requests=record.get('requests')
        if not isinstance(requests,list)or len(requests)!=1:raise ValueError('One native lane required')
        lane=requests[0]
        if not all(u64(lane.get(key))for key in ('request_id','generation','input_rows'))or not lane['request_id']or not lane['generation']:
            raise ValueError('Native request identity invalid')
        if record.get('actual_rows')!=lane['input_rows']:raise ValueError('Actual row count differs from native lane')
        identities.add((lane['request_id'],lane['generation']))
        if record['submission_expected']is False:
            if (record['profile_present']is not False or record.get('event')!='resolved_without_gpu_submit'
                    or record.get('phase')!='target_prefix_restore' or record.get('role')!='target_restore'):
                raise ValueError('Only source zero-work prefix restoration may omit GPU submission')
            continue
        if record['profile_present']is not True or record.get('profiling_mode')!='command' or record.get('hardware_timestamps_valid')is not True:
            raise ValueError('Submitted command needs its actual hardware profile')
        begin,end=(record.get(key)for key in ('gpu_hardware_start_mach_seconds','gpu_hardware_end_mach_seconds'))
        if not all(type(value)in (int,float)and math.isfinite(value)and value>0 for value in (begin,end))or end<begin:
            raise ValueError('Actual GPU command timestamps invalid')
        sequence=record.get('command_sequence')
        if not u64(sequence):raise ValueError('Native command sequence missing')
        sequences.append(sequence)
    if len(identities)!=1 or any(b<=a for a,b in zip(sequences,sequences[1:])):raise ValueError('Native identity/restart/order differs')
    folds=[];followups=[]
    for index,record in enumerate(records):
        if record.get('phase')=='committed_head_fold'and record.get('role')=='mtp_head'and record['actual_rows']==5 and record['submission_expected']:
            folds.append(index)
            later=next((j for j in range(index+1,len(records))if records[j].get('phase')=='target_verify'and records[j].get('role')=='target_trunk'and records[j]['actual_rows']==5 and records[j]['submission_expected']),None)
            if later is not None:followups.append({'fold_record':index,'fold_command_sequence':record['command_sequence'],
                                                 'later_Verify5_record':later,'later_Verify5_command_sequence':records[later]['command_sequence']})
    return {'native_request_id_generation':list(next(iter(identities))), 'records':len(records),
            'committed_head_fold_rows5_events':len(folds),'fold5_with_later_Verify5':followups,
            'direct_event_proved':bool(followups),'scope':'Completed native fold5 command and later Verify5 for the same actual request/generation; no full Head cache-bit proof or performance claim.'}

def validate_command(command):
    if command.get('schema')!='Root-fixed4-head5-single-trace-command-v1':raise ValueError('Unknown diagnostic command')
    for path,digest in command['pins'].items():
        if sha(path)!=digest:raise ValueError('Frozen Root program/control/binding drift:'+path)
    base=json.loads(Path(command['parent_command']).read_text())
    if command['parent_argv']!=base['argv']or command['parent_environment']!=base['environment']:raise ValueError('Canonical parent controls differ')
    if command['budget']!=64 or command['width']!=1 or command['requests']!=1:raise ValueError('Only one bounded64-token B1 diagnostic')
    trace=Path(command['trace']);report=Path(command['report']);log=Path(command['server_log'])
    if not trace.is_absolute()or any(path.exists()for path in (trace,report,log)):raise ValueError('Fresh absolute trace/report/log required')
    if TRACE_FLAG in base['environment']:raise ValueError('Parent already has a trace override')
    return base

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command',type=Path,required=True);parser.add_argument('--command-sha256',required=True)
    parser.add_argument('--run-root-gpu',action='store_true');args=parser.parse_args(argv)
    if not args.run_root_gpu:raise ValueError('Explicit Root GPU diagnostic required')
    if sha(args.command)!=args.command_sha256:raise ValueError('Externally registered command differs')
    command=json.loads(args.command.read_text());base=validate_command(command)
    driver=import_exact(Path(base['argv'][2]),base['pins'][base['argv'][2]],'_head5_frozen_normal_driver')
    original=driver.parse_args(base['argv'][3:]);original.output=Path(command['report']);original.port=command['port']
    # Original prompt builder and mode environment remain unchanged. Root alone
    # executes tokenizer/data work; preparation/tests never call prepare_plan.
    from dev.benchmarks.mtp_fixed4_r5_sep22 import policy_quality as fixed
    runner=fixed.load(original.r5_binding,original.r5_binding_sha256)
    environment,provenance=driver.environment_for(original,'4');environment[TRACE_FLAG]=command['trace']
    plan=driver.prepare_plan(original)
    if len(plan['prompts'])!=1:raise ValueError('One canonical prompt required')
    prompt=plan['prompts'][0]
    if prompt['prompt_tokens']!=2048 or prompt['canonical_recovered_coding2048']is not True:raise ValueError('Exact canonical2K coding prompt required')
    report={'schema':'Root-fixed4-head5-single-diagnostic-v1','completed':False,'evidence_valid':False,'direct_head5_event_proved':False,
            'performance_claim':False,'full_Head_cache_bitproof':False,'main_receipt_changed':False,
            'command_sha256':sha(args.command),'budget':64,'width':1,'warmup':0,'requests':1,
            'canonical_prompt_count':2048,'canonical_prompt_sha256':prompt['prompt_u32le_sha256'],
            'environment_provenance':provenance,'trace_override':{TRACE_FLAG:command['trace']},'trace':command['trace'],'errors':[]}
    path=Path(command['report']);save(path,report);server=None
    server_argv=[sys.executable,'-u',str(ROOT/'server/server.py'),'--local-package',str(original.package),'--tokenizer',str(original.tokenizer),
                 '--model',original.model,'--binary',str(original.binary),'--port',str(original.port),'--max-memory',original.max_memory,
                 '--max-context',str(original.max_context),'--no-webui']
    lockpath=ROOT/'build/splash-tuning-gpu.lock'
    with lockpath.open('a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket()as sock:
            if sock.connect_ex(('127.0.0.1',original.port))==0:raise ValueError('Fresh private diagnostic port required')
        with Path(command['server_log']).open('x')as log:
            try:
                server=subprocess.Popen(server_argv,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                report['server_command']=server_argv;report['server_pid']=server.pid;save(path,report)
                client=driver.HTTPClient(SimpleNamespace(base_url=f'http://127.0.0.1:{original.port}',timeout=original.timeout,model=original.model))
                deadline=time.monotonic()+original.startup_timeout
                while True:
                    if server.poll()is not None:raise RuntimeError('Diagnostic server exited during startup')
                    try:before=driver.wait_idle(client,original,timeout=min(original.idle_timeout,max(.01,deadline-time.monotonic())));break
                    except (OSError,ValueError,TimeoutError):
                        if time.monotonic()>=deadline:raise TimeoutError('Diagnostic startup timeout')
                        time.sleep(.5)
                errors=runner.fixed4_policy_status(before,None,None,'mtp3')
                if errors:raise ValueError(errors)
                witness=driver.frontend_prompt_tokens(client,{'model':original.model,'messages':[{'role':'user','content':prompt['content']}],'reasoning_effort':'none'})
                if driver.token_hash(witness)!=prompt['prompt_u32le_sha256']:raise ValueError('Frontend canonical prompt differs')
                report['status_before']=before
                wave=driver.run_wave(client,prompt,64,1,original);report['request']=wave;save(path,report)
                after=driver.wait_idle(client,original,before['identity'],driver.get_path(before,'status_snapshot.steady_seconds'))
                report['status_after']=after;details,errors=runner.coverage(before,after,{'prompt_token_count':2048,'compact_scope':'singleton-main','body':{'max_completion_tokens':64}},True,'mtp3')
                errors+=runner.fixed4_policy_status(after,None,None,'mtp3')
                delta=driver.counter_delta(before,after)
                for key,wanted in {'requests.submitted':1,'requests.completed':1,'requests.cancelled':0,'requests.failed':0,'metrics.metal_failures':0,'metrics.prefill_input_tokens':2048,'metrics.autoregressive_output_tokens':64}.items():
                    if delta.get(key)!=wanted:errors.append('Diagnostic native counter differs:'+key)
                report['coverage']=details;report['native_counter_delta']=delta;report['errors']+=errors
                for snapshot in (before,after):
                    if driver.get_path(snapshot,'request_command_trace.enabled')is not True:report['errors'].append('Diagnostic sink not enabled')
                    for key in ('missing_profiles','unexpected_profiles'):
                        if driver.get_path(snapshot,'request_command_trace.'+key)!=0:report['errors'].append('Diagnostic source profile count invalid:'+key)
                report['request_and_native_evidence_valid']=wave['valid']and not errors
            except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
            finally:
                if server is not None:report['unload_evidence']=driver.unload(server)
                save(path,report)
    # Read the trace only after Stop/process-group disappearance closes the sink.
    unload=report.get('unload_evidence',{})
    if unload.get('process_group_gone')is not True or unload.get('returncode')!=0 or unload.get('post_parent_exit_sigkill_required')is not False:
        report['errors'].append('Native source/trace EOF requires graceful clean process disappearance')
    if Path(command['trace']).exists()and not report['errors']:
        try:
            records=[json.loads(line)for line in Path(command['trace']).read_text().splitlines()if line.strip()]
            audit=audit_trace(records,report['status_before']['identity']['engine_instance_id']);report['trace_audit']=audit
            report['trace_sha256']=sha(command['trace']);report['direct_head5_event_proved']=audit['direct_event_proved']
        except Exception as error:report['errors'].append('Trace '+type(error).__name__+': '+str(error))
    report['completed']=True;report['evidence_valid']=report.get('request_and_native_evidence_valid')is True and not report['errors']
    report['outcome']='proved_direct_fold5_then_Verify5'if report['evidence_valid']and report['direct_head5_event_proved']else'inconclusive_no_fold5_event'if report['evidence_valid']else'invalid_evidence'
    save(path,report);print(json.dumps({'report':str(path),'outcome':report['outcome'],'performance_claim':False}),flush=True)
    return 0 if report['evidence_valid']else 2

if __name__=='__main__':raise SystemExit(main())
