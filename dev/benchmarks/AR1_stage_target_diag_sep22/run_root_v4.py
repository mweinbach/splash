#!/usr/bin/env python3
"""Root-only one native standard request; imports/tests never read token files."""
import argparse,fcntl,hashlib,json,math,os,pathlib,struct,subprocess,sys,time
ROOT=pathlib.Path('/Users/mweinbach/Projects/splash')
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)
def nested(data,path):
    for key in path.split('.'):data=data[key]
    return data
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.writing');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def audit_profile(data,diagnostic_id):
    wanted={'schema':'private-standard-AR1-stage-diagnostic-sep22-v1','diagnostic_valid':True,
        'request_id':1,'generation':1,'ordinary_AR1':True,'MTP_state_present':False,
        'physical_rows':1,'logit_rows':1,'begin':2050,'returned_length':2051,'AR1_ordinal':3,
        'prior_completed_unprofiled_AR1_calls':2,'maximum_samples':1,'samples_attempted':1,
        'single_whole_target_command':True,'legacy_dispatch_replay':False,'profiling_restored_off':True,
        'stale_profiles_at_activation':0,'profiles_after_target':1,'governor_reservation_bytes':64<<20,
        'reservation_admitted':True,'reservation_released':True,'instrumentation_is_performance_perturbation':True,
        'throughput_baseline':False,'SourceWorld_qualified':False,'Forward_graph_snapshot_hook':False,
        'numeric_inline_ABI_values':None,'tensor_payload_reads':0,'tensor_payload_hashes':0,'token_payload_reads':0,
        'baseline_source_identity_sha256':'162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a',
        'diagnostic_source_id':diagnostic_id}
    need(isinstance(data,dict) and all(type(data.get(k)) is type(v) and data.get(k)==v for k,v in wanted.items()),'Exact selected third ordinaryAR1 metadata required')
    profile=data['raw_command_profile'];count=profile['dispatch_count']
    need(type(count)is int and 1<=count<=4096 and len(profile['dispatches'])==count,'Bounded full actual Core dispatch metadata')
    need(profile['mode']=='stage' and profile['status']=='complete' and profile['encoder_boundaries_altered'] is True and
        profile['sampling_barriers'] is False and profile['dropped_profiles_before']==0 and profile['dispatch_metadata_truncated'] is False and
        profile['dispatches_truncated'] is False and profile['dispatches_total']==profile['dispatches_emitted']==count,'Complete one-command stage/noDrop/noTrunc')
    need(profile['command_kernel_timing_valid'] is True,'Actual command GPU timestamps required')
    for index,dispatch in enumerate(profile['dispatches']):
        need(dispatch['index']==index and isinstance(dispatch['pipeline'],str) and 0<len(dispatch['pipeline'])<=256,'Actual sequential pipeline metadata')
        for name in ('threadgroups','threads_per_threadgroup'):
            need(type(dispatch[name])is list and len(dispatch[name])==3 and all(type(v)is int and v>0 for v in dispatch[name]),'Positive actual Core dispatch geometry')
        bindings=dispatch['bindings']
        need(dispatch['bindings_truncated'] is False and dispatch['bindings_total']==len(bindings)<=32,'Exact bounded Core binding metadata')
        need(len({b['index'] for b in bindings})==len(bindings) and all(type(b['index'])is int and 0<=b['index']<32 and
            type(b['size_bytes'])is int and b['size_bytes']>=0 and type(b['inline_bytes'])is bool for b in bindings),'Exact index/extent/inlinekind bindings')
        need(dispatch['timestamps_valid'] is True and type(dispatch['gpu_seconds'])in(int,float) and
            math.isfinite(dispatch['gpu_seconds']) and dispatch['gpu_seconds']>=0 and dispatch['gpu_start_timestamp']>0 and
            dispatch['gpu_end_timestamp']>=dispatch['gpu_start_timestamp'],'Valid timing for every actual dispatch')
    need(data['governor_snapshots']['reservation_held']['reserved_bytes']>=64<<20,'64MiB admission before active stage')
    need(data['governor_snapshots']['after_reservation_release']['reserved_bytes']==data['governor_snapshots']['before_activation']['reserved_bytes']==0 and
        data['governor_snapshots']['after_reservation_release']['denied_reservations']==0,'Clean diagnostic Gov release')
    return data

def cache_counts_and_standard(result,before,after):
    from server import protocol as wire
    need(result.start is not None and result.start.request_id==1 and result.start.cache_disposition==wire.CacheDisposition.MISS and
        result.start.matched_prompt_tokens==0,'Actual Start cacheMISS/matched0 required')
    need(result.done.prompt_tokens==2048 and result.done.completion_tokens==64 and len(result.tokens)==64,'Actual Done2048/full64')
    need(nested(after,'metrics.prefill_input_tokens')-nested(before,'metrics.prefill_input_tokens')==2048 and
        nested(after,'metrics.autoregressive_output_tokens')-nested(before,'metrics.autoregressive_output_tokens')==64,'Actual full native token counters')
    widths={key:nested(after,'scheduler.decode_batches_by_width.'+key)-nested(before,'scheduler.decode_batches_by_width.'+key) for key in ('b1','b2','b3','b4')}
    need(widths=={'b1':63,'b2':0,'b3':0,'b4':0},'Exactly63 real ordinaryAR1 target calls after prefill seed')
    need(nested(after,'mtp.eligible_requests')-nested(before,'mtp.eligible_requests')==0 and
        nested(after,'mtp.autoregressive_requests')-nested(before,'mtp.autoregressive_requests')==1 and
        nested(after,'mtp.verification_cycles')==nested(before,'mtp.verification_cycles')==0 and
        nested(after,'mtp.drafted_tokens')==nested(before,'mtp.drafted_tokens')==0 and
        nested(after,'mtp.singleton_teacher_bulk.completed_teacher_commands')==nested(before,'mtp.singleton_teacher_bulk.completed_teacher_commands')==0 and
        nested(after,'mtp.enabled') is False and nested(after,'mtp.singleton_teacher_bulk.requested') is False,'No MTP/head/teacherbulk substitution')
    return {'cache_disposition':'MISS','matched_prompt_tokens':0,'Done_prompt_tokens':2048,'Done_completion_tokens':64,
        'native_prefill_counter_delta':2048,'native_output_counter_delta':64,'actual_native_AR1_target_calls':63}

class TrackedProcess:
    def __init__(self,process):
        self.process=process;self.stdin=process.stdin;self.stdout=process.stdout;self.terminate_called=False;self.kill_called=False
        self.graceful_stop_active=False;self.graceful_stop_requested=False;self.stdin_EOF_sent=False;self.generic_terminate_requests_deferred=0
    def __getattr__(self,name):return getattr(self.process,name)
    def terminate(self):
        if self.graceful_stop_active:
            self.generic_terminate_requests_deferred+=1;return None
        self.terminate_called=True;return self.process.terminate()
    def kill(self):self.kill_called=True;return self.process.kill()
def group_gone(pid):
    try:os.killpg(pid,0)
    except ProcessLookupError:return True
    except PermissionError:return False
    return False

def close_with_graceful_EOF(runtime,processes,timeout=30):
    """Native Transport.readLoop treats stdin EOF as clean Stop.

    Fence only concurrent generic SIGTERM while this explicit bounded wait is
    active. A deferred request is recorded; actual signal flags stay truthful.
    Failure/timeout falls through to normal runtime cleanup and never qualifies.
    """
    outcome={'stdin_EOF_Stop_requested':False,'stdin_EOF_sent':False,'exit_observed_before_runtime_close':False,
             'runtime_close_after_process_exit':False,'errors':[]}
    if len(processes)==1:
        process=processes[0]
        if process.poll() is None:
            process.graceful_stop_requested=True;process.graceful_stop_active=True
            outcome['stdin_EOF_Stop_requested']=True
            try:
                need(process.stdin is not None and not process.stdin.closed,'Live stdin required for explicit EOF Stop')
                process.stdin.close();process.stdin_EOF_sent=True;outcome['stdin_EOF_sent']=True
                process.wait(timeout=timeout)
                outcome['exit_observed_before_runtime_close']=process.poll() is not None
            except Exception as error:outcome['errors'].append('EOFStop:'+type(error).__name__+': '+str(error))
            finally:process.graceful_stop_active=False
        else:outcome['exit_observed_before_runtime_close']=True
    if runtime is not None:
        outcome['runtime_close_after_process_exit']=len(processes)==1 and processes[0].poll() is not None
        try:runtime.close()
        except Exception as error:outcome['errors'].append('runtimeclose:'+type(error).__name__+': '+str(error))
    return outcome

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command',type=pathlib.Path,required=True);parser.add_argument('--command-sha256',required=True)
    parser.add_argument('--run-root-gpu',action='store_true');args=parser.parse_args()
    need(args.run_root_gpu,'Explicit Root GPU flag required')
    need(sha(args.command)==args.command_sha256,'External Root command SHA changed')
    command=json.loads(args.command.read_text())
    need(command['schema']=='Root-current-standard-AR1-stage-one-native-command-v1','Unknown command')
    need(command['context']==2048 and command['output_tokens']==64 and command['max_context']==16384 and
        command['request_id']==command['generation']==command['requests']==1,'Exact one2K64 current standard packet')
    for path,digest in command['program_pins'].items():need(sha(path)==digest,'Program pin changed:'+path)
    build=pathlib.Path(command['build'])
    need(sha(build/'splash-flash')==command['exe_sha256'] and sha(build/'splash.metallib')==command['library_sha256'],'Current diagnostic artifacts changed')
    for name in ('report','profile','native_log'):need(not pathlib.Path(command[name]).exists(),'Fresh output required:'+name)
    # Token file access exists only here after explicit Root invocation/closure.
    token_path=pathlib.Path(command['tokens']);need(sha(token_path)==command['tokens_file_sha256'],'Root canonical token file changed')
    tokens=json.loads(token_path.read_text());need(type(tokens)is list and len(tokens)==2048 and all(type(v)is int and 0<=v<248320 for v in tokens),'Exact canonical2048 Root tokens')
    need(hashlib.sha256(struct.pack('<2048I',*tokens)).hexdigest()==command['tokens_u32le_sha256'],'Root frontend55cf witness changed')
    from server.runtime import MultiplexedRuntime,GenerationRequest,Deadline
    from server import protocol as wire
    environment={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_','FLASH_'))};environment.update(command['environment'])
    needed={'SPLASH_FLASH_MTP':'0','SPLASH_FLASH_BATCH_MTP':'0','SPLASH_FLASH_BATCH_MTP_PREFILL':'0',
        'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'0','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY':'0',
        'SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21':'1','SPLASH_FLASH_GDN_PREFILL_FMA_SEP21':'1',
        'SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21':'1','SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21':'1',
        'SPLASH_FLASH_DIAG_AR1_TARGET_STAGE_SEP22':'1','SPLASH_FLASH_DIAG_AR1_STAGE_OUTPUT_SEP22':command['profile']}
    need(all(environment.get(k)==v for k,v in needed.items()),'Exact standard/offteacher/allprefill/profile flags')
    argv=[str(build/'splash-flash'),'serve-flash-native',command['package'],'16384','auto']
    path=pathlib.Path(command['report']);lock=(ROOT/'build/splash-tuning-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    log=pathlib.Path(command['native_log']).open('x');processes=[];runtime=None
    report={'schema':'current-standard-AR1-target-stage-one-native-diagnostic-runner-v4','completed':False,'evidence_valid':False,
        'performance_claim':False,'instrumented_stage_not_canonical_performance':True,'token_payload_exported':False,'tensor_payload_exported':False,
        'tokens_count':2048,'tokens_file_sha256':command['tokens_file_sha256'],'tokens_u32le_sha256':command['tokens_u32le_sha256'],
        'baseline_source_identity_sha256':command['baseline_source_identity_sha256'],'diagnostic_source_id':command['diagnostic_source_id'],
        'diagnostic_exe_sha256':command['exe_sha256'],'library_sha256':command['library_sha256'],'command_sha256':args.command_sha256,'errors':[]}
    def create():
        need(not processes,'Automatic native restart forbidden')
        process=TrackedProcess(subprocess.Popen(argv,cwd=ROOT,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0,start_new_session=True));processes.append(process);return process
    def status():return json.loads(runtime.status(timeout=5).json)
    def idle():
        until=time.monotonic()+30
        while time.monotonic()<until:
            value=status();need(value['ready'] is True and value['memory_pressure']=='normal','Current worker ready/normal required')
            scheduler=value['scheduler']
            if not scheduler['active_requests'] and not scheduler['queued'] and not scheduler['command_in_flight']:return value
            time.sleep(.02)
        raise TimeoutError('Native request not idle')
    try:
        runtime=MultiplexedRuntime(process_factory=create,startup_timeout=600,io_timeout=5);runtime.wait_ready(timeout=600)
        before=idle();report['initial_status']=before
        call=runtime.submit(GenerationRequest(tuple(tokens),64,Deadline.after(180),cohort=wire.Cohort.GREEDY))
        need(call.request_id==1,'First/only native request must ID1');result=call.result(timeout=210);after=idle();report['final_status']=after
        report['actual_cache_and_token_counters']=cache_counts_and_standard(result,before,after)
        report['output_tokens']=len(result.tokens);report['finish_reason']=result.done.reason.name
        need(nested(after,'requests.submitted')-nested(before,'requests.submitted')==1 and
            nested(after,'requests.completed')-nested(before,'requests.completed')==1 and
            nested(after,'requests.failed')==nested(before,'requests.failed') and nested(after,'requests.cancelled')==nested(before,'requests.cancelled'),'Exactly one completed nofailure native request')
        need(nested(after,'metrics.metal_failures')==nested(before,'metrics.metal_failures'),'No Metal failures')
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    finally:
        report['graceful_EOF_stop']=close_with_graceful_EOF(runtime,processes)
        report['errors'].extend(report['graceful_EOF_stop']['errors'])
        report['native_processes']=len(processes);report['native_returncodes']=[p.poll() for p in processes]
        report['native_termination_signals']=[{'terminate_called':p.terminate_called,'kill_called':p.kill_called,'generic_terminate_requests_deferred':p.generic_terminate_requests_deferred,'graceful_stop_requested':p.graceful_stop_requested,'stdin_EOF_sent':p.stdin_EOF_sent} for p in processes]
        report['native_process_group_gone']=len(processes)==1 and group_gone(processes[0].pid)
        report['native_terminal']=report['graceful_EOF_stop']['stdin_EOF_sent'] and report['graceful_EOF_stop']['exit_observed_before_runtime_close'] and report['graceful_EOF_stop']['runtime_close_after_process_exit'] and len(processes)==1 and processes[0].poll()==0 and not processes[0].terminate_called and not processes[0].kill_called and report['native_process_group_gone']
        log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    try:
        need(report['native_terminal'],'Clean one-backend Stop/owner destruction required')
        raw=json.loads(pathlib.Path(command['profile']).read_text());report['profile']=raw
        audit_profile(raw,command['diagnostic_source_id']);report['profile_file_sha256']=sha(command['profile'])
    except Exception as error:report['errors'].append('profile:'+type(error).__name__+': '+str(error))
    report['completed']=True;report['evidence_valid']=not report['errors'];report['backend_process_destroyed']=report['native_terminal']
    save(path,report);print(json.dumps({'report':str(path),'evidence_valid':report['evidence_valid'],'diagnostic_not_performance':True}))
    return 0 if report['evidence_valid'] else 1

if __name__=='__main__':raise SystemExit(main())
