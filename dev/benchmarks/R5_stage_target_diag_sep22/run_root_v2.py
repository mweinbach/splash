#!/usr/bin/env python3
"""Root-only genuine native1 request; source/import/test paths never read tokens."""
import argparse,fcntl,hashlib,json,math,os,pathlib,struct,subprocess,sys,time
ROOT=pathlib.Path('/Users/mweinbach/Projects/splash');sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def nested(d,path):
 for k in path.split('.'):d=d[k]
 return d
def save(path,value):
 tmp=path.with_suffix(path.suffix+'.writing');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def audit_profile(d):
 wanted={'schema':'private-R5-target-stage-diagnostic-sep22-v1','diagnostic_valid':True,'request_id':1,'generation':1,'draft_depth':4,'physical_rows':5,'R5_ordinal':3,'prior_completed_unprofiled_R5_calls':2,'maximum_samples':1,'samples_attempted':1,'profiling_restored_off':True,'stale_profiles_at_activation':0,'profiles_after_target':1,'governor_reservation_bytes':64<<20,'reservation_admitted':True,'reservation_released':True,'tensor_payload_reads':0,'tensor_payload_hashes':0,'legacy_dispatch_replay':False,'single_whole_target_command':True}
 require(isinstance(d,dict) and all(type(d.get(k)) is type(v) and d.get(k)==v for k,v in wanted.items()),'Exactvalid selectedthirdR5 metadata required')
 g=d['Forward_graph'];q=d['raw_command_profile'];count=g['dispatch_count']
 require(type(count) is int and 1<=count<=4096 and g['calls']==1 and g['verification'] is True and g['rows']==5 and g['metadata_truncated'] is False and len(g['dispatches'])==g['metadata_count']==count,'Exactnontruncated genuine Forwardgraph')
 require(q['mode']=='stage' and q['status']=='complete' and q['encoder_boundaries_altered'] is True and q['sampling_barriers'] is False and q['dropped_profiles_before']==0 and q['dispatch_metadata_truncated'] is False and q['dispatch_count']==count and len(q['dispatches'])==count,'Exactone completedstage profile')
 for a,b in zip(g['dispatches'],q['dispatches']):
  require(a['index']==b['index'] and a['pipeline']==b['pipeline'] and a['threadgroups']==b['threadgroups'] and a['threads_per_threadgroup']==b['threads_per_threadgroup'] and a['bindings']==b['bindings'],'Profileactualbindings differfrom recordedgraph')
  require(b['timestamps_valid'] is True and type(b['gpu_seconds']) in (int,float) and math.isfinite(b['gpu_seconds']) and b['gpu_seconds']>=0,'Validnonnegative timestamp foreach dispatch')
 require(d['governor_snapshots']['after_reservation_release']['reserved_bytes']==0 and d['governor_snapshots']['after_reservation_release']['denied_reservations']==0,'Diagnosticreservation/Govresidue')
 return d

class TrackedProcess:
 def __init__(self,process):self.process=process;self.stdin=process.stdin;self.stdout=process.stdout;self.terminate_called=False;self.kill_called=False
 def __getattr__(self,name):return getattr(self.process,name)
 def terminate(self):self.terminate_called=True;return self.process.terminate()
 def kill(self):self.kill_called=True;return self.process.kill()

def check_cache_and_counts(result,before,after):
 from server import protocol as wire
 require(result.start is not None and result.start.request_id==1 and result.start.cache_disposition==wire.CacheDisposition.MISS and result.start.matched_prompt_tokens==0,'ActualStartcacheMISS/matched0 required')
 require(result.done.prompt_tokens==2048 and result.done.completion_tokens==64 and len(result.tokens)==64,'ActualDone2048/full64 required')
 for key,wanted in [('metrics.prefill_input_tokens',2048),('metrics.autoregressive_output_tokens',64)]:require(nested(after,key)-nested(before,key)==wanted,'Actualstatus tokencounter differs:'+key)
 return {'cache_disposition':'MISS','matched_prompt_tokens':0,'Done_prompt_tokens':2048,'Done_completion_tokens':64,'native_prefill_counter_delta':2048,'native_output_counter_delta':64}
def group_gone(pid):
 try:os.killpg(pid,0)
 except ProcessLookupError:return True
 except PermissionError:return False
 return False

def main():
 parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--command',type=pathlib.Path,required=True);parser.add_argument('--command-sha256',required=True);parser.add_argument('--run-root-gpu',action='store_true');a=parser.parse_args()
 require(a.run_root_gpu,'ExplicitRoot GPU permission required');require(sha(a.command)==a.command_sha256,'Externalcommand SHA differs');c=json.loads(a.command.read_text())
 require(c['schema']=='Root-current-fixed4-R5-target-stage-one-native-command-v1','Unknowncommand');require(c['context']==2048 and c['output_tokens']==64 and c['request_id']==1 and c['generation']==1 and c['requests']==1,'Onlyexactone2K64nativepacket')
 for p,s in c['program_pins'].items():require(sha(p)==s,'Programpin changed:'+p)
 b=pathlib.Path(c['build']);require(sha(b/'splash-flash')==c['exe_sha256'] and sha(b/'splash.metallib')==c['library_sha256'],'Diagnostic artifactschanged')
 for name in ['report','profile','native_log']:require(not pathlib.Path(c[name]).exists(),'Freshoutput required:'+name)
 # ROOT invocation only: never executed by source preparation or CPU tests.
 tokens_path=pathlib.Path(c['tokens']);require(sha(tokens_path)==c['tokens_file_sha256'],'Rootexacttokenfile changed');tokens=json.loads(tokens_path.read_text());require(type(tokens) is list and len(tokens)==2048 and all(type(x) is int and 0<=x<248320 for x in tokens),'Exactvalidtokens2048')
 require(hashlib.sha256(struct.pack('<2048I',*tokens)).hexdigest()==c['tokens_u32le_sha256'],'ActualcanonicalHTTPtoken witness differs')
 from server.runtime import MultiplexedRuntime,GenerationRequest,Deadline
 from server import protocol as wire
 env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_','FLASH_'))};env.update(c['environment'])
 require(env['SPLASH_FLASH_MTP']=='1' and env['SPLASH_FLASH_MTP_DRAFT_DEPTH']=='4' and env['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22']=='1','Exactqualifiedfixed4R5flags')
 require(env['SPLASH_FLASH_DIAG_R5_TARGET_STAGE_SEP22']=='1' and env['SPLASH_FLASH_DIAG_R5_STAGE_OUTPUT_SEP22']==c['profile'],'Exactprofileoutput config')
 argv=[str(b/'splash-flash'),'serve-flash-native',c['package'],'16384','auto'];path=pathlib.Path(c['report']);logpath=pathlib.Path(c['native_log']);lock=(ROOT/'build/splash-tuning-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);log=logpath.open('x');processes=[];runtime=None
 report={'schema':'current-fixed4-R5-trained-target-stage-one-native-diagnostic-v1','completed':False,'evidence_valid':False,'performance_claim':False,'instrumented_stage_not_canonical_performance':True,'token_payload_exported':False,'tensor_payload_exported':False,'tokens_count':2048,'tokens_file_sha256':c['tokens_file_sha256'],'tokens_u32le_sha256':c['tokens_u32le_sha256'],'current_baseline_source':'4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a','baseline_exe_sha256':'6f7e22a2ca9c0c9728bf356391d2bde17c4e9bc90ab6295e625cac15e7987d68','diagnostic_exe_sha256':c['exe_sha256'],'library_sha256':c['library_sha256'],'command_sha256':a.command_sha256,'errors':[]}
 def create():
  require(not processes,'Automaticnative restart forbidden');p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0,start_new_session=True);p=TrackedProcess(p);processes.append(p);return p
 def status():return json.loads(runtime.status(timeout=5).json)
 def idle():
  until=time.monotonic()+30
  while time.monotonic()<until:
   s=status();require(s['ready'] is True and s['memory_pressure']=='normal','Workerhealth/memorypressure');q=s['scheduler']
   if not q['active_requests'] and not q['queued'] and not q['command_in_flight']:return s
   time.sleep(.02)
  raise TimeoutError('Nativerequest notidle')
 try:
  runtime=MultiplexedRuntime(process_factory=create,startup_timeout=300,io_timeout=5);runtime.wait_ready(timeout=300);before=idle();report['initial_status']=before
  require(nested(before,'mtp.singleton_maximum_draft_tokens')==4 and nested(before,'mtp.joint_maximum_draft_tokens')==3,'Actualcap4/joint3 required')
  call=runtime.submit(GenerationRequest(tuple(tokens),64,Deadline.after(180),cohort=wire.Cohort.GREEDY));require(call.request_id==1,'FirstandonlynativeRequest IDmust1');result=call.result(timeout=210);after=idle();report['final_status']=after
  report['actual_cache_and_token_counters']=check_cache_and_counts(result,before,after);require(len(result.tokens)==64,'Boundedrequest mustemit64tokens; do notautogrowbudget');report['output_tokens']=len(result.tokens);report['finish_reason']=result.done.reason.name
  hist_a=nested(before,'mtp.completed_cycles_by_proposed_depth');hist_z=nested(after,'mtp.completed_cycles_by_proposed_depth');delta=[z-x for x,z in zip(hist_a,hist_z)];require(len(delta)==16 and delta[4]>=3 and not any(delta[5:]),'Atleast3genuineH4 cycles/nostochasticdeepmode');report['actual_proposed_depth_histogram_delta']=delta
  require(nested(after,'requests.submitted')-nested(before,'requests.submitted')==1 and nested(after,'requests.completed')-nested(before,'requests.completed')==1 and nested(after,'requests.failed')==nested(before,'requests.failed') and nested(after,'requests.cancelled')==nested(before,'requests.cancelled'),'ExactlyonenativecompletednoFailures');require(nested(after,'metrics.metal_failures')==nested(before,'metrics.metal_failures'),'NoMetalFailures')
 except Exception as e:report['errors'].append(type(e).__name__+': '+str(e))
 finally:
  if runtime is not None:
   try:runtime.close()
   except Exception as e:report['errors'].append('close:'+str(e))
  report['native_processes']=len(processes);report['native_returncodes']=[p.poll() for p in processes];report['native_termination_signals']=[{'terminate_called':p.terminate_called,'kill_called':p.kill_called} for p in processes];report['native_process_group_gone']=len(processes)==1 and group_gone(processes[0].pid);report['native_terminal']=len(processes)==1 and processes[0].poll()==0 and not processes[0].kill_called and report['native_process_group_gone']
  if processes and processes[0].poll() is None:report['errors'].append('Nativeprocessstilllive; diagnosticinvalid')
  log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 try:
  require(report['native_terminal'],'CleanoneBackend process terminal required');raw=json.loads(pathlib.Path(c['profile']).read_text());report['profile']=raw;audit_profile(raw);report['profile_file_sha256']=sha(c['profile'])
 except Exception as e:report['errors'].append('profile:'+type(e).__name__+': '+str(e))
 report['completed']=True;report['evidence_valid']=not report['errors'];report['backend_process_destroyed']=report['native_terminal'];save(path,report);print(json.dumps({'report':str(path),'evidence_valid':report['evidence_valid'],'diagnostic_not_performance':True}));return 0 if report['evidence_valid'] else 1
if __name__=='__main__':raise SystemExit(main())
