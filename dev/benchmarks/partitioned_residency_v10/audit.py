from pathlib import Path
import json,hashlib,statistics
root=Path('build/release/flash')
checks=[]
def require(ok,reason):
 checks.append({'check':reason,'passed':bool(ok)})
 if not ok: raise RuntimeError(reason)
summaries=[]
for mode in ['one-set','capped-sets']:
 hp=root/f'v10-residency-{mode}-http.json'; ip=root/f'v10-residency-{mode}-idle9-http.json'; tp=root/f'v10-residency-{mode}.jsonl'
 h=json.loads(hp.read_text()); i=json.loads(ip.read_text()); t=[json.loads(s) for s in tp.read_text().splitlines()]
 require(h['valid'] and i['valid'],mode+' both HTTP reports valid')
 geometry=h['runtime_before']['private_full_original_residency']; after=i['runtime_after']['private_full_original_residency']
 require(geometry['active'] and after['active'] and not geometry['failure_reason'],mode+' standing registration active')
 require(geometry['registered_union_base_allocation_count']==1134 and geometry['registered_union_base_allocation_bytes']==144326852608,mode+' exact union counts bytes')
 require(sum(geometry['residency_set_resource_bytes'])==geometry['registered_union_base_allocation_bytes'],mode+' per-set bytes partition union')
 require(geometry['residency_set_count']==(1 if mode=='one-set' else 25),mode+' exact set count')
 require(geometry['residency_set_cap_bytes']==(0 if mode=='one-set' else 8589934592),mode+' expected cap')
 require(all(b<=8589934592 for b in geometry['residency_set_resource_bytes']) if mode=='capped-sets' else True,mode+' group bound')
 pref=[r for r in t if r.get('phase')=='prefill' and r.get('role')=='target_trunk']
 records=h['records']+i['records']; require(len(pref)==len(records)==10,mode+' all requests joined')
 instance=h['runtime_before']['identity']['engine_instance_id']; require(all(x['instance_id']==instance for x in pref),mode+' same engine instance')
 require([x['requests'][0]['request_id'] for x in pref]==list(range(1,11)),mode+' monotonic native request ids')
 require(all(x['actual_rows']==128 and x['dispatch_count']==1709 for x in pref),mode+' target graph same 128 rows1709 dispatches')
 require(all(x['cross_clock_valid'] and x['driver_kernel_timing_valid'] and x['profile_status']=='complete' and not x['encoder_boundaries_altered'] and not x['sampling_barriers'] and not x['dispatch_metadata_truncated'] for x in pref),mode+' complete unmodified command profiles')
 requests=[]
 for record,trace in zip(records,pref):
  r=record['record']; u=r['usage']; budget=record.get('output_budget',1)
  require(record['valid'] and r['done'] and r['http_status']==200 and not r['errors'] and not r['error_frames'] and u['prompt_tokens']==128 and u['prompt_tokens_details']['cached_tokens']==0 and u['completion_tokens']==budget and r['text']==('1' if budget==1 else '1\n'),mode+' request'+str(trace['requests'][0]['request_id'])+' exact budget output and no cache')
  requests.append({'label':record['label'],'idle_seconds':record['idle_seconds'],'output_budget':budget,'native_request_id':trace['requests'][0]['request_id'],'command_sequence':trace['command_sequence'],'http_first_content_ms':r['first_content_ms'],'target_commit_to_gpu_ms':trace['commit_end_to_gpu_start_seconds']*1000,'target_driver_processing_ms':trace['driver_kernel_processing_seconds']*1000,'target_gpu_ms':trace['gpu_seconds']*1000,'target_command_wall_ms':trace['command_wall_seconds']*1000,'response_sha256':r['response_sha256']})
 status=i['runtime_after']; s=status['scheduler']; require(status['ready'] and status['metal']['healthy'] and s['active_requests']==0 and s['queued']==0 and not s['command_in_flight'],mode+' final healthy idle')
 v=h['vm_samples']; start=v[0]['vm']; end=v[-1]['vm']; ps=start['page_size']; require(ps==end['page_size']==16384,mode+' VM page size consistent')
 deltas={k:end['pages'][k]-start['pages'][k] for k in ['Pageins','Pageouts','Swapins','Swapouts','Compressions','Decompressions','File-backed pages','Pages active','Pages inactive']}
 wiringdrop=start['wired_bytes']-end['wired_bytes']; require(wiringdrop>140000000000,mode+' system wired drop over140 GB')
 require(deltas['Swapins']==deltas['Swapouts']==deltas['Pageouts']==deltas['Compressions']==0,mode+' no swap out or in pageout or compression')
 require(deltas['Pageins']*ps<20000000,mode+' system pageins under20 MB')
 idle=[r for r in requests if r['idle_seconds']==9]; immediate=[r for r in requests if r['label'].startswith('immediate')]
 summaries.append({'mode':mode,'engine_instance_id':instance,'source_identity':h['runtime_before']['identity']['source'],'layout_identity':h['runtime_before']['identity']['loaded_model_layout_sha256'],'kernel_routes':h['runtime_before']['identity']['kernel_routes'],'residency_set_count':geometry['residency_set_count'],'residency_set_cap_bytes':geometry['residency_set_cap_bytes'],'registered_union_base_allocation_count':1134,'registered_union_base_allocation_bytes':144326852608,'per_set_resource_bytes':geometry['residency_set_resource_bytes'],'residency_startup_host_api_ms':geometry['residency_setup_ms'],'requests':requests,'idle9_mean_http_first_content_ms':statistics.mean(r['http_first_content_ms'] for r in idle),'idle9_mean_target_wait_ms':statistics.mean(r['target_commit_to_gpu_ms'] for r in idle),'idle9_mean_target_driver_processing_ms':statistics.mean(r['target_driver_processing_ms'] for r in idle),'idle9_mean_target_gpu_ms':statistics.mean(r['target_gpu_ms'] for r in idle),'immediate_mean_target_wait_ms':statistics.mean(r['target_commit_to_gpu_ms'] for r in immediate),'vm':{'scope':'systemwide; temporal evidence without per-resource causal join','initial_wired_bytes':start['wired_bytes'],'final_wired_bytes':end['wired_bytes'],'wired_drop_bytes':wiringdrop,'wired_drop_gib':wiringdrop/2**30,'page_size':ps,'counter_delta_pages':deltas,'counter_delta_bytes':{k:value*ps for k,value in deltas.items() if k in ['Pageins','Pageouts','Swapins','Swapouts','File-backed pages']},'wired_curve':[{'elapsed_seconds':sample['elapsed'],'wired_bytes':sample['vm']['wired_bytes'],'wired_gib':sample['vm']['wired_bytes']/2**30} for sample in v]},'final_status':{'ready':True,'healthy':True,'active_requests':0,'queued':0,'command_in_flight':False},'artifact_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,ip,tp]}})
require(summaries[0]['source_identity']==summaries[1]['source_identity'] and summaries[0]['layout_identity']==summaries[1]['layout_identity'] and summaries[0]['kernel_routes']==summaries[1]['kernel_routes'],'conditions same source layout and routes')
require([r['response_sha256'] for r in summaries[0]['requests']]==[r['response_sha256'] for r in summaries[1]['requests']],'all20 outputs byte equivalent by corresponding request')
r={'schema':'splash-private-partitioned-residency-v10-cpu-comparison','valid':True,'gpu_work_in_this_audit':False,'diagnostic_not_http_performance_score':True,'conditions':summaries,'checks':checks,'conclusions':{'capped_sets_remove_idle_reclaim':False,'capped_sets_remove_idle_submission_stall':False,'groups_preserve_same_resources_math_graph':True,'system_vm_supports_unwiring_with_RAM_resident_payload':True,'system_vm_excludes_weight_sized_swap_reload_in_observed_window':True,'proprietary_collector_exact_policy_proven':False,'metal4_migration_tested':False},'limitations':['Only one-set then capped-set process order was measured; different wall-time/system activity prevents precise small speed claims.','System-wide VM counters do not identify individual resources; magnitude and timing corroborate retained model-state/source evidence.','Existing allocated bytes and standing residency requests do not establish physical pinning.','The9-second idle probe includes eligibility variation; both budgets1 and2 stall independently, with exact same target graph.','No OS sysctl or wired policy mutation was performed.']}
out=root/'v10-residency-grouping-cpu-comparison.json'; out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'valid':True,'checks':len(checks),'output':str(out),'summary':[{k:x[k] for k in ['mode','residency_set_count','idle9_mean_http_first_content_ms','idle9_mean_target_wait_ms','idle9_mean_target_gpu_ms']} for x in summaries]}))
