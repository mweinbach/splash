"""Synthetic CPU checks; no model/server/capture/response payload access."""
import copy
import argparse
import ast
import contextlib
import hashlib
import io
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from dev.benchmarks.batch_prefill_twopass_sep22 import batch_quality as q

def put(status,path,value):
    parts=path.split(".")
    for part in parts[:-1]:status=status.setdefault(part,{})
    status[parts[-1]]=value
def sample(prompt=2048,schema=False):
    body={"max_completion_tokens":128}
    if schema:body["response_format"]={"type":"json_schema"}
    return {"prompt_token_count":prompt,"body":body}
def wave(prompt,width,role,mode,schema=False):
    case=sample(prompt,schema);windows=[min(2048,prompt-start) for start in range(0,prompt,2048)]
    eligible=mode=="mtp3" and not schema
    # Independent source examples, not a copy of the schedule implementation.
    group={2048:15,2049:16,4096:31,8192:63}[prompt] if eligible else 0
    tail=width if eligible and prompt!=2049 else 0
    groupedPairs=group*128*width
    before={"identity":{"same":"registered"}}
    after=copy.deepcopy(before)
    values={"requests.submitted":width,"requests.completed":width,"requests.cancelled":0,"requests.failed":0,
        "metrics.prefill_input_tokens":prompt*width,"scheduler.prefill_rows":prompt*width,
        "scheduler.prefill_batches":len(windows),"batch_prefill.members_dropped_at_control_boundaries":0,
        "mtp.eligible_requests":width if eligible else 0,"mtp.teacher_cache_only_priming_calls":tail,
        "mtp.batch_teacher_cache_only_priming_calls":group,"mtp.batch_teacher_cache_only_completed_lanes":group*width,
        "mtp.batch_teacher_cache_only_completed_real_pairs":groupedPairs,
        "scheduler.batch_mtp_priming_batches":group,"scheduler.batch_mtp_priming_real_rows":groupedPairs,
        "batch_prefill.true_target_hidden_copied_bytes":prompt*width*10240*2 if eligible else 0,"metrics.metal_failures":0}
    for w in (1,2,3,4):
        values["scheduler.prefill_batches_by_width.b"+str(w)]=len(windows) if w==width else 0
        values["scheduler.batch_mtp_priming_batches_by_width.b"+str(w)]=group if w==width else 0
    large=[n for n in windows if width*n>=256];calls=48*len(large)
    for key,value in {"gate_up_graph_calls":calls,"down_graph_calls":calls,"gate_up_graph_rows":48*width*sum(large),
        "down_graph_rows":48*width*sum(large),"encoded_hit_dispatches":2*calls,"encoded_miss_dispatches":0,
        "full_inventory_graph_calls":2*calls}.items():values["persisted_experts.graph_counters.large_row_"+key]=value
    if role=="new":
        values.update({"batch_prefill_twopass_counters.encoded_QSA_lane_calls":12*width,
            "batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls":12*width,
            "batch_prefill_twopass_counters.completed_native_forwards":1})
    for path,value in values.items():put(before,path,0);put(after,path,value)
    return before,after,case,values
class Tests(unittest.TestCase):
    def test_real_case_schedule_schema_and_modes(self):
        for width in (2,4):
            for prompt,groups,scalar in ((2048,15,width),(2049,16,0),(4096,31,width),(8192,63,width)):
                value=q.schedule(sample(prompt),width,"mtp3")
                self.assertEqual((value["grouped_calls"],value["scalar_calls"]),(groups,scalar))
                self.assertEqual(value["true_pairs"],width*(prompt-1))
                self.assertEqual(value["grouped_pairs"]+127*scalar,value["true_pairs"])
                for mode,schema in (("standard",False),("mtp3",True)):
                    value=q.schedule(sample(prompt,schema),width,mode)
                    self.assertEqual((value["eligible"],value["grouped_calls"],value["scalar_calls"],value["hidden_bytes"]),(0,0,0,0))
                    self.assertTrue(value["first_window_new"])
    def test_every_real_native_counter_and_types(self):
        for width in (2,4):
            for role in ("old","new"):
                for mode in ("standard","mtp3"):
                    for prompt in (2048,2049,4096,8192):
                        for schema in (False,True):
                            before,after,case,values=wave(prompt,width,role,mode,schema)
                            self.assertEqual(q.coverage(before,after,case,width,role,mode)[1],[])
                            for path,value in values.items():
                                bad=copy.deepcopy(after);put(bad,path,value+1)
                                self.assertTrue(q.coverage(before,bad,case,width,role,mode)[1],path)
                                for wrong in (None,True,-1,1.0,"0",2**64):
                                    bad=copy.deepcopy(after);put(bad,path,wrong)
                                    self.assertTrue(q.coverage(before,bad,case,width,role,mode)[1],(path,wrong))
    def test_fallback_width_and_profile_change_rejected(self):
        before,after,case,_=wave(2048,4,"new","mtp3")
        for path,value in (("scheduler.prefill_batches_by_width.b4",0),("scheduler.prefill_batches_by_width.b1",4),
                           ("batch_prefill_twopass_counters.encoded_QSA_lane_calls",0)):
            bad=copy.deepcopy(after);put(bad,path,value)
            self.assertTrue(q.coverage(before,bad,case,4,"new","mtp3")[1])
        bad=copy.deepcopy(after);bad["identity"]["same"]="changed"
        self.assertTrue(q.coverage(before,bad,case,4,"new","mtp3")[1])
    def test_exact_new_identity_and_all_static_tampers(self):
        status={}
        for path,value in {"identity.source":q.SOURCE,"identity.loaded_model_layout_sha256":q.LAYOUT,
            "batch_prefill.enabled":True,"batch_prefill.maximum_lanes":4,"batch_prefill.maximum_real_rows_per_lane":2048,
            "maximum_context_tokens":16384,"scheduler.maximum_prefill_rows":2048,"scheduler.maximum_batch_prefill_rows_per_lane":2048}.items():put(status,path,value)
        status["identity"].update(kernel_routes="same-original-trunk",batch_prefill_kernel_routes="original"+q.MARKER,
            batch_prefill_twopass_requested=True,batch_prefill_twopass_schema=q.SCHEMA,batch_prefill_twopass_policy=q.TEXT_POLICY,
            batch_prefill_twopass_source_sha256=q.POLICY,batch_prefill_twopass_shader_sha256=q.SHADER,
            batch_prefill_twopass_host_sha256=q.HOST,batch_prefill_twopass_arena_plan_bytes=q.ARENA)
        status["identity"]["batch_prefill_twopass_numerical_identity"]=hashlib.sha256("\n".join(("same-original-trunk",q.SCHEMA,q.TEXT_POLICY,q.POLICY,q.SHADER,q.HOST)).encode()).hexdigest()
        status["batch_prefill_twopass_counters"]={"scope":q.COUNTER_SCOPE,"constructed_arenas":1,
            "constructed_arena_bytes":q.ARENA,"encoded_QSA_lane_calls":48,"encoded_QSA_lane_layer_calls":48,"completed_native_forwards":1}
        self.assertEqual(q.batch_identity(status,"new"),[])
        for section in ("identity","batch_prefill_twopass_counters"):
            for key,value in status[section].items():
                if section=="identity" and key=="kernel_routes":continue
                bad=copy.deepcopy(status);bad[section][key]=None
                self.assertTrue(q.batch_identity(bad,"new"),(section,key))
        self.assertTrue(q.batch_identity(status,"old"))
    def test_original_status_gate_errors_kept(self):
        with mock.patch.object(q.original,"gate_status",return_value=["original-context/wire/cache/standard-MTP-error"]):
            self.assertIn("original-context/wire/cache/standard-MTP-error",q.status_errors({},"new",{},{},"standard"))
        self.assertTrue(q.status_errors(None,"new",{},{},"mtp3"))
    def test_record_body_and_prompt_identity_before_unchanged_grade(self):
        case={"body":{"messages":[{"role":"user","content":"original question"}],"max_completion_tokens":32},
              "prompt_u32le_sha256":"a"*64}
        case["request_body_sha256_without_model"]=q.original.digest(case["body"])
        record={"request_body":{**case["body"],"model":"registered"},"prompt_u32le_sha256":"a"*64}
        with mock.patch.object(q.original,"grade_record",return_value=(["original-task-error"],None)) as grade:
            self.assertEqual(q.exact_record(record,case,"registered"),(["original-task-error"],None));grade.assert_called_once_with(record,case)
        for path,value in (("request_body.max_completion_tokens",8),("prompt_u32le_sha256","b"*64),("request_body.model","wrong")):
            bad=copy.deepcopy(record);put(bad,path,value)
            with self.assertRaises(ValueError):q.exact_record(bad,case,"registered")
    def test_saved_snapshot_identity_restart_and_freshness(self):
        initial={"identity":{"registered":"same"},"transport":{"restarts":0}}
        before=copy.deepcopy(initial);put(before,"status_snapshot.steady_seconds",10.0)
        after=copy.deepcopy(initial);put(after,"status_snapshot.steady_seconds",11.0)
        self.assertEqual(q.snapshot_errors(initial,before,after),[])
        for path,value in (("identity.registered","changed"),("transport.restarts",1),
                           ("status_snapshot.steady_seconds",10.0),("status_snapshot.steady_seconds",True)):
            bad=copy.deepcopy(after);put(bad,path,value)
            self.assertTrue(q.snapshot_errors(initial,before,bad),(path,value))
        put(after,"maximum_context_tokens",8192)
        self.assertTrue(q.snapshot_errors(initial,before,after))
    def test_real_HTTP_response_ids_and_documented_parent_identity_only(self):
        records=[{"request_ids":["public-a"]},{"request_ids":["public-b"]}]
        self.assertEqual(q.response_ids_errors(records,2),[])
        for bad in ([{"request_ids":[]},records[1]],[records[0],records[0]],[{"request_ids":[1]},records[1]]):
            self.assertTrue(q.response_ids_errors(bad,2))
        old={"identity":{"kernel_routes":"same","batch_prefill_kernel_routes":"old","target":"unchanged","engine_instance_id":"old-worker"}}
        new=copy.deepcopy(old);new["identity"].update(batch_prefill_kernel_routes="old"+q.MARKER,batch_prefill_twopass_requested=True)
        new["identity"]["engine_instance_id"]="new-worker"
        self.assertEqual(q.parent_identity(old),q.parent_identity(new))
        unexpected=copy.deepcopy(new);unexpected["identity"]["batch_prefill_twopass_unknown"]="unregistered"
        self.assertNotEqual(q.parent_identity(old),q.parent_identity(unexpected))
        new["identity"]["target"]="different";self.assertNotEqual(q.parent_identity(old),q.parent_identity(new))
    def test_actual_server_local_package_parser_accepts_bundled_tokenizer_command(self):
        from dev.benchmarks.batch_prefill_twopass_sep22 import run_batch_quality as launch
        # Compile the literal parser alone: no server import/device/model/tokenizer.
        tree=ast.parse((q.ROOT/"server/server.py").read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="parse_args")
        namespace={"argparse":argparse,"Path":Path,"ROOT":q.ROOT,"sys":sys,"os":os,
            "_parse_model_id":lambda x:x,"_parse_max_context":lambda x:x,"_parse_max_memory":lambda x:x,
            "image_input":SimpleNamespace(MIN_PIXELS=1,MAX_PIXELS=100),
            "validate_api_key":lambda x:None,"is_finite_number":lambda x:math.isfinite(x)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"literal-server-parse-args","exec"),namespace)
        config={"package":"/cpu-only-local-package","model":"local/fixed","port":8058,"tokenizer":"/original-plan-tokenizer"}
        command=launch.server_command(config,Path("/cpu-only-build"))
        self.assertNotIn("--tokenizer",command)
        with mock.patch("install.launcher.local_bundle_manifest",return_value={}) as manifest, mock.patch.dict(os.environ,{},clear=True):
            args=namespace["parse_args"](command[3:])
            self.assertEqual(args.tokenizer,str(Path(config["package"]).resolve()))
            self.assertEqual(args.binary,"/cpu-only-build/splash-flash")
            manifest.assert_called_once_with(Path(config["package"]).resolve())
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                namespace["parse_args"](command[3:]+["--tokenizer",config["tokenizer"]])
if __name__=="__main__":unittest.main()
