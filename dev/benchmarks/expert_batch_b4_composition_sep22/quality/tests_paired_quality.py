"""Source and synthetic metadata tests; no model, task-plan, or report reads."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

core_path=Path(__file__).with_name("paired_quality.py")
core_spec=importlib.util.spec_from_file_location("_sibling_paired_original22_quality_for_cpu",core_path)
q=importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name]=q;core_spec.loader.exec_module(q)
from dev.benchmarks.expert_batch_b4_composition_sep22 import service_adapter as service

FROZEN=q.ROOT/"build/batch-twoPass-original22-quality-sep22-root-v5/source/batch_quality.py"
INNER=q.RUNTIME/"source/dev/benchmarks/expert_batch_b4_composition_sep22/adapter.py"
spec=importlib.util.spec_from_file_location("_test_immutable_composition_adapter",INNER)
inner=importlib.util.module_from_spec(spec);spec.loader.exec_module(inner)

def put(value,path,item):
    keys=path.split(".")
    for key in keys[:-1]:value=value.setdefault(key,{})
    value[keys[-1]]=item

def binding():
    return {"schema":"integer-B4-twopass-adapter-binding-v1","runtime_build":str(q.RUNTIME),
        "overlay_manifest_sha256":"1"*64,"compiled_seal_sha256":"2"*64,
        "worker_sha256":"3"*64,"metallib_sha256":"4"*64,
        "policy_source_sha256":q.POLICY,"policy_text":"synthetic current B4-only source policy",
        "policy_shader_sha256":"5"*64,"policy_host_sha256":"6"*64,
        "integer_source_identity_sha256":q.INTEGER_SOURCE,
        "target_numeric_parent_sha256":q.TARGET,"target_base_numeric_parent_sha256":"7"*64,
        "target_execution_base_child_sha256":"8"*64,"target_execution_child_sha256":"9"*64,
        "original22_plan_content_sha256":q.PLAN,"policy_schema":inner.BQSA_SCHEMA,
        "qualified_shapes_or_scores_inherited":False,
        "inner_adapter_path":str(INNER),"inner_adapter_sha256":"a"*64,
        "service_adapter_path":str(q.ROOT/"build/synthetic-service-adapter.py"),"service_adapter_sha256":"b"*64,
        "frozen_original22_common_status_and_coverage_required":True}

def status(role="new",instance=123):
    b=binding();raw="synthetic-raw-forward-parent-routes";integer=role=="new"
    identity={"source":q.SOURCE,"loaded_model_layout_sha256":q.LAYOUT,"engine_instance_id":instance,
        "batch_prefill_twopass_requested":True,"batch_prefill_twopass_schema":inner.BQSA_SCHEMA,
        "batch_prefill_twopass_policy":b["policy_text"],"batch_prefill_twopass_source_sha256":q.POLICY,
        "batch_prefill_twopass_shader_sha256":b["policy_shader_sha256"],
        "batch_prefill_twopass_host_sha256":b["policy_host_sha256"],
        "batch_prefill_twopass_arena_plan_bytes":inner.ARENA_BYTES,
        "batch_prefill_twopass_numerical_parent_routes":raw,
        "batch_prefill_twopass_numerical_identity":hashlib.sha256("\n".join((raw,inner.BQSA_SCHEMA,b["policy_text"],q.POLICY,b["policy_shader_sha256"],b["policy_host_sha256"])).encode()).hexdigest(),
        "batch_prefill_kernel_routes":"legacy batch route"+inner.BQSA_MARKER,
        "kernel_routes":raw+(q.INTEGER_MARKER if integer else ""),
        "target_numerical_derivative_sha256":q.target_derivative(b,role),
        "target_base_numerical_derivative_sha256":q.target_base_derivative(b,role)}
    s={"identity":identity,"maximum_context_tokens":16384,
        "capabilities":{"mtp":True,"batch_mtp":True,"batch_mtp_prefill":True},
        "batch_prefill":{"enabled":True,"maximum_lanes":4,"maximum_real_rows_per_lane":2048},
        "scheduler":{"maximum_prefill_rows":2048,"maximum_batch_prefill_rows_per_lane":2048},
        "mtp":{"enabled":True,"singleton_maximum_draft_tokens":3,"teacher_cache_only_requested":True,"batch_teacher_cache_only_requested":True},
        "memory_governor":{"host_measurement_valid":True,"growth_allowed":True,"denied_reservations":0},
        "transport":{"restarts":0},"status_snapshot":{"steady_seconds":10.0},
        "batch_prefill_twopass_counters":{"scope":inner.BQSA_SCOPE,"constructed_arenas":1,"constructed_arena_bytes":inner.ARENA_BYTES,
            "encoded_QSA_lane_calls":0,"encoded_QSA_lane_layer_calls":0,"completed_native_forwards":0},
        "compact_native_batch_verify":{"schema":inner.INTEGER_SCHEMA,"scope":inner.INTEGER_SCOPE,
            "enabled":integer,"requested":integer,"source_identity_sha256":q.INTEGER_SOURCE,
            "target_numeric_parent_sha256":q.TARGET,"target_base_numeric_parent_sha256":b["target_base_numeric_parent_sha256"],
            "dispatches_per_layer":6,"base_native_dispatches_per_layer":10,"additional_gpu_allocation_bytes":0,
            "full_model_quality_qualified":False}}
    for rows,tg in ((8,2752),(16,3392)):
        s["compact_native_batch_verify"]["r"+str(rows)]={"physical_rows":rows,"planner_threadgroup_bytes":tg,
            **{role+"_"+kind:0 for role in ("plan","gate","down") for kind in ("graph_calls","graph_rows")}}
    return s

def sample(prompt=2048,schema=False,identifier="synthetic",budget=128):
    body={"messages":[{"role":"user","content":"synthetic original body "+identifier}],"max_completion_tokens":budget}
    if schema:body["response_format"]={"type":"json_schema"}
    return {"id":identifier,"prompt_token_count":prompt,"body":body,"prompt_u32le_sha256":"c"*64,
        "request_body_sha256_without_model":q.original.digest(body),"expected":{"kind":"text","value":"synthetic"}}

def wave(prompt,width,role="new",schema=False):
    case=sample(prompt,schema);before=status(role);after=copy.deepcopy(before)
    windows=[min(2048,prompt-start) for start in range(0,prompt,2048)]
    groups={2048:15,2049:16,4096:31,8192:63}[prompt] if not schema else 0
    scalar=width if not schema and prompt!=2049 else 0
    values={"requests.submitted":width,"requests.completed":width,"requests.cancelled":0,"requests.failed":0,
        "metrics.prefill_input_tokens":width*prompt,"scheduler.prefill_rows":width*prompt,
        "scheduler.prefill_batches":len(windows),"batch_prefill.members_dropped_at_control_boundaries":0,
        "mtp.eligible_requests":0 if schema else width,"mtp.teacher_cache_only_priming_calls":scalar,
        "mtp.batch_teacher_cache_only_priming_calls":groups,"mtp.batch_teacher_cache_only_completed_lanes":width*groups,
        "mtp.batch_teacher_cache_only_completed_real_pairs":width*groups*128,
        "scheduler.batch_mtp_priming_batches":groups,"scheduler.batch_mtp_priming_real_rows":width*groups*128,
        "batch_prefill.true_target_hidden_copied_bytes":0 if schema else width*prompt*10240*2,"metrics.metal_failures":0}
    for n in (1,2,3,4):
        values["scheduler.prefill_batches_by_width.b"+str(n)]=len(windows) if n==width else 0
        values["scheduler.batch_mtp_priming_batches_by_width.b"+str(n)]=groups if n==width else 0
    large=[n for n in windows if width*n>=256];calls=48*len(large)
    for key,value in {"gate_up_graph_calls":calls,"down_graph_calls":calls,"gate_up_graph_rows":48*width*sum(large),
        "down_graph_rows":48*width*sum(large),"encoded_hit_dispatches":2*calls,"encoded_miss_dispatches":0,
        "full_inventory_graph_calls":2*calls}.items():values["persisted_experts.graph_counters.large_row_"+key]=value
    for path,value in values.items():put(before,path,0);put(after,path,value)
    if width==4:after["batch_prefill_twopass_counters"].update(encoded_QSA_lane_calls=48,encoded_QSA_lane_layer_calls=48,completed_native_forwards=1)
    put(after,"status_snapshot.steady_seconds",11.0)
    return before,after,case,values

def add_integer(s,rows):
    for role in ("plan","gate","down"):
        s["compact_native_batch_verify"]["r"+str(rows)][role+"_graph_calls"]=48
        s["compact_native_batch_verify"]["r"+str(rows)][role+"_graph_rows"]=48*rows

class Paired(unittest.TestCase):
    def setUp(self):
        self.service_patch=mock.patch.object(q,"load_service",return_value=service);self.service_patch.start();self.addCleanup(self.service_patch.stop)
        self.inner_patch=mock.patch.object(service,"load_inner",return_value=inner);self.inner_patch.start();self.addCleanup(self.inner_patch.stop)

    def test_frozen_original_functions_are_byte_preserved(self):
        source=FROZEN.read_text();current=Path(q.__file__).read_text()
        def functions(text):
            lines=text.splitlines(keepends=True)
            return {n.name:"".join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        a,b=functions(source),functions(current)
        for name in ("frozen_plan","schedule","exact_record","snapshot_errors","response_ids_errors"):
            self.assertEqual(a[name],b[name],name)
        self.assertEqual(a["coverage"].split("\n",1)[1],b["common_coverage"].split("\n",1)[1])
        def lane_body(text):
            node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=="lane")
            lines=text.splitlines(keepends=True);return "".join(lines[node.lineno-1:node.end_lineno])
        self.assertEqual(lane_body(source),lane_body(current))
        for line in ("if old and not new:regressions.append",'mixed_outcomes.append(not exact_record(r,c,model)[0])'):
            self.assertIn(line,current)

    def test_original_common_status_executes_with_actual_wrapped_role(self):
        for role in q.ROLES:
            b=binding();store={"target_numerical_derivative_sha256":q.target_derivative(b,role)}
            with mock.patch.object(q.original,"gate_status",return_value=["original common failure"]) as common:
                errors=q.status_errors(status(role),role,{"synthetic":True},store,"mtp3",b)
                self.assertIn("original common failure",errors)
                common.assert_called_once_with(status(role),{"synthetic":True},store,"mtp3")
            store["target_numerical_derivative_sha256"]=b["target_base_numeric_parent_sha256"]
            with mock.patch.object(q.original,"gate_status",return_value=[]):self.assertTrue(q.status_errors(status(role),role,{},store,"mtp3",b))

    def test_MTP3_prerequisites_and_common_gate_requirement(self):
        with mock.patch.object(q.original,"gate_status",return_value=[]):
            for path in ("capabilities.mtp","capabilities.batch_mtp","capabilities.batch_mtp_prefill","mtp.enabled",
                "mtp.singleton_maximum_draft_tokens","mtp.teacher_cache_only_requested","mtp.batch_teacher_cache_only_requested","batch_prefill.enabled"):
                s=status();put(s,path,None);self.assertTrue(q.status_errors(s,"new",{}, {"target_numerical_derivative_sha256":"9"*64},"mtp3",binding()),path)
            b=binding();b["frozen_original22_common_status_and_coverage_required"]=False
            self.assertTrue(q.status_errors(status(),"new",{}, {"target_numerical_derivative_sha256":"9"*64},"mtp3",b))

    def test_B4_and_B2_actual_common_and_composition_ledgers(self):
        for width in (2,4):
            for role in q.ROLES:
                for prompt in (2048,2049,4096,8192):
                    for schema in (False,True):
                        before,after,case,_=wave(prompt,width,role,schema)
                        record,errors=q.coverage(before,after,case,width,role,"mtp3",binding())
                        self.assertEqual(errors,[],(width,role,prompt,schema))
                        self.assertTrue(record["original22_common_coverage_invoked"])
                        self.assertEqual(record["composition_route_coverage"]["new_B4_encoded_lane_layer_calls_expected"],48 if width==4 else 0)

    def test_no_silent_common_or_fresh_profile_counter_failure(self):
        before,after,case,values=wave(2048,4)
        for path,value in values.items():
            bad=copy.deepcopy(after);put(bad,path,value+1)
            self.assertTrue(q.coverage(before,bad,case,4,"new","mtp3",binding())[1],path)
        bad=copy.deepcopy(after);bad["batch_prefill_twopass_counters"].update(encoded_QSA_lane_calls=0,encoded_QSA_lane_layer_calls=0,completed_native_forwards=0)
        self.assertTrue(q.coverage(before,bad,case,4,"new","mtp3",binding())[1])
        before,after,case,_=wave(2048,2);after["batch_prefill_twopass_counters"].update(encoded_QSA_lane_calls=48,encoded_QSA_lane_layer_calls=48,completed_native_forwards=1)
        self.assertTrue(q.coverage(before,after,case,2,"new","mtp3",binding())[1])

    def test_integer_control_mask_and_width_geometry(self):
        for width,rows,good in ((2,8,True),(2,16,False),(4,16,True),(4,8,True)):
            before,after,case,_=wave(2048,width);add_integer(after,rows)
            self.assertEqual(not q.coverage(before,after,case,width,"new","mtp3",binding())[1],good)
        before,after,case,_=wave(2048,4,"old");add_integer(after,16)
        self.assertTrue(q.coverage(before,after,case,4,"old","mtp3",binding())[1])
        before,after,case,_=wave(2048,4,"new",True)
        self.assertEqual(q.coverage(before,after,case,4,"new","mtp3",binding())[1],[])
        add_integer(after,16);self.assertTrue(q.coverage(before,after,case,4,"new","mtp3",binding())[1])

    def test_aggregate_new_role_cannot_qualify_all_zero_integer_work(self):
        for width in (2,4):
            initial=status();final=copy.deepcopy(initial)
            record,errors=q.aggregate_integer_coverage(initial,final,width,"new",binding())
            self.assertFalse(record["valid"]);self.assertTrue(errors)
            self.assertEqual(record["actual_integer_graph_calls_by_physical_rows"],{"r8":0,"r16":0})
            add_integer(final,8 if width==2 else 16)
            self.assertEqual(q.aggregate_integer_coverage(initial,final,width,"new",binding())[1],[])

    def test_aggregate_B2_R16_rejected_and_B4_R8_only_insufficient(self):
        initial=status();final=copy.deepcopy(initial);add_integer(final,8)
        self.assertTrue(q.aggregate_integer_coverage(initial,final,4,"new",binding())[1])
        add_integer(final,16)
        self.assertTrue(q.aggregate_integer_coverage(initial,final,2,"new",binding())[1])
        self.assertEqual(q.aggregate_integer_coverage(initial,final,4,"new",binding())[1],[])

    def test_short_schema_zero_integer_waves_remain_legitimate(self):
        for width in (2,4):
            before,after,case,_=wave(2048,width,"new",True)
            self.assertEqual(q.coverage(before,after,case,width,"new","mtp3",binding())[1],[])
            before,after,case,_=wave(2048,width,"new",False)
            self.assertEqual(q.coverage(before,after,case,width,"new","mtp3",binding())[1],[])
            initial=status();final=copy.deepcopy(initial);add_integer(final,8 if width==2 else 16)
            record,errors=q.aggregate_integer_coverage(initial,final,width,"new",binding())
            self.assertEqual(errors,[]);self.assertTrue(record["zero_integer_work_per_short_or_schema_wave_allowed"])

    def test_aggregate_is_delta_not_inherited_previous_width_work(self):
        initial=status();add_integer(initial,16);final=copy.deepcopy(initial)
        self.assertTrue(q.aggregate_integer_coverage(initial,final,4,"new",binding())[1])
        add_integer(final,8)
        self.assertEqual(q.aggregate_integer_coverage(initial,final,2,"new",binding())[1],[])

    def test_aggregate_control_zero_consistency_and_decreases(self):
        for width in (2,4):
            initial=status("old");final=copy.deepcopy(initial)
            self.assertEqual(q.aggregate_integer_coverage(initial,final,width,"old",binding())[1],[])
            add_integer(final,8)
            self.assertTrue(q.aggregate_integer_coverage(initial,final,width,"old",binding())[1])
        initial=status();final=copy.deepcopy(initial);add_integer(final,16)
        for path,value in (("compact_native_batch_verify.r16.down_graph_calls",0),
                           ("compact_native_batch_verify.r16.down_graph_rows",768+1),
                           ("compact_native_batch_verify.r16.plan_graph_calls",49)):
            bad=copy.deepcopy(final);put(bad,path,value)
            self.assertTrue(q.aggregate_integer_coverage(initial,bad,4,"new",binding())[1],path)
        initial=copy.deepcopy(final);add_integer(final,8)
        put(final,"compact_native_batch_verify.r16.plan_graph_calls",0)
        self.assertTrue(q.aggregate_integer_coverage(initial,final,4,"new",binding())[1])

    def test_measure_and_saved_audit_execute_report_aggregate_gate(self):
        source=Path(q.__file__).read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        for name in ("measure","audit"):
            calls=[n for n in ast.walk(nodes[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="aggregate_integer_coverage"]
            self.assertEqual(len(calls),1,name)
        self.assertIn('report["aggregate_integer_coverage"]=aggregate;report["errors"].extend(more)',source)
        self.assertIn('aggregate_integer_coverage(initial,report["final_status"],width,role,binding);errors.extend(more)',source)
        self.assertIn('"aggregate_integer_coverage":a["aggregate_integer_coverage"]',source)

    def test_raw_parent_wrapped_parent_and_child_do_not_alias(self):
        b=binding()
        for role in q.ROLES:
            self.assertEqual(q.composition_identity_errors(status(role),role,b),[])
            for path in ("identity.target_numerical_derivative_sha256","identity.target_base_numerical_derivative_sha256",
                "compact_native_batch_verify.target_numeric_parent_sha256","compact_native_batch_verify.target_base_numeric_parent_sha256"):
                s=status(role);put(s,path,"d"*64);self.assertTrue(q.composition_identity_errors(s,role,b),path)
            s=status(role);s["identity"]["kernel_routes"]=q.INTEGER_MARKER+s["identity"]["kernel_routes"]
            self.assertTrue(q.composition_identity_errors(s,role,b))

    def test_cross_role_normalization_keeps_BQSA_and_other_parent_policy(self):
        b=binding();old=status("old",100);new=status("new",200)
        self.assertEqual(q.parent_identity(old,"old",b),q.parent_identity(new,"new",b))
        new["identity"]["unrelated_parent_policy"]="drift"
        self.assertNotEqual(q.parent_identity(old,"old",b),q.parent_identity(new,"new",b))
        new=status("new",200);new["identity"]["batch_prefill_twopass_policy"]="drift"
        self.assertNotEqual(q.parent_identity(old,"old",b),q.parent_identity(new,"new",b))
        before,after,_,_=wave(2048,4);after["identity"]["engine_instance_id"]+=1
        self.assertTrue(q.snapshot_errors(before,before,after))

    def test_coefficient_store_normalizes_only_bound_execution_derivative(self):
        b=binding();old={"manifest_sha256":"e"*64,"target_numerical_derivative_sha256":q.TARGET}
        new={"manifest_sha256":"e"*64,"target_numerical_derivative_sha256":b["target_execution_child_sha256"]}
        self.assertEqual(q.coefficient_store(old,"old",b),q.coefficient_store(new,"new",b))
        new["manifest_sha256"]="f"*64;self.assertNotEqual(q.coefficient_store(old,"old",b),q.coefficient_store(new,"new",b))
        new["target_numerical_derivative_sha256"]=q.TARGET
        with self.assertRaises(ValueError):q.coefficient_store(new,"new",b)

    def test_external_binding_reaches_immutable_binding_validation(self):
        b=binding();validated=mock.Mock();validated.load_binding.return_value=b
        with mock.patch.object(q,"sha",return_value="e"*64),mock.patch.object(Path,"read_text",return_value=json.dumps(b)),mock.patch.object(service,"load_inner",return_value=validated):
            self.assertEqual(q.load_binding(Path("synthetic-binding.json"),"e"*64),b)
            validated.load_binding.assert_called_once_with(Path("synthetic-binding.json").resolve(),"e"*64)
            for field in ("policy_source_sha256","integer_source_identity_sha256","target_numeric_parent_sha256","target_base_numeric_parent_sha256","target_execution_base_child_sha256","target_execution_child_sha256"):
                wrong=copy.deepcopy(b);wrong[field]=None
                with mock.patch.object(Path,"read_text",return_value=json.dumps(wrong)),self.assertRaises(ValueError):q.load_binding(Path("synthetic-binding.json"),"e"*64)

    def test_source_pin_drift_fails_before_adapter_import(self):
        self.service_patch.stop()
        with mock.patch.object(q,"sha",return_value="f"*64),mock.patch.object(q.importlib.util,"spec_from_file_location") as loader:
            with self.assertRaises(ValueError):q.load_service(binding())
            loader.assert_not_called()
        self.service_patch.start()

    def test_authenticate_is_same_worker_two_explicit_integer_roles(self):
        b=binding()
        with mock.patch.object(q,"load_binding",return_value=b):
            a=q.authenticate("old",Path("synthetic-binding.json"),"e"*64);c=q.authenticate("new",Path("synthetic-binding.json"),"e"*64)
            self.assertEqual(a["worker_sha256"],c["worker_sha256"])
            self.assertEqual((a["integer_enabled"],c["integer_enabled"]),(False,True))
            self.assertTrue(a["batch_prefill_twopass_enabled"] and c["batch_prefill_twopass_enabled"])
            self.assertEqual(a["target_numerical_derivative_sha256"],q.TARGET)
            self.assertEqual(c["target_numerical_derivative_sha256"],b["target_execution_child_sha256"])

    def test_mixed_original_lane_bodies_budgets_and_graders_execute(self):
        before,after,_,_=wave(2048,2);cases=[sample(identifier=q.MIXED_IDS[0],budget=64),sample(identifier=q.MIXED_IDS[1],budget=96)]
        runner=SimpleNamespace(cache_disabled=True,initial=before,wait_idle=mock.Mock(side_effect=[before,after]))
        args=SimpleNamespace(width=2,role="new",execution_mode="mtp3",model="synthetic-model")
        sequence=iter(("HTTP-a","HTTP-b"));lock=threading.Lock();sent=[]
        class Client:
            def __init__(self,args):pass
            def send(self,method,path,body):
                with lock:identifier=next(sequence);sent.append(copy.deepcopy(body))
                return {"request_ids":[identifier],"text":"synthetic"}
        with mock.patch.object(q.http,"HTTPClient",Client),mock.patch.object(q.original,"grade_record",return_value=([],None)) as grade,mock.patch.object(q.original,"gate_status",return_value=[]) as common:
            result=q.send_wave(args,runner,cases,{}, {"target_numerical_derivative_sha256":binding()["target_execution_child_sha256"]},binding())
        self.assertEqual(result["errors"],[]);self.assertEqual(result["lane_case_ids"],[c["id"] for c in cases])
        self.assertEqual(sorted(b["max_completion_tokens"] for b in sent),[64,96])
        self.assertEqual(grade.call_count,2);self.assertEqual(common.call_count,2)
        for r,c in zip(result["records"],cases,strict=True):
            self.assertEqual(r["original_case_id"],c["id"]);self.assertEqual({k:v for k,v in r["request_body"].items() if k!="model"},c["body"])

    def test_only_MTP3_is_admitted(self):
        with self.assertRaises(ValueError):q.status_errors(status(),"new",{}, {},"standard",binding())
        with self.assertRaises(ValueError):q.coverage(status(),status(),sample(),4,"new","standard",binding())
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):q.main(["measure","--execution-mode","standard"])

if __name__=="__main__":unittest.main()
