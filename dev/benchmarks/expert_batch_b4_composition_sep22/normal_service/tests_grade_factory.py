"""Synthetic backup-factory tests; actual task/proof/report files stay unread."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location("_sibling_optional_grade_factory",Path(__file__).with_name("grade_factory.py"))
factory=importlib.util.module_from_spec(spec);sys.modules[spec.name]=factory;spec.loader.exec_module(factory)

def binding():
    return {"worker_sha256":factory.WORKER,"metallib_sha256":factory.LIB,"compiled_seal_sha256":factory.SEAL,
        "integer_source_identity_sha256":factory.SOURCE,"policy_source_sha256":factory.POLICY,
        "target_base_numeric_parent_sha256":factory.RAW_PARENT,"target_numeric_parent_sha256":factory.WRAPPED_PARENT,
        "target_execution_base_child_sha256":factory.RAW_CHILD,"target_execution_child_sha256":factory.WRAPPED_CHILD}

def aggregate(width=4,role="new",r8=0,r16=48):
    deltas={}
    for rows,calls in ((8,r8),(16,r16)):
        for stage in ("plan","gate","down"):
            deltas[f"compact_native_batch_verify.r{rows}.{stage}_graph_calls"]=calls
            deltas[f"compact_native_batch_verify.r{rows}.{stage}_graph_rows"]=calls*rows
    return {"valid":True,"errors":[],"aggregate_integer_coverage":{"valid":True,"width":width,"role":role,"errors":[],
        "actual_initial_to_final_counter_deltas":deltas,"actual_integer_graph_calls_by_physical_rows":{"r8":r8,"r16":r16}}}

def pins():
    return {"schema":factory.PINS_SCHEMA,"task_root_command_sha256":factory.TASK_COMMAND,
        "task_CPU_READY_sha256":factory.TASK_READY,"task_config_sha256":factory.TASK_CONFIG,
        "task_binding_sha256":factory.TASK_BINDING,"files":{str(p):"a"*64 for p in factory.artifact_paths().values()}}

def summary():
    files=pins()["files"];config={"base_environment":{factory.BQSA_FLAG:"1",factory.INTEGER_FLAG:"0","common":"unchanged"}}
    launcher=SimpleNamespace(server_command=lambda c,b,r:["synthetic-server",r],
        role_environment=lambda c,r:{**c["base_environment"],factory.INTEGER_FLAG:"1" if r=="new" else "0"})
    value={"schema":"current-BQSA4-integer-MTP3-paired-original22-summary-v1","completed":True,"valid":True,
        "GPU_executed":True,"execution_modes":["mtp3"],"standard_qualified":False,"native_context":16384,
        "binding_sha256":factory.TASK_BINDING,"native_stage_receipt_sha256":factory.STAGE,
        "same_worker_integer_only_control_delta":True,"original_plan_content_sha256":factory.PLAN,
        "performance_qualified":False,"all_actual_evidence_valid":True,"no_new_task_regressions_at_both_widths":True,
        "old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116,"errors":[],"server_runs":[],"comparisons":[]}
    for role,pid in (("old",100),("new",200)):
        value["server_runs"].append({"role":role,"execution_mode":"mtp3","integer_enabled":role=="new","unloaded":True,
            "pid":pid,"command":["synthetic-server",role],"environment":{"resolved_flash_environment":launcher.role_environment(config,role)},
            "unload_evidence":{"process_group":pid,"process_group_gone":True,"returncode":0,"post_parent_exit_sigkill_required":False},
            "quality_reports":[{"width":w,"path":str(factory.artifact_paths()[f"report.{role}.B{w}"]),
                "sha256":files[str(factory.artifact_paths()[f"report.{role}.B{w}"])]} for w in (4,2)]})
    for width in (4,2):
        path=factory.artifact_paths()[f"comparison.B{width}"]
        value["comparisons"].append({"width":width,"execution_mode":"mtp3","valid":True,"no_new_task_regressions":True,
            "path":str(path),"sha256":files[str(path)]})
    return value,config,launcher,files

class BackupFactory(unittest.TestCase):
    def test_explicit_root_switch_precedes_all_actual_reads_or_hashes(self):
        argv=["--summary","synthetic","--summary-sha256","a"*64,"--actual-artifact-pins","synthetic-pins",
            "--actual-artifact-pins-sha256","b"*64,"--factory-source-sha256","c"*64,"--output","synthetic-output"]
        with mock.patch.object(factory,"sha") as digest,mock.patch.object(Path,"read_text") as read:
            with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):factory.main(argv)
            digest.assert_not_called();read.assert_not_called()

    def test_exact_current_source_and_raw_wrapped_identities(self):
        value=binding();factory.validate_binding(value)
        for key in value:
            wrong=copy.deepcopy(value);wrong[key]="f"*64
            with self.assertRaises(ValueError):factory.validate_binding(wrong)
        self.assertNotEqual(factory.RAW_PARENT,factory.WRAPPED_PARENT)
        self.assertNotEqual(factory.RAW_CHILD,factory.WRAPPED_CHILD)

    def test_external_actual_manifest_requires_all_seven_exact_files(self):
        value=pins();self.assertEqual(len(factory.validate_actual_pins(value,factory.SUMMARY,"a"*64)),7)
        for key in ("task_root_command_sha256","task_CPU_READY_sha256","task_config_sha256","task_binding_sha256"):
            wrong=copy.deepcopy(value);wrong[key]="f"*64
            with self.assertRaises(ValueError):factory.validate_actual_pins(wrong,factory.SUMMARY,"a"*64)
        wrong=copy.deepcopy(value);wrong["files"].pop(str(factory.artifact_paths()["report.new.B2"]))
        with self.assertRaises(ValueError):factory.validate_actual_pins(wrong,factory.SUMMARY,"a"*64)
        with self.assertRaises(ValueError):factory.validate_actual_pins(value,factory.SUMMARY,"b"*64)

    def test_zero_new_role_and_inherited_exposure_never_qualify(self):
        for width in (2,4):
            with self.assertRaises(ValueError):factory.validate_aggregate(aggregate(width,r8=0,r16=0),width,"new")
        self.assertEqual(factory.validate_aggregate(aggregate(2,r8=48,r16=0),2,"new"),{8:48,16:0})
        self.assertEqual(factory.validate_aggregate(aggregate(4,r8=48,r16=96),4,"new"),{8:48,16:96})
        with self.assertRaises(ValueError):factory.validate_aggregate(aggregate(4,r8=48,r16=0),4,"new")
        with self.assertRaises(ValueError):factory.validate_aggregate(aggregate(2,r8=48,r16=48),2,"new")
        self.assertEqual(factory.validate_aggregate(aggregate(4,"old",0,0),4,"old"),{8:0,16:0})
        with self.assertRaises(ValueError):factory.validate_aggregate(aggregate(4,"old",0,48),4,"old")

    def test_exposure_geometry_stage_consistency_and_missing_counter_rejected(self):
        value=aggregate()
        for key,wrong_value in (("compact_native_batch_verify.r16.plan_graph_calls",49),
            ("compact_native_batch_verify.r16.down_graph_calls",0),("compact_native_batch_verify.r16.down_graph_rows",769)):
            wrong=copy.deepcopy(value);wrong["aggregate_integer_coverage"]["actual_initial_to_final_counter_deltas"][key]=wrong_value
            with self.assertRaises(ValueError):factory.validate_aggregate(wrong,4,"new")
        wrong=copy.deepcopy(value);wrong["aggregate_integer_coverage"]["actual_initial_to_final_counter_deltas"].pop("compact_native_batch_verify.r16.plan_graph_calls")
        with self.assertRaises(ValueError):factory.validate_aggregate(wrong,4,"new")

    def test_completed_bothwidth_summary_and_only_integer_flag_delta(self):
        value,config,launcher,files=summary();self.assertEqual(len(factory.validate_summary(value,config,launcher,files)),2)
        relative=copy.deepcopy(value)
        for run in relative["server_runs"]:
            for row in run["quality_reports"]:row["path"]=str(Path(row["path"]).relative_to(factory.ROOT))
        for row in relative["comparisons"]:row["path"]=str(Path(row["path"]).relative_to(factory.ROOT))
        self.assertEqual(len(factory.validate_summary(relative,config,launcher,files)),2)
        for key,bad in (("valid",False),("completed",False),("execution_modes",["standard"]),
            ("native_stage_receipt_sha256","23f15"),("binding_sha256","f"*64),("no_new_task_regressions_at_both_widths",False)):
            wrong=copy.deepcopy(value);wrong[key]=bad
            with self.assertRaises(ValueError):factory.validate_summary(wrong,config,launcher,files)
        wrong=copy.deepcopy(value);wrong["comparisons"]=wrong["comparisons"][:1]
        with self.assertRaises(ValueError):factory.validate_summary(wrong,config,launcher,files)
        wrong=copy.deepcopy(value);wrong["server_runs"][1]["environment"]["resolved_flash_environment"]["unregistered-extra"]="1"
        with self.assertRaises(ValueError):factory.validate_summary(wrong,config,launcher,files)
        wrong=copy.deepcopy(value);wrong["server_runs"][1]["environment"]["resolved_flash_environment"][factory.BQSA_FLAG]="0"
        with self.assertRaises(ValueError):factory.validate_summary(wrong,config,launcher,files)

    def test_clean_both_worker_unload_required_and_bound_to_process_group(self):
        value,config,launcher,files=summary()
        for role_index in (0,1):
            for key,bad in (("process_group_gone",False),("returncode",-9),("post_parent_exit_sigkill_required",True),("process_group",999)):
                wrong=copy.deepcopy(value);wrong["server_runs"][role_index]["unload_evidence"][key]=bad
                with self.assertRaises(ValueError):factory.validate_summary(wrong,config,launcher,files)

    def test_original_and_mixed_outcomes_reconstruct_no_regression(self):
        quality=SimpleNamespace(MIXED_IDS=("mixed-a","mixed-b"));plan={"cases":[{"id":"original-a"}]}
        left={"outcomes":{"original-a":[True,False]},"texts":{"original-a":["same","baseline"]},
            "mixed_outcomes":[True,True],"mixed_texts":["same","same"]}
        right=copy.deepcopy(left)
        self.assertEqual(factory.paired_outcomes(quality,plan,left,right),([],[]))
        right["outcomes"]["original-a"][0]=False;right["mixed_outcomes"][1]=False
        regressions,_=factory.paired_outcomes(quality,plan,left,right)
        self.assertEqual(regressions,[{"id":"original-a","lane":0},{"id":"mixed-b","lane":1,"scope":"mixed_original_task_wave"}])

if __name__=="__main__":unittest.main()
