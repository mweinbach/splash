"""Synthetic normal-service CPU tests; no fixture/model/payload/runtime reads."""
from __future__ import annotations

import builtins
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import urllib.request

HERE = Path(__file__).parent


def guarded_import(filename, name):
    """Load only named source and refuse import-time data I/O or execution."""
    path = HERE / filename
    source = path.read_text()
    module = ModuleType(name)
    module.__file__ = str(path)
    real_import = builtins.__import__
    blocked = {"mlx", "torch", "transformers", "safetensors", "tokenizers", "Metal", "metal"}

    def source_import(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] in blocked:
            raise AssertionError("device/model dependency imported: " + module_name)
        return real_import(module_name, *args, **kwargs)

    def refused(*args, **kwargs):
        raise AssertionError("source import attempted runtime I/O or execution")

    with mock.patch.object(builtins, "__import__", side_effect=source_import), \
            mock.patch.object(builtins, "open", side_effect=refused), \
            mock.patch.object(Path, "read_text", side_effect=refused), \
            mock.patch.object(Path, "read_bytes", side_effect=refused), \
            mock.patch.object(subprocess, "Popen", side_effect=refused), \
            mock.patch.object(subprocess, "run", side_effect=refused), \
            mock.patch.object(os, "system", side_effect=refused), \
            mock.patch.object(socket, "socket", side_effect=refused), \
            mock.patch.object(time, "sleep", side_effect=refused), \
            mock.patch.object(urllib.request, "urlopen", side_effect=refused), \
            mock.patch.dict(sys.modules, {name: module}):
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


common = guarded_import("launch_common.py", "_synthetic_normal_launch_common")
with mock.patch.dict(sys.modules, {"launch_common": common}):
    audit = guarded_import("audit_profile.py", "_synthetic_normal_profile_audit")


def flags64():
    fixed = {
        common.TRACE_FLAG: "/synthetic/baseline.trace.jsonl",
        "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS": "4",
        "SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21": "0",
        "SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS": "0",
        "SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22": "1",
    }
    fixed.update({"SYNTHETIC_RESTORED_FLAG_" + str(index): str(index % 2) for index in range(59)})
    return fixed


def plan():
    baseline = [sys.executable, "-B", "/synthetic/unchanged-original-driver.py",
        "--binary", "/synthetic/baseline/splash-flash", "--output", "/synthetic/baseline.report.json",
        "--batches", "4,2", "--mtp", "3", "--contexts", "2048", "--output-tokens", "256",
        "--warmup", "1", "--trials", "3", "--max-context", "16384", "--lane-variation", "shared",
        "--workloads", "coding", "--temperature", "0", "--cache", "0"]
    for key, value in flags64().items():
        baseline += ["--env", key + "=" + value]
    result = {
        "schema": common.PLAN_SCHEMA, "roles": ["old", "new"],
        "allowed_modes": ["3"], "allowed_native_widths": [4, 2],
        "runtime": "/synthetic/current-composed-runtime",
        "native_stage_receipt": "/synthetic/current-native-stage.json",
        "native_stage_receipt_sha256": common.STAGE_SHA,
        "actual_grade_receipt_sha256": common.GRADE_SHA,
        "frozen_restored_baseline": {"modes": {"3": {"argv": baseline}}}, "profiles": {},
    }
    for role, port in (("old", 8058), ("new", 8059)):
        selected = {"report": f"/synthetic/{role}.report.json", "trace": f"/synthetic/{role}.trace.jsonl",
                    "port": port, "native_audit": f"/synthetic/{role}.native-audit.json",
                    "profile_audit": f"/synthetic/{role}.profile-audit.json"}
        argv = list(baseline)
        argv[argv.index("--binary") + 1] = result["runtime"] + "/splash-flash"
        argv[argv.index("--output") + 1] = selected["report"]
        trace = argv.index(common.TRACE_FLAG + "=" + flags64()[common.TRACE_FLAG])
        argv[trace] = common.TRACE_FLAG + "=" + selected["trace"]
        argv += ["--env", common.BQSA_FLAG + "=1", "--env",
                 common.INTEGER_FLAG + "=" + ("1" if role == "new" else "0"), "--port", str(port)]
        selected["argv"] = argv
        selected["explicit_flags"] = {**flags64(), common.TRACE_FLAG: selected["trace"],
            common.BQSA_FLAG: "1", common.INTEGER_FLAG: "1" if role == "new" else "0"}
        result["profiles"][role] = selected
    return result


def stage():
    return {"schema": common.STAGE_SCHEMA, "pass": True, "Root_GPU_executed": True,
        "candidate_worker_sha256": common.WORKER, "candidate_source_identity_sha256": common.SOURCE,
        "BQSA4_policy_sha256": common.POLICY, "metallib_sha256": common.LIB,
        "compiled_worker_seal_sha256": common.SEAL, "proof_capacity": 4096,
        "all_four_pairs_complete": True, "selected7_fullphysical_qualified": True,
        "all18_logical_output_max16_MoE_rejections_owners_qualified": True,
        "all18_fullphysical_qualified": False, "full_numeric_flags_count": 47,
        "original22_or_service16K_qualified": False, "trained_head_qualified": False,
        "worker_cancel_deadline_qualified": False, "performance_qualified": False,
        "public_promotion_qualified": False,
        "parts": [{"ordinal": index, "pass": True, "backend_destroyed": True} for index in range(1, 5)]}


def grade():
    result = {"schema": common.GRADE_SCHEMA, "pass": True, "execution_mode": "mtp3",
        "qualified_native_widths": [4, 2], "maximum_context_tokens": 16384,
        "worker_sha256": common.WORKER, "metallib_sha256": common.LIB, "source_identity_sha256": common.SOURCE,
        "BQSA4_policy_sha256": common.POLICY, "compiled_worker_seal_sha256": common.SEAL,
        "native_stage_receipt_sha256": common.STAGE_SHA, "original_plan_content_sha256": common.PLAN_CONTENT,
        "performance_qualified": False, "standard_qualified": False, "old_extra_EVERYROW_equivalence": False,
        "both_roles_gracefully_unloaded": True,
        "binding_sha256": "dff527c9c8ea60dbaa64b759ccf2328feda0e0f9aba78d4c189ba03d75d13d82",
        "comparisons": []}
    for width, digest in ((4, "2a9d09a924fa4a4895cc436ea683232e042dd0eaa8cceb140b75a1df23ef215c"),
                          (2, "d945d6bbe0a2b87eaf5262df44818296c488b352ff93c8673f2632efd6c04920")):
        result["comparisons"].append({"width": width, "execution_mode": "mtp3", "valid": True,
            "no_new_task_regressions": True, "sha256": digest})
    return result


def put(value, path, item):
    keys = path.split(".")
    for key in keys[:-1]: value = value.setdefault(key, {})
    value[keys[-1]] = item


def profile_plan():
    return {**plan(), "expected_native_batch_prefill_ctor_bytes": 743964672}


def profile_helper():
    binding = {"synthetic_binding": True}
    adapter = SimpleNamespace(status_errors=mock.Mock(return_value=[]))

    def identity_errors(status, role, supplied):
        assert supplied is binding
        identity = status.get("identity", {})
        expected = common.TARGET_CHILD if role == "new" else common.TARGET_PARENT
        expected_base = common.TARGET_BASE_CHILD if role == "new" else common.TARGET_BASE_PARENT
        if identity.get("target_numerical_derivative_sha256") != expected or \
                identity.get("target_base_numerical_derivative_sha256") != expected_base:
            return ["synthetic pinned quality rejected raw/wrapped target identity"]
        return []

    quality = SimpleNamespace(composition_identity_errors=mock.Mock(side_effect=identity_errors))
    return binding, adapter, quality


def status(role="new"):
    return {"identity": {"target_gathered_mpp_max_physical_rows": 4,
        "mtp_batch_teacher_priming_route": audit.TEACHER_ROUTE,
        "batch_prefill_kernel_routes": "synthetic-existing" + audit.ARENA_ROUTE + audit.BQSA_ROUTE,
        "target_numerical_derivative_sha256": common.TARGET_CHILD if role == "new" else common.TARGET_PARENT,
        "target_base_numerical_derivative_sha256": common.TARGET_BASE_CHILD if role == "new" else common.TARGET_BASE_PARENT},
        "maximum_context_tokens": 16384,
        "memory_audit": {"valid": True, "batch_prefill_workspace_bytes": profile_plan()["expected_native_batch_prefill_ctor_bytes"]},
        "native_lifecycle_timestamps_sep22": {"enabled": True},
        "memory_governor": {"host_available_bytes": 16 << 30, "host_reserve_bytes": 1 << 30}}


def wave(width=4, role="new", selected_calls=48, B4_to_R8=False):
    expected = {f"scheduler.prefill_batches_by_width.b{n}": int(n == width) for n in range(1, 5)}
    expected.update({"batch_prefill_twopass_counters.encoded_QSA_lane_calls": 48 if width == 4 else 0,
        "batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls": 48 if width == 4 else 0,
        "batch_prefill_twopass_counters.completed_native_forwards": int(width == 4),
        "mtp.batch_teacher_cache_only_priming_calls": 15,
        "mtp.batch_teacher_cache_only_completed_lanes": 15 * width,
        "mtp.batch_teacher_cache_only_completed_real_pairs": 1920 * width,
        "mtp.teacher_cache_only_priming_calls": width,
        "batch_prefill.true_target_hidden_copied_bytes": width * 2048 * 10240 * 2})
    expected.update({f"scheduler.batch_mtp_priming_batches_by_width.b{n}": 15 if n == width else 0 for n in range(1, 5)})
    for rows in (8, 16):
        count = selected_calls if role == "new" and rows == (16 if width == 4 else 8) else 0
        if role == "new" and width == 4 and rows == 8 and B4_to_R8: count = 48
        for stage_name in ("plan", "gate", "down"):
            expected[f"compact_native_batch_verify.r{rows}.{stage_name}_graph_calls"] = count
            expected[f"compact_native_batch_verify.r{rows}.{stage_name}_graph_rows"] = count * rows
    for native_width in (2, 4):
        count = expected[f"compact_native_batch_verify.r{native_width * 4}.plan_graph_calls"]
        expected[f"scheduler.decode_batches_by_width.b{native_width}"] = count // 48
    before = status(role); after = copy.deepcopy(before)
    for path, value in expected.items(): put(before, path, 0); put(after, path, value)
    return {"http_width": width, "mtp_setting": "3", "prompt_tokens": 2048, "output_budget_tokens": 256,
        "trial": 0, "warmup": False, "status_before": before, "status_after": after,
        "native_counter_delta": expected}


class CanonicalArgv(unittest.TestCase):
    def test_both_roles_match_independent_64_plus2_flag_argv(self):
        value = plan()
        before = copy.deepcopy(value)
        for role in ("old", "new"):
            with self.subTest(role=role):
                actual = common.canonical(value, role)
                self.assertEqual(actual, value["profiles"][role]["argv"])
                self.assertEqual(actual.count("--env"), 66)
                self.assertEqual(actual[:3], value["frozen_restored_baseline"]["modes"]["3"]["argv"][:3])
        self.assertEqual(value, before)

    def test_role_numerics_differ_only_integer_flag_and_trace_destination(self):
        value = plan()
        old, new = (value["profiles"][role]["explicit_flags"] for role in ("old", "new"))
        numeric = lambda flags: {key: item for key, item in flags.items() if key != common.TRACE_FLAG}
        changed = {key for key in numeric(old) if numeric(old)[key] != numeric(new)[key]}
        self.assertEqual(changed, {common.INTEGER_FLAG})
        self.assertEqual(old[common.INTEGER_FLAG], "0")
        self.assertEqual(new[common.INTEGER_FLAG], "1")
        self.assertEqual(old[common.BQSA_FLAG], "1")
        self.assertEqual(new[common.BQSA_FLAG], "1")

    def test_roles_change_only_binary_output_trace_integer_and_registered_port(self):
        value = plan()
        old, new = (common.canonical(value, role) for role in ("old", "new"))
        changed = [index for index, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        expected = [old.index("--output") + 1, old.index(common.TRACE_FLAG + "=" + value["profiles"]["old"]["trace"]),
                    old.index(common.INTEGER_FLAG + "=0"), old.index("--port") + 1]
        self.assertEqual(changed, sorted(expected))
        self.assertEqual(old[old.index("--binary") + 1], new[new.index("--binary") + 1])

    def test_only_mtp3_width4_then2(self):
        for modes, widths in ((["0"], [4, 2]), (["3", "0"], [4, 2]), (["3"], [2, 4]),
                              (["3"], [4]), (["3"], [2]), (["3"], [4, 3])):
            with self.subTest(modes=modes, widths=widths):
                value = plan(); value["allowed_modes"] = modes; value["allowed_native_widths"] = widths
                with self.assertRaises(ValueError): common.canonical(value, "new")

    def test_unknown_and_standard_roles_refused(self):
        for role in ("std", "standard", "unknown", "OLD", None, 0):
            with self.subTest(role=role):
                with self.assertRaises(ValueError): common.canonical(plan(), role)

    def test_selected_argv_or_explicit_flags_cannot_drift(self):
        for kind in ("argv", "explicit_flags"):
            with self.subTest(kind=kind):
                value = plan()
                if kind == "argv": value["profiles"]["new"][kind].append("--dry-run")
                else: value["profiles"]["new"][kind][common.BQSA_FLAG] = "0"
                with self.assertRaises(ValueError): common.canonical(value, "new")

    def test_unregistered_default_flag_refused(self):
        value = plan()
        del value["profiles"]["new"]["explicit_flags"]["SYNTHETIC_RESTORED_FLAG_58"]
        with self.assertRaisesRegex(ValueError, "66 flag set"): common.canonical(value, "new")

    def test_frozen_workload_defaults_cannot_replace_registered_values(self):
        for flag, invalid in (("--contexts", "1024"), ("--output-tokens", "128"), ("--warmup", "0"),
                ("--trials", "1"), ("--max-context", "4096"), ("--lane-variation", "distinct"),
                ("--workloads", "prose"), ("--batches", "2,4"), ("--mtp", "0")):
            with self.subTest(flag=flag):
                value = plan()
                baseline = value["frozen_restored_baseline"]["modes"]["3"]["argv"]
                baseline[baseline.index(flag) + 1] = invalid
                for selected in value["profiles"].values(): selected["argv"][selected["argv"].index(flag) + 1] = invalid
                with self.assertRaises(ValueError): common.canonical(value, "old")

    def test_frozen_cap_bulk_QA_and_footer_stamps_explicit(self):
        for key, invalid in (("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS", "8"),
                ("SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21", "1"),
                ("SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS", "1"),
                ("SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22", "0")):
            with self.subTest(key=key):
                value = plan(); original = flags64()[key]
                baseline = value["frozen_restored_baseline"]["modes"]["3"]["argv"]
                baseline[baseline.index(key + "=" + original)] = key + "=" + invalid
                for selected in value["profiles"].values():
                    selected["argv"][selected["argv"].index(key + "=" + original)] = key + "=" + invalid
                    selected["explicit_flags"][key] = invalid
                with self.assertRaises(ValueError): common.canonical(value, "new")


class ActualAdmission(unittest.TestCase):
    def test_canonical_actual_stage_and_bothwidth_grade(self):
        self.assertIsNone(common.validate_stage(plan(), stage()))
        self.assertIsNone(common.validate_grade(plan(), grade()))

    def test_exact_current_source_worker_library_seal_and_37b_wrapped_child(self):
        self.assertEqual(common.WORKER, "ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68")
        self.assertEqual(common.SOURCE, "edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94")
        self.assertEqual(common.TARGET_CHILD, "37b89da17689fd1b7f3b61ed3db0c14f743270e16c4dbe7f2803ea918878dad0")
        self.assertNotEqual(common.TARGET_CHILD, common.TARGET_PARENT)
        self.assertNotEqual(common.TARGET_BASE_CHILD, common.TARGET_BASE_PARENT)
        self.assertTrue(common.STAGE_SHA.startswith("7f82"))

    def test_old23f15_and_unknown_state_receipts_refused(self):
        for pin in ("23f15" + "0" * 59, "f" * 64, None):
            with self.subTest(pin=pin):
                value = plan(); value["native_stage_receipt_sha256"] = pin
                with self.assertRaises(ValueError): common.validate_stage(value, stage())

    def test_every_current_stage_field_is_required_and_strict(self):
        for key, valid in stage().items():
            if key == "parts": continue
            with self.subTest(key=key):
                value = stage()
                value[key] = not valid if type(valid) is bool else valid + 1 if type(valid) is int else "unknown"
                with self.assertRaises(ValueError): common.validate_stage(plan(), value)

    def test_all4_current_pairs_and_true_teardown_required(self):
        for count in (0, 3, 5):
            with self.subTest(count=count):
                value = stage(); value["parts"] = [{"pass": True, "backend_destroyed": True}] * count
                with self.assertRaises(ValueError): common.validate_stage(plan(), value)
        for index in range(4):
            for key in ("pass", "backend_destroyed"):
                for invalid in (False, 1, None):
                    with self.subTest(index=index, key=key, invalid=invalid):
                        value = stage(); value["parts"][index][key] = invalid
                        with self.assertRaises(ValueError): common.validate_stage(plan(), value)

    def test_every_top_level_actual_grade_field_required(self):
        for key in grade():
            with self.subTest(key=key):
                value = grade(); del value[key]
                with self.assertRaises(ValueError): common.validate_grade(plan(), value)

    def test_grade_requires_both_actual_root_comparisons_with_no_regressions(self):
        for index in range(2):
            for key in ("valid", "no_new_task_regressions"):
                for invalid in (False, 1, None):
                    with self.subTest(index=index, key=key, invalid=invalid):
                        value = grade(); value["comparisons"][index][key] = invalid
                        with self.assertRaises(ValueError): common.validate_grade(plan(), value)

    def test_both_comparison_widths_modes_and_actual_external_digests_required(self):
        for index, width in enumerate((4, 2)):
            for key, invalids in (("width", (width // 2, str(width), True)),
                    ("execution_mode", ("std", "mtp2", None)),
                    ("sha256", ("", "f" * 64, None))):
                for invalid in invalids:
                    with self.subTest(index=index, key=key, invalid=invalid):
                        value = grade(); value["comparisons"][index][key] = invalid
                        with self.assertRaises(ValueError): common.validate_grade(plan(), value)
        for comparisons in ([], grade()["comparisons"][:1], list(reversed(grade()["comparisons"]))):
            with self.subTest(comparisons=comparisons):
                value = grade(); value["comparisons"] = comparisons
                with self.assertRaises(ValueError): common.validate_grade(plan(), value)

    def test_canonical_actual_grade4133_external_pin_only(self):
        self.assertEqual(common.GRADE_SHA, "413384e51eeae597e5111f350c6f59815d56d93d23f70fce1354c04521449683")
        for pin in ("f" * 64, None):
            with self.subTest(pin=pin):
                value = plan(); value["actual_grade_receipt_sha256"] = pin
                with self.assertRaises(ValueError): common.validate_grade(value, grade())

    def test_unknown_old_or_widened_grade_schema_and_STD_refused(self):
        for key, invalid in (("schema", "old-or-unknown-grade-v1"), ("execution_mode", "std"),
                ("qualified_native_widths", [2, 4]), ("qualified_native_widths", [4]),
                ("standard_qualified", True), ("performance_qualified", True),
                ("old_extra_EVERYROW_equivalence", True), ("native_stage_receipt_sha256", "23f15" + "0" * 59),
                ("source_identity_sha256", "f" * 64), ("worker_sha256", "f" * 64)):
            with self.subTest(key=key, invalid=invalid):
                value = grade(); value[key] = invalid
                with self.assertRaises(ValueError): common.validate_grade(plan(), value)


class SourceImport(unittest.TestCase):
    def test_launch_common_import_has_no_runtime_actions(self):
        self.assertEqual(guarded_import("launch_common.py", "_synthetic_normal_import_check").PLAN_SCHEMA,
                         "current-BQSA4-integer-MTP3-native-normal-plan-v1")

    def test_root_launcher_import_no_main_or_model_load(self):
        with mock.patch.dict(sys.modules, {"launch_common": common}):
            value = guarded_import("run_root_service.py", "_synthetic_normal_root_import_check")
        self.assertTrue(callable(value.main))

    def test_profile_auditor_import_no_report_or_device_reads(self):
        with mock.patch.dict(sys.modules, {"launch_common": common}):
            value = guarded_import("audit_profile.py", "_synthetic_normal_profile_import_check")
        self.assertTrue(callable(value.wave_errors))

    def test_preparer_and_proof_binder_imports_have_no_data_or_runtime_actions(self):
        for filename in ("prepare_plan.py", "bind_root_proofs.py"):
            with self.subTest(filename=filename), mock.patch.dict(sys.modules, {"launch_common": common}):
                value = guarded_import(filename, "_synthetic_normal_" + filename.removesuffix(".py"))
            self.assertTrue(callable(value.main))


class ExternalPins(unittest.TestCase):
    def setUp(self):
        self.value = plan()
        self.witness_sha = "a" * 64
        self.binding_sha = "b" * 64
        self.witness_path = str(common.HERE / "launch-witness.json")
        self.plan_path = str(common.HERE / "service-plan.json")
        self.binding_path = str(common.HERE / "Root-actual-proof-binding.json")
        self.grade_path = "/synthetic/current-task-grade.json"
        self.code_path = "/synthetic/pinned-normal-helper.py"
        self.witness = {"schema": "current-BQSA4-integer-normal-source-witness-v1",
            "files": [{"path": self.code_path, "sha256": "c" * 64},
                      {"path": self.plan_path, "sha256": "d" * 64}]}
        self.binding = {"schema": "current-BQSA4-integer-Root-native-and-task-normal-binding-v1", "pass": True,
            "launch_witness_sha256": self.witness_sha, "native_stage_receipt_sha256": common.STAGE_SHA,
            "grade_receipt": self.grade_path, "grade_receipt_sha256": common.GRADE_SHA}
        self.documents = {self.witness_path: self.witness, self.plan_path: self.value,
            self.binding_path: self.binding, self.value["native_stage_receipt"]: stage(), self.grade_path: grade()}
        self.digests = {self.witness_path: self.witness_sha, self.binding_path: self.binding_sha,
            self.code_path: "c" * 64, self.plan_path: "d" * 64,
            self.value["native_stage_receipt"]: common.STAGE_SHA, self.grade_path: common.GRADE_SHA}
        self.reads = []; self.hashes = []

        def fake_read(path, *args, **kwargs):
            name = str(path); self.reads.append(name)
            self.assertIn(name, self.documents, "unexpected fixture/model/report content read")
            return json.dumps(self.documents[name])

        def fake_sha(path):
            name = str(path); self.hashes.append(name)
            self.assertIn(name, self.digests, "unexpected fixture/model/runtime hash read")
            return self.digests[name]

        for patch in (mock.patch.object(Path, "read_text", autospec=True, side_effect=fake_read),
                mock.patch.object(Path, "read_bytes", side_effect=AssertionError("real data read forbidden")),
                mock.patch.object(common, "sha", side_effect=fake_sha),
                mock.patch.object(common, "module", side_effect=AssertionError("external source import forbidden")),
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError("service launch forbidden")),
                mock.patch.object(subprocess, "run", side_effect=AssertionError("process launch forbidden")),
                mock.patch.object(socket, "socket", side_effect=AssertionError("service read forbidden"))):
            patch.start(); self.addCleanup(patch.stop)

    def test_external_source_witness_loads_only_synthetic_pinned_metadata(self):
        self.assertEqual(common.load_pinned(self.witness_sha), self.value)
        self.assertEqual(self.reads, [self.witness_path, self.plan_path])
        self.assertEqual(self.hashes, [self.witness_path, self.code_path, self.plan_path])

    def test_missing_unknown_or_malformed_external_witness_pin_refused(self):
        for pin in (None, "", "f" * 64, "A" * 64, "0" * 63):
            with self.subTest(pin=pin):
                with self.assertRaises(ValueError): common.load_pinned(pin)
        self.assertEqual(self.reads, [])

    def test_frozen_source_pin_drift_refused(self):
        self.digests[self.code_path] = "f" * 64
        with self.assertRaisesRegex(ValueError, "source/runtime/helper drift"):
            common.load_pinned(self.witness_sha)
        self.assertEqual(self.reads, [self.witness_path])

    def test_unknown_source_witness_schema_refused(self):
        self.witness["schema"] = "unknown-witness-v1"
        with self.assertRaises(ValueError): common.load_pinned(self.witness_sha)

    def test_unknown_or_STD_plan_under_pin_is_refused(self):
        for key, invalid in (("schema", "unknown-plan-v1"), ("roles", ["new", "old"]),
                              ("allowed_modes", ["0"]), ("allowed_modes", ["3", "0"])):
            with self.subTest(key=key):
                self.documents[self.plan_path] = {**self.value, key: invalid}
                with self.assertRaises(ValueError): common.load_pinned(self.witness_sha)

    def test_external_binding_loads_only_exact_synthetic_current_stage_and_grade(self):
        self.assertEqual(common.validate_binding(self.value, self.witness_sha, self.binding_sha), self.binding)
        self.assertEqual(self.reads, [self.binding_path, self.value["native_stage_receipt"], self.grade_path])
        self.assertEqual(self.hashes, [self.binding_path, self.value["native_stage_receipt"], self.grade_path])

    def test_external_actual_proof_binding_pin_required(self):
        for pin in (None, "", "f" * 64, "0" * 63):
            with self.subTest(pin=pin):
                with self.assertRaises(ValueError): common.validate_binding(self.value, self.witness_sha, pin)
        self.assertEqual(self.reads, [])

    def test_binding_schema_pass_and_exact_witness_cannot_drift(self):
        for key, invalid in (("schema", "unknown-binding-v1"), ("pass", False), ("pass", 1),
                              ("launch_witness_sha256", "f" * 64)):
            with self.subTest(key=key, invalid=invalid):
                self.documents[self.binding_path] = {**self.binding, key: invalid}
                with self.assertRaises(ValueError): common.validate_binding(self.value, self.witness_sha, self.binding_sha)

    def test_stage_or_grade_digest_drift_refused_before_actual_metadata_read(self):
        for path in (self.value["native_stage_receipt"], self.grade_path):
            with self.subTest(path=path):
                previous = self.digests[path]; self.digests[path] = "f" * 64
                self.reads.clear()
                with self.assertRaises(ValueError): common.validate_binding(self.value, self.witness_sha, self.binding_sha)
                self.assertEqual(self.reads, [self.binding_path])
                self.digests[path] = previous

    def test_pinned_metadata_still_requires_actual_stage_and_bothwidth_grade(self):
        for path in (self.value["native_stage_receipt"], self.grade_path):
            with self.subTest(path=path):
                self.documents[path]["pass"] = False
                with self.assertRaises(ValueError): common.validate_binding(self.value, self.witness_sha, self.binding_sha)
                self.documents[path]["pass"] = True


class ProfileCounters(unittest.TestCase):
    def test_current_native_status_delegates_pinned_source_identity_and_adapter(self):
        value = profile_plan()
        for role in ("old", "new"):
            with self.subTest(role=role):
                helper = profile_helper(); native = status(role)
                self.assertEqual(audit.status_errors(native, value, role, helper), [])
                helper[1].status_errors.assert_called_once_with(native, helper[0], "mtp3", True, role == "new")
                helper[2].composition_identity_errors.assert_called_once_with(native, role, helper[0])

    def test_native_status_fixed_source_constructor_clocks_headroom_and_routes_required(self):
        for path, invalid in (("identity.target_gathered_mpp_max_physical_rows", 8),
                ("maximum_context_tokens", 4096), ("memory_audit.valid", 1),
                ("memory_audit.batch_prefill_workspace_bytes", 0),
                ("native_lifecycle_timestamps_sep22.enabled", False),
                ("identity.mtp_batch_teacher_priming_route", "old-teacher-route"),
                ("identity.batch_prefill_kernel_routes", audit.ARENA_ROUTE),
                ("identity.batch_prefill_kernel_routes", audit.BQSA_ROUTE),
                ("memory_governor.host_available_bytes", 0), ("memory_governor.host_reserve_bytes", "1")):
            with self.subTest(path=path):
                native = status(); put(native, path, invalid)
                self.assertTrue(audit.status_errors(native, profile_plan(), "new", profile_helper()))

    def test_native_raw_and_37b_wrapped_identities_cannot_be_interchanged(self):
        for role in ("old", "new"):
            with self.subTest(role=role):
                self.assertTrue(audit.status_errors(status("old" if role == "new" else "new"),
                    profile_plan(), role, profile_helper()))

    def test_B4_R16_B2_R8_and_old_zero_route_waves(self):
        for width in (4, 2):
            for role in ("old", "new"):
                with self.subTest(width=width, role=role):
                    row, errors = audit.wave_errors(wave(width, role), profile_plan(), role, profile_helper())
                    self.assertEqual(errors, [])
                    self.assertTrue(row["valid"])
                    self.assertFalse(row["prepared_predictions_used_as_output_numerator"])
                    self.assertEqual(row["new_BQSA4_lane_layer_calls_expected"], 48 if width == 4 else 0)

    def test_legitimate_B4_shrinks_to_R8_after_peer_finish(self):
        row, errors = audit.wave_errors(wave(4, "new", B4_to_R8=True), profile_plan(), "new", profile_helper())
        self.assertEqual(errors, [])
        self.assertTrue(row["legitimate_B4_to_R8_after_peer_finish_allowed"])

    def test_all_EOS_wave_can_have_zero_selected_integer_calls(self):
        for width in (4, 2):
            with self.subTest(width=width):
                value = wave(width, "new", selected_calls=0)
                value["actual_emitted_tokens"] = width
                value["finish_reasons"] = ["stop"] * width
                row, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
                self.assertEqual(errors, [])
                self.assertTrue(row["valid"])

    def test_integer0_control_cannot_encode_new_integer_work(self):
        value = wave(4, "old")
        for stage_name in ("plan", "gate", "down"):
            for kind, count in (("calls", 48), ("rows", 48 * 16)):
                path = f"compact_native_batch_verify.r16.{stage_name}_graph_{kind}"
                put(value["status_after"], path, count); value["native_counter_delta"][path] = count
        _, errors = audit.wave_errors(value, profile_plan(), "old", profile_helper())
        self.assertTrue(any("Integer0 control" in error for error in errors))

    def test_B2_cannot_claim_R16_graphs(self):
        value = wave(2, "new")
        for stage_name in ("plan", "gate", "down"):
            for kind, count in (("calls", 48), ("rows", 48 * 16)):
                path = f"compact_native_batch_verify.r16.{stage_name}_graph_{kind}"
                put(value["status_after"], path, count); value["native_counter_delta"][path] = count
        _, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
        self.assertTrue(any("PhysicalR16" in error for error in errors))

    def test_recorded_deltas_decreasing_counts_and_integer_stage_divergence_refused(self):
        path = "compact_native_batch_verify.r16.gate_graph_calls"
        for kind in ("recorded", "decreasing", "stage"):
            with self.subTest(kind=kind):
                value = wave()
                if kind == "recorded": value["native_counter_delta"][path] = 0
                elif kind == "decreasing": put(value["status_before"], path, 49)
                else:
                    put(value["status_after"], path, 96); value["native_counter_delta"][path] = 96
                _, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
                self.assertTrue(errors)

    def test_integer_graphs_require48layers_exact_physical_rows_and_native_cycles(self):
        for path, count in (("compact_native_batch_verify.r16.gate_graph_calls", 49),
                ("compact_native_batch_verify.r16.down_graph_rows", 1),
                ("scheduler.decode_batches_by_width.b4", 0)):
            with self.subTest(path=path):
                value = wave(); put(value["status_after"], path, count); value["native_counter_delta"][path] = count
                _, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
                self.assertTrue(errors)

    def test_true_cohort_QSA_teacher_counts_and_geometry_cannot_drift(self):
        for path in ("scheduler.prefill_batches_by_width.b4", "mtp.batch_teacher_cache_only_priming_calls",
                "mtp.batch_teacher_cache_only_completed_lanes", "mtp.batch_teacher_cache_only_completed_real_pairs",
                "mtp.teacher_cache_only_priming_calls", "batch_prefill.true_target_hidden_copied_bytes",
                "batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls"):
            with self.subTest(path=path):
                value = wave(); put(value["status_after"], path, 0); value["native_counter_delta"][path] = 0
                _, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
                self.assertTrue(errors)
        for key, invalid in (("http_width", 3), ("mtp_setting", "0"), ("prompt_tokens", 2049),
                              ("output_budget_tokens", 128)):
            with self.subTest(key=key):
                value = wave(); value[key] = invalid
                _, errors = audit.wave_errors(value, profile_plan(), "new", profile_helper())
                self.assertTrue(errors)


def measured_rows(role="new", all_zero=False):
    result = []
    for width in (4, 2):
        for trial in range(3):
            count = 48 if role == "new" and trial == 0 and not all_zero else 0
            row, errors = audit.wave_errors(wave(width, role, selected_calls=count), profile_plan(), role, profile_helper())
            assert not errors, errors
            row["trial"] = trial
            result.append(row)
    return result


class AggregateExposure(unittest.TestCase):
    def test_selected_R16_B4_R8_B2_positive_across_trials_allows_zero_waves(self):
        row, errors = audit.aggregate_integer_errors(measured_rows(), "new")
        self.assertEqual(errors, [])
        self.assertTrue(row["valid"])
        self.assertEqual(row["actual_measured_route_totals"], {
            "b4_r16_measured_graph_calls": 48, "b2_r8_measured_graph_calls": 48})

    def test_allzero_measured_profile_cannot_qualify_new_integer_routes(self):
        row, errors = audit.aggregate_integer_errors(measured_rows(all_zero=True), "new")
        self.assertFalse(row["valid"])
        self.assertEqual(len(errors), 2)

    def test_warmup_only_integer_work_does_not_qualify_measured_trials(self):
        rows = measured_rows(all_zero=True)
        for width in (4, 2):
            row, issues = audit.wave_errors(wave(width), profile_plan(), "new", profile_helper())
            self.assertEqual(issues, []); row["warmup"] = True; rows.append(row)
        result, errors = audit.aggregate_integer_errors(rows, "new")
        self.assertFalse(result["valid"])
        self.assertEqual(len(errors), 2)
        self.assertEqual(set(result["actual_measured_route_totals"].values()), {0})

    def test_missing_or_invalid_source_counters_cannot_qualify_aggregate(self):
        path = "compact_native_batch_verify.r16.plan_graph_calls"
        for value in (None, "48", True, -48, 49):
            with self.subTest(value=value):
                rows = measured_rows()
                if value is None: del rows[0]["actual_native_source_and_priming_counter_deltas"][path]
                else: rows[0]["actual_native_source_and_priming_counter_deltas"][path] = value
                result, errors = audit.aggregate_integer_errors(rows, "new")
                self.assertFalse(result["valid"])
                self.assertTrue(any("Missing/invalid" in error for error in errors))

    def test_old_control_aggregate_requires_zero_integer_work(self):
        rows = measured_rows("old")
        self.assertEqual(audit.aggregate_integer_errors(rows, "old")[1], [])
        rows[0]["actual_native_source_and_priming_counter_deltas"]["compact_native_batch_verify.r16.plan_graph_calls"] = 48
        self.assertTrue(audit.aggregate_integer_errors(rows, "old")[1])

    def test_exactly_three_measured_trials_at_each_width_required(self):
        for count in (0, 2, 4):
            with self.subTest(count=count):
                rows = measured_rows()
                other = [row for row in rows if row["width"] == 2]
                b4 = [row for row in rows if row["width"] == 4]
                selected = [copy.deepcopy(b4[index % 3]) for index in range(count)]
                result, errors = audit.aggregate_integer_errors(selected + other, "new")
                self.assertFalse(result["valid"])
                self.assertTrue(any("three measured trials" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
