"""Synthetic CPU launcher tests; no fixture, plan, model or service reads."""
from __future__ import annotations

import builtins
import copy
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

SOURCE = Path(__file__).with_name("run_paired_quality.py")


def isolated_import():
    """Execute only the launcher source while refusing every runtime action."""
    source = SOURCE.read_text()
    module = ModuleType("_paired_quality_launcher_cpu_test")
    module.__file__ = str(SOURCE)
    real_import = builtins.__import__
    blocked = {"mlx", "torch", "transformers", "safetensors", "tokenizers", "Metal", "metal"}

    def import_source(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError("launcher import attempted a device/model dependency: " + name)
        return real_import(name, *args, **kwargs)

    def refused(*args, **kwargs):
        raise AssertionError("launcher import attempted runtime I/O or execution")

    with mock.patch.object(builtins, "__import__", side_effect=import_source), \
            mock.patch.object(builtins, "open", side_effect=refused), \
            mock.patch.object(Path, "read_text", side_effect=refused), \
            mock.patch.object(Path, "read_bytes", side_effect=refused), \
            mock.patch.object(subprocess, "Popen", side_effect=refused), \
            mock.patch.object(subprocess, "run", side_effect=refused), \
            mock.patch.object(os, "system", side_effect=refused), \
            mock.patch.object(socket, "socket", side_effect=refused), \
            mock.patch.object(time, "sleep", side_effect=refused), \
            mock.patch.object(urllib.request, "urlopen", side_effect=refused), \
            mock.patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


launcher = isolated_import()


def canonical_stage():
    """Independent synthetic dictionary of Root's exact current 7f82 contract."""
    return {
        "schema": "current-BQSA4-plus-integer-batchverify-selected7-all18logical-Root-native-admission-v1",
        "pass": True,
        "Root_GPU_executed": True,
        "candidate_worker_sha256": "ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68",
        "candidate_source_identity_sha256": "edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94",
        "BQSA4_policy_sha256": "b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07",
        "metallib_sha256": "74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8",
        "compiled_worker_seal_sha256": "32338e764d23627d0bfb74f12393c9aa84a0e4aaa37b2ec1da520c3c381f7461",
        "native_QA_seal_sha256": "80d7618e4b002a077ddfc2e9f801e6f5944e5bf40d5bfb61f77b29a5162b55d5",
        "proof_capacity": 4096,
        "all_four_pairs_complete": True,
        "selected7_fullphysical_qualified": True,
        "all18_logical_output_max16_MoE_rejections_owners_qualified": True,
        "all18_fullphysical_qualified": False,
        "full_numeric_flags_count": 47,
        "actual_verify_commands_per_role": 17,
        "actual_commit_commands_per_role": 16,
        "actual_rejection_checks_per_role": 13,
        "BQSA4_actual_fresh_grouped_prefill_layer_calls": 48,
        "B2_oldSG8_BQSA_new_layer_calls": 0,
        "trained_head_qualified": False,
        "worker_cancel_deadline_qualified": False,
        "original22_or_service16K_qualified": False,
        "performance_qualified": False,
        "public_promotion_qualified": False,
        "snapshot_reserved_zero_claimed": False,
        "parts": [
            {"ordinal": ordinal, "pass": True, "backend_destroyed": True, "frames": frames}
            for ordinal, frames in enumerate((158, 158, 158, 157), 1)
        ],
    }


def synthetic_config():
    env = {key: "1" for key in launcher.DEPS}
    env.update({
        launcher.BQSA_FLAG: "1",
        launcher.INTEGER_FLAG: "0",
        "SPLASH_FLASH_MTP_DRAFT_DEPTH": "3",
        "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS": "4",
        "SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21": "0",
        "SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS": "0",
        "SYNTHETIC_PRESERVED_FLAG": "source-only",
    })
    numeric = {"SYNTHETIC_NATIVE_FLAG_" + str(index): str(index % 2) for index in range(47)}
    env.update(numeric)
    return {
        "schema": "current-BQSA4-integer-MTP3-paired-original22-Root-plan-v1",
        "modes": ["mtp3"],
        "width_order": [4, 2],
        "same_worker_integer_only_control_delta": True,
        "ports": {"old": 8058, "new": 8059},
        "base_environment": env,
        "native_stage_numeric_flags": numeric,
        "native_stage_receipt": "/synthetic/current-stage.json",
        "native_stage_receipt_sha256": "7f82c4592d08d7d317b1b7a3d3fa21e7dbb84e61f8e84b1ca68beae629a327cd",
        "artifact_pins": {"/synthetic/launcher-source.py": "a" * 64},
        "quality_driver": "/synthetic/quality-source.py",
        "binding": "/synthetic/binding-metadata.json",
        "binding_sha256": "b" * 64,
        "package": "/synthetic/local-package",
        "model": "synthetic-bundled-tokenizer-model",
        "original_plan": "/synthetic/MUST-NOT-READ-plan.json",
        "original_plan_file_sha256": "c" * 64,
    }


class SyntheticValidation(unittest.TestCase):
    def setUp(self):
        self.config = synthetic_config()
        self.stage = canonical_stage()
        self.binding = {"worker_sha256": launcher.WORKER, "metallib_sha256": launcher.LIB,
                        "runtime_build": "/synthetic/composed-runtime"}
        self.quality = SimpleNamespace(
            load_binding=mock.Mock(return_value=self.binding),
            authenticate=mock.Mock(return_value={"synthetic": True}),
            frozen_plan=mock.Mock(side_effect=AssertionError("plan read forbidden")),
        )
        self.reads = []
        self.hashes = []

        def fake_read(path, *args, **kwargs):
            self.reads.append(str(path))
            self.assertEqual(str(path), "/synthetic/current-stage.json", "unexpected payload/source read")
            return json.dumps(self.stage)

        def fake_sha(path):
            path = str(path)
            self.hashes.append(path)
            if path == "/synthetic/current-stage.json":
                return launcher.STAGE_SHA
            self.assertIn(path, self.config["artifact_pins"], "unexpected fixture/model hash read")
            return self.config["artifact_pins"][path]

        self.patches = [
            mock.patch.object(Path, "read_text", autospec=True, side_effect=fake_read),
            mock.patch.object(Path, "read_bytes", side_effect=AssertionError("content read forbidden")),
            mock.patch.object(launcher, "sha", side_effect=fake_sha),
            mock.patch.object(launcher, "module", return_value=self.quality),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("service launch forbidden")),
            mock.patch.object(subprocess, "run", side_effect=AssertionError("process launch forbidden")),
            mock.patch.object(socket, "socket", side_effect=AssertionError("service access forbidden")),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_canonical_current_stage_source_admission_has_no_payload_reads(self):
        actual_quality, actual_binding = launcher.validate(self.config, payload_allowed=False)
        self.assertIs(actual_quality, self.quality)
        self.assertIs(actual_binding, self.binding)
        self.assertEqual(self.reads, ["/synthetic/current-stage.json"])
        self.assertEqual(self.hashes, ["/synthetic/current-stage.json", "/synthetic/launcher-source.py"])
        self.quality.frozen_plan.assert_not_called()
        self.quality.load_binding.assert_called_once_with(self.config["binding"], self.config["binding_sha256"])
        self.assertEqual(self.quality.authenticate.call_args_list, [
            mock.call("old", self.config["binding"], self.config["binding_sha256"]),
            mock.call("new", self.config["binding"], self.config["binding_sha256"]),
        ])

    def test_current_stage_externally_pinned_exact_7f82(self):
        self.assertEqual(launcher.STAGE_SHA, self.config["native_stage_receipt_sha256"])
        self.assertTrue(launcher.STAGE_SHA.startswith("7f82"))
        self.assertEqual(launcher.STAGE_SCHEMA, self.stage["schema"])

    def test_prior_23f15_stage_digest_refused_before_content_reads(self):
        self.config["native_stage_receipt_sha256"] = "23f15" + "0" * 59
        with self.assertRaisesRegex(ValueError, "current-stage external digest"):
            launcher.validate(self.config)
        self.assertEqual(self.reads, [])
        self.assertEqual(self.hashes, [])

    def test_unknown_stage_digest_refused(self):
        self.config["native_stage_receipt_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            launcher.validate(self.config)
        self.assertEqual(self.reads, [])

    def test_actual_stage_content_digest_drift_refused(self):
        with mock.patch.object(launcher, "sha", return_value="d" * 64):
            with self.assertRaisesRegex(ValueError, "receipt drift"):
                launcher.validate(self.config)
        self.assertEqual(self.reads, [])

    def test_unknown_execution_plan_schema_refused(self):
        self.config["schema"] = "unknown-plan-v1"
        with self.assertRaisesRegex(ValueError, "paired composition plan"):
            launcher.validate(self.config)
        self.assertEqual(self.hashes, [])

    def test_same_worker_binding_mismatch_refused(self):
        self.binding["worker_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Same current composed worker"):
            launcher.validate(self.config)

    def test_same_library_binding_mismatch_refused(self):
        self.binding["metallib_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Same current composed worker"):
            launcher.validate(self.config)

    def test_only_width4_then2_admitted(self):
        for widths in ([2, 4], [4], [2], [4, 2, 1], [4, 3], [True, 2], "4,2", None):
            with self.subTest(widths=widths):
                self.config["width_order"] = widths
                with self.assertRaisesRegex(ValueError, "B4 then B2"):
                    launcher.validate(self.config)
        self.assertEqual(self.hashes, [])

    def test_native_numeric_tuple_requires_all47_string_keys_and_values(self):
        valid = dict(self.config["native_stage_numeric_flags"])
        invalids = [None, [], {}, {key: value for index, (key, value) in enumerate(valid.items()) if index != 0},
                    {**valid, "synthetic48th": "1"}, {**valid, "SYNTHETIC_NATIVE_FLAG_0": 1}]
        bad_key = dict(valid)
        bad_key[0] = bad_key.pop("SYNTHETIC_NATIVE_FLAG_0")
        invalids.append(bad_key)
        for numeric in invalids:
            with self.subTest(numeric=numeric):
                self.config["native_stage_numeric_flags"] = numeric
                with self.assertRaisesRegex(ValueError, "47-flag tuple"):
                    launcher.validate(self.config)
        self.assertEqual(self.hashes, [])

    def test_native_numeric_tuple_must_match_every_base_environment_value(self):
        self.config["base_environment"]["SYNTHETIC_NATIVE_FLAG_46"] = "different"
        with self.assertRaisesRegex(ValueError, "Task numerical tuple differs"):
            launcher.validate(self.config)
        self.assertEqual(self.hashes, [])

    def test_STD_and_mixed_modes_refused(self):
        for modes in (["std"], ["standard"], ["mtp3", "std"], ["mtp2"], ["mtp4"], "mtp3", [], None):
            with self.subTest(modes=modes):
                self.config["modes"] = modes
                with self.assertRaisesRegex(ValueError, "Only paired MTP3"):
                    launcher.validate(self.config)
        self.assertEqual(self.hashes, [])

    def test_control_delta_declaration_must_be_true_boolean(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value):
                self.config["same_worker_integer_only_control_delta"] = value
                with self.assertRaises(ValueError):
                    launcher.validate(self.config)

    def test_private_distinct_nondefault_integer_ports_required(self):
        for ports in ({"old": 8058, "new": 8058}, {"old": 8000, "new": 8059},
                      {"old": 1024, "new": 8059}, {"old": 65536, "new": 8059},
                      {"old": "8058", "new": 8059}, {"old": True, "new": 8059}):
            with self.subTest(ports=ports):
                self.config["ports"] = ports
                with self.assertRaises(ValueError):
                    launcher.validate(self.config)
        self.assertEqual(self.hashes, [])


class StageSchema(unittest.TestCase):
    def test_canonical_all4_selected7_all18logical_is_admitted(self):
        self.assertIsNone(launcher.validate_stage(canonical_stage()))

    def test_unknown_and_prior_stage_schemas_refused(self):
        for schema in ("unknown-v1", "native-selected7-23f15-old-stage-v1", None):
            with self.subTest(schema=schema):
                stage = canonical_stage()
                stage["schema"] = schema
                with self.assertRaisesRegex(ValueError, "Exact current Root stage schema"):
                    launcher.validate_stage(stage)

    def test_nonobject_stage_refused(self):
        for stage in (None, [], "current", True):
            with self.subTest(stage=stage):
                with self.assertRaises(ValueError):
                    launcher.validate_stage(stage)

    def test_all_four_actual_pairs_required(self):
        for parts in ([], canonical_stage()["parts"][:3], canonical_stage()["parts"] + [{}], None):
            with self.subTest(parts=parts):
                stage = canonical_stage()
                stage["parts"] = parts
                with self.assertRaises(ValueError):
                    launcher.validate_stage(stage)

    def test_each_actual_role_pair_requires_true_teardown_and_pass(self):
        for index in range(4):
            for key in ("pass", "backend_destroyed"):
                for invalid in (False, 1, "true", None):
                    with self.subTest(index=index, key=key, invalid=invalid):
                        stage = canonical_stage()
                        stage["parts"][index][key] = invalid
                        with self.assertRaisesRegex(ValueError, "teardown"):
                            launcher.validate_stage(stage)

    def test_pair_order_and_frame_counts_are_exact_and_typed(self):
        for index in range(4):
            for key in ("ordinal", "frames"):
                for invalid in (0, "158", True, None):
                    with self.subTest(index=index, key=key, invalid=invalid):
                        stage = canonical_stage()
                        stage["parts"][index][key] = invalid
                        with self.assertRaises(ValueError):
                            launcher.validate_stage(stage)
        stage = canonical_stage()
        stage["parts"] = list(reversed(stage["parts"]))
        with self.assertRaises(ValueError):
            launcher.validate_stage(stage)

    def test_selected7_fullphysical_and_all18logical_both_required(self):
        for key in ("all_four_pairs_complete", "selected7_fullphysical_qualified",
                    "all18_logical_output_max16_MoE_rejections_owners_qualified"):
            with self.subTest(key=key):
                stage = canonical_stage()
                stage[key] = False
                with self.assertRaisesRegex(ValueError, key):
                    launcher.validate_stage(stage)

    def test_all18_fullphysical_overclaim_is_refused(self):
        stage = canonical_stage()
        stage["all18_fullphysical_qualified"] = True
        with self.assertRaisesRegex(ValueError, "all18_fullphysical_qualified"):
            launcher.validate_stage(stage)

    def test_every_current_identity_and_seal_is_bound(self):
        for key in ("candidate_worker_sha256", "candidate_source_identity_sha256", "BQSA4_policy_sha256",
                    "metallib_sha256", "compiled_worker_seal_sha256", "native_QA_seal_sha256"):
            with self.subTest(key=key):
                stage = canonical_stage()
                stage[key] = "f" * 64
                with self.assertRaisesRegex(ValueError, key):
                    launcher.validate_stage(stage)

    def test_no_task_trained_head_or_performance_claim_is_inherited(self):
        for key in ("trained_head_qualified", "worker_cancel_deadline_qualified", "original22_or_service16K_qualified",
                    "performance_qualified", "public_promotion_qualified", "snapshot_reserved_zero_claimed"):
            with self.subTest(key=key):
                stage = canonical_stage()
                stage[key] = True
                with self.assertRaisesRegex(ValueError, key):
                    launcher.validate_stage(stage)

    def test_proof_counts_and_capacity_are_strict_integers(self):
        for key in ("proof_capacity", "full_numeric_flags_count", "actual_verify_commands_per_role",
                    "actual_commit_commands_per_role", "actual_rejection_checks_per_role",
                    "BQSA4_actual_fresh_grouped_prefill_layer_calls", "B2_oldSG8_BQSA_new_layer_calls"):
            for invalid in (False, str(canonical_stage()[key]), float(canonical_stage()[key]), None):
                with self.subTest(key=key, invalid=invalid):
                    stage = canonical_stage()
                    stage[key] = invalid
                    with self.assertRaisesRegex(ValueError, key):
                        launcher.validate_stage(stage)

    def test_every_required_top_level_field_must_be_present(self):
        for key in canonical_stage():
            with self.subTest(key=key):
                stage = canonical_stage()
                del stage[key]
                with self.assertRaises(ValueError):
                    launcher.validate_stage(stage)


class RoleEnvironment(unittest.TestCase):
    def test_roles_differ_only_by_integer_0_1_flag(self):
        config = synthetic_config()
        before = copy.deepcopy(config)
        old = launcher.role_environment(config, "old")
        new = launcher.role_environment(config, "new")
        changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
        self.assertEqual(changed, {launcher.INTEGER_FLAG})
        self.assertEqual(old[launcher.INTEGER_FLAG], "0")
        self.assertEqual(new[launcher.INTEGER_FLAG], "1")
        self.assertEqual(old[launcher.BQSA_FLAG], "1")
        self.assertEqual(new[launcher.BQSA_FLAG], "1")
        self.assertEqual(config, before)
        self.assertIsNot(old, config["base_environment"])
        self.assertIsNot(new, old)

    def test_mtp3_and_gatheredcap4_and_bulk0_QA0_same_both_roles(self):
        for role in ("old", "new"):
            with self.subTest(role=role):
                env = launcher.role_environment(synthetic_config(), role)
                self.assertEqual(env["SPLASH_FLASH_MTP_DRAFT_DEPTH"], "3")
                self.assertEqual(env["SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS"], "4")
                self.assertEqual(env["SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21"], "0")
                self.assertEqual(env["SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS"], "0")

    def test_unknown_STD_and_nonstring_roles_refused(self):
        for role in ("std", "standard", "unknown", "OLD", None, 0):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    launcher.role_environment(synthetic_config(), role)

    def test_bqsa_must_be_enabled_for_each_role(self):
        for role in ("old", "new"):
            for invalid in ("0", 1, True, None):
                with self.subTest(role=role, invalid=invalid):
                    config = synthetic_config()
                    config["base_environment"][launcher.BQSA_FLAG] = invalid
                    with self.assertRaisesRegex(ValueError, "Same BQSA4"):
                        launcher.role_environment(config, role)

    def test_registered_control_must_start_integer_string0(self):
        for invalid in ("1", 0, False, None):
            with self.subTest(invalid=invalid):
                config = synthetic_config()
                config["base_environment"][launcher.INTEGER_FLAG] = invalid
                with self.assertRaisesRegex(ValueError, "integer0"):
                    launcher.role_environment(config, "new")

    def test_every_prerequisite_string1_required(self):
        self.assertEqual(len(launcher.DEPS), 13)
        self.assertEqual(len(set(launcher.DEPS)), 13)
        for key in launcher.DEPS:
            for invalid in ("0", 1, True, None):
                with self.subTest(key=key, invalid=invalid):
                    config = synthetic_config()
                    config["base_environment"][key] = invalid
                    with self.assertRaisesRegex(ValueError, "prerequisite"):
                        launcher.role_environment(config, "old")

    def test_depth2_depth4_and_STD_depth0_refused(self):
        for invalid in ("0", "2", "4", 3, None):
            with self.subTest(invalid=invalid):
                config = synthetic_config()
                config["base_environment"]["SPLASH_FLASH_MTP_DRAFT_DEPTH"] = invalid
                with self.assertRaisesRegex(ValueError, "native MTP3"):
                    launcher.role_environment(config, "old")

    def test_gathered_cap_bulk_and_pause_mismatches_refused(self):
        for key, values in {
                "SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS": ("2", "8", 4, None),
                "SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21": ("1", 0, False, None),
                "SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS": ("1", 0, False, None),
        }.items():
            for invalid in values:
                with self.subTest(key=key, invalid=invalid):
                    config = synthetic_config()
                    config["base_environment"][key] = invalid
                    with self.assertRaises(ValueError):
                        launcher.role_environment(config, "new")

    def test_ambient_environment_is_not_an_additional_role_delta(self):
        with mock.patch.dict(os.environ, {launcher.INTEGER_FLAG: "1", launcher.BQSA_FLAG: "0",
                                          "SYNTHETIC_AMBIENT_ONLY": "refused"}):
            old = launcher.role_environment(synthetic_config(), "old")
            new = launcher.role_environment(synthetic_config(), "new")
        self.assertNotIn("SYNTHETIC_AMBIENT_ONLY", old)
        self.assertNotIn("SYNTHETIC_AMBIENT_ONLY", new)
        self.assertEqual(old["SYNTHETIC_PRESERVED_FLAG"], "source-only")


class LocalPackageCommand(unittest.TestCase):
    def test_exact_local_package_command_uses_bundled_tokenizer(self):
        config = synthetic_config()
        build = Path("/synthetic/composed-runtime")
        for role in ("old", "new"):
            with self.subTest(role=role):
                command = launcher.server_command(config, build, role)
                self.assertEqual(command, [
                    sys.executable, "-u", str(launcher.ROOT / "server/server.py"),
                    "--local-package", "/synthetic/local-package", "--model", "synthetic-bundled-tokenizer-model",
                    "--binary", "/synthetic/composed-runtime/splash-flash", "--port", str(config["ports"][role]),
                    "--max-memory", "auto", "--max-context", "16384", "--no-webui",
                ])
                self.assertFalse(any("tokenizer" in item and item.startswith("--") for item in command))
                self.assertNotIn("--model-dir", command)
                self.assertNotIn("--hf-model", command)

    def test_same_worker_model_package_command_changes_only_registered_port(self):
        config = synthetic_config()
        build = Path("/synthetic/composed-runtime")
        old = launcher.server_command(config, build, "old")
        new = launcher.server_command(config, build, "new")
        delta = [index for index, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        self.assertEqual(delta, [old.index("--port") + 1])
        self.assertEqual(old[delta[0]], "8058")
        self.assertEqual(new[delta[0]], "8059")


class LauncherImport(unittest.TestCase):
    def test_import_is_source_only_and_no_main_executes(self):
        module = isolated_import()
        self.assertEqual(module.__name__, "_paired_quality_launcher_cpu_test")


if __name__ == "__main__":
    unittest.main()
