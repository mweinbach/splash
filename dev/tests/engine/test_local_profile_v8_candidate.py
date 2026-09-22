"""CPU qualification for accepted and review-copy exact-BF16 vocabulary defaults."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from dev.tests.engine.test_local_profile import (
    EXPECTED_ENVIRONMENT,
    V8_ENVIRONMENT as HISTORICAL_V8_ENVIRONMENT,
    V7_ENVIRONMENT,
    V5_ENVIRONMENT,
    V6_ENVIRONMENT,
)


REGISTER_FLAG = "SPLASH_FLASH_MTP_Q8_BF16_REGISTER"
REGISTER_PARENTS = (
    "SPLASH_FLASH_DENSE_CACHE",
    "SPLASH_FLASH_BATCH_MTP",
    "SPLASH_FLASH_MTP",
    "SPLASH_FLASH_BATCH",
)
V8_ENVIRONMENT = dict(HISTORICAL_V8_ENVIRONMENT)


class LocalProfileV8CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        (self.package / "config.json").write_text('{"model_type":"qwen4_exp"}')

    def candidate(self, environment=None, optional=None):
        environment = {} if environment is None else environment
        optional = {} if optional is None else optional
        with mock.patch.object(
            launcher, "_qualified_saved_operand_defaults", return_value=optional
        ) as qualify:
            candidate = launcher._local_profile_v8_candidate(
                self.package, environment
            )
        qualify.assert_called_once_with(self.package, environment)
        return candidate

    def apply(self, environment):
        original = copy.deepcopy(environment)
        defaults = copy.deepcopy(V8_ENVIRONMENT)
        merged = dict(environment)
        launcher._apply_local_profile_defaults(merged, defaults)
        self.assertEqual(environment, original)
        self.assertEqual(defaults, V8_ENVIRONMENT)
        return merged

    def test_candidate_has_37_static_defaults_and_only_one_change_from_v7(self):
        candidate = self.candidate()
        self.assertEqual(candidate["profile"], "m5-ultra-flash-next-v8")
        self.assertEqual(candidate["environment"], V8_ENVIRONMENT)
        self.assertEqual(candidate["environment"], {**V7_ENVIRONMENT, REGISTER_FLAG: "1"})
        self.assertEqual(len(candidate["environment"]), 37)
        self.assertEqual(
            {key: value for key, value in candidate.items() if key not in ("profile", "environment")},
            {**{key: value for key, value in launcher.LOCAL_PROFILE.items() if key not in ("profile", "environment")}, "minimum_physical_ram_bytes": 192 * 1024**3},
        )

    def test_active_profile_and_process_environment_stay_unchanged(self):
        before = copy.deepcopy(launcher.LOCAL_PROFILE)
        process_before = dict(os.environ)
        first = self.candidate()
        second = self.candidate()
        first["environment"][REGISTER_FLAG] = "0"
        first["environment"]["TASK_SENTINEL"] = "changed"
        self.assertEqual(second["environment"], V8_ENVIRONMENT)
        self.assertEqual(launcher.LOCAL_PROFILE, before)
        self.assertEqual(launcher.LOCAL_PROFILE["profile"], "m5-ultra-flash-next-v12")
        self.assertEqual(launcher.LOCAL_PROFILE["environment"], EXPECTED_ENVIRONMENT)
        self.assertEqual(dict(os.environ), process_before)

    def test_historical_v8_and_v7_are_rejected_by_accepted_v12_gate(self):
        candidate = self.candidate()
        (self.root / ".splash-local-profile.json").write_text(json.dumps(candidate))
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(launcher, "local_bundle_manifest", return_value={
                "schema": launcher.LOCAL_SCHEMA,
                "source_identity_sha256": candidate["source_identity_sha256"],
            }) as manifest,
            mock.patch.object(launcher, "_local_hardware_identity", return_value=("Apple M5 Ultra", 256 * 1024**3)) as hardware,
            mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}),
            mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults", return_value={}),
        ):
            self.assertEqual(launcher._local_profile_defaults(self.package), {})
        manifest.assert_not_called()
        hardware.assert_not_called()
        historical = {**candidate, "profile": "m5-ultra-flash-next-v7", "environment": dict(V7_ENVIRONMENT)}
        (self.root / ".splash-local-profile.json").write_text(json.dumps(historical))
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(launcher, "local_bundle_manifest") as manifest,
            mock.patch.object(launcher, "_local_hardware_identity") as hardware,
        ):
            self.assertEqual(launcher._local_profile_defaults(self.package), {})
        manifest.assert_not_called()
        hardware.assert_not_called()

    def test_each_operative_parent_zero_disables_only_the_implied_register_default(self):
        for parent in REGISTER_PARENTS:
            with self.subTest(parent=parent):
                environment = self.apply({parent: "0", "TASK_SENTINEL": "keep"})
                self.assertEqual(environment[parent], "0")
                self.assertEqual(environment[REGISTER_FLAG], "0")
                self.assertEqual(environment["TASK_SENTINEL"], "keep")
                self.assertEqual(environment["SPLASH_FLASH_HC_UP_F32_MPP"], "1")
                self.assertEqual(environment["SPLASH_FLASH_GDN_LAZY_ROLLBACK"], "1")

    def test_explicit_register_values_survive_disabled_parents(self):
        for value in ("0", "1", "", "invalid"):
            for parent in REGISTER_PARENTS:
                with self.subTest(value=value, parent=parent):
                    environment = self.apply({parent: "0", REGISTER_FLAG: value})
                    self.assertEqual(environment[parent], "0")
                    self.assertEqual(environment[REGISTER_FLAG], value)
        environment = self.apply({**dict.fromkeys(REGISTER_PARENTS, "0"), REGISTER_FLAG: "1"})
        self.assertEqual(environment[REGISTER_FLAG], "1")

    def test_unrelated_route_opt_outs_do_not_disable_register_default(self):
        for parent in (
            "SPLASH_FLASH_INT8_HEAD",
            "SPLASH_FLASH_GPU_GREEDY",
            "SPLASH_FLASH_BATCH_MTP_PREFILL",
            "SPLASH_FLASH_BATCH_PREFILL",
            "SPLASH_FLASH_FLOAT_DENSE_CACHE",
            "SPLASH_FLASH_QMV_F32",
            "SPLASH_FLASH_MTP_QMV_F32",
            "SPLASH_FLASH_FUSE_HC",
            "SPLASH_FLASH_FUSE_GDN",
        ):
            with self.subTest(parent=parent):
                environment = self.apply({parent: "0"})
                self.assertEqual(environment[parent], "0")
                self.assertEqual(environment[REGISTER_FLAG], "1")

    def test_explicit_joint_and_register_overrides_survive_ordinary_batch_opt_out(self):
        environment = self.apply({
            "SPLASH_FLASH_BATCH": "0",
            "SPLASH_FLASH_BATCH_MTP": "1",
            REGISTER_FLAG: "1",
        })
        self.assertEqual(environment["SPLASH_FLASH_BATCH"], "0")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_MTP"], "1")
        self.assertEqual(environment[REGISTER_FLAG], "1")

    def test_nonzero_parent_values_are_not_reinterpreted_by_launcher(self):
        for value in ("1", "", "invalid"):
            with self.subTest(value=value):
                environment = self.apply(dict.fromkeys(REGISTER_PARENTS, value))
                self.assertEqual(environment[REGISTER_FLAG], "1")
                for parent in REGISTER_PARENTS:
                    self.assertEqual(environment[parent], value)

    def test_exact_v5_v6_v7_historical_copies_remain_distinct_from_active_v8(self):
        hypothetical = {**launcher.LOCAL_PROFILE, "profile": "m5-ultra-flash-next-v8", "environment": dict(V8_ENVIRONMENT)}
        for helper, name, expected in (
            (launcher._local_profile_v5_candidate, "m5-ultra-flash-next-v5", V5_ENVIRONMENT),
            (launcher._local_profile_v6_candidate, "m5-ultra-flash-next-v6", V6_ENVIRONMENT),
            (launcher._local_profile_v7_candidate, "m5-ultra-flash-next-v7", V7_ENVIRONMENT),
        ):
            with (
                self.subTest(profile=name),
                mock.patch.object(launcher, "LOCAL_PROFILE", hypothetical),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}),
            ):
                historical = helper(self.package, {})
                self.assertEqual(historical["profile"], name)
                self.assertEqual(historical["environment"], expected)
                self.assertNotIn(REGISTER_FLAG, historical["environment"])
                self.assertNotIn("SPLASH_FLASH_PLE_SSD_STREAMING", historical["environment"])
        self.assertEqual((len(V5_ENVIRONMENT), len(V6_ENVIRONMENT), len(V7_ENVIRONMENT)), (30, 34, 36))

    def test_optional_artifact_defaults_keep_existing_pins_and_explicit_zero_selectors(self):
        pins = copy.deepcopy(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION)
        int8_pins = copy.deepcopy(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION)
        optional = {
            "SPLASH_FLASH_OPERAND_STORE": "/qualified/dense",
            "SPLASH_FLASH_INT8_EXPERT_STORE": "/qualified/top64",
        }
        original = {
            "SPLASH_FLASH_OPERAND_STORE": "0",
            "SPLASH_FLASH_INT8_EXPERT_STORE": "0",
            REGISTER_FLAG: "0",
        }
        original_before = dict(original)
        candidate = self.candidate(original, optional)
        self.assertEqual(original, original_before)
        self.assertEqual(candidate["environment"], {**V8_ENVIRONMENT, **optional})
        launcher._apply_local_profile_defaults(original, candidate["environment"])
        for key, value in original_before.items():
            self.assertEqual(original[key], value)
        self.assertEqual(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION, pins)
        self.assertEqual(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION, int8_pins)

    def test_candidate_keeps_windows_depth_and_excludes_unqualified_experiments(self):
        environment = self.candidate()["environment"]
        for key in ("SPLASH_FLASH_PREFILL_ROWS", "SPLASH_FLASH_BATCH_PREFILL_ROWS"):
            self.assertEqual(environment[key], "2048")
        self.assertEqual(environment["SPLASH_FLASH_MTP_DRAFT_DEPTH"], "15")
        for key in (
            "SPLASH_FLASH_PLE_SSD_STREAMING",
            "SPLASH_FLASH_QSA_OUT_F32_N32",
            "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT",
            "SPLASH_FLASH_GPU_PREFILL_COPY",
            "SPLASH_FLASH_PRIVATE_OWNED_ORIGINAL",
            "SPLASH_FLASH_PRIVATE_COMPLETED_COMMAND_RETENTION",
            "SPLASH_FLASH_PRIVATE_TENSOR_BUFFERS",
        ):
            self.assertNotIn(key, environment)


if __name__ == "__main__":
    unittest.main()
