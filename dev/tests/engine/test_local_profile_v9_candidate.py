"""CPU-only review preparation for local idle residency maintenance."""

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
    V5_ENVIRONMENT,
    V6_ENVIRONMENT,
    V7_ENVIRONMENT,
)


MAINTENANCE_FLAG = "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"
MAINTENANCE_PARENTS = (
    "SPLASH_FLASH_DENSE_CACHE",
    "SPLASH_FLASH_FLOAT_DENSE_CACHE",
    "SPLASH_FLASH_BLOCKED_MOE",
    "SPLASH_FLASH_OPERAND_STORE",
    "SPLASH_FLASH_INT8_EXPERT_STORE",
)
V8_ENVIRONMENT = dict(HISTORICAL_V8_ENVIRONMENT)
V9_ENVIRONMENT = {**V8_ENVIRONMENT, MAINTENANCE_FLAG: "1"}


class LocalProfileV9CandidateTests(unittest.TestCase):
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
            candidate = launcher._local_profile_v9_candidate(self.package, environment)
        qualify.assert_called_once_with(self.package, environment)
        return candidate

    def apply(self, environment):
        original = copy.deepcopy(environment)
        defaults = copy.deepcopy(V9_ENVIRONMENT)
        merged = dict(environment)
        launcher._apply_local_profile_defaults(merged, defaults)
        self.assertEqual(environment, original)
        self.assertEqual(defaults, V9_ENVIRONMENT)
        return merged

    def test_candidate_adds_only_maintenance_flag_to_v8_routes(self):
        candidate = self.candidate()
        self.assertEqual(candidate["profile"], "m5-ultra-flash-next-v9")
        self.assertEqual(candidate["environment"], V9_ENVIRONMENT)
        self.assertEqual(len(candidate["environment"]), 38)
        self.assertEqual(candidate["minimum_physical_ram_bytes"], 256 * 1024**3)
        for key in ("schema_version", "source_identity_sha256", "architecture", "cpu_brand"):
            self.assertEqual(candidate[key], launcher.LOCAL_PROFILE[key])

    def test_review_copy_does_not_activate_write_or_mutate_process(self):
        accepted = copy.deepcopy(launcher.LOCAL_PROFILE)
        process = dict(os.environ)
        first = self.candidate()
        second = self.candidate()
        first["environment"][MAINTENANCE_FLAG] = "0"
        first["minimum_physical_ram_bytes"] = 1
        self.assertEqual(second["environment"], V9_ENVIRONMENT)
        self.assertEqual(second["minimum_physical_ram_bytes"], 256 * 1024**3)
        self.assertEqual(launcher.LOCAL_PROFILE, accepted)
        self.assertEqual(launcher.LOCAL_PROFILE["profile"], "m5-ultra-flash-next-v12")
        self.assertEqual(dict(os.environ), process)
        self.assertFalse((self.root / ".splash-local-profile.json").exists())

    def test_historical_v8_is_rejected_before_model_or_hardware_probes(self):
        with mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}):
            historical = launcher._local_profile_v8_candidate(self.package, {})
        (self.root / ".splash-local-profile.json").write_text(json.dumps(historical))
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(launcher, "local_bundle_manifest") as manifest,
            mock.patch.object(launcher, "_local_hardware_identity") as hardware,
            mock.patch.object(launcher, "_qualified_saved_operand_defaults") as operands,
            mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults") as experts,
        ):
            self.assertEqual(launcher._local_profile_defaults(self.package), {})
        for probe in (manifest, hardware, operands, experts):
            probe.assert_not_called()

    def test_historical_v9_is_rejected_by_active_v12_before_probes(self):
        historical = self.candidate()
        (self.root / ".splash-local-profile.json").write_text(json.dumps(historical))
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(launcher, "local_bundle_manifest") as manifest,
            mock.patch.object(launcher, "_local_hardware_identity") as hardware,
            mock.patch.object(launcher, "_qualified_saved_operand_defaults") as operands,
            mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults") as experts,
        ):
            self.assertEqual(launcher._local_profile_defaults(self.package), {})
        for probe in (manifest, hardware, operands, experts):
            probe.assert_not_called()

    def test_active_v12_preserves_exact_historical_v5_through_v9_routes(self):
        for helper, name, expected, memory in (
            (launcher._local_profile_v5_candidate, "m5-ultra-flash-next-v5", V5_ENVIRONMENT, 192),
            (launcher._local_profile_v6_candidate, "m5-ultra-flash-next-v6", V6_ENVIRONMENT, 192),
            (launcher._local_profile_v7_candidate, "m5-ultra-flash-next-v7", V7_ENVIRONMENT, 192),
            (launcher._local_profile_v8_candidate, "m5-ultra-flash-next-v8", V8_ENVIRONMENT, 192),
            (launcher._local_profile_v9_candidate, "m5-ultra-flash-next-v9", V9_ENVIRONMENT, 256),
        ):
            with (
                self.subTest(profile=name),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}),
            ):
                historical = helper(self.package, {})
            self.assertEqual(historical["profile"], name)
            self.assertEqual(historical["environment"], expected)
            self.assertNotIn("SPLASH_FLASH_PLE_SSD_STREAMING", historical["environment"])
            self.assertEqual(historical["minimum_physical_ram_bytes"], memory * 1024**3)

    def test_each_required_parent_zero_disables_implied_maintenance(self):
        for parent in MAINTENANCE_PARENTS:
            with self.subTest(parent=parent):
                merged = self.apply({parent: "0", "TASK_SENTINEL": "keep"})
                self.assertEqual(merged[parent], "0")
                self.assertEqual(merged[MAINTENANCE_FLAG], "0")
                self.assertEqual(merged["TASK_SENTINEL"], "keep")
                self.assertEqual(merged["SPLASH_FLASH_GDN_LAZY_ROLLBACK"], "1")

    def test_hypothetical_accepted_v9_requires_256_gib_and_keeps_existing_source_gate(self):
        candidate = self.candidate()
        (self.root / ".splash-local-profile.json").write_text(json.dumps(candidate))
        for memory, expected in (
            (192 * 1024**3, {}),
            (255 * 1024**3, {}),
            (256 * 1024**3 - 1, {}),
            (256 * 1024**3, V9_ENVIRONMENT),
        ):
            with (
                self.subTest(memory=memory),
                mock.patch.object(launcher, "ROOT", self.root),
                mock.patch.object(launcher, "LOCAL_PROFILE", candidate),
                mock.patch.object(launcher, "local_bundle_manifest", return_value={
                    "schema": launcher.LOCAL_SCHEMA,
                    "source_identity_sha256": candidate["source_identity_sha256"],
                }),
                mock.patch.object(launcher, "_local_hardware_identity", return_value=("Apple M5 Ultra", memory)),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}) as operands,
                mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults", return_value={}) as experts,
            ):
                self.assertEqual(launcher._local_profile_defaults(self.package), expected)
                if not expected:
                    operands.assert_not_called()
                    experts.assert_not_called()
        for brand, source in (
            ("Apple M5 Max", candidate["source_identity_sha256"]),
            ("Apple M5 Ultra", "different-source"),
        ):
            with (
                self.subTest(brand=brand, source=source),
                mock.patch.object(launcher, "ROOT", self.root),
                mock.patch.object(launcher, "LOCAL_PROFILE", candidate),
                mock.patch.object(launcher, "local_bundle_manifest", return_value={
                    "schema": launcher.LOCAL_SCHEMA, "source_identity_sha256": source,
                }),
                mock.patch.object(launcher, "_local_hardware_identity", return_value=(brand, 256 * 1024**3)),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults") as operands,
                mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults") as experts,
            ):
                self.assertEqual(launcher._local_profile_defaults(self.package), {})
                operands.assert_not_called()
                experts.assert_not_called()

    def test_explicit_child_values_survive_disabled_parents_for_native_validation(self):
        for parent in MAINTENANCE_PARENTS:
            for value in ("0", "1", "", "invalid"):
                with self.subTest(parent=parent, value=value):
                    merged = self.apply({parent: "0", MAINTENANCE_FLAG: value})
                    self.assertEqual(merged[parent], "0")
                    self.assertEqual(merged[MAINTENANCE_FLAG], value)

    def test_independent_route_opt_outs_do_not_invent_maintenance_prerequisites(self):
        for parent in (
            "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT",
            "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE",
            "SPLASH_FLASH_BATCH",
            "SPLASH_FLASH_BATCH_MTP",
            "SPLASH_FLASH_MTP",
            "SPLASH_FLASH_BATCH_PREFILL",
            "SPLASH_FLASH_INT8_HEAD",
            "SPLASH_FLASH_GPU_GREEDY",
            "SPLASH_FLASH_QMV_F32",
            "SPLASH_FLASH_FUSE_HC",
            "SPLASH_FLASH_FUSE_GDN",
        ):
            with self.subTest(parent=parent):
                merged = self.apply({parent: "0"})
                self.assertEqual(merged[parent], "0")
                self.assertEqual(merged[MAINTENANCE_FLAG], "1")

    def test_nonzero_parent_values_remain_authoritative(self):
        for value in ("1", "", "invalid", "/explicit/artifact"):
            with self.subTest(value=value):
                merged = self.apply(dict.fromkeys(MAINTENANCE_PARENTS, value))
                self.assertEqual(merged[MAINTENANCE_FLAG], "1")
                for parent in MAINTENANCE_PARENTS:
                    self.assertEqual(merged[parent], value)

    def test_existing_explicit_interval_is_not_overwritten_or_added_as_a_default(self):
        for value in ("100", "500", "1000", "", "invalid"):
            with self.subTest(value=value):
                merged = self.apply({"SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS": value})
                self.assertEqual(merged["SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS"], value)
        self.assertNotIn("SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS", self.candidate()["environment"])

    def test_historical_helpers_retain_exact_routes_under_hypothetical_v9(self):
        hypothetical = self.candidate()
        for helper, name, expected in (
            (launcher._local_profile_v5_candidate, "m5-ultra-flash-next-v5", V5_ENVIRONMENT),
            (launcher._local_profile_v6_candidate, "m5-ultra-flash-next-v6", V6_ENVIRONMENT),
            (launcher._local_profile_v7_candidate, "m5-ultra-flash-next-v7", V7_ENVIRONMENT),
            (launcher._local_profile_v8_candidate, "m5-ultra-flash-next-v8", V8_ENVIRONMENT),
        ):
            with (
                self.subTest(profile=name),
                mock.patch.object(launcher, "LOCAL_PROFILE", hypothetical),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults", return_value={}),
            ):
                historical = helper(self.package, {})
                self.assertEqual(historical["profile"], name)
                self.assertEqual(historical["environment"], expected)
                self.assertNotIn(MAINTENANCE_FLAG, historical["environment"])
                self.assertNotIn("SPLASH_FLASH_PLE_SSD_STREAMING", historical["environment"])
                self.assertEqual(historical["minimum_physical_ram_bytes"], 192 * 1024**3)

    def test_optional_paths_and_payload_identity_pins_remain_unchanged(self):
        dense = copy.deepcopy(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION)
        experts = copy.deepcopy(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION)
        explicit = {"SPLASH_FLASH_OPERAND_STORE": "0", "SPLASH_FLASH_INT8_EXPERT_STORE": "0"}
        original = dict(explicit)
        candidate = self.candidate(explicit, {"SPLASH_FLASH_OPERAND_STORE": "/qualified/dense"})
        self.assertEqual(explicit, original)
        merged = dict(explicit)
        launcher._apply_local_profile_defaults(merged, candidate["environment"])
        self.assertEqual(merged[MAINTENANCE_FLAG], "0")
        for key, value in explicit.items():
            self.assertEqual(merged[key], value)
        self.assertEqual(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION, dense)
        self.assertEqual(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION, experts)

    def test_candidate_keeps_existing_model_routes_and_excludes_other_remedies(self):
        candidate = self.candidate()["environment"]
        for key in ("SPLASH_FLASH_PREFILL_ROWS", "SPLASH_FLASH_BATCH_PREFILL_ROWS"):
            self.assertEqual(candidate[key], "2048")
        self.assertEqual(candidate["SPLASH_FLASH_MTP_DRAFT_DEPTH"], "15")
        for key in (
            "SPLASH_FLASH_PLE_SSD_STREAMING",
            "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT",
            "SPLASH_FLASH_GPU_PREFILL_COPY",
            "SPLASH_FLASH_QSA_OUT_F32_N32",
            "SPLASH_FLASH_PRIVATE_OWNED_ORIGINAL",
            "SPLASH_FLASH_PRIVATE_COMPLETED_COMMAND_RETENTION",
            "SPLASH_FLASH_PRIVATE_TENSOR_BUFFERS",
            "SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_MAINTENANCE",
        ):
            self.assertNotIn(key, candidate)


if __name__ == "__main__":
    unittest.main()
