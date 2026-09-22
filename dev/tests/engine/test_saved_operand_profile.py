"""CPU-only saved-weight gates, active v12 defaults, and historical review copies."""

import copy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from install import launcher
from dev.tests.engine.test_local_profile import EXPECTED_ENVIRONMENT
from dev.tests.flash import flash_operand_store_fixtures as operands


STORE_ENV = "SPLASH_FLASH_OPERAND_STORE"
DIRECT_A = "SPLASH_FLASH_MOE_DIRECT_A"
QSA_ROWS = "SPLASH_FLASH_QSA_ROW_TILES"
SAVED_RESIDENT = "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT"
V6_NEW_DEFAULTS = {
    "SPLASH_FLASH_SHARED_EXPERT_FUSED": "1",
    "SPLASH_FLASH_DENSE_M64_OUT": "1",
    "SPLASH_FLASH_GDN_BATCH_ILP": "1",
    "SPLASH_FLASH_MTP_QMV_F32": "1",
}
V7_NEW_DEFAULTS = {
    "SPLASH_FLASH_HC_UP_F32_MPP": "1",
    "SPLASH_FLASH_GDN_LAZY_ROLLBACK": "1",
}
V8_NEW_DEFAULTS = {"SPLASH_FLASH_MTP_Q8_BF16_REGISTER": "1"}
V9_NEW_DEFAULTS = {"SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "1"}
V10_NEW_DEFAULTS = {
    "SPLASH_FLASH_PLE_SSD_STREAMING": "1",
    "SPLASH_FLASH_QSA_OUT_F32_N32": "1",
}
V11_NEW_DEFAULTS = {
    "SPLASH_FLASH_PREFILL_DENSE_TILES": "1",
    "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY": "1",
}
V12_NEW_DEFAULTS = {
    "SPLASH_FLASH_QSA_BULK_PREFILL": "1",
    "SPLASH_FLASH_QSA_BULK_PREFILL_SG8": "1",
}
V5_STATIC_ENVIRONMENT = {
    key: value for key, value in EXPECTED_ENVIRONMENT.items()
    if key not in V6_NEW_DEFAULTS and key not in V7_NEW_DEFAULTS
    and key not in V8_NEW_DEFAULTS and key not in V9_NEW_DEFAULTS
    and key not in V10_NEW_DEFAULTS
    and key not in V11_NEW_DEFAULTS
    and key not in V12_NEW_DEFAULTS
}
V5_STATIC_ENVIRONMENT["SPLASH_FLASH_MTP_DRAFT_DEPTH"] = "15"
V6_DEPENDENCIES = {
    "SPLASH_FLASH_SHARED_EXPERT_FUSED": ("SPLASH_FLASH_DENSE_CACHE",),
    "SPLASH_FLASH_DENSE_M64_OUT": ("SPLASH_FLASH_DENSE_CACHE",),
    "SPLASH_FLASH_GDN_BATCH_ILP": ("SPLASH_FLASH_GDN_STAGED", "SPLASH_FLASH_BATCH_PREFILL"),
    "SPLASH_FLASH_MTP_QMV_F32": ("SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_MTP"),
}
V6_STATIC_ENVIRONMENT = {**V5_STATIC_ENVIRONMENT, **V6_NEW_DEFAULTS}
V7_STATIC_ENVIRONMENT = {**V6_STATIC_ENVIRONMENT, **V7_NEW_DEFAULTS}
V7_DEPENDENCIES = {
    "SPLASH_FLASH_HC_UP_F32_MPP": ("SPLASH_FLASH_FUSE_HC", "SPLASH_FLASH_FLOAT_DENSE_CACHE"),
    "SPLASH_FLASH_GDN_LAZY_ROLLBACK": ("SPLASH_FLASH_FUSE_GDN",),
}


class _SavedArtifactFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.package = self.root / "aligned"
        self.package.mkdir()
        self.aligned = {
            "schema": launcher.LOCAL_SCHEMA,
            "alignment": 16384,
            "source_identity_sha256": "a" * 64,
        }
        operands.write_manifest(self.package, self.aligned)
        self.production_pins = copy.deepcopy(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION)
        aligned_digest = self.digest(self.package / "manifest.json")
        source = self.aligned["source_identity_sha256"]
        fingerprint = hashlib.sha256(
            ("splash.native-flash-weights-v1\nsource=" + source
             + "\nmanifest=" + aligned_digest + "\nnorm=one-plus-weight\n").encode()
        ).hexdigest()
        self.store = self.root / self.production_pins["relative_path"]
        self.saved = operands.create_fixture(self.store)
        self.saved["source_identity_sha256"] = source
        self.saved["weights_manifest_fingerprint"] = fingerprint
        operands.write_manifest(self.store, self.saved)
        self.pins = {
            "relative_path": self.production_pins["relative_path"],
            "manifest_sha256": self.digest(self.store / "manifest.json"),
            "source_identity_sha256": source,
            "weights_manifest_fingerprint": fingerprint,
            "aligned_manifest_sha256": aligned_digest,
        }
        for field, value in (("ROOT", self.root), ("LOCAL_SAVED_OPERAND_QUALIFICATION", self.pins)):
            patch = mock.patch.object(launcher, field, value)
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def rewrite_saved(self, manifest, *, repin=False):
        operands.write_manifest(self.store, manifest)
        if repin:
            self.pins["manifest_sha256"] = self.digest(self.store / "manifest.json")

    def rewrite_aligned(self, manifest, *, repin=False):
        operands.write_manifest(self.package, manifest)
        if repin:
            self.pins["aligned_manifest_sha256"] = self.digest(self.package / "manifest.json")

    def defaults(self, environment=None):
        return launcher._qualified_saved_operand_defaults(self.package, environment=environment)

    def candidate(self, environment=None):
        return launcher._local_profile_v7_candidate(self.package, environment=environment)

    def prepare_qualified_profile(self):
        profile = copy.deepcopy(launcher.LOCAL_PROFILE)
        profile["source_identity_sha256"] = self.pins["source_identity_sha256"]
        (self.root / ".splash-local-profile.json").write_bytes(operands.canonical(profile))
        (self.package / "config.json").write_bytes(b'{"model_type":"qwen4_exp"}')
        return profile


class SavedOperandProfileTests(_SavedArtifactFixture):
    def test_production_qualification_has_exact_identity_pins(self):
        self.assertEqual(self.production_pins, {
            "relative_path": "install/local-models/Flash-Next-operands-v1",
            "manifest_sha256": "433e8a0ea5150fc063b7ccd02fc91191ba08032640ec2fd5d63cff5ece129512",
            "source_identity_sha256": "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
            "weights_manifest_fingerprint": "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
            "aligned_manifest_sha256": "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402",
        })

    def test_exact_qualified_artifact_returns_root_relative_absolute_path(self):
        expected = {STORE_ENV: str(self.store.resolve())}
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.defaults({}), expected)
        self.assertTrue(Path(expected[STORE_ENV]).is_absolute())
        self.assertNotEqual(Path(expected[STORE_ENV]).parent, self.package)
        first, second = self.defaults({}), self.defaults({})
        self.assertIsNot(first, second)
        first[STORE_ENV] = "changed"
        self.assertEqual(second, expected)

    def test_absent_store_returns_no_defaults_before_inspecting_package(self):
        self.store.rename(self.store.with_name("absent-original"))
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("package inspected")):
            self.assertEqual(
                launcher._qualified_saved_operand_defaults(self.root / "missing-package", environment={}), {}
            )

    def test_explicit_store_empty_zero_or_custom_path_is_preserved_before_inspection(self):
        for value in ("", "0", "/tmp/custom-operands", "relative/path"):
            environment = {STORE_ENV: value, "TASK_SENTINEL": "keep"}
            before = dict(environment)
            with self.subTest(value=value), mock.patch.object(
                Path, "read_bytes", side_effect=AssertionError("manifest inspected")
            ), mock.patch.object(Path, "exists", side_effect=AssertionError("store inspected")):
                self.assertEqual(self.defaults(environment), {})
            self.assertEqual(environment, before)

    def test_none_environment_uses_process_values_without_mutating_them(self):
        for value in ("", "0", "/tmp/explicit"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {STORE_ENV: value, "TASK_SENTINEL": "keep"}, clear=True
            ):
                self.assertEqual(self.defaults(), {})
                self.assertEqual(dict(os.environ), {STORE_ENV: value, "TASK_SENTINEL": "keep"})

    def test_explicit_empty_environment_does_not_inherit_process_override(self):
        with mock.patch.dict(os.environ, {STORE_ENV: "/tmp/process-only"}, clear=True):
            self.assertEqual(self.defaults({}), {STORE_ENV: str(self.store.resolve())})
            self.assertEqual(dict(os.environ), {STORE_ENV: "/tmp/process-only"})

    def test_unqualified_existing_store_digest_raises_instead_of_falling_back(self):
        changed = copy.deepcopy(self.saved)
        changed["entries"][0]["projection"] = "another.source.view"
        self.rewrite_saved(changed)
        with self.assertRaises(launcher.LauncherError):
            self.defaults({})

    def test_corrupt_existing_manifest_bytes_raise(self):
        for raw in (b"{", b"null", b"[]", b"\xff", b""):
            with self.subTest(raw=raw):
                operands.write_manifest(self.store, self.saved, raw=raw)
                with self.assertRaises(launcher.LauncherError):
                    self.defaults({})

    def test_existing_store_source_and_fingerprint_gates_are_independent_of_hash(self):
        for field, value in (("schema", "another-schema"),
                             ("source_identity_sha256", "d" * 64),
                             ("weights_manifest_fingerprint", "d" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.saved)
                changed[field] = value
                self.rewrite_saved(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.defaults({})

    def test_aligned_package_must_have_exact_raw_manifest_bytes(self):
        changed = copy.deepcopy(self.aligned)
        changed["unused_identity_metadata"] = "different artifact"
        self.rewrite_aligned(changed)
        with self.assertRaises(launcher.LauncherError):
            self.defaults({})

    def test_aligned_package_schema_and_source_gates_are_independent_of_hash(self):
        for field, value in (("schema", "another-schema"), ("source_identity_sha256", "d" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.aligned)
                changed[field] = value
                self.rewrite_aligned(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.defaults({})

    def test_effective_fingerprint_binds_norm_and_layout_identity(self):
        original = self.pins["weights_manifest_fingerprint"]
        self.pins["weights_manifest_fingerprint"] = "d" * 64
        with self.assertRaisesRegex(launcher.LauncherError, "fingerprint"):
            self.defaults({})
        self.pins["weights_manifest_fingerprint"] = original
        self.assertEqual(self.defaults({}), {STORE_ENV: str(self.store.resolve())})

    def test_manifest_checksums_require_exact_digest_and_one_newline(self):
        for directory in (self.package, self.store):
            path = directory / "manifest.sha256"
            original = path.read_bytes()
            for raw in (original[:-1], original + b"\n", b"0" * 64 + b"\n", original.upper()):
                with self.subTest(directory=directory.name, raw=raw[:10]):
                    path.write_bytes(raw)
                    with self.assertRaises(launcher.LauncherError):
                        self.defaults({})
            path.write_bytes(original)

    def test_symlink_or_nonregular_default_store_and_metadata_are_rejected(self):
        moved = self.store.with_name("linked-target")
        self.store.rename(moved)
        self.store.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(launcher.LauncherError):
            self.defaults({})
        self.store.unlink()
        moved.rename(self.store)
        for directory in (self.package, self.store):
            for name in ("manifest.json", "manifest.sha256"):
                path = directory / name
                target = self.root / (directory.name + "-" + name)
                path.rename(target)
                path.symlink_to(target)
                with self.subTest(directory=directory.name, name=name):
                    with self.assertRaises(launcher.LauncherError):
                        self.defaults({})
                path.unlink()
                target.rename(path)

    def test_missing_existing_store_or_aligned_metadata_raises(self):
        for directory in (self.package, self.store):
            for name in ("manifest.json", "manifest.sha256"):
                path = directory / name
                original = path.read_bytes()
                path.unlink()
                with self.subTest(directory=directory.name, name=name):
                    with self.assertRaises(launcher.LauncherError):
                        self.defaults({})
                path.write_bytes(original)

    def test_historical_v7_review_copy_leaves_active_44_defaults_unchanged(self):
        original = copy.deepcopy(launcher.LOCAL_PROFILE)
        with mock.patch.dict(os.environ, {"TASK_SENTINEL": "keep"}, clear=True):
            candidate = self.candidate({})
            self.assertEqual(candidate["profile"], "m5-ultra-flash-next-v7")
            self.assertIsNot(candidate, launcher.LOCAL_PROFILE)
            self.assertIsNot(candidate["environment"], launcher.LOCAL_PROFILE["environment"])
            self.assertEqual(candidate["environment"], {
                **V7_STATIC_ENVIRONMENT, STORE_ENV: str(self.store.resolve()),
            })
            self.assertEqual(len(V7_STATIC_ENVIRONMENT), 36)
            self.assertEqual(
                {k: v for k, v in candidate.items() if k not in ("profile", "environment")},
                {**{k: v for k, v in original.items() if k not in ("profile", "environment")}, "minimum_physical_ram_bytes": 192 * 1024**3},
            )
            candidate["environment"]["SPLASH_FLASH_MTP"] = "0"
            candidate["architecture"] = "changed"
            self.assertEqual(launcher.LOCAL_PROFILE, original)
            self.assertEqual(dict(os.environ), {"TASK_SENTINEL": "keep"})
        self.assertEqual(launcher.LOCAL_PROFILE["profile"], "m5-ultra-flash-next-v12")
        self.assertEqual(launcher.LOCAL_PROFILE["environment"], EXPECTED_ENVIRONMENT)
        self.assertEqual(len(launcher.LOCAL_PROFILE["environment"]), 44)

    def test_absent_saved_store_candidate_preserves_direct_a_default(self):
        self.store.rename(self.store.with_name("absent-original"))
        candidate = self.candidate({})
        self.assertEqual(candidate["environment"], V7_STATIC_ENVIRONMENT)
        self.assertNotIn(STORE_ENV, candidate["environment"])

    def test_explicit_store_key_prevents_candidate_default_and_survives_application(self):
        for value in ("", "0", "/tmp/explicit"):
            environment = {STORE_ENV: value}
            with self.subTest(value=value):
                candidate = self.candidate(environment)
                self.assertNotIn(STORE_ENV, candidate["environment"])
                defaults_before = copy.deepcopy(candidate["environment"])
                launcher._apply_local_profile_defaults(environment, candidate["environment"])
                self.assertEqual(environment[STORE_ENV], value)
                self.assertEqual(candidate["environment"], defaults_before)

    def test_disabling_either_direct_a_parent_disables_only_implied_default(self):
        defaults = {**EXPECTED_ENVIRONMENT, DIRECT_A: "1"}
        original_defaults = dict(defaults)
        for parent in ("SPLASH_FLASH_MOE_Q4X8", "SPLASH_FLASH_BLOCKED_MOE"):
            with self.subTest(parent=parent):
                environment = {parent: "0", "TASK_SENTINEL": "keep"}
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[parent], "0")
                self.assertEqual(environment[DIRECT_A], "0")
                self.assertEqual(environment["TASK_SENTINEL"], "keep")
                self.assertEqual(defaults, original_defaults)

    def test_explicit_direct_a_value_survives_disabled_parents(self):
        defaults = {**EXPECTED_ENVIRONMENT, DIRECT_A: "1"}
        for value in ("", "0", "1", "custom"):
            with self.subTest(value=value):
                environment = {
                    "SPLASH_FLASH_MOE_Q4X8": "0", "SPLASH_FLASH_BLOCKED_MOE": "0", DIRECT_A: value,
                }
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[DIRECT_A], value)
                self.assertEqual(environment["SPLASH_FLASH_MOE_Q4X8"], "0")
                self.assertEqual(environment["SPLASH_FLASH_BLOCKED_MOE"], "0")

    def test_direct_a_parent_values_other_than_explicit_zero_are_preserved(self):
        defaults = {**EXPECTED_ENVIRONMENT, DIRECT_A: "1"}
        for value in ("", "1", "custom"):
            for parent in ("SPLASH_FLASH_MOE_Q4X8", "SPLASH_FLASH_BLOCKED_MOE"):
                with self.subTest(parent=parent, value=value):
                    environment = {parent: value}
                    launcher._apply_local_profile_defaults(environment, defaults)
                    self.assertEqual(environment[parent], value)
                    self.assertEqual(environment[DIRECT_A], "1")

    def test_disabling_qsa_parent_disables_only_implied_row_tiles(self):
        for parent in ("SPLASH_FLASH_QSA_MPP", "SPLASH_FLASH_QSA_F32"):
            with self.subTest(parent=parent):
                defaults = dict(V5_STATIC_ENVIRONMENT)
                environment = {parent: "0"}
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[QSA_ROWS], "0")
                self.assertEqual(environment[parent], "0")
                self.assertEqual(defaults, V5_STATIC_ENVIRONMENT)

    def test_explicit_qsa_row_tiles_values_remain_authoritative(self):
        for value in ("", "0", "1", "custom"):
            with self.subTest(value=value):
                environment = {
                    "SPLASH_FLASH_QSA_MPP": "0", "SPLASH_FLASH_QSA_F32": "0", QSA_ROWS: value,
                }
                launcher._apply_local_profile_defaults(environment, V5_STATIC_ENVIRONMENT)
                self.assertEqual(environment[QSA_ROWS], value)

    def test_saved_residency_explicit_opt_out_survives_candidate_application(self):
        environment = {SAVED_RESIDENT: "0"}
        candidate = self.candidate(environment)
        launcher._apply_local_profile_defaults(environment, candidate["environment"])
        self.assertEqual(environment[SAVED_RESIDENT], "0")
        self.assertEqual(launcher.LOCAL_PROFILE["environment"], EXPECTED_ENVIRONMENT)

    def test_v6_candidate_is_private_and_does_not_mutate_profile_or_environment(self):
        before = copy.deepcopy(launcher.LOCAL_PROFILE)
        with mock.patch.dict(os.environ, {"TASK_SENTINEL": "keep"}, clear=True):
            candidate = launcher._local_profile_v6_candidate(self.package, {})
            second = launcher._local_profile_v6_candidate(self.package, {})
            self.assertEqual(candidate["profile"], "m5-ultra-flash-next-v6")
            self.assertEqual(candidate["environment"], {
                **V6_STATIC_ENVIRONMENT, STORE_ENV: str(self.store.resolve()),
            })
            self.assertIsNot(candidate, launcher.LOCAL_PROFILE)
            self.assertIsNot(candidate["environment"], launcher.LOCAL_PROFILE["environment"])
            self.assertIsNot(candidate["environment"], second["environment"])
            self.assertEqual(
                {k: v for k, v in candidate.items() if k not in ("profile", "environment")},
                {**{k: v for k, v in before.items() if k not in ("profile", "environment")}, "minimum_physical_ram_bytes": 192 * 1024**3},
            )
            candidate["environment"]["SPLASH_FLASH_MTP"] = "0"
            candidate["source_identity_sha256"] = "changed"
            self.assertEqual(second["environment"]["SPLASH_FLASH_MTP"], "1")
            self.assertEqual(launcher.LOCAL_PROFILE, before)
            self.assertEqual(dict(os.environ), {"TASK_SENTINEL": "keep"})
        self.assertEqual(launcher.LOCAL_PROFILE["profile"], "m5-ultra-flash-next-v12")
        self.assertEqual(len(launcher.LOCAL_PROFILE["environment"]), 44)
        historical = launcher._local_profile_v5_candidate(self.package, {STORE_ENV: "0"})
        self.assertEqual(historical["profile"], "m5-ultra-flash-next-v5")
        self.assertEqual(historical["environment"], V5_STATIC_ENVIRONMENT)
        self.assertEqual(len(historical["environment"]), 30)

    def test_v6_candidate_absent_saved_store_has_exact_34_static_defaults(self):
        self.store.rename(self.store.with_name("absent-original"))
        candidate = launcher._local_profile_v6_candidate(self.package, {})
        self.assertEqual(candidate["environment"], V6_STATIC_ENVIRONMENT)
        self.assertEqual(len(candidate["environment"]), 34)
        self.assertNotIn(STORE_ENV, candidate["environment"])
        self.assertNotIn("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT", candidate["environment"])

    def test_v6_candidate_preserves_saved_store_zero_before_inspection(self):
        environment = {STORE_ENV: "0", "TASK_SENTINEL": "keep"}
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("store inspected")):
            candidate = launcher._local_profile_v6_candidate(self.package, environment)
        self.assertEqual(candidate["environment"], V6_STATIC_ENVIRONMENT)
        launcher._apply_local_profile_defaults(environment, candidate["environment"])
        self.assertEqual(environment[STORE_ENV], "0")
        self.assertEqual(environment["TASK_SENTINEL"], "keep")

    def test_v6_candidate_keeps_top64_qualification_and_2048_row_windows(self):
        dense_before = copy.deepcopy(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION)
        int8_before = copy.deepcopy(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION)
        candidate = launcher._local_profile_v6_candidate(self.package, {})
        self.assertEqual(candidate["environment"]["SPLASH_FLASH_PREFILL_ROWS"], "2048")
        self.assertEqual(candidate["environment"]["SPLASH_FLASH_BATCH_PREFILL_ROWS"], "2048")
        self.assertEqual(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION, dense_before)
        self.assertEqual(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION, int8_before)
        self.assertEqual(int8_before["relative_path"], "install/local-models/Flash-Next-int8-experts-top64-v1")
        self.assertEqual(int8_before["selected_experts_per_layer"], 64)
        self.assertNotIn("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT", candidate["environment"])

    def test_v6_implied_children_follow_explicit_parent_opt_outs(self):
        for child, parents in V6_DEPENDENCIES.items():
            for parent in parents:
                with self.subTest(child=child, parent=parent):
                    defaults = dict(V6_STATIC_ENVIRONMENT)
                    environment = {parent: "0", "TASK_SENTINEL": "keep"}
                    launcher._apply_local_profile_defaults(environment, defaults)
                    self.assertEqual(environment[child], "0")
                    self.assertEqual(environment[parent], "0")
                    self.assertEqual(environment["TASK_SENTINEL"], "keep")
                    self.assertEqual(defaults, V6_STATIC_ENVIRONMENT)

    def test_v6_explicit_child_values_survive_disabled_parents(self):
        for child, parents in V6_DEPENDENCIES.items():
            for value in ("", "0", "1", "custom"):
                with self.subTest(child=child, value=value):
                    environment = {**{parent: "0" for parent in parents}, child: value}
                    launcher._apply_local_profile_defaults(environment, V6_STATIC_ENVIRONMENT)
                    self.assertEqual(environment[child], value)
                    for parent in parents:
                        self.assertEqual(environment[parent], "0")

    def test_v7_absent_saved_store_has_exact_36_static_defaults(self):
        self.store.rename(self.store.with_name("absent-original"))
        candidate = self.candidate({})
        self.assertEqual(candidate["environment"], V7_STATIC_ENVIRONMENT)
        self.assertEqual(len(candidate["environment"]), 36)
        self.assertNotIn(STORE_ENV, candidate["environment"])

    def test_v7_saved_store_zero_skips_inspection_and_survives_application(self):
        environment = {STORE_ENV: "0", "TASK_SENTINEL": "keep"}
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("store inspected")):
            candidate = self.candidate(environment)
        self.assertEqual(candidate["environment"], V7_STATIC_ENVIRONMENT)
        launcher._apply_local_profile_defaults(environment, candidate["environment"])
        self.assertEqual(environment[STORE_ENV], "0")
        self.assertEqual(environment["TASK_SENTINEL"], "keep")

    def test_v7_implied_children_follow_only_their_source_parent_opt_outs(self):
        for child, parents in V7_DEPENDENCIES.items():
            for parent in parents:
                with self.subTest(child=child, parent=parent):
                    defaults = dict(V7_STATIC_ENVIRONMENT)
                    environment = {parent: "0"}
                    launcher._apply_local_profile_defaults(environment, defaults)
                    self.assertEqual(environment[child], "0")
                    self.assertEqual(environment[parent], "0")
                    self.assertEqual(defaults, V7_STATIC_ENVIRONMENT)

    def test_v7_explicit_children_remain_authoritative_with_disabled_parents(self):
        for child, parents in V7_DEPENDENCIES.items():
            for value in ("", "0", "1", "custom"):
                with self.subTest(child=child, value=value):
                    environment = {**{parent: "0" for parent in parents}, child: value}
                    launcher._apply_local_profile_defaults(environment, V7_STATIC_ENVIRONMENT)
                    self.assertEqual(environment[child], value)
                    for parent in parents:
                        self.assertEqual(environment[parent], "0")

    def test_v7_does_not_invent_unrelated_route_prerequisites(self):
        for parent in (
            "SPLASH_FLASH_DENSE_CACHE", "SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_MTP",
            "SPLASH_FLASH_GDN_STAGED", "SPLASH_FLASH_BATCH", "SPLASH_FLASH_BATCH_PREFILL",
        ):
            with self.subTest(parent=parent):
                environment = {parent: "0"}
                launcher._apply_local_profile_defaults(environment, V7_STATIC_ENVIRONMENT)
                for child in V7_NEW_DEFAULTS:
                    self.assertEqual(environment[child], "1")

    def test_v7_preserves_pins_windows_and_excludes_other_experiments(self):
        pins = copy.deepcopy(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION)
        int8_pins = copy.deepcopy(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION)
        candidate = self.candidate({})
        self.assertEqual(candidate["environment"]["SPLASH_FLASH_PREFILL_ROWS"], "2048")
        self.assertEqual(candidate["environment"]["SPLASH_FLASH_BATCH_PREFILL_ROWS"], "2048")
        self.assertEqual(candidate["environment"]["SPLASH_FLASH_MTP_DRAFT_DEPTH"], "15")
        self.assertEqual(launcher.LOCAL_SAVED_OPERAND_QUALIFICATION, pins)
        self.assertEqual(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION, int8_pins)
        self.assertEqual(int8_pins["selected_experts_per_layer"], 64)
        for flag in ("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT", "SPLASH_FLASH_GPU_PREFILL_COPY"):
            self.assertNotIn(flag, candidate["environment"])

    @unittest.skipUnless(os.environ.get("FLASH_OPERAND_STORE_NATIVE_CHECKER"), "native CPU checker not configured")
    def test_native_exact_zero_saved_store_selectors_opt_out_without_weights_or_gpu(self):
        environment = {
            **os.environ, STORE_ENV: "0", "SPLASH_FLASH_INT8_EXPERT_STORE": "0",
        }
        environment.pop("SPLASH_FLASH_HOT_EXPERT_PLAN", None)
        result = subprocess.run(
            [os.environ["FLASH_OPERAND_STORE_NATIVE_CHECKER"], "--cpu-self-test"],
            capture_output=True, text=True, timeout=15, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exact-zero selector opt-out: PASS", result.stdout)

    def test_payload_corruption_does_not_make_launcher_claim_payload_verification(self):
        # The pinned manifest still qualifies. Native mapping must reject the
        # corrupted payload; the launcher deliberately avoids hashing GBs.
        path = self.store / "bf16.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        self.assertEqual(self.defaults({}), {STORE_ENV: str(self.store.resolve())})
        with self.assertRaises(operands.StoreError):
            operands.verify_store(
                self.store, source_identity=self.pins["source_identity_sha256"],
                weights_fingerprint=self.pins["weights_manifest_fingerprint"],
            )

    def test_normal_local_serve_loads_qualified_defaults_without_using_review_candidate(self):
        active_profile = copy.deepcopy(launcher.LOCAL_PROFILE)
        qualified_profile = self.prepare_qualified_profile()
        with (
            mock.patch.object(launcher, "LOCAL_PROFILE", qualified_profile),
            mock.patch.object(launcher, "RUNTIME_DIR", self.root / "runtime"),
            mock.patch.object(launcher, "_ensure_local_runtime"),
            mock.patch.object(launcher, "_ensure_local_installed", return_value=self.package),
            mock.patch.object(launcher, "local_bundle_manifest", return_value=self.aligned),
            mock.patch.object(launcher, "_local_hardware_identity", return_value=QUALIFIED_HARDWARE) as hardware,
            mock.patch.object(
                launcher, "_qualified_saved_operand_defaults",
                wraps=launcher._qualified_saved_operand_defaults,
            ) as saved,
            mock.patch.object(
                launcher, "_local_profile_v5_candidate", side_effect=AssertionError("v5 activated"),
            ) as candidate,
            mock.patch.object(
                launcher, "_local_profile_v6_candidate", side_effect=AssertionError("v6 activated"),
            ) as candidate6,
            mock.patch.object(
                launcher, "_local_profile_v7_candidate", side_effect=AssertionError("v7 activated"),
            ) as candidate7,
            mock.patch.object(
                launcher, "_local_profile_v8_candidate", side_effect=AssertionError("v8 review copy activated"),
            ) as candidate8,
            mock.patch.object(
                launcher, "_local_profile_v9_candidate", side_effect=AssertionError("v9 review copy activated"),
            ) as candidate9,
            mock.patch.object(
                launcher, "_local_profile_v10_candidate", side_effect=AssertionError("v10 review copy activated"),
            ) as candidate10,
            mock.patch.object(
                launcher, "_qualified_saved_int8_expert_defaults",
                wraps=launcher._qualified_saved_int8_expert_defaults,
            ) as int8,
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch.dict(os.environ, {"TASK_SENTINEL": "keep"}, clear=True),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            launcher.main([
                "serve", "--model", "local/Flash-Next", "--local-package",
                str(self.package), "--port", "8011",
            ])
            execute.assert_called_once()
            environment = execute.call_args.args[2]
            self.assertEqual(environment[DIRECT_A], "1")
            self.assertEqual(environment[QSA_ROWS], "1")
            self.assertEqual(environment[SAVED_RESIDENT], "1")
            self.assertEqual(environment[STORE_ENV], str(self.store.resolve()))
            self.assertNotIn("SPLASH_FLASH_INT8_EXPERT_STORE", environment)
            for key, value in V6_NEW_DEFAULTS.items():
                self.assertEqual(environment[key], value)
            for key, value in V7_NEW_DEFAULTS.items():
                self.assertEqual(environment[key], value)
            for key, value in V8_NEW_DEFAULTS.items():
                self.assertEqual(environment[key], value)
            for key, value in V9_NEW_DEFAULTS.items():
                self.assertEqual(environment[key], value)
            for key, value in V10_NEW_DEFAULTS.items():
                self.assertEqual(environment[key], value)
            self.assertEqual(dict(os.environ), {"TASK_SENTINEL": "keep"})
            saved.assert_called_once()
            candidate.assert_not_called()
            candidate6.assert_not_called()
            candidate7.assert_not_called()
            candidate8.assert_not_called()
            candidate9.assert_not_called()
            candidate10.assert_not_called()
            int8.assert_called_once()
            self.assertEqual(int8.call_args.kwargs["hardware"], QUALIFIED_HARDWARE)
            hardware.assert_called_once_with()
        self.assertEqual(launcher.LOCAL_PROFILE, active_profile)


INT8_STORE_ENV = "SPLASH_FLASH_INT8_EXPERT_STORE"
QUALIFIED_HARDWARE = ("Apple M5 Ultra", 256 * 1024**3)


class SavedInt8ExpertProfileTests(_SavedArtifactFixture):
    def setUp(self):
        super().setUp()
        self.production_int8_pins = copy.deepcopy(launcher.LOCAL_SAVED_INT8_EXPERT_QUALIFICATION)
        self.int8_store = self.root / self.production_int8_pins["relative_path"]
        self.int8_store.mkdir(parents=True)
        layers = []
        for index in range(48):
            projections, offset = {}, 0
            for role, (n, k) in (
                ("gate_proj", (640, 2560)), ("up_proj", (640, 2560)), ("down_proj", (2560, 640))
            ):
                planes = {}
                for name, dtype, shape, length in (
                    ("codes", "I8", [64, n, k], 64 * n * k),
                    ("scales", "F32", [64, n], 64 * n * 4),
                ):
                    offset = (offset + 16383) // 16384 * 16384
                    planes[name] = {
                        "dtype": dtype, "shape": shape, "offset": offset,
                        "length": length, "sha256": "d" * 64,
                    }
                    offset += length
                projections[role] = {
                    "source_prefix": f"language_model.model.layers.{index}.mlp.switch_mlp.{role}",
                    "dimensions": [64, n, k], **planes,
                }
            layers.append({
                "layer_index": index, "path": f"layer-{index:02d}.bin",
                "bytes": (offset + 16383) // 16384 * 16384,
                "sha256": "d" * 64, "projections": projections,
            })
        self.int8_manifest = {
            "schema": "splash-flash-int8-expert-store-v1",
            "source_identity_sha256": self.pins["source_identity_sha256"],
            "source_manifest_sha256": self.pins["aligned_manifest_sha256"],
            "plan_sha256": self.production_int8_pins["plan_sha256"],
            "alignment": 16384, "target_layers": 48,
            "selected_experts": [list(range(64)) for _ in range(48)],
            "coefficient_policy": "source_q4_g64_f32_separate_multiply_add_then_bf16_rne_v1",
            "quantization_format": "signed-symmetric-int8-rowwise-f32-scale",
            "integer_rounding": "F32 absmax/127; F32 division; nearest-even integer; clamp [-127,127]; zero row scale=1",
            "layers": layers, "total_bytes": sum(layer["bytes"] for layer in layers),
            "planned_allocation_bytes": sum(layer["bytes"] for layer in layers) + 48 * 16384,
        }
        (self.int8_store / "manifest.json").write_bytes(operands.canonical(self.int8_manifest))
        self.int8_pins = {
            **self.production_int8_pins,
            "manifest_sha256": self.digest(self.int8_store / "manifest.json"),
        }
        patch = mock.patch.object(launcher, "LOCAL_SAVED_INT8_EXPERT_QUALIFICATION", self.int8_pins)
        patch.start()
        self.addCleanup(patch.stop)

    def int8_defaults(self, environment=None, hardware=QUALIFIED_HARDWARE):
        return launcher._qualified_saved_int8_expert_defaults(
            self.package, environment=environment, hardware=hardware
        )

    def rewrite_int8(self, manifest, *, repin=False, raw=None):
        (self.int8_store / "manifest.json").write_bytes(
            operands.canonical(manifest) if raw is None else raw
        )
        if repin:
            self.int8_pins["manifest_sha256"] = self.digest(self.int8_store / "manifest.json")

    def test_production_int8_pins_and_256gib_floor_are_exact(self):
        self.assertEqual(self.production_int8_pins, {
            "relative_path": "install/local-models/Flash-Next-int8-experts-top64-v1",
            "manifest_sha256": "12593570ee67b62ddeadb951238879b780368cf99e096aa38a7d7771b9dc5c29",
            "plan_sha256": "d0b1f58c87eb4292ea6e7d04eec55f0c722ee7583468dd58e45c2fa3f476d02d",
            "minimum_physical_ram_bytes": 256 * 1024**3,
            "selected_experts_per_layer": 64,
        })

    def test_qualified_int8_metadata_returns_private_absolute_path_without_payload_files(self):
        self.assertEqual(self.int8_defaults({}), {INT8_STORE_ENV: str(self.int8_store.resolve())})
        self.assertEqual([path.name for path in self.int8_store.iterdir()], ["manifest.json"])
        first, second = self.int8_defaults({}), self.int8_defaults({})
        self.assertIsNot(first, second)
        first[INT8_STORE_ENV] = "changed"
        self.assertEqual(second, {INT8_STORE_ENV: str(self.int8_store.resolve())})

    def test_explicit_int8_store_returns_before_hardware_or_file_inspection(self):
        for value in ("", "0", "relative/custom", "/tmp/int8-custom"):
            environment = {INT8_STORE_ENV: value, "SPLASH_FLASH_BLOCKED_MOE": "0"}
            before = dict(environment)
            with self.subTest(value=value), mock.patch.object(
                launcher, "_local_hardware_identity", side_effect=AssertionError("hardware probed")
            ), mock.patch.object(Path, "read_bytes", side_effect=AssertionError("manifest inspected")):
                self.assertEqual(self.int8_defaults(environment), {})
            self.assertEqual(environment, before)

    def test_absent_int8_store_has_no_default_or_metadata_work(self):
        self.int8_store.rename(self.int8_store.with_name("absent-int8"))
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("manifest inspected")):
            self.assertEqual(self.int8_defaults({}), {})

    def test_top128_store_never_becomes_an_implied_default(self):
        self.int8_store.rename(self.int8_store.with_name("absent-top64"))
        old = self.root / "install/local-models/Flash-Next-int8-experts-top128-v1"
        old.mkdir()
        (old / "manifest.json").write_bytes(b"{")
        self.assertEqual(self.int8_defaults({}), {})

    def test_explicit_top128_path_still_survives_as_manual_override(self):
        path = str(self.root / "install/local-models/Flash-Next-int8-experts-top128-v1")
        environment = {INT8_STORE_ENV: path}
        self.assertEqual(self.int8_defaults(environment), {})
        launcher._apply_local_profile_defaults(
            environment, {**V5_STATIC_ENVIRONMENT, INT8_STORE_ENV: str(self.int8_store.resolve())}
        )
        self.assertEqual(environment[INT8_STORE_ENV], path)

    def test_int8_default_does_not_depend_on_dense_artifact_presence(self):
        self.store.rename(self.store.with_name("absent-dense"))
        self.assertEqual(self.int8_defaults({}), {INT8_STORE_ENV: str(self.int8_store.resolve())})

    def test_provided_hardware_skips_probe_and_default_hardware_probes_once(self):
        with mock.patch.object(launcher, "_local_hardware_identity", side_effect=AssertionError("hardware probed")):
            self.assertEqual(self.int8_defaults({}), {INT8_STORE_ENV: str(self.int8_store.resolve())})
        with mock.patch.object(launcher, "_local_hardware_identity", return_value=QUALIFIED_HARDWARE) as probe:
            self.assertEqual(
                launcher._qualified_saved_int8_expert_defaults(self.package, {}),
                {INT8_STORE_ENV: str(self.int8_store.resolve())},
            )
            probe.assert_called_once_with()

    def test_hardware_floor_brand_and_ram_type_are_strict(self):
        for hardware in (
            None, ("Apple M5 Max", 256 * 1024**3), ("Apple M5 Ultra", 256 * 1024**3 - 1),
            ("Apple M5 Ultra", 192 * 1024**3), ("Apple M5 Ultra", float(256 * 1024**3)),
            ("Apple M5 Ultra", str(256 * 1024**3)), ("Apple M5 Ultra", True),
            ("Apple M5 Ultra", False), ("Apple M5 Ultra", -1),
        ):
            with self.subTest(hardware=hardware):
                self.assertEqual(self.int8_defaults({}, hardware=hardware), {})
        self.assertEqual(
            self.int8_defaults({}, hardware=("Apple M5 Ultra", 512 * 1024**3)),
            {INT8_STORE_ENV: str(self.int8_store.resolve())},
        )

    def test_unavailable_probe_returns_no_default(self):
        with mock.patch.object(launcher, "_local_hardware_identity", return_value=None) as probe:
            self.assertEqual(launcher._qualified_saved_int8_expert_defaults(self.package, {}), {})
            probe.assert_called_once_with()

    def test_effective_blocked_moe_must_be_enabled(self):
        for value in ("0", "", "custom"):
            with self.subTest(value=value):
                self.assertEqual(self.int8_defaults({"SPLASH_FLASH_BLOCKED_MOE": value}), {})
        self.assertEqual(
            self.int8_defaults({"SPLASH_FLASH_BLOCKED_MOE": "1"}),
            {INT8_STORE_ENV: str(self.int8_store.resolve())},
        )

    def test_disabled_hardware_or_parent_avoids_inspecting_corrupt_artifact(self):
        self.rewrite_int8(self.int8_manifest, raw=b"{")
        self.assertEqual(self.int8_defaults({}, hardware=("Apple M5 Max", 256 * 1024**3)), {})
        self.assertEqual(self.int8_defaults({"SPLASH_FLASH_BLOCKED_MOE": "0"}), {})
        with self.assertRaises(launcher.LauncherError):
            self.int8_defaults({})

    def test_corrupt_existing_int8_manifest_bytes_raise(self):
        for raw in (b"{", b"null", b"[]", b"\xff", b""):
            with self.subTest(raw=raw):
                self.rewrite_int8(self.int8_manifest, raw=raw)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})

    def test_int8_schema_source_layout_plan_and_byte_digest_are_bound(self):
        for field, value in (
            ("schema", "other-schema"), ("source_identity_sha256", "d" * 64),
            ("source_manifest_sha256", "d" * 64), ("plan_sha256", "d" * 64),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.int8_manifest)
                changed[field] = value
                self.rewrite_int8(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})
        self.rewrite_int8(self.int8_manifest)
        self.int8_pins["manifest_sha256"] = "d" * 64
        with self.assertRaises(launcher.LauncherError):
            self.int8_defaults({})

    def test_int8_inventory_requires_48_layers_and_48_selected_lists(self):
        for field in ("layers", "selected_experts"):
            for count in (0, 47, 49):
                with self.subTest(field=field, count=count):
                    changed = copy.deepcopy(self.int8_manifest)
                    changed[field] = (
                        changed[field][:count] if count < 48 else changed[field] + [copy.deepcopy(changed[field][-1])]
                    )
                    self.rewrite_int8(changed, repin=True)
                    with self.assertRaises(launcher.LauncherError):
                        self.int8_defaults({})
        for value in (0, 47, 49, 48.0, True, "48"):
            with self.subTest(target_layers=value):
                changed = copy.deepcopy(self.int8_manifest)
                changed["target_layers"] = value
                self.rewrite_int8(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})
        for value in (0, 8192, 16384.0, True, "16384"):
            with self.subTest(alignment=value):
                changed = copy.deepcopy(self.int8_manifest)
                changed["alignment"] = value
                self.rewrite_int8(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})

    def test_int8_expert_selection_requires_exact_64_sorted_unique_integer_ids(self):
        malformed = [
            [], list(range(63)), list(range(65)), list(range(128)), list(reversed(range(64))),
            [0, 0] + list(range(2, 64)), [-1] + list(range(1, 64)),
            list(range(63)) + [512], [False] + list(range(1, 64)),
            [0, True] + list(range(2, 64)), [0.0] + list(range(1, 64)),
            ["0"] + list(range(1, 64)), None,
        ]
        for ids in malformed:
            with self.subTest(ids_type=type(ids).__name__, ids_count=len(ids) if isinstance(ids, list) else None):
                changed = copy.deepcopy(self.int8_manifest)
                changed["selected_experts"][7] = ids
                self.rewrite_int8(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})

    def test_int8_projection_shapes_types_and_source_prefixes_are_exact(self):
        for role in ("gate_proj", "up_proj", "down_proj"):
            for field, value in (
                ("source_prefix", f"language_model.model.layers.8.mlp.switch_mlp.{role}"),
                ("dimensions", [64, 640, 640]),
            ):
                with self.subTest(role=role, field=field):
                    changed = copy.deepcopy(self.int8_manifest)
                    changed["layers"][7]["projections"][role][field] = value
                    self.rewrite_int8(changed, repin=True)
                    with self.assertRaises(launcher.LauncherError):
                        self.int8_defaults({})
            for plane in ("codes", "scales"):
                for mutation in ("shape", "float", "dtype"):
                    changed = copy.deepcopy(self.int8_manifest)
                    target = changed["layers"][7]["projections"][role][plane]
                    if mutation == "shape":
                        target["shape"][0] = 63
                    elif mutation == "float":
                        target["shape"][1] = float(target["shape"][1])
                    else:
                        target["dtype"] = "BF16"
                    with self.subTest(role=role, plane=plane, mutation=mutation):
                        self.rewrite_int8(changed, repin=True)
                        with self.assertRaises(launcher.LauncherError):
                            self.int8_defaults({})
        for value in (True, 1.0, "1"):
            changed = copy.deepcopy(self.int8_manifest)
            changed["layers"][1]["layer_index"] = value
            with self.subTest(layer_index=value):
                self.rewrite_int8(changed, repin=True)
                with self.assertRaises(launcher.LauncherError):
                    self.int8_defaults({})

    def test_int8_default_symlink_manifest_and_store_are_rejected(self):
        target = self.root / "int8-manifest.json"
        path = self.int8_store / "manifest.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(launcher.LauncherError):
            self.int8_defaults({})
        path.unlink()
        target.rename(path)
        moved = self.int8_store.with_name("int8-symlink-target")
        self.int8_store.rename(moved)
        self.int8_store.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(launcher.LauncherError):
            self.int8_defaults({})

    def test_disabling_blocked_moe_removes_implied_path_and_preserves_explicit_path(self):
        path = str(self.int8_store.resolve())
        defaults = {**EXPECTED_ENVIRONMENT, INT8_STORE_ENV: path}
        original = dict(defaults)
        environment = {"SPLASH_FLASH_BLOCKED_MOE": "0"}
        launcher._apply_local_profile_defaults(environment, defaults)
        self.assertNotIn(INT8_STORE_ENV, environment)
        self.assertEqual(defaults, original)
        for value in ("", "0", "/tmp/explicit-int8"):
            with self.subTest(value=value):
                environment = {"SPLASH_FLASH_BLOCKED_MOE": "0", INT8_STORE_ENV: value}
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[INT8_STORE_ENV], value)
                self.assertEqual(defaults, original)

    def test_int8_helper_never_mutates_process_environment_or_caller(self):
        environment = {"TASK_SENTINEL": "keep"}
        with mock.patch.dict(os.environ, {INT8_STORE_ENV: "/tmp/process-path"}, clear=True):
            self.assertEqual(self.int8_defaults(environment), {INT8_STORE_ENV: str(self.int8_store.resolve())})
            self.assertEqual(self.int8_defaults(None), {})
            self.assertEqual(dict(os.environ), {INT8_STORE_ENV: "/tmp/process-path"})
        self.assertEqual(environment, {"TASK_SENTINEL": "keep"})

    def test_v7_review_copy_excludes_dynamic_int8_expert_path(self):
        with mock.patch.object(
            launcher, "_qualified_saved_int8_expert_defaults", side_effect=AssertionError("INT8 activated")
        ) as helper:
            candidate = self.candidate({})
            self.assertNotIn(INT8_STORE_ENV, candidate["environment"])
            helper.assert_not_called()
        self.assertNotIn(INT8_STORE_ENV, launcher.LOCAL_PROFILE["environment"])

    def test_active_qualified_profile_combines_optional_stores_and_preserves_opt_outs(self):
        qualified_profile = self.prepare_qualified_profile()
        dense_path = str(self.store.resolve())
        int8_path = str(self.int8_store.resolve())
        cases = (
            ({}, QUALIFIED_HARDWARE, {STORE_ENV: dense_path, INT8_STORE_ENV: int8_path}),
            ({STORE_ENV: "0"}, QUALIFIED_HARDWARE, {INT8_STORE_ENV: int8_path}),
            ({INT8_STORE_ENV: "0"}, QUALIFIED_HARDWARE, {STORE_ENV: dense_path}),
            ({STORE_ENV: "0", INT8_STORE_ENV: "0"}, QUALIFIED_HARDWARE, {}),
            ({"SPLASH_FLASH_BLOCKED_MOE": "0"}, QUALIFIED_HARDWARE, {STORE_ENV: dense_path}),
        )
        for caller, machine, optional in cases:
            original = {"TASK_SENTINEL": "keep", **caller}
            with (
                self.subTest(caller=caller, machine=machine),
                mock.patch.object(launcher, "LOCAL_PROFILE", qualified_profile),
                mock.patch.object(launcher, "local_bundle_manifest", return_value=self.aligned),
                mock.patch.object(launcher, "_local_hardware_identity", return_value=machine) as hardware,
                mock.patch.object(
                    launcher, "_qualified_saved_operand_defaults",
                    wraps=launcher._qualified_saved_operand_defaults,
                ) as dense,
                mock.patch.object(
                    launcher, "_qualified_saved_int8_expert_defaults",
                    wraps=launcher._qualified_saved_int8_expert_defaults,
                ) as int8,
                mock.patch.dict(os.environ, original, clear=True),
            ):
                defaults = launcher._local_profile_defaults(self.package)
                self.assertEqual(defaults, {**EXPECTED_ENVIRONMENT, **optional})
                self.assertIsNot(defaults, qualified_profile["environment"])
                hardware.assert_called_once_with()
                dense.assert_called_once()
                int8.assert_called_once()
                self.assertEqual(int8.call_args.kwargs["hardware"], machine)
                applied = dict(original)
                launcher._apply_local_profile_defaults(applied, defaults)
                for key, value in caller.items():
                    self.assertEqual(applied[key], value)
                if caller.get("SPLASH_FLASH_BLOCKED_MOE") == "0":
                    self.assertNotIn(INT8_STORE_ENV, applied)
                    self.assertEqual(applied[DIRECT_A], "0")
                self.assertEqual(dict(os.environ), original)
                self.assertEqual(qualified_profile["environment"], EXPECTED_ENVIRONMENT)

        # A present corrupt sidecar must fail the real active-profile path.
        for corrupted in (self.store, self.int8_store):
            path = corrupted / "manifest.json"
            original = path.read_bytes()
            path.write_bytes(b"{")
            with (
                self.subTest(corrupt=corrupted.name),
                mock.patch.object(launcher, "LOCAL_PROFILE", qualified_profile),
                mock.patch.object(launcher, "local_bundle_manifest", return_value=self.aligned),
                mock.patch.object(launcher, "_local_hardware_identity", return_value=QUALIFIED_HARDWARE),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaises(launcher.LauncherError):
                    launcher._local_profile_defaults(self.package)
            path.write_bytes(original)

        # The source/hardware profile gate runs before optional artifact checks.
        for bundle, machine in (
            ({**self.aligned, "source_identity_sha256": "d" * 64}, QUALIFIED_HARDWARE),
            (self.aligned, ("Apple M5 Max", 256 * 1024**3)),
            (self.aligned, ("Apple M5 Ultra", 192 * 1024**3)),
            (self.aligned, ("Apple M5 Ultra", 255 * 1024**3)),
        ):
            with (
                self.subTest(bundle_source=bundle["source_identity_sha256"], machine=machine),
                mock.patch.object(launcher, "LOCAL_PROFILE", qualified_profile),
                mock.patch.object(launcher, "local_bundle_manifest", return_value=bundle),
                mock.patch.object(launcher, "_local_hardware_identity", return_value=machine),
                mock.patch.object(launcher, "_qualified_saved_operand_defaults") as dense,
                mock.patch.object(launcher, "_qualified_saved_int8_expert_defaults") as int8,
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(launcher._local_profile_defaults(self.package), {})
                dense.assert_not_called()
                int8.assert_not_called()

        # Missing optional artifacts retain all qualified static defaults.
        self.store.rename(self.store.with_name("absent-dense"))
        self.int8_store.rename(self.int8_store.with_name("absent-top64"))
        with (
            mock.patch.object(launcher, "LOCAL_PROFILE", qualified_profile),
            mock.patch.object(launcher, "local_bundle_manifest", return_value=self.aligned),
            mock.patch.object(launcher, "_local_hardware_identity", return_value=QUALIFIED_HARDWARE) as hardware,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(launcher._local_profile_defaults(self.package), EXPECTED_ENVIRONMENT)
            hardware.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
