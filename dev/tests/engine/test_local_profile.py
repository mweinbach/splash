import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher


V7_ENVIRONMENT = {
    "SPLASH_FLASH_QMV_F32": "1",
    "SPLASH_FLASH_FUSE_HC": "1",
    "SPLASH_FLASH_FUSE_GDN": "1",
    "SPLASH_FLASH_QSA_F32": "1",
    "SPLASH_FLASH_DENSE_CACHE": "1",
    "SPLASH_FLASH_BLOCKED_MOE": "1",
    "SPLASH_FLASH_PREFILL_ROWS": "2048",
    "SPLASH_FLASH_BATCH": "1",
    "SPLASH_FLASH_MTP": "1",
    "SPLASH_FLASH_EXPERT_QMV": "1",
    "SPLASH_FLASH_GDN_STAGED": "1",
    "SPLASH_FLASH_QSA_MPP": "1",
    "SPLASH_FLASH_MTP_QSA_F32": "1",
    "SPLASH_FLASH_MTP_QSA_MPP": "1",
    "SPLASH_FLASH_FLOAT_DENSE_CACHE": "1",
    "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE": "1",
    "SPLASH_FLASH_MOE_Q4X8": "1",
    "SPLASH_FLASH_MOE_M64": "1",
    "SPLASH_FLASH_BATCH_MTP": "1",
    "SPLASH_FLASH_BATCH_MTP_PREFILL": "1",
    "SPLASH_FLASH_BATCH_PREFILL": "1",
    "SPLASH_FLASH_BATCH_PREFILL_ROWS": "2048",
    "SPLASH_FLASH_MTP_DRAFT_DEPTH": "15",
    "SPLASH_FLASH_PLE_LOOKUP_FUSED": "1",
    "SPLASH_FLASH_PLE_POST_FUSED": "1",
    "SPLASH_FLASH_GPU_GREEDY": "1",
    "SPLASH_FLASH_INT8_HEAD": "1",
    "SPLASH_FLASH_MOE_DIRECT_A": "1",
    "SPLASH_FLASH_QSA_ROW_TILES": "1",
    "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT": "1",
    "SPLASH_FLASH_SHARED_EXPERT_FUSED": "1",
    "SPLASH_FLASH_DENSE_M64_OUT": "1",
    "SPLASH_FLASH_GDN_BATCH_ILP": "1",
    "SPLASH_FLASH_MTP_QMV_F32": "1",
    "SPLASH_FLASH_HC_UP_F32_MPP": "1",
    "SPLASH_FLASH_GDN_LAZY_ROLLBACK": "1",
}
V8_ENVIRONMENT = {
    **V7_ENVIRONMENT,
    "SPLASH_FLASH_MTP_Q8_BF16_REGISTER": "1",
}
V9_ENVIRONMENT = {
    **V8_ENVIRONMENT,
    "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "1",
}
EXPECTED_ENVIRONMENT = {
    **V9_ENVIRONMENT,
    "SPLASH_FLASH_PLE_SSD_STREAMING": "1",
}
V6_ENVIRONMENT = {
    key: value
    for key, value in V7_ENVIRONMENT.items()
    if key not in ("SPLASH_FLASH_HC_UP_F32_MPP", "SPLASH_FLASH_GDN_LAZY_ROLLBACK")
}
V5_ENVIRONMENT = {
    key: value
    for key, value in V6_ENVIRONMENT.items()
    if key not in (
        "SPLASH_FLASH_SHARED_EXPERT_FUSED",
        "SPLASH_FLASH_DENSE_M64_OUT",
        "SPLASH_FLASH_GDN_BATCH_ILP",
        "SPLASH_FLASH_MTP_QMV_F32",
    )
}
V4_ENVIRONMENT = {
    key: value
    for key, value in V5_ENVIRONMENT.items()
    if key not in (
        "SPLASH_FLASH_MOE_DIRECT_A",
        "SPLASH_FLASH_QSA_ROW_TILES",
        "SPLASH_FLASH_SAVED_OPERANDS_RESIDENT",
    )
}
V3_ENVIRONMENT = {
    key: value
    for key, value in V4_ENVIRONMENT.items()
    if key != "SPLASH_FLASH_INT8_HEAD"
}
V2_ENVIRONMENT = {
    key: value
    for key, value in V3_ENVIRONMENT.items()
    if key
    not in (
        "SPLASH_FLASH_PLE_LOOKUP_FUSED",
        "SPLASH_FLASH_PLE_POST_FUSED",
        "SPLASH_FLASH_GPU_GREEDY",
    )
}
V1_ENVIRONMENT = {
    key: EXPECTED_ENVIRONMENT[key]
    for key in (
        "SPLASH_FLASH_QMV_F32",
        "SPLASH_FLASH_FUSE_HC",
        "SPLASH_FLASH_FUSE_GDN",
        "SPLASH_FLASH_QSA_F32",
        "SPLASH_FLASH_DENSE_CACHE",
        "SPLASH_FLASH_BLOCKED_MOE",
        "SPLASH_FLASH_PREFILL_ROWS",
        "SPLASH_FLASH_BATCH",
        "SPLASH_FLASH_MTP",
    )
}


class LocalProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        (self.package / "config.json").write_text('{"model_type":"qwen4_exp"}')
        self.manifest = {
            "schema": launcher.LOCAL_SCHEMA,
            "source_identity_sha256": launcher.LOCAL_PROFILE["source_identity_sha256"],
        }
        self.profile_path = self.root / ".splash-local-profile.json"
        self.write_profile()

    def write_profile(self, profile=None):
        self.profile_path.write_text(
            json.dumps(launcher.LOCAL_PROFILE if profile is None else profile)
        )

    def defaults(self, hardware=("Apple M5 Ultra", 256 * 1024**3), manifest=None):
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(
                launcher,
                "local_bundle_manifest",
                return_value=self.manifest if manifest is None else manifest,
            ),
            mock.patch.object(
                launcher, "_local_hardware_identity", return_value=hardware
            ),
        ):
            return launcher._local_profile_defaults(self.package)

    def exec_environment(self, original=None, defaults=None):
        original = {} if original is None else original
        defaults = dict(EXPECTED_ENVIRONMENT) if defaults is None else defaults
        defaults_before = copy.deepcopy(defaults)
        profile_before = copy.deepcopy(launcher.LOCAL_PROFILE)
        with (
            mock.patch.object(launcher, "ROOT", self.root),
            mock.patch.object(launcher, "RUNTIME_DIR", self.root / "runtime"),
            mock.patch.object(launcher, "_ensure_local_runtime") as build,
            mock.patch.object(
                launcher, "_ensure_local_installed", return_value=self.package
            ),
            mock.patch.object(
                launcher, "_local_profile_defaults", return_value=defaults
            ),
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch.object(launcher.catalog, "spawn_refresh") as refresh,
            mock.patch.dict(os.environ, original, clear=True),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            launcher.main(
                [
                    "serve",
                    "--model",
                    "local/Flash-Next",
                    "--local-package",
                    str(self.package),
                    "--port",
                    "8011",
                ]
            )
            execute.assert_called_once()
            environment = execute.call_args.args[2]
            self.assertEqual(dict(os.environ), original)
            self.assertEqual(defaults, defaults_before)
            self.assertEqual(launcher.LOCAL_PROFILE, profile_before)
        build.assert_called_once_with(self.root / "build/flash-next/splash-flash")
        refresh.assert_not_called()
        return environment

    def test_v10_profile_has_exact_metadata_and_39_static_defaults(self):
        self.assertEqual(
            {key: value for key, value in launcher.LOCAL_PROFILE.items() if key != "environment"},
            {
                "schema_version": 1,
                "profile": "m5-ultra-flash-next-v10",
                "source_identity_sha256": "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
                "architecture": "qwen4_exp",
                "cpu_brand": "Apple M5 Ultra",
                "minimum_physical_ram_bytes": 256 * 1024**3,
            },
        )
        self.assertEqual(len(EXPECTED_ENVIRONMENT), 39)
        self.assertEqual(list(EXPECTED_ENVIRONMENT.values()).count("1"), 36)
        self.assertEqual(launcher.LOCAL_PROFILE["environment"], EXPECTED_ENVIRONMENT)
        self.assertNotIn("SPLASH_FLASH_GPU_PREFILL_COPY", launcher.LOCAL_PROFILE["environment"])

    def test_qualified_profile_matches_exact_hardware_floor_and_returns_private_defaults(
        self,
    ):
        for ram in (256 * 1024**3, 512 * 1024**3):
            with self.subTest(ram=ram):
                defaults = self.defaults(("Apple M5 Ultra", ram))
                self.assertEqual(defaults, EXPECTED_ENVIRONMENT)
                self.assertIsNot(defaults, launcher.LOCAL_PROFILE["environment"])
                other_defaults = self.defaults(("Apple M5 Ultra", ram))
                self.assertIsNot(defaults, other_defaults)
                defaults["SPLASH_FLASH_MTP"] = "0"
                self.assertEqual(other_defaults, EXPECTED_ENVIRONMENT)
                self.assertEqual(
                    launcher.LOCAL_PROFILE["environment"]["SPLASH_FLASH_MTP"], "1"
                )

    def test_saved_v1_through_v9_profiles_never_apply_partial_defaults(self):
        for name, environment in (
            ("m5-ultra-flash-next-v1", EXPECTED_ENVIRONMENT),
            ("m5-ultra-flash-next-v1", V1_ENVIRONMENT),
            ("m5-ultra-flash-next-v2", V1_ENVIRONMENT),
            ("m5-ultra-flash-next-v4", V1_ENVIRONMENT),
            ("m5-ultra-flash-next-v2", EXPECTED_ENVIRONMENT),
            ("m5-ultra-flash-next-v2", V2_ENVIRONMENT),
            ("m5-ultra-flash-next-v4", V2_ENVIRONMENT),
            ("m5-ultra-flash-next-v3", EXPECTED_ENVIRONMENT),
            ("m5-ultra-flash-next-v3", V3_ENVIRONMENT),
            ("m5-ultra-flash-next-v4", V3_ENVIRONMENT),
            ("m5-ultra-flash-next-v4", V4_ENVIRONMENT),
            ("m5-ultra-flash-next-v5", V4_ENVIRONMENT),
            ("m5-ultra-flash-next-v5", V5_ENVIRONMENT),
            ("m5-ultra-flash-next-v6", V5_ENVIRONMENT),
            ("m5-ultra-flash-next-v6", V6_ENVIRONMENT),
            ("m5-ultra-flash-next-v7", V6_ENVIRONMENT),
            ("m5-ultra-flash-next-v7", V7_ENVIRONMENT),
            ("m5-ultra-flash-next-v8", V7_ENVIRONMENT),
            ("m5-ultra-flash-next-v8", V8_ENVIRONMENT),
            ("m5-ultra-flash-next-v9", V8_ENVIRONMENT),
            ("m5-ultra-flash-next-v9", V9_ENVIRONMENT),
        ):
            with self.subTest(name=name, count=len(environment)):
                profile = copy.deepcopy(launcher.LOCAL_PROFILE)
                profile["profile"] = name
                profile["environment"] = dict(environment)
                self.write_profile(profile)
                self.assertEqual(self.defaults(), {})

    def test_historical_v5_through_v9_routes_exclude_ssd_streaming(self):
        for environment, count in (
            (V5_ENVIRONMENT, 30),
            (V6_ENVIRONMENT, 34),
            (V7_ENVIRONMENT, 36),
            (V8_ENVIRONMENT, 37),
            (V9_ENVIRONMENT, 38),
        ):
            with self.subTest(count=count):
                self.assertEqual(len(environment), count)
                self.assertNotIn("SPLASH_FLASH_PLE_SSD_STREAMING", environment)

    def test_other_source_format_or_architecture_never_applies_profile(self):
        for update in (
            {"source_identity_sha256": "b" * 64},
            {"schema": "other-format"},
        ):
            with self.subTest(update=update):
                self.assertEqual(
                    self.defaults(manifest={**self.manifest, **update}), {}
                )
        (self.package / "config.json").write_text('{"model_type":"qwen3_5"}')
        self.assertEqual(self.defaults(), {})

    def test_brand_must_match_exactly_and_ram_must_be_integer_above_floor(self):
        for hardware in (
            ("Apple M5 Max", 256 * 1024**3),
            ("Apple M4 Ultra", 256 * 1024**3),
            ("Apple M5 Ultra extra", 256 * 1024**3),
            ("apple M5 Ultra", 256 * 1024**3),
            ("Apple M5 Ultra", 192 * 1024**3 - 1),
            ("Apple M5 Ultra", float(256 * 1024**3)),
            None,
        ):
            with self.subTest(hardware=hardware):
                self.assertEqual(self.defaults(hardware), {})

    def test_unknown_or_malformed_profile_has_no_partial_defaults(self):
        mutations = (
            {"schema_version": 2},
            {"schema_version": True},
            {"schema_version": 1.0},
            {"profile": "unknown"},
            {"minimum_physical_ram_bytes": float(192 * 1024**3)},
            {"minimum_physical_ram_bytes": 1},
            {"architecture": "other"},
            {"source_identity_sha256": "b" * 64},
            {
                "environment": {
                    **launcher.LOCAL_PROFILE["environment"],
                    "PATH": "/unexpected",
                }
            },
            {
                "environment": {
                    **launcher.LOCAL_PROFILE["environment"],
                    "SPLASH_FLASH_MTP": 1,
                }
            },
            {
                "environment": {
                    **launcher.LOCAL_PROFILE["environment"],
                    "SPLASH_FLASH_MTP": "2",
                }
            },
            {
                "environment": {
                    **launcher.LOCAL_PROFILE["environment"],
                    "SPLASH_FLASH_PREFILL_ROWS": "4096",
                }
            },
        )
        for update in mutations:
            with self.subTest(update=update):
                profile = copy.deepcopy(launcher.LOCAL_PROFILE)
                profile.update(update)
                self.write_profile(profile)
                self.assertEqual(self.defaults(), {})
        for raw in ("{", "[]", "null"):
            with self.subTest(raw=raw):
                self.profile_path.write_text(raw)
                self.assertEqual(self.defaults(), {})
        self.profile_path.unlink()
        self.assertEqual(self.defaults(), {})

    def test_unavailable_hardware_ignores_profile(self):
        for failure in (
            OSError("unavailable"),
            ValueError("invalid RAM"),
            subprocess.TimeoutExpired("sysctl", 2),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                mock.patch.object(launcher, "ROOT", self.root),
                mock.patch.object(
                    launcher, "local_bundle_manifest", return_value=self.manifest
                ),
                mock.patch.object(
                    launcher, "_local_hardware_identity", side_effect=failure
                ),
            ):
                self.assertEqual(launcher._local_profile_defaults(self.package), {})

    def test_exec_profile_preserves_explicit_zero_empty_and_row_overrides_without_mutating_parent(
        self,
    ):
        original = {
            "SPLASH_FLASH_MTP": "0",
            "SPLASH_FLASH_BATCH": "",
            "SPLASH_FLASH_PREFILL_ROWS": "512",
            "SPLASH_FLASH_BATCH_PREFILL_ROWS": "512",
            "SPLASH_FLASH_MTP_DRAFT_DEPTH": "7",
            "UNRELATED": "preserved",
        }
        environment = self.exec_environment(original)
        for key, value in original.items():
            self.assertEqual(environment[key], value)
        self.assertEqual(environment["SPLASH_FLASH_QMV_F32"], "1")
        self.assertEqual(environment["SPLASH_FLASH_FUSE_HC"], "1")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_MTP"], "0")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_MTP_PREFILL"], "0")
        self.assertNotIn("CPPFLAGS", environment)
        self.assertNotIn("CXXFLAGS", environment)

    def test_exec_applies_exact_27_defaults_without_optional_experiments(self):
        environment = self.exec_environment()
        self.assertEqual(
            {key: value for key, value in environment.items() if key.startswith("SPLASH_FLASH_")},
            EXPECTED_ENVIRONMENT,
        )

    def test_legacy_model_never_reads_profile_or_probes_hardware(self):
        with (
            mock.patch.object(launcher, "RUNTIME_DIR", self.root / "runtime"),
            mock.patch.object(launcher, "_ensure_installed"),
            mock.patch.object(launcher, "_local_profile_defaults") as defaults,
            mock.patch.object(launcher, "_local_hardware_identity") as hardware,
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch.object(launcher.catalog, "spawn_refresh"),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            launcher.main(["serve", "--model", "incoai/Qwen3.8-27B-Splash"])
        defaults.assert_not_called()
        hardware.assert_not_called()
        self.assertFalse(
            any(key.startswith("SPLASH_FLASH_") for key in execute.call_args.args[2])
        )

    def test_each_explicit_boolean_zero_is_preserved(self):
        for key, value in EXPECTED_ENVIRONMENT.items():
            if value != "1":
                continue
            with self.subTest(key=key):
                self.assertEqual(self.exec_environment({key: "0"})[key], "0")

    def test_all_explicit_boolean_zeros_are_preserved_together(self):
        original = {key: "0" for key, value in EXPECTED_ENVIRONMENT.items() if value == "1"}
        environment = self.exec_environment(original)
        for key, value in original.items():
            self.assertEqual(environment[key], value)

    def test_disabled_parents_default_dependent_children_to_zero_only(self):
        dependencies = (
            ("SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP"),
            ("SPLASH_FLASH_MTP_QSA_F32", "SPLASH_FLASH_MTP_QSA_MPP"),
            ("SPLASH_FLASH_FLOAT_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"),
            ("SPLASH_FLASH_QMV_F32", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"),
            ("SPLASH_FLASH_MOE_Q4X8", "SPLASH_FLASH_MOE_M64"),
            ("SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH_MTP"),
            ("SPLASH_FLASH_BATCH", "SPLASH_FLASH_BATCH_MTP"),
            ("SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH_MTP_PREFILL"),
            ("SPLASH_FLASH_BATCH_PREFILL", "SPLASH_FLASH_BATCH_MTP_PREFILL"),
        )
        shared_defaults = dict(EXPECTED_ENVIRONMENT)
        for parent, child in dependencies:
            with self.subTest(parent=parent, child=child, explicit_child=False):
                environment = self.exec_environment({parent: "0"}, shared_defaults)
                self.assertEqual(environment[parent], "0")
                self.assertEqual(environment[child], "0")
            with self.subTest(parent=parent, child=child, explicit_child=True):
                environment = self.exec_environment({parent: "0", child: "1"}, shared_defaults)
                self.assertEqual(environment[parent], "0")
                self.assertEqual(environment[child], "1")
        self.assertEqual(shared_defaults, EXPECTED_ENVIRONMENT)

    def test_batch_prefill_remains_enabled_when_batch_is_explicitly_disabled(self):
        environment = self.exec_environment({"SPLASH_FLASH_BATCH": "0"})
        self.assertEqual(environment["SPLASH_FLASH_BATCH"], "0")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_MTP"], "0")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_PREFILL"], "1")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_MTP_PREFILL"], "1")
        self.assertEqual(environment["SPLASH_FLASH_BATCH_PREFILL_ROWS"], "2048")


if __name__ == "__main__":
    unittest.main()
