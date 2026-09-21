"""CPU-only qualified SSD defaults, placement overrides, and cache validation."""

import copy
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from dev.tests.engine.test_local_launcher import local_fixture


class LocalPLESSDLauncherTests(unittest.TestCase):
    def parse(self, *options):
        return launcher.parse_args([
            "serve", "--model", "local/Flash-Next", "--local-package", "/fixture",
            *options,
        ])

    def test_cli_is_unset_and_qualified_profile_selects_ssd_without_forced_cache(self):
        profile = copy.deepcopy(launcher.LOCAL_PROFILE)
        args = self.parse()
        self.assertIsNone(args.ple_ssd_streaming)
        self.assertIsNone(args.ple_ssd_cache_mb)
        self.assertEqual(launcher.LOCAL_PROFILE, profile)
        self.assertEqual(len(profile["environment"]), 39)
        self.assertEqual(profile["environment"]["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertNotIn("SPLASH_FLASH_PLE_SSD_CACHE_MB", profile["environment"])

    def test_explicit_cache_and_streaming_options(self):
        for value in ("0", "1", "64", "1024"):
            with self.subTest(value=value):
                args = self.parse("--ple-ssd-streaming", "--ple-ssd-cache-mb", value)
                self.assertTrue(args.ple_ssd_streaming)
                self.assertEqual(args.ple_ssd_cache_mb, int(value))

    def test_placement_and_cache_options_require_local_model_before_startup(self):
        for options in (
            ["serve", "--model", "org/Remote", "--ple-ssd-streaming"],
            ["serve", "--model", "org/Remote", "--no-ple-ssd-streaming"],
            ["serve", "--model", "org/Remote", "--ple-ssd-cache-mb", "64"],
        ):
            with self.subTest(options=options), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(options)
        # The source/hardware profile gate runs in serve, after parsing.
        self.assertEqual(self.parse("--ple-ssd-cache-mb", "64").ple_ssd_cache_mb, 64)

    def test_opposing_cli_placement_options_are_rejected(self):
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--ple-ssd-streaming", "--no-ple-ssd-streaming")

    def test_cache_size_is_canonical_and_bounded(self):
        for value in ("", "-1", "+1", "01", "00", "1.0", "1e2", "1025", "1G", " 64", "64 ", "٦٤"):
            with self.subTest(value=value), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse("--ple-ssd-streaming", f"--ple-ssd-cache-mb={value}")

    def execute(self, cli=(), environment=None, defaults=None, success=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = local_fixture(root / "bundle")
            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher, "RUNTIME_DIR", root / "runtime"),
                mock.patch.object(launcher, "_ensure_local_runtime"),
                mock.patch.object(launcher, "_ensure_local_installed", return_value=bundle),
                mock.patch.object(launcher, "_local_profile_defaults", return_value=(
                    launcher.LOCAL_PROFILE["environment"] if defaults is None else defaults
                )),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher.os, "execve") as execute,
                mock.patch.dict(os.environ, environment or {}, clear=True),
                mock.patch("sys.stderr", io.StringIO()) as errors,
            ):
                result = launcher.main([
                    "serve", "--model", "local/Flash-Next", "--local-package", str(bundle),
                    "--port", "8011", *cli,
                ])
            if not success:
                self.assertEqual(result, 1)
                execute.assert_not_called()
                self.assertIn("requires SSD streaming to be enabled", errors.getvalue())
                return
            execute.assert_called_once()
            return execute.call_args.args[2]

    def test_cli_values_override_explicit_environment_and_keep_maintenance(self):
        environment = self.execute(("--ple-ssd-streaming", "--ple-ssd-cache-mb", "128"), {
            "SPLASH_FLASH_PLE_SSD_STREAMING": "0", "SPLASH_FLASH_PLE_SSD_CACHE_MB": "32",
        })
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_CACHE_MB"], "128")
        self.assertEqual(environment["SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"], "1")

    def test_unset_cli_preserves_environment_cache_and_native_default_cache(self):
        environment = self.execute(environment={
            "SPLASH_FLASH_PLE_SSD_STREAMING": "1", "SPLASH_FLASH_PLE_SSD_CACHE_MB": "0",
        })
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_CACHE_MB"], "0")
        default = self.execute()
        self.assertEqual(default["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertNotIn("SPLASH_FLASH_PLE_SSD_CACHE_MB", default)

    def test_cache_cli_uses_effective_qualified_default(self):
        environment = self.execute(("--ple-ssd-cache-mb", "128"))
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_CACHE_MB"], "128")

    def test_environment_opt_out_keeps_cache_unset(self):
        environment = self.execute(environment={"SPLASH_FLASH_PLE_SSD_STREAMING": "0"})
        self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_STREAMING"], "0")
        self.assertNotIn("SPLASH_FLASH_PLE_SSD_CACHE_MB", environment)

    def test_cli_opt_out_overrides_default_or_environment(self):
        for explicit in ({}, {"SPLASH_FLASH_PLE_SSD_STREAMING": "1"}):
            with self.subTest(environment=explicit):
                environment = self.execute(("--no-ple-ssd-streaming",), explicit)
                self.assertEqual(environment["SPLASH_FLASH_PLE_SSD_STREAMING"], "0")
                self.assertNotIn("SPLASH_FLASH_PLE_SSD_CACHE_MB", environment)
                self.assertEqual(environment["SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"], "1")

    def test_disabled_streaming_rejects_cache_cli_after_effective_defaults(self):
        self.execute(("--ple-ssd-cache-mb", "64"),
                     {"SPLASH_FLASH_PLE_SSD_STREAMING": "0"}, success=False)
        self.execute(("--no-ple-ssd-streaming", "--ple-ssd-cache-mb", "64"), success=False)

    def test_unqualified_package_has_no_default_and_cache_requires_explicit_streaming(self):
        unknown = self.execute(defaults={})
        self.assertNotIn("SPLASH_FLASH_PLE_SSD_STREAMING", unknown)
        self.assertNotIn("SPLASH_FLASH_PLE_SSD_CACHE_MB", unknown)
        self.execute(("--ple-ssd-cache-mb", "64"), defaults={}, success=False)
        enabled = self.execute(("--ple-ssd-streaming", "--ple-ssd-cache-mb", "64"), defaults={})
        self.assertEqual(enabled["SPLASH_FLASH_PLE_SSD_STREAMING"], "1")
        self.assertEqual(enabled["SPLASH_FLASH_PLE_SSD_CACHE_MB"], "64")
        explicit_environment = self.execute(("--ple-ssd-cache-mb", "0"),
                                           {"SPLASH_FLASH_PLE_SSD_STREAMING": "1"}, defaults={})
        self.assertEqual(explicit_environment["SPLASH_FLASH_PLE_SSD_CACHE_MB"], "0")


if __name__ == "__main__":
    unittest.main()
