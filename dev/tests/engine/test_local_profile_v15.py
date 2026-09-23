"""CPU-only v15 contracts: v14's measured environment with the second optimized runtime pin."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from install import launcher


class V15Tests(unittest.TestCase):
    def test_v15_reuses_v14_environment_with_distinct_runtime(self):
        v13, v14, v15 = launcher.LOCAL_PROFILE_V13, launcher.LOCAL_PROFILE_V14, launcher.LOCAL_PROFILE_V15
        self.assertEqual(v15['profile'], 'm5-ultra-flash-next-v15')
        for key in ('schema_version', 'source_identity_sha256', 'architecture', 'cpu_brand',
                    'minimum_physical_ram_bytes', 'environment', 'serving'):
            self.assertEqual(v15[key], v14[key])
            self.assertEqual(v15[key], v13[key])
        self.assertEqual(v15['runtime']['relative_path'], 'build/flash-opt-sep22-v15/splash-flash')
        self.assertNotEqual(v15['runtime']['executable_sha256'], v14['runtime']['executable_sha256'])
        self.assertEqual(launcher.PINNED_LOCAL_PROFILES[:3], (v13, v14, v15))

    def test_v15_profile_selects_pinned_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            (package / 'config.json').write_text('{"model_type":"qwen4_exp"}')
            (root / '.splash-local-profile.json').write_text(json.dumps(launcher.LOCAL_PROFILE_V15))
            manifest = {'schema': launcher.LOCAL_SCHEMA,
                        'source_identity_sha256': launcher.LOCAL_PROFILE_V15['source_identity_sha256']}
            with mock.patch.object(launcher, 'ROOT', root), \
                    mock.patch.object(launcher, 'local_bundle_manifest', return_value=manifest), \
                    mock.patch.object(launcher, '_local_hardware_identity', return_value=('Apple M5 Ultra', 256 * 1024**3)), \
                    mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={'SPLASH_FLASH_OPERAND_STORE': 'saved'}), \
                    mock.patch.object(launcher, '_qualified_saved_int8_expert_defaults', return_value={'SPLASH_FLASH_INT8_EXPERT_STORE': 'full512'}), \
                    mock.patch.dict(os.environ, {}, clear=True):
                defaults = launcher._local_profile_defaults(package)
                self.assertEqual(defaults.profile, launcher.LOCAL_PROFILE_V15)
                self.assertEqual(defaults['SPLASH_FLASH_INT8_EXPERT_STORE'], 'full512')

    def test_modified_v15_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            changed = json.loads(json.dumps(launcher.LOCAL_PROFILE_V15))
            changed['runtime']['executable_sha256'] = '0' * 64
            (root / '.splash-local-profile.json').write_text(json.dumps(changed))
            with mock.patch.object(launcher, 'ROOT', root):
                self.assertRaisesRegex(launcher.LauncherError, 'differs from the measured configuration',
                                       launcher._local_profile_defaults, package)

    def test_v15_serve_uses_pinned_runtime_and_default_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            defaults = launcher._QualifiedLocalDefaults(launcher.LOCAL_PROFILE_V15['environment'], launcher.LOCAL_PROFILE_V15)
            with mock.patch.object(launcher, 'ROOT', root), mock.patch.object(launcher, 'RUNTIME_DIR', root / 'runtime'), \
                    mock.patch.object(launcher, '_ensure_local_installed', return_value=package), \
                    mock.patch.object(launcher, '_local_profile_defaults', return_value=defaults), \
                    mock.patch.object(launcher, '_qualified_v13_runtime', return_value=root / 'pinned') as pinned, \
                    mock.patch.object(launcher, '_ensure_local_runtime') as build, \
                    mock.patch.object(launcher.socket, 'socket'), mock.patch.object(launcher.os, 'execve') as execute, \
                    mock.patch.dict(os.environ, {}, clear=True), mock.patch('sys.stdout', io.StringIO()):
                launcher.main(['serve', '--model', 'local/Flash-Next', '--local-package', str(package)])
                argv = execute.call_args.args[1]
                self.assertEqual(argv[argv.index('--binary') + 1], str(root / 'pinned'))
                self.assertEqual(argv[argv.index('--max-context') + 1], '16384')
                self.assertEqual(pinned.call_args.args[0], launcher.LOCAL_PROFILE_V15)
                build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
