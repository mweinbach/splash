"""CPU-only v14 contracts: v13's measured environment with the optimized runtime pin."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from install import launcher


class V14Tests(unittest.TestCase):
    def test_v14_reuses_v13_environment_with_distinct_runtime(self):
        v13, v14 = launcher.LOCAL_PROFILE_V13, launcher.LOCAL_PROFILE_V14
        self.assertEqual(v14['profile'], 'm5-ultra-flash-next-v14')
        for key in ('schema_version', 'source_identity_sha256', 'architecture', 'cpu_brand',
                    'minimum_physical_ram_bytes', 'environment', 'serving'):
            self.assertEqual(v14[key], v13[key])
        self.assertEqual(v14['runtime']['relative_path'], 'build/flash-opt-sep22-v14/splash-flash')
        self.assertNotEqual(v14['runtime']['executable_sha256'], v13['runtime']['executable_sha256'])
        self.assertEqual(launcher.PINNED_LOCAL_PROFILES[:2], (v13, v14))

    def test_v14_profile_selects_pinned_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            (package / 'config.json').write_text('{"model_type":"qwen4_exp"}')
            (root / '.splash-local-profile.json').write_text(json.dumps(launcher.LOCAL_PROFILE_V14))
            manifest = {'schema': launcher.LOCAL_SCHEMA,
                        'source_identity_sha256': launcher.LOCAL_PROFILE_V14['source_identity_sha256']}
            with mock.patch.object(launcher, 'ROOT', root), \
                    mock.patch.object(launcher, 'local_bundle_manifest', return_value=manifest), \
                    mock.patch.object(launcher, '_local_hardware_identity', return_value=('Apple M5 Ultra', 256 * 1024**3)), \
                    mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={'SPLASH_FLASH_OPERAND_STORE': 'saved'}), \
                    mock.patch.object(launcher, '_qualified_saved_int8_expert_defaults', return_value={'SPLASH_FLASH_INT8_EXPERT_STORE': 'full512'}), \
                    mock.patch.dict(os.environ, {}, clear=True):
                defaults = launcher._local_profile_defaults(package)
                self.assertEqual(defaults.profile, launcher.LOCAL_PROFILE_V14)
                self.assertEqual(defaults['SPLASH_FLASH_INT8_EXPERT_STORE'], 'full512')

    def test_modified_v14_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            changed = json.loads(json.dumps(launcher.LOCAL_PROFILE_V14))
            changed['runtime']['executable_sha256'] = '0' * 64
            (root / '.splash-local-profile.json').write_text(json.dumps(changed))
            with mock.patch.object(launcher, 'ROOT', root):
                self.assertRaisesRegex(launcher.LauncherError, 'differs from the measured configuration',
                                       launcher._local_profile_defaults, package)

    def test_v14_serve_uses_pinned_runtime_and_default_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            defaults = launcher._QualifiedLocalDefaults(launcher.LOCAL_PROFILE_V14['environment'], launcher.LOCAL_PROFILE_V14)
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
                self.assertEqual(pinned.call_args.args[0], launcher.LOCAL_PROFILE_V14)
                build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
