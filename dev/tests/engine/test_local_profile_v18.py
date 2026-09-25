"""CPU-only contracts for the pinned v18 profile: environment, runtime pin, gates and serve defaults."""
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from install import launcher

MEGAKERNEL = {'SPLASH_OPT_MOE': '1', 'SPLASH_OPT_MOE_TILE': '64', 'SPLASH_MK_MOE': '1',
              'SPLASH_MK_DENSE': '1', 'SPLASH_MK_CONCURRENT': '1', 'SPLASH_MK_QMV': '1',
              'SPLASH_MK_PREFILL': '1', 'SPLASH_MK_MOE_TILED': '1'}
V18 = launcher.LOCAL_PROFILE_V18


class V18Tests(unittest.TestCase):
    def test_environment_is_measured_envelope_plus_megakernel_flags(self):
        command = json.loads((launcher.ROOT / 'build/fixed4-R5-Root-bound-normal-sep22-v2/root-command.json').read_text())
        measured = {}
        for i, value in enumerate(command['argv']):
            if value == '--env':
                key, val = command['argv'][i + 1].split('=', 1)
                measured[key] = val
        for key in ('SPLASH_FLASH_OPERAND_STORE', 'SPLASH_FLASH_INT8_EXPERT_STORE'):
            measured.pop(key)
        self.assertEqual(V18['environment'], {**measured, **MEGAKERNEL})
        self.assertEqual(launcher.LOCAL_PROFILE['profile'], 'm5-ultra-flash-next-v12')

    def test_v18_is_the_only_pinned_profile_and_the_root_default(self):
        self.assertEqual(V18['profile'], 'm5-ultra-flash-next-v18')
        self.assertEqual(V18['runtime']['relative_path'], 'build/flash-opt-sep22-v18/splash-flash')
        self.assertEqual(V18['serving'], {'default_max_context_tokens': 16384})
        self.assertEqual(launcher.PINNED_LOCAL_PROFILES, (V18,))
        profile = json.loads((launcher.ROOT / '.splash-local-profile.json').read_text())
        self.assertEqual(profile, V18)

    def test_explicit_MTP_and_depth_optouts(self):
        defaults = V18['environment']
        before = copy.deepcopy(defaults)
        for supplied in ({'SPLASH_FLASH_MTP': '0'}, {'SPLASH_FLASH_MTP_DRAFT_DEPTH': '3'}):
            env = dict(supplied)
            launcher._apply_local_profile_defaults(env, defaults)
            self.assertEqual(env['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'], '0')
            self.assertEqual(env['SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21'], '0')
            for key, value in supplied.items():
                self.assertEqual(env[key], value)
        for value in ('1', '0', '', 'invalid'):
            env = {'SPLASH_FLASH_MTP': '0', 'SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22': value}
            launcher._apply_local_profile_defaults(env, defaults)
            self.assertEqual(env['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'], value)
        self.assertEqual(defaults, before)

    def test_parent_optouts_suppress_implied_native_contradictions(self):
        cases = {'SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22': ['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22',
                                                                'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22'],
                 'SPLASH_FLASH_MTP_QSA_F32': ['SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21'],
                 'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22': ['SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22',
                                                                 'SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22',
                                                                 'SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22',
                                                                 'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22']}
        for parent, children in cases.items():
            env = {parent: '0'}
            launcher._apply_local_profile_defaults(env, V18['environment'])
            for child in children:
                self.assertEqual(env[child], '0')

    def test_runtime_pin_drift_never_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = copy.deepcopy(V18)
            binary = root / profile['runtime']['relative_path']
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'qualified fixture')
            binary.with_name('splash.metallib').write_bytes(b'library fixture')
            profile['runtime']['executable_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            profile['runtime']['metallib_sha256'] = hashlib.sha256(binary.with_name('splash.metallib').read_bytes()).hexdigest()
            with mock.patch.object(launcher, 'ROOT', root), mock.patch.object(launcher.subprocess, 'run') as run:
                self.assertEqual(launcher._qualified_pinned_runtime(profile), binary)
                binary.write_bytes(b'changed')
                self.assertRaisesRegex(launcher.LauncherError, 'missing or changed',
                                       launcher._qualified_pinned_runtime, profile)
                binary.unlink()
                self.assertRaises(launcher.LauncherError, launcher._qualified_pinned_runtime, profile)
                run.assert_not_called()

    def test_profile_model_hardware_gate_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'package'; package.mkdir()
            (package / 'config.json').write_text('{"model_type":"qwen4_exp"}')
            (root / '.splash-local-profile.json').write_text(json.dumps(V18))
            manifest = {'schema': launcher.LOCAL_SCHEMA, 'source_identity_sha256': V18['source_identity_sha256']}
            with mock.patch.object(launcher, 'ROOT', root), \
                    mock.patch.object(launcher, 'local_bundle_manifest', return_value=manifest), \
                    mock.patch.object(launcher, '_local_hardware_identity', return_value=('Apple M5 Ultra', 256 * 1024**3)) as hardware, \
                    mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={'SPLASH_FLASH_OPERAND_STORE': 'saved'}), \
                    mock.patch.object(launcher, '_qualified_saved_int8_expert_defaults', return_value={'SPLASH_FLASH_INT8_EXPERT_STORE': 'full512'}) as experts, \
                    mock.patch.dict(os.environ, {}, clear=True):
                defaults = launcher._local_profile_defaults(package)
                self.assertEqual(defaults.profile, V18)
                self.assertEqual(defaults['SPLASH_FLASH_INT8_EXPERT_STORE'], 'full512')
                self.assertEqual(experts.call_args.kwargs['qualification']['selected_experts_per_layer'], 512)
                environment = {}
                launcher._apply_local_profile_defaults(environment, defaults)
                for key, value in MEGAKERNEL.items():
                    self.assertEqual(environment[key], value)
                hardware.return_value = ('Apple M5 Max', 256 * 1024**3)
                self.assertEqual(launcher._local_profile_defaults(package), {})
                hardware.return_value = ('Apple M5 Ultra', 128 * 1024**3)
                self.assertEqual(launcher._local_profile_defaults(package), {})

    def test_modified_v18_profile_is_rejected(self):
        for field, value in (('environment', 'SPLASH_MK_MOE_TILED'), ('runtime', 'relative_path')):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); package = root / 'package'; package.mkdir()
                changed = json.loads(json.dumps(V18))
                changed[field][value] = '0'
                (root / '.splash-local-profile.json').write_text(json.dumps(changed))
                with mock.patch.object(launcher, 'ROOT', root):
                    self.assertRaisesRegex(launcher.LauncherError, 'differs from the measured configuration',
                                           launcher._local_profile_defaults, package)

    def test_serve_uses_pinned_runtime_and_default_context(self):
        for override, wanted in ((None, '16384'), ('8192', '8192')):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); package = root / 'package'; package.mkdir()
                defaults = launcher._QualifiedLocalDefaults(V18['environment'], V18)
                with mock.patch.object(launcher, 'ROOT', root), mock.patch.object(launcher, 'RUNTIME_DIR', root / 'runtime'), \
                        mock.patch.object(launcher, '_ensure_local_installed', return_value=package), \
                        mock.patch.object(launcher, '_local_profile_defaults', return_value=defaults), \
                        mock.patch.object(launcher, '_qualified_pinned_runtime', return_value=root / 'pinned') as pinned, \
                        mock.patch.object(launcher, '_ensure_local_runtime') as build, \
                        mock.patch.object(launcher.socket, 'socket'), mock.patch.object(launcher.os, 'execve') as execute, \
                        mock.patch.dict(os.environ, {}, clear=True), mock.patch('sys.stdout', io.StringIO()):
                    args = ['serve', '--model', 'local/Flash-Next', '--local-package', str(package)]
                    if override:
                        args += ['--max-context', override]
                    launcher.main(args)
                    argv = execute.call_args.args[1]
                    self.assertEqual(argv[argv.index('--binary') + 1], str(root / 'pinned'))
                    self.assertEqual(argv[argv.index('--max-context') + 1], wanted)
                    self.assertEqual(pinned.call_args.args[0], V18)
                    build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
