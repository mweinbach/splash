"""CPU-only measured selection/pin/context/store and explicit-opt-out contracts."""
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


class V13Tests(unittest.TestCase):
    def test_static_environment_matches_measured_envelope(self):
        command=json.loads((launcher.ROOT/'build/fixed4-R5-Root-bound-normal-sep22-v2/root-command.json').read_text())
        measured={}
        for i,value in enumerate(command['argv']):
            if value=='--env':key,val=command['argv'][i+1].split('=',1);measured[key]=val
        for key in ('SPLASH_FLASH_OPERAND_STORE','SPLASH_FLASH_INT8_EXPERT_STORE'):measured.pop(key)
        self.assertEqual(launcher.LOCAL_PROFILE_V13['environment'],measured)
        self.assertEqual(launcher.LOCAL_PROFILE['profile'],'m5-ultra-flash-next-v12')

    def test_explicit_MTP_and_depth_optouts(self):
        defaults=launcher.LOCAL_PROFILE_V13['environment'];before=copy.deepcopy(defaults)
        for supplied in ({'SPLASH_FLASH_MTP':'0'},{'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3'}):
            env=dict(supplied);launcher._apply_local_profile_defaults(env,defaults)
            self.assertEqual(env['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'],'0')
            self.assertEqual(env['SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21'],'0')
            for key,value in supplied.items():self.assertEqual(env[key],value)
        for value in ('1','0','','invalid'):
            env={'SPLASH_FLASH_MTP':'0','SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22':value}
            launcher._apply_local_profile_defaults(env,defaults);self.assertEqual(env['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'],value)
        self.assertEqual(defaults,before)

    def test_parent_optouts_suppress_implied_native_contradictions(self):
        cases={'SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22':['SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22'],
            'SPLASH_FLASH_MTP_QSA_F32':['SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21'],
            'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22':['SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22','SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22']}
        for parent,children in cases.items():
            env={parent:'0'};launcher._apply_local_profile_defaults(env,launcher.LOCAL_PROFILE_V13['environment'])
            for child in children:self.assertEqual(env[child],'0')

    def test_runtime_pin_drift_never_builds(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);profile=copy.deepcopy(launcher.LOCAL_PROFILE_V13);binary=root/profile['runtime']['relative_path'];binary.parent.mkdir(parents=True)
            binary.write_bytes(b'qualified fixture');binary.with_name('splash.metallib').write_bytes(b'library fixture')
            profile['runtime']['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();profile['runtime']['metallib_sha256']=hashlib.sha256(binary.with_name('splash.metallib').read_bytes()).hexdigest()
            with mock.patch.object(launcher,'ROOT',root),mock.patch.object(launcher.subprocess,'run')as run:
                self.assertEqual(launcher._qualified_v13_runtime(profile),binary)
                binary.write_bytes(b'changed')
                self.assertRaisesRegex(launcher.LauncherError,'missing or changed',launcher._qualified_v13_runtime,profile)
                binary.unlink();self.assertRaises(launcher.LauncherError,launcher._qualified_v13_runtime,profile)
                run.assert_not_called()

    def test_v13_profile_model_hardware_gate_and_Full512_selection(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);package=root/'package';package.mkdir();(package/'config.json').write_text('{"model_type":"qwen4_exp"}')
            (root/'.splash-local-profile.json').write_text(json.dumps(launcher.LOCAL_PROFILE_V13))
            manifest={'schema':launcher.LOCAL_SCHEMA,'source_identity_sha256':launcher.LOCAL_PROFILE_V13['source_identity_sha256']}
            with mock.patch.object(launcher,'ROOT',root),mock.patch.object(launcher,'local_bundle_manifest',return_value=manifest),mock.patch.object(launcher,'_local_hardware_identity',return_value=('Apple M5 Ultra',256*1024**3))as hardware,mock.patch.object(launcher,'_qualified_saved_operand_defaults',return_value={'SPLASH_FLASH_OPERAND_STORE':'saved'}),mock.patch.object(launcher,'_qualified_saved_int8_expert_defaults',return_value={'SPLASH_FLASH_INT8_EXPERT_STORE':'full512'})as experts,mock.patch.dict(os.environ,{},clear=True):
                defaults=launcher._local_profile_defaults(package);self.assertEqual(defaults['SPLASH_FLASH_INT8_EXPERT_STORE'],'full512')
                self.assertEqual(experts.call_args.kwargs['qualification']['selected_experts_per_layer'],512)
                hardware.return_value=('Apple M5 Max',256*1024**3);self.assertEqual(launcher._local_profile_defaults(package),{})
                hardware.return_value=('Apple M5 Ultra',128*1024**3);self.assertEqual(launcher._local_profile_defaults(package),{})

    def test_default_context16K_explicit_context_authoritative(self):
        for override,wanted in ((None,'16384'),('8192','8192')):
            with tempfile.TemporaryDirectory()as directory:
                root=Path(directory);package=root/'package';package.mkdir();defaults=launcher._QualifiedLocalDefaults(launcher.LOCAL_PROFILE_V13['environment'],launcher.LOCAL_PROFILE_V13)
                with mock.patch.object(launcher,'ROOT',root),mock.patch.object(launcher,'RUNTIME_DIR',root/'runtime'),mock.patch.object(launcher,'_ensure_local_installed',return_value=package),mock.patch.object(launcher,'_local_profile_defaults',return_value=defaults),mock.patch.object(launcher,'_qualified_v13_runtime',return_value=root/'pinned'),mock.patch.object(launcher,'_ensure_local_runtime')as build,mock.patch.object(launcher.socket,'socket'),mock.patch.object(launcher.os,'execve')as execute,mock.patch.dict(os.environ,{},clear=True),mock.patch('sys.stdout',io.StringIO()):
                    args=['serve','--model','local/Flash-Next','--local-package',str(package)]
                    if override:args+=['--max-context',override]
                    launcher.main(args);argv=execute.call_args.args[1];self.assertEqual(argv[argv.index('--max-context')+1],wanted)
                    self.assertEqual(argv[argv.index('--binary')+1],str(root/'pinned'));build.assert_not_called()


if __name__=='__main__':unittest.main()
