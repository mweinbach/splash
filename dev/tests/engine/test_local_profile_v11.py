"""CPU-only versioned safe-prefill defaults and explicit opt-out contracts."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher

TEACHER = 'SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY'
DENSE = 'SPLASH_FLASH_PREFILL_DENSE_TILES'


class LocalProfileV11Tests(unittest.TestCase):
    def test_historical_v11_only_adds_two_flags_to_accepted_v10(self):
        with mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
            current = launcher._local_profile_v11_candidate(Path('/unused'), {})
        with mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
            old = launcher._local_profile_v10_candidate(Path('/unused'), {})
        self.assertEqual(current['profile'], 'm5-ultra-flash-next-v11')
        self.assertEqual(len(current['environment']), 42)
        self.assertEqual(len(old['environment']), 40)
        self.assertEqual(current['environment'], {**old['environment'], TEACHER: '1', DENSE: '1'})
        self.assertEqual(old['environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'], '3')
        self.assertEqual(old['environment']['SPLASH_FLASH_QSA_OUT_F32_N32'], '1')
        self.assertIn(json.loads((launcher.ROOT / '.splash-local-profile.json').read_text()),
                      (launcher.LOCAL_PROFILE, launcher.LOCAL_PROFILE_V13, launcher.LOCAL_PROFILE_V14,
                       launcher.LOCAL_PROFILE_V15, launcher.LOCAL_PROFILE_V16))

    def test_original_placement_snapshot_is_distinct_from_accepted_v10(self):
        with mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
            original = launcher._local_profile_v10_placement_candidate(Path('/unused'), {})
            accepted = launcher._local_profile_v10_candidate(Path('/unused'), {})
        self.assertEqual(len(original['environment']), 39)
        self.assertEqual(original['environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'], '15')
        self.assertNotIn('SPLASH_FLASH_QSA_OUT_F32_N32', original['environment'])
        self.assertEqual(len(accepted['environment']), 40)
        self.assertEqual(accepted['environment'], {**original['environment'], 'SPLASH_FLASH_MTP_DRAFT_DEPTH': '3', 'SPLASH_FLASH_QSA_OUT_F32_N32': '1'})
        for old in [original, accepted]:
            self.assertNotIn(TEACHER, old['environment'])
            self.assertNotIn(DENSE, old['environment'])
            self.assertNotIn('SPLASH_FLASH_QSA_BULK_PREFILL', old['environment'])
            self.assertNotIn('SPLASH_FLASH_QSA_BULK_PREFILL_SG8', old['environment'])

    def test_historical_v5_through_v11_do_not_inherit_new_defaults(self):
        expected = [(5, 30, '15'), (6, 34, '15'), (7, 36, '15'), (8, 37, '15'), (9, 38, '15'), (10, 40, '3'), (11, 42, '3')]
        for version, count, depth in expected:
            with self.subTest(version=version), mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
                old = getattr(launcher, f'_local_profile_v{version}_candidate')(Path('/unused'), {})
            self.assertEqual(old['profile'], f'm5-ultra-flash-next-v{version}')
            self.assertEqual(len(old['environment']), count)
            self.assertEqual(old['environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'], depth)
            if version < 11:
                self.assertNotIn(TEACHER, old['environment'])
                self.assertNotIn(DENSE, old['environment'])
            self.assertNotIn('SPLASH_FLASH_QSA_BULK_PREFILL', old['environment'])
            self.assertNotIn('SPLASH_FLASH_QSA_BULK_PREFILL_SG8', old['environment'])

    def test_parent_optout_suppresses_only_implied_selector(self):
        defaults = copy.deepcopy(launcher.LOCAL_PROFILE['environment'])
        before = copy.deepcopy(defaults)
        for parent, child in [('SPLASH_FLASH_MTP', TEACHER), ('SPLASH_FLASH_DENSE_CACHE', DENSE)]:
            environment = {parent: '0', 'TASK_SENTINEL': 'retained'}
            launcher._apply_local_profile_defaults(environment, defaults)
            self.assertEqual(environment[parent], '0')
            self.assertEqual(environment[child], '0')
            self.assertEqual(environment['TASK_SENTINEL'], 'retained')
            for explicit in ['0', '1', '', 'invalid']:
                environment = {parent: '0', child: explicit}
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[parent], '0')
                self.assertEqual(environment[child], explicit)
        self.assertEqual(defaults, before)

    def test_two_explicit_optouts_survive_and_process_environment_is_unchanged(self):
        source = {TEACHER: '0', DENSE: '0'}
        with mock.patch.dict(os.environ, source, clear=True):
            environment = dict(os.environ)
            launcher._apply_local_profile_defaults(environment, launcher.LOCAL_PROFILE['environment'])
            self.assertEqual(environment[TEACHER], '0')
            self.assertEqual(environment[DENSE], '0')
            self.assertEqual(dict(os.environ), source)

    def test_saved_preceding_v10_is_rejected_before_hardware_or_store_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
                historical = launcher._local_profile_v10_candidate(Path('/unused'), {})
            (root / '.splash-local-profile.json').write_text(json.dumps(historical))
            with mock.patch.object(launcher, 'ROOT', root), mock.patch.object(launcher, 'local_bundle_manifest') as package, mock.patch.object(launcher, '_local_hardware_identity') as hardware, mock.patch.object(launcher, '_qualified_saved_operand_defaults') as operands, mock.patch.object(launcher, '_qualified_saved_int8_expert_defaults') as experts:
                self.assertEqual(launcher._local_profile_defaults(Path('/unused')), {})
                for probe in [package, hardware, operands, experts]:
                    probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
