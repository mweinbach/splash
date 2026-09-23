"""CPU-only exact public bulk QSA/SG8 profile and parent override contracts."""
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from install import launcher

BULK = 'SPLASH_FLASH_QSA_BULK_PREFILL'
SG8 = 'SPLASH_FLASH_QSA_BULK_PREFILL_SG8'
PREREQUISITES = ('SPLASH_FLASH_QSA_F32', 'SPLASH_FLASH_QSA_MPP', 'SPLASH_FLASH_QSA_ROW_TILES')


class LocalProfileV12Tests(unittest.TestCase):
    def test_active_v12_is_exactly_two_additions_to_accepted_v11(self):
        with mock.patch.object(launcher, '_qualified_saved_operand_defaults', return_value={}):
            historical = launcher._local_profile_v11_candidate(Path('/unused'), {})
        current = launcher.LOCAL_PROFILE
        self.assertEqual(current['profile'], 'm5-ultra-flash-next-v12')
        self.assertEqual(len(current['environment']), 44)
        self.assertEqual(len(historical['environment']), 42)
        self.assertEqual(current['environment'], {**historical['environment'], BULK: '1', SG8: '1'})
        self.assertEqual(current['environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'], '3')
        self.assertIn(json.loads((launcher.ROOT / '.splash-local-profile.json').read_text()),
                      (current, launcher.LOCAL_PROFILE_V13, launcher.LOCAL_PROFILE_V14,
                       launcher.LOCAL_PROFILE_V15, launcher.LOCAL_PROFILE_V16))
        for flag in ['SPLASH_FLASH_ALLROWS_FULL512_TARGET', 'SPLASH_FLASH_DENSE_TRAVERSAL', 'SPLASH_FLASH_MOE_GATHERED_MPP', 'SPLASH_FLASH_MOE_QMV_C2', 'SPLASH_FLASH_QSA_NAX']:
            self.assertNotIn(flag, current['environment'])

    def test_each_original_parent_zero_suppresses_both_implied_bulk_selectors(self):
        defaults = copy.deepcopy(launcher.LOCAL_PROFILE['environment'])
        before = copy.deepcopy(defaults)
        for parent in PREREQUISITES:
            with self.subTest(parent=parent):
                environment = {parent: '0', 'TASK_SENTINEL': 'keep'}
                launcher._apply_local_profile_defaults(environment, defaults)
                self.assertEqual(environment[BULK], '0')
                self.assertEqual(environment[SG8], '0')
                self.assertEqual(environment[parent], '0')
                self.assertEqual(environment['TASK_SENTINEL'], 'keep')
        self.assertEqual(defaults, before)

    def test_bulk_zero_suppresses_only_implied_sg8(self):
        environment = {BULK: '0'}
        launcher._apply_local_profile_defaults(environment, launcher.LOCAL_PROFILE['environment'])
        self.assertEqual(environment[BULK], '0')
        self.assertEqual(environment[SG8], '0')
        for parent in PREREQUISITES:
            self.assertEqual(environment[parent], '1')

    def test_explicit_child_values_survive_for_native_validation(self):
        for parent in [*PREREQUISITES, BULK]:
            for child in [BULK, SG8]:
                if parent == child:
                    continue
                for value in ['0', '1', '', 'invalid']:
                    with self.subTest(parent=parent, child=child, value=value):
                        environment = {parent: '0', child: value}
                        launcher._apply_local_profile_defaults(environment, launcher.LOCAL_PROFILE['environment'])
                        self.assertEqual(environment[parent], '0')
                        self.assertEqual(environment[child], value)

    def test_unrelated_batch_mtp_dense_optouts_do_not_disable_bulk_qsa(self):
        for parent in ['SPLASH_FLASH_BATCH', 'SPLASH_FLASH_MTP', 'SPLASH_FLASH_DENSE_CACHE']:
            with self.subTest(parent=parent):
                environment = {parent: '0'}
                launcher._apply_local_profile_defaults(environment, launcher.LOCAL_PROFILE['environment'])
                self.assertEqual(environment[BULK], '1')
                self.assertEqual(environment[SG8], '1')


if __name__ == '__main__':
    unittest.main()
