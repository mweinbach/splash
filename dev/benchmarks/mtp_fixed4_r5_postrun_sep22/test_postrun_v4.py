import copy
import unittest
from dev.benchmarks.mtp_fixed4_r5_postrun_sep22 import postrun_compare_v4 as v4

SOURCE='a'*64
def pair(suffix=v4.BATCH_ILP_SUFFIX):
    before='original-native-routes;current-fixed-math'
    after=before+v4.p.MARKER+SOURCE
    return [{'engine_instance_id':1,'kernel_routes':before,'batch_prefill_kernel_routes':v4.BATCH_PREFIX+before+suffix,'unchanged':{'raw':True,'unknown':17}},
            {'engine_instance_id':2,'kernel_routes':after,'batch_prefill_kernel_routes':v4.BATCH_PREFIX+after+suffix,'unchanged':{'raw':True,'unknown':17}}]

class DisplayTests(unittest.TestCase):
    def test_known_two_displays_normalize_and_inputs_remain_literal(self):
        for suffix in ('',v4.BATCH_ILP_SUFFIX):
            values=pair(suffix);saved=copy.deepcopy(values)
            normalized=v4.normalize_registered_identity_pair(values,SOURCE)
            self.assertEqual(normalized[0],normalized[1]);self.assertEqual(values,saved)
    def test_foreign_marker_source_in_either_display_rejected(self):
        for field in ('kernel_routes','batch_prefill_kernel_routes'):
            values=pair();values[1][field]=values[1][field].replace(SOURCE,'b'*64)
            with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_duplicate_marker_in_either_display_rejected(self):
        for field in ('kernel_routes','batch_prefill_kernel_routes'):
            values=pair();values[1][field]+=v4.p.MARKER+SOURCE
            with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_parent_marker_or_missing_child_marker_rejected(self):
        for index in (0,1):
            values=pair()
            for field in ('kernel_routes','batch_prefill_kernel_routes'):
                values[index][field]=values[index][field]+v4.p.MARKER+SOURCE if index==0 else values[index][field].replace(v4.p.MARKER+SOURCE,'')
            with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_batch_prefix_foreign_tail_and_nonstring_rejected(self):
        for bad in ('foreign:',v4.BATCH_PREFIX+'wrong-body',None,True):
            values=pair();values[1]['batch_prefill_kernel_routes']=bad
            with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_optional_source_suffix_cannot_change_between_runs(self):
        values=pair();values[1]['batch_prefill_kernel_routes']=values[1]['batch_prefill_kernel_routes'].removesuffix(v4.BATCH_ILP_SUFFIX)
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_unknown_identity_key_and_nested_value_drift_rejected(self):
        values=pair();values[1]['new_unknown']=1
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
        values=pair();values[1]['unchanged']['unknown']=18
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
        values=pair();values[1]['unchanged']['raw']=1
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_unrelated_route_drift_rejected_even_display_composes(self):
        values=pair();values[1]['kernel_routes']+=';different-math'
        values[1]['batch_prefill_kernel_routes']=v4.BATCH_PREFIX+values[1]['kernel_routes']+v4.BATCH_ILP_SUFFIX
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
    def test_missing_fields_or_wrong_source_type_rejected(self):
        values=pair();values[0].pop('batch_prefill_kernel_routes')
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(values,SOURCE)
        with self.assertRaises(ValueError):v4.normalize_registered_identity_pair(pair(),'0'*64)

if __name__=='__main__':unittest.main()
