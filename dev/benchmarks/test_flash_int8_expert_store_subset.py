from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("subset", TOOLS / "flash_int8_expert_store_subset.py")
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


class SubsetTests(unittest.TestCase):
    def test_arbitrary_compact_rank_order(self):
        self.assertEqual(subset.subset_ranks([0, 7, 19, 127, 501], [7, 127, 501]), [1, 3, 4])

    def test_not_a_subset_rejected(self):
        with self.assertRaises(ValueError): subset.subset_ranks([0, 7], [0, 19])

    def test_duplicate_unsorted_boolean_ids_rejected(self):
        for source, selected in (([7, 0], [0]), ([0, 0], [0]), ([0, 1], [1, 0]),
                                 ([0, 1], [1, 1]), ([False, 7], [7]), ([0, 7], [False])):
            with self.subTest(source=source, selected=selected), self.assertRaises(ValueError):
                subset.subset_ranks(source, selected)

    def test_inventory_enum_and_consistency(self):
        for count in (32, 64, 128):
            plan = {"schema": subset.store.PLAN_SCHEMA, "source_identity": subset.store.EXPECTED_SOURCE_IDENTITY,
                    "requested_limit": count, "selected_experts": [list(range(count)) for _ in range(48)]}
            self.assertEqual(len(subset.selected_inventory(plan)[47]), count)
        for count in (1, 31, 33, 65, 127):
            plan = {"schema": subset.store.PLAN_SCHEMA, "source_identity": subset.store.EXPECTED_SOURCE_IDENTITY,
                    "requested_limit": count, "selected_experts": [list(range(count)) for _ in range(48)]}
            with self.subTest(count=count), self.assertRaises(ValueError): subset.selected_inventory(plan)
        plan["requested_limit"] = 128
        plan["selected_experts"] = [list(range(128)) for _ in range(48)]
        plan["selected_experts"][7] = list(range(64))
        with self.assertRaises(ValueError): subset.selected_inventory(plan)

    def test_exact_signed_code_and_scale_byte_copy(self):
        # Includes every byte, signed INT8 endpoints, zeros, and arbitrary F32
        # bit patterns. The copy primitive must never interpret those words.
        raw = b"prefix!" + bytes(range(256)) * 4
        with tempfile.TemporaryFile() as source:
            source.write(raw); source.flush()
            output = io.BytesIO(); full = hashlib.sha256(b"padding")
            digest = subset.copy_plane(source.fileno(), output, source_offset=7,
                expert_bytes=64, ranks=[1, 3, 7, 15], file_digest=full)
            expected = b"".join(raw[7 + rank * 64:7 + (rank + 1) * 64] for rank in [1, 3, 7, 15])
            self.assertEqual(output.getvalue(), expected)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            self.assertEqual(full.hexdigest(), hashlib.sha256(b"padding" + expected).hexdigest())

    def test_truncated_source_rejected(self):
        with tempfile.TemporaryFile() as source:
            source.write(b"abc"); source.flush()
            with self.assertRaises(ValueError):
                subset.copy_plane(source.fileno(), io.BytesIO(), source_offset=0,
                    expert_bytes=4, ranks=[0], file_digest=hashlib.sha256())

    def test_short_destination_rejected(self):
        class Short(io.BytesIO):
            def write(self, value): return len(value) - 1
        with tempfile.TemporaryFile() as source:
            source.write(b"abcd"); source.flush()
            with self.assertRaises(ValueError):
                subset.copy_plane(source.fileno(), Short(), source_offset=0,
                    expert_bytes=4, ranks=[0], file_digest=hashlib.sha256())


if __name__ == "__main__": unittest.main()
