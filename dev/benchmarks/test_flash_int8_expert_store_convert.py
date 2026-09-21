"""CPU-only guards and real streaming layout tests for the selected store."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("flash_int8_expert_store_convert", Path(__file__).resolve().parents[1] / "tools" / "flash_int8_expert_store_convert.py")
store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store)


class SelectedStoreTests(unittest.TestCase):
    def plan(self):
        return {"schema": store.PLAN_SCHEMA, "source_identity": store.EXPECTED_SOURCE_IDENTITY, "requested_limit": 128, "selected_experts": [[0, 4, 127] for _ in range(48)]}

    def test_plan_strict_sorted_ids_and_source(self):
        p = self.plan()
        self.assertEqual(store.validate_plan(p, store.EXPECTED_SOURCE_IDENTITY)[0], [0, 4, 127])
        for ids in ([True], [0, 0], [4, 0], [-1], [512], [], list(range(129))):
            broken = self.plan()
            broken["selected_experts"][4] = ids
            with self.assertRaises(store.StoreError):
                store.validate_plan(broken, store.EXPECTED_SOURCE_IDENTITY)
        p["source_identity"] = "0" * 64
        with self.assertRaises(store.StoreError):
            store.validate_plan(p, store.EXPECTED_SOURCE_IDENTITY)

    def test_plan_exact_layers_and_boolean_limit(self):
        for layers in (47, 49):
            p = self.plan()
            p["selected_experts"] = [[0] for _ in range(layers)]
            with self.assertRaises(store.StoreError):
                store.validate_plan(p, store.EXPECTED_SOURCE_IDENTITY)
        p = self.plan()
        p["requested_limit"] = True
        with self.assertRaises(store.StoreError):
            store.validate_plan(p, store.EXPECTED_SOURCE_IDENTITY)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"schema":1,"schema":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(store.StoreError):
                store._json(raw)

    def test_six_planes_and_final_file_alignment(self):
        layer = store.layout_layer(0, 128)
        self.assertEqual(layer["bytes"], 631111680)
        self.assertEqual(sum(store.layout_layer(i, 128)["bytes"] for i in range(48)), 30293360640)
        previous_end = 0
        for projection in layer["projections"].values():
            for name in ("codes", "scales"):
                plane = projection[name]
                self.assertEqual(plane["offset"] % 16384, 0)
                self.assertGreaterEqual(plane["offset"], previous_end)
                previous_end = plane["offset"] + plane["length"]
        self.assertEqual(layer["bytes"] % 16384, 0)

    def test_non128_counts_still_final_aligned(self):
        layer = store.layout_layer(3, 7)
        self.assertEqual(layer["path"], "layer-03.bin")
        self.assertEqual(layer["bytes"] % 16384, 0)
        for projection in layer["projections"].values():
            self.assertEqual(projection["codes"]["shape"][0], 7)
            self.assertEqual(projection["scales"]["shape"], [7, projection["dimensions"][1]])

    def test_invalid_layer_geometry_rejected(self):
        for layer, count in ((True, 1), (48, 1), (0, 0), (0, 129)):
            with self.assertRaises(store.StoreError):
                store.layout_layer(layer, count)

    def test_existing_or_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            source, output = parent / "source", parent / "output"
            source.mkdir()
            output.mkdir()
            plan = parent / "plan.json"
            with self.assertRaises(store.StoreError):
                store.check_destination(source, output, plan, 1)
            with self.assertRaises(store.StoreError):
                store.check_destination(source, source / "nested", plan, 1)

    def test_insufficient_disk_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory).resolve()
            with self.assertRaises(store.StoreError):
                store.check_destination(p / "source", p / "output", p / "plan", 1, disk_usage=lambda _: SimpleNamespace(free=store.MIN_FREE_BYTES - 1))
            self.assertFalse((p / "output").exists())

    def test_source_escape_truncation_and_offset_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory).resolve()
            path = p / "source.bin"
            path.write_bytes(bytes(16384))
            descriptor = {"shard": "source.bin", "dtype": "U32", "shape": [512, 1, 8], "offset": 0, "length": 16384}
            shards = {"source.bin": {"bytes": 16384, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            store.validate_tensor(p, descriptor, (512, 1, 8), "U32", shards, {})
            for key, value in (("offset", True), ("offset", 4), ("offset", 16384), ("length", 10), ("dtype", "F32")):
                bad = {**descriptor, key: value}
                with self.assertRaises(store.StoreError):
                    store.validate_tensor(p, bad, (512, 1, 8), "U32", shards, {})
            with self.assertRaises(store.StoreError):
                store._source_file(p, "../outside.bin")

    def test_streamed_batches_and_plane_hash_match_whole_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            rng = np.random.default_rng(25)
            words = rng.integers(0, 2**32, size=(12, 2, 16), dtype=np.uint32)
            sf = store._PILOT.f32_to_bf16(rng.uniform(-0.03, 0.03, (12, 2, 2)).astype(np.float32))
            bias = store._PILOT.f32_to_bf16(rng.uniform(-0.1, 0.1, (12, 2, 2)).astype(np.float32))
            planes = {}
            for name, values, dtype in (("weight", words, "<u4"), ("scales", sf, "<u2"), ("biases", bias, "<u2")):
                path = p / f"{name}.bin"
                path.write_bytes(values.tobytes())
                planes[name] = {"path": path, "shape": values.shape, "dtype": dtype, "offset": 0}
            experts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
            layer = store.layout_layer(0, len(experts), projections={"gate_proj": (2, 128)})
            layout = layer["projections"]["gate_proj"]
            target = p / "layer.bin"
            with target.open("w+b") as stream:
                stream.truncate(layer["bytes"])
                self.assertEqual(store.write_projection(stream, layout, planes, experts), 8)
            reference = store._PILOT.reconstruct_q4_bf16(words[experts], sf[experts], bias[experts])
            codes, scales = store._PILOT.symmetric_int8(reference)
            raw = target.read_bytes()
            for name, values in (("codes", codes), ("scales", scales.reshape(len(experts), 2))):
                a = layout[name]
                expected = values.tobytes()
                self.assertEqual(raw[a["offset"] : a["offset"] + a["length"]], expected)
                self.assertEqual(a["sha256"], hashlib.sha256(expected).hexdigest())
            self.assertEqual(raw[layout["codes"]["length"] : layout["scales"]["offset"]], bytes(layout["scales"]["offset"] - layout["codes"]["length"]))


if __name__ == "__main__":
    unittest.main()
