"""CPU tests for the private INT8 pilot's actual numerical and file contracts."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("flash_expert_int8_convert", Path(__file__).resolve().parents[1] / "tools" / "flash_expert_int8_convert.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ExpertInt8ConversionTests(unittest.TestCase):
    def test_bf16_nearest_even_and_signed_zero(self):
        words = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000, 0, 0x80000000], dtype=np.uint32)
        np.testing.assert_array_equal(converter.f32_to_bf16(words.view(np.float32)), [0x3F80, 0x3F82, 0xBF80, 0xBF82, 0, 0x8000])

    def test_bf16_finite_roundtrip(self):
        bits = np.arange(65536, dtype=np.uint16)
        finite = bits[(bits & 0x7F80) != 0x7F80]
        np.testing.assert_array_equal(converter.f32_to_bf16(converter.bf16_to_f32(finite)), finite)

    def test_bf16_rejects_nonfinite(self):
        with self.assertRaises(converter.ConversionError):
            converter.f32_to_bf16(np.array([np.inf], dtype=np.float32))

    def test_q4_nibble_order_negative_scale_and_separate_f32_operations(self):
        codes = np.tile(np.arange(16, dtype=np.uint32), 4).reshape(2, 32)
        packed = np.sum(codes.reshape(2, 4, 8) << (np.arange(8, dtype=np.uint32) * 4), axis=-1, dtype=np.uint32)
        scale_values = np.array([[0.015625, -0.00390625], [-0.03125, 0.0078125]], dtype=np.float32)
        bias_values = np.array([[-0.25, 0.1], [0.4, -0.1]], dtype=np.float32)
        scales, biases = converter.f32_to_bf16(scale_values), converter.f32_to_bf16(bias_values)
        result = converter.reconstruct_q4_bf16(packed, scales, biases, group_size=16)
        expected = np.empty((2, 32), dtype=np.uint16)
        for row in range(2):
            for column in range(32):
                sf = converter.bf16_to_f32(scales[row, column // 16])
                bias = converter.bf16_to_f32(biases[row, column // 16])
                value = np.float32(np.float32(codes[row, column]) * sf)
                value = np.float32(value + bias)
                expected[row, column] = converter.f32_to_bf16(value)
        np.testing.assert_array_equal(result, expected)

    def test_q4_invalid_shape_rejected(self):
        with self.assertRaises(converter.ConversionError):
            converter.reconstruct_q4_bf16(np.zeros((2, 8), dtype=np.uint32), np.zeros((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=np.uint16))

    def test_zero_row_and_symmetric_endpoints(self):
        reference = converter.f32_to_bf16(np.array([[0, 0, 0], [-2, 1, 2]], dtype=np.float32))
        codes, scales = converter.symmetric_int8(reference)
        np.testing.assert_array_equal(codes[0], [0, 0, 0])
        np.testing.assert_array_equal(codes[1], [-127, 64, 127])
        self.assertEqual(scales[0, 0], np.float32(1))
        self.assertEqual(scales.dtype, np.dtype("float32"))
        self.assertFalse((codes == -128).any())

    def test_g64_scales_limit_local_error_vs_row_outlier(self):
        values = np.tile(np.linspace(-0.01, 0.01, 64, dtype=np.float32), 2).reshape(1, 128)
        values[0, 0] = 2
        reference = converter.f32_to_bf16(values)
        row_codes, row_scales = converter.symmetric_int8(reference)
        grouped_codes, grouped_scales = converter.symmetric_int8(reference, group_size=64)
        self.assertEqual(row_scales.shape, (1, 1))
        self.assertEqual(grouped_scales.shape, (1, 2))
        row = converter.reconstruct_int8(row_codes, row_scales)
        grouped = converter.reconstruct_int8(grouped_codes, grouped_scales, group_size=64)
        self.assertLess(converter.error_metrics(converter.bf16_to_f32(reference), grouped)["rl2"], converter.error_metrics(converter.bf16_to_f32(reference), row)["rl2"])

    def test_3d_expert_layout_and_error_bound(self):
        values = np.random.default_rng(58).standard_normal((2, 3, 128)).astype(np.float32)
        reference = converter.f32_to_bf16(values)
        codes, scales = converter.symmetric_int8(reference)
        actual = converter.reconstruct_int8(codes, scales)
        self.assertEqual(scales.shape, (2, 3, 1))
        error = np.abs(actual - converter.bf16_to_f32(reference))
        self.assertTrue((error <= np.repeat(scales / 2, 128, axis=-1) + np.finfo(np.float32).eps * 4).all())

    def test_malformed_int8_scales_and_reserved_code_rejected(self):
        with self.assertRaises(converter.ConversionError):
            converter.reconstruct_int8(np.array([[-128]], dtype=np.int8), np.ones((1, 1), dtype=np.float32))
        with self.assertRaises(converter.ConversionError):
            converter.reconstruct_int8(np.array([[1]], dtype=np.int8), np.zeros((1, 1), dtype=np.float32))
        with self.assertRaises(converter.ConversionError):
            converter.symmetric_int8(np.zeros((1, 3), dtype=np.uint16), group_size=2)

    def test_payload_alignment_and_array_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            with path.open("wb") as stream:
                first = converter._write_aligned(stream, np.array([1, -2], dtype=np.int8))
                second = converter._write_aligned(stream, np.array([[0.25]], dtype="<f4"))
            self.assertEqual(first["offset"], 0)
            self.assertEqual(second["offset"], 16384)
            self.assertEqual(path.stat().st_size, 16388)
            self.assertEqual(first["sha256"], converter._digest_array(np.array([1, -2], dtype=np.int8)))

    def test_source_shard_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            package.mkdir()
            sibling = Path(directory) / "outside"
            sibling.write_bytes(b"abc")
            with self.assertRaises(converter.ConversionError):
                converter._source_path(package.resolve(), "../outside")

    def test_model_storage_estimate(self):
        estimate = converter.model_byte_estimate()
        self.assertEqual(estimate["int8_rowwise_f32_scale_bytes"], 123697889280)
        self.assertEqual(estimate["source_q4_g64_bf16_scale_bias_bytes"], 69363302400)
        self.assertLess(estimate["int8_rowwise_f32_scale_bytes"], estimate["bf16_coefficients_bytes"])

    def test_existing_output_and_expert_bool_rejected_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(converter.ConversionError):
                converter.convert(path / "source", path, experts=(0,))
            with self.assertRaises(converter.ConversionError):
                converter.convert(path / "source", path / "new", experts=(True,))

    def test_cpu_proxy_identity_preserves_bf16_outputs(self):
        source = converter.f32_to_bf16(np.random.default_rng(17).standard_normal((2, 4, 16)).astype(np.float32))
        report = converter._output_proxy(source, {"identity": converter.bf16_to_f32(source)}, seed=4)
        self.assertEqual(report["identity"]["bf16_output_bits_changed"], 0)
        self.assertEqual(report["identity"]["before_output_bf16"]["rl2"], 0)

    def test_scale_fitting_zero_rounds_preserves_default_payload(self):
        source = converter.f32_to_bf16(np.random.default_rng(22).standard_normal((2, 4, 64)).astype(np.float32))
        original = converter.symmetric_int8(source)
        explicit = converter.symmetric_int8(source, fit_rounds=0, allow_clipping=True)
        np.testing.assert_array_equal(original[0], explicit[0])
        np.testing.assert_array_equal(original[1], explicit[1])

    def test_least_squares_fitting_reduces_error_and_preserves_constraints(self):
        source = converter.f32_to_bf16(np.random.default_rng(71).standard_normal((2, 16, 128)).astype(np.float32))
        codes, scales = converter.symmetric_int8(source)
        initial_error = converter.error_metrics(converter.bf16_to_f32(source), converter.reconstruct_int8(codes, scales))["rl2"]
        for allow_clipping in (False, True):
            fitted_codes, fitted_scales = converter.symmetric_int8(source, fit_rounds=4, allow_clipping=allow_clipping)
            fitted_error = converter.error_metrics(converter.bf16_to_f32(source), converter.reconstruct_int8(fitted_codes, fitted_scales))["rl2"]
            self.assertLessEqual(fitted_error, initial_error)
            self.assertFalse((fitted_codes == -128).any())
            if not allow_clipping:
                self.assertTrue((fitted_scales >= scales).all())
        zero_codes, zero_scales = converter.symmetric_int8(np.zeros((2, 64), dtype=np.uint16), fit_rounds=4, allow_clipping=True)
        self.assertTrue((zero_codes == 0).all())
        self.assertTrue((zero_scales == 1).all())

    def test_scale_fitting_invalid_options_rejected(self):
        source = np.zeros((1, 64), dtype=np.uint16)
        for rounds in (True, -1, 5):
            with self.assertRaises(converter.ConversionError):
                converter.symmetric_int8(source, fit_rounds=rounds)
        with self.assertRaises(converter.ConversionError):
            converter.symmetric_int8(source, allow_clipping=1)


if __name__ == "__main__":
    unittest.main()
