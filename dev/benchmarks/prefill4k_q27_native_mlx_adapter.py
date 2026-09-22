"""Private native-coefficient Q27 bridge to stock MLX-LM operators.

Import/preflight uses only stdlib. Loading requires an explicit Root GPU gate.
Original upstream A_log/centered norm words are not claimed recovered. Native
operative norm gains and F32 a_scale are exact; only gate construction bridges
pre-exponentiated decay to the unchanged stock GDN state-update kernel.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def preflight(model_dir: Path):
    model_dir = Path(model_dir).resolve()
    config = json.loads((model_dir / "config.json").read_bytes())
    bridge = config.get("splash_native_coefficient_bridge", {})
    if (config.get("model_type") != "splash_native_qwen3_5_bridge" or not bridge.get("required") or
            bridge.get("version") != 1 or bridge.get("original_model_type") != "qwen3_5" or
            not bridge.get("A_log_placeholders_unused") or bridge.get("original_raw_parameters_recovered")):
        raise ValueError("model is not the declared private native-coefficient bridge")
    text = config["text_config"]
    if (text["num_hidden_layers"] != 64 or text["hidden_size"] != 5120 or
            config.get("quantization") != {"bits": 4, "group_size": 64, "mode": "affine"}):
        raise ValueError("native Q27 configuration differs")
    decay_path = (model_dir / bridge["native_decay"]).resolve()
    if not decay_path.is_relative_to(model_dir / "native-bridge") or not decay_path.is_file():
        raise ValueError("mandatory native GDN decay sidecar is missing")
    index = json.loads((model_dir / "model.safetensors.index.json").read_bytes())
    if len(index["weight_map"]) != 1847 or any("mtp." in key for key in index["weight_map"]):
        raise ValueError("native target key coverage differs")
    manifest = json.loads((model_dir / "native-coefficient-manifest.json").read_bytes())
    if (not manifest.get("full_conversion_executed") or not manifest.get("quantized_code_scale_bias_bit_exact") or
            not manifest.get("native_operative_gains_preserved") or not manifest.get("gdn_native_decay_bridge_mandatory")):
        raise ValueError("native coefficient certificates incomplete")
    decay_record = next((entry for entry in manifest["output_files"] if
                         Path(entry["path"]).name == "gdn-decay.safetensors"), None)
    if (decay_record is None or decay_path.stat().st_size != decay_record["bytes"] or
            hashlib.sha256(decay_path.read_bytes()).hexdigest() != decay_record["sha256"]):
        raise ValueError("native decay sidecar byte certificate differs")
    for record in manifest["output_files"]:
        recorded = Path(record["path"])
        current = decay_path if recorded.name == "gdn-decay.safetensors" else model_dir / recorded.name
        if not current.is_file() or current.stat().st_size != record["bytes"]:
            raise ValueError("native coefficient shard extent differs")
    return config, {"source_label": "native operative Q27 coefficients with stock MLX-LM qwen3_5 operators",
                    "upstream_raw_parameter_recovery": False, "coefficient_requantization": False,
                    "norm_gain_source": "native operative BF16 gain, unchanged",
                    "decay_source": "native pre-exponentiated F32 a_scale, unchanged",
                    "stock_GDN_state_update_kernel_unchanged": True,
                    "stock_compute_g_bridge": "exp(native_a_scale.float32 * softplus(a + dt_bias))",
                    "mandatory_bridge": True, "plain_stock_load_rejected_by_custom_model_type": True,
                    "native_source_manifest_sha256": manifest["manifest_sha256"],
                    "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class NativeDecayBinding:
    def __init__(self, module, original, replacement, arrays, report):
        self.module, self.original, self.replacement = module, original, replacement
        self.arrays, self.report = arrays, report
    def __enter__(self):
        return self
    def __exit__(self, *args):
        if self.module.compute_g is self.replacement:
            self.module.compute_g = self.original


def load_model(model_dir: Path, *, run_root_gpu=False, lazy=True):
    config, report = preflight(model_dir)
    if not run_root_gpu:
        raise RuntimeError("loading/evaluating MLX requires Root's explicit GPU slot")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import load_model as stock_load
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from mlx_lm.models import gated_delta

    def stock_classes(config):
        if config.get("model_type") != "splash_native_qwen3_5_bridge":
            raise ValueError("unexpected model type at explicit bridge load")
        # Only the in-memory loader configuration changes. The private artifact
        # remains un-loadable through an accidental unbridged stock CLI call.
        config["model_type"] = "qwen3_5"
        return Model, ModelArgs

    model, loaded_config = stock_load(Path(model_dir), lazy=lazy, strict=True, get_model_classes=stock_classes)
    sidecar = mx.load(Path(model_dir) / config["splash_native_coefficient_bridge"]["native_decay"])
    expected = {f"language_model.model.layers.{index}.linear_attn.native_a_scale" for index in range(64) if (index + 1) % 4}
    if set(sidecar) != expected:
        raise ValueError("native decay sidecar key coverage differs")
    bindings = {}
    for index, layer in enumerate(model.layers):
        if not layer.is_linear:
            continue
        parameter = layer.linear_attn.A_log
        scale = sidecar[f"language_model.model.layers.{index}.linear_attn.native_a_scale"]
        if tuple(scale.shape) != (48,) or scale.dtype != mx.float32 or tuple(parameter.shape) != (48,):
            raise ValueError("native decay shape/dtype differs")
        bindings[id(parameter)] = scale
    if len(bindings) != 48:
        raise ValueError("all native GDN layers must bind before any forward")

    @mx.compile
    def native_g(scale, a, dt_bias):
        return mx.exp(scale.astype(mx.float32) * nn.softplus(a + dt_bias))

    original = gated_delta.compute_g
    def native_compute_g(A_log_placeholder, a, dt_bias):
        scale = bindings.get(id(A_log_placeholder))
        if scale is None:
            raise RuntimeError("unbound GDN parameter or whole-model tracing attempted; no inverse A_log fallback")
        return native_g(scale, a, dt_bias)
    gated_delta.compute_g = native_compute_g
    report.update(native_decay_layers_bound=len(bindings), source_parameter_names_unchanged=True,
                  model_class=type(model).__module__ + "." + type(model).__name__,
                  original_compute_g_module=getattr(original, "__module__", "mlx.core.compile wrapper"))
    handle = NativeDecayBinding(gated_delta, original, native_compute_g, sidecar, report)
    return model, loaded_config, handle
