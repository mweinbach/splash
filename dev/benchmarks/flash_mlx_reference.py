"""ROOT-RUN installed oMLX/MLX Flash-Next greedy reference.

--help and --preflight use only the standard library. Actual model loading and
GPU operators require --run-gpu and must be serialized by the root coordinator.
No server, saved model settings, or checkpoint files are modified. PLE is bound
to the installed disk-mmap implementation before model construction; MTP stays
off. Fresh request-local GDN/QSA/PLE caches are necessary for autoregression,
while cross-request prefix caches and serving caches are absent.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
import time

APP = Path("/Applications/oMLX.app/Contents/Resources")
DEFAULT_MODEL = Path.home() / ".omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = PROJECT / "build/release/flash/native-profile-prompt.json"
DEFAULT_BUNDLE = PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
VENDOR = APP / "omlx/patches/mlx_vlm_qwen4_exp_compat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def preflight(model: Path, prompt: Path, bundle: Path) -> tuple[list[int], dict, dict]:
    model, prompt, bundle = (p.expanduser().resolve() for p in (model, prompt, bundle))
    source_config = model / "config.json"
    config = json.loads(source_config.read_bytes())
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("reference requires the installed Qwen4Exp checkpoint")
    tokens = json.loads(prompt.read_bytes())
    vocabulary = config["text_config"]["vocab_size"]
    if not isinstance(tokens, list) or not tokens or any(
        type(token) is not int or token < 0 or token >= vocabulary for token in tokens
    ):
        raise ValueError("prompt must be a nonempty JSON array of valid UInt32 token IDs")
    raw_manifest = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if (bundle / "manifest.sha256").read_text().strip() != hashlib.sha256(raw_manifest).hexdigest():
        raise ValueError("derived source manifest checksum differs")
    if manifest.get("schema") != "splash-local-qwen4-affine-v1":
        raise ValueError("unsupported derived source manifest")
    records = [
        {"path": item["source_path"], "bytes": item["source_bytes"], "sha256": item["source_sha256"]}
        for item in manifest["shards"]
    ] + [{key: item[key] for key in ("path", "bytes", "sha256")} for item in manifest["small_files"]]
    identity = hashlib.sha256(json.dumps(
        {"schema": manifest["schema"], "source_files": sorted(records, key=lambda item: item["path"])},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if identity != manifest["source_identity_sha256"]:
        raise ValueError("derived source identity does not match manifest records")
    small = {}
    files = {}
    for item in manifest["small_files"]:
        source = model / item["path"]
        before = snapshot(source)
        if before[2] != item["bytes"] or sha256(source) != item["sha256"] or snapshot(source) != before:
            raise ValueError(f"original source file differs from imported source: {source.name}")
        small[item["path"]] = item["sha256"]
        files[str(source)] = before
    expected = {item["source_path"]: item for item in manifest["shards"]}
    observed = {path.name: path for path in model.glob("*.safetensors")}
    if expected.keys() != observed.keys():
        raise ValueError("original safetensors set differs from imported source")
    for name, path in sorted(observed.items()):
        if path.stat().st_size != expected[name]["source_bytes"]:
            raise ValueError(f"original safetensors size differs: {name}")
        files[str(path)] = snapshot(path)
    watched = [prompt, bundle / "manifest.json", bundle / "manifest.sha256"]
    for path in watched:
        files[str(path)] = snapshot(path)
    vendor_files = [
        VENDOR / "__init__.py",
        VENDOR / "vendor/mlx_vlm/models/qwen4_exp/language.py",
        VENDOR / "vendor/mlx_vlm/models/qwen4_exp/qwen4_exp.py",
        VENDOR / "vendor/mlx_vlm/models/qwen4_exp/config.py",
        APP / "omlx/engine/vlm.py",
    ]
    report = {
        "schema": "splash-flash-installed-mlx-reference-v1",
        "source_model_directory": str(model),
        "derived_directory": str(bundle),
        "source_identity": manifest["source_identity_sha256"],
        "source_identity_provenance": "preserved derived import manifest",
        "source_verification": {
            "small_file_sha256": True,
            "source_shard_sizes": True,
            "source_shard_full_hashes_recomputed": False,
            "note": "The importer previously hashed full shards. This oracle does not reread the 99 GiB source merely to hash it and warm PLE pages.",
        },
        "source_small_file_sha256": small,
        "bundle_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "installed_reference_file_sha256": {str(p): sha256(p) for p in vendor_files},
        "oracle_sha256": sha256(Path(__file__)),
        "prompt_path": str(prompt),
        "prompt_file_sha256": sha256(prompt),
        "prompt_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest(),
        "prompt_tokens": tokens,
        "prompt_token_count": len(tokens),
        "mtp_enabled": False,
        "prefix_cache_enabled": False,
        "serving_cache_enabled": False,
        "fresh_request_local_cache": True,
        "ple_mode": "mmap",
        "dense_ple_table_materialized": False,
        "gpu_seconds": None,
        "gpu_timing_available": False,
        "gpu_timing_reason": "Installed MLX 0.32.2 exposes no Python command-buffer timestamp collector; synchronized perf_counter is host wall time.",
        "timing_scope": "per language-model call through mx.eval + mx.synchronize; excludes loading, sampling and reporting",
        "gpu_executed": False,
    }
    return tokens, files, report


def require_unchanged(files: dict) -> None:
    for name, before in files.items():
        if snapshot(Path(name)) != before:
            raise RuntimeError(f"preserved source/input changed during oracle: {name}")


@contextlib.contextmanager
def readonly_loader(utils, mx):
    """Prevent upstream unsupported-dtype fallback from opening shards r+b.

    This checkpoint's BF16/U32/I64/F32 storage is supported by the installed
    MLX loader. An unsupported dtype fails rather than rewriting a header.
    Actual model sanitization and mixed quantization still use installed code.
    """
    previous = utils._load_safetensors
    utils._load_safetensors = lambda path: mx.load(path)
    try:
        yield
    finally:
        utils._load_safetensors = previous


def run(args, tokens: list[int], files: dict, report: dict) -> dict:
    if not args.run_gpu:
        raise RuntimeError("actual MLX reference requires explicit --run-gpu")
    # Nothing above this gate imports or initializes MLX.
    import mlx.core as mx
    import mlx_vlm.utils as utils
    from mlx.utils import tree_flatten
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

    mode = configure_qwen4_exp_runtime(args.model, mode="mmap", mtp_enabled=False)
    from mlx_vlm.models.qwen4_exp.language import (
        DiskBackedShardedEmbedding, get_mtp_runtime, get_ple_runtime_mode,
    )
    from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load

    if mode != "mmap" or get_ple_runtime_mode() != "mmap" or get_mtp_runtime().enabled:
        raise RuntimeError("installed PLE/MTP runtime binding differs from requested reference")
    deadline = time.perf_counter() + args.timeout_seconds

    def check_deadline():
        if time.perf_counter() >= deadline:
            raise TimeoutError("reference exceeded its wall budget between synchronized calls")

    report["mlx_version"] = importlib.metadata.version("mlx")
    report["python_executable"] = sys.executable
    report["installed_model_class"] = "mlx_vlm.models.qwen4_exp.Model"
    report["sampling"] = {"temperature": 0, "top_k": 0, "top_p": 1}
    report["maximum_output_tokens"] = args.max_tokens
    report["eos_token_ids"] = [248046, 248044]
    report["prefill_chunk_rows"] = args.prefill_rows
    report["startup_warmup"] = False
    model_path = args.model.expanduser().resolve()
    loaded_began = time.perf_counter()
    # Context is the actual installed application hook: hide only MLX metadata
    # so sanitize runs BEFORE per-module quantization selection/RMS centering.
    with (
        contextlib.redirect_stdout(sys.stderr),
        mx.stream(mx.gpu),
        readonly_loader(utils, mx),
        _force_qwen4_exp_sanitize_on_load(model_path),
    ):
        model = utils.load_model(model_path, lazy=True, strict=True)
        language = model.language_model
        if hasattr(model, "mtp") or language.get_mtp_module() is not None:
            raise RuntimeError("MTP head was attached to ordinary autoregression reference")
        ple_embeddings = [
            layer.ple.ple_embedding.ngram_embedding
            for layer in language.model.layers if hasattr(layer, "ple")
        ]
        if not ple_embeddings or any(not isinstance(p, DiskBackedShardedEmbedding) for p in ple_embeddings):
            raise RuntimeError("installed model did not retain disk-backed PLE")
        parameters = tree_flatten(model.parameters())
        table_parameters = [
            name for name, value in parameters
            if ".ngram_embedding." in name and not name.endswith(".weight_scale")
        ]
        if table_parameters:
            raise RuntimeError("dense PLE table unexpectedly entered model parameters")
        report["resident_parameter_bytes"] = sum(value.nbytes for _, value in parameters)
        report["ple_embedding_classes"] = [type(p).__name__ for p in ple_embeddings]
        report["ple_parameter_table_tensors"] = len(table_parameters)
        mx.eval(model.parameters())
        mx.synchronize()
    report["model_load_eval_wall_seconds"] = time.perf_counter() - loaded_began
    check_deadline()
    require_unchanged(files)

    cache = language.make_cache()
    if len(cache) != 48:
        raise RuntimeError("reference did not create all 48 request-local layer caches")
    report["request_cache_classes"] = [type(value).__name__ for value in cache]
    output_tokens = []
    calls = []
    generation_began = time.perf_counter()
    logical_length = 0

    def forward(ids, kind):
        nonlocal logical_length
        check_deadline()
        with contextlib.redirect_stdout(sys.stderr), mx.stream(mx.gpu):
            inputs = mx.array([ids], dtype=mx.uint32)
            positions = mx.arange(logical_length, logical_length + len(ids), dtype=mx.int64)[None, :]
            # Canonical text positions are the installed adapter's B1 contract.
            began = time.perf_counter()
            result = language(inputs, cache=cache, position_ids=positions)
            logits = result.logits[:, -1, :]
            mx.eval(logits, [value.state for value in cache])
            mx.synchronize()
            wall = time.perf_counter() - began
        logical_length += len(ids)
        calls.append({"kind": kind, "rows": len(ids), "logical_length": logical_length,
                      "wall_seconds": wall, "gpu_seconds": None,
                      "installed_lm_head_rows": len(ids)})
        print(json.dumps({"reference_progress": kind, "rows": len(ids),
                          "logical_length": logical_length, "wall_seconds": wall}), file=sys.stderr, flush=True)
        return logits

    logits = None
    for offset in range(0, len(tokens), args.prefill_rows):
        logits = forward(tokens[offset:offset + args.prefill_rows], "prefill")
    finish = "length"
    for index in range(args.max_tokens):
        check_deadline()
        with mx.stream(mx.gpu):
            chosen = mx.argmax(logits, axis=-1)
            mx.eval(chosen)
            token = int(chosen.item())
        output_tokens.append(token)
        if token in report["eos_token_ids"]:
            finish = "stop"
            break
        if index + 1 < args.max_tokens:
            logits = forward([token], "decode")
    report.update({
        "gpu_executed": True,
        "valid": True,
        "finish_reason": finish,
        "output_tokens": output_tokens,
        "output_token_count": len(output_tokens),
        "output_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(output_tokens)}I", *output_tokens)).hexdigest(),
        "logical_length": logical_length,
        "calls": calls,
        "prefill_wall_seconds": [c["wall_seconds"] for c in calls if c["kind"] == "prefill"],
        "decode_wall_seconds": [c["wall_seconds"] for c in calls if c["kind"] == "decode"],
        "wall_seconds": sum(c["wall_seconds"] for c in calls),
        "generation_observed_wall_seconds": time.perf_counter() - generation_began,
        "memory": {
            "active_bytes": mx.get_active_memory(),
            "peak_bytes": mx.get_peak_memory(),
            "cache_bytes": mx.get_cache_memory(),
        },
        "ple_last_rows_read": [p.rows_read for p in ple_embeddings],
        "ple_last_uploads": [p.last_uploads for p in ple_embeddings],
        "source_unchanged": True,
    })
    require_unchanged(files)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-rows", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="CPU-only validation; does not import MLX")
    args = parser.parse_args()
    if args.run_gpu == args.preflight:
        parser.error("choose exactly one of --preflight or --run-gpu")
    if not 1 <= args.max_tokens <= 256 or not 1 <= args.prefill_rows <= 128 or not args.timeout_seconds > 0:
        parser.error("tokens must be 1..256, prefill rows 1..128 and timeout positive")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError("choose a fresh private output JSON path")
    for preserved in (args.model, args.bundle):
        if args.output.is_relative_to(preserved.expanduser().resolve()):
            raise ValueError("reference output must not overlap checkpoint or derived bundle")
    tokens, files, report = preflight(args.model, args.prompt, args.bundle)
    report["preflight"] = True
    if args.run_gpu:
        report = run(args, tokens, files, report)
    else:
        require_unchanged(files)
        report["valid"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"valid": report["valid"], "gpu_executed": report["gpu_executed"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
