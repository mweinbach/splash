"""Prepare or ROOT-RUN an isolated installed-oMLX Flash-Next HTTP baseline.

prepare is CPU-only. serve imports the actual installed app, loads weights and
performs GPU work; only the root coordinator may run it. Existing saved settings,
cluster shims/manifests and running services remain outside this private process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
MODEL = Path.home() / ".omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
MODEL_ID = MODEL.name
APP = Path("/Applications/oMLX.app/Contents/Resources")
WATCHED = [Path.home() / ".omlx" / name for name in
           ("settings.json", "model_settings.json", "model_profiles.json", "bin/omlx-cluster-python")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def json_write(path, document):
    with path.open("x") as file:
        json.dump(document, file, indent=2)
        file.write("\n")


def prepare(args):
    root = args.runtime.expanduser().resolve()
    original = Path.home() / ".omlx"
    if root.exists() or root.is_relative_to(original):
        raise ValueError("choose a fresh private runtime outside ~/.omlx")
    model = args.model.expanduser().resolve()
    if json.loads((model / "config.json").read_text()).get("model_type") != "qwen4_exp":
        raise ValueError("expected installed local Qwen4Exp checkpoint")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    root.mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / model.name).symlink_to(model, target_is_directory=True)
    settings = {
        "version": "1.0",
        "server": {
            "host": "127.0.0.1", "port": args.port, "log_level": "info",
            "burst_decode_mode": "balanced", "distributed_inference_enabled": False,
            "server_aliases": ["127.0.0.1"], "sse_keepalive_mode": "comment",
        },
        "model": {"model_dirs": [str(root / "models")], "model_dir": str(root / "models")},
        "scheduler": {
            "max_concurrent_requests": 4, "chunked_prefill": False,
            "prefill_priority": "context", "decode_fairness": False,
        },
        "cache": {
            "enabled": False, "hot_cache_only": False, "hot_cache_max_size": "0",
            "gdn_ssd_split_enabled": False, "ane_compile_cache": False,
        },
        "sampling": {
            "max_context_window": 8192, "max_tokens": 8192,
            "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1,
        },
        "auth": {"skip_api_key_verification": True},
        "memory": {"prefill_memory_guard": True, "memory_guard_tier": "balanced"},
        "mcp": {"config_path": None, "expose_tools": False},
        "huggingface": {"hf_cache_enabled": False},
        "usage": {"usage_history": False},
    }
    model_settings = {
        "version": 1,
        "models": {model.name: {
            "qwen4_ple_ssd_offload": True,
            "mtp_enabled": args.mtp_depth == 3,
            "mtp_num_draft_tokens": 3,
            "vlm_mtp_enabled": False, "dflash_enabled": False,
            "qwen35_oq_a8_enabled": False, "moe_expert_offload_enabled": False,
            "moe_gate_up_fusion_enabled": True,
            "enable_thinking": False, "force_sampling": False,
            "max_context_window": 8192,
            "is_pinned": True, "is_default": True,
            "trust_remote_code": False,
        }},
    }
    json_write(root / "settings.json", settings)
    json_write(root / "model_settings.json", model_settings)
    preserved = {str(path): digest(path) for path in WATCHED}
    json_write(root / "preserved-user-file-hashes.json", preserved)
    launch = {
        "schema": "splash-installed-omlx-flash-baseline-v1",
        "runtime": str(root), "model": str(model), "model_id": model.name,
        "port": args.port, "mtp_depth": args.mtp_depth,
        "mtp_depth_semantics": "adaptive maximum 3" if args.mtp_depth else "ordinary autoregression",
        "installed_app": str(APP),
        "command": [str(Path.home() / ".omlx/bin/omlx-cluster-python"), str(Path(__file__).resolve()),
                    "serve", "--runtime", str(root), "--run-root-gpu"],
        "startup_side_effect_overrides": ["cluster_python_shim", "orphan_distributed_reaper"],
        "cache_enabled": False, "ple_mode": "mmap",
        "source_config_sha256": digest(model / "config.json"),
        "source_index_sha256": digest(model / "model.safetensors.index.json"),
    }
    json_write(root / "launch.json", launch)
    print(json.dumps(launch, indent=2))


def verify_preserved(root):
    records = json.loads((root / "preserved-user-file-hashes.json").read_text())
    mismatches = [name for name, expected in records.items() if digest(Path(name)) != expected]
    if mismatches:
        raise RuntimeError(f"user files changed outside private runtime: {mismatches}")


def serve(args):
    if not args.run_root_gpu:
        raise ValueError("actual server launch requires --run-root-gpu")
    root = args.runtime.expanduser().resolve()
    launch = json.loads((root / "launch.json").read_text())
    if root != Path(launch["runtime"]):
        raise ValueError("private runtime launch identity differs")
    verify_preserved(root)
    for key in list(os.environ):
        if key.startswith("OMLX_"):
            os.environ.pop(key)
    os.environ["OMLX_BASE_PATH"] = str(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMLX_BONJOUR"] = "0"
    os.environ["OMLX_DISCOVERY"] = "0"
    # These are the only startup hooks ignoring the private base directory.
    # Override in this process only, before lifespan imports their functions.
    import omlx.cluster.worker_shim as shim
    import omlx.cluster.launch as cluster_launch
    shim.ensure_cluster_python_shim = lambda **kwargs: None
    cluster_launch.reap_orphaned_launches = lambda *a, **k: {"reaped": [], "failures": [], "active": []}
    import mlx.core as mx
    import mlx_vlm.utils as vlm_utils
    # Unsupported source dtype fails rather than invoking the upstream r+b
    # shard-header fallback. All inspected checkpoint dtypes are supported.
    vlm_utils._load_safetensors = lambda path: mx.load(path)
    import omlx.server as server
    from omlx import cli

    @server.app.get("/__flash_baseline/runtime")
    async def runtime():
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp.language import get_mtp_runtime, get_ple_runtime_mode
        state = server._server_state
        pool = state.engine_pool
        entry = pool._entries.get(launch["model_id"]) if pool is not None else None
        engine = entry.engine if entry is not None else None
        model = getattr(engine, "_vlm_model", None)
        adapter = getattr(engine, "_adapter", None)
        core = getattr(getattr(engine, "_engine", None), "engine", None)
        scheduler = getattr(core, "scheduler", None)
        config = getattr(scheduler, "config", None)
        settings = state.settings_manager.get_settings(launch["model_id"])
        ple = []
        fused = 0
        if model is not None:
            for layer in model.language_model.model.layers:
                if hasattr(layer, "ple"):
                    embedding = layer.ple.ple_embedding.ngram_embedding
                    ple.append({"class": type(embedding).__name__, "rows_read": embedding.rows_read})
                switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
                if switch is not None and hasattr(switch, "gate_up_proj"):
                    fused += 1
        return {
            "schema": "splash-installed-omlx-runtime-v1",
            "model_id": launch["model_id"], "model_directory": launch["model"],
            "runtime": str(root), "installed_app": str(APP),
            "loaded": model is not None and adapter is not None and scheduler is not None,
            "engine_class": type(engine).__name__ if engine is not None else None,
            "ple_mode": get_ple_runtime_mode(), "ple_embeddings": ple,
            "mtp_runtime_enabled": get_mtp_runtime().enabled,
            "mtp_head_present": model is not None and hasattr(model, "mtp"),
            "settings": {name: getattr(settings, name, None) for name in (
                "qwen4_ple_ssd_offload", "mtp_enabled", "mtp_num_draft_tokens",
                "vlm_mtp_enabled", "dflash_enabled", "enable_thinking",
                "qwen35_oq_a8_enabled", "moe_expert_offload_enabled",
            )},
            "scheduler": {name: getattr(config, name, None) for name in (
                "max_num_seqs", "prefill_step_size", "chunked_prefill", "decode_fairness",
                "paged_ssd_cache_dir", "hot_cache_max_size",
            )},
            "prefix_cache": {
                name: getattr(scheduler, name, None) is not None
                for name in ("paged_cache_manager", "block_aware_cache", "paged_ssd_cache_manager")
            },
            "moe_gate_up_fused_layers": fused,
            "m5_gather_qmm_workaround_installed": bool(getattr(mx.gather_qmm, "_omlx_m5_reroute", False)),
            "burst_environment": {key: value for key, value in os.environ.items() if key.startswith("OMLX_DECODE_BURST")},
            "readiness_scope": "loaded actual installed VLMBatchedEngine; no inference performed by this endpoint",
        }

    argv = ["omlx", "serve", "--base-path", str(root), "--model-dir", str(root / "models"),
            "--host", "127.0.0.1", "--port", str(launch["port"]), "--no-cache",
            "--max-concurrent-requests", "4", "--log-level", "info"]
    saved_argv = sys.argv
    sys.argv = argv
    try:
        cli.main()
    finally:
        sys.argv = saved_argv
        verify_preserved(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--runtime", type=Path, required=True)
    make.add_argument("--model", type=Path, default=MODEL)
    make.add_argument("--port", type=int, default=8014)
    make.add_argument("--mtp-depth", type=int, choices=(0, 3), default=0)
    launch = sub.add_parser("serve")
    launch.add_argument("--runtime", type=Path, required=True)
    launch.add_argument("--run-root-gpu", action="store_true")
    check = sub.add_parser("verify-preserved")
    check.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args)
    elif args.mode == "serve": serve(args)
    else:
        verify_preserved(args.runtime.expanduser().resolve())
        print('{"preserved_user_files":true}')


if __name__ == "__main__":
    main()
