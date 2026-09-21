"""Publish CPU-verified v6 singleton and four-lane verifier inputs.

Only local tokenizer and launch metadata are read. This never initializes Metal,
loads tensor payloads, executes inference, or changes profiles/checkpoints.
The root coordinator owns GPU and full-model qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import struct
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dev.benchmarks.flash_v5_attribution_fixture import (
    body_digest, digest, json_bytes, require, witness,
)


def prepare(args: argparse.Namespace) -> dict:
    from install import launcher
    from transformers import AutoTokenizer
    import transformers

    plan_path = args.plan.resolve()
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    require(plan.get("schema") == "splash-flash-matched-http-performance-v1", "unexpected plan schema")
    profile_path = PROJECT / ".splash-local-profile.json"
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    require(profile == launcher.LOCAL_PROFILE, "local profile differs from launcher defaults")
    require(profile.get("profile") == "m5-ultra-flash-next-v6", "expected v6 profile")
    require(len(profile["environment"]) == 34, "expected 34 static v6 defaults")
    package = args.package.resolve()
    source = Path(plan["source_tokenizer"]).resolve()
    manifest = launcher.local_bundle_manifest(package, source, full_source=False)
    require(manifest["source_identity_sha256"] == profile["source_identity_sha256"], "source identity differs")
    config = json.loads((package / "config.json").read_bytes())
    require(config.get("model_type") == profile["architecture"] == "qwen4_exp", "model type differs")
    text_config = config["text_config"]
    require(text_config["vocab_size"] == 248320, "model vocabulary differs")
    require(set(config["eos_token_id"]) == {248044, 248046}, "model stop IDs differ")
    hardware = launcher._local_hardware_identity()
    require(isinstance(hardware, tuple) and hardware[0] == profile["cpu_brand"], "hardware brand differs")
    require(type(hardware[1]) is int and hardware[1] >= profile["minimum_physical_ram_bytes"], "insufficient physical RAM")
    layout_fingerprint = launcher._qualified_saved_operand_layout(package)

    # An empty explicit environment gives qualified defaults, not inherited
    # experimental route flags in the shell that creates this CPU fixture.
    injected = launcher._qualified_saved_operand_defaults(package, {})
    injected.update(launcher._qualified_saved_int8_expert_defaults(package, {}, hardware=hardware))
    require(set(injected) == {"SPLASH_FLASH_OPERAND_STORE", "SPLASH_FLASH_INT8_EXPERT_STORE"}, "both qualified stores are required")
    effective: dict[str, str] = {}
    launcher._apply_local_profile_defaults(effective, {**profile["environment"], **injected})
    require(len(effective) == 36, "expected 36 effective v6 defaults")

    stores = {}
    store_records = []
    for key in injected:
        path = Path(effective[key]) / "manifest.json"
        metadata = json.loads(path.read_bytes())
        stores[key] = metadata
        store_records.append({"environment_key": key, "directory": effective[key], "manifest": witness(path),
            "schema": metadata["schema"], "source_identity_sha256": metadata["source_identity_sha256"]})
    dense = stores["SPLASH_FLASH_OPERAND_STORE"]
    expert = stores["SPLASH_FLASH_INT8_EXPERT_STORE"]
    dense_types = Counter(entry["format"] for entry in dense["entries"])
    require(dense_types == {"BF16": 509, "F32": 508}, "saved dense operand types differ")
    expert_types = Counter(projection[field]["dtype"] for layer in expert["layers"]
        for projection in layer["projections"].values() for field in ("codes", "scales"))
    require(expert_types == {"I8": 144, "F32": 144}, "saved expert operand types differ")
    require(len(expert["selected_experts"]) == 48 and all(len(ids) == 64 for ids in expert["selected_experts"]), "saved selected expert count differs")
    tensor_types = Counter(entry["dtype"] for entry in manifest["tensors"].values())
    require(tensor_types == {"BF16": 2725, "U32": 1020, "I64": 3}, "aligned source tensor types differ")
    integer_parameters = [{"tensor": name, "dtype": entry["dtype"], "shape": entry["shape"]}
        for name, entry in manifest["tensors"].items() if entry["dtype"] == "I64"]

    statuses = []
    for role, path in (("initial", args.initial_status.resolve()), ("final", args.final_status.resolve())):
        raw = path.read_bytes()
        status = json.loads(raw)
        require(status["identity"]["source"] == profile["source_identity_sha256"], f"{role} status source differs")
        require(status["identity"]["loaded_model_layout_sha256"] == layout_fingerprint, f"{role} status layout differs")
        require(status["persisted_operands"]["schema"] == dense["schema"], f"{role} status saved operand schema differs")
        require(status["persisted_operands"]["store_manifest_sha256"] == witness(Path(injected["SPLASH_FLASH_OPERAND_STORE"]) / "manifest.json")["sha256"], f"{role} status dense manifest differs")
        require(status["persisted_operands"]["bf16_tensors"] == dense_types["BF16"] and status["persisted_operands"]["f32_tensors"] == dense_types["F32"], f"{role} status dense type counts differ")
        require(status["persisted_experts"]["store_manifest_sha256"] == witness(Path(injected["SPLASH_FLASH_INT8_EXPERT_STORE"]) / "manifest.json")["sha256"], f"{role} status expert manifest differs")
        require(status["persisted_experts"]["selection_plan_sha256"] == expert["plan_sha256"], f"{role} status expert plan differs")
        require(status["persisted_experts"]["expert_count"] == 3072, f"{role} status selected expert count differs")
        require(status["mtp"]["singleton_depth_override"] == int(effective["SPLASH_FLASH_MTP_DRAFT_DEPTH"]), f"{role} status MTP depth differs")
        require(status["identity"]["gpu_prefill_feature_copy"] is False, f"{role} status does not reflect default-off GPU feature copy")
        statuses.append({"role": role, "path": str(path), "sha256": digest(raw), "identity": status["identity"],
            "persisted_operands": status["persisted_operands"], "persisted_experts": status["persisted_experts"],
            "saved_operands_residency": status["saved_operands_residency"], "requests": status["requests"],
            "scheduler": status["scheduler"], "ready": status["ready"]})
    require(statuses[0]["identity"] == statuses[1]["identity"], "v6 initial/final route or engine identities changed")
    require(statuses[0]["persisted_operands"] == statuses[1]["persisted_operands"], "v6 initial/final dense identities changed")
    require(statuses[0]["persisted_experts"] == statuses[1]["persisted_experts"], "v6 initial/final expert identities changed")

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    output = args.output.resolve()
    allowed = (PROJECT / "build/flash-v6-verifier-attribution").resolve()
    require(output == allowed or allowed in output.parents, "outputs must stay under build/flash-v6-verifier-attribution")
    require(not output.exists() and not output.is_symlink(), "fixture output directory already exists; choose a fresh child")
    pending: dict[str, bytes] = {}
    lane_records = []
    wave_records = []
    for context in (128, 2048):
        for width in (1, 4):
            waves = [wave for wave in plan["waves"] if wave.get("sample") == 0 and wave.get("context") == context and wave.get("width") == width]
            require(len(waves) == 1 and len(waves[0]["lanes"]) == width, "saved exact wave missing or malformed")
            combined = []
            for lane_index in range(width):
                lanes = [lane for lane in waves[0]["lanes"] if lane.get("lane") == lane_index]
                require(len(lanes) == 1, "saved exact lane missing or duplicated")
                lane = lanes[0]
                body = lane["body"]
                require(body_digest(body) == lane["body_sha256_without_model"], "saved body checksum differs")
                tokens = tokenizer.apply_chat_template(body["messages"], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                require(len(tokens) == lane["prompt_token_count"] == context, "retokenized count differs")
                require(tokens == lane["prompt_tokens"], f"token IDs differ at ctx{context}/width{width}/lane{lane_index}")
                require(all(type(token) is int and 0 <= token < text_config["vocab_size"] for token in tokens), "invalid token ID")
                packed = struct.pack(f"<{len(tokens)}I", *tokens)
                require(digest(packed) == lane["prompt_u32le_sha256"], "retokenized U32LE checksum differs")
                stem = f"ctx{context}-width{width}-lane{lane_index}"
                names = {"tokens_json": stem + ".tokens.json", "native_alias_json": stem + ".json",
                    "tokens_u32le": stem + ".tokens.u32le", "request_json": stem + ".request.json"}
                pending[names["tokens_json"]] = pending[names["native_alias_json"]] = json_bytes(tokens)
                pending[names["tokens_u32le"]] = packed
                pending[names["request_json"]] = json_bytes(body)
                lane_records.append({"sample": 0, "context": context, "width": width, "lane": lane_index,
                    "prompt_token_count": len(tokens), "all_saved_token_ids_equal": True,
                    "prompt_u32le_sha256": digest(packed), "body_sha256_without_model": lane["body_sha256_without_model"],
                    "prompt_content_utf8_sha256": digest(body["messages"][0]["content"].encode()),
                    "artifacts": {key: {"path": str(output / name), "bytes": len(pending[name]), "sha256": digest(pending[name])} for key, name in names.items()}})
                combined.append(tokens)
            combined_name = f"ctx{context}-width{width}.lanes.tokens.json"
            pending[combined_name] = json_bytes(combined)
            wave_records.append({"sample": 0, "context": context, "width": width,
                "lanes_tokens_json": {"path": str(output / combined_name), "bytes": len(pending[combined_name]), "sha256": digest(pending[combined_name])}})

    pending["v6-environment.json"] = json_bytes(effective)
    pending["v6-environment.sh"] = ("# Exact 34 v6 static defaults plus two launcher-qualified saved-store paths.\n"
        "# Remove inherited experimental SPLASH_FLASH_* flags before sourcing.\n"
        + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in effective.items())).encode()
    provenance = {"schema": "splash-flash-v6-verifier-fixture-v1", "gpu_executed": False,
        "inference_executed": False, "payloads_scanned": False,
        "plan": {"path": str(plan_path), "sha256": digest(plan_raw), "schema": plan["schema"], "nonce": plan["nonce"]},
        "profile": {"path": str(profile_path), "sha256": digest(profile_raw), "value": profile},
        "launcher": witness(PROJECT / "install/launcher.py"), "preparer": witness(Path(__file__)),
        "shared_cpu_helpers": witness(PROJECT / "dev/benchmarks/flash_v5_attribution_fixture.py"),
        "package": {"path": str(package), "manifest": witness(package / "manifest.json"),
            "config": witness(package / "config.json"), "index": witness(package / "model.safetensors.index.json"),
            "source_identity_sha256": manifest["source_identity_sha256"], "weights_manifest_fingerprint": layout_fingerprint},
        "hardware": {"cpu_brand": hardware[0], "physical_ram_bytes": hardware[1], "probe": "CPU sysctl only"},
        "type_and_token_ids": {"model_type": config["model_type"], "text_model_type": text_config["model_type"],
            "activation_dtype": text_config["dtype"], "mamba_ssm_dtype": text_config["mamba_ssm_dtype"],
            "vocabulary_size": text_config["vocab_size"], "eos_token_ids": config["eos_token_id"],
            "bos_token_id": text_config["bos_token_id"], "aligned_tensor_type_counts": dict(tensor_types),
            "integer_parameter_metadata": integer_parameters, "saved_dense_type_counts": dict(dense_types),
            "saved_expert_type_counts": dict(expert_types), "selected_expert_count": 3072,
            "selected_expert_ids_per_layer": expert["selected_experts"]},
        "tokenizer": {"source": str(source), "class": type(tokenizer).__name__, "transformers_version": transformers.__version__,
            "local_files_only": True, "trust_remote_code": False,
            "chat_template": {"tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": False},
            "files": [witness(source / name) for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja") if (source / name).is_file()]},
        "saved_stores": store_records, "saved_statuses": statuses,
        "initial_final_identity_equal": True, "initial_final_store_identities_equal": True,
        "environment": {"static_count": 34, "effective_count": 36, "caller_overrides": {}, "effective": effective,
            "json": {"path": str(output / "v6-environment.json"), "sha256": digest(pending["v6-environment.json"])},
            "shell": {"path": str(output / "v6-environment.sh"), "sha256": digest(pending["v6-environment.sh"])},
            "invocation_requirement": "Remove inherited SPLASH_FLASH_* variables, apply these exact 36 defaults, then add explicit experiment/profiling overrides. GPU prefill copy remains default off."},
        "lanes": lane_records, "waves": wave_records,
        "scope": "Ten independently retokenized sample0 saved prompts, contexts128/2048 widths1/4. Saved initial/final statuses are historical provenance, not a live health check. Native/root GPU runs must verify actual loaded identities."}
    pending["fixture-provenance.json"] = json_bytes(provenance)
    # A fresh directory and exclusive file opens never replace earlier evidence.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in pending.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return {"gpu_executed": False, "inference_executed": False, "verified_prompt_count": len(lane_records),
        "plan_sha256": digest(plan_raw), "provenance": str(output / "fixture-provenance.json"),
        "provenance_sha256": digest(pending["fixture-provenance.json"]), "static_flag_count": 34, "effective_flag_count": 36,
        "environment_json": str(output / "v6-environment.json"), "environment_json_sha256": digest(pending["v6-environment.json"]),
        "environment_shell": str(output / "v6-environment.sh")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT / "build/release/flash/http-performance-plan.json")
    parser.add_argument("--package", type=Path, default=PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1")
    parser.add_argument("--initial-status", type=Path, default=PROJECT / "build/release/flash/default-v6-initial-status.json")
    parser.add_argument("--final-status", type=Path, default=PROJECT / "build/release/flash/default-v6-final-idle-status.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "build/flash-v6-verifier-attribution/fixture")
    print(json.dumps(prepare(parser.parse_args()), allow_nan=False))


if __name__ == "__main__":
    main()
