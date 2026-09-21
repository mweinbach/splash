"""Prepare exact saved v5 attribution inputs using CPU tokenizer/metadata only.

This command does not load weights, create a Metal backend, call a server, or
change the local profile. Native/GPU execution remains the root coordinator's
responsibility. Outputs are created exclusively so earlier evidence survives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import struct
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def body_digest(body: dict) -> str:
    # Exact canonicalization used by the frozen HTTP plan generator.
    return digest(json.dumps(body, sort_keys=True, ensure_ascii=False).encode())


def witness(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": digest(raw)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    require(profile.get("profile") == "m5-ultra-flash-next-v5", "expected the v5 profile")

    source = Path(plan["source_tokenizer"]).resolve()
    package = args.package.resolve()
    # full_source=False checks tokenizer/config bytes and weight shard sizes;
    # it does not scan the multi-GB weight payloads or instantiate the model.
    manifest = launcher.local_bundle_manifest(package, source, full_source=False)
    require(manifest["source_identity_sha256"] == profile["source_identity_sha256"], "source identity differs")
    config = json.loads((package / "config.json").read_bytes())
    require(config.get("model_type") == profile["architecture"], "model architecture differs")
    hardware = launcher._local_hardware_identity()
    require(isinstance(hardware, tuple) and hardware[0] == profile["cpu_brand"], "hardware brand differs")
    require(type(hardware[1]) is int and hardware[1] >= profile["minimum_physical_ram_bytes"], "insufficient physical RAM")
    layout_fingerprint = launcher._qualified_saved_operand_layout(package)

    # Empty explicit environment reproduces ordinary qualified defaults,
    # independently of any experimental SPLASH_FLASH_* flags in this shell.
    injected = launcher._qualified_saved_operand_defaults(package, {})
    injected.update(launcher._qualified_saved_int8_expert_defaults(package, {}, hardware=hardware))
    require(set(injected) == {"SPLASH_FLASH_OPERAND_STORE", "SPLASH_FLASH_INT8_EXPERT_STORE"}, "both qualified stores are required")
    effective: dict[str, str] = {}
    launcher._apply_local_profile_defaults(effective, {**profile["environment"], **injected})
    require(len(profile["environment"]) == 30 and len(effective) == 32, "unexpected v5 flag count")

    status_path = args.status.resolve()
    status_raw = status_path.read_bytes()
    status = json.loads(status_raw)
    require(status["identity"]["source"] == profile["source_identity_sha256"], "saved status source differs")
    require(status["identity"]["loaded_model_layout_sha256"] == layout_fingerprint, "saved status layout differs")
    store_records = []
    for key, status_field in (
        ("SPLASH_FLASH_OPERAND_STORE", "persisted_operands"),
        ("SPLASH_FLASH_INT8_EXPERT_STORE", "persisted_experts"),
    ):
        store_manifest = witness(Path(effective[key]) / "manifest.json")
        require(store_manifest["sha256"] == status[status_field]["store_manifest_sha256"], f"saved status store differs: {key}")
        store_records.append({"environment_key": key, "directory": effective[key], "manifest": store_manifest, "saved_status": status[status_field]})

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    tokenizer_witnesses = [witness(source / name) for name in (
        "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja"
    ) if (source / name).is_file()]
    output = args.output.resolve()
    allowed = (PROJECT / "build/flash-v5-attribution").resolve()
    require(output == allowed or allowed in output.parents, "outputs must stay under build/flash-v5-attribution")
    pending: dict[str, bytes] = {}
    lane_records = []
    width_records = []
    for width in (1, 4):
        waves = [wave for wave in plan["waves"] if wave.get("sample") == 0 and wave.get("context") == 2048 and wave.get("width") == width]
        require(len(waves) == 1, f"expected one exact saved wave for width {width}")
        wave = waves[0]
        require(len(wave["lanes"]) == width, f"saved wave lane count differs for width {width}")
        width_tokens = []
        for lane_index in range(width):
            lanes = [lane for lane in wave["lanes"] if lane.get("lane") == lane_index]
            require(len(lanes) == 1, f"expected exact saved lane {lane_index}")
            lane = lanes[0]
            body = lane["body"]
            require(body_digest(body) == lane["body_sha256_without_model"], "saved body checksum differs")
            tokens = tokenizer.apply_chat_template(
                body["messages"], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            require(len(tokens) == lane["prompt_token_count"] == 2048, "retokenized prompt count differs")
            require(tokens == lane["prompt_tokens"], f"retokenized token IDs differ at width{width}/lane{lane_index}")
            require(all(type(token) is int and 0 <= token < 248320 for token in tokens), "invalid model token ID")
            u32le = struct.pack(f"<{len(tokens)}I", *tokens)
            require(digest(u32le) == lane["prompt_u32le_sha256"], "retokenized U32LE checksum differs")
            stem = f"ctx2048-sample0-width{width}-lane{lane_index}"
            artifacts = {
                "tokens_json": stem + ".tokens.json",
                "tokens_u32le": stem + ".tokens.u32le",
                "request_json": stem + ".request.json",
            }
            pending[artifacts["tokens_json"]] = json_bytes(tokens)
            pending[artifacts["tokens_u32le"]] = u32le
            pending[artifacts["request_json"]] = json_bytes(body)
            lane_records.append({
                "context": 2048, "sample": 0, "width": width, "lane": lane_index,
                "prompt_token_count": len(tokens), "all_saved_token_ids_equal": True,
                "prompt_u32le_sha256": digest(u32le),
                "body_sha256_without_model": lane["body_sha256_without_model"],
                "prompt_content_utf8_sha256": digest(body["messages"][0]["content"].encode()),
                "artifacts": {key: {"path": str(output / name), "bytes": len(pending[name]), "sha256": digest(pending[name])} for key, name in artifacts.items()},
            })
            width_tokens.append(tokens)
        combined_name = f"ctx2048-sample0-width{width}.lanes.tokens.json"
        pending[combined_name] = json_bytes(width_tokens)
        width_records.append({"context": 2048, "sample": 0, "width": width,
            "lanes_tokens_json": {"path": str(output / combined_name), "sha256": digest(pending[combined_name]), "bytes": len(pending[combined_name])}})

    pending["v5-environment.json"] = json_bytes(effective)
    pending["v5-environment.sh"] = (
        "# Reproduces 30 static v5 defaults and two CPU-qualified store paths.\n"
        "# Source after removing any inherited experimental SPLASH_FLASH_* flags.\n"
        + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in effective.items())
    ).encode()
    provenance = {
        "schema": "splash-flash-v5-attribution-fixture-v1",
        "gpu_executed": False, "inference_executed": False, "payloads_scanned": False,
        "plan": {"path": str(plan_path), "sha256": digest(plan_raw), "schema": plan["schema"], "nonce": plan["nonce"]},
        "profile": {"path": str(profile_path), "sha256": digest(profile_raw), "value": profile},
        "launcher": witness(PROJECT / "install/launcher.py"),
        "preparer": witness(Path(__file__)),
        "package": {"path": str(package), "manifest": witness(package / "manifest.json"), "source_identity_sha256": manifest["source_identity_sha256"], "weights_manifest_fingerprint": layout_fingerprint},
        "hardware": {"cpu_brand": hardware[0], "physical_ram_bytes": hardware[1], "probe": "CPU sysctl only"},
        "tokenizer": {"source": str(source), "class": type(tokenizer).__name__, "transformers_version": transformers.__version__, "local_files_only": True, "trust_remote_code": False, "chat_template": {"tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": False}, "files": tokenizer_witnesses},
        "saved_status": {"path": str(status_path), "sha256": digest(status_raw), "identity": status["identity"], "saved_operands_residency": status["saved_operands_residency"]},
        "saved_stores": store_records,
        "environment": {"static_count": 30, "effective_count": 32, "caller_overrides": {}, "effective": effective,
            "json": {"path": str(output / "v5-environment.json"), "sha256": digest(pending["v5-environment.json"])},
            "shell": {"path": str(output / "v5-environment.sh"), "sha256": digest(pending["v5-environment.sh"])},
            "invocation_requirement": "Remove inherited SPLASH_FLASH_* variables, then apply this exact 32-key environment. Experimental flags/profile controls are separate overrides; GPU prefill copy remains default off."},
        "lanes": lane_records, "widths": width_records,
        "scope": "Five independently retokenized exact ctx2048/sample0 saved prompts for widths1/4. Saved idle status is historical provenance, not a live health check. Full payload verification and GPU execution belong to the native/root run.",
    }
    pending["fixture-provenance.json"] = json_bytes(provenance)
    require(all(not (output / name).exists() for name in pending), "fixture output already exists; select a fresh child directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in pending.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return {"gpu_executed": False, "inference_executed": False, "verified_prompt_count": len(lane_records), "tokens_per_prompt": 2048,
        "provenance": str(output / "fixture-provenance.json"), "provenance_sha256": digest(pending["fixture-provenance.json"]),
        "environment_json": str(output / "v5-environment.json"), "environment_shell": str(output / "v5-environment.sh"),
        "plan_sha256": digest(plan_raw), "static_flag_count": 30, "effective_flag_count": len(effective)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT / "build/release/flash/http-performance-plan.json")
    parser.add_argument("--package", type=Path, default=PROJECT / "install/local-models/Flash-Next-oQ4e-mtp-v1")
    parser.add_argument("--status", type=Path, default=PROJECT / "build/release/flash/default-v5-final-idle-status.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "build/flash-v5-attribution/fixture")
    args = parser.parse_args()
    print(json.dumps(prepare(args), allow_nan=False))


if __name__ == "__main__":
    main()
