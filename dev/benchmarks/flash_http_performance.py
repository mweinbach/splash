"""Persist matched Flash HTTP prompts or ROOT-RUN their full-budget benchmark.

plan imports only CPU tokenizer code. measure sends actual inference requests;
the root coordinator must serialize server runs and choose the active endpoint.
Primary throughput is full-budget output tokens divided by complete wave wall
time. Native model counters and server-reported decode timing remain separate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dev.benchmarks.qualify_flash_http import HTTPClient

MODEL = Path.home() / ".omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
CORPUS = Path("/Applications/oMLX.app/Contents/Resources/omlx/admin/bench_corpora/code_python.txt")


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_fresh(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def make_plan(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    corpus = CORPUS.read_text()
    filler = tokenizer.encode(corpus[:32768], add_special_tokens=False)
    contexts = [128, 2048]
    plan = {
        "schema": "splash-flash-matched-http-performance-v1",
        "source_tokenizer": str(args.model.resolve()),
        "corpus": str(CORPUS), "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "corpus_prefix_characters_tokenized": 32768,
        "nonce": args.nonce, "maximum_output_tokens": 128, "contexts": contexts,
        "samples": args.samples, "batch_widths": [1, 4], "waves": [],
    }
    for sample in range(args.samples):
        for context in contexts:
            for width in (1, 4):
                lanes = []
                for lane in range(width):
                    nonce = hashlib.sha256(f"{args.nonce}/{sample}/{context}/{width}/{lane}".encode()).hexdigest()[:32]
                    prefix = f"Benchmark {nonce}. Read the following Python context.\n\x60\x60\x60python\n"
                    suffix = "\n\x60\x60\x60\nWrite consecutive integers from 1 to 200, one per line. Include no other text."
                    rows = context
                    for _ in range(16):
                        content = prefix + tokenizer.decode((filler * (rows // len(filler) + 1))[:rows]) + suffix
                        tokens = tokenizer.apply_chat_template(
                            [{"role": "user", "content": content}], tokenize=True,
                            return_dict=False, add_generation_prompt=True, enable_thinking=False,
                        )
                        if len(tokens) == context: break
                        rows -= len(tokens) - context
                        if rows < 0: raise ValueError("prompt overhead exceeds target context")
                    else:
                        # Code punctuation can make token-boundary feedback
                        # alternate by one. Keep a real corpus prefix and use
                        # a short benign code-comment pad to reach exact rows.
                        rows = max(0, rows - 32)
                        source = tokenizer.decode((filler * (rows // len(filler) + 1))[:rows])
                        for padding in range(96):
                            content = prefix + source + "\n# padding" + " x" * padding + suffix
                            tokens = tokenizer.apply_chat_template(
                                [{"role": "user", "content": content}], tokenize=True,
                                return_dict=False, add_generation_prompt=True, enable_thinking=False,
                            )
                            if len(tokens) == context: break
                        else: raise ValueError("failed to form exact target token count")
                    body = {
                        "messages": [{"role": "user", "content": content}],
                        "max_completion_tokens": 128, "temperature": 0, "seed": 0,
                        "reasoning_effort": "none", "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    lanes.append({
                        "lane": lane, "body": body, "body_sha256_without_model": hash_json(body),
                        "prompt_tokens": tokens, "prompt_token_count": len(tokens),
                        "prompt_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest(),
                    })
                plan["waves"].append({"sample": sample, "context": context, "width": width, "lanes": lanes})
    write_fresh(args.output, plan)
    print(json.dumps({"plan": str(args.output), "waves": len(plan["waves"]), "gpu_executed": False}))


def get(client, path):
    record = client.send("GET", path)
    if record["http_status"] != 200 or not isinstance(record.get("response"), dict):
        raise ValueError(f"status/readiness request failed at {path}")
    return record["response"]


def check_omlx(runtime, depth):
    errors = []
    if runtime.get("loaded") is not True or runtime.get("engine_class") != "VLMBatchedEngine":
        errors.append("actual installed VLMBatchedEngine is not loaded")
    if runtime.get("ple_mode") != "mmap" or not runtime.get("ple_embeddings") or any(
        item["class"] != "DiskBackedShardedEmbedding" for item in runtime.get("ple_embeddings", [])
    ):
        errors.append("actual PLE route is not disk mmap")
    expected = depth == 3
    for name in ("mtp_runtime_enabled", "mtp_head_present"):
        if runtime.get(name) is not expected: errors.append(f"actual {name} differs")
    settings = runtime.get("settings", {})
    if settings.get("mtp_enabled") is not expected or settings.get("mtp_num_draft_tokens") != 3:
        errors.append("actual MTP settings differ")
    if settings.get("qwen4_ple_ssd_offload") is not True or settings.get("enable_thinking") is not False:
        errors.append("actual PLE/thinking settings differ")
    if any(runtime.get("prefix_cache", {}).values()) or runtime.get("scheduler", {}).get("paged_ssd_cache_dir") is not None:
        errors.append("cross-request prefix cache is enabled")
    if runtime.get("m5_gather_qmm_workaround_installed") is not True:
        errors.append("installed M5 sorted gather workaround is absent")
    return errors


def native_mtp_policy(runtime):
    """Read declared native caps without treating the oMLX CLI as native state."""
    errors = []
    mtp = runtime.get("mtp", {})
    capabilities = runtime.get("capabilities", {})
    if not isinstance(mtp, dict) or not isinstance(capabilities, dict):
        return {}, ["native MTP policy or capabilities is not an object"]
    enabled = mtp.get("enabled")
    joint_enabled = capabilities.get("batch_mtp")
    if type(enabled) is not bool or capabilities.get("mtp") is not enabled:
        errors.append("native MTP enabled state is missing or contradicts capabilities")
    if type(joint_enabled) is not bool:
        errors.append("native joint MTP enabled state is unavailable")
    if joint_enabled is True and enabled is not True:
        errors.append("native joint MTP is enabled without the singleton head")

    def cap(name):
        value = mtp.get(name)
        if type(value) is not int or value < (1 if enabled is True else 0) or value >= 4096:
            errors.append(f"native {name} is missing or outside protocol bounds")
            return None
        if enabled is False and value != 0:
            errors.append(f"disabled native MTP reports nonzero {name}")
        return value

    singleton_cap = cap("singleton_maximum_draft_tokens")
    concurrent_cap = cap("singleton_concurrent_draft_cap")
    if singleton_cap is not None and concurrent_cap is not None and concurrent_cap > singleton_cap:
        errors.append("native concurrent singleton cap exceeds its maximum")
    override = mtp.get("singleton_depth_override")
    if override is not None and (type(override) is not int or not 1 <= override < 4096):
        errors.append("native singleton depth override is malformed")
    if enabled is True and override is not None and override != singleton_cap:
        errors.append("native singleton depth override contradicts its maximum")
    controller = mtp.get("depth_controller_semantics")
    if controller is not None and (not isinstance(controller, str) or not controller):
        errors.append("native depth controller semantics is malformed")
    policy = mtp.get("policy")
    if enabled is True and (not isinstance(policy, str) or not policy):
        errors.append("native singleton policy is unavailable")
    joint_policy = mtp.get("joint_policy")
    joint_cap = 0 if joint_enabled is False else mtp.get("joint_maximum_draft_tokens")
    joint_cap_source = "capabilities.batch_mtp=false" if joint_enabled is False else "mtp.joint_maximum_draft_tokens"
    if joint_enabled is True and joint_cap is None:
        # Current workers explicitly declare "fixed cap3" in this field.
        # Read that declared number rather than guessing from the global cap7.
        match = re.search(r"\bfixed\s*cap\s*([0-9]+)\b", joint_policy) if isinstance(joint_policy, str) else None
        if match:
            joint_cap = int(match.group(1))
            joint_cap_source = "explicit fixed cap in mtp.joint_policy"
    if joint_enabled is True and (type(joint_cap) is not int or not 1 <= joint_cap < 4096):
        errors.append("native joint draft cap is unavailable or outside protocol bounds")
        joint_cap = None
    return {
        "enabled": enabled, "singleton_maximum_draft_tokens": singleton_cap,
        "singleton_depth_override": override, "singleton_concurrent_draft_cap": concurrent_cap,
        "singleton_policy": "disabled" if enabled is False else "adaptive" if controller else "fixed",
        "singleton_policy_details": policy, "depth_controller_semantics": controller,
        "joint_enabled": joint_enabled, "joint_maximum_draft_tokens": joint_cap,
        "joint_cap_source": joint_cap_source, "joint_policy": joint_policy,
    }, errors


def native_mtp_depth_metadata(before, after):
    before_policy, before_errors = native_mtp_policy(before)
    after_policy, after_errors = native_mtp_policy(after)
    errors = [f"runtime_before: {error}" for error in before_errors]
    errors.extend(f"runtime_after: {error}" for error in after_errors)
    stable = before_policy == after_policy
    if not stable:
        errors.append("native MTP policy changed during measurement")
    return {
        "scope": "native /status runtime policy before and after; independent of CLI --mtp-depth",
        "verified": not errors, "stable": stable,
        "runtime_before": before_policy, "runtime_after": after_policy, "errors": errors,
    }


def run_wave(client, wave):
    def one(lane):
        body = copy.deepcopy(lane["body"]); body["model"] = client.model
        record = client.send("POST", "/v1/chat/completions", body)
        usage = record.get("usage") or {}
        errors = list(record["errors"])
        if record.get("http_status") != 200 or not record.get("done"): errors.append("incomplete HTTP stream")
        if record.get("error_frames"): errors.append("server error frame")
        if record.get("reasoning_text"): errors.append("unexpected thinking output")
        if usage.get("prompt_tokens") != wave["context"]: errors.append("prompt token count differs from plan")
        if usage.get("completion_tokens") != 128 or record.get("finish_reason") != "length":
            errors.append("request did not complete the full 128-token budget")
        if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
            errors.append("request reused prefix or cached-token count unavailable")
        record["performance_validation_errors"] = errors
        record["plan_lane"] = lane["lane"]
        record["prompt_u32le_sha256"] = lane["prompt_u32le_sha256"]
        return record
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=wave["width"]) as executor:
        records = list(executor.map(one, wave["lanes"]))
    wall = time.monotonic() - began
    valid = all(not record["performance_validation_errors"] for record in records)
    completed = sum((record.get("usage") or {}).get("completion_tokens", 0) for record in records)
    return {
        "sample": wave["sample"], "context": wave["context"], "width": wave["width"],
        "valid": valid, "wall_seconds": wall, "completion_tokens": completed,
        "full_wave_output_tokens_per_second": completed / wall,
        "mean_client_first_content_ms": statistics.mean(record["first_content_ms"] for record in records
            if record["first_content_ms"] is not None) if any(record["first_content_ms"] is not None for record in records) else None,
        "records": records,
    }


def measure(args):
    if not args.run_root_gpu: raise ValueError("actual inference requires --run-root-gpu")
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or not endpoint.port:
        raise ValueError("benchmark endpoint must be explicit local HTTP host and port")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "splash-flash-matched-http-performance-v1": raise ValueError("unsupported plan")
    client = HTTPClient(SimpleNamespace(base_url=args.base_url, timeout=args.timeout, model=args.model))
    metadata = get(client, "/__flash_baseline/runtime" if args.backend == "omlx" else "/status")
    errors = check_omlx(metadata, args.mtp_depth) if args.backend == "omlx" else []
    if args.backend == "splash" and (metadata.get("ready") is not True or metadata.get("metal", {}).get("healthy") is not True):
        errors.append("Splash native route is not ready and healthy")
    if errors: raise ValueError(f"runtime qualification failed: {errors}")
    report = {
        "schema": "splash-flash-http-performance-report-v1", "backend": args.backend,
        "endpoint": args.base_url, "model": args.model, "mtp_depth": args.mtp_depth,
        "mtp_depth_scope": "oMLX comparison CLI setting; native actual caps are in native_mtp_depth_policy"
            if args.backend == "splash" else "oMLX CLI setting checked against the loaded runtime",
        "plan": str(args.plan.resolve()), "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "runtime_before": metadata, "warmups": [], "waves": [],
        "primary_metric": "actual full-budget output tokens / complete HTTP wave wall time",
        "timing_notes": [
            "All requests must produce exactly 128 completion tokens, finish length, and report zero cached tokens.",
            "Client first-content latency includes queue/prefill/transport. Balanced oMLX bursts can group several tokens.",
            "Server-reported generation_tokens_per_second and native model counters have different timing scopes and are preserved separately.",
            "Warmup compiles the active backend and warms OS file pages; every request remains cold with respect to cross-request prefix/KV caches.",
            "Native singleton and joint MTP caps and policies are verified separately from the comparison CLI setting."
                if args.backend == "splash" else "oMLX MTP depth3 is an adaptive maximum, not three guaranteed accepted drafts.",
        ],
    }
    if args.warmup:
        warm = next(wave for wave in plan["waves"] if wave["context"] == 2048 and wave["width"] == 1)
        report["warmups"].append(run_wave(client, warm))
    for wave in plan["waves"]:
        wave_before = get(client, "/status") if args.backend == "splash" and args.status_per_wave else None
        result = run_wave(client, wave)
        if wave_before is not None:
            result["runtime_before"] = wave_before
            result["runtime_after"] = get(client, "/status")
            result["runtime_snapshot_scope"] = "native safe-point snapshots outside measured HTTP wave; may precede final terminal bookkeeping"
        report["waves"].append(result)
        print(json.dumps({key: result[key] for key in
                         ("sample", "context", "width", "valid", "full_wave_output_tokens_per_second")}), flush=True)
    report["runtime_after"] = get(client, "/__flash_baseline/runtime" if args.backend == "omlx" else "/status")
    report["valid"] = all(wave["valid"] for wave in report["waves"] + report["warmups"])
    if args.backend == "splash":
        report["native_mtp_depth_policy"] = native_mtp_depth_metadata(metadata, report["runtime_after"])
        report["valid"] = report["valid"] and report["native_mtp_depth_policy"]["verified"]
    write_fresh(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--model", type=Path, default=MODEL)
    plan.add_argument("--samples", type=int, default=2)
    plan.add_argument("--nonce", default="flash-performance-20260919")
    plan.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("measure")
    test.add_argument("--backend", choices=("splash", "omlx"), required=True)
    test.add_argument("--base-url", required=True)
    test.add_argument("--model", required=True)
    test.add_argument("--mtp-depth", type=int, choices=(0, 3), default=0)
    test.add_argument("--plan", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--status-per-wave", action="store_true", help="capture Splash native safe-point counters outside each measured wave")
    test.add_argument("--timeout", type=float, default=600)
    test.add_argument("--warmup", action="store_true")
    test.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("choose a fresh report path")
    if args.command == "plan":
        if not 1 <= args.samples <= 8: raise ValueError("samples must be 1..8")
        make_plan(args)
    else: measure(args)


if __name__ == "__main__":
    main()
