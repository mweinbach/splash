#!/usr/bin/env python3
"""Reproducible fresh-process Flash-Next tuning matrix.

Preparation is CPU-only. Actual serving/inference requires --run-root-gpu.
Examples (choose a fresh --output for each run):
  .venv/bin/python dev/benchmarks/splash_tuning_sep21.py --binary BINARY \
    --output build/tuning-plan.json --dry-run
  .venv/bin/python dev/benchmarks/splash_tuning_sep21.py --binary BINARY \
    --output build/tuning-report.json --mtp standard,3 --contexts 2048 \
    --batches 1,2,4 --workloads coding --run-root-gpu

One process owns the GPU matrix lock, and each MTP setting gets a completely
unloaded server process before the next setting. Concurrent HTTP requests do
not prove native batch width; actual scheduler counters are saved separately.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from install import launcher
from dev.benchmarks.flash_http_performance import native_mtp_depth_metadata
from dev.benchmarks.flash_context_grid import frontend_prompt_tokens
from dev.benchmarks.qualify_flash_http import (
    HTTPClient, counter_delta, get_path, idle, maximum_overlap, validate_status,
)

PACKAGE = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"
MODEL = "local/Qwen3.8-Flash-Next-oQ4e-mtp"
CANONICAL = Path(__file__).with_name("fixtures") / "splash_tuning_sep21_code2048.txt"
CANONICAL_SHA256 = "ce28c1b48bbed10a157c683f0cbc78372021041b28c9bf372b763c8df3029dbf"
FILLER = '''
def drain_pending(queue, sink, retries=3):
    """Flush queued events while retaining failed work for a later retry."""
    pending = list(queue)
    queue.clear()
    for event in pending:
        for attempt in range(retries):
            try:
                sink.write(event)
                break
            except OSError:
                if attempt + 1 == retries:
                    queue.append(event)
    return len(pending) - len(queue)
'''
HEADER = "Review this Python event-worker snapshot. Repeated helper copies represent generated source.\n"
QUESTION = "\nFind correctness and cancellation bugs. Provide a complete robust replacement, explain failure handling, and write comprehensive pytest tests. Include all code; do not abbreviate."
PROSE = "Regional warehouse operations handled 14500 shipments with 27 late deliveries. Staffing costs rose seven percent while revenue rose four percent. Customer interviews identify delayed status updates and inconsistent handoffs. Managers should compare service quality, delivery time, and costs before changing staffing levels.\n"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def save(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def render_tokens(tokenizer, content):
    # This is the recovered benchmark's exact render-then-tokenize procedure.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        return_dict=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def exact_prompt(tokenizer, target, task, nonce=None):
    if task == "coding" and target == 2048 and nonce is None:
        if sha256_file(CANONICAL) != CANONICAL_SHA256:
            raise ValueError("frozen recovered coding2048 fixture changed")
        content = CANONICAL.read_text()
        if len(render_tokens(tokenizer, content)) != target:
            raise ValueError("canonical coding2048 tokenizer count differs")
        return content
    if task == "coding":
        prefix = f"ultra-locality-fixed-code-{target}-20260921\n" + HEADER
        filler, suffix = FILLER, QUESTION
    elif task == "count":
        prefix = "Read the following Python context.\n```python\n"
        filler = FILLER
        suffix = "\n```\nWrite consecutive integers from 1 to 100000, one per line. Include no other text. Continue until the output limit."
    else:
        prefix = "Write a detailed operational review using these fictional records.\n"
        filler = PROSE
        suffix = "\nProduce at least 100 numbered paragraphs explaining the findings, uncertainty, tradeoffs and practical next actions. Continue until the output limit. Begin directly with paragraph 1."
    if nonce is not None:
        prefix = nonce + "\n" + prefix
    baseline = len(render_tokens(tokenizer, prefix + suffix))
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    pool = filler_ids * ((target + 128) // len(filler_ids) + 3)
    rows, tried = target - baseline, set()

    def content_at(count, padding=None):
        source = tokenizer.decode(pool[:count], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        pad = "" if padding is None else "\n#" + " a" * padding
        return prefix + source + pad + suffix

    for _ in range(40):
        if rows < 0 or rows in tried:
            break
        tried.add(rows)
        content = content_at(rows)
        actual = len(render_tokens(tokenizer, content))
        if actual == target:
            return content
        if 0 < target - actual <= 8:
            for padding in range(9):
                padded = content_at(rows, padding)
                if len(render_tokens(tokenizer, padded)) == target:
                    return padded
        rows += target - actual
    for count in range(max(0, rows - 64), rows + 65):
        content = content_at(count)
        actual = len(render_tokens(tokenizer, content))
        if actual == target:
            return content
        if 0 < target - actual <= 8:
            for padding in range(9):
                padded = content_at(count, padding)
                if len(render_tokens(tokenizer, padded)) == target:
                    return padded
    raise ValueError(f"cannot form exact {task}/{target}-token prompt")


def prepare_plan(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                            trust_remote_code=False)
    prompts = []
    for task in args.workloads:
        for context in args.contexts:
            content = exact_prompt(tokenizer, context, task)
            tokens = render_tokens(tokenizer, content)
            prompts.append({"task": task, "prompt_tokens": context, "content": content,
                            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                            "prompt_u32le_sha256": token_hash(tokens), "token_ids": tokens,
                            "canonical_recovered_coding2048": task == "coding" and context == 2048})
            if args.lane_variation == "seeded":
                lanes = []
                for lane in range(max(args.batches)):
                    nonce = hashlib.sha256(f"{args.lane_seed}/{task}/{context}/{lane}".encode()).hexdigest()[:24]
                    varied = exact_prompt(tokenizer, context, task, nonce)
                    varied_tokens = render_tokens(tokenizer, varied)
                    lanes.append({"lane": lane, "nonce": nonce, "content": varied,
                                  "content_sha256": hashlib.sha256(varied.encode()).hexdigest(),
                                  "prompt_u32le_sha256": token_hash(varied_tokens), "token_ids": varied_tokens})
                prompts[-1]["seeded_batch_lanes"] = lanes
    return {"prompts": prompts, "plan_sha256": json_hash(prompts)}


def environment_for(args, mode):
    inherited = dict(os.environ)
    removed = sorted(key for key in inherited if key.startswith(("SPLASH_FLASH_", "FLASH_")))
    clean = {key: value for key, value in inherited.items() if key not in removed}
    # Profile artifact resolution consults os.environ. Keep stale tuning flags
    # out of that resolution, then restore the caller's environment immediately.
    os.environ.clear()
    os.environ.update(clean)
    try:
        defaults = launcher._local_profile_defaults(args.package)
    finally:
        os.environ.clear()
        os.environ.update(inherited)
    if not defaults:
        raise ValueError("launcher did not qualify a local profile for this package/hardware")
    explicit = dict(args.environment_overrides)
    if mode == "standard":
        explicit.update({"SPLASH_FLASH_MTP": "0", "SPLASH_FLASH_BATCH_MTP": "0",
                         "SPLASH_FLASH_BATCH_MTP_PREFILL": "0",
                         "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY": "0",
                         "SPLASH_FLASH_GPU_PREFILL_COPY": "0"})
        explicit.pop("SPLASH_FLASH_MTP_DRAFT_DEPTH", None)
    else:
        explicit.update({"SPLASH_FLASH_MTP": "1", "SPLASH_FLASH_MTP_DRAFT_DEPTH": mode})
    clean.update(explicit)
    launcher._apply_local_profile_defaults(clean, defaults)
    flags = {key: value for key, value in sorted(clean.items())
             if key.startswith(("SPLASH_FLASH_", "FLASH_"))}
    return clean, {"removed_inherited_flag_names": removed,
                   "qualified_launcher_defaults": defaults, "explicit_environment": explicit,
                   "resolved_flash_environment": flags, "resolved_environment_sha256": json_hash(flags)}


def provenance(args):
    paths = {"binary": args.binary, "metallib": args.binary.with_name("splash.metallib"),
             "launcher_profile": ROOT / ".splash-local-profile.json",
             "model_config": args.package / "config.json"}
    for name in ("manifest.json", "local-manifest.json", "manifest.sha256"):
        path = args.package / name
        if path.is_file():
            paths["package_" + name] = path
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "chat_template.jinja"):
        path = args.tokenizer / name
        if path.is_file():
            paths["actual_tokenizer_" + name] = path
    if args.semantic_plan is not None:
        paths["semantic_plan"] = args.semantic_plan
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                         capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                           capture_output=True, check=True).stdout
    source_paths = [path for directory in (ROOT / "server", ROOT / "runtime")
                    for path in directory.rglob("*") if path.is_file()
                    and path.suffix in (".py", ".mm", ".cpp", ".hpp", ".h", ".metal")]
    source_paths.extend((ROOT / "install").glob("*.py"))
    source_paths.extend(Path(__file__).with_name(name) for name in
                        ("splash_tuning_sep21.py", "flash_http_performance.py",
                         "flash_context_grid.py", "qualify_flash_http.py",
                         "prefill4k_attribution_quality.py", "flash_precision_quality.py",
                         "prefill4k_attribution_quality_python.py",
                         "prefill_decode_phase_quality.py", "singleton_teacher_bulk_quality.py",
                         "phase_saved_only_resource_quality.py", "teacher_singleton_lease_quality.py"))
    source_paths.append(CANONICAL)
    sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(source_paths))}
    return {"git_head": git, "git_status_porcelain": dirty,
            "source_file_sha256": sources, "source_manifest_sha256": json_hash(sources),
            "python": sys.executable, "python_version": sys.version,
            "package": str(args.package), "tokenizer": str(args.tokenizer),
            "files": {key: {"path": str(path), "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)} for key, path in paths.items()}}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def wait_idle(client, args, identity=None, after_snapshot=None, timeout=None):
    deadline = time.monotonic() + (args.idle_timeout if timeout is None else timeout)
    status_client = copy.copy(client)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("server idle status deadline expired")
        status_client.timeout = min(args.status_timeout, remaining)
        status = status_client.status()
        errors = validate_status(status, identity)
        if get_path(status, "instance.model") != args.model:
            errors.append("server model differs")
        if get_path(status, "capabilities.native_route") != "flash-next":
            errors.append("native route is not Flash-Next")
        if get_path(status, "capabilities.prefix_cache") is not False:
            errors.append("native prefix cache is not explicitly disabled")
        if status.get("maximum_context_tokens") != args.max_context:
            errors.append("native maximum context differs")
        if errors:
            raise ValueError("; ".join(errors))
        snapshot = get_path(status, "status_snapshot.steady_seconds")
        fresh = after_snapshot is None or (finite(snapshot) and snapshot > after_snapshot)
        requests = status.get("requests", {})
        accounted = requests.get("submitted") == sum(requests.get(key, -1) for key in ("completed", "failed", "cancelled"))
        if idle(status) and fresh and accounted and time.monotonic() < deadline:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError("server did not reach a fresh, fully accounted idle status")
        time.sleep(0.1)


def all_numeric_deltas(before, after, prefix=""):
    result = {}
    for key, right in after.items():
        left = before.get(key)
        path = prefix + key
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(all_numeric_deltas(left, right, path + "."))
        elif finite(left) and finite(right):
            result[path] = right - left
    return result


def native_exact(record, prompt, budget):
    errors = list(record.get("errors", []))
    if record.get("http_status") != 200 or not record.get("done") or record.get("error_frames"):
        errors.append("incomplete/error HTTP stream")
    if record.get("reasoning_text") or record.get("tool_calls"):
        errors.append("unexpected reasoning/tool output")
    usage = record.get("usage") or {}
    for path, expected in (("prompt_tokens", prompt), ("completion_tokens", budget),
                           ("total_tokens", prompt + budget),
                           ("prompt_tokens_details.cached_tokens", 0),
                           ("completion_tokens_details.reasoning_tokens", 0)):
        value = get_path(usage, path)
        if type(value) is not int or value != expected:
            errors.append(f"usage {path} differs: {value!r}")
    if record.get("finish_reason") != "length":
        errors.append("request did not use full output budget")
    metrics = record.get("metrics") or {}
    if get_path(metrics, "cache.matched_tokens") != 0 or get_path(metrics, "prefill.tokens") != prompt:
        errors.append("native cache/prefill accounting differs")
    tokens = get_path(metrics, "decode.tokens")
    interval = get_path(metrics, "request_latency.first_token_to_done_ms")
    rate = get_path(metrics, "request_latency.stream_tokens_per_second")
    first = budget - tokens if type(tokens) is int else None
    exact_rate = tokens * 1000 / interval if type(tokens) is int and finite(interval) and interval > 0 else None
    if type(tokens) is not int or tokens <= 0 or first is None or not 1 <= first <= 16:
        errors.append("native exact first-emission token accounting invalid")
    if exact_rate is None or not finite(rate) or not math.isclose(rate, exact_rate, rel_tol=1e-9, abs_tol=1e-7):
        errors.append("native decode rate disagrees with exact token/Done interval")
    actual_output = usage.get("completion_tokens")
    return {"valid": not errors, "errors": errors,
            "full_output_budget": actual_output == budget and record.get("finish_reason") == "length",
            "shortened": type(actual_output) is int and 0 <= actual_output < budget,
            "actual_completion_tokens": actual_output,
            "native_post_first_emission_tokens": tokens, "native_first_emission_tokens": first,
            "native_first_emission_to_done_ms": interval, "native_exact_decode_tokens_per_second": exact_rate}


def run_wave(client, prompt, budget, width, args):
    barrier = threading.Barrier(width)
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt["content"]}],
            "max_completion_tokens": budget, "temperature": 0, "seed": 0,
            "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}

    def one(lane):
        request_body = copy.deepcopy(body)
        lane_prompt = prompt["seeded_batch_lanes"][lane] if args.lane_variation == "seeded" and width > 1 else prompt
        request_body["messages"][0]["content"] = lane_prompt["content"]
        began = time.monotonic()
        try:
            barrier.wait(timeout=10)
            record = client.send("POST", "/v1/chat/completions", request_body)
        except Exception as error:
            record = {"http_status": None, "done": False, "errors": [repr(error)],
                      "usage": None, "metrics": {}, "request_body": request_body,
                      "client_started_monotonic": began, "client_ended_monotonic": time.monotonic()}
        record["lane"] = lane
        record["lane_prompt_content_sha256"] = lane_prompt["content_sha256"]
        record["lane_prompt_u32le_sha256"] = lane_prompt["prompt_u32le_sha256"]
        record["measurement"] = native_exact(record, prompt["prompt_tokens"], budget)
        return record

    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=width) as pool:
        records = list(pool.map(one, range(width)))
    wall = time.monotonic() - began
    output = sum(value for record in records
                 if type(value := (record.get("usage") or {}).get("completion_tokens")) is int)
    exact = [record["measurement"] for record in records]
    decode_tokens = sum(row["native_post_first_emission_tokens"] for row in exact
                        if type(row["native_post_first_emission_tokens"]) is int)
    intervals = [row["native_first_emission_to_done_ms"] for row in exact]
    native_sum_ms = sum(intervals) if all(finite(value) for value in intervals) else None
    firsts = [r["client_started_monotonic"] + r["first_content_ms"] / 1000
              for r in records if finite(r.get("first_content_ms"))]
    dones = [r["client_started_monotonic"] + r["done_ms"] / 1000
             for r in records if finite(r.get("done_ms"))]
    client_interval = max(dones) - min(firsts) if len(firsts) == width and len(dones) == width else None
    return {"valid": all(row["valid"] for row in exact), "records": records,
            "lane_content_policy": "seeded varied prompts" if args.lane_variation == "seeded" and width > 1 else "identical shared prompt",
            "lane_prompt_identity": [{"lane": row["lane"], "content_sha256": row["lane_prompt_content_sha256"],
                                      "u32le_sha256": row["lane_prompt_u32le_sha256"]} for row in records],
            "actual_http_maximum_overlap": maximum_overlap(records), "full_wave_wall_seconds": wall,
            "completion_tokens": output, "full_wave_output_tokens_per_second": output / wall,
            "native_post_first_emission_tokens_sum": decode_tokens,
            "native_request_decode_intervals_sum_ms": native_sum_ms,
            "native_tokens_per_sum_request_decode_interval": decode_tokens * 1000 / native_sum_ms
                if native_sum_ms and native_sum_ms > 0 else None,
            "client_first_content_to_last_done_wave_seconds": client_interval,
            "client_wave_post_first_emission_tokens_per_second": decode_tokens / client_interval
                if client_interval and client_interval > 0 else None}


def counter_rates(delta):
    def rate(tokens, milliseconds):
        n, ms = delta.get(tokens), delta.get(milliseconds)
        return n * 1000 / ms if finite(n) and finite(ms) and ms > 0 else None
    return {"native_prefill_tokens_per_summed_command_wall_second": rate("metrics.prefill_input_tokens", "metrics.prefill_wall_ms"),
            "native_prepared_target_predictions_per_summed_decode_command_wall_second": rate("metrics.decode_output_tokens", "metrics.decode_wall_ms"),
            "native_prepared_target_predictions_per_summed_decode_gpu_second": rate("metrics.decode_output_tokens", "model_timing.decode.total_gpu_ms"),
            "native_committed_output_tokens_per_summed_decode_command_wall_second": rate("metrics.autoregressive_output_tokens", "metrics.decode_wall_ms"),
            "native_committed_output_tokens_per_summed_decode_gpu_second": rate("metrics.autoregressive_output_tokens", "model_timing.decode.total_gpu_ms")}


def summary(report):
    grouped = {}
    for row in report["waves"]:
        if row["warmup"] or not row.get("valid"):
            continue
        key = (row["mtp_setting"], row["task"], row["prompt_tokens"], row["output_budget_tokens"], row["http_width"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (mtp, task, context, budget, width), rows in grouped.items():
        entry = {"mtp_setting": mtp, "task": task, "prompt_tokens": context,
                 "output_budget_tokens": budget, "http_width": width, "valid_trials": len(rows)}
        paths = {"native_prefill": "counter_rates.native_prefill_tokens_per_summed_command_wall_second",
                 "native_committed_decode_command": "counter_rates.native_committed_output_tokens_per_summed_decode_command_wall_second",
                 "client_wave_decode": "client_wave_post_first_emission_tokens_per_second",
                 "full_wave_output": "full_wave_output_tokens_per_second"}
        for label, path in paths.items():
            values = [get_path(row, path) for row in rows if finite(get_path(row, path))]
            entry[label + "_median_tokens_per_second"] = statistics.median(values) if values else None
            entry[label + "_mean_tokens_per_second"] = statistics.mean(values) if values else None
        rates = [r["measurement"]["native_exact_decode_tokens_per_second"] for row in rows for r in row["records"]
                 if finite(r["measurement"]["native_exact_decode_tokens_per_second"])]
        entry["native_exact_request_decode_median_tokens_per_second"] = statistics.median(rates) if rates else None
        result.append(entry)
    return result


def unload(server):
    # Kill the whole session, including serve-native descendants; waiting only
    # for Python does not establish that GPU/model ownership has been unloaded.
    group = server.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)
        server.wait(timeout=10)
    deadline = time.monotonic() + 10
    escalated = False
    while True:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return {"process_group": group, "returncode": server.returncode, "process_group_gone": True,
                    "post_parent_exit_sigkill_required": escalated}
        if time.monotonic() >= deadline:
            if escalated:
                return {"process_group": group, "returncode": server.returncode, "process_group_gone": False,
                        "post_parent_exit_sigkill_required": True, "error": "server descendants still exist after unload timeout"}
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            escalated = True
            deadline = time.monotonic() + 10
        time.sleep(0.1)


def launch_measure(args, report, environments):
    lockpath = ROOT / "build/splash-tuning-gpu.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode, environment in environments.items():
            for name, path in (("binary", args.binary), ("metallib", args.binary.with_name("splash.metallib"))):
                if sha256_file(path) != report["provenance"]["files"][name]["sha256"]:
                    raise ValueError(f"{name} changed after provenance capture; choose a fresh report")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", args.port)) == 0:
                    raise RuntimeError(f"port {args.port} already has a listener; refusing to reuse another server")
            logpath = args.output.with_name(args.output.stem + f"-{mode}.server.log")
            command = [sys.executable, "-u", str(ROOT / "server/server.py"),
                       "--local-package", str(args.package), "--tokenizer", str(args.tokenizer),
                       "--model", args.model, "--binary", str(args.binary), "--port", str(args.port),
                       "--max-memory", args.max_memory, "--max-context", str(args.max_context), "--no-webui"]
            run = {"mtp_setting": mode, "command": command, "log": str(logpath),
                   "started_utc": utc_now(), "unloaded": False}
            report["server_runs"].append(run)
            save(args.output, report)
            with logpath.open("x") as log:
                server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                run["pid"] = server.pid
                try:
                    client = HTTPClient(SimpleNamespace(base_url=f"http://127.0.0.1:{args.port}", timeout=args.timeout, model=args.model))
                    deadline = time.monotonic() + args.startup_timeout
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"server exited during startup; see {logpath}")
                        try:
                            initial = wait_idle(client, args, timeout=min(args.idle_timeout, max(0.01, deadline - time.monotonic())))
                            break
                        except (OSError, ValueError, TimeoutError):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f"server startup timeout; see {logpath}")
                            time.sleep(0.5)
                    run["initial_status"] = initial
                    identity = initial["identity"]
                    policy = native_mtp_depth_metadata(initial, initial)
                    if not policy["verified"]:
                        raise ValueError(policy["errors"])
                    actual = policy["runtime_before"]
                    if actual["enabled"] != (mode != "standard") or (mode != "standard" and actual["singleton_maximum_draft_tokens"] != int(mode)):
                        raise ValueError(f"requested MTP setting differs from native policy: {actual}")
                    run["frontend_prompt_witnesses"] = []
                    for prompt in report["plan"]["prompts"]:
                        variants = [prompt] + prompt.get("seeded_batch_lanes", [])
                        for variant in variants:
                            body = {"model": args.model, "messages": [{"role": "user", "content": variant["content"]}], "reasoning_effort": "none"}
                            tokens = frontend_prompt_tokens(client, body)
                            if token_hash(tokens) != variant["prompt_u32le_sha256"]:
                                raise ValueError("frontend rendered prompt tokens differ from CPU plan")
                            run["frontend_prompt_witnesses"].append({"task": prompt["task"], "lane": variant.get("lane"), "count": len(tokens), "u32le_sha256": token_hash(tokens)})
                    save(args.output, report)
                    for trial in range(args.warmup + args.trials):
                        for prompt in report["plan"]["prompts"]:
                            for budget in args.output_tokens:
                                for width in args.batches:
                                    before = wait_idle(client, args, identity)
                                    wave = run_wave(client, prompt, budget, width, args)
                                    wave.update(mtp_setting=mode, task=prompt["task"], prompt_tokens=prompt["prompt_tokens"],
                                                output_budget_tokens=budget, http_width=width, trial=trial - args.warmup + 1,
                                                warmup=trial < args.warmup, status_before=before,
                                                request_validation_valid=wave["valid"], post_wave_idle_pending=True)
                                    wave["valid"] = False
                                    report["waves"].append(wave)
                                    save(args.output, report)
                                    after = wait_idle(client, args, identity, get_path(before, "status_snapshot.steady_seconds"))
                                    delta = all_numeric_deltas(before, after)
                                    delta.update(counter_delta(before, after))
                                    expected = {"requests.submitted": width, "requests.completed": width,
                                                "requests.cancelled": 0, "requests.failed": 0,
                                                "metrics.metal_failures": 0, "metrics.prefill_input_tokens": width * prompt["prompt_tokens"],
                                                "metrics.autoregressive_output_tokens": width * budget}
                                    errors = [f"counter {key} differs: {delta.get(key)!r} != {value}" for key, value in expected.items() if delta.get(key) != value]
                                    policy = native_mtp_depth_metadata(before, after)
                                    errors.extend(policy["errors"])
                                    wave.update(mtp_setting=mode, task=prompt["task"], prompt_tokens=prompt["prompt_tokens"],
                                                output_budget_tokens=budget, http_width=width, trial=trial - args.warmup + 1,
                                                warmup=trial < args.warmup, status_before=before, status_after=after,
                                                native_counter_delta=delta, counter_rates=counter_rates(delta),
                                                native_mtp_policy=policy, wave_validation_errors=errors,
                                                actual_native_decode_batches_by_width={f"b{n}": delta.get(f"scheduler.decode_batches_by_width.b{n}") for n in range(1, 5)},
                                                mtp_counters={key: value for key, value in delta.items() if key.startswith("mtp.") or key in ("metrics.drafted_tokens", "metrics.accepted_draft_tokens")})
                                    wave["post_wave_idle_pending"] = False
                                    wave["valid"] = wave["request_validation_valid"] and not errors
                                    report["summary"] = summary(report)
                                    save(args.output, report)
                                    print(json.dumps({key: wave[key] for key in ("mtp_setting", "task", "prompt_tokens", "output_budget_tokens", "http_width", "trial", "warmup", "valid", "counter_rates", "actual_native_decode_batches_by_width")}), flush=True)
                                    if not wave["valid"]:
                                        raise ValueError(f"invalid full-budget wave; checkpoint retained at {args.output}")
                    if args.semantic_plan is not None:
                        if mode not in ("standard", "3"):
                            raise ValueError("frozen semantic qualification supports --mtp standard and/or 3")
                        semantic_output = args.output.with_name(args.output.stem + f"-{mode}.semantic.json")
                        store = environment.get("SPLASH_FLASH_INT8_EXPERT_STORE")
                        if not store:
                            raise ValueError("semantic qualification requires an explicit expert store")
                        inventory = get_path(initial, "persisted_experts.expert_count")
                        if type(inventory) is not int or inventory % 48:
                            raise ValueError("semantic qualification expert inventory is unavailable")
                        quality_command = [sys.executable, str(ROOT / "dev/benchmarks/prefill4k_attribution_quality.py"),
                                           "measure", "--plan", str(args.semantic_plan), "--expert-store", store,
                                           "--runtime-build", str(args.binary.parent), "--inventory", str(inventory // 48),
                                           "--base-url", f"http://127.0.0.1:{args.port}", "--model", args.model,
                                           "--label", args.output.stem + "-" + mode, "--output", str(semantic_output),
                                           "--execution-mode", "standard" if mode == "standard" else "mtp3",
                                           "--run-root-gpu"]
                        derivative = get_path(initial, "identity.target_numerical_derivative_sha256")
                        if derivative:
                            quality_command.extend(["--target-derivative-sha256", derivative])
                        quality_result = subprocess.run(quality_command, cwd=ROOT, env=environment, check=False)
                        run["semantic_quality"] = {"command": quality_command, "exit_code": quality_result.returncode,
                                                   "report": str(semantic_output),
                                                   "report_sha256": sha256_file(semantic_output) if semantic_output.exists() else None}
                        save(args.output, report)
                    run["final_status"] = wait_idle(client, args, identity)
                finally:
                    run["unload_evidence"] = unload(server)
                    run["unloaded"] = run["unload_evidence"]["process_group_gone"]
                    run["ended_utc"] = utc_now()
                    save(args.output, report)
                    if not run["unloaded"]:
                        raise RuntimeError(run["unload_evidence"]["error"])


def csv_values(raw, allowed):
    values = raw.split(",")
    if not values or len(set(values)) != len(values) or any(value not in allowed for value in values):
        raise argparse.ArgumentTypeError(f"choose unique comma-separated values from {','.join(allowed)}")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mtp", type=lambda v: csv_values(v, ["standard", "off", "1", "2", "3", "4", "7", "8", "15"]), default=["3"])
    parser.add_argument("--contexts", type=lambda v: [int(x) for x in csv_values(v, ["2048", "4096", "8192"])], default=[2048, 4096, 8192])
    parser.add_argument("--batches", type=lambda v: [int(x) for x in csv_values(v, ["1", "2", "4"])], default=[1, 2, 4])
    parser.add_argument("--workloads", type=lambda v: csv_values(v, ["coding", "code", "count", "prose"]), default=["coding", "count", "prose"])
    parser.add_argument("--output-tokens", type=lambda v: [int(x) for x in csv_values(v, ["256", "1024"])], default=[256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", "--samples", type=int, default=3)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--lane-variation", choices=("shared", "seeded"), default="shared",
                        help="B1 retains the frozen prompt; seeded B2/B4 use independently nonced exact-count lanes")
    parser.add_argument("--lane-seed", default="splash-sep21-seeded-lanes-v1")
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-memory", default="auto")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--semantic-plan", type=Path, help="Run frozen semantic qualification in the same loaded process after measured waves")
    parser.add_argument("--idle-timeout", type=float, default=30)
    parser.add_argument("--status-timeout", type=float, default=5)
    parser.add_argument("--dry-run", "--prepare-only", action="store_true")
    parser.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args(argv)
    args.binary, args.output, args.package = (path.resolve() for path in (args.binary, args.output, args.package))
    args.tokenizer = (args.tokenizer or args.package).resolve()
    if args.semantic_plan is not None:
        args.semantic_plan = args.semantic_plan.resolve(strict=True)
    args.mtp = list(dict.fromkeys("standard" if mode == "off" else mode for mode in args.mtp))
    if args.semantic_plan is not None and any(mode not in ("standard", "3") for mode in args.mtp):
        parser.error("--semantic-plan supports --mtp standard and/or 3")
    args.workloads = list(dict.fromkeys("coding" if task == "code" else task for task in args.workloads))
    args.environment_overrides = {}
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"(?:SPLASH_FLASH_|FLASH_)[A-Z0-9_]+", key):
            parser.error("--env requires a SPLASH_FLASH_* or FLASH_* assignment")
        args.environment_overrides[key] = value
    if args.warmup < 0 or args.trials < 1 or not 1 <= args.port <= 65535:
        parser.error("warmup >= 0, trials >= 1, and a valid port are required")
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.startup_timeout, args.idle_timeout, args.status_timeout)):
        parser.error("timeouts must be finite and positive")
    if max(args.contexts) + max(args.output_tokens) > args.max_context:
        parser.error("maximum prompt plus full output budget exceeds --max-context")
    if args.dry_run and args.run_root_gpu:
        parser.error("--dry-run and --run-root-gpu are mutually exclusive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise ValueError("report already exists; choose a fresh --output path")
    plan = prepare_plan(args)
    environments, env_reports = {}, {}
    for mode in args.mtp:
        environments[mode], env_reports[mode] = environment_for(args, mode)
    report = {"schema": "splash-tuning-sep21-v1", "created_utc": utc_now(),
              "gpu_executed": False, "completed": False, "provenance": provenance(args),
              "settings": {"mtp": args.mtp, "contexts": args.contexts, "batches": args.batches,
                           "workloads": args.workloads, "output_tokens": args.output_tokens,
                           "warmup": args.warmup, "trials": args.trials, "max_context": args.max_context,
                           "lane_variation": args.lane_variation, "lane_seed": args.lane_seed,
                           "cache_tokens": 0, "reasoning": "none", "temperature": 0},
              "rate_definitions": {
                  "native_exact_request_decode": "metrics.decode.tokens / native first_token_to_done_ms; excludes the entire first speculative emission",
                  "native_prefill_command": "delta prefill_input_tokens / delta summed native prefill host command wall_ms; not request/wave wall time",
                  "native_prepared_target_predictions": "delta decode_output_tokens counts prepared verifier/AR target predictions, including rejected MTP rows; this is not output-token throughput",
                  "native_committed_output_command": "delta autoregressive_output_tokens / delta summed native decode command wall_ms or GPU_ms; full emitted usage includes the initial prompt prediction, and timing excludes prefill; use exact request decode for aligned steady output speed",
                  "native_sum_request_decode_interval": "sum exact post-first-emission tokens / sum per-request native decode intervals; concurrent intervals overlap, so this is not aggregate batch throughput",
                  "client_wave_decode": "sum exact native post-first-emission tokens / actual client earliest nonempty content to latest SSE DONE interval; HTTP clock, not native/GPU timing",
                  "full_wave_output": "sum actual full-usage completion tokens / actual concurrent HTTP wave wall interval"},
              "environments": env_reports, "plan": plan, "server_runs": [], "waves": [], "summary": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    if args.dry_run or not args.run_root_gpu:
        report["preparation_completed"] = True
        save(args.output, report)
        print(json.dumps({"report": str(args.output), "plan_sha256": plan["plan_sha256"],
                          "exact_prompts": len(plan["prompts"]), "gpu_executed": False,
                          "planned_waves": len(plan["prompts"]) * len(args.output_tokens) * len(args.batches) * len(args.mtp) * (args.warmup + args.trials)}))
        return 0
    try:
        report["gpu_executed"] = True
        launch_measure(args, report, environments)
        report["completed"] = True
        report["ended_utc"] = utc_now()
    except BaseException as error:
        report["error"] = repr(error)
        report["ended_utc"] = utc_now()
        save(args.output, report)
        raise
    save(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
