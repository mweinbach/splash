#!/usr/bin/env python3
"""Root-only projection of benchmark controls and numeric timing metadata.

The authoring agent must not run this on original reports. Root runs it and
shares the small envelope. No prompt, generation, token or tensor arrays are
exported, and no file referenced by a report is opened.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re

FORBIDDEN = {
    "content", "prompt", "prompts", "messages", "text", "output_text",
    "generated_text", "generation", "generations", "reasoning_text", "tool_calls",
    "token_ids", "tokens", "prompt_tokens_ids", "completion_token_ids",
    "input", "inputs", "tensor", "tensors", "capture", "captures", "data",
    "events", "chunks", "raw_stream", "responses", "response", "body",
}
STRING_KEYS = {
    "schema", "created_utc", "ended_utc", "model", "model_id", "package",
    "tokenizer", "binary", "metallib", "profile", "profile_name", "local_profile",
    "mtp_setting", "task", "workload", "reasoning", "finish_reason", "status",
    "lane_variation", "python", "python_version", "git_head", "path", "name",
    "type", "mode", "quantization", "quantization_policy", "norm_convention",
    "profile_mode", "profiling_mode", "command_dispatch_profiling_mode",
    "target_policy_routes", "policy", "source_id", "source_layout_id",
}
METADATA_CONTAINERS = {
    "settings", "provenance", "files", "environments", "resolved_flash_environment",
    "explicit_environment", "qualified_launcher_defaults", "identity", "metrics",
    "counter_rates", "mtp_counters", "native_counter_delta", "status_snapshot",
    "request_latency", "prefill", "decode", "cache", "measurement", "usage",
    "prompt_tokens_details", "completion_tokens_details", "native_mtp_policy",
    "acceptance_histogram", "accepted_prefix_histogram", "verification_histogram",
    "summary", "waves", "records", "server_runs", "initial_status", "final_status",
    "frontend_prompt_witnesses", "lane_prompt_identity", "samples", "cases",
    "results", "controls", "timings", "rate_definitions", "variant", "variants",
}
NUMERIC_KEYS = {
    "completed", "gpu_executed", "valid", "warmup", "trial", "trials", "valid_trials",
    "warmups", "max_context", "max_context_tokens", "prompt_tokens", "completion_tokens",
    "total_tokens", "output_budget_tokens", "output_tokens", "http_width", "batch",
    "width", "lane", "count", "bytes", "temperature", "cache_tokens", "cached_tokens",
    "reasoning_tokens", "done", "http_status", "returncode", "exitcode", "mtp_depth",
    "draft_depth", "verification_cycles", "accepted_committed_drafts", "shortened",
    "full_output_budget", "actual_completion_tokens", "actual_http_maximum_overlap",
    "encoder_boundaries_altered", "sampling_barriers", "lane_seed", "benchmark_index",
}
NUMERIC_NAME = re.compile(r"(tokens_per_second|tokens_per_summed|tokens_per_|_tps$|_ms$|_seconds$|_seconds_sum$|_calls$|_rows$|_tokens$|_cycles$|_drafts$|_batches$|^mtp\.|^decode\.|^prefill\.|^cache\.|^request_latency\.|^target_verify\.|^head\.)")
DIGEST_NAME = re.compile(r"(sha256|fingerprint|source_id|source_layout_id)$")
DIGEST_VALUE = re.compile(r"[0-9a-f]{64}$")
NUMERIC_LIST_KEYS = {"contexts", "batches", "output_tokens", "actual_native_decode_batches_by_width"}
ENUM_LIST_KEYS = {"mtp", "workloads"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    return value is None or type(value) is bool or type(value) is int or (type(value) is float and math.isfinite(value))


def project(value, parent="", allow_numeric_map=False):
    """Only explicitly named metadata sections and scalar leaf types survive."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key == "tokens" and parent in {"decode", "prefill", "cache"} and scalar(child):
                out[key] = child
                continue
            if not isinstance(key, str) or len(key) > 256 or key in FORBIDDEN:
                continue
            if key in METADATA_CONTAINERS:
                selected = project(child, key, key.endswith("histogram") or key == "actual_native_decode_batches_by_width")
                if selected not in ({}, []): out[key] = selected
            elif key.startswith("SPLASH_") and parent in {
                    "environments", "resolved_flash_environment", "explicit_environment", "qualified_launcher_defaults"}:
                if type(child) in (str, int, float, bool) and len(str(child)) <= 4096: out[key] = child
            elif DIGEST_NAME.search(key) and isinstance(child, str) and DIGEST_VALUE.fullmatch(child):
                out[key] = child
            elif key == "source_file_sha256" and isinstance(child, dict):
                out[key] = {k: v for k, v in child.items() if isinstance(k, str) and len(k) < 4096
                            and isinstance(v, str) and DIGEST_VALUE.fullmatch(v)}
            elif key in STRING_KEYS and isinstance(child, str) and len(child) <= 8192:
                out[key] = child
            elif (key in NUMERIC_KEYS or NUMERIC_NAME.search(key) or (allow_numeric_map and key.isdecimal())) and scalar(child):
                out[key] = child
            elif key in NUMERIC_LIST_KEYS and parent == "settings" and isinstance(child, list) and len(child) <= 256 and all(scalar(x) for x in child):
                out[key] = child
            elif key in ENUM_LIST_KEYS and isinstance(child, list) and len(child) <= 64:
                out[key] = [x for x in child if isinstance(x, str) and re.fullmatch(r"standard|[1-9][0-9]?|coding|general|math|reasoning", x)]
            elif parent in {"environments", "files"} and isinstance(child, dict):
                selected = project(child, parent)
                if selected: out[key] = selected
            elif parent == "rate_definitions" and isinstance(child, str) and len(child) <= 1024:
                out[key] = child
        return out
    if isinstance(value, list) and parent in METADATA_CONTAINERS:
        return [selected for item in value if isinstance(item, dict)
                and (selected := project(item, parent))]
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-metadata-only", action="store_true")
    args = parser.parse_args()
    if not args.root_metadata_only:
        raise ValueError("Original-report reading is Root-only; pass --root-metadata-only explicitly")
    output = args.output.resolve()
    if output.exists(): raise ValueError("A fresh metadata envelope path is required")
    reports = []
    for supplied in args.report:
        path = supplied.resolve()
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict): raise ValueError("Benchmark report root must be an object")
        reports.append({"report_path": str(path), "report_bytes": path.stat().st_size,
                        "report_sha256": digest(path),
                        "top_level_key_types": {k: type(v).__name__ for k, v in raw.items()},
                        "selected_metadata": project(raw),
                        "exported_prompt_generation_token_tensor_arrays": False})
    envelope = {"schema": "Root-benchmark-native-baseline-continuity-metadata-v1",
                "Root_extracted_original_reports": True, "extractor_sha256": digest(Path(__file__)),
                "referenced_report_model_capture_tokenizer_files_opened": False,
                "reports": reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"envelope": str(output), "sha256": digest(output), "reports": len(reports),
                      "prompt_generation_token_tensor_arrays_exported": False}))


if __name__ == "__main__":
    main()
