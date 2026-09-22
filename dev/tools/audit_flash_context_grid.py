"""CPU-only independent audit and corrected summary of a Flash context grid.

Reads an immutable snapshot of the benchmark JSON and its CPU input plan. It
does not contact a server, load a model/tokenizer, or change the source report.
Output speed uses native exact post-first-emission token accounting; TTFT uses
the client's first nonempty SSE content timestamp. Incomplete runs require
--allow-partial and are prominently identified as partial in both outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = (1024, 4096, 16384, 32768, 65536)
OUTPUTS = (128, 512, 4096)
TRIALS = 3
CONTEXT = 262144
SCHEMA = "splash-flash-context-grid-independent-audit-v1"


def at(document, path):
    for key in path.split("."):
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (
        value > 0 if positive else value >= 0
    )


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def stats(values):
    return {"mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def audit_status(status, identity, model, label):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(f"{label}: {message}")
    check(isinstance(status, dict), "status missing")
    check(at(status, "identity") == identity, "native identity changed")
    check(at(status, "maximum_context_tokens") == CONTEXT, "context differs from 256K")
    check(at(status, "instance.model") == model, "model identity differs")
    for path in ("ready", "metal.healthy", "transport.ready", "memory_audit.valid"):
        check(at(status, path) is True, f"{path} is not true")
    for path in ("transport.status_stale", "transport.recovering",
                 "capabilities.prefix_cache", "ple_storage.table_gpu_buffers_present"):
        check(at(status, path) is False, f"{path} is not false")
    check(at(status, "capabilities.native_route") == "flash-next", "native route differs")
    check(at(status, "ple_storage.ssd_streaming_enabled") is True, "SSD streaming disabled")
    check(at(status, "mtp.enabled") is True, "MTP disabled")
    check(at(status, "memory_pressure") == "normal", "memory pressure is not normal")
    for path in ("scheduler.active_requests", "scheduler.prefilling", "scheduler.decoding",
                 "scheduler.waiting_mask", "scheduler.queued", "transport.pending",
                 "frontend.active", "frontend.waiting", "http.requests.active"):
        check(at(status, path) == 0, f"{path} is not idle")
    check(at(status, "scheduler.command_in_flight") is False, "model command in flight")
    current, limit = at(status, "memory_actual.current_bytes"), at(status, "memory_governor.limit_bytes")
    check(finite(current) and finite(limit, positive=True) and current <= limit,
          "native memory allocation/limit invalid")
    return errors


def audit_row(row, item, identity, model):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    response = row.get("response", {})
    measurements = row.get("measurements", {})
    check(row.get("valid") is True and row.get("errors") == [], "original attempt invalid")
    check(row.get("full_output_budget") is True and row.get("early_eos") is False,
          "full output budget not proved")
    for key in ("trial", "prompt_tokens", "output_budget_tokens", "prompt_u32le_sha256",
                "request_body_sha256"):
        check(row.get(key) == item[key], f"{key} differs from input plan")
    check(row.get("frontend_prompt_token_count") == item["prompt_tokens"], "frontend input length differs")
    check(row.get("frontend_prompt_u32le_sha256") == item["prompt_u32le_sha256"],
          "frontend rendered token digest differs")
    check(response.get("http_status") == 200, "HTTP status not 200")
    check(response.get("stream") is True and response.get("done") is True, "incomplete SSE stream")
    check(response.get("content_type", "").startswith("text/event-stream"), "response is not SSE")
    check(response.get("finish_reason") == "length", "finish reason is not length")
    check(response.get("errors") == [] and response.get("error_frames") == [], "SSE parser/server errors")
    check(isinstance(response.get("text"), str) and bool(response.get("text")), "output content missing")
    check(response.get("reasoning_text") == "" and response.get("tool_calls") == [],
          "unexpected reasoning/tool output")
    request_ids = response.get("request_ids", [])
    check(isinstance(request_ids, list) and len(request_ids) == 1 and bool(request_ids[0]),
          "stream request identity inconsistent")
    check(response.get("model_ids") == [model], "stream model identity inconsistent")
    prompt, output = item["prompt_tokens"], item["output_budget_tokens"]
    for path, expected in (("usage.prompt_tokens", prompt), ("usage.completion_tokens", output),
                           ("usage.total_tokens", prompt + output),
                           ("usage.prompt_tokens_details.cached_tokens", 0),
                           ("usage.completion_tokens_details.reasoning_tokens", 0),
                           ("metrics.cache.matched_tokens", 0), ("metrics.cache.capacity", CONTEXT),
                           ("metrics.prefill.tokens", prompt)):
        check(type(at(response, path)) is int and at(response, path) == expected,
              f"{path} differs from exact accounting")
    check(measurements.get("measured_completion_tokens") == output, "measurement output count differs")
    ttft = measurements.get("ttft_ms")
    check(finite(ttft, positive=True) and ttft == response.get("first_content_ms"),
          "client first-content TTFT invalid or inconsistent")
    for key in ("output_tokens_per_second", "http_total_seconds"):
        check(finite(measurements.get(key), positive=True), f"auxiliary client {key} invalid")
    decode = at(response, "metrics.decode.tokens")
    latency = at(response, "metrics.request_latency") or {}
    rate, interval = latency.get("stream_tokens_per_second"), latency.get("first_token_to_done_ms")
    first_batch = output - decode if type(decode) is int else None
    check(type(decode) is int and decode > 0 and 1 <= first_batch <= 16,
          "native exact first-emission accounting invalid")
    check(finite(rate, positive=True) and finite(interval, positive=True), "native output rate/interval invalid")
    if finite(rate, positive=True) and finite(interval, positive=True) and type(decode) is int:
        check(math.isclose(rate, decode * 1000 / interval, rel_tol=1e-9, abs_tol=1e-7),
              "native rate differs from exact token/interval computation")
    for key in ("wall_ms", "ttft_ms", "queue_to_start_ms", "start_to_first_token_ms"):
        check(finite(latency.get(key)), f"native {key} invalid")
    if all(finite(latency.get(key)) for key in ("wall_ms", "ttft_ms", "first_token_to_done_ms")):
        check(math.isclose(latency["wall_ms"], latency["ttft_ms"] + latency["first_token_to_done_ms"],
                           rel_tol=1e-9, abs_tol=1e-6), "native lifecycle intervals inconsistent")
    before, after = row.get("status_before", {}), row.get("status_after", {})
    errors += audit_status(before, identity, model, "before")
    errors += audit_status(after, identity, model, "after")
    btime, atime = at(before, "status_snapshot.steady_seconds"), at(after, "status_snapshot.steady_seconds")
    check(finite(btime) and finite(atime) and atime > btime, "post-request native snapshot not fresh")
    delta = row.get("native_counter_delta", {})
    expected = {"requests.submitted": 1, "requests.completed": 1, "requests.cancelled": 0,
                "requests.failed": 0, "metrics.metal_failures": 0,
                "scheduler.prefill_rows": prompt, "metrics.prefill_input_tokens": prompt,
                "metrics.autoregressive_output_tokens": output,
                **{f"scheduler.decode_batches_by_width.b{width}": 0 for width in (2, 3, 4)}}
    for path, expected_value in expected.items():
        left, right = at(before, path), at(after, path)
        actual = right - left if finite(left) and finite(right) else None
        check(actual == expected_value and delta.get(path) == actual, f"isolated counter {path} invalid")
    drafted, accepted = delta.get("metrics.drafted_tokens"), delta.get("metrics.accepted_draft_tokens")
    check(finite(drafted) and finite(accepted) and accepted <= drafted, "MTP acceptance counters invalid")
    for path in ("metrics.drafted_tokens", "metrics.accepted_draft_tokens"):
        left, right = at(before, path), at(after, path)
        check(finite(left) and finite(right) and delta.get(path) == right - left,
              f"MTP counter {path} differs from snapshots")
    return {"id": row.get("id"), "trial": item["trial"], "prompt_tokens": prompt,
            "output_budget_tokens": output, "valid": not errors, "errors": errors,
            "client_ttft_ms": ttft, "native_output_tokens_per_second": rate,
            "native_post_first_emission_tokens": decode, "native_first_emission_tokens": first_batch,
            "native_output_interval_ms": interval,
            "native_queue_to_start_ms": latency.get("queue_to_start_ms"),
            "native_start_to_first_token_ms": latency.get("start_to_first_token_ms"),
            "native_ttft_ms": latency.get("ttft_ms"),
            "http_full_usage_output_tokens_per_second_auxiliary": measurements.get("output_tokens_per_second"),
            "http_total_seconds": measurements.get("http_total_seconds"),
            "mtp_drafted_tokens": drafted, "mtp_accepted_committed_drafts": accepted,
            "mtp_acceptance_fraction": accepted / drafted if finite(drafted, positive=True) and finite(accepted) else None}


def audit(source, plan_path, allow_partial):
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != "splash-flash-context-grid-report-v1":
        raise ValueError("unsupported source report schema")
    plan_path = plan_path or Path(report["plan_path"])
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    saved_hash = plan.pop("plan_sha256", None)
    if (plan.get("schema") != "splash-flash-context-grid-v1" or digest(plan) != saved_hash
            or report.get("plan_sha256") != saved_hash):
        raise ValueError("CPU plan/report hash proof failed")
    if (tuple(plan["prompt_token_targets"]) != PROMPTS or tuple(plan["output_token_budgets"]) != OUTPUTS
            or plan["trials"] != TRIALS or plan["configured_context_tokens"] != CONTEXT
            or plan.get("concurrency") != 1):
        raise ValueError("CPU plan does not describe the complete requested 15-cell, 3-trial, 256K matrix")
    items = {item["id"]: item for item in plan["requests"]}
    expected_ids = {f"p{prompt}-o{output}-t{trial}" for prompt in PROMPTS for output in OUTPUTS
                    for trial in range(1, TRIALS + 1)}
    if len(plan["requests"]) != 45 or set(items) != expected_ids:
        raise ValueError("CPU plan request IDs are incomplete or duplicated")
    model = plan["model"]
    for item in items.values():
        if digest(item["body"]) != item["request_body_sha256"]:
            raise ValueError(f"CPU body hash invalid: {item['id']}")
        body = item["body"]
        if (item["id"] != f"p{item['prompt_tokens']}-o{item['output_budget_tokens']}-t{item['trial']}"
                or item["prompt_tokens"] + item["output_budget_tokens"] > CONTEXT
                or body.get("model") != model or body.get("max_completion_tokens") != item["output_budget_tokens"]
                or body.get("reasoning_effort") != "none" or body.get("temperature") != 0
                or body.get("stream") is not True or at(body, "stream_options.include_usage") is not True):
            raise ValueError(f"CPU request configuration invalid: {item['id']}")
    rows = report.get("results", [])
    ids = [row.get("id") for row in rows]
    coverage_complete = len(rows) == 45 and set(ids) == expected_ids and len(set(ids)) == 45
    if not allow_partial and (report.get("state") != "complete" or not coverage_complete):
        raise ValueError(f"source run is incomplete ({report.get('state')}, {len(rows)}/45); use --allow-partial explicitly")
    errors = []
    if len(set(ids)) != len(ids) or any(identifier not in items for identifier in ids):
        errors.append("source contains duplicated or unknown request IDs")
    if report.get("errors"):
        errors.append("source run contains errors: " + "; ".join(report["errors"]))
    for key in ("prompt_token_targets", "output_token_budgets", "trials", "configured_context_tokens"):
        if report.get(key) != plan[key]:
            errors.append(f"source {key} differs from CPU plan")
    initial = report.get("initial_status", {})
    identity = initial.get("identity")
    if not isinstance(identity, dict) or type(identity.get("engine_instance_id")) is not int:
        errors.append("initial native engine identity missing")
    errors += audit_status(initial, identity, model, "initial")
    derived = [audit_row(row, items[row["id"]], identity, model) for row in rows if row.get("id") in items]
    stream_ids = [at(row, "response.request_ids")[0] for row in rows
                  if isinstance(at(row, "response.request_ids"), list) and len(at(row, "response.request_ids")) == 1]
    if len(set(stream_ids)) != len(stream_ids):
        errors.append("HTTP stream request IDs repeated across attempts")
    if report.get("state") == "complete":
        errors += audit_status(report.get("final_status", {}), identity, model, "final")
    cells = []
    for prompt in PROMPTS:
        for output in OUTPUTS:
            attempts = [row for row in derived if row["prompt_tokens"] == prompt and row["output_budget_tokens"] == output]
            valid = [row for row in attempts if row["valid"]]
            valid_trials = [row["trial"] for row in valid]
            complete = sorted(valid_trials) == [1, 2, 3]
            tokens = sum(row["native_post_first_emission_tokens"] for row in valid)
            interval = sum(row["native_output_interval_ms"] for row in valid)
            drafted = sum(row["mtp_drafted_tokens"] for row in valid)
            accepted = sum(row["mtp_accepted_committed_drafts"] for row in valid)
            cells.append({"prompt_tokens": prompt, "output_budget_tokens": output,
                          "expected_trials": TRIALS, "attempted_trials": len(attempts),
                          "valid_trials": len(valid), "valid_trial_ids": valid_trials,
                          "state": "complete" if complete else "partial" if valid else "failed" if attempts else "pending",
                          "client_ttft_ms": stats([row["client_ttft_ms"] for row in valid]),
                          "native_output_tokens_per_second": stats([row["native_output_tokens_per_second"] for row in valid]),
                          "native_output_tokens_per_second_pooled": tokens * 1000 / interval if interval else None,
                          "native_post_first_emission_tokens_total": tokens,
                          "native_output_interval_ms_total": interval,
                          "native_queue_to_start_ms": stats([row["native_queue_to_start_ms"] for row in valid]),
                          "native_start_to_first_token_ms": stats([row["native_start_to_first_token_ms"] for row in valid]),
                          "http_full_usage_output_tokens_per_second_auxiliary": stats([
                              row["http_full_usage_output_tokens_per_second_auxiliary"] for row in valid]),
                          "http_total_seconds": stats([row["http_total_seconds"] for row in valid]),
                          "mtp_drafted_tokens": drafted, "mtp_accepted_committed_drafts": accepted,
                          "mtp_acceptance_fraction": accepted / drafted if drafted else None})
    statuses = [initial, report.get("final_status", {})] + [row.get(key, {}) for row in rows
                                                            for key in ("status_before", "status_after")]
    memory = {}
    for path in ("memory_actual.current_bytes", "memory_actual.peak_bytes", "memory_audit.state_bytes_per_request",
                 "memory_audit.mtp_extra_state_bytes_per_eligible_request", "memory_governor.host_available_bytes",
                 "memory_governor.limit_bytes", "memory_governor.host_reserve_bytes"):
        values = [at(status, path) for status in statuses if finite(at(status, path))]
        memory[path] = {"initial": at(initial, path), "final": at(report.get("final_status", {}), path),
                        "min_observed_snapshot": min(values) if values else None,
                        "max_observed_snapshot": max(values) if values else None}
    passed = not errors and all(row["valid"] for row in derived)
    complete = passed and report.get("state") == "complete" and coverage_complete and all(cell["state"] == "complete" for cell in cells)
    return {"schema": SCHEMA, "audited_utc": datetime.now(timezone.utc).isoformat(),
            "state": "complete" if complete else "partial" if passed else "audit_failed",
            "audit_passed": passed, "coverage_complete": coverage_complete,
            "source_report": str(source.resolve()), "source_report_sha256": hashlib.sha256(raw).hexdigest(),
            "source_state": report.get("state"), "source_updated_utc": report.get("updated_utc"),
            "source_plan": str(plan_path.resolve()), "source_plan_file_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "plan_sha256": saved_hash, "model": model, "configured_context_tokens": CONTEXT,
            "concurrency": 1, "planned_attempts": 45, "audited_attempts": len(derived),
            "valid_attempts": sum(row["valid"] for row in derived), "errors": errors,
            "rate_definition": "Exact native tokens after first native emission / first native emission to Done interval; arithmetic mean across valid full-budget trials. Pooled rate uses summed exact token and interval counts.",
            "ttft_definition": "Client HTTP request start to first nonempty SSE content delivery; includes HTTP transfer, frontend prompt preparation, native admission/cache allocation, prefill and first output delivery.",
            "diagnostic_caveats": ["queue_to_start includes native admission and full-capacity cache allocation/zeroing; it is not solely contention.",
                                   "HTTP full-usage first-to-last-content rate includes the first chunk in its numerator and is auxiliary only.",
                                   "Memory snapshot ranges are observations at safe points; native peak_bytes is the process lifetime GPU allocation peak and includes pre-run warmup."],
            "initial_identity": identity, "memory": memory, "cells": cells, "results": derived}


def markdown(document):
    def fmt(value, scale=1):
        return f"{value * scale:.2f}" if isinstance(value, (int, float)) else "—"
    lines = [f"# Flash-Next independent context audit ({document['state']})", "",
             f"256K configured context; one request at a time; SSD n-gram streaming; reasoning disabled; no prefix reuse. "
             f"Audited **{document['audited_attempts']}/45** attempts; **{document['valid_attempts']}** valid. "
             f"Three trials per input/output cell.", "",
             "TTFT is measured at the HTTP client. Output speed is the native exact rate after the first native token emission; "
             "tokens in that first emission are excluded from its interval. All means exclude incomplete or invalid attempts.", "",
             "| Input | Output | Valid trials | Mean TTFT (s) | Mean output (tok/s) | Median output | Output range | Pooled output | MTP accepted/drafted |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in document["cells"]:
        rate = cell["native_output_tokens_per_second"]
        lines.append(f"| {cell['prompt_tokens']:,} | {cell['output_budget_tokens']:,} | {cell['valid_trials']}/3 | "
                     f"{fmt(cell['client_ttft_ms']['mean'], .001)} | {fmt(rate['mean'])} | {fmt(rate['median'])} | "
                     f"{fmt(rate['min'])}–{fmt(rate['max'])} | {fmt(cell['native_output_tokens_per_second_pooled'])} | "
                     f"{fmt(cell['mtp_acceptance_fraction'], 100)}% |")
    lines += ["", "Native queue-to-start timing includes full-capacity cache allocation and zeroing. "
              "MTP acceptance counts committed verifier drafts. Auxiliary HTTP rates and memory observations remain in JSON."]
    if document["state"] != "complete":
        lines += ["", "**This is not a complete 45-attempt benchmark. Pending or invalid cells require qualification.**"]
    problems = document["errors"] + [f"{row['id']}: " + "; ".join(row["errors"])
                                      for row in document["results"] if row["errors"]]
    if problems:
        lines += ["", "Audit errors:", ""] + ["- " + problem for problem in problems]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--plan", type=Path, help="Override source report's CPU plan location")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.source.with_name(args.source.stem + "-audited.json")
    md = args.markdown or output.with_suffix(".md")
    if output.resolve() == args.source.resolve() or md.resolve() == args.source.resolve() or output.resolve() == md.resolve():
        parser.error("derived outputs must differ from the source and each other")
    if not args.overwrite and (output.exists() or md.exists()):
        parser.error("derived output already exists; use fresh paths or --overwrite")
    try:
        document = audit(args.source, args.plan, args.allow_partial)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    if Path(document["source_plan"]).resolve() in (output.resolve(), md.resolve()):
        parser.error("derived outputs must not overwrite the CPU input plan")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown(document))
    print(json.dumps({"state": document["state"], "audit_passed": document["audit_passed"],
                      "valid_attempts": document["valid_attempts"], "planned_attempts": 45,
                      "output": str(output), "markdown": str(md), "gpu_executed": False}))
    return 0 if document["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
