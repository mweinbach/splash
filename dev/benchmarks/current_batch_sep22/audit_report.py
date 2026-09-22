#!/usr/bin/env python3
"""Audit emitted-token batch reports; read JSON metadata only, submit no work."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import statistics

COUNTERS = ("completed_teacher_commands", "completed_pairs",
    "completed_original128_cache_prefix_windows", "completed_original128_cache_prefix_rows",
    "completed_bulk_commands", "completed_bulk_pairs", "completed_tail_commands", "completed_tail_pairs")


def get(record, path):
    for part in path.split("."):
        if not isinstance(record, dict):
            return None
        record = record.get(part)
    return record


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def trace_groups(path, widths):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    requests = {}
    owners = set()
    footer = None
    for row in records:
        if row.get("schema") != "splash-native-lifecycle-trace-sep22-v1":
            raise ValueError("native lifecycle trace schema differs")
        clock = row.get("clock_domain")
        if row.get("clock_kind") != "native-worker-std-steady-nanoseconds-v1" or not isinstance(clock, str):
            raise ValueError("native lifecycle trace clock domain differs")
        pid, start = row.get("native_worker_pid"), row.get("native_worker_start_token_ns")
        if type(pid) is not int or pid <= 0 or type(start) is not int or start <= 0 or clock != f"native-worker-std-steady-nanoseconds-v1:{pid}:{start}":
            raise ValueError("native lifecycle trace clock owner token differs")
        if type(row.get("steady_nanoseconds")) is not int or row["steady_nanoseconds"] < 0:
            raise ValueError("native lifecycle trace clock value is invalid")
        if type(row.get("instance_id")) is not int or row["instance_id"] <= 0:
            raise ValueError("native lifecycle trace worker identity is invalid")
        source = row.get("source_identity_sha256")
        if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{64}", source):
            raise ValueError("native lifecycle trace source identity is invalid")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("emitted_tokens", "prompt_tokens", "request_id", "generation")):
            raise ValueError("native lifecycle trace request/token metadata is invalid")
        owners.add((row["instance_id"], row["clock_domain"], source))
        key = (row["instance_id"], row["request_id"], row["generation"])
        event = row["event"]
        if event == "stop_flush_complete":
            if footer is not None or row is not records[-1] or row.get("flushed_lifecycle_records") != len(records) - 1:
                raise ValueError("native lifecycle stop-flush footer is incomplete/duplicated")
            footer = row
            continue
        if event not in ("first_emission", "done"):
            raise ValueError("unexpected native lifecycle event")
        if event in requests.setdefault(key, {}):
            raise ValueError("duplicate native lifecycle event")
        requests[key][event] = row
    if len(owners) != 1:
        raise ValueError("native lifecycle trace mixes worker/source clock identities")
    if footer is None:
        raise ValueError("native lifecycle trace has no completed graceful-stop flush footer")
    complete = []
    for key, pair in requests.items():
        if set(pair) != {"first_emission", "done"}:
            raise ValueError("native lifecycle trace has an incomplete request")
        first, done = pair["first_emission"], pair["done"]
        if done["steady_nanoseconds"] <= first["steady_nanoseconds"] or done["finish_reason"] != "length" or done["emitted_tokens"] <= first["emitted_tokens"]:
            raise ValueError("native lifecycle request interval/finish is invalid")
        complete.append((key, first, done))
    complete.sort(key=lambda row: row[1]["steady_nanoseconds"])
    if complete and footer["steady_nanoseconds"] < max(row[2]["steady_nanoseconds"] for row in complete):
        raise ValueError("native lifecycle flush footer precedes request Done")
    if len(complete) != sum(widths):
        raise ValueError("native trace request count differs from benchmark waves")
    groups, start = [], 0
    for width in widths:
        groups.append(complete[start:start + width])
        start += width
    for left, right in zip(groups, groups[1:]):
        if max(row[2]["steady_nanoseconds"] for row in left) >= min(row[1]["steady_nanoseconds"] for row in right):
            raise ValueError("native trace groups overlap across benchmark idle boundaries")
    return groups


def audit(report, groups=None):
    errors, waves, summarized = [], [], {}
    if report.get("schema") != "splash-tuning-sep21-v1" or report.get("completed") is not True or report.get("gpu_executed") is not True or report.get("error"):
        errors.append("original report is incomplete, failed, or did not execute GPU measurement")
    settings = report.get("settings", {})
    modes = settings.get("mtp", [])
    if len(modes) != 1 or modes[0] not in ("standard", "3"):
        errors.append("expected one fresh worker per standard/MTP3 mode")
    requirements = {"contexts": [2048], "workloads": ["coding"], "output_tokens": [256],
        "warmup": 1, "trials": 3, "max_context": 16384, "lane_variation": "shared",
        "cache_tokens": 0, "reasoning": "none", "temperature": 0}
    for key, value in requirements.items():
        if settings.get(key) != value:
            errors.append("frozen batch benchmark setting differs: " + key)
    if set(settings.get("batches", [])) != {2, 4}:
        errors.append("full B2/B4 benchmark matrix is absent")
    actual_keys = [(wave.get("mtp_setting"), wave.get("http_width"), wave.get("trial"), wave.get("warmup")) for wave in report.get("waves", [])]
    expected_keys = {(mode, width, trial, trial == 0) for mode in modes for width in (2, 4) for trial in range(4)}
    if len(actual_keys) != 8 or len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        errors.append("expected exactly one warmup and three measured trials for both native widths")
    if len(report.get("server_runs", [])) != 1:
        errors.append("fresh benchmark server-run evidence is absent or ambiguous")
    if any("semantic_quality" in run for run in report.get("server_runs", [])):
        errors.append("batch benchmark worker unexpectedly ran a semantic suite")
    if any(not run.get("unloaded") for run in report.get("server_runs", [])):
        errors.append("worker unload evidence is incomplete")
    for run in report.get("server_runs", []):
        evidence = run.get("unload_evidence", {})
        if evidence.get("process_group_gone") is not True or evidence.get("post_parent_exit_sigkill_required") is not False or evidence.get("returncode") == -9:
            errors.append("worker process group was not proven fully unloaded without SIGKILL")
    for index, wave in enumerate(report.get("waves", [])):
        problems = []
        width, budget = wave["http_width"], wave["output_budget_tokens"]
        if width not in (2, 4) or budget != 256 or wave["prompt_tokens"] != 2048:
            problems.append("wave geometry differs from frozen B2/B4 baseline")
        if not wave.get("valid") or not wave.get("request_validation_valid") or wave.get("post_wave_idle_pending"):
            problems.append("original strict request/idle/counter audit did not pass")
        delta = wave.get("native_counter_delta", {})
        records = wave.get("records", [])
        if len(records) != width:
            problems.append("real output record count differs from requested lanes")
        completion = sum(get(row, "measurement.actual_completion_tokens") or 0 for row in records)
        post_first = sum(get(row, "measurement.native_post_first_emission_tokens") or 0 for row in records)
        if completion != width * budget or delta.get("metrics.autoregressive_output_tokens") != completion:
            problems.append("actual emitted output does not reconcile with native completed tokens")
        for side in ("status_before", "status_after"):
            status = wave.get(side, {})
            if get(status, "identity.singleton_teacher_bulk_enabled") is not False or get(status, "mtp.singleton_teacher_bulk.requested") is not False:
                problems.append(side + ": singleton bulk route is unexpectedly active")
            for counter in COUNTERS:
                if get(status, "mtp.singleton_teacher_bulk." + counter) != 0:
                    problems.append(side + ": inactive singleton bulk counter is nonzero: " + counter)
        decode_widths = {f"b{n}": delta.get(f"scheduler.decode_batches_by_width.b{n}") for n in range(1, 5)}
        prefill_widths = {f"b{n}": delta.get(f"scheduler.prefill_batches_by_width.b{n}") for n in range(1, 5)}
        if not number(decode_widths[f"b{width}"]) or decode_widths[f"b{width}"] <= 0:
            problems.append("requested width was never an actual native decode/verifier graph")
        decode_ms = delta.get("metrics.decode_wall_ms")
        native_command_rate = post_first * 1000 / decode_ms if number(decode_ms) and decode_ms > 0 else None
        if not number(native_command_rate):
            problems.append("native emitted-token decode command duration is unavailable/invalid")
        exact_span = None
        if groups is not None:
            group = groups[index]
            first_ns = min(row[1]["steady_nanoseconds"] for row in group)
            last_ns = max(row[2]["steady_nanoseconds"] for row in group)
            seconds = (last_ns - first_ns) / 1e9
            trace_completed = sum(row[2]["emitted_tokens"] for row in group)
            trace_post_first = sum(row[2]["emitted_tokens"] - row[1]["emitted_tokens"] for row in group)
            clock_owner = group[0][1]
            for side in ("status_before", "status_after"):
                provenance = wave.get(side, {}).get("native_lifecycle_timestamps_sep22", {})
                if (provenance.get("enabled") is not True or
                    provenance.get("clock_domain") != clock_owner["clock_domain"] or
                    provenance.get("native_worker_pid") != clock_owner["native_worker_pid"] or
                    provenance.get("native_worker_start_token_ns") != clock_owner["native_worker_start_token_ns"] or
                    provenance.get("native_worker_instance_id") != clock_owner["instance_id"] or
                    provenance.get("source_identity_sha256") != clock_owner["source_identity_sha256"]):
                    problems.append(side + ": native trace owner/clock provenance differs from Worker status")
            if trace_completed != completion or trace_post_first != post_first:
                problems.append("native absolute trace emitted counts differ from strict request metrics")
            recorded_intervals = sorted(get(row, "measurement.native_first_emission_to_done_ms") for row in records)
            traced_intervals = sorted((row[2]["steady_nanoseconds"] - row[1]["steady_nanoseconds"]) / 1e6 for row in group)
            if any(not number(a) or abs(a - b) > 0.0011 for a, b in zip(recorded_intervals, traced_intervals)):
                problems.append("absolute native trace intervals differ from authoritative Done micros")
            exact_span = {"earliest_native_first_emission_nanoseconds": first_ns,
                "latest_native_done_nanoseconds": last_ns, "native_decode_span_seconds": seconds,
                "emitted_tokens_after_each_lane_first_sum": trace_post_first,
                "native_aggregate_decode_tokens_per_second": trace_post_first / seconds,
                "clock_domain": clock_owner["clock_domain"],
                "native_worker_instance_id": clock_owner["instance_id"],
                "native_worker_pid": clock_owner["native_worker_pid"],
                "native_worker_start_token_ns": clock_owner["native_worker_start_token_ns"],
                "source_identity_sha256": clock_owner["source_identity_sha256"],
                "scope": "actual emitted tokens after each lane first burst / common earliest native first emission to latest native Done; instrumented metadata trace"}
        row = {"index": index, "mode": wave["mtp_setting"], "width": width,
            "trial": wave["trial"], "warmup": wave["warmup"], "valid": not problems,
            "errors": problems, "actual_completion_tokens": completion,
            "actual_native_post_first_emission_tokens": post_first,
            "actual_native_decode_graph_counts_by_width": decode_widths,
            "actual_native_prefill_graph_counts_by_width": prefill_widths,
            "native_emitted_decode_tokens_per_summed_decode_host_second": native_command_rate,
            "native_prefill_tokens_per_summed_prefill_host_second": get(wave, "counter_rates.native_prefill_tokens_per_summed_command_wall_second"),
            "client_aggregate_decode_tokens_per_second": wave.get("client_wave_post_first_emission_tokens_per_second"),
            "exact_native_common_span": exact_span,
            "prepared_target_prediction_counts_used_as_decode_numerator": False}
        waves.append(row)
        errors.extend(f"wave {index}: {error}" for error in problems)
        if not wave["warmup"] and not problems:
            summarized.setdefault((row["mode"], width), []).append(row)
    summary = []
    for (mode, width), rows in summarized.items():
        if len(rows) != 3:
            errors.append(f"{mode}/B{width}: expected exactly three valid measured trials")
        rates = [get(row, "exact_native_common_span.native_aggregate_decode_tokens_per_second") for row in rows]
        summary.append({"mode": mode, "width": width, "measured_trials": len(rows),
            "exact_native_common_span_median_tokens_per_second": statistics.median(rates) if all(number(rate) for rate in rates) else None,
            "native_emitted_decode_command_median_tokens_per_second": statistics.median(row["native_emitted_decode_tokens_per_summed_decode_host_second"] for row in rows),
            "native_prefill_median_tokens_per_second": statistics.median(row["native_prefill_tokens_per_summed_prefill_host_second"] for row in rows)})
    return {"schema": "splash-current-batch-emitted-token-audit-sep22-v1", "valid": not errors,
        "errors": errors, "gpu_executed_by_auditor": False, "payload_reads": 0, "hashes_computed": 0,
        "B2_B4_semantic_qualification_claimed": False, "supported_native_lane_width_maximum": 4,
        "native_common_span_available": groups is not None,
        "native_common_span_unavailable_reason": None if groups is not None else "current wire Done exposes request-relative intervals; no shared absolute native emission clock was recorded",
        "waves": waves, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--native-lifecycle-trace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh metadata audit output")
    report = json.loads(args.report.read_text())
    groups = trace_groups(args.native_lifecycle_trace, [wave["http_width"] for wave in report["waves"]]) if args.native_lifecycle_trace else None
    result = audit(report, groups)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("waves", "errors")}))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
