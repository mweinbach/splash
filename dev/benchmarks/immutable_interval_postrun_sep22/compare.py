#!/usr/bin/env python3
"""Root-only postrun auditing of the frozen samebinary interval0/1 campaign.

No measured helper, runtime, report, task body, budget or grader is modified.
Actual normal/semantic reports contain generation data and are read only when
Root executes this program with four externally typed complete-file hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path("/Users/mweinbach/Projects/splash")
sys.path.insert(0, str(ROOT))
NORMAL = ROOT / "build/immutable96-index-Q4-normal-original22-sep22-v3"
HELPER_SHA = "11238da14eb743f62111e9fc7576ebcc4bc5d08e902ea762832a35d50d485dff"
DRIVER_SHA = "b37e3714043bed599636c41363a50af49323fe22a22be928bf644746c33fcb80"
NATIVE = ROOT / "build/release/flash/sep22-immutable96-index-Q4-Root-native-admission-v1.json"
NATIVE_SHA = "bbe18982a9e8dc2c9296e3ad7f5169af1e1ea63e00a9003d5ea84f36902bcde6"
PLAN_SHA = "a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac"
FLAG = "SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22"
BEST_PARENT_NATIVE_DECODE = 53.225974

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def module(path, expected, name):
    require(sha(path) == expected, "immutable source dependency drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def load_original_chain():
    own = module(NORMAL / "semantic_quality.py", HELPER_SHA, "_postrun_frozen_immutable96")
    own.authenticate(own.BUILD, NATIVE, NATIVE_SHA, True)
    raw = own.parent_module()
    # This admits the unchanged raw parent solely for its own origin. The new
    # child was separately admitted above; no old receipt admits the new child.
    runner = raw.load(own.PARENT, expected=True, require_state=True)
    return own, runner

class RecordedIndexFlags:
    """Only complete explicitly registered contexts can dispatch index0/1."""
    def __init__(self, own, http):
        self.own, self.http = own, http
        self.hooks = {flag: own.index_hooks(http, flag) for flag in (False, True)}
        self.registered = {}

    def bind(self, reports, plan):
        require(len(reports) == 2, "one original-loop and one indexed report required")
        frozen = {case["id"]: case for case in plan["cases"]}
        require(len(frozen) == 22 and plan["content_sha256"] == PLAN_SHA, "all22 unchanged original cases required")
        staged = {}
        for report, flag in zip(reports, (False, True)):
            required = {"schema": "splash-prefill4k-semantic-report-v1", "execution_mode": "mtp3",
                        "completed": True, "full_plan_coverage": True,
                        "strict_cache_graph_coverage_required": True,
                        "runtime_file_sha256": {"splash-flash": self.own.EXE, "splash.metallib": self.own.LIB},
                        "plan_content_sha256": PLAN_SHA}
            require(all(self.own.same(report.get(k), v) for k, v in required.items()),
                    "complete exact current runtime/original22 report required")
            cases = report.get("cases")
            require(isinstance(cases, list) and len(cases) == 22 and {case.get("id") for case in cases} == set(frozen),
                    "all22 unique original case contexts required")
            initial = report.get("initial_status")
            require(isinstance(initial, dict) and isinstance(initial.get("identity"), dict), "initial per-run identity required")
            values = [initial]
            for case in cases:
                before, after = case.get("status_before"), case.get("status_after")
                _, errors = self.hooks[flag][1](before, after, frozen[case["id"]], True, "mtp3")
                require(not errors, "saved index case profile/counters invalid: " + "; ".join(errors))
                values.extend((before, after))
            values.append(report.get("final_status"))
            previous = None
            for status in values:
                errors = self.hooks[flag][0](status, plan, report.get("store_witness"), "mtp3")
                require(not errors, "saved index source/flag/census invalid: " + "; ".join(errors))
                require(status.get("identity") == initial["identity"], "exact within-run identity drift")
                current = {key: self.http.get_path(status, self.own.SECTION + "." + key) for key in self.own.COUNTERS}
                if previous is not None:
                    require(all(current[k] >= previous[k] for k in current), "process index counters decreased")
                previous = current
                key = fingerprint(status)
                require(key not in staged or staged[key] is flag, "conflicting saved flag context")
                staged[key] = flag
        self.registered = staged
        return self

    def flag_for(self, status):
        key = fingerprint(status)
        require(key in self.registered, "unregistered complete saved status")
        return self.registered[key]

    def status(self, status, plan, store, execution_mode="mtp3"):
        try:
            flag = self.flag_for(status)
        except (ValueError, TypeError):
            return ["unregistered saved index status"]
        return self.hooks[flag][0](status, plan, store, execution_mode)

    def coverage(self, before, after, case, require_counters=True, execution_mode="mtp3"):
        try:
            a, b = self.flag_for(before), self.flag_for(after)
        except (ValueError, TypeError):
            return {}, ["unregistered saved index coverage context"]
        if a is not b:
            return {}, ["mixed index flags within a request"]
        return self.hooks[a][1](before, after, case, require_counters, execution_mode)

    def ownership(self, status):
        return self.hooks[self.flag_for(status)][2](status)

def install(runner, dispatch):
    old_status, old_coverage, old_ownership = runner.gate_status, runner.coverage, runner.ownership_policy
    def status(*args, **kwargs):
        return old_status(*args, **kwargs) + dispatch.status(*args, **kwargs)
    def coverage(*args, **kwargs):
        details, errors = old_coverage(*args, **kwargs)
        extra, added = dispatch.coverage(*args, **kwargs)
        return {**details, **extra}, errors + added
    runner.gate_status, runner.coverage = status, coverage
    runner.ownership_policy = lambda value: {**old_ownership(value), **dispatch.ownership(value)}
    return runner

def normal_audit(report, semantic, semantic_path, semantic_sha, own, runner, driver, expected, plan):
    require(report.get("schema") == "splash-tuning-sep21-v1" and report.get("completed") is True and report.get("gpu_executed") is True and "error" not in report,
            "completed normal Root run required")
    wanted = {"mtp": ["3"], "contexts": [2048], "batches": [1], "workloads": ["coding"],
              "output_tokens": [256], "warmup": 1, "trials": 3, "max_context": 16384,
              "cache_tokens": 0, "reasoning": "none", "temperature": 0}
    require(all(own.same(report["settings"].get(k), v) for k, v in wanted.items()), "canonical normal controls differ")
    files = report["provenance"]["files"]
    require(files["binary"]["sha256"] == own.EXE and files["metallib"]["sha256"] == own.LIB, "normal exact samebinary artifacts differ")
    runs = report.get("server_runs")
    require(isinstance(runs, list) and len(runs) == 1 and runs[0]["mtp_setting"] == "3", "one normal MTP3 server required")
    run = runs[0]
    sq = run["semantic_quality"]
    require(Path(sq["report"]).resolve() == semantic_path.resolve() and sq["report_sha256"] == semantic_sha,
            "normal run must bind its exact externally pinned semantic report")
    # Known original task failures are retained; semantic exit is not forged.
    unload = run["unload_evidence"]
    require(unload["returncode"] == 0 and unload["process_group_gone"] is True and unload["post_parent_exit_sigkill_required"] is False, "actual clean normal unload required")
    hooks = own.make_hooks(runner.http, expected)
    initial = run["initial_status"]
    require(initial["identity"] == semantic["initial_status"]["identity"] == semantic["final_status"]["identity"], "normal/semantic must belong to the same real worker instance")
    values = [initial]
    waves = report["waves"]
    require(len(waves) == 4 and sum(w["warmup"] is True for w in waves) == 1 and sum(w["warmup"] is False for w in waves) == 3, "one warmup/three measured full waves required")
    measured = []
    query_details = []
    for wave in waves:
        require(all(own.same(wave.get(k), v) for k, v in {"mtp_setting": "3", "task": "coding", "prompt_tokens": 2048, "output_budget_tokens": 256, "http_width": 1, "valid": True, "request_validation_valid": True, "post_wave_idle_pending": False}.items()),
                "real full-budget normal wave shape/evidence differs")
        before, after = wave["status_before"], wave["status_after"]
        details, errors = hooks[1](before, after, {"body": {}}, True, "mtp3")
        require(not errors, "normal query/H3/raw graph coverage differs: " + "; ".join(errors))
        query_details.append(details)
        values.extend((before, after))
        recomputed = runner.http.counter_delta(before, after)
        complete_delta = driver.all_numeric_deltas(before, after)
        complete_delta.update(recomputed)
        require(complete_delta == wave["native_counter_delta"], "normal saved native deltas differ from literal recomputation")
        for path, n in {"requests.submitted": 1, "requests.completed": 1, "requests.cancelled": 0, "requests.failed": 0, "metrics.prefill_input_tokens": 2048, "metrics.autoregressive_output_tokens": 256}.items():
            require(recomputed.get(path) == n, "normal recomputed complete native terminal/budget differs:" + path)
        require(driver.counter_rates(wave["native_counter_delta"]) == wave["counter_rates"], "normal saved rates differ from literal original rate function")
        require(driver.native_mtp_depth_metadata(before, after) == wave["native_mtp_policy"], "normal actual-depth policy differs from literal original source")
        records = wave["records"]
        require(isinstance(records, list) and len(records) == 1, "one actual response per singleton wave")
        r = records[0]
        require(driver.native_exact(r, 2048, 256) == r["measurement"], "normal native interval/emission measurement differs")
        if not wave["warmup"]:
            measured.append(wave)
    values.extend((semantic["initial_status"], semantic["final_status"], run["final_status"]))
    previous = None
    for status in values:
        errors = hooks[0](status, None, None, "mtp3")
        require(not errors, "normal static source/flag/raw profile differs:" + "; ".join(errors))
        errors = runner.gate_status(status, plan, semantic["store_witness"], "mtp3")
        require(not errors, "original common/raw/C1/compact/HC normal status gate failed:" + "; ".join(errors))
        require(not runner.http.validate_status(status, initial["identity"]) and runner.http.idle(status),
                "original normal protocol/identity/idle status gate failed")
        require(status["identity"] == initial["identity"], "normal per-run identity drift")
        current = [runner.http.get_path(status, own.SECTION + "." + k) for k in own.COUNTERS]
        if previous is not None:
            require(all(b >= a for a, b in zip(previous, current)), "normal/semantic process index counters decreased")
        previous = current
    env = report["environments"]["3"]["resolved_flash_environment"]
    require(env.get(FLAG) == ("1" if expected else "0") and env.get("SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22") == "1", "actual normal flag/raw policy differs")
    rates = [w["counter_rates"]["native_prefill_tokens_per_summed_command_wall_second"] for w in measured]
    decode = [r["measurement"]["native_exact_decode_tokens_per_second"] for w in measured for r in w["records"]]
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in rates + decode), "finite measured full native rates required")
    return {"clean_unload": True, "measured_trials": 3, "prefill_mean_tok_s": statistics.mean(rates),
            "prefill_median_tok_s": statistics.median(rates), "all_measured_prefill_above4000": all(v > 4000 for v in rates),
            "native_exact_decode_median_tok_s": statistics.median(decode), "actual_query_coverage_all4waves": query_details,
            "resolved_flags": env}

def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--normal-reports", type=Path, nargs=2, required=True)
    p.add_argument("--normal-sha256", nargs=2, required=True)
    p.add_argument("--semantic-reports", type=Path, nargs=2, required=True)
    p.add_argument("--semantic-sha256", nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    require(not a.output.exists(), "fresh postrun summary required")
    comparison = a.output.with_name(a.output.stem + ".original22-comparison.json")
    require(not comparison.exists(), "fresh literal original comparison required")
    for path, expected in zip(a.normal_reports + a.semantic_reports, a.normal_sha256 + a.semantic_sha256):
        require(len(expected) == 64 and sha(path) == expected, "external whole-report SHA mismatch")
    own, runner = load_original_chain()
    normal = [runner.http.strict_json(path.read_text()) for path in a.normal_reports]
    semantic = [runner.http.strict_json(path.read_text()) for path in a.semantic_reports]
    for r in normal + semantic:
        runner.strict_numbers(r)
    plan = runner.read_plan(Path(semantic[0]["plan"]))
    dispatch = RecordedIndexFlags(own, runner.http).bind(semantic, plan)
    driver = module(NORMAL / "tuning.py", DRIVER_SHA, "_postrun_literal_normal_rates")
    audits = [normal_audit(n, s, path, digest, own, runner, driver, flag, plan)
              for n, s, path, digest, flag in zip(normal, semantic, a.semantic_reports, a.semantic_sha256, (False, True))]
    left, right = (dict(audit["resolved_flags"]) for audit in audits)
    require(left.pop(FLAG) == "0" and right.pop(FLAG) == "1" and left == right, "all other actual resolved numerical/runtime controls must match")
    runner = install(runner, dispatch)
    # Exact original comparator and grade function are invoked unchanged.
    rc = runner.compare(SimpleNamespace(reports=a.semantic_reports, output=comparison, allow_runtime_change=False))
    formal = runner.http.strict_json(comparison.read_text())
    for audit in audits:
        del audit["resolved_flags"]
    decode = [audit["native_exact_decode_median_tok_s"] for audit in audits]
    result = {"schema": "immutable96-samebinary-flag0-flag1-postrun-comparison-v1",
              "cpu_only": True, "source_policy_sha256": own.SOURCE, "worker_sha256": own.EXE,
              "metallib_sha256": own.LIB, "fresh_native_receipt_sha256": NATIVE_SHA,
              "report_sha256": {"normal": a.normal_sha256, "semantic": a.semantic_sha256},
              "normal_audits": audits, "original_comparison_path": str(comparison), "original_comparison_sha256": sha(comparison),
              "valid": formal["valid"], "no_new_task_regressions": formal["no_new_task_regressions"],
              "generation_differences": [row["generation_differences"] for row in formal["comparisons"]],
              "literal_original_allow_runtime_change": False,
              "samebinary_flag0_shared_getenv_atomic_costs_included": True,
              "native_decode_change_pct_vs_instrumented0": 100 * (decode[1] / decode[0] - 1),
              "best_parent_native_decode_tok_s": BEST_PARENT_NATIVE_DECODE,
              "native_decode_change_pct_vs_separate_best_parent": 100 * (decode[1] / BEST_PARENT_NATIVE_DECODE - 1),
              "prefill_variation_not_attributed_to_index": True, "whole_performance_promotion": False,
              "uninitialized_production_tail_parity_claimed": False}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid": result["valid"], "no_new_task_regressions": result["no_new_task_regressions"],
                      "output": str(a.output), "whole_performance_promotion": False}))
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
