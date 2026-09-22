#!/usr/bin/env python3
"""Summarize warmed, byte-exact dense candidates versus runtime baselines."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("screen",type=Path)
parser.add_argument("--trace",type=Path,default=Path("build/release/flash/prefill4k-current-stage.json.trace.jsonl"))
parser.add_argument("--out",type=Path)
args = parser.parse_args()
report = json.loads(args.screen.read_text())
trace = next(json.loads(line) for line in args.trace.open() if json.loads(line)["phase"] == "target_prefill")
costs = collections.defaultdict(lambda:{"calls":0,"gpu_ms":0.0})
for dispatch in trace["command"]["dispatches"]:
    match = re.search(r"flash_dense_cache_m(\d+)_n(\d+)$",dispatch["pipeline"])
    if not match:
        continue
    m,n = map(int,match.groups())
    columns,rowtiles,_ = dispatch["threadgroups"]
    rows,outputs = rowtiles*m,columns*n
    bindings = {b["index"]:b["size_bytes"] for b in dispatch["bindings"]}
    inputs = bindings[1]//(outputs*2)
    key = (rows,inputs,outputs)
    costs[key]["calls"] += 1
    costs[key]["gpu_ms"] += dispatch["gpu_seconds"]*1000


def warmed(variant):
    values = variant["gpu_ms"][1:] if len(variant["gpu_ms"]) > 3 else variant["gpu_ms"]
    return statistics.median(values)


results = []
for case in report["cases"]:
    baseline = case["variants"][0]
    exact = [v for v in case["variants"] if v["baseline_error"]["bf16_mismatches"] == 0]
    fastest = min(case["variants"],key=warmed)
    winner = min(exact,key=warmed)
    comparisons = [b/w for b,w in zip(baseline["gpu_ms"][1:],winner["gpu_ms"][1:])]
    original = costs[(case["rows"],case["input_size"],case["output_size"])]
    ratio = warmed(baseline)/warmed(winner)
    result = {
        "projection":case["projection"],"rows":case["rows"],
        "input_size":case["input_size"],"output_size":case["output_size"],
        "runtime_baseline":baseline["name"],"baseline_warm_ms":warmed(baseline),
        "fastest_exact":winner["name"],"exact_warm_ms":warmed(winner),
        "exact_gpu_speedup":ratio,
        "exact_wall_speedup":baseline["median_wall_ms"]/winner["median_wall_ms"],
        "paired_warm_min_speedup":min(comparisons) if comparisons else None,
        "paired_warm_positive":sum(v > 1 for v in comparisons),
        "paired_warm_samples":len(comparisons),
        "fastest_any":fastest["name"],
        "fastest_any_bf16_mismatches":fastest["baseline_error"]["bf16_mismatches"],
        "trace_calls":original["calls"],"trace_gpu_ms":original["gpu_ms"],
        "isolated_saved_ms_estimate":original["gpu_ms"]*(1-1/ratio),
        "full_model_qualified":False,
    }
    results.append(result)
    print(f"K{case['input_size']} N{case['output_size']} rows{case['rows']}: "
          f"{winner['name']} {ratio:.3f}xGPU "
          f"{result['exact_wall_speedup']:.3f}xwall "
          f"positive={result['paired_warm_positive']}/{result['paired_warm_samples']} "
          f"BF16mismatch=0 est_saved={result['isolated_saved_ms_estimate']:.2f}ms")
summary = {"schema":"splash-prefill4k-dense-screen-summary-v1",
           "gpu_executed_by_analysis":False,"cases":results,
           "scope":"Isolated projection estimates; full-model qualification required"}
if args.out:
    args.out.write_text(json.dumps(summary,indent=2)+"\n")
