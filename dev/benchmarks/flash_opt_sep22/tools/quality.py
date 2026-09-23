#!/usr/bin/env python3
"""Run the frozen 22-case semantic plan against one worker binary.

Launches server.py with the qualified v13 environment for MTP depth 4 (plus
optional overrides), grades every case with the repository grader, unloads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from dev.benchmarks import splash_tuning_sep21 as tuning
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.prefill4k_attribution_quality import grade_record

PLAN = ROOT / "build/release/flash/prefill4k-semantic-plan-v1.json"
DECODE_PROMPTS = [
    "Write a complete Python implementation of a red-black tree with insert, delete and in-order traversal. Include docstrings.",
    "Explain in detail how the TCP congestion control algorithms Reno, CUBIC and BBR differ, with examples.",
    "Write a SQL schema for an online bookstore with authors, books, customers, orders and reviews, then write ten useful queries.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle. Make it vivid and detailed.",
    "Implement a thread pool in modern C++20 with a work-stealing queue. Provide the full code.",
    "Summarize the main events of the French Revolution between 1789 and 1799 as a detailed timeline.",
    "Solve step by step: a train leaves city A at 60 km/h and another leaves city B, 300 km away, at 90 km/h toward A. When and where do they meet? Then generalize.",
    "Write a React component in TypeScript for a sortable, filterable data table with pagination. Include the full code.",
    "Describe the architecture of a modern GPU, including SIMD groups, memory hierarchy, and how matrix units accelerate machine learning.",
    "Write a bash script that backs up a directory incrementally with rsync, rotates old backups, and logs results. Explain each part.",
]


EXTRA_PROMPTS = [
    "Explain how a B-tree index works in a relational database and why it suits disk storage.",
    "Write a Rust function that parses a CSV line with quoted fields, with tests.",
    "Describe the causes and consequences of the 2008 financial crisis in detail.",
    "Write a haiku sequence of ten poems about the four seasons.",
    "Explain the difference between TCP and UDP, including when to use each, with examples.",
    "Implement Dijkstra's algorithm in Python with a heap and explain its complexity.",
    "Write a cover letter for a senior backend engineer position at a fintech startup.",
    "Explain how transformers use attention, including multi-head attention and positional encodings.",
    "Write a Go HTTP server with graceful shutdown, structured logging and a health endpoint.",
    "Describe the water cycle and its role in climate, in a detailed explanation for students.",
    "Write a detailed recipe for sourdough bread, including starter maintenance.",
    "Compare PostgreSQL and MongoDB for an e-commerce backend, with trade-offs.",
    "Write a JavaScript debounce and throttle implementation with explanations and tests.",
    "Explain how public-key cryptography works, including RSA and elliptic curves.",
    "Write a short science fiction story about the first human colony on Europa.",
    "Explain the CAP theorem and give examples of systems that choose each trade-off.",
    "Write a Python script that watches a directory and syncs changed files to S3.",
    "Summarize the history of the Roman Empire from Augustus to the fall of the West.",
    "Explain how a CPU pipeline works, including hazards, forwarding and branch prediction.",
    "Write a SQL query set that analyzes monthly retention for a subscription product.",
]


TOPICS = ["binary search trees", "the French Revolution", "photosynthesis", "garbage collection in Java",
          "the stock market", "black holes", "REST API design", "the Silk Road", "neural network training",
          "climate change mitigation", "Kubernetes scheduling", "the printing press", "vaccines",
          "compiler optimization", "ocean currents", "the Apollo program", "database transactions",
          "impressionist painting", "quantum computing", "urban planning"]
STYLES = ["Write a detailed explanation of {} for a curious beginner.",
          "Write a Python program related to {}, with comments and tests.",
          "Write a short essay arguing about the most important aspect of {}."]
MORE_PROMPTS = [style.format(topic) for topic in TOPICS for style in STYLES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--mtp", default="4")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--more", action="store_true")
    args = parser.parse_args()
    targs = SimpleNamespace(package=tuning.PACKAGE, environment_overrides=dict(
        item.split("=", 1) for item in args.env))
    environment, _ = tuning.environment_for(targs, args.mtp)
    for item in args.env:
        key, value = item.split("=", 1)
        environment[key] = value
    plan = json.loads(PLAN.read_text())
    command = [sys.executable, "-u", str(ROOT / "server/server.py"),
               "--local-package", str(tuning.PACKAGE), "--tokenizer", str(tuning.PACKAGE),
               "--model", tuning.MODEL, "--binary", str(args.binary.resolve()), "--port", str(args.port),
               "--max-memory", "auto", "--max-context", "16384", "--no-webui"]
    log = args.output.with_suffix(".server.log").open("w")
    server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    results = []
    try:
        client = http.HTTPClient(SimpleNamespace(base_url=f"http://127.0.0.1:{args.port}",
                                                 timeout=600, model=tuning.MODEL))
        deadline = time.monotonic() + 400
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                status = client.status()
                if status.get("ready"):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("startup")
            time.sleep(1)
        for case in ([] if args.decode_only else plan["cases"]):
            body = dict(case["body"])
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            began = time.monotonic()
            record = client.send("POST", "/v1/chat/completions", body)
            errors, details = grade_record(record, case)
            results.append({"id": case["id"], "pass": not errors, "errors": errors,
                            "text": record.get("text", ""), "seconds": time.monotonic() - began,
                            "decode": (record.get("metrics") or {}).get("request_latency", {}).get(
                                "stream_tokens_per_second")})
            print(json.dumps({k: results[-1][k] for k in ("id", "pass", "errors", "decode")})[:300], flush=True)
        # Decode suite: aggregate throughput and MTP acceptance over varied prompts.
        before = client.status()
        decode_tokens, decode_ms = 0, 0.0
        prompts = MORE_PROMPTS if args.more else DECODE_PROMPTS + (EXTRA_PROMPTS if args.extended else [])
        for prompt in prompts * args.repeat:
            body = {"model": tuning.MODEL, "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 256, "temperature": 0, "seed": 0,
                    "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
            record = client.send("POST", "/v1/chat/completions", body)
            metrics = record.get("metrics") or {}
            tokens = (metrics.get("decode") or {}).get("tokens") or 0
            ms = (metrics.get("request_latency") or {}).get("first_token_to_done_ms") or 0
            decode_tokens += tokens; decode_ms += ms
            results.append({"id": "decode:" + prompt[:40], "pass": True, "errors": [], "text": record.get("text", ""),
                            "seconds": 0, "decode": tokens * 1000 / ms if ms else None})
        after = client.status()
        delta = lambda path: (http.get_path(after, path) or 0) - (http.get_path(before, path) or 0)
        suite = {"decode_tokens": decode_tokens, "decode_ms": decode_ms,
                 "aggregate_decode_tokens_per_second": decode_tokens * 1000 / decode_ms if decode_ms else None,
                 "cycles": delta("mtp.verification_cycles"), "drafted": delta("mtp.drafted_tokens"),
                 "accepted": delta("mtp.accepted_committed_drafts"),
                 "verify_gpu_ms": delta("mtp.target_verify.total_gpu_ms"),
                 "draft_gpu_ms": delta("mtp.head_decode.total_gpu_ms")}
        if suite["cycles"]:
            suite["tokens_per_cycle"] = (suite["accepted"] + suite["cycles"]) / suite["cycles"]
            suite["verify_gpu_ms_per_cycle"] = suite["verify_gpu_ms"] / suite["cycles"]
            suite["draft_gpu_ms_per_cycle"] = suite["draft_gpu_ms"] / suite["cycles"]
        suite["all_deltas"] = {k: v for k, v in tuning.all_numeric_deltas(before, after).items() if v}
        print("DECODE_SUITE", json.dumps({k: v for k, v in suite.items() if k != "all_deltas"}), flush=True)
    finally:
        tuning.unload(server)
        args.output.write_text(json.dumps({"binary": str(args.binary), "env": args.env,
                                           "decode_suite": locals().get("suite"),
                                           "passed": sum(r["pass"] for r in results if not r["id"].startswith("decode:")),
                                           "total": len([r for r in results if not r["id"].startswith("decode:")]),
                                           "results": results}, indent=1))
    graded = [r for r in results if not r["id"].startswith("decode:")]
    print("PASSED", sum(r["pass"] for r in graded), "of", len(graded))


if __name__ == "__main__":
    main()
