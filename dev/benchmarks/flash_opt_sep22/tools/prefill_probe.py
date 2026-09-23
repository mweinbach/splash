#!/usr/bin/env python3
"""Short-prompt prefill probe: fresh prompts of several lengths, one output token.

Starts server.py once with the given binary/env, sends distinct prompts (so PLE
rows and caches are cold), and reports median native prefill GPU/wall time.
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from dev.benchmarks import splash_tuning_sep21 as tuning
from dev.benchmarks import qualify_flash_http as http

WORDS = ("system memory kernel thread cache vector matrix tensor model token layer "
         "compute shader buffer queue signal router expert weight scale bias latency "
         "throughput bandwidth register pipeline prefetch branch predictor scheduler "
         "garden river mountain window candle harbor lantern meadow orchard valley").split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--lengths", default="30,90,180,360")
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    targs = SimpleNamespace(package=tuning.PACKAGE, environment_overrides=dict(
        item.split("=", 1) for item in args.env))
    environment, _ = tuning.environment_for(targs, "4")
    for item in args.env:
        key, value = item.split("=", 1)
        environment[key] = value
    command = [sys.executable, "-u", str(ROOT / "server/server.py"),
               "--local-package", str(tuning.PACKAGE), "--tokenizer", str(tuning.PACKAGE),
               "--model", tuning.MODEL, "--binary", str(args.binary.resolve()), "--port", str(args.port),
               "--max-memory", "auto", "--max-context", "16384", "--no-webui"]
    log = args.output.with_suffix(".server.log").open("w")
    server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    report = {}
    try:
        client = http.HTTPClient(SimpleNamespace(base_url=f"http://127.0.0.1:{args.port}",
                                                 timeout=600, model=tuning.MODEL))
        deadline = time.monotonic() + 400
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                if client.status().get("ready"):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("startup")
            time.sleep(1)
        rng = random.Random(1234)
        # Warm the pipelines once.
        for length in [30, 200]:
            body = {"model": tuning.MODEL, "messages": [{"role": "user", "content": " ".join(rng.choice(WORDS) for _ in range(length))}],
                    "max_completion_tokens": 1, "temperature": 0, "reasoning_effort": "none", "stream": True}
            client.send("POST", "/v1/chat/completions", body)
        for length in [int(x) for x in args.lengths.split(",")]:
            rows = []
            for trial in range(args.trials):
                words = " ".join(rng.choice(WORDS) for _ in range(length))
                body = {"model": tuning.MODEL, "messages": [{"role": "user", "content": f"Summarize: {words}"}],
                        "max_completion_tokens": 1, "temperature": 0, "reasoning_effort": "none", "stream": True}
                before = client.status()
                client.send("POST", "/v1/chat/completions", body)
                after = client.status()
                delta = lambda path: (http.get_path(after, path) or 0) - (http.get_path(before, path) or 0)
                rows.append({"tokens": delta("metrics.prefill_input_tokens"),
                             "gpu_ms": delta("model_timing.prefill.total_gpu_ms"),
                             "wall_ms": delta("metrics.prefill_wall_ms")})
            report[length] = {"tokens": statistics.median(r["tokens"] for r in rows),
                              "gpu_ms": statistics.median(r["gpu_ms"] for r in rows),
                              "wall_ms": statistics.median(r["wall_ms"] for r in rows)}
            print(length, json.dumps(report[length]), flush=True)
    finally:
        tuning.unload(server)
        args.output.write_text(json.dumps({"binary": str(args.binary), "env": args.env, "report": report}, indent=1))


if __name__ == "__main__":
    main()
