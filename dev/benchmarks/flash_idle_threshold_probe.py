"""Root-only diagnostic of the native idle threshold with fixed HTTP work."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

from dev.benchmarks.qualify_flash_http import HTTPClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-root-gpu", action="store_true")
    args = p.parse_args()
    if not args.run_root_gpu:
        p.error("Only the serial GPU coordinator may execute this probe")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.load(open("build/release/flash/http-performance-plan.json"))
    body = copy.deepcopy(plan["waves"][0]["lanes"][0]["body"])
    body.update(model="local/Qwen3.8-Flash-Next-oQ4e-mtp", max_completion_tokens=1)
    client = HTTPClient(SimpleNamespace(base_url="http://127.0.0.1:8011", timeout=120, model=body["model"]))
    status = lambda: json.load(urllib.request.urlopen("http://127.0.0.1:8011/status", timeout=10))
    before = status()
    if not before["request_command_trace"]["enabled"]:
        raise RuntimeError("Start a fresh opt-in request-command trace")
    durations = [0, 0, .25, .5, 1, 2, 3, 5, 9, 5, 3, 2, 1, .5, 0]
    records = []
    for index, duration in enumerate(durations):
        idle_start = time.monotonic()
        if duration:
            time.sleep(duration)
        sent = time.monotonic()
        record = client.send("POST", "/v1/chat/completions", body)
        usage = record.get("usage") or {}
        valid = record["http_status"] == 200 and record["done"] and not record["errors"] and usage.get("prompt_tokens") == 128 and usage.get("completion_tokens") == 1 and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0
        records.append(dict(index=index, requested_idle_seconds=duration,
                            elapsed_idle_seconds=sent-idle_start, sent_monotonic=sent,
                            valid=valid, record=record))
        print(json.dumps(dict(index=index, idle_seconds=duration, valid=valid,
                              first_content_ms=record.get("first_content_ms"))), flush=True)
    after = status()
    deadline = time.monotonic() + 10
    while after["scheduler"]["active_requests"] or after["scheduler"]["command_in_flight"]:
        if time.monotonic() >= deadline:
            break
        time.sleep(.05)
        after = status()
    valid = all(r["valid"] for r in records) and not after["scheduler"]["active_requests"] and not after["scheduler"]["command_in_flight"]
    with args.output.open("x") as f:
        json.dump(dict(schema="splash-idle-threshold-probe-v1", diagnostic_trace_on=True,
                       runtime_before=before, records=records, runtime_after=after, valid=valid), f, indent=2)
        f.write("\n")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
