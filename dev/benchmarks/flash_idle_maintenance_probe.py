"""Root-only long-idle qualification; does not schedule future agent work."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace
import urllib.request
from dev.benchmarks.qualify_flash_http import HTTPClient


def wired_bytes():
    raw = subprocess.check_output(["vm_stat"], text=True)
    size = int(re.search(r"page size of (\d+)", raw).group(1))
    pages = int(re.search(r"^Pages wired down:\s*(\d+)", raw, re.M).group(1))
    return pages * size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--idle-seconds", type=float, default=30)
    p.add_argument("--run-root-gpu", action="store_true")
    args = p.parse_args()
    if not args.run_root_gpu or not 1 <= args.idle_seconds <= 45 or args.output.exists():
        p.error("Root-only, fresh output and idle1..45 seconds required")
    status = lambda: json.load(urllib.request.urlopen("http://127.0.0.1:8011/status", timeout=10))
    before = status()
    plan = json.load(open("build/release/flash/http-performance-plan.json"))
    body = copy.deepcopy(plan["waves"][0]["lanes"][0]["body"])
    body.update(model="local/Qwen3.8-Flash-Next-oQ4e-mtp", max_completion_tokens=16)
    client = HTTPClient(SimpleNamespace(base_url="http://127.0.0.1:8011", timeout=120, model=body["model"]))
    records = []
    for label, idle in (("warmup", 0), ("after_long_idle", args.idle_seconds), ("immediate_repeat", 0)):
        snapshots = []
        if idle:
            began = time.monotonic()
            for target in (idle/3, idle*2/3, idle):
                remaining = began + target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                snapshots.append(dict(elapsed=time.monotonic()-began, wired_bytes=wired_bytes(), runtime=status()))
        record = client.send("POST", "/v1/chat/completions", body)
        usage = record.get("usage") or {}
        valid = record["http_status"] == 200 and record["done"] and not record["errors"] and usage.get("prompt_tokens") == 128 and usage.get("completion_tokens") == 16 and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0
        records.append(dict(label=label, idle_seconds=idle, valid=valid, snapshots=snapshots, record=record))
        print(json.dumps(dict(label=label, valid=valid, first_content_ms=record.get("first_content_ms"))), flush=True)
    after = status()
    deadline = time.monotonic() + 10
    while after["scheduler"]["active_requests"] or after["scheduler"]["command_in_flight"]:
        if time.monotonic() >= deadline:
            break
        time.sleep(.05)
        after = status()
    valid = all(x["valid"] for x in records) and after["metal"]["healthy"] and not after["scheduler"]["active_requests"] and not after["scheduler"]["command_in_flight"] and all(x["record"]["text"] == records[0]["record"]["text"] for x in records)
    with args.output.open("x") as f:
        json.dump(dict(schema="splash-idle-maintenance-probe-v1", valid=valid,
                       vm_scope="systemwide", runtime_before=before, records=records, runtime_after=after), f, indent=2)
        f.write("\n")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
