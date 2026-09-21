"""Root-only matched idle versus MTP-eligibility prefill control."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

from dev.benchmarks.qualify_flash_http import HTTPClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=9)
    parser.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args()
    if not args.run_root_gpu or not 1 <= args.idle_seconds <= 20:
        parser.error("Root-only execution and an idle period1..20 seconds required")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.load(open("build/release/flash/http-performance-plan.json"))
    body = next(w for w in plan["waves"] if (w["sample"], w["context"], w["width"]) == (0, 128, 1))["lanes"][0]["body"]
    client = HTTPClient(SimpleNamespace(base_url="http://127.0.0.1:8011", timeout=120,
                                       model="local/Qwen3.8-Flash-Next-oQ4e-mtp"))
    status = lambda: json.load(urllib.request.urlopen("http://127.0.0.1:8011/status", timeout=10))
    before = status()
    records = []
    cases = [("first_ineligible", 1, 0), ("immediate_ineligible", 1, 0),
             ("idle_ineligible", 1, args.idle_seconds), ("immediate_eligible", 2, 0),
             ("idle_eligible", 2, args.idle_seconds), ("immediate_ineligible_after_head", 1, 0)]
    for label, budget, idle in cases:
        if idle:
            time.sleep(idle)
        request = copy.deepcopy(body)
        request.update(model=client.model, max_completion_tokens=budget)
        record = client.send("POST", "/v1/chat/completions", request)
        usage = record.get("usage") or {}
        valid = (record["http_status"] == 200 and record["done"] and not record["errors"]
                 and usage.get("prompt_tokens") == 128 and usage.get("completion_tokens") == budget
                 and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0)
        records.append(dict(label=label, output_budget=budget, idle_seconds=idle,
                            valid=valid, record=record))
        print(json.dumps(dict(label=label, valid=valid, first_content_ms=record.get("first_content_ms"))), flush=True)
    after = status()
    until = time.monotonic() + 10
    while after["scheduler"]["active_requests"] or after["scheduler"]["command_in_flight"]:
        if time.monotonic() >= until:
            break
        time.sleep(.05)
        after = status()
    valid = all(r["valid"] for r in records) and not after["scheduler"]["active_requests"] and not after["scheduler"]["command_in_flight"]
    with args.output.open("x") as file:
        json.dump(dict(schema="splash-idle-prefill-probe-v1", diagnostic_trace_on=True,
                       runtime_before=before, records=records, runtime_after=after, valid=valid), file, indent=2)
        file.write("\n")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
