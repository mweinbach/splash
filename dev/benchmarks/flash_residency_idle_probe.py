"""Root-only residency grouping diagnostic; fixed prompt and system VM samples."""
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


def vm():
    text = subprocess.check_output(["vm_stat"], text=True)
    size = int(re.search(r"page size of (\d+)", text).group(1))
    values = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s*(\d+)\.", text, re.M)}
    return dict(page_size=size, wired_bytes=values["Pages wired down"]*size, pages=values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-root-gpu", action="store_true")
    args = p.parse_args()
    if not args.run_root_gpu or args.output.exists():
        p.error("Root-only execution and a fresh output required")
    status = lambda: json.load(urllib.request.urlopen("http://127.0.0.1:8011/status", timeout=10))
    before = status()
    plan = json.load(open("build/release/flash/http-performance-plan.json"))
    body = copy.deepcopy(plan["waves"][0]["lanes"][0]["body"])
    body.update(model="local/Qwen3.8-Flash-Next-oQ4e-mtp", max_completion_tokens=1)
    client = HTTPClient(SimpleNamespace(base_url="http://127.0.0.1:8011", timeout=120, model=body["model"]))
    records = []
    for label, pause in (("first", 0), ("immediate", 0), ("idle2", 2), ("immediate_after_idle", 0)):
        if pause:
            time.sleep(pause)
        r = client.send("POST", "/v1/chat/completions", body)
        u = r.get("usage") or {}
        valid = r["http_status"] == 200 and r["done"] and not r["errors"] and u.get("prompt_tokens") == 128 and u.get("completion_tokens") == 1 and u.get("prompt_tokens_details", {}).get("cached_tokens") == 0
        records.append(dict(label=label, idle_seconds=pause, valid=valid, record=r))
        print(json.dumps(dict(label=label, valid=valid, first_content_ms=r.get("first_content_ms"))), flush=True)
    began = time.monotonic()
    samples = []
    for delay in (0, .5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 5):
        remaining = began + delay - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        samples.append(dict(elapsed=time.monotonic()-began, vm=vm()))
    after = status()
    valid = all(x["valid"] for x in records) and after["metal"]["healthy"] and not after["scheduler"]["active_requests"] and not after["scheduler"]["command_in_flight"]
    with args.output.open("x") as f:
        json.dump(dict(schema="splash-residency-grouping-idle-v1", valid=valid,
                       diagnostic_trace_on=True, vm_scope="systemwide resident-page counters",
                       runtime_before=before, records=records, vm_samples=samples, runtime_after=after), f, indent=2)
        f.write("\n")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
