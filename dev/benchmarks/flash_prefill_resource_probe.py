"""Root-only prefill experiment with ordinary HTTP and opt-in command tracing."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import urllib.request

from dev.benchmarks.qualify_flash_http import HTTPClient


def run(args):
    if not args.run_root_gpu:
        raise SystemExit("Use --run-root-gpu only from the serial GPU coordinator")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    client = HTTPClient(SimpleNamespace(base_url=args.base_url, timeout=120,
                                       model="local/Qwen3.8-Flash-Next-oQ4e-mtp"))
    status = lambda: json.load(urllib.request.urlopen(args.base_url + "/status", timeout=10))
    before = status()
    if not before.get("request_command_trace", {}).get("enabled"):
        raise RuntimeError("Start the server with a fresh request command trace")
    cases = [(128, 1, 0), (2048, 1, 0), (2048, 1, 1),
             (128, 4, 0), (2048, 4, 0), (2048, 4, 1)]
    reports = []
    for context, width, sample in cases:
        wave = next(w for w in plan["waves"]
                    if (w["context"], w["width"], w["sample"]) == (context, width, sample))
        bodies = []
        for lane in wave["lanes"]:
            body = copy.deepcopy(lane["body"])
            body.update(model=client.model, max_completion_tokens=1)
            bodies.append(body)
        with ThreadPoolExecutor(max_workers=width) as pool:
            records = list(pool.map(lambda body: client.send("POST", "/v1/chat/completions", body), bodies))
        errors = []
        for record in records:
            usage = record.get("usage") or {}
            if record.get("http_status") != 200 or not record.get("done") or record["errors"]:
                errors.append("HTTP request did not terminate successfully")
            if usage.get("prompt_tokens") != context or usage.get("completion_tokens") != 1:
                errors.append("real token counts differ from the probe")
            if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
                errors.append("prompt cache reuse detected or unknown")
        reports.append(dict(context=context, width=width, sample=sample,
                            records=records, errors=errors, valid=not errors))
        print(json.dumps(dict(context=context, width=width, sample=sample,
                              valid=not errors,
                              first_content_ms=[r.get("first_content_ms") for r in records])), flush=True)
    deadline = time.monotonic() + 10
    after = status()
    while after["scheduler"]["active_requests"] or after["scheduler"]["command_in_flight"]:
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
        after = status()
    scheduler_idle = not after["scheduler"]["active_requests"] and not after["scheduler"]["command_in_flight"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(dict(schema="splash-prefill-resource-probe-v1",
                       decode_scope="one output token; MTP ineligible, head priming omitted",
                       diagnostic_trace_on=True, runtime_before=before, runtime_after=after,
                       waves=reports, scheduler_idle=scheduler_idle,
                       valid=scheduler_idle and all(r["valid"] for r in reports)), file, indent=2)
        file.write("\n")
    if not scheduler_idle:
        raise RuntimeError("Final scheduler is not idle; retained partial report and live service")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--plan", type=Path, default=Path("build/release/flash/http-performance-plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root-gpu", action="store_true")
    run(parser.parse_args())
