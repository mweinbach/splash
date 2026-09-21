"""CPU-only join of idle HTTP probes, native command profiles, and refresh API trace."""
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(http, commands, refreshes):
    if not http.get("valid"):
        raise ValueError("HTTP idle probe is not valid")
    records = http["records"]
    targets = [r for r in commands if r.get("phase") == "prefill" and r.get("role") == "target_trunk"]
    if len(targets) != len(records):
        raise ValueError("target prefill count does not match HTTP probes")
    by_sequence = {r["sequence"]: r for r in refreshes}
    if len(by_sequence) != len(refreshes):
        raise ValueError("duplicate refresh trace sequences")
    results = []
    for record, target in zip(records, targets, strict=True):
        sequence = target["command_sequence"]
        refresh = by_sequence.get(sequence)
        if refresh is None:
            raise ValueError("missing matching refresh trace")
        response = record["record"]
        usage = response.get("usage") or {}
        if not (record["valid"] and response["http_status"] == 200 and response["done"]
                and not response["errors"] and usage.get("prompt_tokens") == 128
                and usage.get("completion_tokens") == record["output_budget"]
                and usage.get("prompt_tokens_details", {}).get("cached_tokens") == 0
                and target["profile_present"] and target["profile_status"] == "complete"
                and target["cross_clock_valid"] and target["actual_rows"] == 128
                and target["lanes"] == 1 and not target["encoder_boundaries_altered"]):
            raise ValueError("HTTP/profile correctness or timing precondition failed")
        if refresh["requested"] and not refresh["succeeded"]:
            raise ValueError("refresh API did not succeed")
        fields = {
            "label": record["label"], "requested_idle_seconds": record["idle_seconds"],
            "output_budget": record["output_budget"], "native_sequence": sequence,
            "dispatches": target["dispatch_count"], "response_text": response.get("text"),
            "refresh_requested": refresh["requested"], "refresh_succeeded": refresh["succeeded"],
            "refresh_count": refresh["refresh_count"],
            "registered_base_count": refresh["registered_base_count"],
            "registered_base_bytes": refresh["registered_base_bytes"],
            "measured_idle_ms": refresh["idle_seconds"] * 1000,
            "refresh_api_ms": refresh["api_seconds"] * 1000,
            "commit_end_to_gpu_start_ms": target["commit_end_to_gpu_start_seconds"] * 1000,
            "submit_entry_to_gpu_start_ms": target["backend_submit_entry_to_gpu_start_seconds"] * 1000,
            "gpu_ms": target["gpu_seconds"] * 1000,
            "command_wall_ms": target["command_wall_seconds"] * 1000,
            "first_content_ms": response.get("first_content_ms"),
            "total_ms": response.get("http_total_ms"),
            "driver_kernel_processing_ms": (target.get("driver_kernel_processing_seconds") or 0) * 1000
                if target.get("driver_kernel_timing_valid") else None,
        }
        results.append(fields)
    scheduler = http["runtime_after"]["scheduler"]
    idle = not scheduler["active_requests"] and not scheduler["command_in_flight"]
    if not idle or not http["runtime_after"]["metal"]["healthy"]:
        raise ValueError("final service is not healthy and idle")
    return {"schema": "splash-private-idle-refresh-http-cpu-audit-v9", "valid": True,
            "gpu_work": False, "final_service_healthy_idle": idle,
            "target_dispatches_unchanged": len({r["dispatches"] for r in results}) == 1,
            "diagnostic_not_steady_http_performance": True, "samples": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--refresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.http.read_text()), load_jsonl(args.commands), load_jsonl(args.refresh))
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"valid": True, "samples": len(report["samples"])}))


if __name__ == "__main__":
    main()
