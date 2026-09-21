"""Preview or bracket one temporary, administrator-only IOGPU policy experiment.

This tool never invokes sudo, edits persistent OS configuration, or starts GPU
work itself. The optional child command belongs to the serial GPU coordinator.
The proprietary collector semantics are not a public Apple API contract.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

POLICY = "iogpu.disable_wired_collector"
OBSERVED_KEYS = (POLICY, "iogpu.wired_lwm_mb", "iogpu.dynamic_lwm", "iogpu.wired_limit_mb")


def read_setting(key: str) -> int:
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n", key], text=True).strip())


def write_setting(key: str, value: int) -> None:
    if key != POLICY or value not in (0, 1):
        raise ValueError("Only the collector boolean may be changed")
    subprocess.run(["/usr/sbin/sysctl", "-w", f"{key}={value}"], check=True,
                   capture_output=True, text=True)
    if read_setting(key) != value:
        raise RuntimeError("sysctl did not retain the requested value")


def vm_snapshot() -> dict:
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+)", raw).group(1))
    values = {key: int(value) for key, value in re.findall(r"^([^:\n]+):\s*(\d+)\.", raw, re.M)}
    return {"page_size": page_size, "wired_bytes": values["Pages wired down"] * page_size,
            "pages": values}


def snapshot() -> dict:
    return {"monotonic_seconds": time.monotonic(),
            "settings": {key: read_setting(key) for key in OBSERVED_KEYS},
            "physical_memory_bytes": read_setting("hw.memsize"), "vm": vm_snapshot()}


@contextmanager
def temporary_collector_policy(read=read_setting, write=write_setting, state=None):
    """Restore after a successful write, child failure, KeyboardInterrupt or signal.

    If another administrator changes the boolean concurrently, the tool refuses
    to overwrite that new value. SIGKILL or a host crash cannot run this cleanup;
    the report records the exact original value for manual restoration.
    """
    state = {} if state is None else state
    original = read(POLICY)
    if original not in (0, 1):
        raise ValueError("Unrecognized collector policy; refusing to write")
    state.update(original_value=original, temporary_value=1, applied=False,
                 restore_attempted=False, restored=False, already_disabled=original == 1)
    if original == 1:
        raise RuntimeError("Collector is already disabled; this is not a controlled experiment")
    try:
        # Restore even if a write applies but its readback fails.
        state["restore_attempted"] = True
        write(POLICY, 1)
        state["applied"] = True
        yield state
    finally:
        current = read(POLICY)
        if current == original:
            state["restored"] = True
        elif current == 1:
            write(POLICY, original)
            state["restored"] = read(POLICY) == original
        else:
            state["restore_conflict_value"] = current
            raise RuntimeError("Concurrent policy change; original value not overwritten")


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def execute_child(command: list[str], timeout_seconds: float) -> dict:
    """Do not expose child credentials or use shell command interpolation."""
    started = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    try:
        code = child.wait(timeout=timeout_seconds)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        raise
    return {"argv": command, "exit_code": code, "elapsed_seconds": time.monotonic() - started}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-temporarily", action="store_true")
    parser.add_argument("--run-root-gpu", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--report", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not 1 <= args.timeout_seconds <= 180:
        parser.error("Timeout must be 1..180 seconds")
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    report = {"schema": "splash-temporary-iogpu-policy-v11", "policy": POLICY,
              "changes_persistent_configuration": False,
              "public_apple_contract": False, "before": snapshot(),
              "global_effect": "May retain GPU wiring for other applications as well as Splash",
              "potential_extra_wired_bytes": 145_943_855_104,
              "manual_restore_argv": ["/usr/sbin/sysctl", "-w",
                  f"{POLICY}={read_setting(POLICY)}"]}
    if not args.apply_temporarily:
        report["preview_only"] = True
        print(json.dumps(report, indent=2))
        return 0
    if os.geteuid() != 0:
        parser.error("Administrator execution is required; this tool will not request a password")
    if not args.run_root_gpu or not args.command or args.report is None:
        parser.error("Temporary execution requires a coordinator command, --run-root-gpu and a fresh --report")
    state = {}
    report["transaction"] = state
    # Reserve the evidence file before any write; preserve it if the process dies.
    with args.report.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    error = None
    try:
        for sig in handlers:
            signal.signal(sig, _interrupt)
        with temporary_collector_policy(state=state):
            report["temporary_before_command"] = snapshot()
            report["child"] = execute_child(args.command, args.timeout_seconds)
            report["temporary_after_command"] = snapshot()
    except BaseException as exception:
        error = exception
        report["error"] = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        try:
            report["after_restore"] = snapshot()
        except Exception as exception:
            # Preserve the transaction state even if post-test diagnostics fail.
            report["after_restore_snapshot_error"] = {
                "type": type(exception).__name__, "message": str(exception)}
            if error is None:
                error = exception
        report["valid_transaction"] = (error is None and state.get("restored") is True
                                      and report.get("child", {}).get("exit_code") == 0)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "valid_transaction": report["valid_transaction"],
                      "restored": state.get("restored")}))
    return 0 if report["valid_transaction"] else 1


if __name__ == "__main__":
    sys.exit(main())
