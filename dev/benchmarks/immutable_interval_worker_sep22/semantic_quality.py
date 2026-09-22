#!/usr/bin/env python3
"""New host-only interval admission layered over every original raw-Q4 gate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path("/Users/mweinbach/Projects/splash")
BUILD = ROOT / "build/immutable96-index-Q4-sep22-worker-v1"
PARENT = ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
SOURCE = "0abf03000e859c1233a538d394e8a68137b6f986188de14233895b8983c5f98b"
SEAL = "c63eb7a6a34a624e44e7d8128faa39183e28729339238613b7634f9308beb899"
EXE = "96b52486bf1ffa4a1b6ecc6b4528cf10c211d1c11b43989d49a2db0dc7d49d23"
LIB = "7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8"
QA_EXE = "4a3f5e94449890b9af0a7bf635986fa132922f2e6718858bdde6b161878a4800"
QA_READY = "88819d924023ff39682b63de8a860672a034de3681c3e42636212f55577bb833"
PARENT_ADAPTER = "be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e"
SECTION = "immutable_interval_index_guard"
SCHEMA = "CPU-immutable96-exact-positive-index-original-fallback-v1"
SCOPE = "CPU-only immutable96 alias lookup; positive bounded disjoint accepts indexed; rejected/odd/nonindexable queries retain literal original callback; FP/graph/public-guards unchanged"
COUNTER_SCOPE = "process-cumulative CPU metadata lookups; not GPU completion"
STATIC = ("schema", "requested", "source_policy_sha256", "scope",
          "GPU_allocation_bytes_added", "FP_graph_public_guards_unchanged", "counter_scope")
COUNTERS = ("finalized_tables", "finalized_spans", "indexable_tables", "indexed_accepts", "original_callbacks")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def same(a, b):
    return type(a) is type(b) and a == b

def u64(value):
    return type(value) is int and 0 <= value < 2 ** 64

def parent_module():
    path = PARENT / "source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py"
    if digest(path) != PARENT_ADAPTER:
        raise ValueError("immutable original raw-Q4 adapter drift")
    spec = importlib.util.spec_from_file_location("_immutable96_original_rawQ4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def index_hooks(http, expected):
    if type(expected) is not bool:
        raise ValueError("expected index must be Boolean")
    def status(value, plan, store, execution_mode="mtp3"):
        p = value.get(SECTION) if isinstance(value, dict) else None
        if not isinstance(p, dict):
            return ["immutable96 registered CPU profile missing"]
        wanted = {"schema": SCHEMA, "requested": expected, "source_policy_sha256": SOURCE,
                  "scope": SCOPE, "GPU_allocation_bytes_added": 0,
                  "FP_graph_public_guards_unchanged": True, "counter_scope": COUNTER_SCOPE}
        errors = ["immutable96 profile differs: " + key for key, target in wanted.items() if not same(p.get(key), target)]
        if any(not u64(p.get(key)) for key in COUNTERS):
            return errors + ["immutable96 cumulative counter type/range invalid"]
        n = int(expected)
        if (p["finalized_tables"], p["finalized_spans"], p["indexable_tables"]) != (n, 96 * n, n):
            errors.append("actual finalized immutable96 Store census differs")
        if not expected and p["indexed_accepts"]:
            errors.append("disabled index recorded fast accepts")
        return errors
    def coverage(before, after, case, require_counters=True, execution_mode="mtp3"):
        errors = []
        for side, value in (("before", before), ("after", after)):
            errors.extend(side + ": " + e for e in status(value, None, None, execution_mode))
        # No source or runtime identity fields are removed or normalized.
        for key in STATIC:
            if not http.same_json(http.get_path(before, SECTION + "." + key), http.get_path(after, SECTION + "." + key)):
                errors.append("immutable96 per-run provenance drift: " + key)
        deltas = {}
        for key in COUNTERS:
            a, b = (http.get_path(v, SECTION + "." + key) for v in (before, after))
            if not u64(a) or not u64(b) or b < a:
                errors.append("immutable96 decreasing/invalid cumulative counter: " + key)
            else:
                deltas[key] = b - a
        if any(deltas.get(key) for key in COUNTERS[:3]):
            errors.append("immutable96 Store constructor census changed within case")
        old, new = (http.get_path(v, "mtp.completed_cycles_by_proposed_depth") for v in (before, after))
        cycles = None
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new) == 16 and all(u64(x) for x in old + new) and all(b >= a for a, b in zip(old, new)):
            cycles = new[3] - old[3]
        else:
            errors.append("immutable96 real actual-depth histogram invalid/decreasing")
        minimum = 624 * cycles if cycles is not None else None
        relevant = deltas.get("indexed_accepts" if expected else "original_callbacks")
        if minimum is not None and relevant is not None and relevant < minimum:
            errors.append("immutable96 lookup work below624 per actual singleton H3 VerifyR4 cycle")
        if not expected and deltas.get("indexed_accepts"):
            errors.append("original-loop case recorded indexed work")
        if require_counters and sum(deltas.get(k, 0) for k in ("indexed_accepts", "original_callbacks")) == 0:
            errors.append("real case recorded no immutable query decisions")
        return {"immutable96_lookup_counter_deltas": deltas,
                "immutable96_actual_H3_cycles": cycles,
                "immutable96_minimum_VerifyR4_queries": minimum,
                "immutable96_other_prefill_AR_queries_included": True,
                "samebinary_flag0_shared_getenv_counter_cost_included": True}, errors
    def ownership(value):
        return {SECTION + "." + key: http.get_path(value, SECTION + "." + key) for key in STATIC}
    return status, coverage, ownership

def make_hooks(http, expected):
    # Timing admission includes the unchanged rawQ4 status checks as well.
    original = parent_module().make_hooks(http, True)
    own = index_hooks(http, expected)
    def status(*args, **kwargs):
        return original[0](*args, **kwargs) + own[0](*args, **kwargs)
    def coverage(*args, **kwargs):
        d, e = original[1](*args, **kwargs)
        extra, added = own[1](*args, **kwargs)
        return {**d, **extra}, e + added
    def ownership(value):
        return {**original[2](value), **own[2](value)}
    return status, coverage, ownership

def authenticate(build, native_receipt=None, native_receipt_sha256=None, require_state=False):
    build = Path(build).resolve()
    if build != BUILD or digest(build / "compiled-cpu-seal.json") != SEAL:
        raise ValueError("unknown immutable96 worker source/seal")
    m = json.loads((build / "compiled-cpu-seal.json").read_text())
    if not m["pass"] or m["source_policy_sha256"] != SOURCE or m["changed_TUs"] != ["runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashWorker.mm"]:
        raise ValueError("unknown immutable96 source/2TU closure")
    for r in m["objects"] + m["artifacts"]:
        if digest(build / r["path"]) != r["sha256"]:
            raise ValueError("immutable96 object/artifact drift: " + r["path"])
    for r in m["source_files"]:
        if digest(build / "source" / r["path"]) != r["sha256"]:
            raise ValueError("immutable96 frozen source drift: " + r["path"])
    if digest(build / "splash-flash") != EXE or digest(build / "splash.metallib") != LIB:
        raise ValueError("immutable96 exact executable/library drift")
    if require_state:
        if not native_receipt or not native_receipt_sha256 or digest(native_receipt) != native_receipt_sha256:
            raise ValueError("fresh externally pinned Root native receipt required")
        r = json.loads(Path(native_receipt).read_text())
        wanted = {"schema": "immutable96-index-bounded-current-native-admission-v1", "pass": True,
                  "Root_GPU_executed": True, "source_policy_sha256": SOURCE, "worker_sha256": EXE,
                  "worker_seal_sha256": SEAL, "metallib_sha256": LIB, "oracle_sha256": QA_EXE,
                  "CPU_READY_sha256": QA_READY, "frames": 10, "repeated_frames": 0,
                  "all134_state_and216_physical_tapes_compared": True,
                  "original_unused_PLE_count_tails_known_initialized_QA_only": True,
                  "VerifyR4_indexed_accepts": 624, "VerifyR4_original_callbacks": 0,
                  "actual_owned_alias_guard_cases": 8, "all8_errors_exact_category_and_text": True,
                  "actual_final_zero_governor_reservations": True, "backend_destroyed": True,
                  "original22_or_whole_performance_qualified": False}
        if any(not same(r.get(key), value) for key, value in wanted.items()) or not u64(r.get("bytes_compared")) or not 0 < r["bytes_compared"] < 4 << 30:
            raise ValueError("fresh native receipt does not bind this bounded current worker")
    return build

def load(build, expected=False, require_state=False, native_receipt=None, native_receipt_sha256=None):
    authenticate(build, native_receipt, native_receipt_sha256, require_state)
    parent = parent_module()
    # Parent proof admits only immutable original parent, never the new child.
    runner = parent.load(PARENT, expected=True, require_state=True)
    original_install = runner.gate_status, runner.coverage, runner.ownership_policy
    status, coverage, ownership = index_hooks(runner.http, expected)
    runner.gate_status = lambda *a, **k: original_install[0](*a, **k) + status(*a, **k)
    def combined(*a, **k):
        d, e = original_install[1](*a, **k)
        extra, added = coverage(*a, **k)
        return {**d, **extra}, e + added
    runner.coverage = combined
    runner.ownership_policy = lambda value: {**original_install[2](value), **ownership(value)}
    return runner

def main(argv=None):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--expected-index", choices=("0", "1"), required=True)
    p.add_argument("--native-receipt", type=Path, required=True)
    p.add_argument("--native-receipt-sha256", required=True)
    args, rest = p.parse_known_args(argv)
    measured = bool(rest and rest[0] == "measure")
    if measured:
        rest = parent_module().runtime_args(rest, args.build.resolve())
    return load(args.build, args.expected_index == "1", True, args.native_receipt, args.native_receipt_sha256).main(rest)

if __name__ == "__main__":
    raise SystemExit(main())
