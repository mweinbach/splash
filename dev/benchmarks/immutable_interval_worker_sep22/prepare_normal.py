#!/usr/bin/env python3
"""CPU source preparation only; fresh native admission is required at execution."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path("/Users/mweinbach/Projects/splash")
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2/source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def transform(source):
    journal = []
    def change(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError("private normal source anchor drift: " + old[:100])
        journal.append({"old": old, "new": new, "count": count})
        source = source.replace(old, new)
    change('args.binary.parent / "source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py"',
           '(Path(__file__).parent / "semantic_quality.py")', 3)
    change("args.binary.parent / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py'",
           'Path(__file__).with_name("semantic_quality.py")')
    change("SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22", "SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22", 3)
    change('"--expected-rowpair"', '"--expected-index"')
    change('    args = parse_args(argv)\n    expected =',
           '''    admission = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    admission.add_argument("--native-receipt", type=Path, required=True)
    admission.add_argument("--native-receipt-sha256", required=True)
    admission_args, rest = admission.parse_known_args(argv)
    args = parse_args(rest)
    args.native_receipt = admission_args.native_receipt.resolve()
    args.native_receipt_sha256 = admission_args.native_receipt_sha256
    expected =''')
    change('raw.load(args.binary.parent, expected=expected == "1", require_state=True)',
           'raw.load(args.binary.parent, expected=expected == "1", require_state=True, native_receipt=args.native_receipt, native_receipt_sha256=args.native_receipt_sha256)')
    change('"--build", str(args.binary.parent), "--expected-index",',
           '"--native-receipt", str(args.native_receipt), "--native-receipt-sha256", args.native_receipt_sha256, "--build", str(args.binary.parent), "--expected-index",')
    inverse = source
    for edit in reversed(journal):
        if inverse.count(edit["new"]) != edit["count"]:
            raise ValueError("inverse normal source journal drift")
        inverse = inverse.replace(edit["new"], edit["old"])
    return source, journal, inverse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, required=True)
    a = p.parse_args()
    build = a.build.resolve()
    if build.exists():
        raise ValueError("fresh Python-only normal source build required")
    if sha(PARENT) != "8361f2cad577da735d828e52cd22ef3998ac81192a9941917df1ab688caaa4b2":
        raise ValueError("frozen original normal driver drift")
    original = PARENT.read_text()
    driver, journal, inverse = transform(original)
    if inverse != original:
        raise ValueError("non-admission original driver behavior changed")
    build.mkdir(parents=True)
    (build / "tuning.py").write_text(driver)
    for name in ("semantic_quality.py", "test_semantic_quality.py", "make_native_admission.py"):
        shutil.copy2(HERE / name, build / name)
    compile(driver, str(build / "tuning.py"), "exec")
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-B", "-m", "unittest",
                             "dev.benchmarks.immutable_interval_worker_sep22.test_semantic_quality"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    ready = {"schema": "immutable96-host-only-normal-original22-Python-source-v1", "pass": True,
             "GPU_executed": False, "model_fixture_tokenizer_response_capture_operand_payload_read_or_hashed": False,
             "parent_driver_path": str(PARENT), "parent_driver_sha256": sha(PARENT),
             "literal_inverse_source_journal_pass": True, "journal": journal,
             "original22_graders_bodies_plan_budgets_cache_numeric_descriptor_gates_unchanged": True,
             "samebinary_flag0_shared_getenv_counter_cost_included": True,
             "separate_best_parent53_226_performance_comparison_required": True,
             "fresh_Root_bounded_native_receipt_required_before_execution": True,
             "files": [{"path": str(path), "sha256": sha(path)} for path in (build / "tuning.py", build / "semantic_quality.py", build / "test_semantic_quality.py", build / "make_native_admission.py", HERE / "prepare_normal.py", HERE / "semantic_quality.py", HERE / "test_semantic_quality.py", HERE / "make_native_admission.py")],
             "CPU_unit_test_stdout": result.stdout, "CPU_unit_test_stderr": result.stderr,
             "Root_native_model_quality_and_performance_pending": True}
    (build / "CPU_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps({"pass": True, "CPU_READY": str(build / "CPU_READY.json"), "CPU_READY_sha256": sha(build / "CPU_READY.json")}))

if __name__ == "__main__":
    main()
