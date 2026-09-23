#!/usr/bin/env python3
"""Generate exact untimed QMV_F32 sum taps from an immutable original source."""
import argparse
import hashlib
import json
from pathlib import Path
import re

EXPECTED_SOURCE_SHA256 = "4c271477b47d037cbe4018f73e6ba8f1a27dbfca4c9a0e98a6fce93b0155e73a"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def extract_function(source, name):
    found = re.search(r"inline void " + re.escape(name) + r"\s*\(", source)
    if found is None:
        raise ValueError("missing original function: " + name)
    begin = source.rfind("template <", 0, found.start())
    opening = source.index("{", found.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def replace_exact(source, before, after, expected):
    count = source.count(before)
    if count != expected:
        raise ValueError(f"exact source replacement count {count} != {expected}")
    return source.replace(before, after)


def tokens(source):
    return "".join(re.sub(r"//[^\n]*", "", source).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", required=True)
    args = parser.parse_args()
    original, output, journal = map(Path, (args.original, args.output, args.journal))
    source = original.read_text()
    if sha(source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("frozen original QMV_F32 source hash mismatch")
    if output.exists() or journal.exists():
        raise SystemExit("fresh generated source and journal paths required")
    math = extract_function(source, "project_math")
    driver = extract_function(source, "project")
    original_math, original_driver = math, driver
    changes = []

    def change(text, before, after, expected, reason):
        changed = replace_exact(text, before, after, expected)
        changes.append({"before": before, "after": after,
                        "count": expected, "reason": reason})
        return changed

    math = change(math, "inline void project_math(", "inline void project_math_tap(",
                  1, "private instrumentation name")
    math = change(math, "device bfloat *output, device atomic_uint *diagnostics,\n",
                  "device bfloat *output, device atomic_uint *diagnostics,\n"
                  "    device float *raw_f32,\n", 1, "raw tap pointer argument")
    math = change(math, "      if (lane == 0) {\n        const bfloat value = bfloat(sum);",
                  "      if (lane == 0) {\n"
                  "        raw_f32[route * p.output_size + out_row + row] = sum;\n"
                  "        const bfloat value = bfloat(sum);",
                  1, "one raw full-F32 sum store; no arithmetic changes")
    math_changes = changes[:]
    restored = math
    for record in reversed(math_changes):
        restored = replace_exact(restored, record["after"], record["before"],
                                 record["count"])
    if restored != original_math or tokens(restored) != tokens(original_math):
        raise ValueError("project_math byte/token restoration failed")

    driver = change(driver, "inline void project(", "inline void project_tap(",
                    1, "private driver name")
    driver = change(driver,
                    "device atomic_uint *diagnostics, constant FlashAffineParams &p,\n",
                    "device atomic_uint *diagnostics, device float *raw_f32,\n"
                    "    constant FlashAffineParams &p,\n",
                    1, "raw tap pointer argument plumbing")
    driver = change(driver, "project_math<Bits, GroupSize,",
                    "project_math_tap<Bits, GroupSize,", 2, "instrumented driver call")
    driver = change(driver,
                    "x, weights, scales, biases, output, diagnostics, p, out_row,",
                    "x, weights, scales, biases, output, diagnostics, raw_f32, p, out_row,",
                    2, "raw pointer forwarded; preserve original fast/fallback dispatch")
    restored_driver = driver
    for record in reversed(changes[len(math_changes):]):
        restored_driver = replace_exact(restored_driver, record["after"],
                                        record["before"], record["count"])
    if restored_driver != original_driver or tokens(restored_driver) != tokens(original_driver):
        raise ValueError("project driver byte/token restoration failed")
    generated = ("// Generated untimed instrumentation; source journal sealed beside build.\n"
                 "namespace gdn_ab_original_qmv {\n" + math + "\n\n" + driver +
                 "\n} // namespace gdn_ab_original_qmv\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated)
    result = {"schema": "splash-gdn-ab-qmv-f32-tap-source-journal-v1",
              "gpu_executed": False, "model_payload_read": False,
              "original_source": str(original.resolve()),
              "original_source_sha256": sha(source),
              "generated_source": str(output.resolve()),
              "generated_source_sha256": sha(generated),
              "math_restored_byte_exact": True, "math_restored_tokens_exact": True,
              "driver_restored_byte_exact": True, "driver_restored_tokens_exact": True,
              "original_math_sha256": sha(original_math),
              "original_driver_sha256": sha(original_driver),
              "added_math_operations": "one F32 sum store; no arithmetic",
              "changes": changes}
    journal.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": True, "gpu_executed": False,
                      "generated_sha256": sha(generated)}))


if __name__ == "__main__":
    main()
