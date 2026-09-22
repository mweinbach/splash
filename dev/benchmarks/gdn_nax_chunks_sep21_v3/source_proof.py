#!/usr/bin/env python3
"""CPU-only frozen native replay closure and bounded selector/model proof."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bits(x):
    return struct.unpack("I", struct.pack("f", x))[0]


def range_select(alphas, betas, ftz=False):
    segment = f32(1.0)
    selected = any(not math.isfinite(b) or b < 0 or b > 1 for b in betas)
    selected |= any(0 < (bits(b) & 0x7fffffff) < 0x00800000 for b in betas)
    threshold = f32(f32(2.0**-126) * f32(1.000008))
    for alpha in alphas:
        raw = bits(alpha) & 0x7fffffff
        selected |= not math.isfinite(alpha) or alpha < 0 or alpha > 1
        if not raw:
            segment = f32(1.0)
        else:
            selected |= raw < 0x00800000
            operand = 0.0 if ftz and raw < 0x00800000 else abs(alpha)
            segment = f32(segment * operand)
            if ftz and 0 < abs(segment) < 2.0**-126:
                segment = 0.0
            selected |= not math.isfinite(segment) or segment < threshold
    return bool(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to replace a source witness")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "pass": bool(passed)})
        if not passed:
            raise RuntimeError(name)

    original = (SOURCE / "frozen/canonical.metal").read_bytes()
    helper = original[:original.index(b"#define GDS_ENTRY")]
    fallback = (SOURCE / "native_fallback.metal").read_bytes()
    check("native_helper_literal_byte_prefix", fallback[:len(helper)] == helper)
    audit = (SOURCE / "frozen/canonical_audit.metal").read_bytes()
    actual = (SOURCE / "native_fallback_audit.metal").read_bytes()
    check("native_audit_literal_full_source_prefix", actual[:len(audit)] == audit)
    check("native_fallback_exact_V16_T16_call", b"gds_recurrence<16,16>" in fallback)
    check("audit_fallback_exact_V16_T16_call", b"gds_audit_recurrence<16,16>" in actual)
    check("fallback_native_no_explicit_fma", b"fma(" not in helper and b"fma(" not in audit)
    check("native_no_contract_or_reassociate", b"fp contract(off)" in helper and b"fp reassociate(off)" in helper)
    check("audit_head_zero_guard", b"group.x != 0" in actual)
    snapshot_source = (SOURCE / "snapshot.metal").read_bytes()
    check("snapshot_copies_integer_words", b"device uint *state" in snapshot_source and b"device uint *snapshot" in snapshot_source)

    alpha_cases = [
        ("normal", [f32(.90)] * 32, [f32(.5)] * 32, False),
        ("actual_zero", [f32(0)] * 32, [f32(.5)] * 32, False),
        ("actual_negative_zero", [f32(-0.0)] * 32, [f32(.5)] * 32, False),
        ("nonzero_prefix_underflow", [f32(2.0**-100)] * 2 + [f32(1)] * 30, [f32(.5)] * 32, True),
        ("post_zero_relative_underflow", [f32(0)] + [f32(2.0**-100)] * 2 + [f32(1)] * 29, [f32(.5)] * 32, True),
        ("subnormal_operand_bits", [f32(2.0**-149)] + [f32(1)] * 31, [f32(.5)] * 32, True),
        ("decay_outside_domain", [f32(2)] + [f32(1)] * 31, [f32(.5)] * 32, True),
        ("beta_outside_domain", [f32(.9)] * 32, [f32(2)] + [f32(.5)] * 31, True),
        ("subnormal_beta_bits", [f32(.9)] * 32, [f32(2.0**-133)] + [f32(.5)] * 31, True),
    ]
    for ftz in [False, True]:
        for name, alphas, betas, expect in alpha_cases:
            check(f"range_model_{name}_ftz{int(ftz)}", range_select(alphas, betas, ftz) == expect)

    layout_cases = tiles = 0
    for rows in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2047, 2048]:
        for lanes in [1, 2, 4, 32]:
            chunks = (rows + 31) // 32
            stride = 3 * 32 * 128 + 32 * 32 + 32
            for lane in range(lanes):
                for head in range(48):
                    flag = lane * 48 + head
                    assert flag < lanes * 48
                    for tile in range(64):
                        saved = (lane * 48 + head) * 16384 + tile * 256
                        assert saved + 256 <= lanes * 48 * 16384
                        tiles += 1
                    for chunk in range(chunks):
                        base = ((lane * chunks + chunk) * 48 + head) * stride
                        assert base + stride <= lanes * chunks * 48 * stride
                        tiles += 1
            layout_cases += 1
    check("snapshot_flags_workspace_bound_cases", layout_cases == 48 and tiles > 0)

    round16k = lambda n: ((n + 16383) // 16384) * 16384
    coefficients = 64 * 48 * (3 * 32 * 128 + 32 * 32 + 32) * 4
    snapshot = 48 * 128 * 128 * 4
    flags = 48 * 4
    logical = coefficients + snapshot + flags
    physical = sum(map(round16k, [coefficients, snapshot, flags]))
    check("singleton_governor_arena", logical == 167116992 and physical == 167133184)
    policy = {
        "source_native_reference_equivalence": True,
        "gpu_native_reference_equivalence_proven": False,
        "strict_f64_accuracy_proven": False,
        "selector_is_whole_history_certificate": False,
        "selector_rounding_model": "MSL permits RTZ u=2^-23, FTZ, finite overflow saturation; selector heuristic only",
        "raw_tiny_transform_failure_discarded": False,
        "gpu_executed": False,
    }
    witness = {"schema": "splash.gdn-wy-native-replay-source-proof.v1", "pass": True,
               "checks": checks, "layout_cases": layout_cases, "checked_tiles": tiles,
               "arena": {"coefficients": coefficients, "snapshot": snapshot, "flags": flags,
                         "logical_bytes": logical, "rounded_physical_bytes": physical},
               "native_helper_sha256": hashlib.sha256(helper).hexdigest(),
               "native_audit_sha256": hashlib.sha256(audit).hexdigest(), "policy": policy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "checks": len(checks), "pass": True,
                      "logical_bytes": logical, "physical_bytes": physical}))


if __name__ == "__main__":
    main()
