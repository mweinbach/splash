#!/usr/bin/env python3
"""Materialize the reviewed private source plan. No compiler or device work."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2"
SOURCE = PARENT / "source/dev/benchmarks/prefill4k_attention/bulk_attention_sg8.metal"
SOURCE_SHA = "9f5a1da89e05a20b109330eaced63a645000a5728538c8e9cafc738ce9ae0905"
AIR = PARENT / "reused/air/120-prefill4k_bulk_attention_sg8.air"
AIR_SHA = "7e560b9b3e4c598fa04d3d061ec88b93ca20d705849b550e6390158d97a352d2"

ADDRESS_EDITS = [
    ("values[(ulong(token) * 2 + kv) * 256 + dimension + d]",
     "values[(ulong(kv) * 256 + dimension + d) * 2048 + token]", 1),
    ("values[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i / 64]",
     "values[(ulong(kv) * 256 + chunk * 64 + i / 64) * 2048 + token]", 2),
]
HELPERS = [
    "bulk_qsa_mpp_failure", "bulk_qsa_mpp_visible", "bulk_qsa_mpp_token",
    "bulk_qsa_online_partition", "bulk_rowtile_error", "BulkQueryStorage",
    "bulk_rowtile_online", "BulkOnlineScratch", "BulkTemporalScratch",
    "BulkAttentionScratch", "bulk_temporal_sg8_error",
    "BulkTemporalSG8QueryStorage", "bulk_temporal_sg8_online", "bulk_qsa_fast_gate",
]
ENTRIES = {
    "flash_qsa_mpp_prefill_bulk_early_2048": "early",
    "flash_qsa_mpp_prefill_bulk_temporal_sg8_2048": "temporal",
    "flash_qsa_fast_prefill_bulk_reduce_2048": "reduce",
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def kernel_span(source, name):
    name_start = source.index("kernel void " + name + "(")
    start = source.rfind("[[max_total_threads_per_threadgroup", 0, name_start)
    opening = source.index("{", name_start)
    depth = 1
    cursor = opening + 1
    while depth:
        if source[cursor] == "{": depth += 1
        elif source[cursor] == "}": depth -= 1
        cursor += 1
    return start, cursor

def omit_entry(source, name, journal):
    start, end = kernel_span(source, name)
    original = source[start:end]
    marker = "// Private component omits the unselected entry: " + name + "."
    journal.append({"kind": "unselected_entry", "old": original, "new": marker, "count": 1})
    return source[:start] + marker + source[end:]

def isolate(source, role, taps=False):
    prefix = "qsa_online_packed_v_" + role + ("_tap" if taps else "") + "_"
    mapping = {name: prefix + name for name in HELPERS}
    mapping.update({name: "qsa_online_packed_v_" + role + "_" + phase + ("_tap" if taps else "")
                    for name, phase in ENTRIES.items()})
    for old, new in mapping.items():
        source = re.sub(r"\b" + re.escape(old) + r"\b", new, source)
    restored = source
    for old, new in mapping.items():
        restored = re.sub(r"\b" + re.escape(new) + r"\b", old, restored)
    return source, mapping, restored

def main():
    literal = SOURCE.read_bytes()
    if sha(literal) != SOURCE_SHA or sha(AIR.read_bytes()) != AIR_SHA:
        raise ValueError("Authoritative current-parent source/AIR changed")
    original = literal.decode()
    (HERE / "authoritative.metal").write_bytes(literal)
    common_journal = []
    selected = omit_entry(original, "flash_qsa_mpp_prefill_bulk_2048", common_journal)
    restored = selected
    for edit in reversed(common_journal):
        if restored.count(edit["new"]) != edit["count"]: raise ValueError("Ambiguous wrapper inverse")
        restored = restored.replace(edit["new"], edit["old"])
    if restored != original: raise ValueError("Selected-source inverse differs")
    changed = selected
    address_journal = []
    for old, new, count in ADDRESS_EDITS:
        if changed.count(old) != count or new in changed: raise ValueError("V gather anchor drift")
        changed = changed.replace(old, new)
        address_journal.append({"kind": "global_V_address", "old": old, "new": new, "count": count})
    inverse = changed
    for edit in reversed(address_journal): inverse = inverse.replace(edit["new"], edit["old"])
    if inverse != selected: raise ValueError("Address inverse differs")
    namespace_maps = {}
    for role, source in (("native", selected), ("candidate", changed)):
        isolated, mapping, back = isolate(source, role)
        if back != source: raise ValueError("Namespace inverse differs")
        (HERE / (role + ".metal")).write_text(isolated)
        namespace_maps[role] = mapping
        tap_journal = []
        tap = source
        for name in tuple(ENTRIES)[:2]: tap = omit_entry(tap, name, tap_journal)
        old = "    constant FlashQSAFastParams &params [[buffer(5)]],"
        new = old + ("\n    device float *raw_quotient [[buffer(6)]],"
                     "\n    device bfloat *rounded_attention [[buffer(7)]],")
        if tap.count(old) != 1: raise ValueError("Reducer ABI tap anchor drift")
        tap = tap.replace(old, new)
        tap_journal.append({"kind": "untimed_raw_quotient_binding", "old": old, "new": new, "count": 1})
        old = "  const bfloat attention = bfloat(value / sum);"
        new = ("  const float quotient = value / sum;\n"
               "  const bfloat attention = bfloat(quotient);\n"
               "  raw_quotient[(ulong(row) * 24 + head) * 256 + tid] = quotient;\n"
               "  rounded_attention[(ulong(row) * 24 + head) * 256 + tid] = attention;")
        if tap.count(old) != 1: raise ValueError("Reducer live-quotient tap anchor drift")
        tap = tap.replace(old, new)
        tap_journal.append({"kind": "untimed_live_quotient_store", "old": old, "new": new, "count": 1})
        tap_back = tap
        for edit in reversed(tap_journal):
            if tap_back.count(edit["new"]) != edit["count"]: raise ValueError("Ambiguous tap inverse")
            tap_back = tap_back.replace(edit["new"], edit["old"])
        if tap_back != source: raise ValueError("Tap inverse differs")
        isolated, mapping, back = isolate(tap, role, taps=True)
        if back != tap: raise ValueError("Tap namespace inverse differs")
        (HERE / (role + "_reduce_tap.metal")).write_text(isolated)
        namespace_maps[role + "_tap"] = mapping
        (HERE / (role + "_TAP_JOURNAL.json")).write_text(json.dumps(tap_journal, indent=2) + "\n")
    journal = {"schema": "online-packed-V-QSA-literal-source-journal-v1", "pass": True,
               "compiler_work": False, "GPU_work": False, "authoritative_source": str(SOURCE),
               "authoritative_source_sha256": SOURCE_SHA, "original_AIR": str(AIR),
               "original_AIR_sha256": AIR_SHA, "selected_wrapper_journal": common_journal,
               "candidate_address_journal": address_journal, "namespace_maps": namespace_maps,
               "candidate_changes_only_global_V_address": True,
               "exact_inverse_restores_entire_authoritative_source": True,
               "unselected_entry_omission_identical_native_candidate": True,
               "all_active_entry_bodies_otherwise_literal": True,
               "tap_quotient_computed_once": True}
    (HERE / "SOURCE_JOURNAL.json").write_text(json.dumps(journal, indent=2) + "\n")
    print(json.dumps({"pass": True, "compiler_work": False, "GPU_work": False,
                      "source_SHA": SOURCE_SHA, "AIR_SHA": AIR_SHA,
                      "candidate_global_V_address_occurrences": 3}))

if __name__ == "__main__": main()
