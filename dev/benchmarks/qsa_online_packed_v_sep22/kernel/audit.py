#!/usr/bin/env python3
"""Bounded actual AIR graph audit, admitting only the declared V address map."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import ir_graph as g

def wide_terms(function, value, scale=1, seen=None):
    """Recognize the actual i64 affine address nodes; stop at i32 coordinates."""
    seen = set() if seen is None else seen
    if value in seen: raise g.Refusal("Cycle in declared V address")
    seen = seen | {value}
    rhs = function.defs[value]["rhs"]
    match = re.fullmatch(r"add(?: nuw| nsw)* i64 (%[-\w.]+), (%[-\w.]+)", rhs)
    if match:
        result = wide_terms(function, match[1], scale, seen)
        for ref, coefficient in wide_terms(function, match[2], scale, seen).items():
            result[ref] = result.get(ref, 0) + coefficient
        return result
    match = re.fullmatch(r"shl(?: nuw| nsw)* i64 (%[-\w.]+), (\d+)", rhs)
    if match: return wide_terms(function, match[1], scale * (1 << int(match[2])), seen)
    match = re.fullmatch(r"zext i32 (%[-\w.]+) to i64", rhs)
    if match: return {match[1]: scale}
    raise g.Refusal("Unexpected actual V i64 address node: " + rhs)

def dimension_roles(function, refs):
    if len(refs) == 1:
        rhs = function.defs[refs[0]]["rhs"]
        match = re.fullmatch(r"add(?: nuw| nsw)* i32 (%[-\w.]+), (%[-\w.]+)", rhs)
        if not match: raise g.Refusal("Expected original online dimension sum")
        refs = [match[1], match[2]]
    if len(refs) != 2: raise g.Refusal("Wrong declared dimension coordinate count")
    chunk, d = None, None
    for ref in refs:
        rhs = function.defs[ref]["rhs"]
        match = re.fullmatch(r"shl(?: nuw| nsw)* i32 (%[-\w.]+), 6", rhs)
        if match:
            if chunk is not None: raise g.Refusal("Duplicated dimension chunk")
            chunk = match[1]
        elif re.fullmatch(r"lshr i32 (%[-\w.]+), 6", rhs):
            if d is not None: raise g.Refusal("Duplicated within-chunk dimension")
            d = ref
        else: raise g.Refusal("Unexpected actual dimension role: " + rhs)
    if chunk is None or d is None: raise g.Refusal("Incomplete actual dimension roles")
    return chunk, d

def map_V_pointer(function, packed):
    loads = []
    for record in function.instructions:
        if record["rhs"].startswith("load bfloat, "):
            pointers = [v for v in function.value_refs(record["rhs"]) if function.pointer_base(v) == 2]
            if pointers: loads.append((record, pointers))
    if len(loads) != 1 or len(loads[0][1]) != 1: raise g.Refusal("Expected one static global BF16 V load")
    load, pointers = loads[0]
    pointer = pointers[0]
    record = function.defs[pointer]
    match = re.fullmatch(r"getelementptr inbounds bfloat, bfloat addrspace\(1\)\* " + re.escape(function.args[2]) + r", i64 (%[-\w.]+)", record["rhs"])
    if not match: raise g.Refusal("Global V load has unexpected typed pointer")
    terms = wide_terms(function, match[1])
    token_coefficient, kv_coefficient, dim_coefficient = (1, 524288, 2048) if packed else (512, 256, 1)
    token = [v for v, coefficient in terms.items() if coefficient == token_coefficient]
    kv = [v for v, coefficient in terms.items() if coefficient == kv_coefficient]
    dims = [v for v, coefficient in terms.items() if coefficient == dim_coefficient and v not in token]
    if len(token) != 1 or len(kv) != 1 or len(dims) not in (1, 2): raise g.Refusal("Actual V polynomial coefficients differ")
    if set(terms) != set(token + kv + dims): raise g.Refusal("Undeclared V address coordinate")
    kv_rhs = function.defs[kv[0]]["rhs"]
    if kv_rhs != "extractelement <3 x i32> " + function.args[8] + ", i64 1": raise g.Refusal("V kv is not the original logical group.y")
    chunk, d = dimension_roles(function, dims)
    old_pointer = record["rhs"]
    # Keep the four actual coordinate dependencies, type, address space and base.
    # Only this certified integer GEP is replaced in the comparison projection.
    record["rhs"] = ("V_bit_permuted_pointer bfloat addrspace(1)* " + function.args[2] +
                     ", token i32 " + token[0] + ", kv i32 " + kv[0] +
                     ", chunk i32 " + chunk + ", d i32 " + d)
    return {"function": function.name, "packed": packed, "original_actual_GEP": old_pointer,
            "actual_load": load["rhs"], "wide_coefficients": terms,
            "coordinates": {"token": token[0], "kv": kv[0], "chunk": chunk, "d": d},
            "pointer_projection": record["rhs"], "all_coordinate_dependencies_retained": True,
            "source_bound": "Fresh begin0/rows2048; source128-window mapping and unchanged guarded token<=query or commonStop masks imply token<2048; kv<2; chunk<4; d<64",
            "no_address_overflow_in_domain": True}

def require_equal(left, right, what):
    difference = g.difference(left, right)
    if difference: raise g.Refusal(what + ": " + json.dumps(difference))

def declarations_and_globals(original, other):
    records = []
    for name, declaration in other.declarations.items():
        key = g.helper_alias(name)
        if key not in original.declarations: raise g.Refusal("Unknown imported declaration: " + key)
        original.begin_metadata_context(); other.begin_metadata_context()
        require_equal(original.normalize(original.declarations[key]), other.normalize(declaration), "Imported declaration/attributes")
    for name, body in other.globals.items():
        key = g.helper_alias(name)
        if key not in original.globals: raise g.Refusal("Unknown private TG/type allocation: " + key)
        original.begin_metadata_context(); other.begin_metadata_context()
        require_equal(original.normalize(original.globals[key]), other.normalize(body), "Original TG/type allocation")
        records.append({"private": name, "original": key, "allocation_or_type_exact": True})
    return records

def selected_functions(original, other, map_packed):
    results, certificates = [], []
    for name, function in other.functions.items():
        key = g.helper_alias(name)
        if key not in original.functions: raise g.Refusal("Unknown selected SDK/helper function: " + key)
        left = original.functions[key]
        if map_packed and ("bulk_qsa_online_partition" in key or "bulk_temporal_sg8_online" in key):
            certificates.append(map_V_pointer(left, False))
            certificates.append(map_V_pointer(function, True))
        graph_a = left.graph()[0]
        graph_b = function.graph()[0]
        require_equal(graph_a, graph_b, "Actual selected FP/load/pointer/call/control/attribute graph " + key)
        has_V_projection = bool(map_packed and ("bulk_qsa_online_partition" in key or "bulk_temporal_sg8_online" in key))
        if not has_V_projection:
            require_equal(left.full(), function.full(), "Literal complete actual helper " + key)
        results.append({"original": key, "private": name, "pass": True,
                        "actual_graph": graph_a, "complete_graph": not has_V_projection,
                        "explicit_V_pointer_projection": has_V_projection})
    return results, certificates

def reducer_tap(original, tapped):
    name = next(name for name in tapped.functions if name.endswith("_reduce_tap"))
    function = tapped.functions[name]
    left = original.functions["flash_qsa_fast_prefill_bulk_reduce_2048"]
    if len(function.args) != 11 or len(left.args) != 9: raise g.Refusal("Reducer tap actual ABI differs")
    # Exact Metal ABI: original buffers0..5 then three builtins; tap buffers6/7
    # precede builtins in AIR. Move only those two known sinks to the graph tail.
    order = list(range(6)) + [8, 9, 10, 6, 7]
    function.args = [function.args[i] for i in order]
    function.arg_types = [function.arg_types[i] for i in order]
    stores = function.stores()
    sinks = [entry for entry in stores if entry[3] in (9, 10)]
    if sorted(entry[3] for entry in sinks) != [9, 10]: raise g.Refusal("Tap has undeclared/missing stores")
    raw = next(entry for entry in sinks if entry[3] == 9)
    rounded = next(entry for entry in sinks if entry[3] == 10)
    if not raw[1].startswith("float ") or not rounded[1].startswith("bfloat "): raise g.Refusal("Tap sink type differs")
    raw_refs = function.value_refs(raw[1])
    rounded_refs = function.value_refs(rounded[1])
    if len(raw_refs) != 1 or len(rounded_refs) != 1: raise g.Refusal("Tap live value relation differs")
    if re.match(r"^fdiv\b.*\bfloat\b", function.defs[raw_refs[0]]["rhs"]) is None: raise g.Refusal("Tap raw value does not use the sole original quotient")
    division = [r for r in function.instructions if re.match(r"^fdiv\b.*\bfloat\b", r["rhs"])]
    if len(division) != 1: raise g.Refusal("Tap recomputes division")
    if raw_refs[0] not in function.value_refs(function.defs[rounded_refs[0]]["rhs"]): raise g.Refusal("Rounded tap does not consume the same quotient")
    gated = [entry for entry in stores if entry[3] == 3]
    if len(gated) != 1: raise g.Refusal("Original gated output store count differs")
    offsets = []
    for entry in (raw, rounded, gated[0]):
        rhs = function.defs[entry[2]]["rhs"]
        match = re.fullmatch(r"getelementptr inbounds (?:float|bfloat), (?:float|bfloat) addrspace\(1\)\* %[-\w.]+, i64 (%[-\w.]+)", rhs)
        if not match: raise g.Refusal("Unexpected typed live-tap output pointer")
        offsets.append(match[1])
    if len(set(offsets)) != 1: raise g.Refusal("Tap output indices do not use the identical original live index")
    # Metal adds exact noalias scopes for the two new disjoint output buffers.
    # Preserve all original scopes, distinct-node sharing and payloads; project
    # only these declared sink memberships from list metadata, never FP attrs.
    extra_scopes, domains = [], set()
    for identifier, body in tapped.metadata.items():
        match = re.fullmatch(r'distinct !\{!' + identifier + r', !(\d+), !"air-alias-scope-arg\(([67])\)"\}', body)
        if match:
            extra_scopes.append((identifier, int(match[2]))); domains.add(match[1])
    if sorted(slot for identifier, slot in extra_scopes) != [6, 7] or len(domains) != 1: raise g.Refusal("Tap extra alias scopes differ from buffers6/7")
    domain = next(iter(domains))
    expected = 'distinct !{!' + domain + ', !"air-alias-scopes(' + name + ')"}'
    if tapped.metadata[domain] != expected: raise g.Refusal("Tap alias domain does not belong to the exact declared entry")
    extra = {identifier for identifier, slot in extra_scopes}
    alias_projection = []
    for identifier, body in tuple(tapped.metadata.items()):
        if re.fullmatch(r'!\{(?:!\d+(?:, )?)+\}', body):
            members = re.findall(r'!(\d+)', body)
            removed = [member for member in members if member in extra]
            if removed:
                after = '!{' + ', '.join('!' + member for member in members if member not in extra) + '}'
                tapped.metadata[identifier] = after
                alias_projection.append({"metadata_ID": identifier, "actual": body,
                                         "original_scope_memberships_retained": after,
                                         "only_declared_sink_memberships_projected": removed})
    graph, _, diagnostic = function.graph(base_args=9)
    if len(diagnostic) != 2: raise g.Refusal("Only two explicit tap stores may be projected")
    require_equal(left.graph()[0], graph, "Actual reducer shipping graph after two live tap sinks")
    other = next(f for n, f in tapped.functions.items() if "bulk_qsa_fast_gate" in n)
    key = g.helper_alias(other.name)
    require_equal(original.functions[key].full(), other.full(), "Tap gate complete body/attributes")
    return {"pass": True, "pipeline": name, "two_tap_sinks_only": [entry[0]["rhs"] for entry in sinks],
            "one_original_division": division[0]["rhs"], "rounded_conversion": function.defs[rounded_refs[0]]["rhs"],
            "shipping_graph": graph, "builtin_argument_mapping": order,
            "all_three_stores_use_identical_original_live_index": offsets[0],
            "exact_two_sink_alias_scope_addition": alias_projection,
            "host_requirement": "Raw and rounded sink extents disjoint from all original buffers and each other"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    args = p.parse_args(); build = args.build.resolve()
    report = {"schema": "online-packed-V-QSA-actual-AIR-arithmetic-audit-v1", "GPU_work": False,
              "candidate_native_FP_tree_match": False, "taps_shipping_FP_tree_match": False,
              "V_bit_permutation_proved": False, "failures": [],
              "normalization": ["Exact journaled private namespace and Itanium identifier lengths",
                                "SSA alpha identity with sharing/control/load/call/attributes preserved",
                                "Only two actual certified V integer GEPs mapped to the four original coordinate dependencies",
                                "Only the two live reducer tap stores and exact builtin ABI mapping"]}
    try:
        original, native, candidate, native_tap, candidate_tap = [g.Module(build / (n + ".ll")) for n in ("original", "native", "candidate", "native_reduce_tap", "candidate_reduce_tap")]
        report["native_allocation_type_proof"] = declarations_and_globals(original, native)
        report["candidate_allocation_type_proof"] = declarations_and_globals(original, candidate)
        report["native_complete_selected_graphs"], _ = selected_functions(original, native, False)
        report["candidate_selected_graphs"], report["actual_V_GEP_certificates"] = selected_functions(original, candidate, True)
        report["candidate_native_FP_tree_match"] = True
        # Fresh original module: the declared virtual V pointer is not relevant
        # to reducers and must not leak into the untimed tap admission.
        original = g.Module(build / "original.ll")
        report["native_reducer_tap"] = reducer_tap(original, native_tap)
        report["candidate_reducer_tap"] = reducer_tap(original, candidate_tap)
        report["taps_shipping_FP_tree_match"] = True
        permutation = json.loads((build / "V-permutation-proof.json").read_text())
        if permutation.get("pass") is not True or permutation["words"] != 1048576: raise g.Refusal("Integer V permutation proof differs")
        pack = g.Module(build / "pack.ll")
        function = pack.functions["qsa_online_packed_v_pack"]
        if any(g.FP.match(r["rhs"]) or g.FP_LOAD.match(r["rhs"]) for r in function.instructions): raise g.Refusal("Pack contains floating arithmetic/load")
        if re.search(r"\b(?:half|bfloat|float|double)\b", function.body): raise g.Refusal("Pack contains a floating type")
        report["compiled_pack_graph"] = function.full()
        report["V_bit_permutation_proved"] = True
    except (g.Refusal, ValueError, KeyError, StopIteration) as error:
        report["failures"].append(str(error))
    report["pass"] = all(report[key] for key in ("candidate_native_FP_tree_match", "taps_shipping_FP_tree_match", "V_bit_permutation_proved")) and not report["failures"]
    (build / "arithmetic-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("pass", "candidate_native_FP_tree_match", "taps_shipping_FP_tree_match", "V_bit_permutation_proved", "failures")}))
    return 0 if report["pass"] else 2

if __name__ == "__main__": sys.exit(main())
