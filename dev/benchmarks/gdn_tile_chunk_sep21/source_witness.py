#!/usr/bin/env python3
"""Independent CPU source/ABI/address/ODR witness; never creates a Metal device."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
POLICY = "gdn-t32-v32-sg8-local-guard-095-deferred-state-commit-native-v8-t16-four-waves-from-hybrid-seed-v1"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def part(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a + len(start))]

def run(command, stdin=None):
    result = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=60)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}

def weak_definitions(ir):
    definitions = {}
    attributes = dict(re.findall(r'^attributes\s+#([0-9]+)\s*=\s*(\{[^\n]*\})', ir, re.M))
    pattern = r'^define\s+(?P<header>[^\n]*\b(?:linkonce_odr|weak_odr)\b[^\n]*)\{\n(?P<body>.*?)^\}'
    for m in re.finditer(pattern, ir, re.M | re.S):
        symbol = re.search(r'@(?:"([^"]+)"|([^ (]+))\(', m['header'])
        if not symbol:
            continue
        # Metadata/attribute numbering is module-local. Keep arithmetic, calls,
        # argument/SSA identities and constants in the actual definition body.
        normalized = m['header']+'\n'+m['body']
        normalized = re.sub(r'![0-9]+', '!MD', normalized)
        normalized = re.sub(r'#([0-9]+)', lambda a: '#ATTR'+attributes.get(a.group(1), 'MISSING'), normalized)
        definitions[symbol.group(1) or symbol.group(2)] = sha(normalized.encode())
    return definitions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/gdn-tile-chunk-sep21")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-ir", action="store_true", help="CPU-compile candidate/control LLVM IR and compare weak helper symbols")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh CPU witness filename")
    build = args.build.resolve()
    if ROOT / "build" not in build.parents:
        raise ValueError("Diagnostic outputs must stay under private build/")
    files = ["candidate.metal", "abi.hpp", "native_helper.metal", "native_audit_helper.metal",
             "frozen/canonical.metal", "frozen/canonical_audit.metal", "frozen/metal/abi/FlashGDN.h",
             "frozen/v6_math_control.metal", "frozen/v6_math_control.provenance.json"]
    data = {name: (SOURCE / name).read_bytes() for name in files}
    checks, errors = {}, {}
    def check(name, callback):
        try:
            checks[name] = bool(callback())
        except Exception as error:
            checks[name] = False; errors[name] = str(error)
    candidate = data["candidate.metal"].decode()
    abi = data["abi.hpp"].decode()
    prep = part(candidate, "inline void gtc_prepare(", "template <ushort Values")
    apply = part(candidate, "inline void gtc_apply(", "// Explicit packed buffer ABI")
    expected_native = data["frozen/canonical.metal"].decode().split("#define GDS_ENTRY")[0]
    expected_native = expected_native.replace("gds_", "gtcn_").replace("constant FlashGDNParams &p", "thread const FlashGDNParams &p")
    # Wrapper dispatch documentation and its max-thread attribute are not part
    # of the literal helper. V8 callers have their own 256-thread entrypoints.
    expected_audit = data["frozen/canonical_audit.metal"].decode().split("// Dispatch only {1, 8, lanes}")[0]
    expected_audit = expected_audit.replace("gds_", "gtca_").replace("constant FlashGDNParams &p", "thread const FlashGDNParams &p")
    check("native_helper_literal_arithmetic_after_name_addressspace_adapter", lambda: expected_native.strip() == data["native_helper.metal"].decode().strip())
    check("audit_helper_literal_arithmetic_after_name_addressspace_adapter", lambda: expected_audit.strip() == data["native_audit_helper.metal"].decode().strip())
    check("native_no_explicit_fma_or_reassociation", lambda: all(b"fma(" not in data[n] and b"fp contract(off)" in data[n] and b"fp reassociate(off)" in data[n] for n in ("native_helper.metal", "native_audit_helper.metal")))
    check("main_scope_explicit_b1", lambda: "p.lanes!=1" in apply and "control.gdn.lanes!=1" in candidate)
    check("immutable_initial_branch_decision", lambda: "const uint initialRange=range[chunk*48+head]" in apply and "const bool attemptWY=control.mode==3 || (!forced && !initialRange)" in apply)
    check("local_reason_separate_from_phase_arena", lambda: "threadgroup uint scratch[2332];threadgroup atomic_uint local" in candidate and "threadgroup atomic_uint &local" in apply)
    check("range_return_before_gram_and_inverse", lambda: prep.index("if (rangeReason) return") < prep.index("gop.run(k,k,kk)") < prep.index("inverse[token * Time + tid] = value"))
    check("deferred_state_commit", lambda: apply.index("update[i] = value;") < apply.index("if (control.mode==3 || !atomic_load_explicit(&local") < apply.index("recurrent[stateBase+ix[1]*128+ix[0]]=update[i]"))
    check("no_speculative_persistent_state_store", lambda: len(re.findall(r'recurrent\s*\[[^\n]*?\]\s*=', apply)) == 1)
    check("guard_threshold_unmodified", lambda: "sumD <= 0.095f * sumC" in apply and "sqrt(norm) * 1.000125f" in candidate)
    check("native_same_seed_four_v8_waves_two_t16_blocks", lambda: "cp.rows=count; cp.lanes=1" in apply and "wave<4" in apply and "const uint3 ng{head,group.y*4+wave,0}" in apply and "gtcn_recurrence<8,16>" in apply and "gtca_audit_recurrence<8,16>" in apply)
    check("chunk_local_input_output_tapes_rebased", lambda: all(x in apply for x in ("mixed+ulong(begin)*10240", "decay+ulong(begin)*48", "beta+ulong(begin)*48", "output+ulong(begin)*6144", "history+ulong(begin)*128*128", "deltaAudit+ulong(begin)*128", "outputAudit+ulong(begin)*128")))
    check("overwrite_order_device_fence_before_native", lambda: "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);" in part(apply, "} // speculative WY region", "if (native)"))
    check("seed_copy_device_fence_before_commit", lambda: "if constexpr (Audit || Probe)\n      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);" in apply)
    check("end_chunk_state_visibility", lambda: "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n  }" in apply)
    check("decision_single_leader_fixed_uint_layout", lambda: "if (!tid) decisions[(ulong(chunk)*48+head)*4+group.y]=" in apply and "reason|(native?256u:0u)|(forced?512u:0u)" in apply)
    check("audit_rejects_nonzero_heads", lambda: "(Audit && group.x != 0)" in apply)
    check("packed_cpp_abi_assertions", lambda: "sizeof(TileChunkParams)==56" in abi and "offsetof(TileChunkParams,mode)==48" in abi and "offsetof(TileChunkParams,reserved)==52" in abi)
    abi_probe = run(["xcrun", "-sdk", "macosx", "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I"+str(SOURCE), "-x", "c++", "-fsyntax-only", "-"], '#include "abi.hpp"\nstatic_assert(sizeof(FlashGDNParams)==48);\nstatic_assert(sizeof(uint32_t)==4);\n')
    check("cpp_abi_compiler_pass", lambda: abi_probe["exit_code"] == 0)

    geometry_cases, tiles = 0, 0
    def address_model():
        nonlocal geometry_cases, tiles
        for rows in (1,15,16,17,31,32,33,63,64,65,127,128,2047,2048):
            chunks=(rows+31)//32
            for stride in (3145728,3145732,3162112):
                owners=set(); decision=set(); range_words=set()
                for head in range(48):
                    for tile in range(4):
                        start=head*65536+tile*16384
                        assert start>=0 and start+16384<=stride
                        for wave in range(4):
                            value_group=4*tile+wave
                            for sg in range(8):
                                row=8*value_group+sg
                                assert 32*tile<=row<32*tile+32
                                owner=(head,row); assert owner not in owners; owners.add(owner)
                        for chunk in range(chunks):
                            begin=chunk*32; count=min(32,rows-begin)
                            assert 0<count<=32 and begin+count<=rows
                            for token in (0,count-1):
                                for dimension in (0,127):
                                    row=32*tile+dimension//4
                                    assert head*16384+row*128+127 < 48*16384
                                    assert (begin+token)*10240+4096+head*128+127 < rows*10240
                                    assert (begin+token)*48+head < rows*48
                                    assert (begin+token)*6144+head*128+32*tile+31 < rows*6144
                                    if head==0:
                                        assert ((begin+token)*128+32*tile+31)*128+127 < rows*128*128
                            index=(chunk*48+head)*4+tile
                            assert index not in decision; decision.add(index)
                            range_words.add(chunk*48+head); tiles+=1
                assert len(owners)==48*128 and len(decision)==chunks*48*4 and len(range_words)==chunks*48
                assert max(decision)==chunks*48*4-1 and max(range_words)==chunks*48-1
                geometry_cases+=1
        return True
    check("exhaustive_tail_tile_wave_stride_address_model", address_model)
    control=data["frozen/v6_math_control.metal"].decode()
    provenance=json.loads(data["frozen/v6_math_control.provenance.json"])
    original=Path(ROOT/provenance["source_path"]).read_bytes()
    original_text=original.decode()
    v6_seal=json.loads((ROOT/provenance["v6_cpu_seal_path"]).read_text())
    sealed_sources={r['path']:r['sha256'] for r in v6_seal['sources']}
    captured_names=("frozen/canonical.metal","frozen/canonical_audit.metal","frozen/metal/abi/FlashGDN.h")
    check("captured_native_and_abi_authenticated_against_v6_seal",lambda:all(
        sha(data[n])==sealed_sources['dev/benchmarks/gdn_nax_chunks_sep21_v6/'+n] for n in captured_names))
    def localize(text):
        return text.replace('gwy_','gtc_').replace('GWY_','GTC_').replace('gtc_ineligible(flags,batch,head,','gtc_mark(local,')
    check("cached_w_norm_predicates_literal_v6_order",lambda:
        localize(part(original_text,'inline void gwy_weight_norm(','template <ushort Time, ushort Groups>')).strip()==
        part(candidate,'inline void gtc_weight_norm(','template <ushort Time, ushort Groups>').strip())
    check("state_norm_predicates_literal_v6_order",lambda:
        localize(part(original_text,'    for (uint value = simdGroup;','    for (uint token = simdGroup;')).strip()==
        part(apply,'    for (uint value = simdGroup;','    for (uint token = simdGroup;').strip())
    check("delta_conditioning_predicates_literal_v6_order",lambda:
        localize(part(original_text,'    float localDelta2 =','    auto qt = tensor(')).strip()==
        part(apply,'    float localDelta2 =','    auto qt = tensor(').strip())
    expected_control=original[:original.index(b"#define GWY_PREP_ENTRY")].decode().replace("gwy_","gtcv6_")
    check("v6_control_sealed_source_authenticated", lambda: sha(original)==provenance["source_sha256"] and sha(data["frozen/v6_math_control.metal"])==provenance["control_sha256"] and sha((ROOT/provenance["v6_cpu_seal_path"]).read_bytes())==provenance["v6_cpu_seal_sha256"])
    check("v6_control_helper_math_literal_after_namespace_rename", lambda: control.startswith('#include "../abi.hpp"\n'+expected_control))
    check("control_helper_names_disjoint", lambda: not re.search(r'\bgwy_[A-Za-z_]+',control) and "gtcv6_prepare<32,8>" in control and "gtcv6_apply<32,32,8,true>" in control)
    check("safe_control_old_abi_and_tile_mapping", lambda: "private_gdn_tile_v6_prepare_control" in control and "[[buffer(11)]]" in control and "const uint3 selectedGroup{group.x,control.reserved,group.z}" in control)
    ir_records=[]
    if args.emit_ir:
        build.mkdir(parents=True,exist_ok=True)
        definitions=[]
        for name,filename in (("candidate",SOURCE/"candidate.metal"),("v6-math-control",SOURCE/"frozen/v6_math_control.metal")):
            ir_path=build/(name+"-source-witness.ll")
            source_sha=sha(filename.read_bytes())
            result=run(["xcrun","-sdk","macosx","metal","-std=metal4.1","-O3","-Wall","-Wextra","-Werror","-I"+str(SOURCE/"frozen"),"-mmacosx-version-min=27.0","-S","-emit-llvm",str(filename),"-o",str(ir_path)])
            check(name+"_ir_cpu_compile",lambda r=result:r["exit_code"]==0)
            check(name+"_source_stable_during_compile",lambda p=filename,h=source_sha:sha(p.read_bytes())==h)
            if result["exit_code"]==0:
                ir=ir_path.read_text(); definitions.append(weak_definitions(ir))
                ir_records.append({**result,"source_sha256":source_sha,"ir_path":str(ir_path),"ir_sha256":sha(ir.encode()),"weak_symbols":sorted(definitions[-1])})
            else: definitions.append({});ir_records.append(result)
        overlap=set(definitions[0])&set(definitions[1])
        incompatible=[symbol for symbol in sorted(overlap) if definitions[0][symbol]!=definitions[1][symbol]]
        user_overlap=[s for s in overlap if any(tag in s for tag in ("gtc_","gtcn_","gtca_","gtcv6_","gwy_"))]
        check("compiled_user_helper_odr_domains_disjoint",lambda:not user_overlap)
        check("compiled_shared_weak_helpers_body_compatible",lambda:not incompatible)
    else:
        overlap=set();incompatible=[]
    check("all_kernel_abi_helper_control_inputs_stable_during_witness",lambda:
          all((SOURCE/name).read_bytes()==data[name] for name in files))
    source_entries=[{"path":n,"sha256":sha(data[n]),"bytes":len(data[n])} for n in files]
    result={"schema":"splash.gdn-tile-chunk-cpu-source-witness.v1","pass":all(checks.values()),
        "checks":checks,"errors":errors,"sources":source_entries,
        "source_identity_sha256":sha("\n".join(e['path']+' '+e['sha256'] for e in source_entries).encode()),
        "numerical_policy":POLICY,"abi_probe":abi_probe,"geometry_cases":geometry_cases,"checked_tiles":tiles,
        "r2048_b1":{"range_uint_words":3072,"range_bytes":12288,"decision_uint_words":12288,"decision_bytes":49152,
            "combined_decision_bytes":61440,"coefficient_bytes":164757504,"prep_declared_tg_bytes":12804,
            "apply_declared_tg_bytes":9332,"audit_seed_bytes":4194304,"probe_seed_bytes":201326592},
        "ir_checked":args.emit_ir,"ir_modules":ir_records,"shared_weak_symbols":sorted(overlap),"incompatible_weak_symbols":incompatible,
        "gpu_executed":False,"payload_bytes_read":0,"native_v8_gpu_equivalence_proven":False,
        "safe_wy_gpu_equivalence_proven":False,"strict_f64_qualified":False,"whole_worker_allowed":False,
        "mode3_range_rejected_math_qualified":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"pass":result['pass'],"checks":len(checks),"failed_checks":[n for n,v in checks.items() if not v],"ir_checked":args.emit_ir,"gpu_executed":False,"output":str(args.output)}))
    if not result['pass']:raise SystemExit(1)

if __name__=="__main__":main()
