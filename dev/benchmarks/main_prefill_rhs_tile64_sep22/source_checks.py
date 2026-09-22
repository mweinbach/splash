"""Callable CPU/source checks; never compile, initialize devices or read payloads."""
from pathlib import Path
import ast
import hashlib
import json
import re

def host_contract(text):
    case = text[text.index("for(unsigned test=0;test<4;++test)"):text.index("mutableX[0]=word;mutableIDs[0]=id;for(unsigned i=0;i<2;++i)")]
    return [
        'FlashMoEPointwiseParams{2048,2560,10,512},{1,2048,1},{256,1,1}' in text and 'FlashMoEPointwiseParams{2048,2560,10,512},{2048,1,1}' not in text,
        'void poisonShipping' in text and '{a.s.packedActivated,a.s.scatteredDown,a.out}' in text and text.count('poisonShipping(a[i])') >= 2 and 'finite(a[i].out,true,uint64_t(mp::rows)*2560)' in text,
        all(token in text for token in ('row<mp::rows','column<2560','slot=lane;slot<10;slot+=8','partial[lane]=bf(number(partial[lane])+number(v))','routed=bf(number(routed)+number(partial[lane]))')) and text.count('canonicalCombine(f,a[i])') >= 4,
        'record.expected=test<2?0x80000004u:0x80000005u;' in case and 'require(record.shipping[0]==record.expected,' in case and 'require(record.shipping[0]==record.shipping[1]&&record.probe[0]==record.probe[1]&&record.shipping[0]==record.probe[0],' in case,
        case.index('record.expected=') < case.index('p.publish(false,false)') and 'record.shippingObserved[i]=true' in case and 'record.probeObserved[i]=true' in case and 'record.submissionsAfter=backend.submissionCount();p.publish(false,false);' in case and 'p.publish(false,false);require(record.shipping[0]' in case,
        'else f<<"null"' in text and 'c.shippingObserved[arm]' in text and 'c.probeObserved[arm]' in text and 'record.excluded[i]+=inverse[r]>=mp::routes' in case,
        text.index('ScopeFinalizer teardown') < text.index('std::vector<Guarded>all') and 'native.reset();host.reset();const auto sample=gov.snapshot();' in text and 'p.teardownSampled=true' in text and 'catch(const std::exception&e){p.teardownError=e.what();}' in text,
        'if(after==UINT64_MAX)f<<"null"' in text and 'if(reserved==UINT64_MAX)f<<"null"' in text and 'teardown_sampled' in text and 'teardown_sampling_error' in text]

def run_checks(directory=None):
    here = Path(directory) if directory is not None else Path(__file__).parent
    texts = {name: (here / name).read_text() for name in ("candidate.metal", "abi.hpp", "packing.hpp", "oracle.mm", "prepare.py", "probe-source-journal.json")}
    shader, abi, packing, oracle, prepare = [texts[name] for name in ("candidate.metal", "abi.hpp", "packing.hpp", "oracle.mm", "prepare.py")]
    journal = json.loads(texts["probe-source-journal.json"]); count = 0
    def require(value, message):
        nonlocal count
        if not value: raise AssertionError(message)
        count += 1
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    current = Path(journal["current_source_path"]).read_text()
    for value, key in ((shader, "candidate_metal_sha256"), (abi, "abi_hpp_sha256"), (current, "current_source_sha256")):
        require(digest(value) == journal[key], "Frozen source hash drift: " + key)
    marker = "template <ushort M, bool Probe>\ninline void adaptive_expert_tail_sg2k128_gate_math("
    original = current[current.index(marker):current.index("kernel void adaptive_expert_tail_sg2k128_sep21_gate_up_m32_control(")]
    baseline = shader[shader.index(marker):shader.index("\n// Candidate:")]
    candidate_start = shader.index("template <ushort M, bool Probe>", shader.index("\n// Candidate:"))
    candidate = shader[candidate_start:shader.index("\nkernel void ")]
    for value, key in ((original, "current_source_helper_region_sha256"), (baseline, "observed_baseline_helper_region_sha256"), (candidate, "candidate_helper_region_sha256")):
        require(digest(value) == journal[key], "Literal helper hash drift: " + key)
    recovered = candidate.replace("main_prefill_rhs_tile64_sep22_candidate_", "adaptive_expert_tail_sg2k128_")
    for change in reversed(journal["allowed_rhs_changes"]):
        require(recovered.count(change["after"]) == change["expected_count"], "RHS change anchor count")
        recovered = recovered.replace(change["after"], change["before"])
    require(recovered == baseline, "Reverse RHS diff must recover literal baseline")
    erase = lambda text: re.sub(r"    if constexpr \(Probe\) \{\n.*?\n    \}\n", "", text, flags=re.S)
    require(erase(baseline) == erase(original), "Optional tap erasure must recover original arithmetic")
    require(current[:current.index(marker)] in shader, "Original sigmoid/job validator prelude")
    for anchor in journal["unchanged_anchors"]:
        require(baseline.count(anchor["literal"]) == anchor["baseline_count"] and candidate.count(anchor["literal"]) == anchor["candidate_count"], "Preserved arithmetic anchor")
    require(baseline.count("constexpr ushort SG = 2, K = 128;") == 2 and baseline.count("if (valid_rows <= 16)") == 2, "Literal SG2/K128 adaptive tail")
    require(baseline.index("raw_g[at] = gd[i]") < baseline.index("const float gs") and baseline.index("scaled_g[at] = gv") < baseline.index("const bfloat silu"), "GU raw/late-scale/BF16/SwiGLU chronology")
    require(baseline.index("raw[at] = dot[i]") < baseline.index("const float scale") < baseline.index("const float result = dot[i] * scale"), "Down literal dot/late-scale chronology")
    concepts = 0
    for valid in range(1, 33):
        for inner in (2560, 640):
            math_rows = 16 if valid <= 16 else 32
            require(valid <= math_rows and inner % 128 == 0 and list(range(0, inner, 128))[-1] + 128 == inner, "Logical row/K128 tail concept")
            concepts += 1
    entries = re.findall(r"kernel void (\w+)\(.*?\n\}", shader, re.S)
    require(entries == [row["name"] for row in journal["pipeline_inventory"]] and len(entries) == 8, "Exact private entry inventory")
    for row in journal["pipeline_inventory"]:
        entry = re.search(r"kernel void " + row["name"] + r"\(.*?\n\}", shader, re.S).group()
        params = 11 if row["phase"] == "gate_up" else 10; extra = 4 if params == 11 else 2
        require(f"constant FlashInt8ExpertStoreParams &p [[buffer({params})]]" in entry, "Original normal parameter slot")
        slots = [int(n) for n in re.findall(r"\[\[buffer\((\d+)\)\]\]", entry)]
        require(slots == list(range(params + 1 + (extra if row["probe"] else 0))), "Probe ABI slots")
        require([slot for slot in slots if slot != params] == list(range(params)) + (list(range(params + 1, params + extra + 1)) if row["probe"] else []), "Device probe buffers skip params")
    require("result.owner=std::move(g)" in oracle and "result.commands.assign(result.owner.dispatches().begin()" in oracle, "Copied plans retain graph-owned byte buffers")
    require("dispatch.buffers.push_back({12+i,a.g[i]})" in oracle and "dispatch.buffers.push_back({11+i,a.down[i]})" in oracle, "Harness probe slots skip graph-owned inline params")
    require("void poisonProbes" in oracle and oracle.count("poisonProbes(a[i])") >= 2 and "0x7fc10000u" in oracle and "uint16_t(0x7fc1)" in oracle, "Poison all raw/BF16 probe interiors before healthy and malformed taps")
    require("malformed shipping/probe fullchain coupling" in oracle and "malformed GU probe including untouched NaN cells" in oracle and "malformed down probe including untouched NaN cells" in oracle, "Malformed literal probe/output coupling and untouched-cell parity")
    rows, routes, alignment = 2048, 20480, 16384
    guarded = lambda n: (n + 128 + alignment - 1) // alignment * alignment
    fixture = [rows*2560*2, routes*8, routes*2, rows*2560*2, rows*2]
    scratch = [512*4, 513*4, routes*4, routes*4, (routes+63)*2560*2, 513*4, 4, 3071*8, (routes+63)*640*2, routes*2560*2, rows*2560*2, 4]
    probes = [routes*640*4]*2 + [routes*640*2]*2 + [routes*2560*4, routes*2560*2]
    totals = {"fixtureBytes": sum(map(guarded, fixture)), "scratchBytes": sum(map(guarded, scratch)), "probeBytes": sum(map(guarded, probes))}
    totals["nativePlannedBytes"] = 2524446720 + alignment + guarded(2524446720) + totals["fixtureBytes"] + 2*(totals["scratchBytes"] + totals["probeBytes"])
    env = {"rows": rows, "routes": routes, "jobCapacity": 3071, "alignment": alignment, "guarded": guarded, **{key: (lambda n=value: n) for key, value in totals.items()}}
    for name, expected in totals.items():
        expression = re.search(r"constexpr uint64_t " + name + r"\(\)\{return (.*?);\}", packing).group(1)
        expression = re.sub(r"uint64_t\(([^()]+)\)", r"(\1)", expression).replace("ULL", "")
        tree = ast.parse(expression, mode="eval")
        require(all(isinstance(node, (ast.Expression, ast.BinOp, ast.Constant, ast.Name, ast.Call, ast.Load, ast.Add, ast.Mult)) for node in ast.walk(tree)) and all(node.id in env for node in ast.walk(tree) if isinstance(node, ast.Name)), "Pure memory arithmetic source")
        require(eval(expression, {"__builtins__": {}}, env) == expected, "Independent guarded memory formula: " + name)
    require("alignment=16384,guard=64,hostAllowance=512ULL<<20" in packing and "return rounded(bytes+2*guard)" in packing, "Guard64/alignment16K/host512MiB")
    require(len(probes) == 6 and oracle.count("arm(backend,all)") == 2, "Two arms with six distinct probe planes each")
    require("p.actualInput=argc==7" in oracle and "if(p.actualInput)" in oracle and "synthetic RMS" in oracle and "std::sqrt(sum/2560)" in oracle, "Honest synthetic RMS versus actual mode")
    require("external Root actual fixture manifest digest" in oracle and "Root_actual_model_capture" in oracle and "exact current completed normalized source position" in oracle and "current high-acceptance FMA-only capture policy" in oracle, "Retained actual current manifest gate")
    require("source_review_GO_required_before_compile" in prepare, "Independent GO before any build")
    require("actual_current_fixture_manifest_required_before_actual_input_label_or_integration" in prepare and "explicitly synthetic per-row RMS BF16 A" in prepare, "Synthetic ROI admission distinct from retained actual-capture gate")
    for index, passed in enumerate(host_contract(oracle)): require(passed, "LIVE host source contract " + str(index))
    mutations = [('},{1,2048,1},{256,1,1}', '},{2048,1,1},{256,1,1}'), ('void poisonShipping', 'void removedPoison'), ('slot=lane;slot<10;slot+=8', 'slot=lane;slot<10;slot+=1'), ('record.expected=test<2?0x80000004u:0x80000005u;', 'record.expected=test<2?0x80000004u:0x80000001u;'), ('record.submissionsAfter=backend.submissionCount();p.publish(false,false);', 'record.submissionsAfter=backend.submissionCount();'), ('record.excluded[i]+=inverse[r]>=mp::routes', 'record.excluded[i]+=0'), ('native.reset();host.reset();const auto sample=gov.snapshot();', 'const auto sample=gov.snapshot();'), ('if(after==UINT64_MAX)f<<"null"', 'if(after==UINT64_MAX)f<<after')]
    for before, after in mutations: require(before in oracle and not all(host_contract(oracle.replace(before, after))), "Synthetic host source mutation refused")
    return {"pass": True, "SourceNoCompile": True, "CPU_checks": count, "host_source_negative_cases": len(mutations), "conceptual_K128_tail_cases": concepts, "native_plan_bytes": totals["nativePlannedBytes"], "host_plan_bytes": 512 << 20, "GPU_executed": False, "payload_reads": False}

if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True))
