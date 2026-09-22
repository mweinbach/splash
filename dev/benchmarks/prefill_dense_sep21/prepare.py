#!/usr/bin/env python3
"""CPU-only precise/relaxed BF16 dense prefill screen source preparation.

Model and captured tensor payloads are never opened by this generator.
The oracle itself reads tensors only when root explicitly runs GPU qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def replace_once(text: str,before: str,after: str) -> str:
    if text.count(before)!=1:
        raise ValueError(f"Source anchor drift: {before[:120]}")
    return text.replace(before,after,1)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shader(text: str) -> str:
    text=replace_once(text,"template <ushort Groups>","template <ushort Groups, bool Relaxed>")
    text=replace_once(text,"      false,true,false,matmul2d_descriptor::mode::multiply);",
        "      false,true,Relaxed,matmul2d_descriptor::mode::multiply);")
    text=replace_once(text,"#define PREFILL_DENSE_ENTRY(G)","#define PREFILL_DENSE_ENTRY(G,R,LABEL)")
    text=replace_once(text,"kernel void flash_dense_cache_prefill_m128_n64_sg##G(",
        "kernel void prefill_dense_precision_m128_n64_sg##G##_##LABEL(")
    text=replace_once(text,"  prefill_dense_tile<G>(input,weights,output,diagnostics,p,group,threads,tid);",
        "  prefill_dense_tile<G,R>(input,weights,output,diagnostics,p,group,threads,tid);")
    text=replace_once(text,"PREFILL_DENSE_ENTRY(4)\nPREFILL_DENSE_ENTRY(8)",
        "PREFILL_DENSE_ENTRY(4,false,strict)\nPREFILL_DENSE_ENTRY(8,false,strict)\n"
        "PREFILL_DENSE_ENTRY(4,true,relaxed)\nPREFILL_DENSE_ENTRY(8,true,relaxed)")
    return ("// Private numerical-alternative screen. Only descriptor relaxedprecision\n"
        "// differs; BF16 inputs/weights, wholeK,F32 destination, finalBF16 cast,\n"
        "// semantic shape guards and traversal are retained from production.\n"+text)


def oracle(text: str) -> str:
    text=replace_once(text,"namespace {\nusing namespace splash::metal;",
        "namespace {\nbool includeRegisterCandidates = false;\nusing namespace splash::metal;")
    text=replace_once(text,
        "  bool production = false, vector = false, block512 = false, prefillProduction = false;",
        "  bool production = false, vector = false, block512 = false, prefillProduction = false;\n"
        "  bool precisionCandidate = false, relaxedPrecision = false, registerCandidate = false;")
    text=replace_once(text,
        "  Error baselineError, geometryError, oracleError;",
        "  Error baselineError, geometryError, oracleError, strictPrecisionError;")
    text=replace_once(text,
        "      chosen.prefillProduction = true; result.push_back(std::move(chosen));",
        "      chosen.prefillProduction = true; result.push_back(std::move(chosen));\n"
        "      for (uint32_t groups : {4u,8u}) for (bool relaxed : {false,true}) {\n"
        "        auto alternative = variant(128,64,static_cast<uint32_t>(plan.traversal),false,groups);\n"
        "        alternative.precisionCandidate = true; alternative.relaxedPrecision = relaxed;\n"
        "        alternative.name = std::string(relaxed ? \"relaxed_\" : \"strict_\") + alternative.name;\n"
        "        result.push_back(std::move(alternative));\n"
        "      }\n"
        "      if (includeRegisterCandidates && shape.k%512 == 0)\n"
        "        for (uint32_t columns : {32u,64u}) for (bool relaxed : {false,true}) {\n"
        "          auto alternative = variant(32,columns,static_cast<uint32_t>(plan.traversal),false,1,true);\n"
        "          alternative.precisionCandidate = true; alternative.relaxedPrecision = relaxed;\n"
        "          alternative.registerCandidate = true;\n"
        "          alternative.name = std::string(relaxed ? \"register_relaxed_\" : \"register_strict_\") + alternative.name;\n"
        "          result.push_back(std::move(alternative));\n"
        "        }")
    text=replace_once(text,
        "        const std::string pipeline = v.prefillProduction\n",
        "        const std::string pipeline = v.registerCandidate\n"
        "            ? \"prefill_dense_register_m32_n\" + std::to_string(v.n) + \"_sg1_bk512_sk16_\" +\n"
        "                (v.relaxedPrecision ? \"relaxed\" : \"strict\")\n"
        "            : v.precisionCandidate\n"
        "            ? \"prefill_dense_precision_m128_n64_sg\" + std::to_string(v.groups) +\n"
        "                (v.relaxedPrecision ? \"_relaxed\" : \"_strict\")\n"
        "            : v.prefillProduction\n")
    text=replace_once(text,
        "      std::map<std::tuple<uint32_t,uint32_t,uint32_t,bool>, std::vector<uint16_t>> geometryReferences;",
        "      std::map<std::tuple<uint32_t,uint32_t,uint32_t,bool,bool>, std::vector<uint16_t>> geometryReferences;")
    text=replace_once(text,
        "        const auto key = std::tuple{candidate.m,candidate.n,candidate.groups,candidate.block512};",
        "        const auto key = std::tuple{candidate.m,candidate.n,candidate.groups,candidate.block512,candidate.relaxedPrecision};")
    text=replace_once(text,
        "        b.reset(); auto warm = graphFor(candidate, b, rows, shape, 1);\n"
        "        (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();",
        "        b.reset(); auto warm = graphFor(candidate, b, rows, shape, 1);\n"
        "        (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();\n"
        "        b.reset(); (void)backend.submitCommand(warm.dispatches()); b.guards();\n"
        "        require(compare(b.result(),actual).mismatches == 0, \"candidate deterministic repeat changed output\");")
    text=replace_once(text,
        "        << \"{\\\"experiment\\\":\\\"prefill4k_dense_v1\\\",\\\"arithmetic\\\":\\\"whole_k_bf16_mpp\\\",\"",
        "        << \"{\\\"experiment\\\":\\\"prefill_dense_precision_sep21_v1\\\",\\\"arithmetic\\\":\\\"whole_k_bf16_mpp_strict_vs_relaxed\\\",\"\n"
        "        << \"\\\"numerical_alternative\\\":true,\\\"full_model_quality_qualified\\\":false,\\\"deterministic_repeats_checked\\\":true,\"")
    text=replace_once(text,
        "            << \",\\\"median_gpu_ms\\\":\" << candidate.medianGPU",
        "            << \",\\\"precision_candidate\\\":\" << (candidate.precisionCandidate ? \"true\" : \"false\")\n"
        "            << \",\\\"register_candidate\\\":\" << (candidate.registerCandidate ? \"true\" : \"false\")\n"
        "            << \",\\\"relaxed_precision\\\":\" << (candidate.relaxedPrecision ? \"true\" : \"false\")\n"
        "            << \",\\\"full_output_sha256\\\":\" << splash::json::quote(outputHashes.at(candidate.name))\n"
        "            << \",\\\"median_gpu_ms\\\":\" << candidate.medianGPU")
    text=replace_once(text,
        "      std::vector<uint16_t> reference;",
        "      std::vector<uint16_t> reference;\n      std::map<std::string,std::string> outputHashes;\n"
        "      std::map<uint32_t,std::vector<uint16_t>> strictPrecisionReferences;")
    text=replace_once(text,
        "        candidate.baselineError = compare(actual, reference);",
        "        outputHashes[candidate.name] = sha256(actual.data(), actual.size()*sizeof(uint16_t));\n"
        "        candidate.baselineError = compare(actual, reference);\n"
        "        if (candidate.precisionCandidate) {\n"
        "          if (!candidate.relaxedPrecision) strictPrecisionReferences[candidate.groups] = actual;\n"
        "          require(strictPrecisionReferences.contains(candidate.groups), \"strict paired reference missing\");\n"
        "          candidate.strictPrecisionError = compare(actual,strictPrecisionReferences.at(candidate.groups));\n"
        "        }")
    text=replace_once(text,
        "        report << \",\\\"same_geometry_error\\\":\"; candidate.geometryError.write(report);",
        "        report << \",\\\"same_geometry_error\\\":\"; candidate.geometryError.write(report);\n"
        "        report << \",\\\"paired_strict_precision_error\\\":\"; candidate.strictPrecisionError.write(report);")
    text=replace_once(text,
        "uint64_t shaderBoundaryTests(MetalBackend &backend, bool productionTraversal, bool productionPrefill) {",
        "uint64_t shaderBoundaryTests(MetalBackend &backend, bool productionTraversal, bool productionPrefill,\n"
        "    const std::string &overridePipeline = {}, uint32_t overrideGroups = 4,\n"
        "    uint32_t overrideM = 0, uint32_t overrideN = 0) {")
    text=replace_once(text,
        "  Shape boundary; boundary.k = 32; boundary.n = 320;",
        "  Shape boundary; boundary.k = overrideM ? 512 : 32; boundary.n = 320;")
    text=replace_once(text,
        "  const uint32_t tileM = productionPrefill ? 128 : 32;",
        "  const uint32_t tileM = overrideM ? overrideM : productionPrefill ? 128 : 32;")
    text=replace_once(text,
        "  const uint32_t tileN = productionPrefill ? 64 : 128;",
        "  const uint32_t tileN = overrideN ? overrideN : productionPrefill ? 64 : 128;")
    text=replace_once(text,
        "  FlashDenseCacheParams valid{rows,32,320,64,128,tileM,tileN,3};",
        "  FlashDenseCacheParams valid{rows,boundary.k,320,64,128,tileM,tileN,3};")
    text=replace_once(text,
        "    case 4: p.output_count = 32; break;",
        "    case 4: p.output_count = tileN/2; break;")
    text=replace_once(text,
        "  const std::string pipeline = productionPrefill\n",
        "  const std::string pipeline = !overridePipeline.empty() ? overridePipeline : productionPrefill\n")
    text=replace_once(text,
        "      valid, traversal(rows/tileM,128/tileN,3), {128, 1, 1});",
        "      valid, traversal(rows/tileM,128/tileN,3), {overrideGroups*32, 1, 1});")
    text=replace_once(text,
        "        p, {1, 1, 1}, {128, 1, 1});",
        "        p, {1, 1, 1}, {overrideGroups*32, 1, 1});")
    text=replace_once(text,
        "    const auto boundaryChecks = shaderBoundaryTests(backend, productionTraversal, productionPrefill);",
        "    auto boundaryChecks = shaderBoundaryTests(backend, productionTraversal, productionPrefill);\n"
        "    for (uint32_t groups : {4u,8u}) for (const char *mode : {\"strict\",\"relaxed\"})\n"
        "      boundaryChecks += shaderBoundaryTests(backend,false,true,\n"
        "          \"prefill_dense_precision_m128_n64_sg\" + std::to_string(groups) + \"_\" + mode,groups);\n"
        "    if (includeRegisterCandidates) for (uint32_t columns : {32u,64u})\n"
        "      for (const char *mode : {\"strict\",\"relaxed\"})\n"
        "        boundaryChecks += shaderBoundaryTests(backend,false,true,\n"
        "            \"prefill_dense_register_m32_n\" + std::to_string(columns) + \"_sg1_bk512_sk16_\" + mode,1,32,columns);")
    text=replace_once(text,
        "      if (arg == \"--production-prefill\") { productionPrefill = true; continue; }",
        "      if (arg == \"--production-prefill\") { productionPrefill = true; continue; }\n"
        "      if (arg == \"--register-candidates\") { includeRegisterCandidates = true; continue; }")
    # Error covers every output cell, with finite/sign/near-zero checks alongside
    # sampled independently serial FP64 reductions. No universal exactness claim.
    text=replace_once(text,
        "  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;",
        "  double maximumAbsolute = 0, squaredError = 0, squaredReference = 0;\n"
        "  double productSum = 0, squaredActual = 0, maximumNearZeroAbsolute = 0;\n"
        "  uint64_t signFlips = 0, nearZeroElements = 0;")
    text=replace_once(text,
        "    squaredError += delta * delta; squaredReference += b * b;",
        "    squaredError += delta * delta; squaredReference += b * b;\n"
        "    productSum += a*b; squaredActual += a*a;\n"
        "    signFlips += a != 0 && b != 0 && std::signbit(a) != std::signbit(b);\n"
        "    if (std::abs(b) <= 1e-3) { ++nearZeroElements; maximumNearZeroAbsolute =\n"
        "        std::max(maximumNearZeroAbsolute,std::abs(delta)); }")
    text=replace_once(text,
        "        << \",\\\"relative_l2\\\":\" << relativeL2() << '}';",
        "        << \",\\\"relative_l2\\\":\" << relativeL2()\n"
        "        << \",\\\"cosine\\\":\" << productSum/std::sqrt(std::max(1e-30,squaredReference*squaredActual))\n"
        "        << \",\\\"sign_flips\\\":\" << signFlips << \",\\\"near_zero_elements\\\":\" << nearZeroElements\n"
        "        << \",\\\"near_zero_max_abs\\\":\" << maximumNearZeroAbsolute << '}';")
    return text


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"build/prefill-dense-sep21")
    args=parser.parse_args();output=args.output.resolve()
    if ROOT/"build" not in output.parents: raise ValueError("Private output must be under build")
    paths={"shader":ROOT/"runtime/metal/kernels/shared/flash_dense_cache_prefill.metal",
        "oracle":ROOT/"dev/benchmarks/prefill4k_dense/oracle.mm",
        "capture_manifest":ROOT/"build/prefill4k-dense/actual-activations-v1/manifest.json"}
    source={key:path.read_bytes() for key,path in paths.items()}
    output.mkdir(parents=True,exist_ok=True)
    files={"precision.metal":shader(source["shader"].decode()).encode(),
        "oracle.mm":oracle(source["oracle"].decode()).encode()}
    manifest=json.loads(source["capture_manifest"])
    # Only rows/shapes with actual current M128 cached-dense policy use this screen.
    shapes={(10240,320),(2560,10240),(2560,6144),(6144,2560),(2560,12288),(2560,512),(2560,640)}
    manifest["cases"]=[case for case in manifest["cases"] if (case["input_size"],case["output_size"]) in shapes]
    files["actual-captures.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    for name,data in files.items(): (output/name).write_bytes(data)
    report={"schema":"splash-prefill-dense-precision-source-v1","gpu_executed":False,
        "model_or_capture_payload_bytes_read":0,"normal_sources_modified":False,
        "numerical_alternative":True,"model_quality_qualified":False,"actual_capture_cases":len(manifest["cases"]),
        "source_hashes":{key:sha(value) for key,value in source.items()},
        "generated_hashes":{key:sha(value) for key,value in files.items()},
        "symbols":[f"prefill_dense_precision_m128_n64_sg{groups}_{mode}" for groups in (4,8) for mode in ("strict","relaxed")],
        "changed_arithmetic":"MPP descriptor relaxedprecision false versus true; unchanged BF16 operands,F32destination,wholeK,finalBF16 cast",
        "qualification_required":["full_output_error","sampled_serial_fp64","guards_sticky_diagnostics","input_weight_immutability","deterministic_repeats","whole_model_generation_quality"]}
    (output/"source-manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"prepared":str(output),**report}))


if __name__=="__main__":main()
