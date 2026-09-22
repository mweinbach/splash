#!/usr/bin/env python3
"""Freeze selected SG1 full-K HC-down screen; CPU build only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def replace(text:str,a:str,b:str)->str:
    if text.count(a)!=1:raise ValueError(f"SG1oracle sourceanchor drift:{a!r}")
    return text.replace(a,b)
def oracle()->str:
    text=(ROOT/"dev/benchmarks/hc_down_rowreuse_sep21/flash_hc_down_packed_oracle.mm").read_text()
    for a,b in [("FlashHCDownPacked.hpp","bridge.hpp"),("FlashHCDownPackedABI.h","abi.hpp"),("FlashHCDownPackedHostProvenance.hpp","HCDownSG1HostProvenance.hpp"),
        ("kFlashOracleHostObjectProvenance","kHCDownSG1Provenance"),("splash::flash::candidate","splash::flash::hcdown_sg1"),
        ("HCDownPackedMode","Mode"),("HCDownPackedDebug","Debug"),("hcDownPackedModeName","modeName"),
        ("addHCDownPackedLiteralWitness","addWitness"),("addHCDownPackedCandidate","addCandidate"),
        ("FLASH_HC_DOWN_PACKED_","FLASH_HC_DOWN_SG1_")]:text=text.replace(a,b)
    a=text.index("std::string metadata(");z=text.index("void midpoint(",a)
    text=text[:a]+'''std::string metadata(const char *path) {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metalunavailable");NSError *error=nil;
  id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];require(library!=nil,"SG1librarymissing");
  std::ostringstream out;out<<"{\\"pipelines\\":[";bool first=true;
  for(uint32_t bits:{4u,5u,6u,8u}) {
    for(uint32_t group:{32u,64u,128u}) {
      for(uint32_t kind:{0u,1u,4u}) {
    const auto name=kind==0 ? "flash_hc_down_sg1_q"+std::to_string(bits)+"_g"+std::to_string(group) : "flash_hc_down_probe_q"+std::to_string(bits)+"_g"+std::to_string(group)+"_s"+std::to_string(kind);
    id<MTLFunction> function=[library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];require(function!=nil,"SG1entrymissing");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];require(pipeline!=nil&&pipeline.threadExecutionWidth==32&&pipeline.maxTotalThreadsPerThreadgroup>=(kind==4?128u:32u),"SG1pipelinegeometryunsupported");
    if(!first)out<<',';first=false;out<<"{\\"name\\":"<<splash::json::quote(name)<<",\\"static_threadgroup_bytes\\":"<<pipeline.staticThreadgroupMemoryLength<<'}';
      }
    }
  }
  out<<"]}";return out.str();
}
''' + text[z:]
    # Raw taps use compact320/324 widths, including the injection-absent case.
    text=text.replace("uint64_t{row} * 324 + n","uint64_t{row} * (p.outputSize + 4) + n",1)
    # Midpoint reference function receives down only; no-injection final mixer
    # needs actual tap stride passed explicitly.
    text=replace(text,"MetalBuffer rawControl, MetalBuffer rawCandidate, uint32_t rows) {","MetalBuffer rawControl, MetalBuffer rawCandidate, uint32_t rows, uint32_t rawStride) {")
    text=text.replace("uint64_t{row} * (p.outputSize + 4) + n","uint64_t{row} * rawStride + n",1)
    text=replace(text,"const FlashAffineProjection *inj = weights.contains(prefix + \".block_inject_weight.weight\") ? &weights.projection(prefix + \".block_inject_weight\") : nullptr;",
                 "const FlashAffineProjection *inj = weights.contains(prefix + \".block_inject_weight.weight\") ? &weights.projection(prefix + \".block_inject_weight\") : nullptr;\n  const uint32_t rawStride = 320 + (inj ? 4 : 0);")
    text=text.replace("rows * 324 * 2","rows * rawStride * 2").replace("rows * 324 * 4","rows * rawStride * 4")
    text=text.replace("rows, 320, 324","rows, rawStride, rawStride").replace("uint64_t{row} * 324 + n","uint64_t{row} * rawStride + n")
    text=replace(text,"HCDownPackedParams post{}; post.literal.rows = rows; post.literal.has_injection = inj != nullptr;",
                 "FlashHCFusedParams post{}; post.rows = rows; post.has_injection = inj != nullptr; post.width = 2560; post.streams = 4; post.lowrank = 320; post.simdgroups = 4; post.norm_epsilon = 1e-6f;")
    text=text.replace('"flash_hc_down_packed_epilog_from_raw"','"flash_hc_down_sg1_epilog_from_raw"').replace("uint64_t{rows} * 324","uint64_t{rows} * rawStride")
    text=replace(text,"rawBF[0].view, rawBF[1].view, rows);","rawBF[0].view, rawBF[1].view, rows, rawStride);")
    text=replace(text,"sizeof(HCDownPackedParams) == 176 && sizeof(FlashHCFusedParams) == 160","sizeof(CommandTiming) == 200 && sizeof(FlashHCFusedParams) == 160")
    text=replace(text,"for (uint32_t i = 0; i < 2; ++i) require(*c::modeName","for (uint32_t i = 0; i < 1; ++i) require(*c::modeName")
    text=replace(text,"require(argc == 4, \"usage: flash-hc-down-packed-oracle PRIVATE_METALLIB PACKAGE REPORT_JSON\");",
                 "require(argc == 5 && std::string_view(argv[1]) == \"--gpu\", \"usage: oracle --gpu PRIVATE_METALLIB PACKAGE NEW_REPORT_JSON\");\n      ++argv; --argc;\n      require(!std::filesystem::exists(argv[3]) && !std::filesystem::exists(std::string(argv[3])+\".partial\"), \"choose freshSG1report\");")
    text=text.replace('"FLASH_HC_DOWN_SG1_ROWS", {4, 8, 16}','"FLASH_HC_DOWN_SG1_ROWS", {1, 4}').replace('"FLASH_HC_DOWN_SG1_MODES", {0, 1}, 1','"FLASH_HC_DOWN_SG1_MODES", {0}, 0')
    text=text.replace('r == 4 || r == 8 || r == 16','r == 1 || r == 4').replace('row-reuse qualifier supports physical R4/R8/R16 only','SG1qualifier supportsR1/R4only')
    text=text.replace("splash-sep21-hc-down-packed-rowreuse-qualified-v2","splash-sep21-hc-down-fullK-sg1-exact-v1")
    text=text.replace(">=150ms GPU warm per route; even balanced matched commands; no CPU tensor access between warm and timing","samefullKdot SIMDchronology; >=150ms GPUwarmperroute; evenbalancedSG4/SG1 commands; noCPUtensoraccessbetweenwarmandtiming")
    # Keep all timing behind the existing exact raw/F32/activation/injection gates.
    start=text.index("  while (warmedGPU[0]");end=text.index("  healthy(); // Payload",start)
    for token in ["contents()","healthy()","hash(","memcpy","memset"]:
        if token in text[start:end]:raise AssertionError("CPUmodelbufferaccessduringwarming/timing")
    return text

def main()->None:
    args=argparse.ArgumentParser(description=__doc__);args.add_argument("--build",type=Path,default=ROOT/"build/hc-down-sg1-sep21-v1");args.add_argument("--base",type=Path,default=ROOT/"build/moe-pointwise-sep21-worker-v1");args=args.parse_args()
    build,base=args.build.resolve(),args.base.resolve()
    if build.exists():raise ValueError("freshSG1buildrequired")
    build.mkdir(parents=True);shutil.copytree(base/"source",build/"source")
    src=build/"source/dev/benchmarks/hc_down_sg1_sep21";shutil.copytree(HERE,src)
    (src/"oracle.mm").write_text(oracle());shutil.copy2(ROOT/"dev/benchmarks/hc_down_rowreuse_sep21/FlashFloatBoundaryAudit.hpp",src/"FlashFloatBoundaryAudit.hpp")
    reference=build/"source/runtime/metal/kernels/shared/flash_hc_fused.metal";reference.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/"runtime/metal/kernels/shared/flash_hc_fused.metal",reference)
    names=["FlashAffine","FlashHC","FlashHCFused","FlashFloatDenseCache","FlashOperandStore","FlashDenseCache","FlashAffineMPP","FlashDescriptor","FlashWeights","FlashPLESSDStore","FlashInt8ExpertStoreMetadata"]
    objects=[base/"reused/base-host"/(name+".o") for name in names]+[base/"reused/core/engine/metal"/(name+".o") for name in ["MetalBackend","DeviceCapabilities"]]
    airLine=next(line for line in (base/"link-inputs.mk").read_text().splitlines() if line.startswith("AIRS := "))
    airs=[base/token.removeprefix("$(BUILD)/") for token in airLine.removeprefix("AIRS := ").split()]+[base/"pointwise.air"]
    sealed=[];frozenObjects=[];frozenAirs=[]
    for index,path in enumerate(objects+airs):
        dest=build/"reused"/f"{index:03d}-{path.name}";dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);sealed.append({"original":str(path),"frozen":str(dest),"sha256":sha(dest)})
        (frozenObjects if index<len(objects) else frozenAirs).append(dest)
    provenance={"schema":"splash-hcSG1-frozen-object-closure-v1","base_binary_sha256":sha(base/"splash-flash"),"reference_shader_sha256":sha(reference),"objects":sealed[:len(objects)]}
    (build/"HCDownSG1HostProvenance.hpp").write_text('#pragma once\ninline constexpr const char* kHCDownSG1Provenance = R"CLOSURE('+json.dumps(provenance,sort_keys=True)+')CLOSURE";\n')
    flags=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-Wno-deprecated-declarations","-I"+str(build/"source/runtime"),"-I"+str(src),"-I"+str(build),"-I"+str(build/"source"),"-mmacosx-version-min=27.0","-DSPLASH_INT8_EXPERIMENT=1","-fobjc-arc"]
    commands=[["xcrun","-sdk","macosx","metal","-std=metal4.1","-O3","-Wall","-Wextra","-Werror","-I"+str(build/"source/runtime"),"-I"+str(src),"-mmacosx-version-min=27.0","-c",str(src/"candidate.metal"),"-o",str(build/"candidate.air")],
      ["xcrun","-sdk","macosx","metallib",*map(str,frozenAirs),str(build/"candidate.air"),"-o",str(build/"splash.metallib")],
      ["xcrun","-sdk","macosx","clang++",*flags,str(src/"oracle.mm"),*map(str,frozenObjects),"-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(build/"oracle")]]
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)
    result=subprocess.run([str(build/"oracle"),"--cpu-self-test"],cwd=ROOT,text=True,capture_output=True,check=True);(build/"cpu-self-test.json").write_text(result.stdout)
    manifest={"schema":"splash-hc-down-sg1-frozen-cpu-preparation-v1","gpu_work":False,"model_payload_bytes_read":0,"source_files":[{"path":str(path),"sha256":sha(path)} for path in sorted((build/"source").rglob("*")) if path.is_file()],"frozen_inputs":sealed,"compiler_commands":commands,"runtime_sha256":{"oracle":sha(build/"oracle"),"splash.metallib":sha(build/"splash.metallib")},"reference_shader_sha256":sha(reference),"no_math_split":"full10240termdot/lane chronology/simd_sum/originalscalarpost unchanged; onlyCTA/SIMD grouping differs","cpu_model_buffer_touches_between_warm_and_timing":0}
    (build/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"pass":True,"build":str(build),"gpu_work":False,"model_payload_bytes_read":0,"sources":len(manifest["source_files"]),"frozen_inputs":len(sealed)}))
if __name__=="__main__":main()
