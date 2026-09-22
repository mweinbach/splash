#!/usr/bin/env python3
"""Seal a scheduling-only GDN A/B pair on the completed exact HC worker."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PRIVATE=Path("dev/benchmarks/gdn_ab_merge_sep21")
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def replace(text:str,a:str,b:str)->str:
    if text.count(a)!=1:raise ValueError(f"GDNABworkeranchor drift:{a!r}")
    return text.replace(a,b)
def transform(relative:str,text:str)->str:
    if relative not in ["runtime/flash/FlashForward.cpp","runtime/flash/FlashWorker.mm"]:return text
    text='#include "dev/benchmarks/gdn_ab_merge_sep21/bridge.hpp"\n'+text
    if relative.endswith("FlashForward.cpp"):
        text=replace(text,'  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");','''  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");
  const bool mergeGDNAB = gdn_ab_merge::requested();''')
        anchor='''      affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
      affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));'''
        text=replace(text,anchor,'''      if (impl_->mergeGDNAB && (rows == 1 || rows == 4)) {
        (void)gdn_ab_merge::requested();
        const auto &a = impl_->weights.projection(attention + ".in_proj_a");
        const auto &b = impl_->weights.projection(attention + ".in_proj_b");
        if (gdn_ab_merge::geometry(a, rows) && gdn_ab_merge::geometry(b, rows) && flashAffineFastEnabled()) {
          gdn_ab_merge::add(graph, mixed, a, b, bf(Scratch::GDNA, 48), bf(Scratch::GDNB, 48), diag, rows);
        } else {
          affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
          affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));
        }
      } else {
        affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
        affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));
      }''')
        return replace(text,'  return std::string(flashAffineSemantics()) +','''  return std::string(flashAffineSemantics()) +
      (impl_->mergeGDNAB ? std::string(";") + gdn_ab_merge::kSemantics : "") +''')
    text=replace(text,"      std::signal(SIGPIPE, SIG_IGN);","      (void)gdn_ab_merge::requested(); // Freezebefore paths/backend/model.\n      std::signal(SIGPIPE, SIG_IGN);")
    marker='      << R"(,"kernel_routes":)"'
    text=replace(text,marker,'''      << R"(,"gdn_ab_merge_enabled":)" << (gdn_ab_merge::requested() ? "true" : "false")
      << R"(,"gdn_ab_merge_semantics":)" << json::quote(gdn_ab_merge::kSemantics)
'''+marker)
    anchor="  const auto requestTrace = requestCommandTraceInfo();"
    text=replace(text,anchor,'''  const auto abGraphs = gdn_ab_merge::counters();
  out << R"(,"gdn_ab_merge_route_counters":{"scope":"main-target ordinary/verify physicalR1/R4 only; graph construction, notGPUcompletion; 36pairedgraphs replace72projections perfull eligiblecall; batchpolicyunchanged","paired_graph_calls":)"
      << abGraphs.calls << R"(,"paired_graph_rows":)" << abGraphs.rows << '}';
'''+anchor)
    start=text.index('      << R"(,"identity":{"source":)"');end=text.index('      << R"(,"weight_format":"mlx_affine_preconverted"})"',start)
    assert "abGraphs" not in text[start:end] and "paired_graph_calls" not in text[start:end]
    return text

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=ROOT/"build/prefill-hc-inject-norm-sep21-worker-v3");p.add_argument("--build",type=Path,default=ROOT/"build/gdn-ab-merge-hc-v3-sep21-worker-v1");args=p.parse_args();base,build=args.base.resolve(),args.build.resolve()
    if build.exists():raise ValueError("freshGDNABworkerbuild required")
    parentBytes=(base/"overlay-manifest.json").read_bytes();parent=json.loads(parentBytes);build.mkdir(parents=True);shutil.copytree(base/"source",build/"source")
    records=[]
    for entry in parent["files"]:
        relative=entry["path"];src=base/"source"/relative;assert sha(src)==entry["overlay_sha256"]
        data=transform(relative,src.read_text()).encode();dest=build/"source"/relative;dest.write_bytes(data)
        records.append({**entry,"gdn_ab_base_overlay_sha256":sha(src),"gdn_ab_changed":data!=src.read_bytes(),"overlay_sha256":hashlib.sha256(data).hexdigest()})
    component=build/"source"/PRIVATE
    if component.exists():raise ValueError("base already containsGDNABcomponent")
    shutil.copytree(HERE,component)
    reference=build/"source/runtime/metal/kernels/shared/flash_affine_qmv_f32.metal";reference.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/"runtime/metal/kernels/shared/flash_affine_qmv_f32.metal",reference)
    subprocess.run([str(ROOT/".venv/bin/python"),str(component/"shader_generate.py"),"--original",str(reference),"--output",str(component/"gdn_ab_qmv_probe_generated.metal"),"--journal",str(build/"probe-source-journal.json")],cwd=ROOT,check=True)
    known={e["path"] for e in records}
    for path in sorted((build/"source").rglob("*")):
        if path.is_file():
            relative=path.relative_to(build/"source").as_posix()
            if relative not in known:records.append({"path":relative,"new_gdn_ab_file":True,"overlay_sha256":sha(path)})
    paths={"REUSED":[],"CORE":[],"AIRS":[]};inputs=[]
    def freeze(path:Path,kind:str)->None:
        dest=build/"reused"/f"{len(inputs):03d}-{path.name}";dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);paths[kind].append(dest.relative_to(build).as_posix());inputs.append({"original":str(path),"private_path":dest.relative_to(build).as_posix(),"sha256":sha(path),"kind":kind})
    make=(base/"link-inputs.mk").read_text()
    for path in sorted((base/"host").glob("*.o")):
        if path.stem not in ["FlashForward","FlashWorker"]:freeze(path,"REUSED")
    for kind in paths:
        line=next(line for line in make.splitlines() if line.startswith(kind+" := "))
        for token in line.removeprefix(kind+" := ").split():
            path=base/token.removeprefix("$(BUILD)/")
            if kind=="REUSED" and path.stem in ["FlashForward","FlashWorker"]:continue
            freeze(path,kind)
    # Base private shaders are link prerequisites outside the reused AIR list.
    for path in sorted(base.glob("*.air")):freeze(path,"AIRS")
    link="\n".join(f"{kind} := "+" ".join("$(BUILD)/"+name for name in names) for kind,names in paths.items())+"\n";(build/"link-inputs.mk").write_text(link)
    manifest=copy.deepcopy(parent);manifest.update({"route":"exact-GDN-AB-merged-scheduling-on-HCv3-v1","gdn_ab_composed":True,"gdn_ab_base_build":str(base),"gdn_ab_base_manifest_sha256":hashlib.sha256(parentBytes).hexdigest(),"gdn_ab_transform_sha256":sha(Path(__file__)),"gdn_ab_added_gpu_bytes":0,"gdn_ab_numerical_derivative_unchanged":True,"gdn_ab_scope":"mainForwardR1/R4only; batchunchanged; otherrows/formatsfallback","files":records,"gdn_ab_frozen_inputs":inputs,"gdn_ab_link_make_sha256":sha(build/"link-inputs.mk"),"gpu_work":False,"model_payload_reads":0})
    (build/"overlay-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"prepared":str(build),"frozen_sources":len(records),"frozen_inputs":len(inputs),"gpu_work":False,"model_payload_reads":0}))
if __name__=="__main__":main()
