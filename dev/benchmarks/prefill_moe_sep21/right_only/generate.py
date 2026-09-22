#!/usr/bin/env python3
"""Derive a bounded one-layer device-A/register-B screen, no model/GPU reads."""
from pathlib import Path
import argparse
import importlib.util


def replace(text,before,after,count=1):
    if text.count(before) !=count:raise RuntimeError(f'right-only adapter source drift:{before!r}')
    return text.replace(before,after)


INVENTORY=r'''
constexpr std::array<prefill_moe_sep21::Variant,3> kRightVariants{{
  {"right_only",64,1,true,true,true}, {"right_only",128,1,true,true,true},
  {"right_only",256,1,true,true,true}
}};
std::string rightPipelineName(const prefill_moe_sep21::Variant &v,bool gate) {
  return std::string("prefill_moe_sep21_right_only_") +(gate ? "gate_up" :"down_scatter") +
      "_m32_n64_k" +std::to_string(v.k) +"_sg1";
}
'''


def generate(destination):
    path=Path('dev/benchmarks/prefill_moe_sep21/one_layer/generate.py')
    spec=importlib.util.spec_from_file_location('prefill_moe_sep21_one_layer_source',path)
    if not spec or not spec.loader:raise RuntimeError('Bounded one-layer adapter unavailable')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.generate(destination)
    source=(destination/'oracle.mm').read_text()
    point='constexpr uint32_t kSelections = 10, kSticky = 0x80000000u;'
    source=replace(source,point,point +INVENTORY)
    source=replace(source,'  using prefill_moe_sep21::variants;','  const auto &variants=kRightVariants;')
    source=replace(source,'      plans.emplace_back(graphs[1].dispatches(),rows,variants[index]);',r'''      const prefill_moe_sep21::Variant controlScope{"whole",0,1,false,false,false};
      plans.emplace_back(graphs[1].dispatches(),rows,controlScope);
      for (auto &dispatch :plans.back().commands)
        if (dispatch.pipelineName.starts_with("prefill_moe_sep21_memory_"))
          dispatch.pipelineName=rightPipelineName(variants[index],dispatch.pipelineName.find("gate_up") !=std::string::npos);''')
    source=source.replace('prefill-moe-sep21-eleven-variant-one-layer-v1','prefill-moe-sep21-right-only-register-one-layer-v1')
    source=source.replace('eleven_variant_gpu_parity','right_only_register_gpu_parity')
    point='  out<<",\\\"variants\\\":[";'
    source=replace(source,point,'  out<<",\\\"right_only_device_A_bf16_register_B\\\":true,\\\"actual_descriptor_K64_K128_K256\\\":true,\\\"K256_down_tail_masked\\\":true";\n' +point)
    (destination/'oracle.mm').write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
