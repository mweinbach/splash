#!/usr/bin/env python3
"""Derive the strict adaptive-M16 oracle against existing SG2/static-K128 math."""
from pathlib import Path
import argparse
import importlib.util


FIXED_COMMANDS = r'''
// The native store creates original M32 jobs/parameters/grids. Adapt BOTH
// scratch graphs to the existing variant7 baseline before any GPU submission.
// Only the producer pipeline and the shared 64-thread SG2 launch differ from
// the native SG4 launch; ownership, argument bytes, buffers and grids do not.
std::vector<ComputeDispatch> fixedCommands(std::span<const ComputeDispatch> source,uint32_t rows) {
  std::vector<ComputeDispatch> result;uint32_t producers=0;
  for (const auto &original:source) {
    auto d=original;
    const bool gate=d.pipelineName=="flash_int8_expert_store_gate_up_m32_n64";
    const bool down=d.pipelineName=="flash_int8_expert_store_down_scatter_m32_n64";
    if (gate || down) {
      require(d.bytes.size()==1 && d.bytes[0].data &&
          d.bytes[0].sizeBytes==sizeof(FlashInt8ExpertStoreParams),"SG2 baseline producer parameter ABI differs");
      FlashInt8ExpertStoreParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));
      require(validTailParams(p,rows) && d.threadgroups.x==(gate ? 10u : 40u) &&
          d.threadgroups.y==p.job_capacity && d.threadgroups.z==1 &&
          d.threadsPerThreadgroup.x==128 && d.threadsPerThreadgroup.y==1 &&
          d.threadsPerThreadgroup.z==1,"SG2 baseline source requires original native M32 graph");
      d.pipelineName=gate ? "prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2" :
          "prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2";
      d.threadsPerThreadgroup.x=64;++producers;
    }
    result.push_back(d);
  }
  require(producers==2 && result.size()==source.size(),"SG2 baseline producer inventory differs");
  return result;
}
'''


def replace(source, before, after, count=1):
    actual = source.count(before)
    if actual != count:
        raise RuntimeError(f'combined oracle source drift: {before!r}: {actual} != {count}')
    return source.replace(before, after)


def generate(destination: Path):
    parent_path = Path(__file__).resolve().parents[1] / 'generate_oracle.py'
    spec = importlib.util.spec_from_file_location('combined_adaptive_parent_oracle', parent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination.mkdir(parents=True, exist_ok=True)
    module.generate(destination)
    source = (destination / 'oracle.mm').read_text()
    source = replace(source,
        'constexpr std::array<TailVariant,2> tailVariants{{{"m16-tail",16},{"m8-tail",8}}};',
        'constexpr std::array<TailVariant,1> tailVariants{{{"sg2-k128-m16-tail",16}}};')
    source = replace(source,
        'require(tailRows==0 || tailRows==16 || tailRows==8,"private tail selector differs");',
        'require(tailRows==0 || tailRows==16,"private SG2/K128 tail selector differs");')
    source = source.replace('adaptive_expert_tail_sep21_', 'adaptive_expert_tail_sg2k128_sep21_')
    source = replace(source,
        'std::vector<ComputeDispatch> tailCommands(std::span<const ComputeDispatch> source,',
        FIXED_COMMANDS + '\nstd::vector<ComputeDispatch> tailCommands(std::span<const ComputeDispatch> source,')
    # Change only private replacement matching, leaving fixedCommands native
    # entry validation above untouched.
    begin = source.index('std::vector<ComputeDispatch> tailCommands(')
    end = source.index('void matchingNativeGraphs(', begin)
    adapter = source[begin:end]
    adapter = replace(adapter, '"flash_int8_expert_store_gate_up_m32_n64"',
                      '"prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2"')
    adapter = replace(adapter, '"flash_int8_expert_store_down_scatter_m32_n64"',
                      '"prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2"')
    adapter = replace(adapter, 'd.threadsPerThreadgroup.x==128', 'd.threadsPerThreadgroup.x==64')
    adapter = replace(adapter, 'untouched M32 parameters/grid/128 threads',
                      'untouched original M32 parameters/grid and shared SG2/64 threads')
    source = source[:begin] + adapter + source[end:]
    # Every baseline submission, hash, timing and report uses variant7.
    begin = source.index('bool runCase(')
    end = source.index('} // namespace\n\nint main(', begin)
    case = source[begin:end]
    case = replace(case,
        '  matchingNativeGraphs(graphs[0].dispatches(),graphs[1].dispatches());',
        '  matchingNativeGraphs(graphs[0].dispatches(),graphs[1].dispatches());\n'
        '  const std::array<std::vector<ComputeDispatch>,2> fixedGraphs{\n'
        '      fixedCommands(graphs[0].dispatches(),rows),fixedCommands(graphs[1].dispatches(),rows)};\n'
        '  matchingNativeGraphs(fixedGraphs[0],fixedGraphs[1]);')
    insertion_end = case.index('  const uint64_t routes=')
    case = case[:insertion_end] + case[insertion_end:].replace('graphs[0].dispatches()', 'fixedGraphs[0]')\
        .replace('graphs[1].dispatches()', 'fixedGraphs[1]')
    case = replace(case, '"ADAPTIVE_EXPERT_TAIL_SEP21_VARIANT"',
                   '"ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21_VARIANT"', 2)
    case = replace(case, 'envNumber("ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21_VARIANT",1,2)',
                   'envNumber("ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21_VARIANT",1,1)')
    case = case.replace('Native baseline and copied M32 probe', 'Existing SG2/static-K128 variant7 and copied M32 probe')
    case = case.replace('native M32', 'existing variant7 SG2/K128 M32')
    case = case.replace('native baseline', 'existing variant7 baseline')
    case = case.replace('matched production control', 'matched existing variant7 control')
    case = replace(case,
        r'      <<",\"same_original_m32_jobs_params_grids_threads\":true,\"producer_threads\":128"',
        r'      <<",\"original_m32_jobs_params_grids_preserved\":true,\"same_control_candidate_threads\":true,\"producer_threads\":64,\"producer_simdgroups\":2,\"fixed_k\":128,\"static_full_tiles\":true"')
    case = case.replace('copied_m32_probe_qualified_against_native_control',
                        'copied_m32_probe_qualified_against_existing_variant7_control')
    source = source[:begin] + case + source[end:]
    source = source.replace('adaptive-expert-tail-sep21-one-layer-v1',
                            'adaptive-expert-tail-sg2k128-sep21-one-layer-v1')
    source = source.replace('adaptive-expert-tail-oracle', 'adaptive-expert-tail-sg2k128-oracle')
    source = source.replace('adaptive_tail_gpu_parity', 'adaptive_sg2k128_tail_gpu_parity')
    source = replace(source,
        r'\"uniform_effective_matrix_rows_m32_m16_tail_m8_tail\":[32768,24576,20480]',
        r'\"uniform_effective_matrix_rows_m32_m16_tail\":[32768,24576],\"producer_threads\":64,\"producer_simdgroups\":2,\"fixed_k\":128')
    source = source.replace('complete native M32 control and private adaptive original-M32-job chains;',
        'complete existing variant7 SG2/static-K128 M32 control and private SG2/static-K128 adaptive original-M32-job chains;')
    (destination / 'oracle.mm').write_text(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    generate(parser.parse_args().destination)
