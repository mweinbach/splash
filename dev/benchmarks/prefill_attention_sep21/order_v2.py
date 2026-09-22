#!/usr/bin/env python3
"""Make a fresh balanced timing oracle while preserving the v1 GPU binary.

CPU-only source/report/artifact reads. Tensor payloads and GPU work are excluded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from prepare import ROOT,once


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def transform(text:str)->str:
    text=once(text,
        '      const uint32_t repeats=single("PREFILL4K_BULK_REPEATS",5,100);require(repeats,"Bulk repeats must be positive");',
        '      const uint32_t repeats=single("PREFILL4K_BULK_REPEATS",10,100);\n'
        '      require(repeats>=4 && repeats%2==0,"Balanced timing requires an even4..100 AB/BA pair count");')
    text=once(text,
        '''      Times oldTimes,newTimes;
      for (uint32_t i=0;i<repeats+2;++i) {
        clearState(a);clearState(b);CommandTiming x,y;
        if (i%2) {y=backend.submitCommand(test.dispatches());x=backend.submitCommand(exactControl.dispatches());}
        else {x=backend.submitCommand(exactControl.dispatches());y=backend.submitCommand(test.dispatches());}
        stateEqual(a,b);control.output.check(true);candidate.output.check(true);
        controlStats.check(false);controlNums.check(false);candidateStats.check(false);candidateNums.check(false);
        if(i>=2){oldTimes.add(x);newTimes.add(y);}
      }''',
        '''      // Host writes and checks belong outside the timed command train.
      // Clear once, warm both variants in two ABBA blocks (eight GPU commands),
      // then time balanced AB/BA pairs without any CPU tensor access between
      // calls. Every fresh-begin0 graph overwrites its complete active prefix.
      clearState(a);clearState(b);
      for (uint32_t warm=0;warm<4;++warm) {
        if (warm%2) {
          (void)backend.submitCommand(test.dispatches());
          (void)backend.submitCommand(exactControl.dispatches());
        } else {
          (void)backend.submitCommand(exactControl.dispatches());
          (void)backend.submitCommand(test.dispatches());
        }
      }
      Times oldTimes,newTimes;
      for (uint32_t i=0;i<repeats;++i) {
        CommandTiming x,y;
        if (i%2) {y=backend.submitCommand(test.dispatches());x=backend.submitCommand(exactControl.dispatches());}
        else {x=backend.submitCommand(exactControl.dispatches());y=backend.submitCommand(test.dispatches());}
        oldTimes.add(x);newTimes.add(y);
      }
      // Only after every timed command completes may CPU validation read
      // mutable caches, partials, outputs or canaries again.
      stateEqual(a,b);control.output.check(true);candidate.output.check(true);
      controlStats.check(false);controlNums.check(false);candidateStats.check(false);candidateNums.check(false);
      require(compare(control.output.values(),ordinaryOutput).mismatches==0,"Timed exact SG8 output changed");
      require(compare(candidate.output.values(),deterministic).mismatches==0,"Timed BF16P output changed");
      std::vector<double> aFirst,aSecond,bFirst,bSecond,firstOrderRatios,secondOrderRatios,abbaRatios;
      double firstLog=0,secondLog=0,abbaLog=0,wallABBAlog=0;
      for (uint32_t pair=0;pair<repeats;pair+=2) {
        const double first=oldTimes.gpu[pair]/newTimes.gpu[pair+1];
        const double second=oldTimes.gpu[pair+1]/newTimes.gpu[pair];
        const double block=std::sqrt(first*second);
        firstOrderRatios.push_back(first);secondOrderRatios.push_back(second);abbaRatios.push_back(block);
        aFirst.push_back(oldTimes.gpu[pair]);bSecond.push_back(newTimes.gpu[pair]);
        bFirst.push_back(newTimes.gpu[pair+1]);aSecond.push_back(oldTimes.gpu[pair+1]);
        firstLog+=std::log(first);secondLog+=std::log(second);abbaLog+=std::log(block);
        wallABBAlog+=.5*std::log((oldTimes.wall[pair]/newTimes.wall[pair+1])*
            (oldTimes.wall[pair+1]/newTimes.wall[pair]));
      }
      const double blocks=repeats/2;
      const double firstOrderSpeedup=std::exp(firstLog/blocks),secondOrderSpeedup=std::exp(secondLog/blocks);
      const double balancedSpeedup=std::exp(abbaLog/blocks),balancedWallSpeedup=std::exp(wallABBAlog/blocks);''')
    text=once(text,'      report << ",\\\"control\\\":\\\"exact_bulk_sg8_f32P\\\",\\\"model_quality_qualified\\\":false,\\\"paired_order\\\":\\\"AB/BA\\\",\\\"exact_control_dispatches\\\":"<<exactControl.dispatches().size()',
        '      report << ",\\\"control\\\":\\\"exact_bulk_sg8_f32P\\\",\\\"model_quality_qualified\\\":false,\\\"paired_order\\\":\\\"balanced_ABBA\\\",\\\"exact_control_dispatches\\\":"<<exactControl.dispatches().size()')
    text=once(text,'      report<<"]}";',
        '''      report<<"],\\\"timing_protocol\\\":{\\\"version\\\":2,\\\"balanced_pair_count\\\":"<<repeats
            <<",\\\"AB_pairs\\\":"<<repeats/2<<",\\\"BA_pairs\\\":"<<repeats/2
            <<",\\\"warmup_GPU_commands\\\":8,\\\"host_state_clears_before_warmups\\\":1,\\\"CPU_tensor_reads_writes_between_timed_calls\\\":0,"
            <<"\\\"validation_after_all_timed_calls\\\":true,\\\"decision_metric\\\":\\\"balanced_geometric_ABBA_order_strata\\\","
            <<"\\\"v1_odd_pair_pooled_median_order_bias\\\":true},\\\"order_stratified\\\":{"
            <<"\\\"baseline_first_median_gpu_seconds\\\":"<<median(aFirst)<<",\\\"candidate_first_median_gpu_seconds\\\":"<<median(bFirst)
            <<",\\\"baseline_second_median_gpu_seconds\\\":"<<median(aSecond)<<",\\\"candidate_second_median_gpu_seconds\\\":"<<median(bSecond)
            <<",\\\"first_vs_first_geometric_speedup\\\":"<<firstOrderSpeedup
            <<",\\\"second_vs_second_geometric_speedup\\\":"<<secondOrderSpeedup
            <<",\\\"baseline_first_over_second_median_penalty\\\":"<<median(aFirst)/median(aSecond)
            <<",\\\"candidate_first_over_second_median_penalty\\\":"<<median(bFirst)/median(bSecond)
            <<"},\\\"balanced_geometric_gpu_speedup\\\":"<<balancedSpeedup
            <<",\\\"balanced_geometric_wall_speedup\\\":"<<balancedWallSpeedup
            <<",\\\"above_20percent_order_consistent_win\\\":"
            <<((balancedSpeedup>1.2&&firstOrderSpeedup>1.2&&secondOrderSpeedup>1.2)?"true":"false")
            <<",\\\"ABBA_blocks\\\":[";
      for (uint32_t i=0;i<abbaRatios.size();++i) {
        if (i) report<<',';
        report<<"{\\\"first_order_speedup\\\":"<<firstOrderRatios[i]
              <<",\\\"second_order_speedup\\\":"<<secondOrderRatios[i]
              <<",\\\"geometric_gpu_speedup\\\":"<<abbaRatios[i]<<'}';
      }
      report<<"]}";''')
    return text


def cpu_statistics()->dict:
    checks=0
    for speedup in (.75,1,1.05,1.25,2):
        for first_penalty in (1,1.5,2,3):
            for common_drift in (.9,1,1.1):
                aFirst=10*first_penalty;bSecond=10/speedup
                bFirst=10/speedup*first_penalty*common_drift;aSecond=10*common_drift
                estimate=math.sqrt((aFirst/bFirst)*(aSecond/bSecond))
                if not math.isclose(estimate,speedup,rel_tol=1e-12):raise AssertionError('Balanced estimator biased by common order penalty/drift')
                checks+=1
    return {'estimator_synthetic_combinations_checked':checks,'multiplicative_order_penalty_cancelled':True,
        'common_pair_drift_cancelled':True,'decision_requires_both_order_strata_and_balanced_speedup_over_1point2':True}


def bias_report(raw:bytes)->dict:
    original=json.loads(raw);pairs=original['paired_samples'];first=[];second=[];block=[]
    for i in range(0,len(pairs)-1,2):
        x,y=pairs[i:i+2]
        first.append(x['baseline_gpu_seconds']/y['candidate_gpu_seconds'])
        second.append(y['baseline_gpu_seconds']/x['candidate_gpu_seconds'])
        block.append(math.sqrt(first[-1]*second[-1]))
    geometric=lambda values:math.exp(statistics.mean(math.log(value) for value in values))
    return {'schema':'splash-prefill-sg8-bf16P-v1-order-bias-audit','source_report_sha256':sha(raw),
        'source_report_execution_complete':original['execution_complete'],'source_output_error':original['output_error'],
        'reported_pooled_median_speedup':original['gpu_speedup'],'reported_speedup_is_order_biased':True,
        'AB_pairs':(len(pairs)+1)//2,'BA_pairs':len(pairs)//2,'balanced_complete_ABBA_blocks':len(block),
        'unmatched_final_AB_pair_excluded_from_stratified_estimate':len(pairs)%2==1,
        'first_vs_first_geometric_speedup':geometric(first),'second_vs_second_geometric_speedup':geometric(second),
        'balanced_complete_blocks_geometric_speedup':geometric(block),'model_overlay_justified':False,
        'cause_observed':'v1 CPU state clears/cache reads between pairs; first command slow whichever variant',
        'required_next_measurement':'GPU-only warm/timed train, balanced8/10pairs, matched order strata'}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill-attention-bf16p-sep21-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-attention-bf16p-sep21-v2')
    parser.add_argument('--v1-report',type=Path,default=ROOT/'build/release/flash/prefill-attention-sg8-bf16P-sep21-v1.json')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if ROOT/'build' not in output.parents or output.exists() or output==base:raise ValueError('Choose fresh separate v2 output')
    manifest_raw=(base/'source-manifest.json').read_bytes();parent=json.loads(manifest_raw)
    source=(base/'oracle.mm').read_bytes();candidate=(base/'candidate.metal').read_bytes()
    for name,data in [('oracle.mm',source),('candidate.metal',candidate)]:
        if sha(data)!=parent['generated_hashes'][name]:raise ValueError(f'Sealed v1 source drift:{name}')
    files={'candidate.metal':candidate,'oracle.mm':transform(source.decode()).encode()}
    for name in ('candidate.air','candidate.metallib'):
        data=(base/name).read_bytes()
        if sha(data)!=parent['compiled_hashes'][name]:raise ValueError(f'Sealed GPU binary drift:{name}')
        files[name]=data
    bias=bias_report(args.v1_report.read_bytes());checks=cpu_statistics()
    manifest={'schema':'splash-prefill-sg8-bf16P-v2-balanced-source','parent_manifest_sha256':sha(manifest_raw),
        'parent_build':str(base),'GPU_shader_source_and_binaries_preserved_exact':True,
        'generated_hashes':{name:sha(data) for name,data in files.items()},'gpu_executed':False,'payload_bytes_read':0,
        'numerical_alternative':True,'model_quality_qualified':False,'timing_pair_default':10,
        'timing_pair_requires_even_4_to_100':True,'no_CPU_tensor_access_after_warmups_until_timing_complete':True,**checks}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prefill-bf16P-order-v2-',dir=output.parent) as temporary:
        staged=Path(temporary)/'output';staged.mkdir()
        for name,data in files.items():(staged/name).write_bytes(data)
        (staged/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (staged/'v1-order-bias-audit.json').write_text(json.dumps(bias,indent=2)+'\n')
        staged.rename(output)
    print(json.dumps({'prepared':str(output),'v1_order_bias':bias,**manifest}))


if __name__=='__main__':main()
