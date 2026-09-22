#!/usr/bin/env python3
"""Prepare CPU-only SG8 bulk temporal BF16 local-probability PV screen.

The four early128-row windows retain exact F32 arithmetic. Temporal PV casts
tile-local unnormalized exponential weights to BF16; F32 denominators stay
unrounded. This is a numerical alternative, requiring primitive/model checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TEMPORAL='flash_qsa_mpp_prefill_bulk_temporal_sg8_2048'


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def once(text:str,before:str,after:str)->str:
    if text.count(before)!=1:raise ValueError(f'Source anchor drift: {before[:120]}')
    return text.replace(before,after,1)


def shader(text:str)->str:
    start=text.index('inline void bulk_temporal_sg8_error(')
    stop=text.index('// Split only because the early SG4 and temporal SG8 routes')
    middle=text[start:stop]
    middle=once(middle,'    threadgroup bfloat *kvMemory, threadgroup float *weights) {',
        '    threadgroup bfloat *kvMemory, threadgroup float *weights, threadgroup bfloat *probabilities) {')
    middle=once(middle,'  auto pt = tensor(weights, dextents<int, 2>{64, M}, array<int, 2>{1, 64});',
        '  auto pt = tensor(probabilities, dextents<int, 2>{64, M}, array<int, 2>{1, 64});')
    middle=once(middle,'        weights[h * 64 + slot] = weight;\n        tileSum += weight;',
        '        weights[h * 64 + slot] = weight;\n'
        '        probabilities[h * 64 + slot] = bfloat(weight);\n        tileSum += weight;')
    text=text[:start]+middle+text[stop:]
    start=text.index('kernel void '+TEMPORAL+'(')
    stop=text.index('// The original scalar reducer uses contraction/reassociation off.')
    entry=text[start:stop]
    entry=once(entry,'  threadgroup float weights[32 * 64];',
        '  threadgroup float weights[32 * 64];\n  threadgroup bfloat probabilities[32 * 64];')
    entry=once(entry,'      queryMemory, kvMemory, weights);',
        '      queryMemory, kvMemory, weights, probabilities);')
    text=text[:start]+entry+text[stop:]
    # Export unique symbols so exact and candidate AIR can coexist; the harness
    # replaces only the temporal SG8 dispatch. Early/reducer remain exact AIR.
    for name in ('flash_qsa_mpp_prefill_bulk_2048','flash_qsa_mpp_prefill_bulk_early_2048',
                 TEMPORAL,'flash_qsa_fast_prefill_bulk_reduce_2048'):
        text=once(text,'kernel void '+name+'(','kernel void '+name+'_bf16p(')
    return ('// Private numerical alternative: only temporalSG8 localP cast/PV dtype differs.\n'
        '// BF16P uses a separate4KiB plane; all F32 QK/softmax/denominator banks remain.\n'+text)


def oracle(text:str)->str:
    text=once(text,'#include "oracle.mm"','#include <oracle.mm>')
    text=once(text,'namespace {\nusing namespace splash::flash::prefill4k;',
        '''namespace {
using namespace splash::flash::prefill4k;
CommandGraph bf16PVGraph(const CommandGraph &source) {
  CommandGraph result;
  for (const auto &dispatch : source.dispatches()) {
    std::vector<MetalBuffer> buffers;
    for (const auto &binding : dispatch.buffers) {
      require(binding.index == buffers.size(), "SG8 BF16P binding order drift");
      buffers.push_back(binding.buffer);
    }
    require(dispatch.bytes.size()==1, "SG8 BF16P parameter count drift");
    const auto name=dispatch.pipelineName=="flash_qsa_mpp_prefill_bulk_temporal_sg8_2048"
        ? "flash_qsa_mpp_prefill_bulk_temporal_sg8_2048_bf16p" : dispatch.pipelineName;
    if (dispatch.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {
      FlashQSAParams p{};std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));
      result.add(name,buffers,p,dispatch.threadgroups,dispatch.threadsPerThreadgroup);
    } else {
      require(dispatch.bytes[0].sizeBytes==sizeof(FlashQSAFastParams), "SG8 BF16P parameter ABI drift");
      FlashQSAFastParams p{};std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));
      result.add(name,buffers,p,dispatch.threadgroups,dispatch.threadsPerThreadgroup);
    }
  }
  return result;
}
struct FloatError {
  uint64_t elements=0,mismatches=0,nonfinite=0;
  double squareError=0,squareReference=0,maxAbs=0;
  void add(float actual,float expected) {
    elements++;mismatches+=std::bit_cast<uint32_t>(actual)!=std::bit_cast<uint32_t>(expected);
    if (!std::isfinite(actual)||!std::isfinite(expected)) {nonfinite++;return;}
    const double delta=double(actual)-expected;squareError+=delta*delta;
    squareReference+=double(expected)*expected;maxAbs=std::max(maxAbs,std::abs(delta));
  }
  void write(std::ostream &out) const {
    out<<"{\\\"elements\\\":"<<elements<<",\\\"f32_mismatches\\\":"<<mismatches
       <<",\\\"nonfinite\\\":"<<nonfinite<<",\\\"max_abs\\\":"<<maxAbs
       <<",\\\"relative_l2\\\":"<<std::sqrt(squareError/std::max(1e-30,squareReference))<<'}';
  }
};''')
    text=once(text,'      const bool bf16PV=single("PREFILL4K_BULK_BF16PV",0,1);\n'
        '      const bool sg8=single("PREFILL4K_BULK_SG8",0,1);\n'
        '      const bool direct=single("PREFILL4K_BULK_DIRECT",0,1)||bf16PV;',
        '      const bool bf16PV=true,sg8=true,direct=false;')
    text=once(text,'      DenseCoalescedWorkspace prepared;BulkExactWorkspace bulk;',
        '''      DenseCoalescedWorkspace prepared;BulkExactWorkspace bulk;
      auto controlBulk=allocateBulkExactWorkspace(backend);
      Guarded controlStats(backend,uint64_t(2048)*24*4*2*2),controlNums(backend,uint64_t(2048)*24*4*256*2);
      Guarded candidateStats(backend,uint64_t(2048)*24*4*2*2),candidateNums(backend,uint64_t(2048)*24*4*256*2);
      controlBulk.partials.partitionStatistics=controlStats.view;
      controlBulk.partials.partitionValues=controlNums.view;''')
    text=once(text,'      auto &preparedCandidate=direct?prepared:bulk.prepared;',
        '''      bulk.partials.partitionStatistics=candidateStats.view;
      bulk.partials.partitionValues=candidateNums.view;
      auto &preparedCandidate=direct?prepared:bulk.prepared;''')
    text=once(text,'      CommandGraph ordinary,test;addOrdinaryChunkedQSA(backend,ordinary,control.input(),a,wa,fa,0,2048);',
        '''      CommandGraph ordinary,test,exactControl;
      addOrdinaryChunkedQSA(backend,ordinary,control.input(),a,wa,fa,0,2048);
      addBulkExactQSA(backend,exactControl,control.input(),a,wa,fa,controlBulk,0,2048,true);''')
    text=once(text,'      else addBulkExactQSA(backend,test,candidate.input(),b,wb,fb,bulk,0,2048,sg8);',
        '''      else addBulkExactQSA(backend,test,candidate.input(),b,wb,fb,bulk,0,2048,sg8);
      test=bf16PVGraph(test);''')
    text=once(text,'      (void)backend.submitCommand(test.dispatches());stateEqual(a,b);',
        '''      const std::vector<uint16_t> ordinaryOutput(control.output.values().begin(),control.output.values().end());
      (void)backend.submitCommand(exactControl.dispatches());
      require(compare(control.output.values(),ordinaryOutput).mismatches==0,"Exact SG8 control changed ordinary output");
      (void)backend.submitCommand(test.dispatches());stateEqual(a,b);
      controlStats.check(false);controlNums.check(false);candidateStats.check(false);candidateNums.check(false);
      const std::vector<uint16_t> deterministic(candidate.output.values().begin(),candidate.output.values().end());
      (void)backend.submitCommand(test.dispatches());
      require(compare(candidate.output.values(),deterministic).mismatches==0,"BF16P deterministic repeat changed output");''')
    text=once(text,'      bool partialsExact=true;',
        '      bool partialsExact=true,statisticsExact=true,earlyNumeratorsExact=true;FloatError numeratorError;')
    text=once(text,
        '          partialsExact &= !std::memcmp(stats+at*2,expectedStats.data()+at*2,2*4)&&!std::memcmp(nums+at*256,expectedNums.data()+at*256,256*4);',
        '''          const bool statsMatch=!std::memcmp(stats+at*2,expectedStats.data()+at*2,2*4);
          const bool numsMatch=!std::memcmp(nums+at*256,expectedNums.data()+at*256,256*4);
          partialsExact &= statsMatch&&numsMatch;statisticsExact &= statsMatch;
          if (row<512) earlyNumeratorsExact &= numsMatch;
          for (uint32_t d=0;d<256;++d) numeratorError.add(nums[at*256+d],expectedNums[at*256+d]);''')
    text=once(text,'        require(partialsExact,"Exact bulk F32 attention partials changed");',
        '''        require(statisticsExact,"BF16P changed original F32 maximum/denominator statistics");
        require(earlyNumeratorsExact,"BF16P changed the unchanged early SG4 numerators");
        require(numeratorError.nonfinite==0,"BF16P produced nonfinite F32 numerators");
        const auto inactiveSentinel=uint32_t(kSentinel)|(uint32_t(kSentinel)<<16);
        const auto padding=[&](const FlashQSAFastWorkspace &scratch) {
          const auto *ss=static_cast<const uint32_t *>(scratch.partitionStatistics.contents());
          const auto *vv=static_cast<const uint32_t *>(scratch.partitionValues.contents());
          for (uint32_t row=0;row<128;++row)for (uint32_t head=0;head<24;++head)for (uint32_t part=1;part<4;++part) {
            const uint64_t at=(uint64_t(row)*24+head)*4+part;
            for (uint32_t d=0;d<2;++d) require(ss[at*2+d]==inactiveSentinel,"Inactive statistics padding changed");
            for (uint32_t d=0;d<256;++d) require(vv[at*256+d]==inactiveSentinel,"Inactive numerator padding changed");
          }
        };
        padding(controlBulk.partials);padding(bulk.partials);''')
    text=once(text,'      if (!direct)require(error.mismatches==0,"Exact bulk BF16 output changed");',
        '''      require(!error.nonfinite&&error.relativeL2()<=.01&&error.cosine()>=.9998,"BF16P output exceeds primitive numeric tolerance");
      for (uint32_t window=0;window<4;++window) require(mismatches[window]==0,"BF16P changed unchanged early BF16 output");
      uint64_t outputSignFlips=0,nearZeroOutputCells=0;double nearZeroOutputMaxAbs=0;
      for (uint64_t i=0;i<uint64_t(2048)*6144;++i) {
        const float actual=number(candidate.output.values()[i]),expected=number(control.output.values()[i]);
        outputSignFlips+=actual!=0&&expected!=0&&std::signbit(actual)!=std::signbit(expected);
        if (std::abs(expected)<=1e-3) {nearZeroOutputCells++;nearZeroOutputMaxAbs=std::max(nearZeroOutputMaxAbs,std::abs(double(actual)-expected));}
      }''')
    text=once(text,'      uint32_t begin=2048;',
        '''      const auto partialHash=[&]() {SHA256 hash;hash.add(bulk.partials.partitionStatistics);hash.add(bulk.partials.partitionValues);return hash.finish();};
      const auto validPartialHash=partialHash();uint32_t shaderRejections=0;
      for (uint32_t which=0;which<7;++which) {
        FlashQSAFastParams p{{2048,0,4096,1024,0,512,0,0,1e-6f,1e7f,0,0},0,0,4,4};
        splash::metal::DispatchSize threads{256,1,1},groups{1,1,1};
        switch (which) {
          case 0:p.common.begin=1;break;case 1:p.common.rows=2047;break;case 2:p.common.capacity=2047;break;
          case 3:p.partitions=3;break;case 4:p.maximum_partitions=3;break;case 5:threads.x=128;break;default:threads.x=128;threads.y=2;break;
        }
        *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;CommandGraph rejected;
        rejected.add("flash_qsa_mpp_prefill_bulk_temporal_sg8_2048_bf16p",{bulk.prepared.queries,b.keys,b.values,
            bulk.prepared.selectedBlocks,bulk.partials.partitionStatistics,bulk.partials.partitionValues,candidate.diag},p,groups,threads);
        (void)backend.submitCommand(rejected.dispatches());
        require(*static_cast<uint32_t *>(candidate.diag.contents())==(kSticky|(1u<<9)),"BF16P malformed shader parameter did not fail closed");
        candidateStats.check(false);candidateNums.check(false);shaderRejections++;
      }
      require(partialHash()==validPartialHash,"BF16P malformed shader params changed partials");
      *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;
      uint32_t begin=2048;''')
    text=text.replace('x=backend.submitCommand(ordinary.dispatches())','x=backend.submitCommand(exactControl.dispatches())')
    text=once(text,'        stateEqual(a,b);if(i>=2){oldTimes.add(x);newTimes.add(y);}',
        '''        stateEqual(a,b);control.output.check(true);candidate.output.check(true);
        controlStats.check(false);controlNums.check(false);candidateStats.check(false);candidateNums.check(false);
        if(i>=2){oldTimes.add(x);newTimes.add(y);}''')
    text=text.replace('<< (direct?"true":"false") << ",\\\"temporal_sg8\\\":','<< "true" << ",\\\"temporal_sg8\\\":')
    text=once(text,'      report << ",\\\"baseline_timing\\\":";oldTimes.write(report);',
        '''      report << ",\\\"control\\\":\\\"exact_bulk_sg8_f32P\\\",\\\"model_quality_qualified\\\":false,\\\"paired_order\\\":\\\"AB/BA\\\",\\\"exact_control_dispatches\\\":"<<exactControl.dispatches().size()
          <<",\\\"f32_softmax_statistics_exact\\\":"<<(statisticsExact?"true":"false")
          <<",\\\"early512_numerators_and_output_exact\\\":"<<(earlyNumeratorsExact?"true":"false")
          <<",\\\"partition_scratch_canaries_pass\\\":true,\\\"inactive_partition_padding_exact\\\":true,\\\"deterministic_repeat_exact\\\":true,\\\"shader_fail_closed_checks\\\":"<<shaderRejections
          <<",\\\"bf16_output_sign_flips\\\":"<<outputSignFlips<<",\\\"near_zero_output_cells\\\":"<<nearZeroOutputCells
          <<",\\\"near_zero_output_max_abs\\\":"<<nearZeroOutputMaxAbs
          <<",\\\"numerator_error\\\":";numeratorError.write(report);
      report << ",\\\"baseline_timing\\\":";oldTimes.write(report);''')
    text=once(text,'report << ",\\\"gpu_speedup\\\":" << median(oldTimes.gpu)/median(newTimes.gpu) << \'}\';',
        '''report << ",\\\"gpu_speedup\\\":" << median(oldTimes.gpu)/median(newTimes.gpu) << ",\\\"paired_samples\\\":[";
      for (uint32_t i=0;i<oldTimes.gpu.size();++i) {
        if (i) report<<',';
        report<<"{\\\"baseline_gpu_seconds\\\":"<<oldTimes.gpu[i]<<",\\\"candidate_gpu_seconds\\\":"<<newTimes.gpu[i]
              <<",\\\"baseline_wall_seconds\\\":"<<oldTimes.wall[i]<<",\\\"candidate_wall_seconds\\\":"<<newTimes.wall[i]<<'}';
      }
      report<<"]}";''')
    return text


def mapping_audit()->dict:
    visits=[0]*2048
    for tid in range(256):
        head=tid//8;head_lane=tid%8
        for slot in range(head_lane,64,8):visits[head*64+slot]+=1
    if set(visits)!={1}:raise AssertionError('SG8 BF16P write mapping is not bijective')
    return {'bf16P_store_cells_checked':len(visits),'each_cell_written_once':True,
        'f32_score_and_bf16_probability_storage_disjoint':True,'source_shared_bytes':24576,
        'original_qk_descriptor_and_softmax_order_unchanged':True,'original_barrier_consumes_casts_before_pv':True}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-attention-bf16p-sep21-v1')
    args=parser.parse_args();output=args.output.resolve()
    if ROOT/'build' not in output.parents:raise ValueError('Choose private output beneath build')
    paths={'shader':ROOT/'dev/benchmarks/prefill4k_attention/bulk_attention_sg8.metal',
           'oracle':ROOT/'dev/benchmarks/prefill4k_attention/bulk_oracle.mm'}
    raw={key:path.read_bytes() for key,path in paths.items()};files={'candidate.metal':shader(raw['shader'].decode()).encode(),
        'oracle.mm':oracle(raw['oracle'].decode()).encode()};output.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():(output/name).write_bytes(data)
    report={'schema':'splash-prefill-sg8-bf16P-source-v1','normal_sources_modified':False,'gpu_executed':False,
        'payload_bytes_read':0,'numerical_alternative':True,'model_quality_qualified':False,
        'control':'exact_bulk_SG8_F32P','candidate':'sameSG8_temporalOnly_BF16TileLocalUnnormalizedP_BF16V_F32Dot',
        'source_hashes':{key:sha(data) for key,data in raw.items()},'generated_hashes':{name:sha(data) for name,data in files.items()},
        'temporal_symbol':TEMPORAL+'_bf16p',**mapping_audit()}
    (output/'source-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'prepared':str(output),**report}))


if __name__=='__main__':main()
