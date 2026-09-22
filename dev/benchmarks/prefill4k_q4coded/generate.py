from pathlib import Path
import argparse

def replace(text,before,after,count=1):
    if text.count(before) !=count:raise RuntimeError(f'Q4coded oracle source drift: {before!r}')
    return text.replace(before,after)

def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    shader=Path('dev/benchmarks/prefill4k_q4coded/candidate.metal').read_text()
    (destination /'audit.metal').write_text('#define Q4CODED_AUDIT 1\n' +shader.replace('prefill4k_q4coded_','prefill4k_q4audit_'))
    text=Path('build/prefill4k-int8tiles/oracle.mm').read_text()
    text=replace(text,'#include "dev/benchmarks/prefill4k_int8tiles/bridge.hpp"','#include "dev/benchmarks/prefill4k_q4coded/bridge.hpp"\n#include "dev/benchmarks/prefill4k_q4coded/precision.hpp"')
    begin=text.index('  const uint32_t candidateTile = envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=text.index('  const auto healthy = [&] {',begin)
    block=r'''  const bool all =std::getenv("PREFILL4K_Q4CODED_ALL") !=nullptr;
  const auto rounded =[](uint64_t n) { return (n +16383) &~uint64_t(16383); };
  uint64_t planned =0;
  for (uint32_t width :{40u,10u,640u,640u,2560u}) planned +=rounded(uint64_t(rows) *10 *width *4 +64);
  auto admission =governor.tryReserve(planned);require(bool(admission),"Q4coded sums/audit buffers denied before allocation");
  prefill4k_q4coded::Commands hitPlan(backend,graphs[1].dispatches(),rows,all);admission->commit();
  std::array<std::vector<ComputeDispatch>,2> commands;
  commands[0].assign(graphs[0].dispatches().begin(),graphs[0].dispatches().end());commands[1]=hitPlan.commands;
  const auto checkHitJobs =[&] { require(hitPlan.canaries(),"Q4coded sums/audit canary changed"); };
'''
    text=text[:begin]+block+text[end:]
    text=text.replace('prefill4k_int8tiles::HitCommands hitPlan;','prefill4k_q4coded::Commands hitPlan;')
    # The alternative intentionally changes both coefficients and association.
    text=replace(text,'  require(!activation.mismatches && !downComparison.mismatches && !combined.mismatches,\n      "all-miss chain must be bit-exact to current Q4x8/DirectA control");','  require(std::isfinite(activation.relativeL2) && std::isfinite(downComparison.relativeL2) && std::isfinite(combined.relativeL2),"Q4coded alternative nonfinite comparison");')
    text=replace(text,'  const uint64_t misses = exactMissSlices(scratch, packed, ids, hot);','  const uint64_t misses =std::count_if(ids.begin(),ids.end(),[&](int64_t id) { return !std::binary_search(hot.begin(),hot.end(),uint32_t(id)); });')
    text=replace(text,'  require(exactMissSlices(scratch, packed, ids, hot) == misses, "timed miss ownership changed");','  // Changed Q4 miss arithmetic is a numerical alternative, not byte parity.')
    begin=text.index('  require(compare(scratch[0].packedActivated,scratch[1].packedActivated,routes *640,maxL2,minCosine).mismatches ==0')
    end=text.index('  uint32_t hitJobs = 0',begin)
    audit=r'''  const std::array<std::string,3> fastHashes{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
  for (uint32_t i=2;i <5;++i) std::fill_n(static_cast<float *>(hitPlan.values[i].contents()),hitPlan.values[i].sizeBytes() /4,std::numeric_limits<float>::quiet_NaN());
  std::vector<ComputeDispatch> auditCommands =hitPlan.commands;
  for (auto &d :auditCommands) if (d.pipelineName.starts_with("prefill4k_q4coded_"))
    d.pipelineName.replace(0,std::strlen("prefill4k_q4coded_"),"prefill4k_q4audit_");
  (void)backend.submitCommand(auditCommands);healthy();checkHitJobs();
  require(std::array<std::string,3>{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])} ==fastHashes,
      "Q4coded audit outputs differ from timed fast path");
  uint64_t certified =0;double maximumAffineRatio =0,maximumSourceDifference =0;
  const auto certifyPlane =[&](uint32_t plane) {
    const auto &p=weights.projection(prefixFor(layer) +(plane ==0 ? ".gate_proj" :plane ==1 ? ".up_proj" :".down_proj"));
    const auto *x=static_cast<const uint16_t *>(plane ==2 ? scratch[1].packedActivated.contents() :scratch[1].buckets.packedInputs.contents());
    const auto *actual=static_cast<const float *>(hitPlan.values[plane +2].contents());
    uint32_t expertsChecked=0;
    for (uint32_t expert=0;expert <512 && expertsChecked <16;++expert) {
      if (packed.offsets[expert] ==packed.offsets[expert +1] || (!all && std::binary_search(hot.begin(),hot.end(),expert))) continue;
      ++expertsChecked;
      for (uint32_t row :{packed.offsets[expert],packed.offsets[expert +1] -1})
        for (uint32_t n :{0u,1u,31u,63u,64u,p.outputSize -1}) {
          const auto certificate=prefill4k_q4coded::certificate(p,expert,n,x +uint64_t(row) *p.inputSize);
          const double got=actual[uint64_t(row) *p.outputSize +n];
          require(std::isfinite(got),"Q4coded linear projection audit is nonfinite or producer absent");
          const double error=std::abs(got -certificate.affine),sourceError=std::abs(got -certificate.sourceBF16);
          require(error <=certificate.candidateEnvelope && sourceError <=certificate.candidateEnvelope +certificate.coefficientEnvelope,
              "Q4coded sampled FP64 affine/BF16 coefficient absolute envelope failed");
          if (certificate.candidateEnvelope >0) maximumAffineRatio=std::max(maximumAffineRatio,error /certificate.candidateEnvelope);
          maximumSourceDifference=std::max(maximumSourceDifference,sourceError);++certified;
        }
    }
  };
  certifyPlane(0);certifyPlane(1);certifyPlane(2);
  uint32_t changedExperts=0;
  for (uint32_t expert=0;expert <512;++expert)
    changedExperts +=uint32_t(packed.counts[expert] &&(all || !std::binary_search(hot.begin(),hot.end(),expert)));
  require(certified ==uint64_t(std::min(changedExperts,16u)) *2 *6 *3,"Q4coded audit projection sample cardinality differs");
  // All inputs/parameters are independently decoded in the certificate.
  // This bounded sample covers16 active experts/plane, first+last packed rows,
  // six output columns. Full fused activations remain separate observations.
  (void)compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);
'''
    text=text[:begin]+audit+text[end:]
    text=replace(text,'<< ",\\\"candidate_hit_tile\\\":" << candidateTile << ",\\\"candidate_hit_sg\\\":" << candidateSG << ",\\\"candidate_relaxed_precision\\\":" << (relaxed ? "true" :"false") << ",\\\"cancellation_fixture\\\":" << (edge ? "true" :"false")','<< ",\\\"q4coded_all_experts\\\":" << (all ? "true" :"false") << ",\\\"fp64_certified_projection_samples\\\":" << certified << ",\\\"maximum_affine_error_over_envelope\\\":" << maximumAffineRatio << ",\\\"maximum_difference_from_source_bf16_dot\\\":" << maximumSourceDifference << ",\\\"cancellation_fixture\\\":" << (edge ? "true" :"false")')
    text=text.replace('\\\"miss_activation_and_down_exact\\\":true','\\\"miss_activation_and_down_exact\\\":false')
    text=replace(text,'<< (misses == routes ? "true" : "null")','<< (misses ==routes ? "false" :"null")')
    text=text.replace('\\\"numerical_alternative\\\":false','\\\"numerical_alternative\\\":true').replace('prefill4k-current-int8-hit-only-tile-oracle-v1','prefill4k-packed-u4-group-affine-numerical-expert-oracle-v1')
    text=text.replace('uint64_t exactMissSlices(', '[[maybe_unused]] uint64_t exactMissSlices(')
    begin=text.index('  const uint32_t candidateTile =envNumber("PREFILL4K_INT8_HIT_TILE",64,128);')
    end=text.index('  functions.push_back(direct ? "flash_moe_direct_a_prepare_down"',begin)
    text=text[:begin]+r'''  for (const char *prefix :{"prefill4k_q4coded_","prefill4k_q4audit_"})
    for (const char *phase :{"gate_up_m32_n64","down_scatter_m32_n64","input_sums"}) functions.push_back(std::string(prefix) +phase);
'''+text[end:]
    old='const uint32_t requested =(name.starts_with("prefill4k_int8tiles_") || name.starts_with("prefill4k_int8relaxed_")) && name.find("_sg") !=std::string::npos ? candidateSG *32 : name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;'
    text=replace(text,old,'const uint32_t requested =name.find("q4coded_") !=std::string::npos || name.find("q4audit_") !=std::string::npos ? (name.ends_with("input_sums") ? 256 :name.find("down_scatter") !=std::string::npos ? 32 :128) :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;')
    (destination /'oracle.mm').write_text(text)

if __name__ =='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);generate(p.parse_args().destination)
