from pathlib import Path
import argparse

def replace(text,before,after,count=1):
    if text.count(before) !=count:raise RuntimeError(f'NAX oracle source drift: {before!r}')
    return text.replace(before,after)

def generate(destination):
    destination.mkdir(parents=True,exist_ok=True)
    shader=Path('dev/benchmarks/prefill4k_q4nax/candidate.metal').read_text()
    # Metal AIR emits the large templated helper bodies as weak symbols.
    # Distinct kernel entry points alone do not keep fast/audit bodies apart:
    # linking can otherwise coalesce the fast body and omit the audit stores.
    (destination /'audit.metal').write_text('#define Q4NAX_AUDIT 1\n' +shader.replace('q4nax_','q4naxaudit_'))
    text=Path('build/prefill4k-q4coded/oracle.mm').read_text()
    before='  std::array<std::vector<ComputeDispatch>,2> commands;\n  commands[0].assign(graphs[0].dispatches().begin(),graphs[0].dispatches().end());commands[1]=hitPlan.commands;'
    after=r'''  const bool relaxed=std::getenv("PREFILL4K_Q4NAX_RELAXED") !=nullptr;
  for (auto &d :hitPlan.commands) {
    if (d.pipelineName =="prefill4k_q4coded_gate_up_m32_n64") d.pipelineName=relaxed ? "prefill4k_q4nax_gate_up_relaxed_m32_n64" :"prefill4k_q4nax_gate_up_strict_m32_n64";
    if (d.pipelineName =="prefill4k_q4coded_down_scatter_m32_n64") { d.pipelineName=relaxed ? "prefill4k_q4nax_down_scatter_relaxed_m32_n64" :"prefill4k_q4nax_down_scatter_strict_m32_n64";d.threadsPerThreadgroup={128,1,1}; }
  }
  std::erase_if(hitPlan.commands,[](const ComputeDispatch &d) { return d.pipelineName =="prefill4k_q4coded_input_sums"; });
  std::array<std::vector<ComputeDispatch>,2> commands;
  commands[0].assign(graphs[0].dispatches().begin(),graphs[0].dispatches().end());commands[1]=hitPlan.commands;'''
    text=replace(text,before,after)
    text=replace(text,'for (auto &d :auditCommands) if (d.pipelineName.starts_with("prefill4k_q4coded_"))\n    d.pipelineName.replace(0,std::strlen("prefill4k_q4coded_"),"prefill4k_q4audit_");','for (auto &d :auditCommands) if (d.pipelineName.starts_with("prefill4k_q4nax_"))\n    d.pipelineName.replace(0,std::strlen("prefill4k_q4nax_"),"prefill4k_q4naxaudit_");')
    text=replace(text,'          require(error <=certificate.candidateEnvelope && sourceError <=certificate.candidateEnvelope +certificate.coefficientEnvelope,\n              "Q4coded sampled FP64 affine/BF16 coefficient absolute envelope failed");',r'''          const double sourceEnvelope=prefill4k_q4coded::addUp(prefill4k_q4coded::mulUp(prefill4k_q4coded::gammaUp(2 *p.inputSize,0x1p-24),certificate.sourceNorm),
              prefill4k_q4coded::addUp(certificate.diagnosticEnvelope,prefill4k_q4coded::underflowUp(2 *p.inputSize,0x1p-24,0x1p-150)));
          require(sourceError <=sourceEnvelope &&error <=prefill4k_q4coded::addUp(sourceEnvelope,certificate.coefficientEnvelope),
              "NAX sampled FP64 original-BF16 coefficient absolute envelope failed");''')
    text=replace(text,'<< ",\\\"q4coded_all_experts\\\":" << (all ? "true" :"false")','<< ",\\\"same_original_bf16_coefficients\\\":true,\\\"nax_relaxed_precision\\\":" << (relaxed ? "true" :"false") << ",\\\"changed_source_reduction_grouping\\\":true,\\\"q4coded_all_experts\\\":" << (all ? "true" :"false")')
    text=replace(text,'  for (const char *prefix :{"prefill4k_q4coded_","prefill4k_q4audit_"})\n    for (const char *phase :{"gate_up_m32_n64","down_scatter_m32_n64","input_sums"}) functions.push_back(std::string(prefix) +phase);',r'''  for (const char *prefix :{"prefill4k_q4nax_","prefill4k_q4naxaudit_"})
    for (const char *phase :{"gate_up_strict_m32_n64","gate_up_relaxed_m32_n64","down_scatter_strict_m32_n64","down_scatter_relaxed_m32_n64"}) functions.push_back(std::string(prefix) +phase);''')
    text=replace(text,'const uint32_t requested =name.find("q4coded_") !=std::string::npos || name.find("q4audit_") !=std::string::npos ? (name.ends_with("input_sums") ? 256 :name.find("down_scatter") !=std::string::npos ? 32 :128) :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;','const uint32_t requested =name.find("q4nax_") !=std::string::npos ||name.find("q4naxaudit_") !=std::string::npos ? 128 :name.ends_with(suffix) ? (m ==64 ? 256 :128) :256;')
    text=text.replace('prefill4k-packed-u4-group-affine-numerical-expert-oracle-v1','prefill4k-original-bf16-coefficients-register-nax-expert-oracle-v1')
    point='  const std::array<std::string,3> fastHashes{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};'
    beforeAudit=r'''  const auto diagnosticPath=std::getenv("PREFILL4K_Q4NAX_DIAGNOSTIC");
  const auto writeDiagnostic=[&](const char *status,uint32_t plane,uint32_t expert,uint32_t packedRow,uint32_t column,double got) {
    if (!diagnosticPath) return;
    std::ofstream debug(diagnosticPath);require(bool(debug),"cannot open fresh NAX diagnostic output");
    debug <<std::setprecision(17) <<"{\"status\":" <<splash::json::quote(status) <<",\"layer\":" <<layer
        <<",\"rows\":" <<rows <<",\"pattern\":" <<splash::json::quote(pattern) <<",\"plane\":" <<plane
        <<",\"expert\":" <<expert <<",\"packed_row\":" <<packedRow <<",\"column\":" <<column
        <<",\"got_finite\":" <<(std::isfinite(got) ? "true" :"false") <<",\"got\":";
    if (std::isfinite(got)) debug <<got;else debug <<"null";
    const uint32_t first=packed.offsets[expert],last=packed.offsets[expert +1];
    debug <<",\"expert_packed_begin\":" <<first <<",\"expert_packed_end\":" <<last
        <<",\"candidate_activation_bf16_mismatches\":" <<activation.mismatches <<",\"candidate_down_bf16_mismatches\":" <<downComparison.mismatches
        <<",\"candidate_combined_bf16_mismatches\":" <<combined.mismatches <<",\"activation_relative_l2\":" <<activation.relativeL2
        <<",\"down_relative_l2\":" <<downComparison.relativeL2 <<",\"combined_relative_l2\":" <<combined.relativeL2
        <<",\"control_gpu_ms\":";times(debug,timing[0],true);debug <<",\"candidate_gpu_ms\":";times(debug,timing[1],true);
    debug <<",\"control_wall_ms\":";times(debug,timing[0],false);debug <<",\"candidate_wall_ms\":";times(debug,timing[1],false);
    debug <<",\"sticky_diagnostics\":[" <<*static_cast<const uint32_t *>(diagnostic[0].contents()) <<',' <<*static_cast<const uint32_t *>(diagnostic[1].contents()) <<"]"
        <<",\"job\":{\"index\":";
    uint32_t jobIndex=0;
    for (;jobIndex <jobs.count;++jobIndex) if (jobs.entries[jobIndex].expert ==expert &&packedRow >=jobs.entries[jobIndex].rowBegin &&packedRow <jobs.entries[jobIndex].rowBegin +32) break;
    debug <<jobIndex <<",\"packed_start\":" <<(jobIndex <jobs.count ? jobs.entries[jobIndex].rowBegin :UINT32_MAX) <<"},\"linear_samples\":[";
    const auto *raw=static_cast<const float *>(hitPlan.values[plane +2].contents());
    const uint32_t width=plane ==2 ? 2560 :640;
    for (uint32_t n=0;n <std::min(width,80u);++n) { if (n) debug <<',';const double value=raw[uint64_t(packedRow) *width +n];if (std::isfinite(value)) debug <<value;else debug <<"null"; }
    debug <<"],\"input_bf16_words\":[";
    const auto *input=static_cast<const uint16_t *>(plane ==2 ? scratch[1].packedActivated.contents() :scratch[1].buckets.packedInputs.contents());
    const uint32_t k=plane ==2 ? 640 :2560;
    for (uint32_t j=0;j <64;++j) { if (j) debug <<',';debug <<input[uint64_t(packedRow) *k +j]; }
    debug <<"],\"candidate_pipeline_names\":";names(debug,hitPlan.commands);debug <<"}\n";debug.flush();
  };
  writeDiagnostic("timed_fast_before_audit",0,0,0,0,0);
'''
    text=replace(text,point,beforeAudit+point)
    text=replace(text,'          require(std::isfinite(got),"Q4coded linear projection audit is nonfinite or producer absent");',r'''          if (!std::isfinite(got)) {
            writeDiagnostic("nonfinite_or_absent_linear_projection",plane,expert,row,n,got);
            throw std::runtime_error("NAX linear audit failure plane=" +std::to_string(plane) +" expert=" +std::to_string(expert) +" packed_row=" +std::to_string(row) +" col=" +std::to_string(n));
          }''')
    text=text.replace('exact Q4 miss slices','Q4 miss tensor differences observed, exactness reported separately')
    # Capture whether every live full-chain output remains byte-identical.
    text=replace(text,'  const auto combined = compare(output[0], output[1], uint64_t{rows} * 2560, maxL2, minCosine);','  const auto combined =compare(output[0],output[1],uint64_t(rows) *2560,maxL2,minCosine);\n  const bool strictExact=!activation.mismatches && !downComparison.mismatches && !combined.mismatches;')
    text=replace(text,'\\\"canaries_clean\\\":true,\\\"activation\\\":";', '\\\"strict_full_bf16_exact\\\":" << (strictExact ? "true" :"false") << ",\\\"canaries_clean\\\":true,\\\"activation\\\":";')
    (destination /'oracle.mm').write_text(text)

if __name__ =='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);generate(p.parse_args().destination)
