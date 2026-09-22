#!/usr/bin/env python3
"""Generate a source-only bounded W8A8 numerical prefill oracle."""
from pathlib import Path
import argparse
import importlib.util


HELPERS = r'''
namespace w8= splash::flash::prefill_moe_w8a8;

std::vector<uint16_t> normalizedHidden(uint32_t rows,const NativeStoreView &store,
    std::span<const int64_t> ids) {
  std::vector<uint16_t> result(uint64_t{rows}*2560);
  for (uint32_t row=0;row<rows;++row) {
    std::array<float,2560> values{};
    for (uint32_t k=0;k<2560;++k)
      values[k]=float(int((uint64_t(row)*2560*73+k*73+row*17)%257)-128);
    if (!row) {
      // Two actual equal-magnitude I8 coefficients let opposite BF16 operands
      // cancel exactly even after normalization. A tiny third operand makes
      // the original F64 reference near zero while activation quantization
      // may erase that residual. The raw certificate explicitly samples n0.
      const uint32_t expert=uint32_t(ids[0]);
      const auto *codes=static_cast<const int8_t *>(store.layer.codes[0].contents())+
          uint64_t(expert)*640*2560;
      std::array<uint32_t,128> first;first.fill(UINT32_MAX);
      uint32_t k1=UINT32_MAX,k2=UINT32_MAX;
      for (uint32_t k=0;k<2560;++k) {
        const uint32_t magnitude=uint32_t(std::abs(int(codes[k])));
        if (!magnitude || magnitude>127) continue;
        if (first[magnitude]!=UINT32_MAX) {k1=first[magnitude];k2=k;break;}
        first[magnitude]=k;
      }
      require(k1!=UINT32_MAX && k2!=UINT32_MAX,"actual-code cancellation fixture has no coefficient pair");
      values.fill(0);values[k1]=float(codes[k2]);values[k2]=-float(codes[k1]);
      uint32_t residual=0;
      while (residual<2560 && (residual==k1 || residual==k2 || !codes[residual])) ++residual;
      require(residual<2560,"actual-code cancellation fixture has no residual coefficient");
      values[residual]=0x1p-16f;
    }
    double squares=0;
    for (float value:values) squares+=double(value)*value;
    const double rms=std::sqrt(squares/2560);
    require(rms>0 && std::isfinite(rms),"synthetic row normalization failed");
    for (uint32_t k=0;k<2560;++k) result[uint64_t(row)*2560+k]=bf16(float(double(values[k])/rms));
  }
  return result;
}

struct Certificate final {
  std::array<w8::precision::QuantizationReport,2> quantization;
  std::array<uint64_t,3> samples{},integerFailures{},scaledFailures{},envelopeFailures{};
  std::array<double,3> minimumOriginalAbs{
      std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
  std::array<double,3> maximumOriginalError{};
  std::vector<std::string> failures;
  std::string cancellation;
  bool pass() const {
    return quantization[0].pass() && quantization[1].pass() && failures.empty() &&
        std::all_of(samples.begin(),samples.end(),[](uint64_t count){return count>0;});
  }
  void write(std::ostream &out) const {
    out<<"{\"pass\":"<<(pass()?"true":"false")<<",\"all_active_and_padding_quantized_rows_checked\":true"
        <<",\"quantization\":[";quantization[0].write(out);out<<',';quantization[1].write(out);
    out<<"],\"sample_policy\":\"every output column of first/last packed rows in first/last nonempty experts; original control BF16 source including gup change in down envelope\""
        <<",\"projections\":[";
    for (uint32_t plane=0;plane<3;++plane) {
      if (plane) out<<',';
      out<<"{\"plane\":"<<plane<<",\"samples\":"<<samples[plane]
          <<",\"integer_dot_failures\":"<<integerFailures[plane]
          <<",\"scaled_f32_bit_failures\":"<<scaledFailures[plane]
          <<",\"original_f64_absolute_quantization_envelope_failures\":"<<envelopeFailures[plane]
          <<",\"minimum_abs_original_scaled_f64\":"<<finiteJSON(minimumOriginalAbs[plane])
          <<",\"maximum_abs_error_from_original_f64\":"<<finiteJSON(maximumOriginalError[plane])<<'}';
    }
    out<<"],\"actual_code_cancellation_sample\":"<<(cancellation.empty()?"null":cancellation)<<",\"first_failures\":[";
    for (uint32_t i=0;i<failures.size();++i) {if(i)out<<',';out<<splash::json::quote(failures[i]);}
    out<<"]}";
  }
};

Certificate certify(const NativeStoreView &store,const std::array<FlashMoEBlockedScratch,2> &scratch,
    const ref::Packed &packed,const w8::Workspace &workspace) {
  Certificate result;
  for (uint32_t activation=0;activation<2;++activation) {
    const auto &source=activation ? scratch[1].packedActivated : scratch[1].buckets.packedInputs;
    result.quantization[activation]=w8::precision::certifyQuantizedRows(
        static_cast<const uint16_t *>(source.contents()),
        static_cast<const int8_t *>(workspace.quantized[activation].contents()),
        static_cast<const float *>(workspace.activationScale[activation].contents()),
        workspace.paddedRoutes,activation ? 640 : 2560);
  }
  uint32_t first=0,last=511;
  while (first<512 && !packed.counts[first]) ++first;
  while (last>first && !packed.counts[last]) --last;
  require(first<512,"F64 certificate has no active expert");
  std::vector<std::pair<uint32_t,uint32_t>> sampleRows;
  for (uint32_t expert:{first,last}) {
    for (uint32_t row:{packed.offsets[expert],packed.offsets[expert+1]-1}) {
      const std::pair<uint32_t,uint32_t> sample{expert,row};
      if (std::find(sampleRows.begin(),sampleRows.end(),sample)==sampleRows.end()) sampleRows.push_back(sample);
    }
  }
  const auto *ranks=static_cast<const uint32_t *>(store.layer.ranks.contents());
  for (uint32_t plane=0;plane<3;++plane) {
    const uint32_t activation=plane==2 ? 1 : 0,width=plane==2 ? 640 : 2560,n=plane==2 ? 2560 : 640;
    const auto &original=plane==2 ? scratch[0].packedActivated : scratch[0].buckets.packedInputs;
    const auto *originalA=static_cast<const uint16_t *>(original.contents());
    const auto *quantized=static_cast<const int8_t *>(workspace.quantized[activation].contents());
    const auto *activationScales=static_cast<const float *>(workspace.activationScale[activation].contents());
    const auto *coefficients=static_cast<const int8_t *>(store.layer.codes[plane].contents());
    const auto *weightScales=static_cast<const float *>(store.layer.scales[plane].contents());
    const auto *integer=static_cast<const int32_t *>(workspace.integerAudit[plane].contents());
    const auto *scaled=static_cast<const float *>(workspace.scaledAudit[plane].contents());
    for (const auto &[expert,row]:sampleRows) {
      const uint32_t rank=ranks[expert];require(rank<512,"certificate coefficient rank differs");
      const uint32_t outputRow=plane==2 ? packed.routeMap[row] : row;
      for (uint32_t column=0;column<n;++column) {
        const auto certificate=w8::precision::certifyProjection(originalA+uint64_t(row)*width,
            quantized+uint64_t(row)*width,activationScales[row],
            coefficients+(uint64_t(rank)*n+column)*width,width,weightScales[uint64_t(rank)*n+column],
            integer[uint64_t(outputRow)*n+column],scaled[uint64_t(outputRow)*n+column]);
        ++result.samples[plane];result.integerFailures[plane]+=!certificate.integerExact;
        result.scaledFailures[plane]+=!certificate.scaledBitsExact;
        result.envelopeFailures[plane]+=!certificate.withinOriginalEnvelope;
        result.minimumOriginalAbs[plane]=std::min(result.minimumOriginalAbs[plane],std::abs(certificate.originalScaledF64));
        result.maximumOriginalError[plane]=std::max(result.maximumOriginalError[plane],certificate.rawScaledAbsoluteErrorFromOriginal);
        if (!certificate.pass() && result.failures.size()<16) {
          std::ostringstream detail;detail<<"plane="<<plane<<" expert="<<expert<<" packedrow="<<row<<" n="<<column<<' ';
          certificate.write(detail);result.failures.push_back(detail.str());
        }
        if (!plane && row==packed.offsets[first] && !column && result.cancellation.empty()) {
          std::ostringstream detail;certificate.write(detail);result.cancellation=detail.str();
        }
      }
    }
  }
  return result;
}

std::string quantizerEdges(MetalBackend &backend) {
  std::vector<Guard> guards;
  std::vector<uint16_t> source(3*2560,0);
  source[2560]=bf16(127);source[2561]=bf16(.5f);source[2562]=bf16(1.5f);
  source[2563]=bf16(-.5f);source[2564]=bf16(-1.5f);source[2565]=bf16(2.5f);
  source[2*2560]=bf16(127);source[2*2560+1]=0x7f80;source[2*2560+2]=0xff80;
  source[2*2560+3]=0x7fc0;source[2*2560+4]=0x8000;
  auto input=upload(backend,source,"W8A8 quantizer zero/tie/nonfinite unit fixture");
  auto quantized=guarded(backend,source.size(),guards),scales=guarded(backend,3*4,guards),diag=guarded(backend,4,guards);
  *static_cast<uint32_t *>(diag.contents())=kSticky;CommandGraph graph;
  graph.add("prefill_moe_sep21_w8a8_quantize_gate_t256",{input,quantized,scales,diag},
      w8::QuantParams{3,2560,0,0},{3,1,1},{256,1,1});
  (void)backend.submitCommand(graph.dispatches());
  const auto certificate=w8::precision::certifyQuantizedRows(source.data(),
      static_cast<const int8_t *>(quantized.contents()),static_cast<const float *>(scales.contents()),3,2560);
  const bool pass=certificate.pass() && *static_cast<const uint32_t *>(diag.contents())==(kSticky|4u) &&
      std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();}) &&
      !std::memcmp(source.data(),input.contents(),source.size()*2);
  std::ostringstream out;out<<"{\"pass\":"<<(pass?"true":"false")
      <<",\"expected_diagnostic\":"<<(kSticky|4u)<<",\"actual_diagnostic\":"<<*static_cast<const uint32_t *>(diag.contents())
      <<",\"scope\":\"untimed independent zero/power-of-two RNE tie/nonfinite/negative-zero quantizer fixture\",\"certificate\":";
  certificate.write(out);out<<'}';require(pass,"W8A8 independent quantizer edge GPU check failed");return out.str();
}
'''


def replace(source,before,after,count=1):
    if source.count(before)!=count:raise RuntimeError(f'W8A8 one-layer source drift: {before!r}')
    return source.replace(before,after)


def generate(destination):
    spec=importlib.util.spec_from_file_location('one_layer_generator',Path('dev/benchmarks/prefill_moe_sep21/one_layer/generate.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.generate(destination)
    source=(destination/'oracle.mm').read_text()
    source=replace(source,'#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"',
        '#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"\n'
        '#include "dev/benchmarks/prefill_moe_sep21/w8a8/precision.hpp"')
    source=source.replace('prefill_moe_sep21::','prefill_moe_w8a8::')
    source=replace(source,'prefill_moe_w8a8::cpuSelfTest();',
        'prefill_moe_w8a8::cpuSelfTest();prefill_moe_w8a8::precision::cpuSelfTest();')
    source=replace(source,'bool runCase(MetalBackend &backend, const NativeStoreView &store,',
        HELPERS+'\nbool runCase(MetalBackend &backend, const NativeStoreView &store,\n'
        '    const prefill_moe_w8a8::Workspace &workspace,')
    source=replace(source,
        '} else for (uint64_t i=0;i<hidden.size();++i)\n'
        '    hidden[i]=bf16(float(int((i*73+i/2560*17)%257)-128)/(std::getenv("PREFILL_MOE_SEP21_NORMALIZED") ? 74.0f : 512.0f));',
        '} else hidden=normalizedHidden(rows,store,ids);\n'
        '  double hiddenRMSmin=std::numeric_limits<double>::infinity(),hiddenRMSmax=0;\n'
        '  for (uint32_t row=0;row<rows;++row) {\n'
        '    double squares=0;for(uint32_t k=0;k<2560;++k){const double value=number(hidden[uint64_t(row)*2560+k]);squares+=value*value;}\n'
        '    const double rms=std::sqrt(squares/2560);hiddenRMSmin=std::min(hiddenRMSmin,rms);hiddenRMSmax=std::max(hiddenRMSmax,rms);\n'
        '  }')
    source=source.replace('prefill_moe_w8a8::HitCommands','prefill_moe_w8a8::QuantizedCommands')
    source=replace(source,'plans.emplace_back(graphs[1].dispatches(),rows,variants[index]);',
        'plans.emplace_back(graphs[1].dispatches(),rows,variants[index],store.layer,workspace);')
    source=replace(source,'    std::string outputHash;',
        '    std::string outputHash,initialCertificate,finalCertificate;\n'
        '    bool initialRawPass=false,finalRawPass=false;')
    source=replace(source,'return std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});',
        'return workspace.guardsClean() && std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});')
    source=replace(source,
        '(void)backend.submitCommand(plans[position].commands);\n      inspect(position,true);',r'''
      (void)backend.submitCommand(plans[position].auditCommands);
      const auto certificate=certify(store,scratch,packed,workspace);
      r.initialRawPass=certificate.pass();std::ostringstream detail;certificate.write(detail);r.initialCertificate=detail.str();
      if (!r.initialRawPass) fail(r,"independent initial quantization/I32/scaled-F32/original-F64-envelope certificate failed");
      inspect(position,true);
      const std::array<std::string,3> auditHashes{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])};
      (void)backend.submitCommand(plans[position].commands);
      if (auditHashes!=std::array<std::string,3>{digest(scratch[1].packedActivated),digest(scratch[1].scatteredDown),digest(output[1])}) {
        fail(r,"audit and timed W8A8 BF16 producers differ");r.eligible=false;
      }''')
    source=replace(source,
        'r.eligible=r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);',
        'r.eligible=r.initialRawPass && r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);')
    source=replace(source,
        'r.eligible=r.eligible && r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);',
        'r.eligible=r.eligible && r.finalRawPass && r.diagnostic==kSticky && r.buckets && r.canaries && guard && (!strict || exact);')
    source=replace(source,
        '(void)backend.submitCommand(plans[position].commands);\n        inspect(position,false);',r'''
        (void)backend.submitCommand(plans[position].auditCommands);
        const auto certificate=certify(store,scratch,packed,workspace);
        r.finalRawPass=certificate.pass();std::ostringstream detail;certificate.write(detail);r.finalCertificate=detail.str();
        if (!r.finalRawPass) fail(r,"independent final quantization/I32/scaled-F32/original-F64-envelope certificate failed");
        inspect(position,false);''')
    source=source.replace('r.initialChecked && r.finalChecked && r.errorGuard &&',
        'r.initialChecked && r.finalChecked && r.initialRawPass && r.finalRawPass && r.errorGuard &&')
    source=replace(source,
        '<<",\\\"synthetic_input_policy\\\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" :\n'
        '          std::getenv("PREFILL_MOE_SEP21_NORMALIZED") ? "private normalized /74 BF16 fixture" : "frozen inherited /512 BF16 fixture")',
        '<<",\\\"synthetic_input_policy\\\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" : "true per-row RMS normalization with actual-code near-zero cancellation row, then BF16 rounding")\n'
        '      <<",\\\"true_row_rms_normalized_fixture\\\":"<<(!rawInput ? "true" : "false")\n'
        '      <<",\\\"actual_code_cancellation_fixture\\\":"<<(!rawInput ? "true" : "false")\n'
        '      <<",\\\"hidden_row_rms_min\\\":"<<finiteJSON(hiddenRMSmin)<<",\\\"hidden_row_rms_max\\\":"<<finiteJSON(hiddenRMSmax)')
    source=replace(source,'<<",\\\"additional_prefix_dispatches\\\":0,\\\"original_dispatch_count\\\":"',
        '<<",\\\"additional_quantizer_dispatches\\\":2,\\\"original_dispatch_count\\\":"')
    source=replace(source,'splash::json::quote(v.kind)','splash::json::quote("w8a8_whole")')
    source=replace(source,'<<v.k<<",\\\"sg\\\":"<<v.sg','<<0<<",\\\"sg\\\":"<<v.sg')
    source=replace(source,'<<(v.registerOperand ? "true" : "false")','<<"false"')
    source=replace(source,'<<(v.staticExtent ? "true" : "false")','<<"false"')
    source=replace(source,'<<(v.rowParts ? 16 : 32)','<<32')
    source=replace(source,'<<(v.differentReduction ? "true" : "false")','<<"true"')
    source=replace(source,'<<(exact ? "false" : "true")','<<"true"')
    source=replace(source,'<<",\\\"initial_activation\\\":";r.initial[0].write(out);',
        '<<",\\\"quantizer_commands_included_in_timing\\\":2,\\\"activation_quantization_numerical_alternative\\\":true"\n'
        '        <<",\\\"initial_raw_f64_certificate\\\":"<<(r.initialCertificate.empty()?"null":r.initialCertificate)\n'
        '        <<",\\\"final_raw_f64_certificate\\\":"<<(r.finalCertificate.empty()?"null":r.finalCertificate)\n'
        '        <<",\\\"initial_activation\\\":";r.initial[0].write(out);')
    source=source.replace('PREFILL_MOE_SEP21_VARIANT','PREFILL_MOE_W8A8_VARIANT')
    source=source.replace('PREFILL_MOE_SEP21_STRICT','PREFILL_MOE_W8A8_STRICT')
    source=source.replace('0.001,1.0','0.05,1.0').replace('0.999999,1.0','0.9985,1.0')
    source=source.replace('.001 relative-L2/.999999 cosine','.05 relative-L2/.9985 cosine')
    source=replace(source,'      std::ofstream out(argv[6]);require(bool(out),"cannot create native report");',r'''
      const uint64_t quantPlanned=w8::Workspace::plannedBytes(rows);
      auto quantAdmission=governor.tryReserve(quantPlanned);
      require(bool(quantAdmission),"independent W8A8 activation/audit workspace admission denied");
      auto workspace=w8::Workspace::allocate(backend,rows);quantAdmission->commit();
      const auto quantizerEdgeReport=quantizerEdges(backend);
      std::ofstream out(argv[6]);require(bool(out),"cannot create native report");''')
    source=replace(source,'<<",\\\"planned_bytes\\\":"<<planned<<",\\\"host_reserve_bytes\\\":"<<reserve',
        '<<",\\\"planned_bytes\\\":"<<planned+quantPlanned<<",\\\"host_reserve_bytes\\\":"<<reserve\n'
        '          <<",\\\"independent_activation_and_audit_admitted_bytes\\\":"<<quantPlanned\n'
        '          <<",\\\"quantizer_edge_fixture\\\":"<<quantizerEdgeReport\n'
        '          <<",\\\"numerical_alternative\\\":true,\\\"model_semantics_qualified\\\":false,\\\"mtp_acceptance_qualified\\\":false"')
    source=replace(source,'runCase(backend,store,rows,pattern,pairs,maxL2,minCosine,strict,entry)',
        'runCase(backend,store,workspace,rows,pattern,pairs,maxL2,minCosine,strict,entry)')
    source=replace(source,'after-before<=planned,"one-layer native ledger exceeded reservation"',
        'after-before<=planned+quantPlanned,"one-layer W8A8 ledger exceeded independent reservations"')
    source=replace(source,'<<",\\\"final_allocated_bytes\\\":"<<after-before<<"}\\n";',
        '<<",\\\"activation_and_audit_guards_clean\\\":"<<(workspace.guardsClean()?"true":"false")\n'
        '          <<",\\\"final_allocated_bytes\\\":"<<after-before<<"}\\n";')
    source=source.replace('prefill-moe-sep21-eleven-variant-one-layer-v1','prefill-moe-sep21-w8a8-one-layer-v1')
    source=source.replace('eleven_variant_gpu_parity','w8a8_gpu_certificate_and_parity')
    source=source.replace('warm rotating private variants paired with native M32',
        'warm rotating W8A8 variants paired with native M32; both actual quantizer dispatches included')
    (destination/'oracle.mm').write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
