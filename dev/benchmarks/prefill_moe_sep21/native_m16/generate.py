#!/usr/bin/env python3
"""Generate a bounded one-layer shipping M32/M16 oracle; never load payloads."""
from pathlib import Path
import argparse
import importlib.util


INPUT = r'''
enum class InputPolicy { RowRMS, Inherited512, Divisor74 };
InputPolicy inputPolicy() {
  const char *raw=std::getenv("PREFILL_MOE_NATIVE_M16_INPUT_POLICY");
  if (!raw || std::string_view(raw)=="row-rms") return InputPolicy::RowRMS;
  if (std::string_view(raw)=="inherited512") return InputPolicy::Inherited512;
  if (std::string_view(raw)=="divisor74") return InputPolicy::Divisor74;
  throw std::runtime_error("unknown bounded M16 input policy");
}
std::vector<uint16_t> syntheticHidden(uint32_t rows,InputPolicy policy) {
  std::vector<uint16_t> hidden(uint64_t{rows}*2560);
  for (uint32_t row=0;row<rows;++row) {
    double square=0;
    for (uint32_t column=0;column<2560;++column) {
      const uint64_t i=uint64_t{row}*2560+column;
      const double value=int((i*73+i/2560*17)%257)-128;
      square+=value*value;
    }
    const double divisor=policy==InputPolicy::RowRMS ? std::sqrt(square/2560) :
        policy==InputPolicy::Divisor74 ? 74 : 512;
    require(divisor>0 && std::isfinite(divisor),"invalid synthetic row RMS");
    for (uint32_t column=0;column<2560;++column) {
      const uint64_t i=uint64_t{row}*2560+column;
      hidden[i]=bf16(float(int((i*73+i/2560*17)%257)-128)/float(divisor));
    }
  }
  return hidden;
}
std::array<double,2> rowRMS(std::span<const uint16_t> hidden,uint32_t rows) {
  require(hidden.size()==uint64_t{rows}*2560,"RMS input extent differs");
  std::array<double,2> bounds{std::numeric_limits<double>::infinity(),0};
  for (uint32_t row=0;row<rows;++row) {
    double square=0;
    for (uint32_t column=0;column<2560;++column) {
      const double value=number(hidden[uint64_t{row}*2560+column]);
      require(std::isfinite(value),"nonfinite hidden fixture");
      square+=value*value;
    }
    const double rms=std::sqrt(square/2560);
    bounds[0]=std::min(bounds[0],rms);bounds[1]=std::max(bounds[1],rms);
  }
  return bounds;
}
'''


COMPARISON = r'''
struct Comparison final {
  uint64_t elements=0,mismatches=0,nonfinite=0;
  double relativeL2=0,cosine=1,maxAbs=0;
  bool guardPass=false;
  void write(std::ostream &out) const {
    const auto value=[&](double v) {
      if (std::isfinite(v)) out<<v; else out<<"null";
    };
    out<<"{\"elements\":"<<elements<<",\"bf16_mismatches\":"<<mismatches
        <<",\"nonfinite\":"<<nonfinite<<",\"relative_l2\":";value(relativeL2);
    out<<",\"cosine\":";value(cosine);out<<",\"max_abs\":";value(maxAbs);
    out<<",\"error_guard_pass\":"<<(guardPass ? "true" : "false")<<'}';
  }
};
Comparison compare(const MetalBuffer &a,const MetalBuffer &b,uint64_t elements,
                   double maxL2,double minCosine) {
  require(a.sizeBytes()>=elements*2 && b.sizeBytes()>=elements*2,
      "output comparison extent differs");
  const auto *av=static_cast<const uint16_t *>(a.contents());
  const auto *bv=static_cast<const uint16_t *>(b.contents());
  Comparison result;result.elements=elements;
  double diff=0,normA=0,normB=0,dot=0;
  for (uint64_t i=0;i<elements;++i) {
    result.mismatches+=av[i]!=bv[i];
    const double x=number(av[i]),y=number(bv[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) {++result.nonfinite;continue;}
    const double delta=x-y;diff+=delta*delta;normA+=x*x;normB+=y*y;dot+=x*y;
    result.maxAbs=std::max(result.maxAbs,std::abs(delta));
  }
  result.relativeL2=normA ? std::sqrt(diff/normA) :
      diff ? std::numeric_limits<double>::infinity() : 0;
  result.cosine=normA && normB ? dot/std::sqrt(normA*normB) : normA==normB ? 1 : 0;
  result.guardPass=!result.nonfinite && std::isfinite(result.relativeL2) &&
      result.relativeL2<=maxL2 && std::isfinite(result.cosine) && result.cosine>=minCosine;
  return result; // Always retain complete comparisons, including rejected candidates.
}
'''


CPU = r'''
  require(moEBucketJobCapacity(2048,10,16)==1791 &&
      moEBucketJobCapacity(2048,10,8)==3071,"M16 bounded scratch capacity differs");
  ref::Packed uniform;uniform.rows=2048;uniform.selections=10;
  const auto uniformIDs=patternIDs(2048,hot,"spread-all");
  for (int64_t expert:uniformIDs) ++uniform.counts[expert];
  for (uint32_t expert=0;expert<512;++expert) {
    require(uniform.counts[expert]==40,"R2048 spread-all is not forty rows/expert");
    uniform.offsets[expert+1]=uniform.offsets[expert]+uniform.counts[expert];
  }
  const auto uniformM32=ref::makeJobs(uniform,32),uniformM16=ref::makeJobs(uniform,16);
  require(uniformM32.count==1024 && uniformM16.count==1536 &&
      uniformM32.count*32==32768 && uniformM16.count*16==24576,
      "M16/M32 uniform padding golden differs");
  const auto normalized=syntheticHidden(2048,InputPolicy::RowRMS);
  const auto rms=rowRMS(normalized,2048);
  require(rms[0]>.999 && rms[1]<1.001,"normalized BF16 row RMS differs");
  const auto inherited=syntheticHidden(2,InputPolicy::Inherited512);
  for (uint64_t i=0;i<inherited.size();++i)
    require(inherited[i]==bf16(float(int((i*73+i/2560*17)%257)-128)/512.0f),
        "inherited /512 fixture drift");
'''


def replace(source, old, new, count=1):
    actual=source.count(old)
    if actual != count:
        raise RuntimeError(f'native M64 adapter drift: {old!r}: {actual} != {count}')
    return source.replace(old,new)


def generate(destination: Path):
    path=Path('dev/benchmarks/prefill_moe_sep21/native_m64/generate.py')
    spec=importlib.util.spec_from_file_location('native_m64_source',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body=module.BODY
    body=replace(body,'m == 64 ? "_m64_n64_sg8" : "_m32_n64"',
        '"_m" + std::to_string(m) + "_n64"',2)
    body=replace(body,'m == 64 ? 256u : 128u','128u',2)
    body=replace(body,'{32u,64u}','{32u,16u}')
    body=replace(body,'ref::makeJobs(packed,64)','ref::makeJobs(packed,16)')
    body=replace(body,'FlashMoEBlockedTile::M64N64','FlashMoEBlockedTile::M16N64')
    body=replace(body,'[32,64]','[32,16]')
    body=replace(body,'[4,8]','[4,4]')
    body=replace(body,' &&\n          setenv("SPLASH_FLASH_MOE_M64","1",1)==0','')
    body=replace(body,'const bool strict=std::getenv("PREFILL_MOE_NATIVE_M64_STRICT")!=nullptr;',
        'const bool strict=true; // Complete BF16 equality is mandatory before timing.')
    body=body.replace('M32/M64','M32/M16').replace('m32_m64','m32_m16')
    body=body.replace('m32_and_m64','m32_and_m16').replace('m32-m64','m32-m16')
    body=body.replace('native-m64-oracle','native-m16-oracle')
    body=replace(body,'namespace one = splash::flash::qmv_one_layer;',
        'namespace one = splash::flash::qmv_one_layer;\n'+INPUT)
    body=replace(body,'  require(sizeof(FlashInt8ExpertStoreParams)==32,"native I8 parameter ABI differs");',
        CPU+'  require(sizeof(FlashInt8ExpertStoreParams)==32,"native I8 parameter ABI differs");')
    body=replace(body,'  } else for (uint64_t i=0;i<hidden.size();++i)\n'
        '    hidden[i]=bf16(float(int((i*73+i/2560*17)%257)-128)/512.0f);',
        '  } else hidden=syntheticHidden(rows,inputPolicy());\n'
        '  const auto rms=rowRMS(hidden,rows);')
    body=replace(body,'!activation.mismatches && !down.mismatches && !combined.mismatches',
        '!activation.mismatches && !activation.nonfinite && !down.mismatches && '
        '!down.nonfinite && !combined.mismatches && !combined.nonfinite',2)
    body=replace(body,r'      <<",\"synthetic_input_policy\":\"frozen inherited /512 BF16 fixture; not actual model activation\""',
        r'''      <<",\"synthetic_input_policy\":"<<splash::json::quote(rawInput ? "caller BF16 fixture" :
          inputPolicy()==InputPolicy::RowRMS ? "per-row RMS normalized before BF16 rounding; not actual model activation" :
          inputPolicy()==InputPolicy::Divisor74 ? "private approximate normalized /74 BF16 fixture" : "frozen inherited /512 BF16 fixture")
      <<",\"input_row_rms_min\":"<<rms[0]<<",\"input_row_rms_max\":"<<rms[1]
      <<",\"padded_matrix_rows\":["<<jobs[0].count*32<<','<<jobs[1].count*16<<']'
      <<",\"scratch_job_capacity\":["<<scratch[0].buckets.jobCapacity<<','<<scratch[1].buckets.jobCapacity<<']'
      <<",\"coefficients_scales_and_bf16_boundaries_shared\":true,\"additional_prefix_dispatches\":0"''')
    body=replace(body,r'\"native_m32_m16_gpu_parity\":\"pending\"',
        r'\"native_m32_m16_gpu_parity\":\"pending\",\"uniform_rows_per_expert\":40,\"m32_padded_rows\":32768,\"m16_padded_rows\":24576,\"m16_job_capacity\":1791,\"scratch_job_capacity\":3071,\"bf16_row_rms_bounds\":[0.999,1.001]')
    module.BODY=body
    module.generate(destination)
    generated=destination/'oracle.mm'
    source=generated.read_text()
    start=source.index('struct Comparison final {')
    end=source.index('void times(',start)
    source=source[:start]+COMPARISON+source[end:]
    generated.write_text(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
